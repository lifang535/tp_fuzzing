# TileSmith Experiment Report

**Date**: 2026-07-11  
**Setup**: 3 GPUs (RTX4060 / RTX4090 / A800), 2 backends (TileLang / Triton), 2 shape modes (easy-shape / hard-shape), seed=42  
**Changes from last run**: Test scale doubled (TileLang 10,000→20,000 iterations/config, Triton 20,000→40,000); input parameter ranges expanded; constraints moved from fuzzer frontend to backend.

---

## 1. Trigger Count Overview

### TileLang Backend (20,000 iterations per configuration)

| GPU | Shape Mode | Total Tested | Triggered | Rate | wrong_result | dtype_mismatch | ptx_async_boundary | warp_partition | shared_memory_overflow | timeout | assertion_failure | gpu_oom |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| RTX4060 | easy-shape | 20,000 | 3,805 | 19.0% | 2,962 | 698 | 100 | 34 | — | 9 | 2 | — |
| RTX4060 | hard-shape | 20,000 | 3,830 | 19.2% | 3,315 | 476 | 1 | 20 | — | 15 | 2 | 1 |
| RTX4090 | easy-shape | 20,000 | 3,985 | 19.9% | 3,128 | 745 | 72 | 40 | — | — | — | — |
| RTX4090 | hard-shape | 20,000 | 3,816 | 19.1% | 3,287 | 501 | 2 | 24 | 2 | — | — | — |
| A800    | easy-shape | 20,000 | 3,713 | 18.6% | 2,840 | 750 | 99 | 24 | — | — | — | — |
| A800    | hard-shape | 20,000 | 3,554 | 17.8% | 3,057 | 474 | 1 | 22 | — | — | — | — |

### Triton Backend (40,000 iterations per configuration)

| GPU | Shape Mode | Total Tested | Triggered | Rate | wrong_result | shared_memory_overflow | timeout | assertion_failure | gpu_oom |
|---|---|---|---|---|---|---|---|---|---|
| RTX4060 | easy-shape | 40,000 | 6,191 | 15.5% | 6,155 | 25 | 10 | 1 | — |
| RTX4060 | hard-shape | 40,000 | 6,571 | 16.4% | 6,521 | 34 | 13 | 1 | 2 |
| RTX4090 | easy-shape | 40,000 | 5,987 | 15.0% | 5,949 | 38 | — | — | — |
| RTX4090 | hard-shape | 40,000 | 6,667 | 16.7% | 6,619 | 48 | — | — | — |
| A800    | easy-shape | 40,000 | 5,977 | 14.9% | 5,963 | 14 | — | — | — |
| A800    | hard-shape | 40,000 | 6,470 | 16.2% | 6,455 | 15 | — | — | — |

---

## 2. Analysis of `wrong_result` Cases

Consistent with the previous run, `wrong_result` counts remain large (TileLang ~2,800–3,300 per configuration, Triton ~5,900–6,600 per configuration) and are **all false positives** from the same three sources: incorrect oracle reference semantics, floating-point errors amplified by conditional operators like `WHERE`, and thresholds that are too tight for fp32 GEMM with large K. The expanded parameter ranges increased absolute counts but the trigger rate remained stable. All subsequent analysis excludes `wrong_result`.

---

## 3. Real Bug Summary

Excluding `wrong_result`, this run detected **8 distinct root causes** — three new ones compared to the previous run: `timeout`, `assertion_failure`, and `gpu_oom`. Totals across all 3 GPUs:

| root_cause | Backend | Crash Stage | Total (3 GPUs) |
|---|---|---|---|
| `dtype_mismatch` | TileLang | **runtime fail** | 3,644 |
| `ptx_async_boundary` | TileLang | **compile fail** | 275 |
| `shared_memory_overflow` | Triton | **runtime fail** | 174 |
| `warp_partition` | TileLang | **compile fail** | 164 |
| `timeout` | TileLang + Triton | — | 47 |
| `shared_memory_overflow` | TileLang | **runtime fail** | 2 |
| `assertion_failure` | TileLang + Triton | **runtime fail** | 6 |
| `gpu_oom` | TileLang + Triton | **runtime fail** | 3 |

> **Comparison with previous run**: `dtype_mismatch` grew from 454 to 3,644 (~8×), `ptx_async_boundary` from 8 to 275 (~34×), `warp_partition` from 40 to 164 (~4×), and `shared_memory_overflow` (Triton) from 72 to 174 (~2.4×). The exceptional growth in `ptx_async_boundary` — far beyond the 2× scale increase — is explained in Section 4.

---

## 4. Compile Fail: Detailed Analysis

### 4.1 `warp_partition`

Behavior is identical to the previous run: TileLang's `LayoutInference` pass cannot factor the thread count into a valid `m_warp × n_warp` warp grid when block_M and block_N are too small (e.g., 16×16), causing a compile-time assertion failure.

```
InternalError: m_warp * n_warp must equal num_warps,
               m_warp: 1, n_warp: 1, num_warps: 4
```

The total across 3 GPUs is 164 (up from 40), an ~4× increase consistent with the doubled test scale and expanded parameter space sampling more small-block configurations.

---

### 4.2 `ptx_async_boundary`

**Previous run**: triggered only in hard-shape mode (8 total), because odd/prime K values were exclusive to hard-shape.  
**This run**: triggers heavily in easy-shape mode (RTX4060: 100, RTX4090: 72, A800: 99) while hard-shape shows almost none (1–2 per GPU).

The cause is the **expanded parameter ranges**: `block_K` is no longer restricted to powers of 2. Certain `block_K` values cause the per-row transfer size `block_K × sizeof(dtype)` to fall outside the {4, 8, 16} bytes required by the `cp.async` PTX instruction, triggering a codegen assertion even when K itself is a power of 2.

```
InternalError: tl::ptx_cp_async requires PTX byte width in {4, 8, 16}, but got 2
```

Many triggering cases have non-standard `block_K` values (e.g., block_K=1), confirming that the root cause has shifted from shape-mode-dependent K alignment to unconstrained block_K sampling.

---

## 5. Runtime Fail: Detailed Analysis

### 5.1 `dtype_mismatch`

Behavior is unchanged from the previous run: mixed-precision kernels (`dtype="float16"`, `accum_dtype="float32"`) cause TileLang's type inference to infer float32 as the expected input type, conflicting with the float16 tensors passed by the caller.

```
RuntimeError: kernel impl input A dtype mismatch, expected float32
```

The total across 3 GPUs is 3,644 (up from 454, ~8× increase) — well above the expected 2× from the test scale increase. The expanded parameter ranges cause the fuzzer to sample more mixed-precision kernel configurations, covering more trigger paths. This remains a systemic defect in TileLang's type inference for mixed-precision patterns.

---

### 5.2 `shared_memory_overflow`

Triton backend behavior is identical to the previous run: the kernel compiles successfully, and `_init_handles` discovers the shared memory overcommitment only on the first execution call.

```
triton.runtime.errors.OutOfResources:
  out of resource: shared memory, Required: 110592, Hardware limit: 101376
```

The cross-GPU frequency ordering this run is RTX4090 > RTX4060 > A800 (RTX4090: 86, RTX4060: 59, A800: 29). This differs slightly from the previous run (RTX4060 > RTX4090 > A800); with the expanded parameter space, RTX4090 happened to sample more configurations near its hardware limit.

TileLang also triggered `shared_memory_overflow` twice on RTX4090 hard-shape, consistent with the sparse occurrence seen in the previous run.

---

### 5.3 `timeout` (new)

**Only observed on RTX4060** (TileLang: easy-shape 9 + hard-shape 15; Triton: easy-shape 10 + hard-shape 13; RTX4090 and A800: 0 each). Total: 47 across 3 GPUs.

**Example**: `failed_dynamic_gemm+copy_f2g_M32,N256,K32,bM16,bN128,bK32,t128,pipelined,s4,float16.py`

```
error_message: "Execution timed out"
```

These kernels typically involve deep pipelines (`num_stages=4`) or `copy_f2g` in dynamic sequences. A timeout could indicate either a genuine GPU hang (unmatched barrier, warp starvation) or simply that RTX4060's lower compute throughput caused the same kernel to exceed the fuzzer's time limit. Since RTX4090 and A800 did not time out on the same configurations, it is premature to classify these as compiler bugs. Recommended follow-up: reproduce with `CUDA_LAUNCH_BLOCKING=1` and a GPU watchdog to distinguish true hangs from performance-limited runs.

---

### 5.4 `assertion_failure` (new)

6 total, all on RTX4060 (TileLang easy/hard: 2 each; Triton easy/hard: 1 each). RTX4090 and A800: 0.

**Typical error message**:
```
RuntimeError: CUDA error: CUDA-capable device(s) is/are busy or unavailable
```

Despite the name, these are not compiler assertion failures. The actual mechanism is: **a preceding kernel crash (e.g., `warp_partition`) leaves the GPU in an error state; the next test's `torch.randn(..., device='cuda')` call fails to acquire a CUDA context and surfaces this secondary error**. This is a cascading crash effect, not an independent bug. It exposes a gap in the fuzzer: there is no GPU state reset between tests after a crash. RTX4060's weaker post-crash CUDA context recovery likely explains why this only appears on that card.

---

### 5.5 `gpu_oom` (new)

3 total, all on RTX4060 (TileLang hard-shape: 1; Triton hard-shape: 2). RTX4090 and A800: 0.

**Example**: `M=6528, N=9414, K=9399, dtype=float32` (dynamic sequence with 9 operators)

```
RuntimeError: CUDA error: CUBLAS_STATUS_ALLOC_FAILED when calling `cublasCreate(handle)`
```

This is the **oracle reference implementation** (PyTorch/cuBLAS) running out of VRAM on RTX4060's 8 GB when allocating matrices in the thousands for both dimensions — not a defect in the kernel under test. RTX4090 (24 GB) and A800 (80 GB) have sufficient VRAM for the same shapes. The expanded parameter ranges introduced extreme M×N×K combinations that were previously excluded by frontend constraints. The fix is to add a pre-check in the fuzzer frontend that bounds matrix size combinations based on available GPU VRAM.

---

## 6. Correspondence with Paper Bug Taxonomy

Reference paper: *Characterizing Real-World Bugs in Tile Programs for Automated Bug Detection* (ISSTA '26)

| root_cause | Crash Stage | Paper Category | Corresponding Paper Case |
|---|---|---|---|
| `warp_partition` | compile fail | Tile Mapping and Launch Bugs (6.31%) | Triton #5265 (num_warps assertion failure) |
| `ptx_async_boundary` | compile fail | Memory Bugs (19.27%) + Device-Specific Bugs (3.99%) | PTX alignment constraint; non-standard block_K triggers codegen assertion |
| `dtype_mismatch` | runtime fail | Type and Operator Bugs (48.84%) — Data-Type Semantics | Apache TVM #14112 (dtype mismatch causes transform_layout failure) |
| `shared_memory_overflow` | runtime fail | Memory Bugs (19.27%) — Resource Allocation | Triton `OutOfResources`; TileLang dynamic shared memory allocation failure |
| `timeout` | — | Crash (58.14%) — potential deadlock | Requires further reproduction to distinguish hang from performance limit |
| `assertion_failure` | runtime fail | — | Cascading GPU crash effect; not an independent compiler bug |
| `gpu_oom` | runtime fail | — | Oracle (cuBLAS) OOM on large matrices; not a defect in the tested kernel |

**Key findings**:

1. **`dtype_mismatch` remains the most frequent real bug** (3,644 occurrences), growing ~8× from the previous run. The larger parameter space samples more mixed-precision configurations, further exposing the systemic defect in TileLang's type inference for `dtype` + `accum_dtype` combinations.

2. **`ptx_async_boundary` is no longer shape-mode-specific**: with expanded parameter ranges, easy-shape configurations now trigger it via non-standard `block_K` values, revealing that the root cause is broader than "odd/prime K" — the compiler's codegen has no pre-check for arbitrary `block_K` alignment against the `cp.async` constraint.

3. **`timeout` requires disambiguation**: all 47 cases are on RTX4060 and could be genuine deadlocks or simply performance-limited behavior. Using `CUDA_LAUNCH_BLOCKING=1` with a watchdog is the recommended next step before treating these as compiler bugs.

4. **`assertion_failure` and `gpu_oom` are tool-robustness issues**, not new compiler bugs. `assertion_failure` indicates the fuzzer needs a GPU state reset after crashes; `gpu_oom` indicates the oracle reference implementation needs VRAM-aware matrix size bounds before allocation.
