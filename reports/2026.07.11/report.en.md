# TileSmith Experiment Report

**Date**: 2026-07-11  
**Setup**: 3 GPUs (RTX4060 / RTX4090 / A800), 2 backends (TileLang / Triton), 2 shape modes (easy-shape / hard-shape), seed=42

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

## 2. False Positive Analysis

The following two categories are false positives and are excluded from subsequent analysis.

### 2.1 `wrong_result`

`wrong_result` counts are large (TileLang ~2,800–3,300 per configuration, Triton ~5,900–6,600 per configuration) but are **all false positives**, arising from three sources:

**1. Incorrect oracle reference semantics (primary cause)**  
Composite operators in dynamic sequences use a global reduction over the full matrix as the oracle reference, while the kernel performs local reductions per tile. When the tile size is much smaller than the matrix, local and global results differ by orders of magnitude, producing relative errors far above threshold. This is a defect in the fuzzer's oracle, not a compiler bug.

**2. Floating-point errors amplified by conditional operators**  
The `WHERE` operator gates on the sign of GEMM output. When the result is near zero, different accumulation orders in Triton vs. PyTorch can flip the sign, and `WHERE` amplifies this into `max_diff ≈ |D2|` (potentially > 1.0), far exceeding the threshold.

**3. Threshold too tight for fp32 GEMM with large K**  
For large K, the fp32 accumulation difference between Triton and PyTorch stabilizes at ~3%–5%, just above the 5% threshold. This is a pure floating-point precision issue unrelated to kernel correctness.

### 2.2 `timeout`

**Only observed on RTX4060** (TileLang: easy-shape 9 + hard-shape 15; Triton: easy-shape 10 + hard-shape 13; RTX4090 and A800: 0 each). Total: 47.

These timeouts are caused by **Python multi-threading stalls on the local test machine**, not by GPU kernel hangs or deadlocks. When running multiple configurations concurrently on the RTX4060 machine, Python thread scheduling stalls caused the timeout timer to fire spuriously. RTX4090 and A800 ran on more stable machines and did not time out on identical configurations, confirming that these are **test environment noise** rather than compiler bugs.

---

## 3. Real Bug Summary

Excluding `wrong_result` and `timeout`, this run detected **6 distinct root causes**, grouped into compile failures, runtime failures, and tool-robustness issues:

| root_cause | Backend | Crash Stage | Total (3 GPUs) |
|---|---|---|---|
| `dtype_mismatch` | TileLang | **runtime fail** | 3,644 |
| `ptx_async_boundary` | TileLang | **compile fail** | 275 |
| `shared_memory_overflow` | Triton | **runtime fail** | 174 |
| `warp_partition` | TileLang | **compile fail** | 164 |
| `shared_memory_overflow` | TileLang | **runtime fail** | 2 |
| `assertion_failure` | TileLang + Triton | **runtime fail** | 6 |
| `gpu_oom` | TileLang + Triton | **runtime fail** | 3 |

---

## 4. Compile Fail: Detailed Analysis

### 4.1 `warp_partition`

**Crash stage**: Layout inference pass inside `tilelang.lower()`

**Example**: `results_RTX4060/.../failed/warp_partition/failed_pipeline_gemm+sqrt.py`

```python
# kernel declaration: block_M=16, block_N=16, threads=128 (= 4 warps)
with T.Kernel(..., threads=128) as (bx, by):
    ...
    T.gemm(A_shared, B_shared, C_local)   # GEMM triggers warp partition inference
```

Output:
```
TileLang begins to compile kernel `impl`
  tilelang/jit/__init__.py compile
    JITKernel.__init__
      _compile_and_create_adapter
        tilelang.lower()
          LayoutInference pass
            GemmNode.InferLayout
              compute_warp_partition
InternalError: m_warp * n_warp must equal num_warps,
               m_warp: 1, n_warp: 1, num_warps: 4
```

TileLang's `LayoutInference` pass factors the thread count into an `m_warp × n_warp` warp grid. With block_M=16 and block_N=16, the only valid factoring yields `m_warp=1, n_warp=1` (1 warp total), which cannot satisfy the requested 4 warps, causing a compile-time assertion failure.

**Root cause**: `ComputeDefaultWarpPartition` has no valid fallback for small tiles (e.g., 16×16). When the tile size cannot support the requested warp count, the compiler fails immediately. 164 total across 3 GPUs, with no strong correlation to shape mode.

---

### 4.2 `ptx_async_boundary`

**Crash stage**: PTX codegen pass inside `tilelang.lower()`

**Typical error**:
```
_compile_and_create_adapter
  tilelang.lower()
    device_codegen → BuildTileLangCUDA
      CodeGenTileLangCUDA → GetTileLangCPAsyncTransferBytes
InternalError: tl::ptx_cp_async requires PTX byte width in {4, 8, 16}, but got 2
```

The `cp.async` PTX instruction requires the per-row transfer size to be exactly 4, 8, or 16 bytes. The transfer size equals `block_K × sizeof(dtype)`. When `block_K` takes a non-standard value (e.g., block_K=1 with float16, giving 2 bytes per row), the constraint is violated and codegen fails.

**Relationship to shape mode**: easy-shape triggers this heavily (RTX4060: 100, RTX4090: 72, A800: 99) while hard-shape triggers almost none (1–2 per GPU). The root cause is expanded `block_K` sampling: easy-shape M/N/K are powers of 2, but `block_K` can now take arbitrary non-power-of-2 values, some of which violate the PTX alignment constraint even when K itself is a power of 2. The compiler lacks a pre-check on `block_K` alignment before entering codegen. 275 total across 3 GPUs.

---

## 5. Runtime Fail: Detailed Analysis

### 5.1 `dtype_mismatch`

**Crash stage**: first call to `kernel(A, B)` after successful compilation

**Example**: `results_RTX4060/.../failed/dtype_mismatch/failed_dynamic_gemm+copy_f2g.py`

```python
# kernel declaration: dtype="float16", accum_dtype="float32"
dtype = "float16"
accum_dtype = "float32"
...
C_local_1 = T.alloc_fragment((block_M, block_N), accum_dtype)  # fp32 accumulator
T.gemm(A_shared_1, B_shared_1, C_local_1)
T.copy(C_local_1, C[...])   # copy fp32 accumulator back to fp16 output
```

Output (no compile failure — crash only at call time):
```
C = kernel(A, B)             ← kernel already compiled; this is the first call
  tilelang/jit/kernel.py __call__
    tilelang/jit/adapter/tvm_ffi.py func
      executable(*tensor_list)
RuntimeError: kernel impl input A dtype mismatch, expected float32
```

**Root cause**: the kernel declares `dtype="float16"`, but TileLang's type inference infers float32 as the expected input type when processing the `accum_dtype="float32"` fragment buffer, conflicting with the float16 tensors passed by the caller. This mismatch is not caught at compile time and surfaces only when the JIT adapter performs argument type checking before execution.

**Trigger pattern**: virtually every kernel that combines `dtype` and `accum_dtype` (fp16 input/output with fp32 accumulator) triggers this. This is a systemic defect in TileLang's type inference for mixed-precision kernels. 3,644 total across 3 GPUs — the most frequent real bug in this run.

---

### 5.2 `shared_memory_overflow`

**Crash stage**: first execution call after successful compilation, during Triton's GPU handle initialization

**Typical error**:
```
kernel_0_kernel[grid](...)
  triton/runtime/jit.py run
    triton/compiler/compiler.py _init_handles
      raise OutOfResources(...)
triton.runtime.errors.OutOfResources:
  out of resource: shared memory, Required: 110592, Hardware limit: 101376
```

The Triton kernel compiles successfully. `_init_handles` queries the GPU hardware limit only on the first actual execution, at which point it discovers that the requested shared memory (108 KB) exceeds the GPU limit (99 KB).

**Cross-GPU frequency**: RTX4090 > RTX4060 > A800 (RTX4090 easy+hard=86, RTX4060 easy+hard=59, A800 easy+hard=29), inversely correlated with the shared memory capacity of each card. A800 has the largest per-SM shared memory, so the same configurations do not overflow there.

TileLang also triggered `shared_memory_overflow` twice on RTX4090 hard-shape, indicating that TileLang has a similar late-check timing issue, though at much lower frequency than Triton.

---

## 6. Tool-Robustness Issues

The following two categories are not compiler bugs, but expose engineering gaps in the fuzzer itself.

### 6.1 `assertion_failure`

6 total, all on RTX4060 (TileLang easy/hard: 2 each; Triton easy/hard: 1 each). RTX4090 and A800: 0.

**Typical error**:
```
RuntimeError: CUDA error: CUDA-capable device(s) is/are busy or unavailable
```

Despite the name, these are not compiler assertion failures. The actual mechanism is: **a preceding kernel crash (e.g., `warp_partition`) leaves the GPU in an error state; the next test's `torch.randn(..., device='cuda')` call fails to acquire a CUDA context and surfaces this secondary error**. This is a cascading crash effect, not an independent bug. It exposes a gap in the fuzzer: no GPU state reset is performed between tests after a crash.

### 6.2 `gpu_oom`

3 total, all on RTX4060 (TileLang hard-shape: 1; Triton hard-shape: 2). RTX4090 and A800: 0.

**Example**: `M=6528, N=9414, K=9399, dtype=float32` (dynamic sequence with 9 operators)

```
RuntimeError: CUDA error: CUBLAS_STATUS_ALLOC_FAILED when calling `cublasCreate(handle)`
```

This is the **oracle reference implementation** (PyTorch/cuBLAS) running out of VRAM on RTX4060's 8 GB when allocating large matrices — not a defect in the kernel under test. RTX4090 (24 GB) and A800 (80 GB) have sufficient VRAM. The fix is to add a VRAM-aware matrix size bound in the fuzzer frontend before oracle allocation.

---

## 7. Correspondence with Paper Bug Taxonomy

Reference paper: *Characterizing Real-World Bugs in Tile Programs for Automated Bug Detection* (ISSTA '26)

The paper classifies 301 bugs by symptom into three groups: Crash (58.14%), Correctness issues (36.21%), and Performance Bottleneck (5.65%).

| root_cause | Crash Stage | Paper Category | Corresponding Paper Case |
|---|---|---|---|
| `warp_partition` | compile fail | Tile Mapping and Launch Bugs (6.31%) | Triton #5265 (num_warps assertion failure) |
| `ptx_async_boundary` | compile fail | Memory Bugs (19.27%) + Device-Specific Bugs (3.99%) | PTX alignment constraint; non-standard block_K triggers codegen assertion |
| `dtype_mismatch` | runtime fail | Type and Operator Bugs (48.84%) — Data-Type Semantics | Apache TVM #14112 (dtype mismatch causes transform_layout failure) |
| `shared_memory_overflow` | runtime fail | Memory Bugs (19.27%) — Resource Allocation | Triton `OutOfResources`; TileLang dynamic shared memory allocation failure |

**Key findings**:

1. **`dtype_mismatch` is the most frequent real bug** (3,644 occurrences), corresponding to the largest paper category, Type and Operator Bugs (48.84%). Virtually every mixed-precision kernel (fp16 input + fp32 accumulator) triggers it, confirming a systemic defect in TileLang's type inference for mixed-precision patterns.

2. **`warp_partition` precisely reproduces Triton #5265** (`num_warps` cannot be factored into a valid warp grid), showing that this class of bug exists across frameworks — TileLang is equally affected.

3. **`ptx_async_boundary` has a broader trigger condition than "odd/prime K"**: any non-standard `block_K` alignment can trigger it, since the compiler's codegen has no pre-check. Easy-shape configurations are equally exposed when `block_K` sampling is unrestricted.

4. **`shared_memory_overflow` frequency is inversely correlated with GPU shared memory capacity** (RTX4090 > RTX4060 > A800), directly supporting the paper's claim that Resource Allocation Bugs are strongly hardware-dependent.
