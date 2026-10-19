# TileSmith Latest Experiment Audit

Date: 2026-10-19. All times are China Standard Time (UTC+8). [中文报告](REPORT.md)

**The campaign started around noon on October 16. At 18:25 on October 19, it had run for approximately 78 hours 16 minutes; all four experiment processes were still running, and the latest checkpoints totaled 164,000 tests.** Bug classification uses a separate, fixed snapshot of 6,176 failure records collected at 17:35: 4,669 records match **43 distinct compiler or downstream toolchain mechanisms**, and 87 records correspond to **four fuzzer defects**. Another 19 numerical discrepancies and eight nvcc crashes remain unresolved.

This report assigns independent, consecutive identifiers: **TileLang#1–TileLang#26 and Triton#1–Triton#17**. They do not retain the historical catalogue numbering. The Chinese report, English report, data tables and reproduction instructions use the same identifiers, which are scoped to this report. Under the same classification rules, the preceding long campaign reached 46 mechanisms; the current audited sample reaches 43, three fewer.

## 1. Runtime and experiment scope

The experiment source revision is `5a790b0235ec59e927583f7a02ac64763f17ebef`, on branch `tilelang-0.1.15-triton-3.8_2026.10.15`. Each server runs one TileLang campaign and one Triton campaign, with a target of 500,000 programs per run.

| Server | GPU / seed | Campaign start | Status observation | Wall-clock duration |
| --- | --- | --- | --- | --- |
| 41790 | RTX 4090 / 43 | 2026-10-16 12:09:22 | 2026-10-19 18:25:32 | 78 h 16 min 10 s |
| 32906 | 32 GB vGPU / 42 | 2026-10-16 12:10:02 | 2026-10-19 18:25:32 | 78 h 15 min 31 s |

### 1.1 Live progress and the fixed audit sample

Live progress establishes that the campaigns continue to advance. Failures generated after the fixed snapshot are not added to its classification totals.

| Server / backend | Latest test count read at 18:25 | Checkpoint time | Test count associated with the 17:35 audit | Saved failures in the fixed sample |
| --- | ---: | --- | ---: | ---: |
| 41790 / TileLang | 19,000 | 18:08:53 | 18,800 | 1,225 |
| 41790 / Triton | 42,600 | 18:19:25 | 42,100 | 1,312 |
| 32906 / TileLang | 32,300 | 18:18:12 | 31,900 | 2,136 |
| 32906 / Triton | 70,100 | 18:23:15 | 69,400 | 1,503 |
| Total | **164,000** | | **162,200** | **6,176** |

Failure files were collected during **2026-10-19 17:35:22–17:35:41**. The corresponding progress checkpoints were written at 17:26–17:33, approximately 77.3 hours into the campaign. They contain 156,027 passes and 6,173 failures. Three additional failure files appeared between the checkpoint reads and file collection, so classification uses the 6,176 saved JSON records. All were readable.

Completed tests are counted using `coverage_progress.json.total_tested`, rather than generation-attempt sequence numbers in logs. A pass can mean successful compilation for another architecture; it does not necessarily mean numerical correctness was tested on a GPU of that architecture.

### 1.2 Environment and process health

- Both servers use TileLang **0.1.15**, Triton **3.8.0**, PyTorch **2.4.0+cu124**, apache-tvm-ffi **0.1.11**, Python **3.11**, and system nvcc **12.4.131**.
- Result directories are `2026.10.16-12.09_{tilelang,triton}_hard-shape_seed=43` and `2026.10.16-12.10_{tilelang,triton}_hard-shape_seed=42`. Two short runs earlier that morning are excluded.
- All four main processes were present at the 18:25 observation. At the 17:51 health check, exit codes and stop requests were null, with approximately 44.20 / 49.92 GiB free disk space.
- During the source audit, all 152 commonly sampled source, documentation and test files matched between the local and remote checkouts. Remote source hashes were unchanged across the inspection. SSH subprocesses connected directly with proxies disabled.

Timestamps, counts, versions and sampling statistics are preserved in [audit_summary.json](data/audit_summary.json).

## 2. Classification of failure records

| Category | Records | Deduplication or attribution |
| --- | ---: | --- |
| Matches to real compiler / toolchain mechanisms | 4,669 | 26 TileLang-path mechanisms and 17 Triton-path mechanisms |
| Confirmed generator / checker defects | 87 | Four mechanisms, described in Section 5 |
| Resource limits, unsupported inputs, unstable references or timeouts | 1,393 | Neither counted as real compiler bugs nor collectively blamed on the fuzzer |
| nvcc crashes with unresolved attribution | 8 | Separate toolchain candidates |
| Numerical discrepancies with unresolved attribution | 19 | Reference-model and floating-point explanations are not fully excluded |
| Total | **6,176** | |

A record is a repeated trigger; a mechanism is a deduplicated bug category. There are 267 distinct `(backend, failure_bucket)` diagnostic buckets across the servers. One mechanism can produce several buckets, and one bucket can contain different mechanisms.

Evidence has two levels: saved diagnostics, parameters and reduced cores match independently verified mechanisms; this follow-up also ran independent examples and controls for **five** of the 43 mechanisms. It did not independently revalidate every one of the 4,669 records. Many cross-architecture findings are established through target compilation. See [classification.csv](data/classification.csv) for record assignments and [bug_catalog.json](data/bug_catalog.json) for the mechanism catalogue.

## 3. Real mechanisms reached in this campaign

### 3.1 TileLang path: 26 mechanisms, 3,018 records

| Identifier | Mechanism | 41790 records | 32906 records | Total |
| --- | --- | ---: | ---: | ---: |
| TileLang#1 | CUDA codegen cannot print vectorized bool types | 43 | 68 | 111 |
| TileLang#2 | An inferred reduction layout cannot be lowered | 38 | 52 | 90 |
| TileLang#3 | cp.async is emitted with an illegal transfer size | 17 | 40 | 57 |
| TileLang#4 | Vectorized nested integer division causes a conflicting Let binding | 2 | 6 | 8 |
| TileLang#5 | Ambiguous C++ constructors for int64/uint64 to f16/bf16 casts | 352 | 687 | 1039 |
| TileLang#6 | CUDA template type mismatch for int64 atomic min/max/or | 45 | 57 | 102 |
| TileLang#7 | Integer to bf16 conversion truncates instead of rounding to nearest | 9 | 35 | 44 |
| TileLang#8 | Missing synchronization in while-loop shared-memory copies followed by GEMM | 38 | 112 | 150 |
| TileLang#9 | Narrow integer arithmetic loses wraparound when widened | 73 | 152 | 225 |
| TileLang#10 | Invalid half-precision math intrinsic codegen; hpow/htanh overloads in this run | 6 | 33 | 39 |
| TileLang#11 | Missing lowering for popcount/copysign/nextafter/ldexp/hypot | 40 | 51 | 91 |
| TileLang#12 | Shared-memory remapping incorrectly removes the column offset | 29 | 92 | 121 |
| TileLang#13 | Region-copy lowering to ldmatrix drops the column offset | 9 | 5 | 14 |
| TileLang#14 | ldgstg configuration produces incompatible vector-lane/reinterpret widths | 3 | 5 | 8 |
| TileLang#15 | index64 rewriting changes values or vector types and can break codegen | 40 | 74 | 114 |
| TileLang#16 | CUDA codegen cannot print Blackwell 256-bit int16/uint16 vector types | 35 | 68 | 103 |
| TileLang#17 | HIP has no lowering rules for tl.ieee_* intrinsics | 121 | 149 | 270 |
| TileLang#18 | sm80 fp8 GEMM selects an unsupported MMA and fails at runtime | 12 | 17 | 29 |
| TileLang#19 | Narrow-N sm90 WGMMA triggers descriptor-layout assertions | 0 | 3 | 3 |
| TileLang#20 | HIP fp8 constants and MFMA datatype tables have coverage gaps | 47 | 68 | 115 |
| TileLang#21 | HIP int8/uint8 xor fails the codegen Check(p) | 42 | 66 | 108 |
| TileLang#22 | bf16 pow is rejected by the is_float type check | 5 | 11 | 16 |
| TileLang#23 | Explicit pipeline stage/order multibuffers fragments and breaks layouts | 58 | 71 | 129 |
| TileLang#24 | Floating-point x-x is folded to zero for infinite/NaN inputs | 1 | 1 | 2 |
| TileLang#25 | GEMM accumulator plus a separate fragment row reduction triggers has_best | 5 | 4 | 9 |
| TileLang#26 | sm75 bf16 atomic_min/max selects an incompatible CUDA 12.4 atomic_ref branch | 8 | 13 | 21 |

The TileLang#15 record `194926aa6a83ea49` retains both index64 and while/GEMM conditions and may overlap TileLang#8. Its current assignment is a record index, not evidence of an additional distinct mechanism.

### 3.2 Triton path: 17 mechanisms, 1,651 records

| Identifier | Mechanism | 41790 records | 32906 records | Total |
| --- | --- | ---: | ---: | ---: |
| Triton#1 | Half-precision abs retains negative signs in some lanes (ptxas 12.9) | 24 | 49 | 73 |
| Triton#2 | Double rounding in float32 to fp8 conversion | 0 | 17 | 17 |
| Triton#3 | CUDA lowering failure or abort for float64 and fp8 conversions | 319 | 317 | 636 |
| Triton#4 | Rounding-mode assertion for conversions between fp8 formats | 93 | 104 | 197 |
| Triton#5 | rtz downcast lowering crashes, including AMD maybeOutVals assertions | 65 | 72 | 137 |
| Triton#6 | bf16 custom reduce/scan combine has an inconsistent promoted type | 52 | 74 | 126 |
| Triton#7 | Garbage lanes after conversion of folded constants to fp8 (ptxas) | 0 | 6 | 6 |
| Triton#8 | Incorrect negative masked-load fill values for narrow integers | 0 | 12 | 12 |
| Triton#9 | Half-precision or IEEE fdiv has incompatible intrinsic widths or return types | 18 | 19 | 37 |
| Triton#10 | Gather layout triggers an isWarpLocal assertion | 26 | 37 | 63 |
| Triton#11 | Widened signed libdevice.mul24 returns an incorrect full product | 7 | 4 | 11 |
| Triton#12 | Scan inside a warp-specialized loop reaches an unreachable path | 8 | 0 | 8 |
| Triton#13 | AMD atomic_xchg emits a nonexistent exch intrinsic | 15 | 16 | 31 |
| Triton#14 | bf16 atomic_add loses CAS updates with maxnreg=32 (ptxas) | 0 | 5 | 5 |
| Triton#15 | sm80 fp8e5 software conversion mishandles ties and NaNs | 14 | 19 | 33 |
| Triton#16 | LLVM rejects a gfx1100 DPP instruction; Triton reports an LLD failure | 95 | 120 | 215 |
| Triton#17 | Blackwell warp specialization with unrolling hits a dyn_cast assertion | 19 | 25 | 44 |

“Path” includes the DSL compiler and its dependencies. For example, Triton#1, #7 and #14 involve ptxas; Triton#11 involves libdevice; Triton#16 involves the LLVM AMDGPU backend. These are not all Triton frontend defects.

## 4. Concrete examples, reproduction and observed results

### 4.1 Prerequisites

The small GPU examples ran locally on an **RTX 4060 Laptop, sm89, Python 3.10.20**, with the same TileLang, Triton and PyTorch versions as the servers. The local system nvcc and Triton's bundled ptxas are both **12.9.86**. TileLang#26 was checked separately using the server's existing **nvcc 12.4.131**, through CPU compilation only; no extra GPU tests were run on the servers. The full local environment is recorded in [environment.json](validation/environment.json).

Use an isolated Python environment with the project's [dependency versions](../../../requirements.txt). From the repository root, enter the report directory; all commands below assume this directory:

```bash
cd reports/2026.10.19/latest_experiment_audit
python -c "import importlib.metadata as m; print({p: m.version(p) for p in ('tilelang', 'triton', 'torch', 'apache-tvm-ffi')})"
```

The first five scripts do not import the fuzzer generator or its oracle. They print observations: exit status 0 means that the checking script completed, not that the compiler produced correct results. **The following outcomes were observed in this follow-up's actual tests.**

### 4.2 Triton#1: abs preserves negative signs

The kernel converts 256 negative values to fp16, applies `tl.abs`, and converts the result to integers before storing it. For example, `[-1, -2, -3, -4]` should produce `[1, 2, 3, 4]`; it produces `[-1, 2, -3, 4]`.

```bash
python -B repro/triton_abs.py
```

Observed: **128 of 256 outputs remain negative**. An otherwise equivalent float32 control has **zero mismatches**. Earlier identical-PTX comparisons established the ptxas attribution; this follow-up rechecked the numerical symptom and datatype control, rather than repeating every assembler-version comparison.

Source: [triton_abs.py](repro/triton_abs.py). Output: [triton_abs.py.txt](validation/triton_abs.py.txt).

### 4.3 Triton#8: negative masked-load fills are packed incorrectly

The kernel loads an int8 array using `tl.load(..., mask=r < 48, other=-2)`. Every element in a masked row should be `-2`; the observed row is `[-2, -1, -1, -1, -2, -1, -1, -1]`.

```bash
python -B repro/triton_masked_load.py
```

Observed: **96 of 128 masked elements are wrong**. The int32 control with `other=-2` has zero mismatches, as does int8 with `other=+2`. This matches the failure to repeat the fill bit pattern correctly when packing narrow integers.

Source: [triton_masked_load.py](repro/triton_masked_load.py). Output: [triton_masked_load.py.txt](validation/triton_masked_load.py.txt).

### 4.4 TileLang#22: bf16 pow is rejected as non-floating-point

A small kernel computes `T.pow(X[i], Y[i])` for bfloat16 tensors. Construction fails with:

```text
Check failed: (x.dtype().is_float()) is false: power only applies to float
```

```bash
python -B repro/tilelang_pow_bf16.py
```

Observed: **bf16 pow is rejected**. Both controls, float32 pow and bf16 exp, compile and run successfully. The type check excludes bfloat16 from the accepted floating-point category.

Source: [tilelang_pow_bf16.py](repro/tilelang_pow_bf16.py). Output: [tilelang_pow_bf16.py.txt](validation/tilelang_pow_bf16.py.txt).

### 4.5 TileLang#24: floating-point x−x is folded to constant zero

For input `[+inf, -inf, NaN, 1.5]`, the kernel computes `X[i] - X[i]` without enabling fast math. The expected output is `[NaN, NaN, NaN, 0]`; the actual output is **`[0, 0, 0, 0]`**.

```bash
python -B repro/tilelang_self_subtract.py
```

All three exceptional-value cases are wrong; the finite control is correct. The emitted CUDA directly stores zero and no longer has an input parameter. This locates the error in simplification/lowering rather than numerical variability during GPU execution.

Source and generated CUDA: [tilelang_self_subtract.py](repro/tilelang_self_subtract.py), [output](validation/tilelang_self_subtract.py.txt).

### 4.6 TileLang#26: sm75 bf16 atomic min/max selects the wrong CUDA template branch

TileLang 0.1.15's `atomic.h` retains `cutlass::bfloat16_t` for sm75. It misses the existing 16-bit CAS branch and instantiates generic `cuda::atomic_ref`, which CUDA 12.4 rejects with `atomic_ref does not support 1 or 2 byte types`.

Run in an environment with **CUDA 12.4**, setting `NVCC` to that installation:

```bash
NVCC=/usr/local/cuda/bin/nvcc python -B repro/tilelang_atomic_compile.py
```

The script compiles tiny CUDA kernels through stdin and writes cubins to `/dev/null`. It does not launch a GPU kernel. The corrected control changes only an in-memory header string, leaving installed files intact.

| Condition | Observed result |
| --- | --- |
| Original header, bf16 Min / sm75 | Compilation fails with atomic_ref static assertion |
| Original header, bf16 Max / sm75 | Compilation fails with the same assertion |
| Original header, fp16 Min / sm75 | Compiles |
| Original header, bf16 Min / sm80 | Compiles |
| Include bf16 in the existing CAS branch, Min / sm75 | Compiles |
| Same control, Max / sm75 | Compiles |

This is a toolchain-dependent compatibility issue. Earlier local CUDA 12.9 controls compiled successfully, so failure is not expected under every CUDA version. There are 21 matching records in the current run and 14 in the preceding TileLang 0.1.14 run. It is **newly isolated in this audit, not newly triggered by this campaign or established as a 0.1.15 regression**.

Source: [tilelang_atomic_compile.py](repro/tilelang_atomic_compile.py). All six compilation results: [tl26_cuda124.json](validation/tl26_cuda124.json).

## 5. Fuzzer defects and a false-positive control

| Identifier | Defect | Records |
| --- | --- | ---: |
| Fuzzer#1 | Generates gfx942/gfx950 f32 MFMA + kpack=2 configurations outside supported k_dim layouts | 21 |
| Fuzzer#2 | Enables CUDA-only nan_propagate=True for HIP half/bf16 reductions | 30 |
| Fuzzer#3 | Forces cfg_amd=async on gfx90a/gfx1100 targets that do not support the selected async-copy path | 29 |
| Fuzzer#4 | The floating-point oracle does not adequately model permitted multiply-add contraction | 7 |
| Total | | **87** |

The first three are generator legalization gaps in target/datatype handling in [gemm.py](../../../src/workflow/slices/gemm.py), [reduce.py](../../../src/workflow/slices/reduce.py) and [runtime.py](../../../src/workflow/slices/runtime.py). They are not new compiler mechanisms.

### 5.1 An actual paired reproduction of Fuzzer#4

Case `7c40f5b1a54c5f71` subtracts fp16 products and subsequently applies a scan. The controls retain the original input, failing identity variant, all seed/runtime cases and checker; only the nvcc contraction setting changes:

```bash
python -B repro/fz04_default.py
python -B repro/fz04_no_fma.py
```

The first command **exits 1 as expected**, reporting `actual=0.00103759765625, expected=0.0`. With `--fmad=false`, the second **exits 0 and prints ALL PASSED**. These are larger saved-program controls, not independent minimal examples.

Passing after an option change alone would not establish a false positive. In the earlier audit, six identity-control pairs emitted byte-identical CUDA. Their half-precision multiply/subtract operations use contraction-permitting `__hmul2` / `__hsub2`; CUDA provides `_rn` alternatives to prevent the corresponding contraction. The mismatch is between allowed contraction and the oracle's fixed computation model. [CUDA half2 arithmetic](https://docs.nvidia.com/cuda/archive/12.3.2/cuda-math-api/group__CUDA__MATH____HALF2__ARITHMETIC.html), [nvcc fmad option](https://docs.nvidia.com/cuda/archive/12.9.1/cuda-compiler-driver-nvcc/index.html#fmad-true-false-fmad).

Outputs: [default contraction](validation/fz04_default.py.txt), [contraction disabled](validation/fz04_no_fma.py.txt). Summary of the audit's full set of controls: [fusion_controls.json](data/fusion_controls.json). Only the seven records with adequate paired evidence are assigned to Fuzzer#4; the other numerical discrepancies are not collectively dismissed as false positives.

### 5.2 Corrections to automatic attribution

- `other:3bb67ac4a2` contains both AMD atomic_xchg and IEEE fdiv failures; it is split between Triton#13 and Triton#9.
- An overly broad half-precision atomic rule mixed bf16 min/max compilation errors with atomic_add. The former are now TileLang#26.
- A float32-to-fp8 rule also captured AMD rtz assertions and sm80 fp8e5 software conversion failures. Those are assigned to Triton#5 and Triton#15.
- All five matching bf16 CAS records are assigned to Triton#14.
- Generic missing intrinsic lowering and HIP IEEE lowering are separated as TileLang#11 and TileLang#17. Two pipeline-fragment assertions are merged as TileLang#23.

These are record-level corrections in the report. The original monitoring rules and project source were not changed.

## 6. Unresolved findings and resource-related records

| Type | Records | Treatment |
| --- | ---: | --- |
| oracle_unstable | 247 | The reference is unstable; no numerical bug verdict |
| unsupported_feature | 981 | Explicitly unsupported target, datatype or configuration |
| shared_memory_overflow | 116 | Hardware shared-memory limit exceeded |
| register_limit | 17 | Insufficient register budget |
| warp_partition | 11 | Invalid thread or tile partition |
| timeout | 20 | Slow compilation/execution has not been separated from a true hang |
| gpu_oom | 1 | Insufficient GPU memory |
| Total | **1,393** | |

The unsupported group includes 101 HIP records with a block smaller than a complete wavefront. The 11 invalid warp partitions are separate. These indicate further legalization opportunities, but are already counted here and are not added again to the 87 confirmed fuzzer-defect records.

Another **eight sm90 + nvcc 12.4 segmentation faults** lack independently isolated root causes and remain toolchain candidates. **Nineteen numerical discrepancies** remain unresolved; see [numeric_pending.csv](data/numeric_pending.csv). In the earlier replay, one case became ORACLE UNSTABLE and another exceeded 120 seconds. Its no-FMA control later exhausted the local 8 GiB GPU memory, yielding no valid conclusion about the remote failure.

Disabling fast math does not disable FMA. Direct variant-invariance comparisons also do not automatically receive the structured reference's stability protection. Accumulation order, contraction and error amplification must still be excluded case by case.

## 7. Comparison with the preceding long campaign

The preceding campaign started at 00:45 / 00:47 on October 12 and stopped at 02:02 on October 16: approximately 97 wall-clock hours and 204,285 tests. Its vGPU / Triton `campaign_seconds` was 94.86 hours, while the other three were approximately 97.2 hours; their effective runtimes should not be described as identical.

Applying the corrected classification rules to all 8,225 saved failure JSON records from that campaign gives the following mechanism matches. This comparison did not independently replay every previous record.

| Scope | TileLang path | Triton path | Total |
| --- | ---: | ---: | ---: |
| Previous campaign, final sample | 28 | 18 | **46** |
| Previous campaign, restricted to the current audit's approximately 77.3-hour duration | 28 | 18 | **46** |
| Current fixed audit sample | 26 | 17 | **43** |

The current sample reaches **three fewer mechanisms, a difference of approximately 6.5%**. The campaigns share 41 mechanisms. TileLang#13, the ldmatrix column-offset bug, and Triton#7, the constant-fp8 ptxas bug, appear in the current sample but were not matched in the preceding campaign's saved failures. Five appear only in the preceding campaign: fp16 atomic_add index mapping, vectorized bitwise_not, SSA breakage from loop unswitching, maxnreg-constrained dot illegal memory access, and the AMD fp32 dot + kpack=2 assertion. The two current-only matches had already been confirmed in earlier experiments; they are not first discoveries.

TileLang#26 occurs in both campaigns and is counted on both sides. The historical catalogue's 51 mechanisms combine multiple campaigns and local probes, so 51 is not the count for the preceding campaign alone. TileLang changed from 0.1.14 to 0.1.15, and the generator also changed. The count difference alone does not establish a loss of fuzzer effectiveness or prove that unobserved bugs were fixed.

Supporting statistics: [comparison.json](data/comparison.json).

## 8. Sampling checks and evidence scope

All four runs cover all ten slices. Slice generation accounts for approximately 85% of tests, with fresh, mutate and dsl_extend also active. Non-native targets account for approximately 43.8%–54.3%; native + sm80 accounts for approximately 50.5%–62.7%. The all-default share for 41790 / TileLang is 29.63%, slightly below the original 30% observation threshold, with no evidence of a slice ceasing to be sampled.

There are zero observed cases of the previously addressed non-HIP cfg_amd leakage and zero errors involving obsolete TileLang `k_pack/sync/group` keywords. The oracle_unstable rate is 0.141%–0.191%. Per-run statistics are in [audit_summary.json](data/audit_summary.json).

This work edits the local reports and adds reproduction material. It does not change original project source, remote source or experiment results, and does not stop, restart or deploy campaigns. The only extra remote execution consists of the small CPU compilation controls for TileLang#26.

Published data and validation material:

- [43-mechanism catalogue](data/bug_catalog.json), [6,176 record classifications](data/classification.csv), [19 pending numerical records](data/numeric_pending.csv).
- [Runtime, checkpoints and versions](data/audit_summary.json), [campaign comparison](data/comparison.json).
- [Reproduction guide](repro/README.md), [local run exit codes](validation/local_runs.json), [paired contraction-control exit codes](validation/fusion_runs.json), [six CUDA 12.4 compilation controls](validation/tl26_cuda124.json).

Complete raw remote snapshots, batch replay artifacts and compiler caches remain in the local `evidence/` archive. GitHub contains the compact data, reproduction scripts and test outputs referenced by this report. Case labels and filenames in the test outputs were normalized to the final report numbering; numerical outputs, diagnostics, exit codes and timestamps are unchanged. See [validation/README.md](validation/README.md).
