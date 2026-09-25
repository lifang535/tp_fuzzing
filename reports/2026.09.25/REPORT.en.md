# 2026-09-25 Audit: real bugs under the newest tilelang / triton

Audited: four campaigns, `2026.09.24-00.45` and `2026.09.24-00.46`, one tilelang and one triton each.

Machines (both ran the same seed and flags, hence the same generated program stream):

| Machine | GPU | Architecture (sm, CUDA compute capability) |
|---|---|---|
| A | NVIDIA GeForce RTX 4090, 24564 MiB | sm_89 (Ada, cc 8.9) |
| B | NVIDIA vGPU-32GB, 32760 MiB | sm_89 (Ada, cc 8.9) |

Environment: tilelang **0.1.14**, triton **3.8.0**, torch 2.4.0+cu124.
Identical command line for all four: `main.py --backend <b> --seed 42 --function-min-count 1 --function-max-count 2 --function-call-prob 0.05 -n 100000 --no-save-artifacts`.

## tilelang

| Label | Symptom (trigger → basis for the verdict) | Real bug | Count |
|---|---|---:|--:|
| `tilelang_codegen_error` | Compile-time failure, no runnable kernel: internal CUDA codegen / lowering error. The message carries either `Cannot convert type boolxN to CUDA type` (a vectorized bool cannot be printed as a CUDA type) or `ReduceOp cannot lower a layout where a source index depends on a thread-owned reduce segment` (a reduce layout-lowering assertion). Hit by both region and extended programs | ✅ yes | 415 |
| `ptx_async_boundary` | Compile-time failure: `T.ptx_cp_async` reaches PTX with a transfer byte count outside {4, 8, 16}, so the `IsValidCPAsyncTransferBytes(total_bytes)` assertion fails. Triggered by non-divisible vectorized copies (int8 / small tail blocks) | ✅ yes | 84 |
| `wrong_result` | Compiles and runs, but the output disagrees with the fp64 reference by far more than the tolerance (not ULP-level noise) → the result is computed wrong | ✅ yes | 75 |
| `layout_inference` | Compile-time failure: the layout inference pass reports `no available layout found` — it cannot assign a layout to some fragment (`has_best` assertion) | ✅ yes (candidate) | 32 |
| `schedule_mismatch` | Numerical-invariance violation: the same kernel with only the thread partition changed (threads sweep, which does not change the math) is run twice, and results that must be bit-identical differ | ✅ yes | 19 |
| `layout_mismatch` | Numerical-invariance violation: the same kernel under a different input layout (contiguous / offset; the strides in the emitted source change) must produce identical results but does not → indexing bug | ✅ yes | 12 |
| `pass_config_mismatch` | Numerical-invariance violation: the same kernel source, recompiled with only `@tilelang.jit(pass_configs=...)` changed (numerically neutral passes such as `tl.enable_async_copy`, `tl.disable_shared_memory_reuse`), must produce identical results but does not → some pass changed the semantics | ✅ yes | 1 |
| import `tilelang_callback_cuda_compile` fails | Harness defect: the generated reproducer script imports an internal symbol that 0.1.14 removed, so the script cannot even start | ❌ no | 878 |
| `shared_memory_overflow` | The generated program requests more dynamic shared memory than the device allows (e.g. 262144 B against a 101376 B/SM hardware limit); the generator applies no bound | ❌ no | 154 |
| `No valid warp partition for T.gemm ... M=16, N=32 ... 8 warps` | Generator-side hole: with `block_M=16` no warp partition is feasible (the generator's own `check_warp_partition` can tell, but the hard-shape path never calls it) | ❌ no | 2 |
| `timeout` / `gpu_oom` | Resource limits: compile or execution timeout; VRAM held by other processes on the same host | ❌ no | 6 / 8 |
| `oracle_unstable` | Trust gate: the fp64 reference disagrees with its own / jittered copy by >1e-2 relative (chaotic recurrence), so the numeric check is skipped by design | ❌ no | 175 |
| `other` (region side, unclassified) | No specific error attributed | ❓ unknown | 54 |

## triton

| Label | Symptom (trigger → basis for the verdict) | Real bug | Count |
|---|---|---:|--:|
| `wrong_result` | Compiles and runs, but the output disagrees with the fp64 reference by far more than the tolerance | ✅ yes | 234 |
| `pass_config_mismatch` | Numerical-invariance violation: the same kernel recompiled with a different compilation configuration (the `enable_fp_fusion` switch, randomly sampled pass combinations) must produce identical results but does not. The largest error of any class here: `67108864.0` against a `0.001` tolerance | ✅ yes | 20 |
| `layout_mismatch` | Numerical-invariance violation: the same kernel under a different input layout (contiguous / offset) must produce identical results but does not → indexing bug | ✅ yes | 20 |
| `schedule_mismatch` | Numerical-invariance violation: the same kernel with only `num_warps` / `num_stages` changed must produce bit-identical results but does not | ✅ yes | 7 |
| `ASTSource` signature keyed by int | Harness defect: the `ASTSource` signature is built with integer keys such as `0:'*fp16'`, while triton 3.8 requires parameter-name strings. It raises `Signature keys must be string` outright, so not one extended program ever reaches compilation | ❌ no | 3591 |
| `shared_memory_overflow` | The generated program requests more shared memory than the device allows (`OutOfResources: Required: 131072–262144, Hardware limit: 101376`); the generator applies no bound | ❌ no | 172 |
| `timeout` / `gpu_oom` | Resource limits | ❌ no | 6 / 7 |
| `oracle_unstable` | Trust gate skips the numeric check by design | ❌ no | 648 |
| `other` (region side, unclassified) | No specific error attributed | ❓ unknown | 13 |

## Examples of the real bugs

**tilelang `tilelang_codegen_error`** — two signatures, on region and extended programs alike:

```
Cannot convert type boolx16 to CUDA type
Cannot convert type boolx8 to CUDA type
Check failed: (analyzer->CanProveEqual(projected_index, simplified_index)) is false:
  ReduceOp cannot lower a layout where a source index depends on a thread-owned reduce segment
```

The first two are the CUDA printer refusing a vectorized bool; the third is an internal assertion in the reduce layout lowering. Representative programs:
`M=163 N=4673 K=1 block 256x128x128 threads=128 stages=4 fp16`, `M=4968 N=122 K=1240 block 16x256x32 threads=256 stages=1 fp16`.

**tilelang `ptx_async_boundary`** — the cp.async transfer width reaching PTX falls outside {4, 8, 16}:

```
Check failed: (IsValidCPAsyncTransferBytes(total_bytes)) is false:
T.ptx_cp_async(T.address_of(As[T.shift_right(thread_binding, 3) * 64 + ...
```

Representative program: `M=1 N=9282 K=10937 block 16x64x64 threads=128 stages=4 fp16`.

**Numerical-invariance violations on both backends** — the same program under a different schedule / pass config / layout disagrees with its reference or its variant. The oracle gate (fp64 plus a 1-ulp jitter self-comparison) has already ruled out an unstable reference:

| Class | Largest error / tolerance |
|---|---|
| triton `pass_config_mismatch` | `67108864.0 / 0.001` (2^26), `841.6 / 0.1`, `6.89 / 0.001` |
| tilelang `schedule_mismatch` | `16.98 / 0.1`, `5.17 / 0.1`, `1.64 / 0.1` |
| tilelang `layout_mismatch` | `4.0 / 0.001`, `1.0 / 0.001`, `0.5 / 0.001` |
| triton `layout_mismatch` | `8.0 / 0.001`, `1.0 / 0.001` |
| triton `wrong_result` | `2.79 / 0.05`, `1.80 / 0.05` |
| tilelang `wrong_result` | `5.12 / 0.1`, `1.32 / 0.05` |

Tolerance scale: `elemwise_atol=1e-3` ≈ 1 ULP of fp16 in [1,2), `region_rtol_fp16=0.10` / `fp32=0.05`. Samples such as `error=0.001953125` (2 ULP) and `0.00390625` (4 ULP) sit at the tolerance edge and are counted as noise, not as bugs.

**A harness defect takes down the whole extended track** — not one extended program passed in these runs. On the triton side 100% die at the integer-keyed signature in `src/backends/triton/extended.py:194`; on the tilelang side they die on the function-body import of `tilelang_callback_cuda_compile` in `src/backends/tilelang/extended.py:328`, a symbol 0.1.14 removed. A further 146/67 tilelang extended programs exit during lowering with **genuine compile errors** (before that import line runs) and are included in the tilelang counts above.

## Fix: adaptation to the new releases (branch `tilelang-0.1.14-triton-3.8`, commit `2fe7c95b`)

Both harness defects in the tables above — the tilelang callback import and the triton integer-keyed signature — are fixed on this branch, with measurements.

**Root cause, now pinned down further than at audit time**

- **tilelang**: 0.1.14 moved `tilelang_callback_cuda_compile` from `tilelang.engine.lower` to `tilelang.cuda.backend` (same name and signature). In the emitted program that import sits **after the per-variant compile loop and before `make_launch`**, so what it discards is programs whose variants did compile and would have run; a program that raises inside `tilelang.compile` is classified at `device_compile:<variant>` and never reaches the line. The emitted program now does `try: from tilelang.cuda.backend ... except ImportError: from tilelang.engine.lower ...`, so older releases keep taking the fallback.
- **triton**: 3.8 requires the `ASTSource` signature to be keyed by **parameter-name strings** (`triton/compiler/compiler.py:69`, `Signature keys must be string`); integer keys raise TypeError. The signature is now keyed by the generated program's real parameter names, which 3.0 accepts as well.

**Before the fix** (both servers, `tp_fuzzing_latest`, started 2026-09-24 23:39 and still running; counted from the `.json` of each saved case on disk, a different basis from the "occurrences" of the tables above)

| Machine | Backend | Passed | extended passed | extended failed |
|---|---|---:|---:|---|
| A | tilelang | 2441 | **0** | 458 = 370 `other` + 88 codegen |
| A | triton | 8537 | **0** | 1499 (all TypeError) |
| B | tilelang | 3850 | **0** | 745 = 611 `other` + 134 codegen |
| B | triton | 13684 | **0** | 2324 (all TypeError) |

369 of 370 of A's `other` cases (610 of 611 on B) are that ImportError. The codegen part (88 / 134, e.g. the `boolx16` PrintType crash) is a **genuine bug** that dies before the import — those still fail after the fix and are not counted as a gain.

**After the fix** (machine A, target pair, 30 iterations, same flags as the live campaigns)

- The eight extended fixture variants pass 8/8 on both machines;
- 30 iterations: 4 extended programs pass on tilelang and 10 on triton, with 0 extended failures on either (plus 1 `oracle_unstable`, non-extended) — against zero on both tracks of the live campaigns;
- The same harness passes all 6 fixtures on the legacy pair (0.1.11 / 3.0.0), so the adaptation does not break the old releases.

**Harness defects fixed along the way**

- **Probe bit comparison**: a degenerate-strided reference could not be compared at all — torch's contiguity check skips size-1 dimensions, so `.contiguous()` left `stride(-1) == 0` and `view(uint8)` raised `stride(-1) must be 1 to view Float as Byte`. Five live cases were misevaluated this way (1 tilelang, 4 triton). The tensor is now materialized only when the byte view would be illegal.
- **Diagnostics wording**: tilelang gains the 0.1.14 layout-inference and warp-partition rewordings, triton gains a class for `PassManager::run failed`. The dotted `'triton.compiler'` pattern is deliberately left alone, since widening it to the source-path form would relabel harness-side errors — such as that signature TypeError — out of `other`.
- **Version marking**: `src/backends/common/versions.py` plus the startup banner and `environment` / `target_versions` in `summary.json`; a resume under a different environment now warns.
- **Both re-verification candidates stay excluded after measurement**: `tl.config_index_bitwidth` is still a universal breaker on 0.1.14 (MakePackedAPI: `impl variables (limit,) are used, but are not passed in as API arguments`, `make_packed_api.cc:1060`; `:577` on 0.1.11), and bfloat16 is not a pool edit at all (the `DataType` in `ir/ir.py`, `DTYPES` in `ir/extended.py` and the triton signature map all reject it) — it needs the IR whitelists, the emitters and the tolerances extended first.

Tests: 303 OK on the branch against 282 OK on main, with 21 new tests. The four fixture cases whose emission changed on purpose have their digests updated after diffing every case; the other 23 emissions are byte-identical.
