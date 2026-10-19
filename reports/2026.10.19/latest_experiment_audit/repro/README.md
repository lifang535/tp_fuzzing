# Reproducing the October 19 audit examples

Identifiers are local to this report: **TileLang#1–TileLang#26** and **Triton#1–Triton#17**. See the [Chinese report](../REPORT.md) or [English report](../REPORT_EN.md) for the complete catalogue.

Use TileLang 0.1.15, Triton 3.8.0, PyTorch 2.4.0+cu124 and apache-tvm-ffi 0.1.11 in an isolated environment. The small GPU examples were tested on an RTX 4060 Laptop (sm89), Python 3.10.20, system nvcc 12.9.86 and Triton's bundled ptxas 12.9.86. Actual environment details are in [environment.json](../validation/environment.json). Compiler bugs can depend on architecture and toolchain versions; a different environment may produce different observations.

From the repository root:

```bash
cd reports/2026.10.19/latest_experiment_audit
python -c "import importlib.metadata as m; print({p: m.version(p) for p in ('tilelang', 'triton', 'torch', 'apache-tvm-ffi')})"
```

Run examples individually. They report observations; their process exit code is not a compiler-correctness verdict.

| Mechanism | Command from the report directory | Observed failure | Control |
| --- | --- | --- | --- |
| Triton#1 | `python -B repro/triton_abs.py` | 128/256 fp16 abs outputs remain negative | float32: 0 mismatches |
| Triton#8 | `python -B repro/triton_masked_load.py` | 96/128 int8 masked fills are wrong for other=-2 | int32/-2 and int8/+2: 0 mismatches |
| TileLang#22 | `python -B repro/tilelang_pow_bf16.py` | bf16 pow is rejected by is_float | float32 pow and bf16 exp compile and run |
| TileLang#24 | `python -B repro/tilelang_self_subtract.py` | x-x returns zero for +inf, -inf and NaN | finite input 1.5 returns the expected zero |
| TileLang#26 | `NVCC=/usr/local/cuda/bin/nvcc python -B repro/tilelang_atomic_compile.py` | CUDA 12.4 rejects bf16 atomic Min/Max on sm75 | fp16/sm75, bf16/sm80 and both in-memory CAS controls compile |

The first four require a CUDA GPU. TileLang#26 requires **CUDA 12.4** for the recorded failure and uses compilation only. Set `NVCC` to the actual CUDA 12.4 compiler. It reads TileLang's installed header, compiles kernels through stdin, and sends cubins to `/dev/null`. The corrected header exists only in memory; installed headers and campaign source are not modified. CUDA 12.9 need not reproduce this compilation failure.

The five compiler examples are standalone and do not import TileSmith's generator or oracle.

## Fuzzer#4 paired contraction control

These two files are saved-program controls for case `7c40f5b1a54c5f71`, not minimal kernels. They preserve the failing identity variant, its inputs and checker. The no-FMA file changes the nvcc contraction setting.

```bash
python -B repro/fz04_default.py
python -B repro/fz04_no_fma.py
```

The default command is expected to exit **1**, reporting `actual=0.00103759765625, expected=0.0`. The no-FMA command is expected to exit **0** and print `ALL PASSED`. This is evidence for the specific oracle mismatch described in the reports; it is not a general rule that errors disappearing under a compiler option change are fuzzer bugs.

Captured outputs, elapsed execution times and exit codes are under [validation](../validation/README.md). These scripts use small local tests; they do not start or resume the fuzzing campaigns.
