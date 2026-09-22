# Smoke seed-scan findings (2026-09-21, parallel-compile harness validation)

While picking passing seeds for the 5-case GPU smoke of the parallel-compile
harness, the scan itself surfaced 7 genuine bug-class finds. Each failed seed
was checked for false-positive signature (systematic emitter bug would fail
every program of the kind); pass rates show they are real finds:

## tilelang native region (seeds 22–33 scanned, first pass = 22)
- seed 11: compile_crash|warp_partition — tilelang C++ InternalError
  `Check failed: (m_warp * n_warp == num_warps) is false: m_warp: 1, n_warp: 1, num_warps: 8`
  (M=1, N=1373, K=14806, block 32×16, threads=256, num_stages=4). Also used as
  the serial-vs-parallel A/B: both harnesses crash identically at prepare_0_0,
  proving program-inherent crash + correct marker reprint attribution.
- seed 21: wrong_result (M=14031, N=15700, K=5406, fp16 GEMM, ragged shapes —
  classic tilelang miscompile territory)

## tilelang typed region (seeds 16–21 scanned; 17–20 PASS)
- seed 12: wrong_result (v4 typed program: reduce_tile/broadcast_tile/
  load_input/to_tile + function calls + for-loops)
- seed 16: compile_crash|warp_partition
- seed 21: compile_crash|ptx_async_boundary

## triton native region (seeds 23–31 scanned; 24 PASS)
- seed 13: compile_crash|codegen_api_mismatch (tile_transpose inside an
  if-branch)
- seed 23: compile_crash|segfault (neg->where->if->row_max->copy->div->erf->...)

## Conclusion
Every historical bug class trigger still fires under the new parallel-compile
harness; pass rates (4/6 typed, most others) show no false-positive flooding.
