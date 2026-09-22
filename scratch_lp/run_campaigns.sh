#!/bin/bash
# Phase 4 gate + Phase 5: dual-backend short campaigns (GPU-serialized).
# tilelang: confirms dtype_mismatch still reachable with current emitter;
# triton: Phase 5 sanity run. seed 42 matches the historical dtype_mismatch
# finding conditions.
set -u
cd /home/lifang535/fdu_lab/project/tile_program_fuzzing/tp_fuzzing || exit 1
echo "=== tilelang campaign $(date -Is) ==="
python3 main.py --backend tilelang --seed 42 -n 100 -o results
echo "tilelang exit=$?"
echo "=== triton campaign $(date -Is) ==="
python3 main.py --backend triton --seed 42 -n 100 -o results
echo "triton exit=$?"
echo "=== done $(date -Is) ==="
