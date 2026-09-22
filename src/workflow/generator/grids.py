"""Bounded instance grids for new operations.

MLIRSmith exhaustively enumerates small instance domains; this is the targeted
version. Each grid lists every corner of an operation's attribute domain, and
GridState round-robins one cell per (op, backend) so every corner is emitted
exactly once per round instead of being sampled uniformly at random (which can
miss corners for long stretches). Cells are pure attribute dictionaries; the
generators interpret them at their emission sites and fall back to
`random.choice` when no GridState is attached (instance_grid off, or the
mutation path, which never consumes grids).

Grid cursors persist in rng_state.json alongside the random state, so a resumed
campaign continues the exact same cell sequence.
"""
import itertools

ATOMIC_GRID = tuple(
    {'dtype': dtype, 'index': index, 'value': value, 'mask': 'all' if i % 2 else 'half'}
    for i, (dtype, index, value) in enumerate(itertools.product(
        ('int32', 'float32'),
        ('uniform', 'paired', 'unique'),
        ('positive', 'negative', 'mixed', 'zero')))
)

# fp16/fp32 x sign/zero operand triples. A zero entry pins the corner where the
# fused operation degenerates (x=0 -> fma(y,z)=z, y=0 -> fma(x,0,z)=z).
FMA_GRID = tuple(
    {'dtype': dtype, 'x': x, 'y': y, 'z': z}
    for dtype in ('float16', 'float32')
    for x, y, z in (
        (1.0, 0.125, -0.125), (1.0, 1.0, -1.0), (-0.5, 0.25, 0.0),
        (0.0, 0.125, -0.125), (0.125, 0.0, 0.125), (1.0, -1.0, 0.5))
)

# Pre-validated shapes for int8 x int8 matmuls: M,N >= 16 and K >= 32
# (the shared-memory pipelined path requires K >= 32 for the s8 mma pipeline).
INT8_MATMUL_GRID = (
    {'m': 16, 'n': 16, 'k': 32},
    {'m': 16, 'n': 32, 'k': 64},
    {'m': 32, 'n': 16, 'k': 32},
    {'m': 32, 'n': 32, 'k': 64},
)

# Pre-validated full specs for int8 x int8 region GEMMs. Every cell satisfies
# block_K in {32, 64} (the s8 mma shared path requires K >= 32), block_M % 16
# == 0, block_N % 8 == 0, the tilelang warp partition (m_warp * n_warp =
# threads / 32 with m_warp <= block_M/16 and n_warp <= block_N/8) and the
# shared-memory budget for stages 2..3. K is always a multiple of 4 so every
# tl.dot int8 shape is legal; cells 2/3/7/8 carry M/N/K tails for the
# boundary-copy paths.
INT8_SPEC_GRID = (
    {'m': 64, 'n': 64, 'k': 128, 'block_m': 32, 'block_n': 32, 'block_k': 32, 'threads': 128, 'stages': 2},
    {'m': 48, 'n': 32, 'k': 96, 'block_m': 32, 'block_n': 16, 'block_k': 32, 'threads': 128, 'stages': 2},
    {'m': 32, 'n': 48, 'k': 64, 'block_m': 16, 'block_n': 32, 'block_k': 64, 'threads': 128, 'stages': 2},
    {'m': 64, 'n': 64, 'k': 64, 'block_m': 32, 'block_n': 32, 'block_k': 64, 'threads': 256, 'stages': 3},
    {'m': 16, 'n': 64, 'k': 32, 'block_m': 16, 'block_n': 32, 'block_k': 32, 'threads': 128, 'stages': 2},
    {'m': 64, 'n': 16, 'k': 64, 'block_m': 32, 'block_n': 16, 'block_k': 32, 'threads': 128, 'stages': 2},
    {'m': 32, 'n': 32, 'k': 160, 'block_m': 32, 'block_n': 32, 'block_k': 32, 'threads': 128, 'stages': 2},
    {'m': 48, 'n': 48, 'k': 100, 'block_m': 32, 'block_n': 32, 'block_k': 32, 'threads': 128, 'stages': 2},
)

# Per-op source shape patterns. split sources always end in the required
# trailing 2; interleave/join sources keep the doubled minor dimension <= 64
# (the extended TensorType per-dimension bound).
SHAPE_OP_GRID = {
    'flip': ({},),  # flip applies to the existing answer chain in place
    'interleave': ({'shape': (8,)}, {'shape': (16,)}, {'shape': (32,)}),
    'join': ({'shape': (8,)}, {'shape': (16,)}, {'shape': (32,)}),
    'split': ({'shape': (8, 2)}, {'shape': (16, 2)}, {'shape': (32, 2)}),
}


class GridState:
    """Round-robin cursors, one per (op, backend), over the grid tables."""

    def __init__(self):
        self.cursors = {}

    def next_cell(self, op, backend, grid):
        key = (op, backend)
        cursor = self.cursors.get(key, 0)
        self.cursors[key] = cursor + 1
        return grid[cursor % len(grid)]

    def save(self):
        return {f'{op}:{backend}': cursor for (op, backend), cursor in self.cursors.items()}

    def load(self, data):
        for key, cursor in data.items():
            op, backend = key.rsplit(':', 1)
            self.cursors[(op, backend)] = int(cursor)
