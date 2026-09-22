"""Shared operation contracts for template selection and semantic validation.

All non-entry values use the program's block_M x block_N fp32 tile type.
Row reductions broadcast back to a tile; transpose is tile-local. Calls have
signature-dependent arity and are checked against the program's function pool.
"""
from dataclasses import dataclass

@dataclass(frozen=True)
class OpContract:
    arity: int
    regions: int = 0
    entry: bool = False

OPS = {name: OpContract(1) for name in (
    'scale', 'neg', 'abs', 'sqrt', 'round', 'copy', 'exp', 'tile_transpose', 'index_add',
    'row_sum', 'row_max', 'row_min', 'row_softmax')}
OPS.update({name: OpContract(2) for name in ('add', 'mul', 'sub', 'maximum', 'minimum', 'div')})
OPS.update({'where': OpContract(3), 'for': OpContract(1, 1), 'if': OpContract(1, 2),
            'load': OpContract(0, entry=True), 'gemm': OpContract(0, entry=True)})
# Transcendental elementwise surface: each name is a compiler code path not
# exercised by the basic arithmetic set. Domain sanitization (clamps) lives in
# the backend op tables and is mirrored identically in the interpreters.
OPS.update({name: OpContract(1) for name in (
    'tanh', 'erf', 'log', 'log2', 'exp2', 'rsqrt', 'sin', 'cos', 'floor', 'ceil')})

# v4 operations. Keep the old registry separate so historical mutation cannot
# accidentally introduce operations unsupported by its saved execution harness.
TYPED_OPS = {
    'cast': OpContract(1), 'reduce_tile': OpContract(1),
    'broadcast_tile': OpContract(1), 'to_tile': OpContract(1),
    'load_input': OpContract(0), 'store_tile': OpContract(1),
    'load_tile': OpContract(1), 'write_tile': OpContract(2),
}
