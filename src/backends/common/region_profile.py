"""Frozen target restrictions of persisted region v1-v4 programs.

Both built-in backends use this historical domain. Keeping it explicit preserves
old validation/replay; expanding the IR domain is a separate versioned change.
"""


def validate_dimensions(spec):
    if min(spec.M, spec.N, spec.K) < 1 or any(v < 8 or v & (v-1) for v in (spec.block_M, spec.block_N, spec.block_K)):
        raise ValueError('Region requires positive dimensions and power-of-two tiles')


def validate_target(program):
    from src.ir import DataType
    p = program.spec
    if any(o.kind == 'tile_transpose' for o in program.all_operations()) and p.block_M != p.block_N:
        raise ValueError('Tile transpose requires a square tile')
    # int8 is only reachable through the gemm-only int8 generation path; the
    # mutation path never produces it (supported_dtypes stays float-only).
    if p.dtype not in (DataType.FLOAT16, DataType.FLOAT32, DataType.INT8) or p.threads not in (128, 256):
        raise ValueError('Unsupported structured dtype/thread count')
