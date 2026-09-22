"""Triton tile constraints and shared-memory checks."""


from src.backends.common.cuda import dtype_bytes, TRITON_MAX_SHARED



# ── Constraint checks ────────────────────────────────────────────────────────

def check_shared_memory(block_M: int, block_N: int, block_K: int,
                        dtype, num_stages: int = 1) -> bool:
    """
    Check Triton shared memory usage for a GEMM kernel.

    A_tile + B_tile are duplicated by num_stages for software pipelining.
    Accumulator is always float32 and not duplicated per stage.
    """
    elem_size = dtype_bytes(dtype)
    acc_size = 4  # accumulator is always float32
    tiles_per_stage = (block_M * block_K + block_K * block_N) * elem_size
    acc_total = block_M * block_N * acc_size
    shared_bytes = tiles_per_stage * num_stages + acc_total
    return shared_bytes <= TRITON_MAX_SHARED
