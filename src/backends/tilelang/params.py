"""Tilelang tile constraints and shared-memory checks."""


from src.backends.common.cuda import dtype_bytes, TILELANG_MAX_SHARED, WARP_SIZE, MMA_M, MMA_N


def valid_block_k(dtype) -> list:
    """block_K must be a multiple of 8 (stride alignment for tensor core);
    the int8 s8-mma pipeline additionally requires K >= 32, so int8 GEMMs
    restrict block_K to {32, 64}."""
    val = dtype.value if hasattr(dtype, 'value') else str(dtype)
    return [32, 64] if val == 'int8' else [8, 16, 32, 64, 128]

# ── Constraint checks ────────────────────────────────────────────────────────

def check_shared_memory(block_M: int, block_N: int, block_K: int, dtype, num_stages: int = 2) -> bool:
    """
    Check TileLang shared memory usage including pipeline multi-buffering.
    TileLang allocates (A_shared + B_shared) * num_stages bytes for pipelining.
    """
    elem_size = dtype_bytes(dtype)
    per_stage = (block_M * block_K + block_K * block_N) * elem_size
    shared_bytes = per_stage * num_stages
    return shared_bytes <= TILELANG_MAX_SHARED


def check_warp_partition(block_M: int, block_N: int, threads: int, policy: str = "square") -> bool:
    """
    Check that warp partition is valid:
    num_warps = threads / 32 must be factorable into m_warp * n_warp
    where m_warp <= block_M/16 and n_warp <= block_N/8.

    With a non-square GemmWarpPolicy the partition is fixed by the policy:
    FullRow assigns every warp to rows, needing block_M/16 >= num_warps;
    FullCol assigns every warp to columns, needing block_N/8 >= num_warps.
    """
    num_warps = threads // WARP_SIZE
    m_tiles = block_M // MMA_M
    n_tiles = block_N // MMA_N
    if policy == "full_row":
        return m_tiles >= num_warps
    if policy == "full_col":
        return n_tiles >= num_warps
    for m_warp in range(1, num_warps + 1):
        if num_warps % m_warp == 0:
            n_warp = num_warps // m_warp
            if m_warp <= m_tiles and n_warp <= n_tiles:
                return True
    return False
