"""
TileLang parameter generation and hardware constraint checking.

All TileLang-specific knowledge lives here:
  - Valid block sizes (MMA instruction requirements)
  - Shared memory calculation and limit
  - Warp partition validity
  - Parameter generation with retry loop
"""

import random

from src.constraints import dtype_bytes, TILELANG_MAX_SHARED, WARP_SIZE, MMA_M, MMA_N


# ── Valid candidate lists ────────────────────────────────────────────────────

def valid_block_m_n() -> list:
    """block_M and block_N must be multiples of 16 (MMA m16n8k16 requires this)."""
    return [16, 32, 64, 128, 256]


def valid_block_k(dtype) -> list:
    """block_K must be a multiple of 8 (stride alignment for tensor core)."""
    return [8, 16, 32, 64, 128]


def valid_threads() -> list:
    """threads must be a multiple of warp_size (32), and num_warps must factor into m_warp * n_warp."""
    return [128, 256]


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


def check_warp_partition(block_M: int, block_N: int, threads: int) -> bool:
    """
    Check that warp partition is valid:
    num_warps = threads / 32 must be factorable into m_warp * n_warp
    where m_warp <= block_M/16 and n_warp <= block_N/8.
    """
    num_warps = threads // WARP_SIZE
    m_tiles = block_M // MMA_M
    n_tiles = block_N // MMA_N
    for m_warp in range(1, num_warps + 1):
        if num_warps % m_warp == 0:
            n_warp = num_warps // m_warp
            if m_warp <= m_tiles and n_warp <= n_tiles:
                return True
    return False


# ── Parameter generation ─────────────────────────────────────────────────────

def generate_valid_params(dim_pool: list, dtype) -> dict:
    """Generate a valid parameter set for TileLang, satisfying all constraints."""
    _valid_block_mn = valid_block_m_n()
    _valid_block_k = valid_block_k(dtype)
    _valid_threads = valid_threads()

    for _ in range(100):
        M = random.choice(dim_pool)
        N = random.choice(dim_pool)
        K = random.choice(dim_pool)

        bm_candidates = [b for b in _valid_block_mn if b <= max(M, 16)]
        bn_candidates = [b for b in _valid_block_mn if b <= max(N, 16)]
        if not bm_candidates:
            bm_candidates = [16]
        if not bn_candidates:
            bn_candidates = [16]

        block_M = random.choice(bm_candidates)
        block_N = random.choice(bn_candidates)

        bk_candidates = [b for b in _valid_block_k if b <= max(K, 8)]
        if not bk_candidates:
            bk_candidates = [8]
        block_K = random.choice(bk_candidates)

        threads = random.choice(_valid_threads)
        if not check_warp_partition(block_M, block_N, threads):
            found = False
            for t in _valid_threads:
                if check_warp_partition(block_M, block_N, t):
                    threads = t
                    found = True
                    break
            if not found:
                continue

        max_stages = 4
        if not check_shared_memory(block_M, block_N, block_K, dtype, num_stages=max_stages):
            for bk in sorted(_valid_block_k):
                if bk <= K and check_shared_memory(block_M, block_N, bk, dtype, num_stages=max_stages):
                    block_K = bk
                    break
            else:
                continue

        return {
            "M": M, "N": N, "K": K,
            "block_M": block_M, "block_N": block_N, "block_K": block_K,
            "threads": threads,
        }

    return {
        "M": 128, "N": 128, "K": 128,
        "block_M": 64, "block_N": 64, "block_K": 32,
        "threads": 128,
    }
