"""
Triton parameter generation and hardware constraint checking.

All Triton-specific knowledge lives here:
  - Valid block sizes (power-of-2, tl.dot requirements)
  - Shared memory calculation and limit
  - Parameter generation with retry loop
"""

import random

from src.constraints import dtype_bytes, TRITON_MAX_SHARED


# ── Valid candidate lists ────────────────────────────────────────────────────

def valid_block_sizes() -> list:
    """Triton requires tile sizes to be powers of 2."""
    return [16, 32, 64, 128, 256]


def valid_block_k() -> list:
    """Triton tl.dot needs K dim to be power of 2 and >= 16."""
    return [16, 32, 64, 128]


def valid_threads() -> list:
    return [128]


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


# ── Parameter generation ─────────────────────────────────────────────────────

def generate_valid_params(dim_pool: list, dtype) -> dict:
    """Generate valid parameters for Triton. Triton needs power-of-2 tile sizes."""
    _valid_blocks = valid_block_sizes()
    _valid_block_k = valid_block_k()

    for _ in range(100):
        M = random.choice(dim_pool)
        N = random.choice(dim_pool)
        K = random.choice(dim_pool)

        bm_candidates = [b for b in _valid_blocks if b <= max(M, 16)]
        bn_candidates = [b for b in _valid_blocks if b <= max(N, 16)]
        if not bm_candidates:
            bm_candidates = [16]
        if not bn_candidates:
            bn_candidates = [16]

        block_M = random.choice(bm_candidates)
        block_N = random.choice(bn_candidates)

        bk_candidates = [b for b in _valid_block_k if b <= max(K, 16)]
        if not bk_candidates:
            bk_candidates = [16]
        block_K = random.choice(bk_candidates)

        if not check_shared_memory(block_M, block_N, block_K, dtype):
            for bk in sorted(_valid_block_k):
                if bk <= K and check_shared_memory(block_M, block_N, bk, dtype):
                    block_K = bk
                    break
            else:
                continue

        num_stages = random.choice([1, 2, 3, 4])
        if not check_shared_memory(block_M, block_N, block_K, dtype, num_stages):
            for ns in [1, 2]:
                if check_shared_memory(block_M, block_N, block_K, dtype, ns):
                    num_stages = ns
                    break
            else:
                for bk in sorted(_valid_block_k):
                    if bk <= K and check_shared_memory(block_M, block_N, bk, dtype, num_stages):
                        block_K = bk
                        break
                else:
                    num_stages = 1

        threads = 128

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
