"""
Hardware Constraints — Ensures generated parameters are valid for each backend.

Analogous to MLIRSmith's TypedValuePool type-checking: MLIRSmith ensures every
operand has the correct type BEFORE emitting. We ensure every parameter satisfies
hardware constraints BEFORE compiling.

Constraints are derived from:
- TileLang: MMA instruction requirements, shared memory limits, warp partition rules
- Triton: tl.dot shape requirements, shared memory limits, power-of-2 tile sizes

Shared memory limits (for reference):
  sm_80 (A100):           163840 bytes (160 KB)
  sm_86 (RTX 3090):        99328 bytes ( 97 KB)
  sm_89 (RTX 4090/4060L):  99328 bytes ( 97 KB)
  sm_90 (H100):            227328 bytes (222 KB)

Hardware limits are detected at import time via torch.cuda and stored in
HARDWARE_INFO. A 10% safety margin is applied to leave room for driver/runtime
overhead (compiler-generated temporary buffers, etc.).
"""

import random

# DataType is defined in src/ir/ir.py but we can't import it directly
# because ir/pipeline.py imports from constraints → circular.
# Use duck typing: accept any object whose .value is "float16"/"float32",
# or a plain string, or the actual DataType enum.
# dtype_bytes() below handles all three cases.


# ============================================================
# Dynamic hardware detection
# ============================================================

def _detect_hardware() -> dict:
    """
    Detect GPU hardware limits at import time.

    Strategy (in order):
    1. Try pynvml (most accurate, optional dependency)
    2. Use sm_major.minor → known shmem table (covers all common GPUs)
    3. Fall back to 48 KB (safe for all sm_80+ GPUs)

    Returns a dict with:
      max_shared_memory  — usable shmem per block (90% of hardware max)
      max_threads        — max threads per block (always 1024 for sm_80+)
      warp_size          — always 32 for NVIDIA
      compute_capability — e.g. "sm_89"
      device_name        — human-readable GPU name
    """
    # Known max shared memory per block (bytes) by compute capability.
    # Source: NVIDIA CUDA Programming Guide, Appendix H.
    # Format: (major, minor) → bytes
    _SHMEM_BY_SM = {
        (8, 0): 163840,   # sm_80: A100         160 KB
        (8, 6): 99328,    # sm_86: RTX 3090/A40  97 KB
        (8, 7): 99328,    # sm_87: Jetson Orin   97 KB
        (8, 9): 99328,    # sm_89: RTX 4090/4060  97 KB
        (9, 0): 227328,   # sm_90: H100          222 KB
    }
    _DEFAULT_SHMEM = 48 * 1024  # safe fallback for unknown architectures

    fallback = {
        "max_shared_memory": _DEFAULT_SHMEM,
        "max_threads": 1024,
        "warp_size": 32,
        "compute_capability": "unknown",
        "device_name": "unknown (CUDA unavailable)",
    }

    try:
        import torch
        if not torch.cuda.is_available():
            return fallback

        props = torch.cuda.get_device_properties(0)
        sm_key = (props.major, props.minor)
        device_name = props.name

        # Strategy 1: pynvml (gives exact value)
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            raw_shmem = pynvml.nvmlDeviceGetAttribute(
                handle, pynvml.NVML_DEVICE_ATTR_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN
            )
            pynvml.nvmlShutdown()
        except Exception:
            raw_shmem = None

        # Strategy 2: lookup table
        if raw_shmem is None:
            raw_shmem = _SHMEM_BY_SM.get(sm_key, _DEFAULT_SHMEM)

        # Apply safety margin from config (default 50%).
        # Compilers use extra shared memory internally beyond A/B tiles and
        # accumulators (barrier metadata, alignment padding, warp buffers).
        # Config.shmem_safety_fraction controls what fraction of hardware max
        # is safe to use; 0.50 eliminates shared_memory_overflow false positives.
        try:
            from src.config import DEFAULT_CONFIG
            fraction = DEFAULT_CONFIG.shmem_safety_fraction
        except Exception:
            fraction = 0.50
        usable = int(raw_shmem * fraction)

        return {
            "max_shared_memory": usable,
            "max_threads": 1024,
            "warp_size": 32,
            "compute_capability": f"sm_{props.major}{props.minor}",
            "device_name": device_name,
            "raw_shmem_bytes": raw_shmem,
        }

    except Exception:
        return fallback


# Detected at import time — used by all constraint functions below.
HARDWARE_INFO = _detect_hardware()

MAX_SHARED_MEMORY_BYTES = HARDWARE_INFO["max_shared_memory"]
TILELANG_MAX_SHARED     = HARDWARE_INFO["max_shared_memory"]
TRITON_MAX_SHARED       = HARDWARE_INFO["max_shared_memory"]

WARP_SIZE = HARDWARE_INFO["warp_size"]

# MMA instruction tile sizes supported by NVIDIA GPUs (sm_80+)
# m16n8k16 (fp16), m16n8k8 (fp32/tf32)
MMA_M = 16
MMA_N = 8


def dtype_bytes(dtype) -> int:
    """Return bytes per element. Accepts DataType enum, DataType.value string, or plain string."""
    val = dtype.value if hasattr(dtype, "value") else str(dtype)
    return {"float16": 2, "float32": 4, "bfloat16": 2, "int8": 1, "int32": 4}.get(val, 2)


# ============================================================
# TileLang constraints
# ============================================================

def tilelang_valid_block_m_n() -> list:
    """block_M and block_N must be multiples of 16 (MMA m16n8k16 requires this)."""
    return [16, 32, 64, 128, 256]


def tilelang_valid_block_k(dtype) -> list:
    """block_K must be a multiple of 8 (stride alignment for tensor core)."""
    return [8, 16, 32, 64, 128]


def tilelang_valid_threads() -> list:
    """threads must be a multiple of warp_size (32), and num_warps must factor into m_warp * n_warp."""
    return [128, 256]  # 4 or 8 warps — safe for most block_M x block_N


def tilelang_check_shared_memory(block_M: int, block_N: int, block_K: int, dtype, num_stages: int = 2) -> bool:
    """
    Check TileLang shared memory usage including pipeline multi-buffering.
    TileLang allocates (A_shared + B_shared) * num_stages bytes for pipelining.
    Hardware limit: TILELANG_MAX_SHARED (96 KB on sm_89).
    """
    elem_size = dtype_bytes(dtype)
    per_stage = (block_M * block_K + block_K * block_N) * elem_size
    shared_bytes = per_stage * num_stages
    return shared_bytes <= TILELANG_MAX_SHARED


def tilelang_check_warp_partition(block_M: int, block_N: int, threads: int) -> bool:
    """
    Check that warp partition is valid:
    - num_warps = threads / 32
    - Must be able to partition block_M/16 x block_N/16 tiles across num_warps
    - Specifically: (block_M/16) * (block_N/16) >= num_warps or factorizable
    """
    num_warps = threads // WARP_SIZE
    m_tiles = block_M // MMA_M  # How many 16-wide tiles in M
    n_tiles = block_N // MMA_N  # How many 8-wide tiles in N

    # Simple check: num_warps must divide evenly into a m_warp * n_warp grid
    # where m_warp <= m_tiles and n_warp <= n_tiles
    for m_warp in range(1, num_warps + 1):
        if num_warps % m_warp == 0:
            n_warp = num_warps // m_warp
            if m_warp <= m_tiles and n_warp <= n_tiles:
                return True
    return False


def tilelang_generate_valid_params(dim_pool: list, dtype) -> dict:
    """Generate a valid parameter set for TileLang, satisfying all constraints."""
    valid_block_mn = tilelang_valid_block_m_n()
    valid_block_k = tilelang_valid_block_k(dtype)
    valid_threads = tilelang_valid_threads()

    for _ in range(100):  # Retry until we find a valid combination
        M = random.choice(dim_pool)
        N = random.choice(dim_pool)
        K = random.choice(dim_pool)

        # block_M, block_N: must be multiples of 16, must be <= M, N
        bm_candidates = [b for b in valid_block_mn if b <= max(M, 16)]
        bn_candidates = [b for b in valid_block_mn if b <= max(N, 16)]
        if not bm_candidates:
            bm_candidates = [16]
        if not bn_candidates:
            bn_candidates = [16]

        block_M = random.choice(bm_candidates)
        block_N = random.choice(bn_candidates)

        # block_K: must be multiple of 8, must be <= K
        bk_candidates = [b for b in valid_block_k if b <= max(K, 8)]
        if not bk_candidates:
            bk_candidates = [8]
        block_K = random.choice(bk_candidates)

        # threads: must pass warp partition check
        threads = random.choice(valid_threads)
        if not tilelang_check_warp_partition(block_M, block_N, threads):
            # Try other thread counts
            found = False
            for t in valid_threads:
                if tilelang_check_warp_partition(block_M, block_N, t):
                    threads = t
                    found = True
                    break
            if not found:
                continue  # Retry with different block sizes

        # Check shared memory (assume max 4 pipeline stages for worst case)
        max_stages = 4
        if not tilelang_check_shared_memory(block_M, block_N, block_K, dtype, num_stages=max_stages):
            # Reduce block_K until it fits
            for bk in sorted(valid_block_k):
                if bk <= K and tilelang_check_shared_memory(block_M, block_N, bk, dtype, num_stages=max_stages):
                    block_K = bk
                    break
            else:
                continue  # Retry

        return {
            "M": M, "N": N, "K": K,
            "block_M": block_M, "block_N": block_N, "block_K": block_K,
            "threads": threads,
        }

    # Fallback: guaranteed valid config
    return {
        "M": 128, "N": 128, "K": 128,
        "block_M": 64, "block_N": 64, "block_K": 32,
        "threads": 128,
    }


# ============================================================
# Triton constraints
# ============================================================

def triton_valid_block_sizes() -> list:
    """Triton requires tile sizes to be powers of 2."""
    return [16, 32, 64, 128, 256]


def triton_check_shared_memory(block_M: int, block_N: int, block_K: int,
                               dtype, num_stages: int = 1) -> bool:
    """
    Check Triton shared memory usage for a GEMM kernel.

    Triton GEMM requires:
      - A_tile: block_M * block_K * dtype_bytes  (input dtype)
      - B_tile: block_K * block_N * dtype_bytes  (input dtype)
      - acc:    block_M * block_N * 4            (always float32, not duplicated per stage)
    A_tile + B_tile are duplicated by num_stages for software pipelining.

    Hardware limit: TRITON_MAX_SHARED (87 KB on sm_89, 10% safety margin applied).
    """
    elem_size = dtype_bytes(dtype)
    acc_size = 4  # accumulator is always float32
    tiles_per_stage = (block_M * block_K + block_K * block_N) * elem_size
    acc_total = block_M * block_N * acc_size
    shared_bytes = tiles_per_stage * num_stages + acc_total
    return shared_bytes <= TRITON_MAX_SHARED


def triton_generate_valid_params(dim_pool: list, dtype) -> dict:
    """Generate valid parameters for Triton. Triton needs power-of-2 tile sizes."""
    valid_blocks = triton_valid_block_sizes()
    valid_block_k = [16, 32, 64, 128]  # Triton tl.dot needs K dim to be power of 2 and >= 16

    for _ in range(100):
        M = random.choice(dim_pool)
        N = random.choice(dim_pool)
        K = random.choice(dim_pool)

        bm_candidates = [b for b in valid_blocks if b <= max(M, 16)]
        bn_candidates = [b for b in valid_blocks if b <= max(N, 16)]
        if not bm_candidates:
            bm_candidates = [16]
        if not bn_candidates:
            bn_candidates = [16]

        block_M = random.choice(bm_candidates)
        block_N = random.choice(bn_candidates)

        bk_candidates = [b for b in valid_block_k if b <= max(K, 16)]
        if not bk_candidates:
            bk_candidates = [16]
        block_K = random.choice(bk_candidates)

        # Check shared memory
        if not triton_check_shared_memory(block_M, block_N, block_K, dtype):
            for bk in sorted(valid_block_k):
                if bk <= K and triton_check_shared_memory(block_M, block_N, bk, dtype):
                    block_K = bk
                    break
            else:
                continue

        # Triton num_stages — must also fit in shared memory
        num_stages = random.choice([1, 2, 3, 4])
        # Re-check with actual num_stages (multi-stage doubles/triples shmem)
        if not triton_check_shared_memory(block_M, block_N, block_K, dtype, num_stages):
            # Try to reduce num_stages first, then block_K
            for ns in [1, 2]:
                if triton_check_shared_memory(block_M, block_N, block_K, dtype, ns):
                    num_stages = ns
                    break
            else:
                for bk in sorted(valid_block_k):
                    if bk <= K and triton_check_shared_memory(block_M, block_N, bk, dtype, num_stages):
                        block_K = bk
                        break
                else:
                    num_stages = 1  # fallback: single stage always uses least shmem

        threads = 128  # Triton typically uses 128 threads (4 warps)

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


# ============================================================
# Unified interface
# ============================================================

def generate_valid_params(backend: str, dim_pool: list, dtype,
                          transpose: bool = False) -> dict:
    """Generate hardware-valid parameters for the given backend."""
    if backend == "tilelang":
        params = tilelang_generate_valid_params(dim_pool, dtype)
    elif backend == "triton":
        params = triton_generate_valid_params(dim_pool, dtype)
    else:
        params = tilelang_generate_valid_params(dim_pool, dtype)

    # Transpose needs two shared buffers (A + B), so halve the allowed block size
    if transpose:
        elem = dtype_bytes(dtype)
        while params["block_M"] * params["block_N"] * elem * 2 > MAX_SHARED_MEMORY_BYTES:
            valid_mn = tilelang_valid_block_m_n() if backend == "tilelang" else triton_valid_block_sizes()
            smaller = [b for b in sorted(valid_mn) if b < params["block_N"]]
            if smaller:
                params["block_N"] = smaller[-1]
            else:
                params["block_M"] = max(16, params["block_M"] // 2)
                break

    return params
