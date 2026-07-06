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
# Unified interface — delegates to backend-specific params modules
# ============================================================

def generate_valid_params(backend: str, dim_pool: list, dtype,
                          transpose: bool = False) -> dict:
    """Generate hardware-valid parameters for the given backend."""
    from src.workflow.emitter.tilelang import params as tl_params
    from src.workflow.emitter.triton import params as triton_params

    if backend == "triton":
        params = triton_params.generate_valid_params(dim_pool, dtype)
        valid_mn_fn = triton_params.valid_block_sizes
    else:
        params = tl_params.generate_valid_params(dim_pool, dtype)
        valid_mn_fn = tl_params.valid_block_m_n

    # Transpose needs two shared buffers (A + B), so halve the allowed block size
    if transpose:
        elem = dtype_bytes(dtype)
        while params["block_M"] * params["block_N"] * elem * 2 > MAX_SHARED_MEMORY_BYTES:
            smaller = [b for b in sorted(valid_mn_fn()) if b < params["block_N"]]
            if smaller:
                params["block_N"] = smaller[-1]
            else:
                params["block_M"] = max(16, params["block_M"] // 2)
                break

    return params
