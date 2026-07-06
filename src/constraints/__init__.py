from .constraints import (
    generate_valid_params,
    dtype_bytes,
    MAX_SHARED_MEMORY_BYTES,
    TILELANG_MAX_SHARED,
    TRITON_MAX_SHARED,
    HARDWARE_INFO,
    WARP_SIZE,
    MMA_M,
    MMA_N,
)

# Compatibility shims — delegate to backend params modules so existing
# callers (mutator.py) don't need to be updated.
def tilelang_check_shared_memory(block_M, block_N, block_K, dtype, num_stages=2):
    from src.workflow.emitter.tilelang.params import check_shared_memory
    return check_shared_memory(block_M, block_N, block_K, dtype, num_stages)

def tilelang_check_warp_partition(block_M, block_N, threads):
    from src.workflow.emitter.tilelang.params import check_warp_partition
    return check_warp_partition(block_M, block_N, threads)

def tilelang_valid_block_m_n():
    from src.workflow.emitter.tilelang.params import valid_block_m_n
    return valid_block_m_n()

def tilelang_valid_block_k(dtype):
    from src.workflow.emitter.tilelang.params import valid_block_k
    return valid_block_k(dtype)

def triton_valid_block_sizes():
    from src.workflow.emitter.triton.params import valid_block_sizes
    return valid_block_sizes()

def triton_check_shared_memory(block_M, block_N, block_K, dtype, num_stages=1):
    from src.workflow.emitter.triton.params import check_shared_memory
    return check_shared_memory(block_M, block_N, block_K, dtype, num_stages)
