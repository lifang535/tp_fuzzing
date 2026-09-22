"""Compiler diagnostic patterns; ordering is part of historical compatibility."""
def classify(message):
    err = message.lower()
    if 'm_warp * n_warp' in err:
        return 'warp_partition'
    if 'm must be divisible' in err or 'kmperwarp' in err:
        return 'alignment'
    if 'unsupported k_dim' in err:
        return 'unsupported_k_dim'
    if 'stride' in err and 'check failed' in err:
        return 'stride_alignment'
    if 'no available layout' in err:
        return 'layout_inference'
    if 'shared memory' in err or 'shared_memory' in err or 'out of resource' in err:
        return 'shared_memory_overflow'
    if 'dtype mismatch' in err:
        return 'dtype_mismatch'
    if 'isvalidcpasync' in err or 'ptx_cp_async' in err or 'cp_async' in err:
        return 'ptx_async_boundary'
    if 'internalerror' in err and ('check failed' in err or 'codegen' in err):
        return 'tilelang_codegen_error'
