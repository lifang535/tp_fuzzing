"""Compiler diagnostic patterns; ordering is part of historical compatibility."""
def classify(message):
    err = message.lower()
    if 'multiple values for argument' in err:
        return 'codegen_duplicate_arg'
    if 'expected dtype' in err and 'but got' in err:
        return 'dtype_unsupported_op'
    if "object has no attribute 'clamp'" in err or 'has no attribute' in err:
        return 'codegen_api_mismatch'
    if 'compilationerror' in err or 'triton.compiler' in err:
        return 'triton_compile_error'
    if 'out of resources' in err:
        return 'shared_memory_overflow'
