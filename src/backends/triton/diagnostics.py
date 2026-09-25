"""Compiler diagnostic patterns; ordering is part of historical compatibility.

Branches list every known spelling of a condition (Triton 3.0 and 3.8); a
missing spelling does not error, it falls through to `other`, so
tests/test_classifier.py pins the live wording of every reworded diagnostic.
"""
def classify(message):
    err = message.lower()
    if 'multiple values for argument' in err:
        return 'codegen_duplicate_arg'
    if 'expected dtype' in err and 'but got' in err:
        return 'dtype_unsupported_op'
    if "object has no attribute 'clamp'" in err or 'has no attribute' in err:
        return 'codegen_api_mismatch'
    # A failed MLIR pass (triton/compiler/compiler.py make_ttgir / make_llir
    # raise it after the offending pass) is an upstream compiler-internal
    # failure: distinct from the frontend/lowering errors below, and the only
    # triton class whose stored message carries an MLIR verifier diagnostic
    # (e.g. "'arith.addf' op requires the same encoding for all operands").
    # Must precede the traceback-path rule: this failure's traceback lives in
    # triton/compiler/ too.
    if 'passmanager::run failed' in err:
        return 'triton_pass_failure'
    # Kept verbatim: the dotted 'triton.compiler' is dead in both pairs (paths
    # in a traceback use slashes), but widening it to the path form would
    # swallow harness-side errors raised from inside that package -- most
    # visibly "Signature keys must be string", which is this harness's own
    # defect and must stay visible in `other`.
    if 'compilationerror' in err or 'triton.compiler' in err:
        return 'triton_compile_error'
    if 'out of resources' in err:
        return 'shared_memory_overflow'
