"""Preserve historical classification order across compiler diagnostics."""

import re

from src.backends.tilelang.diagnostics import classify as _tilelang
from src.backends.triton.diagnostics import classify as _triton

_INVARIANCE_LABELS = ('repeat determinism', 'schedule invariance', 'stage invariance',
                      'loop-kind invariance', 'pass-config invariance', 'swizzle invariance',
                      'configuration/observation invariance',
                      'layout invariance')


def _failure_location(error_message):
    """Smallest failing unit the harness knows about, from the most precise
    source down: the invariance label itself, the last TILESMITH_STAGE marker
    printed before the crash, a TVM pass or source file named in the
    diagnostic, or nothing."""
    err = error_message.lower()
    for label in _INVARIANCE_LABELS:
        if 'wrong result' in err and label in err:
            return label
    stages = re.findall(r'^TILESMITH_STAGE=(.*)$', error_message, flags=re.MULTILINE)
    if stages:
        return stages[-1]
    passes = re.findall(r'\b(?:tirx|tir|tvm|s_tir)\.(?:transform\.)?[A-Za-z_]\w*', error_message)
    if passes:
        return passes[-1]
    sources = re.findall(r'/([a-z_]+)\.cc:\d+', error_message)
    if sources:
        return sources[-1]
    return ''


def classify_with_location(error_message):
    """(root_cause, location) pair; classification itself is unchanged."""
    return classify_root_cause(error_message), _failure_location(error_message)


def classify_root_cause(error_message):
    err = error_message.lower()
    # The oracle trust gate must precede every wrong_result rule (including the
    # invariance wrappers): a chaotic reference cannot be reproduced by any
    # kernel, so its failure is oracle noise, never a compiler bug.
    if 'oracle unstable' in err:
        return 'oracle_unstable'
    if 'wrong result' in err and 'repeat determinism' in err:
        return 'nondeterminism'
    if 'wrong result' in err and 'schedule invariance' in err:
        return 'schedule_mismatch'
    # Schedule sweep rules must precede the generic wrong_result rule, and the
    # 'stage' rule must precede 'loop-kind': both diagnostics share the
    # 'invariance' wording but never share the leading kind label.
    if 'wrong result' in err and 'stage invariance' in err:
        return 'stage_mismatch'
    if 'wrong result' in err and 'loop-kind invariance' in err:
        return 'loop_kind_mismatch'
    if 'wrong result' in err and 'pass-config invariance' in err:
        return 'pass_config_mismatch'
    if 'wrong result' in err and 'swizzle invariance' in err:
        return 'swizzle_mismatch'
    if 'wrong result' in err and 'warp-policy invariance' in err:
        return 'warp_policy_mismatch'
    if 'wrong result' in err and 'configuration/observation invariance' in err:
        return 'configuration_mismatch'
    # Identity variants (distributivity copies) report with the `identity:`
    # check prefix; their failures are algebraic-identity regressions (RC5).
    if 'wrong result' in err and 'identity:' in err:
        return 'algebraic_identity'
    if 'wrong result' in err and 'precision:' in err:
        return 'precision_mismatch'
    # Op-surface checks report with `fma:` / `atomic:` prefixes. They follow
    # the identity/precision rules so an fp16-accumulation variant that fails
    # its fma check stays a precision_mismatch, and follow the invariance
    # labels so a racy repeat stays nondeterminism.
    if 'wrong result' in err and 'fma:' in err:
        return 'fma_mismatch'
    if 'wrong result' in err and 'atomic:' in err:
        return 'atomic_mismatch'
    if 'wrong result' in err and 'layout invariance' in err:
        return 'layout_mismatch'
    if 'wrong result' in err and 'scratch canary' in err:
        return 'scratch_out_of_bounds'
    if 'wrong result' in err and 'canary' in err:
        return 'output_out_of_bounds'
    if 'wrong result' in err and 'input storage modified' in err:
        return 'input_corruption'
    if 'wrong result' in err:
        return 'wrong_result'
    for classify in (_tilelang, _triton):
        result = classify(error_message)
        if result is not None:
            return result
    if 'segfault' in err or 'segmentation fault' in err or 'signal 11' in err:
        return 'segfault'
    # CUDA OOM rules must precede the assertion rule: an OOM traceback carries
    # the "device-side assertions" hint from torch, which would otherwise be
    # misclassified as assertion_failure (observed in the 2026.09.17-00.09 run).
    if 'cuda error' in err and ('alloc' in err or 'out of memory' in err or 'oom' in err):
        return 'gpu_oom'
    if 'cublas_status_alloc_failed' in err or 'cublascreate' in err:
        return 'gpu_oom'
    if 'out of memory' in err:
        return 'gpu_oom'
    if 'assertion' in err and ('failed' in err or 'error' in err):
        return 'assertion_failure'
    if 'timeout' in err:
        return 'timeout'
    return 'other'
