"""Compiler diagnostic patterns; ordering is part of historical compatibility.

The patterns are wording-coupled: TileLang 0.1.14 reworded the failures this
module labels without renaming the conditions, so each branch lists every
known spelling (0.1.11 and 0.1.14). A missing spelling does not error -- the
message falls through to `other` and the bug class silently disappears from
the campaign's root-cause counts, so tests/test_classifier.py pins the live
wording of every reworded diagnostic.
"""
def classify(message):
    err = message.lower()
    # 0.1.11: "Check failed: m_warp * n_warp == num_warps".
    # 0.1.14: "No valid warp partition for T.gemm: M=.., N=.. cannot be evenly
    # covered by N warps (policy=..). Each warp must own a multiple of 16 rows
    # and 8 columns; adjust `threads` or the block tile shape." Raised from
    # tilelang/ir.py GemmWarpPolicyComputeWarpPartition.
    if 'm_warp * n_warp' in err or 'no valid warp partition' in err:
        return 'warp_partition'
    if 'm must be divisible' in err or 'kmperwarp' in err:
        return 'alignment'
    if 'unsupported k_dim' in err:
        return 'unsupported_k_dim'
    if 'stride' in err and 'check failed' in err:
        return 'stride_alignment'
    # 0.1.11: "no available layout found" from the fragment->MMA staging path.
    # 0.1.14: the inferencer instead reports a conflict between two fragments
    # it must reconcile in one T.Parallel loop ("Layout infer conflict between
    # e11 and e24 in T.Parallel loop", raised by tilelang.transform.
    # LayoutInference() inside CUDAPassPipelineBodyPrologue). Same pass, same
    # bug class: a layout the emitter's parallel loops cannot satisfy.
    if 'no available layout' in err or 'layout infer conflict' in err:
        return 'layout_inference'
    if 'shared memory' in err or 'shared_memory' in err or 'out of resource' in err:
        return 'shared_memory_overflow'
    if 'dtype mismatch' in err:
        return 'dtype_mismatch'
    if 'isvalidcpasync' in err or 'ptx_cp_async' in err or 'cp_async' in err:
        return 'ptx_async_boundary'
    if 'internalerror' in err and ('check failed' in err or 'codegen' in err):
        return 'tilelang_codegen_error'
