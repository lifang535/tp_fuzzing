"""Regression tests for common root-cause classification ordering.

The 2026.09.17-00.09 run misclassified CUDA OOMs as assertion_failure: the
torch traceback carries the "device-side assertions" hint, which matched the
assertion rule before the gpu_oom rules. These tests lock the corrected order.
"""
import unittest

from src.backends.common.diagnostics import classify_root_cause, classify_with_location

_TORCH_HINT = ("CUDA kernel errors might be asynchronously reported at some other "
               "API call, so the stacktrace below might be incorrect.\nFor debugging "
               "consider passing CUDA_LAUNCH_BLOCKING=1\nCompile with "
               "`TORCH_USE_CUDA_DSA` to enable device-side assertions.")


class ClassifierTests(unittest.TestCase):
    def test_cuda_oom_with_torch_hint_is_gpu_oom(self):
        msg = f"RuntimeError: CUDA error: out of memory\n{_TORCH_HINT}"
        self.assertEqual(classify_root_cause(msg), 'gpu_oom')

    def test_cuda_alloc_failure_is_gpu_oom(self):
        msg = ("RuntimeError: CUDA error: CUBLAS_STATUS_ALLOC_FAILED when calling "
               "`cublasCreate(handle)`\n" + _TORCH_HINT)
        self.assertEqual(classify_root_cause(msg), 'gpu_oom')

    def test_illegal_memory_access_stays_assertion_failure(self):
        """Historical records: kernel OOB poisons the context; the next CUDA
        call raises illegal memory access. Must not become gpu_oom."""
        msg = ("Traceback (most recent call last):\n  File \"/tmp/tilesmith_x.py\", "
               "line 26, in _finite_compare\n    diff = (c[mask] - r[mask]).abs()\n"
               "RuntimeError: CUDA error: an illegal memory access was encountered\n"
               + _TORCH_HINT)
        self.assertEqual(classify_root_cause(msg), 'assertion_failure')

    def test_tilelang_diagnostics_precede_common_rules(self):
        self.assertEqual(
            classify_root_cause("Check failed: (IsValidCPAsyncTransferBytes(total_bytes)) "
                                "is false: tl::ptx_cp_async requires a final PTX byte "
                                "width in {4, 8, 16}, but got 2"),
            'ptx_async_boundary')
        self.assertEqual(
            classify_root_cause("Check failed: m_warp * n_warp == num_warps"),
            'warp_partition')

    def test_tilelang_0114_rewordings_keep_their_classes(self):
        """0.1.14 reworded two failures this harness labels; both messages are
        verbatim from saved 2026.09.24 samples. A lost pattern falls through to
        `other`, which is how a real bug disappears from the root-cause counts."""
        self.assertEqual(
            classify_root_cause(
                "tvm.error.InternalError: No valid warp partition for T.gemm: M=16, N=16 "
                "cannot be evenly covered by 4 warps (policy=Square). Each warp must own a "
                "multiple of 16 rows and 8 columns; adjust `threads` or the block tile shape."),
            'warp_partition')
        conflict = ("tvm.error.InternalError: Layout infer conflict between e11 and e24 in "
                    "T.Parallel loop:\n    loop Fragment((16, 16) -> (8,), replicate: 1, "
                    "thread: 32, forward_thread: _i % 8 * 4 + _j % 8 // 2)")
        self.assertEqual(classify_root_cause(conflict), 'layout_inference')

    def test_triton_pass_failure_is_its_own_class(self):
        """An MLIR pass failure names the failing pass and keeps the verifier
        diagnostic; it is not a frontend/lowering error."""
        message = ("triton/compiler/compiler.py\", line 189, in make_ttgir\n"
                   "    pm.run(mod, 'make_ttgir')\n"
                   "RuntimeError: PassManager::run failed\n"
                   "loc(fused[..]): error: 'arith.addf' op requires the same encoding "
                   "for all operands and results")
        self.assertEqual(classify_root_cause(message), 'triton_pass_failure')

    def test_harness_side_triton_errors_stay_unclassified(self):
        """The triton branch must not claim errors raised from inside the
        triton package: a harness-generated call that triton rejects is a
        harness defect, and labelling it a DSL bug hides it."""
        message = ("File \"/tmp/triton/compiler/compiler.py\", line 69, in __init__\n"
                   "    raise TypeError(\"Signature keys must be string\")\n"
                   "TypeError: Signature keys must be string")
        self.assertEqual(classify_root_cause(message), 'other')
        self.assertEqual(
            classify_root_cause("triton.runtime.errors.OutOfResources: out of resource: "
                                "shared memory, Required: 196608, Hardware limit: 99328"),
            'shared_memory_overflow')

    def test_triton_out_of_resources_is_shared_memory_overflow(self):
        self.assertEqual(
            classify_root_cause("triton.runtime.errors.OutOfResources: out of resource: "
                                "shared memory, Required: 196608, Hardware limit: 99328"),
            'shared_memory_overflow')

    def test_schedule_sweep_invariance_routes_to_own_causes(self):
        """The schedule sweep knobs must not fold into the generic wrong_result."""
        self.assertEqual(
            classify_root_cause('WRONG RESULT: stage invariance: error=0.12, tolerance=0.02'),
            'stage_mismatch')
        self.assertEqual(
            classify_root_cause('WRONG RESULT: loop-kind invariance: error=0.12, tolerance=0.02'),
            'loop_kind_mismatch')
        self.assertEqual(
            classify_root_cause('WRONG RESULT: structured reference error=0.5'),
            'wrong_result')
        self.assertEqual(
            classify_root_cause('WRONG RESULT: schedule invariance: error=0.12, tolerance=0.02'),
            'schedule_mismatch')

    def test_compilation_knob_invariance_routes_to_own_causes(self):
        """Pass-config and swizzle pairs get their own root causes and
        location labels, distinct from the spec-diff sweep knobs."""
        for label, cause in (('pass-config', 'pass_config_mismatch'),
                             ('swizzle', 'swizzle_mismatch')):
            msg = f'WRONG RESULT: {label} invariance: error=0.12, tolerance=0.02'
            self.assertEqual(classify_root_cause(msg), cause)
            self.assertEqual(classify_with_location(msg), (cause, f'{label} invariance'))

    def test_op_surface_check_prefixes_route_to_own_causes(self):
        """fma:/atomic: check labels separate the op-surface bug classes from
        the generic wrong_result."""
        self.assertEqual(
            classify_root_cause('WRONG RESULT: fma:extended_0_0:out:seed=0; max_abs=1.0'),
            'fma_mismatch')
        self.assertEqual(
            classify_root_cause('WRONG RESULT: atomic:scratch:max_abs=0.5'),
            'atomic_mismatch')

    def test_transformed_variants_keep_their_own_labels(self):
        """An fp16-accumulation or identity copy that fails its fma/atomic
        check still classifies as precision/identity; a racy repeat stays
        nondeterminism even though the memory label mentions atomics."""
        self.assertEqual(
            classify_root_cause('WRONG RESULT: precision:fma:extended_0_prec:out; max_abs=1.0'),
            'precision_mismatch')
        self.assertEqual(
            classify_root_cause('WRONG RESULT: identity:atomic:scratch:max_abs=0.5'),
            'algebraic_identity')
        self.assertEqual(
            classify_root_cause('WRONG RESULT: repeat determinism:atomic:scratch'),
            'nondeterminism')

    def test_layout_invariance_routes_to_layout_mismatch(self):
        """Alternate-layout failures get their own root cause; the relabel
        prefix carries the failing pair for the location field."""
        self.assertEqual(
            classify_root_cause('layout invariance: offset/strided: WRONG RESULT: '
                                'structured reference error=0.5'),
            'layout_mismatch')
        self.assertEqual(
            classify_with_location('layout invariance: strided/contiguous: WRONG RESULT: '
                                   'structured reference error=0.5'),
            ('layout_mismatch', 'layout invariance'))

    def test_schedule_sweep_labels_win_over_layout_relabel(self):
        """A cross-variant invariance failure inside an alternate-layout case
        still keeps its historical label: the schedule/stage/loop-kind rules
        precede the layout rule."""
        msg = ('layout invariance: offset/strided: WRONG RESULT: '
               'stage invariance: error=0.12, tolerance=0.02')
        self.assertEqual(classify_root_cause(msg), 'stage_mismatch')
        msg = ('layout invariance: offset/strided: WRONG RESULT: '
               'repeat determinism')
        self.assertEqual(classify_root_cause(msg), 'nondeterminism')

    def test_location_from_invariance_label(self):
        """Location source 1: the invariance label itself, and it beats any
        TILESMITH_STAGE marker printed before the failing check."""
        self.assertEqual(
            classify_with_location('WRONG RESULT: stage invariance: error=0.12, tolerance=0.02'),
            ('stage_mismatch', 'stage invariance'))
        msg = ('TILESMITH_STAGE=execute_variant_1\n'
               'WRONG RESULT: loop-kind invariance: error=0.12, tolerance=0.02\n')
        self.assertEqual(classify_with_location(msg), ('loop_kind_mismatch', 'loop-kind invariance'))
        self.assertEqual(
            classify_with_location('WRONG RESULT: repeat determinism'), ('nondeterminism', 'repeat determinism'))

    def test_location_from_last_tilesmith_stage_marker(self):
        """Location source 2: the last stderr stage marker names the failing
        variant even when the traceback itself carries no location."""
        msg = ('TILESMITH_STAGE=reference\n'
               'TILESMITH_STAGE=prepare_0\n'
               'TILESMITH_STAGE=execute_variant_0\n'
               'TILESMITH_STAGE=execute_variant_2\n'
               f'RuntimeError: CUDA error: out of memory\n{_TORCH_HINT}')
        self.assertEqual(classify_with_location(msg), ('gpu_oom', 'execute_variant_2'))

    def test_location_from_tvm_pass_name(self):
        """Location source 3a: the diagnostic names the TVM pass that failed."""
        msg = ('TVMError: Check failed: (tiles.size()) is false\n'
               'An error occurred inside s_tir.transform.UnrollLoop')
        self.assertEqual(classify_with_location(msg)[1], 's_tir.transform.UnrollLoop')

    def test_location_from_tvm_source_file(self):
        """Location source 3b: a C++ source path narrows the failing pass."""
        msg = ('RuntimeError: Check failed at '
               '/usr/lib/python3/dist-packages/tilelang/src/transform/loop_partition.cc:120')
        self.assertEqual(classify_with_location(msg)[1], 'loop_partition')

    def test_location_empty_for_plain_diagnostics_and_classification_unchanged(self):
        """Legacy messages without markers keep location '' and the exact
        historical root_cause."""
        for msg in ('WRONG RESULT: structured reference error=0.5',
                    f'RuntimeError: CUDA error: out of memory\n{_TORCH_HINT}',
                    'Execution timed out'):
            cause, location = classify_with_location(msg)
            self.assertEqual(location, '')
            self.assertEqual(cause, classify_root_cause(msg))


if __name__ == '__main__':
    unittest.main()
