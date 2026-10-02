"""Region oracle: elementwise mixed tolerance, admissible TF32 GEMM semantics
and the elementwise explanation of a mismatch by perturbed references.

CPU-only. fp32 GEMMs run on TF32 tensor cores (tl.dot's default input
precision, T.gemm), which drop or round the 13 low mantissa bits of each
operand, so a strict fp32 reference reports a wrong_result for any program
that amplifies the difference. Those readings are admissible references;
everything outside them still fails.
"""
import ast
import math
import unittest
from dataclasses import asdict

import torch

from src.config import Config
from src.ir import TileKernel, DataType, ComputeKind
from src.ir.region import Operation as Op, Region, RegionProgram, RegionExecution
from src.workflow.emitter.region_checks import (_reference_bounds, _reference_explained, _region_check,
                                                _run_region)
from src.workflow.emitter.region_runtime import _region_reference
from src.workflow.emitter.runtime import _finite_compare
from src.workflow.emitter.typed_region_runtime import _typed_region_reference
from src.workflow.oracle import Oracle


def gemm_program(dtype='float32', typed=False):
    spec = TileKernel('kernel_0', M=16, N=16, K=16, block_M=16, block_N=16, block_K=16,
                      threads=128, num_stages=2, dtype=DataType(dtype),
                      compute_kind=ComputeKind.GEMM)
    operations = [Op('gemm', 'matmul')]
    if typed:
        operations.append(Op('cast', 'c1', ['matmul'], {'dtype': 'float32'}))
    operations.append(Op('copy', 'v_out', [operations[-1].result]))
    p = RegionProgram(spec, Region([], operations, 'v_out'), typed=typed)
    p.execution = RegionExecution(input_pattern='normal', input_seed_count=1, repeat_count=1)
    p.validate()
    return p


class MixedToleranceTests(unittest.TestCase):
    def test_rtol_absorbs_one_ulp_at_large_magnitude(self):
        reference = torch.tensor([16384.0, 1.0])
        actual = torch.tensor([16384.0 + 2 ** -7, 1.0])  # 4 fp32 ulps above
        maximum, _, _ = _finite_compare(actual, reference)
        self.assertEqual(maximum, 2 ** -7)
        # Errors inside the relative band count as zero.
        maximum, _, _ = _finite_compare(actual, reference, rtol=1e-5)
        self.assertEqual(maximum, 0.0)
        with self.assertRaisesRegex(RuntimeError, 'WRONG RESULT: structured reference'):
            _region_check(actual, reference, False, 1e-3, 'structured reference')
        _region_check(actual, reference, False, 1e-3, 'structured reference', rtol=1e-5)
        # The slack scales with each element's own magnitude: the same
        # absolute error at 1.0 still fails.
        with self.assertRaisesRegex(RuntimeError, 'WRONG RESULT'):
            _region_check(torch.tensor([16384.0, 1.0 + 2 ** -7]), reference, False, 1e-3, 'x', rtol=1e-5)

    def test_rtol_never_relaxes_exceptional_values(self):
        reference = torch.tensor([1.0, float('inf')])
        for actual in (torch.tensor([float('nan'), float('inf')]), torch.tensor([1.0, float('-inf')])):
            with self.assertRaisesRegex(RuntimeError, 'WRONG RESULT'):
                _region_check(actual, reference, False, 1e-3, 'x', rtol=1.0)


class Tf32ReferenceTests(unittest.TestCase):
    one = 1.0
    ulp = 2.0 ** -10  # TF32 keeps 10 explicit mantissa bits

    def convert(self, values, mode, typed):
        # One value per row times the identity reads each converted operand
        # back (1.0 and 0.0 are exact in TF32); column 0 never multiplies an
        # infinity by zero.
        p = gemm_program(typed=typed)
        a = torch.zeros(16, 16)
        a[:len(values), 0] = torch.tensor(values, dtype=torch.float32)
        interpreter = _typed_region_reference if typed else _region_reference
        return interpreter(a, torch.eye(16), asdict(p.body), 16, 16, 'float32', tf32=mode)[:len(values), 0].tolist()

    def test_truncation_and_round_to_nearest_even(self):
        u = self.ulp
        values = [1 + u / 2 + 2 ** -20,  # above the half step
                  1 + u / 2,              # tie, even neighbour below
                  1 + u + u / 2,          # tie, odd: rounds up
                  -(1 + u / 2 + 2 ** -20),
                  1 + u - 2 ** -23]       # largest fp32 below the next TF32 value
        for typed in (False, True):
            with self.subTest(typed=typed):
                self.assertEqual(self.convert(values, None, typed), torch.tensor(values).tolist())
                self.assertEqual(self.convert(values, 'truncate', typed), [1, 1, 1 + u, -1, 1])
                self.assertEqual(self.convert(values, 'nearest', typed), [1 + u, 1, 1 + 2 * u, -(1 + u), 1 + u])

    def test_exceptional_operands_pass_through(self):
        big = torch.finfo(torch.float32).max
        for typed in (False, True):
            for mode in ('truncate', 'nearest'):
                with self.subTest(typed=typed, mode=mode):
                    out = self.convert([float('inf'), float('-inf'), 0.0, float('nan')], mode, typed)
                    self.assertEqual(out[:3], [float('inf'), float('-inf'), 0.0])
                    self.assertTrue(math.isnan(out[3]))
            # Rounding the largest finite value to nearest overflows like any
            # IEEE rounding; truncation keeps it finite.
            self.assertEqual(self.convert([big], 'truncate', typed), [(2 - 2 ** -10) * 2.0 ** 127])
            self.assertEqual(self.convert([big], 'nearest', typed), [float('inf')])


def floor_cos_program(typed=False):
    """floor(cos(x)): exact except where cos(x) is within its approximation
    error of an integer, e.g. x = pi/2, where it decides between -1 and 0."""
    spec = TileKernel('kernel_0', M=16, N=16, K=16, block_M=16, block_N=16, block_K=16,
                      threads=128, num_stages=2, dtype=DataType('float32'),
                      compute_kind=ComputeKind.COPY)
    body = Region([], [Op('load', 'entry'), Op('cos', 'c', ['entry']), Op('floor', 'value', ['c'])], 'value')
    p = RegionProgram(spec, body, typed=typed)
    p.execution = RegionExecution(input_pattern='normal', input_seed_count=1, repeat_count=1)
    p.validate()
    return p


class NudgeTests(unittest.TestCase):
    def setUp(self):
        self.a = torch.full((16, 16), 0.5)
        self.a[0, 0] = math.pi / 2  # cos is -4.4e-8 in fp32

    def evaluate(self, typed, perturb):
        interpreter = _typed_region_reference if typed else _region_reference
        p = floor_cos_program(typed)
        return interpreter(self.a, torch.empty(16, 16), asdict(p.body), 16, 16, 'float32', perturb=perturb)

    def test_patterns_move_inexact_results_and_propagate_through_exact_ones(self):
        for typed in (False, True):
            with self.subTest(typed=typed):
                exact = self.evaluate(typed, None)
                self.assertEqual(exact[0, 0].item(), -1.0)
                self.assertTrue(torch.equal(exact[1:], torch.zeros(15, 16)))
                # Moving cos up by its error bound flips the floor at pi/2
                # only; down keeps it.
                self.assertEqual(self.evaluate(typed, 0)[0, 0].item(), 0.0)
                self.assertTrue(torch.equal(self.evaluate(typed, 1), exact))
                for pattern in range(8):
                    self.assertTrue(torch.equal(self.evaluate(typed, pattern)[1:], exact[1:]))
                # Sign patterns are seeded: the same pattern repeats exactly.
                self.assertTrue(torch.equal(self.evaluate(typed, 5), self.evaluate(typed, 5)))

    def test_moves_are_bounded_by_the_operation_error(self):
        p = RegionProgram(
            TileKernel('kernel_0', M=16, N=16, K=16, block_M=16, block_N=16, block_K=16, threads=128,
                       num_stages=2, dtype=DataType('float32'), compute_kind=ComputeKind.COPY),
            Region([], [Op('load', 'entry'), Op('exp', 'value', ['entry'])], 'value'))
        p.validate()
        x = torch.linspace(-3, 3, 256).reshape(16, 16)
        exact = _region_reference(x, x, asdict(p.body), 16, 16, 'float32')
        for pattern in (0, 1, 2):
            moved = _region_reference(x, x, asdict(p.body), 16, 16, 'float32', perturb=pattern)
            relative = ((moved - exact).abs() / exact.abs()).max().item()
            self.assertGreater(relative, 0)
            self.assertLessEqual(relative, 17 * 2.0 ** -23)

    def test_half_precision_rounding_absorbs_moves_away_from_its_boundaries(self):
        """Backends evaluate typed operations in fp32 and round to the SSA
        dtype: floor(x / x) flips under an approximate fp32 division, but an
        fp16 quotient rounds back to 1 first."""
        def program(half):
            spec = TileKernel('kernel_0', M=16, N=16, K=16, block_M=16, block_N=16, block_K=16,
                              threads=128, num_stages=2, dtype=DataType('float32'),
                              compute_kind=ComputeKind.COPY)
            ops = [Op('load', 'entry')]
            if half:
                ops.append(Op('cast', 'x', ['entry'], {'dtype': 'float16'}))
            source = ops[-1].result
            ops += [Op('div', 'q', [source, source]), Op('floor', 'f', ['q'])]
            if half:
                ops.append(Op('cast', 'value', ['f'], {'dtype': 'float32'}))
            p = RegionProgram(spec, Region([], ops, ops[-1].result), typed=True)
            p.execution = RegionExecution(input_pattern='normal', input_seed_count=1, repeat_count=1)
            p.validate()
            return asdict(p.body)
        a = torch.full((16, 16), 0.5)
        for half, flipped in ((False, 0.0), (True, 1.0)):
            with self.subTest(half=half):
                body = program(half)
                exact = _typed_region_reference(a, a, body, 16, 16, 'float32')
                self.assertTrue(torch.equal(exact, torch.ones(16, 16)))
                self.assertTrue(torch.equal(_typed_region_reference(a, a, body, 16, 16, 'float32', perturb=0), exact))
                down = _typed_region_reference(a, a, body, 16, 16, 'float32', perturb=1)
                self.assertTrue(torch.equal(down, torch.full((16, 16), flipped)))

    def test_moves_stay_finite_near_the_fp32_maximum(self):
        """ldexp builds its power of two in fp32: an unsplit shift by the
        exponent 128 of a value near the maximum is an infinite move."""
        from src.workflow.emitter.extended_runtime import _reference_nudge
        big = torch.full((16, 16), 1.5e19)  # squared: 2.25e38, exponent 128
        p = RegionProgram(
            TileKernel('kernel_0', M=16, N=16, K=16, block_M=16, block_N=16, block_K=16, threads=128,
                       num_stages=2, dtype=DataType('float32'), compute_kind=ComputeKind.COPY),
            Region([], [Op('load', 'entry'), Op('mul', 'value', ['entry', 'entry'])], 'value'))
        p.validate()
        for typed in (False, True):
            p.typed = typed
            interpreter = _typed_region_reference if typed else _region_reference
            exact = interpreter(big, big, asdict(p.body), 16, 16, 'float32')
            for pattern in (0, 1, 2):
                with self.subTest(typed=typed, pattern=pattern):
                    moved = interpreter(big, big, asdict(p.body), 16, 16, 'float32', perturb=pattern)
                    self.assertTrue(torch.isfinite(moved).all())
                    self.assertLessEqual(((moved - exact).abs() / exact).max().item(), 3 * 2.0 ** -23)
        value = torch.tensor([2.25e38, -2.25e38])
        moved = _reference_nudge('mul', {}, [value, value], value, lambda shape: 1)
        self.assertTrue(torch.isfinite(moved).all())
        self.assertTrue(torch.all(moved != value))

    def test_bounds_skip_failing_patterns(self):
        values = {0: torch.tensor([1.0, float('nan'), 3.0]), 2: torch.tensor([2.0, float('nan'), float('inf')])}
        def evaluate(pattern):
            if pattern not in values:
                raise RuntimeError('pattern failed')
            return values[pattern]
        low, high = _reference_bounds(evaluate, patterns=4)
        self.assertEqual(low.tolist(), [1.0, float('inf'), 3.0])
        self.assertEqual(high.tolist(), [2.0, float('-inf'), 3.0])
        self.assertIsNone(_reference_bounds(evaluate, patterns=0))
        # Bounds widen the hull; a value outside them stays unexplained.
        expected = torch.tensor([1.0, 0.0, 3.0])
        self.assertEqual(_reference_explained(torch.tensor([1.5, 0.0, 3.0]), [expected], False, 1e-3,
                                              bounds=(low, high)), (1, 0))
        self.assertEqual(_reference_explained(torch.tensor([2.5, 0.0, 3.0]), [expected], False, 1e-3,
                                              bounds=(low, high)), (1, 1))

    def test_run_region_explains_a_flip_inside_the_envelope_only(self):
        expected = self.evaluate(False, None)
        bases = []
        def nudged(pattern, base):
            bases.append(base)
            return self.evaluate(False, pattern)
        def check(value, **kwargs):
            def launch(c):
                c.copy_(expected)
                c[0, 0] = value
            _run_region([lambda: launch], (self.a,), expected, 1, 1e-3, rtol=1e-5,
                        reference_verify=lambda: expected, reference_nudged=nudged, **kwargs)
        with self.assertRaisesRegex(RuntimeError, 'ORACLE UNSTABLE: 1 mismatched elements lie within'):
            check(0.0)
        self.assertEqual(bases, [0] * 8)
        with self.assertRaisesRegex(RuntimeError, 'WRONG RESULT: structured reference'):
            check(1.0)
        # The base is the reference the kernel mismatches least: here the
        # model, which the kernel follows except at the flip.
        model = expected.clone()
        model[3, 3] = 7.0
        bases.clear()
        def launch(c):
            c.copy_(model)
            c[0, 0] = 0.0
        def nudged_model(pattern, base):
            bases.append(base)
            value = self.evaluate(False, pattern)
            value[3, 3] = 7.0
            return value
        with self.assertRaisesRegex(RuntimeError, 'ORACLE UNSTABLE: 2 mismatched elements lie within'):
            _run_region([lambda: launch], (self.a,), expected, 1, 1e-3, rtol=1e-5,
                        reference_verify=lambda: expected, reference_models=lambda: [model],
                        reference_nudged=nudged_model)
        self.assertEqual(bases, [1] * 8)


class ExplanationTests(unittest.TestCase):
    def test_counts_mismatched_and_unexplained_elements(self):
        expected = torch.tensor([0.0, 0.0, 0.0, 0.0])
        fp64 = torch.tensor([0.0, 1.0, 0.0, 0.0])
        actual = torch.tensor([0.0, 0.5, 2.0, 1e-4])
        # Element 1 lies between the references, element 2 nowhere near them,
        # element 3 is inside tolerance and not counted at all.
        self.assertEqual(_reference_explained(actual, [expected, fp64], False, 1e-3), (2, 1))
        # The tolerance band widens the hull.
        self.assertEqual(_reference_explained(torch.tensor([0.0, 1.0005, 0.0, 0.0]),
                                              [expected, fp64], False, 1e-3), (1, 0))
        self.assertEqual(_reference_explained(torch.tensor([0.0, 1.01, 0.0, 0.0]),
                                              [expected, fp64], False, 1e-3), (1, 1))

    def test_relative_mode_uses_the_implied_absolute_allowance(self):
        expected = torch.full((4,), 100.0)
        # Normalized tolerance 0.05 of mean |ref| 100 allows 5 per element.
        self.assertEqual(_reference_explained(torch.tensor([104.0, 100, 100, 100]), [expected], True, 0.05), (0, 0))
        self.assertEqual(_reference_explained(torch.tensor([106.0, 100, 100, 100]), [expected], True, 0.05), (1, 1))

    def test_non_finite_values_are_explained_only_by_repetition(self):
        expected = torch.tensor([1.0, 1.0, 1.0])
        jitter = torch.tensor([float('nan'), float('inf'), 1.0])
        actual = torch.tensor([float('nan'), float('inf'), float('nan')])
        self.assertEqual(_reference_explained(actual, [expected, jitter], False, 1e-3), (3, 1))
        # A finite value is never explained by a hull of non-finite references.
        self.assertEqual(_reference_explained(torch.tensor([5.0]), [torch.tensor([float('nan')])], False, 1e-3),
                         (1, 1))

    def test_fp16_actual_compares_against_rounded_references(self):
        expected = torch.tensor([1.0, 2.0], dtype=torch.float32)
        fp64 = torch.tensor([1.0, 2.5], dtype=torch.float64)
        actual = torch.tensor([1.0, 2.25], dtype=torch.float16)
        self.assertEqual(_reference_explained(actual, [expected, fp64], False, 1e-3), (1, 0))


class RunRegionVerdictTests(unittest.TestCase):
    def setUp(self):
        self.expected = torch.full((2, 3), 100.0)
        self.source = torch.ones(2, 3)

    def check(self, value, **kwargs):
        def launch(c):
            c.copy_(self.expected)
            c[0, 0] = value
        return _run_region([lambda: launch], (self.source,), self.expected, 1, 0.01, **kwargs)

    def test_admissible_model_passes_and_others_fail(self):
        model = self.expected.clone()
        model[0, 0] = 100.5
        calls = []
        def models():
            calls.append(1)
            return [model]
        self.check(100.5, reference_models=models)
        self.assertEqual(calls, [1])
        with self.assertRaisesRegex(RuntimeError, 'WRONG RESULT: structured reference'):
            self.check(100.5)
        with self.assertRaisesRegex(RuntimeError, 'WRONG RESULT: structured reference'):
            self.check(101.0, reference_models=models)
        # Models are only built on a failure.
        calls.clear()
        self.check(100.0, reference_models=models)
        self.assertEqual(calls, [])

    def test_mismatch_inside_the_perturbed_references_is_oracle_unstable(self):
        fp64 = self.expected.clone()
        fp64[0, 0] = 100.5  # a self-error of 5e-3: _reference_stable passes
        with self.assertRaisesRegex(RuntimeError, 'ORACLE UNSTABLE: 1 mismatched elements lie within'):
            self.check(100.25, reference_verify=lambda: fp64)
        with self.assertRaisesRegex(RuntimeError, 'WRONG RESULT: structured reference') as raised:
            self.check(101.0, reference_verify=lambda: fp64)
        # The original failure is re-raised: no ORACLE text in its traceback.
        self.assertIsNone(raised.exception.__context__)

    def test_large_outputs_evaluate_only_the_uniform_patterns(self):
        # Only the envelope (pattern 1 moves the element down) explains 99.
        for limit, wanted in ((6, list(range(8))), (5, [0, 1])):
            patterns = []
            def nudged(pattern, base):
                patterns.append(pattern)
                value = self.expected.clone()
                value[0, 0] = 99.0 if pattern == 1 else 100.0
                return value
            with self.subTest(limit=limit), self.assertRaisesRegex(RuntimeError, 'ORACLE UNSTABLE: 1 mismatched'):
                self.check(99.0, reference_verify=lambda: self.expected, reference_nudged=nudged, nudge_limit=limit)
            self.assertEqual(patterns, wanted)

    def test_out_of_memory_falls_back_to_the_strict_check(self):
        def exhausted(*_):
            raise torch.cuda.OutOfMemoryError('CUDA out of memory')
        fp64 = self.expected.clone()
        with self.assertRaisesRegex(RuntimeError, 'WRONG RESULT: structured reference'):
            self.check(100.5, reference_models=exhausted, reference_verify=lambda: fp64,
                       reference_nudged=exhausted)
        # Envelope patterns that fail one by one are skipped the same way.
        fp64[0, 0] = 100.5
        with self.assertRaisesRegex(RuntimeError, 'ORACLE UNSTABLE: 1 mismatched elements lie within'):
            self.check(100.25, reference_models=exhausted, reference_verify=lambda: fp64,
                       reference_nudged=exhausted)

    def test_global_gate_still_precedes_the_explanation(self):
        fp64 = self.expected * 2
        with self.assertRaisesRegex(RuntimeError, 'ORACLE UNSTABLE: reference disagrees'):
            self.check(101.0, reference_verify=lambda: fp64)


class EmissionTests(unittest.TestCase):
    def test_tf32_models_only_for_fp32_gemm_and_nudges_for_floats(self):
        config = Config()
        nudged = "tf32=(None, 'truncate', 'nearest')[base], perturb=pattern))"
        for backend in ('tilelang', 'triton'):
            for typed in (False, True):
                interpreter = '_typed_region_reference' if typed else '_region_reference'
                with self.subTest(backend=backend, typed=typed):
                    code = Oracle(config, backend)._emit_code(gemm_program('float32', typed))
                    ast.parse(code)
                    self.assertIn(f", rtol={config.region_elem_rtol_fp32}, reference_models=lambda: "
                                  f"[{interpreter}(A, B, ", code)
                    self.assertIn("tf32=mode) for mode in ('truncate', 'nearest')], reference_nudged=lambda "
                                  f"pattern, base: {interpreter}(A, B, ", code)
                    self.assertIn(nudged, code)
                    self.assertIn('def _reference_explained(', code)
                    self.assertIn('def _reference_bounds(', code)
                    code = Oracle(config, backend)._emit_code(gemm_program('float16', typed))
                    self.assertIn(f", rtol={config.region_elem_rtol_fp16}, reference_nudged=lambda pattern, base: ",
                                  code)
                    self.assertIn(nudged, code)
                    self.assertNotIn('reference_models=lambda', code)


if __name__ == '__main__':
    unittest.main()
