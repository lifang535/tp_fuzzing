"""Coverage expansion: reusable templates, scoped mutations, and arithmetic."""
import ast
import copy
import random
import unittest
from dataclasses import asdict
from unittest.mock import patch

import torch
from src.config import Config
from src.ir import ComputeKind
from src.ir.region import Operation, Region, RegionProgram
from src.workflow.generator.region_generator import RegionGenerator, TemplateOp
from src.workflow.mutator.mutator import Mutator
from src.workflow.feedback import StructuralFeedback, key, program_features
from src.workflow.oracle import Oracle
from src.workflow.emitter.region_runtime import _region_reference
from test_functions import function_program


def arithmetic_program(dtype='float32'):
    p = function_program(dtype)
    fn = p.functions[0]
    fn.body = Region(['x', 'y'], [Operation('div', 'quotient', ['x', 'y']),
                                Operation('minimum', 'clipped', ['quotient', 'x'])], 'clipped')
    p.validate()
    return p


class GenerationDiversityTests(unittest.TestCase):
    def setUp(self):
        state = random.getstate()
        self.addCleanup(random.setstate, state)
        random.seed(218)

    def test_complete_template_precedes_instantiation_and_is_reusable(self):
        gen = RegionGenerator(Config(coverage_probe_prob=0, function_min_count=3))
        with patch.object(gen, 'instantiate', side_effect=AssertionError('too early')):
            template = gen.program_template()
        before = copy.deepcopy(template)
        programs = [gen.instantiate_program(template) for _ in range(12)]
        self.assertEqual(template, before)
        self.assertEqual(len({p.call_label() for p in programs}), 1)
        self.assertGreater(len({str(p.to_dict()) for p in programs}), 1)
        for p in programs:
            p.validate()
            self.assertEqual([o.kind for o in p.all_operations()],
                             [o.kind for o in programs[0].all_operations()])

    def test_load_and_gemm_entry_controls(self):
        for probability, entry, kind in ((0, 'load', ComputeKind.COPY), (1, 'gemm', ComputeKind.GEMM)):
            gen = RegionGenerator(Config(coverage_probe_prob=0, region_gemm_prob=probability))
            for _ in range(10):
                p = gen.generate()
                self.assertEqual(p.body.operations[0].kind, entry)
                self.assertEqual(p.spec.compute_kind, kind)

    def test_data_feedback_guides_reuse_of_earlier_values(self):
        # Isolates the passed-count decay: the uncovered boost is a one-shot
        # mechanism that vanishes after the first observe(), while this test
        # never observes, so both edges would stay boosted and dilute the
        # passed-count signal (see test_uncovered_boost_prefers_fresh_ops).
        gen = RegionGenerator(Config(latest_value_prob=0, uncovered_boost=0))
        gen.feedback = StructuralFeedback()
        # observe() records both counters together (attempted then passed);
        # a passed edge with no attempted entry would be treated as fresh.
        gen.feedback.attempted[key('data', 'neg', 'abs')] = 10000
        gen.feedback.passed[key('data', 'neg', 'abs')] = 10000
        template = [TemplateOp('neg'), TemplateOp('abs')]
        # With only load and neg available, the under-tested load -> abs edge
        # should be preferred over neg -> abs, including the first operand.
        earlier = 0
        for _ in range(300):
            body = gen.instantiate(template)
            earlier += body.operations[-1].operands[0] == body.operations[0].result
        self.assertGreater(earlier, 210)

    def test_uncovered_boost_prefers_fresh_ops(self):
        # MLIRSmith DiversityCriteria-style: a never-attempted op kind gets a
        # large fixed boost and must dominate heavily-passed (but attempted)
        # kinds in template() op selection.
        gen = RegionGenerator(Config(coverage_probe_prob=0, region_typed_prob=0,
                                     region_control_prob=0, uncovered_boost=1000))
        gen.feedback = StructuralFeedback()
        for kind in RegionGenerator.LEAVES:
            if kind != 'neg':
                gen.feedback.attempted[key('op', kind)] = 1
                gen.feedback.passed[key('op', kind)] = 10000
                gen.feedback.attempted[key('nest', 'function', kind)] = 1
                gen.feedback.passed[key('nest', 'function', kind)] = 10000
        kinds = []
        for _ in range(100):
            kinds.extend(n.kind for n in gen.template(budget=[4]))
        self.assertTrue(kinds)
        self.assertGreater(sum(k == 'neg' for k in kinds) / len(kinds), 0.9)

    def test_local_mutations_preserve_scope_and_call_graph(self):
        config = Config(dtype_mutate_prob=0, local_mutate_prob=1)
        seed = function_program(reductions=True)
        before = seed.to_dict()
        changed_ops = changed_attrs = changed_operands = False
        for backend in ('tilelang', 'triton'):
            for _ in range(80):
                mutated = Mutator(config, backend).mutate(seed)
                mutated.validate()
                self.assertNotEqual(mutated.to_dict(), before)
                self.assertEqual(mutated.spec, seed.spec)
                self.assertEqual(mutated.call_label(), seed.call_label())
                for a, b in zip(seed.all_operations(), mutated.all_operations()):
                    changed_ops |= a.kind != b.kind
                    changed_attrs |= a.attrs != b.attrs
                    changed_operands |= a.operands != b.operands
                ast.parse(Oracle(config, backend)._emit_code(mutated))
                self.assertEqual(RegionProgram.from_dict(mutated.to_dict()).to_dict(), mutated.to_dict())
        self.assertEqual(seed.to_dict(), before)
        self.assertTrue(changed_ops and changed_attrs and changed_operands)

    def test_division_and_minimum_reference_with_zero_denominators(self):
        p = function_program()
        p.functions = []
        p.spec.M, p.spec.N = 2, 3
        p.body = Region([], [Operation('load', 'x'), Operation('neg', 'y', ['x']),
                            Operation('div', 'q', ['x', 'y']),
                            Operation('minimum', 'z', ['q', 'x'])], 'z')
        a = torch.tensor([[0., 0.0005, -0.0005], [2., -2., 1.]])
        expected = torch.tensor([[0., 0.0005, -0.5], [1., -2., 1.]])
        for dtype in ('float16', 'float32'):
            actual = _region_reference(a, torch.empty(1, 3), asdict(p.body), 32, 32, dtype)
            torch.testing.assert_close(actual, expected.to(getattr(torch, dtype)))
        for backend in ('tilelang', 'triton'):
            ast.parse(Oracle(Config(), backend)._emit_code(arithmetic_program()))

    def test_attribute_variants_contribute_to_feedback(self):
        p = function_program()
        before = program_features(p)
        loop = p.functions[1].body.operations[1]
        loop.attrs['trip_count'] = 4
        after = program_features(p)
        self.assertIn(key('attribute', 'for', 4), after - before)


if __name__ == '__main__':
    unittest.main()
