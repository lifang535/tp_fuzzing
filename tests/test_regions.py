"""Structured scope, execution, persistence, and compatibility regression tests."""
import ast
import copy
import json
import random
import unittest
from dataclasses import asdict
import torch
from src.config import Config
from src.ir import TileKernel, DataType, ComputeKind
from src.ir.region import Operation as Op, Region, RegionProgram
from src.workflow.generator.region_generator import RegionGenerator, TemplateOp
from src.workflow.generator import ProgramGenerator
from src.workflow.emitter.region_runtime import _region_reference
from src.workflow.oracle import Oracle
from src.workflow.fuzzer.fuzzer import TileSmith
from src.workflow.mutator.mutator import Mutator
from src.workflow.feedback import program_features, key


def nested_program(dtype='float32', initial='load'):
    # Each loop iteration uses the immutable entry tile as well as the carried tile.
    loop = Region(['carry'], [Op('add', 'sum', ['carry', 'entry'])], 'sum')
    yes = Region(['yes_arg'], [Op('for', 'loop_out', ['yes_arg'], {'trip_count': 3}, [loop])], 'loop_out')
    no = Region(['no_arg'], [Op('neg', 'negative', ['no_arg'])], 'negative')
    body = Region([], [Op(initial, 'entry'), Op('if', 'chosen', ['entry'], {'parity': 0}, [yes, no]),
                       Op('add', 'final', ['chosen', 'entry'])], 'final')
    spec = TileKernel('kernel_0', M=33, N=35, K=33, block_M=32, block_N=32, block_K=32,
                      threads=128, num_stages=2, dtype=DataType(dtype),
                      compute_kind=ComputeKind.COPY if initial == 'load' else ComputeKind.GEMM)
    # Keep baseline GEMM copies 4-byte aligned, while retaining M/N/K tails.
    # Odd fp16 strides remain in the random generator to exercise cp.async lowering.
    if initial == 'gemm':
        spec.N = spec.K = 34
    return RegionProgram(spec, body)


def loop_outer_program(dtype='float32', initial='load'):
    p = nested_program(dtype, initial)
    gen = RegionGenerator(Config())
    state = random.getstate()
    random.seed(91)
    try:
        p.body = gen.instantiate([TemplateOp('for', [[
            TemplateOp('if', [[TemplateOp('scale'), TemplateOp('round')],
                              [TemplateOp('neg'), TemplateOp('abs')]]),
            TemplateOp('sqrt'), TemplateOp('mul'), TemplateOp('add')]])], initial)
    finally:
        random.setstate(state)
    p.validate()
    return p


def migrated_ops_program(dtype='float32', reductions=False):
    p = nested_program(dtype)
    if reductions:
        ops = [Op('tile_transpose', 'transposed', ['arg']),
               Op('row_max', 'largest', ['transposed']), Op('row_min', 'smallest', ['arg']),
               Op('row_sum', 'total', ['arg']), Op('row_softmax', 'prob', ['transposed']),
               Op('add', 'extrema', ['largest', 'smallest']), Op('add', 'stats', ['extrema', 'total']),
               Op('mul', 'answer', ['stats', 'prob'])]
    else:
        ops = [Op('exp', 'exponent', ['arg']), Op('sub', 'diff', ['exponent', 'arg']),
               Op('maximum', 'maxval', ['diff', 'arg']), Op('where', 'selected', ['arg', 'maxval', 'exponent']),
               Op('copy', 'answer', ['selected'])]
    yes = Region(['arg'], ops, 'answer')
    no = Region(['other'], [Op('neg', 'negative', ['other'])], 'negative')
    p.body = Region([], [Op('load', 'entry'), Op('if', 'choice', ['entry'], {'parity':0}, [yes,no])], 'choice')
    p.validate()
    return p


class RegionTests(unittest.TestCase):
    def test_tile_padding_reduction_and_transpose_reference(self):
        a = torch.tensor([[1., 2., 3.], [4., 5., 6.]])
        p = nested_program()
        p.spec.M, p.spec.N = 2, 3
        for kind in ('row_sum', 'row_max', 'row_min', 'row_softmax', 'tile_transpose'):
            p.body = Region([], [Op('load', 'entry'), Op(kind, 'out', ['entry'])], 'out')
            p.validate()
            actual = _region_reference(a, torch.empty(33,3), asdict(p.body), 32, 32, 'float32')
            padded = torch.zeros(32,32)
            padded[:2,:3] = a
            if kind == 'tile_transpose':
                expected = padded.T
            elif kind == 'row_softmax':
                expected = torch.softmax(padded, 1)
            else:
                fn = {'row_sum':'sum', 'row_max':'amax', 'row_min':'amin'}[kind]
                expected = getattr(padded, fn)(1, keepdim=True).expand(32,32)
            torch.testing.assert_close(actual, expected[:2,:3])

    def test_loop_carry_and_branch_semantics(self):
        p = nested_program()
        p.validate()
        a = torch.arange(33*35).reshape(33,35).float() / 100
        actual = _region_reference(a, torch.empty(33,35), asdict(p.body), 32, 32, 'float32')
        expected = a * 5
        expected[32:] = 0
        torch.testing.assert_close(actual, expected)

    def test_scope_rejects_branch_escape(self):
        p = nested_program()
        p.body.operations[-1].operands[1] = 'negative'
        with self.assertRaisesRegex(ValueError, 'not visible'):
            p.validate()

    def test_rejects_duplicate_and_unbounded_loop(self):
        p = nested_program()
        p.body.operations[-1].result = 'entry'
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            p.validate()
        p = nested_program()
        p.body.operations[1].regions[0].operations[0].attrs['trip_count'] = 100
        with self.assertRaisesRegex(ValueError, 'bounded'):
            p.validate()

    def test_persistence_and_bug_metadata(self):
        p = nested_program()
        raw = json.loads(json.dumps(p.to_dict()))
        restored = TileSmith._dict_to_program(raw)
        self.assertEqual(restored.to_dict(), raw)
        sig = TileSmith._make_sig(p)
        self.assertEqual(sig, TileSmith._make_sig_from_dict(raw))
        params, dtype, kind = Oracle(Config(), 'triton')._get_meta(p)
        self.assertEqual(sig, TileSmith._make_sig_from_dict({'params':params, 'dtype':dtype, 'compute_kind':kind}))
        altered = copy.deepcopy(p)
        altered.body.operations[1].attrs['parity'] = 1
        self.assertNotEqual(sig, TileSmith._make_sig(altered))

    def test_real_nesting_and_data_feedback(self):
        p = nested_program()
        features = program_features(p)
        self.assertIn(key('nest', 'if', 'for'), features)
        self.assertIn(key('nest', 'for', 'add'), features)
        self.assertIn(key('data', 'load', 'add'), features)
        for backend in ('triton', 'tilelang'):
            code = Oracle(Config(), backend)._emit_code(p)
            ast.parse(code)
            self.assertIn('if by % 2 == 0:', code)
            self.assertIn('for iter_loop_out in ', code)

    def test_template_and_random_mutation(self):
        random.seed(70)
        config = Config(coverage_probe_prob=0)
        gen = ProgramGenerator(config)
        template = [TemplateOp('if', [[TemplateOp('for', [[TemplateOp('add')]])], []])]
        body = gen.region_gen.instantiate(template)
        p = nested_program()
        p.body = body
        p.validate()
        self.assertFalse(hasattr(template[0], 'operands'))
        for _ in range(100):
            p = gen.generate()
            self.assertIsInstance(p, RegionProgram)
            p = Mutator(config).mutate(p)
            p.validate()
            RegionProgram.from_dict(p.to_dict())
            for backend in ('triton', 'tilelang'):
                ast.parse(Oracle(config, backend)._emit_code(p))

    def test_campaign_save_and_resume(self):
        import tempfile
        import contextlib
        import io
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            config = Config(output_dir=directory, seed=17, coverage_probe_prob=0,
                            seed_add_prob=1, backends=['triton'])
            with patch.object(Oracle, 'test', return_value=None):
                campaign = TileSmith(config)
                campaign.run(2, verbose=False)
                restored = TileSmith(config, resume_dir=str(campaign.output_dir))
                self.assertEqual(restored.stats.total_tested, 2)
                self.assertTrue(restored.seed_pool)
                self.assertTrue(all(isinstance(p, RegionProgram) for p in restored.seed_pool))
                self.assertEqual(campaign.tested_configs, restored.tested_configs)
                restored.run(1, verbose=False)
                self.assertEqual(restored.stats.total_tested, 3)


    def test_function_template_is_instantiated_in_one_route(self):
        gen = RegionGenerator(Config(coverage_probe_prob=0))
        for first in ('load', 'gemm'):
            for suffix in ([], [TemplateOp('row_sum')], [TemplateOp('for', [[TemplateOp('row_softmax')]])]):
                template = [TemplateOp(first)] + suffix
                before = copy.deepcopy(template)
                p = gen.instantiate_function(template)
                p.validate()
                self.assertEqual(template, before)

    def test_region_parameter_diversity(self):
        from collections import Counter
        config = Config(coverage_probe_prob=0, dim_range=(17, 2048))
        generator = ProgramGenerator(config, 'tilelang')
        observed = [generator.generate().spec for _ in range(120)]
        self.assertGreater(len({(p.M, p.N, p.K) for p in observed}), 80)
        self.assertGreater(len({(p.block_M, p.block_N, p.block_K) for p in observed}), 8)
        self.assertGreater(max(p.M for p in observed), 256)
        self.assertGreater(len({len(list(__import__('src.ir.region', fromlist=['walk']).walk(generator.generate().body))) for _ in range(20)}), 1)

    def test_native_and_probe_generation_routes(self):
        for prob in (0, 1):
            generator = ProgramGenerator(Config(coverage_probe_prob=prob))
            self.assertFalse(hasattr(generator, 'dynamic_gen'))
            self.assertFalse(hasattr(generator, 'pipeline_gen'))
            self.assertFalse(hasattr(generator, 'generate_legacy'))
            for _ in range(30):
                p = generator.generate()
                self.assertEqual(p.body.operations[0].kind == 'probe', prob == 1)
                p = RegionProgram.from_dict(p.to_dict())
                Mutator(generator.config).mutate(p).validate()
                for backend in ('triton', 'tilelang'):
                    ast.parse(Oracle(generator.config, backend)._emit_code(p))

class DivRelativeModeTests(unittest.TestCase):
    """Division amplifies magnitudes (the reference clamps the denominator to
    1e-3, so chained divs grow ~1e3 per op): div programs must use the
    relative error mode, or every large-magnitude result becomes a false
    wrong_result. The rule is pattern-based, so a div inside a function body
    counts as well."""

    def test_div_programs_use_relative_error_mode(self):
        from test_generation_diversity import arithmetic_program
        from src.ir.region import RegionExecution
        config = Config()
        rtol = str(config.region_rtol_fp32)
        atol = str(config.elemwise_atol)
        for backend in ('triton', 'tilelang'):
            for execution in (None, RegionExecution()):
                with self.subTest(backend=backend, execution=execution):
                    p = arithmetic_program()
                    p.execution = execution
                    code = Oracle(config, backend)._emit_code(p)
                    ast.parse(code)
                    plain = migrated_ops_program()
                    plain.execution = execution
                    plain_code = Oracle(config, backend)._emit_code(plain)
                    ast.parse(plain_code)
                    if execution is None:
                        # Legacy single-execution harness: mode and tolerance
                        # are inline conditionals on the emitted error lines.
                        self.assertIn('relative_error if True else max_diff', code)
                        self.assertIn(f'if True else {atol}', code)
                        self.assertIn('relative_error if False else max_diff', plain_code)
                        self.assertIn(f'if False else {atol}', plain_code)
                    else:
                        # Checked harness: mode is a call argument and the
                        # tolerance is baked in as the selected value.
                        self.assertIn('relative=True', code)
                        self.assertIn(f'tolerance={rtol}', code)
                        self.assertIn('relative=False', plain_code)
                        self.assertIn(f'tolerance={atol}', plain_code)


if __name__ == '__main__':
    unittest.main()
