"""Contracts for the compositional IR, independent oracle, and campaign state."""
import ast
import contextlib
import copy
import io
import json
import random
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from src.backends import get_backend
from src.config import Config
from src.ir.extended import TensorType as Ty, Value, Node, Block, Buffer, Helper, ExtendedProgram, analyze
from src.ir.serialization import program_from_dict, program_to_dict
from src.workflow.emitter.extended_runtime import extended_inputs, extended_reference, extended_check, run_extended
from src.workflow.feedback import StructuralFeedback, program_features, key
from src.workflow.fuzzer.fuzzer import TileSmith
from src.workflow.generator.extended import Builder, ExtendedGenerator, FAMILIES, mutate_extended
from src.workflow.generator.generator import ProgramGenerator
from src.workflow.oracle import Oracle


def nan_arithmetic_program(config, backend):
    """Candidate: half overflow then Inf-Inf must produce NaN."""
    generator = ExtendedGenerator(config, backend)
    program = generator.generate('shape_matmul')
    pool = [v for n in program.body.operations for v in n.results]
    builder = Builder(generator, program.body, pool)
    answer = next(v for v in pool if v.name == program.body.returns[0])
    half = builder.cast(answer, 'float16')
    largest = builder.constant(half.type, 65504.)
    infinity = builder.binary('mul', largest, largest)
    # Use distinct producers: TIR simplifies x-x to zero even when x is Inf.
    # That separate compiler candidate is retained in the validation artifacts.
    other = builder.constant(half.type, 32752.)
    other_infinity = builder.binary('mul', largest, other)
    nan = builder.binary('sub', infinity, other_infinity)
    mask = builder.binary('eq', builder.indices(half.type.shape, shuffled=False), builder.constant(Ty('int32'), 0))
    selected = builder.emit('select', [mask, nan, half], [half.type])
    maximum = builder.reduce(selected, 0, 'max')
    minimum = builder.reduce(selected, 0, 'min')
    program.body.returns += [maximum.name, minimum.name]
    program.validate()
    return program


def nan_reduction_program(config, backend):
    """Input NaN after MMA, with finite neighbours on both reduction axes."""
    generator = ExtendedGenerator(config, backend)
    builder = Builder(generator)
    ty = Ty('float16', (16, 16))
    buf = generator.buffer('float16', ty.size)
    index = builder.indices(ty.shape, shuffled=False)
    mask = builder.binary('eq', index, builder.constant(Ty('int32'), 0))
    loaded = builder.load(buf, ty.shape, index, mask)
    lhs, rhs = builder.constant(ty, 0.125), builder.constant(ty, 0.5)
    acc = builder.constant(Ty('float32', ty.shape), 0.125)
    product = builder.emit('matmul', [lhs, rhs, acc], [acc.type])
    selected = builder.emit('select', [mask, loaded, builder.cast(product, 'float16')], [ty])
    reductions = [builder.reduce(selected, axis, kind) for axis in (0, 1) for kind in ('max', 'min')]
    builder.block.returns = [v.name for v in reductions]
    program = ExtendedProgram(builder.block, generator.buffers, observations=[selected.name],
                              input_pattern='special', family='nan_reduction',
                              configuration_pair=config.extended_configuration_pair,
                              observation_pair=config.extended_observation_pair)
    get_backend(backend).validate_program(program)
    return program


class ExtendedTests(unittest.TestCase):
    def setUp(self):
        state = random.getstate()
        self.addCleanup(random.setstate, state)
        random.seed(0)
        self.config = Config(extended_prob=1, region_repeat_count=2)

    def program(self, family='mixed', backend='triton'):
        return ExtendedGenerator(self.config, backend).generate(family)

    def test_generation_mutation_roundtrip_and_both_emitters(self):
        for backend in ('triton', 'tilelang'):
            for family in FAMILIES:
                for seed in range(4):
                    random.seed(seed)
                    program = self.program(family, backend)
                    before = copy.deepcopy(program.to_dict())
                    mutated = mutate_extended(program, self.config, backend)
                    self.assertEqual(before, program.to_dict())
                    for p in (program, mutated):
                        encoded = json.loads(json.dumps(program_to_dict(p)))
                        restored = program_from_dict(encoded)
                        self.assertEqual(p.to_dict(), restored.to_dict())
                        ast.parse(get_backend(backend).make_emitter(self.config).emit(restored))
                        inputs = extended_inputs(encoded)
                        for steps, limit in p.runtime_cases:
                            outputs, memory = extended_reference(encoded, inputs, steps, limit)
                            self.assertEqual(set(outputs), set(p.body.returns + p.observations))
                            for b in p.buffers:
                                if b.base is None and b.role == 'input':
                                    self.assertTrue(torch.equal(inputs[b.name], memory[b.name]))

    def test_fresh_route_can_force_extended_or_native(self):
        for backend in ('triton', 'tilelang'):
            self.assertIsInstance(ProgramGenerator(self.config, backend).generate(), ExtendedProgram)
            self.assertNotIsInstance(ProgramGenerator(Config(extended_prob=0), backend).generate(), ExtendedProgram)
        with self.assertRaises(ValueError):
            ProgramGenerator(Config(compile_only=True, extended_prob=0))

    def test_missing_operands_are_synthesized_and_reused(self):
        gen = ExtendedGenerator(self.config, 'triton')
        builder = Builder(gen)
        for dtype in ('float16', 'float32', 'int32', 'bool'):
            value = builder.get_or_create(Ty(dtype, (16, 16)))
            self.assertEqual(value.type, Ty(dtype, (16, 16)))
            with patch('random.random', return_value=0):
                self.assertIn(builder.get_or_create(value.type), builder.pool)
        self.assertLess(len(builder.block.operations), 24)

    def test_alias_snapshot_mask_and_complete_memory_reference(self):
        gen = ExtendedGenerator(self.config, 'triton')
        b = Builder(gen)
        root = gen.buffer('int32', 8, 'scratch')
        left = gen.buffer('int32', 4, 'scratch', base=root.name, offset=0, stride=2)
        right = gen.buffer('int32', 4, 'scratch', base=root.name, offset=2)
        idx = b.indices((4,), shuffled=False)
        mask = b.binary('lt', idx, b.constant(Ty('int32'), 3))
        b.emit('store', [idx, mask, idx], buffer=left.name)
        b.emit('barrier')
        snapshot = b.load(left, (4,), idx)
        b.emit('barrier')
        b.emit('store', [idx, mask, snapshot], buffer=right.name)
        b.emit('barrier')
        reread = b.load(left, (4,), idx)
        b.block.returns = [snapshot.name, reread.name]
        p = ExtendedProgram(b.block, gen.buffers)
        p.validate()
        inputs = extended_inputs(p.to_dict())
        outputs, memory = extended_reference(p.to_dict(), inputs, 0, 3)
        torch.testing.assert_close(outputs[snapshot.name][0], torch.tensor([0, 1, 2, 11], dtype=torch.int32))
        torch.testing.assert_close(outputs[reread.name][0], torch.tensor([0, 0, 2, 11], dtype=torch.int32))
        torch.testing.assert_close(memory[root.name][0, 16:-16], torch.tensor([0, 11, 0, 1, 2, 11, 11, 11], dtype=torch.int32))
        self.assertTrue(torch.all(memory[root.name][:, :16] == 19))
        self.assertTrue(torch.all(memory[root.name][:, -16:] == 19))
        # A constant/scattered address cannot be certified as a race-free store.
        store = next(n for n in b.block.operations if n.op == 'store')
        store.operands[0] = next(n.results[0].name for n in b.block.operations if n.op == 'constant')
        with self.assertRaises(ValueError):
            p.validate()

    def test_mixed_loop_carriers_zero_trip_and_runtime_bound(self):
        p = self.program('control_calls')
        loops = [n for n in p.body.operations if n.op in ('for', 'while')]
        self.assertEqual([n.op for n in loops], ['for', 'while'])
        self.assertEqual([v.type.dtype for v in loops[-1].results], ['float16', 'float32', 'int32'])
        counter = loops[-1].results[-1].name
        inputs = extended_inputs(p.to_dict())
        for steps in (0, 1, 3, 8):
            output, _ = extended_reference(p.to_dict(), inputs, steps, 0)
            n = min(steps, 4)
            self.assertTrue(torch.all(output[counter] == n * (n + 1)))

    def test_invalid_scope_signature_and_alias_are_rejected(self):
        for change in ('scope', 'signature', 'view', 'version'):
            p = self.program()
            if change == 'scope':
                p.body.operations[-1].operands = ['missing']
            elif change == 'signature':
                loop = next(n for n in p.body.operations if n.op == 'while')
                loop.regions[0].returns.pop()
            elif change == 'view':
                next(b for b in p.buffers if b.base).offset = 4096
            else:
                data = p.to_dict()
                data['version'] = 100
                with self.assertRaises(ValueError):
                    ExtendedProgram.from_dict(data)
                continue
            with self.assertRaises(ValueError):
                p.validate()

    def test_observability_excludes_dead_values_and_disabled_observations(self):
        x, y = Value('x', Ty('float32')), Value('y', Ty('int32'))
        first, dead = Node('constant', [x], attrs={'value': 1}), Node('constant', [y], attrs={'value': 7})
        helper_value = Value('h', Ty('float16'))
        helper_node = Node('constant', [helper_value], attrs={'value': 2})
        p = ExtendedProgram(Block(operations=[first, dead], returns=['x']), [Buffer('input', 'float32', 1)],
                            [Helper('unused', Block(operations=[helper_node], returns=['h']))], ['y'])
        _, live = analyze(p)
        self.assertIn(('main', id(dead)), live)
        self.assertNotIn(('unused', id(helper_node)), live)
        p.observation_pair = False
        _, live = analyze(p)
        self.assertIn(('main', id(first)), live)
        self.assertNotIn(('main', id(dead)), live)
        features = program_features(p)
        signature = key('extended_signature', 'constant', (('int32', ()),))
        self.assertIn(signature, features)
        self.assertNotIn(key('observable', signature), features)

    def test_called_function_and_region_dependencies_remain_live(self):
        p = self.program('control_calls')
        _, live = analyze(p)
        for fn in p.functions:
            self.assertIn((fn.name, id(fn.body.operations[-1])), live)
        loop = next(n for n in p.body.operations if n.op == 'while')
        self.assertIn(('main', id(loop)), live)
        self.assertIn(('main', id(loop.regions[0].operations[-1])), live)

    def test_numeric_oracle_rejects_special_and_integer_errors(self):
        expected = torch.tensor([float('nan'), float('inf'), -float('inf'), 1.])
        extended_check(expected.clone(), expected, 'equal')
        for index, value in ((0, 0.), (1, -float('inf')), (3, 2.)):
            actual = expected.clone()
            actual[index] = value
            with self.assertRaisesRegex(RuntimeError, 'WRONG RESULT'):
                extended_check(actual, expected, 'fault')
        with self.assertRaisesRegex(RuntimeError, 'index=1'):
            extended_check(torch.tensor([1, 4], dtype=torch.int32), torch.tensor([1, 3], dtype=torch.int32), 'integer')

    def test_column_reduction_propagates_nan_without_poisoning_other_lanes(self):
        for backend in ('triton', 'tilelang'):
            random.seed(1)
            p = nan_reduction_program(self.config, backend)
            raw = p.to_dict()
            outputs, _ = extended_reference(raw, extended_inputs(raw), 1, 256)
            for name in p.body.returns:
                self.assertTrue(torch.all(torch.isnan(outputs[name][:, 0])))
                self.assertTrue(torch.all(torch.isfinite(outputs[name][:, 1:])))
            ast.parse(get_backend(backend).make_emitter(self.config).emit(p))

    def test_oracle_checks_unwritten_outputs_and_scratch_corruption(self):
        p = self.program('indexed_memory').to_dict()
        watched = p['body']['returns'] + p['observations']
        def prepare(fault):
            def launch(memory, outputs, steps, limit):
                expected, final = extended_reference(p, memory, steps, limit)
                for name, value in final.items():
                    memory[name].copy_(value)
                for name, value in outputs.items():
                    if fault != 'unwritten':
                        value[:, 16:-16].copy_(expected[name].reshape(p['blocks'], -1))
                if fault == 'scratch':
                    root = next(b['name'] for b in p['buffers'] if b['role'] == 'scratch' and b['base'] is None)
                    memory[root][:, 16] = 100
            return lambda: [('simulated', watched, launch)]
        run_extended(p, prepare(None), repeats=2, device='cpu')
        for fault in ('unwritten', 'scratch'):
            with self.assertRaisesRegex(RuntimeError, 'WRONG RESULT'):
                run_extended(p, prepare(fault), repeats=2, device='cpu')

    def test_compilation_feedback_is_separate_from_execution_and_persists(self):
        p, feedback = self.program(), StructuralFeedback()
        records = [{'features': [key('compiler', 'ttir', 'tt.dot')]}]
        feedback.observe(p, False)
        self.assertEqual(feedback.observe_compilation(p, records, True), 1)
        self.assertTrue(feedback.compiled)
        self.assertFalse(feedback.passed)
        self.assertGreater(feedback.observe(p, True), 0)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'feedback.json'
            feedback.save(path)
            restored = StructuralFeedback()
            restored.restore(path)
            self.assertEqual(restored.compiled, feedback.compiled)
            self.assertEqual(restored.compiler, feedback.compiler)

    def test_campaign_saves_restores_and_deduplicates_extended_program(self):
        p = self.program()
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            config = Config(output_dir=directory, backends=['triton'], extended_prob=1, seed=10)
            fuzzer = TileSmith(config)
            with patch.object(fuzzer, '_generate_test_case', return_value=p), patch.object(fuzzer.oracle, 'test', return_value=None):
                fuzzer.run(1, verbose=False)
            self.assertTrue(list((fuzzer.output_dir / 'passed').glob('*.json')))
            restored = TileSmith(config, resume_dir=str(fuzzer.output_dir))
            self.assertIn(restored._make_sig(p), restored.tested_configs)
            self.assertEqual(restored.seed_pool[0].to_dict(), p.to_dict())
            self.assertEqual(restored._make_sig_from_dict({'params': p.params_dict}), restored._make_sig(p))

    def test_pass_configuration_sweep_depths(self):
        """extended_config_depth gates the pass-configuration pair tier: 0 =
        single configuration, 1 = the historical threads/stages pair, 2 =
        additionally a second pass configuration (RC2 reachability)."""
        from src.backends import get_backend
        tilelang, triton = get_backend('tilelang'), get_backend('triton')
        # Pin the numeric sweeps off: they would append precision variants to
        # this matmul family and change the counts asserted below. Random
        # pass-pipeline sampling is also off (it appends plain variants).
        depth2_cfg = Config(extended_prob=1, extended_config_depth=2,
                            extended_precision_pair=False, extended_identity_pair=False,
                            random_config_count=0)
        tl_program = ExtendedGenerator(depth2_cfg, 'tilelang').generate('mixed')
        tr_program = ExtendedGenerator(depth2_cfg, 'triton').generate('mixed')

        variants = tilelang.extended_variants(
            tl_program, Config(extended_prob=1, extended_configuration_pair=False,
                               extended_config_depth=0,
                               extended_precision_pair=False, extended_identity_pair=False,
                               random_config_count=0))
        self.assertEqual(len(variants), 2)  # one configuration, both observation modes

        variants = tilelang.extended_variants(
            tl_program, Config(extended_prob=1, extended_config_depth=1,
                               extended_precision_pair=False, extended_identity_pair=False,
                               random_config_count=0))
        self.assertEqual(len(variants), 4)  # threads/stages pair x observation modes
        self.assertEqual([v[1]['pass_configs'] for v in variants],
                         [{}, {}] + [{'tirx.disable_vectorize': True}] * 2)

        variants = tilelang.extended_variants(tl_program, depth2_cfg)
        self.assertEqual(len(variants), 6)
        self.assertIn({'tl.disable_loop_unswitching': True}, [v[1]['pass_configs'] for v in variants])
        self.assertEqual(len({lower.name for lower, _ in variants}), len(variants))

        tvariants = triton.extended_variants(tr_program, depth2_cfg)
        self.assertEqual(len(tvariants), 6)
        fused = [v for v in tvariants if v[1].get('enable_fp_fusion')]
        self.assertEqual(len(fused), 2)
        self.assertTrue(all(not v[1].get('enable_fp_fusion') for v in tvariants[:2]))
        self.assertEqual(len({lower.name for lower, _ in tvariants}), len(tvariants))

    def test_precision_pair_variants_are_opt_in_and_matmul_gated(self):
        """extended_precision_pair appends one fp16-accumulation variant per
        base configuration (plus one tf32 variant on triton); matmul-less
        programs and configs without the flag get none."""
        from src.backends import get_backend
        tilelang, triton = get_backend('tilelang'), get_backend('triton')
        config = Config(extended_prob=1, extended_precision_pair=True, extended_config_depth=1,
                    extended_int8_prob=0, extended_fma_prob=0, extended_shape_op_prob=0, extended_atomic_prob=0)
        tl_program = ExtendedGenerator(config, 'tilelang').generate('shape_matmul')
        tr_program = ExtendedGenerator(config, 'triton').generate('shape_matmul')

        tl_variants = tilelang.extended_variants(tl_program, config)
        tl_prec = [v for v in tl_variants if v[1].get('precision') == 'fp16']
        self.assertEqual(len(tl_prec), 2)  # one per base configuration
        self.assertTrue(all(lower.name.endswith('_prec') for lower, _ in tl_prec))
        # Precision options extend, not replace, the base configuration knobs.
        self.assertTrue(all(set(options) >= {'threads', 'stages', 'pass_configs', 'precision'}
                            for _, options in tl_prec))
        self.assertEqual(len({lower.name for lower, _ in tl_variants}), len(tl_variants))

        tr_variants = triton.extended_variants(tr_program, config)
        tr_prec = [v for v in tr_variants if v[1].get('precision') == 'fp16']
        tf32 = [v for v in tr_variants if v[1].get('input_precision') == 'tf32']
        self.assertEqual(len(tr_prec), 2)
        self.assertEqual(len(tf32), 1)
        self.assertEqual(tf32[0][0].name, 'extended_tf32')
        self.assertTrue(all(set(options) >= {'num_warps', 'num_stages', 'enable_fp_fusion', 'precision'}
                            for _, options in tr_prec))
        self.assertEqual(len({lower.name for lower, _ in tr_variants}), len(tr_variants))

        off = Config(extended_prob=1, extended_precision_pair=False, extended_identity_pair=False)
        self.assertFalse(any(v[1].get('precision') or v[1].get('input_precision')
                             for v in tilelang.extended_variants(tl_program, off)))
        self.assertFalse(any(v[1].get('precision') or v[1].get('input_precision')
                             for v in triton.extended_variants(tr_program, off)))
        # No constant-accumulator matmul → no precision variants even with the flag.
        arith = ExtendedGenerator(config, 'triton').generate('arithmetic')
        self.assertFalse(any(v[1].get('precision') or v[1].get('input_precision')
                             for v in triton.extended_variants(arith, config)))

    def test_fast_math_pair_is_opt_in_on_both_sides(self):
        from src.backends import get_backend
        adapter = get_backend('tilelang')
        config = Config(extended_prob=1, extended_fast_math_pair=True, extended_config_depth=1)
        program = ExtendedGenerator(config, 'tilelang').generate('arithmetic')
        program.fast_math_pair = True
        program.validate()
        with_fast = adapter.extended_variants(program, config)
        self.assertIn({'tl.enable_fast_math': True}, [v[1]['pass_configs'] for v in with_fast])
        without = adapter.extended_variants(
            program, Config(extended_prob=1, extended_fast_math_pair=False, extended_config_depth=1))
        self.assertNotIn({'tl.enable_fast_math': True}, [v[1]['pass_configs'] for v in without])

    def test_old_extended_dicts_restore_with_new_field_defaults(self):
        """Programs saved before the pass configuration sweep lack the new
        keys; from_dict must default them and validate."""
        p = ExtendedGenerator(Config(extended_prob=1, extended_config_depth=2),
                              'triton').generate('mixed')
        old = copy.deepcopy(p.to_dict())
        for field in ('pass_config_pair', 'fast_math_pair', 'precision_pair', 'identity_pair'):
            del old[field]
        restored = program_from_dict(old)
        self.assertTrue(restored.pass_config_pair)
        self.assertFalse(restored.fast_math_pair)
        self.assertTrue(restored.precision_pair)
        self.assertTrue(restored.identity_pair)
        self.assertEqual(restored.to_dict(), p.to_dict())

    def test_depth_two_emission_carries_the_pass_pairs(self):
        config = Config(extended_prob=1, extended_config_depth=2)
        oracle = Oracle(config, 'tilelang')
        program = ExtendedGenerator(config, 'tilelang').generate('mixed')
        code = oracle._emit_code(program)
        ast.parse(code)
        self.assertIn("{'tl.disable_loop_unswitching': True}", code)
        code = Oracle(config, 'triton')._emit_code(
            ExtendedGenerator(config, 'triton').generate('mixed'))
        ast.parse(code)
        self.assertIn("'enable_fp_fusion': True", code)

    def test_precision_emission_ships_the_transformed_reference(self):
        from src.backends import get_backend
        from src.workflow.generator.identities import extended_variant_label
        config = Config(extended_prob=1, extended_precision_pair=True, extended_identity_pair=False,
                    extended_int8_prob=0, extended_fma_prob=0, extended_shape_op_prob=0, extended_atomic_prob=0)
        # REFERENCE_PROGRAMS keys every precision variant's compile label to
        # the transformed program dict (labels carry the global variant index).
        program = ExtendedGenerator(config, 'tilelang').generate('shape_matmul')
        expected = [extended_variant_label('tilelang', i, options)
                    for i, (_, options) in enumerate(get_backend('tilelang').extended_variants(program, config))
                    if options.get('precision') == 'fp16']
        self.assertTrue(expected)
        code = Oracle(config, 'tilelang')._emit_code(program)
        ast.parse(code)
        for label in expected:
            self.assertIn(f"'{label}':", code)
        self.assertIn("'precision': 'fp16'", code)
        self.assertIn('reference_programs=REFERENCE_PROGRAMS', code)
        code = Oracle(config, 'triton')._emit_code(
            ExtendedGenerator(config, 'triton').generate('shape_matmul'))
        ast.parse(code)
        self.assertIn('extended_tf32', code)
        self.assertIn('reference_programs=REFERENCE_PROGRAMS', code)
        # Matmul-less programs ship an empty reference table.
        code = Oracle(config, 'triton')._emit_code(
            ExtendedGenerator(config, 'triton').generate('arithmetic'))
        ast.parse(code)
        self.assertIn('REFERENCE_PROGRAMS = {}', code)
