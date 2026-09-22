"""Unit tests for the Phase 2c algebraic-identity sweep (identity pair).

The identity pair rewrites a matmul-less program copy so one float
mul(x, add/sub(y, z)) becomes add/sub(mul(x, y), mul(x, z)) (RC5 reachability):
the transform must keep the IR valid, leave the original untouched, insert
fresh intermediate SSA names before the rewritten node, and the interpreter
must agree with the original program within floating-point tolerance.
"""
import ast
import random
import unittest

import torch

from src.backends.common.diagnostics import classify_root_cause
from src.config import Config
from src.ir.extended import TensorType as Ty, Block
from src.ir.serialization import program_to_dict
from src.workflow.emitter.extended_runtime import extended_inputs, extended_reference
from src.workflow.generator.extended import Builder, ExtendedGenerator
from src.workflow.generator.identities import extended_variant_label, identity_variant

MATMUL_FAMILIES = ('shape_matmul', 'mixed')


def hand_built_program(inner_op, dtype='float32'):
    """`x * (y OP z)` (or its mirrored form) returning the result, with the
    entry buffer kept so analyze() has at least one allocation."""
    generator = ExtendedGenerator(Config(extended_prob=1), 'triton')
    program = generator.generate('arithmetic')
    program.body = Block()
    program.observations = []
    program.functions = []
    program.blocks = 2
    program.input_pattern = 'integer'
    program.runtime_cases = [(0, 1), (1, 15)]
    generator.serial, generator.buffers = 0, []
    generator.buffer(dtype, 16)
    program.buffers = generator.buffers
    builder = Builder(generator, program.body, [])
    x = builder.constant(Ty(dtype, (4, 4)), 0.25)
    y = builder.constant(Ty(dtype, (4, 4)), 0.5)
    z = builder.constant(Ty(dtype, (4, 4)), -0.125)
    t = builder.binary(inner_op, y, z)
    r = builder.binary('mul', x, t)
    program.body.returns = [r.name]
    program.validate()
    return program, r.name


def generated_with_match(backend, family='arithmetic', max_seed=40):
    """First seed in range(max_seed) whose generated program matches the
    identity pattern; deterministic because random is reseeded per iteration."""
    for seed in range(max_seed):
        random.seed(seed)
        program = ExtendedGenerator(Config(extended_prob=1), backend).generate(family)
        if identity_variant(program) is not None:
            return program
    raise AssertionError(f'No {backend} {family} program matched in {max_seed} seeds')


class TestIdentityVariant(unittest.TestCase):
    def test_mul_of_add_becomes_distributed_add(self):
        program, name = hand_built_program('add')
        transformed = identity_variant(program)
        self.assertIsNotNone(transformed)
        transformed.validate()
        nodes = transformed.body.operations
        self.assertEqual([n.op for n in nodes],
                         ['constant'] * 3 + ['add', 'mul', 'mul', 'add'])
        final = next(n for n in nodes if n.results[0].name == name)
        self.assertEqual(final.op, 'add')
        self.assertEqual([nodes[-3].results[0].name, nodes[-2].results[0].name], final.operands)
        self.assertTrue(all(n.op == 'mul' for n in nodes[-3:-1]))
        self.assertEqual(final.results[0].type, program.body.operations[-1].results[0].type)
        self.assertEqual(program_to_dict(program), program_to_dict(hand_built_program('add')[0]))

    def test_mul_of_sub_becomes_distributed_sub(self):
        program, name = hand_built_program('sub')
        transformed = identity_variant(program)
        self.assertIsNotNone(transformed)
        transformed.validate()
        final = next(n for n in transformed.body.operations if n.results[0].name == name)
        self.assertEqual(final.op, 'sub')
        self.assertEqual(len(final.operands), 2)

    def test_mirrored_factor_also_distributes(self):
        """mul(add(y, z), x) distributes too: the add/sub operand may sit in
        either position."""
        generator = ExtendedGenerator(Config(extended_prob=1), 'triton')
        program = generator.generate('arithmetic')
        program.body = Block()
        program.observations = []
        program.functions = []
        generator.serial, generator.buffers = 0, []
        generator.buffer('float32', 16)
        program.buffers = generator.buffers
        builder = Builder(generator, program.body, [])
        x = builder.constant(Ty('float32', (4, 4)), 0.25)
        y = builder.constant(Ty('float32', (4, 4)), 0.5)
        z = builder.constant(Ty('float32', (4, 4)), -0.125)
        t = builder.binary('add', y, z)
        r = builder.binary('mul', t, x)
        program.body.returns = [r.name]
        program.validate()
        transformed = identity_variant(program)
        self.assertIsNotNone(transformed)
        transformed.validate()
        final = next(n for n in transformed.body.operations if n.results[0].name == r.name)
        self.assertEqual(final.op, 'add')

    def test_integer_distribution_is_left_alone(self):
        # int32 a*(b+c) != a*b+a*c under wrapping overflow, so only float
        # patterns are rewritten.
        program, _ = hand_built_program('add', dtype='int32')
        self.assertIsNone(identity_variant(program))

    def test_matmul_families_never_transform(self):
        for backend in ('tilelang', 'triton'):
            for family in MATMUL_FAMILIES:
                with self.subTest(backend=backend, family=family):
                    random.seed(5)
                    program = ExtendedGenerator(Config(extended_prob=1), backend).generate(family)
                    self.assertIsNone(identity_variant(program))

    def test_generated_arithmetic_program_transforms_and_preserves_the_original(self):
        for backend in ('tilelang', 'triton'):
            with self.subTest(backend=backend):
                program = generated_with_match(backend)
                before = program_to_dict(program)
                transformed = identity_variant(program)
                self.assertIsNotNone(transformed)
                transformed.validate()
                self.assertEqual(program_to_dict(program), before)
                # Exactly two extra nodes, and the final result name survives.
                self.assertEqual(len(transformed.body.operations),
                                 len(program.body.operations) + 2)
                self.assertEqual(transformed.body.returns, program.body.returns)
                self.assertEqual(transformed.observations, program.observations)

    def test_intermediate_names_are_fresh_and_reference_stays_within_tolerance(self):
        for backend in ('tilelang', 'triton'):
            with self.subTest(backend=backend):
                program = generated_with_match(backend)
                raw = program.to_dict()
                transformed = identity_variant(program)
                distributed = transformed.to_dict()
                # The interpreter agrees with the original program: fp32
                # distributivity differs by ~1 ulp, far below this tolerance.
                inputs = extended_inputs(raw, 0, 'cpu')
                expected, _ = extended_reference(raw, inputs, 0, 3)
                got, _ = extended_reference(distributed, inputs, 0, 3)
                self.assertEqual(set(got), set(expected))
                for name, value in expected.items():
                    self.assertEqual(value.dtype, got[name].dtype)
                    self.assertTrue(torch.allclose(got[name], value, rtol=1e-4, atol=1e-4), name)


class TestIdentityVariantsAndEmission(unittest.TestCase):
    def test_identity_variant_is_gated_by_program_and_config(self):
        from src.backends import get_backend
        adapter = get_backend('triton')
        config = Config(extended_prob=1, extended_identity_pair=True,
                        extended_precision_pair=False, extended_config_depth=1)
        program = generated_with_match('triton')
        variants = adapter.extended_variants(program, config)
        identities = [v for v in variants if v[1].get('identity')]
        self.assertTrue(identities)
        self.assertEqual(len(identities), 1)
        self.assertEqual(identities[0][1]['num_warps'], 4)
        self.assertEqual(len({lower.name for lower, _ in variants}), len(variants))
        program.identity_pair = False
        program.validate()
        self.assertFalse(any(v[1].get('identity')
                             for v in adapter.extended_variants(program, config)))
        other = generated_with_match('triton')
        self.assertFalse(any(v[1].get('identity') for v in adapter.extended_variants(
            other, Config(extended_prob=1, extended_identity_pair=False,
                          extended_precision_pair=False))))
        # TileLang mirrors the variant on its own base configuration.
        tlang = get_backend('tilelang')
        program = generated_with_match('tilelang')
        variants = tlang.extended_variants(program, config)
        identities = [v for v in variants if v[1].get('identity')]
        self.assertTrue(identities)
        self.assertTrue(all(set(options) >= {'threads', 'stages', 'pass_configs', 'identity'}
                            for _, options in identities))

    def test_emission_ships_the_identity_reference(self):
        from src.backends import get_backend
        from src.workflow.oracle import Oracle
        config = Config(extended_prob=1, extended_precision_pair=False)
        program = generated_with_match('triton')
        expected = [extended_variant_label('triton', i, options)
                    for i, (_, options) in enumerate(get_backend('triton').extended_variants(program, config))
                    if options.get('identity')]
        self.assertTrue(expected)
        code = Oracle(config, 'triton')._emit_code(program)
        ast.parse(code)
        for label in expected:
            self.assertIn(f"'{label}':", code)
        self.assertIn('reference_programs=REFERENCE_PROGRAMS', code)


class TestIdentityDiagnostics(unittest.TestCase):
    def test_label_and_classification(self):
        self.assertEqual(extended_variant_label('triton', 0, {'identity': True}), 'triton_0_ident')
        self.assertEqual(extended_variant_label('tilelang', 3, {'identity': True}), 'tilelang_3_ident')
        self.assertEqual(extended_variant_label('triton', 2, {'num_warps': 4}), 'triton_2')
        message = ('RuntimeError: WRONG RESULT: identity:triton_4_ident:out_0:seed=0:steps=1:limit=3; '
                   'max_abs=1.5; index=7; actual=3.0; expected=4.5')
        self.assertEqual(classify_root_cause(message), 'algebraic_identity')
        # The precision prefix keeps its own rule.
        self.assertEqual(classify_root_cause(
            'RuntimeError: WRONG RESULT: precision:triton_4_prec:out_0:seed=0'), 'precision_mismatch')
        # The generic extended label still classifies as wrong_result.
        self.assertEqual(classify_root_cause(
            'RuntimeError: WRONG RESULT: triton_4:out_0:seed=0'), 'wrong_result')


if __name__ == '__main__':
    unittest.main()
