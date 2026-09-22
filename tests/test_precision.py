"""Unit tests for the Phase 2a accumulator-width sweep (precision pair).

The precision pair rewrites a program copy so every constant-accumulator
matmul accumulates in float16 (RC5 reachability): the transform must keep the
IR valid, leave the original untouched, insert casts for dtype-fixed
consumers, and the interpreter must model the per-k=16 MMA rounding that
hardware performs on an fp16 accumulator.
"""
import random
import unittest

import torch

from src.backends.common.diagnostics import classify_root_cause
from src.config import Config
from src.ir.extended import TensorType as Ty
from src.ir.serialization import program_to_dict
from src.workflow.emitter.extended_runtime import extended_inputs, extended_reference
from src.workflow.generator.extended import Builder, ExtendedGenerator
from src.workflow.generator.identities import extended_variant_label, precision_program

MATMUL_FAMILIES = ('shape_matmul', 'mixed')
MATMUL_LESS_FAMILIES = ('arithmetic', 'indexed_memory', 'control_calls')


def chunked_fp16_matmul_program():
    """Minimal fp16-accumulation program with k=32, so the interpreter must
    model two hardware rounding steps. The two k=16 chunks carry different
    products (2048.0 and ~0.8), which makes the chunked rounding observable:
    single-shot fp32 rounding yields 2050.0 per lane, chunked 2048.0."""
    generator = ExtendedGenerator(Config(extended_prob=1), 'triton')
    program = generator.generate('arithmetic')
    program.body.operations.clear()
    program.body.returns.clear()
    program.observations.clear()
    program.functions.clear()
    program.buffers.clear()
    generator.buffer('float32', 32)  # analyze() requires at least one buffer
    builder = Builder(generator, program.body, [])
    half, i32 = 'float16', 'int32'
    lhs_idx = builder.indices((16, 32), shuffled=False)
    col = builder.binary('mod', lhs_idx, builder.constant(Ty(i32), 32))
    lhs_mask = builder.binary('lt', col, builder.constant(Ty(i32), 16))
    lhs = builder.emit('select', [lhs_mask, builder.constant(Ty(half, (16, 32)), 0.5),
                                  builder.constant(Ty(half, (16, 32)), 0.5)],
                       [Ty(half, (16, 32))])
    rhs_idx = builder.indices((32, 16), shuffled=False)
    rhs_mask = builder.binary('lt', rhs_idx, builder.constant(Ty(i32), 256))
    rhs = builder.emit('select', [rhs_mask, builder.constant(Ty(half, (32, 16)), 256.0),
                                  builder.constant(Ty(half, (32, 16)), 0.1)],
                       [Ty(half, (32, 16))])
    acc = builder.constant(Ty(half, (16, 16)), 1.0)
    matmul = builder.emit('matmul', [lhs, rhs, acc], [Ty(half, (16, 16))])
    program.body.returns = [matmul.name]
    program.validate()
    return program, matmul.name


class TestPrecisionProgram(unittest.TestCase):
    def test_matmul_families_transform_and_validate_without_touching_the_original(self):
        for backend in ('tilelang', 'triton'):
            for family in MATMUL_FAMILIES:
                with self.subTest(backend=backend, family=family):
                    random.seed(3)
                    program = ExtendedGenerator(Config(extended_prob=1), backend).generate(family)
                    before = program_to_dict(program)
                    transformed = precision_program(program)
                    self.assertIsNotNone(transformed)
                    transformed.validate()
                    self.assertEqual(program_to_dict(program), before)
                    matmuls = [n for n in transformed.all_operations() if n.op == 'matmul']
                    self.assertTrue(matmuls)
                    self.assertTrue(all(n.results[0].type.dtype == 'float16' for n in matmuls))
                    # The shared accumulator constant itself is retyped.
                    self.assertTrue(any(
                        n.op == 'constant' and n.results[0].type.dtype == 'float16'
                        and n.results[0].name in {m.operands[2] for m in matmuls}
                        for n in transformed.all_operations()))
                    # The original keeps its fp32 accumulation.
                    self.assertFalse(any(
                        n.op == 'matmul' and n.results[0].type.dtype == 'float16'
                        for n in program.all_operations()))
                    # Dtype-fixed consumers (sub with the fp32 column, fp32
                    # reduce, stores) are repaired with explicit casts.
                    self.assertTrue(any(n.op == 'cast' for n in transformed.all_operations()))

    def test_matmul_less_families_have_no_precision_variant(self):
        for backend in ('tilelang', 'triton'):
            for family in MATMUL_LESS_FAMILIES:
                with self.subTest(backend=backend, family=family):
                    random.seed(5)
                    program = ExtendedGenerator(Config(extended_prob=1), backend).generate(family)
                    self.assertIsNone(precision_program(program))

    def test_precision_pair_program_flag_gates_the_sweep(self):
        from src.backends import get_backend
        program = ExtendedGenerator(Config(extended_prob=1), 'triton').generate('shape_matmul')
        program.precision_pair = False
        program.validate()
        variants = get_backend('triton').extended_variants(
            program, Config(extended_prob=1, extended_precision_pair=True))
        self.assertFalse(any(v[1].get('precision') or v[1].get('input_precision') for v in variants))


class TestFp16AccumulationReference(unittest.TestCase):
    def test_chunked_fp16_reference_matches_hardware_rounding_model(self):
        program, name = chunked_fp16_matmul_program()
        raw = program.to_dict()
        outputs, _ = extended_reference(raw, extended_inputs(raw, 0, 'cpu'), 0, 3)
        got = outputs[name][0]
        self.assertEqual(got.dtype, torch.float16)
        self.assertTrue(torch.all(got == got[0, 0]))
        chunked = torch.full((16, 16), 1.0, dtype=torch.float16).float()
        lhs = torch.full((16, 32), 0.5, dtype=torch.float16).float()
        rhs = torch.full((32, 16), 0.1, dtype=torch.float16).float()
        rhs[:16] = 256.0
        for start in range(0, 32, 16):
            chunked = (chunked + lhs[:, start:start + 16] @ rhs[start:start + 16, :]).to(torch.float16)
        single = (torch.full((16, 16), 1.0, dtype=torch.float16).float() + lhs @ rhs).to(torch.float16)
        self.assertTrue(torch.equal(got, chunked))
        # The two semantics genuinely disagree on this program, so the test
        # proves the interpreter rounds once per k=16 MMA step.
        self.assertFalse(torch.equal(chunked, single))


class TestPrecisionDiagnostics(unittest.TestCase):
    def test_precision_label_and_classification(self):
        self.assertEqual(extended_variant_label('triton', 4, {'precision': 'fp16'}), 'triton_4_prec')
        self.assertEqual(extended_variant_label('triton', 4, {'num_warps': 4}), 'triton_4')
        self.assertEqual(extended_variant_label('tilelang', 0, {'precision': 'fp16'}), 'tilelang_0_prec')
        message = ('RuntimeError: WRONG RESULT: precision:triton_4_prec:out_0:seed=0:steps=1:limit=3; '
                   'max_abs=1.5; index=7; actual=3.0; expected=4.5')
        self.assertEqual(classify_root_cause(message), 'precision_mismatch')
        # The generic extended label still classifies as wrong_result.
        self.assertEqual(classify_root_cause(
            'RuntimeError: WRONG RESULT: triton_4:out_0:seed=0:steps=1:limit=3'), 'wrong_result')


if __name__ == '__main__':
    unittest.main()
