"""New extended op surfaces: atomics, scalar FMA, shape primitives, int8 matmul.

Covers the analyze() accept/reject matrix, both DSL lowerings, the JSON-driven
CPU reference, the race/fusion-aware checkers, the per-(op, backend) instance
grids, serialization round trips, and the coverage audit tags.
"""
import ast
import json
import re
import unittest

import torch

from src.backends import get_backend
from src.config import Config
from src.ir.extended import TensorType as Ty, ExtendedProgram
from src.ir.serialization import program_from_dict, program_to_dict
from src.workflow.coverage_audit import program_capabilities
from src.workflow.emitter.extended_runtime import (extended_inputs, extended_reference,
                                                   extended_check_atomic, extended_check_fma)
from src.workflow.generator.extended import Builder, ExtendedGenerator
from src.workflow.generator.grids import (ATOMIC_GRID, FMA_GRID, SHAPE_OP_GRID, GridState)


def validated(gen, builder, observations=()):
    program = ExtendedProgram(builder.block, gen.buffers, observations=list(observations),
                              input_pattern='integer', family='hand')
    program.validate()
    return program


def emit(backend, program):
    code = get_backend(backend).make_emitter(Config()).emit(program)
    ast.parse(code)
    return code


def reference(program, steps=0, limit=1):
    encoded = json.loads(json.dumps(program.to_dict()))
    inputs = extended_inputs(encoded)
    return extended_reference(encoded, inputs, steps, limit)


class AnalyzeContractTests(unittest.TestCase):
    """Accept/reject matrix exercised through program.validate() -> analyze()."""

    def _loads(self, b, buf, count, ty):
        return [b.load(buf, (), b.constant(Ty('int32'), i), b.constant(Ty('bool'), True))
                for i in range(count)]

    def test_fma_accepts_three_scalar_floats(self):
        for backend in ('triton', 'tilelang'):
            for dtype in ('float16', 'float32'):
                gen = ExtendedGenerator(Config(extended_prob=1), backend)
                b = Builder(gen)
                buf = gen.buffer(dtype, 4)
                x, y, z = self._loads(b, buf, 3, Ty(dtype))
                result = b.emit('fma', [x, y, z], [Ty(dtype)])
                b.block.returns = [result.name]
                validated(gen, b)

    def test_fma_rejects_wrong_arity_dtype_and_shape(self):
        for backend in ('triton', 'tilelang'):
            gen = ExtendedGenerator(Config(extended_prob=1), backend)
            b = Builder(gen)
            buf = gen.buffer('float32', 4)
            x, y, z = self._loads(b, buf, 3, Ty('float32'))
            b.emit('fma', [x, y], [Ty('float32')])
            b.block.returns = [x.name]
            with self.assertRaisesRegex(ValueError, 'Invalid fma'):
                validated(gen, b)

            gen = ExtendedGenerator(Config(extended_prob=1), backend)
            b = Builder(gen)
            x, y = self._loads(b, gen.buffer('float32', 4), 2, Ty('float32'))
            half = b.load(gen.buffer('float16', 4), (), b.constant(Ty('int32'), 0),
                          b.constant(Ty('bool'), True))
            b.emit('fma', [x, y, half], [Ty('float32')])
            b.block.returns = [x.name]
            with self.assertRaisesRegex(ValueError, 'Invalid fma'):
                validated(gen, b)

            gen = ExtendedGenerator(Config(extended_prob=1), backend)
            b = Builder(gen)
            loaded = b.load(gen.buffer('float32', 8), (8,))
            b.emit('fma', [b.reduce(loaded, 0), b.reduce(loaded, 0), b.reduce(loaded, 0)], [Ty('float32')])
            b.block.returns = [loaded.name]
            validated(gen, b)  # scalars produced by a reduction are fine

            gen = ExtendedGenerator(Config(extended_prob=1), backend)
            b = Builder(gen)
            x, y, z = self._loads(b, gen.buffer('float32', 4), 3, Ty('float32'))
            b.emit('fma', [x, y, z], [Ty('float32', (4,))])
            b.block.returns = [x.name]
            with self.assertRaisesRegex(ValueError, 'Incorrect result types for fma'):
                validated(gen, b)

    def test_atomic_requires_scratch_buffer_of_supported_dtype(self):
        for role, dtype in (('input', 'int32'), ('scratch', 'bool'), ('scratch', 'int8')):
            gen = ExtendedGenerator(Config(extended_prob=1), 'triton')
            b = Builder(gen)
            buf = gen.buffer(dtype, 8, role)
            idx = b.indices((8,), shuffled=False)
            mask = b.constant(Ty('bool', (8,)), True)
            payload = b.emit('broadcast', [b.constant(Ty('int32'), 1)], [Ty('int32', (8,))])
            b.emit('atomic_add', [idx, mask, payload], buffer=buf.name)
            b.block.returns = [payload.name]
            with self.assertRaisesRegex(ValueError, 'Invalid atomic operation'):
                validated(gen, b)

    def test_atomic_rejects_wrong_operand_types_and_arity(self):
        cases = []
        # index must be int32
        gen = ExtendedGenerator(Config(extended_prob=1), 'triton')
        b = Builder(gen)
        buf = gen.buffer('int32', 8, 'scratch')
        idx = b.emit('cast', [b.indices((8,), shuffled=False)], [Ty('float32', (8,))])
        mask = b.constant(Ty('bool', (8,)), True)
        payload = b.emit('broadcast', [b.constant(Ty('int32'), 1)], [Ty('int32', (8,))])
        b.emit('atomic_add', [idx, mask, payload], buffer=buf.name)
        b.block.returns = [payload.name]
        with self.assertRaisesRegex(ValueError, 'Invalid atomic operation'):
            validated(gen, b)
        # mask must be bool of the index shape
        gen = ExtendedGenerator(Config(extended_prob=1), 'triton')
        b = Builder(gen)
        buf = gen.buffer('int32', 8, 'scratch')
        idx = b.indices((8,), shuffled=False)
        mask = b.constant(Ty('int32', (8,)), 1)
        payload = b.emit('broadcast', [b.constant(Ty('int32'), 1)], [Ty('int32', (8,))])
        b.emit('atomic_add', [idx, mask, payload], buffer=buf.name)
        b.block.returns = [payload.name]
        with self.assertRaisesRegex(ValueError, 'Invalid atomic operation'):
            validated(gen, b)
        # value must match the buffer dtype
        gen = ExtendedGenerator(Config(extended_prob=1), 'triton')
        b = Builder(gen)
        buf = gen.buffer('float32', 8, 'scratch')
        idx = b.indices((8,), shuffled=False)
        mask = b.constant(Ty('bool', (8,)), True)
        payload = b.emit('broadcast', [b.constant(Ty('int32'), 1)], [Ty('int32', (8,))])
        b.emit('atomic_add', [idx, mask, payload], buffer=buf.name)
        b.block.returns = [payload.name]
        with self.assertRaisesRegex(ValueError, 'Invalid atomic operation'):
            validated(gen, b)
        # two operands are not enough
        gen = ExtendedGenerator(Config(extended_prob=1), 'triton')
        b = Builder(gen)
        buf = gen.buffer('int32', 8, 'scratch')
        idx = b.indices((8,), shuffled=False)
        mask = b.constant(Ty('bool', (8,)), True)
        b.emit('atomic_add', [idx, mask], buffer=buf.name)
        b.block.returns = [idx.name]
        with self.assertRaisesRegex(ValueError, 'Invalid atomic operation'):
            validated(gen, b)

    def test_join_split_shape_rules(self):
        # 2-D join operands are rejected; the result type is a placeholder
        # because the 3-D join output is itself unrepresentable.
        gen = ExtendedGenerator(Config(extended_prob=1), 'triton')
        b = Builder(gen)
        left = b.load(gen.buffer('float32', 16), (8, 2))
        right = b.load(gen.buffer('float32', 16), (8, 2))
        b.emit('join', [left, right], [Ty('float32', (8, 2))])
        b.block.returns = [left.name]
        with self.assertRaisesRegex(ValueError, 'Join requires 1-D operands'):
            validated(gen, b)
        # split requires a trailing extent of exactly 2.
        gen = ExtendedGenerator(Config(extended_prob=1), 'triton')
        b = Builder(gen)
        loaded = b.load(gen.buffer('float32', 16), (8, 4))
        b.emit('split', [loaded], [Ty('float32', (8,)), Ty('float32', (8,))])
        b.block.returns = [loaded.name]
        with self.assertRaisesRegex(ValueError, 'Invalid split'):
            validated(gen, b)
        # split must produce exactly two results.
        gen = ExtendedGenerator(Config(extended_prob=1), 'triton')
        b = Builder(gen)
        loaded = b.load(gen.buffer('float32', 16), (8, 2))
        b.emit('split', [loaded], [Ty('float32', (8,))])
        b.block.returns = [loaded.name]
        with self.assertRaisesRegex(ValueError, 'Invalid split'):
            validated(gen, b)

    def test_int8_matmul_bounds_and_dtype_rules(self):
        def build(m, n, k, a_dtype='int8', b_dtype='int8', acc_dtype='int32', message='Invalid int8 matmul'):
            gen = ExtendedGenerator(Config(extended_prob=1), 'triton')
            b = Builder(gen)
            a = b.load(gen.buffer(a_dtype, m * k), (m, k))
            c = b.load(gen.buffer(b_dtype, k * n), (k, n))
            acc = b.constant(Ty(acc_dtype, (m, n)), 0)
            product = b.emit('matmul', [a, c, acc], [acc.type])
            b.block.returns = [product.name]
            with self.assertRaisesRegex(ValueError, message):
                validated(gen, b)

        # The minimum valid int8 gemm shape is accepted.
        gen = ExtendedGenerator(Config(extended_prob=1), 'triton')
        b = Builder(gen)
        a = b.load(gen.buffer('int8', 16 * 32), (16, 32))
        c = b.load(gen.buffer('int8', 32 * 16), (32, 16))
        acc = b.constant(Ty('int32', (16, 16)), 0)
        product = b.emit('matmul', [a, c, acc], [acc.type])
        b.block.returns = [product.name]
        validated(gen, b)

        build(16, 16, 16)  # K too small for the pipelined s8 path
        build(8, 16, 32)   # M below the MMA minimum
        build(16, 16, 32, b_dtype='float16', message='Invalid matmul operands')
        build(16, 16, 32, acc_dtype='float32')
        build(16, 16, 32, a_dtype='float16', message='Invalid matmul operands')

    def test_int8_arithmetic_and_reduction_are_rejected(self):
        gen = ExtendedGenerator(Config(extended_prob=1), 'triton')
        b = Builder(gen)
        x = b.cast(b.load(gen.buffer('int32', 8), (8,)), 'int8')
        y = b.cast(b.load(gen.buffer('int32', 8), (8,)), 'int8')
        b.binary('add', x, y)
        b.block.returns = [x.name]
        with self.assertRaisesRegex(ValueError, 'int8 arithmetic is unsupported'):
            validated(gen, b)

        gen = ExtendedGenerator(Config(extended_prob=1), 'triton')
        b = Builder(gen)
        x = b.cast(b.load(gen.buffer('int32', 8), (8,)), 'int8')
        b.reduce(x, 0)
        b.block.returns = [x.name]
        with self.assertRaisesRegex(ValueError, 'Invalid reduction'):
            validated(gen, b)


class LoweringSourceTests(unittest.TestCase):
    def _fma_program(self, backend):
        gen = ExtendedGenerator(Config(extended_prob=1), backend)
        b = Builder(gen)
        buf = gen.buffer('float32', 4)
        loads = [b.load(buf, (), b.constant(Ty('int32'), i), b.constant(Ty('bool'), True))
                 for i in range(3)]
        result = b.emit('fma', loads, [Ty('float32')])
        b.block.returns = [result.name]
        return validated(gen, b)

    def _atomic_program(self, backend, fn='add', dtype='int32'):
        gen = ExtendedGenerator(Config(extended_prob=1), backend)
        b = Builder(gen)
        buf = gen.buffer(dtype, 8, 'scratch')
        idx = b.indices((8,), shuffled=False)
        mask = b.constant(Ty('bool', (8,)), True)
        payload = b.emit('broadcast', [b.constant(Ty(dtype), 1)], [Ty(dtype, (8,))])
        b.emit(f'atomic_{fn}', [idx, mask, payload], buffer=buf.name)
        b.block.returns = [payload.name]
        return validated(gen, b)

    def _shape_program(self, backend, ops):
        gen = ExtendedGenerator(Config(extended_prob=1), backend)
        b = Builder(gen)
        returns = []
        left = b.load(gen.buffer('float32', 8), (8,))
        right = b.load(gen.buffer('float32', 8), (8,))
        if 'flip' in ops:
            returns.append(b.emit('flip', [left], [left.type]))
        if 'join' in ops:
            returns.append(b.emit('join', [left, right], [Ty('float32', (8, 2))]))
        if 'split' in ops:
            source = b.load(gen.buffer('float32', 16), (8, 2))
            first, second = b.emit('split', [source], [Ty('float32', (8,)), Ty('float32', (8,))])
            returns += [first, second]
        if 'interleave' in ops:
            returns.append(b.emit('interleave', [left, right], [Ty('float32', (16,))]))
        b.block.returns = [v.name for v in returns]
        return validated(gen, b)

    def test_triton_fma_is_fused(self):
        code = emit('triton', self._fma_program('triton'))
        self.assertIn('tl.fma(', code)

    def test_tilelang_fma_is_a_plain_contraction(self):
        code = emit('tilelang', self._fma_program('tilelang'))
        self.assertIsNotNone(re.search(r'e\d+\[0\] = T\.cast\(\(e\d+\[0\] \* e\d+\[0\] \+ e\d+\[0\]\)', code))

    def test_triton_atomic_emits_masked_atomic_call(self):
        code = emit('triton', self._atomic_program('triton'))
        self.assertIn('tl.atomic_add(', code)
        self.assertIn('bid *', code)
        # The mask argument guards both the predicate and the bounds.
        self.assertIn('& (e', code)

    def test_tilelang_atomic_emits_guarded_atomic_call(self):
        code = emit('tilelang', self._atomic_program('tilelang', 'max'))
        self.assertIn('T.atomic_max(', code)
        self.assertIn('memory_order="relaxed"', code)
        # Masked lanes never reach the atomic: the guard must be a branch.
        self.assertIn('if (', code)

    def test_triton_shape_primitives(self):
        code = emit('triton', self._shape_program('triton', ('flip', 'join', 'split', 'interleave')))
        self.assertIn('tl.flip(', code)
        self.assertIn('tl.join(', code)
        self.assertIn('= tl.split(', code)
        self.assertIn('tl.interleave(', code)

    def test_tilelang_flip_reverses_the_minor_index(self):
        gen = ExtendedGenerator(Config(extended_prob=1), 'tilelang')
        b = Builder(gen)
        loaded = b.load(gen.buffer('int32', 8), (8,))
        flipped = b.emit('flip', [loaded], [loaded.type])
        b.block.returns = [flipped.name]
        code = emit('tilelang', validated(gen, b))
        self.assertIn('[7 - i]', code)

    def test_tilelang_rejects_triton_only_shape_ops(self):
        # split fails at index synthesis (its two outputs have different
        # ranks than the source), join/interleave at the op dispatch itself.
        for op, message in (('join', 'Unsupported TileLang operation: join'),
                            ('interleave', 'Unsupported TileLang operation: interleave'),
                            ('split', 'Missing indices in TileLang lowering')):
            with self.assertRaisesRegex(ValueError, re.escape(message)):
                emit('tilelang', self._shape_program('tilelang', (op,)))

    def test_int8_matmul_lowering(self):
        gen = ExtendedGenerator(Config(extended_prob=1), 'triton')
        b = Builder(gen)
        a = b.load(gen.buffer('int8', 16 * 32), (16, 32))
        c = b.load(gen.buffer('int8', 32 * 16), (32, 16))
        acc = b.constant(Ty('int32', (16, 16)), 0)
        product = b.emit('matmul', [a, c, acc], [acc.type])
        b.block.returns = [product.name]
        program = validated(gen, b)
        triton = emit('triton', program)
        self.assertIn('tl.dot(', triton)
        self.assertIn('out_dtype=tl.int32', triton)
        self.assertNotIn('input_precision', triton)
        tilelang = emit('tilelang', program)
        self.assertIn('T.gemm(', tilelang)
        self.assertIn('T.alloc_shared((16, 32), "int8")', tilelang)
        self.assertIn('T.alloc_fragment((16, 16), "int32")', tilelang)


class ReferenceConsistencyTests(unittest.TestCase):
    """extended_reference against a manual torch computation of each new op."""

    def test_fma_reference_matches_double_evaluation(self):
        gen = ExtendedGenerator(Config(extended_prob=1), 'triton')
        b = Builder(gen)
        buf = gen.buffer('float32', 4)
        loads = [b.load(buf, (), b.constant(Ty('int32'), i), b.constant(Ty('bool'), True))
                 for i in range(3)]
        result = b.emit('fma', loads, [Ty('float32')])
        b.block.returns = [result.name]
        outputs, memory = reference(validated(gen, b))
        x, y, z = (memory[buf.name][0, 16 + i] for i in range(3))
        expected = (x.double() * y.double() + z.double()).to(torch.float32)
        self.assertTrue(torch.equal(outputs[result.name][0], expected))

    def test_flip_reference(self):
        gen = ExtendedGenerator(Config(extended_prob=1), 'tilelang')
        b = Builder(gen)
        buf = gen.buffer('float32', 8)
        loaded = b.load(buf, (8,), b.indices((8,), shuffled=False))
        flipped = b.emit('flip', [loaded], [loaded.type])
        b.block.returns = [flipped.name]
        outputs, memory = reference(validated(gen, b))
        expected = torch.flip(memory[buf.name][0, 16:24], [-1])
        self.assertTrue(torch.equal(outputs[flipped.name][0], expected))

    def test_join_split_interleave_reference(self):
        gen = ExtendedGenerator(Config(extended_prob=1), 'triton')
        b = Builder(gen)
        left_buf, right_buf = gen.buffer('float32', 8), gen.buffer('float32', 8)
        left = b.load(left_buf, (8,), b.indices((8,), shuffled=False))
        right = b.load(right_buf, (8,), b.indices((8,), shuffled=False))
        joined = b.emit('join', [left, right], [Ty('float32', (8, 2))])
        first, second = b.emit('split', [joined], [Ty('float32', (8,)), Ty('float32', (8,))])
        mixed = b.emit('interleave', [left, right], [Ty('float32', (16,))])
        b.block.returns = [joined.name, first.name, second.name, mixed.name]
        outputs, memory = reference(validated(gen, b))
        l, r = memory[left_buf.name][0, 16:24], memory[right_buf.name][0, 16:24]
        self.assertTrue(torch.equal(outputs[joined.name][0], torch.stack((l, r), dim=-1)))
        self.assertTrue(torch.equal(outputs[first.name][0], l))
        self.assertTrue(torch.equal(outputs[second.name][0], r))
        self.assertTrue(torch.equal(outputs[mixed.name][0], torch.stack((l, r), dim=-1).reshape(16)))

    def test_int8_matmul_reference_is_exact(self):
        gen = ExtendedGenerator(Config(extended_prob=1), 'triton')
        b = Builder(gen)
        a_buf, c_buf = gen.buffer('int8', 16 * 32), gen.buffer('int8', 32 * 16)
        a = b.load(a_buf, (16, 32), b.indices((16, 32), shuffled=False))
        c = b.load(c_buf, (32, 16), b.indices((32, 16), shuffled=False))
        acc = b.constant(Ty('int32', (16, 16)), 0)
        product = b.emit('matmul', [a, c, acc], [acc.type])
        b.block.returns = [product.name]
        outputs, memory = reference(validated(gen, b))
        left = memory[a_buf.name][0, 16:16 + 16 * 32].reshape(16, 32).to(torch.int32)
        right = memory[c_buf.name][0, 16:16 + 32 * 16].reshape(32, 16).to(torch.int32)
        self.assertTrue(torch.equal(outputs[product.name][0], left @ right))

    def test_atomic_reference_scatter_reduces_with_include_self(self):
        gen = ExtendedGenerator(Config(extended_prob=1), 'triton')
        b = Builder(gen)
        scratch = gen.buffer('int32', 8, 'scratch')
        idx = b.indices((8,), shuffled=False)
        mask = b.constant(Ty('bool', (8,)), True)
        payload = b.emit('broadcast', [b.constant(Ty('int32'), 1)], [Ty('int32', (8,))])
        b.emit('atomic_add', [idx, mask, payload], buffer=scratch.name)
        b.block.returns = [payload.name]
        _, memory = reference(validated(gen, b))
        # The int32 scratch input pattern initializes the payload to 11; the
        # race folds it in exactly like the hardware.
        expected = torch.full((8,), 12, dtype=torch.int32)
        self.assertTrue(torch.equal(memory[scratch.name][0, 16:24], expected))


class CheckerTests(unittest.TestCase):
    def test_check_fma_accepts_fused_and_unfused(self):
        x, y, z = (torch.tensor([v], dtype=torch.float32) for v in (0.125, 0.25, -0.125))
        expected = (x.double() * y.double() + z.double()).to(torch.float32)
        unfused = x * y + z
        fused = torch.addcmul(z, x, y)
        extended_check_fma(fused, expected, 'fma:fused')
        extended_check_fma(unfused, expected, 'fma:unfused', matmul=True)  # flag is accepted, ignored
        with self.assertRaisesRegex(RuntimeError, 'WRONG RESULT'):
            extended_check_fma(unfused + 0.01, expected, 'fma:forged')

    def test_check_fma_accepts_fp16_fused_against_fp32_reference(self):
        x, y, z = (torch.tensor([v], dtype=torch.float16) for v in (0.5, 1.0, -0.25))
        expected = (x.double() * y.double() + z.double()).to(torch.float16)
        fused = torch.addcmul(z, x, y)
        extended_check_fma(fused, expected, 'fma:fp16')

    def test_check_atomic_int32_is_exact(self):
        expected = torch.tensor([12, 12, 11], dtype=torch.int32)
        extended_check_atomic(expected.clone(), expected, 'atomic:int32', 'add')
        with self.assertRaisesRegex(RuntimeError, 'WRONG RESULT'):
            extended_check_atomic(torch.tensor([12, 13, 11], dtype=torch.int32), expected, 'atomic:int32', 'add')

    def test_check_atomic_float_max_normalizes_signed_zero(self):
        actual = torch.tensor([1.0, -0.0, 3.0])
        expected = torch.tensor([1.0, 0.0, 3.0])
        extended_check_atomic(actual, expected, 'atomic:max', 'max')
        with self.assertRaisesRegex(RuntimeError, 'WRONG RESULT'):
            extended_check_atomic(torch.tensor([1.0, 0.0, 2.0]), expected, 'atomic:max', 'max')

    def test_check_atomic_float_add_tolerates_reorder_but_not_lost_contribution(self):
        values = torch.tensor([0.125, -0.25, 0.5, 0.125], dtype=torch.float32)
        exact = torch.sum(values.double()).to(torch.float32)
        reordered = values[2] + values[0] + values[3] + values[1]
        extended_check_atomic(reordered, exact, 'atomic:add', 'add')
        with self.assertRaisesRegex(RuntimeError, 'WRONG RESULT'):
            extended_check_atomic(torch.tensor([0.125]), exact, 'atomic:add', 'add')


class GridTests(unittest.TestCase):
    def test_round_robin_visits_each_cell_once_per_round(self):
        state = GridState()
        grid = ATOMIC_GRID
        seen = [state.next_cell('atomic', 'triton', grid) for _ in range(2 * len(grid))]
        self.assertEqual(seen[:len(grid)], list(grid))
        self.assertEqual(seen[len(grid):], list(grid))

    def test_cursors_are_per_op_and_backend(self):
        state = GridState()
        grid = FMA_GRID
        self.assertEqual(state.next_cell('fma', 'triton', grid), grid[0])
        self.assertEqual(state.next_cell('fma', 'tilelang', grid), grid[0])
        self.assertEqual(state.next_cell('fma', 'triton', grid), grid[1])
        self.assertEqual(state.next_cell('fma', 'tilelang', grid), grid[1])

    def test_save_load_continues_the_sequence(self):
        state = GridState()
        grid = SHAPE_OP_GRID['join']
        state.next_cell('shape_join', 'triton', grid)
        restored = GridState()
        restored.load(state.save())
        self.assertEqual(restored.next_cell('shape_join', 'triton', grid), grid[1])
        state.next_cell('shape_join', 'triton', grid)
        self.assertEqual(restored.save(), state.save())

    def test_generator_consumes_grid_cells_when_attached(self):
        gen = ExtendedGenerator(Config(extended_prob=1), 'triton', grids=GridState())
        b = Builder(gen)
        value = b.constant(Ty('float32', (8,)), 0.125)
        gen.atomic(b, value, fn='add')
        scratch = [buf for buf in gen.buffers if buf.role == 'scratch'][0]
        self.assertEqual(scratch.dtype, ATOMIC_GRID[0]['dtype'])
        self.assertEqual(gen.grids.cursors[('atomic', 'triton')], 1)


class SerializationAndAuditTests(unittest.TestCase):
    def test_new_op_programs_serialize_round_trip(self):
        gen = ExtendedGenerator(Config(extended_prob=1), 'triton')
        b = Builder(gen)
        buf = gen.buffer('float32', 8)
        flipped = b.emit('flip', [b.load(buf, (8,))], [Ty('float32', (8,))])
        scratch = gen.buffer('int32', 8, 'scratch')
        idx = b.indices((8,), shuffled=False)
        mask = b.constant(Ty('bool', (8,)), True)
        payload = b.emit('broadcast', [b.constant(Ty('int32'), 1)], [Ty('int32', (8,))])
        b.emit('atomic_max', [idx, mask, payload], buffer=scratch.name)
        source = gen.buffer('float32', 4)
        loads = [b.load(source, (), b.constant(Ty('int32'), i), b.constant(Ty('bool'), True))
                 for i in range(3)]
        fused = b.emit('fma', loads, [Ty('float32')])
        b.block.returns = [flipped.name, fused.name]
        program = validated(gen, b)
        restored = program_from_dict(json.loads(json.dumps(program_to_dict(program))))
        self.assertEqual(program.to_dict(), restored.to_dict())
        ast.parse(emit('triton', restored))

    def test_coverage_audit_flags_each_new_op_surface(self):
        gen = ExtendedGenerator(Config(extended_prob=1), 'triton')
        b = Builder(gen)
        buf = gen.buffer('float32', 8)
        loaded = b.load(buf, (8,))
        flipped = b.emit('flip', [loaded], [loaded.type])
        right = b.load(gen.buffer('float32', 8), (8,))
        joined = b.emit('join', [loaded, right], [Ty('float32', (8, 2))])
        first, second = b.emit('split', [joined], [Ty('float32', (8,)), Ty('float32', (8,))])
        scratch = gen.buffer('int32', 8, 'scratch')
        idx = b.indices((8,), shuffled=False)
        mask = b.constant(Ty('bool', (8,)), True)
        payload = b.emit('broadcast', [b.constant(Ty('int32'), 1)], [Ty('int32', (8,))])
        b.emit('atomic_add', [idx, mask, payload], buffer=scratch.name)
        source = gen.buffer('float32', 4)
        loads = [b.load(source, (), b.constant(Ty('int32'), i), b.constant(Ty('bool'), True))
                 for i in range(3)]
        fused = b.emit('fma', loads, [Ty('float32')])
        a = b.load(gen.buffer('int8', 16 * 32), (16, 32))
        c = b.load(gen.buffer('int8', 32 * 16), (32, 16))
        acc = b.constant(Ty('int32', (16, 16)), 0)
        product = b.emit('matmul', [a, c, acc], [acc.type])
        b.block.returns = [flipped.name, first.name, second.name, fused.name, product.name]
        capabilities = program_capabilities(validated(gen, b))
        for tag in ('global_atomics', 'scalar_fma', 'shape_join_split',
                    'shape_flip_interleave', 'int8_matmul'):
            self.assertIn(tag, capabilities)


if __name__ == '__main__':
    unittest.main()
