"""Rank-2 operands and drawn attributes (axis, direction, order, k) of DSL calls."""
import ast
import functools
import json
import random
import unittest
from unittest.mock import patch

import torch

from src.backends import get_backend
from src.config import Config
from src.ir.extended import ExtendedProgram, TensorType as Ty
from src.workflow.emitter.extended_runtime import extended_inputs, extended_reference
from src.workflow.extended_feedback import extended_features
from src.workflow.generator import dsl_extend
from src.workflow.generator.dsl_extend import (MATRIX_OPS, eligible_ops, extend_passed, loop_target,
                                               respell_target, target_attributes, target_attribute_cells)
from src.workflow.generator.extended import Builder, ExtendedGenerator
from src.workflow.generator.grids import GridState


def loaded(backend, dtype, shape):
    """A builder whose block loads one row-major buffer of `shape`."""
    gen = ExtendedGenerator(Config(extended_prob=1), backend)
    builder = Builder(gen)
    buf = gen.buffer(dtype, shape[0] * shape[1] if len(shape) == 2 else shape[0])
    return gen, builder, buf, builder.load(buf, shape, builder.indices(shape, shuffled=False))


def target_program(backend, dtype, calls, shape=(4, 8)):
    """Apply each (op, result type, attrs) to the loaded value; validated."""
    gen, builder, buf, source = loaded(backend, dtype, shape)
    results = [builder.emit(op, [source], [ty], **attrs) for op, ty, attrs in calls]
    builder.block.returns = [v.name for v in results]
    program = ExtendedProgram(builder.block, gen.buffers, input_pattern='integer', family='hand')
    program.validate()
    return program, buf, source, results


def encoded(program):
    return json.loads(json.dumps(program.to_dict()))


def run(program, buf, shape=(4, 8)):
    """Block 0 of every watched value, and the loaded input of block 0."""
    program = encoded(program)
    outputs, memory = extended_reference(program, extended_inputs(program, seed=3), 0, 1)
    size = shape[0] * shape[1]
    return ({name: value[0] for name, value in outputs.items()},
            memory[buf.name][0, 16:16 + size].reshape(shape))


def emit(backend, program):
    code = get_backend(backend).make_emitter(Config()).emit(program)
    ast.parse(code)
    return code


def attribute_features(program):
    return {f for f in extended_features(program) if f.startswith('["attribute"')}


class AttributeContractTests(unittest.TestCase):
    def test_softmax_row_cells_and_saved_program_broadcast(self):
        ty = Ty('float32', (4, 8))
        with patch.object(dsl_extend, '_accepts', return_value=True):
            cells = target_attribute_cells('softmax', ty, 'triton')
            self.assertEqual({cell['axis'] for cell in cells}, {-2, -1, 0, 1})
            self.assertTrue(all(cell['keep_dims'] for cell in cells if cell['axis'] % 2 == 1))
            for _ in range(100):
                self.assertTrue(target_attributes('softmax', ty, 'triton', axes=[1])['keep_dims'])
        for shape in ((4, 8), (8, 8)):
            for axis in (1, -1):
                program, _, source, _ = target_program('triton', 'float32', [
                    ('softmax', Ty('float32', shape), {'axis': axis, 'keep_dims': False})], shape)
                self.assertIn(f'tl.softmax({source.name}, dim={axis}, keep_dims=True)', emit('triton', program))

    def test_histogram_reference_drops_out_of_range_values(self):
        program, buf, _, out = target_program('triton', 'int32', [
            ('histogram', Ty('int32', (16,)), {})], shape=(8,))
        data = encoded(program)
        for values in ([-7, -1, 0, 1, 15, 16, 480, 0], [-1] * 8):
            memory = extended_inputs(data, seed=3)
            memory[buf.name][:, 16:24] = torch.tensor(values)
            outputs, _ = extended_reference(data, memory, 0, 1)
            expected = torch.tensor([values.count(i) for i in range(16)], dtype=torch.int32)
            self.assertTrue(torch.equal(outputs[out[0].name][0], expected))

    def test_ir_accepts_axis_spellings_and_rejects_invalid_attributes(self):
        target_program('triton', 'float32', [
            ('scan_sum', Ty('float32', (4, 8)), {'axis': -2, 'reverse': True}),
            ('softmax', Ty('float32', (4, 8)), {'axis': 1, 'keep_dims': False}),
            ('topk', Ty('float32', (4, 8)), {'k': 8, 'descending': False}),
            ('reduce_absmax', Ty('float32', (8,)), {'axis': 0})])
        target_program('tilelang', 'int32', [('reduce_bitxor', Ty('int32', (4,)), {'axis': -1}),
                                             ('argmax', Ty('int32', (8,)), {'axis': 0})])
        for op, ty, attrs, dtype in (
                ('scan_sum', Ty('float32', (4, 8)), {'axis': 2}, 'float32'),
                ('scan_max', Ty('float32', (4, 8)), {'axis': 0, 'reverse': 1}, 'float32'),
                ('sort', Ty('float32', (4, 8)), {'descending': 'yes'}, 'float32'),
                ('softmax', Ty('float32', (4, 8)), {'axis': True}, 'float32'),
                ('topk', Ty('float32', (4, 16)), {'k': 16}, 'float32'),
                ('topk', Ty('float32', (4, 4)), {'k': 3}, 'float32'),
                ('argmin', Ty('int32', (4,)), {'axis': 0}, 'int32'),
                ('xor_sum', Ty('int32', ()), {'axis': 0}, 'int32'),
                ('histogram', Ty('int32', (16,)), {}, 'int32')):
            with self.subTest(op=op, attrs=attrs), self.assertRaises(ValueError):
                target_program('triton', dtype, [(op, ty, attrs)])
        # gather keeps its 1-D contract.
        gen, builder, _, source = loaded('triton', 'float32', (4, 8))
        out = builder.emit('gather', [source, builder.indices((4, 8))], [Ty('float32', (4, 8))])
        builder.block.returns = [out.name]
        with self.assertRaises(ValueError):
            ExtendedProgram(builder.block, gen.buffers, family='hand').validate()

    def test_reference_follows_axis_direction_and_order(self):
        program, buf, _, out = target_program('triton', 'float32', [
            ('scan_sum', Ty('float32', (4, 8)), {'axis': -2, 'reverse': True}),
            ('scan_max', Ty('float32', (4, 8)), {'axis': 1, 'reverse': True}),
            ('scan_product', Ty('float32', (4, 8)), {'axis': 0}),
            ('sort', Ty('float32', (4, 8)), {'descending': True}),
            ('softmax', Ty('float32', (4, 8)), {'axis': 0, 'keep_dims': True}),
            ('topk', Ty('float32', (4, 2)), {'k': 2, 'descending': False}),
            ('reduce_abssum', Ty('float32', (8,)), {'axis': 0}),
            ('reduce_absmax', Ty('float32', (4,)), {'axis': -1})])
        outputs, x = run(program, buf)
        rows, cols = x.shape
        suffix_sum = torch.stack([x[i:].sum(0) for i in range(rows)])
        suffix_max = torch.stack([x[:, j:].max(1).values for j in range(cols)], dim=1)
        prefix_product = torch.stack([x[:i + 1].prod(0) for i in range(rows)])
        exp = torch.exp(x - x.max(0, keepdim=True).values)
        expected = [suffix_sum, suffix_max, prefix_product, torch.sort(x, -1, descending=True).values,
                    exp / exp.sum(0, keepdim=True), torch.sort(x, -1).values[:, :2],
                    x.abs().sum(0), x.abs().max(1).values]
        for value, wanted in zip(out, expected):
            with self.subTest(op=value.name):
                torch.testing.assert_close(outputs[value.name], wanted)

    def test_integer_reductions_follow_axis_and_first_index_ties(self):
        program, buf, _, out = target_program('triton', 'int32', [
            ('argmax', Ty('int32', (8,)), {'axis': 0}),
            ('argmin', Ty('int32', (4,)), {'axis': -1}),
            ('xor_sum', Ty('int32', (8,)), {'axis': -2}),
            ('reduce_bitand', Ty('int32', (4,)), {'axis': 1}),
            ('reduce_bitor', Ty('int32', (8,)), {'axis': 0})])
        outputs, x = run(program, buf)
        # Inputs in [-7, 7] tie often; the first extreme index wins.
        def first(values, better):
            values = values.tolist()
            return functools.reduce(lambda best, i: i if better(values[i], values[best]) else best,
                                    range(len(values)), 0)
        columns, rows = x.t(), x
        expected = [torch.tensor([first(c, int.__gt__) for c in columns], dtype=torch.int32),
                    torch.tensor([first(r, int.__lt__) for r in rows], dtype=torch.int32),
                    functools.reduce(torch.bitwise_xor, rows),
                    functools.reduce(torch.bitwise_and, columns),
                    functools.reduce(torch.bitwise_or, rows)]
        self.assertTrue(any(len(set(c.tolist())) < len(c) for c in columns))
        for value, wanted in zip(out, expected):
            with self.subTest(op=value.name):
                self.assertTrue(torch.equal(outputs[value.name], wanted))

    def test_lowerings_print_the_spelled_attributes(self):
        program, _, source, _ = target_program('triton', 'float32', [
            ('scan_sum', Ty('float32', (4, 8)), {'axis': -2, 'reverse': True}),
            ('softmax', Ty('float32', (4, 8)), {'axis': 1, 'keep_dims': True}),
            ('softmax', Ty('float32', (4, 8)), {'axis': 0, 'keep_dims': False}),
            ('sort', Ty('float32', (4, 8)), {'descending': True}),
            ('topk', Ty('float32', (4, 2)), {'k': 2, 'descending': False})])
        code = emit('triton', program)
        x = source.name
        for text in (f'tl.cumsum({x}, -2, reverse=True))', f'tl.softmax({x}, dim=1, keep_dims=True))',
                     f'tl.softmax({x}, 0))', f'tl.sort({x}, descending=True))',
                     # Triton 3.8 rejects dim=-1 for topk and sort.
                     f'tl.topk({x}, 2, dim=1, descending=False))'):
            self.assertIn(text, code)
        program, _, source, out = target_program('tilelang', 'float32', [
            ('scan_max', Ty('float32', (4, 8)), {'axis': -1, 'reverse': True}),
            ('reduce_abssum', Ty('float32', (8,)), {'axis': -2}),
            ('reduce_absmax', Ty('float32', (4,)), {'axis': 1})])
        code = emit('tilelang', program)
        scan, column, row = (v.name for v in out)
        self.assertIn(f'T.cummax({source.name}, {scan}, dim=-1, reverse=True)\n', code)
        # A column reduction stages through shared memory like reduce.
        self.assertIn(f'T.reduce_abssum({column}_wide, {column}, dim=-2)\n', code)
        self.assertIn(f'{column}_reduce_shared = T.alloc_shared', code)
        self.assertIn(f'T.reduce_absmax({row}_wide, {row}, dim=1)\n', code)
        self.assertNotIn(f'{row}_reduce_shared', code)

    def test_absent_attributes_keep_the_historical_spelling(self):
        program, _, source, out = target_program('triton', 'float32', [
            ('scan_sum', Ty('float32', (16,)), {}), ('sort', Ty('float32', (16,)), {}),
            ('softmax', Ty('float32', (16,)), {}), ('topk', Ty('float32', (4,)), {'k': 4})],
            shape=(16,))
        code = emit('triton', program)
        x = source.name
        for text in (f'tl.cumsum({x}, 0))', f'tl.sort({x}, descending=False))',
                     f'tl.softmax({x}, 0))', f'tl.topk({x}, 4, dim=0))'):
            self.assertIn(text, code)
        program, _, source, out = target_program('tilelang', 'float32', [
            ('scan_sum', Ty('float32', (16,)), {}), ('reduce_abssum', Ty('float32', ()), {})],
            shape=(16,))
        code = emit('tilelang', program)
        scan, total = (v.name for v in out)
        self.assertIn(f'T.cumsum({source.name}, {scan}, dim=0)\n', code)
        self.assertIn(f'T.reduce_abssum({total}_wide, {total}, dim=0)\n', code)
        program, _, _, (bits,) = target_program('tilelang', 'int32', [
            ('reduce_bitand', Ty('int32', ()), {})], shape=(16,))
        self.assertIn(f'T.reduce_bitand({bits.name}_wide, {bits.name}, dim=0, clear=True)\n',
                      emit('tilelang', program))


class AttributeDrawTests(unittest.TestCase):
    def setUp(self):
        state = random.getstate()
        self.addCleanup(random.setstate, state)
        random.seed(5)

    def test_draws_cover_spellings_within_the_installed_front_end(self):
        matrix = Ty('float32', (4, 8))
        with patch.object(dsl_extend, '_accepts', return_value=True):
            draws = {op: [target_attributes(op, matrix, 'triton') for _ in range(64)]
                     for op in ('scan_sum', 'softmax', 'argmax', 'topk')}
        for op in ('scan_sum', 'softmax', 'argmax'):
            self.assertEqual({d['axis'] for d in draws[op]}, {-2, -1, 0, 1})
        self.assertEqual({d['reverse'] for d in draws['scan_sum']}, {False, True})
        self.assertEqual({d['keep_dims'] for d in draws['softmax']}, {False, True})
        self.assertEqual({d['descending'] for d in draws['topk']}, {False, True})
        self.assertEqual({d['k'] for d in draws['topk']}, {2, 4, 8})
        # Triton 3.0 names neither softmax dim/keep_dims nor a topk
        # direction; the draw keeps to what the front end accepts.
        with patch.object(dsl_extend, '_accepts', return_value=False):
            for _ in range(16):
                self.assertEqual(target_attributes('softmax', matrix, 'triton'), {'axis': 0})
                self.assertNotIn('reverse', target_attributes('scan_sum', matrix, 'triton'))
                self.assertNotIn('descending', target_attributes('topk', matrix, 'triton'))
        with self.assertRaises(ValueError):
            target_attributes('topk', Ty('float32', (4, 1)), 'triton')
        # A restricted reduction still draws both spellings of its axis.
        self.assertEqual({target_attributes('argmax', Ty('int32', (4, 4)), 'tilelang', axes=[1])['axis']
                          for _ in range(32)}, {1, -1})

    def test_matrix_route_and_historical_knobs(self):
        legacy = Config(extended_prob=1)
        config = Config(extended_prob=1, dsl_matrix_prob=1, dsl_attributes=True)
        seen = set()
        for backend in ('triton', 'tilelang'):
            for seed in range(6):
                random.seed(seed)
                parent = ExtendedGenerator(legacy, backend).generate('mixed')
                for op in set(eligible_ops(parent, backend)) & MATRIX_OPS:
                    with self.subTest(backend=backend, seed=seed, op=op):
                        old = extend_passed(parent, backend, op, legacy)
                        node = old.body.operations[-1]
                        self.assertEqual(len(old.validate()[0]['main', node.operands[0]].shape), 1)
                        self.assertEqual(set(node.attrs), {'k'} if op == 'topk' else set())
                        new = extend_passed(parent, backend, op, config)
                        node = new.body.operations[-1]
                        seen.add((backend, op, len(new.validate()[0]['main', node.operands[0]].shape)))
                        self.assertEqual(new.body.returns[:-1], parent.body.returns)
                        emit(backend, new)
        for backend, ops in (('triton', ('scan_sum', 'sort', 'softmax', 'argmax', 'xor_sum', 'dsl_clamp')),
                             ('tilelang', ('scan_max', 'reduce_abssum', 'reduce_bitor', 'dsl_sigmoid'))):
            for op in ops:
                self.assertIn((backend, op, 2), seen)

    def test_matrix_softmax_without_attributes_keeps_the_historical_call(self):
        # Without drawn attributes, softmax prints tl.softmax(x, 0) at any
        # rank, which Triton 3.0 and 3.8 both read as axis 0.
        config = Config(extended_prob=1, dsl_matrix_prob=1)
        ranks = set()
        for seed in range(12):
            random.seed(seed)
            parent = ExtendedGenerator(Config(extended_prob=1), 'triton').generate('mixed')
            child = extend_passed(parent, 'triton', 'softmax', config)
            node = child.body.operations[-1]
            rank = len(child.validate()[0]['main', node.operands[0]].shape)
            ranks.add(rank)
            self.assertEqual(node.attrs, {'axis': 0} if rank == 2 else {})
            self.assertIn(f'tl.softmax({node.operands[0]}, 0))', emit('triton', child))
            raw = encoded(child)
            outputs, _ = extended_reference(raw, extended_inputs(raw, seed=1), 0, 1)
            # Block-stacked: dim 1 is the operand's axis 0.
            total = outputs[node.results[0].name].sum(1)
            torch.testing.assert_close(total, torch.ones_like(total))
        self.assertIn(2, ranks)

    def test_respelling_keeps_every_type(self):
        config = Config(extended_prob=1, dsl_matrix_prob=1, dsl_attributes=True)
        parent = ExtendedGenerator(config, 'tilelang').generate('arithmetic')
        for op in ('reduce_abssum', 'reduce_bitxor', 'scan_sum'):
            child = extend_passed(parent, 'tilelang', op, config)
            types = child.validate()[0]
            spellings = set()
            current = child
            for _ in range(24):
                respelled = respell_target(current, 'tilelang')
                self.assertEqual(respelled.validate()[0], types)
                self.assertNotEqual(respelled.to_dict(), current.to_dict())
                spellings.add(json.dumps(respelled.body.operations[-1].attrs, sort_keys=True))
                current = respelled
            self.assertGreater(len(spellings), 1, op)
        with self.assertRaises(ValueError):
            respell_target(parent, 'tilelang')

    def test_attribute_grid_visits_every_legal_cell_and_resumes(self):
        ty = Ty('float32', (4, 8))
        grids = GridState()
        with patch.object(dsl_extend, '_accepts', return_value=True):
            for op in ('scan_sum', 'softmax', 'topk'):
                cells = target_attribute_cells(op, ty, 'triton')
                first = target_attributes(op, ty, 'triton', grids=grids)
                restored = GridState()
                restored.load(json.loads(json.dumps(grids.save())))
                rest = [target_attributes(op, ty, 'triton', grids=restored) for _ in range(len(cells) - 1)]
                encode = lambda values: {json.dumps(value, sort_keys=True) for value in values}
                self.assertEqual(encode([first, *rest]), encode(cells))
                self.assertEqual(target_attributes(op, ty, 'triton', grids=restored), first)
        with patch.object(dsl_extend, '_accepts', return_value=False):
            self.assertEqual(target_attributes('softmax', ty, 'triton', grids=grids), {'axis': 0})
            self.assertNotIn('reverse', target_attributes('scan_sum', ty, 'triton', grids=grids))

    def test_grid_respelling_preserves_topk_shape_and_changes_the_program(self):
        program, _, _, _ = target_program('triton', 'float32', [
            ('topk', Ty('float32', (4, 4)), {'k': 4, 'descending': True})])
        types = program.validate()[0]
        grids = GridState()
        with patch.object(dsl_extend, '_accepts', return_value=True):
            for _ in range(8):
                changed = respell_target(program, 'triton', grids)
                self.assertEqual(changed.validate()[0], types)
                self.assertNotEqual(changed.to_dict(), program.to_dict())
                program = changed

    def test_loop_repeats_the_scan_as_spelled(self):
        config = Config(extended_prob=1, dsl_matrix_prob=1, dsl_attributes=True)
        parent = ExtendedGenerator(config, 'tilelang').generate('arithmetic')
        scan = extend_passed(parent, 'tilelang', 'scan_sum', config)
        node = scan.body.operations[-1]
        scan.body.operations[-1].attrs = {'axis': -2, 'reverse': True}
        scan.observations = [node.operands[0]]
        wrapped = encoded(loop_target(scan, 'tilelang'))
        memory = extended_inputs(wrapped, seed=4)
        for steps in (0, 1, 2, 3):
            outputs, _ = extended_reference(wrapped, memory, steps, 1)
            expected = outputs[node.operands[0]]
            for _ in range(min(steps, 2)):
                expected = expected.flip(-2).cumsum(-2).flip(-2)
            torch.testing.assert_close(outputs[node.results[0].name], expected)

    def test_feedback_distinguishes_spellings(self):
        def features(attrs):
            program, *_ = target_program('triton', 'float32', [('scan_sum', Ty('float32', (4, 8)), attrs)])
            return attribute_features(program)
        self.assertNotEqual(features({'axis': 0}), features({'axis': -2}))
        self.assertNotEqual(features({'axis': 1}), features({'axis': 1, 'reverse': True}))
        self.assertEqual(features({'axis': 1}), features({'axis': 1, 'reverse': False}))
        program, *_ = target_program('triton', 'float32', [('topk', Ty('float32', (4, 8)), {'k': 8}),
                                                           ('topk', Ty('float32', (4, 2)), {'k': 2})])
        self.assertEqual(len(attribute_features(program)), 2)


if __name__ == '__main__':
    unittest.main()
