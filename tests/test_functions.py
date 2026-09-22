"""Multi-function scope, call semantics, serialization and backend regression tests."""
import ast
import copy
import random
import unittest
from dataclasses import asdict

import torch
from src.config import Config
from src.ir.region import Function, Operation as Op, Region, RegionProgram, walk
from src.workflow.emitter.region_runtime import _region_reference
from src.workflow.fuzzer.fuzzer import TileSmith
from src.workflow.generator.region_generator import RegionGenerator
from src.workflow.feedback import program_features, key
from src.workflow.oracle import Oracle
from test_regions import nested_program


def function_program(dtype='float32', initial='load', reductions=False):
    p = nested_program(dtype, initial)
    first = Region(['x', 'y'], [Op('add', 'a', ['x', 'y'])], 'a')
    if reductions:
        first.operations += [Op('tile_transpose', 'b', ['a']), Op('row_sum', 'c', ['b']),
                             Op('scale', 'd', ['c'], {'alpha': 1 / 32})]
        first.yield_value = 'd'
    loop = Region(['carry'], [Op('call', 'c', ['carry', 'z'], {'callee': 'fn_0'})], 'c')
    yes = Region(['arg'], [Op('call', 'yes', ['arg', 'x'], {'callee': 'fn_0'})], 'yes')
    no = Region(['other'], [Op('call', 'no', ['other', 'y'], {'callee': 'fn_0'})], 'no')
    second = Region(['x', 'y', 'z'], [
        Op('call', 'a', ['x', 'y'], {'callee': 'fn_0'}),
        Op('for', 'b', ['a'], {'trip_count': 2}, [loop]),
        Op('if', 'out', ['b'], {'parity': 0}, [yes, no])], 'out')
    p.functions = [Function('fn_0', first), Function('fn_1', second)]
    p.body = Region([], [Op(initial, 'v1'), Op('neg', 'v2', ['v1']),
                        Op('call', 'v3', ['v1', 'v2', 'v1'], {'callee': 'fn_1'}),
                        Op('call', 'v4', ['v3', 'v1'], {'callee': 'fn_0'})], 'v4')
    p.validate()
    return p


class FunctionTests(unittest.TestCase):
    def test_reference_calls_bind_arguments_and_isolate_locals(self):
        p = function_program()
        a = torch.arange(33 * 35).reshape(33, 35).float() / 1024
        before = a.clone()
        result = _region_reference(a, torch.empty(33, 35), asdict(p.body), 32, 32,
                                   'float32', [asdict(fn) for fn in p.functions])
        factor = torch.where(torch.arange(33) // 32 % 2 == 0, 4, 2)[:, None]
        torch.testing.assert_close(result, a * factor)
        torch.testing.assert_close(a, before)

    def test_invalid_calls_and_function_scope(self):
        mutations = [
            lambda p: p.functions[0].body.operations.append(Op('call', 'recursive', ['x', 'y'], {'callee': 'fn_0'})),
            lambda p: p.functions[0].body.operations.append(Op('call', 'forward', ['x', 'y', 'x'], {'callee': 'fn_1'})),
            lambda p: p.body.operations[-1].attrs.update(callee='fn_missing'),
            lambda p: p.body.operations[-1].operands.pop(),
            lambda p: p.functions[0].body.operations[0].operands.__setitem__(0, 'v1'),
            lambda p: p.body.operations[-1].operands.__setitem__(0, 'x'),
            lambda p: setattr(p.functions[0].body.operations[0], 'kind', 'gemm'),
            lambda p: setattr(p.functions[1], 'name', 'fn_0'),
        ]
        for mutate in mutations:
            p = function_program()
            mutate(p)
            with self.assertRaises(ValueError):
                p.validate()

    def test_roundtrip_hash_and_call_only_filename(self):
        p = function_program()
        self.assertEqual(p.to_dict()['version'], 2)
        self.assertEqual(RegionProgram.from_dict(p.to_dict()).to_dict(), p.to_dict())
        old = nested_program()
        self.assertEqual(old.to_dict()['version'], 1)
        self.assertFalse(RegionProgram.from_dict(old.to_dict()).functions)
        fuzzer = object.__new__(TileSmith)
        label = fuzzer._kind_label(p)
        self.assertTrue(label.startswith('calls_main(f1+f0)__f0()__f1(f0+f0+f0+f0)_'))
        self.assertNotIn('float32', label)
        self.assertNotIn('M33', label)
        other = copy.deepcopy(p)
        other.functions[1].body.operations[1].attrs['trip_count'] = 3
        self.assertEqual(p.call_label(), other.call_label())
        self.assertNotEqual(label, fuzzer._kind_label(other))
        # Many repeated call sites remain distinguishable without exceeding NAME_MAX.
        for i in range(100):
            p.body.operations.append(Op('call', f'extra{i}', ['v1', 'v2'], {'callee': 'fn_0'}))
        p.validate()
        self.assertLess(len(('failed_' + fuzzer._kind_label(p) + '.json').encode()), 255)

    def test_backends_emit_real_calls_and_reference_definitions(self):
        p = function_program()
        for backend in ('triton', 'tilelang'):
            code = Oracle(Config(), backend)._emit_code(p)
            ast.parse(code)
            self.assertIn('def fn_0(', code)
            self.assertIn('def fn_1(', code)
            self.assertIn("'callee': 'fn_1'", code)
            if backend == 'triton':
                self.assertIn('v3 = fn_1(v1, v2, v1, by, bx, 0)', code)
            else:
                self.assertEqual(code.count('@T.macro'), 2)
                self.assertIn('fn_1(v1, v2, v1, by, bx, 0, v3)', code)

    def test_generation_reachability_budget_and_feedback(self):
        random.seed(321)
        gen = RegionGenerator(Config(coverage_probe_prob=0, function_min_count=3))
        nested_calls = False
        for _ in range(40):
            p = gen.generate()
            p.validate()
            self.assertEqual(len(p.functions), 3)
            self.assertLessEqual(len(list(p.all_operations())), gen.config.region_max_ops + 1)
            reached = set()
            pool = {fn.name: fn for fn in p.functions}
            def visit(body):
                nonlocal nested_calls
                for op in body.operations:
                    for child in op.regions:
                        nested_calls |= any(o.kind == 'call' for o in walk(child))
                        visit(child)
                    if op.kind == 'call' and op.attrs['callee'] not in reached:
                        reached.add(op.attrs['callee'])
                        visit(pool[op.attrs['callee']].body)
            visit(p.body)
            self.assertEqual(reached, set(pool))
            self.assertIn(key('function_count', 3), program_features(p))
        self.assertTrue(nested_calls)


if __name__ == '__main__':
    unittest.main()
