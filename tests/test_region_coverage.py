"""Executable semantics, persistence and fault injection for expanded regions."""
import ast
import copy
import random
import types
import unittest
from dataclasses import asdict
from itertools import product

import torch

from src.config import Config
from src.ir.layout import MATRIX_LAYOUTS
from src.ir.region import Function, Operation as Op, Region, RegionExecution, RegionProgram, walk
from src.workflow.emitter.region_checks import _region_input, _region_input_storage, _run_region
from src.backends.common.region_emitter import region_variants
from src.backends.triton.region import triton_code
from src.workflow.emitter.region_runtime import _region_reference
from src.workflow.feedback import program_features, key
from src.workflow.fuzzer.fuzzer import TileSmith
from src.workflow.generator.region_generator import RegionGenerator
from src.workflow.mutator import Mutator
from src.workflow.oracle import Oracle
from test_regions import nested_program


def coverage_program(dtype='float32', initial='load'):
    p = nested_program(dtype, initial)
    p.spec.M, p.spec.N, p.spec.K = 65, 70, 34
    fn = Function('fn_0', Region(['x'], [
        Op('index_add', 'out', ['x'], {'axis': 'iteration', 'scale': 0.25})], 'out'))
    yes = Region(['yes'], [Op('neg', 'negated', ['yes'])], 'negated')
    no = Region(['no'], [Op('index_add', 'shifted', ['no'], {'axis': 'column', 'scale': 0.5})], 'shifted')
    loop = Region(['carry'], [
        Op('call', 'called', ['carry'], {'callee': 'fn_0'}),
        Op('if', 'chosen', ['called'], {'predicate': 'iteration', 'modulus': 3, 'parity': 0}, [yes, no])], 'chosen')
    even = Region(['even'], [Op('index_add', 'row_added', ['even'], {'axis': 'row', 'scale': 0.25})], 'row_added')
    odd = Region(['odd'], [], 'odd')
    skipped = Region(['unused'], [Op('scale', 'bad', ['unused'], {'alpha': 100.0})], 'bad')
    p.functions = [fn]
    p.body = Region([], [
        Op(initial, 'entry'),
        Op('for', 'iterated', ['entry'], {'trip_count': 3, 'start': 1, 'step': 2}, [loop]),
        Op('if', 'tiled', ['iterated'], {'predicate': 'checkerboard', 'modulus': 2, 'parity': 0}, [even, odd]),
        Op('for', 'zero', ['tiled'], {'trip_count': 0}, [skipped]),
        Op('call', 'answer', ['zero'], {'callee': 'fn_0'})], 'answer')
    p.execution = RegionExecution()
    p.validate()
    return p


class RegionCoverageTests(unittest.TestCase):
    def setUp(self):
        state = random.getstate()
        self.addCleanup(random.setstate, state)
        random.seed(509)

    def test_reference_induction_calls_zero_loop_and_grid_predicate(self):
        p = coverage_program()
        a = torch.arange(p.spec.M * p.spec.N).reshape(p.spec.M, p.spec.N).float() / 8
        bx = (torch.arange(p.spec.N) // 32)[None, :]
        by = (torch.arange(p.spec.M) // 32)[:, None]
        expected = a.clone()
        for i in (1, 3, 5):
            expected = expected + i * 0.25
            expected = -expected if i % 3 == 0 else expected + bx * 0.5
        expected = torch.where((by + bx) % 2 == 0, expected + by * 0.25, expected)
        result = _region_reference(a, torch.empty(34, 70), asdict(p.body), 32, 32,
                                   'float32', [asdict(fn) for fn in p.functions])
        torch.testing.assert_close(result, expected, rtol=0, atol=0)

    def test_all_predicates_and_nested_induction_restore(self):
        p = nested_program()
        a = torch.ones(33, 35)
        by = (torch.arange(33) // 32)[:, None]
        bx = (torch.arange(35) // 32)[None, :]
        for axis, indices in (('row', by), ('column', bx), ('checkerboard', by+bx), ('iteration', torch.tensor(0))):
            yes = Region(['y'], [Op('neg', 'negative', ['y'])], 'negative')
            no = Region(['n'], [], 'n')
            p.body = Region([], [Op('load', 'x'), Op('if', 'out', ['x'],
                {'predicate': axis, 'modulus': 3, 'parity': 1}, [yes, no])], 'out')
            p.validate()
            result = _region_reference(a, torch.empty(33,35), asdict(p.body), 32, 32, 'float32')
            torch.testing.assert_close(result, torch.where(indices % 3 == 1, -a, a))
        inner = Region(['inner_arg'], [Op('index_add', 'inner_out', ['inner_arg'],
                                        {'axis': 'iteration', 'scale': 1.0})], 'inner_out')
        outer = Region(['outer_arg'], [
            Op('for', 'nested', ['outer_arg'], {'trip_count': 2, 'start': 2, 'step': 3}, [inner]),
            Op('index_add', 'outer_out', ['nested'], {'axis': 'iteration', 'scale': 1.0})], 'outer_out')
        p.body = Region([], [Op('load', 'x'), Op('for', 'out', ['x'],
                           {'trip_count': 2, 'start': 1, 'step': 2}, [outer])], 'out')
        p.validate()
        result = _region_reference(a, torch.empty(33,35), asdict(p.body), 32, 32, 'float32')
        torch.testing.assert_close(result, a + (2+5+1) + (2+5+3))

    def test_emitted_triton_executes_on_cpu_tile_model(self):
        # Execute the emitted statements, not a second walk over the IR. This
        # catches wrong context passing, induction binding and masked writeback.
        class Pointer:
            def __init__(self, value, offset=0):
                self.value, self.offset = value.reshape(-1), offset
            def __add__(self, offset):
                return Pointer(self.value, self.offset + offset)
        def load(ptr, mask, other=0):
            indices = torch.where(mask, ptr.offset, 0)
            return torch.where(mask, ptr.value[indices], other)
        def store(ptr, value, mask):
            ptr.value[ptr.offset[mask]] = value[mask].to(ptr.value.dtype)
        tile = [0, 0]
        tl = types.SimpleNamespace(program_id=lambda axis: tile[axis], arange=torch.arange,
            load=load, store=store, float32=torch.float32, full=torch.full,
            range=lambda start, stop, **kwargs: range(start, stop),
            dot=lambda a,b,c: a.float() @ b.float() + c)
        tl.full = lambda shape, value, dtype: torch.full(shape, value, dtype=dtype)
        for dtype in ('float16', 'float32'):
            for initial in ('load', 'gemm'):
                b_layouts = MATRIX_LAYOUTS if initial == 'gemm' else ('contiguous',)
                for layout_a, layout_b in product(MATRIX_LAYOUTS, b_layouts):
                    with self.subTest(dtype=dtype, initial=initial, a=layout_a, b=layout_b):
                        p = coverage_program(dtype, initial)
                        p.execution.input_layout_a, p.execution.input_layout_b = layout_a, layout_b
                        p.validate()
                        a_storage, a = _region_input_storage((65, 34 if initial == 'gemm' else 70),
                            getattr(torch, dtype), 'integer', 0.125, layout_a, 'cpu')
                        b_storage, b = _region_input_storage((34, 70), getattr(torch, dtype),
                            'integer', 0.125, layout_b, 'cpu')
                        a_before, b_before = a_storage.clone(), b_storage.clone()
                        c = torch.full((65,70), float('nan'), dtype=getattr(torch,dtype))
                        namespace = {'triton': types.SimpleNamespace(jit=lambda fn: fn), 'tl': tl}
                        exec(triton_code(p), namespace)
                        for by in range(3):
                            for bx in range(3):
                                tile[:] = [by, bx]
                                namespace['kernel'](Pointer(a_storage), Pointer(b_storage), Pointer(c))
                        expected = _region_reference(a,b,asdict(p.body),32,32,dtype,[asdict(fn) for fn in p.functions])
                        torch.testing.assert_close(c, expected, rtol=0, atol=0)
                        self.assertTrue(torch.equal(a_storage, a_before))
                        self.assertTrue(torch.equal(b_storage, b_before))

    def test_layout_identity_feedback_and_validation(self):
        for initial in ('load', 'gemm'):
            p = coverage_program(initial=initial)
            original = p.to_dict()
            self.assertNotIn('input_layout_a', original['execution'])
            self.assertNotIn('input_layout_b', original['execution'])
            signatures = set()
            b_layouts = MATRIX_LAYOUTS if initial == 'gemm' else ('contiguous',)
            for layout_a, layout_b in product(MATRIX_LAYOUTS, b_layouts):
                with self.subTest(initial=initial, a=layout_a, b=layout_b):
                    p.execution.input_layout_a, p.execution.input_layout_b = layout_a, layout_b
                    raw = p.to_dict()
                    restored = RegionProgram.from_dict(raw)
                    self.assertEqual(restored.to_dict(), raw)
                    signature = TileSmith._make_sig(p)
                    self.assertEqual(signature, TileSmith._make_sig_from_dict(raw))
                    signatures.add(signature)
                    self.assertIn(key('region_layout', initial, layout_a, layout_b, p.spec.dtype.value),
                                  program_features(restored))
            self.assertEqual(len(signatures), 6 if initial == 'load' else 36)
            p.execution.input_layout_a = p.execution.input_layout_b = 'contiguous'
            self.assertEqual(p.to_dict(), original)
        with self.assertRaisesRegex(ValueError, 'Unsupported region input layout'):
            RegionExecution(input_layout_a='unknown').validate()
        p = coverage_program()
        p.execution.input_layout_b = 'transposed'
        with self.assertRaisesRegex(ValueError, 'second input layout'):
            p.validate()

    def test_ir_identity_legacy_compatibility_and_validation(self):
        p = coverage_program()
        raw = p.to_dict()
        self.assertEqual(raw['version'], 3)
        self.assertEqual(RegionProgram.from_dict(raw).to_dict(), raw)
        self.assertEqual(TileSmith._make_sig(p), TileSmith._make_sig_from_dict(raw))
        old = nested_program().to_dict()
        self.assertEqual(old['version'], 1)
        self.assertEqual(RegionProgram.from_dict(old).to_dict(), old)
        self.assertIsNone(RegionProgram.from_dict(old).execution)
        changed = copy.deepcopy(p)
        changed.execution.input_pattern = 'integer'
        self.assertNotEqual(TileSmith._make_sig(p), TileSmith._make_sig(changed))
        self.assertIn(key('region_input','integer'), program_features(changed))
        for update in ({'step': 0}, {'start': -1}, {'trip_count': -1}, {'trip_count': True}):
            bad = copy.deepcopy(p)
            bad.body.operations[1].attrs.update(update)
            with self.assertRaises(ValueError):
                bad.validate()
        for update in ({'modulus': 1}, {'predicate': 'unknown'}, {'parity': 2}):
            bad = copy.deepcopy(p)
            bad.body.operations[2].attrs.update(update)
            with self.assertRaises(ValueError):
                bad.validate()
        for update in ({'input_seed_count': 0}, {'repeat_count': 9}, {'schedule_pair': 1}):
            settings = asdict(p.execution)
            settings.update(update)
            with self.assertRaises(ValueError):
                RegionExecution(**settings).validate()

    def test_generation_mutation_feedback_and_schedule_geometry(self):
        # Fixed seed: the body/spec samplers share one global RNG stream, and
        # sampler edits shift every downstream draw; seed 0 covers all axes,
        # trip counts and patterns within 40 programs per backend.
        random.seed(0)
        self.addCleanup(random.seed)
        axes, trips, patterns = set(), set(), set()
        for backend in ('tilelang', 'triton'):
            config = Config(coverage_probe_prob=0, dim_range=(1,129))
            gen = RegionGenerator(config, backend)
            mutator = Mutator(config, backend)
            for _ in range(40):
                p = gen.generate()
                for op in p.all_operations():
                    if op.kind == 'if':
                        axes.add(op.attrs['predicate'])
                    if op.kind == 'for':
                        trips.add(op.attrs['trip_count'])
                        child = op.regions[0]
                        self.assertIn(child.arguments[0], child.operations[-1].operands)
                        self.assertEqual(child.yield_value, child.operations[-1].result)
                self.assertLessEqual(len(list(p.all_operations())), config.region_max_ops+1)
                patterns.add(p.execution.input_pattern)
                for variant in region_variants(p,backend):
                    self.assertEqual((variant.spec.block_M,variant.spec.block_N,variant.spec.block_K),
                                     (p.spec.block_M,p.spec.block_N,p.spec.block_K))
                for candidate in (p, mutator.mutate(p)):
                    candidate.validate()
                    self.assertEqual(RegionProgram.from_dict(candidate.to_dict()).to_dict(),candidate.to_dict())
                    program_features(candidate)
                    ast.parse(Oracle(config,backend)._emit_code(candidate))
        self.assertEqual(axes, {'row','column','checkerboard','iteration'})
        self.assertEqual(trips,{0,1,2,3,4})
        self.assertEqual(patterns, {'normal','integer','alternating'})


class RegionCheckTests(unittest.TestCase):
    def setUp(self):
        self.source = torch.ones(2,3)
        self.reference = self.source.clone()

    def run_check(self, launches, repeats=2, reference=None):
        _run_region(launches, (self.source,), self.reference if reference is None else reference,
                    repeats, tolerance=0.01)

    def test_input_patterns_reproducible_and_factories_compile_all_then_run(self):
        state = torch.random.get_rng_state()
        self.addCleanup(torch.random.set_rng_state,state)
        for pattern in ('normal','integer','alternating'):
            torch.manual_seed(4)
            first = _region_input((3,5),torch.float16,pattern,0.1,'cpu')
            torch.manual_seed(4)
            self.assertTrue(torch.equal(first,_region_input((3,5),torch.float16,pattern,0.1,'cpu')))
        events = []
        def factory(name):
            def prepare():
                events.append('prepare'+name)
                def launch(c):
                    events.append('run'+name)
                    c.copy_(self.source)
                return launch
            return prepare
        self.run_check([factory('A'),factory('B')])
        # Cold compiles dominate the harness wall time, so all prepares run up
        # front on a thread pool (any prepare order) while execution keeps the
        # strict serial launch order.
        self.assertEqual(sorted(events[:2]), ['prepareA','prepareB'])
        self.assertEqual(events[2:], ['runA','runA','runB','runB'])

    def test_parallel_prepare_reports_first_error_in_variant_order(self):
        """Concurrent prepares keep the serial first-failure contract: every
        variant still compiles, and the earliest failing variant (in launch
        order) is the error raised, with its prepare marker re-printed so the
        last-marker location inference names the crashing variant."""
        import contextlib
        import io
        calls = []
        def factory(name, fail=False):
            def prepare():
                calls.append('prepare' + name)
                if fail:
                    raise RuntimeError('boom ' + name)
                def launch(c):
                    calls.append('run' + name)
                    c.copy_(self.source)
                return launch
            return prepare
        err = io.StringIO()
        with self.assertRaisesRegex(RuntimeError, 'boom B'), contextlib.redirect_stderr(err):
            _run_region([factory('A'), factory('B', fail=True), factory('C')], (self.source,),
                        self.reference, repeats=1, tolerance=0.01)
        # Every variant compiles, earlier variants still execute, and the
        # failing variant surfaces at its own position in the serial order.
        self.assertEqual(sorted(calls), ['prepareA', 'prepareB', 'prepareC', 'runA'])
        self.assertNotIn('runB', calls)
        self.assertNotIn('runC', calls)
        self.assertIn('TILESMITH_STAGE=prepare_1', err.getvalue())
        # Strided workers: with more variants than compile threads, every
        # variant still compiles (variant 8 rides thread 0's stride), and the
        # first failure in order wins.
        calls.clear()
        with self.assertRaisesRegex(RuntimeError, 'boom v0'):
            _run_region([factory(f'v{i}', fail=(i == 0)) for i in range(9)], (self.source,),
                        self.reference, repeats=1, tolerance=0.01)
        self.assertEqual(sorted(calls), [f'preparev{i}' for i in range(9)])

    def test_layout_initialization_and_padding_corruption(self):
        state = torch.random.get_rng_state()
        self.addCleanup(torch.random.set_rng_state, state)
        for dtype, shape, pattern, layout in product(
                (torch.float16, torch.float32), ((1, 1), (3, 5)),
                ('normal', 'integer', 'alternating'), MATRIX_LAYOUTS):
            with self.subTest(dtype=dtype, shape=shape, pattern=pattern, layout=layout):
                unique_shape = ((1, shape[1]) if layout == 'broadcast_rows' else
                                (shape[0], 1) if layout == 'broadcast_cols' else shape)
                torch.manual_seed(71)
                expected = _region_input(unique_shape, dtype, pattern, 0.125, 'cpu').expand(shape)
                torch.manual_seed(71)
                storage, view = _region_input_storage(shape, dtype, pattern, 0.125, layout, 'cpu')
                torch.testing.assert_close(view, expected, rtol=0, atol=0)
                addresses = (view.storage_offset() + torch.arange(shape[0])[:, None] * view.stride(0)
                             + torch.arange(shape[1])[None, :] * view.stride(1))
                unused = torch.ones(storage.numel(), dtype=torch.bool)
                unused[addresses.reshape(-1)] = False
                self.assertTrue(torch.all(storage[unused] == 19))
                self.assertGreaterEqual(int(unused.sum()), 16)

        storage, view = _region_input_storage((3, 5), torch.float32, 'integer', 0.125, 'strided', 'cpu')
        reference = view.clone()
        def corrupt_padding(c):
            c.copy_(reference)
            storage[1] = 0  # A hole between two logical input elements.
        with self.assertRaisesRegex(RuntimeError, 'input storage modified'):
            _run_region([lambda: corrupt_padding], (storage,), reference, 1, tolerance=0)
        torch.testing.assert_close(view, reference, rtol=0, atol=0)

    def test_detects_missing_writes_including_expected_nan(self):
        for ref in (self.reference, torch.full_like(self.reference,float('nan'))):
            with self.assertRaisesRegex(RuntimeError,'WRONG RESULT'):
                self.run_check([lambda: lambda c: None],reference=ref)

    def test_detects_input_corruption_and_guard_writes(self):
        def corrupt(c):
            c.copy_(self.reference)
            self.source[0,0] = 2
        with self.assertRaisesRegex(RuntimeError,'input storage modified'):
            self.run_check([lambda: corrupt])
        self.source.fill_(1)
        def overrun(c):
            c.copy_(self.reference)
            c.as_strided((1,), (1,), c.storage_offset()+c.numel()).fill_(0)
        with self.assertRaisesRegex(RuntimeError,'canary'):
            self.run_check([lambda: overrun])

    def test_detects_repeat_and_schedule_disagreement(self):
        counter = [0]
        def nondeterministic(c):
            c.copy_(self.reference + counter[0] * 0.001)
            counter[0] += 1
        with self.assertRaisesRegex(RuntimeError,'repeat determinism'):
            self.run_check([lambda: nondeterministic])
        # Each schedule is within reference tolerance, but not of each other.
        with self.assertRaisesRegex(RuntimeError,'schedule invariance'):
            self.run_check([lambda: lambda c:c.copy_(self.reference-0.009),
                            lambda: lambda c:c.copy_(self.reference+0.009)])


if __name__ == '__main__':
    unittest.main()
