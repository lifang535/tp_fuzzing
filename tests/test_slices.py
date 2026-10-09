"""Feature-slice programs: legalization, harness emission, exact references,
scheduling, persistence and campaign integration (CPU only)."""
import json
import random
import tempfile
import unittest
from collections import Counter
from unittest.mock import patch

import torch

from src.config import Config
from src.ir.serialization import program_from_dict, program_to_dict
from src.ir.slice import SliceProgram
from src.workflow.fuzzer.fuzzer import TileSmith
from src.workflow.oracle import BugReport, BugType
from src.workflow.slices import SLICES, available, emit_slice, make_program, slice_origin, validate_program
from src.workflow.slices.chain import transfer
from src.workflow.slices.dtypes import DTYPES, Domain
from src.workflow.slices.scheduler import SliceScheduler, pairs
from src.workflow.triage import failure_key
from src.backends.common.diagnostics import classify_root_cause

BACKENDS = ('triton', 'tilelang')


class Cfg:
    input_seed = 3


def programs(count=24, seed=0):
    rng = random.Random(seed)
    for backend in BACKENDS:
        for name, slice_ in SLICES.items():
            for _ in range(count):
                yield make_program(name, slice_.sample(rng, backend), backend)


def harness_namespace(program):
    """Execute a harness module body without running main (no GPU)."""
    code = emit_slice(program, Cfg)
    code = code.replace('import triton.language as tl', '').replace('import triton\n', '')
    code = code.replace('import tilelang.language as T', '').replace('import tilelang\n', '')
    code = code.replace('@triton.jit', '')
    namespace = {'__name__': 'slice_harness'}
    exec(compile(code, 'harness', 'exec'), namespace)
    return namespace


class DomainTests(unittest.TestCase):
    def test_representability(self):
        self.assertTrue(Domain(-8, 8).fits('f8e5'))
        self.assertFalse(Domain(-9, 9).fits('f8e5'))
        self.assertTrue(Domain(-16, 16).fits('f8e4'))
        self.assertFalse(Domain(-4, 4, 3).fits('f8e4'))
        self.assertTrue(Domain(-256, 256).fits('bf16'))
        self.assertFalse(Domain(-257, 257).fits('bf16'))
        self.assertFalse(Domain(-1, 2).fits('u8'))
        self.assertFalse(Domain(0, 1, 1).fits('i32'))

    def test_transfer_division_semantics(self):
        x, c = Domain(-7, 7), Domain(-3, -3)
        self.assertEqual(transfer('mod', x, c, 'floor'), Domain(-2, 0))
        self.assertEqual(transfer('mod', x, c, 'trunc'), Domain(-2, 2))
        self.assertEqual(transfer('half', Domain(-3, 3, 1), None, 'trunc'), Domain(-1.5, 1.5, 2))


class LegalizationTests(unittest.TestCase):
    def test_programs_are_fixed_points_and_round_trip(self):
        for program in programs():
            with self.subTest(slice=program.slice, backend=program.backend):
                validate_program(program)
                restored = program_from_dict(json.loads(json.dumps(program_to_dict(program))))
                self.assertEqual(restored.to_dict(), program.to_dict())
                report = {'params': program.params_dict}
                self.assertEqual(program_from_dict(report).to_dict(), program.to_dict())

    def test_unlegalized_parameters_are_rejected(self):
        program = next(programs(1))
        broken = SliceProgram(program.slice, dict(program.params, out_dt='nonsense'), program.backend)
        with self.assertRaises(ValueError):
            broken.validate()

    def test_every_chain_value_is_representable(self):
        for program in programs(16, seed=1):
            params = dict(program.params)
            plan = SLICES[program.slice].legalize(params, program.backend)
            for op, dtype, target, source, constant in plan['pre'] + plan['post']:
                self.assertIn(dtype, DTYPES)
                self.assertIn(target, DTYPES)
                if op != 'none':
                    self.assertFalse(DTYPES[dtype].is_fp8, (program.slice, op, dtype))
            self.assertTrue(DTYPES[plan['out_dt']].torch)

    def test_backend_restrictions(self):
        for program in programs(16, seed=2):
            params = program.params
            if program.backend == 'triton':
                plan = SLICES[program.slice].legalize(dict(params), 'triton')
                for op, dtype, *_ in plan['pre'] + plan['post']:
                    if op in ('floor', 'ceil'):
                        self.assertIn(dtype, ('f32', 'f64'))
            if program.slice == 'reduce' and program.backend == 'tilelang' and params['batch'] > 1:
                self.assertGreater(params['batch'] * params['threads'], 0)

    def test_available_and_origin(self):
        self.assertEqual(set(available('triton')), set(SLICES))
        with self.assertRaises(ValueError):
            available('triton', ('nope',))
        for program in programs(2, seed=3):
            self.assertTrue(slice_origin(program).startswith(program.slice))


class HarnessTests(unittest.TestCase):
    def test_emitted_harnesses_compile(self):
        for program in programs(12, seed=4):
            code = emit_slice(program, Cfg)
            compile(code, program.slice, 'exec')
            self.assertIn('ALL PASSED', code)
            imports = 'import triton' if program.backend == 'triton' else 'import tilelang'
            self.assertIn(imports, code)

    def test_references_are_exact_and_representable(self):
        for program in programs(10, seed=5):
            with self.subTest(slice=program.slice, backend=program.backend, params=program.params):
                ns = harness_namespace(program)
                params = dict(program.params)
                plan = SLICES[program.slice].legalize(params, program.backend)
                x_shape, y_shape = SLICES[program.slice].input_shapes(plan)
                d = plan['input_domain']
                x = ns['_slice_values'](x_shape, d.lo, d.hi, d.frac, 3, 1)
                y = ns['_slice_values'](y_shape, d.lo, d.hi, d.frac, 3, 2)
                expected = ns['reference'](x, y)
                self.assertFalse(torch.isnan(expected).any())
                out = DTYPES[plan['out_dt']]
                if out.is_float:
                    stored = expected.to(getattr(torch, out.torch)).double()
                    self.assertTrue(torch.equal(stored, expected), 'reference not representable in output')
                else:
                    self.assertTrue(torch.equal(expected, torch.trunc(expected)))
                    self.assertGreaterEqual(float(expected.min()), out.minval)
                    self.assertLessEqual(float(expected.max()), out.maxval)

    def test_scan_reference_semantics(self):
        program = make_program('scan', dict(SLICES['scan'].sample(random.Random(0), 'triton'),
                                            kind='cpair', reverse=1, shape='32', tail='none', in_dt='i32',
                                            values='tiny', pre1_op='none', pre2_op='none', post1_op='none',
                                            core_dt='i32', out_dt='i32', pre1_dt='i32', pre2_dt='i32',
                                            post1_dt='i32'), 'triton')
        ns = harness_namespace(program)
        x = torch.tensor([1.0, 2.0, 2.0, 0.0] * 8)
        r = ns['reference'](x, x)
        # suffix argmax with ties to the smallest index
        self.assertEqual(r[0].item(), 1.0)
        self.assertEqual(r[2].item(), 2.0)
        self.assertEqual(r[-1].item(), 31.0)

    def test_rejection_marker(self):
        ns = harness_namespace(next(programs(1)))
        import io
        import contextlib
        stream = io.StringIO()
        with contextlib.redirect_stderr(stream):
            ns['_slice_reject'](ValueError('atomic_max does not support fp16'))
            ns['_slice_reject'](RuntimeError('PassManager::run failed'))
        self.assertEqual(stream.getvalue().count('TILESMITH_REJECTED='), 1)


class TriageTests(unittest.TestCase):
    def test_classification(self):
        self.assertEqual(classify_root_cause('TILESMITH_REJECTED=cannot cast\nCompilationError: x'),
                         'unsupported_feature')
        nvcc = ('RuntimeError: Compilation error:\n/tmp/a/tvm_kernels.cu(24): error: more than one '
                'constructor applies to convert from "long long" to "cutlass::half_t":\n')
        self.assertEqual(classify_root_cause(nvcc), 'tilelang_cuda_compile_error')
        self.assertIn('nvcc more than one constructor', failure_key(nvcc))

    def test_pass_failures_keep_their_mlir_diagnosis(self):
        def message(detail):
            return (f'{detail}\nLLVM ERROR: Unsupported rounding mode for conversion.\n'
                    "x.py:3:1: error: Failures have been detected while processing an MLIR pass pipeline\n"
                    "x.py:3:1: note: Pipeline failed while executing [`ConvertTritonGPUToLLVM` on 'builtin.module' operation]\n"
                    'Traceback (most recent call last):\n  File "c.py", line 1, in make_llir\n'
                    'RuntimeError: PassManager::run failed\n')
        a = failure_key(message('Unsupported conversion from f64 to f8E5M2 with rounding mode rtne'))
        b = failure_key(message('Unsupported conversion from f8E4M3FN to f64'))
        self.assertNotEqual(a, b)
        self.assertIn('pass ConvertTritonGPUToLLVM', a)
        self.assertIn('f64 to f8E5M2', a)


class SchedulerTests(unittest.TestCase):
    def test_pairs_and_observation(self):
        scheduler = SliceScheduler('triton')
        program = scheduler.next(random.Random(1))
        self.assertIsInstance(program, SliceProgram)
        cells = pairs(program.params)
        self.assertEqual(len(cells), len(program.params) * (len(program.params) - 1) // 2)
        scheduler.observe(program, None, None, 2.0)
        state = scheduler.states[program.slice]
        self.assertEqual(state.single, cells)
        scheduler.observe(program, 'triton_pass_failure', 'b1', 2.0)
        self.assertEqual(state.single, set())
        self.assertEqual(state.buckets, Counter({'b1': 1}))
        scheduler.observe(program, 'unsupported_feature', 'u1', 2.0)
        self.assertEqual(state.rejected, 1)
        self.assertNotIn('u1', state.buckets)

    def test_warmup_tries_every_slice_first(self):
        scheduler = SliceScheduler('triton', warmup=2)
        rng = random.Random(3)
        chosen = []
        while sum(s.tests for s in scheduler.states.values()) < 2 * len(scheduler.names):
            name = scheduler.choose_slice(type('R', (), {'random': lambda self: 0.99,
                                                          'choice': rng.choice, 'choices': rng.choices})())
            chosen.append(name)
            program = make_program(name, SLICES[name].sample(rng, 'triton'), 'triton')
            scheduler.observe(program, None, None, 1.0)
        self.assertEqual(Counter(chosen), Counter({name: 2 for name in scheduler.names}))

    def test_discovery_rewards_new_buckets(self):
        scheduler = SliceScheduler('triton', ['cast', 'reduce'], warmup=0)
        rng = random.Random(8)
        for name, bucket in (('cast', 'b'), ('reduce', None)):
            for i in range(10):
                program = make_program(name, SLICES[name].sample(rng, 'triton'), 'triton')
                scheduler.observe(program, 'x' if bucket else None, f'{bucket}{i}' if bucket else None, 1.0)
        self.assertGreater(scheduler.estimate('cast'), scheduler.estimate('reduce'))

    def test_known_cores_are_avoided_and_found(self):
        scheduler = SliceScheduler('triton', ['cast'], min_explore=0.0)
        program = make_program('cast', SLICES['cast'].sample(random.Random(9), 'triton'), 'triton')
        kept = {'in_dt': program.params['in_dt']}
        scheduler.add_core(program, kept, 'b', 'wrong_result')
        scheduler.add_core(program, kept, 'b', 'wrong_result')
        self.assertEqual(scheduler.cores_of(program, 'b'), 1)
        self.assertEqual(scheduler.find_core(program, 'wrong_result'), (kept, 'b'))
        self.assertIsNone(scheduler.find_core(program, 'other'))
        self.assertIsNone(scheduler.find_core(program, 'wrong_result', 'c'))
        always = type('R', (), {'random': lambda self: 0.999})()
        self.assertFalse(scheduler.known(program, always))
        for _ in range(30):
            scheduler.observe(program, 'wrong_result', 'b', 1.0)
        self.assertTrue(scheduler.known(program, always))
        self.assertEqual(scheduler.avoided['b'], 1)

    def test_greedy_candidates_prefer_uncovered_pairs(self):
        scheduler = SliceScheduler('tilelang', ['cast'], candidates=8, mutate_prob=0)
        rng = random.Random(4)
        for _ in range(30):
            state = random.Random(rng.random())
            draws = [make_program('cast', SLICES['cast'].sample(random.Random(state.random()), 'tilelang'),
                                  'tilelang') for _ in range(8)]
            covered = scheduler.states['cast'].covered
            best = max(len(pairs(p.params) - covered) for p in draws)
            samples = iter([p.params for p in draws])
            with patch.object(SLICES['cast'], 'sample', side_effect=lambda *_: dict(next(samples))):
                chosen = scheduler.next(rng)
            self.assertEqual(len(pairs(chosen.params) - covered), best)
            scheduler.observe(chosen, None, None, 1.0)

    def test_accept_filter_and_snapshot(self):
        scheduler = SliceScheduler('triton', ['reduce', 'gemm'])
        self.assertIsNone(scheduler.next(random.Random(2), accept=lambda p: False)
                          if scheduler.mutate_prob == 0 else None)
        rng = random.Random(5)
        for _ in range(10):
            program = scheduler.next(rng)
            scheduler.observe(program, 'wrong_result' if rng.random() < 0.3 else None,
                              f'w{rng.randrange(3)}', 1.5)
        snapshot = json.loads(json.dumps(scheduler.snapshot()))
        restored = SliceScheduler('triton', ['reduce', 'gemm'])
        restored.restore(snapshot)
        self.assertEqual(restored.snapshot(), scheduler.snapshot())
        with self.assertRaises(ValueError):
            SliceScheduler('tilelang', ['reduce']).restore(snapshot)
        stats = scheduler.stats()
        self.assertEqual(set(stats), {'reduce', 'gemm'})


class MinimizeTests(unittest.TestCase):
    def test_reduction_keeps_only_essential_knobs(self):
        from src.workflow.slices.minimize import core, minimize, matches, signature
        rng = random.Random(11)
        while True:
            program = make_program('cast', SLICES['cast'].sample(rng, 'triton'), 'triton')
            if program.params['in_dt'] == 'f16' or rng.random() < 0.02:
                break
        in_dt = program.params['in_dt']
        calls = []

        def still_fails(candidate):
            calls.append(candidate)
            return candidate.params['in_dt'] == in_dt
        reduced, kept, used = minimize(program, still_fails, budget=64)
        self.assertEqual(used, len(calls))
        self.assertEqual(reduced.params['in_dt'], in_dt)
        self.assertIn('in_dt', kept)
        self.assertTrue(matches(program.params, {'in_dt': in_dt}))
        for knob in ('pre1_op', 'pre2_op', 'pre3_op', 'tail', 'pair', 'dynamic'):
            self.assertNotIn(knob, kept, kept)
        self.assertEqual(core(reduced), kept)
        self.assertTrue(signature(reduced).startswith(f'cast {in_dt}'))

    def test_budget_bounds_the_tests(self):
        from src.workflow.slices.minimize import minimize
        program = make_program('reduce', SLICES['reduce'].sample(random.Random(12), 'tilelang'), 'tilelang')
        _, _, used = minimize(program, lambda candidate: True, budget=3)
        self.assertLessEqual(used, 3)


class CampaignTests(unittest.TestCase):
    def setUp(self):
        state = random.getstate()
        self.addCleanup(random.setstate, state)
        random.seed(7)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)

    def test_campaign_tests_saves_and_resumes_slices(self):
        config = Config(backends=['triton'], output_dir=self.temp.name, seed=7, slice_prob=1.0,
                        quarantine=True, structural_feedback=False, slice_minimize=False)
        fuzzer = TileSmith(config)
        calls = []

        def fake_test(program):
            calls.append(program)
            if len(calls) % 3:
                return None
            report = BugReport(BugType.COMPILE_CRASH, 'RuntimeError: PassManager::run failed\n',
                               params=program.params_dict, generated_code='pass\n')
            report.classify_root_cause('triton')
            from src.workflow.triage import failure_bucket
            report.failure_bucket, report.failure_key = failure_bucket(report.error_message, report.root_cause)
            return report

        with patch.object(fuzzer.oracle, 'test', side_effect=fake_test), \
                patch.object(fuzzer.oracle, '_emit_code', return_value='pass\n'):
            fuzzer.run(9, verbose=False)
        self.assertEqual(len(calls), 9)
        self.assertTrue(all(isinstance(p, SliceProgram) for p in calls))
        passed = list((fuzzer.output_dir / 'passed').glob('passed_slice_*.json'))
        failed = list((fuzzer.output_dir / 'failed').rglob('failed_slice_*.json'))
        self.assertEqual((len(passed), len(failed)), (6, 3))
        summary = json.loads((fuzzer.output_dir / 'summary.json').read_text())
        self.assertEqual(sum(s['tests'] for s in summary['slices'].values()), 9)
        self.assertEqual(summary['route_stats']['slice']['tests'], 9)
        self.assertTrue(all(first['route'] == 'slice' for first in summary['failure_bucket_first_seen'].values()))
        self.assertEqual(summary['generation_config']['slice_prob'], 1.0)
        self.assertTrue((fuzzer.output_dir / 'slice_state.json').exists())
        restored = TileSmith(config, resume_dir=str(fuzzer.output_dir))
        self.assertEqual(sum(s.tests for s in restored.slice_scheduler.states.values()), 9)
        self.assertEqual(len(restored.tested_configs), 9)
        self.assertEqual(restored.route_stats['slice']['tests'], 9)

    def test_wrong_results_are_bucketed_by_their_reduced_core(self):
        config = Config(backends=['triton'], output_dir=self.temp.name, seed=8, slice_prob=1.0,
                        slice_names=('cast',), quarantine=False, structural_feedback=False)
        fuzzer = TileSmith(config)
        calls = []

        def fake_test(program):
            calls.append(program)
            if program.params['out_dt'] not in ('i8', 'u8'):
                return None
            report = BugReport(BugType.WRONG_RESULT, 'RuntimeError: WRONG RESULT: slice=cast variant=w4 '
                               'differs at 3/64 elements', params=program.params_dict, generated_code='pass\n')
            report.classify_root_cause('triton')
            from src.workflow.triage import failure_bucket
            from src.workflow.slices import slice_origin
            report.failure_bucket, report.failure_key = failure_bucket(
                report.error_message, report.root_cause, origin=slice_origin(program))
            return report

        with patch.object(fuzzer.oracle, 'test', side_effect=fake_test), \
                patch.object(fuzzer.oracle, '_emit_code', return_value='pass\n'):
            fuzzer.run(60, verbose=False)
        wrong = {b: n for b, n in fuzzer.failure_buckets.items() if b.startswith('wrong_result:')}
        self.assertTrue(wrong)
        keys = {fuzzer.failure_bucket_keys[b] for b in wrong}
        # Conversions at any chain step reduce to the output conversion; an
        # unsigned output also keeps what makes its inputs non-negative.
        self.assertTrue(all('origin cast f32 ' in key and ('| >i8' in key or '| >u8' in key) for key in keys), keys)
        self.assertIn('origin cast f32 | >i8', ' '.join(keys))
        self.assertLessEqual(len(wrong), 4)
        self.assertGreater(fuzzer.slice_reduction_tests, 0)
        self.assertEqual(fuzzer.route_stats['slice_reduction']['tests'], fuzzer.slice_reduction_tests)
        self.assertGreater(len(calls), 60)
        minimized = [json.loads(p.read_text()) for p in (fuzzer.output_dir / 'failed').rglob('*.json')]
        self.assertTrue(any('minimized' in record for record in minimized))
        self.assertTrue(list((fuzzer.output_dir / 'failed').rglob('*.min.py')))


if __name__ == '__main__':
    unittest.main()
