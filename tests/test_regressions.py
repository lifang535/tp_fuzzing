"""Numeric checks, native emission, subprocess diagnostics and campaign resume."""
import ast
import contextlib
import io
import json
import pickle
import random
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import torch
from src.config import Config
from src.ir import TileKernel, ComputeKind
from src.backends.common.probes import probe_program, KINDS
from src.workflow.emitter import _threshold_header, TritonEmitter, TileLangEmitter
from src.workflow.emitter.runtime import _finite_compare
from src.workflow.fuzzer.fuzzer import TileSmith
from src.workflow.generator import ProgramGenerator
from src.workflow.oracle import Oracle
from test_regions import nested_program


class NumericTests(unittest.TestCase):
    def test_nonfinite_mismatches(self):
        for c, r in [([float('nan')], [1.]), ([float('inf')], [1.]),
                     ([float('inf')], [-float('inf')]), ([1., float('nan')], [1., 2.]),
                     ([1.], [float('nan')])]:
            with self.subTest(c=c, r=r), self.assertRaisesRegex(RuntimeError, 'WRONG RESULT'):
                _finite_compare(torch.tensor(c), torch.tensor(r))

    def test_matching_nonfinite_and_finite_error(self):
        self.assertEqual(_finite_compare(torch.tensor([float('nan'), float('inf')]),
                                         torch.tensor([float('nan'), float('inf')]))[0], 0.)
        self.assertEqual(_finite_compare(torch.tensor([float('inf'), 3.]),
                                         torch.tensor([float('inf'), 1.]))[0], 2.)

    def test_output_rounding_and_overflow(self):
        ref = torch.tensor([1.0001, 70000.])
        self.assertEqual(_finite_compare(ref.half(), ref)[0], 0.)
        with self.assertRaisesRegex(RuntimeError, 'infinity'):
            _finite_compare(torch.tensor([float('inf')], dtype=torch.float16), torch.tensor([1.]))

    def test_signed_zero_and_shape(self):
        with self.assertRaisesRegex(RuntimeError, 'signed zeros'):
            _finite_compare(torch.tensor([-0.]), torch.tensor([0.]), check_signed_zero=True)
        with self.assertRaisesRegex(RuntimeError, 'shape'):
            _finite_compare(torch.ones(2, 1), torch.ones(2))

    def test_finite_fp32_difference_does_not_overflow(self):
        error, _, relative = _finite_compare(torch.tensor([3e38]), torch.tensor([-3e38]))
        self.assertGreater(error, 5e38)
        self.assertAlmostEqual(relative, 2.)

    def test_chunk_boundary_keeps_all_errors(self):
        ref = torch.ones((1 << 20) + 1)
        actual = ref.clone()
        actual[-1] = 3
        self.assertEqual(_finite_compare(actual, ref)[0], 2.)
        actual[-1] = float('nan')
        with self.assertRaisesRegex(RuntimeError, 'NaN'):
            _finite_compare(actual, ref)

    def test_seed_is_embedded_and_reproducible(self):
        header = _threshold_header(Config(input_seed=123))
        exec(header, {'torch': torch})
        first = torch.randn(10)
        exec(header, {'torch': torch})
        torch.testing.assert_close(first, torch.randn(10), rtol=0, atol=0)



class ResumeTests(unittest.TestCase):
    def test_removed_seed_formats_abort_resume_without_partial_pool(self):
        with tempfile.TemporaryDirectory() as tmp:
            fuzzer = TileSmith.__new__(TileSmith)
            fuzzer.output_dir = Path(tmp)
            fuzzer.seed_pool = []
            records = [nested_program().to_dict(), {'type': 'dynamic', 'params': {}}]
            (Path(tmp) / 'seed_pool.json').write_text(json.dumps(records))
            with self.assertRaisesRegex(ValueError, 'Cannot resume seed pool'):
                fuzzer._restore_seed_pool()
            self.assertEqual(fuzzer.seed_pool, [])

    def test_pending_native_program_is_normalized_and_replayed(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            fuzzer = TileSmith.__new__(TileSmith)
            fuzzer.output_dir = Path(tmp)
            state = random.getstate()
            self.addCleanup(random.setstate, state)
            (Path(tmp) / 'rng_state.json').write_text(json.dumps(dict(
                version=state[0], internalstate=state[1], gauss_next=state[2])))
            program = nested_program()
            program.spec.alpha = 1.0  # Native specs saved before cleanup.
            program.legacy = None
            pending = Path(tmp) / 'pending_program.pkl'
            pending.write_bytes(pickle.dumps((7, program)))
            fuzzer._restore_rng_state()
            self.assertEqual(fuzzer._resume_pending_i, 7)
            self.assertEqual(fuzzer._resume_pending_program.to_dict(), program.to_dict())
            pending.write_bytes(pickle.dumps((8, object())))
            with self.assertRaisesRegex(ValueError, 'Cannot resume pending program'):
                fuzzer._restore_rng_state()

    def test_summary_counts_survive_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp) / '2026.07.08-15.44_tilelang_hard-shape_seed=42'
            folder.mkdir()
            (folder / 'summary.json').write_text(json.dumps(dict(
                total_tested=30058, root_causes={'wrong_result': 4992, 'dtype_mismatch': 731}, input_seed=7)))
            config = Config(seed=42, input_seed=7)
            with contextlib.redirect_stdout(io.StringIO()):
                fuzzer = TileSmith(config, resume_dir=str(folder))
                fuzzer.run(0, verbose=False)
            saved = json.loads((folder / 'summary.json').read_text())
            self.assertEqual(saved['total_tested'], 30058)
            self.assertEqual(saved['bugs_total'], 5723)
            self.assertEqual(saved['bugs_unique'], 2)
            self.assertEqual(saved['input_seed'], 7)
            with self.assertRaisesRegex(ValueError, 'input seed mismatch'):
                TileSmith(Config(seed=42, input_seed=8), resume_dir=str(folder))



class EmissionTests(unittest.TestCase):
    def test_probe_warps_and_standalone_entry(self):
        for kind in KINDS:
            for threads in (128, 256):
                program = probe_program(TileKernel('probe', compute_kind=kind, threads=threads, N=64))
                code = TritonEmitter().emit(program)
                tree = ast.parse(code)
                launches = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Subscript)]
                self.assertEqual(len(launches), 2, kind)
                self.assertEqual(next(k.value.value for k in launches[0].keywords if k.arg == 'num_warps'), threads // 32)
                self.assertIn('    test_probe()', code)
                ast.parse(TileLangEmitter().emit(program))

    def test_softmax_uses_explicit_row_axis(self):
        program = probe_program(TileKernel('probe', compute_kind=ComputeKind.SOFTMAX, N=64))
        code = TritonEmitter().emit(program)
        self.assertNotIn('tl.softmax(', code)
        self.assertIn('axis=1', code)

    def test_random_emission(self):
        random.seed(17)
        for backend in ('tilelang', 'triton'):
            config = Config(dim_range=(16, 128))
            generator = ProgramGenerator(config, backend)
            oracle = Oracle(config, backend)
            for _ in range(100):
                code = oracle._emit_code(generator.generate())
                ast.parse(code)
                self.assertNotIn('max_diff = (C.to(torch.float32) - ref', code)


    def test_signal_crash_has_diagnostic(self):
        oracle = Oracle(Config(), 'triton')
        completed = type('Result', (), dict(returncode=-11, stderr='', stdout=''))()
        with patch('src.workflow.oracle.oracle.subprocess.run', return_value=completed):
            report = oracle.test(nested_program())
        self.assertEqual(report.root_cause, 'segfault')
        self.assertIn('SIGSEGV', report.error_message)



if __name__ == '__main__':
    unittest.main()
