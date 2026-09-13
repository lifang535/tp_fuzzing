"""CPU regressions for generated-code semantics; no GPU/compiler execution required."""
import ast
import contextlib
import io
import json
import tempfile
from pathlib import Path
import copy
import random
import unittest
from unittest.mock import patch

import torch

from src.config import Config
from src.ir import TileKernel, TileProgram, ComputeKind, DataType, TilePipeline, PipelineStep
from src.ir.dynamic_seq import (TileBuffer, TileValuePool, KernelStep, DynamicSequence,
                                GemmOpGen, DoublePipelineOpGen, SoftmaxOpGen)
from src.workflow.emitter import (_threshold_header, TritonEmitter, TileLangEmitter,
                                 TritonPipelineEmitter, TileLangPipelineEmitter,
                                 TritonDynamicEmitter, TileLangDynamicEmitter)
from src.workflow.emitter.runtime import _finite_compare, _dynamic_reference
from src.workflow.fuzzer.fuzzer import TileSmith
from src.workflow.generator import ProgramGenerator
from src.workflow.oracle import Oracle


def sequence(mode='subtract_max', n=4, block_n=2, dtype='float32'):
    a = TileBuffer('A', (1, 1), dtype, 'global', 'A.float()')
    b = TileBuffer('B', (1, n), dtype, 'global', 'B.float()')
    frag = TileBuffer('C_local_1', (1, block_n), 'float32', 'fragment', '')
    steps = [
        KernelStep('gemm', [a, b], [frag], dict(a_shared='As', b_shared='Bs', c_local=frag.name,
                   loop_kind='serial', num_stages=1, block_M=1, block_N=block_n, block_K=1), ''),
        KernelStep('accumulate_reduce', [frag], [frag], dict(mode=mode, frag_name=frag.name, row_stat_name='stat'), ''),
        KernelStep('copy_f2g', [frag], [], dict(frag_name=frag.name), ''),
    ]
    return DynamicSequence(steps, TileValuePool(global_in=[a, b], fragment=[frag]),
                           M=1, N=n, K=1, block_M=1, block_N=block_n, block_K=1,
                           loop_kind='serial', num_stages=1, dtype=dtype)


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


class DynamicTests(unittest.TestCase):
    def test_tile_local_max(self):
        seq = sequence()
        result = _dynamic_reference({'A': torch.ones(1, 1), 'B': torch.tensor([[1., 2., 10., 20.]])}, seq.step_specs, 2, 'float32')
        torch.testing.assert_close(result, torch.tensor([[-1., 0., -10., 0.]]))

    def test_negative_sum_is_not_clamped(self):
        seq = sequence('divide_sum', n=2)
        result = _dynamic_reference({'A': torch.ones(1, 1), 'B': torch.tensor([[-2., -1.]])}, seq.step_specs, 2, 'float32')
        torch.testing.assert_close(result, torch.tensor([[2/3, 1/3]]))

    def test_nonlinear_padding_is_preserved_until_writeback(self):
        seq = sequence('divide_sum', n=3)
        frag = seq.pool.fragment[0]
        seq.steps.insert(1, KernelStep('exp', [frag], [frag], {'frag_name': frag.name}, ''))
        result = _dynamic_reference({'A': torch.ones(1, 1), 'B': torch.zeros(1, 3)}, seq.step_specs, 2, 'float32')
        torch.testing.assert_close(result, torch.full((1, 3), 0.5))

    def test_double_pipeline_keeps_updated_fragment_active(self):
        seq = sequence()
        pool = seq.pool
        frag = pool.fragment[-1]
        step = DoublePipelineOpGen().apply(pool, {'acc_dtype': 'float32'}, {})
        self.assertIs(pool.fragment[-1], frag)
        seq.steps = [seq.steps[0], step, seq.steps[-1]]
        result = _dynamic_reference({'A': torch.ones(1, 1), 'B': torch.tensor([[1., 2., 3., 4.]])}, seq.step_specs, 2, 'float32')
        torch.testing.assert_close(result, torch.tensor([[2., 4., 6., 8.]]))

    def test_buffer_copies_and_binary_operands(self):
        seq = sequence(n=4)
        original = seq.pool.fragment[0]
        shared = TileBuffer('shared', (1, 2), 'float32', 'shared', '')
        copied = TileBuffer('copied', (1, 2), 'float32', 'fragment', '')
        # B is (M,N) here and can be loaded by a copy_g2s.
        seq.steps = [seq.steps[0],
                     KernelStep('copy_g2s', [seq.pool.global_in[1]], [shared], {'src_name': 'B', 'shared_name': 'shared'}, ''),
                     KernelStep('copy_s2f', [shared], [copied], {'src_name': 'shared', 'frag_name': 'copied'}, ''),
                     KernelStep('elemwise_mul', [copied, original], [copied], {'use_global': False, 'frag_a_name': 'copied', 'frag_b_name': original.name}, ''),
                     KernelStep('copy_f2g', [copied], [], {'frag_name': 'copied'}, '')]
        seq.pool.fragment.append(copied)
        inputs = {'A': torch.tensor([[2.]]), 'B': torch.tensor([[1., 2., 3., 4.]])}
        torch.testing.assert_close(_dynamic_reference(inputs, seq.step_specs, 2, 'float32'), 2 * inputs['B'].square())
        code = TritonDynamicEmitter().emit(seq)
        self.assertIn('copied = shared.to(tl.float32)', code)
        self.assertIn('copied.to(tl.float32) * C_local_1.to(tl.float32)', code)

    def test_softmax_writes_output(self):
        seq = sequence(n=2)
        step = SoftmaxOpGen().apply(seq.pool, {}, {})
        seq.steps = [seq.steps[0], step]
        code = TileLangDynamicEmitter().emit(seq)
        self.assertIn('T.copy(C_local_1, C[by * block_M, bx * block_N])', code)

    def test_dedup_includes_attrs_and_buffer_identity(self):
        fuzzer = TileSmith.__new__(TileSmith)
        seq = sequence()
        changed = copy.deepcopy(seq)
        changed.steps[1].attrs['mode'] = 'divide_sum'
        self.assertNotEqual(fuzzer._make_sig(seq), fuzzer._make_sig(changed))
        self.assertNotEqual(fuzzer._kind_label(seq), fuzzer._kind_label(changed))
        meta = fuzzer._program_to_dict(seq)
        self.assertEqual(fuzzer._make_sig(seq), fuzzer._make_sig_from_dict(meta))
        restored = fuzzer._dict_to_program(meta)
        self.assertEqual(fuzzer._make_sig(seq), fuzzer._make_sig(restored))
        changed = copy.deepcopy(seq)
        changed.steps[-1].inputs[0] = TileBuffer('other', (1, 2), 'float32', 'fragment', '')
        self.assertNotEqual(fuzzer._make_sig(seq), fuzzer._make_sig(changed))


class ResumeTests(unittest.TestCase):
    def test_dynamic_seed_restores_executable_dataflow(self):
        fuzzer = TileSmith.__new__(TileSmith)
        seq = sequence()
        restored = fuzzer._dict_to_program(json.loads(json.dumps(fuzzer._program_to_dict(seq))))
        self.assertEqual(restored.pool.global_in[0].shape, (1, 1))
        self.assertEqual(restored.output_buffer.shape, (1, 2))
        for emitter in (TritonDynamicEmitter(), TileLangDynamicEmitter()):
            self.assertEqual(emitter.emit(seq), emitter.emit(restored))

    def test_legacy_seed_can_still_be_used_for_mutation(self):
        fuzzer = TileSmith.__new__(TileSmith)
        meta = fuzzer._program_to_dict(sequence())
        del meta['params']['sequence_steps']
        restored = fuzzer._dict_to_program(meta)
        self.assertEqual([s.op_kind for s in restored.steps], ['gemm', 'accumulate_reduce', 'copy_f2g'])
        self.assertNotEqual(fuzzer._make_sig_from_dict(meta), fuzzer._make_sig(sequence()))

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

    def test_legacy_pending_double_pipeline_temporary(self):
        seq = sequence()
        double = DoublePipelineOpGen().apply(seq.pool, {'acc_dtype': 'float32'}, {})
        temporary = TileBuffer(double.attrs['c2_name'], (1, 2), 'float32', 'fragment', '')
        seq.steps = [seq.steps[0], double, KernelStep('copy_f2g', [temporary], [], {'frag_name': temporary.name}, '')]
        inputs = {'A': torch.ones(1, 1), 'B': torch.tensor([[1.,2.,3.,4.]])}
        torch.testing.assert_close(_dynamic_reference(inputs, seq.step_specs, 2, 'float32'), inputs['B'])


class EmissionTests(unittest.TestCase):
    def test_all_single_ops_pass_warps_and_parse(self):
        for kind in ComputeKind:
            for threads in (128, 256):
                kernel = TileKernel('probe', compute_kind=kind, threads=threads, N=64)
                program = TileProgram([kernel])
                code = TritonEmitter().emit(program)
                tree = ast.parse(code)
                launches = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Subscript)]
                self.assertEqual(len(launches), 2 if kernel.coverage_probe else 1, kind)
                self.assertEqual(next(k.value.value for k in launches[0].keywords if k.arg == 'num_warps'), threads // 32)
                ast.parse(TileLangEmitter().emit(program))

    def test_dynamic_reference_is_embedded_and_executable(self):
        seq = sequence()
        for emitter in (TritonDynamicEmitter(), TileLangDynamicEmitter()):
            tree = ast.parse(emitter.emit(seq))
            helpers = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                       and node.name in ('_finite_compare', '_max_diff', '_dynamic_reference')]
            self.assertEqual(len(helpers), 3)
            namespace = {}
            exec(compile(ast.Module(body=helpers, type_ignores=[]), '<helpers>', 'exec'), namespace)
            out = namespace['_dynamic_reference']({'A': torch.ones(1, 1), 'B': torch.tensor([[1.,2.,10.,20.]])},
                                                  seq.step_specs, 2, 'float32')
            torch.testing.assert_close(out, torch.tensor([[-1.,0.,-10.,0.]]))

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

    def test_softmax_uses_explicit_row_axis(self):
        kernel = TileKernel('probe', compute_kind=ComputeKind.SOFTMAX, N=64)
        programs = [TritonEmitter().emit(TileProgram([kernel]))]
        for first in (ComputeKind.GEMM, ComputeKind.COPY):
            pipeline = TilePipeline([PipelineStep(first), PipelineStep(ComputeKind.SOFTMAX)], N=64)
            programs.append(TritonPipelineEmitter().emit(pipeline))
        for code in programs:
            self.assertNotIn('tl.softmax(', code)
            self.assertIn('axis=1', code)

    def test_signal_crash_has_diagnostic(self):
        oracle = Oracle(Config(), 'triton')
        completed = type('Result', (), dict(returncode=-11, stderr='', stdout=''))()
        with patch('src.workflow.oracle.oracle.subprocess.run', return_value=completed):
            report = oracle.test(TileProgram([TileKernel('probe')]))
        self.assertEqual(report.root_cause, 'segfault')
        self.assertIn('SIGSEGV', report.error_message)


if __name__ == '__main__':
    unittest.main()
