"""Directed probe invariants, persistence, and injected oracle faults on CPU."""
import ast
import copy
import random
import unittest
from unittest.mock import patch

import torch
from src.config import Config
from src.ir import TileKernel, ComputeKind, DataType
from src.backends.common.probes import generate_probe, mutate_probe, repair_probe, probe_program, KINDS, patterns
from src.workflow.emitter.probe_runtime import _probe_input, _probe_exact, _run_probe
from src.workflow.emitter import get_emitter
from src.workflow.fuzzer.fuzzer import TileSmith


class ProbeTests(unittest.TestCase):
    def test_layout_values_and_storage_bounds(self):
        for layout in ('contiguous', 'transposed', 'strided', 'offset'):
            storage, view, stride, offset = _probe_input(3, 5, torch.float16, layout, 'indexed', 'cpu')
            self.assertEqual(view.stride(), stride)
            self.assertEqual(view.storage_offset(), offset)
            torch.testing.assert_close(view, torch.arange(-510, -495, dtype=torch.float16).reshape(3, 5))
            self.assertTrue(torch.all(storage[-16:] == 19))
            if layout != 'contiguous':
                self.assertFalse(view.is_contiguous())

    def test_special_values_are_not_normalized(self):
        x = _probe_input(2, 8, torch.float32, 'offset', 'special', 'cpu')[1]
        self.assertTrue(torch.isnan(x).any())
        self.assertTrue(torch.isinf(x).any())
        self.assertTrue(torch.signbit(x[x == 0]).any())
        y = _probe_input(2, 8, torch.float16, 'strided', 'subnormal', 'cpu')[1]
        self.assertTrue(((y.abs() < torch.finfo(y.dtype).tiny) & (y != 0)).any())
        _probe_exact(x, x.clone(), 'clone')
        with self.assertRaisesRegex(RuntimeError, 'bits differ'):
            _probe_exact(torch.tensor([0.]), torch.tensor([-0.]), 'signed zero')

    def test_generation_mutation_and_emission(self):
        random.seed(73)
        config = Config(coverage_probe_prob=1)
        seen = set()
        for _ in range(100):
            program = mutate_probe(generate_probe(config))
            k = program.spec
            seen.add(k.compute_kind)
            self.assertLessEqual(k.N, k.block_N)
            self.assertEqual(k.block_N & (k.block_N - 1), 0)
            self.assertIn(k.input_pattern, patterns(k.compute_kind))
            for backend in ('triton', 'tilelang'):
                ast.parse(get_emitter(backend, config).emit(program))
        self.assertEqual(seen, set(KINDS))

    def test_wide_gemm_repairs_resource_usage(self):
        from src.backends.tilelang.params import check_shared_memory as tilelang_check_shared_memory
        from src.backends.triton.params import check_shared_memory as triton_check_shared_memory
        for dtype in DataType:
            k = repair_probe(TileKernel('wide', compute_kind=ComputeKind.GEMM_ARGMAX,
                                       N=129, dtype=dtype, num_stages=3))
            for check in (tilelang_check_shared_memory, triton_check_shared_memory):
                self.assertTrue(check(k.block_M, k.block_N, k.block_K, k.dtype, k.num_stages))

    def test_resume_and_dedup_include_probe_semantics(self):
        fuzzer = TileSmith.__new__(TileSmith)
        program = generate_probe(Config())
        meta = fuzzer._program_to_dict(program)
        restored = fuzzer._dict_to_program(meta)
        self.assertEqual(fuzzer._make_sig(program), fuzzer._make_sig(restored))
        self.assertEqual(fuzzer._make_sig(program), fuzzer._make_sig_from_dict(meta))
        self.assertEqual(program.spec.params_dict, restored.spec.params_dict)
        for attr, value in [('input_layout', 'different'), ('input_pattern', 'different'),
                            ('repeat_count', 8), ('schedule_pair', False), ('cache_cycle', False)]:
            changed = copy.deepcopy(program)
            setattr(changed.spec, attr, value)
            self.assertNotEqual(fuzzer._make_sig(program), fuzzer._make_sig(changed))
            self.assertNotEqual(fuzzer._kind_label(program), fuzzer._kind_label(changed))

    def test_broadcast_storage_is_initialized_without_overlapping_writes(self):
        from src.backends.common.probe_emitter import _layout
        for layout in ('broadcast_rows', 'broadcast_cols'):
            storage, view, strides, offset = _probe_input(3, 5, torch.float32, layout, 'indexed', 'cpu')
            self.assertEqual((*strides, offset, storage.numel()), _layout(3, 5, layout))
            if layout == 'broadcast_rows':
                self.assertTrue(torch.equal(view[0], view[2]))
            else:
                self.assertTrue(torch.equal(view[:, 0], view[:, 4]))

    @patch('torch.cuda.synchronize')
    def test_cache_cycle_instantiates_in_execution_order(self, sync):
        a = _probe_input(2, 3, torch.float32, 'offset', 'integer', 'cpu')
        events = []
        def factory(name):
            def build():
                events.append('compile-' + name)
                def run(out):
                    events.append('run-' + name)
                    out[16:-16].copy_(a[1].reshape(-1))
                return run
            return build
        _run_probe([factory('A'), factory('B'), factory('A')], [a], a[1], 'copy', 2, 0,
                   instantiate=True)
        self.assertEqual(events, ['compile-A', 'run-A', 'run-A', 'compile-B', 'run-B',
                                  'run-B', 'compile-A', 'run-A', 'run-A'])

    @patch('torch.cuda.synchronize')
    def test_oracles_detect_injected_faults(self, sync):
        a = _probe_input(2, 3, torch.float32, 'strided', 'integer', 'cpu')
        ref = a[1].clone()
        def good(out):
            out[16:-16].copy_(ref.reshape(-1))
        _run_probe([good, good], [a], ref, 'copy', 3, 0.1)
        def canary(out):
            good(out)
            out[0] = 0
        with self.assertRaisesRegex(RuntimeError, 'canary'):
            _run_probe([canary], [a], ref, 'copy', 2, 0.1)
        with self.assertRaisesRegex(RuntimeError, 'reference'):
            _run_probe([lambda out: None], [a], ref, 'copy', 2, 0.1)
        def corrupt_input(out):
            good(out)
            a[0][-1] = 0
        with self.assertRaisesRegex(RuntimeError, 'input storage'):
            _run_probe([corrupt_input], [a], ref, 'copy', 2, 0.1)
        count = 0
        def unstable(out):
            nonlocal count
            count += 1
            out[16:-16].copy_(ref.reshape(-1) + count * .001)
        with self.assertRaisesRegex(RuntimeError, 'repeat determinism'):
            _run_probe([unstable], [a], ref, 'softmax', 3, .1)
        # Both variants individually pass reference tolerance; pairing fails.
        zero_ref = torch.zeros_like(ref)
        def plus(out):
            out[16:-16].fill_(.075)
        def minus(out):
            out[16:-16].fill_(-.075)
        with self.assertRaisesRegex(AssertionError, 'schedule invariance'):
            _run_probe([plus, minus], [a], zero_ref, 'softmax', 2, .1)


if __name__ == '__main__':
    unittest.main()
