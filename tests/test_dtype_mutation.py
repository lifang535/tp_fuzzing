"""Storage dtype mutation preserves semantics and repairs backend resources."""
import ast
import random
import unittest
from unittest.mock import patch

from src.config import Config
from src.ir import DataType, TileKernel, ComputeKind
from src.ir.region import RegionProgram
from src.workflow.fuzzer.fuzzer import TileSmith
from src.backends.common.probes import mutate_probe, repair_probe, probe_program, KINDS
from src.workflow.mutator.mutator import Mutator
from src.workflow.oracle import Oracle
from test_functions import function_program


class DtypeMutationTests(unittest.TestCase):
    def setUp(self):
        state = random.getstate()
        self.addCleanup(random.setstate, state)
        random.seed(93)
        self.config = Config(dtype_mutate_prob=1, coverage_probe_prob=0)

    def test_multifunction_dtype_switch_and_replay(self):
        for backend in ('tilelang', 'triton'):
            for initial in ('load', 'gemm'):
                # INT8 is generation-only (the gemm-only int8 path): mutation
                # never produces it, so it has no seed/roundtrip here.
                for dtype in (DataType.FLOAT16, DataType.FLOAT32):
                    with self.subTest(backend=backend, initial=initial, dtype=dtype):
                        seed = function_program(dtype.value, initial)
                        before = seed.to_dict()
                        result = Mutator(self.config, backend).mutate(seed)
                        self.assertNotEqual(result.spec.dtype, dtype)
                        self.assertEqual(seed.to_dict(), before)
                        expected = seed.to_dict()
                        expected['spec']['dtype'] = result.spec.dtype.value
                        self.assertEqual(result.to_dict(), expected)
                        result.validate()
                        restored = RegionProgram.from_dict(result.to_dict())
                        sig = TileSmith._make_sig(result)
                        self.assertNotEqual(sig, TileSmith._make_sig(seed))
                        self.assertEqual(sig, TileSmith._make_sig_from_dict(restored.to_dict()))
                        code = Oracle(self.config, backend)._emit_code(restored)
                        ast.parse(code)
                        self.assertIn('dtype=torch.' + result.spec.dtype.value, code)
                        # Switching back preserves the original program exactly.
                        self.assertEqual(Mutator(self.config, backend).mutate(result).to_dict(), before)

    def test_wider_dtype_repairs_schedule_without_changing_structure(self):
        for backend, limit in (('tilelang', 24576), ('triton', 40960)):
            module = 'src.backends.' + backend + '.params'
            constant = 'TILELANG_MAX_SHARED' if backend == 'tilelang' else 'TRITON_MAX_SHARED'
            with patch(module + '.' + constant, limit):
                seed = function_program('float16', 'gemm')
                seed.spec.block_M = seed.spec.block_N = 64
                seed.spec.block_K = 32
                # fp16 fits, fp32 does not fit this schedule.
                result = Mutator(self.config, backend).mutate(seed)
                self.assertEqual(result.spec.dtype, DataType.FLOAT32)
                self.assertEqual((result.spec.M, result.spec.N, result.spec.K),
                                 (seed.spec.M, seed.spec.N, seed.spec.K))
                self.assertEqual(result.body, seed.body)
                self.assertEqual(result.functions, seed.functions)
                from importlib import import_module
                check = import_module(module).check_shared_memory
                p = result.spec
                self.assertTrue(check(p.block_M, p.block_N, p.block_K, p.dtype, p.num_stages))
                result.validate()

    def test_probe_dtype_switch_all_kinds_and_resource_repair(self):
        from src.backends.tilelang.params import check_shared_memory as tilelang_check_shared_memory
        from src.backends.triton.params import check_shared_memory as triton_check_shared_memory
        for kind in KINDS:
            seed = probe_program(TileKernel('probe', compute_kind=kind,
                                coverage_probe=True, N=129, dtype=DataType.FLOAT16))
            before = seed.spec.params_dict
            mutated = mutate_probe(seed, self.config)
            k = mutated.spec
            self.assertEqual(k.dtype, DataType.FLOAT32)
            self.assertEqual(seed.spec.params_dict, before)
            self.assertEqual((k.M, k.N, k.K), (seed.spec.M, 129, seed.spec.K))
            if kind == ComputeKind.GEMM_ARGMAX:
                for check in (tilelang_check_shared_memory, triton_check_shared_memory):
                    self.assertTrue(check(k.block_M, k.block_N, k.block_K, k.dtype, k.num_stages))
            for backend in ('tilelang', 'triton'):
                ast.parse(Oracle(self.config, backend)._emit_code(mutated))

    def test_campaign_probe_dispatch_and_single_dtype_config(self):
        from src.workflow.generator import ProgramGenerator
        config = Config(coverage_probe_prob=1, dtype_mutate_prob=1)
        seed = ProgramGenerator(config).generate()
        result = Mutator(config).mutate(seed)
        self.assertNotEqual(result.spec.dtype, seed.spec.dtype)
        for configured in (['float16'], [DataType.FLOAT16]):
            config = Config(supported_dtypes=configured, dtype_mutate_prob=1)
            for probe_prob in (0, 1):
                config.coverage_probe_prob = probe_prob
                gen = ProgramGenerator(config)
                for _ in range(10):
                    result = Mutator(config).mutate(gen.generate())
                    self.assertEqual(result.spec.dtype, DataType.FLOAT16)
                    result.validate()

    def test_probe_dtype_mutation_can_be_disabled(self):
        seed = probe_program(TileKernel('probe', compute_kind=ComputeKind.COPY,
                                                   coverage_probe=True))
        for _ in range(30):
            result = mutate_probe(seed, Config(dtype_mutate_prob=0))
            self.assertEqual(result.spec.dtype, seed.spec.dtype)


if __name__ == '__main__':
    unittest.main()
