"""Phase 3 gate: the int8 x int8 region GEMM surface.

Generation stays inside the pre-validated INT8_SPEC_GRID and never mixes an
int32 accumulator with float32 fragment ops (gemm-only body, no functions).
Emission accumulates in int32 on both backends; the reference is the exact
integer matmul compared with a one-unit absolute slack; mutation never
rewrites a spec into int8 (supported_dtypes excludes it), and int8 inputs
must survive input scaling (sub-unit scales truncate everything to zero).
"""
import ast
import random
import unittest
from dataclasses import asdict

import torch

from src.config import Config
from src.ir import DataType
from src.ir.serialization import program_from_dict, program_to_dict
from src.workflow.emitter.region_checks import _region_input
from src.workflow.emitter.region_runtime import _region_reference
from src.workflow.generator.grids import INT8_SPEC_GRID
from src.workflow.generator.region_generator import RegionGenerator
from src.workflow.mutator.mutator import Mutator
from src.workflow.oracle import Oracle

GRID_CELLS = {(c['m'], c['n'], c['k'], c['block_m'], c['block_n'], c['block_k'], c['threads'], c['stages'])
              for c in INT8_SPEC_GRID}


def int8_config(**overrides):
    config = Config(region_int8_prob=1, coverage_probe_prob=0, function_min_count=0,
                    region_typed_prob=0)
    for key, value in overrides.items():
        setattr(config, key, value)
    return config


def int8_program(backend='tilelang', seed=0):
    config = int8_config()
    random.seed(seed)
    return RegionGenerator(config, backend).generate()


class Int8GenerationTests(unittest.TestCase):
    def test_generation_rolls_gemm_only_entries(self):
        for seed in range(8):
            with self.subTest(seed=seed):
                program = int8_program(seed=seed)
                self.assertEqual(program.spec.dtype, DataType.INT8)
                self.assertEqual([op.kind for op in program.body.operations], ['gemm'])
                self.assertEqual(program.functions, [])
                self.assertFalse(program.typed)
                self.assertGreaterEqual(program.input_scale, 1.0)

    def test_spec_stays_inside_prevalidated_grid(self):
        for backend in ('tilelang', 'triton'):
            for seed in range(8):
                with self.subTest(backend=backend, seed=seed):
                    program = int8_program(backend, seed)
                    spec = program.spec
                    cell = (spec.M, spec.N, spec.K, spec.block_M, spec.block_N,
                            spec.block_K, spec.threads, spec.num_stages)
                    self.assertIn(cell, GRID_CELLS)
                    self.assertIn(spec.block_K, (32, 64))
                    self.assertGreaterEqual(spec.K, 32)

    def test_pass_config_sampling_never_disables_vectorization_for_int8(self):
        """tirx.disable_vectorize de-vectorizes the shared-memory copy, and an
        int8 cp_async transfer then falls to a 1-byte width that tilelang's
        CUDA codegen rejects outright ({4, 8, 16} required) — the key must
        never reach an int8 kernel. GPU-verified: seed 5's sampled variant
        crashed every int8 draw with this key present."""
        from src.backends.common.knobs import TILELANG_REGION_PASS_POOL, region_pass_configs
        for seed in range(16):
            with self.subTest(seed=seed):
                program = int8_program('tilelang', seed)
                options = region_pass_configs(program, Config(seed=seed))
                self.assertNotIn('tirx.disable_vectorize', options)
        # The float32 pool keeps the key: it is a real coverage corner there.
        self.assertIn('tirx.disable_vectorize', TILELANG_REGION_PASS_POOL)
        # The emitted int8 pass-config decorator must never carry it either.
        program = int8_program('tilelang', seed=5)
        code = Oracle(int8_config(), 'tilelang')._emit_code(program)
        ast.parse(code)
        for line in code.splitlines():
            if line.startswith('@tilelang.jit(pass_configs='):
                self.assertNotIn('tirx.disable_vectorize', line)

    def test_int8_inputs_survive_scaling(self):
        program = int8_program(seed=1)
        spec = program.spec
        a = _region_input((spec.M, spec.K), torch.int8, 'normal', program.input_scale, device='cpu')
        b = _region_input((spec.K, spec.N), torch.int8, 'normal', program.input_scale, device='cpu')
        self.assertEqual(a.dtype, torch.int8)
        self.assertEqual(b.dtype, torch.int8)
        self.assertGreater((a != 0).float().mean().item(), 0.5)
        self.assertGreater((b != 0).float().mean().item(), 0.5)
        self.assertLessEqual(a.abs().max().item(), 7 * program.input_scale + 1)

    def test_reference_is_the_exact_integer_matmul(self):
        for seed in range(4):
            with self.subTest(seed=seed):
                program = int8_program(seed=seed)
                spec = program.spec
                a = _region_input((spec.M, spec.K), torch.int8, 'normal', program.input_scale, device='cpu')
                b = _region_input((spec.K, spec.N), torch.int8, 'normal', program.input_scale, device='cpu')
                ref = _region_reference(a, b, asdict(program.body), spec.block_M, spec.block_N, 'int32')
                # The interpreter pads internally for tile execution and slices
                # the result back to (M, N), matching the kernel output shape.
                expected = a.to(torch.int32) @ b.to(torch.int32)
                self.assertEqual(ref.shape, (spec.M, spec.N))
                self.assertTrue(torch.equal(ref, expected))

    def test_roundtrip_preserves_int8_spec(self):
        program = int8_program(seed=2)
        restored = program_from_dict(program_to_dict(program))
        self.assertEqual(restored.spec.dtype, DataType.INT8)
        self.assertEqual(restored.spec.block_K, program.spec.block_K)
        self.assertEqual(restored.input_scale, program.input_scale)
        self.assertEqual([op.kind for op in restored.body.operations], ['gemm'])
        for backend in ('tilelang', 'triton'):
            from src.backends import get_backend
            get_backend(backend).validate_program(restored)


class Int8EmissionTests(unittest.TestCase):
    def test_tilelang_emission_accumulates_in_int32(self):
        from src.backends.tilelang.region import tilelang_code
        program = int8_program('tilelang', seed=3)
        code = tilelang_code(program)
        ast.parse(code)
        self.assertIn('dtype = "int8"', code)
        self.assertIn('c_dtype = "int32" if dtype == "int8" else dtype', code)
        self.assertIn('frag_dtype = "int32" if dtype == "int8" else "float32"', code)
        self.assertIn('T.gemm(As, Bs,', code)
        self.assertIn(f'As = T.alloc_shared(({program.spec.block_M}, {program.spec.block_K}), dtype)', code)

    def test_triton_emission_accumulates_in_int32(self):
        from src.backends.triton.region import triton_code
        program = int8_program('triton', seed=3)
        code = triton_code(program)
        ast.parse(code)
        self.assertIn('tl.full((', code)
        self.assertIn('tl.int32', code)
        self.assertIn('out_dtype=tl.int32', code)

    def test_oracle_reproducer_uses_int8_inputs_and_exact_tolerance(self):
        for backend in ('tilelang', 'triton'):
            with self.subTest(backend=backend):
                program = int8_program(backend, seed=4)
                code = Oracle(int8_config(), backend)._emit_code(program)
                ast.parse(code)
                self.assertIn('dtype=torch.int8', code)
                self.assertIn('tolerance=1.0', code)
                # The checked runner allocates output with reference.dtype; the
                # int32 accumulator flows through the reference's output dtype.
                self.assertIn("'int32'", code)


class Int8MutationTests(unittest.TestCase):
    def test_mutation_never_produces_int8(self):
        for backend in ('tilelang', 'triton'):
            with self.subTest(backend=backend):
                mutator = Mutator(int8_config(region_int8_prob=0, dtype_mutate_prob=1), backend)
                for seed in range(10):
                    random.seed(seed)
                    program = RegionGenerator(int8_config(region_int8_prob=0), backend).generate()
                    for _ in range(6):
                        mutated = mutator.mutate(program)
                        self.assertNotEqual(mutated.spec.dtype, DataType.INT8)
                        program = mutated

    def test_mutating_an_int8_program_stays_valid_and_never_returns_to_int8(self):
        for backend in ('tilelang', 'triton'):
            with self.subTest(backend=backend):
                mutator = Mutator(int8_config(dtype_mutate_prob=1), backend)
                program = int8_program(backend, seed=5)
                for _ in range(6):
                    mutated = mutator.mutate(program)
                    self.assertNotEqual(mutated.spec.dtype, DataType.INT8)
                    program = mutated


if __name__ == '__main__':
    unittest.main()
