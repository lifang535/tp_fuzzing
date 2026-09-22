"""Transcendental elementwise surface: registry, emission, and reference branches.

The 10 ops (tanh/erf/log/log2/exp2/rsqrt/sin/cos/floor/ceil) extend the region
op surface with new compiler code paths. Domain sanitization (clamps) must be
identical between the backend op tables and the reference interpreters; the
emission assertions pin the exact source text and the reference tests pin the
semantics each interpreter attaches to it.
"""
import ast
import unittest
from dataclasses import asdict

import torch
from src.config import Config
from src.ir import TileKernel, DataType, ComputeKind
from src.ir.region import Operation as Op, Region, RegionProgram, RegionExecution
from src.ir.region_ops import OPS
from src.ir.region_types import result_type, ValueType, FULL
from src.ir.serialization import program_from_dict, program_to_dict
from src.workflow.emitter.region_runtime import _region_reference
from src.workflow.emitter.typed_region_runtime import _typed_region_reference
from src.workflow.oracle import Oracle
from src.workflow.coverage_audit import program_capabilities, TRANSCENDENTAL_OPS
from src.workflow.generator.region_generator import RegionGenerator
from src.workflow.fuzzer.fuzzer import TileSmith

TRANSCENDENTALS = ('tanh', 'erf', 'log', 'log2', 'exp2', 'rsqrt', 'sin', 'cos', 'floor', 'ceil')

TORCH_EXPR = {
    'tanh': lambda x: x.tanh(),
    'erf': lambda x: torch.erf(x),
    'log': lambda x: x.abs().clamp_min(1e-3).log(),
    'log2': lambda x: x.abs().clamp_min(1e-3).log2(),
    'exp2': lambda x: x.clamp(-10, 10).exp2(),
    'rsqrt': lambda x: x.abs().clamp_min(1e-6).rsqrt(),
    'sin': lambda x: x.sin(),
    'cos': lambda x: x.cos(),
    'floor': lambda x: x.floor(),
    'ceil': lambda x: x.ceil(),
}

TILELANG_EXPR = {
    'tanh': 'T.tanh(',
    'erf': 'T.erf(',
    'log': 'T.log(T.max(T.abs(',
    'log2': 'T.log2(T.max(T.abs(',
    'exp2': 'T.exp2(T.min(T.max(',
    'rsqrt': 'T.rsqrt(T.max(T.abs(',
    'sin': 'T.sin(',
    'cos': 'T.cos(',
    'floor': 'T.floor(',
    'ceil': 'T.ceil(',
}

TRITON_EXPR = {
    'tanh': 'tl.extra.libdevice.tanh(',
    'erf': 'tl.erf(',
    'log': 'tl.log(tl.maximum(tl.abs(',
    'log2': 'tl.log2(tl.maximum(tl.abs(',
    'exp2': 'tl.exp2(tl.minimum(tl.maximum(',
    'rsqrt': 'tl.rsqrt(tl.maximum(tl.abs(',
    'sin': 'tl.sin(',
    'cos': 'tl.cos(',
    'floor': 'tl.floor(',
    'ceil': 'tl.ceil(',
}


def transcendental_program(kind, typed=False):
    body = Region([], [Op('load', 'entry'), Op(kind, 'value', ['entry'])], 'value')
    spec = TileKernel('kernel_0', M=16, N=16, K=16, block_M=16, block_N=16, block_K=16,
                      threads=128, num_stages=2, dtype=DataType('float32'),
                      compute_kind=ComputeKind.COPY)
    p = RegionProgram(spec, body, typed=typed)
    if typed:
        p.execution = RegionExecution(input_pattern='integer', input_seed_count=1, repeat_count=1)
    p.validate()
    return p


class TranscendentalRegistryTests(unittest.TestCase):
    def test_ops_are_registered_as_unary_leaves(self):
        for kind in TRANSCENDENTALS:
            with self.subTest(kind=kind):
                self.assertIn(kind, OPS)
                self.assertEqual(OPS[kind].arity, 1)
                self.assertIn(kind, RegionGenerator.LEAVES)

    def test_result_type_follows_operand_dtype(self):
        for kind in TRANSCENDENTALS:
            with self.subTest(kind=kind):
                out = result_type(kind, [ValueType('float16')], {})
                self.assertEqual(out, ValueType('float16'))
                out = result_type(kind, [FULL], {})
                self.assertEqual(out, FULL)


class TranscendentalReferenceTests(unittest.TestCase):
    def test_native_reference_matches_torch_formulas(self):
        a = torch.randn(2, 3) * 0.5
        b = torch.randn(3, 3)
        for kind in TRANSCENDENTALS:
            with self.subTest(kind=kind):
                p = transcendental_program(kind)
                ref = _region_reference(a, b, asdict(p.body), 2, 3, 'float32')
                self.assertTrue(torch.allclose(ref, TORCH_EXPR[kind](a.float()), atol=1e-5))
                self.assertTrue(torch.isfinite(ref).all())

    def test_typed_reference_matches_torch_formulas(self):
        a = torch.randn(2, 3) * 0.5
        b = torch.randn(3, 3)
        for kind in TRANSCENDENTALS:
            with self.subTest(kind=kind):
                p = transcendental_program(kind, typed=True)
                ref = _typed_region_reference(a, b, asdict(p.body), 2, 3, 'float32')
                self.assertTrue(torch.allclose(ref, TORCH_EXPR[kind](a.float()), atol=1e-5))


class TranscendentalEmissionTests(unittest.TestCase):
    def test_native_emission_pins_backend_source_text(self):
        for kind in TRANSCENDENTALS:
            for backend, needle in (('tilelang', TILELANG_EXPR[kind]), ('triton', TRITON_EXPR[kind])):
                with self.subTest(kind=kind, backend=backend):
                    if kind == 'tanh' and backend == 'triton':
                        # tanh has no tl.tanh; the emitter probes the libdevice
                        # binding available in this Triton (2.x vs 3.x) and
                        # falls back to an exp-based identity when neither
                        # exists. The pin must follow the probed binding.
                        from src.backends.triton.ops import _TANH_BINDING
                        needle = (f'{_TANH_BINDING}(' if _TANH_BINDING
                                  else '(1.0 - 2.0 / (1.0 + tl.exp(2.0 * ')
                    code = Oracle(Config(), backend)._emit_code(transcendental_program(kind))
                    ast.parse(code)
                    self.assertIn(needle, code)

    def test_typed_emission_pins_backend_source_text(self):
        for kind in TRANSCENDENTALS:
            for backend, needle in (('tilelang', TILELANG_EXPR[kind]), ('triton', TRITON_EXPR[kind])):
                with self.subTest(kind=kind, backend=backend):
                    if kind == 'tanh' and backend == 'triton':
                        from src.backends.triton.ops import _TANH_BINDING
                        needle = (f'{_TANH_BINDING}(' if _TANH_BINDING
                                  else '(1.0 - 2.0 / (1.0 + tl.exp(2.0 * ')
                    code = Oracle(Config(), backend)._emit_code(transcendental_program(kind, typed=True))
                    ast.parse(code)
                    self.assertIn(needle, code)

    def test_serialization_round_trip_preserves_new_ops(self):
        for typed in (False, True):
            p = transcendental_program('tanh', typed=typed)
            restored = program_from_dict(program_to_dict(p))
            self.assertEqual(TileSmith._make_sig(p), TileSmith._make_sig(restored))

    def test_coverage_audit_reports_the_capability(self):
        for kind in TRANSCENDENTALS:
            with self.subTest(kind=kind):
                self.assertIn(kind, TRANSCENDENTAL_OPS)
        self.assertIn('transcendental_elementwise',
                      program_capabilities(transcendental_program('tanh')))


if __name__ == '__main__':
    unittest.main()
