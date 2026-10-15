"""Feature-slice programs: legalization, harness emission, exact references,
scheduling, persistence and campaign integration (CPU only)."""
import json
import random
import re
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
    # only the module's own imports: helpers import the DSLs inside functions
    code = re.sub(r'^(?:import (?:triton|tilelang)(?:\.language as \w+)?|from triton\.language\.extra\.\w+ import libdevice)$',
                  '', code, flags=re.M)
    code = code.replace('@triton.jit', '')
    # kernel signatures may annotate tl.constexpr, evaluated at definition
    namespace = {'__name__': 'slice_harness', 'tl': type('tl', (), {'constexpr': object})}
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

    def test_legalization_is_idempotent(self):
        # b1eded9f vGPU Triton crashed in emit: a cumsum layout program's
        # second legalization read arg1 after the first had zeroed it.
        crashed = {'source': 'cumsum', 'dt': 'f32', 'shape': '64x8', 'k': 16, 'store': 'native', 'out_dt': 'i8',
                   'pair': 0, 'warps': 1, 'warps2': 1, 'op1': 'none', 'arg1': 1, 'op2': 'split', 'arg2': 3,
                   'op3': 'flip', 'arg3': 2, 'cfg_maxnreg': 0, 'cfg_fusion': 0, 'cfg_target': 'native',
                   'cfg_amd': 'none'}
        program = make_program('layout', crashed, 'triton')
        emit_slice(program, Cfg)
        rng = random.Random(7)
        for backend in BACKENDS:
            for name, slice_ in SLICES.items():
                if backend not in slice_.backends:
                    continue
                for _ in range(3000):
                    params = slice_.sample(rng, backend)
                    slice_.legalize(params, backend)
                    again = dict(params)
                    slice_.legalize(again, backend)
                    self.assertEqual(again, params, f'{name}/{backend}')

    def test_unlegalized_parameters_are_rejected(self):
        program = next(programs(1))
        broken = SliceProgram(program.slice, dict(program.params, out_dt='nonsense'), program.backend)
        with self.assertRaises(ValueError):
            broken.validate()

    def test_every_chain_value_is_representable(self):
        for program in programs(16, seed=1):
            params = dict(program.params)
            plan = SLICES[program.slice].legalize(params, program.backend)
            if 'pre' not in plan:
                continue
            for op, dtype, target, source, constant in plan['pre'] + plan['post']:
                self.assertIn(dtype, DTYPES)
                self.assertIn(target, DTYPES)
                if op != 'none':
                    self.assertFalse(DTYPES[dtype].is_fp8, (program.slice, op, dtype))
            self.assertTrue(DTYPES[plan['out_dt']].torch)

    def test_triton_max_min_widen_narrow_types(self):
        base = SLICES['reduce'].sample(random.Random(0), 'triton')
        for core, widened in (('u16', 'i32'), ('i8', 'i32'), ('f16', 'f32'), ('bf16', 'f32'), ('i64', 'i64')):
            params = dict(base, in_dt=core, core_dt=core, kind='max', values='nonneg', pre1_op='none',
                          pre2_op='none', pre1_dt=core, pre2_dt=core, post1_op='none', post1_dt=widened,
                          out_dt='f64')
            plan = SLICES['reduce'].legalize(params, 'triton')
            self.assertEqual(params['kind'], 'max')
            self.assertEqual(plan['final_dt'], widened, core)

    def test_backend_restrictions(self):
        for program in programs(16, seed=2):
            params = program.params
            if program.backend == 'triton' and program.slice != 'round':
                plan = SLICES[program.slice].legalize(dict(params), 'triton')
                for op, dtype, *_ in plan.get('pre', []) + plan.get('post', []):
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
            if program.slice in ('round', 'math'):
                continue  # inexact by design (RoundTests) or per-element (MathTests)
            with self.subTest(slice=program.slice, backend=program.backend, params=program.params):
                ns = harness_namespace(program)
                params = dict(program.params)
                plan = SLICES[program.slice].legalize(params, program.backend)
                x_shape, y_shape = SLICES[program.slice].input_shapes(plan)
                d = plan['input_domain']
                x = ns['_slice_values'](x_shape, d.lo, d.hi, d.frac, 3, 1)
                y = ns['_slice_values'](y_shape, d.lo, d.hi, d.frac, 3, 2)
                expected = ns['reference'](x, y)
                # a layout view may reinterpret integer bits as NaN or infinity
                special = program.params.get('padding') == 'nan' or any(
                    step['op'] == 'view' and DTYPES[step['dt']].is_float for step in plan.get('steps', []))
                if not special:
                    self.assertFalse(torch.isnan(expected).any())
                expected = torch.nan_to_num(expected, nan=0.0, posinf=float('inf'), neginf=float('-inf'))
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


class RoundTests(unittest.TestCase):
    def test_rounding_matches_correct_single_rounding(self):
        import numpy as np
        from src.workflow.slices.runtime import _slice_round, _slice_round_values
        g = torch.Generator().manual_seed(1)
        x = torch.randn(20000, generator=g, dtype=torch.float64) * torch.pow(2.0, torch.randint(-30, 20, (20000,), generator=g)).double()
        self.assertTrue(torch.equal(_slice_round(x, 'f16'), torch.from_numpy(x.numpy().astype(np.float16).astype(np.float64))))
        x32 = x.float()
        for fmt, dtype in (('bf16', torch.bfloat16), ('f16', torch.float16)):
            ref = _slice_round(x32.double(), fmt)
            got = x32.to(dtype).double()
            self.assertTrue(bool(((ref == got) | (torch.isnan(ref) & torch.isnan(got))).all()), fmt)
        small = x32[x32.abs() <= 448]
        self.assertTrue(torch.equal(_slice_round(small.double(), 'f8e4'), small.to(torch.float8_e4m3fn).double()))
        # toward zero never increases a magnitude and stays on the grid
        rtz = _slice_round(x, 'f16', 'rtz')
        self.assertTrue(bool((rtz.abs() <= x.abs()).all()))
        self.assertTrue(torch.equal(_slice_round(rtz, 'f16'), rtz))
        # generated inputs are exact in their own format and inside both ranges
        for src, dst, mix in (('f32', 'f8e4', 'ties'), ('f64', 'f16', 'edges'), ('i32', 'bf16', 'ties'),
                              ('f32', 'i32', 'wide'), ('u8', 'f16', 'ties'), ('bf16', 'i8', 'wide')):
            values = _slice_round_values(src, dst, 512, 3, mix)
            self.assertEqual(len(values), 512)
            self.assertTrue(bool(torch.isfinite(values).all()))
            if src[0] in 'iu':
                self.assertTrue(torch.equal(values, torch.trunc(values)))
            else:
                self.assertTrue(torch.equal(_slice_round(values, src), values), (src, dst))
            if dst[0] in 'iu':
                bits = int(dst[1:])
                self.assertTrue(bool((values.abs() < 2.0 ** (bits - 1)).all()))

    def test_legalization_keeps_conversions_meaningful(self):
        rng = random.Random(6)
        for backend in BACKENDS:
            for _ in range(200):
                program = make_program('round', SLICES['round'].sample(rng, backend), backend)
                p = program.params
                self.assertNotEqual(p['src'], p['dst'])
                self.assertFalse(p['src'][0] in 'iu' and p['dst'][0] in 'iu')
                if p.get('mode') == 'rtz':
                    self.assertTrue(p['src'].startswith(('f', 'b')) and p['dst'].startswith(('f', 'b')))
                if p['via'] != 'none':
                    self.assertNotEqual(p['via'], p['dst'])


class MathTests(unittest.TestCase):
    def test_rounding_helpers_match_numpy(self):
        import numpy as np
        from fractions import Fraction
        from src.workflow.slices.runtime import _slice_fnext, _slice_fround, _slice_fsqrt
        rng = random.Random(2)
        for _ in range(2000):
            a = np.float32(rng.uniform(-1, 1) * 2.0 ** rng.randint(-40, 40))
            b = np.float32(rng.uniform(-1, 1) * 2.0 ** rng.randint(-40, 40))
            self.assertEqual(_slice_fround(Fraction(float(a)) * Fraction(float(b)), 'f32'), float(a * b))
            self.assertEqual(_slice_fround(Fraction(float(a)) / Fraction(float(b)), 'f32'), float(a / b))
            self.assertEqual(_slice_fsqrt(abs(float(a)), 'f32'), float(np.sqrt(np.abs(a))))
            self.assertEqual(_slice_fnext(float(a), float(b), 'f32'), float(np.nextafter(a, b)))
            h = np.float16(rng.uniform(-1, 1) * 2.0 ** rng.randint(-20, 15))
            self.assertEqual(_slice_fround(Fraction(float(h)) * 3, 'f16'), float(np.float16(h * np.float16(3))))
            v = Fraction(rng.randint(1, 10 ** 9), rng.randint(1, 10 ** 9)) * rng.choice((-1, 1))
            down, up = _slice_fround(v, 'f32', 'rd'), _slice_fround(v, 'f32', 'ru')
            self.assertTrue(down <= v <= up)
            self.assertIn(_slice_fround(v, 'f32', 'rn'), (down, up))
            self.assertLessEqual(abs(_slice_fround(v, 'f32', 'rz')), abs(v))
        self.assertEqual(_slice_fround(65520, 'f16'), float('inf'))
        self.assertEqual(_slice_fround(65520, 'f16', 'rz'), 65504.0)

    def test_philox_matches_triton(self):
        # Values produced by Triton 3.8 tl.randint / tl.rand on an RTX 4060.
        from src.workflow.slices.runtime import _slice_meval
        offsets = [0, 1, 2, 1000, 2 ** 31 - 1, -5, 123456789, -2 ** 31]
        expected = [3522838145, 11954473, 1814877333, 2817279776, 3552396717, 866953835, 4131302393, 4117041148]
        self.assertEqual([_slice_meval('philox', '12345:10:0', [v], ['i32'], 'u32') for v in offsets], expected)
        self.assertEqual([_slice_meval('philox', f'{0x9876543210}:10:0', [v], ['i64'], 'u32')
                          for v in (0, 1, 2 ** 40 + 7, -3)], [3098728694, 3079670387, 2806096737, 348501100])
        self.assertEqual(_slice_meval('rand', '4242:10:0', [0], ['i32'], 'f32'), 0.8976094126701355)

    def test_reference_cases_cover_the_catalog(self):
        """Every (function, dtype) of both catalogs yields operands of its
        types and results representable in its output type."""
        import math
        from src.workflow.slices.mathfn import CATALOGS
        from src.workflow.slices.runtime import _SLICE_FORMATS, _slice_fround, _slice_int_range
        for backend, catalog in CATALOGS.items():
            space = SLICES['math'].space(backend)
            base = dict(SLICES['math'].sample(random.Random(0), backend), shape=space['shape'][0], tail='none')
            for name, fn in catalog.items():
                for dt in fn.dtypes:
                    for values in sorted({r.split(':')[0] for r in fn.allowed_regimes(dt)}):
                        program = make_program('math', dict(base, fn=name, in_dt=dt, values=values), backend)
                        with self.subTest(backend=backend, fn=name, dt=dt, values=values):
                            ns = harness_namespace(program)
                            args, want = ns['reference_case'](3)
                            plan = SLICES['math'].legalize(dict(program.params), backend)
                            self.assertEqual(len(args), fn.arity)
                            self.assertEqual(len(want), math.prod(plan['valid']))
                            for operand, kind in zip(args, plan['types']):
                                if kind[0] in 'iu':
                                    lo, hi = _slice_int_range(kind)
                                    self.assertTrue(all(lo <= v <= hi for v in operand), kind)
                                else:
                                    self.assertTrue(all(not math.isfinite(v) or _slice_fround(v, kind) == v
                                                        for v in operand), kind)
                            out = plan['out']
                            for w in want:
                                if w is None:
                                    continue
                                if out[0] in 'iu':
                                    lo, hi = _slice_int_range(out)
                                    self.assertTrue(lo <= w <= hi, (w, out))
                                elif fn.ulps == 0 and math.isfinite(w) and out in _SLICE_FORMATS:
                                    self.assertEqual(_slice_fround(w, out), w, (w, out))


class HarnessModelTests(unittest.TestCase):
    """Semantics the b9e08b4c campaign showed the harness must model."""

    def test_libdevice_unsigned_conversions_are_int32(self):
        from src.workflow.slices.mathfn import CATALOGS
        fn = CATALOGS['triton']['double2uint_rn']
        self.assertEqual((fn.out, fn.wraps), ('i32', 'u32'))
        program = make_program('math', dict(SLICES['math'].sample(random.Random(0), 'triton'), fn='double2uint_rn',
                                            in_dt='f64', values='wide', widen=1), 'triton')
        code = emit_slice(program, Cfg)
        self.assertIn("_slice_wrap(w, 'i32')", code)
        ns = harness_namespace(program)
        self.assertEqual(ns['_slice_wrap'](ns['_slice_meval']('convert', 'rn', [3464260925.0], ['f64'], 'u32'),
                                           'i32'), 3464260925 - 2 ** 32)

    def test_approximations_may_overflow_to_infinity(self):
        from src.workflow.slices.runtime import _slice_mcheck
        inf = float('inf')
        _slice_mcheck([inf, -inf], [4.52e38, -1.47e40], 'x', 'bf16', 16, False)
        _slice_mcheck([inf], [65504.0 * (1 + 2 ** -12)], 'x', 'f16', 1, False)
        for got, want in (([-inf], [4.52e38]), ([inf], [1.0e30]), ([inf], [65504.0 / 2])):
            with self.assertRaises(RuntimeError):
                _slice_mcheck(got, want, 'x', 'f16' if want[0] < 1e5 else 'bf16', 16, False)

    def test_fma_of_special_values_is_exact(self):
        import math
        from src.workflow.slices.runtime import _slice_meval
        big = 1.7976931348623157e308

        def fma(x, y, z, dt='f64', mode='rn'):
            return _slice_meval('fma', mode, [x, y, z], [dt] * 3, dt)
        self.assertEqual(fma(big, -1478974.0, math.inf), math.inf)
        self.assertEqual(fma(big, 2.0, -math.inf, mode='rz'), -math.inf)
        self.assertEqual(fma(math.inf, 2.0, 1.0), math.inf)
        self.assertTrue(math.isnan(fma(-math.inf, 2.0, math.inf)))
        self.assertTrue(math.isnan(fma(math.inf, 0.0, 1.0)))
        self.assertTrue(math.isnan(fma(1.0, math.nan, 1.0)))
        self.assertEqual(fma(math.inf, -0.5, -math.inf), -math.inf)

    def test_triton_libdevice_f32_is_flush_to_zero(self):
        from src.workflow.slices.runtime import _slice_ftz
        self.assertIsNone(_slice_ftz(2.0 ** -149, [0.0, 1.0], ['f32', 'f32'], 'f32'))
        self.assertIsNone(_slice_ftz(1.0, [2.0 ** -130, 1.0], ['f32', 'f32'], 'f32'))
        self.assertEqual(_slice_ftz(2.0 ** -149, [0.0, 1.0], ['f64', 'f64'], 'f64'), 2.0 ** -149)
        self.assertEqual(_slice_ftz(0.5, [0.0, 1.0], ['f32', 'f32'], 'f32'), 0.5)
        for backend, expected in (('triton', True), ('tilelang', False)):
            program = make_program('math', dict(SLICES['math'].sample(random.Random(1), backend), fn='nextafter',
                                                in_dt='f32', values='small'), backend)
            self.assertEqual('_slice_ftz(w,' in emit_slice(program, Cfg), expected, backend)

    def test_triton_results_take_the_declared_type(self):
        program = make_program('math', dict(SLICES['math'].sample(random.Random(2), 'triton'), fn='maximum',
                                            in_dt='u16', yform='const', widen=0, values='small'), 'triton')
        code = emit_slice(program, Cfg)
        self.assertIn('r = r.to(tl.uint16)\n    r = r.to(tl.int16, bitcast=True)', code)

    def test_libdevice_integer_abs_is_undefined_at_the_minimum(self):
        from src.workflow.slices.mathfn import CATALOGS
        from src.workflow.slices.runtime import _slice_meval
        self.assertEqual(CATALOGS['triton']['ld_abs'].sem, 'cabs')
        self.assertIsNone(_slice_meval('cabs', 'rn', [-2 ** 31], ['i32'], 'i32'))
        self.assertIsNone(_slice_meval('cabs', 'rn', [-2 ** 63], ['i64'], 'i64'))
        self.assertEqual(_slice_meval('cabs', 'rn', [-7], ['i32'], 'i32'), 7)
        self.assertEqual(_slice_meval('cabs', 'rn', [-1.5], ['f32'], 'f32'), 1.5)
        self.assertEqual(_slice_meval('abs', 'rn', [-2 ** 31], ['i32'], 'i32'), -2 ** 31)  # tl.abs wraps

    def test_cdiv_follows_the_triton_formula(self):
        from src.workflow.slices.runtime import _slice_meval
        self.assertEqual(_slice_meval('cdiv', 'rn', [93, 43], ['i8', 'i8'], 'i8'), -2)
        self.assertEqual(_slice_meval('cdiv', 'rn', [7, 2], ['i32', 'i32'], 'i32'), 4)
        self.assertEqual(_slice_meval('cdiv', 'rn', [100, 60], ['u8', 'u8'], 'u8'), 2)
        self.assertEqual(_slice_meval('cdiv', 'rn', [0, 5], ['i16', 'i16'], 'i16'), 0)

    def test_tilelang_float_constants_are_typed(self):
        program = make_program('math', dict(SLICES['math'].sample(random.Random(3), 'tilelang'), fn='div',
                                            in_dt='f16', yform='const', values='small'), 'tilelang')
        self.assertEqual(program.params['yform'], 'const')
        self.assertIn('T.cast(0.75, "float16")', emit_slice(program, Cfg))
        program = make_program('math', dict(program.params, fn='mul', in_dt='i8'), 'tilelang')
        self.assertIn('* 3)', emit_slice(program, Cfg))

    def test_loop_envelope_counts_both_inputs(self):
        from src.workflow.slices.loop import envelope, fits
        rng = random.Random(4)
        for _ in range(400):
            params = dict(SLICES['loop'].sample(rng, 'tilelang'), inputs=2, body='axpy')
            program = make_program('loop', params, 'tilelang')
            p = program.params
            plan = SLICES['loop'].legalize(dict(p), 'tilelang')
            if p['inputs'] != 2:
                continue
            nonneg = DTYPES[plan['in_dt']].kind == 'uint'
            bound = envelope(plan['body'], p['coef'], p['trips'], (0 if nonneg else -4, 4), p['k'], 1, plan['init'])
            self.assertTrue(fits(*bound, plan['acc_dt']), p)

    def test_memory_coalesced_width_fits_each_thread(self):
        rng = random.Random(5)
        seen = Counter()
        for _ in range(3000):
            program = make_program('memory', SLICES['memory'].sample(rng, 'tilelang'), 'tilelang')
            p = program.params
            if not p['coalesced']:
                continue
            rows, cols = (int(v) for v in p['tile'].split('x'))
            seen[p['coalesced']] += 1
            for threads in {p['threads'], p['threads2']}:
                self.assertEqual((rows * cols) % (threads * p['coalesced']), 0, p)
            self.assertEqual(cols % p['coalesced'], 0)
            self.assertLessEqual(p['coalesced'] * DTYPES[p['dt']].bits // 8, 16)
        self.assertGreaterEqual(len(seen), 3, seen)

    def test_tilelang_uint64_neutral_is_not_a_long_literal(self):
        import re
        base = SLICES['reduce'].sample(random.Random(6), 'tilelang')
        for kind in ('min', 'bitand'):
            params = dict(base, kind=kind, core_dt='u64', in_dt='u8', values='nonneg', pre1_op='none',
                          pre2_op='none', post1_op='none', out_dt='u64', clear=1)
            program = make_program('reduce', params, 'tilelang')
            if program.params['kind'] != kind or program.params['core_dt'] != 'u64':
                continue
            code = emit_slice(program, Cfg)
            self.assertIn('T.max_value("uint64")', code, kind)
            self.assertFalse([v for v in re.findall(r'\d{19,}', code) if int(v) >= 2 ** 63], kind)


class NewSliceTests(unittest.TestCase):
    def test_layout_reference_has_the_stored_shape(self):
        import math
        rng = random.Random(21)
        for backend in BACKENDS:
            for _ in range(60):
                program = make_program('layout', SLICES['layout'].sample(rng, backend), backend)
                plan = SLICES['layout'].legalize(dict(program.params), backend)
                ns = harness_namespace(program)
                x_shape, y_shape = SLICES['layout'].input_shapes(plan)
                x = ns['_slice_values'](x_shape, -3, 3, 0, 1, 1)
                y = ns['_slice_values'](y_shape, -3, 3, 0, 1, 2)
                expected = ns['reference'](x, y)
                if program.params.get('store') == 'flat':
                    self.assertEqual(tuple(expected.shape), (math.prod(plan['final_shape']),))
                else:
                    self.assertEqual(tuple(expected.shape), tuple(plan['final_shape']), program.params)
                for step in plan['steps']:
                    self.assertNotEqual(step['op'], 'none')

    def test_loop_reference_runs_the_loop(self):
        base = SLICES['loop'].sample(random.Random(0), 'triton')
        params = dict(base, loop='range', body='axpy', coef=2, trips=3, in_dt='f32', acc_dt='f32', order='rev',
                      addr='index', mask='none', inner=0, shape='64')
        program = make_program('loop', params, 'triton')
        self.assertEqual(program.params['order'], 'rev')
        ns = harness_namespace(program)
        x = torch.arange(3 * 64, dtype=torch.float64).view(3, 64)
        expected = ns['reference'](x, torch.zeros(1, 64))
        # reverse order: acc = ((x2) * 2 + x1) * 2 + x0
        self.assertTrue(torch.equal(expected, x[2] * 4 + x[1] * 2 + x[0]))

    def test_memory_reference_fills_masked_lanes(self):
        base = SLICES['memory'].sample(random.Random(0), 'triton')
        params = dict(base, access='ptr', dt='f32', tile='16x16', grid=1, origin=1, row_pad=4, view='row',
                      mask='bound', row_extent='cut', col_extent='cut', fill=7, transform='none', hint='none')
        program = make_program('memory', params, 'triton')
        plan = SLICES['memory'].legalize(dict(program.params), 'triton')
        ns = harness_namespace(program)
        x_shape, _ = SLICES['memory'].input_shapes(plan)
        x = torch.arange(x_shape[0] * x_shape[1], dtype=torch.float64).view(x_shape)
        expected = ns['reference'](x, None)
        rows, cols = plan['cut_rows'], plan['cut_cols']
        self.assertTrue(torch.equal(expected[:rows, :cols], x[1:rows + 1, 1:cols + 1]))
        self.assertTrue(bool((expected[rows:] == 7).all()) and bool((expected[:, cols:] == 7).all()))

    def test_input_operands_convert_legally(self):
        from src.workflow.slices.chain import cast_allowed
        rng = random.Random(23)
        for _ in range(3000):
            backend = rng.choice(BACKENDS)
            name = rng.choice(('cast', 'reduce', 'scan', 'atomic'))
            program = make_program(name, SLICES[name].sample(rng, backend), backend)
            plan = SLICES[name].legalize(dict(program.params), backend)
            for op, dtype, target, source, constant in plan.get('pre', []):
                if source == 'input':
                    self.assertTrue(cast_allowed(plan['in_dt'], dtype, backend), program.params)

    def test_new_slices_reduce_to_their_cores(self):
        from src.workflow.slices.minimize import minimize
        rng = random.Random(24)
        for name, knob in (('math', 'fn'), ('layout', 'source'), ('loop', 'body'), ('memory', 'access')):
            for backend in BACKENDS:
                program = make_program(name, SLICES[name].sample(rng, backend), backend)
                value = program.params[knob]
                reduced, kept, _ = minimize(program, lambda c: c.params[knob] == value, budget=200)
                self.assertEqual(reduced.params[knob], value)
                simple = SLICES[name].SIMPLEST.get(knob)
                if isinstance(simple, dict):
                    simple = simple[backend]
                if value != simple:
                    self.assertIn(knob, kept, (name, backend))


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
        self.assertEqual(scheduler.avoided['cast', 'b'], 1)
        self.assertEqual(scheduler.stats()['cast']['avoided'], {'b': 1})
        restored = SliceScheduler('triton', ['cast'])
        restored.restore(json.loads(json.dumps(scheduler.snapshot())))
        self.assertEqual(restored.avoided, scheduler.avoided)

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


class TargetTests(unittest.TestCase):
    """Compilation targets are a knob every slice samples and every harness
    honours before its first compilation."""

    def test_targets_are_sampled_beside_the_native_gpu(self):
        from src.workflow.slices.base import TARGET_PROB, TARGETS
        rng = random.Random(41)
        seen = Counter()
        for backend in BACKENDS:
            for name, slice_ in SLICES.items():
                self.assertEqual(slice_.space(backend)['cfg_target'][0], 'native')
                for _ in range(40):
                    program = make_program(name, slice_.sample(rng, backend), backend)
                    seen[program.params['cfg_target'] != 'native'] += 1
                    self.assertIn(program.params['cfg_target'], TARGETS[backend])
        share = seen[True] / sum(seen.values())
        self.assertLess(abs(share - TARGET_PROB), 0.08, share)

    def test_harness_redirects_before_compiling_and_skips_checks_when_compile_only(self):
        rng = random.Random(42)
        for backend in BACKENDS:
            for name, slice_ in SLICES.items():
                params = dict(slice_.sample(rng, backend), cfg_target='sm90')
                code = emit_slice(make_program(name, params, backend), Cfg)
                main = code[code.index('def main():'):]
                redirect = main.index(f"compile_only = _slice_target('sm90', '{backend}', ")
                self.assertLess(redirect, main.index('launch('))
                self.assertIn('if compile_only:\n                break', main)
                self.assertLess(main.index('if compile_only:'), main.index('guard'))

    def test_amd_math_keeps_only_entries_with_an_amd_lowering(self):
        from src.workflow.slices.mathfn import CATALOGS, HIP_TARGETS, hip_supports
        slice_ = SLICES['math']
        moved = kept = 0
        for fn_name, fn in CATALOGS['triton'].items():
            params = dict(slice_.sample(random.Random(fn_name), 'triton'), fn=fn_name, cfg_target='gfx942')
            program = make_program('math', params, 'triton')
            if hip_supports(fn):
                kept += 1
                self.assertEqual(program.params['cfg_target'], 'gfx942')
                code = emit_slice(program, Cfg)
                self.assertIn('from triton.language.extra.hip import libdevice', code)
            else:
                moved += 1
                self.assertNotIn(program.params['cfg_target'], HIP_TARGETS)
        self.assertGreater(moved, 0)
        self.assertGreater(kept, 0)

    def test_reduction_resets_the_target_first(self):
        from src.workflow.slices.minimize import ORDER, minimize
        self.assertEqual(ORDER[0], 'cfg_target')
        rng = random.Random(43)
        program = make_program('cast', dict(SLICES['cast'].sample(rng, 'triton'), cfg_target='gfx942'), 'triton')
        tried = []

        def still_fails(candidate):
            tried.append(candidate.params['cfg_target'])
            return True
        reduced, core, _ = minimize(program, still_fails)
        self.assertEqual(tried[0], 'native')
        self.assertEqual(reduced.params['cfg_target'], 'native')
        self.assertNotIn('cfg_target', core)

    def test_amd_configs_never_reach_a_cuda_target(self):
        """The b1eded9f campaign ran 2818 Triton programs with an AMD lowering
        choice on a CUDA target: knob mutation bypassed sample()'s reset, and
        those inert knobs inflated the pair-cell coverage."""
        from src.workflow.slices.base import HIP_TARGETS
        rng = random.Random(44)
        for name, slice_ in SLICES.items():
            if 'triton' not in slice_.backends:
                continue
            params = dict(slice_.sample(rng, 'triton'), cfg_target='gfx942', cfg_amd='kpack2')
            hip = make_program(name, params, 'triton')
            if hip.params['cfg_target'] in HIP_TARGETS:
                self.assertEqual(hip.params['cfg_amd'], 'kpack2')
            mutated = make_program(name, dict(hip.params, cfg_target='sm90'), 'triton')
            self.assertEqual(mutated.params['cfg_amd'], 'none', name)
            with self.assertRaises(ValueError):
                validate_program(SliceProgram(name, dict(mutated.params, cfg_amd='kpack2'), 'triton'))
        scheduler = SliceScheduler('triton', candidates=4)
        for _ in range(400):
            program = scheduler.next(rng)
            if program.params['cfg_target'] not in HIP_TARGETS:
                self.assertEqual(program.params['cfg_amd'], 'none')
            scheduler.observe(program, seconds=0.1)

    def test_what_a_foreign_target_lacks_is_noise(self):
        amd = ("TILESMITH_TARGET=gfx942:compile-only\nerror: Unsupported conversion from 'f8E4M3FN' to 'f64'\n"
               'RuntimeError: PassManager::run failed')
        self.assertEqual(classify_root_cause(amd), 'unsupported_feature')
        cap = ('TILESMITH_TARGET=sm100:compile-only\nptxas-blackwell fatal : (C7602) Insufficient registers (32)\n'
               'triton.runtime.errors.PTXASError: PTXAS error: Internal Triton PTX codegen error')
        self.assertEqual(classify_root_cause(cap), 'register_limit')
        # the same diagnostics without a foreign target stay failures
        self.assertNotEqual(classify_root_cause(amd.split('\n', 1)[1]), 'unsupported_feature')
        self.assertNotEqual(classify_root_cause(cap.split('\n', 1)[1]), 'register_limit')

    def test_markers_survive_a_long_traceback(self):
        from src.workflow.oracle.oracle import _bounded
        frames = '\n'.join(f'  File "x.py", line {i}, in f' for i in range(400))
        rejected = ('TILESMITH_TARGET=sm80\nTILESMITH_REJECTED=not supported in this architecture\n' + frames
                    + '\ntriton.compiler.errors.CompilationError: at 3:4:')
        self.assertEqual(classify_root_cause(rejected[-2000:]), 'triton_compile_error')
        self.assertEqual(classify_root_cause(_bounded(rejected)), 'unsupported_feature')
        amd = ("TILESMITH_TARGET=gfx1100:compile-only\nk.py:9:1: error: Unsupported conversion from 'f64' to "
               "'f8E5M2'\n" + frames + '\nRuntimeError: PassManager::run failed')
        self.assertEqual(classify_root_cause(_bounded(amd)), 'unsupported_feature')
        crash = 'TILESMITH_TARGET=sm100:compile-only\n' + frames + '\nRuntimeError: PassManager::run failed'
        self.assertEqual(failure_key(_bounded(crash)), failure_key(crash[-2000:]))
        self.assertEqual(_bounded('short'), 'short')

    def test_non_native_targets_are_path_features(self):
        from src.workflow.slices import path_features
        rng = random.Random(44)
        program = make_program('scan', dict(SLICES['scan'].sample(rng, 'triton'), cfg_target='gfx1100'), 'triton')
        self.assertIn('path:target gfx1100', path_features(program))
        native = make_program('scan', dict(program.params, cfg_target='native'), 'triton')
        self.assertFalse(any(f.startswith('path:target') for f in path_features(native)))


class ConfigurationTests(unittest.TestCase):
    """Compiler configurations are knobs that reach the launch options."""

    def test_configurations_reach_the_launch(self):
        from src.workflow.slices.base import CONFIG_KNOBS, TILELANG_PASSES
        rng = random.Random(30)
        defaults = Counter()
        for backend in BACKENDS:
            for name, slice_ in SLICES.items():
                space = slice_.space(backend)
                self.assertTrue(any(k in space for k in CONFIG_KNOBS), (name, backend))
                for _ in range(20):
                    program = make_program(name, slice_.sample(rng, backend), backend)
                    p = program.params
                    defaults[all(p[k] == space[k][0] for k in CONFIG_KNOBS if k in space)] += 1
                    variants = slice_.variants(p, backend)
                    for label, options in variants:
                        self.assertTrue(label[0] in 'wt' and label[1:].isdigit(), label)
                        if backend == 'tilelang':
                            self.assertEqual(options.get('pass_configs', {}), TILELANG_PASSES[p['cfg_pass']])
                        else:
                            self.assertEqual(options.get('maxnreg', 0), p['cfg_maxnreg'])
                            self.assertEqual(options.get('enable_fp_fusion', True), bool(p['cfg_fusion']))
                            self.assertEqual(options.get('num_stages', 0), p.get('cfg_stages', 0))
                            self.assertEqual(options.get('enable_reflect_ftz', True), bool(p.get('cfg_ftz', 1)))
        self.assertGreater(defaults[True], defaults[False] / 3)
        self.assertGreater(defaults[False], defaults[True] / 3)

    def test_tilelang_builds_are_keyed_by_configuration(self):
        program = make_program('cast', dict(SLICES['cast'].sample(random.Random(31), 'tilelang'),
                                            cfg_pass='index64'), 'tilelang')
        code = emit_slice(program, Cfg)
        self.assertIn('@tilelang.jit(pass_configs=configs)', code)
        self.assertIn("'pass_configs': {'tl.config_index_bitwidth': 64}", code)
        ns = harness_namespace(program)
        built = []
        ns['build'] = lambda threads, configs: built.append((threads, configs)) or (lambda *a: None)
        options = {'threads': 64, 'pass_configs': {'tl.config_index_bitwidth': 64}}
        ns['launch'](None, None, None, options)
        ns['launch'](None, None, None, options)
        ns['launch'](None, None, None, {'threads': 64})
        self.assertEqual(built, [(64, {'tl.config_index_bitwidth': 64}), (64, None)])

    def test_math_flush_to_zero_follows_the_configuration(self):
        base = dict(SLICES['math'].sample(random.Random(32), 'triton'), fn='nextafter', in_dt='f32', values='small')
        on = make_program('math', dict(base, cfg_ftz=1), 'triton')
        off = make_program('math', dict(base, cfg_ftz=0), 'triton')
        self.assertIn('_slice_ftz(w,', emit_slice(on, Cfg))
        self.assertNotIn('_slice_ftz(w,', emit_slice(off, Cfg))
        self.assertIn("'enable_reflect_ftz': False", emit_slice(off, Cfg))

    def test_configurations_reduce_first(self):
        from src.workflow.slices.minimize import minimize
        program = make_program('loop', dict(SLICES['loop'].sample(random.Random(33), 'triton'), cfg_maxnreg=32,
                                            cfg_stages=2, cfg_fusion=0), 'triton')
        reduced, kept, _ = minimize(program, lambda c: c.params['cfg_stages'] == 2, budget=200)
        self.assertEqual(kept.get('cfg_stages'), 2)
        self.assertEqual((reduced.params['cfg_maxnreg'], reduced.params['cfg_fusion']), (0, 1))


class LoweringFeedbackTests(unittest.TestCase):
    def test_harness_reports_lowering_features(self):
        from src.workflow.slices.runtime import _slice_lowering
        import contextlib
        import io
        ptx = '\n'.join(['.reg .b32 %r<4>;', '$L__BB0_1:', '\tld.global.v4.b32 {%r1, %r2}, [%rd1];',
                         '\t@%p1 st.global.b32 [%rd2], %r1;', '\tmma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 '
                         '{%f1}, {%r1}, {%r2}, {%f2};', '\tret;'])
        ttgir = '\n'.join(['#blocked = #ttg.blocked<{sizePerThread = [1]}>', '#mma = #ttg.nvidia_mma<{version = 2}>',
                           'module {', '  %0 = tt.load %arg0 : tensor<64x!tt.ptr<f16>, #blocked>',
                           '  %1 = ttg.convert_layout %0 : tensor<64xf16, #blocked> -> tensor<64xf16, #mma>',
                           '  %2 = "tt.reduce"(%1) <{axis = 0 : i32}> ({', '  tt.return', '}'])
        compiled = type('K', (), {'asm': {'ptx': ptx, 'ttgir': ttgir}})()
        jit = type('J', (), {'device_caches': {0: ({'key': compiled}, None, None, None, None)}})()
        cuda = ('extern "C" __global__ void __launch_bounds__(128) main_kernel(half_t* A) {\n'
                '  #pragma unroll\n  *(uint4*)(A + 8) = *(uint4*)(A);\n'
                '  tl::cp_async_gs<16>(A, A);\n  tl::AllReduce<tl::SumOp, 128, 1, 0>::run(x, red);\n'
                '  asm volatile("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(r) : "f"(a), "f"(b));\n'
                '  __syncthreads();\n  if (x < 3) { y = max(x, y); }\n}')
        tl_kernel = type('T', (), {'get_kernel_source': lambda self: cuda})()
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            _slice_lowering({'kernel': jit, '_KERNELS': {(128, '[]'): tl_kernel}, 'other': 3})
        line = out.getvalue().strip()
        self.assertTrue(line.startswith('TILESMITH_LOWERING='))
        features = set(line.split('=', 1)[1].split(','))
        for f in ('ptx:ld.global.v4.b32', 'ptx:st.global.b32', 'ptx:ret',
                  'ptx:mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32', 'layout:blocked', 'layout:nvidia_mma',
                  'convert:blocked>nvidia_mma', 'ttgir:tt.load', 'ttgir:tt.reduce', 'ttgir:tt.return',
                  'call:tl::cp_async_gs<16>', 'call:tl::AllReduce<tl::SumOp/N/1/0>::run', 'call:__syncthreads',
                  'call:max', 'cast:uint4', 'asm:cvt.rn.f16x2.f32', 'pragma:unroll'):
            self.assertIn(f, features)
        self.assertFalse([f for f in features if f.startswith('ptx:.') or f.startswith('ptx:$')
                          or 'main_kernel' in f or f in ('call:if', 'call:volatile', 'call:__launch_bounds__')
                          or ' ' in f], features)
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            _slice_lowering({'kernel': object()})
        self.assertEqual(out.getvalue(), '')

    def test_oracle_takes_the_lowering_line(self):
        from src.workflow.oracle.oracle import _take_lowering
        features, rest = _take_lowering('x\nTILESMITH_LOWERING=ptx:ret,call:max\nALL PASSED\n')
        self.assertEqual(features, frozenset({'ptx:ret', 'call:max'}))
        self.assertEqual(rest, 'x\nALL PASSED\n')
        self.assertEqual(_take_lowering('ALL PASSED\n'), (None, 'ALL PASSED\n'))

    def test_new_lowering_makes_rarity_weighted_seeds(self):
        scheduler = SliceScheduler('triton', ['cast', 'reduce'], warmup=0, seeds_max=3)
        rng = random.Random(34)
        programs = [make_program('cast', SLICES['cast'].sample(rng, 'triton'), 'triton') for _ in range(6)]
        state = scheduler.states['cast']
        self.assertTrue(scheduler.observe(programs[0], None, None, 1.0, lowering={'a', 'b'}))
        scheduler.observe(programs[1], None, None, 1.0, lowering={'a', 'b'})
        scheduler.observe(programs[2], 'wrong_result', 'w', 1.0, lowering={'a', 'c'})
        self.assertEqual(state.seeds, [[programs[0].params, ['a', 'b']]])
        self.assertEqual((state.lowered, state.lowering_single()), (3, 1))
        for i, program in enumerate(programs[3:]):
            scheduler.observe(program, None, None, 1.0, lowering={'a', f'n{i}'})
        self.assertEqual(len(state.seeds), 3)
        self.assertNotIn([programs[0].params, ['a', 'b']], state.seeds)  # its features are no longer rare
        before = scheduler.estimate('cast')
        scheduler.observe(programs[3], None, None, 1.0, lowering={'a', 'n0', 'n1', 'n2'})
        self.assertLess(scheduler.estimate('cast'), before)
        stats = scheduler.lowering_stats()
        self.assertEqual((stats['features'], stats['seeds'], stats['tests']), (6, 3, 7))
        restored = SliceScheduler('triton', ['cast', 'reduce'])
        restored.restore(json.loads(json.dumps(scheduler.snapshot())))
        self.assertEqual(restored.snapshot(), scheduler.snapshot())
        draws = Counter()
        mutate = SliceScheduler('triton', ['cast'], mutate_prob=1.0, warmup=0)
        mutate.restore(json.loads(json.dumps(scheduler.snapshot())))
        mutate.states['cast'].interesting = []
        with patch.object(random.Random, 'choices', autospec=True,
                          side_effect=lambda self, population, weights=None, k=1:
                          draws.update([len(population)]) or [population[0]]):
            mutate.next(random.Random(35))
        self.assertEqual(draws[3], 1)  # one rarity-weighted draw among the three seeds


class PathFeatureTests(unittest.TestCase):
    def test_conversions_are_position_free(self):
        from src.workflow.slices import path_features
        rng = random.Random(14)
        base = SLICES['cast'].sample(rng, 'triton')
        chain = dict(base, in_dt='f32', values='small', operand='const', out_dt='f32',
                     pre1_op='none', pre2_op='none', pre3_op='none', pre1_dt='f32', pre2_dt='f32', pre3_dt='f32')
        early = make_program('cast', dict(chain, pre1_dt='f64', pre2_dt='f16', pre3_dt='f16', out_dt='f16'), 'triton')
        late = make_program('cast', dict(chain, pre3_dt='f64', out_dt='f16'), 'triton')
        for program in (early, late):
            self.assertIn('path:conv f64>f16', path_features(program), program.params)
            self.assertIn('path:conv f64>f16', program.features())
        reduce_ = make_program('reduce', dict(SLICES['reduce'].sample(rng, 'triton'), in_dt='f32', core_dt='bf16',
                                              kind='cmax', pre1_op='none', pre2_op='none', pre1_dt='f32',
                                              pre2_dt='f32'), 'triton')
        features = path_features(reduce_)
        self.assertIn('path:conv f32>bf16', features)
        self.assertIn('path:reduce cmax@bf16', features)
        for program in programs(4, seed=15):
            self.assertTrue(path_features(program), program.slice)


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
