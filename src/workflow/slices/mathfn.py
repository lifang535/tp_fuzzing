"""Library math over full-precision values, with exact references.

The cast slice keeps every value small and exact; the round slice covers
conversions only. This slice calls the math libraries of each DSL directly:
Triton's tl.* math and libdevice (rounding-mode arithmetic, float/int
conversions, bit manipulation, exact float functions, approximations) and
TileLang's T.ieee_* arithmetic, rounding functions, bit operations,
reinterpretation and packed x2 arithmetic. Inputs are full-precision values
(random significands, integer bit patterns, grid midpoints, special values);
the reference evaluates each element exactly with rationals and rounds once
in the requested mode, so a correctly rounded operation is checked bit for
bit, while approximations get a tolerance of a few units in the last place.

Each catalog entry lists only the operand dtypes the front end accepts
(probed on Triton 3.8 / TileLang 0.1.14): a dtype the front end rejects is
noise, one it accepts and then miscompiles is the target.
"""
import math
import zlib

from .base import GUARD, Slice, TILELANG_SHAPES, TRITON_SHAPES, parse_shape, triton_offsets, valid_extents
from .runtime import SOURCES

FLOAT = ('f16', 'bf16', 'f32', 'f64')
F32_64 = ('f32', 'f64')
SIGNED = ('i8', 'i16', 'i32', 'i64')
UNSIGNED = ('u8', 'u16', 'u32', 'u64')
INTEGER = SIGNED + UNSIGNED
TRITON_TYPE = {'f16': 'tl.float16', 'bf16': 'tl.bfloat16', 'f32': 'tl.float32', 'f64': 'tl.float64',
               'f8e4': 'tl.float8e4nv', 'f8e5': 'tl.float8e5', 'i8': 'tl.int8', 'i16': 'tl.int16',
               'i32': 'tl.int32', 'i64': 'tl.int64', 'u8': 'tl.uint8', 'u16': 'tl.uint16',
               'u32': 'tl.uint32', 'u64': 'tl.uint64'}
TILELANG_TYPE = {'f16': 'float16', 'bf16': 'bfloat16', 'f32': 'float32', 'f64': 'float64',
                 'f8e4': 'float8_e4m3', 'f8e5': 'float8_e5m2', 'i8': 'int8', 'i16': 'int16', 'i32': 'int32',
                 'i64': 'int64', 'u8': 'uint8', 'u16': 'uint16', 'u32': 'uint32', 'u64': 'uint64'}
TORCH_TYPE = {'f16': 'float16', 'bf16': 'bfloat16', 'f32': 'float32', 'f64': 'float64',
              'f8e4': 'float8_e4m3fn', 'f8e5': 'float8_e5m2', 'i8': 'int8', 'i16': 'int16', 'i32': 'int32',
              'i64': 'int64', 'u8': 'uint8', 'u16': 'uint16', 'u32': 'uint32', 'u64': 'uint64'}
# Same-width twins for reinterpretation.
TWIN = {'f16': 'i16', 'bf16': 'i16', 'f32': 'i32', 'f64': 'i64', 'f8e4': 'i8', 'f8e5': 'u8',
        'i16': 'f16', 'u16': 'bf16', 'i32': 'f32', 'u32': 'f32', 'i64': 'f64', 'u64': 'f64', 'i8': 'f8e4',
        'u8': 'f8e5'}
WIDE = {'f16': 'f64', 'bf16': 'f64', 'f32': 'f64', 'i8': 'i64', 'i16': 'i64', 'i32': 'i64', 'u8': 'i64',
        'u16': 'i64', 'u32': 'i64'}
FLOAT_REGIMES = ('small', 'wide', 'specials')
INT_REGIMES = ('small', 'bits', 'edges')


def is_float(dt):
    return dt[0] in 'fb'


class Fn:
    """One catalog entry. template names its operands {x}, {y}, {z};
    types are per operand ('$' is the entry's input dtype), out is the
    result dtype ('$' likewise), domains constrain each operand, ulps is 0
    for an exact result and the tolerance of an approximation otherwise."""

    def __init__(self, template, dtypes, sem, mode='rn', out='$', types=('$', '$', '$'),
                 domains=('any', 'any', 'any'), regimes=None, ulps=0, signed=False, const=None, packed=False,
                 helper=''):
        self.template, self.dtypes, self.sem, self.mode = template, tuple(dtypes), sem, mode
        self.helper = helper
        self.out, self.types, self.domains, self.ulps = out, types, domains, ulps
        self.regimes, self.signed, self.const, self.packed = regimes, signed, const, packed
        self.arity = sum(f'{{{v}}}' in template for v in 'xyz')

    def operand_types(self, dt):
        return [dt if t == '$' else t for t in self.types[:self.arity]]

    def out_type(self, dt):
        if self.out == '$':
            return dt
        if self.out == 'twin':
            return TWIN[dt]
        return self.out

    def allowed_regimes(self, dt):
        if self.regimes is not None:
            return self.regimes
        return FLOAT_REGIMES if is_float(dt) else INT_REGIMES


def rounding(prefix, sem, dtypes, arity, extra=None):
    """libdevice entries prefix_rn/_rz/_rd/_ru of one correctly rounded op.
    Triton links libdevice with flush-to-zero, so the operands avoid the
    range edges where results become subnormal."""
    operands = ', '.join('{' + v + '}' for v in 'xyz'[:arity])
    entries = {}
    for mode in ('rn', 'rz', 'rd', 'ru'):
        entries[f'{prefix}_{mode}'] = Fn(f'libdevice.{prefix}_{mode}({operands})', dtypes, sem, mode,
                                         regimes=('small', 'wide'), **(extra or {}))
    return entries


def conversions(name, src, dst, modes=('rn', 'rz', 'rd', 'ru'), regimes=None, domain='any'):
    return {f'{name}_{m}': Fn(f'libdevice.{name}_{m}({{x}})', (src,), 'convert', m, out=dst,
                              regimes=regimes or (f'ties:{dst}', 'wide', 'small'), domains=(domain,))
            for m in modes}


def approximation(name, dtypes, sem, domain, ulps, template=None):
    return Fn(template or f'libdevice.{name}({{x}})', dtypes, sem, regimes=(f'range:{domain[0]}:{domain[1]}',),
              ulps=ulps)


def triton_catalog():
    c = {}
    # native tl operations
    for name, symbol, sem in (('add', '+', 'add'), ('sub', '-', 'sub'), ('mul', '*', 'mul')):
        c[name] = Fn(f'{{x}} {symbol} {{y}}', FLOAT + INTEGER, sem, const=3)
    c['div'] = Fn('{x} / {y}', FLOAT, 'div_approx', domains=('any', 'nonzero'), ulps=2, const=0.75)
    c['div_f64'] = Fn('{x} / {y}', ('f64',), 'div', domains=('any', 'nonzero'), const=0.75)
    c['div_rn'] = Fn('tl.div_rn({x}, {y})', ('f32',), 'div', domains=('any', 'nonzero'))
    c['fdiv'] = Fn('tl.fdiv({x}, {y})', FLOAT, 'div_approx', domains=('any', 'nonzero'), ulps=2)
    c['fdiv_ieee'] = Fn('tl.fdiv({x}, {y}, ieee_rounding=True)', FLOAT, 'div', domains=('any', 'nonzero'))
    c['sqrt'] = Fn('tl.sqrt({x})', F32_64, 'sqrt_approx', domains=('nonneg',), ulps=2)
    c['sqrt_rn'] = Fn('tl.sqrt_rn({x})', ('f32',), 'sqrt', domains=('nonneg',))
    c['rsqrt'] = Fn('tl.rsqrt({x})', F32_64, 'rsqrt_approx', domains=('pos',), ulps=4)
    c['fma'] = Fn('tl.fma({x}, {y}, {z})', F32_64, 'fma')
    c['fma_half'] = Fn('tl.fma({x}, {y}, {z})', ('f16', 'bf16'), 'fma', ulps=1)
    for name in ('exp', 'exp2', 'log', 'log2', 'sin', 'cos', 'erf', 'sigmoid'):
        domain = (1 / 64, 64) if name.startswith('log') else (-8, 8)
        c[name] = approximation(name, F32_64, name, domain, 16, template=f'tl.{name}({{x}})')
    c['floor'] = Fn('tl.floor({x})', F32_64, 'floor', regimes=('ties:i32', 'wide', 'specials'), signed=True)
    c['ceil'] = Fn('tl.ceil({x})', F32_64, 'ceil', regimes=('ties:i32', 'wide', 'specials'), signed=True)
    c['abs'] = Fn('tl.abs({x})', FLOAT + INTEGER, 'abs', signed=True)
    c['neg'] = Fn('-{x}', FLOAT + INTEGER, 'neg', signed=True)
    c['maximum'] = Fn('tl.maximum({x}, {y})', FLOAT + INTEGER, 'max', regimes=('small', 'wide', 'bits', 'edges'),
                      const=3)
    c['minimum'] = Fn('tl.minimum({x}, {y})', FLOAT + INTEGER, 'min', regimes=('small', 'wide', 'bits', 'edges'),
                      const=3)
    c['maximum_nan'] = Fn('tl.maximum({x}, {y}, propagate_nan=tl.PropagateNan.ALL)', FLOAT, 'pmax',
                          regimes=('specials',))
    c['minimum_nan'] = Fn('tl.minimum({x}, {y}, propagate_nan=tl.PropagateNan.ALL)', FLOAT, 'pmin',
                          regimes=('specials',))
    c['clamp'] = Fn('tl.clamp({x}, {y}, {z})', FLOAT, 'clamp', regimes=('small', 'wide'))
    # LLVM's NVPTX lowering of frem is x - trunc(x / y) * y: exact only for
    # small quotients and it drops the sign of a zero result (known)
    c['mod'] = Fn('{x} % {y}', FLOAT, 'fmod', domains=('any', 'nonzero'), const=0.75, regimes=('small',))
    c['imod'] = Fn('{x} % {y}', INTEGER, 'imod', domains=('any', 'nonzero'), const=3)
    c['idiv'] = Fn('{x} // {y}', INTEGER, 'idiv', domains=('any', 'nonzero'), const=3)
    c['cdiv'] = Fn('tl.cdiv({x}, {y})', INTEGER, 'cdiv', domains=('nonneg', 'small_pos'), regimes=('small',))
    c['umulhi'] = Fn('tl.umulhi({x}, {y})', ('i32', 'u32', 'i64', 'u64'), 'umulhi')
    c['not'] = Fn('~{x}', INTEGER, 'not')
    for name, symbol in (('and', '&'), ('or', '|'), ('xor', '^')):
        c[name] = Fn(f'{{x}} {symbol} {{y}}', INTEGER, name, const=3)
    for name, symbol in (('shl', '<<'), ('shr', '>>')):
        for dt in INTEGER:
            c[f'{name}_{dt}'] = Fn(f'{{x}} {symbol} {{y}}', (dt,), name, domains=('any', f'shift:{dt}'), const=3)
    c['bitcast'] = Fn('{x}.to({twin}, bitcast=True)', FLOAT + ('f8e4', 'f8e5', 'i16', 'u16', 'i32', 'u32', 'i64',
                                                               'u64', 'i8', 'u8'),
                      'bitcast', out='twin', regimes=('wide', 'specials', 'bits', 'edges'), signed=True)
    # libdevice: correctly rounded arithmetic in four rounding modes
    for prefix, sem, arity, extra in (('add', 'add', 2, None), ('sub', 'sub', 2, None), ('mul', 'mul', 2, None),
                                      ('div', 'div', 2, {'domains': ('any', 'nonzero')}),
                                      ('fma', 'fma', 3, None), ('rcp', 'rcp', 1, {'domains': ('nonzero',)}),
                                      ('sqrt', 'sqrt', 1, {'domains': ('nonneg',)})):
        c.update(rounding(prefix, sem, F32_64, arity, extra))
    c['rsqrt_rn'] = Fn('libdevice.rsqrt_rn({x})', ('f32',), 'rsqrt', domains=('pos',))
    # libdevice conversions
    c.update(conversions('double2float', 'f64', 'f32'))
    c.update(conversions('double2int', 'f64', 'i32'))
    c.update(conversions('double2uint', 'f64', 'u32', domain='nonneg'))
    c.update(conversions('double2ll', 'f64', 'i64'))
    c.update(conversions('float2int', 'f32', 'i32'))
    c.update(conversions('float2uint', 'f32', 'u32', domain='nonneg'))
    c.update(conversions('float2ll', 'f32', 'i64'))
    for name, src, dst in (('int2float', 'i32', 'f32'), ('uint2float', 'u32', 'f32'), ('ll2float', 'i64', 'f32'),
                           ('ull2float', 'u64', 'f32'), ('ll2double', 'i64', 'f64'), ('ull2double', 'u64', 'f64')):
        c.update(conversions(name, src, dst, regimes=(f'ties:{dst}', 'bits', 'edges')))
    c['int2double_rn'] = Fn('libdevice.int2double_rn({x})', ('i32',), 'convert', out='f64', regimes=INT_REGIMES)
    c['uint2double_rn'] = Fn('libdevice.uint2double_rn({x})', ('u32',), 'convert', out='f64', regimes=INT_REGIMES)
    # libdevice exact float functions
    rounding_regimes = ('ties:i32', 'wide', 'specials')
    for name, sem in (('rint', 'rint'), ('nearbyint', 'rint'), ('round', 'round'), ('trunc', 'trunc'),
                      ('floor', 'floor'), ('ceil', 'ceil')):
        c[f'ld_{name}'] = Fn(f'libdevice.{name}({{x}})', F32_64, sem, regimes=rounding_regimes, signed=True)
    c['llrint'] = Fn('libdevice.llrint({x})', F32_64, 'convert', out='i64', regimes=('ties:i64', 'small'))
    c['llround'] = Fn('libdevice.llround({x})', F32_64, 'round', out='i64', regimes=('ties:i64', 'small'))
    c['ld_abs'] = Fn('libdevice.abs({x})', F32_64 + ('i32', 'i64'), 'abs', signed=True)
    c['copysign'] = Fn('libdevice.copysign({x}, {y})', F32_64, 'copysign', signed=True)
    c['fmod'] = Fn('libdevice.fmod({x}, {y})', F32_64, 'fmod', domains=('any', 'nonzero'), signed=True,
                   regimes=('small', 'wide'))
    c['remainder'] = Fn('libdevice.remainder({x}, {y})', F32_64, 'remainder', domains=('any', 'nonzero'),
                        signed=True, regimes=('small', 'wide'))
    c['fdim'] = Fn('libdevice.fdim({x}, {y})', F32_64, 'fdim', regimes=('small', 'wide'))
    c['nextafter'] = Fn('libdevice.nextafter({x}, {y})', F32_64, 'nextafter', regimes=('small', 'wide'))
    c['ldexp'] = Fn('libdevice.ldexp({x}, {y})', F32_64, 'ldexp', types=('$', 'i32'), domains=('any', 'exp'),
                    regimes=('small', 'wide'))
    c['scalbn'] = Fn('libdevice.scalbn({x}, {y})', F32_64, 'ldexp', types=('$', 'i32'), domains=('any', 'exp'),
                     regimes=('small', 'wide'))
    c['ilogb'] = Fn('libdevice.ilogb({x})', F32_64, 'ilogb', out='i32', regimes=('small', 'wide'))
    c['logb'] = Fn('libdevice.logb({x})', F32_64, 'logb', regimes=('small', 'wide'))
    c['signbit'] = Fn('(libdevice.signbit({x}) != 0).to(tl.int32)', F32_64, 'signbit', out='i32',
                      regimes=('specials', 'wide'))
    c['isnan'] = Fn('libdevice.isnan({x}).to(tl.int32)', F32_64, 'isnan', out='i32', regimes=('specials',))
    c['isinf'] = Fn('libdevice.isinf({x}).to(tl.int32)', F32_64, 'isinf', out='i32', regimes=('specials',))
    c['finitef'] = Fn('(libdevice.finitef({x}) != 0).to(tl.int32)', ('f32',), 'isfinite', out='i32',
                      regimes=('specials',))
    c['isfinited'] = Fn('(libdevice.isfinited({x}) != 0).to(tl.int32)', ('f64',), 'isfinite', out='i32',
                        regimes=('specials',))
    c['saturatef'] = Fn('libdevice.saturatef({x})', ('f32',), 'saturate', regimes=('small', 'wide'))
    # libdevice integer and bit functions
    c['popc'] = Fn('libdevice.popc({x})', ('i32', 'i64'), 'popc', out='i32')
    c['clz'] = Fn('libdevice.clz({x})', ('i32', 'i64'), 'clz', out='i32')
    c['ffs'] = Fn('libdevice.ffs({x})', ('i32', 'i64'), 'ffs', out='i32')
    c['brev'] = Fn('libdevice.brev({x})', ('i32', 'i64'), 'brev')
    c['byte_perm'] = Fn('libdevice.byte_perm({x}, {y}, {z})', ('i32',), 'byte_perm',
                        domains=('any', 'any', 'selector'), regimes=('bits',))
    c['mulhi'] = Fn('libdevice.mulhi({x}, {y})', ('i32', 'u32', 'i64', 'u64'), 'mulhi', regimes=('bits', 'edges'))
    c['mul24'] = Fn('libdevice.mul24({x}, {y})', ('i32', 'u32'), 'mul24', regimes=('bits', 'small'))
    c['hadd'] = Fn('libdevice.hadd({x}, {y})', ('i32', 'u32'), 'hadd', regimes=('bits', 'edges'))
    c['rhadd'] = Fn('libdevice.rhadd({x}, {y})', ('i32', 'u32'), 'rhadd', regimes=('bits', 'edges'))
    c['sad'] = Fn('libdevice.sad({x}, {y}, {z})', ('i32', 'u32'), 'sad', types=('$', '$', 'u32'),
                  regimes=('small',))
    c['float_as_int'] = Fn('libdevice.float_as_int({x})', ('f32',), 'bitcast', out='i32',
                           regimes=('wide', 'specials'))
    c['int_as_float'] = Fn('libdevice.int_as_float({x})', ('i32',), 'bitcast', out='f32', regimes=('bits',))
    c['double_as_longlong'] = Fn('libdevice.double_as_longlong({x})', ('f64',), 'bitcast', out='i64',
                                 regimes=('wide', 'specials'))
    c['longlong_as_double'] = Fn('libdevice.longlong_as_double({x})', ('i64',), 'bitcast', out='f64',
                                 regimes=('bits',))
    c['double2hiint'] = Fn('libdevice.double2hiint({x})', ('f64',), 'hiint', out='i32', regimes=('wide', 'specials'))
    c['double2loint'] = Fn('libdevice.double2loint({x})', ('f64',), 'loint', out='i32', regimes=('wide',))
    c['hiloint2double'] = Fn('libdevice.hiloint2double({x}, {y})', ('i32',), 'hilo', out='f64', regimes=('bits',))
    # libdevice approximations
    for name, domain in (('exp', (-8, 8)), ('exp2', (-8, 8)), ('exp10', (-4, 4)), ('expm1', (-4, 4)),
                         ('log', (0.01, 100)), ('log2', (0.01, 100)), ('log10', (0.01, 100)), ('log1p', (-0.5, 50)),
                         ('sin', (-6, 6)), ('cos', (-6, 6)), ('tan', (-1.4, 1.4)), ('sinh', (-4, 4)),
                         ('cosh', (-4, 4)), ('tanh', (-4, 4)), ('asin', (-1, 1)), ('acos', (-1, 1)),
                         ('atan', (-8, 8)), ('asinh', (-8, 8)), ('acosh', (1, 8)), ('atanh', (-0.9, 0.9)),
                         ('erf', (-3, 3)), ('erfc', (-3, 3)), ('cbrt', (-8, 8)), ('sinpi', (-2, 2)),
                         ('cospi', (-2, 2)), ('tgamma', (0.5, 6)), ('lgamma', (0.5, 6))):
        c[f'ld_{name}'] = approximation(name, F32_64, name, domain, 16)
    c['atan2'] = Fn('libdevice.atan2({x}, {y})', F32_64, 'atan2', regimes=('range:-4:4',), ulps=16)
    c['hypot'] = Fn('libdevice.hypot({x}, {y})', F32_64, 'hypot', regimes=('range:-8:8',), ulps=16)
    c['pow'] = Fn('libdevice.pow({x}, {y})', F32_64, 'pow', regimes=('range:0.5:3',), ulps=32)
    for name, sem, domain in (('fast_expf', 'exp', (-4, 4)), ('fast_logf', 'log', (0.1, 10)),
                              ('fast_sinf', 'sin', (-3, 3)), ('fast_cosf', 'cos', (-3, 3)),
                              ('fast_log2f', 'log2', (0.1, 10)), ('fast_exp10f', 'exp10', (-2, 2))):
        c[name] = approximation(name, ('f32',), sem, domain, 4096)
    c['fast_dividef'] = Fn('libdevice.fast_dividef({x}, {y})', ('f32',), 'div_approx', regimes=('range:0.5:8',),
                           ulps=8)
    # random numbers (Philox 4x32, exact reference)
    offsets = ('i32', 'u32', 'i64', 'u64')
    for name, seed, rounds in (('randint', 12345, 10), ('randint_seed64', 0x9876543210, 10),
                               ('randint_r7', 3000000000, 7)):
        c[name] = Fn(f'tl.randint({seed}, {{x}}, n_rounds={rounds})', offsets, 'philox', f'{seed}:{rounds}:0',
                     out='u32', regimes=('small', 'bits', 'edges'))
    for k in range(4):
        c[f'randint4x_{k}'] = Fn(f'tl.randint4x(777, {{x}})[{k}]', offsets, 'philox', f'777:10:{k}', out='u32',
                                 regimes=('small', 'bits'))
    c['rand'] = Fn('tl.rand(4242, {x})', offsets, 'rand', '4242:10:0', out='f32', regimes=('small', 'bits'))
    c['rand4x_3'] = Fn('tl.rand4x(99, {x})[3]', offsets, 'rand', '99:10:3', out='f32', regimes=('small', 'bits'))
    # inline PTX: rounding-mode arithmetic, packed lanes, conversions, two outputs
    asm = 'tl.inline_asm_elementwise('
    for name, text, constraints, dtypes, sem, mode, extra in (
            ('asm_add_rz', 'add.rz.f32 $0, $1, $2;', '=f,f,f', ('f32',), 'add', 'rz', {}),
            ('asm_div_rn', 'div.rn.f32 $0, $1, $2;', '=f,f,f', ('f32',), 'div', 'rn', {'domains': ('any', 'nonzero')}),
            ('asm_sqrt_rp', 'sqrt.rp.f32 $0, $1;', '=f,f', ('f32',), 'sqrt', 'ru', {'domains': ('nonneg',)}),
            ('asm_fma_rm', 'fma.rm.f64 $0, $1, $2, $3;', '=d,d,d,d', ('f64',), 'fma', 'rd', {}),
            ('asm_mul_hi', 'mul.hi.s32 $0, $1, $2;', '=r,r,r', ('i32',), 'mulhi', 'rn', {'regimes': ('bits',)}),
            ('asm_mad_lo', 'mad.lo.s64 $0, $1, $2, $3;', '=l,l,l,l', ('i64',), 'mad', 'rn', {'regimes': ('bits',)}),
            ('asm_brev', 'brev.b32 $0, $1;', '=r,r', ('i32',), 'brev', 'rn', {'regimes': ('bits',)})):
        c[name] = Fn(f'{asm}"{text}", "{constraints}", [{", ".join("{" + v + "}" for v in "xyz"[:constraints.count(",")])}], '
                     f'dtype={{tt}}, is_pure=True, pack=1)', dtypes, sem, mode, **extra)
    for name, text, constraints, dtypes, sem, pack, extra in (
            ('asm_f16x2_add', 'add.rn.f16x2 $0, $1, $2;', '=r,r,r', ('f16',), 'add', 2, {}),
            ('asm_f16x2_fma', 'fma.rn.f16x2 $0, $1, $2, $3;', '=r,r,r,r', ('f16',), 'fma', 2, {}),
            ('asm_bf16x2_fma', 'fma.rn.bf16x2 $0, $1, $2, $3;', '=r,r,r,r', ('bf16',), 'fma', 2, {}),
            ('asm_b8x4_and', 'and.b32 $0, $1, $2;', '=r,r,r', ('i8', 'u8'), 'and', 4, {'regimes': ('bits',)}),
            ('asm_b16x2_xor', 'xor.b32 $0, $1, $2;', '=r,r,r', ('i16', 'u16'), 'xor', 2, {'regimes': ('bits',)})):
        operands = ', '.join('{' + v + '}' for v in 'xyz'[:constraints.count(',')])
        c[name] = Fn(f'{asm}"{text}", "{constraints}", [{operands}], dtype={{tt}}, is_pure=True, pack={pack})',
                     dtypes, sem, **extra)
    for name, text, constraints, src, dst, mode in (
            ('asm_cvt_rni', 'cvt.rni.s32.f32 $0, $1;', '=r,f', 'f32', 'i32', 'rn'),
            ('asm_cvt_rzi', 'cvt.rzi.s64.f64 $0, $1;', '=l,d', 'f64', 'i64', 'rz'),
            ('asm_cvt_f16', 'cvt.rn.f16.f32 $0, $1;', '=h,f', 'f32', 'f16', 'rn')):
        c[name] = Fn(f'{asm}"{text}", "{constraints}", [{{x}}], dtype={TRITON_TYPE[dst]}, is_pure=True, pack=1)',
                     (src,), 'convert', mode, out=dst, regimes=(f'ties:{dst}', 'small'))
    c['asm_two_out'] = Fn(f'{asm}"{{{{ mul.lo.s32 $0, $2, $3; mul.hi.s32 $1, $2, $3; }}}}", "=r,=r,r,r", [{{x}}, {{y}}], '
                          'dtype=(tl.int32, tl.int32), is_pure=True, pack=1)[1]', ('i32',), 'mulhi',
                          regimes=('bits',))
    # map_elementwise over a scalar function with control flow
    c['map_pick'] = Fn('tl.map_elementwise(_slice_pick, {x}, {y})', ('f32', 'f16', 'i32', 'i64', 'f64'), 'pick',
                       regimes=('small', 'wide', 'bits'), helper=MAP_PICK)
    return c


MAP_PICK = '''@triton.jit
def _slice_pick(a, b):
    if a > b:
        return a - b
    else:
        return b * 2'''


def tilelang_catalog():
    c = {}
    for name, symbol, sem in (('add', '+', 'add'), ('sub', '-', 'sub'), ('mul', '*', 'mul')):
        c[name] = Fn(f'({{x}} {symbol} {{y}})', FLOAT + INTEGER, sem, const=3)
    c['div'] = Fn('({x} / {y})', FLOAT, 'div', domains=('any', 'nonzero'), const=0.75)
    for op, sem, arity, extra in (('add', 'add', 2, {}), ('sub', 'sub', 2, {}), ('mul', 'mul', 2, {}),
                                  ('fdiv', 'div', 2, {'domains': ('any', 'nonzero')}), ('fmaf', 'fma', 3, {}),
                                  ('fsqrt', 'sqrt', 1, {'domains': ('nonneg',)}),
                                  ('frcp', 'rcp', 1, {'domains': ('nonzero',)})):
        operands = ', '.join('{' + v + '}' for v in 'xyz'[:arity])
        for mode in ('rn', 'rz', 'rd', 'ru'):
            dtypes = F32_64 if mode != 'rn' else FLOAT
            c[f'ieee_{op}_{mode}'] = Fn(f'T.ieee_{op}({operands}, "{mode}")', dtypes, sem, mode, **extra)
    c['ieee_frsqrt'] = Fn('T.ieee_frsqrt({x})', ('f32',), 'rsqrt', domains=('pos',))
    c['sqrt'] = Fn('T.sqrt({x})', F32_64, 'sqrt', domains=('nonneg',))
    c['sqrt_half'] = Fn('T.sqrt({x})', ('f16', 'bf16'), 'sqrt', domains=('nonneg',), ulps=1)
    c['rsqrt'] = Fn('T.rsqrt({x})', FLOAT, 'rsqrt_approx', domains=('pos',), ulps=4)
    for name in ('exp', 'exp2', 'log', 'log2', 'sin', 'cos', 'tanh', 'erf', 'sigmoid'):
        domain = (1 / 64, 64) if name.startswith('log') else (-4, 4)
        c[name] = approximation(name, FLOAT, name, domain, 16, template=f'T.{name}({{x}})')
    # half-precision variants of these lack CUDA intrinsic rules (one
    # representative half type keeps that class reachable)
    HALF = ('f16', 'f32', 'f64')
    c['pow'] = Fn('T.pow({x}, {y})', FLOAT, 'pow', regimes=('range:0.5:3',), ulps=32)
    c['hypot'] = Fn('T.hypot({x}, {y})', HALF, 'hypot', regimes=('range:-8:8',), ulps=16)
    for name, sem in (('round', 'rint'), ('nearbyint', 'rint'), ('trunc', 'trunc'), ('floor', 'floor'),
                      ('ceil', 'ceil')):
        c[name] = Fn(f'T.{name}({{x}})', FLOAT, sem, regimes=('ties:i32', 'wide', 'specials'), signed=True)
    c['fabs'] = Fn('T.fabs({x})', FLOAT, 'abs', signed=True)
    c['abs'] = Fn('T.abs({x})', FLOAT + SIGNED, 'abs', regimes=('small', 'wide', 'specials'), signed=True)
    c['neg'] = Fn('(-{x})', FLOAT + SIGNED, 'neg', regimes=('small', 'wide', 'specials'), signed=True)
    c['copysign'] = Fn('T.copysign({x}, {y})', HALF, 'copysign', signed=True)
    c['fmod'] = Fn('T.fmod({x}, {y})', FLOAT, 'fmod', domains=('any', 'nonzero'), signed=True,
                   regimes=('small', 'wide'))
    c['nextafter'] = Fn('T.nextafter({x}, {y})', HALF, 'nextafter', regimes=('small', 'wide'))
    c['ldexp'] = Fn('T.ldexp({x}, {y})', HALF, 'ldexp', types=('$', 'i32'), domains=('any', 'exp'))
    for name in ('isnan', 'isinf', 'isfinite'):  # bfloat16 is rejected with a clear message
        c[name] = Fn(f'T.cast(T.{name}({{x}}), "int32")', ('f16', 'f32', 'f64'), name, out='i32',
                     regimes=('specials',))
    c['clamp'] = Fn('T.clamp({x}, {y}, {z})', FLOAT + INTEGER, 'clamp', regimes=('small', 'wide', 'bits'))
    c['max'] = Fn('T.max({x}, {y})', FLOAT + INTEGER, 'max', regimes=('small', 'wide', 'bits', 'edges'), const=3)
    c['min'] = Fn('T.min({x}, {y})', FLOAT + INTEGER, 'min', regimes=('small', 'wide', 'bits', 'edges'), const=3)
    c['clz'] = Fn('T.cast(T.clz({x}), "int32")', ('i32', 'i64', 'u32', 'u64'), 'clz', out='i32')
    c['popcount'] = Fn('T.cast(T.popcount({x}), "int32")', ('i32', 'i64', 'u32', 'u64', 'u16', 'u8'), 'popc',
                       out='i32')
    c['not'] = Fn('T.bitwise_not({x})', INTEGER, 'not')
    for name in ('and', 'or', 'xor'):
        c[name] = Fn(f'T.bitwise_{name}({{x}}, {{y}})', INTEGER, name, const=3)
    for name, fn in (('shl', 'shift_left'), ('shr', 'shift_right')):
        for dt in INTEGER:
            c[f'{name}_{dt}'] = Fn(f'T.{fn}({{x}}, {{y}})', (dt,), name, domains=('any', f'shift:{dt}'), const=3)
    for name, sem in (('truncdiv', 'idiv'), ('truncmod', 'imod'), ('floordiv', 'floordiv'),
                      ('floormod', 'floormod')):
        c[name] = Fn(f'T.{name}({{x}}, {{y}})', INTEGER, sem, domains=('any', 'nonzero'), const=3,
                     regimes=('small', 'bits'))
    c['reinterpret'] = Fn('T.reinterpret({x}, "{twin}")', ('f16', 'bf16', 'f32', 'f64', 'i16', 'u16', 'i32', 'u32',
                                                          'i64', 'u64'),
                          'bitcast', out='twin', regimes=('wide', 'specials', 'bits', 'edges'), signed=True)
    for name, sem in (('add2', 'add'), ('sub2', 'sub'), ('mul2', 'mul'), ('max2', 'max'), ('min2', 'min'),
                      ('fma2', 'fma'), ('abs2', 'abs')):
        operands = {'fma2': '{x}, {y}, {z}', 'abs2': '{x}'}.get(name, '{x}, {y}')
        c[name] = Fn(f'T.{name}({operands})', ('f16', 'bf16', 'f32'), sem, packed=True,
                     regimes=('small', 'wide'), signed=name == 'abs2')
    return c


CATALOGS = {'triton': triton_catalog(), 'tilelang': tilelang_catalog()}
# Constant operands of the clamp entries (lower bound, upper bound).
CLAMP = {'float': (-1.5, 2.25), 'int': (-3, 5), 'uint': (1, 5)}


def clamp_bounds(dt):
    return CLAMP['float' if is_float(dt) else 'uint' if dt[0] == 'u' else 'int']


class MathSlice(Slice):
    name = 'math'
    SIMPLEST = {'yform': 'tensor', 'widen': 0, 'values': 'small', 'stage': 'global'}

    def space(self, backend):
        catalog = CATALOGS[backend]
        dtypes = sorted({dt for fn in catalog.values() for dt in fn.dtypes})
        shapes = TRITON_SHAPES if backend == 'triton' else TILELANG_SHAPES
        space = {'fn': tuple(sorted(catalog)), 'in_dt': tuple(dtypes),
                 'values': ('small', 'wide', 'specials', 'ties', 'bits', 'edges', 'range'),
                 'yform': ('tensor', 'const'), 'widen': (0, 1), 'shape': shapes, 'tail': ('none', 'last', 'all'),
                 'pair': (0, 1)}
        if backend == 'triton':
            space.update(warps=(1, 2, 4, 8), warps2=(1, 2, 4, 8), dynamic=(0, 1))
        else:
            space.update(threads=(32, 64, 128, 256), threads2=(32, 64, 128, 256),
                         stage=('global', 'fragment', 'shared'))
        return space

    def sample(self, rng, backend):
        params = super().sample(rng, backend)
        fn = CATALOGS[backend][params['fn']]
        params['in_dt'] = rng.choice(fn.dtypes)
        params['values'] = rng.choice(fn.allowed_regimes(params['in_dt'])).split(':')[0]
        return params

    def legalize(self, params, backend):
        space = self.space(backend)
        for knob, values in space.items():
            if params.get(knob) not in values:
                params[knob] = values[0]
        for knob in list(params):
            if knob not in space:
                del params[knob]
        fn = CATALOGS[backend][params['fn']]
        if params['in_dt'] not in fn.dtypes:
            params['in_dt'] = fn.dtypes[0]
        dt = params['in_dt']
        regimes = fn.allowed_regimes(dt)
        names = [r.split(':')[0] for r in regimes]
        if params['values'] not in names:
            params['values'] = names[0]
        regime = regimes[names.index(params['values'])]
        if fn.arity < 2 or fn.const is None:
            params['yform'] = 'tensor'
        out = fn.out_type(dt)
        if out not in WIDE or fn.sem == 'bitcast' or fn.packed:
            params['widen'] = 0
        if fn.packed:
            # x2 vector views of a one-dimensional global tensor
            if len(parse_shape(params['shape'])) != 1:
                params['shape'] = '128'
            params['tail'], params['stage'] = 'none', 'global'
        shape = parse_shape(params['shape'])
        if backend == 'tilelang':
            size = math.prod(shape) // (2 if fn.packed else 1)
            for knob in ('threads', 'threads2'):
                if size % params[knob]:
                    params[knob] = max(t for t in (32, 64, 128, 256) if size % t == 0 or t == 32)
        if not params['pair']:
            params['warps2' if backend == 'triton' else 'threads2'] = params['warps' if backend == 'triton'
                                                                            else 'threads']
        store = WIDE[out] if params['widen'] else out
        return {'backend': backend, 'fn': fn, 'in_dt': dt, 'types': fn.operand_types(dt), 'out': out,
                'store': store, 'regime': regime, 'shape': shape, 'valid': valid_extents(shape, params['tail'])}

    def plan_data(self, params, plan):
        return {}

    # ---- harness ------------------------------------------------------------
    def operand_lists(self, params, plan):
        """Harness lines building the operand value lists a0, a1, a2."""
        fn, types = plan['fn'], plan['types']
        n = math.prod(plan['valid'])
        lines = []
        salt = zlib.crc32(params['fn'].encode()) % 7919
        for k, dt in enumerate(types):
            domain = fn.domains[k] if k < len(fn.domains) else 'any'
            regime = plan['regime'] if k == 0 or domain == 'any' or domain in ('nonzero', 'pos', 'nonneg') \
                else 'small'
            if fn.sem == 'clamp' and k > 0:
                lines.append(f'    a{k} = [{clamp_bounds(plan["in_dt"])[k - 1]!r}] * {n}')
                continue
            if dt != plan['in_dt'] and regime.startswith('ties'):
                regime = 'small'
            lines.append(f'    a{k} = _slice_mvalues({dt!r}, {n}, {regime!r}, {domain!r}, seed, {salt + k})')
            if k == 1 and params['yform'] == 'const':
                lines.append(f'    a1 = [{fn.const!r}] * {n}')
        return lines

    def expression(self, params, plan, names):
        fn = plan['fn']
        values = dict(zip('xyz', names))
        if fn.sem == 'clamp':
            lo, hi = clamp_bounds(plan['in_dt'])
            if params['yform'] == 'const' or plan['backend'] == 'tilelang':
                t = (TILELANG_TYPE if plan['backend'] == 'tilelang' else TRITON_TYPE)[plan['in_dt']]
                values['y'], values['z'] = ((f'T.cast({lo!r}, "{t}")', f'T.cast({hi!r}, "{t}")')
                                            if plan['backend'] == 'tilelang' else (repr(lo), repr(hi)))
        elif params['yform'] == 'const':
            values['y'] = repr(fn.const)
        twin = TWIN.get(plan['in_dt'], '')
        types = TRITON_TYPE if plan['backend'] == 'triton' else TILELANG_TYPE
        return fn.template.format(twin=types.get(twin, ''), tt=types[plan['in_dt']], **values)

    def triton_kernel(self, params, plan):
        fn, types = plan['fn'], plan['types']
        shape, valid = plan['shape'], plan['valid']
        offs, mask, names = triton_offsets(shape, valid, params['dynamic'])
        args = ', '.join(f'n{i}' for i in range(len(shape)))
        body = [f'    offs = {offs}', f'    mask = {mask}']
        operands = []
        for k, dt in enumerate(types):
            other = '1.0' if is_float(dt) else '1'
            load = f'tl.load(A{k} + offs, mask=mask, other={other})'
            if dt in ('u16', 'u32', 'u64'):
                load += f'.to({TRITON_TYPE[dt]}, bitcast=True)'
            body.append(f'    a{k} = {load}')
            operands.append(f'a{k}')
        body.append(f'    r = {self.expression(params, plan, operands)}')
        if plan['store'] != plan['out']:
            body.append(f'    r = r.to({TRITON_TYPE[plan["store"]]})')
        if plan['store'] in ('u16', 'u32', 'u64'):
            body.append(f'    r = r.to({TRITON_TYPE["i" + plan["store"][1:]]}, bitcast=True)')
        body.append('    tl.store(OUT + offs, r, mask=mask)')
        extents = ', '.join(str(v) for v in valid)
        inputs = ', '.join(f'A{k}' for k in range(3))
        helper = [fn.helper, ''] if fn.helper else []
        return '\n'.join(helper + ['@triton.jit', f'def kernel({inputs}, OUT, {args}):'] + body + [
            '', 'def launch(A, out, options):',
            f'    kernel[(1,)](*(A + [A[0]] * (3 - len(A))), out, {extents}, **options)'])

    def tilelang_kernel(self, params, plan):
        from .cast import tilelang_index, tilelang_launch
        fn, types = plan['fn'], plan['types']
        shape, valid = plan['shape'], plan['valid']
        out_t = TILELANG_TYPE[plan['store']]
        lines = []
        if fn.packed:
            vector = TILELANG_TYPE[plan['in_dt']] + 'x2'
            half = shape[0] // 2
            for k in range(len(types)):
                lines.append(f'v{k} = T.view(A{k}, ({half},), "{vector}")')
            lines.append(f'vo = T.view(O, ({half},), "{vector}")')
            expr = self.expression(params, plan, [f'v{k}[i]' for k in range(len(types))])
            lines += [f'for i in T.Parallel({half}):', f'    vo[i] = {expr}']
        else:
            idx = tilelang_index(len(shape))
            at = '[' + ', '.join(idx) + ']'
            origin = '[' + ', '.join('0' for _ in shape) + ']'
            guard = ' and '.join(f'{i} < {v}' for i, v in zip(idx, valid))
            stage = params['stage']
            if stage == 'global':
                names = [f'A{k}{at}' for k in range(len(types))]
            else:
                alloc = 'T.alloc_fragment' if stage == 'fragment' else 'T.alloc_shared'
                for k, dt in enumerate(types):
                    lines += [f's{k} = {alloc}({shape!r}, "{TILELANG_TYPE[dt]}")', f'T.copy(A{k}{origin}, s{k})']
                names = [f's{k}{at}' for k in range(len(types))]
            expr = self.expression(params, plan, names)
            if plan['store'] != plan['out']:
                expr = f'T.cast({expr}, "{out_t}")'
            loop = f'for {", ".join(idx)} in T.Parallel({", ".join(map(str, shape))}):'
            if stage == 'global':
                lines += [loop, f'    if {guard}:', f'        O{at} = {expr}']
            else:
                lines += [f'of = T.alloc_fragment({shape!r}, "{out_t}")', loop, f'    of{at} = {expr}',
                          f'T.copy(of, O{origin})']
        signature = ', '.join(f'A{k}: T.Tensor({valid!r}, "{TILELANG_TYPE[dt]}")' for k, dt in enumerate(types))
        signature += f', O: T.Tensor({valid!r}, "{out_t}")'
        text = tilelang_launch(signature, lines)
        return text.replace('def launch(X, Y, out, options):', 'def launch(A, out, options):').replace(
            '_KERNELS[threads](X, Y, out)', '_KERNELS[threads](*A, out)')

    def emit(self, program, config):
        params = dict(program.params)
        plan = self.legalize(params, program.backend)
        if params != program.params:
            raise ValueError('slice program parameters are not legalized')
        seed = config.input_seed if config is not None else 0
        backend = program.backend
        fn = plan['fn']
        kernel = self.triton_kernel(params, plan) if backend == 'triton' else self.tilelang_kernel(params, plan)
        imports = ('import triton\nimport triton.language as tl\nfrom triton.language.extra.cuda import libdevice'
                   if backend == 'triton' else 'import tilelang\nimport tilelang.language as T')
        valid = tuple(plan['valid'])
        n = math.prod(valid)
        arity = len(plan['types'])
        # Triton reads unsigned 16-64 bit operands from signed buffers of the
        # same width; TileLang takes torch's unsigned tensors.
        storage = [('i' + dt[1:] if backend == 'triton' and dt in ('u16', 'u32', 'u64') else dt)
                   for dt in plan['types']]
        store = plan['store']
        store_storage = 'i' + store[1:] if backend == 'triton' and store in ('u16', 'u32', 'u64') else store
        if fn.sem == 'bitcast':
            twin = TORCH_TYPE[storage_twin(plan, backend)]
            reference = [f"    want = _slice_values_of(_slice_tensor(a0, {storage[0]!r}, 'cpu').view(torch.{twin}), "
                         f"{plan['out']!r})"]
        else:
            reference = [f'    want = [_slice_meval({fn.sem!r}, {fn.mode!r}, [a[i] for a in args], '
                         f'{plan["types"]!r}, {plan["out"]!r}) for i in range({n})]']
        if store != plan['out']:
            reference.append(f'    want = [None if w is None else (float(w) if {is_float(store)!r} else int(w)) '
                             f'for w in want]')
        label = f"slice=math fn={params['fn']} {plan['in_dt']}"
        lines = [
            'import sys', 'import torch', imports, '', SOURCES, '', kernel, '',
            'def reference_case(seed):',
            '    """Operand values and the expected result values (CPU only)."""',
        ] + self.operand_lists(params, plan) + [
            f'    args = [{", ".join(f"a{k}" for k in range(arity))}]',
        ] + reference + [
            '    return args, want',
            '',
            'def main():',
            f'    seed = {seed!r}',
            "    print('TILESMITH_STAGE=reference', file=sys.stderr, flush=True)",
            '    args, want = reference_case(seed)',
            f'    A = [_slice_tensor(a, s).view({valid!r}) for a, s in zip(args, {storage!r})]',
            f'    for label, options in {self.variants(params, backend)!r}:',
            '        previous = None',
            '        for run in range(2):',
            f'            storage = _slice_tensor([7] * {n + 2 * GUARD}, {store_storage!r})',
            f'            out = storage[{GUARD}:{GUARD + n}].view({valid!r})',
            "            print(f'TILESMITH_STAGE=execute:{label}:{run}', file=sys.stderr, flush=True)",
            '            try:',
            '                launch(A, out, options)',
            '                torch.cuda.synchronize()',
            '            except Exception as exc:',
            '                _slice_reject(exc)',
            '                raise',
            f'            guard = _slice_values_of(storage[:{GUARD}], {store_storage!r}) + '
            f'_slice_values_of(storage[-{GUARD}:], {store_storage!r})',
            '            if any(g != 7 for g in guard):',
            f"                raise RuntimeError(f'WRONG RESULT: output canary modified by {label} variant={{label}}')",
            '            if previous is not None and not torch.equal(_slice_bits(previous), _slice_bits(out)):',
            f"                raise RuntimeError(f'WRONG RESULT: repeat determinism of {label} variant={{label}}')",
            f'            got = _slice_values_of(out, {store!r})',
            f"            _slice_mcheck(got, want, f'{label} variant={{label}}', {plan['out']!r}, {fn.ulps!r}, "
            f'{fn.signed!r})',
            '            previous = out.clone()',
            "    print('ALL PASSED')",
            '',
            "if __name__ == '__main__':",
            '    main()',
        ]
        return '\n'.join(lines) + '\n'


def storage_twin(plan, backend):
    """Torch storage type of a bitcast's result (Triton keeps unsigned bits
    in signed buffers)."""
    out = plan['out']
    if backend == 'triton' and out in ('u16', 'u32', 'u64'):
        return 'i' + out[1:]
    return out
