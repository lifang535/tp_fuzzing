"""Standalone helpers embedded into every slice harness.

The emitter copies these functions' source at import time (see
SOURCES below), so the generated scripts stay self-contained and never read
this file while a campaign runs.
"""
import inspect


def _slice_values(shape, lo, hi, frac, seed, salt):
    """Exact inputs on the grid k * 2**-frac within [lo, hi], as float64."""
    import torch
    g = torch.Generator().manual_seed(seed * 1000003 + salt)
    scale = 2 ** frac
    k = torch.randint(int(round(lo * scale)), int(round(hi * scale)) + 1, tuple(shape), generator=g)
    return k.double() / scale


def _slice_storage(values, name):
    """Convert exact float64 values to a torch storage dtype on the GPU."""
    import torch
    dtype = getattr(torch, name)
    if dtype.is_floating_point:
        return values.to(dtype).cuda()
    return values.to(torch.int64).to(dtype).cuda()


def _slice_cast(values, dtype):
    """Reference conversion of exact values: integer targets truncate."""
    import torch
    if dtype.startswith('f') or dtype == 'bf16':
        return values
    return torch.trunc(values)


def _slice_step(values, step, operand, semantics):
    """Apply one legalized step (op, dtype, target, source, constant)."""
    import torch
    op, dtype, target, source, c = step
    x = values
    if op != 'none':
        y = operand if source == 'input' else torch.full_like(x, float(c) if c is not None else 0.0)
        if op == 'add':
            x = x + y
        elif op == 'sub':
            x = x - y
        elif op == 'mul':
            x = x * y
        elif op == 'max':
            x = torch.maximum(x, y)
        elif op == 'min':
            x = torch.minimum(x, y)
        elif op == 'select':
            x = torch.where(x > y, x, y)
        elif op == 'abs':
            x = x.abs()
        elif op == 'neg':
            x = -x
        elif op == 'half':
            x = x * 0.5
        elif op == 'floor':
            x = torch.floor(x)
        elif op == 'ceil':
            x = torch.ceil(x)
        elif op in ('and', 'or', 'xor'):
            a, b = x.to(torch.int64), y.to(torch.int64)
            x = (a & b if op == 'and' else a | b if op == 'or' else a ^ b).double()
        elif op == 'shl':
            x = x * 2.0 ** c
        elif op == 'shr':
            x = torch.floor(x / 2.0 ** c)
        elif op == 'div':
            x = torch.div(x, y, rounding_mode=semantics)
        elif op == 'mod':
            x = torch.fmod(x, y) if semantics == 'trunc' else torch.remainder(x, y)
        else:
            raise ValueError('unknown slice step ' + op)
    if target != dtype:
        x = _slice_cast(x, target)
    return x


def _slice_output(shape, name, guard):
    """A guarded output allocation filled with 7 and the view a kernel writes."""
    import math
    import torch
    size = math.prod(shape)
    storage = torch.full((size + 2 * guard,), 7.0, dtype=torch.float32, device='cuda').to(getattr(torch, name))
    return storage, storage[guard:guard + size].view(tuple(shape))


def _slice_bits(tensor):
    """An integer view of a tensor for bitwise run-to-run comparison."""
    import torch
    width = {1: torch.uint8, 2: torch.int16, 4: torch.int32, 8: torch.int64}[tensor.element_size()]
    return tensor.contiguous().view(width)


def _slice_compare(actual, expected, label):
    """Bit-exact value comparison; NaNs match NaNs."""
    import torch
    got = actual.detach().cpu()
    got = got.double() if got.dtype.is_floating_point else got.to(torch.int64).double()
    want = expected.detach().cpu().double()
    if tuple(got.shape) != tuple(want.shape):
        raise RuntimeError(f'WRONG RESULT: {label} shape {tuple(got.shape)} != {tuple(want.shape)}')
    same = (got == want) | (torch.isnan(got) & torch.isnan(want))
    if not bool(same.all()):
        bad = (~same).nonzero()
        first = tuple(int(v) for v in bad[0])
        raise RuntimeError(
            f'WRONG RESULT: {label} differs at {bad.shape[0]}/{same.numel()} elements; '
            f'first index {first}: got {got[first].item()!r}, expected {want[first].item()!r}')


# (significand bits, minimum normal exponent, largest finite, has infinities)
_SLICE_FORMATS = {'f64': (53, -1022, 1.7976931348623157e308, True), 'f32': (24, -126, 3.4028234663852886e38, True),
                  'f16': (11, -14, 65504.0, True), 'bf16': (8, -126, 3.3895313892515355e38, True),
                  'f8e4': (4, -6, 448.0, False), 'f8e5': (3, -14, 57344.0, True)}


def _slice_round(x, fmt, mode='rtne'):
    """Correctly rounded conversion of float64 values to a binary format,
    subnormals included (one rounding; numpy agrees, torch's float64
    conversions round twice through float32)."""
    import torch
    mant, emin, largest, has_inf = _SLICE_FORMATS[fmt]
    a = x.abs()
    _, exponent = torch.frexp(a)
    e = torch.clamp(exponent.to(torch.float64) - 1, min=emin)
    quantum = torch.pow(torch.full_like(a, 2.0), e - (mant - 1))
    q = a / quantum
    out = (torch.round(q) if mode == 'rtne' else torch.trunc(q)) * quantum
    over = out > largest
    if mode == 'rtne':
        out = torch.where(over, torch.full_like(out, float('inf') if has_inf else float('nan')), out)
    else:
        out = torch.where(over, torch.full_like(out, largest), out)
    out = torch.where(a == 0, a, out)
    out = torch.where(torch.isinf(a), a if has_inf else torch.full_like(a, float('nan')), out)
    out = torch.where(torch.isnan(a), a, out)
    return torch.copysign(out, x)


def _slice_round_values(src, dst, n, seed, mix):
    """Exact values of format/integer type src that stress a conversion to
    dst: ties and near-ties of the dst grid, a log-uniform spread over the
    dst range including subnormals, or its edges (largest values, smallest
    normals, signed zeros). Integer types are 'i<bits>' or 'u<bits>'."""
    import torch
    g = torch.Generator().manual_seed(seed)

    def integer(t):
        return t[0] in 'iu' and t[1:].isdigit()

    def bounds(t):
        if integer(t):
            bits = int(t[1:])
            return (0, 2 ** bits - 1) if t[0] == 'u' else (-2 ** (bits - 1), 2 ** (bits - 1) - 1)
        largest = _SLICE_FORMATS[t][2]
        return -largest, largest
    lo, hi = bounds(src)
    dlo, dhi = bounds(dst)
    lo, hi = max(lo, dlo, -2.0 ** 53), min(hi, dhi, 2.0 ** 53)
    if integer(dst):
        # float -> integer truncates; keep away from the integer's ends
        lo, hi = lo + 1, hi - 1
    sign = torch.where(torch.rand(n, generator=g) < 0.5, -1.0, 1.0).double()
    if lo >= 0:
        sign = torch.ones(n, dtype=torch.float64)
    if integer(dst) or integer(src) and not integer(dst) and mix != 'ties':
        top = max(abs(lo), abs(hi))
        mag = torch.pow(2.0, torch.rand(n, generator=g).double() * torch.log2(torch.tensor(top + 1.0)).item())
        values = sign * mag
    else:
        mant, emin, largest, _ = _SLICE_FORMATS[dst if not integer(dst) else 'f64']
        top = min(largest, max(abs(lo), abs(hi)))
        etop = int(torch.floor(torch.log2(torch.tensor(top))).item())
        low = emin if not integer(src) else max(emin, mant - 1)
        if mix == 'ties' and low > etop:
            mix = 'wide'
        if mix == 'ties':
            e = torch.randint(low, etop + 1, (n,), generator=g).double()
            quantum = torch.pow(2.0, e - (mant - 1))
            k = torch.randint(2 ** (mant - 1), 2 ** mant, (n,), generator=g).double()
            k = torch.where(e == emin, torch.randint(0, 2 ** mant, (n,), generator=g).double(), k)
            nudge = torch.randint(-2, 3, (n,), generator=g).double() * 2.0 ** -12
            values = sign * (k + 0.5 + nudge) * quantum
        elif mix == 'edges':
            picks = torch.tensor([0.0, 2.0 ** emin, 2.0 ** (emin - mant + 1), top, top * (1 - 2.0 ** -mant),
                                  2.0 ** emin * 1.5, 1.0, 2.0 ** etop], dtype=torch.float64)
            values = sign * picks[torch.randint(0, len(picks), (n,), generator=g)]
            values = values * torch.pow(2.0, torch.randint(-1, 1, (n,), generator=g).double())
        else:
            e = torch.rand(n, generator=g).double() * (etop - emin + mant) + emin - mant
            values = sign * torch.pow(2.0, e)
    values = torch.clamp(values, lo, hi)
    values = torch.trunc(values) if integer(src) else _slice_round(values, src)
    # Rounding to src may leave the common range (2**31 - 1 rounds up in f32).
    inside = torch.isfinite(values) & (values >= lo) & (values <= hi)
    return torch.where(inside, values, torch.zeros_like(values))


def _slice_fround(value, fmt, mode='rn'):
    """The correctly rounded value of an exact rational (int, float or
    Fraction) in a binary format, subnormals included: to nearest even (rn),
    toward zero (rz), up (ru) or down (rd). Returns a Python float."""
    from fractions import Fraction
    mant, emin, largest, has_inf = _SLICE_FORMATS[fmt]
    v = Fraction(value)
    if v == 0:
        return 0.0
    negative = v < 0
    a = -v if negative else v
    e = a.numerator.bit_length() - a.denominator.bit_length()
    if a < Fraction(2) ** e:
        e -= 1
    quantum = Fraction(2) ** (max(e, emin) - mant + 1)
    q = a / quantum
    n, rem = divmod(q.numerator, q.denominator)
    if mode == 'rn':
        up = 2 * rem > q.denominator or (2 * rem == q.denominator and n % 2 == 1)
    else:
        up = rem > 0 and ((mode == 'ru' and not negative) or (mode == 'rd' and negative))
    result = (n + up) * quantum
    if result > largest:
        away = mode == 'rn' or (mode == 'ru' and not negative) or (mode == 'rd' and negative)
        result = (float('inf') if has_inf else float('nan')) if away else largest
    result = float(result)
    return -result if negative else result


def _slice_fsqrt(value, fmt, mode='rn'):
    """Correctly rounded square root of an exact non-negative rational."""
    import math
    from fractions import Fraction
    a = Fraction(value)
    if a == 0:
        return float(value)
    if a < 0:
        return float('nan')
    e = a.numerator.bit_length() - a.denominator.bit_length()
    k = 200 + max(0, -e)
    scaled = a * (1 << (2 * k))
    n = scaled.numerator // scaled.denominator
    s = math.isqrt(n)
    exact = scaled.denominator == 1 and s * s == n
    # An inexact root lies strictly inside (s, s + 1) / 2**k, an interval no
    # rounding boundary of the format splits; its midpoint rounds the same.
    root = Fraction(s, 1 << k) if exact else Fraction(2 * s + 1, 1 << (k + 1))
    return _slice_fround(root, fmt, mode)


def _slice_fnext(x, toward, fmt):
    """nextafter(x, toward) on the grid of a binary format."""
    import math
    from fractions import Fraction
    mant, emin, largest, _ = _SLICE_FORMATS[fmt]
    if math.isnan(x) or math.isnan(toward):
        return float('nan')
    if x == toward:
        return toward
    up = toward > x
    if x == 0:
        tiny = 2.0 ** (emin - mant + 1)
        return tiny if up else -tiny
    if math.isinf(x):
        return math.copysign(largest, x)
    a = abs(Fraction(x))
    away = up == (x > 0)
    e = math.frexp(abs(x))[1] - 1
    q = Fraction(2) ** (max(e, emin) - mant + 1)
    if not away and a == Fraction(2) ** e and e > emin:
        q /= 2
    r = a + q if away else a - q
    return math.copysign(float(r) if r <= largest else float('inf'), x)


def _slice_int_range(dt):
    bits = int(dt[1:])
    return (0, (1 << bits) - 1) if dt[0] == 'u' else (-(1 << (bits - 1)), (1 << (bits - 1)) - 1)


def _slice_wrap(value, dt):
    """Two's complement wrap of a Python int into integer type dt."""
    bits = int(dt[1:])
    value &= (1 << bits) - 1
    if dt[0] == 'i' and value >= 1 << (bits - 1):
        value -= 1 << bits
    return value


def _slice_mvalues(dt, n, regime, domain, seed, salt):
    """Python values of type dt for one math operand. regime: small, wide,
    specials, ties:<target> (midpoints of the target's grid), bits, edges or
    range:<lo>:<hi>; domain: any, pos, nonneg, nonzero, unit, shift (0 to
    bits - 1 of the type named after it, e.g. shift:i32), exp (-8..8)."""
    import math
    import random
    rng = random.Random(seed * 1000003 + salt)
    if domain.startswith('shift'):
        bits = int(domain.split(':')[1][1:])
        return [rng.randint(0, bits - 1) for _ in range(n)]
    if domain == 'exp':
        return [rng.randint(-8, 8) for _ in range(n)]
    if domain == 'selector':
        return [rng.randint(0, 0x7777) for _ in range(n)]
    integer = dt[0] in 'iu'
    if integer:
        lo, hi = _slice_int_range(dt)
        if domain in ('pos', 'nonneg') and lo < 0:
            lo = 0
        if domain == 'small_pos':
            lo, hi = 1, min(hi, 60)
        if regime.startswith('ties:') and regime[5] in 'fb':
            mant = _SLICE_FORMATS[regime[5:]][0]
            top = max(lo, hi).bit_length()
        values = []
        for _ in range(n):
            if regime == 'bits':
                v = rng.randint(lo, hi)
            elif regime == 'edges':
                v = rng.choice([lo, hi, 0, 1, lo + 1, hi - 1, rng.randint(lo, hi), rng.randint(lo, hi)])
            elif regime.startswith('ties:') and regime[5] in 'fb' and top > mant + 1:
                # an integer halfway between two values of the float grid
                e = rng.randint(mant + 1, top - 1)
                v = (2 * rng.randint(1 << (mant - 1), (1 << mant) - 1) + 1) << (e - mant - 1)
                v = v if lo == 0 or rng.random() < 0.5 else -v
                v = v if lo <= v <= hi else rng.randint(lo, hi)
            else:
                v = rng.randint(max(lo, -50), min(hi, 100 if lo >= 0 else 50))
            if domain in ('pos', 'nonzero', 'small_pos') and v == 0:
                v = 1
            values.append(v)
        return values
    mant, emin, largest, has_inf = _SLICE_FORMATS[dt]
    span = {'f64': 40, 'f32': 30, 'bf16': 30, 'f16': 6, 'f8e4': 3, 'f8e5': 6}[dt]
    values = []
    for _ in range(n):
        if regime.startswith('range:'):
            lo, hi = (float(t) for t in regime.split(':')[1:])
            v = _slice_fround(rng.uniform(lo, hi), dt)
        elif regime == 'small':
            v = rng.randint(-32, 32) / 4
        elif regime.startswith('ties:'):
            target = regime[5:]
            if target[0] in 'iu':
                v = rng.randint(-60, 60) + rng.choice((0.5, 0.5, 0.25, 0.75, 0.0))
            else:
                tm, temin = _SLICE_FORMATS[target][:2]
                top = min(span, 20)
                e = rng.randint(max(-top, temin), top)
                k = rng.randint(1 << (tm - 1), (1 << tm) - 1)
                nudge = rng.choice((0, 0, 1, -1))
                v = (2 * k + 1 + nudge * 2.0 ** -6) * 2.0 ** (e - tm)
                v = _slice_fround(v, dt) * rng.choice((-1, 1))
        elif regime == 'specials' and rng.random() < 0.4:
            v = rng.choice([0.0, -0.0, math.inf, -math.inf, math.nan, 1.0, -1.0, largest, -largest,
                            2.0 ** emin, -2.0 ** emin, 0.5])
            if not has_inf and math.isinf(v):
                v = math.nan
        else:
            m = rng.randint(1 << (mant - 1), (1 << mant) - 1)
            v = rng.choice((-1, 1)) * m * 2.0 ** (rng.randint(-span, span) - mant + 1)
        if domain in ('pos', 'nonneg') and not math.isnan(v):
            v = abs(v)
        if domain in ('pos', 'nonzero') and v == 0:
            v = 1.0
        if domain == 'unit' and not abs(v) <= 1:
            v = math.copysign(1.0, v) / 2 ** rng.randint(1, 4) if not math.isnan(v) else 0.5
        values.append(v)
    return values


def _slice_meval(sem, mode, args, types, out):
    """Exact (or float64) reference of one element of a math slice program.
    args are Python values of the operand types; returns the result as a
    Python value of type out, or None when the result is undefined for these
    arguments (the element is not checked)."""
    import math
    from fractions import Fraction
    x = args[0]
    y = args[1] if len(args) > 1 else None
    z = args[2] if len(args) > 2 else None
    t = types[0]
    integer = t[0] in 'iu'
    finite = all(not isinstance(a, float) or math.isfinite(a) for a in args)

    def rnd(value):
        return _slice_fround(value, out, mode)
    if sem in ('add', 'sub', 'mul', 'div', 'fma', 'rcp') and not integer:
        if not finite or (sem == 'div' and y == 0) or (sem == 'rcp' and x == 0):
            a = float(x)
            r = {'add': lambda: a + y, 'sub': lambda: a - y, 'mul': lambda: a * y,
                 'div': lambda: a / y if y else math.copysign(math.inf, a) * math.copysign(1, y) if a else math.nan,
                 'fma': lambda: a * y + z, 'rcp': lambda: 1 / a if a else math.copysign(math.inf, a)}[sem]()
            return r
        X = Fraction(x)
        exact = {'add': lambda: X + Fraction(y), 'sub': lambda: X - Fraction(y), 'mul': lambda: X * Fraction(y),
                 'div': lambda: X / Fraction(y), 'fma': lambda: X * Fraction(y) + Fraction(z),
                 'rcp': lambda: 1 / X}[sem]()
        return rnd(exact)
    if sem in ('add', 'sub', 'mul', 'neg', 'abs') and integer:
        r = {'add': lambda: x + y, 'sub': lambda: x - y, 'mul': lambda: x * y, 'neg': lambda: -x,
             'abs': lambda: abs(x)}[sem]()
        return _slice_wrap(r, out)
    if sem == 'sqrt':
        return _slice_fsqrt(x, out, mode) if math.isfinite(x) else (x if x > 0 else math.nan)
    if sem == 'rsqrt':
        return _slice_fsqrt(1 / Fraction(x), out, mode) if x > 0 and math.isfinite(x) else None
    if sem == 'neg':
        return -x
    if sem == 'abs':
        return abs(x)
    if sem in ('max', 'min', 'fmax', 'fmin', 'pmax', 'pmin'):
        if isinstance(x, float) and (math.isnan(x) or math.isnan(y)):
            if sem[0] == 'f':
                return y if math.isnan(x) else x
            return math.nan if sem[0] == 'p' else None
        return max(x, y) if sem.endswith('max') else min(x, y)
    if sem == 'clamp':
        if isinstance(x, float) and math.isnan(x):
            return None
        return min(max(x, y), z)
    if sem in ('floor', 'ceil', 'trunc', 'rint', 'round'):
        if not math.isfinite(x):
            return x
        X = Fraction(x)
        f = X.numerator // X.denominator
        r = {'floor': f, 'ceil': -((-X.numerator) // X.denominator), 'trunc': int(X),
             'rint': round(X), 'round': int(abs(X) + Fraction(1, 2)) * (1 if X >= 0 else -1)}[sem]
        if out[0] in 'iu':
            return r
        return math.copysign(float(r), x) if r == 0 else float(r)
    if sem == 'fmod' and not integer:
        if not finite or y == 0:
            return math.fmod(x, y) if y != 0 and not math.isnan(x) and not math.isnan(y) and not math.isinf(x) else math.nan
        X, Y = Fraction(x), Fraction(y)
        r = X - int(X / Y) * Y
        return math.copysign(float(r), x) if r == 0 else float(r)
    if sem == 'remainder':
        if not finite or y == 0:
            return None
        X, Y = Fraction(x), Fraction(y)
        r = X - round(X / Y) * Y
        return math.copysign(float(r), x) if r == 0 else float(r)
    if sem == 'copysign':
        return math.copysign(x, y)
    if sem == 'fdim':
        if not finite:
            return None
        return rnd(Fraction(x) - Fraction(y)) if x > y else 0.0
    if sem == 'nextafter':
        return _slice_fnext(x, y, out)
    if sem == 'ldexp':
        return rnd(Fraction(x) * Fraction(2) ** y) if math.isfinite(x) else x
    if sem in ('ilogb', 'logb'):
        if not math.isfinite(x) or x == 0:
            return None
        e = math.frexp(abs(x))[1] - 1
        return e if sem == 'ilogb' else float(e)
    if sem == 'signbit':
        return int(math.copysign(1, x) < 0)
    if sem == 'isnan':
        return int(math.isnan(x))
    if sem == 'isinf':
        return int(math.isinf(x))
    if sem == 'isfinite':
        return int(math.isfinite(x))
    if sem == 'saturate':
        return 0.0 if math.isnan(x) else min(max(x, 0.0), 1.0)
    if sem == 'convert':
        # float/int -> float/int with a rounding mode; float -> int is
        # undefined outside the integer's range
        if out[0] in 'iu':
            if not math.isfinite(x):
                return None
            X = Fraction(x)
            r = {'rn': round(X), 'rz': int(X), 'rd': X.numerator // X.denominator,
                 'ru': -((-X.numerator) // X.denominator)}[mode]
            lo, hi = _slice_int_range(out)
            return r if lo <= r <= hi else None
        return rnd(x) if not isinstance(x, float) or math.isfinite(x) else x
    # ---- integers --------------------------------------------------------------
    if sem in ('idiv', 'imod'):
        if y == 0 or (y == -1 and x == _slice_int_range(t)[0]):
            return None
        q = abs(x) // abs(y)
        q = q if (x >= 0) == (y > 0) else -q
        r = q if sem == 'idiv' else x - q * y
        return _slice_wrap(r, out)
    if sem in ('floordiv', 'floormod'):
        if y == 0:
            return None
        return _slice_wrap(x // y if sem == 'floordiv' else x % y, out)
    if sem == 'cdiv':
        return -((-x) // y) if y > 0 and x >= 0 else None
    if sem in ('and', 'or', 'xor'):
        return _slice_wrap(x & y if sem == 'and' else x | y if sem == 'or' else x ^ y, out)
    if sem == 'not':
        return _slice_wrap(~x, out)
    if sem == 'shl':
        return _slice_wrap(x << y, out)
    if sem == 'shr':
        return _slice_wrap(x >> y, out)
    bits = int(t[1:]) if integer else 0
    mask = (1 << bits) - 1
    if sem == 'popc':
        return bin(x & mask).count('1')
    if sem == 'clz':
        return bits - (x & mask).bit_length()
    if sem == 'ffs':
        u = x & mask
        return (u & -u).bit_length()
    if sem == 'brev':
        return _slice_wrap(int(format(x & mask, f'0{bits}b')[::-1], 2), out)
    if sem == 'mulhi':
        return _slice_wrap((x * y) >> bits, out)
    if sem == 'umulhi':
        return _slice_wrap(((x & mask) * (y & mask)) >> bits, out)
    if sem == 'mul24':
        def s24(v):
            v &= 0xffffff
            return v - (1 << 24) if t[0] == 'i' and v >= 1 << 23 else v
        return _slice_wrap(s24(x) * s24(y), out)
    if sem == 'hadd':
        return _slice_wrap((x + y) >> 1, out)
    if sem == 'rhadd':
        return _slice_wrap((x + y + 1) >> 1, out)
    if sem == 'sad':
        return _slice_wrap(abs(x - y) + (z & 0xffffffff), out)
    if sem == 'byte_perm':
        data = ((y & 0xffffffff) << 32) | (x & 0xffffffff)
        r = 0
        for i in range(4):
            r |= ((data >> (8 * ((z >> (4 * i)) & 7))) & 0xff) << (8 * i)
        return _slice_wrap(r, out)
    if sem == 'bitcast':
        import struct
        codes = {'f16': 'e', 'f32': 'f', 'f64': 'd', 'i16': 'h', 'i32': 'i', 'i64': 'q', 'u16': 'H',
                 'u32': 'I', 'u64': 'Q', 'i8': 'b', 'u8': 'B'}
        if t in ('bf16', 'f8e4', 'f8e5') or out in ('bf16', 'f8e4', 'f8e5'):
            return None  # checked bit-for-bit by the harness instead
        value = struct.unpack(codes[out], struct.pack(codes[t], x))[0]
        return value
    if sem == 'hiint':
        import struct
        return struct.unpack('<ii', struct.pack('<d', x))[1]
    if sem == 'loint':
        import struct
        return struct.unpack('<ii', struct.pack('<d', x))[0]
    if sem == 'hilo':
        import struct
        return struct.unpack('<d', struct.pack('<ii', y, x))[0]
    if sem == 'mad':
        return _slice_wrap(x * y + z, out)
    if sem == 'pick':
        if integer:
            return _slice_wrap(x - y if x > y else 2 * y, out)
        if not finite:
            return None
        return rnd(Fraction(x) - Fraction(y)) if x > y else rnd(2 * Fraction(y))
    if sem in ('philox', 'rand'):
        # Triton's Philox 4x32 (random.py): mode is 'seed:rounds:output'
        seed, rounds, which = (int(v) for v in mode.split(':'))
        m32 = 0xffffffff
        seed &= (1 << 64) - 1
        c = [x & m32, (x >> 32) & m32 if int(t[1:]) > 32 else 0, 0, 0]
        k0, k1 = seed & m32, (seed >> 32) & m32
        for _ in range(rounds):
            a0, a2 = c[0], c[2]
            c[0] = ((0xCD9E8D57 * a2) >> 32) ^ c[1] ^ k0
            c[2] = ((0xD2511F53 * a0) >> 32) ^ c[3] ^ k1
            c[1] = (0xCD9E8D57 * a2) & m32
            c[3] = (0xD2511F53 * a0) & m32
            k0, k1 = (k0 + 0x9E3779B9) & m32, (k1 + 0xBB67AE85) & m32
        r = c[which]
        if sem == 'philox':
            return r if out == 'u32' else _slice_wrap(r, out)
        r = _slice_wrap(r, 'i32')
        r = ~r if r < 0 else r
        scale = Fraction(_slice_fround(4.6566127342e-10, 'f32'))
        return _slice_fround(Fraction(_slice_fround(r, 'f32')) * scale, 'f32')
    # ---- approximations (float64 reference, compared with a tolerance) ----------
    approx = {'exp': math.exp, 'exp2': lambda v: 2.0 ** v, 'exp10': lambda v: 10.0 ** v, 'expm1': math.expm1,
              'log': math.log, 'log2': math.log2, 'log10': math.log10, 'log1p': math.log1p,
              'sin': math.sin, 'cos': math.cos, 'tan': math.tan, 'sinh': math.sinh, 'cosh': math.cosh,
              'tanh': math.tanh, 'asin': math.asin, 'acos': math.acos, 'atan': math.atan,
              'asinh': math.asinh, 'acosh': math.acosh, 'atanh': math.atanh, 'erf': math.erf,
              'erfc': math.erfc, 'cbrt': lambda v: math.copysign(abs(v) ** (1 / 3), v),
              'sigmoid': lambda v: 1 / (1 + math.exp(-v)), 'sinpi': lambda v: math.sin(math.pi * v),
              'cospi': lambda v: math.cos(math.pi * v), 'tgamma': math.gamma, 'lgamma': math.lgamma,
              'rsqrt_approx': lambda v: 1 / math.sqrt(v), 'sqrt_approx': math.sqrt,
              'div_approx': None, 'atan2': None, 'hypot': None, 'pow': None}
    if sem in approx:
        if sem == 'div_approx':
            return x / y
        if sem == 'atan2':
            return math.atan2(x, y)
        if sem == 'hypot':
            return math.hypot(x, y)
        if sem == 'pow':
            return math.pow(x, y)
        return approx[sem](x)
    raise ValueError('unknown math semantics ' + sem)


def _slice_tensor(values, dt, device='cuda'):
    """A tensor of Python values (exact in dt); unsigned 16/32/64-bit types
    use torch's unsigned dtypes, fp8 goes through float64."""
    import torch
    names = {'f16': torch.float16, 'bf16': torch.bfloat16, 'f32': torch.float32, 'f64': torch.float64,
             'f8e4': torch.float8_e4m3fn, 'f8e5': torch.float8_e5m2, 'i8': torch.int8, 'i16': torch.int16,
             'i32': torch.int32, 'i64': torch.int64, 'u8': torch.uint8, 'u16': torch.uint16,
             'u32': torch.uint32, 'u64': torch.uint64}
    if dt[0] in 'iu':
        if dt == 'u64':
            signed = [_slice_wrap(v, 'i64') for v in values]
            return torch.tensor(signed, dtype=torch.int64).view(torch.uint64).to(device)
        if dt[0] == 'i':
            values = [_slice_wrap(v, dt) for v in values]
        return torch.tensor(values, dtype=torch.int64).to(names[dt]).to(device)
    return torch.tensor(values, dtype=torch.float64).to(names[dt]).to(device)


def _slice_values_of(tensor, dt):
    """Python values of a result tensor of type dt (unsigned types read
    unsigned, whatever torch dtype holds their bits)."""
    import torch
    t = tensor.detach().cpu().contiguous().view(-1)
    if dt[0] in 'iu':
        bits = int(dt[1:])
        signed = t.view({8: torch.int8, 16: torch.int16, 32: torch.int32, 64: torch.int64}[bits]) \
            if t.element_size() * 8 == bits else t.to(torch.int64)
        values = signed.to(torch.int64).tolist()
        return [v & ((1 << bits) - 1) for v in values] if dt[0] == 'u' else values
    return t.double().tolist()


def _slice_mcheck(got, want, label, fmt, ulps, signed_zero):
    """Compare result values with the reference: exactly (NaN matches NaN,
    the sign of zero when signed_zero), or within ulps units in the last
    place of fmt for approximations; None in want is not checked."""
    import math
    bad = []
    for i, (g, w) in enumerate(zip(got, want)):
        if w is None:
            continue
        if isinstance(w, float) and math.isnan(w):
            ok = isinstance(g, float) and math.isnan(g)
        elif isinstance(g, float) and math.isnan(g):
            ok = False
        elif ulps and fmt in _SLICE_FORMATS and math.isfinite(w):
            mant, emin = _SLICE_FORMATS[fmt][:2]
            # the float64 reference of an approximation carries its own
            # error (sin(pi * x) near an integer), so f64 checks ~40 bits
            mant = min(mant, 40)
            e = math.frexp(abs(w))[1] - 1 if w else emin
            tolerance = ulps * 2.0 ** (max(e, emin) - mant + 1) + ulps * 2.0 ** (-mant - 6)
            ok = abs(g - w) <= tolerance
        else:
            ok = g == w and (not signed_zero or g != 0 or math.copysign(1, g) == math.copysign(1, w))
        if not ok:
            bad.append(i)
    if bad:
        i = bad[0]
        raise RuntimeError(f'WRONG RESULT: {label} differs at {len(bad)}/{len(want)} elements; '
                           f'first index {i}: got {got[i]!r}, expected {want[i]!r}')


_SLICE_REJECTIONS = ('not supported', 'does not support', 'unsupported dtype', 'cannot cast',
                     'expected dtype', 'only supports', 'only support', 'is not supported',
                     'got an unexpected keyword argument', 'not implemented for')
_SLICE_INTERNAL = ('passmanager', 'llvm error', "error: '", 'assert', 'check failed',
                   'internalerror', 'segmentation', 'core dumped', 'tvm_kernels.cu', 'ptxas',
                   'illegal', 'mlir')


def _slice_reject(exc):
    """Mark a clean front-end rejection of an unsupported feature."""
    import sys
    texts, seen = [], exc
    while seen is not None and len(texts) < 4:
        texts.append(f'{type(seen).__name__}: {seen}')
        seen = seen.__cause__ or seen.__context__
    text = '\n'.join(texts).lower()
    if any(p in text for p in _SLICE_REJECTIONS) and not any(p in text for p in _SLICE_INTERNAL):
        reason = next(p for p in _SLICE_REJECTIONS if p in text)
        print(f'TILESMITH_REJECTED={reason}', file=sys.stderr, flush=True)


HELPERS = (_slice_values, _slice_storage, _slice_cast, _slice_step, _slice_output, _slice_bits,
           _slice_compare, _slice_reject, _slice_round, _slice_round_values, _slice_fround, _slice_fsqrt,
           _slice_fnext, _slice_int_range, _slice_wrap, _slice_mvalues, _slice_meval, _slice_tensor,
           _slice_values_of, _slice_mcheck)
SOURCES = '\n\n'.join([f'_SLICE_REJECTIONS = {_SLICE_REJECTIONS!r}', f'_SLICE_INTERNAL = {_SLICE_INTERNAL!r}',
                        f'_SLICE_FORMATS = {_SLICE_FORMATS!r}']
                       + [inspect.getsource(fn) for fn in HELPERS])
