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
           _slice_compare, _slice_reject, _slice_round, _slice_round_values)
SOURCES = '\n\n'.join([f'_SLICE_REJECTIONS = {_SLICE_REJECTIONS!r}', f'_SLICE_INTERNAL = {_SLICE_INTERNAL!r}',
                        f'_SLICE_FORMATS = {_SLICE_FORMATS!r}']
                       + [inspect.getsource(fn) for fn in HELPERS])
