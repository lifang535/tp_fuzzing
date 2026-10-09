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
           _slice_compare, _slice_reject)
SOURCES = '\n\n'.join([f'_SLICE_REJECTIONS = {_SLICE_REJECTIONS!r}', f'_SLICE_INTERNAL = {_SLICE_INTERNAL!r}']
                      + [inspect.getsource(fn) for fn in HELPERS])
