"""Conversion and elementwise chains across the full dtype lattice.

Bug study (ISSTA'26): data-type semantics are 19% of tile codegen bugs and
both DSLs specialize conversions per (source, target, vector width). The
focus here is the chain itself: three steps, each an operation in the
current dtype followed by a conversion, over every storage and compute dtype
the installed DSLs expose (bf16, fp8, fp64, 16/64-bit and unsigned ints).
"""
from .base import zero, Slice, triton_offsets
from .dtypes import DTYPES


class CastSlice(Slice):
    name = 'cast'
    pre_steps = 3
    post_steps = 0

    def triton_kernel(self, params, plan):
        shape, valid = plan['shape'], plan['valid']
        offs, mask, names = triton_offsets(shape, valid, params['dynamic'])
        args = ', '.join(f'n{i}' for i in range(len(shape)))
        body = [f'    offs = {offs}', f'    mask = {mask}',
                f'    v = tl.load(X + offs, mask=mask, other={zero(plan["in_dt"])})',
                f'    w = tl.load(Y + offs, mask=mask, other={zero(plan["in_dt"])})']
        body += ['    ' + line for line in self.triton_chain('v', plan['pre'], shape, 'w')]
        body.append(f'    tl.store(OUT + offs, v.to({DTYPES[plan["out_dt"]].triton}), mask=mask)')
        extents = ', '.join(str(v) for v in valid)
        return '\n'.join(['@triton.jit', f'def kernel(X, Y, OUT, {args}):'] + body + [
            '', 'def launch(X, Y, out, options):',
            f'    kernel[(1,)](X, Y, out, {extents}, **options)'])

    def tilelang_kernel(self, params, plan):
        return tilelang_elementwise(self, params, plan, lambda src, operand: self.tilelang_chain(src, plan['pre'], operand))

    def reference(self, params, plan):
        return '\n'.join(['def reference(x, y):', '    v = x'] + self.reference_chain('v', 'pre')
                         + ["    return _slice_cast(v, PLAN['out_dt'])"])

    def plan_data(self, params, plan):
        data = super().plan_data(params, plan)
        data['out_dt'] = plan['out_dt']
        return data


def tilelang_index(rank):
    return ['i'] if rank == 1 else ['i', 'j'] if rank == 2 else ['i', 'j', 'k']


def tilelang_elementwise(slice_, params, plan, chain):
    """A one-block TileLang kernel computing chain(X, Y) elementwise."""
    shape, valid = plan['shape'], plan['valid']
    rank = len(shape)
    idx = tilelang_index(rank)
    at = '[' + ', '.join(idx) + ']'
    origin = '[' + ', '.join('0' for _ in shape) + ']'
    in_t, out_t = DTYPES[plan['in_dt']].tilelang, DTYPES[plan['out_dt']].tilelang
    guard = ' and '.join(f'{i} < {v}' for i, v in zip(idx, valid))
    stage = params['stage']
    lines = []
    if stage == 'global':
        src = f'T.if_then_else({guard}, X{at}, T.cast(0, "{in_t}"))'
        operand = f'T.if_then_else({guard}, Y{at}, T.cast(0, "{in_t}"))'
    else:
        alloc = 'T.alloc_fragment' if stage == 'fragment' else 'T.alloc_shared'
        lines += [f'xs = {alloc}({shape!r}, "{in_t}")', f'ys = {alloc}({shape!r}, "{in_t}")',
                  f'T.copy(X{origin}, xs)', f'T.copy(Y{origin}, ys)']
        src, operand = f'xs{at}', f'ys{at}'
    expr = f'T.cast({chain(src, operand)}, "{out_t}")'
    loop = f'for {", ".join(idx)} in T.Parallel({", ".join(map(str, shape))}):'
    if stage == 'global':
        lines += [loop, f'    if {guard}:', f'        O{at} = {expr}']
    else:
        lines += [f'of = T.alloc_fragment({shape!r}, "{out_t}")', loop, f'    of{at} = {expr}',
                  f'T.copy(of, O{origin})']
    return tilelang_launch(f'X: T.Tensor({valid!r}, "{in_t}"), Y: T.Tensor({valid!r}, "{in_t}"), '
                           f'O: T.Tensor({tuple(plan["out_shape"]) if "out_shape" in plan else valid!r}, "{out_t}")',
                           lines)


def tilelang_launch(signature, lines, grid='1'):
    body = '\n'.join('                ' + line for line in lines)
    return f'''def build(threads):
    @tilelang.jit
    def kern():
        @T.prim_func
        def main({signature}):
            with T.Kernel({grid}, threads=threads) as bx:
{body}
        return main
    return kern()


_KERNELS = {{}}


def launch(X, Y, out, options):
    threads = options['threads']
    if threads not in _KERNELS:
        _KERNELS[threads] = build(threads)
    _KERNELS[threads](X, Y, out)'''
