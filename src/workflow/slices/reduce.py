"""Reductions: built-in, user-combined and multi-operand, per dtype and axis.

Operator implementation is the largest bug class of the tile-bug study
(27%); reductions are synthesized per layout, axis and dtype. Triton adds
user-defined combine functions and tuple reductions; TileLang adds the
clear/batch/nan-propagation options and shared-memory sources.
"""
import math

from .base import zero, Slice, literal, neutral, triton_offsets
from .cast import tilelang_index, tilelang_launch
from .chain import cast_allowed, cast_domain
from .dtypes import ARITHMETIC, DTYPES, Domain, bit_bound

TRITON_KINDS = ('sum', 'max', 'min', 'argmax', 'argmin', 'xor_sum', 'cadd', 'cmax', 'cpair', 'cminmax')
TILELANG_KINDS = ('sum', 'max', 'min', 'abssum', 'absmax', 'bitand', 'bitor', 'bitxor')
NEUTRAL_KIND = {'sum': 'sum', 'cadd': 'sum', 'xor_sum': 'xor', 'max': 'max', 'cmax': 'max', 'argmax': 'max',
                'cpair': 'max', 'min': 'min', 'argmin': 'min', 'abssum': 'sum', 'absmax': 'sum',
                'bitand': 'and', 'bitor': 'or', 'bitxor': 'xor'}

TRITON_COMBINES = '''@triton.jit
def _comb_add(a, b):
    return a + b


@triton.jit
def _comb_max(a, b):
    return tl.maximum(a, b)


@triton.jit
def _comb_min(a, b):
    return tl.minimum(a, b)


@triton.jit
def _comb_pair(v1, i1, v2, i2):
    take = (v1 > v2) | ((v1 == v2) & (i1 < i2))
    return tl.where(take, v1, v2), tl.where(take, i1, i2)


@triton.jit
def _comb_minmax(a1, b1, a2, b2):
    return tl.minimum(a1, a2), tl.maximum(b1, b2)


@triton.jit
def _comb_lin(a1, b1, a2, b2):
    return a1 * a2, b1 * a2 + b2
'''


def sum_domain(domain, n):
    return Domain(n * min(domain.lo, 0), n * max(domain.hi, 0), domain.frac)


def small_int_accumulator(dtype):
    """Triton's tl.sum/cumsum widen sub-32-bit integers to 32 bits."""
    d = DTYPES[dtype]
    if d.is_float or d.bits >= 32:
        return dtype
    return 'i32' if d.kind == 'int' else 'u32'


def core_dtype(params, dtype, domain, backend):
    """Legalize params['core_dt'] for the conversion from dtype."""
    core = params['core_dt']
    converted = cast_domain(domain, core)
    if not (cast_allowed(dtype, core, backend) and converted.fits(core)
            and (DTYPES[core].kind != 'uint' or domain.lo >= 0)):
        core = next(c for c in (dtype, 'f32', 'f64') if c in ARITHMETIC and cast_allowed(dtype, c, backend)
                    and cast_domain(domain, c).fits(c))
        params['core_dt'] = core
    return core, cast_domain(domain, core)


def tile_index(shape, axis):
    """Index of `axis` broadcast over a rank-1..3 tile, for tuple reductions."""
    rank = len(shape)
    if rank == 1:
        return f'tl.arange(0, {shape[0]})'
    index = ['None'] * rank
    index[axis] = ':'
    return f'tl.broadcast_to(tl.arange(0, {shape[axis]})[{", ".join(index)}], {shape!r})'


class ReduceSlice(Slice):
    name = 'reduce'
    pre_steps = 2
    post_steps = 1

    def core_space(self, backend):
        if backend == 'triton':
            return {'core_dt': ARITHMETIC, 'kind': TRITON_KINDS, 'axis': (0, 1, 2), 'keep': (0, 1)}
        return {'core_dt': ARITHMETIC, 'kind': TILELANG_KINDS, 'axis': (0, 1), 'clear': (0, 1),
                'scope': ('fragment', 'shared'), 'batch': (1, 2, 4), 'nanprop': (0, 1)}

    def legalize_core(self, params, backend, plan, dtype, domain):
        shape = plan['shape']
        rank = len(shape)
        core, domain = core_dtype(params, dtype, domain, backend)
        if params['axis'] >= rank:
            params['axis'] = rank - 1
        axis = params['axis']
        kind = params['kind']
        d = DTYPES[core]
        n = shape[axis]
        if backend == 'tilelang':
            params['stage'] = 'global'
            if kind.startswith('bit') and d.is_float:
                kind = 'sum'
            if kind == 'abssum':
                params['clear'] = 1
            if not (kind in ('max', 'min', 'absmax') and core in ('f16', 'bf16')):
                params['nanprop'] = 0
            if rank == 1:
                params['axis'] = axis = 0
            # A batched AllReduce needs `batch` outputs per thread.
            outputs = 1 if rank == 1 else shape[1 - axis]
            if outputs < params['batch'] * max(params['threads'], params['threads2']):
                params['batch'] = 1
        else:
            if kind == 'xor_sum' and d.is_float:
                kind = 'sum'
        result = core
        init = Domain(1, 1)
        clear = params.get('clear', 1)
        if kind in ('sum', 'cadd', 'abssum'):
            acc = small_int_accumulator(core) if (backend == 'triton' and kind == 'sum') else core
            base = Domain(0, domain.magnitude, domain.frac) if kind == 'abssum' else domain
            out = sum_domain(base, n)
            if not clear:
                out = Domain(out.lo + 1, out.hi + 1, out.frac)
            if not out.fits(acc) or (DTYPES[acc].kind == 'uint' and out.lo < 0):
                kind, acc, out = 'max', core, domain if clear else domain.join(init)
            result = acc if kind != 'max' else core
        elif kind in ('argmax', 'argmin', 'cpair'):
            out, result = Domain(0, n - 1), 'i32'
        elif kind in ('xor_sum', 'bitand', 'bitor', 'bitxor'):
            out = bit_bound(domain.join(init) if not clear else domain)
            if not out.fits(core):
                kind, out = 'max', domain if clear else domain.join(init)
        elif kind == 'cminmax':
            out = Domain(0, domain.hi - domain.lo, domain.frac)
            if not out.fits(core):
                kind, out = 'max', domain
        elif kind == 'absmax':
            out = Domain(0, domain.magnitude, domain.frac)
            if not clear:
                out = out.join(init)
        else:
            out = domain if clear else domain.join(init)
        if backend == 'triton' and kind in ('max', 'min') and DTYPES[core].bits < 32:
            # tl.max/tl.min widen narrow inputs before reducing: floats to
            # float32 and integers, unsigned ones too, to int32.
            result = 'f32' if DTYPES[core].is_float else 'i32'
        if backend == 'triton' and kind == 'cmax' and core == 'bf16':
            result = 'f32'
        params['kind'] = kind
        plan['core_dt'], plan['kind'], plan['axis'] = core, kind, axis
        keep = params.get('keep', 0)
        out_tile = list(shape)
        out_valid = list(plan['valid'])
        if keep:
            out_tile[axis] = out_valid[axis] = 1
        else:
            del out_tile[axis], out_valid[axis]
        if backend == 'tilelang' and not out_tile:
            out_tile, out_valid = [1], [1]
        plan['out_tile'], plan['out_shape'] = tuple(out_tile), tuple(out_valid)
        return result, out

    def plan_data(self, params, plan):
        data = super().plan_data(params, plan)
        data.update(core_dt=plan['core_dt'], kind=plan['kind'], axis=plan['axis'],
                    keep=params.get('keep', 0), clear=params.get('clear', 1), out_dt=plan['out_dt'],
                    rank1_out=plan['backend'] == 'triton' and len(plan['shape']) == 1 and not params['keep'])
        return data

    # ---- Triton --------------------------------------------------------------
    def triton_kernel(self, params, plan):
        shape, valid, axis, kind = plan['shape'], plan['valid'], plan['axis'], plan['kind']
        core = DTYPES[plan['core_dt']].triton
        offs, mask, _ = triton_offsets(shape, valid, params['dynamic'])
        args = ', '.join(f'n{i}' for i in range(len(shape)))
        keep = bool(params['keep'])
        body = [f'    offs = {offs}', f'    mask = {mask}',
                f'    v = tl.load(X + offs, mask=mask, other={zero(plan["in_dt"])})',
                f'    w = tl.load(Y + offs, mask=mask, other={zero(plan["in_dt"])})']
        body += ['    ' + line for line in self.triton_chain('v', plan['pre'], shape, 'w')]
        body.append(f'    v = v.to({core})')
        fill = literal(neutral(NEUTRAL_KIND.get(kind, 'max'), plan['core_dt']))
        if kind == 'cminmax':
            low = literal(neutral('min', plan['core_dt']))
            body += [f'    a = tl.where(mask, v, tl.full({shape!r}, {low}, {core}))',
                     f'    b = tl.where(mask, v, tl.full({shape!r}, {fill}, {core}))',
                     f'    mn, mx = tl.reduce((a, b), {axis}, _comb_minmax, keep_dims={keep})',
                     '    r = mx - mn']
        else:
            body.append(f'    v = tl.where(mask, v, tl.full({shape!r}, {fill}, {core}))')
            if kind in ('sum', 'max', 'min', 'xor_sum'):
                body.append(f'    r = tl.{kind}(v, axis={axis}, keep_dims={keep})')
            elif kind in ('argmax', 'argmin'):
                body.append(f'    r = tl.{kind}(v, axis={axis}, keep_dims={keep})')
            elif kind in ('cadd', 'cmax'):
                fn = '_comb_add' if kind == 'cadd' else '_comb_max'
                body.append(f'    r = tl.reduce(v, {axis}, {fn}, keep_dims={keep})')
            else:
                body += [f'    idx = {tile_index(shape, axis)}',
                         f'    _, r = tl.reduce((v, idx), {axis}, _comb_pair, keep_dims={keep})']
        out_tile, out_valid = plan['out_tile'], plan['out_shape']
        body += ['    ' + line for line in self.triton_chain('r', plan['post'], out_tile, None)]
        out_t = DTYPES[plan['out_dt']].triton
        if not out_tile:
            body.append(f'    tl.store(OUT, r.to({out_t}))')
        else:
            out_offs, out_mask, _ = triton_offsets(out_tile, out_valid, False)
            body.append(f'    tl.store(OUT + {out_offs}, r.to({out_t}), mask={out_mask})')
        extents = ', '.join(str(v) for v in valid)
        return '\n'.join([TRITON_COMBINES, '', '@triton.jit', f'def kernel(X, Y, OUT, {args}):'] + body + [
            '', 'def launch(X, Y, out, options):',
            f'    kernel[(1,)](X, Y, out, {extents}, **options)'])

    # ---- TileLang ------------------------------------------------------------
    def tilelang_kernel(self, params, plan):
        shape, valid, axis, kind = plan['shape'], plan['valid'], plan['axis'], plan['kind']
        core = DTYPES[plan['core_dt']].tilelang
        in_t, out_t = DTYPES[plan['in_dt']].tilelang, DTYPES[plan['out_dt']].tilelang
        idx = tilelang_index(len(shape))
        at = '[' + ', '.join(idx) + ']'
        guard = ' and '.join(f'{i} < {v}' for i, v in zip(idx, valid))
        chain = self.tilelang_chain(f'X{at}', plan['pre'], f'Y{at}')
        fill = neutral(NEUTRAL_KIND[kind], plan['core_dt'])
        fill_expr = (f'T.cast({literal(fill)}, "{core}")' if not (isinstance(fill, float) and math.isinf(fill))
                     else f'{"-" if fill < 0 else ""}T.infinity("{core}")')
        alloc = 'T.alloc_fragment' if params['scope'] == 'fragment' else 'T.alloc_shared'
        out_tile, out_valid = plan['out_tile'], plan['out_shape']
        lines = [f'xf = {alloc}({shape!r}, "{core}")', f'of = T.alloc_fragment({out_tile!r}, "{core}")',
                 f'for {", ".join(idx)} in T.Parallel({", ".join(map(str, shape))}):',
                 f'    xf{at} = T.if_then_else({guard}, T.cast({chain}, "{core}"), {fill_expr})']
        if not params['clear']:
            lines += [f'for k in T.Parallel({out_tile[0]}):', f'    of[k] = T.cast(1, "{core}")']
        options = [f'dim={axis}']
        if kind != 'abssum':
            options.append(f'clear={bool(params["clear"])}')
        if params['batch'] != 1:
            options.append(f'batch={params["batch"]}')
        if params['nanprop']:
            options.append('nan_propagate=True')
        lines.append(f'T.reduce_{kind}(xf, of, {", ".join(options)})')
        post = self.tilelang_chain('of[k]', plan['post'], None)
        lines += [f'for k in T.Parallel({out_tile[0]}):', f'    if k < {out_valid[0]}:',
                  f'        O[k] = T.cast({post}, "{out_t}")']
        signature = (f'X: T.Tensor({valid!r}, "{in_t}"), Y: T.Tensor({valid!r}, "{in_t}"), '
                     f'O: T.Tensor({out_valid!r}, "{out_t}")')
        return tilelang_launch(signature, lines)

    # ---- reference -----------------------------------------------------------
    def reference(self, params, plan):
        return '\n'.join(['def reference(x, y):', '    import torch', '    v = x']
                         + self.reference_chain('v', 'pre') + [
            "    v = _slice_cast(v, PLAN['core_dt'])",
            "    axis, kind = PLAN['axis'], PLAN['kind']",
            "    if kind in ('sum', 'cadd'):",
            '        r = v.sum(axis)',
            "    elif kind == 'abssum':",
            '        r = v.abs().sum(axis)',
            "    elif kind in ('max', 'cmax'):",
            '        r = v.amax(axis)',
            "    elif kind == 'min':",
            '        r = v.amin(axis)',
            "    elif kind == 'absmax':",
            '        r = v.abs().amax(axis)',
            "    elif kind in ('argmax', 'cpair'):",
            '        r = v.argmax(axis).double()',
            "    elif kind == 'argmin':",
            '        r = v.argmin(axis).double()',
            "    elif kind == 'cminmax':",
            '        r = v.amax(axis) - v.amin(axis)',
            '    else:',
            '        ints = v.to(torch.int64).movedim(axis, 0)',
            '        r = ints[0].clone()',
            '        for row in ints[1:]:',
            "            r = r & row if kind == 'bitand' else r | row if kind == 'bitor' else r ^ row",
            '        r = r.double()',
            "    if not PLAN['clear']:",
            "        if kind in ('sum', 'abssum'):",
            '            r = r + 1',
            "        elif kind in ('max', 'absmax'):",
            '            r = torch.clamp(r, min=1.0)',
            "        elif kind == 'min':",
            '            r = torch.clamp(r, max=1.0)',
            "        elif kind == 'bitand':",
            '            r = (r.to(torch.int64) & 1).double()',
            "        elif kind == 'bitor':",
            '            r = (r.to(torch.int64) | 1).double()',
            "        elif kind == 'bitxor':",
            '            r = (r.to(torch.int64) ^ 1).double()',
            "    if PLAN['keep']:",
            '        r = r.unsqueeze(axis)',
            "    if r.dim() == 0 and not PLAN['rank1_out']:",
            '        r = r.reshape(1)',
        ] + self.reference_chain('r', 'post', 'None') + [
            "    return _slice_cast(r, PLAN['out_dt'])"])
