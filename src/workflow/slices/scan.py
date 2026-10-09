"""Scans: built-in, user-combined, tuple and non-commutative combines.

Triton lowers associative_scan per layout with warp shuffles and shared
memory; a user combine function must be associative but not commutative,
so a linear recurrence and a running argmax pair also exercise operand
order and reverse scans. TileLang lowers T.cumsum/T.cummax through shared
memory for fragment sources, in place or into a separate buffer.
"""
from .base import zero, Slice, literal, neutral, triton_offsets
from .cast import tilelang_index, tilelang_launch
from .dtypes import ARITHMETIC, DTYPES, Domain
from .reduce import TRITON_COMBINES, core_dtype, small_int_accumulator, sum_domain, tile_index

TRITON_KINDS = ('cumsum', 'cumprod', 'cmax', 'cmin', 'cadd', 'cpair', 'linrec')
TILELANG_KINDS = ('cumsum', 'cummax')


class ScanSlice(Slice):
    name = 'scan'
    pre_steps = 2
    post_steps = 1

    def core_space(self, backend):
        if backend == 'triton':
            return {'core_dt': ARITHMETIC, 'kind': TRITON_KINDS, 'axis': (0, 1, 2), 'reverse': (0, 1)}
        return {'core_dt': ARITHMETIC, 'kind': TILELANG_KINDS, 'axis': (0, 1), 'reverse': (0, 1),
                'scope': ('fragment', 'shared'), 'dst': ('inplace', 'separate')}

    def legalize_core(self, params, backend, plan, dtype, domain):
        shape = plan['shape']
        core, domain = core_dtype(params, dtype, domain, backend)
        if params['axis'] >= len(shape):
            params['axis'] = len(shape) - 1
        axis, kind = params['axis'], params['kind']
        n = shape[axis]
        if backend == 'tilelang':
            params['stage'] = 'global'
        result = core
        if kind in ('cumsum', 'cadd', 'linrec'):
            acc = small_int_accumulator(core) if (backend == 'triton' and kind == 'cumsum') else core
            out = sum_domain(domain, n)
            if not out.fits(acc) or (DTYPES[acc].kind == 'uint' and out.lo < 0) or (
                    kind == 'linrec' and DTYPES[core].kind == 'uint'):
                kind, acc, out = ('cmax' if backend == 'triton' else 'cummax'), core, domain
            result = acc
            if backend == 'triton' and kind == 'cumsum' and core == 'bf16':
                result = 'f32'
        elif kind == 'cumprod':
            if domain.frac or domain.magnitude > 1:
                kind, out = 'cmax', domain
            else:
                out = Domain(-1 if domain.lo < 0 else 0, 1)
                result = 'f32' if core == 'bf16' else core
        elif kind == 'cpair':
            out, result = Domain(0, n - 1), 'i32'
        else:
            out = domain
            if backend == 'triton' and kind in ('cmax', 'cmin') and core == 'bf16':
                result = 'f32'
        params['kind'] = kind
        plan['core_dt'], plan['kind'], plan['axis'] = core, kind, axis
        plan['out_tile'], plan['out_shape'] = shape, plan['valid']
        return result, out

    def plan_data(self, params, plan):
        data = super().plan_data(params, plan)
        data.update(core_dt=plan['core_dt'], kind=plan['kind'], axis=plan['axis'],
                    reverse=params['reverse'], out_dt=plan['out_dt'])
        return data

    def triton_kernel(self, params, plan):
        shape, valid, axis, kind = plan['shape'], plan['valid'], plan['axis'], plan['kind']
        core = DTYPES[plan['core_dt']].triton
        offs, mask, _ = triton_offsets(shape, valid, params['dynamic'])
        args = ', '.join(f'n{i}' for i in range(len(shape)))
        rev = bool(params['reverse'])
        body = [f'    offs = {offs}', f'    mask = {mask}',
                f'    v = tl.load(X + offs, mask=mask, other={zero(plan["in_dt"])})',
                f'    w = tl.load(Y + offs, mask=mask, other={zero(plan["in_dt"])})']
        body += ['    ' + line for line in self.triton_chain('v', plan['pre'], shape, 'w')]
        body.append(f'    v = v.to({core})')
        fills = {'cumsum': 'sum', 'cadd': 'sum', 'linrec': 'sum', 'cumprod': 'prod', 'cmax': 'max',
                 'cpair': 'max', 'cmin': 'min'}
        fill = literal(neutral(fills[kind], plan['core_dt']))
        body.append(f'    v = tl.where(mask, v, tl.full({shape!r}, {fill}, {core}))')
        if kind in ('cumsum', 'cumprod'):
            body.append(f'    r = tl.{kind}(v, axis={axis}, reverse={rev})')
        elif kind in ('cmax', 'cmin', 'cadd'):
            fn = {'cmax': '_comb_max', 'cmin': '_comb_min', 'cadd': '_comb_add'}[kind]
            body.append(f'    r = tl.associative_scan(v, {axis}, {fn}, reverse={rev})')
        elif kind == 'cpair':
            body += [f'    idx = {tile_index(shape, axis)}',
                     f'    _, r = tl.associative_scan((v, idx), {axis}, _comb_pair, reverse={rev})']
        else:
            body += [f'    a = tl.where(mask & (v > 0), tl.full({shape!r}, 1, {core}), tl.full({shape!r}, 0, {core}))',
                     f'    a = tl.where(mask, a, tl.full({shape!r}, 1, {core}))',
                     f'    _, r = tl.associative_scan((a, v), {axis}, _comb_lin, reverse={rev})']
        body += ['    ' + line for line in self.triton_chain('r', plan['post'], shape, None)]
        body.append(f'    tl.store(OUT + offs, r.to({DTYPES[plan["out_dt"]].triton}), mask=mask)')
        extents = ', '.join(str(v) for v in valid)
        return '\n'.join([TRITON_COMBINES, '', '@triton.jit', f'def kernel(X, Y, OUT, {args}):'] + body + [
            '', 'def launch(X, Y, out, options):',
            f'    kernel[(1,)](X, Y, out, {extents}, **options)'])

    def tilelang_kernel(self, params, plan):
        shape, valid, axis, kind = plan['shape'], plan['valid'], plan['axis'], plan['kind']
        core = DTYPES[plan['core_dt']].tilelang
        in_t, out_t = DTYPES[plan['in_dt']].tilelang, DTYPES[plan['out_dt']].tilelang
        idx = tilelang_index(len(shape))
        at = '[' + ', '.join(idx) + ']'
        guard = ' and '.join(f'{i} < {v}' for i, v in zip(idx, valid))
        chain = self.tilelang_chain(f'X{at}', plan['pre'], f'Y{at}')
        fill = neutral('sum' if kind == 'cumsum' else 'max', plan['core_dt'])
        fill_expr = (f'T.cast({literal(fill)}, "{core}")' if not isinstance(fill, float)
                     else '-T.infinity("%s")' % core)
        alloc = 'T.alloc_fragment' if params['scope'] == 'fragment' else 'T.alloc_shared'
        loop = f'for {", ".join(idx)} in T.Parallel({", ".join(map(str, shape))}):'
        lines = [f'xf = {alloc}({shape!r}, "{core}")', loop,
                 f'    xf{at} = T.if_then_else({guard}, T.cast({chain}, "{core}"), {fill_expr})']
        rev = bool(params['reverse'])
        if params['dst'] == 'separate':
            lines += [f'rf = {alloc}({shape!r}, "{core}")', f'T.{kind}(xf, rf, dim={axis}, reverse={rev})']
            result = 'rf'
        else:
            lines.append(f'T.{kind}(xf, dim={axis}, reverse={rev})')
            result = 'xf'
        post = self.tilelang_chain(f'{result}{at}', plan['post'], None)
        lines += [loop, f'    if {guard}:', f'        O{at} = T.cast({post}, "{out_t}")']
        signature = (f'X: T.Tensor({valid!r}, "{in_t}"), Y: T.Tensor({valid!r}, "{in_t}"), '
                     f'O: T.Tensor({valid!r}, "{out_t}")')
        return tilelang_launch(signature, lines)

    def reference(self, params, plan):
        return '\n'.join(['def reference(x, y):', '    import torch', '    v = x']
                         + self.reference_chain('v', 'pre') + [
            "    v = _slice_cast(v, PLAN['core_dt'])",
            "    axis, kind, reverse = PLAN['axis'], PLAN['kind'], PLAN['reverse']",
            '    s = v.flip(axis) if reverse else v',
            '    s = s.movedim(axis, -1)',
            '    outs = []',
            '    for t in range(s.shape[-1]):',
            '        x_t = s[..., t]',
            '        if t == 0:',
            "            acc = x_t.clone()",
            "            best = torch.zeros_like(x_t)",
            "            h = x_t.clone()",
            "        elif kind in ('cumsum', 'cadd'):",
            '            acc = acc + x_t',
            "        elif kind == 'cumprod':",
            '            acc = acc * x_t',
            "        elif kind in ('cmax', 'cummax'):",
            '            acc = torch.maximum(acc, x_t)',
            "        elif kind == 'cmin':",
            '            acc = torch.minimum(acc, x_t)',
            "        elif kind == 'cpair':",
            '            index = float(s.shape[-1] - 1 - t) if reverse else float(t)',
            '            take = (x_t > acc) | ((x_t == acc) & (index < best))',
            '            best = torch.where(take, torch.full_like(best, index), best)',
            '            acc = torch.where(take, x_t, acc)',
            "        elif kind == 'linrec':",
            '            h = torch.where(x_t > 0, h, torch.zeros_like(h)) + x_t',
            "        if kind == 'cpair':",
            '            if t == 0:',
            '                best = torch.full_like(x_t, float(s.shape[-1] - 1) if reverse else 0.0)',
            '            outs.append(best.clone())',
            "        elif kind == 'linrec':",
            '            outs.append(h.clone())',
            '        else:',
            '            outs.append(acc.clone())',
            '    r = torch.stack(outs, -1).movedim(-1, axis)',
            '    r = r.flip(axis) if reverse else r',
        ] + self.reference_chain('r', 'post', 'None') + [
            "    return _slice_cast(r, PLAN['out_dt'])"])
