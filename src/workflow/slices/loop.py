"""Loops that carry tile state across iterations.

Software pipelining, unrolling, flattening, warp specialization and load
scheduling rewrite loops whose tensor state is carried across iterations;
their bugs show as a wrong accumulator, a race on a multi-buffered operand
or a stale value after the loop. Triton loops here are Python range,
tl.range (num_stages, loop_unroll_factor, flatten,
disallow_acc_multi_buffer, warp_specialize), tl.static_range and while
loops, iterating forward, backward or by two, addressing their tiles by
index, by a carried pointer or by an advanced block pointer, with nested
inner loops, conditionals, scans, two carried values and dot accumulation.
TileLang loops are T.Pipelined (num_stages, order, stage, sync, group),
T.serial, T.unroll, T.Unroll(explicit), while loops over T.alloc_var and
early exits with T.loop_break, with copies into shared memory or fragments
and fragment updates, reductions or T.gemm in the body. Values are small
integers whose growth legalization bounds; the reference runs the loop in
torch.
"""
import math

from .base import Slice, parse_shape
from .dtypes import DTYPES, Domain, bit_bound
from .gemm import warp_partition
from .layout import nd_offsets

LOOP_DT = ('f32', 'f16', 'bf16', 'f64', 'i8', 'i16', 'i32', 'i64', 'u8')
TRITON_LOOP_SHAPES = ('64', '128', '256', '1024', '16x16', '32x32', '16x64', '64x16')
TILELANG_LOOP_SHAPES = ('16x32', '32x32', '32x64', '64x32', '64x64', '16x128')
TRITON_BODIES = ('axpy', 'max', 'xor', 'dot', 'cond', 'ifelse', 'cumsum', 'count')
TILELANG_BODIES = ('axpy', 'max', 'gemm', 'cond', 'rsum')


def exact_range(dt):
    d = DTYPES[dt]
    if d.is_float:
        return -2 ** d.mantissa, 2 ** d.mantissa
    return d.minval, d.maxval


def fits(lo, hi, dt):
    a, b = exact_range(dt)
    return a <= lo and hi <= b


def envelope(body, coef, trips, value, k, size, init):
    """Interval of every value the accumulator (and its scaled form) takes."""
    lo = hi = init
    worst_lo, worst_hi = lo, hi
    vlo, vhi = value
    for _ in range(trips):
        if body == 'axpy':
            scaled = sorted((coef * lo, coef * hi))
            worst_lo, worst_hi = min(worst_lo, scaled[0]), max(worst_hi, scaled[1])
            lo, hi = scaled[0] + vlo, scaled[1] + vhi
        elif body == 'max':
            lo, hi = max(lo, vlo), max(hi, vhi)
        elif body == 'xor':
            b = bit_bound(Domain(min(lo, vlo), max(hi, vhi)))
            lo, hi = b.lo, b.hi
        elif body in ('dot', 'gemm'):
            lo, hi = lo - 4 * k, hi + 4 * k
        elif body in ('cond', 'ifelse'):
            lo, hi = min(lo + vlo, lo - vhi), max(hi + vhi, hi - vlo)
        elif body == 'cumsum':
            lo, hi = lo + size * min(vlo, 0), hi + size * max(vhi, 0)
        elif body in ('count', 'rsum'):
            factor = size if body == 'rsum' else 1
            lo, hi = lo + factor * min(vlo, 0), hi + factor * max(vhi, 0)
        worst_lo, worst_hi = min(worst_lo, lo), max(worst_hi, hi)
    return worst_lo, worst_hi


class LoopSlice(Slice):
    name = 'loop'
    SIMPLEST = {'coef': 1, 'in_dt': 'f32', 'acc_dt': 'f32', 'k': 16,
                'loop': {'triton': 'range', 'tilelang': 'serial'}, 'dyn': 0, 'order': 'fwd',
                'accform': 'arg', 'stages': 0, 'unroll': 0, 'flatten': 0, 'disallow': 0, 'ws': 0,
                'body': 'axpy', 'inner': 0, 'addr': 'index', 'mask': 'none', 'sched': 'none',
                'inputs': 1, 'stage_in': 'shared'}

    def space(self, backend):
        space = {'trips': (1, 2, 3, 4, 5, 8), 'coef': (1, 2, -1), 'in_dt': LOOP_DT, 'acc_dt': LOOP_DT,
                 'k': (16, 32, 64), 'pair': (0, 1)}
        if backend == 'triton':
            space.update(loop=('range', 'tl_range', 'static', 'while'), dyn=(0, 1), order=('fwd', 'rev', 'step2'),
                         accform=('arg', 'add'),
                         stages=(0, 1, 2, 3, 4), unroll=(0, 2, 4), flatten=(0, 1), disallow=(0, 1), ws=(0, 1),
                         body=TRITON_BODIES, inner=(0, 2, 3), addr=('index', 'ptr', 'bptr'),
                         mask=('none', 'ragged'), shape=TRITON_LOOP_SHAPES, warps=(1, 2, 4, 8),
                         warps2=(1, 2, 4, 8))
        else:
            space.update(loop=('pipelined', 'serial', 'unroll', 'unroll_explicit', 'while', 'break'),
                         stages=(0, 1, 2, 3), sched=('none', 'order', 'reorder', 'sync', 'group'),
                         body=TILELANG_BODIES, inputs=(1, 2), stage_in=('shared', 'fragment'),
                         shape=TILELANG_LOOP_SHAPES, threads=(32, 64, 128, 256), threads2=(32, 64, 128, 256))
            space['trips'] = (1, 2, 3, 4, 6)
        return space

    # ---- legalization -------------------------------------------------------
    def legalize(self, params, backend):
        space = self.space(backend)
        for knob, values in space.items():
            if params.get(knob) not in values:
                params[knob] = values[0]
        for knob in list(params):
            if knob not in space:
                del params[knob]
        shape = parse_shape(params['shape'])
        body = params['body']
        in_dt, acc_dt = params['in_dt'], params['acc_dt']
        integer = not DTYPES[acc_dt].is_float
        if body in ('dot', 'gemm'):
            if in_dt not in ('f16', 'bf16', 'i8') + (('f32',) if backend == 'triton' else ()):
                params['in_dt'] = in_dt = 'f16'
            params['acc_dt'] = acc_dt = 'i32' if in_dt == 'i8' else 'f32'
            if in_dt == 'i8':
                params['k'] = max(params['k'], 32)
            if len(shape) != 2 or min(shape) < (16 if backend == 'triton' else 32):
                params['shape'] = '32x32'
                shape = (32, 32)
        else:
            params['k'] = 16
            if body == 'xor' and not integer:
                params['body'] = body = 'axpy'
            if body in ('count', 'cumsum') and len(shape) != 1:
                params['body'] = body = 'axpy'
        if DTYPES[in_dt].kind == 'uint' or DTYPES[acc_dt].kind == 'uint':
            # unsigned values stay non-negative: no subtraction, no negation
            if body in ('cond', 'ifelse'):
                params['body'] = body = 'axpy'
            if params['coef'] == -1:
                params['coef'] = 1
            if DTYPES[acc_dt].kind == 'uint' and DTYPES[in_dt].kind != 'uint':
                params['in_dt'] = in_dt = 'u8'
        if body != 'axpy':
            params['coef'] = 1
        if backend == 'triton':
            self.legalize_triton_loop(params, shape)
        else:
            self.legalize_tilelang_loop(params, shape)
        trips = params['trips']
        inner = params.get('inner', 0)
        nonneg = DTYPES[in_dt].kind == 'uint'
        value = (0 if nonneg else -2 * (1 + inner), 2 * (1 + inner))
        size = math.prod(shape) if body == 'cumsum' else shape[-1] if body == 'rsum' else 1
        init = -3 if body == 'max' and not nonneg else 0
        bound = envelope(body, params['coef'], trips, value, params['k'], size, init)
        if not fits(*bound, acc_dt):
            if params['coef'] != 1:
                params['coef'] = 1
                return self.legalize(params, backend)
            if body in ('cumsum', 'rsum', 'count') or not fits(*bound, 'f32' if not integer else 'i32'):
                if params['trips'] > 2:
                    params['trips'] = 2
                    return self.legalize(params, backend)
            params['acc_dt'] = acc_dt = 'i32' if integer and not nonneg else 'i64' if integer else 'f32'
            if not fits(*bound, acc_dt):
                params['acc_dt'] = acc_dt = 'i64' if integer else 'f64'
        if not params['pair']:
            params['warps2' if backend == 'triton' else 'threads2'] = params['warps' if backend == 'triton'
                                                                            else 'threads']
        domain = Domain(0, 2) if nonneg else Domain(-2, 2)
        return {'backend': backend, 'shape': shape, 'in_dt': in_dt, 'acc_dt': acc_dt, 'out_dt': acc_dt,
                'body': body, 'input_domain': domain, 'init': init, 'trips': params['trips'], 'k': params['k'],
                'inner': inner}

    @staticmethod
    def legalize_triton_loop(params, shape):
        loop = params['loop']
        if loop != 'tl_range':
            for knob in ('stages', 'unroll', 'flatten', 'disallow', 'ws'):
                params[knob] = 0
        if loop == 'static':
            params['dyn'] = 0
        if loop == 'while' and params['order'] != 'fwd':
            params['order'] = 'fwd'
        if not params['inner']:
            params['flatten'] = 0
        if params['body'] == 'dot':
            params['mask'] = 'none'
            params['inner'] = 0
            params['flatten'] = 0
        else:
            params['accform'] = 'arg'
        if params['addr'] == 'bptr' and params['mask'] == 'ragged':
            params['mask'] = 'none'
        if params['addr'] != 'index' and params['order'] == 'step2':
            params['order'] = 'fwd'
        if params['body'] == 'dot' and len(shape) == 2:
            # multi-buffered operands (num_stages, unrolled copies) fit in shared memory
            size = DTYPES[params['in_dt']].bits // 8
            tile = (shape[0] + shape[1]) * params['k'] * size

            def staged():
                return tile * max(1, params['stages']) * max(1, params['unroll'])
            while params['stages'] > 1 and staged() > 64 * 1024:
                params['stages'] -= 1
            while params['unroll'] > 2 and staged() > 64 * 1024:
                params['unroll'] //= 2

    @staticmethod
    def legalize_tilelang_loop(params, shape):
        loop = params['loop']
        if loop != 'pipelined':
            params['stages'], params['sched'] = 0, 'none'
        body = params['body']
        if body == 'gemm':
            params['stage_in'] = 'shared'
            params['inputs'] = 1
        if body == 'rsum':
            params['stage_in'] = 'fragment'
            params['inputs'] = 1
        if loop == 'break' and params['trips'] < 2:
            params['trips'] = 2
        if params['sched'] != 'none' and params['stages'] < 2:
            params['stages'] = 2
        r, c = shape
        size = DTYPES[params['in_dt']].bits // 8
        k = params['k']
        tile = (r * k + k * c) if body == 'gemm' else r * c * params['inputs']
        while params['stages'] > 1 and tile * size * params['stages'] > 96 * 1024:
            params['stages'] -= 1
        if params['sched'] != 'none' and params['stages'] < 2:
            params['sched'] = 'none'
        for knob in ('threads', 'threads2'):
            while params[knob] > 32 and ((r * c) % params[knob] or (body == 'rsum' and r % params[knob] and
                                                                     params[knob] > r)
                                         or (body == 'gemm' and not warp_partition(r, c, params[knob] // 32,
                                                                                   'Square'))):
                params[knob] //= 2
            if body == 'gemm' and not warp_partition(r, c, params[knob] // 32, 'Square'):
                params[knob] = 128

    # ---- harness ------------------------------------------------------------
    def input_shapes(self, plan):
        # TileLang checks argument shapes: its tiles are stacked along rows
        shape, trips, k = plan['shape'], plan['trips'], plan['k']
        tilelang = plan['backend'] == 'tilelang'
        if plan['body'] in ('dot', 'gemm'):
            m, n = shape
            if tilelang:
                return (trips * m, k), (trips * k, n)
            return (trips, m, k), (trips, k, n)
        if tilelang:
            return (trips * shape[0], shape[1]), (trips * shape[0], shape[1])
        return (trips,) + shape, (max(1, trips * plan['inner']),) + shape

    def plan_data(self, params, plan):
        return {}

    def order_indices(self, p):
        trips = p['trips']
        order = p.get('order', 'fwd')
        if order == 'rev':
            return list(range(trips - 1, -1, -1))
        return list(range(trips))

    def triton_kernel(self, params, plan):
        p = params
        shape, body = plan['shape'], plan['body']
        acc_t = DTYPES[plan['acc_dt']].triton
        size = math.prod(shape)
        trips = p['trips']
        lines = []
        if body == 'dot':
            m, n = shape
            k = p['k']
            a_offs, b_offs = nd_offsets((m, k)), nd_offsets((k, n))
            a_size, b_size = m * k, k * n
            lines.append(f'    acc = tl.zeros({shape!r}, {acc_t})')
        else:
            offs = nd_offsets(shape)
            lines += [f'    offs = {offs}', f'    acc = tl.full({shape!r}, {plan["init"]}, {acc_t})']
            if body == 'count':
                lines.append(f'    cnt = tl.zeros({shape!r}, tl.int32)')
        loop_var = 't'
        order = p['order']
        if order == 'rev':
            bounds = 'n - 1, -1, -1'
        elif order == 'step2':
            bounds = '0, 2 * n, 2'
        else:
            bounds = '0, n'
        index = 't // 2' if order == 'step2' else 't'
        addr = p['addr']
        if addr == 'ptr':
            start = '(n - 1)' if order == 'rev' else '0'
            if body == 'dot':
                lines += [f'    pa = X + {start} * {a_size} + {a_offs}', f'    pb = Y + {start} * {b_size} + {b_offs}']
            else:
                lines.append(f'    px = X + {start} * {size} + offs')
        elif addr == 'bptr':
            start = 'n - 1' if order == 'rev' else '0'
            rows = shape[0] if len(shape) == 2 else 1
            tile = shape if len(shape) == 2 else (shape[0],)
            if body == 'dot':
                lines += [f'    pa = tl.make_block_ptr(X, shape=(n * {m}, {k}), strides=({k}, 1), '
                          f'offsets=(({start}) * {m}, 0), block_shape=({m}, {k}), order=(1, 0))',
                          f'    pb = tl.make_block_ptr(Y, shape=(n * {k}, {n}), strides=({n}, 1), '
                          f'offsets=(({start}) * {k}, 0), block_shape=({k}, {n}), order=(1, 0))']
            elif len(shape) == 2:
                lines.append(f'    px = tl.make_block_ptr(X, shape=(n * {rows}, {shape[1]}), strides=({shape[1]}, 1), '
                             f'offsets=(({start}) * {rows}, 0), block_shape={tile!r}, order=(1, 0))')
            else:
                lines.append(f'    px = tl.make_block_ptr(X, shape=(n * {size},), strides=(1,), '
                             f'offsets=(({start}) * {size},), block_shape=({size},), order=(0,))')
        step = -1 if order == 'rev' else 1
        loop = p['loop']
        if loop == 'while':
            head = ['    t = 0', '    while t < n:']
        else:
            if loop == 'range':
                it = f'range({bounds})'
            elif loop == 'static':
                it = f'tl.static_range({bounds})'
            else:
                options = []
                for knob, name in (('stages', 'num_stages'), ('unroll', 'loop_unroll_factor')):
                    if p[knob]:
                        options.append(f'{name}={p[knob]}')
                for knob, name in (('flatten', 'flatten'), ('disallow', 'disallow_acc_multi_buffer'),
                                   ('ws', 'warp_specialize')):
                    if p[knob]:
                        options.append(f'{name}=True')
                it = f"tl.range({', '.join([bounds] + options)})"
            head = [f'    for {loop_var} in {it}:']
        lines += head
        inside = []
        if body == 'dot':
            if addr == 'index':
                inside += [f'a = tl.load(X + {index} * {a_size} + {a_offs})',
                           f'b = tl.load(Y + {index} * {b_size} + {b_offs})']
            else:
                inside += ['a = tl.load(pa)', 'b = tl.load(pb)']
            inside.append('acc = tl.dot(a, b, acc)' if p['accform'] == 'arg' else 'acc += tl.dot(a, b)')
            if addr == 'ptr':
                inside += [f'pa += {step * a_size}', f'pb += {step * b_size}']
            elif addr == 'bptr':
                inside += [f'pa = tl.advance(pa, ({step * m}, 0))', f'pb = tl.advance(pb, ({step * k}, 0))']
        else:
            mask = ''
            if p['mask'] == 'ragged':
                inside.append(f'm = offs < {size} - {index} * 3')
                mask = ', mask=m, other=0'
            if addr == 'index':
                inside.append(f'v = tl.load(X + {index} * {size} + offs{mask}).to({acc_t})')
            elif addr == 'ptr':
                inside += [f'v = tl.load(px{mask}).to({acc_t})', f'px += {step * size}']
            elif len(shape) == 2:
                inside += [f'v = tl.load(px).to({acc_t})', f'px = tl.advance(px, ({step * shape[0]}, 0))']
            else:
                inside += [f'v = tl.load(px).to({acc_t})', f'px = tl.advance(px, ({step * size},))']
            if p['inner']:
                inside += [f"for u in range({p['inner']}):",
                           f"    v = v + tl.load(Y + ({index} * {p['inner']} + u) * {size} + offs).to({acc_t})"]
            inside += self.triton_update(body, p['coef'], index)
        lines += ['        ' + line for line in inside]
        if loop == 'while':
            lines.append('        t += 1')
        if body == 'count':
            lines += ['    tl.store(OUT + offs, acc)', f'    tl.store(OUT + {size} + offs, cnt.to({acc_t}))']
        else:
            lines.append(f'    tl.store(OUT + {nd_offsets(shape)}, acc)')
        n_param = 'n' if p['dyn'] else 'n: tl.constexpr'
        return '\n'.join(['@triton.jit', f'def kernel(X, Y, OUT, {n_param}):'] + lines + [
            '', 'def launch(X, Y, out, options):', f'    kernel[(1,)](X, Y, out, {trips}, **options)'])

    @staticmethod
    def triton_update(body, coef, index):
        # Triton computes bf16 max/min and narrow integer arithmetic wider;
        # the carried value keeps its dtype
        if body == 'axpy':
            return [{1: 'acc = (acc + v).to(acc.dtype)', 2: 'acc = (acc * 2 + v).to(acc.dtype)',
                     -1: 'acc = (v - acc).to(acc.dtype)'}[coef]]
        if body == 'max':
            return ['acc = tl.maximum(acc, v).to(acc.dtype)']
        if body == 'xor':
            return ['acc = (acc ^ v).to(acc.dtype)']
        if body == 'cond':
            return [f'acc = tl.where({index} % 2 == 0, acc + v, acc - v).to(acc.dtype)']
        if body == 'ifelse':
            return [f'if {index} % 3 == 0:', '    acc = (acc + v).to(acc.dtype)', 'else:',
                    '    acc = (acc - v).to(acc.dtype)']
        if body == 'cumsum':
            return ['acc = (acc + tl.cumsum(v, 0).to(acc.dtype)).to(acc.dtype)']
        if body == 'count':
            return ['acc = (acc + v).to(acc.dtype)', 'cnt = cnt + (v > 0).to(tl.int32)']
        raise ValueError(body)

    def tilelang_kernel(self, params, plan):
        from .cast import tilelang_launch
        p = params
        shape, body = plan['shape'], plan['body']
        r, c = shape
        k = p['k']
        trips = p['trips']
        t_in = DTYPES[plan['in_dt']].tilelang
        t_acc = DTYPES[plan['acc_dt']].tilelang
        alloc_in = 'T.alloc_shared' if p['stage_in'] == 'shared' else 'T.alloc_fragment'
        lines = []
        if body == 'gemm':
            lines += [f'xs = T.alloc_shared(({r}, {k}), "{t_in}")', f'ws = T.alloc_shared(({k}, {c}), "{t_in}")']
            copies = [f'T.copy(X[t * {r}, 0], xs)', f'T.copy(Y[t * {k}, 0], ws)']
            signature = (f'X: T.Tensor(({trips * r}, {k}), "{t_in}"), Y: T.Tensor(({trips * k}, {c}), "{t_in}"), '
                         f'O: T.Tensor(({r}, {c}), "{t_acc}")')
        else:
            lines.append(f'xs = {alloc_in}(({r}, {c}), "{t_in}")')
            copies = [f'T.copy(X[t * {r}, 0], xs)']
            if p['inputs'] == 2:
                lines.append(f'ys = {alloc_in}(({r}, {c}), "{t_in}")')
                copies.append(f'T.copy(Y[t * {r}, 0], ys)')
            out_shape = (r,) if body == 'rsum' else (r, c)
            signature = (f'X: T.Tensor(({trips * r}, {c}), "{t_in}"), Y: T.Tensor(({trips * r}, {c}), "{t_in}"), '
                         f'O: T.Tensor({out_shape!r}, "{t_acc}")')
        acc_shape = (r,) if body == 'rsum' else (r, c)
        lines.append(f'acc = T.alloc_fragment({acc_shape!r}, "{t_acc}")')
        lines.append(f'T.fill(acc, T.cast({plan["init"]}, "{t_acc}"))' if plan['init'] else 'T.clear(acc)')
        if body == 'rsum':
            lines.append(f'row = T.alloc_fragment(({r},), "{t_in}")')
        compute = self.tilelang_compute(p, plan, r, c)
        statements = copies + compute
        loop = p['loop']
        if loop == 'pipelined':
            options = [f"num_stages={p['stages']}"]
            # order/stage/sync/group annotate top-level statements, not lines
            ncopy = len(copies)
            count = ncopy + sum(1 for line in compute if not line.startswith((' ', 'else')))
            stage = [0] * ncopy + [1] * (count - ncopy)
            if p['sched'] == 'order':
                options += [f'order={list(range(count))}', f'stage={stage}']
            elif p['sched'] == 'reorder':
                order = list(range(ncopy))[::-1] + list(range(ncopy, count))
                options += [f'order={order}', f'stage={stage}']
            elif p['sched'] == 'sync':
                options += [f'order={list(range(count))}', f'stage={stage}', f'sync={[[0, ncopy]]}']
            elif p['sched'] == 'group':
                options += [f'order={list(range(count))}', f'stage={stage}',
                            f'group={[[i] for i in range(count)]}']
            head = [f"for t in T.Pipelined({trips}, {', '.join(options)}):"]
        elif loop == 'serial':
            head = [f'for t in T.serial({trips}):']
        elif loop == 'unroll':
            head = [f'for t in T.unroll({trips}):']
        elif loop == 'unroll_explicit':
            head = [f'for t in T.Unroll({trips}, explicit=True):']
        elif loop == 'while':
            lines.append('t = T.alloc_var("int32", init=0)')
            head = [f'while t < {trips}:']
        else:
            head = [f'for t in T.serial({trips}):', f'    if t == {self.break_at(p)}:', '        T.loop_break()']
        lines += head
        lines += ['    ' + line for line in statements]
        if loop == 'while':
            lines.append('    t = t + 1')
        lines.append('T.copy(acc, O)')
        return tilelang_launch(signature, lines)

    @staticmethod
    def break_at(p):
        return max(1, p['trips'] - 1 - p['coef'] % 2)

    @staticmethod
    def tilelang_compute(p, plan, r, c):
        body = plan['body']
        t_acc = DTYPES[plan['acc_dt']].tilelang
        if body == 'gemm':
            return ['T.gemm(xs, ws, acc)']
        if body == 'rsum':
            return ['T.reduce_sum(xs, row, dim=1)', f'for i in T.Parallel({r}):',
                    f'    acc[i] = acc[i] + T.cast(row[i], "{t_acc}")']
        v = f'T.cast(xs[i, j], "{t_acc}")'
        if p['inputs'] == 2:
            v = f'({v} + T.cast(ys[i, j], "{t_acc}"))'
        loop = f'for i, j in T.Parallel({r}, {c}):'
        if body == 'axpy':
            coef = p['coef']
            expr = {1: f'acc[i, j] + {v}', 2: f'acc[i, j] * T.cast(2, "{t_acc}") + {v}',
                    -1: f'{v} - acc[i, j]'}[coef]
            return [loop, f'    acc[i, j] = {expr}']
        if body == 'max':
            return [loop, f'    acc[i, j] = T.max(acc[i, j], {v})']
        if body == 'cond':
            return ['if t % 2 == 0:', '    ' + loop, f'        acc[i, j] = acc[i, j] + {v}', 'else:', '    ' + loop,
                    f'        acc[i, j] = acc[i, j] - {v}']
        raise ValueError(body)

    def reference(self, params, plan):
        p = params
        body = plan['body']
        backend = plan['backend']
        lines = ['def reference(x, y):']
        shape = plan['shape']
        if backend == 'triton':
            indices = self.order_indices(p)
        else:
            indices = list(range(p['trips']))
            if p['loop'] == 'break':
                indices = indices[:self.break_at(p)]
        if backend == 'tilelang':
            trips, k = p['trips'], p['k']
            if body == 'gemm':
                lines += [f'    x = x.view({trips}, {shape[0]}, {k})', f'    y = y.view({trips}, {k}, {shape[1]})']
            else:
                lines += [f'    x = x.view({trips}, {shape[0]}, {shape[1]})',
                          f'    y = y.view({trips}, {shape[0]}, {shape[1]})']
        lines.append(f'    acc = torch.full({shape[:1] if body == "rsum" else shape!r}, {float(plan["init"])!r}, '
                     f'dtype=torch.float64)')
        if body == 'count':
            lines.append(f'    cnt = torch.zeros({shape!r}, dtype=torch.float64)')
        size = math.prod(shape)
        lines.append(f'    for t in {indices!r}:')
        if body in ('dot', 'gemm'):
            lines.append('        acc = acc + x[t] @ y[t]')
        else:
            lines.append('        v = x[t].clone()')
            if backend == 'tilelang' and p['inputs'] == 2:
                lines.append('        v = v + y[t]')
            if backend == 'triton' and p['mask'] == 'ragged':
                lines.append(f'        v = torch.where(torch.arange({size}).view({shape!r}) < {size} - t * 3, v, '
                             f'torch.zeros_like(v))')
            if backend == 'triton' and p['inner']:
                lines += [f"        for u in range({p['inner']}):", f"            v = v + y[t * {p['inner']} + u]"]
            if body == 'axpy':
                lines.append('        acc = ' + {1: 'acc + v', 2: 'acc * 2 + v', -1: 'v - acc'}[p['coef']])
            elif body == 'max':
                lines.append('        acc = torch.maximum(acc, v)')
            elif body == 'xor':
                lines.append('        acc = (acc.long() ^ v.long()).double()')
            elif body == 'cond':
                lines.append('        acc = acc + v if t % 2 == 0 else acc - v')
            elif body == 'ifelse':
                lines.append('        acc = acc + v if t % 3 == 0 else acc - v')
            elif body == 'cumsum':
                lines.append('        acc = acc + torch.cumsum(v, 0)')
            elif body == 'count':
                lines += ['        acc = acc + v', '        cnt = cnt + (v > 0).double()']
            elif body == 'rsum':
                lines.append('        acc = acc + v.sum(1)')
        lines.append('    return torch.cat([acc, cnt])' if body == 'count' else '    return acc')
        return '\n'.join(lines)
