"""Matrix multiply: operand dtype x transposition x placement x pipelining.

MMA lowering is specialized per operand dtype (fp16, bf16, fp8 e4m3/e5m2,
int8, tf32), per operand layout and per software pipeline. Inputs are small
integers or dyadic values, so every accumulator stays exact and the float64
product is the reference. An epilogue chain then converts the accumulator.
"""
from .base import Slice, zero
from .chain import OPS, cast_allowed, cast_domain, legalize_steps
from .dtypes import DTYPES, STORAGE, Domain, value_regimes

MMA = ('f16', 'bf16', 'f8e4', 'f8e5', 'i8', 'f32')
ACCUMULATORS = ('f32', 'f16', 'i32')


def mma_k_min(dtype):
    return 32 if dtype in ('f8e4', 'f8e5', 'i8') else 16


def warp_partition(bm, bn, warps, policy):
    """True when T.gemm can split a (bm, bn) tile over `warps` warps."""
    options = [(m, warps // m) for m in range(1, warps + 1) if warps % m == 0]
    if policy == 'FullRow':
        options = [(warps, 1)]
    elif policy == 'FullCol':
        options = [(1, warps)]
    return any(bm % (16 * m) == 0 and bn % (8 * n) == 0 for m, n in options)


class GemmSlice(Slice):
    name = 'gemm'
    post_steps = 2

    def space(self, backend):
        space = {'mma_dt': MMA, 'values': ('tiny', 'small', 'nonneg', 'half', 'unit'),
                 'tail': ('none', 'last', 'all'), 'trans_a': (0, 1), 'trans_b': (0, 1),
                 'acc': ACCUMULATORS, 'out_dt': STORAGE, 'pair': (0, 1)}
        for i in range(1, self.post_steps + 1):
            space[f'post{i}_op'] = OPS
            space[f'post{i}_dt'] = tuple(DTYPES)
        if backend == 'triton':
            space.update(m=(16, 32, 64, 128), n=(16, 32, 64, 128), k=(16, 32, 64, 128), batch=(1, 2, 4),
                         kloop=(1, 2, 4), stages=(1, 2, 3, 4), acc_mode=('ret', 'arg', 'plus'),
                         prec=('ieee', 'tf32', 'tf32x3'), warps=(1, 2, 4, 8), warps2=(1, 2, 4, 8))
        else:
            space.update(bm=(16, 32, 64, 128), bn=(16, 32, 64, 128), bk=(16, 32, 64), gm=(1, 2), gn=(1, 2),
                         gk=(1, 2, 3), areg=(0, 1), kpack=(1, 2), policy=('Square', 'FullRow', 'FullCol'),
                         stages=(1, 2, 3), loop=('pipelined', 'serial'), clear_accum=(0, 1),
                         threads=(64, 128, 256), threads2=(64, 128, 256))
        return space

    def legalize(self, params, backend):
        space = self.space(backend)
        for knob, values in space.items():
            if params.get(knob) not in values:
                params[knob] = values[0]
        for knob in list(params):
            if knob not in space:
                del params[knob]
        mma = params['mma_dt']
        regimes = value_regimes()
        domain = regimes[params['values']]
        if not domain.fits(mma):
            params['values'], domain = 'tiny', regimes['tiny']
        if mma == 'i8':
            params['acc'] = 'i32'
        elif params['acc'] == 'i32' or (params['acc'] == 'f16' and mma != 'f16'):
            params['acc'] = 'f32'
        kmin = mma_k_min(mma)
        if backend == 'triton':
            params['k'] = max(params['k'], kmin * params['kloop'])
            while params['k'] // params['kloop'] < kmin:
                params['kloop'] //= 2
            if params['kloop'] == 1:
                params['stages'] = 1
            if mma != 'f32':
                params['prec'] = 'ieee'
            if mma == 'i8' and params['acc_mode'] == 'arg':
                params['acc_mode'] = 'ret'
            tile = (params['m'], params['n'], params['k'])
            # One program holds the whole tile in registers: bound its size,
            # and keep the pipelined operand buffers within shared memory.
            while params['batch'] > 1 and params['batch'] * params['m'] * params['n'] > 8192:
                params['batch'] //= 2
            size = DTYPES[mma].bits // 8
            chunk = params['k'] // params['kloop']

            def staged():
                return params['batch'] * (params['m'] + params['n']) * chunk * size * max(params['stages'], 1)
            while params['stages'] > 1 and staged() > 64 * 1024:
                params['stages'] -= 1
            while params['batch'] > 1 and staged() > 64 * 1024:
                params['batch'] //= 2
        else:
            params['bk'] = max(params['bk'], kmin)
            for knob in ('threads', 'threads2'):
                if not warp_partition(params['bm'], params['bn'], params[knob] // 32, params['policy']):
                    params[knob] = next((t for t in (128, 64, 256) if warp_partition(
                        params['bm'], params['bn'], t // 32, params['policy'])), params[knob])
            if not warp_partition(params['bm'], params['bn'], params['threads'] // 32, params['policy']):
                params['policy'] = 'Square'
                params['bm'], params['bn'] = max(params['bm'], 32), max(params['bn'], 32)
                params['threads'] = params['threads2'] = 128
            if not warp_partition(params['bm'], params['bn'], params['threads2'] // 32, params['policy']):
                params['threads2'] = params['threads']
            size = DTYPES[mma].bits // 8
            while params['stages'] > 1 and (params['bm'] + params['bn']) * params['bk'] * size * params['stages'] > 96 * 1024:
                params['stages'] -= 1
            if params['loop'] == 'serial':
                params['stages'] = 1
            tile = (params['bm'] * params['gm'], params['bn'] * params['gn'], params['bk'] * params['gk'])
        m, n, k = tile
        cut = {'none': (0, 0, 0), 'last': (0, 0, 1), 'all': (1, 1, 1)}[params['tail']]
        valid = tuple(max(1, size - c * (size // 4 + 1)) for size, c in zip((m, n, k), cut))
        products = Domain(-domain.magnitude ** 2 if domain.lo < 0 else 0, domain.magnitude ** 2, 2 * domain.frac)
        acc_domain = Domain(valid[2] * products.lo, valid[2] * products.hi, products.frac)
        if not acc_domain.fits(params['acc']):
            params['acc'] = 'f32' if mma != 'i8' else 'i32'
        if not acc_domain.fits(params['acc']):
            params['values'], domain = 'unit', regimes['unit']
            acc_domain = Domain(-valid[2], valid[2])
        plan = {'backend': backend, 'semantics': 'trunc' if backend == 'triton' else 'floor', 'in_dt': mma,
                'input_domain': domain, 'tile': tile, 'valid3': valid, 'pre': [], 'operand': 'const'}
        dtype, out, post = legalize_steps(params, 'post', self.post_steps, params['acc'], acc_domain, backend,
                                          None, plan['semantics'])
        plan['post'] = post
        out_dt = params['out_dt']
        converted = cast_domain(out, out_dt)
        if (not cast_allowed(dtype, out_dt, backend) or not converted.fits(out_dt)
                or (DTYPES[out_dt].kind == 'uint' and out.lo < 0)):
            out_dt = next(d for d in (dtype, 'f32', 'f64', 'i64') if DTYPES[d].torch and cast_allowed(dtype, d, backend)
                          and cast_domain(out, d).fits(d) and (DTYPES[d].kind != 'uint' or out.lo >= 0))
            params['out_dt'] = out_dt
        plan['out_dt'], plan['final_dt'] = out_dt, dtype
        batch = params.get('batch', 1)
        plan['batch'] = batch
        plan['valid'] = (batch,) + valid if batch > 1 else valid
        if not params['pair']:
            params['warps2' if backend == 'triton' else 'threads2'] = params['warps' if backend == 'triton' else 'threads']
        return plan

    def input_shapes(self, plan):
        m, n, k = plan['valid3']
        lead = (plan['batch'],) if plan['batch'] > 1 else ()
        return lead + (m, k), lead + (k, n)

    def plan_data(self, params, plan):
        return {'post': plan['post'], 'semantics': plan['semantics'], 'out_dt': plan['out_dt'],
                'trans_a': params['trans_a'], 'trans_b': params['trans_b']}

    def reference(self, params, plan):
        return '\n'.join([
            'def reference(x, y):',
            '    r = x @ y',
        ] + self.reference_chain('r', 'post', 'None') + ["    return _slice_cast(r, PLAN['out_dt'])"])

    def emit(self, program, config):
        # Stored operands are transposed copies when trans_a/trans_b is set;
        # the kernel undoes the transposition.
        code = super().emit(program, config)
        return code.replace(
            '    X = _slice_storage(x,',
            "    x_store = x.transpose(-1, -2).contiguous() if PLAN['trans_a'] else x\n"
            "    y_store = y.transpose(-1, -2).contiguous() if PLAN['trans_b'] else y\n"
            '    X = _slice_storage(x_store,').replace('    Y = _slice_storage(y,', '    Y = _slice_storage(y_store,')

    # ---- Triton --------------------------------------------------------------
    def triton_kernel(self, params, plan):
        m, n, k = plan['tile']
        mv, nv, kv = plan['valid3']
        batch, kloop = plan['batch'], params['kloop']
        kc = k // kloop
        acc = DTYPES[params['acc']].triton
        b3 = batch > 1
        z = zero(params['mma_dt'])

        def tile2(rows, cols, rv, cv, row_name, col_name, stride):
            # Offsets/mask of a row-major (rows x cols) view with valid (rv, cv).
            if b3:
                return (f'(tl.arange(0, {batch})[:, None, None] * {rv * cv} + {row_name}[None, :, None] * {stride} + {col_name}[None, None, :])',
                        f'(({row_name}[None, :, None] < {rv}) & ({col_name}[None, None, :] < {cv}))')
            return (f'({row_name}[:, None] * {stride} + {col_name}[None, :])',
                    f'(({row_name}[:, None] < {rv}) & ({col_name}[None, :] < {cv}))')

        def trans(expr):
            return f'tl.permute({expr}, (0, 2, 1))' if b3 else f'tl.trans({expr})'

        body = ['    rm = tl.arange(0, %d)' % m, '    rn = tl.arange(0, %d)' % n,
                f'    acc = tl.zeros({(batch, m, n) if b3 else (m, n)!r}, dtype={acc})',
                f'    for kk in tl.range(0, {kloop}, num_stages={params["stages"]}):',
                f'        rk = kk * {kc} + tl.arange(0, {kc})']
        if params['trans_a']:
            offs, mask = tile2(kc, m, kv, mv, 'rk', 'rm', mv)
            body.append(f'        a = {trans(f"tl.load(A + {offs}, mask={mask}, other={z})")}')
        else:
            offs, mask = tile2(m, kc, mv, kv, 'rm', 'rk', kv)
            body.append(f'        a = tl.load(A + {offs}, mask={mask}, other={z})')
        if params['trans_b']:
            offs, mask = tile2(n, kc, nv, kv, 'rn', 'rk', kv)
            body.append(f'        b = {trans(f"tl.load(B + {offs}, mask={mask}, other={z})")}')
        else:
            offs, mask = tile2(kc, n, kv, nv, 'rk', 'rn', nv)
            body.append(f'        b = tl.load(B + {offs}, mask={mask}, other={z})')
        options = f', out_dtype={acc}'
        if params['mma_dt'] == 'f32':
            options += f", input_precision='{params['prec']}'"
        if params['acc_mode'] == 'arg':
            body.append(f'        acc = tl.dot(a, b, acc{options})')
        elif params['acc_mode'] == 'plus':
            body.append(f'        acc = acc + tl.dot(a, b{options})')
        else:
            body.append(f'        acc += tl.dot(a, b{options})')
        out_shape = (batch, m, n) if b3 else (m, n)
        body += ['    ' + line for line in self.triton_chain('acc', plan['post'], out_shape, None)]
        offs, mask = tile2(m, n, mv, nv, 'rm', 'rn', nv)
        body.append(f'    tl.store(OUT + {offs}, acc.to({DTYPES[plan["out_dt"]].triton}), mask={mask})')
        return '\n'.join(['@triton.jit', 'def kernel(A, B, OUT):'] + body + [
            '', 'def launch(X, Y, out, options):',
            '    kernel[(1,)](X, Y, out, num_stages=1, **options)'])

    # ---- TileLang ------------------------------------------------------------
    def tilelang_kernel(self, params, plan):
        bm, bn, bk = params['bm'], params['bn'], params['bk']
        mv, nv, kv = plan['valid3']
        mma, acc = DTYPES[params['mma_dt']].tilelang, DTYPES[params['acc']].tilelang
        out_t = DTYPES[plan['out_dt']].tilelang
        ta, tb = params['trans_a'], params['trans_b']
        a_shape, b_shape = ((kv, mv) if ta else (mv, kv)), ((nv, kv) if tb else (kv, nv))
        a_tile, b_tile = ((bk, bm) if ta else (bm, bk)), ((bn, bk) if tb else (bk, bn))
        a_at = f'A[k * {bk}, by * {bm}]' if ta else f'A[by * {bm}, k * {bk}]'
        b_at = f'B[bx * {bn}, k * {bk}]' if tb else f'B[k * {bk}, bx * {bn}]'
        loop = (f'T.Pipelined(T.ceildiv({kv}, {bk}), num_stages={params["stages"]})'
                if params['loop'] == 'pipelined' else f'T.serial(T.ceildiv({kv}, {bk}))')
        options = [f'transpose_A={bool(ta)}', f'transpose_B={bool(tb)}',
                   f'policy=T.GemmWarpPolicy.{params["policy"]}']
        if params['kpack'] != 1:
            options.append(f'k_pack={params["kpack"]}')
        if params['clear_accum']:
            options.append('clear_accum=(k == 0)')
        lines = [f'As = T.alloc_shared({a_tile!r}, "{mma}")', f'Bs = T.alloc_shared({b_tile!r}, "{mma}")',
                 f'Cl = T.alloc_fragment(({bm}, {bn}), "{acc}")']
        operand = 'As'
        if params['areg']:
            lines.append(f'Af = T.alloc_fragment({a_tile!r}, "{mma}")')
            operand = 'Af'
        lines.append('T.fill(Cl, 7)' if params['clear_accum'] else 'T.clear(Cl)')
        lines += [f'for k in {loop}:', f'    T.copy({a_at}, As)', f'    T.copy({b_at}, Bs)']
        if params['areg']:
            lines.append('    T.copy(As, Af)')
        lines.append(f'    T.gemm({operand}, Bs, Cl, {", ".join(options)})')
        post = self.tilelang_chain('Cl[i, j]', plan['post'], None)
        lines += [f'for i, j in T.Parallel({bm}, {bn}):',
                  f'    if by * {bm} + i < {mv} and bx * {bn} + j < {nv}:',
                  f'        O[by * {bm} + i, bx * {bn} + j] = T.cast({post}, "{out_t}")']
        signature = (f'A: T.Tensor({a_shape!r}, "{mma}"), B: T.Tensor({b_shape!r}, "{mma}"), '
                     f'O: T.Tensor(({mv}, {nv}), "{out_t}")')
        body = '\n'.join('                ' + line for line in lines)
        grid = f'T.ceildiv({nv}, {bn}), T.ceildiv({mv}, {bm})'
        return f'''def build(threads):
    @tilelang.jit
    def kern():
        @T.prim_func
        def main({signature}):
            with T.Kernel({grid}, threads=threads) as (bx, by):
{body}
        return main
    return kern()


_KERNELS = {{}}


def launch(X, Y, out, options):
    threads = options['threads']
    if threads not in _KERNELS:
        _KERNELS[threads] = build(threads)
    _KERNELS[threads](X, Y, out)'''
