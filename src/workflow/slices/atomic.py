"""Global-memory atomics: dtype x operation x contention x ordering.

Atomic lowering is selected per (operation, dtype, width): CUDA has no
native 16-bit max/min, 64-bit integer overloads differ between `long` and
`long long`, and TileLang packs vectorized adds. Every contribution is a
small integer-valued element, so the final contents are order-independent
and exact; the harness output buffer starts at 7 in every slot.
"""
from .base import Slice
from .chain import OPS, legalize_steps
from .dtypes import DTYPES, Domain, bit_bound, value_regimes

TRITON_OPS = ('add', 'max', 'min', 'and', 'or', 'xor', 'xchg')
TILELANG_OPS = ('add', 'max', 'min', 'or', 'addx2', 'addx4')
TRITON_DTYPES = ('f16', 'bf16', 'f32', 'f64', 'i32', 'i64')
TILELANG_DTYPES = ('f16', 'bf16', 'f32', 'f64', 'i32', 'i64')
INIT = 7


class AtomicSlice(Slice):
    name = 'atomic'
    pre_steps = 1
    post_steps = 0

    def space(self, backend):
        space = {'in_dt': TRITON_DTYPES if backend == 'triton' else TILELANG_DTYPES,
                 'values': ('tiny', 'small', 'nonneg', 'unit', 'half'), 'operand': ('const', 'input'),
                 'op': TRITON_OPS if backend == 'triton' else TILELANG_OPS,
                 'n': (32, 128, 256, 1024), 'blocks': (1, 2, 4), 'slots': (1, 4, 16, 64),
                 'contention': ('mod', 'block', 'reverse'), 'mask': (0, 1), 'pair': (0, 1),
                 'pre1_op': OPS, 'pre1_dt': ('same',)}
        if backend == 'triton':
            space.update(sem=('relaxed', 'acquire', 'release', 'acq_rel'), scope=('gpu', 'cta', 'sys'),
                         warps=(1, 2, 4, 8), warps2=(1, 2, 4, 8))
        else:
            space.update(threads=(32, 64, 128, 256), threads2=(32, 64, 128, 256))
        return space

    def legalize(self, params, backend):
        space = self.space(backend)
        for knob, values in space.items():
            if params.get(knob) not in values:
                params[knob] = values[0]
        for knob in list(params):
            if knob not in space:
                del params[knob]
        dt, op = params['in_dt'], params['op']
        d = DTYPES[dt]
        if op in ('and', 'or', 'xor') and d.is_float:
            op = 'add'
        if backend == 'triton' and op in ('max', 'min', 'xchg') and dt in ('f16', 'bf16'):
            op = 'add'  # Triton rejects 16-bit float max/min/xchg
        if op in ('addx2', 'addx4') and dt not in ('f16', 'bf16', 'f32'):
            op = 'add'
        lanes = {'addx2': 2, 'addx4': 4}.get(op, 1)
        if lanes > 1:
            params['slots'] = max(params['slots'], lanes)
            params['pre1_op'] = 'none'
            params['mask'] = 0
            params['contention'] = 'mod'
        params['op'] = op
        regimes = value_regimes()
        domain = regimes[params['values']]
        if op in ('and', 'or', 'xor') or not domain.fits(dt) or (d.kind == 'uint' and domain.lo < 0):
            params['values'], domain = 'nonneg', regimes['nonneg']
            if not domain.fits(dt):
                params['values'], domain = 'unit', regimes['unit']
        total = params['n'] * params['blocks']
        if op == 'xchg':
            # A unique destination per element keeps the final contents defined.
            params['slots'], params['contention'], params['mask'] = 64, 'mod', 0
            params['n'], params['blocks'] = 32, 2
            total = 64
        operand = domain if params['operand'] == 'input' else None
        semantics = 'trunc' if backend == 'triton' else 'floor'
        params['pre1_dt'] = 'same'
        chain_params = {'pre1_op': params['pre1_op'], 'pre1_dt': dt}
        _, value, pre = legalize_steps(chain_params, 'pre', 1, dt, domain, backend, operand, semantics,
                                       operand_dt=dt)
        params['pre1_op'] = chain_params['pre1_op']
        hits = -(-total // params['slots'])
        init = Domain(INIT, INIT)
        if op in ('add', 'addx2', 'addx4'):
            result = Domain(INIT + hits * min(value.lo, 0), INIT + hits * max(value.hi, 0), value.frac)
            if not result.fits(dt):
                params['values'], value = 'unit', regimes['unit']
                pre, params['pre1_op'] = [('none', dt, dt, 'const', None)], 'none'
                result = Domain(INIT - hits, INIT + hits)
            while not result.fits(dt) and params['slots'] < 64:
                params['slots'] *= 4
                hits = -(-total // params['slots'])
                result = Domain(INIT - hits, INIT + hits)
            while not result.fits(dt) and params['blocks'] > 1:
                params['blocks'] //= 2
                total = params['n'] * params['blocks']
                hits = -(-total // params['slots'])
                result = Domain(INIT - hits, INIT + hits)
        elif op in ('and', 'or', 'xor'):
            result = bit_bound(value.join(init))
        else:
            result = value.join(init)
        if not params['pair']:
            params['warps2' if backend == 'triton' else 'threads2'] = params['warps' if backend == 'triton' else 'threads']
        return {'backend': backend, 'semantics': semantics, 'input_domain': domain, 'in_dt': dt, 'out_dt': dt,
                'pre': pre, 'post': [], 'operand': params['operand'], 'valid': (params['n'] * params['blocks'],),
                'lanes': lanes, 'result': result}

    def plan_data(self, params, plan):
        return {'pre': plan['pre'], 'post': [], 'semantics': plan['semantics'], 'op': params['op'],
                'n': params['n'], 'blocks': params['blocks'], 'slots': params['slots'],
                'contention': params['contention'], 'mask': params['mask'], 'lanes': plan['lanes'],
                'init': INIT, 'out_dt': plan['out_dt']}

    @staticmethod
    def index_expr(params, flat):
        """Destination slot of element `flat` (an expression)."""
        slots, n, blocks = params['slots'], params['n'], params['blocks']
        if params['contention'] == 'block':
            return f'(({flat}) // {max(1, (n * blocks) // slots)}) % {slots}'
        if params['contention'] == 'reverse':
            return f'({n * blocks - 1} - ({flat})) % {slots}'
        return f'({flat}) % {slots}'

    def triton_kernel(self, params, plan):
        n, t = params['n'], DTYPES[params['in_dt']].triton
        body = ['    pid = tl.program_id(0)', f'    offs = pid * {n} + tl.arange(0, {n})',
                '    v = tl.load(X + offs)', '    w = tl.load(Y + offs)']
        body += ['    ' + line for line in self.triton_chain('v', plan['pre'], (n,), 'w')]
        body.append(f'    idx = {self.index_expr(params, "offs")}')
        mask = ', mask=(offs % 3) != 1' if params['mask'] else ''
        op = params['op']
        sem = f", sem='{params['sem']}', scope='{params['scope']}'"
        if op == 'xchg':
            body.append(f'    tl.atomic_xchg(OUT + idx, v.to({t}){mask}{sem})')
        else:
            body.append(f'    tl.atomic_{op}(OUT + idx, v.to({t}){mask}{sem})')
        return '\n'.join(['@triton.jit', 'def kernel(X, Y, OUT):'] + body + [
            '', 'def launch(X, Y, out, options):',
            f'    out.fill_({INIT})',
            f'    kernel[({params["blocks"]},)](X, Y, out, **options)'])

    def tilelang_kernel(self, params, plan):
        n, blocks, slots = params['n'], params['blocks'], params['slots']
        t = DTYPES[params['in_dt']].tilelang
        op, lanes = params['op'], plan['lanes']
        total = n * blocks
        if lanes > 1:
            loop = [f'for i in T.Parallel({n // lanes}):',
                    f'    T.atomic_{op}(O[(bx * {n} + i * {lanes}) % {slots}], X[bx * {n} + i * {lanes}])']
        else:
            value = self.tilelang_chain(f'X[bx * {n} + i]', plan['pre'], f'Y[bx * {n} + i]')
            target = f'O[{self.index_expr(params, f"bx * {n} + i")}]'
            call = f'T.atomic_{op}({target}, T.cast({value}, "{t}"))'
            loop = [f'for i in T.Parallel({n}):']
            if params['mask']:
                loop += [f'    if (bx * {n} + i) % 3 != 1:', f'        {call}']
            else:
                loop.append(f'    {call}')
        body = '\n'.join('                ' + line for line in loop)
        return f'''def build(threads):
    @tilelang.jit
    def kern():
        @T.prim_func
        def main(X: T.Tensor(({total},), "{t}"), Y: T.Tensor(({total},), "{t}"), O: T.Tensor(({slots},), "{t}")):
            with T.Kernel({blocks}, threads=threads) as bx:
{body}
        return main
    return kern()


_KERNELS = {{}}


def launch(X, Y, out, options):
    threads = options['threads']
    if threads not in _KERNELS:
        _KERNELS[threads] = build(threads)
    out.fill_({INIT})
    _KERNELS[threads](X, Y, out)'''

    def reference(self, params, plan):
        return '\n'.join(['def reference(x, y):', '    import torch', '    v = x'] + self.reference_chain('v', 'pre') + [
            "    n, blocks, slots, lanes = PLAN['n'], PLAN['blocks'], PLAN['slots'], PLAN['lanes']",
            '    flat = torch.arange(n * blocks)',
            "    if lanes > 1:",
            '        base = (flat // lanes) * lanes',
            '        idx = (base % slots) + (flat - base)',
            "    elif PLAN['contention'] == 'block':",
            '        idx = (flat // max(1, (n * blocks) // slots)) % slots',
            "    elif PLAN['contention'] == 'reverse':",
            '        idx = (n * blocks - 1 - flat) % slots',
            '    else:',
            '        idx = flat % slots',
            "    keep = (flat % 3) != 1 if PLAN['mask'] else torch.ones_like(flat, dtype=torch.bool)",
            "    out = torch.full((slots,), float(PLAN['init']), dtype=torch.float64)",
            "    op = PLAN['op']",
            "    if op in ('add', 'addx2', 'addx4'):",
            '        out.index_add_(0, idx[keep], v[keep])',
            "    elif op in ('max', 'min'):",
            "        out = out.scatter_reduce(0, idx[keep], v[keep], 'amax' if op == 'max' else 'amin', include_self=True)",
            "    elif op == 'xchg':",
            '        out[idx[keep]] = v[keep]',
            '    else:',
            '        acc = out.to(torch.int64)',
            '        for i, value in zip(idx[keep].tolist(), v[keep].to(torch.int64).tolist()):',
            "            acc[i] = acc[i] & value if op == 'and' else acc[i] | value if op == 'or' else acc[i] ^ value",
            '        out = acc.double()',
            '    return out'])
