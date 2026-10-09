"""Rounding conversions: inexact values through every conversion path.

The cast slice keeps every value exactly representable, so it cannot see a
conversion that rounds wrongly. Here the inputs are ties and near-ties of
the destination grid, subnormals and range edges, and the reference is the
correctly rounded result (one rounding, ties to even, or toward zero for
Triton's fp_downcast_rounding='rtz'); float-to-integer conversions truncate.
A conversion can go through a wider format first, which must not change it.
"""
import zlib

from .base import GUARD, Slice
from .runtime import SOURCES
from .dtypes import DTYPES

FLOATS = ('f64', 'f32', 'f16', 'bf16', 'f8e4', 'f8e5')
INTEGERS = ('i8', 'i16', 'i32', 'i64', 'u8')
TRITON_BLOCKS = (64, 128, 256, 1024)
TILELANG_BLOCKS = (128, 256, 512)
WIDTH = {'f64': 64, 'f32': 32, 'f16': 16, 'bf16': 16, 'f8e4': 8, 'f8e5': 8}


def wider(via, src):
    """True when float format via represents every value of src exactly."""
    order = {'f8e4': ('f16', 'bf16', 'f32', 'f64'), 'f8e5': ('f16', 'bf16', 'f32', 'f64'),
             'f16': ('f32', 'f64'), 'bf16': ('f32', 'f64'), 'f32': ('f64',), 'f64': (),
             'i8': ('f16', 'bf16', 'f32', 'f64'), 'u8': ('f16', 'bf16', 'f32', 'f64'),
             'i16': ('f32', 'f64'), 'i32': ('f64',), 'i64': ()}
    return via in order[src]


class RoundSlice(Slice):
    name = 'round'

    def space(self, backend):
        space = {'src': FLOATS + INTEGERS, 'dst': FLOATS[1:] + INTEGERS, 'via': ('none', 'f32', 'f64'),
                 'mix': ('ties', 'wide', 'edges'), 'rows': (1, 2, 4), 'tail': ('none', 'last'), 'pair': (0, 1)}
        if backend == 'triton':
            space.update(mode=('rtne', 'rtz'), block=TRITON_BLOCKS, warps=(1, 2, 4, 8), warps2=(1, 2, 4, 8))
        else:
            space.update(block=TILELANG_BLOCKS, stage=('global', 'fragment', 'shared'),
                         threads=(32, 64, 128, 256), threads2=(32, 64, 128, 256))
        return space

    def legalize(self, params, backend):
        space = self.space(backend)
        for knob, values in space.items():
            if params.get(knob) not in values:
                params[knob] = values[0]
        for knob in list(params):
            if knob not in space:
                del params[knob]
        src, dst = params['src'], params['dst']
        if src == dst or (src in INTEGERS and dst in INTEGERS):
            params['dst'] = dst = 'f16' if src != 'f16' else 'bf16'
        # Neither front end converts between integers and fp8.
        if (src in INTEGERS and DTYPES[dst].is_fp8) or (dst in INTEGERS and DTYPES[src].is_fp8):
            params['dst'] = dst = 'f32'
        if src in INTEGERS and dst in INTEGERS:
            params['dst'] = dst = 'f32'
        if params['via'] != 'none' and (not wider(params['via'], src) or params['via'] == dst):
            params['via'] = 'none'
        rounding = src in FLOATS and dst in FLOATS and WIDTH[dst] < WIDTH[src if params['via'] == 'none'
                                                                         else params['via']]
        if backend == 'triton' and (params['mode'] == 'rtz') and not rounding:
            params['mode'] = 'rtne'
        if dst in INTEGERS:
            params['mix'] = 'wide'
        if backend == 'tilelang':
            for knob in ('threads', 'threads2'):
                params[knob] = min(params[knob], params['block'])
        if not params['pair']:
            params['warps2' if backend == 'triton' else 'threads2'] = params['warps' if backend == 'triton' else 'threads']
        rows = params['rows']
        n = rows * params['block'] - (params['block'] // 4 + 1 if params['tail'] == 'last' else 0)
        return {'backend': backend, 'n': n, 'mode': params.get('mode', 'rtne')}

    def plan_data(self, params, plan):
        return {}

    def emit(self, program, config):
        params = dict(program.params)
        plan = self.legalize(params, program.backend)
        if params != program.params:
            raise ValueError('slice program parameters are not legalized')
        seed = config.input_seed if config is not None else 0
        backend = program.backend
        src, dst, n = params['src'], params['dst'], plan['n']
        kernel = self.triton_kernel(params, plan) if backend == 'triton' else self.tilelang_kernel(params, plan)
        imports = ('import triton\nimport triton.language as tl' if backend == 'triton'
                   else 'import tilelang\nimport tilelang.language as T')
        if dst in INTEGERS:
            expected = 'torch.trunc(x)'
        else:
            expected = f"_slice_round(x, {dst!r}, {plan['mode']!r})"
        lines = [
            'import sys', 'import torch', imports, '', SOURCES, '', kernel, '',
            'def main():',
            f'    x = _slice_round_values({src!r}, {dst!r}, {n}, {seed!r} * 7919 + {zlib.crc32((src + dst).encode()) % 7919}, '
            f'{params["mix"]!r})',
            "    print('TILESMITH_STAGE=reference', file=sys.stderr, flush=True)",
            f'    expected = {expected}',
            f'    X = _slice_storage(x, {DTYPES[src].torch!r})',
            f'    for label, options in {self.variants(params, backend)!r}:',
            '        previous = None',
            '        for run in range(2):',
            f'            storage, out = _slice_output(({n},), {DTYPES[dst].torch!r}, {GUARD})',
            "            print(f'TILESMITH_STAGE=execute:{label}:{run}', file=sys.stderr, flush=True)",
            '            try:',
            '                launch(X, out, options)',
            '                torch.cuda.synchronize()',
            '            except Exception as exc:',
            '                _slice_reject(exc)',
            '                raise',
            f'            guard = torch.cat([storage[:{GUARD}].float(), storage[-{GUARD}:].float()])',
            '            if not bool((guard == 7).all()):',
            "                raise RuntimeError(f'WRONG RESULT: output canary modified by slice=round variant={label}')",
            '            if previous is not None and not torch.equal(_slice_bits(previous), _slice_bits(out)):',
            "                raise RuntimeError(f'WRONG RESULT: repeat determinism of slice=round variant={label}')",
            "            _slice_compare(out, expected, f'slice=round variant={label}')",
            '            previous = out.clone()',
            "    print('ALL PASSED')",
            '',
            "if __name__ == '__main__':",
            '    main()',
        ]
        return '\n'.join(lines) + '\n'

    def triton_kernel(self, params, plan):
        block, src, dst, via = params['block'], params['src'], params['dst'], params['via']
        convert = 'x'
        if via != 'none':
            convert = f'{convert}.to({DTYPES[via].triton})'
        rounding = ", fp_downcast_rounding='rtz'" if params['mode'] == 'rtz' else ''
        convert = f'{convert}.to({DTYPES[dst].triton}{rounding})'
        other = '0.0' if src in FLOATS else '0'
        return f'''@triton.jit
def kernel(X, OUT, n):
    offs = tl.program_id(0) * {block} + tl.arange(0, {block})
    mask = offs < n
    x = tl.load(X + offs, mask=mask, other={other})
    tl.store(OUT + offs, {convert}, mask=mask)


def launch(X, out, options):
    kernel[({params['rows']},)](X, out, {plan['n']}, **options)'''

    def tilelang_kernel(self, params, plan):
        block, src, dst, via, n = params['block'], params['src'], params['dst'], params['via'], plan['n']
        s_t, d_t = DTYPES[src].tilelang, DTYPES[dst].tilelang
        value = 'X[bx * %d + i]' % block if params['stage'] == 'global' else 'xs[i]'
        if via != 'none':
            value = f'T.cast({value}, "{DTYPES[via].tilelang}")'
        value = f'T.cast({value}, "{d_t}")'
        guard = f'bx * {block} + i < {n}'
        if params['stage'] == 'global':
            body = [f'for i in T.Parallel({block}):', f'    if {guard}:', f'        O[bx * {block} + i] = {value}']
        else:
            alloc = 'T.alloc_fragment' if params['stage'] == 'fragment' else 'T.alloc_shared'
            body = [f'xs = {alloc}(({block},), "{s_t}")', f'of = T.alloc_fragment(({block},), "{d_t}")',
                    f'for i in T.Parallel({block}):',
                    f'    xs[i] = T.if_then_else({guard}, X[T.min(bx * {block} + i, {n - 1})], T.cast(0, "{s_t}"))',
                    f'for i in T.Parallel({block}):', f'    of[i] = {value}',
                    f'for i in T.Parallel({block}):', f'    if {guard}:', f'        O[bx * {block} + i] = of[i]']
        text = '\n'.join('                ' + line for line in body)
        return f'''def build(threads):
    @tilelang.jit
    def kern():
        @T.prim_func
        def main(X: T.Tensor(({n},), "{s_t}"), O: T.Tensor(({n},), "{d_t}")):
            with T.Kernel({params['rows']}, threads=threads) as bx:
{text}
        return main
    return kern()


_KERNELS = {{}}


def launch(X, out, options):
    threads = options['threads']
    if threads not in _KERNELS:
        _KERNELS[threads] = build(threads)
    _KERNELS[threads](X, out)'''
