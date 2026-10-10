"""Shared slice machinery: knob spaces, shapes, chains and harness assembly."""
import math

from .chain import OPS, legalize_steps, triton_step, tilelang_step
from .dtypes import DTYPES, STORAGE, value_regimes
from .runtime import SOURCES

GUARD = 64  # guard elements on each side of every output allocation

TRITON_SHAPES = ('32', '64', '128', '256', '1024', '4x64', '16x16', '16x32', '32x16', '64x8', '2x128',
                 '2x4x32', '4x8x8', '2x16x4', '8x2x16')
TILELANG_SHAPES = ('32', '64', '128', '256', '512', '96', '16x16', '16x32', '32x64', '8x64', '64x8',
                   '4x128', '128x4', '64x64')


def parse_shape(text):
    return tuple(int(n) for n in text.split('x'))


def valid_extents(shape, tail):
    """Logical extents inside the tile; 'last' and 'all' leave a ragged tail."""
    def cut(n):
        return max(1, n - n // 4 - 1)
    if tail == 'none':
        return shape
    if tail == 'last':
        return shape[:-1] + (cut(shape[-1]),)
    return tuple(cut(n) for n in shape)


def neutral(kind, dtype):
    """Identity element of a reduction kind in dtype, as a Python literal."""
    d = DTYPES[dtype]
    if kind in ('sum', 'xor', 'or'):
        return 0
    if kind == 'and':
        return -1 if d.kind == 'int' else int(d.maxval)
    if kind == 'max':
        return float('-inf') if d.is_float else int(d.minval)
    if kind == 'min':
        return float('inf') if d.is_float else int(d.maxval)
    if kind == 'prod':
        return 1
    raise ValueError(kind)


def zero(dtype):
    """The literal a masked load fills with (fp8 pointers reject an int 0)."""
    return '0.0' if DTYPES[dtype].is_float else '0'


def literal(value):
    if isinstance(value, float) and math.isinf(value):
        return "float('inf')" if value > 0 else "float('-inf')"
    return repr(value)


class Slice:
    """A feature slice: a knob space plus per-backend lowering.

    Subclasses define `core_space(backend)`, `legalize_core`, the kernel
    lowering for each backend and the reference expression of the core."""
    name = ''
    backends = ('triton', 'tilelang')
    pre_steps = 2
    post_steps = 1
    # Simplest knob values that differ from the shared table in minimize.py
    # (a value may be a {backend: value} dict).
    SIMPLEST = {}

    # ---- knob space ---------------------------------------------------------
    def space(self, backend):
        shapes = TRITON_SHAPES if backend == 'triton' else TILELANG_SHAPES
        space = {
            'in_dt': STORAGE,
            'values': tuple(value_regimes()),
            'operand': ('const', 'input'),
            'shape': shapes,
            'tail': ('none', 'last', 'all'),
            'out_dt': STORAGE,
            'pair': (0, 1),
        }
        for i in range(1, self.pre_steps + 1):
            space[f'pre{i}_op'] = OPS
            space[f'pre{i}_dt'] = tuple(DTYPES)
        for i in range(1, self.post_steps + 1):
            space[f'post{i}_op'] = OPS
            space[f'post{i}_dt'] = tuple(DTYPES)
        if backend == 'triton':
            space['warps'] = (1, 2, 4, 8)
            space['warps2'] = (1, 2, 4, 8)
            space['dynamic'] = (0, 1)
        else:
            space['threads'] = (32, 64, 128, 256)
            space['threads2'] = (32, 64, 128, 256)
            space['stage'] = ('global', 'fragment', 'shared')
        space.update(self.core_space(backend))
        return space

    def core_space(self, backend):
        return {}

    def sample(self, rng, backend):
        return {knob: rng.choice(values) for knob, values in self.space(backend).items()}

    # ---- legalization -------------------------------------------------------
    def legalize(self, params, backend):
        """Repair params in place into an executable program; returns its plan."""
        space = self.space(backend)
        for knob, values in space.items():
            if params.get(knob) not in values:
                params[knob] = values[0]
        for knob in list(params):
            if knob not in space:
                del params[knob]
        regimes = value_regimes()
        in_dt = params['in_dt']
        domain = regimes[params['values']]
        if not domain.fits(in_dt) or (DTYPES[in_dt].kind == 'uint' and domain.lo < 0):
            for name in ('nonneg', 'tiny', 'unit'):
                candidate = regimes[name]
                if candidate.fits(in_dt) and (DTYPES[in_dt].kind != 'uint' or candidate.lo >= 0):
                    params['values'], domain = name, candidate
                    break
        operand = domain if params['operand'] == 'input' else None
        semantics = 'trunc' if backend == 'triton' else 'floor'
        dtype, domain, pre = legalize_steps(params, 'pre', self.pre_steps, in_dt, domain, backend,
                                            operand, semantics, operand_dt=in_dt)
        plan = {'backend': backend, 'semantics': semantics, 'input_domain': regimes[params['values']],
                'in_dt': in_dt, 'pre': pre, 'shape': parse_shape(params['shape']),
                'operand': params['operand']}
        plan['valid'] = valid_extents(plan['shape'], params['tail'])
        dtype, domain = self.legalize_core(params, backend, plan, dtype, domain)
        dtype, domain, post = legalize_steps(params, 'post', self.post_steps, dtype, domain, backend,
                                             None, semantics)
        plan['post'] = post
        out_dt = params['out_dt']
        from .chain import cast_allowed, cast_domain
        converted = cast_domain(domain, out_dt)
        if (not DTYPES[out_dt].torch or not cast_allowed(dtype, out_dt, backend) or not converted.fits(out_dt)
                or (DTYPES[out_dt].kind == 'uint' and domain.lo < 0)):
            out_dt = next((d for d in (dtype, 'f32', 'f64', 'i64') if DTYPES[d].torch
                           and cast_allowed(dtype, d, backend) and cast_domain(domain, d).fits(d)
                           and (DTYPES[d].kind != 'uint' or domain.lo >= 0)), 'f64')
            params['out_dt'] = out_dt
        plan['out_dt'] = out_dt
        plan['final_dt'] = dtype
        if backend == 'tilelang':
            # TileLang lays a fragment out over all threads of the block; a
            # tile whose size the block does not divide has no layout.
            size = math.prod(plan['shape'])
            for knob in ('threads', 'threads2'):
                if size % params[knob]:
                    params[knob] = max(t for t in (32, 64, 128, 256) if size % t == 0 or t == 32)
        if not params['pair']:
            if backend == 'triton':
                params['warps2'] = params['warps']
            else:
                params['threads2'] = params['threads']
        return plan

    def legalize_core(self, params, backend, plan, dtype, domain):
        """Legalize core knobs; returns the dtype and domain after the core."""
        return dtype, domain

    # ---- harness ------------------------------------------------------------
    def variants(self, params, backend):
        if backend == 'triton':
            first = (f"w{params['warps']}", {'num_warps': params['warps']})
            second = (f"w{params['warps2']}", {'num_warps': params['warps2']})
        else:
            first = (f"t{params['threads']}", {'threads': params['threads']})
            second = (f"t{params['threads2']}", {'threads': params['threads2']})
        return [first] if not params['pair'] or first[0] == second[0] else [first, second]

    def emit(self, program, config):
        params = dict(program.params)
        plan = self.legalize(params, program.backend)
        if params != program.params:
            raise ValueError('slice program parameters are not legalized')
        seed = config.input_seed if config is not None else 0
        backend = program.backend
        kernel = self.triton_kernel(params, plan) if backend == 'triton' else self.tilelang_kernel(params, plan)
        reference = self.reference(params, plan)
        imports = ('import triton\nimport triton.language as tl' if backend == 'triton'
                   else 'import tilelang\nimport tilelang.language as T')
        d = plan['input_domain']
        x_shape, y_shape = self.input_shapes(plan)
        lines = [
            'import sys', 'import torch', imports, '', SOURCES, '',
            f'PLAN = {self.plan_data(params, plan)!r}', '', kernel, '', reference, '',
            'def main():',
            f'    seed = {seed!r}',
            f'    x = _slice_values({x_shape!r}, {d.lo!r}, {d.hi!r}, {d.frac!r}, seed, 1)',
            f'    y = _slice_values({y_shape!r}, {d.lo!r}, {d.hi!r}, {d.frac!r}, seed, 2)',
            "    print('TILESMITH_STAGE=reference', file=sys.stderr, flush=True)",
            '    expected = reference(x, y)',
            f'    X = _slice_storage(x, {DTYPES[plan["in_dt"]].torch!r})',
            f'    Y = _slice_storage(y, {DTYPES[plan["in_dt"]].torch!r})',
            f'    for label, options in {self.variants(params, backend)!r}:',
            '        previous = None',
            '        for run in range(2):',
            f'            storage, out = _slice_output(tuple(expected.shape), {DTYPES[plan["out_dt"]].torch!r}, {GUARD})',
            "            print(f'TILESMITH_STAGE=execute:{label}:{run}', file=sys.stderr, flush=True)",
            '            try:',
            '                launch(X, Y, out, options)',
            '                torch.cuda.synchronize()',
            '            except Exception as exc:',
            '                _slice_reject(exc)',
            '                raise',
            f'            guard = torch.cat([storage[:{GUARD}].float(), storage[-{GUARD}:].float()])',
            '            if not bool((guard == 7).all()):',
            f"                raise RuntimeError(f'WRONG RESULT: output canary modified by slice={self.name} variant={{label}}')",
            '            if previous is not None and not torch.equal(_slice_bits(previous), _slice_bits(out)):',
            f"                raise RuntimeError(f'WRONG RESULT: repeat determinism of slice={self.name} variant={{label}}')",
            f"            _slice_compare(out, expected, f'slice={self.name} variant={{label}}')",
            '            previous = out.clone()',
            "    print('ALL PASSED')",
            '',
            "if __name__ == '__main__':",
            '    main()',
        ]
        return '\n'.join(lines) + '\n'

    def input_shapes(self, plan):
        return plan['valid'], plan['valid']

    def plan_data(self, params, plan):
        return {'pre': plan['pre'], 'post': plan['post'], 'semantics': plan['semantics'],
                'operand': plan['operand'], 'valid': list(plan['valid'])}

    # ---- chain helpers for subclasses ---------------------------------------
    @staticmethod
    def triton_chain(value, steps, shape, operand_expr):
        lines = []
        for step in steps:
            lines += triton_step(value, step, shape, operand_expr)
        return lines

    @staticmethod
    def tilelang_chain(expr, steps, operand_expr):
        for step in steps:
            expr = tilelang_step(expr, step, operand_expr)
        return expr

    @staticmethod
    def reference_chain(value, steps_name, operand='y'):
        return [f"    for step in PLAN[{steps_name!r}]:",
                f"        {value} = _slice_step({value}, step, {operand}, PLAN['semantics'])"]


def triton_offsets(shape, valid, dynamic):
    """Offsets and mask expressions of a rank-1..3 tile in row-major order."""
    rank = len(shape)
    names = [f'n{i}' for i in range(rank)] if dynamic else [str(v) for v in valid]
    aranges = []
    for i, n in enumerate(shape):
        index = ['None'] * rank
        index[i] = ':'
        suffix = '' if rank == 1 else '[' + ', '.join(index) + ']'
        aranges.append(f'tl.arange(0, {n}){suffix}')
    strides = [math.prod(valid[i + 1:]) for i in range(rank)]
    if dynamic:
        strides = ['*'.join(names[i + 1:]) or '1' for i in range(rank)]
    offs = ' + '.join(f'{a} * {s}' for a, s in zip(aranges, strides))
    mask = ' & '.join(f'({a} < {n})' for a, n in zip(aranges, names))
    return f'({offs})', f'({mask})', names
