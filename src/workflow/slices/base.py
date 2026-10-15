"""Shared slice machinery: knob spaces, shapes, chains and harness assembly."""
import math

from .chain import OPS, legalize_steps, triton_step, tilelang_step
from .dtypes import DTYPES, STORAGE, value_regimes
from src.backends.common.knobs import declared_pass_keys

from .runtime import SOURCES, _SLICE_AMD, _SLICE_TARGETS

GUARD = 64  # guard elements on each side of every output allocation

# Compiler configurations a program is compiled under; the first value of
# each knob is the default. They change lowering decisions, not semantics:
# Triton's register cap (ptxas spills), FMA contraction, libdevice
# flush-to-zero and default software-pipelining depth; TileLang's pass
# configurations (index width, vectorization, CSE, cp.async and LDG/STG
# lowering, shared-memory planning, let inlining, in-place storage reuse,
# loop unswitching, the reducer's baseline plan, the layout cost model,
# aliasing assumptions and ptxas register usage). A differential oracle per
# configuration exposes miscompilations that the default never reaches.
TRITON_CONFIGS = {'cfg_maxnreg': (0, 32, 64, 128), 'cfg_fusion': (1, 0), 'cfg_stages': (0, 1, 2, 4),
                  'cfg_ftz': (1, 0)}
TILELANG_PASSES = {
    'none': {}, 'index64': {'tl.config_index_bitwidth': 64}, 'novec': {'tirx.disable_vectorize': True},
    'nocse': {'tirx.disable_cse_tir': True}, 'noasync': {'tl.enable_async_copy': False},
    'ldgstg': {'tl.enable_lower_ldgstg': True, 'tl.enable_lower_ldgstg_predicated': True},
    'smem_merge': {'tl.enable_aggressive_shared_memory_merge': True},
    'smem_noreuse': {'tl.disable_shared_memory_reuse': True}, 'let_inline': {'tl.force_let_inline': True},
    'inplace': {'tl.storage_rewrite_detect_inplace': True}, 'no_unswitch': {'tl.disable_loop_unswitching': True},
    'unswitch_else': {'tl.loop_unswitching_allow_non_trivial_else': True},
    'reducer_baseline': {'tl.reducer_force_baseline': True}, 'io_aware': {'tl.layout_cost_model': 'io-aware'},
    'alias': {'tirx.noalias': False}, 'regs0': {'tl.ptxas_register_usage_level': 0},
    'regs10': {'tl.ptxas_register_usage_level': 10}}
# Configurations introduced after TileLang 0.1.14, which rejects their unknown
# keys ("Invalid config option"): offered only where the installed release
# declares them. The automatic warp-specialization scheduler (0.1.15) splits
# producer and consumer warps of pipelined kernels on sm_90 and newer, so the
# cross-target programs reach it; on sm_89 it is a no-op.
VERSIONED_PASSES = {'auto_ws': {'tl.enable_auto_warp_specialization': 'role_based'}}
TILELANG_PASSES.update({name: configs for name, configs in VERSIONED_PASSES.items()
                        if set(configs) <= (declared_pass_keys() or set())})
CONFIG_KNOBS = tuple(TRITON_CONFIGS) + ('cfg_pass', 'cfg_amd')
DEFAULT_CONFIG_PROB = 0.5  # share of sampled programs compiled under the default configuration

# Targets a program is compiled for besides the GPU in use (see
# _SLICE_TARGETS in runtime.py). Lowering of the campaign GPU saturates
# within hours; other architectures reach code the native target never
# runs: wgmma/TMA (sm_90), tcgen05 (sm_100), mma without cp.async (sm_75)
# and the AMD backends (Triton 3.x supports only compute capability 8.0 and
# up, so Turing is a TileLang target alone). Same-major older targets (sm_80 on sm_89) execute
# under the full oracle; the rest are compiled only, so a failure is a
# crash or a rejected build.
TARGETS = {'triton': ('native', 'sm80', 'sm90', 'sm100', 'sm120', 'gfx90a', 'gfx942', 'gfx950', 'gfx1100'),
           'tilelang': ('native', 'sm80', 'sm75', 'sm90', 'sm100', 'sm120', 'gfx942', 'gfx950')}
TARGET_PROB = 0.35  # share of sampled programs compiled for another target
HIP_TARGETS = tuple(name for name, (vendor, _, _) in _SLICE_TARGETS.items() if vendor == 'hip')
AMD_CONFIGS = tuple(_SLICE_AMD)  # Triton's AMD lowering choices; inert on CUDA targets


def launch_options(params, backend):
    """Compiler options of a program's configuration knobs."""
    if backend == 'tilelang':
        configs = TILELANG_PASSES[params.get('cfg_pass', 'none')]
        return {'pass_configs': dict(configs)} if configs else {}
    options = {}
    if params.get('cfg_maxnreg'):
        options['maxnreg'] = params['cfg_maxnreg']
    if params.get('cfg_fusion', 1) == 0:
        options['enable_fp_fusion'] = False
    if params.get('cfg_stages'):
        options['num_stages'] = params['cfg_stages']
    if params.get('cfg_ftz', 1) == 0:
        options['enable_reflect_ftz'] = False
    return options


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
    # Triton configuration knobs beyond the register cap and FMA contraction
    # that matter to this slice (pipelining depth for loops, flush-to-zero
    # for libdevice math).
    CONFIGS = ()

    def config_space(self, backend):
        if backend == 'tilelang':
            return {'cfg_pass': tuple(TILELANG_PASSES), 'cfg_target': TARGETS[backend]}
        space = {knob: TRITON_CONFIGS[knob] for knob in ('cfg_maxnreg', 'cfg_fusion') + self.CONFIGS}
        space['cfg_target'] = TARGETS[backend]
        space['cfg_amd'] = AMD_CONFIGS
        return space

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
        space.update(self.config_space(backend))
        return space

    def core_space(self, backend):
        return {}

    def sample(self, rng, backend):
        space = self.space(backend)
        params = {knob: rng.choice(values) for knob, values in space.items()}
        if rng.random() < DEFAULT_CONFIG_PROB:
            params.update({knob: space[knob][0] for knob in CONFIG_KNOBS if knob in space})
        if 'cfg_target' in space:
            params['cfg_target'] = (rng.choice(space['cfg_target'][1:]) if rng.random() < TARGET_PROB
                                    else 'native')
        if 'cfg_amd' in space and params['cfg_target'] not in HIP_TARGETS:
            params['cfg_amd'] = 'none'
        return params

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
        # The configuration stays out of the label, which failure keys keep:
        # one mechanism under several configurations is one bucket.
        config = launch_options(params, backend)
        if backend == 'triton':
            first = (f"w{params['warps']}", {'num_warps': params['warps'], **config})
            second = (f"w{params['warps2']}", {'num_warps': params['warps2'], **config})
        else:
            first = (f"t{params['threads']}", {'threads': params['threads'], **config})
            second = (f"t{params['threads2']}", {'threads': params['threads2'], **config})
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
            f"    compile_only = _slice_target({params.get('cfg_target', 'native')!r}, {backend!r}, {params.get('cfg_amd', 'none')!r})",
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
            '            if compile_only:',
            '                break',
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
            '    try:',
            '        main()',
            '    finally:',
            '        _slice_lowering(globals())',
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
