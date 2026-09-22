"""TileLang target rules, emitters, and launch conventions."""
from src.backends.common.builtin import CudaTileBackend
from . import params


class TileLangBackend(CudaTileBackend):
    name = 'tilelang'
    params = params
    imports = 'import tilelang\nimport tilelang.language as T'
    min_block_k = 8

    def extended_variants(self, program, config=None):
        from .extended import ExtendedLowering
        if config is None:
            from src.config import DEFAULT_CONFIG
            config = DEFAULT_CONFIG
        # Each warp requires a 16x16 output tile on the MMA path. Small
        # internal matmuls cannot use the ordinary 128/256-thread defaults.
        products = [n.results[0].type for n in program.all_operations() if n.op == 'matmul']
        threads = min([128] + [max(32, t.size // 256 * 32) for t in products])
        configurations = [(threads, 1, {})]
        if program.configuration_pair and config.extended_config_depth >= 1:
            configurations.append((threads if products else 256, 2, {'tirx.disable_vectorize': True}))
        if program.pass_config_pair and config.extended_config_depth >= 2:
            # tl.disable_loop_unswitching toggles a real C++ transform pass
            # (loop_unswitching.cc). The previous tirx.disable_cse_tir key was
            # inert: only the TVM s_tir pipeline reads it, and tilelang.compile
            # runs the CUDA pipeline, so the depth-2 pair compiled identically
            # to the base. Fallback if this proves unstable: tl.force_let_inline.
            configurations.append((threads, 1, {'tl.disable_loop_unswitching': True}))
        if program.fast_math_pair and config.extended_fast_math_pair:
            configurations.append((threads, 1, {'tl.enable_fast_math': True}))
        variants = [(ExtendedLowering(program, f'extended_{i}_{int(observe)}', observe, threads, stages),
                     {'threads': threads, 'stages': stages, 'pass_configs': passes})
                    for i, (threads, stages, passes) in enumerate(configurations)
                    for observe in ([False, True] if program.observation_pair else [False])]
        # Random pass-pipeline sampling (MLIRSmith-style): plain deterministic-
        # random configurations appended after the fixed sweep tiers, so the
        # precision/identity blocks below (which iterate `configurations`) do
        # not multiply. Plain variants are baseline-checked at runtime.
        from src.backends.common.knobs import sample_configs
        sampled = sample_configs(program, config, self.name)
        start = len(configurations)
        for j, options in enumerate(sampled):
            for observe in ([False, True] if program.observation_pair else [False]):
                variants.append((
                    ExtendedLowering(program, f'extended_{start + j}_{int(observe)}', observe,
                                     options['threads'], options['stages']),
                    options))
        if program.precision_pair and config.extended_precision_pair:
            # Accumulator-width sweep: one fp16-accumulation copy per base
            # configuration (T.gemm on an fp16 accumulator fragment); its own
            # interpretation supplies the reference.
            from src.workflow.generator.identities import precision_program
            transformed = precision_program(program)
            if transformed is not None:
                for i, (threads, stages, passes) in enumerate(configurations):
                    variants.append((
                        ExtendedLowering(transformed, f'extended_{i}_prec', observe=True,
                                         threads=threads, stages=stages),
                        {'threads': threads, 'stages': stages, 'pass_configs': passes,
                         'precision': 'fp16'}))
        if program.identity_pair and config.extended_identity_pair:
            # Algebraic-identity sweep: a distributivity copy of the program
            # on the base configuration; its own interpretation supplies the
            # reference. Matmul programs and patterns without a match get none.
            from src.workflow.generator.identities import identity_variant
            distributed = identity_variant(program)
            if distributed is not None:
                threads, stages, passes = configurations[0]
                variants.append((
                    ExtendedLowering(distributed, 'extended_0_ident', observe=True,
                                     threads=threads, stages=stages),
                    {'threads': threads, 'stages': stages, 'pass_configs': passes,
                     'identity': True}))
        return variants

    def extended_compile_source(self, entries, program):
        from .extended import compile_source
        return compile_source(entries, program)


    def filter_block_k(self, values, dtype):
        return [v for v in values if v in self.params.valid_block_k(dtype)]

    def valid_mutated_block_k(self, value, dtype):
        return value in self.params.valid_block_k(dtype)

    def schedule_supported(self, block_m, block_n, threads, policy="square"):
        return self.params.check_warp_partition(block_m, block_n, threads, policy)

    def region_warp_policies(self, program):
        # MLIRSmith-style warp-partition sweep: GemmWarpPolicy variants of the
        # same gemm (FullRow splits along M, FullCol along N). The per-tile
        # math is untouched, so all variants share the one reference. This is
        # the sm_89-active warp-level knob — tilelang's warp *specialization*
        # itself is TMA-gated (sm_90+), reachable there via the
        # tl.disable_warp_specialized pass key in knobs.py.
        return ('full_row', 'full_col')

    def make_emitter(self, config=None):
        from .emitter import TileLangEmitter
        return TileLangEmitter(config, backend=self.name)


    def native_region(self, program):
        from .region import tilelang_code
        return tilelang_code(program), 'compiled = make_kernel()', 'compiled(A, B, C)'

    def region_pass_options(self, program, config):
        # Deterministic 1-3 key subset of the curated numerically-neutral
        # region pool; pure in (program, config) like extended sampling.
        from src.backends.common.knobs import region_pass_configs
        return {'pass_configs': region_pass_configs(program, config)}

    def region_swizzle_options(self, program):
        # Threadblock rasterization swizzle (panel size 10, row order): the
        # standard tilelang idiom, numerically neutral, gemm-gated upstream.
        return {'swizzle': {'panel_size': 10, 'order': 'row'}}

    def checked_region(self, program, index, options=None):
        from .region import tilelang_code
        options = options or {}
        name = f'make_kernel_variant_{index}'
        pass_configs = options.get('pass_configs')
        swizzle = options.get('swizzle')
        # A pass-config variant compiles the same kernel source through a
        # different pipeline: the jit decorator carries the configs into
        # tilelang.compile, so the prepare stays identical to the base.
        decorator = (f'@tilelang.jit(pass_configs={pass_configs!r})' if pass_configs
                     else '@tilelang.jit')
        code = tilelang_code(program, name, decorator=decorator,
                             swizzle=swizzle is not None)
        return code, f'def prepare_{index}(A, B):\n    compiled = {name}()\n    return lambda C: compiled(A, B, C)\n'

    def typed_lowering(self, program, name, suffix='', options=None):
        from .typed import TypedLowering
        return TypedLowering(program, self.name, name, suffix,
                             decorator=(f'@tilelang.jit(pass_configs={options["pass_configs"]!r})'
                                        if options and options.get('pass_configs') else '@tilelang.jit'),
                             swizzle=bool(options and options.get('swizzle')))

    def typed_prepare(self, program, name, argv, options=None):
        return f'    kernel = {name}()'

    def probe_code(self, kernel, a, b, outsize):
        from .probes import _tilelang
        return _tilelang(kernel, a, b, outsize)

    def probe_launch(self, k, threads):
        return [f'    def instantiate_{threads}():',
                f'        compiled = make_{k.name}_probe({threads})',
                '        return lambda out: compiled(a[0], b[0], out)',
                f'    launches.append(instantiate_{threads})']


BACKEND = TileLangBackend()
