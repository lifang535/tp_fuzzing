"""Triton target rules, emitters, and launch conventions."""
from src.backends.common.builtin import CudaTileBackend
from . import params


class TritonBackend(CudaTileBackend):
    name = 'triton'
    params = params
    imports = 'import triton\nimport triton.language as tl'
    min_block_k = 16

    def extended_variants(self, program, config=None):
        from .extended import ExtendedLowering
        if config is None:
            from src.config import DEFAULT_CONFIG
            config = DEFAULT_CONFIG
        configurations = [(4, 1, {})]
        if program.configuration_pair and config.extended_config_depth >= 1:
            configurations.append((8, 2, {}))
        if program.pass_config_pair and config.extended_config_depth >= 2:
            # The base configurations pin enable_fp_fusion=False, so the fused
            # pair runs a genuinely different compilation pipeline.
            configurations.append((4, 1, {'enable_fp_fusion': True}))
        variants = [(ExtendedLowering(program, f'extended_{i}_{int(observe)}', observe, stages),
                     {'num_warps': warps, 'num_stages': stages, 'enable_fp_fusion': False, **passes})
                    for i, (warps, stages, passes) in enumerate(configurations)
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
                                     options['num_stages']),
                    options))
        if program.precision_pair and config.extended_precision_pair:
            # Accumulator-width sweep: an fp16-accumulation copy per base
            # configuration (its own interpretation supplies the reference),
            # plus one tf32 input-precision variant on the base configuration
            # (tf32 error ~5e-4 is far below the matmul tolerance).
            from src.workflow.generator.identities import precision_program
            transformed = precision_program(program)
            if transformed is not None:
                for i, (warps, stages, passes) in enumerate(configurations):
                    variants.append((
                        ExtendedLowering(transformed, f'extended_{i}_prec', observe=True, stages=stages),
                        {'num_warps': warps, 'num_stages': stages, 'enable_fp_fusion': False,
                         **passes, 'precision': 'fp16'}))
                variants.append((
                    ExtendedLowering(program, 'extended_tf32', observe=False, stages=1,
                                     input_precision='tf32'),
                    {'num_warps': 4, 'num_stages': 1, 'enable_fp_fusion': False,
                     'input_precision': 'tf32'}))
        if program.identity_pair and config.extended_identity_pair:
            # Algebraic-identity sweep: a distributivity copy of the program
            # on the base configuration; its own interpretation supplies the
            # reference. Matmul programs and patterns without a match get none.
            from src.workflow.generator.identities import identity_variant
            distributed = identity_variant(program)
            if distributed is not None:
                variants.append((
                    ExtendedLowering(distributed, 'extended_0_ident', observe=True, stages=1),
                    {'num_warps': 4, 'num_stages': 1, 'enable_fp_fusion': False,
                     'identity': True}))
        return variants

    def extended_compile_source(self, entries, program):
        from .extended import compile_source
        return compile_source(entries, program)


    def filter_block_k(self, values, dtype):
        return values

    def valid_mutated_block_k(self, value, dtype):
        return value >= 16

    def schedule_supported(self, block_m, block_n, threads, policy="square"):
        return True

    def make_emitter(self, config=None):
        from .emitter import TritonEmitter
        return TritonEmitter(config, backend=self.name)


    def native_region(self, program):
        from .region import triton_code
        p = program.spec
        launch = f'kernel[({(p.M+p.block_M-1)//p.block_M}, {(p.N+p.block_N-1)//p.block_N})](A, B, C, num_warps={p.threads//32})'
        return triton_code(program), '', launch

    def region_pass_options(self, program, config):
        # The base launch leaves enable_fp_fusion at triton's default (off);
        # the pair runs the fused pipeline on the same kernel source.
        return {'enable_fp_fusion': True}

    def checked_region(self, program, index, options=None):
        from .region import triton_code
        p = program.spec
        name = f'kernel_variant_{index}'
        code = triton_code(program, name, suffix=f'_variant_{index}')
        fusion = ', enable_fp_fusion=True' if options and options.get('enable_fp_fusion') else ''
        launch = (f'{name}[({(p.M+p.block_M-1)//p.block_M}, {(p.N+p.block_N-1)//p.block_N})]'
                  f'(A, B, C, num_warps={p.threads // 32}{fusion})')
        return code, f'def prepare_{index}(A, B):\n    return lambda C: {launch}\n'

    def typed_lowering(self, program, name, suffix='', options=None):
        from .typed import TypedLowering
        return TypedLowering(program, self.name, name, suffix)

    def typed_prepare(self, program, name, argv, options=None):
        p = program.spec
        tm, tn = (p.M+p.block_M-1)//p.block_M, (p.N+p.block_N-1)//p.block_N
        fusion = 'True' if options and options.get('enable_fp_fusion') else 'False'
        return f'    kernel = lambda {argv}: {name}[({tm}, {tn})]({argv}, num_warps={p.threads//32}, enable_fp_fusion={fusion})'

    def probe_code(self, kernel, a, b, outsize):
        from .probes import _triton
        return _triton(kernel, a, b)

    def probe_launch(self, k, threads):
        return [f'    def launch_{threads}(out):',
                f'        {k.name}_probe[({(k.M + k.block_M - 1) // k.block_M},)](a[0], b[0], out, num_warps={threads // 32})',
                f'    launches.append(lambda: launch_{threads})']


BACKEND = TritonBackend()
