"""Existing CUDA region/probe policies, shared by Triton and TileLang.

Other DSLs may implement Backend directly instead of inheriting this profile.
"""
import random
from dataclasses import replace
from src.backends.base import Backend


class CudaTileBackend(Backend):
    supports_extended = True
    min_block_k = 16
    imports = ''


    def sample_region_spec(self, generator, body, initial, functions=(), dtype=None, unchecked_ok=True):
        from src.ir import DataType, TileKernel, ComputeKind, LoopKind
        from src.ir.region import walk
        config = generator.config
        random_dtype = dtype is None
        dtype = generator.type_gen.random_dtype() if dtype is None else DataType(dtype)
        if dtype == DataType.INT8:
            return self._sample_int8_spec(generator)
        params = {}
        for dim in ('M', 'N', 'K'):
            params[dim] = random.choice(generator.type_gen.dim_pool)
        blocks = [v for v in config.tile_size_choices if v >= 16 and v & (v-1) == 0]
        ks = [v for v in config.block_k_choices if v >= self.min_block_k and v & (v-1) == 0]
        ks = self.filter_block_k(ks, dtype)
        tiles = [(m,n,k,t) for m in blocks for n in blocks for k in ks for t in config.thread_choices
                 if t in (128,256)]
        if unchecked_ok and random.random() < config.unchecked_spec_prob:
            # Historical trigger mode: sample the raw tile/thread/stage domain
            # without schedule/shared-memory pre-validation. The validation
            # filters below otherwise exclude every combination that reaches
            # the warp_partition and shared_memory_overflow bug classes.
            check = lambda *args: True
        else:
            tiles = [v for v in tiles if self.schedule_supported(v[0], v[1], v[3])]
            check = self.params.check_shared_memory
            tiles = [v for v in tiles if check(v[0],v[1],v[2],dtype,1)]
        if any(o.kind == 'tile_transpose' for region in [body] + [fn.body for fn in functions] for o in walk(region)):
            tiles = [v for v in tiles if v[0] == v[1]]
        if not tiles:
            raise ValueError('No valid region tiles in configured choices')
        params['block_M'], params['block_N'], params['block_K'], params['threads'] = random.choice(tiles)
        loop = random.choice(list(LoopKind))
        # Bias M below a full tile: the boundary copy lowerings this reaches
        # historically triggered ptx_async_boundary (tilelang cp.async
        # byte-width crash) and other tail-handling bugs that random dim pools
        # almost never hit. GEMM entries carry the cp.async trigger, so they
        # get twice the base probability.
        bias = config.boundary_shape_prob * (2 if initial == 'gemm' else 1)
        if random.random() < bias:
            params['M'] = random.randint(1, params['block_M'] - 1)
            if initial == 'gemm':
                # Complete the historical trigger shape. The cp.async
                # byte-width crash (GPU-verified grid probe) needs a tail of
                # exactly ONE element in M or K: M=1 crashes for every K tail,
                # and K=1 crashes for every M, while uniform sub-tile draws
                # (M=2..bM-1) compile fine. fp16 + pipelined are required,
                # so prefer them when dtype was randomly drawn.
                roll = random.random()
                if roll < 0.70:
                    params['M'] = 1
                elif roll < 0.85:
                    params['K'] = 1
                if params['K'] % params['block_K'] == 0:
                    params['K'] -= 1
                if random_dtype and random.random() < 0.5:
                    dtype = DataType('float16')
                if random.random() < 0.7:
                    loop = LoopKind.PIPELINED
        stages = [n for n in config.pipeline_stages_choices if n >= 1 and check(params['block_M'],params['block_N'],params['block_K'],dtype,n)]
        if not stages:
            raise ValueError('No valid pipeline stages')
        return TileKernel('kernel_0', **params, dtype=dtype, loop_kind=loop,
                          num_stages=random.choice(stages) if loop == LoopKind.PIPELINED else 1,
                          compute_kind=ComputeKind.GEMM if initial == 'gemm' else ComputeKind.COPY)

    def _sample_int8_spec(self, generator):
        """int8 x int8 GEMM spec from the pre-validated grid (block_K in
        {32, 64}, warp-partition-legal tiles, shared-memory-budget stages).
        The boundary-shape bias below would break the M,N >= 16 / K >= 32
        int8 constraints, so this branch never reaches it."""
        from src.ir import DataType, TileKernel, ComputeKind, LoopKind
        from src.workflow.generator.grids import INT8_SPEC_GRID
        grids = getattr(generator, 'grids', None)
        if grids is not None:
            cell = grids.next_cell('int8_region', generator.backend, INT8_SPEC_GRID)
        else:
            cell = random.choice(INT8_SPEC_GRID)
        return TileKernel('kernel_0', M=cell['m'], N=cell['n'], K=cell['k'],
                          block_M=cell['block_m'], block_N=cell['block_n'],
                          block_K=cell['block_k'], threads=cell['threads'],
                          dtype=DataType.INT8, loop_kind=LoopKind.PIPELINED,
                          num_stages=cell['stages'], compute_kind=ComputeKind.GEMM)

    def dtype_parameters_valid(self, spec):
        return self.valid_mutated_block_k(spec.block_K, spec.dtype) and self.params.check_shared_memory(
            spec.block_M, spec.block_N, spec.block_K, spec.dtype, spec.num_stages)

    def region_variants(self, program, config=None):
        """Deterministic schedule sweep around one program: the base variant,
        the historical 128/256-thread pair, then alternate num_stages and
        loop_kind configurations. All variants share one reference — the
        reference interpreter is schedule-independent.
        """
        return [variant for variant, _ in self._region_variant_pairs(program, config)]

    def region_variant_options(self, program, config=None):
        """Per-variant compilation options matching region_variants order."""
        return [options for _, options in self._region_variant_pairs(program, config)]

    def _region_variant_pairs(self, program, config=None):
        """(variant_program, options) pairs for the checked harness.

        options carries the per-variant compilation knobs that do not change
        the kernel source (pass_configs, enable_fp_fusion, swizzle); the
        variant program itself carries the source-changing schedule knobs
        (threads, num_stages, loop_kind). Backends add their own knobs through
        region_pass_options / region_swizzle_options.
        """
        from src.ir import LoopKind
        from src.ir.region import walk
        if config is None:
            from src.config import DEFAULT_CONFIG
            config = DEFAULT_CONFIG
        pairs = [(program, {})]
        execution = program.execution
        spec = program.spec
        if execution is None:
            return pairs
        if execution.schedule_pair:
            threads = 256 if spec.threads == 128 else 128
            if self.schedule_supported(spec.block_M, spec.block_N, threads):
                pairs.append((replace(program, spec=replace(spec, threads=threads)),
                              {'threads': threads}))
        # num_stages and loop_kind only shape the pipelined gemm loop; a sweep
        # of load-entry programs would recompile identical kernels.
        gemm = any(o.kind == 'gemm' for region in [program.body] + [fn.body for fn in program.functions]
                   for o in walk(region))
        if gemm:
            if execution.stage_sweep and spec.loop_kind == LoopKind.PIPELINED:
                for stages in config.pipeline_stages_choices:
                    if stages == spec.num_stages:
                        continue
                    if (self.schedule_supported(spec.block_M, spec.block_N, spec.threads)
                            and self.params.check_shared_memory(spec.block_M, spec.block_N, spec.block_K,
                                                                spec.dtype, stages)):
                        pairs.append((replace(program, spec=replace(spec, num_stages=stages)),
                                      {'num_stages': stages}))
            if execution.loop_sweep:
                if spec.loop_kind == LoopKind.PIPELINED:
                    # Serial always fits: there are no pipelined stages to share.
                    pairs.append((replace(program, spec=replace(spec, loop_kind=LoopKind.SERIAL, num_stages=1)),
                                  {'loop_kind': 'serial'}))
                else:
                    stages = [n for n in config.pipeline_stages_choices if n >= 1
                              and self.params.check_shared_memory(spec.block_M, spec.block_N, spec.block_K,
                                                                  spec.dtype, n)]
                    if stages:
                        num_stages = spec.num_stages if spec.num_stages in stages else max(stages)
                        pairs.append((replace(program, spec=replace(spec, loop_kind=LoopKind.PIPELINED,
                                                                    num_stages=num_stages)),
                                      {'loop_kind': 'pipelined'}))
        # Compilation-knob pairs: the variant program is the base program (the
        # kernel source is unchanged), so they never multiply with the sweep
        # tiers above and every pair shares the single reference.
        if execution.pass_sweep:
            pass_options = self.region_pass_options(program, config)
            if pass_options is not None:
                pairs.append((program, pass_options))
        if execution.swizzle_sweep and gemm:
            swizzle_options = self.region_swizzle_options(program)
            if swizzle_options is not None:
                pairs.append((program, swizzle_options))
        # Warp-partition policy pairs (MLIRSmith-style): the variant program
        # carries a non-square GemmWarpPolicy in its spec — source-changing,
        # like threads/stages — but the per-tile math is untouched, so all
        # policies share the one reference. Gated per-policy on the same
        # warp-partition feasibility check as the base configuration.
        if execution.warp_policy_sweep and gemm:
            for policy in self.region_warp_policies(program):
                if self.schedule_supported(spec.block_M, spec.block_N, spec.threads, policy):
                    pairs.append((replace(program, spec=replace(spec, warp_policy=policy)),
                                  {'warp_policy': policy}))
        return pairs

    def region_pass_options(self, program, config):
        """Backend hook: options for the pass-config invariance pair, or None
        when the backend has no such knob."""
        return None

    def region_swizzle_options(self, program):
        """Backend hook: options for the swizzle invariance pair, or None when
        the backend has no such knob."""
        return None

    def region_warp_policies(self, program):
        """Backend hook: the GemmWarpPolicy values to sweep as invariance
        variants (each replaces the spec's warp_policy). Empty for backends
        without a policy knob."""
        return ()

    def generate_probe(self, config):
        from .probes import generate_probe
        return generate_probe(config)

    def mutate_probe(self, program, config):
        from .probes import mutate_probe
        return mutate_probe(program, config)

    def classify_error(self, message):
        from src.workflow.oracle.oracle import BugType
        lower = message.lower()
        # Oracle trust gate: chaotic references are noise, not wrong results.
        if 'oracle unstable' in lower:
            return BugType.ORACLE_UNSTABLE
        if 'wrong result' in lower:
            return BugType.WRONG_RESULT
        if 'cuda' in lower and 'runtime' in lower:
            return BugType.RUNTIME_CRASH
        return BugType.COMPILE_CRASH
