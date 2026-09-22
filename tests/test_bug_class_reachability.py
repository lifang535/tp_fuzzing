"""Regression tests for the historical bug-class triggers restored on 2026-09-18.

GPU crashes themselves are exercised by tests/test_dtype_mismatch.py and manual
campaign probes; these tests lock in the emission/sampling invariants that make
dtype_mismatch, ptx_async_boundary, warp_partition, and shared_memory_overflow
reachable from the fuzzer's fresh-generation path.
"""
import ast
import copy
import random
import unittest

from src.config import Config
from src.ir import DataType, TileKernel, ComputeKind, LoopKind
from src.ir.region import Region, RegionProgram, Operation, RegionExecution
from src.workflow.generator import ProgramGenerator
from src.workflow.oracle import Oracle


def typed_gemm(dtype='float16', loop=LoopKind.PIPELINED, stages=2, M=65, N=64, K=64,
               bM=16, bN=32, bK=16, threads=128, layout='contiguous'):
    spec = TileKernel('kernel_0', M=M, N=N, K=K, block_M=bM, block_N=bN, block_K=bK,
                      threads=threads, num_stages=stages, dtype=DataType(dtype),
                      compute_kind=ComputeKind.GEMM, loop_kind=loop)
    program = RegionProgram(spec, Region([], [Operation('gemm', 'matmul', []),
                                               Operation('cast', 'c1', ['matmul'], {'dtype': 'float32'}),
                                               Operation('copy', 'v_out', ['c1'])], 'v_out'))
    program.typed = True
    program.execution = RegionExecution(input_layout_a=layout, input_layout_b=layout)
    program.validate()
    return program


def impl_source(code):
    for node in ast.walk(ast.parse(code)):
        if isinstance(node, ast.FunctionDef) and node.name == 'impl':
            return ast.unparse(node)
    raise AssertionError('no impl function in emitted code')


class BugClassReachabilityTests(unittest.TestCase):
    def test_typed_dtype_only_variants_share_impl_source(self):
        """dtype_mismatch: dtype-mutated programs must collide in tilelang's frontend cache."""
        oracle = Oracle(Config(), 'tilelang')
        p = typed_gemm(dtype='float16')
        q = copy.deepcopy(p)
        q.spec.dtype = DataType('float32')
        q.validate()
        code1, code2 = oracle._emit_code(p), oracle._emit_code(q)
        self.assertIn('dtype = "float16"', code1)
        self.assertIn('dtype = "float32"', code2)
        self.assertEqual(impl_source(code1), impl_source(code2))

    def test_typed_contiguous_gemm_uses_logical_copy(self):
        """ptx_async_boundary: contiguous typed gemms must take the T.copy lowering."""
        code = Oracle(Config(), 'tilelang')._emit_code(typed_gemm())
        self.assertIn('T.copy(A[by * 16, ki * 16], As)', code)
        self.assertIn('T.copy(B[ki * 16, bx * 32], Bs)', code)
        self.assertIn('T.Buffer((65, 64), dtype)', code)  # 2D logical A
        self.assertNotIn('T.cast(0, dtype)', impl_source(code))  # guarded loads only for physical

    def test_typed_physical_layout_keeps_guarded_flat_loads(self):
        """Physical layouts still need flat storage and guarded loads (no OOB)."""
        code = Oracle(Config(), 'tilelang')._emit_code(typed_gemm(layout='strided'))
        self.assertIn('T.cast(0, dtype)', code)
        self.assertNotIn('T.copy(A[by *', code)
        self.assertIn('T.Buffer((', code)

    def test_unchecked_sampling_restores_invalid_schedule_domain(self):
        """warp_partition/shared_memory_overflow: unchecked mode draws combos the
        checked filters exclude, and checked mode never does."""
        from src.backends import get_backend
        adapter = get_backend('tilelang')
        body = typed_gemm().body
        unchecked, checked = 0, 0
        for mode, cfg in ((True, Config(unchecked_spec_prob=1.0, seed=11)),
                          (False, Config(unchecked_spec_prob=0.0, seed=11))):
            gen = ProgramGenerator(cfg, 'tilelang')
            invalid = 0
            for _ in range(80):
                spec = adapter.sample_region_spec(gen, body, 'gemm')
                if not adapter.schedule_supported(spec.block_M, spec.block_N, spec.threads):
                    invalid += 1
            (unchecked if mode else checked)  # noqa
            if mode:
                unchecked = invalid
            else:
                checked = invalid
        self.assertGreater(unchecked, 0)
        self.assertEqual(checked, 0)

    def test_boundary_shape_bias_draws_shapes_below_tile(self):
        """ptx_async_boundary trigger needs M < block_M; the bias must provide it."""
        adapter = __import__('src.backends', fromlist=['get_backend']).get_backend('tilelang')
        gen = ProgramGenerator(Config(boundary_shape_prob=1.0, unchecked_spec_prob=0.0, seed=5), 'tilelang')
        body = typed_gemm().body
        small = 0
        for _ in range(60):
            spec = adapter.sample_region_spec(gen, body, 'gemm')
            if spec.M < spec.block_M:
                small += 1
        self.assertEqual(small, 60)

    def test_region_sampler_repair_mode_stays_valid(self):
        """The dtype-mutation repair demands a validated schedule even with
        unchecked sampling enabled globally."""
        from src.backends import get_backend
        adapter = get_backend('tilelang')
        gen = ProgramGenerator(Config(unchecked_spec_prob=1.0, seed=9), 'tilelang')
        body = typed_gemm().body
        spec = adapter.sample_region_spec(gen, body, 'gemm', unchecked_ok=False)
        self.assertTrue(adapter.schedule_supported(spec.block_M, spec.block_N, spec.threads))
        self.assertTrue(adapter.params.check_shared_memory(
            spec.block_M, spec.block_N, spec.block_K, spec.dtype, spec.num_stages))

    def test_boundary_branch_completes_ptx_trigger_shape(self):
        """Inside the boundary branch, gemm specs get the full historical
        ptx_async trigger shape: a one-element M or K tail (GPU-verified:
        M=1 or K=1 crashes the cp.async byte-width check, other tails pass),
        fp16 preference, pipelined."""
        from src.backends import get_backend
        adapter = get_backend('tilelang')
        gen = ProgramGenerator(Config(boundary_shape_prob=1.0, unchecked_spec_prob=0.0,
                                       seed=21), 'tilelang')
        body = typed_gemm().body
        specs = [adapter.sample_region_spec(gen, body, 'gemm') for _ in range(60)]
        self.assertTrue(all(s.M < s.block_M or s.K == 1 for s in specs))
        self.assertTrue(all(s.K % s.block_K != 0 for s in specs))
        self.assertGreater(sum(s.M == 1 or s.K == 1 for s in specs), 40)
        self.assertGreater(sum(s.dtype == DataType('float16') for s in specs), 36)
        self.assertGreater(sum(s.loop_kind == LoopKind.PIPELINED for s in specs), 30)

    def test_boundary_branch_keeps_pinned_repair_dtype(self):
        """The mutation repair path pins dtype; the boundary branch must not
        re-roll it even when it fires."""
        from src.backends import get_backend
        adapter = get_backend('tilelang')
        gen = ProgramGenerator(Config(boundary_shape_prob=1.0, unchecked_spec_prob=1.0,
                                       seed=13), 'tilelang')
        body = typed_gemm().body
        for _ in range(40):
            spec = adapter.sample_region_spec(gen, body, 'gemm', dtype='float32',
                                              unchecked_ok=False)
            self.assertEqual(spec.dtype, DataType('float32'))

    def test_probe_impl_source_is_dtype_free(self):
        """Probes bind dtype at module scope so dtype-only probe variants collide too."""
        from src.backends.common.probes import probe_program
        k = TileKernel('probe', compute_kind=ComputeKind.COPY, coverage_probe=True,
                       M=33, N=33, K=33, dtype=DataType.FLOAT16)
        p = probe_program(k)
        code = Oracle(Config(), 'tilelang')._emit_code(p)
        self.assertIn('dtype = "float16"', code)
        self.assertIn('T.Buffer(', code)
        # Every float16 literal in the probe impl belongs to the module binding.
        for node in ast.walk(ast.parse(code)):
            if isinstance(node, ast.FunctionDef) and node.name == 'impl':
                self.assertNotIn('float16', ast.unparse(node))

    def test_stage_sweep_lists_only_checked_alternate_stages(self):
        """MLIRSmith-style schedule sweep: one variant per alternate num_stages
        that passes schedule and shared-memory checks, nothing else."""
        from src.backends import get_backend
        adapter = get_backend('tilelang')
        p = typed_gemm(stages=2)
        p.execution.schedule_pair = False
        p.execution.stage_sweep = True
        p.execution.loop_sweep = False
        p.validate()
        spec = p.spec
        expected = [s for s in Config().pipeline_stages_choices if s != spec.num_stages
                    and adapter.schedule_supported(spec.block_M, spec.block_N, spec.threads)
                    and adapter.params.check_shared_memory(spec.block_M, spec.block_N,
                                                           spec.block_K, spec.dtype, s)]
        self.assertGreaterEqual(len(expected), 1)
        variants = adapter.region_variants(p, Config())
        self.assertEqual([v.spec.num_stages for v in variants], [spec.num_stages] + expected)
        for variant in variants[1:]:
            self.assertEqual((variant.spec.block_M, variant.spec.block_N, variant.spec.block_K),
                             (spec.block_M, spec.block_N, spec.block_K))
            self.assertEqual(variant.spec.threads, spec.threads)
            self.assertEqual(variant.spec.loop_kind, spec.loop_kind)

    def test_stage_sweep_skips_stages_that_overflow_shared_memory(self):
        """Big tiles: only the checked alternate stages may be swept."""
        from src.backends import get_backend
        adapter = get_backend('tilelang')
        p = typed_gemm(stages=1, bM=64, bN=64, bK=64, dtype='float32')
        p.execution.schedule_pair = False
        p.execution.stage_sweep = True
        p.execution.loop_sweep = False
        p.validate()
        variants = adapter.region_variants(p, Config())
        self.assertLess(len(variants) - 1, len(Config().pipeline_stages_choices) - 1)
        for variant in variants[1:]:
            self.assertTrue(adapter.params.check_shared_memory(
                variant.spec.block_M, variant.spec.block_N, variant.spec.block_K,
                variant.spec.dtype, variant.spec.num_stages))

    def test_loop_sweep_serial_variant_emits_serial_loop(self):
        """The PIPELINED -> SERIAL variant must actually lower to T.serial."""
        from src.backends import get_backend
        p = typed_gemm(loop=LoopKind.PIPELINED, stages=2)
        p.execution.schedule_pair = False
        p.execution.stage_sweep = False
        p.execution.loop_sweep = True
        p.validate()
        adapter = get_backend('tilelang')
        variants = adapter.region_variants(p, Config())
        self.assertEqual([v.spec.loop_kind for v in variants], [LoopKind.PIPELINED, LoopKind.SERIAL])
        self.assertEqual(variants[-1].spec.num_stages, 1)
        code = Oracle(Config(), 'tilelang')._emit_code(p)
        self.assertIn('T.Pipelined(4, num_stages=2)', code)
        self.assertIn('T.serial(4)', code)

    def test_loop_sweep_serial_base_gets_pipelined_variant(self):
        """SERIAL -> PIPELINED keeps the current stages when they pass the check."""
        from src.backends import get_backend
        p = typed_gemm(loop=LoopKind.SERIAL, stages=1)
        p.execution.schedule_pair = False
        p.execution.stage_sweep = False
        p.execution.loop_sweep = True
        p.validate()
        adapter = get_backend('tilelang')
        variants = adapter.region_variants(p, Config())
        self.assertEqual([v.spec.loop_kind for v in variants], [LoopKind.SERIAL, LoopKind.PIPELINED])
        self.assertEqual(variants[-1].spec.num_stages, 1)

    def test_load_entry_programs_skip_stage_and_loop_sweep(self):
        """num_stages/loop_kind only shape the gemm loop; recompiling identical
        load-entry kernels would waste compilation without new coverage."""
        from src.backends import get_backend
        spec = TileKernel('kernel_0', M=32, N=32, K=32, block_M=16, block_N=16, block_K=16,
                          threads=128, num_stages=2, dtype=DataType.FLOAT32,
                          compute_kind=ComputeKind.COPY, loop_kind=LoopKind.PIPELINED)
        program = RegionProgram(spec, Region([], [Operation('load', 'x'),
                                                  Operation('copy', 'out', ['x'])], 'out'))
        program.execution = RegionExecution(schedule_pair=False, stage_sweep=True, loop_sweep=True)
        program.validate()
        self.assertEqual(len(get_backend('tilelang').region_variants(program, Config())), 1)

    def test_schedule_sweep_shares_one_reference_and_labels_variants(self):
        """One reference per input_case drives every variant; the invariance
        labels separate the new knobs from the historical schedule pair."""
        p = typed_gemm(stages=2)
        p.execution.schedule_pair = False
        p.execution.stage_sweep = True
        p.execution.loop_sweep = True
        p.validate()
        code = Oracle(Config(), 'tilelang')._emit_code(p)
        self.assertEqual(code.count('ref = _typed_region_reference'), 1)
        self.assertIn('variant_kinds=', code)
        self.assertIn("'stage'", code)
        self.assertIn("'loop-kind'", code)

    def test_stage_markers_name_each_variant_on_stderr(self):
        """Location-aware root causes: reference, prepare_i and
        execute_variant_v markers must precede their steps on stderr."""
        from src.backends import get_backend
        p = typed_gemm(stages=2)
        code = Oracle(Config(), 'tilelang')._emit_code(p)
        ast.parse(code)
        for variant in range(len([v for v in get_backend('tilelang').region_variants(p)])):
            self.assertIn(f"print('TILESMITH_STAGE=prepare_{variant}', file=sys.stderr, flush=True)", code)
        self.assertIn("print('TILESMITH_STAGE=reference', file=sys.stderr, flush=True)", code)
        self.assertIn("print(f'TILESMITH_STAGE=execute_variant_{variant}', file=sys.stderr, flush=True)", code)

    def test_layout_sweep_emits_alternate_pair_only_for_physical_programs(self):
        """Layout sweep (RC1/RC3): physical programs run each input case against
        the primary pair and one alternate. Layouts are baked into the kernel
        source, so each pair compiles its own kernel set; contiguous programs
        keep the plain input_case loop, and layout_sweep off keeps the
        historical text."""
        p = typed_gemm(layout='offset')
        p.execution.layout_sweep = True
        p.validate()
        code = Oracle(Config(), 'tilelang')._emit_code(p)
        ast.parse(code)
        self.assertIn("print('TILESMITH_STAGE=layout:offset/offset', file=sys.stderr, flush=True)", code)
        self.assertIn("print('TILESMITH_STAGE=layout:offset/strided', file=sys.stderr, flush=True)", code)
        self.assertIn('_run_layout_case([run_variant_0_0], (a_storage, b_storage), ref, '
                      '3, 0.1, relative=True, layout_a=\'offset\', layout_b=\'offset\', '
                      'alternate=False, prepare_markers=[\'prepare_0_0\']', code)
        # The oracle trust gate ships both copies of the reference with every
        # checked call (fp64 self-check + one-ulp-jitter self-check).
        self.assertIn('reference_verify=lambda: _typed_region_reference_double(', code)
        self.assertIn('reference_jitter=lambda: _typed_region_reference(_ulp_jitter(A), _ulp_jitter(B)', code)
        self.assertIn('alternate=True, prepare_markers=[\'prepare_1_0\']', code)  # the alternate pair is relabeled
        self.assertIn('layout invariance', code)  # relabel lives in the embedded helper
        # One compiled kernel set per layout pair: global prepare indices.
        self.assertIn('typed_kernel_0', code)
        self.assertIn('typed_kernel_1', code)
        self.assertIn('def prepare_0(A, B)', code)
        self.assertIn('def prepare_1(A, B)', code)
        self.assertIn('*67+', code)  # offset row stride baked into pair 0
        self.assertIn('*131+', code)  # strided row stride baked into pair 1
        contiguous = typed_gemm(layout='contiguous')
        contiguous.execution.layout_sweep = True
        contiguous.validate()
        code = Oracle(Config(), 'tilelang')._emit_code(contiguous)
        self.assertNotIn('TILESMITH_STAGE=layout:', code)
        off = typed_gemm(layout='offset')
        off.execution.layout_sweep = False
        off.validate()
        code = Oracle(Config(), 'tilelang')._emit_code(off)
        self.assertNotIn('TILESMITH_STAGE=layout:', code)
        self.assertIn('_region_input_storage((65, 64), torch.float16, \'normal\', 0.1, \'offset\')', code)

    def test_emitted_harnesses_are_self_contained(self):
        """A harness must carry every helper it calls: missing definitions pass
        compile() but NameError at runtime, and a gate trip (the only moment
        _reference_stable runs) then lands as 'other' noise instead of a
        wrong_result/oracle_unstable verdict. symtable resolves each nested
        function's global references against the module scope."""
        import builtins
        import symtable
        cases = []
        native_plain = RegionProgram(
            TileKernel('kernel_0', M=65, N=64, K=64, block_M=16, block_N=32, block_K=16,
                       threads=128, num_stages=2, dtype=DataType('float16'),
                       compute_kind=ComputeKind.GEMM),
            Region([], [Operation('gemm', 'matmul', []), Operation('copy', 'v_out', ['matmul'])], 'v_out'))
        native_plain.validate()
        native_checked = RegionProgram(
            TileKernel('kernel_0', M=65, N=64, K=64, block_M=16, block_N=32, block_K=16,
                       threads=128, num_stages=2, dtype=DataType('float16'),
                       compute_kind=ComputeKind.GEMM),
            Region([], [Operation('gemm', 'matmul', []), Operation('copy', 'v_out', ['matmul'])], 'v_out'))
        native_checked.execution = RegionExecution(input_pattern='normal', input_seed_count=1, repeat_count=1)
        native_checked.validate()
        typed = typed_gemm()
        sweep = typed_gemm(layout='offset')
        sweep.execution.layout_sweep = True
        sweep.validate()
        for backend in ('triton', 'tilelang'):
            cases += [(f'{backend} native plain', backend, native_plain),
                      (f'{backend} native checked', backend, native_checked),
                      (f'{backend} typed checked', backend, typed),
                      (f'{backend} typed sweep', backend, sweep)]
        builtins_names = set(dir(builtins)) | {'__name__'}
        for name, backend, program in cases:
            with self.subTest(case=name):
                code = Oracle(Config(), backend)._emit_code(program)
                ast.parse(code)
                table = symtable.symtable(code, '<harness>', 'exec')
                defined = set(table.get_identifiers()) | builtins_names
                unresolved = []
                def walk(scope):
                    unresolved.extend(n for n in scope.get_globals() if n not in defined)
                    for child in scope.get_children():
                        walk(child)
                for child in table.get_children():
                    walk(child)
                self.assertEqual(sorted(set(unresolved)), [], name)

    def test_layout_sweep_dedupes_when_primary_is_the_gemm_alternate(self):
        """Primary pair == gemm alternate runs a single pair: every failure
        keeps its historical label instead of being re-raised."""
        p = typed_gemm(layout='offset')
        p.execution.input_layout_b = 'strided'
        p.execution.layout_sweep = True
        p.validate()
        code = Oracle(Config(), 'tilelang')._emit_code(p)
        self.assertIn("print('TILESMITH_STAGE=layout:offset/strided', file=sys.stderr, flush=True)", code)
        self.assertNotIn('layout:offset/offset', code)
        self.assertNotIn('alternate=True', code)
        self.assertNotIn('def prepare_1(A, B)', code)  # only one pair's kernels

    def test_load_entry_layout_sweep_picks_offset_or_strided_alternate(self):
        """Load-entry programs sweep A against the offset/strided alternate
        (strided when the primary is offset), B stays contiguous."""
        spec = TileKernel('kernel_0', M=32, N=32, K=32, block_M=16, block_N=16, block_K=16,
                          threads=128, num_stages=2, dtype=DataType.FLOAT32,
                          compute_kind=ComputeKind.COPY, loop_kind=LoopKind.PIPELINED)
        program = RegionProgram(spec, Region([], [Operation('load', 'x'),
                                                  Operation('copy', 'out', ['x'])], 'out'))
        program.execution = RegionExecution(input_layout_a='offset', layout_sweep=True)
        program.validate()
        code = Oracle(Config(), 'tilelang')._emit_code(program)
        ast.parse(code)
        self.assertIn("print('TILESMITH_STAGE=layout:offset/contiguous', file=sys.stderr, flush=True)", code)
        self.assertIn("print('TILESMITH_STAGE=layout:strided/contiguous', file=sys.stderr, flush=True)", code)
        strided = RegionProgram(spec, Region([], [Operation('load', 'x'),
                                                   Operation('copy', 'out', ['x'])], 'out'))
        strided.execution = RegionExecution(input_layout_a='strided', layout_sweep=True)
        strided.validate()
        code = Oracle(Config(), 'tilelang')._emit_code(strided)
        self.assertIn("print('TILESMITH_STAGE=layout:strided/contiguous', file=sys.stderr, flush=True)", code)
        self.assertIn("print('TILESMITH_STAGE=layout:offset/contiguous', file=sys.stderr, flush=True)", code)

    def test_thread_variant_keeps_historical_schedule_label(self):
        """schedule invariance -> schedule_mismatch is a historical label: the
        thread pair must keep kind 'schedule' while new knobs get new kinds."""
        from dataclasses import replace
        from src.backends.common.region_emitter import _variant_kinds
        p = typed_gemm(stages=2)
        base = p.spec
        variants = [p,
                    replace(p, spec=replace(base, threads=256)),
                    replace(p, spec=replace(base, num_stages=3)),
                    replace(p, spec=replace(base, loop_kind=LoopKind.SERIAL, num_stages=1))]
        options = [{}, {'threads': 256}, {'num_stages': 3}, {'loop_kind': 'serial'}]
        self.assertEqual(_variant_kinds(variants, options, base), ['schedule', 'schedule', 'stage', 'loop-kind'])

    def test_compilation_knob_variants_reuse_base_spec_and_get_own_labels(self):
        """Pass-config and swizzle pairs keep the base program's spec (their
        knobs are compilation options, not source changes) and are labeled by
        options, never by the spec-diff fallback."""
        from src.backends import get_backend
        from src.backends.common.region_emitter import _variant_kinds
        from src.backends.common.knobs import TILELANG_REGION_PASS_POOL
        p = typed_gemm(stages=2)
        p.execution.schedule_pair = False
        p.execution.stage_sweep = False
        p.execution.loop_sweep = False
        p.execution.pass_sweep = True
        p.execution.swizzle_sweep = True
        p.validate()
        adapter = get_backend('tilelang')
        pairs = adapter._region_variant_pairs(p, Config())
        variants = [v for v, _ in pairs]
        options = [o for _, o in pairs]
        self.assertEqual([v.spec for v in variants], [p.spec] * len(variants))
        self.assertEqual(_variant_kinds(variants, options, p.spec),
                         ['schedule', 'pass-config', 'swizzle'])
        self.assertGreaterEqual(len(options[1]['pass_configs']), 1)
        self.assertTrue(set(options[1]['pass_configs']) <= set(TILELANG_REGION_PASS_POOL))
        self.assertEqual(options[2], {'swizzle': {'panel_size': 10, 'order': 'row'}})

    def test_pass_config_pair_emits_jit_decorator_with_configs(self):
        """The tilelang pass-config variant compiles the same kernel source
        through tilelang.jit(pass_configs=...) — the decorator must carry the
        sampled configs and the prepare must stay a plain call."""
        p = typed_gemm(stages=2)
        p.execution.schedule_pair = False
        p.execution.stage_sweep = False
        p.execution.loop_sweep = False
        p.execution.pass_sweep = True
        p.validate()
        code = Oracle(Config(), 'tilelang')._emit_code(p)
        ast.parse(code)
        # The '@' must be present: without it the line is a bare expression
        # statement and the variant function is left undecorated, so the
        # prepare returns the raw PrimFunc instead of a compiled kernel.
        self.assertIn('@tilelang.jit(pass_configs={', code)
        self.assertRegex(code, r'@tilelang\.jit\(pass_configs=\{[^)]*\}\)\s*\n\s*def typed_kernel_1\(\):')
        self.assertIn('def typed_kernel_1():', code)
        self.assertIn('kernel = typed_kernel_1()', code)
        self.assertIn("'pass-config'", code)
        self.assertIn('variant_kinds=', code)

    def test_triton_pass_config_pair_launches_with_fp_fusion(self):
        """The triton pass-config variant toggles enable_fp_fusion on the
        launch while the base launch keeps its pinned-off historical text."""
        p = typed_gemm(stages=2)
        p.execution.schedule_pair = False
        p.execution.stage_sweep = False
        p.execution.loop_sweep = False
        p.execution.pass_sweep = True
        p.validate()
        code = Oracle(Config(), 'triton')._emit_code(p)
        ast.parse(code)
        self.assertIn('num_warps=4, enable_fp_fusion=True)', code)
        self.assertIn('num_warps=4, enable_fp_fusion=False)', code)

    def test_swizzle_pair_emits_use_swizzle_in_gemm_source(self):
        """The swizzle variant annotates the gemm kernel with T.use_swizzle
        and the plain variants do not."""
        p = typed_gemm(stages=2)
        p.execution.schedule_pair = False
        p.execution.stage_sweep = False
        p.execution.loop_sweep = False
        p.execution.swizzle_sweep = True
        p.validate()
        code = Oracle(Config(), 'tilelang')._emit_code(p)
        ast.parse(code)
        self.assertEqual(code.count('T.use_swizzle(panel_size=10, order="row")'), 1)
        self.assertIn("'swizzle'", code)
        # Triton has no swizzle knob: the pair must not appear there.
        triton_code = Oracle(Config(), 'triton')._emit_code(p)
        self.assertNotIn('swizzle invariance', triton_code)
        self.assertNotIn("'swizzle'", triton_code)


class StepOpNoiseTrapTests(unittest.TestCase):
    """fp32 GEMM + boundary step ops are oracle noise (TF32 kernel math vs
    exact-fp32 reference flips ceil/floor/round/cast boundaries by one ulp on
    ~1% of elements), so generation re-rolls them and mutation never flips a
    gemm+step program to fp32."""

    def _program(self, dtype, kinds, typed=False, functions=()):
        spec = TileKernel('kernel_0', M=65, N=64, K=64, block_M=16, block_N=32,
                          block_K=16, threads=128, num_stages=2,
                          dtype=DataType(dtype), compute_kind=ComputeKind.GEMM,
                          loop_kind=LoopKind.PIPELINED)

        def chain(kinds, entry=True):
            # Validation forbids load/gemm anywhere inside helper bodies, so
            # function chains start from the argument instead of a gemm entry.
            ops = [Operation('gemm', 'v1', [])] if entry else []
            for i, kind in enumerate(kinds):
                source = ops[-1].result if ops else 'arg0'
                ops.append(Operation(kind, f'v{i + 2}', [source],
                                     {'dtype': 'float16'} if kind in ('round', 'cast') else {}))
            return Region([], ops, ops[-1].result)

        from src.ir.region import Function
        program = RegionProgram(spec, chain(kinds))
        program.functions = []
        for i, fn_kinds in enumerate(functions):
            body = chain(fn_kinds, entry=False)
            body.arguments = ['arg0']
            program.functions.append(Function(f'fn_{i}', body))
        program.typed = typed
        program.execution = RegionExecution(input_layout_a='contiguous', input_layout_b='contiguous')
        program.validate()
        return program

    def test_predicate_matrix(self):
        from src.workflow.generator.region_generator import step_op_noise_trap
        self.assertTrue(step_op_noise_trap(self._program('float32', ['ceil'])))
        self.assertTrue(step_op_noise_trap(self._program('float32', ['copy', 'floor'])))
        # Typed programs reject a cast-to-fp16 entry output ("Entry output must
        # be a full float32 tile"), so exercise the typed branch with round,
        # whose dtype attribute leaves the inferred type untouched.
        self.assertTrue(step_op_noise_trap(self._program('float32', ['round'], typed=True)))
        self.assertFalse(step_op_noise_trap(self._program('float16', ['ceil'])))
        self.assertFalse(step_op_noise_trap(self._program('float32', ['copy', 'exp'])))
        # Coarse any-any semantics: a step op anywhere in a GEMM program is
        # treated as trapped, even when it sits in an unrelated function.
        self.assertTrue(step_op_noise_trap(self._program('float32', [], functions=[['ceil']])))

    def test_generation_never_emits_trap_programs(self):
        from src.workflow.generator.region_generator import RegionGenerator, step_op_noise_trap
        for backend in ('tilelang', 'triton'):
            for typed in (0, 1):
                config = Config(coverage_probe_prob=0, region_int8_prob=0,
                                region_typed_prob=typed, region_layout_prob=0)
                for seed in range(20):
                    random.seed(seed)
                    program = RegionGenerator(config, backend).generate()
                    self.assertFalse(step_op_noise_trap(program),
                                     f'{backend} typed={typed} seed={seed} emitted a trap program')

    def test_mutation_never_introduces_trap_programs(self):
        from src.workflow.mutator import Mutator
        from src.workflow.generator.region_generator import step_op_noise_trap
        program = self._program('float16', ['ceil'])
        config = Config(dtype_mutate_prob=1)
        for _ in range(20):
            mutated = Mutator(config, 'tilelang').mutate(copy.deepcopy(program))
            self.assertFalse(step_op_noise_trap(mutated))


if __name__ == '__main__':
    unittest.main()
