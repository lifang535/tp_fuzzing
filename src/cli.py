"""Command line of main.py: arguments, resume inheritance, validation and Config.

A campaign records its generation settings in summary.json. Resuming it must
keep the generation distribution of the earlier segments, so every option
that changed default over time has two defaults: one for new campaigns and one
for a resumed campaign that did not record the option (it predates it).
"""
import argparse
import json
from pathlib import Path

from src.config import Config


def build_parser(backend_names) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="TileSmith: Tile Program Fuzzer")
    add = parser.add_argument
    toggle = argparse.BooleanOptionalAction

    add("-n", "--iterations", type=int, default=100,
        help="Number of NEW test cases to execute (dedup-skipped cases do not count). "
             "When resuming, the fuzzer continues until this many "
             "previously-unseen programs have been tested.")
    add("--probe-prob", type=float, default=0.20,
        help="Conditional probe probability within the Region route (use --extended-prob 0 --probe-prob 1 for probes only)")
    add("--dtype-mutate-prob", type=float, default=0.25,
        help="Probability of explicitly switching storage dtype during mutation")
    add("--gemm-prob", type=float, default=0.50,
        help="GEMM entry probability for fresh native regions (otherwise load)")
    add('--typed-op-prob', type=float, default=0.35,
        help='Probability of type/shape/memory operations in fresh regions; 0 generates v3')

    # Options whose default depends on whether a campaign is resumed (see INHERITED).
    add('--extended-prob', type=float, default=None,
        help='Fresh typed exploration programs: default 0.25 (0 when resuming an old campaign)')
    add('--dsl-extend-prob', type=float, default=None,
        help='Probability of deriving a target-DSL test from a passing common Extended case; '
             'default 0.35 for new campaigns, 0 for old campaigns on resume')
    add('--slice-prob', type=float, default=None,
        help='Probability that a test is a feature-slice program (dtype chains, reductions, '
             'scans, GEMM over bf16/fp8/fp64/int16/int64/unsigned dtypes); default 0.4 for '
             'new campaigns, 0 when resuming a campaign that predates slices')
    add('--slices', default=None,
        help='Comma-separated slice names (default: every slice the backend supports)')
    add('--slice-adaptive', action=toggle, default=None,
        help='Choose slices by Good-Turing discovery per second and candidates by uncovered '
             'knob pairs (default on); off samples slices and knobs uniformly')
    add('--dsl-evolve-prob', type=float, default=None,
        help='Within DSL tests, probability of evolving a passing target-specific seed '
             '(default 0.5; old campaigns resume with 0)')
    add('--dsl-max-depth', type=int, default=None,
        help='Maximum number of DSL derivation steps (default 3)')
    add('--dsl-source-variants', type=int, default=None,
        help='Bounded attempts per common parent / target operation '
             '(default 4; old campaigns resume with 1)')
    add('--dsl-adaptive-schedule', action=toggle, default=None,
        help='Adapt DSL actions to recent novelty, cost and audited repeats '
             '(default on; old campaigns resume with off unless explicitly enabled)')
    add('--dsl-matrix-prob', type=float, default=None,
        help='Probability that a DSL target call takes a rank-2 operand '
             '(default 0.5; old campaigns resume with 0)')
    add('--dsl-attributes', action=toggle, default=None,
        help='Draw the axis, direction, order and topk size of DSL target calls '
             '(default on; old campaigns resume with off)')
    add('--corpus-feedback', action=toggle, default=None,
        help='Preserve rare-feature representatives in bounded seed pools '
             '(default on; old campaigns resume with off)')

    add('--legacy-extended-mix', action='store_true',
        help='Restore the historical Extended generator that mixes DSL-specific shape operations into fresh cases')
    add('--compile-only', action='store_true',
        help='Compile extended programs without GPU execution; implies --extended-prob 1')
    add('--no-save-artifacts', action='store_true',
        help='Use temporary compilation evidence and delete it after each test instead of saving artifacts/')
    add('--no-save-passed-code', action='store_true',
        help='Save passing programs only as IR JSON, without their Python reproducers '
             '(about 80%% of passed/ disk use); failures always keep both')
    add('--no-extended-observations', action='store_true',
        help='Disable additional intermediate-output variants')
    add('--no-extended-configurations', action='store_true',
        help='Disable paired backend compilation configurations (alias for --extended-config-depth 0)')
    add('--extended-config-depth', type=int, choices=(0, 1, 2), default=1,
        help='Extended configuration sweep depth: 0 = single configuration; '
             '1 = threads/stages pair; 2 = additionally a second pass '
             'configuration (tl.disable_loop_unswitching / enable_fp_fusion)')
    add('--extended-fast-math', action='store_true',
        help='Additionally compile extended programs with tl.enable_fast_math '
             '(changes numerics; off by default)')
    add('--no-extended-precision', action='store_true',
        help='Disable the accumulator-width sweep: no fp16-accumulation copies of '
             'extended matmul programs and no triton ieee->tf32 variant (on by default)')
    add('--no-extended-identities', action='store_true',
        help='Disable the algebraic-identity sweep: no distributivity copies of '
             'extended matmul-less programs (on by default)')
    add('--random-config-count', type=int, default=2,
        help='Random pass-pipeline configurations sampled per extended program '
             '(deterministic per program + seed; 0 disables; max 8)')
    add('--no-instance-grids', action='store_true',
        help='Disable the per-(op, backend) round-robin instance grids; new op '
             'attribute corners are then sampled uniformly at random')
    add('--extended-atomic-prob', type=float, default=0.25,
        help='Probability of global-memory atomics per extended program')
    add('--extended-elementwise-prob', type=float, default=0.50,
        help='Probability of adding a shared float32 math operation to an extended program')
    add('--extended-fma-prob', type=float, default=0.30,
        help='Probability of scalar fused multiply-add per extended program')
    add('--extended-shape-op-prob', type=float, default=0.30,
        help='Probability of triton shape primitives per extended program')
    add('--extended-int8-prob', type=float, default=0.30,
        help='Probability that an extended matmul program uses int8 x int8')
    add('--region-int8-prob', type=float, default=0.15,
        help='Probability that a native region is an int8 GEMM-only program')
    add('--no-region-pass-config', action='store_true',
        help='Disable the region pass-configuration variant pair')
    add('--no-region-swizzle', action='store_true',
        help='Disable the tilelang T.use_swizzle region variant pair')
    add('--no-region-warp-policy', action='store_true',
        help='Disable the tilelang GemmWarpPolicy region variant pair (FullRow/FullCol)')
    add("--local-mutate-prob", type=float, default=0.35,
        help="Local mutation probability after skipping dtype mutation")
    add("--function-min-count", type=int, default=1,
        help="Minimum auxiliary functions per native program (default: 1)")
    add("--function-max-count", type=int, default=3,
        help="Maximum auxiliary functions per native program (default: 3, limit: 8)")
    add("--function-call-prob", type=float, default=0.30,
        help="Probability of replacing a non-control operation with a call when callees exist")
    add("--no-structural-feedback", action="store_true",
        help="Disable feature-guided selection for ablation experiments")
    add("--uncovered-boost", type=float, default=50.0,
        help="Additive weight boost for never-attempted structural features "
             "(MLIRSmith DiversityCriteria-style; 0 restores legacy weighting)")
    add("--quarantine", action=toggle, default=None,
        help="Redraw fresh/mutated candidates matching a learned rule of a recurring "
             "failure bucket (default: on; a resumed campaign keeps its setting)")
    add("--quarantine-retries", type=int, default=8,
        help="Redraws before a quarantined candidate is tested anyway (default: 8)")
    add("--explained-feedback", action=toggle, default=None,
        help="Treat features of duplicate failures as exercised in rarity weights "
             "(default: on; a resumed campaign keeps its setting)")
    add("--swarm-prob", type=float, default=None,
        help="Probability that a native region draws from a random half of the "
             "operations (default: 0.5; a resumed campaign keeps its setting)")
    add("--seed", type=int, default=None)
    add("--input-seed", type=int, default=0,
        help="Independent PyTorch tensor seed embedded in saved tests (default: 0)")
    add("--region-input-seeds", type=int, default=2,
        help="Input seeds per fresh native region (1..8, starting at --input-seed)")
    add("--region-repeat-count", type=int, default=3,
        help="Executions per input and thread configuration for fresh regions (1..8)")
    add("--no-region-schedule-pair", action="store_true",
        help="Disable paired 128/256-thread checks for fresh native regions")
    add("--no-region-stage-sweep", action="store_true",
        help="Disable the alternate num_stages sweep for fresh native regions")
    add("--no-region-loop-sweep", action="store_true",
        help="Disable the alternate loop_kind sweep for fresh native regions")
    add("--no-region-layout-sweep", action="store_true",
        help="Disable the alternate layout pair sweep for fresh physical regions")
    add("--region-layout-prob", type=float, default=0.35,
        help="Non-contiguous layout probability per used input of a fresh region")
    add("--unchecked-spec-prob", type=float, default=0.30,
        help="Probability that region spec sampling skips schedule/"
             "shared-memory pre-validation (keeps warp_partition and "
             "shared_memory_overflow generable)")
    add("--boundary-shape-prob", type=float, default=0.10,
        help="Probability of forcing M below block_M (x2 for GEMM "
             "entries) to reach ptx_async_boundary and tail bugs")
    add("-o", "--output", type=str, default="results")
    add("--dump", action="store_true", help="Print generated code without executing")
    add('--backend-plugin', action='append', default=[], metavar='MODULE',
        help='Import a Python module that registers a backend (repeatable)')
    add("--backend", type=str, default="tilelang", choices=backend_names())
    add("-q", "--quiet", action="store_true")
    add("--list-kernels", action="store_true", help="List all supported kernel kinds and exit")
    add("--easy-shape", action="store_true",
        help="Use power-of-two shapes (native: 1..16384; fresh probes: 16/32/64/128). "
             "Shapes smaller than a tile can still require boundary handling.")
    add("--resume", type=str, default=None,
        help="Resume fuzzing from a previous run directory. Pass the directory name "
             "(e.g., '2026.06.29-16.41_triton_easy-shape_seed=42') or full path. "
             "Historical results are loaded to avoid re-testing, and new results "
             "are appended to the same directory.")
    return parser


# (argument, default for a new campaign, default when a resumed campaign did
# not record it), in resolution order: a callable new-campaign default reads
# arguments resolved before it. The summary.json key is the argument name.
INHERITED = (
    ('extended_prob', 0.25, 0.0),
    ('dsl_extend_prob', lambda args: 0.0 if args.compile_only or args.extended_prob == 0 else 0.35, 0.0),
    ('dsl_evolve_prob', 0.5, 0.0),
    ('slice_prob', lambda args: 0.0 if args.compile_only or not _slices_available(args.backend) else 0.4, 0.0),
    ('slice_adaptive', True, True),
    ('dsl_max_depth', 3, 3),
    ('dsl_source_variants', 4, 1),
    ('dsl_adaptive_schedule', True, False),
    ('dsl_matrix_prob', 0.5, 0.0),
    ('dsl_attributes', True, False),
    ('corpus_feedback', True, False),
    # Campaigns that predate failure triage resume without it.
    ('quarantine', True, False),
    ('explained_feedback', True, False),
    ('swarm_prob', 0.5, 0.0),
)

# Arguments that must lie in [0, 1].
PROBABILITIES = ('slice_prob', 'dsl_evolve_prob', 'dsl_matrix_prob', 'swarm_prob', 'dsl_extend_prob',
                 'extended_prob', 'typed_op_prob', 'region_layout_prob', 'unchecked_spec_prob',
                 'boundary_shape_prob', 'probe_prob', 'dtype_mutate_prob', 'gemm_prob', 'local_mutate_prob',
                 'function_call_prob', 'extended_atomic_prob', 'extended_elementwise_prob', 'extended_fma_prob',
                 'extended_shape_op_prob', 'extended_int8_prob', 'region_int8_prob')

# Config field <- argument, where only the name differs.
RENAMED = {
    'coverage_probe_prob': 'probe_prob',
    'region_gemm_prob': 'gemm_prob',
    'region_typed_prob': 'typed_op_prob',
    'region_input_seed_count': 'region_input_seeds',
    'output_dir': 'output',
}
# Config field <- not --no-<argument>.
NEGATED = {
    'structural_feedback': 'no_structural_feedback',
    'extended_common_only': 'legacy_extended_mix',
    'extended_configuration_pair': 'no_extended_configurations',
    'extended_precision_pair': 'no_extended_precision',
    'extended_identity_pair': 'no_extended_identities',
    'extended_observation_pair': 'no_extended_observations',
    'instance_grid': 'no_instance_grids',
    'region_pass_config': 'no_region_pass_config',
    'region_swizzle_pair': 'no_region_swizzle',
    'region_warp_policy_pair': 'no_region_warp_policy',
    'save_artifacts': 'no_save_artifacts',
    'save_passed_code': 'no_save_passed_code',
    'region_schedule_pair': 'no_region_schedule_pair',
    'region_stage_sweep': 'no_region_stage_sweep',
    'region_loop_sweep': 'no_region_loop_sweep',
    'region_layout_sweep': 'no_region_layout_sweep',
}
# Config fields taken unchanged from the argument of the same name.
DIRECT = ('seed', 'uncovered_boost', 'quarantine', 'quarantine_retries', 'explained_feedback', 'swarm_prob',
          'dtype_mutate_prob', 'extended_prob', 'dsl_extend_prob', 'slice_prob', 'slice_adaptive',
          'dsl_evolve_prob', 'dsl_max_depth', 'dsl_source_variants', 'dsl_adaptive_schedule', 'dsl_matrix_prob',
          'dsl_attributes', 'corpus_feedback', 'random_config_count', 'extended_atomic_prob',
          'extended_elementwise_prob', 'extended_fma_prob', 'extended_shape_op_prob', 'extended_int8_prob',
          'region_int8_prob', 'compile_only', 'local_mutate_prob', 'function_min_count', 'function_max_count',
          'function_call_prob', 'input_seed', 'region_repeat_count', 'region_layout_prob', 'unchecked_spec_prob',
          'boundary_shape_prob', 'easy_shape')


def _slices_available(backend, names=()):
    from src.workflow.slices import available
    return available(backend, names)


def saved_generation(args) -> dict:
    """generation_config of the campaign being resumed ({} for a new one)."""
    if not args.resume:
        return {}
    saved = Path(args.resume)
    if not saved.exists():
        saved = Path(args.output) / args.resume
    summary = saved / 'summary.json'
    return json.loads(summary.read_text()).get('generation_config', {}) if summary.exists() else {}


def resolve_defaults(args, saved: dict):
    """Fill unset options with the new-campaign or the resumed-campaign default."""
    # Old campaigns predate the separated common/DSL stages: keep their mix.
    if saved and not args.legacy_extended_mix:
        args.legacy_extended_mix = not saved.get('extended_common_only', False)
    for name, fresh, resumed in INHERITED:
        if getattr(args, name) is None:
            value = saved.get(name, resumed) if args.resume else (fresh(args) if callable(fresh) else fresh)
            setattr(args, name, value)
    if args.slices is None:
        args.slices = ','.join(saved.get('slice_names', ())) if args.resume else ''


def validate(parser, args, get_backend):
    """Reject inconsistent arguments; --compile-only then implies --extended-prob 1."""
    for name in PROBABILITIES:
        if not 0 <= getattr(args, name) <= 1:
            parser.error(f'--{name.replace("_", "-")} must be between 0 and 1')
    ranges = (('dsl_max_depth', 1, 8), ('dsl_source_variants', 1, 32), ('region_input_seeds', 1, 8),
              ('region_repeat_count', 1, 8), ('random_config_count', 0, 8))
    for name, low, high in ranges:
        if not low <= getattr(args, name) <= high:
            parser.error(f'--{name.replace("_", "-")} must be in [{low}, {high}]')
    if not 0 <= args.function_min_count <= args.function_max_count <= 8:
        parser.error("Require 0 <= --function-min-count <= --function-max-count <= 8")
    for name in ('quarantine_retries', 'uncovered_boost'):
        if getattr(args, name) < 0:
            parser.error(f'--{name.replace("_", "-")} must be >= 0')
    if args.compile_only:
        for name in ('slice_prob', 'dsl_extend_prob'):
            if getattr(args, name):
                parser.error(f'--{name.replace("_", "-")} requires execution mode')
        args.extended_prob = 1.0
    if args.slice_prob:
        try:
            _slices_available(args.backend, slice_names(args))
        except ValueError as exc:
            parser.error(str(exc))
    if args.extended_prob and not get_backend(args.backend).supports_extended:
        parser.error('This backend does not support extended programs; use --extended-prob 0')


def slice_names(args) -> tuple:
    return tuple(name for name in args.slices.split(',') if name)


def make_config(args) -> Config:
    fields = {name: getattr(args, name) for name in DIRECT}
    fields.update({field: getattr(args, name) for field, name in RENAMED.items()})
    fields.update({field: not getattr(args, name) for field, name in NEGATED.items()})
    fields.update(
        slice_names=slice_names(args),
        extended_config_depth=0 if args.no_extended_configurations else args.extended_config_depth,
        extended_fast_math_pair=args.extended_fast_math,
        backends=[args.backend],
    )
    return Config(**fields)


def list_kernels():
    from src.ir.region_ops import OPS, TYPED_OPS
    from src.ir.extended import FLOAT_UNARY_OPS, FLOAT_BINARY_OPS
    from src.workflow.generator.extended import FAMILIES
    print("Registered region operations:")
    for kind, contract in {**OPS, **TYPED_OPS}.items():
        print(f"  {kind}: operands={contract.arity}, regions={contract.regions}, entry={contract.entry}")
    print("  probe: restricted whole-function template with dedicated oracle")
    print("  call: 1-3 tile operands, one tile result, previously generated callee")
    print("Extended families: " + ', '.join(FAMILIES))
    print("Extended shared float32 math: " + ', '.join(FLOAT_UNARY_OPS + FLOAT_BINARY_OPS))
