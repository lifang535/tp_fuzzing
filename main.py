"""
TileSmith — A Structure-Aware Fuzzer for Tile-Based GPU Programs.

Usage:
    python main.py                          # Run with defaults
    python main.py -n 500 --seed 42         # 500 iterations, reproducible
    python main.py --dump                    # Print generated code (no execution)
    python main.py --backend triton          # Target Triton backend
    python main.py --list-kernels            # List all supported kernel kinds
    python main.py --easy-shape              # Use power-of-2 shapes (higher pass rate)
    python main.py --easy-shape --seed 42 -n 100  # Compare pass rate with regular mode
    python main.py --resume 2026.06.29-16.41_triton_easy-shape_seed=42 -n 200  # Continue previous run

python main.py --backend tilelang              --seed 42 --function-min-count 1 --function-max-count 2 --function-call-prob 0.05 -n 100000 --no-save-artifacts --resume 2026.09.21-14.02_tilelang_hard-shape_seed=42
python main.py --backend triton                --seed 42 --function-min-count 1 --function-max-count 2 --function-call-prob 0.05 -n 100000 --no-save-artifacts --resume 2026.09.18-23.23_triton_hard-shape_seed=42
"""

import argparse
import sys

from src.config import Config
from src.workflow.fuzzer import TileSmith


def main():
    from src.backends import backend_names, get_backend, load_backend_plugins
    # --backend must not be interpreted as an abbreviation of --backend-plugin.
    plugin_parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    plugin_parser.add_argument('--backend-plugin', action='append', default=[], metavar='MODULE')
    plugin_args, _ = plugin_parser.parse_known_args()
    load_backend_plugins(plugin_args.backend_plugin)
    parser = argparse.ArgumentParser(description="TileSmith: Tile Program Fuzzer")
    parser.add_argument(
        "-n", "--iterations", type=int, default=100,
        help="Number of NEW test cases to execute (dedup-skipped cases do not count). "
             "When resuming, the fuzzer continues until this many "
             "previously-unseen programs have been tested.",
    )
    parser.add_argument("--probe-prob", type=float, default=0.20,
                        help="Conditional probe probability within the Region route (use --extended-prob 0 --probe-prob 1 for probes only)")
    parser.add_argument("--dtype-mutate-prob", type=float, default=0.25,
                        help="Probability of explicitly switching storage dtype during mutation")
    parser.add_argument("--gemm-prob", type=float, default=0.50,
                        help="GEMM entry probability for fresh native regions (otherwise load)")
    parser.add_argument('--typed-op-prob', type=float, default=0.35,
                        help='Probability of type/shape/memory operations in fresh regions; 0 generates v3')
    parser.add_argument('--extended-prob', type=float, default=None,
                        help='Fresh typed exploration programs: default 0.25 (0 when resuming an old campaign)')
    parser.add_argument('--compile-only', action='store_true',
                        help='Compile extended programs without GPU execution; implies --extended-prob 1')
    parser.add_argument('--no-save-artifacts', action='store_true',
                        help='Use temporary compilation evidence and delete it after each test instead of saving artifacts/')
    parser.add_argument('--no-extended-observations', action='store_true',
                        help='Disable additional intermediate-output variants')
    parser.add_argument('--no-extended-configurations', action='store_true',
                        help='Disable paired backend compilation configurations (alias for --extended-config-depth 0)')
    parser.add_argument('--extended-config-depth', type=int, choices=(0, 1, 2), default=1,
                        help='Extended configuration sweep depth: 0 = single configuration; '
                             '1 = threads/stages pair; 2 = additionally a second pass '
                             'configuration (tl.disable_loop_unswitching / enable_fp_fusion)')
    parser.add_argument('--extended-fast-math', action='store_true',
                        help='Additionally compile extended programs with tl.enable_fast_math '
                             '(changes numerics; off by default)')
    parser.add_argument('--no-extended-precision', action='store_true',
                        help='Disable the accumulator-width sweep: no fp16-accumulation copies of '
                             'extended matmul programs and no triton ieee->tf32 variant (on by default)')
    parser.add_argument('--no-extended-identities', action='store_true',
                        help='Disable the algebraic-identity sweep: no distributivity copies of '
                             'extended matmul-less programs (on by default)')
    parser.add_argument('--random-config-count', type=int, default=2,
                        help='Random pass-pipeline configurations sampled per extended program '
                             '(deterministic per program + seed; 0 disables; max 8)')
    parser.add_argument('--no-instance-grids', action='store_true',
                        help='Disable the per-(op, backend) round-robin instance grids; new op '
                             'attribute corners are then sampled uniformly at random')
    parser.add_argument('--extended-atomic-prob', type=float, default=0.25,
                        help='Probability of global-memory atomics per extended program')
    parser.add_argument('--extended-fma-prob', type=float, default=0.30,
                        help='Probability of scalar fused multiply-add per extended program')
    parser.add_argument('--extended-shape-op-prob', type=float, default=0.30,
                        help='Probability of triton shape primitives per extended program')
    parser.add_argument('--extended-int8-prob', type=float, default=0.30,
                        help='Probability that an extended matmul program uses int8 x int8')
    parser.add_argument('--region-int8-prob', type=float, default=0.15,
                        help='Probability that a native region is an int8 GEMM-only program')
    parser.add_argument('--no-region-pass-config', action='store_true',
                        help='Disable the region pass-configuration variant pair')
    parser.add_argument('--no-region-swizzle', action='store_true',
                        help='Disable the tilelang T.use_swizzle region variant pair')
    parser.add_argument('--no-region-warp-policy', action='store_true',
                        help='Disable the tilelang GemmWarpPolicy region variant pair (FullRow/FullCol)')
    parser.add_argument("--local-mutate-prob", type=float, default=0.35,
                        help="Local mutation probability after skipping dtype mutation")
    parser.add_argument("--function-min-count", type=int, default=1,
                        help="Minimum auxiliary functions per native program (default: 1)")
    parser.add_argument("--function-max-count", type=int, default=3,
                        help="Maximum auxiliary functions per native program (default: 3, limit: 8)")
    parser.add_argument("--function-call-prob", type=float, default=0.30,
                        help="Probability of replacing a non-control operation with a call when callees exist")
    parser.add_argument("--no-structural-feedback", action="store_true",
                        help="Disable feature-guided selection for ablation experiments")
    parser.add_argument("--uncovered-boost", type=float, default=50.0,
                        help="Additive weight boost for never-attempted structural features "
                             "(MLIRSmith DiversityCriteria-style; 0 restores legacy weighting)")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--input-seed", type=int, default=0,
                        help="Independent PyTorch tensor seed embedded in saved tests (default: 0)")
    parser.add_argument("--region-input-seeds", type=int, default=2,
                        help="Input seeds per fresh native region (1..8, starting at --input-seed)")
    parser.add_argument("--region-repeat-count", type=int, default=3,
                        help="Executions per input and thread configuration for fresh regions (1..8)")
    parser.add_argument("--no-region-schedule-pair", action="store_true",
                        help="Disable paired 128/256-thread checks for fresh native regions")
    parser.add_argument("--no-region-stage-sweep", action="store_true",
                        help="Disable the alternate num_stages sweep for fresh native regions")
    parser.add_argument("--no-region-loop-sweep", action="store_true",
                        help="Disable the alternate loop_kind sweep for fresh native regions")
    parser.add_argument("--no-region-layout-sweep", action="store_true",
                        help="Disable the alternate layout pair sweep for fresh physical regions")
    parser.add_argument("--region-layout-prob", type=float, default=0.35,
                        help="Non-contiguous layout probability per used input of a fresh region")
    parser.add_argument("--unchecked-spec-prob", type=float, default=0.30,
                        help="Probability that region spec sampling skips schedule/"
                             "shared-memory pre-validation (keeps warp_partition and "
                             "shared_memory_overflow generable)")
    parser.add_argument("--boundary-shape-prob", type=float, default=0.10,
                        help="Probability of forcing M below block_M (x2 for GEMM "
                             "entries) to reach ptx_async_boundary and tail bugs")
    parser.add_argument("-o", "--output", type=str, default="results")
    parser.add_argument("--dump", action="store_true", help="Print generated code without executing")
    parser.add_argument('--backend-plugin', action='append', default=[], metavar='MODULE',
                        help='Import a Python module that registers a backend (repeatable)')
    parser.add_argument("--backend", type=str, default="tilelang", choices=backend_names())
    parser.add_argument("-q", "--quiet", action="store_true")
    parser.add_argument("--list-kernels", action="store_true", help="List all supported kernel kinds and exit")
    parser.add_argument(
        "--easy-shape", action="store_true",
        help="Use power-of-two shapes (native: 1..16384; fresh probes: 16/32/64/128). "
             "Shapes smaller than a tile can still require boundary handling.",
    )
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Resume fuzzing from a previous run directory. Pass the directory name "
             "(e.g., '2026.06.29-16.41_triton_easy-shape_seed=42') or full path. "
             "Historical results are loaded to avoid re-testing, and new results "
             "are appended to the same directory.",
    )
    args = parser.parse_args()
    if args.extended_prob is None:
        args.extended_prob = 0.0 if args.resume else 0.25
        if args.resume:
            import json
            from pathlib import Path
            saved = Path(args.resume)
            if not saved.exists():
                saved = Path(args.output) / args.resume
            if (saved / 'summary.json').exists():
                args.extended_prob = json.loads((saved / 'summary.json').read_text()).get('generation_config', {}).get('extended_prob', 0.0)
    if not 0 <= args.extended_prob <= 1:
        parser.error('--extended-prob must be between 0 and 1')
    if args.compile_only:
        args.extended_prob = 1.0
    if args.extended_prob and not get_backend(args.backend).supports_extended:
        parser.error('This backend does not support extended programs; use --extended-prob 0')
    if not 0 <= args.typed_op_prob <= 1:
        parser.error('--typed-op-prob must be between 0 and 1')

    if not 1 <= args.region_input_seeds <= 8 or not 1 <= args.region_repeat_count <= 8:
        parser.error("--region-input-seeds and --region-repeat-count must be in 1..8")
    if not 0 <= args.region_layout_prob <= 1:
        parser.error("--region-layout-prob must be between 0 and 1")
    if not 0 <= args.unchecked_spec_prob <= 1 or not 0 <= args.boundary_shape_prob <= 1:
        parser.error("--unchecked-spec-prob and --boundary-shape-prob must be between 0 and 1")
    if not 0 <= args.probe_prob <= 1:
        parser.error("--probe-prob must be between 0 and 1")
    if not 0 <= args.dtype_mutate_prob <= 1:
        parser.error("--dtype-mutate-prob must be between 0 and 1")
    if not 0 <= args.gemm_prob <= 1 or not 0 <= args.local_mutate_prob <= 1:
        parser.error("--gemm-prob and --local-mutate-prob must be between 0 and 1")
    if not 0 <= args.function_min_count <= args.function_max_count <= 8:
        parser.error("Require 0 <= --function-min-count <= --function-max-count <= 8")
    if not 0 <= args.function_call_prob <= 1:
        parser.error("--function-call-prob must be between 0 and 1")
    if args.uncovered_boost < 0:
        parser.error("--uncovered-boost must be >= 0")
    if not 0 <= args.random_config_count <= 8:
        parser.error("--random-config-count must be between 0 and 8")
    for flag, name in ((args.extended_atomic_prob, '--extended-atomic-prob'),
                       (args.extended_fma_prob, '--extended-fma-prob'),
                       (args.extended_shape_op_prob, '--extended-shape-op-prob'),
                       (args.extended_int8_prob, '--extended-int8-prob'),
                       (args.region_int8_prob, '--region-int8-prob')):
        if not 0 <= flag <= 1:
            parser.error(f'{name} must be between 0 and 1')
    if args.list_kernels:
        from src.ir.region_ops import OPS, TYPED_OPS
        print("Registered region operations:")
        for kind, contract in {**OPS, **TYPED_OPS}.items():
            print(f"  {kind}: operands={contract.arity}, regions={contract.regions}, entry={contract.entry}")
        print("  probe: restricted whole-function template with dedicated oracle")
        print("  call: 1-3 tile operands, one tile result, previously generated callee")
        return 0

    config = Config(
        seed=args.seed,
        structural_feedback=not args.no_structural_feedback,
        uncovered_boost=args.uncovered_boost,
        coverage_probe_prob=args.probe_prob,
        dtype_mutate_prob=args.dtype_mutate_prob,
        region_gemm_prob=args.gemm_prob,
        region_typed_prob=args.typed_op_prob,
        extended_prob=args.extended_prob,
        extended_configuration_pair=not args.no_extended_configurations,
        extended_config_depth=0 if args.no_extended_configurations else args.extended_config_depth,
        extended_fast_math_pair=args.extended_fast_math,
        extended_precision_pair=not args.no_extended_precision,
        extended_identity_pair=not args.no_extended_identities,
        extended_observation_pair=not args.no_extended_observations,
        random_config_count=args.random_config_count,
        instance_grid=not args.no_instance_grids,
        extended_atomic_prob=args.extended_atomic_prob,
        extended_fma_prob=args.extended_fma_prob,
        extended_shape_op_prob=args.extended_shape_op_prob,
        extended_int8_prob=args.extended_int8_prob,
        region_int8_prob=args.region_int8_prob,
        region_pass_config=not args.no_region_pass_config,
        region_swizzle_pair=not args.no_region_swizzle,
        region_warp_policy_pair=not args.no_region_warp_policy,
        compile_only=args.compile_only,
        save_artifacts=not args.no_save_artifacts,
        local_mutate_prob=args.local_mutate_prob,
        function_min_count=args.function_min_count,
        function_max_count=args.function_max_count,
        function_call_prob=args.function_call_prob,
        input_seed=args.input_seed,
        region_input_seed_count=args.region_input_seeds,
        region_repeat_count=args.region_repeat_count,
        region_schedule_pair=not args.no_region_schedule_pair,
        region_stage_sweep=not args.no_region_stage_sweep,
        region_loop_sweep=not args.no_region_loop_sweep,
        region_layout_sweep=not args.no_region_layout_sweep,
        region_layout_prob=args.region_layout_prob,
        unchecked_spec_prob=args.unchecked_spec_prob,
        boundary_shape_prob=args.boundary_shape_prob,
        output_dir=args.output,
        backends=[args.backend],
        easy_shape=args.easy_shape,
    )

    if args.dump:
        import random
        if args.seed is not None:
            random.seed(args.seed)
        from src.workflow.emitter import get_emitter
        gen = get_backend(args.backend).make_generator(config)
        emitter = get_emitter(args.backend, config=config)
        print(emitter.emit(gen.generate()))
        return 0

    fuzzer = TileSmith(config, resume_dir=args.resume)
    fuzzer.run(num_iterations=args.iterations, verbose=not args.quiet)
    return 0


if __name__ == "__main__":
    sys.exit(main())
