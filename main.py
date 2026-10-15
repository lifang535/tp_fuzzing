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

from src import cli
from src.workflow.fuzzer import TileSmith


def main():
    from src.backends import backend_names, get_backend, load_backend_plugins
    # Backends registered by plugins must exist before --backend is parsed;
    # --backend must not be read as an abbreviation of --backend-plugin.
    plugin_parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    plugin_parser.add_argument('--backend-plugin', action='append', default=[], metavar='MODULE')
    load_backend_plugins(plugin_parser.parse_known_args()[0].backend_plugin)

    parser = cli.build_parser(backend_names)
    args = parser.parse_args()
    cli.resolve_defaults(args, cli.saved_generation(args))
    cli.validate(parser, args, get_backend)
    if args.list_kernels:
        cli.list_kernels()
        return 0
    config = cli.make_config(args)

    if args.dump:
        import random
        if args.seed is not None:
            random.seed(args.seed)
        from src.workflow.emitter import get_emitter
        generator = get_backend(args.backend).make_generator(config)
        print(get_emitter(args.backend, config=config).emit(generator.generate()))
        return 0

    TileSmith(config, resume_dir=args.resume).run(num_iterations=args.iterations, verbose=not args.quiet)
    return 0


if __name__ == "__main__":
    sys.exit(main())
