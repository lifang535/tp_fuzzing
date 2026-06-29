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
"""

import argparse
import sys

from src.config import Config
from src.workflow.fuzzer import TileSmith


def main():
    parser = argparse.ArgumentParser(description="TileSmith: Tile Program Fuzzer")
    parser.add_argument("-n", "--iterations", type=int, default=100)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("-o", "--output", type=str, default="results")
    parser.add_argument("--dump", action="store_true", help="Print generated code without executing")
    parser.add_argument("--backend", type=str, default="tilelang", choices=["tilelang", "triton"])
    parser.add_argument("-q", "--quiet", action="store_true")
    parser.add_argument("--list-kernels", action="store_true", help="List all supported kernel kinds and exit")
    parser.add_argument(
        "--easy-shape", action="store_true",
        help="Use only power-of-2 shapes (128/256/512/1024/2048). "
             "These are always divisible by block sizes, so fewer kernels hit "
             "boundary code paths. Useful to measure baseline pass rate or "
             "build a clean seed corpus.",
    )
    args = parser.parse_args()

    if args.list_kernels:
        from src.ir import ComputeKind
        print("Supported kernel kinds:")
        for kind in ComputeKind:
            print(f"  {kind.value}")
        return 0

    config = Config(
        seed=args.seed,
        output_dir=args.output,
        backends=[args.backend],
        easy_shape=args.easy_shape,
    )

    if args.dump:
        import random
        if args.seed:
            random.seed(args.seed)
        from src.workflow.generator import ProgramGenerator
        from src.workflow.emitter import get_emitter
        from src.ir import TilePipeline
        from src.workflow.emitter import TileLangPipelineEmitter, TritonPipelineEmitter
        from src.ir import DynamicSequence
        from src.workflow.emitter import TileLangDynamicEmitter, TritonDynamicEmitter
        gen = ProgramGenerator(config, backend=args.backend)
        emitter = get_emitter(args.backend, config=config)
        if args.backend == "tilelang":
            pipeline_emitter = TileLangPipelineEmitter(config=config)
            dynamic_emitter = TileLangDynamicEmitter(config=config)
        else:
            pipeline_emitter = TritonPipelineEmitter(config=config)
            dynamic_emitter = TritonDynamicEmitter(config=config)
        program = gen.generate()
        if isinstance(program, DynamicSequence):
            print(f"# Dynamic: {' -> '.join(s.op_kind for s in program.steps)}")
            print(dynamic_emitter.emit(program))
        elif isinstance(program, TilePipeline):
            print(f"# Pipeline: {' -> '.join(s.kind.value for s in program.steps)}")
            print(pipeline_emitter.emit(program))
        else:
            print(emitter.emit(program))
        return 0

    fuzzer = TileSmith(config)
    fuzzer.run(num_iterations=args.iterations, verbose=not args.quiet)
    return 0


if __name__ == "__main__":
    sys.exit(main())
