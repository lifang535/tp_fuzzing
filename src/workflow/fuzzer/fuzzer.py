"""
TileSmith Fuzzer — Main fuzzing loop.
"""

import json
import random
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from src.config import Config, DEFAULT_CONFIG
from src.workflow.generator import ProgramGenerator
from src.workflow.mutator import Mutator
from src.workflow.oracle import Oracle, BugReport, BugType
from src.ir import TileProgram
from src.ir import TilePipeline
from src.ir import DynamicSequence


class FuzzingStats:
    def __init__(self):
        self.total_generated = 0
        self.total_tested = 0
        self.bugs_found: List[BugReport] = []
        self.unique_bugs: List[BugReport] = []
        self.start_time = time.time()

    def summary(self) -> str:
        elapsed = time.time() - self.start_time
        return (
            f"=== TileSmith Fuzzing Stats ===\n"
            f"Time: {elapsed:.1f}s\n"
            f"Generated: {self.total_generated}\n"
            f"Tested: {self.total_tested}\n"
            f"Bugs (total): {len(self.bugs_found)}\n"
            f"Bugs (unique): {len(self.unique_bugs)}\n"
            f"Throughput: {self.total_tested / max(elapsed, 1):.2f} tests/sec\n"
        )


class TileSmith:
    def __init__(self, config: Config = DEFAULT_CONFIG):
        self.config = config
        self.backend = config.backends[0] if config.backends else "tilelang"
        self.generator = ProgramGenerator(config, backend=self.backend)
        self.mutator = Mutator(config, backend=self.backend)
        self.oracle = Oracle(config, backend=self.backend)
        self.stats = FuzzingStats()
        self.seed_pool: List = []  # holds TileProgram or TilePipeline
        self.tested_configs: set = set()
        self.known_root_causes: dict = {}

        if config.seed is not None:
            random.seed(config.seed)

        timestamp = datetime.now().strftime("%Y.%m.%d-%H.%M")
        seed_str = f"seed={config.seed}" if config.seed is not None else "seed=random"
        shape_str = "easy-shape" if config.easy_shape else "hard-shape"
        run_dir_name = f"{timestamp}_{self.backend}_{shape_str}_{seed_str}"
        self.output_dir = Path(config.output_dir) / run_dir_name
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run(self, num_iterations: int = 1000, verbose: bool = True):
        if verbose:
            print(f"TileSmith: {num_iterations} iterations, backend={self.backend}")
            print(f"Output: {self.output_dir}")
            print()

        pool_rotation_interval = self.config.pool_rotation_interval

        for i in range(num_iterations):
            # Rotate dim_pool periodically
            if i > 0 and i % pool_rotation_interval == 0:
                self.generator.type_gen._init_pool()
                if verbose:
                    print(f"[{i}] dim_pool rotated → {self.generator.type_gen.dim_pool[:5]}...")

            program = self._generate_test_case()
            self.stats.total_generated += 1

            # Dedup — works for both TileProgram and TilePipeline
            config_sig = self._make_sig(program)
            if config_sig in self.tested_configs:
                continue
            self.tested_configs.add(config_sig)

            # Test
            bug = self.oracle.test(program)
            self.stats.total_tested += 1

            if bug:
                self.stats.bugs_found.append(bug)
                is_new = self.known_root_causes.get(bug.root_cause, 0) < self.config.max_same_root_cause
                if is_new:
                    self.stats.unique_bugs.append(bug)
                    self._save_bug(bug, i, program)
                self.known_root_causes[bug.root_cause] = self.known_root_causes.get(bug.root_cause, 0) + 1
                if verbose:
                    marker = "NEW" if is_new else "dup"
                    print(f"[{i}] BUG ({marker}): {bug.summary()}")
            else:
                self._save_passed(program, i)
                if random.random() < self.config.seed_add_prob:
                    self.seed_pool.append(program)
                    if len(self.seed_pool) > self.config.seed_pool_max:
                        self.seed_pool.pop(random.randint(0, len(self.seed_pool) - 1))

            if verbose and i > 0 and i % 100 == 0:
                print(f"[{i}] tested={self.stats.total_tested} bugs={len(self.stats.bugs_found)} unique={len(self.stats.unique_bugs)}")

        if verbose:
            print()
            print(self.stats.summary())

        # Save coverage summary
        summary = {
            "backend": self.backend,
            "total_tested": self.stats.total_tested,
            "bugs_total": len(self.stats.bugs_found),
            "bugs_unique": len(self.stats.unique_bugs),
            "root_causes": self.known_root_causes,
        }
        with open(self.output_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        return self.stats

    def _make_sig(self, program):
        """Create a hashable dedup signature for TileProgram, TilePipeline, or DynamicSequence."""
        if isinstance(program, DynamicSequence):
            steps_sig = tuple(s.op_kind for s in program.steps)
            return (steps_sig, program.M, program.N, program.K,
                    program.block_M, program.block_N, program.block_K,
                    program.threads, program.loop_kind, program.num_stages, program.dtype)
        if isinstance(program, TilePipeline):
            steps_sig = tuple(s.kind for s in program.steps)
            return (steps_sig, program.M, program.N, program.K,
                    program.block_M, program.block_N, program.block_K,
                    program.threads, program.loop_kind, program.num_stages, program.dtype)
        kernel = program.kernels[0]
        return (kernel.compute_kind, kernel.M, kernel.N, kernel.K,
                kernel.block_M, kernel.block_N, kernel.block_K,
                kernel.threads, kernel.loop_kind, kernel.num_stages, kernel.dtype)

    def _generate_test_case(self):
        if not self.seed_pool:
            return self.generator.generate()
        strategy = random.choices(
            ["fresh", "mutate"],
            weights=[1.0 - self.config.mutate_prob, self.config.mutate_prob],
            k=1,
        )[0]
        if strategy == "mutate":
            seed = random.choice(self.seed_pool)
            return self.mutator.mutate(seed)
        return self.generator.generate()

    def _kind_label(self, program) -> str:
        """Return a human-readable label for use in filenames.
        Format:
          single_{op}                e.g. single_gemm
          pipeline_{op1}+{op2}+...  e.g. pipeline_gemm+scale+softmax
          dynamic_{op1}+{op2}+...   e.g. dynamic_gemm+exp+copy_f2g
        """
        if isinstance(program, DynamicSequence):
            ops = "+".join(s.op_kind for s in program.steps)
            return f"dynamic_{ops}"
        if isinstance(program, TilePipeline):
            ops = "+".join(s.kind.value for s in program.steps)
            return f"pipeline_{ops}"
        kernel = program.kernels[0] if program.kernels else None
        op = kernel.compute_kind.value if kernel else "unknown"
        return f"single_{op}"

    def _save_bug(self, bug: BugReport, iteration: int, program):
        """
        Save failed programs under:
          failed/{root_cause}/failed_{type_label}.{json,py}
        """
        root_cause = bug.root_cause or "other"
        kind_label = self._kind_label(program)
        failed_dir = self.output_dir / "failed" / root_cause
        failed_dir.mkdir(parents=True, exist_ok=True)

        name = f"failed_{kind_label}"
        with open(failed_dir / f"{name}.json", "w") as f:
            json.dump(bug.to_dict(), f, indent=2)
        with open(failed_dir / f"{name}.py", "w") as f:
            f.write(bug.generated_code)

    def _save_passed(self, program, iteration: int):
        """
        Save passing programs under:
          passed/passed_{type_label}.{json,py}
        """
        passed_dir = self.output_dir / "passed"
        passed_dir.mkdir(exist_ok=True)

        kind_label = self._kind_label(program)
        name = f"passed_{kind_label}"
        code = self.oracle._emit_code(program)

        if isinstance(program, DynamicSequence):
            meta = {
                "type": "dynamic",
                "sequence": [s.op_kind for s in program.steps],
                "params": program.params_dict,
                "dtype": program.dtype,
            }
        elif isinstance(program, TilePipeline):
            meta = {
                "type": "pipeline",
                "pipeline": [s.kind.value for s in program.steps],
                "params": program.params_dict,
                "dtype": program.dtype.value,
            }
        else:
            kernel = program.kernels[0]
            meta = {
                "type": "single_op",
                "compute_kind": kernel.compute_kind.value,
                "params": kernel.params_dict,
                "dtype": kernel.dtype.value,
            }

        with open(passed_dir / f"{name}.json", "w") as f:
            json.dump(meta, f, indent=2)
        with open(passed_dir / f"{name}.py", "w") as f:
            f.write(code)
