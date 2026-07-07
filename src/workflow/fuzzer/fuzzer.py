"""
TileSmith Fuzzer — Main fuzzing loop.
"""

import json
import pickle
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

    def summary(self, historical_bugs_total: int = 0, historical_bugs_unique: int = 0) -> str:
        elapsed = time.time() - self.start_time
        return (
            f"=== TileSmith Fuzzing Stats ===\n"
            f"Time: {elapsed:.1f}s\n"
            f"Generated: {self.total_generated}\n"
            f"Tested: {self.total_tested}\n"
            f"Bugs (total): {historical_bugs_total + len(self.bugs_found)}\n"
            f"Bugs (unique): {historical_bugs_unique + len(self.unique_bugs)}\n"
            f"Throughput: {self.total_tested / max(elapsed, 1):.2f} tests/sec\n"
        )


class TileSmith:
    def __init__(self, config: Config = DEFAULT_CONFIG, resume_dir: str = None):
        self.config = config
        self.backend = config.backends[0] if config.backends else "tilelang"

        if config.seed is not None:
            random.seed(config.seed)

        self.generator = ProgramGenerator(config, backend=self.backend)
        self.mutator = Mutator(config, backend=self.backend)
        self.oracle = Oracle(config, backend=self.backend)
        self.stats = FuzzingStats()
        self.seed_pool: List = []
        self.tested_configs: set = set()
        self.known_root_causes: dict = {}
        self._historical_bugs_total = 0
        self._historical_bugs_unique = 0

        if resume_dir:
            # Resume mode: use the specified directory
            self.output_dir = Path(resume_dir)
            if not self.output_dir.exists():
                # Try as a subdirectory name under output_dir
                self.output_dir = Path(config.output_dir) / resume_dir
            if not self.output_dir.exists():
                raise FileNotFoundError(f"Cannot find resume directory: {resume_dir}")
            # Validate consistency: parse dir name and check against current config
            self._validate_resume_config(self.output_dir.name, config)
            self._load_history()
            self._restore_rng_state()
            self._restore_dim_pool()
            self._restore_seed_pool()
        else:
            # New run: create fresh directory
            timestamp = datetime.now().strftime("%Y.%m.%d-%H.%M")
            seed_str = f"seed={config.seed}" if config.seed is not None else "seed=random"
            shape_str = "easy-shape" if config.easy_shape else "hard-shape"
            run_dir_name = f"{timestamp}_{self.backend}_{shape_str}_{seed_str}"
            self.output_dir = Path(config.output_dir) / run_dir_name
            self.output_dir.mkdir(parents=True, exist_ok=True)

    def _load_history(self):
        """Load previous results from resume directory to avoid re-testing."""
        passed_count = 0
        failed_count = 0

        passed_dir = self.output_dir / "passed"
        if passed_dir.exists():
            for json_file in passed_dir.rglob("*.json"):
                try:
                    with open(json_file) as f:
                        d = json.load(f)
                    sig = self._make_sig_from_dict(d)
                    self.tested_configs.add(sig)
                    passed_count += 1
                except (json.JSONDecodeError, KeyError):
                    pass

        failed_dir = self.output_dir / "failed"
        if failed_dir.exists():
            for root_cause_dir in failed_dir.iterdir():
                if not root_cause_dir.is_dir():
                    continue
                root_cause = root_cause_dir.name
                dir_count = 0
                for json_file in root_cause_dir.rglob("*.json"):
                    try:
                        with open(json_file) as f:
                            d = json.load(f)
                        sig = self._make_sig_from_dict(d)
                        self.tested_configs.add(sig)
                        failed_count += 1
                        dir_count += 1
                    except (json.JSONDecodeError, KeyError):
                        pass
                if dir_count > 0:
                    self.known_root_causes[root_cause] = self.known_root_causes.get(root_cause, 0) + dir_count

        total_count = passed_count + failed_count

        self.stats.total_tested = total_count
        self.stats.total_generated = self.stats.total_tested

        # bugs_total = sum of all trigger counts; bugs_unique = distinct root cause categories
        self._historical_bugs_total = sum(self.known_root_causes.values())
        self._historical_bugs_unique = len(self.known_root_causes)

        print(f"[resume] Loaded {total_count} historical results from {self.output_dir}")
        print(f"[resume]   passed={passed_count}, failed={failed_count} (across {len(self.known_root_causes)} root causes)")
        print(f"[resume] Known root causes: {self.known_root_causes}")
        print(f"[resume] Previous tests: {self.stats.total_tested}")
        print()

    def _restore_rng_state(self):
        rng_path = self.output_dir / "rng_state.json"
        if not rng_path.exists():
            print("[resume] No rng_state.json found, keeping current random state")
            return
        try:
            with open(rng_path) as f:
                s = json.load(f)
            state = (s["version"], tuple(s["internalstate"]), s["gauss_next"])
            random.setstate(state)
            self._resume_generation_attempts = s.get("generation_attempts", 0)
            pending_path = self.output_dir / "pending_program.pkl"
            if pending_path.exists():
                try:
                    with open(pending_path, "rb") as pf:
                        pending_i, pending_program = pickle.load(pf)
                    self._resume_pending_i = pending_i
                    self._resume_pending_program = pending_program
                    print(f"[resume] Restored pending program [{pending_i}] (interrupted test will be re-run)")
                except Exception as pe:
                    print(f"[resume] Failed to restore pending program: {pe}")
            print(f"[resume] Restored random state from rng_state.json (generation_attempts={self._resume_generation_attempts})")
        except Exception as e:
            print(f"[resume] Failed to restore random state: {e}")

    def _save_dim_pool(self):
        with open(self.output_dir / "dim_pool.json", "w") as f:
            json.dump(self.generator.type_gen.dim_pool, f)

    def _restore_dim_pool(self):
        pool_path = self.output_dir / "dim_pool.json"
        if not pool_path.exists():
            return
        try:
            with open(pool_path) as f:
                self.generator.type_gen.dim_pool = json.load(f)
            print(f"[resume] Restored dim_pool ({len(self.generator.type_gen.dim_pool)} entries)")
        except Exception as e:
            print(f"[resume] Failed to restore dim_pool: {e}")

    def _save_seed_pool(self):
        if not self.seed_pool:
            return
        entries = []
        for program in self.seed_pool:
            entries.append(self._program_to_dict(program))
        with open(self.output_dir / "seed_pool.json", "w") as f:
            json.dump(entries, f, indent=2)

    def _restore_seed_pool(self):
        pool_path = self.output_dir / "seed_pool.json"
        if not pool_path.exists():
            return
        try:
            with open(pool_path) as f:
                entries = json.load(f)
            for d in entries:
                program = self._dict_to_program(d)
                if program is not None:
                    self.seed_pool.append(program)
            print(f"[resume] Restored seed_pool with {len(self.seed_pool)} entries")
        except Exception as e:
            print(f"[resume] Failed to restore seed_pool: {e}")

    def _program_to_dict(self, program) -> dict:
        """Serialize a program to a JSON-compatible dict (same format as passed files)."""
        if isinstance(program, DynamicSequence):
            return {
                "type": "dynamic",
                "sequence": [s.op_kind for s in program.steps],
                "params": program.params_dict,
                "dtype": program.dtype,
            }
        if isinstance(program, TilePipeline):
            return {
                "type": "pipeline",
                "pipeline": [s.kind.value for s in program.steps],
                "params": program.params_dict,
                "dtype": program.dtype.value if hasattr(program.dtype, "value") else program.dtype,
            }
        # TileProgram (single_op)
        kernel = program.kernels[0]
        return {
            "type": "single_op",
            "compute_kind": kernel.compute_kind.value,
            "params": kernel.params_dict,
            "dtype": kernel.dtype.value if hasattr(kernel.dtype, "value") else kernel.dtype,
        }

    @staticmethod
    def _dict_to_program(d: dict):
        """Deserialize a dict back to a program object."""
        from src.ir import TileKernel, ComputeKind, LoopKind, DataType, TileProgram
        from src.ir import TilePipeline, PipelineStep
        from src.ir import DynamicSequence, KernelStep, TileValuePool

        params = d.get("params", {})
        type_ = d.get("type", "")

        if type_ == "pipeline":
            ops = params.get("pipeline", [])
            alphas = params.get("pipeline_alphas", [1.0] * len(ops))
            steps = [PipelineStep(kind=ComputeKind(op), alpha=a) for op, a in zip(ops, alphas)]
            lk = params.get("loop_kind", "pipelined")
            dtype_str = d.get("dtype", "float16")
            return TilePipeline(
                steps=steps,
                M=params["M"], N=params["N"], K=params["K"],
                block_M=params["block_M"], block_N=params["block_N"], block_K=params["block_K"],
                threads=params["threads"], num_stages=params["num_stages"],
                loop_kind=LoopKind(lk) if isinstance(lk, str) else lk,
                dtype=DataType(dtype_str) if isinstance(dtype_str, str) else dtype_str,
            )

        if type_ == "single_op":
            ck = d.get("compute_kind", params.get("compute_kind", "gemm"))
            lk = params.get("loop_kind", "pipelined")
            dtype_str = d.get("dtype", "float16")
            kernel = TileKernel(
                name="kernel_0",
                compute_kind=ComputeKind(ck),
                M=params["M"], N=params["N"], K=params["K"],
                block_M=params["block_M"], block_N=params["block_N"], block_K=params["block_K"],
                threads=params["threads"], num_stages=params["num_stages"],
                loop_kind=LoopKind(lk) if isinstance(lk, str) else lk,
                dtype=DataType(dtype_str) if isinstance(dtype_str, str) else dtype_str,
                alpha=params.get("alpha", 1.0),
            )
            return TileProgram(kernels=[kernel])

        if type_ == "dynamic":
            # Rebuild a minimal DynamicSequence with just the scalar fields.
            # The pool and step buffers are not needed by the mutator (it only
            # reads op_kind and scalar params before regenerating the sequence).
            ops = params.get("sequence", [])
            alphas = params.get("sequence_alphas", [1.0] * len(ops))
            steps = [
                KernelStep(op_kind=op, inputs=[], outputs=[],
                           attrs={"alpha": a} if op == "scale" else {},
                           torch_ref_update="")
                for op, a in zip(ops, alphas)
            ]
            lk = params.get("loop_kind", "pipelined")
            return DynamicSequence(
                steps=steps,
                pool=TileValuePool(),
                M=params["M"], N=params["N"], K=params["K"],
                block_M=params["block_M"], block_N=params["block_N"], block_K=params["block_K"],
                threads=params["threads"], num_stages=params["num_stages"],
                loop_kind=lk if isinstance(lk, str) else lk.value,
                dtype=d.get("dtype", "float16"),
            )

        return None

    @staticmethod
    def _validate_resume_config(dir_name: str, config):
        """
        Parse the directory name and check that current config matches.
        Directory format: {date-time}_{backend}_{easy/hard-shape}_seed={seed}
        Example: 2026.06.29-16.41_triton_easy-shape_seed=42
        """
        parts = dir_name.split("_")
        # Expected parts: [date-time, backend, shape-mode, seed=N]
        # But date-time itself contains no underscore (uses dots and dash)
        # So: parts[0]=date-time, parts[1]=backend, parts[2]=shape-mode, parts[3]=seed=N
        # However backend could be "tilelang" or "triton" (no underscore)

        errors = []

        # Check backend
        current_backend = config.backends[0] if config.backends else "tilelang"
        if current_backend not in dir_name:
            errors.append(
                f"Backend mismatch: directory is for "
                f"'{'triton' if 'triton' in dir_name else 'tilelang'}' "
                f"but current config uses '{current_backend}'"
            )

        # Check easy/hard shape
        if "easy-shape" in dir_name and not config.easy_shape:
            errors.append(
                "Shape mode mismatch: directory used --easy-shape but current config does not"
            )
        elif "hard-shape" in dir_name and config.easy_shape:
            errors.append(
                "Shape mode mismatch: directory used hard-shape but current config uses --easy-shape"
            )

        # Check seed
        if "seed=" in dir_name:
            dir_seed_str = dir_name.split("seed=")[-1]
            if dir_seed_str == "random":
                if config.seed is not None:
                    errors.append(
                        f"Seed mismatch: directory used seed=random but current config uses seed={config.seed}"
                    )
            else:
                try:
                    dir_seed = int(dir_seed_str)
                    if config.seed is not None and config.seed != dir_seed:
                        errors.append(
                            f"Seed mismatch: directory used seed={dir_seed} but current config uses seed={config.seed}"
                        )
                except ValueError:
                    pass

        if errors:
            msg = "\n".join(f"  - {e}" for e in errors)
            raise ValueError(
                f"Resume directory '{dir_name}' does not match current config:\n{msg}\n"
                f"Please use matching --backend, --easy-shape, and --seed options."
            )

    def run(self, num_iterations: int = 1000, verbose: bool = True):
        if verbose:
            print(f"TileSmith: {num_iterations} iterations, backend={self.backend}")
            print(f"Output: {self.output_dir}")
            print()

        pool_rotation_interval = self.config.pool_rotation_interval
        # i counts all generation attempts (including dedup skips); new_tested counts
        # only cases actually tested this session — loop runs until new_tested == num_iterations
        i = getattr(self, '_resume_generation_attempts', 0)
        new_tested = 0
        # If a previous run was interrupted mid-test, resume that program first.
        _pending_program = getattr(self, '_resume_pending_program', None)
        _pending_i = getattr(self, '_resume_pending_i', None)
        _inflight_i, _inflight_program = None, None  # set around oracle.test(); used by finally

        try:
            while new_tested < num_iterations:
                if _pending_program is not None:
                    # Resume the interrupted program directly (already past dedup).
                    program = _pending_program
                    i = _pending_i
                    self.tested_configs.add(self._make_sig(program))
                    self.stats.total_generated += 1
                    _pending_program = None
                    _pending_i = None
                else:
                    # Rotate dim_pool periodically based on generation attempts
                    if i > 0 and i % pool_rotation_interval == 0:
                        self.generator.type_gen._init_pool()
                        if verbose:
                            print(f"[{i}] dim_pool rotated → {self.generator.type_gen.dim_pool[:5]}...")

                    program = self._generate_test_case()
                    self.stats.total_generated += 1
                    i += 1

                    # Dedup
                    config_sig = self._make_sig(program)
                    if config_sig in self.tested_configs:
                        if verbose:
                            print(f"[{i}] [DUPLICATE] {self._kind_label(program)}")
                        continue
                    self.tested_configs.add(config_sig)

                # Test — track inflight program so interrupt can resume it
                _inflight_i, _inflight_program = i, program
                bug = self.oracle.test(program)
                _inflight_i, _inflight_program = None, None
                self.stats.total_tested += 1
                new_tested += 1

                if bug:
                    self.stats.bugs_found.append(bug)
                    is_new = self.known_root_causes.get(bug.root_cause, 0) < self.config.max_same_root_cause
                    if is_new:
                        self.stats.unique_bugs.append(bug)
                        self._save_bug(bug, i, program)
                    self.known_root_causes[bug.root_cause] = self.known_root_causes.get(bug.root_cause, 0) + 1
                    if verbose:
                        marker = "NEW" if is_new else "dup"
                        print(f"[{i}] [FAILED] ({marker} / {bug.root_cause}) {self._kind_label(program)}")
                else:
                    self._save_passed(program, i)
                    if random.random() < self.config.seed_add_prob:
                        self.seed_pool.append(program)
                        if len(self.seed_pool) > self.config.seed_pool_max:
                            self.seed_pool.pop(random.randint(0, len(self.seed_pool) - 1))
                    if verbose:
                        print(f"[{i}] [PASSED] {self._kind_label(program)}")

                if verbose and new_tested % 100 == 0:
                    total_bugs = self._historical_bugs_total + len(self.stats.bugs_found)
                    total_unique = self._historical_bugs_unique + len(self.stats.unique_bugs)
                    print(f"[{new_tested}] tested={self.stats.total_tested} bugs={total_bugs} unique={total_unique}")

        finally:
            bugs_total = sum(self.known_root_causes.values())
            bugs_unique = len(self.known_root_causes)

            summary = {
                "backend": self.backend,
                "total_tested": self.stats.total_tested,
                "bugs_total": bugs_total,
                "bugs_unique": bugs_unique,
                "root_causes": self.known_root_causes,
            }
            with open(self.output_dir / "summary.json", "w") as f:
                json.dump(summary, f, indent=2)

            rng_state = random.getstate()
            with open(self.output_dir / "rng_state.json", "w") as f:
                json.dump({
                    "version": rng_state[0],
                    "internalstate": list(rng_state[1]),
                    "gauss_next": rng_state[2],
                    "generation_attempts": i,
                }, f)
            pending_path = self.output_dir / "pending_program.pkl"
            # _inflight_program is non-None only when interrupt happened inside oracle.test()
            if _inflight_program is not None:
                with open(pending_path, "wb") as f:
                    pickle.dump((_inflight_i, _inflight_program), f)
            elif pending_path.exists():
                pending_path.unlink()

            self._save_dim_pool()
            self._save_seed_pool()

        if verbose:
            print()
            print(self.stats.summary(
                max(0, bugs_total - len(self.stats.bugs_found)),
                max(0, bugs_unique - len(self.stats.unique_bugs)),
            ))

        return self.stats

    def _make_sig(self, program):
        """Create a hashable dedup signature for TileProgram, TilePipeline, or DynamicSequence.
        All enum fields are converted to their string .value so the sig matches _make_sig_from_dict."""
        if isinstance(program, DynamicSequence):
            steps_sig = tuple(
                (s.op_kind, s.attrs.get("alpha", 1.0) if s.op_kind == "scale" else 1.0)
                for s in program.steps
            )
            return (steps_sig, program.M, program.N, program.K,
                    program.block_M, program.block_N, program.block_K,
                    program.threads, program.loop_kind, program.num_stages, program.dtype)
        if isinstance(program, TilePipeline):
            steps_sig = tuple(
                (s.kind.value if hasattr(s.kind, "value") else s.kind, s.alpha)
                for s in program.steps
            )
            return (steps_sig, program.M, program.N, program.K,
                    program.block_M, program.block_N, program.block_K,
                    program.threads,
                    program.loop_kind.value if hasattr(program.loop_kind, "value") else program.loop_kind,
                    program.num_stages,
                    program.dtype.value if hasattr(program.dtype, "value") else program.dtype)
        kernel = program.kernels[0]
        return (
            kernel.compute_kind.value if hasattr(kernel.compute_kind, "value") else kernel.compute_kind,
            kernel.M, kernel.N, kernel.K,
            kernel.block_M, kernel.block_N, kernel.block_K,
            kernel.threads,
            kernel.loop_kind.value if hasattr(kernel.loop_kind, "value") else kernel.loop_kind,
            kernel.num_stages,
            kernel.dtype.value if hasattr(kernel.dtype, "value") else kernel.dtype,
            kernel.alpha,
        )

    @staticmethod
    def _make_sig_from_dict(d: dict):
        """Reconstruct the same dedup sig from a saved JSON dict.
        Handles both passed-file format (has 'type' key) and
        BugReport.to_dict() format (has 'compute_kind' key, no 'type')."""
        params = d.get("params", {})
        M = params.get("M", 0)
        N = params.get("N", 0)
        K = params.get("K", 0)
        bM = params.get("block_M", 0)
        bN = params.get("block_N", 0)
        bK = params.get("block_K", 0)
        threads = params.get("threads", 0)
        num_stages = params.get("num_stages", 1)
        loop_kind = params.get("loop_kind", "pipelined")
        dtype = d.get("dtype", params.get("dtype", "float16"))

        # Determine type from 'type' key (passed files) or 'compute_kind' prefix (bug files)
        type_ = d.get("type", "")
        if not type_:
            compute_kind_str = d.get("compute_kind", "")
            if compute_kind_str.startswith("dynamic:"):
                type_ = "dynamic"
            elif compute_kind_str.startswith("pipeline:"):
                type_ = "pipeline"
            else:
                type_ = "single_op"

        if type_ == "dynamic":
            ops = params.get("sequence", [])
            alphas = params.get("sequence_alphas", [1.0] * len(ops))
            steps_sig = tuple(
                (op, a if op == "scale" else 1.0)
                for op, a in zip(ops, alphas)
            )
            return (steps_sig, M, N, K, bM, bN, bK, threads, loop_kind, num_stages, dtype)

        if type_ == "pipeline":
            ops = params.get("pipeline", [])
            alphas = params.get("pipeline_alphas", [1.0] * len(ops))
            steps_sig = tuple(zip(ops, alphas))
            return (steps_sig, M, N, K, bM, bN, bK, threads, loop_kind, num_stages, dtype)

        # single_op
        compute_kind = d.get("compute_kind", params.get("compute_kind", ""))
        alpha = params.get("alpha", 1.0)
        return (compute_kind, M, N, K, bM, bN, bK, threads, loop_kind, num_stages, dtype, alpha)

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
          dynamic_{ops}_M{m},N{n},K{k},bM{bm},bN{bn},bK{bk},t{threads},{loop_kind},s{stages},{dtype}
          pipeline_{ops}_M{m},N{n},K{k},bM{bm},bN{bn},bK{bk},t{threads},{loop_kind},s{stages},{dtype}
          single_{op}_M{m},N{n},K{k},bM{bm},bN{bn},bK{bk},t{threads},{loop_kind},s{stages},{dtype}
        """
        def _step_label_dynamic(s) -> str:
            if s.op_kind == "scale":
                return f"scale_a{s.attrs.get('alpha', 1.0)}"
            return s.op_kind

        def _step_label_pipeline(s) -> str:
            kind = s.kind.value if hasattr(s.kind, "value") else s.kind
            if kind == "scale":
                return f"scale_a{s.alpha}"
            return kind

        if isinstance(program, DynamicSequence):
            ops = "+".join(_step_label_dynamic(s) for s in program.steps)
            params = (f"M{program.M},N{program.N},K{program.K},"
                      f"bM{program.block_M},bN{program.block_N},bK{program.block_K},"
                      f"t{program.threads},{program.loop_kind},s{program.num_stages},{program.dtype}")
            return f"dynamic_{ops}_{params}"
        if isinstance(program, TilePipeline):
            ops = "+".join(_step_label_pipeline(s) for s in program.steps)
            lk = program.loop_kind.value if hasattr(program.loop_kind, "value") else program.loop_kind
            dt = program.dtype.value if hasattr(program.dtype, "value") else program.dtype
            params = (f"M{program.M},N{program.N},K{program.K},"
                      f"bM{program.block_M},bN{program.block_N},bK{program.block_K},"
                      f"t{program.threads},{lk},s{program.num_stages},{dt}")
            return f"pipeline_{ops}_{params}"
        kernel = program.kernels[0] if program.kernels else None
        if kernel:
            op = kernel.compute_kind.value
            lk = kernel.loop_kind.value if hasattr(kernel.loop_kind, "value") else kernel.loop_kind
            dt = kernel.dtype.value if hasattr(kernel.dtype, "value") else kernel.dtype
            alpha_suffix = f",a{kernel.alpha}" if op == "scale" else ""
            params = (f"M{kernel.M},N{kernel.N},K{kernel.K},"
                      f"bM{kernel.block_M},bN{kernel.block_N},bK{kernel.block_K},"
                      f"t{kernel.threads},{lk},s{kernel.num_stages},{dt}{alpha_suffix}")
            return f"single_{op}_{params}"
        return "single_unknown"

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
