"""
TileSmith Fuzzer — Main fuzzing loop.
"""

import hashlib
import json
import pickle
import random
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from src.config import Config, DEFAULT_CONFIG
from src.workflow.oracle import Oracle, BugReport, BugType


class FuzzingStats:
    def __init__(self):
        self.total_generated = 0
        self.total_tested = 0
        self.programs_compiled = 0
        self.programs_passed = 0
        self.bugs_found: List[BugReport] = []
        self.unique_bugs: List[BugReport] = []
        # MLIRSmith-style wasted-effort counter: chaotic programs whose
        # reference is numerically unstable are oracle noise, not bugs.
        self.oracle_unstable = 0
        self.start_time = time.time()

    def summary(self, historical_bugs_total: int = 0, historical_bugs_unique: int = 0,
                total_categories: int = None) -> str:
        if total_categories is None:
            total_categories = historical_bugs_unique + len({b.root_cause for b in self.bugs_found})
        elapsed = time.time() - self.start_time
        return (
            f"=== TileSmith Fuzzing Stats ===\n"
            f"Time: {elapsed:.1f}s\n"
            f"Generated: {self.total_generated}\n"
            f"Tested: {self.total_tested}\n"
            f"Bugs (total): {historical_bugs_total + len(self.bugs_found)}\n"
            f"Failure categories: {total_categories}\n"
            f"Throughput: {self.total_tested / max(elapsed, 1):.2f} tests/sec\n"
        )


class TileSmith:
    def __init__(self, config: Config = DEFAULT_CONFIG, resume_dir: str = None):
        self.config = config
        self.backend = config.backends[0] if config.backends else "tilelang"

        if config.seed is not None:
            random.seed(config.seed)

        from src.backends import get_backend
        backend_impl = get_backend(self.backend)
        self.generator = backend_impl.make_generator(config)
        self.mutator = backend_impl.make_mutator(config)
        self.mutator.type_gen = self.generator.type_gen
        self.oracle = Oracle(config, backend=self.backend)
        from src.workflow.feedback import StructuralFeedback
        self.feedback = StructuralFeedback()
        if config.structural_feedback:
            self.generator.region_gen.feedback = self.feedback
            self.mutator.feedback = self.feedback
        self.stats = FuzzingStats()
        self.seed_pool: List = []
        self.tested_configs: set = set()
        self.known_root_causes: dict = {}
        # root_cause -> location -> count (location-aware bug kinds)
        self.root_cause_locations: dict = {}
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
            self.feedback.restore(self.output_dir / "structural_feedback.json")
        else:
            # New run: create fresh directory
            timestamp = datetime.now().strftime("%Y.%m.%d-%H.%M")
            seed_str = f"seed={config.seed}" if config.seed is not None else "seed=random"
            shape_str = "easy-shape" if config.easy_shape else "hard-shape"
            run_dir_name = f"{timestamp}_{self.backend}_{shape_str}_{seed_str}"
            self.output_dir = Path(config.output_dir) / run_dir_name
            self.output_dir.mkdir(parents=True, exist_ok=True)
        self.oracle.artifact_root = self.output_dir / 'artifacts'

    def _load_history(self):
        """Load previous results from resume directory to avoid re-testing."""
        passed_count = 0
        failed_count = 0

        for passed_dir in (self.output_dir / 'passed', self.output_dir / 'compiled'):
            for json_file in passed_dir.glob("*.json"):
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
                        self.root_cause_locations.setdefault(root_cause, Counter())[d.get('location', '')] += 1
                    except (json.JSONDecodeError, KeyError):
                        pass
                if dir_count > 0:
                    self.known_root_causes[root_cause] = self.known_root_causes.get(root_cause, 0) + dir_count

        total_count = passed_count + failed_count
        # Reports are deduplicated/limited, so their file count may be much
        # smaller than the actual number of historical executions.
        summary_path = self.output_dir / "summary.json"
        if summary_path.exists():
            try:
                with open(summary_path) as f:
                    summary = json.load(f)
            except (OSError, json.JSONDecodeError):
                summary = {}
            if "input_seed" in summary and summary["input_seed"] != self.config.input_seed:
                raise ValueError(f"Resume input seed mismatch: use --input-seed {summary['input_seed']}")
            if summary.get('compile_only', False) != self.config.compile_only:
                raise ValueError('Cannot mix compile-only and execution results in one campaign')
            self.stats.programs_compiled = summary.get('programs_compiled', 0)
            self.stats.programs_passed = summary.get('programs_passed', passed_count if not self.config.compile_only else 0)
            total_count = max(total_count, summary.get("total_tested", 0))
            for cause, count in summary.get("root_causes", {}).items():
                self.known_root_causes[cause] = max(self.known_root_causes.get(cause, 0), count)
            for cause, counts in summary.get("root_cause_locations", {}).items():
                if isinstance(counts, dict) and counts:
                    current = self.root_cause_locations.get(cause, Counter())
                    if sum(counts.values()) > sum(current.values()):
                        self.root_cause_locations[cause] = Counter(counts)

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
            if self.generator.grids is not None:
                self.generator.grids.load(s.get("grid_cursors", {}))
            print(f"[resume] Restored random state from rng_state.json (generation_attempts={self._resume_generation_attempts})")
        except Exception as e:
            print(f"[resume] Failed to restore random state: {e}")

        pending_path = self.output_dir / "pending_program.pkl"
        if pending_path.exists():
            try:
                with open(pending_path, "rb") as pf:
                    pending_i, program = pickle.load(pf)
                # Normalize previous native specs and validate the executable IR.
                program = self._dict_to_program(self._program_to_dict(program))
            except Exception as error:
                raise ValueError(f'Cannot resume pending program {pending_path}: {error}') from error
            self._resume_pending_i = pending_i
            self._resume_pending_program = program
            print(f"[resume] Restored pending program [{pending_i}] (interrupted test will be re-run)")

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
            self.seed_pool = [self._dict_to_program(d) for d in entries]
            print(f"[resume] Restored seed_pool with {len(self.seed_pool)} entries")
        except Exception as error:
            raise ValueError(f'Cannot resume seed pool {pool_path}: {error}') from error

    def _program_to_dict(self, program) -> dict:
        from src.ir.serialization import program_to_dict
        return program_to_dict(program)

    @staticmethod
    def _dict_to_program(d: dict):
        from src.ir.serialization import program_from_dict
        return program_from_dict(d)

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
        # Backend registry names cannot contain underscores.

        errors = []

        # Check backend
        current_backend = config.backends[0] if config.backends else "tilelang"
        if current_backend not in dir_name:
            errors.append(
                f"Backend mismatch: directory is for "
                f"'{parts[1] if len(parts) > 1 else 'unknown'}' "
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

                novelty = self.feedback.observe(program, passed=bug is None and not self.config.compile_only)
                compiler_novelty = self.feedback.observe_compilation(
                    program, self.oracle.last_compilation, self.oracle.compilation_complete)
                if self.oracle.compilation_complete:
                    self.stats.programs_compiled += 1
                if bug is None and not self.config.compile_only:
                    self.stats.programs_passed += 1
                if bug:
                    if bug.root_cause == 'oracle_unstable':
                        # The numeric check was skipped: no implementation could
                        # pass it, so this is wasted fuzzing effort (MLIRSmith
                        # counts invalid programs the same way), not a bug.
                        # A few reproducers are saved for auditing, and the
                        # failed-program feedback demotion still applies.
                        self.stats.oracle_unstable += 1
                        if self.stats.oracle_unstable <= self.config.max_same_root_cause:
                            self._save_bug(bug, i, program)
                        if verbose:
                            print(f"[{i}] [ORACLE UNSTABLE] {self._kind_label(program)}")
                        continue
                    self.stats.bugs_found.append(bug)
                    is_new = self.known_root_causes.get(bug.root_cause, 0) < self.config.max_same_root_cause
                    if is_new:
                        self.stats.unique_bugs.append(bug)
                        self._save_bug(bug, i, program)
                    self.known_root_causes[bug.root_cause] = self.known_root_causes.get(bug.root_cause, 0) + 1
                    self.root_cause_locations.setdefault(bug.root_cause, Counter())[bug.location] += 1
                    if verbose:
                        marker = "NEW" if is_new else "dup"
                        print(f"[{i}] [FAILED] ({marker} / {bug.root_cause}) {self._kind_label(program)}")
                else:
                    self._save_passed(program, i)
                    if (self.config.structural_feedback and (novelty or compiler_novelty)) or random.random() < self.config.seed_add_prob:
                        self.seed_pool.append(program)
                        if len(self.seed_pool) > self.config.seed_pool_max:
                            self.seed_pool.pop(random.randint(0, len(self.seed_pool) - 1))
                    if verbose:
                        status = 'COMPILED' if self.config.compile_only else 'PASSED'
                        print(f"[{i}] [{status}] {self._kind_label(program)}")

                if verbose and new_tested % 100 == 0:
                    total_bugs = self._historical_bugs_total + len(self.stats.bugs_found)
                    total_unique = len(self.known_root_causes)
                    print(f"[{new_tested}] tested={self.stats.total_tested} bugs={total_bugs} categories={total_unique}")

        finally:
            bugs_total = sum(self.known_root_causes.values())
            bugs_unique = len(self.known_root_causes)

            summary = {
                "backend": self.backend,
                "input_seed": self.config.input_seed,
                "compile_only": self.config.compile_only,
                "save_artifacts": self.config.save_artifacts,
                "programs_compiled": self.stats.programs_compiled,
                "programs_passed": self.stats.programs_passed,
                "coverage_probe_prob": self.config.coverage_probe_prob,
                "dtype_mutate_prob": self.config.dtype_mutate_prob,
                "generation_config": {
                    "extended_prob": self.config.extended_prob,
                    "extended_configuration_pair": self.config.extended_configuration_pair,
                    "extended_config_depth": self.config.extended_config_depth,
                    "extended_fast_math_pair": self.config.extended_fast_math_pair,
                    "extended_precision_pair": self.config.extended_precision_pair,
                    "extended_identity_pair": self.config.extended_identity_pair,
                    "extended_observation_pair": self.config.extended_observation_pair,
                    "random_config_count": self.config.random_config_count,
                    "instance_grid": self.config.instance_grid,
                    "extended_atomic_prob": self.config.extended_atomic_prob,
                    "extended_fma_prob": self.config.extended_fma_prob,
                    "extended_shape_op_prob": self.config.extended_shape_op_prob,
                    "extended_int8_prob": self.config.extended_int8_prob,
                    "region_int8_prob": self.config.region_int8_prob,
                    "region_pass_config": self.config.region_pass_config,
                    "region_swizzle_pair": self.config.region_swizzle_pair,
                    "region_gemm_prob": self.config.region_gemm_prob,
                    "region_typed_prob": self.config.region_typed_prob,
                    "region_scratch_max_bytes": self.config.region_scratch_max_bytes,
                    "local_mutate_prob": self.config.local_mutate_prob,
                    "latest_value_prob": self.config.latest_value_prob,
                    "function_min_count": self.config.function_min_count,
                    "function_max_count": self.config.function_max_count,
                    "function_call_prob": self.config.function_call_prob,
                    "region_input_seed_count": self.config.region_input_seed_count,
                    "region_repeat_count": self.config.region_repeat_count,
                    "region_schedule_pair": self.config.region_schedule_pair,
                    "region_layout_prob": self.config.region_layout_prob,
                    "uncovered_boost": self.config.uncovered_boost,
                },
                "structural_features_attempted": len(self.feedback.attempted),
                "structural_features_passed": len(self.feedback.passed),
                "structural_features_compiled": len(self.feedback.compiled),
                "compiler_ir_features": len(self.feedback.compiler),
                "total_tested": self.stats.total_tested,
                "oracle_unstable": self.stats.oracle_unstable,
                "bugs_total": bugs_total,
                "bugs_unique": bugs_unique,
                "root_causes": self.known_root_causes,
                "root_cause_locations": {cause: dict(counts) for cause, counts in self.root_cause_locations.items()},
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
                    "grid_cursors": self.generator.grids.save() if self.generator.grids is not None else {},
                }, f)
            pending_path = self.output_dir / "pending_program.pkl"
            # _inflight_program is non-None only when interrupt happened inside oracle.test()
            if _inflight_program is not None:
                with open(pending_path, "wb") as f:
                    pickle.dump((_inflight_i, _inflight_program), f)
            elif pending_path.exists():
                pending_path.unlink()

            self.feedback.save(self.output_dir / "structural_feedback.json")
            self._save_dim_pool()
            self._save_seed_pool()

        if verbose:
            print()
            print(self.stats.summary(
                max(0, bugs_total - len(self.stats.bugs_found)),
                total_categories=bugs_unique,
            ))

        return self.stats

    @staticmethod
    def _make_sig(program):
        """Canonical executable IR identity, shared with persisted records."""
        from src.ir.serialization import program_to_dict
        data = program_to_dict(program)
        return (data['type'], json.dumps(data, sort_keys=True, separators=(',', ':')))

    @staticmethod
    def _make_sig_from_dict(data):
        from src.ir.serialization import program_from_dict
        return TileSmith._make_sig(program_from_dict(data))

    def _generate_test_case(self):
        if not self.seed_pool:
            return self.generator.generate()
        strategy = random.choices(
            ["fresh", "mutate"],
            weights=[1.0 - self.config.mutate_prob, self.config.mutate_prob],
            k=1,
        )[0]
        if strategy == "mutate":
            seed = (random.choices(self.seed_pool, weights=[self.feedback.seed_weight(p) for p in self.seed_pool], k=1)[0]
                    if self.config.structural_feedback else random.choice(self.seed_pool))
            return self.mutator.mutate(seed)
        return self.generator.generate()

    def _kind_label(self, program) -> str:
        """Summarize static function calls in filenames, with a full IR hash.

        Shapes, operations and control-flow nesting remain in the saved IR.
        """
        from src.ir.region import RegionProgram
        from src.ir.extended import ExtendedProgram
        if isinstance(program, ExtendedProgram):
            digest = hashlib.sha256(repr(self._make_sig(program)).encode()).hexdigest()[:16]
            return f'extended_{program.family}_{digest}'
        if isinstance(program, RegionProgram):
            calls = program.call_label()
            if len(calls) > 190:
                calls = calls[:190] + '~'
            digest = hashlib.sha256(repr(self._make_sig(program)).encode()).hexdigest()[:16]
            return f"calls_{calls}_{digest}"

        raise TypeError(f'Unsupported program: {type(program).__name__}')

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
        status = 'compiled' if self.config.compile_only else 'passed'
        passed_dir = self.output_dir / status
        passed_dir.mkdir(exist_ok=True)

        kind_label = self._kind_label(program)
        name = f"{status}_{kind_label}"
        code = self.oracle._emit_code(program)

        meta = self._program_to_dict(program)
        meta["input_seed"] = self.config.input_seed
        meta['validation_mode'] = 'compile_only' if self.config.compile_only else 'execute'

        with open(passed_dir / f"{name}.json", "w") as f:
            json.dump(meta, f, indent=2)
        with open(passed_dir / f"{name}.py", "w") as f:
            f.write(code)
