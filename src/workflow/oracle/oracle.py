"""
Test Oracle — Executes generated programs and detects bugs.
"""

import os
import hashlib
import json
import signal
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
from pathlib import Path

from src.config import Config, DEFAULT_CONFIG
from src.backends.common.region import _layout_sweep_pairs


def _region_timeout_multiplier(program, config, backend_impl):
    """Number of kernel compilations the checked harness performs: every
    variant compiles once per layout pair. Layout pairs can be empty (logical
    programs, sweep off), so the max(1, ...) must apply to the pair count, not
    the product — the historical formula underestimated non-physical programs
    with several variants to a compile_timeout * 1 budget."""
    return len(backend_impl.region_variants(program, config)) * max(1, len(_layout_sweep_pairs(program)))


class BugType(Enum):
    COMPILE_CRASH = "compile_crash"
    RUNTIME_CRASH = "runtime_crash"
    WRONG_RESULT = "wrong_result"
    TIMEOUT = "timeout"
    # Oracle trust gate: the fp32 reference disagrees with its fp64 copy, so
    # the numeric check is meaningless (chaotic program) and this is wasted
    # fuzzing effort, not a bug.
    ORACLE_UNSTABLE = "oracle_unstable"


@dataclass
class BugReport:
    bug_type: BugType
    error_message: str
    params: dict = field(default_factory=dict)
    dtype: str = ""
    compute_kind: str = ""
    generated_code: str = ""
    timestamp: float = field(default_factory=time.time)
    root_cause: str = ""
    location: str = ""

    def classify_root_cause(self, backend=None):
        from src.backends.common.diagnostics import _failure_location
        if backend is None:
            from src.backends.common.diagnostics import classify_root_cause
            self.root_cause = classify_root_cause(self.error_message)
        else:
            from src.backends import get_backend
            self.root_cause = get_backend(backend).classify_root_cause(self.error_message)
        self.location = _failure_location(self.error_message)

    def to_dict(self) -> dict:
        return {
            "bug_type": self.bug_type.value,
            "root_cause": self.root_cause,
            "location": self.location,
            "compute_kind": self.compute_kind,
            "params": self.params,
            "dtype": self.dtype,
            "error_message": self.error_message[:2000],
            "timestamp": self.timestamp,
        }

    def summary(self) -> str:
        return f"[{self.bug_type.value}|{self.root_cause}] {self.compute_kind} params={self.params}"


class Oracle:
    def __init__(self, config: Config = DEFAULT_CONFIG, backend: str = "tilelang"):
        self.config = config
        self.backend = backend
        from src.backends import get_backend
        self.backend_impl = get_backend(backend)
        self.emitter = self.backend_impl.make_emitter(config)
        self.artifact_root = None
        self.last_compilation = []
        self.compilation_complete = False
        self.last_artifact_dir = None

    def _emit_code(self, program) -> str:
        return self.emitter.emit(program)

    def _get_meta(self, program):
        """Return (params_dict, dtype_str, compute_kind_str) for bug report."""
        from src.ir.region import RegionProgram
        from src.ir.extended import ExtendedProgram
        if isinstance(program, ExtendedProgram):
            dtypes = sorted({v.type.dtype for n in program.all_operations() for v in n.results})
            return program.params_dict, '+'.join(dtypes), 'extended:' + program.family
        if isinstance(program, RegionProgram):
            return program.params_dict, program.spec.dtype.value, "region:" + "->".join(o.kind for o in program.all_operations())
        raise TypeError(f'Unsupported program: {type(program).__name__}')

    def _read_extended_evidence(self, program, directory):
        """Read subprocess evidence without letting interrupted writes abort fuzzing."""
        from .evidence import read_extended_evidence
        evidence = read_extended_evidence(directory, len(self.backend_impl.extended_variants(program, self.config)))
        self.last_compilation = evidence.records
        self.compilation_complete = evidence.compiled
        return evidence

    def test(self, program) -> Optional[BugReport]:
        """Test a Region or Extended program."""
        code = self._emit_code(program)
        self.last_compilation = []
        self.compilation_complete = False
        self.last_artifact_dir = None
        from src.ir.extended import ExtendedProgram
        extended = isinstance(program, ExtendedProgram)
        if self.config.compile_only and not extended:
            raise ValueError('compile_only requires an extended program')
        temporary_artifacts = None
        options = dict(self.backend_impl.execution_options(self.config))
        if extended:
            if not self.config.save_artifacts or self.artifact_root is None:
                temporary_artifacts = tempfile.TemporaryDirectory(prefix='tilesmith_compilation_')
                artifact_dir = Path(temporary_artifacts.name)
            else:
                artifact_dir = Path(self.artifact_root) / hashlib.sha256(code.encode()).hexdigest()[:24]
                artifact_dir.mkdir(parents=True, exist_ok=True)
            self.last_artifact_dir = artifact_dir
            for name in ('compilation.json', 'progress.json'):
                (artifact_dir / name).unlink(missing_ok=True)
            (artifact_dir / 'repro.py').write_text(code)
            (artifact_dir / 'program.json').write_text(json.dumps(program.to_dict(), indent=2))
            options['env'] = dict(options.get('env', os.environ),
                                  TILESMITH_ARTIFACT_DIR=str(artifact_dir.resolve()),
                                  TILESMITH_COMPILE_ONLY=str(int(self.config.compile_only)))

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, prefix="tilesmith_") as f:
            f.write(code)
            tmp_path = f.name

        try:
            from .process import run_isolated
            execute = run_isolated if extended else subprocess.run
            if not extended:
                options.update(capture_output=True, text=True)
            result = execute(
                self.backend_impl.execution_command(tmp_path),
                timeout=self.config.compile_timeout * (len(self.backend_impl.extended_variants(program, self.config)) if extended
                                                      else _region_timeout_multiplier(program, self.config, self.backend_impl))
                        + self.config.execute_timeout,
                **options,
            )
            if extended:
                evidence = self._read_extended_evidence(program, artifact_dir)
                progress = evidence.progress
                output = result.stdout + result.stderr
                if result.returncode:
                    output += f'\nProcess exited with return code {result.returncode}\n'
                evidence_error = ''
                if result.returncode == 0:
                    mode = 'compile_only' if self.config.compile_only else 'execute'
                    if not evidence.completed(mode, result.stdout):
                        evidence_error = 'Incomplete extended compilation/execution evidence'
                        if evidence.diagnostic:
                            evidence_error += ': ' + evidence.diagnostic
                        output += '\n' + evidence_error + '\n'
                (artifact_dir / 'run.log').write_text(output)
            else:
                # Successful execution is also evidence of compilation. Failed
                # Region/probe reproducers do not expose a reliable stage marker.
                # Post-execute failures (e.g. the oracle-unstable gate, which
                # trips after every variant compiled and ran) print an execute
                # marker, so they still count as compiled.
                self.compilation_complete = (result.returncode == 0
                                             or 'TILESMITH_STAGE=execute' in (result.stderr or ''))

            if result.returncode != 0 or (extended and evidence_error):
                error_msg = (evidence_error if extended and evidence_error else
                             result.stderr.strip() or result.stdout.strip())
                if not error_msg:
                    error_msg = f'Process exited with return code {result.returncode}'
                if result.returncode < 0:
                    signum = -result.returncode
                    try:
                        signame = signal.Signals(signum).name
                    except ValueError:
                        signame = "unknown"
                    error_msg += f"\nProcess terminated by signal {signum} ({signame})"
                bug_type = self._classify_error(error_msg)
                if extended:
                    if progress.get('stage') in ('execute', 'complete') and not self.config.compile_only and bug_type != BugType.WRONG_RESULT:
                        bug_type = BugType.RUNTIME_CRASH
                params, dtype_str, compute_kind_str = self._get_meta(program)
                params["input_seed"] = self.config.input_seed
                report = BugReport(
                    bug_type=bug_type,
                    error_message=error_msg[-2000:],
                    params=params,
                    dtype=dtype_str,
                    compute_kind=compute_kind_str,
                    generated_code=code,
                )
                report.classify_root_cause(self.backend)
                if extended and progress:
                    # The progress manifest names the exact variant and stage
                    # the subprocess reached, which beats message inference.
                    report.location = f"{progress.get('stage')}:{progress.get('variant')}"
                return report

        except subprocess.TimeoutExpired as exc:
            if extended:
                def decode(value):
                    return value.decode(errors='replace') if isinstance(value, bytes) else value or ''
                (artifact_dir / 'run.log').write_text(decode(exc.stdout) + decode(exc.stderr) + '\nExecution timed out')
            params, dtype_str, compute_kind_str = self._get_meta(program)
            params["input_seed"] = self.config.input_seed
            report = BugReport(
                bug_type=BugType.TIMEOUT,
                error_message="Execution timed out",
                params=params,
                dtype=dtype_str,
                compute_kind=compute_kind_str,
                generated_code=code,
            )
            report.root_cause = "timeout"
            if extended:
                progress = self._read_extended_evidence(program, artifact_dir).progress
                if progress:
                    report.location = f"{progress.get('stage')}:{progress.get('variant')}"
            else:
                # Region/probe harnesses print TILESMITH_STAGE markers; the
                # last one names where the subprocess hung (prepare, execute,
                # reference, ...). Previously always '' for non-extended.
                from src.backends.common.diagnostics import _failure_location
                stderr = exc.stderr
                if isinstance(stderr, bytes):
                    stderr = stderr.decode(errors='replace')
                report.location = _failure_location(stderr or '')
            return report
        finally:
            if extended:
                try:
                    self._read_extended_evidence(program, artifact_dir)
                finally:
                    if temporary_artifacts is not None:
                        temporary_artifacts.cleanup()
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

        return None

    def _classify_error(self, error_msg: str) -> BugType:
        return self.backend_impl.classify_error(error_msg)
