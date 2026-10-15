"""
Test Oracle — Executes generated programs and detects bugs.
"""

import os
import hashlib
import json
import shutil
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


def _slice_timeout_multiplier(program):
    """A slice harness compiles one kernel per launch configuration."""
    from src.workflow.slices import SLICES
    return len(SLICES[program.slice].variants(program.params, program.backend))


_MARKERS = ('TILESMITH_TARGET=', 'TILESMITH_REJECTED=')


def _bounded(message, limit=2000):
    """The message's last `limit` characters, led by what classification
    reads that is printed before a traceback that can outgrow the tail: the
    harness markers (the compilation target, a clean rejection) and, for a
    foreign target, the backend's unsupported-conversion diagnostic."""
    tail = message[-limit:]
    foreign = 'TILESMITH_TARGET=' in message
    kept = [line for line in message.splitlines() if line not in tail and (
        line.startswith(_MARKERS) or (foreign and 'error: Unsupported conversion from' in line))][:4]
    return '\n'.join(kept + [tail]) if kept else tail


def _process_error(result) -> str:
    """The diagnostic of a failed subprocess, with the signal that killed it."""
    message = result.stderr.strip() or result.stdout.strip()
    if not message:
        message = f'Process exited with return code {result.returncode}'
    if result.returncode < 0:
        signum = -result.returncode
        try:
            name = signal.Signals(signum).name
        except ValueError:
            name = "unknown"
        message += f"\nProcess terminated by signal {signum} ({name})"
    return message


def _take_lowering(stdout):
    """(lowering features, stdout without them) of a slice harness run; the
    harness prints its compiled kernels' features on one stdout line."""
    features, kept = None, []
    for line in (stdout or '').splitlines(keepends=True):
        if line.startswith('TILESMITH_LOWERING='):
            features = frozenset(f for f in line.strip().split('=', 1)[1].split(',') if f)
        else:
            kept.append(line)
    return features, ''.join(kept)


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
    confirmed_signature: str = ""
    # Normalized diagnostic grouping (src.workflow.triage), computed from the
    # complete message; the root cause alone merges unrelated mechanisms.
    failure_bucket: str = ""
    failure_key: str = ""

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
        result = {
            "bug_type": self.bug_type.value,
            "root_cause": self.root_cause,
            "location": self.location,
            "compute_kind": self.compute_kind,
            "params": self.params,
            "dtype": self.dtype,
            "error_message": self.error_message[:2000],
            "timestamp": self.timestamp,
        }
        if self.confirmed_signature:
            result['confirmed_signature'] = self.confirmed_signature
        if self.failure_bucket:
            result['failure_bucket'] = self.failure_bucket
            result['failure_key'] = self.failure_key
        return result

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
        self.last_lowering = None  # lowering features of the last slice test

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
        from src.ir.slice import SliceProgram
        if isinstance(program, SliceProgram):
            dtypes = sorted({str(v) for k, v in program.params.items() if k.endswith('_dt') or k == 'dt'})
            return program.params_dict, '+'.join(dtypes), 'slice:' + program.slice
        raise TypeError(f'Unsupported program: {type(program).__name__}')

    def _read_extended_evidence(self, program, directory):
        """Read subprocess evidence without letting interrupted writes abort fuzzing."""
        from .evidence import read_extended_evidence
        evidence = read_extended_evidence(directory, len(self.backend_impl.extended_variants(program, self.config)))
        self.last_compilation = evidence.records
        self.compilation_complete = evidence.compiled
        return evidence

    def test(self, program) -> Optional[BugReport]:
        """Run a Region, Extended or slice program in a subprocess; None when it passes."""
        from src.ir.extended import ExtendedProgram
        from src.ir.slice import SliceProgram
        code = self._emit_code(program)
        self.last_compilation = []
        self.compilation_complete = False
        self.last_artifact_dir = None
        self.last_lowering = None
        extended = isinstance(program, ExtendedProgram)
        sliced = isinstance(program, SliceProgram)
        if self.config.compile_only and not extended:
            raise ValueError('compile_only requires an extended program')
        options = dict(self.backend_impl.execution_options(self.config))
        artifact_dir, temporary_artifacts = (self._artifact_dir(program, code, options) if extended
                                             else (None, None))
        # Compilers leave temporaries in TMPDIR when they fail or are killed
        # on timeout (Triton keeps its ptxas input and output, megabytes per
        # kernel), so every test gets a private TMPDIR removed with it.
        scratch = tempfile.mkdtemp(prefix='tilesmith_tmp_')
        options['env'] = dict(options.get('env', os.environ), TMPDIR=scratch)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, prefix="tilesmith_") as f:
            f.write(code)
            tmp_path = f.name
        try:
            from .process import run_isolated
            if not extended:
                options.update(capture_output=True, text=True)
            result = (run_isolated if extended else subprocess.run)(
                self.backend_impl.execution_command(tmp_path), timeout=self._timeout(program), **options)
            if extended:
                evidence = self._read_extended_evidence(program, artifact_dir)
                error = self._extended_outcome(result, evidence, artifact_dir)
                if error is not None:
                    return self._failure(program, code, error, progress=evidence.progress)
            else:
                if sliced:
                    self.last_lowering, result.stdout = _take_lowering(result.stdout)
                # Successful execution is also evidence of compilation. Failed
                # Region/probe reproducers do not expose a reliable stage marker.
                # Post-execute failures (e.g. the oracle-unstable gate, which
                # trips after every variant compiled and ran) print an execute
                # marker, so they still count as compiled.
                self.compilation_complete = (result.returncode == 0
                                             or 'TILESMITH_STAGE=execute' in (result.stderr or ''))
                if result.returncode != 0:
                    return self._failure(program, code, _process_error(result))
        except subprocess.TimeoutExpired as exc:
            return self._timeout_report(program, code, exc, artifact_dir if extended else None)
        finally:
            shutil.rmtree(scratch, ignore_errors=True)
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

    def _artifact_dir(self, program, code, options):
        """Directory the Extended harness writes its evidence to, and its owner if temporary."""
        temporary = None
        if not self.config.save_artifacts or self.artifact_root is None:
            temporary = tempfile.TemporaryDirectory(prefix='tilesmith_compilation_')
            directory = Path(temporary.name)
        else:
            directory = Path(self.artifact_root) / hashlib.sha256(code.encode()).hexdigest()[:24]
            directory.mkdir(parents=True, exist_ok=True)
        self.last_artifact_dir = directory
        for name in ('compilation.json', 'progress.json'):
            (directory / name).unlink(missing_ok=True)
        (directory / 'repro.py').write_text(code)
        (directory / 'program.json').write_text(json.dumps(program.to_dict(), indent=2))
        options['env'] = dict(options.get('env', os.environ),
                              TILESMITH_ARTIFACT_DIR=str(directory.resolve()),
                              TILESMITH_COMPILE_ONLY=str(int(self.config.compile_only)))
        return directory, temporary

    def _timeout(self, program) -> float:
        """Compile budget per compiled variant plus the execution budget."""
        from src.ir.extended import ExtendedProgram
        from src.ir.slice import SliceProgram
        if isinstance(program, ExtendedProgram):
            variants = len(self.backend_impl.extended_variants(program, self.config))
        elif isinstance(program, SliceProgram):
            variants = _slice_timeout_multiplier(program)
        else:
            variants = _region_timeout_multiplier(program, self.config, self.backend_impl)
        return self.config.compile_timeout * variants + self.config.execute_timeout

    def _extended_outcome(self, result, evidence, artifact_dir):
        """Error message of an Extended run (None when it passed); writes run.log."""
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
        if evidence_error:
            return evidence_error
        return _process_error(result) if result.returncode != 0 else None

    def _report(self, program, code, bug_type, message) -> BugReport:
        params, dtype, compute_kind = self._get_meta(program)
        params["input_seed"] = self.config.input_seed
        return BugReport(bug_type=bug_type, error_message=message, params=params, dtype=dtype,
                         compute_kind=compute_kind, generated_code=code)

    def _failure(self, program, code, error_msg, progress=None) -> BugReport:
        """Classify a failed run: bug type, root cause, location and failure bucket."""
        from src.ir.extended import ExtendedProgram
        from src.ir.slice import SliceProgram
        from src.workflow.feedback import confirmed_failure
        from src.workflow.triage import failure_bucket, wrong_result_origin
        extended = isinstance(program, ExtendedProgram)
        bug_type = self._classify_error(error_msg)
        # A skipped numeric check (the oracle trust gate) also ends in the
        # execute stage but is not a crash.
        if (extended and progress.get('stage') in ('execute', 'complete') and not self.config.compile_only
                and bug_type not in (BugType.WRONG_RESULT, BugType.ORACLE_UNSTABLE)):
            bug_type = BugType.RUNTIME_CRASH
        report = self._report(program, code, bug_type, _bounded(error_msg))
        # Classify against the complete diagnostic before keeping its bounded
        # tail; pass names can otherwise be truncated away.
        report.confirmed_signature = confirmed_failure(error_msg, self.backend) or ''
        report.classify_root_cause(self.backend)
        if extended and progress:
            # The progress manifest names the exact variant and stage the
            # subprocess reached, which beats message inference.
            report.location = f"{progress.get('stage')}:{progress.get('variant')}"
        else:
            # Stage markers precede a traceback that can outgrow the kept
            # tail (region harnesses print the program).
            from src.backends.common.diagnostics import _failure_location
            report.location = _failure_location(error_msg)
        # Every wrong value shares the checker's message; the operation
        # producing the earliest one separates the mechanisms.
        origin = (wrong_result_origin(report.params['extended_program'], error_msg)
                  if extended and bug_type is BugType.WRONG_RESULT else '')
        if isinstance(program, SliceProgram) and bug_type is BugType.WRONG_RESULT:
            from src.workflow.slices import slice_origin
            origin = slice_origin(program)
        report.failure_bucket, report.failure_key = failure_bucket(
            error_msg, report.root_cause, report.confirmed_signature, report.location, origin)
        return report

    def _timeout_report(self, program, code, exc, artifact_dir) -> BugReport:
        from src.workflow.triage import failure_bucket
        def decode(value):
            return value.decode(errors='replace') if isinstance(value, bytes) else value or ''
        if artifact_dir is not None:
            (artifact_dir / 'run.log').write_text(decode(exc.stdout) + decode(exc.stderr) + '\nExecution timed out')
        report = self._report(program, code, BugType.TIMEOUT, "Execution timed out")
        report.root_cause = "timeout"
        if artifact_dir is not None:
            progress = self._read_extended_evidence(program, artifact_dir).progress
            if progress:
                report.location = f"{progress.get('stage')}:{progress.get('variant')}"
        else:
            # Region/probe harnesses print TILESMITH_STAGE markers; the last
            # one names where the subprocess hung (prepare, execute, ...).
            from src.backends.common.diagnostics import _failure_location
            report.location = _failure_location(decode(exc.stderr))
        report.failure_bucket, report.failure_key = failure_bucket(
            report.error_message, report.root_cause, location=report.location)
        return report

    def _classify_error(self, error_msg: str) -> BugType:
        return self.backend_impl.classify_error(error_msg)
