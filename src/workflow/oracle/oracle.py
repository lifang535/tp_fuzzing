"""
Test Oracle — Executes generated programs and detects bugs.
"""

import os
import sys
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from src.ir import TileProgram
from src.workflow.emitter import get_emitter
from src.ir import TilePipeline
from src.workflow.emitter import TileLangPipelineEmitter, TritonPipelineEmitter
from src.ir import DynamicSequence
from src.workflow.emitter import TileLangDynamicEmitter, TritonDynamicEmitter
from src.config import Config, DEFAULT_CONFIG


class BugType(Enum):
    COMPILE_CRASH = "compile_crash"
    RUNTIME_CRASH = "runtime_crash"
    WRONG_RESULT = "wrong_result"
    TIMEOUT = "timeout"


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

    def classify_root_cause(self):
        err = self.error_message.lower()
        raw = self.error_message  # preserve case for some checks

        if "wrong result" in err:
            self.root_cause = "wrong_result"

        # TileLang / TVM internal errors
        elif "m_warp * n_warp" in err:
            self.root_cause = "warp_partition"
        elif "m must be divisible" in err or "kmperwarp" in err:
            self.root_cause = "alignment"
        elif "unsupported k_dim" in err:
            self.root_cause = "unsupported_k_dim"
        elif "stride" in err and "check failed" in err:
            self.root_cause = "stride_alignment"
        elif "no available layout" in err:
            self.root_cause = "layout_inference"
        elif "shared memory" in err or "shared_memory" in err or "out of resource" in err:
            self.root_cause = "shared_memory_overflow"
        elif "dtype mismatch" in err:
            self.root_cause = "dtype_mismatch"

        # Triton errors
        elif "multiple values for argument" in err:
            # Our code generation bug: duplicate kernel arguments
            self.root_cause = "codegen_duplicate_arg"
        elif "expected dtype" in err and "but got" in err:
            # tl.exp / tl.sqrt don't support fp16
            self.root_cause = "dtype_unsupported_op"
        elif "object has no attribute 'clamp'" in err or "has no attribute" in err:
            # Triton tensor API mismatch in our generated code
            self.root_cause = "codegen_api_mismatch"
        elif "compilationerror" in err or "triton.compiler" in err:
            self.root_cause = "triton_compile_error"
        elif "out of resources" in err:
            self.root_cause = "shared_memory_overflow"
        elif "segfault" in err or "segmentation fault" in err or "signal 11" in err:
            self.root_cause = "segfault"
        elif "assertion" in err and ("failed" in err or "error" in err):
            self.root_cause = "assertion_failure"
        elif "timeout" in err:
            self.root_cause = "timeout"
        # GPU out-of-memory errors (cublas, cuda, torch allocator)
        elif "cublas_status_alloc_failed" in err or "cublascreate" in err:
            self.root_cause = "gpu_oom"
        elif "cuda error" in err and ("alloc" in err or "out of memory" in err or "oom" in err):
            self.root_cause = "gpu_oom"
        elif "out of memory" in err:
            self.root_cause = "gpu_oom"
        else:
            self.root_cause = "other"

    def to_dict(self) -> dict:
        return {
            "bug_type": self.bug_type.value,
            "root_cause": self.root_cause,
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
        self.emitter = get_emitter(backend, config=config)
        # Pipeline emitters
        if backend == "tilelang":
            self.pipeline_emitter = TileLangPipelineEmitter(config=config)
            self.dynamic_emitter = TileLangDynamicEmitter(config=config)
        else:
            self.pipeline_emitter = TritonPipelineEmitter(config=config)
            self.dynamic_emitter = TritonDynamicEmitter(config=config)

    def _emit_code(self, program) -> str:
        """Emit code for a TileProgram, TilePipeline, or DynamicSequence."""
        if isinstance(program, DynamicSequence):
            return self.dynamic_emitter.emit(program)
        if isinstance(program, TilePipeline):
            return self.pipeline_emitter.emit(program)
        return self.emitter.emit(program)

    def _get_meta(self, program):
        """Return (params_dict, dtype_str, compute_kind_str) for bug report."""
        if isinstance(program, DynamicSequence):
            return (
                program.params_dict,
                program.dtype,
                "dynamic:" + "->".join(s.op_kind for s in program.steps),
            )
        if isinstance(program, TilePipeline):
            return (
                program.params_dict,
                program.dtype.value,
                "pipeline:" + "->".join(s.kind.value for s in program.steps),
            )
        kernel = program.kernels[0] if program.kernels else None
        return (
            kernel.params_dict if kernel else {},
            kernel.dtype.value if kernel else "",
            kernel.compute_kind.value if kernel else "",
        )

    def test(self, program) -> Optional[BugReport]:
        """Test a TileProgram or TilePipeline."""
        code = self._emit_code(program)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, prefix="tilesmith_") as f:
            f.write(code)
            tmp_path = f.name

        try:
            result = subprocess.run(
                [sys.executable, tmp_path],
                capture_output=True, text=True,
                timeout=self.config.compile_timeout + self.config.execute_timeout,
            )

            if result.returncode != 0:
                error_msg = result.stderr.strip()
                bug_type = self._classify_error(error_msg)
                params, dtype_str, compute_kind_str = self._get_meta(program)
                report = BugReport(
                    bug_type=bug_type,
                    error_message=error_msg[-2000:],
                    params=params,
                    dtype=dtype_str,
                    compute_kind=compute_kind_str,
                    generated_code=code,
                )
                report.classify_root_cause()
                return report

        except subprocess.TimeoutExpired:
            params, dtype_str, compute_kind_str = self._get_meta(program)
            report = BugReport(
                bug_type=BugType.TIMEOUT,
                error_message="Execution timed out",
                params=params,
                dtype=dtype_str,
                compute_kind=compute_kind_str,
                generated_code=code,
            )
            report.root_cause = "timeout"
            return report
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

        return None

    def _classify_error(self, error_msg: str) -> BugType:
        lower = error_msg.lower()
        if "wrong result" in lower:
            return BugType.WRONG_RESULT
        if "cuda" in lower and "runtime" in lower:
            return BugType.RUNTIME_CRASH
        return BugType.COMPILE_CRASH
