"""
Pipeline IR and Generator — Multi-step op sequence support.

A TilePipeline is a sequence of compute steps that execute on the same tile,
exposing bugs at op boundaries (type mismatches, precision loss, shape errors).

Valid pipeline patterns:
  1. GEMM epilogue:    GEMM → [epilogue ops]* → [terminal]?
  2. Elementwise chain: COPY → [shape-preserving ops]+
  3. Single op:        Any single ComputeKind (backward compat wrapper)
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import List

from .ir import ComputeKind, LoopKind, DataType, REDUCE_OPS
from src.constraints import generate_valid_params
from src.config import Config, DEFAULT_CONFIG


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline IR
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PipelineStep:
    """One step in a compute pipeline."""
    kind: ComputeKind
    # For SCALE: scalar multiplier
    alpha: float = 1.0


@dataclass
class TilePipeline:
    """
    A sequence of compute steps operating on the same tile.
    All steps after the first operate on the accumulator fragment in-place.

    Valid pipeline patterns:
    1. GEMM epilogue: GEMM → [SCALE|ELEMWISE_ADD|ELEMWISE_MUL|UNARY_EXP|UNARY_SQRT|WHERE]* → [SOFTMAX|REDUCE_SUM|REDUCE_MAX|REDUCE_MIN]?
    2. Elementwise chain: COPY → [any shape-preserving op]+
    3. Single op: any single ComputeKind (backward compat)
    """
    steps: List[PipelineStep] = field(default_factory=list)

    # Kernel dimensions
    M: int = 128
    N: int = 128
    K: int = 128
    block_M: int = 64
    block_N: int = 64
    block_K: int = 32
    threads: int = 128
    loop_kind: LoopKind = LoopKind.PIPELINED
    num_stages: int = 2
    dtype: DataType = DataType.FLOAT16
    name: str = "kernel_0"

    @property
    def first_kind(self) -> ComputeKind:
        return self.steps[0].kind

    @property
    def last_kind(self) -> ComputeKind:
        return self.steps[-1].kind

    @property
    def is_gemm_pipeline(self) -> bool:
        return bool(self.steps) and self.steps[0].kind == ComputeKind.GEMM

    @property
    def is_pipeline(self) -> bool:
        """True if this is a multi-step pipeline (not single-op)."""
        return len(self.steps) > 1

    @property
    def acc_dtype(self) -> str:
        return "float32" if self.dtype == DataType.FLOAT16 else self.dtype.value

    @property
    def output_shape(self) -> tuple:
        if not self.steps:
            return (self.M, self.N)
        last = self.steps[-1].kind
        if last in REDUCE_OPS:
            return (self.M,)
        elif last == ComputeKind.TRANSPOSE:
            return (self.N, self.M)
        else:
            return (self.M, self.N)

    @property
    def params_dict(self) -> dict:
        return {
            "M": self.M, "N": self.N, "K": self.K,
            "block_M": self.block_M, "block_N": self.block_N, "block_K": self.block_K,
            "threads": self.threads, "num_stages": self.num_stages,
            "pipeline": [s.kind.value for s in self.steps],
            "pipeline_alphas": [s.alpha for s in self.steps],
        }


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline Generator
# ─────────────────────────────────────────────────────────────────────────────

class PipelineGenerator:
    """Generates valid TilePipelines."""

    # Shape-preserving ops that can appear in epilogue (after GEMM or in elem chain)
    EPILOGUE_OPS = [
        ComputeKind.SCALE,
        ComputeKind.ELEMWISE_ADD,
        ComputeKind.ELEMWISE_MUL,
        ComputeKind.UNARY_EXP,
        ComputeKind.UNARY_SQRT,
        ComputeKind.WHERE,
    ]

    # Terminal ops — can only appear at end (change shape or complex)
    TERMINAL_OPS = [
        ComputeKind.SOFTMAX,
        ComputeKind.REDUCE_SUM,
        ComputeKind.REDUCE_MAX,
        ComputeKind.REDUCE_MIN,
    ]

    def __init__(self, config: Config = DEFAULT_CONFIG, backend: str = "tilelang"):
        self.config = config
        self.backend = backend

    def generate(self, dim_pool: list, dtype: DataType) -> TilePipeline:
        """Generate a random valid pipeline."""
        strategy = random.choices(
            ["gemm_epilogue", "elemwise_chain"],
            weights=[self.config.pipeline_gemm_epilogue_prob, 1.0 - self.config.pipeline_gemm_epilogue_prob],
            k=1,
        )[0]

        if strategy == "gemm_epilogue":
            steps = self._gemm_epilogue()
        else:
            steps = self._elemwise_chain()

        # Check if terminal (reduce/softmax) — need N == block_N
        last_kind = steps[-1].kind
        needs_reduce_constraint = (last_kind in REDUCE_OPS or last_kind == ComputeKind.SOFTMAX)

        # Generate hardware-valid parameters for GEMM (most restrictive)
        params = generate_valid_params(
            self.backend, dim_pool, dtype,
            transpose=False,
        )

        # For REDUCE/SOFTMAX at end, N must equal block_N
        if needs_reduce_constraint:
            params["N"] = params["block_N"]

        # Loop kind
        loop_kind = random.choice([LoopKind.PIPELINED, LoopKind.SERIAL])
        num_stages = (
            random.choice(self.config.pipeline_stages_choices)
            if loop_kind == LoopKind.PIPELINED
            else 1
        )

        return TilePipeline(
            steps=steps,
            M=params["M"],
            N=params["N"],
            K=params["K"],
            block_M=params["block_M"],
            block_N=params["block_N"],
            block_K=params["block_K"],
            threads=params["threads"],
            loop_kind=loop_kind,
            num_stages=num_stages,
            dtype=dtype,
            name="kernel_0",
        )

    def _gemm_epilogue(self) -> List[PipelineStep]:
        """GEMM followed by 1-2 epilogue ops, optionally ending with terminal."""
        steps = [PipelineStep(ComputeKind.GEMM)]
        # Add 0-2 epilogue ops
        n_epilogue = random.randint(0, self.config.pipeline_max_epilogue_ops)
        for _ in range(n_epilogue):
            kind = random.choice(self.EPILOGUE_OPS)
            alpha = round(random.uniform(self.config.scale_alpha_min, self.config.scale_alpha_max), 4) if kind == ComputeKind.SCALE else 1.0
            steps.append(PipelineStep(kind=kind, alpha=alpha))
        # Optionally add terminal
        if random.random() < self.config.pipeline_terminal_prob:
            steps.append(PipelineStep(kind=random.choice(self.TERMINAL_OPS)))
        return steps

    def _elemwise_chain(self) -> List[PipelineStep]:
        """COPY followed by 1-2 more shape-preserving ops."""
        # Start with COPY
        steps = [PipelineStep(ComputeKind.COPY)]
        for _ in range(random.randint(1, self.config.pipeline_max_elemwise_ops)):
            kind = random.choice(self.EPILOGUE_OPS)
            alpha = round(random.uniform(self.config.scale_alpha_min, self.config.scale_alpha_max), 4) if kind == ComputeKind.SCALE else 1.0
            steps.append(PipelineStep(kind=kind, alpha=alpha))
        return steps
