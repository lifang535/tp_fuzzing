"""
Tile Program IR — Abstract representation of tile programs.

Design principle: The IR fully determines the computation semantics.
Both the Emitter (generates backend code) and the Oracle (generates reference)
derive their behavior from the SAME IR. This guarantees consistency.

Computation model:
  A TileKernel operates on 2D tiles with a grid of thread blocks.
  Each block processes one output tile C[by*block_M : (by+1)*block_M, bx*block_N : (bx+1)*block_N].
  The computation is a sequence of ComputeSteps that define what happens to each tile.

ComputeStep types (each has clear semantics for both code generation and reference):
  - GEMM: C += A_tile @ B_tile (matmul accumulation over K dimension)
  - COPY: B = A (identity copy, tests memory path)
  - ELEMWISE: B = f(A) where f is a pointwise function
  - REDUCE: B = reduce(A, axis) where reduce is sum/max/min
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, List


class ComputeKind(Enum):
    """What computation the kernel performs — determines BOTH code gen and reference."""
    # Original 4
    GEMM = "gemm"              # C = A @ B
    COPY = "copy"              # B = A (identity)
    ELEMWISE_ADD = "add"       # C = A + B
    ELEMWISE_MUL = "mul"       # C = A * B

    # Batch 1 — simple elementwise
    ELEMWISE_MAX = "max"       # C = max(A, B)
    ELEMWISE_SUB = "sub"       # C = A - B
    SCALE = "scale"            # B = alpha * A
    UNARY_EXP = "exp"          # B = exp(A)
    UNARY_SQRT = "sqrt"        # B = sqrt(|A|)
    TRANSPOSE = "transpose"    # B[j,i] = A[i,j]

    # Batch 2 — reduce (output shape is (M,) not (M,N))
    REDUCE_SUM = "reduce_sum"  # B[i] = sum(A[i,:])
    REDUCE_MAX = "reduce_max"  # B[i] = max(A[i,:])
    REDUCE_MIN = "reduce_min"  # B[i] = min(A[i,:])

    # Batch 3 — compound
    SOFTMAX = "softmax"        # B[i,j] = softmax(A[i,:])
    WHERE = "where"            # C = where(A > 0, A, B)


# Sets of ops by output shape / structural category
REDUCE_OPS = {
    ComputeKind.REDUCE_SUM,
    ComputeKind.REDUCE_MAX,
    ComputeKind.REDUCE_MIN,
}

UNARY_OPS = {
    ComputeKind.COPY,
    ComputeKind.SCALE,
    ComputeKind.UNARY_EXP,
    ComputeKind.UNARY_SQRT,
    ComputeKind.SOFTMAX,
}

BINARY_OPS = {
    ComputeKind.ELEMWISE_ADD,
    ComputeKind.ELEMWISE_MUL,
    ComputeKind.ELEMWISE_MAX,
    ComputeKind.ELEMWISE_SUB,
    ComputeKind.WHERE,
}


class LoopKind(Enum):
    """How the K-dimension loop is scheduled — structural variation."""
    PIPELINED = "pipelined"
    SERIAL = "serial"


class DataType(Enum):
    FLOAT16 = "float16"
    FLOAT32 = "float32"


@dataclass
class TileKernel:
    """
    A complete tile kernel with clear semantics.

    The compute_kind determines:
      - What TileLang/Triton code to emit
      - What reference to compare against
    The structural params (loop_kind, num_stages, threads) vary HOW
    the computation is executed without changing WHAT is computed.
    """
    name: str

    # WHAT to compute (determines reference)
    compute_kind: ComputeKind = ComputeKind.GEMM

    # Problem shape
    M: int = 128
    N: int = 128
    K: int = 128  # Only used for GEMM

    # Tile shape (how to partition the work)
    block_M: int = 64
    block_N: int = 64
    block_K: int = 32  # Only used for GEMM

    # Structural parameters (HOW to execute — vary without changing semantics)
    loop_kind: LoopKind = LoopKind.PIPELINED
    num_stages: int = 2
    threads: int = 128
    dtype: DataType = DataType.FLOAT16

    # Op-specific parameters
    alpha: float = 1.0  # Used by SCALE op

    @property
    def acc_dtype(self) -> str:
        return "float32" if self.dtype == DataType.FLOAT16 else self.dtype.value

    @property
    def output_shape(self) -> tuple:
        """Shape of the kernel's primary output tensor."""
        if self.compute_kind in REDUCE_OPS:
            return (self.M,)
        elif self.compute_kind == ComputeKind.TRANSPOSE:
            return (self.N, self.M)
        else:
            return (self.M, self.N)

    @property
    def params_dict(self) -> dict:
        return {
            "M": self.M, "N": self.N, "K": self.K,
            "block_M": self.block_M, "block_N": self.block_N, "block_K": self.block_K,
            "threads": self.threads, "num_stages": self.num_stages,
            "loop_kind": self.loop_kind.value,
            "compute_kind": self.compute_kind.value,
            "alpha": self.alpha,
        }


@dataclass
class TileProgram:
    """A complete tile program."""
    kernels: List[TileKernel] = field(default_factory=list)
