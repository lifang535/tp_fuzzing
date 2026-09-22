"""Shape, dtype and scheduling parameters shared by native regions and probes."""
from __future__ import annotations
from dataclasses import dataclass
from enum import Enum


class ComputeKind(Enum):
    """Region entry computation or directed probe operation."""
    GEMM = 'gemm'
    COPY = 'copy'
    REDUCE_SUM = 'reduce_sum'
    REDUCE_MAX = 'reduce_max'
    REDUCE_MIN = 'reduce_min'
    SOFTMAX = 'softmax'
    ARGMAX = 'argmax'
    GEMM_ARGMAX = 'gemm_argmax'


class LoopKind(Enum):
    """How the K-dimension loop is scheduled — structural variation."""
    PIPELINED = "pipelined"
    SERIAL = "serial"


class DataType(Enum):
    FLOAT16 = "float16"
    FLOAT32 = "float32"
    # A/B inputs store int8; the GEMM accumulator (and C) is int32. There is
    # deliberately no INT32 enum: C's dtype is derived, never generated.
    INT8 = "int8"


@dataclass
class TileKernel:
    """Launch specification attached to a RegionProgram.

    Region operations define the dataflow. compute_kind selects its entry or
    whole-function probe, while tile geometry and scheduling control execution.
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

    # Directed paper-derived probes use explicit physical strides and oracles.
    coverage_probe: bool = False
    input_layout: str = "contiguous"
    input_pattern: str = "normal"
    repeat_count: int = 3
    schedule_pair: bool = True
    cache_cycle: bool = False
    # MLIRSmith-style warp-partition policy sweep (tilelang GemmWarpPolicy):
    # 'square' (balanced, the default), 'full_row' (all warps along M) and
    # 'full_col' (all warps along N). Purely structural — the per-tile math is
    # untouched, so policy variants share the one reference.
    warp_policy: str = "square"

    def __post_init__(self):
        if self.compute_kind in (ComputeKind.ARGMAX, ComputeKind.GEMM_ARGMAX):
            self.coverage_probe = True
            if self.input_pattern == "normal":
                self.input_pattern = "integer"
            self.block_N = max(32, self.block_N, 1 << (self.N - 1).bit_length())
        if self.warp_policy not in ("square", "full_row", "full_col"):
            raise ValueError("warp_policy must be square/full_row/full_col")


    @property
    def params_dict(self) -> dict:
        return {
            "M": self.M, "N": self.N, "K": self.K,
            "block_M": self.block_M, "block_N": self.block_N, "block_K": self.block_K,
            "threads": self.threads, "num_stages": self.num_stages,
            "loop_kind": self.loop_kind.value,
            "compute_kind": self.compute_kind.value,
            **({"coverage_probe": True, "input_layout": self.input_layout,
                "input_pattern": self.input_pattern, "repeat_count": self.repeat_count,
                "schedule_pair": self.schedule_pair,
                **({"cache_cycle": True} if self.cache_cycle else {})} if self.coverage_probe else {}),
            **({"warp_policy": self.warp_policy} if self.warp_policy != "square" else {}),
        }
