"""
Program Generator — Produces TilePrograms with valid parameters.

Randomizes:
  1. compute_kind: all 15 supported ops, weighted by complexity
  2. Problem shapes: M, N, K (from dim pool)
  3. Tile sizes: block_M, block_N, block_K (hardware-valid)
  4. Structural params: loop_kind (pipelined/serial), num_stages, threads
  5. Op-specific params: alpha (for SCALE)

Pipeline generation (40% of kernels):
  - GEMM epilogue: GEMM → [epilogue ops]* → [terminal]?
  - Elementwise chain: COPY → [shape-preserving ops]+
"""

import random
from typing import List, Union

from src.ir import TileProgram, TileKernel, ComputeKind, LoopKind, DataType
from src.constraints import generate_valid_params
from src.config import Config, DEFAULT_CONFIG
from src.ir import TilePipeline, PipelineGenerator
from src.ir import DynamicSequence, DynamicSequenceGenerator


# Weighted distribution over compute kinds
_KIND_WEIGHTS = [
    (ComputeKind.GEMM,        20),
    (ComputeKind.COPY,        10),
    (ComputeKind.ELEMWISE_ADD,  8),
    (ComputeKind.ELEMWISE_MUL,  8),
    (ComputeKind.ELEMWISE_MAX,  8),
    (ComputeKind.ELEMWISE_SUB,  8),
    (ComputeKind.SCALE,         8),
    (ComputeKind.UNARY_EXP,     6),
    (ComputeKind.UNARY_SQRT,    6),
    (ComputeKind.TRANSPOSE,     8),
    (ComputeKind.REDUCE_SUM,    6),
    (ComputeKind.REDUCE_MAX,    6),
    (ComputeKind.REDUCE_MIN,    6),
    (ComputeKind.SOFTMAX,       5),
    (ComputeKind.WHERE,         5),
]

_KINDS = [k for k, _ in _KIND_WEIGHTS]
_WEIGHTS = [w for _, w in _KIND_WEIGHTS]


class TypeGenerator:
    """Generates shapes from a shared pool — like MLIRSmith's TypeGeneration."""

    def __init__(self, config: Config = DEFAULT_CONFIG):
        self.config = config
        self.dim_pool: List[int] = []
        self._init_pool()

    def _init_pool(self):
        self.dim_pool.clear()
        if self.config.easy_shape:
            # Easy-shape mode: use only power-of-2 values so that M/N/K are
            # always divisible by common block sizes. This avoids boundary
            # handling code paths and raises the pass rate.
            for _ in range(self.config.dim_pool_size):
                self.dim_pool.append(random.choice(self.config.easy_shape_values))
        else:
            lo, hi = self.config.dim_range
            for _ in range(self.config.dim_pool_size):
                self.dim_pool.append(random.randint(lo, hi))

    def random_dtype(self) -> DataType:
        choice = random.choice(self.config.supported_dtypes)
        if isinstance(choice, str):
            return DataType(choice)
        return choice


class ProgramGenerator:
    """Top-level generator — produces valid TilePrograms or TilePipelines."""

    def __init__(self, config: Config = DEFAULT_CONFIG, backend: str = "tilelang"):
        self.config = config
        self.backend = backend
        self.type_gen = TypeGenerator(config)
        self.pipeline_gen = PipelineGenerator(config, backend=backend)
        self.dynamic_gen = DynamicSequenceGenerator(config, backend=backend)

    def generate(self) -> Union[TileProgram, TilePipeline, DynamicSequence]:
        """
        Generate a test case:
          40% → TilePipeline (template-based)
          30% → DynamicSequence (pool-driven, MLIRSmith-style)
          30% → single-kernel TileProgram
        """
        r = random.random()
        if r < self.config.pipeline_prob:
            dtype = self.type_gen.random_dtype()
            return self.pipeline_gen.generate(self.type_gen.dim_pool, dtype)
        elif r < self.config.pipeline_prob + self.config.dynamic_prob:
            dtype = self.type_gen.random_dtype()
            return self._generate_dynamic(dtype)
        program = TileProgram()
        kernel = self._generate_kernel("kernel_0")
        program.kernels.append(kernel)
        return program

    def _generate_dynamic(self, dtype: DataType) -> DynamicSequence:
        """Generate a DynamicSequence using pool-driven op selection."""
        from src.ir import REDUCE_OPS

        params = generate_valid_params(self.backend, self.type_gen.dim_pool, dtype)
        loop_kind = random.choice([LoopKind.PIPELINED, LoopKind.SERIAL])
        num_stages = (
            random.choice(self.config.pipeline_stages_choices)
            if loop_kind == LoopKind.PIPELINED else 1
        )
        # Re-validate with actual num_stages (generate_valid_params uses stages=1 default)
        if loop_kind == LoopKind.PIPELINED and num_stages > 1:
            from src.constraints.constraints import (
                tilelang_check_shared_memory, triton_check_shared_memory,
            )
            check_fn = tilelang_check_shared_memory if self.backend == "tilelang" else triton_check_shared_memory
            if not check_fn(params["block_M"], params["block_N"], params["block_K"], dtype, num_stages):
                for ns in sorted(self.config.pipeline_stages_choices):
                    if check_fn(params["block_M"], params["block_N"], params["block_K"], dtype, ns):
                        num_stages = ns
                        break
                else:
                    num_stages = 1
        params["loop_kind"] = loop_kind.value
        params["num_stages"] = num_stages
        params["name"] = "kernel_0"
        # Pass config-driven hyperparameters into params so dynamic_seq can use them
        params["diversity_boost"] = self.config.diversity_boost
        params["scale_alpha_min"] = self.config.scale_alpha_min
        params["scale_alpha_max"] = self.config.scale_alpha_max

        max_steps = random.randint(self.config.dynamic_max_steps_min, self.config.dynamic_max_steps_max)
        return self.dynamic_gen.generate(params, dtype.value, max_steps=max_steps)

    def _generate_kernel(self, name: str) -> TileKernel:
        dtype = self.type_gen.random_dtype()
        compute_kind = random.choices(_KINDS, weights=_WEIGHTS, k=1)[0]

        # Generate hardware-valid parameters
        params = generate_valid_params(
            self.backend, self.type_gen.dim_pool, dtype,
            transpose=(compute_kind == ComputeKind.TRANSPOSE),
        )

        # Structural variation
        loop_kind = random.choice([LoopKind.PIPELINED, LoopKind.SERIAL])
        num_stages = random.choice(self.config.pipeline_stages_choices) if loop_kind == LoopKind.PIPELINED else 1

        # Re-validate shared memory with the actual num_stages.
        # generate_valid_params checks with num_stages=1 by default, but the
        # actual num_stages may be larger and require more shared memory.
        if loop_kind == LoopKind.PIPELINED and num_stages > 1:
            from src.constraints.constraints import (
                tilelang_check_shared_memory, triton_check_shared_memory,
                tilelang_valid_block_k, triton_valid_block_sizes,
            )
            check_fn = tilelang_check_shared_memory if self.backend == "tilelang" else triton_check_shared_memory
            valid_k = tilelang_valid_block_k(dtype) if self.backend == "tilelang" else [16, 32, 64, 128]
            if not check_fn(params["block_M"], params["block_N"], params["block_K"], dtype, num_stages):
                # Reduce num_stages until shared memory fits
                for ns in sorted(self.config.pipeline_stages_choices):
                    if check_fn(params["block_M"], params["block_N"], params["block_K"], dtype, ns):
                        num_stages = ns
                        break
                else:
                    num_stages = 1  # fallback to single stage

        # SOFTMAX and REDUCE ops process one full row per block.
        # N must equal block_N so there is exactly one block per row —
        # otherwise multiple blocks would overwrite the same output element.
        from src.ir import REDUCE_OPS
        if compute_kind == ComputeKind.SOFTMAX or compute_kind in REDUCE_OPS:
            params["N"] = params["block_N"]

        # Op-specific parameters
        alpha = round(random.uniform(self.config.scale_alpha_min, self.config.scale_alpha_max), 4) if compute_kind == ComputeKind.SCALE else 1.0

        return TileKernel(
            name=name,
            compute_kind=compute_kind,
            M=params["M"], N=params["N"], K=params["K"],
            block_M=params["block_M"], block_N=params["block_N"], block_K=params["block_K"],
            loop_kind=loop_kind,
            num_stages=num_stages,
            threads=params["threads"],
            dtype=dtype,
            alpha=alpha,
        )
