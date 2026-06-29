"""
Mutation Engine — Applies parametric and structural mutations.
All mutations preserve the compute_kind (WHAT is computed doesn't change),
only HOW it's computed varies.
"""

import random
import copy

from src.ir import TileProgram, TileKernel, ComputeKind, LoopKind, DataType
from src.config import Config, DEFAULT_CONFIG
from src.constraints import (
    tilelang_check_shared_memory, tilelang_check_warp_partition,
    tilelang_valid_block_m_n, tilelang_valid_block_k,
    triton_valid_block_sizes, triton_check_shared_memory,
    generate_valid_params,
)
from src.ir import TilePipeline, PipelineStep


class Mutator:
    def __init__(self, config: Config = DEFAULT_CONFIG, backend: str = "tilelang"):
        self.config = config
        self.backend = backend

    def mutate(self, program):
        """Mutate a TileProgram, TilePipeline, or DynamicSequence."""
        # Import here to avoid circular imports
        from src.ir import DynamicSequence, DynamicSequenceGenerator
        if isinstance(program, DynamicSequence):
            return self._mutate_dynamic(program)
        if isinstance(program, TilePipeline):
            return self._mutate_pipeline(program)
        return self._mutate_program(program)

    def _mutate_dynamic(self, seq):
        """Mutate a DynamicSequence by regenerating with perturbed parameters."""
        from src.ir import DynamicSequenceGenerator
        from src.ir import DataType

        gen = DynamicSequenceGenerator(self.config, backend=self.backend)

        # Mutate the dimensions
        dim_attrs = ["M", "N", "K", "block_M", "block_N", "block_K", "threads", "num_stages"]
        mutation = random.choice(["shape", "tile_size", "num_stages", "regenerate"])

        def _fix_warp(params):
            """Ensure threads is compatible with block_M and block_N."""
            if not tilelang_check_warp_partition(params["block_M"], params["block_N"], params["threads"]):
                for t in [128, 256]:
                    if tilelang_check_warp_partition(params["block_M"], params["block_N"], t):
                        params["threads"] = t
                        return
                # If neither threads value works, fall back to smaller tiles
                params["block_M"] = 32
                params["block_N"] = 32
                params["threads"] = 128

        if mutation == "regenerate":
            params = {
                "M": seq.M, "N": seq.N, "K": seq.K,
                "block_M": seq.block_M, "block_N": seq.block_N, "block_K": seq.block_K,
                "threads": seq.threads, "num_stages": seq.num_stages,
                "loop_kind": seq.loop_kind, "name": seq.name,
            }
            _fix_warp(params)
            return gen.generate(params, seq.dtype, max_steps=random.randint(self.config.dynamic_max_steps_min, self.config.dynamic_max_steps_max))

        elif mutation == "shape":
            attr = random.choice(["M", "N", "K"])
            val = 2 ** random.randint(4, 9)
            params = {
                "M": seq.M, "N": seq.N, "K": seq.K,
                "block_M": seq.block_M, "block_N": seq.block_N, "block_K": seq.block_K,
                "threads": seq.threads, "num_stages": seq.num_stages,
                "loop_kind": seq.loop_kind, "name": seq.name,
            }
            params[attr] = max(16, val)
            if any(s.op_kind in ("softmax", "reduce_sum", "reduce_max", "reduce_min")
                   for s in seq.steps):
                params["N"] = params["block_N"]
            _fix_warp(params)
            return gen.generate(params, seq.dtype, max_steps=random.randint(self.config.dynamic_max_steps_min, self.config.dynamic_max_steps_max))

        elif mutation == "tile_size":
            valid_mn = tilelang_valid_block_m_n() if self.backend == "tilelang" else triton_valid_block_sizes()
            from src.ir.ir import DataType
            dtype_obj = DataType(seq.dtype) if isinstance(seq.dtype, str) else seq.dtype
            valid_k = tilelang_valid_block_k(dtype_obj) if self.backend == "tilelang" else [16, 32, 64, 128]

            # Try random tile sizes, validate shared memory before accepting
            block_M = random.choice(valid_mn)
            block_N = random.choice(valid_mn)
            block_K = seq.block_K
            num_stages = seq.num_stages

            # Reduce block sizes until shared memory fits
            check_fn = (tilelang_check_shared_memory if self.backend == "tilelang"
                        else triton_check_shared_memory)
            for bk in [block_K] + sorted(valid_k):
                if check_fn(block_M, block_N, bk, dtype_obj, num_stages=max(1, num_stages)):
                    block_K = bk
                    break
            else:
                # Fall back to smallest possible tile sizes
                block_M, block_N, block_K = 16, 16, valid_k[0]

            params = {
                "M": seq.M, "N": seq.N, "K": seq.K,
                "block_M": block_M, "block_N": block_N, "block_K": block_K,
                "threads": seq.threads, "num_stages": num_stages,
                "loop_kind": seq.loop_kind, "name": seq.name,
            }
            _fix_warp(params)
            return gen.generate(params, seq.dtype, max_steps=random.randint(self.config.dynamic_max_steps_min, self.config.dynamic_max_steps_max))

        else:  # num_stages
            params = {
                "M": seq.M, "N": seq.N, "K": seq.K,
                "block_M": seq.block_M, "block_N": seq.block_N, "block_K": seq.block_K,
                "threads": seq.threads, "num_stages": random.choice([1, 2, 3, 4]),
                "loop_kind": seq.loop_kind, "name": seq.name,
            }
            _fix_warp(params)
            return gen.generate(params, seq.dtype, max_steps=random.randint(self.config.dynamic_max_steps_min, self.config.dynamic_max_steps_max))

    def _mutate_program(self, program: TileProgram) -> TileProgram:
        new_prog = copy.deepcopy(program)
        if not new_prog.kernels:
            return new_prog

        kernel = random.choice(new_prog.kernels)
        mutation = random.choices(
            [
                self._mutate_shape,
                self._mutate_tile_size,
                self._mutate_dtype,
                self._mutate_loop_kind,
                self._mutate_num_stages,
                self._mutate_threads,
                self._mutate_compute_kind,
                self._mutate_boundary,
            ],
            weights=[0.15, 0.15, 0.10, 0.15, 0.10, 0.10, 0.10, 0.15],
            k=1,
        )[0]
        mutation(kernel)
        self._enforce_constraints(kernel)
        return new_prog

    def _mutate_pipeline(self, pipeline: TilePipeline) -> TilePipeline:
        """Mutate a pipeline: change shape params or add/remove an epilogue step."""
        p = copy.deepcopy(pipeline)
        strategy = random.choices(
            ["shape", "tile_size", "dtype", "loop_kind", "num_stages", "epilogue_step"],
            weights=[0.15, 0.15, 0.10, 0.15, 0.10, 0.35],
            k=1,
        )[0]

        EPILOGUE_OPS = [
            ComputeKind.SCALE, ComputeKind.ELEMWISE_ADD, ComputeKind.ELEMWISE_MUL,
            ComputeKind.UNARY_EXP, ComputeKind.UNARY_SQRT, ComputeKind.WHERE,
        ]

        if strategy == "shape":
            attr = random.choice(["M", "N", "K"])
            val = 2 ** random.randint(4, 9)
            setattr(p, attr, max(16, val))
        elif strategy == "tile_size":
            valid_mn = tilelang_valid_block_m_n() if self.backend == "tilelang" else triton_valid_block_sizes()
            attr = random.choice(["block_M", "block_N"])
            setattr(p, attr, random.choice(valid_mn))
        elif strategy == "dtype":
            choice = random.choice(self.config.supported_dtypes)
            if isinstance(choice, str):
                choice = DataType(choice)
            p.dtype = choice
        elif strategy == "loop_kind":
            p.loop_kind = LoopKind.SERIAL if p.loop_kind == LoopKind.PIPELINED else LoopKind.PIPELINED
            if p.loop_kind == LoopKind.PIPELINED:
                p.num_stages = random.choice(self.config.pipeline_stages_choices)
        elif strategy == "num_stages":
            p.num_stages = random.choice([1, 2, 3, 4])
        elif strategy == "epilogue_step":
            # Find epilogue range (after first step, before any terminal)
            from src.ir import REDUCE_OPS
            TERMINAL_OPS = {ComputeKind.SOFTMAX} | REDUCE_OPS
            # Identify epilogue steps (not first, not terminal at end)
            terminal_at_end = len(p.steps) > 1 and p.steps[-1].kind in TERMINAL_OPS
            if terminal_at_end:
                epilogue_range = p.steps[1:-1]
                terminal = p.steps[-1]
                first = p.steps[0]
            else:
                epilogue_range = p.steps[1:]
                terminal = None
                first = p.steps[0]

            action = random.choice(["add", "remove", "replace"])
            if action == "add":
                kind = random.choice(EPILOGUE_OPS)
                alpha = round(random.uniform(self.config.scale_alpha_min, self.config.scale_alpha_max), 4) if kind == ComputeKind.SCALE else 1.0
                new_step = PipelineStep(kind=kind, alpha=alpha)
                insert_pos = random.randint(0, len(epilogue_range))
                epilogue_range.insert(insert_pos, new_step)
            elif action == "remove" and epilogue_range:
                epilogue_range.pop(random.randint(0, len(epilogue_range) - 1))
            elif action == "replace" and epilogue_range:
                idx = random.randint(0, len(epilogue_range) - 1)
                kind = random.choice(EPILOGUE_OPS)
                alpha = round(random.uniform(self.config.scale_alpha_min, self.config.scale_alpha_max), 4) if kind == ComputeKind.SCALE else 1.0
                epilogue_range[idx] = PipelineStep(kind=kind, alpha=alpha)

            # Rebuild steps
            p.steps = [first] + epilogue_range
            if terminal is not None:
                p.steps.append(terminal)

        # Enforce constraints
        from src.ir import REDUCE_OPS
        TERMINAL_OPS = {ComputeKind.SOFTMAX} | REDUCE_OPS
        last_kind = p.steps[-1].kind if p.steps else None
        if last_kind in TERMINAL_OPS:
            p.N = p.block_N

        # Clamp block sizes
        valid_mn = tilelang_valid_block_m_n() if self.backend == "tilelang" else triton_valid_block_sizes()
        valid_k = tilelang_valid_block_k(p.dtype) if self.backend == "tilelang" else [16, 32, 64, 128]

        if p.block_M not in valid_mn:
            p.block_M = min(valid_mn, key=lambda x: abs(x - p.block_M))
        if p.block_N not in valid_mn:
            p.block_N = min(valid_mn, key=lambda x: abs(x - p.block_N))

        # Enforce shared memory constraint
        check_fn = (tilelang_check_shared_memory if self.backend == "tilelang"
                    else triton_check_shared_memory)
        stages = p.num_stages if hasattr(p, 'num_stages') else 1
        if not check_fn(p.block_M, p.block_N, p.block_K, p.dtype, num_stages=max(1, stages)):
            for bk in sorted(valid_k):
                if check_fn(p.block_M, p.block_N, bk, p.dtype, num_stages=max(1, stages)):
                    p.block_K = bk
                    break

        # Enforce warp partition constraint
        if not tilelang_check_warp_partition(p.block_M, p.block_N, p.threads):
            for t in [128, 256]:
                if tilelang_check_warp_partition(p.block_M, p.block_N, t):
                    p.threads = t
                    break

        return p

    def _mutate_shape(self, k: TileKernel):
        attr = random.choice(["M", "N", "K"])
        strategy = random.choice(["power2", "boundary", "prime", "extreme"])
        if strategy == "power2":
            val = 2 ** random.randint(4, 11)
        elif strategy == "boundary":
            val = 2 ** random.randint(5, 10) + random.choice([-1, 0, 1])
        elif strategy == "prime":
            val = random.choice([17, 31, 67, 127, 257, 509, 1021])
        else:
            val = random.choice([1, 3, 7, 4096])
        setattr(k, attr, max(1, val))

    def _mutate_tile_size(self, k: TileKernel):
        valid_mn = tilelang_valid_block_m_n() if self.backend == "tilelang" else triton_valid_block_sizes()
        # Triton tl.dot requires BLOCK_K >= 16
        valid_k = tilelang_valid_block_k(k.dtype) if self.backend == "tilelang" else [16, 32, 64, 128]
        attr = random.choice(["block_M", "block_N", "block_K"])
        if attr == "block_K":
            setattr(k, attr, random.choice(valid_k))
        else:
            setattr(k, attr, random.choice(valid_mn))

    def _mutate_dtype(self, k: TileKernel):
        choice = random.choice(self.config.supported_dtypes)
        if isinstance(choice, str):
            choice = DataType(choice)
        k.dtype = choice

    def _mutate_loop_kind(self, k: TileKernel):
        k.loop_kind = LoopKind.SERIAL if k.loop_kind == LoopKind.PIPELINED else LoopKind.PIPELINED
        if k.loop_kind == LoopKind.PIPELINED:
            k.num_stages = random.choice(self.config.pipeline_stages_choices)

    def _mutate_num_stages(self, k: TileKernel):
        k.num_stages = random.choice([1, 2, 3, 4])
        if k.num_stages > 1:
            k.loop_kind = LoopKind.PIPELINED

    def _mutate_threads(self, k: TileKernel):
        k.threads = random.choice(self.config.thread_choices)

    def _mutate_compute_kind(self, k: TileKernel):
        k.compute_kind = random.choice([
            ComputeKind.GEMM,
            ComputeKind.COPY,
            ComputeKind.ELEMWISE_ADD,
            ComputeKind.ELEMWISE_MUL,
            ComputeKind.ELEMWISE_MAX,
            ComputeKind.ELEMWISE_SUB,
            ComputeKind.SCALE,
            ComputeKind.UNARY_EXP,
            ComputeKind.UNARY_SQRT,
            ComputeKind.TRANSPOSE,
            ComputeKind.REDUCE_SUM,
            ComputeKind.REDUCE_MAX,
            ComputeKind.REDUCE_MIN,
            ComputeKind.SOFTMAX,
            ComputeKind.WHERE,
        ])
        # Regenerate op-specific parameters when kind changes
        if k.compute_kind == ComputeKind.SCALE:
            k.alpha = round(random.uniform(self.config.scale_alpha_min, self.config.scale_alpha_max), 4)
        from src.ir import REDUCE_OPS
        if k.compute_kind == ComputeKind.SOFTMAX or k.compute_kind in REDUCE_OPS:
            k.N = k.block_N  # single block per row required

    def _mutate_boundary(self, k: TileKernel):
        """Make shape not divisible by tile — tests boundary handling."""
        attr = random.choice(["M", "N", "K"])
        tile_attr = "block_" + attr
        tile_val = getattr(k, tile_attr)
        new_val = tile_val * random.randint(1, 5) + random.randint(1, max(1, tile_val - 1))
        setattr(k, attr, new_val)

    def _enforce_constraints(self, k: TileKernel):
        from src.ir import REDUCE_OPS
        valid_mn = tilelang_valid_block_m_n() if self.backend == "tilelang" else triton_valid_block_sizes()
        valid_k = tilelang_valid_block_k(k.dtype) if self.backend == "tilelang" else [16, 32, 64, 128]

        if k.block_M not in valid_mn:
            k.block_M = min(valid_mn, key=lambda x: abs(x - k.block_M))
        if k.block_N not in valid_mn:
            k.block_N = min(valid_mn, key=lambda x: abs(x - k.block_N))
        if k.block_K not in valid_k:
            k.block_K = min(valid_k, key=lambda x: abs(x - k.block_K))

        if not tilelang_check_warp_partition(k.block_M, k.block_N, k.threads):
            for t in [128, 256]:
                if tilelang_check_warp_partition(k.block_M, k.block_N, t):
                    k.threads = t
                    break

        # Softmax and reduce ops require N == block_N (single block per row)
        if k.compute_kind == ComputeKind.SOFTMAX or k.compute_kind in REDUCE_OPS:
            k.N = k.block_N

        # Transpose allocates two shared buffers (A + B transposed), double the cost
        shmem_multiplier = 2 if k.compute_kind == ComputeKind.TRANSPOSE else 1
        if not tilelang_check_shared_memory(k.block_M, k.block_N, k.block_K, k.dtype, num_stages=4):
            for bk in sorted(valid_k):
                if tilelang_check_shared_memory(k.block_M, k.block_N, bk, k.dtype, num_stages=4):
                    k.block_K = bk
                    break
        # For transpose: check A_shared + B_shared both fit
        if k.compute_kind == ComputeKind.TRANSPOSE:
            from src.constraints import dtype_bytes, MAX_SHARED_MEMORY_BYTES
            elem = dtype_bytes(k.dtype)
            needed = k.block_M * k.block_N * elem * 2  # A_shared + B_shared
            if needed > MAX_SHARED_MEMORY_BYTES:
                # Reduce block_N until it fits
                for bn in sorted(valid_mn):
                    if k.block_M * bn * elem * 2 <= MAX_SHARED_MEMORY_BYTES:
                        k.block_N = bn
                        break
