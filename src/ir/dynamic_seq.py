"""
Dynamic Sequence Generator — TypedValuePool-equivalent dynamic op sequence generation.

Analogous to MLIRSmith's TypedValuePool + RegionGen:
  1. Maintains TileValuePool of available buffers
  2. Each step: finds all applicable op generators
  3. Selects one by weighted random (with diversity boost for uncovered ops)
  4. Generates the op, adds result buffer to pool
  5. Repeats for max_steps — sequence is never pre-planned

This replaces template-based pipeline.py with truly dynamic generation.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import List, Optional

from src.constraints import generate_valid_params
from .ir import DataType, LoopKind


# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TileBuffer:
    """
    A concrete buffer available in the current kernel scope.
    Analogous to TypeValue in MLIRSmith.
    """
    name: str         # variable name in generated code, e.g. "C_local_1", "A_shared_1"
    shape: tuple      # (M, N) or (M, K) or (K, N) or (block_M, block_N) etc.
    dtype: str        # "float16", "float32"
    scope: str        # "global", "shared", "fragment"
    torch_ref: str    # Python expression computing this buffer's value using input tensor names


@dataclass
class TileValuePool:
    """
    Tracks available buffers at each scope.
    Analogous to MLIRSmith's TypedValuePool.
    """
    global_in: List[TileBuffer] = field(default_factory=list)
    global_out: List[TileBuffer] = field(default_factory=list)
    shared: List[TileBuffer] = field(default_factory=list)
    fragment: List[TileBuffer] = field(default_factory=list)

    def find(self, scope: str, shape: tuple = None, dtype: str = None) -> List[TileBuffer]:
        """Find compatible buffers — like TypedValuePool.getCandidatesFrom."""
        pool = getattr(self, scope)
        result = [
            b for b in pool
            if (shape is None or b.shape == shape)
            and (dtype is None or b.dtype == dtype)
        ]
        return result

    def find_or_none(self, scope: str, shape: tuple = None, dtype: str = None) -> Optional[TileBuffer]:
        candidates = self.find(scope, shape, dtype)
        return random.choice(candidates) if candidates else None

    def add(self, scope: str, buf: TileBuffer):
        getattr(self, scope).append(buf)


@dataclass
class KernelStep:
    """One concrete operation in the generated sequence."""
    op_kind: str           # "gemm", "copy_g2s", "copy_s2f", "copy_f2g", "scale", etc.
    inputs: List[TileBuffer]
    outputs: List[TileBuffer]
    attrs: dict
    torch_ref_update: str  # Description of what changed (for documentation)


@dataclass
class DynamicSequence:
    """
    A dynamically-generated op sequence.
    Analogous to MLIRSmith's generated function body.
    """
    steps: List[KernelStep]
    pool: TileValuePool

    M: int = 128
    N: int = 128
    K: int = 128
    block_M: int = 64
    block_N: int = 64
    block_K: int = 32
    threads: int = 128
    loop_kind: str = "pipelined"
    num_stages: int = 2
    dtype: str = "float16"
    name: str = "kernel_0"

    @property
    def acc_dtype(self) -> str:
        return "float32" if self.dtype == "float16" else self.dtype

    @property
    def extra_inputs(self) -> List[TileBuffer]:
        """Global input buffers beyond A and B — needed by ELEMWISE_BINARY ops."""
        return self.pool.global_in[2:]

    @property
    def output_buffer(self) -> Optional[TileBuffer]:
        """The final fragment buffer to write back."""
        frags = self.pool.fragment
        return frags[-1] if frags else None

    @property
    def output_shape(self) -> tuple:
        ob = self.output_buffer
        if ob is None:
            return (self.M, self.N)
        return ob.shape

    @property
    def has_terminal(self) -> bool:
        return any(s.op_kind in ("softmax", "reduce_sum", "reduce_max", "reduce_min")
                   for s in self.steps)

    @property
    def final_torch_ref(self) -> str:
        """The torch expression for the final output buffer."""
        ob = self.output_buffer
        if ob is None:
            return "torch.zeros(M, N, device='cuda')"
        return ob.torch_ref

    @property
    def params_dict(self) -> dict:
        return {
            "M": self.M, "N": self.N, "K": self.K,
            "block_M": self.block_M, "block_N": self.block_N, "block_K": self.block_K,
            "threads": self.threads, "num_stages": self.num_stages,
            "sequence": [s.op_kind for s in self.steps],
        }


# ─────────────────────────────────────────────────────────────────────────────
# Op Generators — one per op kind
# ─────────────────────────────────────────────────────────────────────────────

class OpGenBase:
    kind: str = ""
    base_weight: float = 1.0

    def is_applicable(self, pool: TileValuePool, params: dict) -> bool:
        raise NotImplementedError

    def apply(self, pool: TileValuePool, params: dict, counter: dict) -> Optional[KernelStep]:
        raise NotImplementedError

    def _next_name(self, counter: dict, prefix: str) -> str:
        counter[prefix] = counter.get(prefix, 0) + 1
        return f"{prefix}_{counter[prefix]}"


class GemmOpGen(OpGenBase):
    """GEMM: alloc shared A/B, accumulate into C_local fragment."""
    kind = "gemm"
    base_weight = 20.0

    def is_applicable(self, pool: TileValuePool, params: dict) -> bool:
        # Need at least A and B global inputs
        return len(pool.global_in) >= 2

    def apply(self, pool: TileValuePool, params: dict, counter: dict) -> Optional[KernelStep]:
        A = pool.global_in[0]
        B = pool.global_in[1]

        n = counter.get("gemm", 0) + 1
        counter["gemm"] = n

        block_M = params["block_M"]
        block_N = params["block_N"]
        block_K = params["block_K"]
        num_stages = params.get("num_stages", 2)
        loop_kind = params.get("loop_kind", "pipelined")
        acc_dtype = "float32" if params.get("dtype", "float16") == "float16" else params.get("dtype", "float32")

        a_shared = f"A_shared_{n}"
        b_shared = f"B_shared_{n}"
        c_local = f"C_local_{n}"

        torch_ref = f"({A.torch_ref}) @ ({B.torch_ref})"

        out_buf = TileBuffer(
            name=c_local,
            shape=(block_M, block_N),
            dtype=acc_dtype,
            scope="fragment",
            torch_ref=torch_ref,
        )
        pool.add("fragment", out_buf)

        return KernelStep(
            op_kind="gemm",
            inputs=[A, B],
            outputs=[out_buf],
            attrs={
                "a_shared": a_shared,
                "b_shared": b_shared,
                "c_local": c_local,
                "loop_kind": loop_kind,
                "num_stages": num_stages,
                "block_M": block_M,
                "block_N": block_N,
                "block_K": block_K,
            },
            torch_ref_update=f"{c_local} = A @ B",
        )


class CopyG2SOpGen(OpGenBase):
    """Copy from global (M, N)-shaped input to shared memory tile."""
    kind = "copy_g2s"
    base_weight = 10.0

    def _mn_globals(self, pool: TileValuePool, params: dict) -> list:
        """Return global_in buffers that have (M, N) shape — safe to tile with block_M x block_N."""
        M, N = params["M"], params["N"]
        return [g for g in pool.global_in if g.shape == (M, N)]

    def is_applicable(self, pool: TileValuePool, params: dict) -> bool:
        if not self._mn_globals(pool, params):
            return False
        # Check that adding a new shared buffer of shape (block_M, block_N)
        # won't push total shared memory over the hardware limit.
        # Existing shared usage: sum of all current shared buffers
        from src.constraints.constraints import TILELANG_MAX_SHARED
        bM = params["block_M"]
        bN = params["block_N"]
        dtype = params.get("dtype", "float16")
        bytes_per_elem = 2 if dtype == "float16" else 4
        new_buf_bytes = bM * bN * bytes_per_elem
        existing = sum(
            b.shape[0] * b.shape[1] * (2 if b.dtype == "float16" else 4)
            for b in pool.shared
        )
        return (existing + new_buf_bytes) <= TILELANG_MAX_SHARED

    def apply(self, pool: TileValuePool, params: dict, counter: dict) -> Optional[KernelStep]:
        candidates = self._mn_globals(pool, params)
        if not candidates:
            return None
        src = random.choice(candidates)
        block_M = params["block_M"]
        block_N = params["block_N"]

        self._next_name(counter, "g2s_shared")
        shared_name = f"X_shared_{counter['g2s_shared']}"

        out_buf = TileBuffer(
            name=shared_name,
            shape=(block_M, block_N),
            dtype=src.dtype,
            scope="shared",
            torch_ref=src.torch_ref,
        )
        pool.add("shared", out_buf)

        return KernelStep(
            op_kind="copy_g2s",
            inputs=[src],
            outputs=[out_buf],
            attrs={"shared_name": shared_name, "src_name": src.name},
            torch_ref_update=f"{shared_name} = {src.name}[tile]",
        )


class CopyS2FOpGen(OpGenBase):
    """Copy from shared memory to fragment (register tile)."""
    kind = "copy_s2f"
    base_weight = 8.0

    def is_applicable(self, pool: TileValuePool, params: dict) -> bool:
        return len(pool.shared) > 0

    def apply(self, pool: TileValuePool, params: dict, counter: dict) -> Optional[KernelStep]:
        src = random.choice(pool.shared)

        n = self._next_name(counter, "s2f_frag")
        frag_name = f"X_frag_{counter['s2f_frag']}"

        out_buf = TileBuffer(
            name=frag_name,
            shape=src.shape,
            dtype=src.dtype,
            scope="fragment",
            torch_ref=src.torch_ref,
        )
        pool.add("fragment", out_buf)

        return KernelStep(
            op_kind="copy_s2f",
            inputs=[src],
            outputs=[out_buf],
            attrs={"frag_name": frag_name, "src_name": src.name},
            torch_ref_update=f"{frag_name} = {src.name}",
        )


class CopyF2GOpGen(OpGenBase):
    """Write fragment back to global output."""
    kind = "copy_f2g"
    base_weight = 8.0

    def is_applicable(self, pool: TileValuePool, params: dict) -> bool:
        return len(pool.fragment) > 0

    def apply(self, pool: TileValuePool, params: dict, counter: dict) -> Optional[KernelStep]:
        frag = pool.fragment[-1]  # Use most recently produced fragment

        return KernelStep(
            op_kind="copy_f2g",
            inputs=[frag],
            outputs=[],
            attrs={"frag_name": frag.name},
            torch_ref_update=f"C = {frag.name}",
        )

    def _force_apply(self, frag: TileBuffer, pool: TileValuePool, params: dict, counter: dict) -> Optional[KernelStep]:
        """Force a write-back of a specific fragment."""
        return KernelStep(
            op_kind="copy_f2g",
            inputs=[frag],
            outputs=[],
            attrs={"frag_name": frag.name},
            torch_ref_update=f"C = {frag.name}",
        )


class ScaleOpGen(OpGenBase):
    """In-place scale (multiply by alpha) of a fragment."""
    kind = "scale"
    base_weight = 8.0

    def is_applicable(self, pool: TileValuePool, params: dict) -> bool:
        return len(pool.fragment) > 0

    def apply(self, pool: TileValuePool, params: dict, counter: dict) -> Optional[KernelStep]:
        frag = pool.fragment[-1]
        alpha = round(random.uniform(params.get("scale_alpha_min", 0.1), params.get("scale_alpha_max", 10.0)), 4)

        old_ref = frag.torch_ref
        frag.torch_ref = f"({old_ref}) * {alpha}"

        return KernelStep(
            op_kind="scale",
            inputs=[frag],
            outputs=[frag],
            attrs={"alpha": alpha, "frag_name": frag.name},
            torch_ref_update=f"{frag.name} *= {alpha}",
        )


class ExpOpGen(OpGenBase):
    """In-place exp of a fragment."""
    kind = "exp"
    base_weight = 6.0

    def is_applicable(self, pool: TileValuePool, params: dict) -> bool:
        return len(pool.fragment) > 0

    def apply(self, pool: TileValuePool, params: dict, counter: dict) -> Optional[KernelStep]:
        frag = pool.fragment[-1]

        old_ref = frag.torch_ref
        frag.torch_ref = f"torch.exp(({old_ref}).float())"

        return KernelStep(
            op_kind="exp",
            inputs=[frag],
            outputs=[frag],
            attrs={"frag_name": frag.name},
            torch_ref_update=f"{frag.name} = exp({frag.name})",
        )


class SqrtOpGen(OpGenBase):
    """In-place sqrt(abs()) of a fragment."""
    kind = "sqrt"
    base_weight = 6.0

    def is_applicable(self, pool: TileValuePool, params: dict) -> bool:
        return len(pool.fragment) > 0

    def apply(self, pool: TileValuePool, params: dict, counter: dict) -> Optional[KernelStep]:
        frag = pool.fragment[-1]

        old_ref = frag.torch_ref
        frag.torch_ref = f"torch.sqrt(({old_ref}).float().abs())"

        return KernelStep(
            op_kind="sqrt",
            inputs=[frag],
            outputs=[frag],
            attrs={"frag_name": frag.name},
            torch_ref_update=f"{frag.name} = sqrt(abs({frag.name}))",
        )


class ElemwiseAddOpGen(OpGenBase):
    """Element-wise add of two fragments, or fragment + new global input."""
    kind = "elemwise_add"
    base_weight = 8.0

    def is_applicable(self, pool: TileValuePool, params: dict) -> bool:
        if not pool.fragment:
            return False
        # Either 2 fragments, or we can add a new global input
        return len(pool.fragment) >= 2 or True  # always applicable if fragment exists (will add global)

    def apply(self, pool: TileValuePool, params: dict, counter: dict) -> Optional[KernelStep]:
        fragA = pool.fragment[-1]
        block_M = params["block_M"]
        block_N = params["block_N"]
        dtype = params.get("dtype", "float16")

        if len(pool.fragment) >= 2:
            fragB = pool.fragment[-2]
            old_ref_a = fragA.torch_ref
            old_ref_b = fragB.torch_ref
            fragA.torch_ref = f"({old_ref_a}).float() + ({old_ref_b}).float()"
            return KernelStep(
                op_kind="elemwise_add",
                inputs=[fragA, fragB],
                outputs=[fragA],
                attrs={"use_global": False, "frag_a_name": fragA.name, "frag_b_name": fragB.name},
                torch_ref_update=f"{fragA.name} += {fragB.name}",
            )
        else:
            d_idx = len(pool.global_in)
            d_name = f"D{d_idx}"
            d_frag_name = f"D{d_idx}_local"

            d_global = TileBuffer(
                name=d_name,
                shape=(params["M"], params["N"]),
                dtype=dtype,
                scope="global",
                torch_ref=f"{d_name}.float()",
            )
            pool.add("global_in", d_global)

            old_ref_a = fragA.torch_ref
            fragA.torch_ref = f"({old_ref_a}).float() + {d_name}.float()"

            return KernelStep(
                op_kind="elemwise_add",
                inputs=[fragA, d_global],
                outputs=[fragA],
                attrs={"use_global": True, "d_idx": d_idx, "d_name": d_name, "d_frag_name": d_frag_name, "frag_a_name": fragA.name},
                torch_ref_update=f"{fragA.name} += {d_name}",
            )


class ElemwiseMulOpGen(OpGenBase):
    """Element-wise multiply of two fragments, or fragment * new global input."""
    kind = "elemwise_mul"
    base_weight = 8.0

    def is_applicable(self, pool: TileValuePool, params: dict) -> bool:
        return len(pool.fragment) > 0

    def apply(self, pool: TileValuePool, params: dict, counter: dict) -> Optional[KernelStep]:
        fragA = pool.fragment[-1]
        block_M = params["block_M"]
        block_N = params["block_N"]
        dtype = params.get("dtype", "float16")

        if len(pool.fragment) >= 2:
            fragB = pool.fragment[-2]
            old_ref_a = fragA.torch_ref
            old_ref_b = fragB.torch_ref
            fragA.torch_ref = f"({old_ref_a}).float() * ({old_ref_b}).float()"
            return KernelStep(
                op_kind="elemwise_mul",
                inputs=[fragA, fragB],
                outputs=[fragA],
                attrs={"use_global": False, "frag_a_name": fragA.name, "frag_b_name": fragB.name},
                torch_ref_update=f"{fragA.name} *= {fragB.name}",
            )
        else:
            d_idx = len(pool.global_in)
            d_name = f"D{d_idx}"
            d_frag_name = f"D{d_idx}_local"

            d_global = TileBuffer(
                name=d_name,
                shape=(params["M"], params["N"]),
                dtype=dtype,
                scope="global",
                torch_ref=f"{d_name}.float()",
            )
            pool.add("global_in", d_global)

            old_ref_a = fragA.torch_ref
            fragA.torch_ref = f"({old_ref_a}).float() * {d_name}.float()"

            return KernelStep(
                op_kind="elemwise_mul",
                inputs=[fragA, d_global],
                outputs=[fragA],
                attrs={"use_global": True, "d_idx": d_idx, "d_name": d_name, "d_frag_name": d_frag_name, "frag_a_name": fragA.name},
                torch_ref_update=f"{fragA.name} *= {d_name}",
            )


class ElemwiseMaxOpGen(OpGenBase):
    """Element-wise maximum with a new global input."""
    kind = "elemwise_max"
    base_weight = 6.0

    def is_applicable(self, pool: TileValuePool, params: dict) -> bool:
        return len(pool.fragment) > 0

    def apply(self, pool: TileValuePool, params: dict, counter: dict) -> Optional[KernelStep]:
        fragA = pool.fragment[-1]
        block_M = params["block_M"]
        block_N = params["block_N"]
        dtype = params.get("dtype", "float16")

        d_idx = len(pool.global_in)
        d_name = f"D{d_idx}"
        d_frag_name = f"D{d_idx}_local"

        d_global = TileBuffer(
            name=d_name,
            shape=(params["M"], params["N"]),
            dtype=dtype,
            scope="global",
            torch_ref=f"{d_name}.float()",
        )
        pool.add("global_in", d_global)

        old_ref_a = fragA.torch_ref
        fragA.torch_ref = f"torch.maximum(({old_ref_a}).float(), {d_name}.float())"

        return KernelStep(
            op_kind="elemwise_max",
            inputs=[fragA, d_global],
            outputs=[fragA],
            attrs={"d_idx": d_idx, "d_name": d_name, "d_frag_name": d_frag_name, "frag_a_name": fragA.name},
            torch_ref_update=f"{fragA.name} = max({fragA.name}, {d_name})",
        )


class SoftmaxOpGen(OpGenBase):
    """In-place softmax on fragment rows (TERMINAL op — changes semantics)."""
    kind = "softmax"
    base_weight = 5.0

    def is_applicable(self, pool: TileValuePool, params: dict) -> bool:
        # Need a fragment, and N must equal block_N for correctness
        return len(pool.fragment) > 0 and params.get("N") == params.get("block_N")

    def apply(self, pool: TileValuePool, params: dict, counter: dict) -> Optional[KernelStep]:
        frag = pool.fragment[-1]

        old_ref = frag.torch_ref
        frag.torch_ref = f"torch.softmax(({old_ref}).float(), dim=-1)"

        return KernelStep(
            op_kind="softmax",
            inputs=[frag],
            outputs=[frag],
            attrs={"frag_name": frag.name},
            torch_ref_update=f"{frag.name} = softmax({frag.name})",
        )


class ReduceSumOpGen(OpGenBase):
    """Reduce fragment to 1D sum (TERMINAL op)."""
    kind = "reduce_sum"
    base_weight = 5.0

    def is_applicable(self, pool: TileValuePool, params: dict) -> bool:
        return len(pool.fragment) > 0 and params.get("N") == params.get("block_N")

    def apply(self, pool: TileValuePool, params: dict, counter: dict) -> Optional[KernelStep]:
        frag = pool.fragment[-1]
        acc_dtype = "float32" if params.get("dtype", "float16") == "float16" else params.get("dtype", "float32")

        n = self._next_name(counter, "reduce")
        reduce_name = f"C_reduce_{counter['reduce']}"

        old_ref = frag.torch_ref
        reduced_buf = TileBuffer(
            name=reduce_name,
            shape=(params["block_M"],),
            dtype=acc_dtype,
            scope="fragment",
            torch_ref=f"({old_ref}).float().sum(dim=-1)",
        )
        pool.fragment[-1] = reduced_buf

        return KernelStep(
            op_kind="reduce_sum",
            inputs=[frag],
            outputs=[reduced_buf],
            attrs={"frag_name": frag.name, "reduce_name": reduce_name},
            torch_ref_update=f"{reduce_name} = sum({frag.name})",
        )


class ReduceMaxOpGen(OpGenBase):
    """Reduce fragment to 1D max (TERMINAL op)."""
    kind = "reduce_max"
    base_weight = 5.0

    def is_applicable(self, pool: TileValuePool, params: dict) -> bool:
        return len(pool.fragment) > 0 and params.get("N") == params.get("block_N")

    def apply(self, pool: TileValuePool, params: dict, counter: dict) -> Optional[KernelStep]:
        frag = pool.fragment[-1]
        acc_dtype = "float32" if params.get("dtype", "float16") == "float16" else params.get("dtype", "float32")

        n = self._next_name(counter, "reduce")
        reduce_name = f"C_reduce_{counter['reduce']}"

        old_ref = frag.torch_ref
        reduced_buf = TileBuffer(
            name=reduce_name,
            shape=(params["block_M"],),
            dtype=acc_dtype,
            scope="fragment",
            torch_ref=f"({old_ref}).float().max(dim=-1).values",
        )
        pool.fragment[-1] = reduced_buf

        return KernelStep(
            op_kind="reduce_max",
            inputs=[frag],
            outputs=[reduced_buf],
            attrs={"frag_name": frag.name, "reduce_name": reduce_name},
            torch_ref_update=f"{reduce_name} = max({frag.name})",
        )


# ─────────────────────────────────────────────────────────────────────────────
# Nested Op Generators — analogous to MLIRSmith's scf.if / scf.for nesting
# ─────────────────────────────────────────────────────────────────────────────

class IfEpilogueOpGen(OpGenBase):
    """
    Conditional epilogue — analogous to MLIRSmith's scf.if.
    Splits per-element computation into two branches based on a condition:
      if frag[i,j] > threshold: branch_a (e.g. exp)
      else:                      branch_b (e.g. sqrt)
    Both branches operate on the same fragment in-place via T.Parallel.
    """
    kind = "if_epilogue"
    base_weight = 8.0

    def is_applicable(self, pool: TileValuePool, params: dict) -> bool:
        return len(pool.fragment) > 0

    def apply(self, pool: TileValuePool, params: dict, counter: dict) -> Optional[KernelStep]:
        import random
        frag = pool.fragment[-1]

        # Randomly pick two different branch ops to create variety
        branch_pairs = [
            ("exp",  "sqrt",  "exp(x)",       "sqrt(|x|)"),
            ("exp",  "neg",   "exp(x)",       "-x"),
            ("sqrt", "scale", "sqrt(|x|)",    "x*0.5"),
            ("abs",  "neg",   "|x|",          "-x"),
        ]
        a_op, b_op, a_desc, b_desc = random.choice(branch_pairs)

        def torch_expr(op, val):
            if op == "exp":  return f"torch.exp({val}.clamp(-80,80))"
            if op == "sqrt": return f"torch.sqrt({val}.abs())"
            if op == "neg":  return f"(-{val})"
            if op == "scale": return f"({val} * 0.5)"
            if op == "abs":  return f"{val}.abs()"
            return val

        threshold = round(random.uniform(-1.0, 1.0), 3)

        old_ref = frag.torch_ref
        frag.torch_ref = (
            f"torch.where(({old_ref}).float() > {threshold}, "
            f"{torch_expr(a_op, '(' + old_ref + ').float()')}, "
            f"{torch_expr(b_op, '(' + old_ref + ').float()')})"
        )

        return KernelStep(
            op_kind="if_epilogue",
            inputs=[frag],
            outputs=[frag],
            attrs={"threshold": threshold, "branch_a": a_op, "branch_b": b_op, "frag_name": frag.name},
            torch_ref_update=f"{frag.name} = if({a_op},{b_op})",
        )


class DoublePipelineOpGen(OpGenBase):
    """
    Double pipeline — analogous to MLIRSmith's nested affine.for.
    Runs a SECOND independent K-loop over a different region of A and B,
    accumulates into a second fragment, then adds the two results:
      C2 = A[:, K//2:] @ B[K//2:, :]
      C_final = C1 + C2
    This tests whether TileLang correctly handles multiple pipeline stages
    writing to the same output tile.
    """
    kind = "double_pipeline"
    base_weight = 5.0

    def is_applicable(self, pool: TileValuePool, params: dict) -> bool:
        # Need at least one fragment (from prior GEMM) and K large enough to split
        return len(pool.fragment) > 0 and params.get("K", 0) >= 32

    def apply(self, pool: TileValuePool, params: dict, counter: dict) -> Optional[KernelStep]:
        frag = pool.fragment[-1]
        n = self._next_name(counter, "dp")
        c2 = f"C2_{counter['dp']}"
        a2 = f"A2_{counter['dp']}"
        b2 = f"B2_{counter['dp']}"
        acc_dtype = params.get("acc_dtype", "float32")
        num_stages = params.get("num_stages", 2)
        loop_kind = params.get("loop_kind", "pipelined")

        old_ref = frag.torch_ref
        A_ref = pool.global_in[0].torch_ref
        B_ref = pool.global_in[1].torch_ref
        c2_ref = f"(({A_ref}) @ ({B_ref}))"
        frag.torch_ref = f"(({old_ref}).float() + {c2_ref}.float())"

        c2_buf = TileBuffer(name=c2, shape=frag.shape, dtype=acc_dtype, scope="fragment", torch_ref=c2_ref)
        pool.fragment.append(c2_buf)

        return KernelStep(
            op_kind="double_pipeline",
            inputs=[frag],
            outputs=[frag],
            attrs={
                "a2_name": a2, "b2_name": b2, "c2_name": c2,
                "frag_name": frag.name,
                "num_stages": num_stages,
                "loop_kind": loop_kind,
            },
            torch_ref_update=f"{frag.name} += second_gemm",
        )


class AccumulateReduceOpGen(OpGenBase):
    """
    Accumulate-then-reduce — analogous to MLIRSmith's nested scf.for + reduce.
    Applies an elementwise transform (scale/exp) to each fragment, then
    reduces row-wise, yielding a 1D result that feeds back into a new 2D fragment
    via broadcast. Tests reduce → broadcast data flow.

    Sequence:
      row_max = reduce_max(frag, axis=1)     # (block_M,)
      frag[i,j] = frag[i,j] - row_max[i]    # subtract row max (like online softmax)
      (result stays 2D but uses the 1D reduced intermediate)
    """
    kind = "accumulate_reduce"
    base_weight = 6.0

    def is_applicable(self, pool: TileValuePool, params: dict) -> bool:
        return len(pool.fragment) > 0

    def apply(self, pool: TileValuePool, params: dict, counter: dict) -> Optional[KernelStep]:
        frag = pool.fragment[-1]
        n = self._next_name(counter, "ar")
        row_stat = f"row_stat_{counter['ar']}"

        import random
        mode = random.choice(["subtract_max", "divide_sum"])

        old_ref = frag.torch_ref
        if mode == "subtract_max":
            frag.torch_ref = (
                f"(({old_ref}).float() - "
                f"({old_ref}).float().max(dim=-1, keepdim=True).values)"
            )
        else:  # divide_sum
            frag.torch_ref = (
                f"(({old_ref}).float() / "
                f"({old_ref}).float().sum(dim=-1, keepdim=True).clamp(min=1e-6))"
            )

        return KernelStep(
            op_kind="accumulate_reduce",
            inputs=[frag],
            outputs=[frag],
            attrs={"mode": mode, "row_stat_name": row_stat, "frag_name": frag.name},
            torch_ref_update=f"{frag.name} = {mode}",
        )


# ─────────────────────────────────────────────────────────────────────────────
# Main Generator
# ─────────────────────────────────────────────────────────────────────────────

class DynamicSequenceGenerator:
    """
    Analogous to MLIRSmith's RegionGen.apply() — generates op sequences
    dynamically based on what's in the pool, not from pre-planned templates.
    """

    ALL_OP_GENS = [
        GemmOpGen(),
        CopyG2SOpGen(),
        CopyS2FOpGen(),
        CopyF2GOpGen(),
        ScaleOpGen(),
        ExpOpGen(),
        SqrtOpGen(),
        ElemwiseAddOpGen(),
        ElemwiseMulOpGen(),
        ElemwiseMaxOpGen(),
        SoftmaxOpGen(),
        ReduceSumOpGen(),
        ReduceMaxOpGen(),
        # ── Nested ops (analogous to MLIRSmith's scf.if / scf.for nesting) ──
        IfEpilogueOpGen(),        # if/else per-element branching
        DoublePipelineOpGen(),    # two independent K-loops, results summed
        AccumulateReduceOpGen(),  # reduce → broadcast back (online softmax pattern)
    ]

    TERMINAL_OPS = {"softmax", "reduce_sum", "reduce_max", "reduce_min"}

    def __init__(self, config=None, backend: str = "tilelang"):
        self.config = config
        self.backend = backend

    def generate(self, params: dict, dtype: str, max_steps: int = 8) -> DynamicSequence:
        """
        Generate a DynamicSequence by dynamically picking ops based on pool state.
        Analogue of MLIRSmith's RegionGen.apply().
        """
        block_M = params["block_M"]
        block_N = params["block_N"]
        acc_dtype = "float32" if dtype == "float16" else dtype

        # Initialize pool with kernel inputs (A and B are always present)
        pool = TileValuePool(
            global_in=[
                TileBuffer("A", (params["M"], params["K"]), dtype, "global", "A.float()"),
                TileBuffer("B", (params["K"], params["N"]), dtype, "global", "B.float()"),
            ],
            global_out=[
                TileBuffer("C", (params["M"], params["N"]), dtype, "global", ""),
            ],
            shared=[],
            fragment=[],
        )

        # Working copy of params to allow N override for terminal ops
        working_params = dict(params)
        working_params["dtype"] = dtype
        working_params["acc_dtype"] = acc_dtype

        steps = []
        covered_ops = set()
        counter = {}
        has_terminal = False
        has_writeback = False

        # Always start with GEMM — it produces the main computation result
        # and populates pool.fragment with the accumulator.
        gemm_gen = GemmOpGen()
        gemm_step = gemm_gen.apply(pool, working_params, counter)
        if gemm_step:
            steps.append(gemm_step)
            covered_ops.add("gemm")

        # Dynamic loop: pick ops by pool state (max_steps - 1 remaining after GEMM)
        for step_i in range(max(0, max_steps - 1)):
            # Find applicable ops
            applicable = []
            for gen in self.ALL_OP_GENS:
                # After terminal, only allow copy_f2g
                if has_terminal and gen.kind not in {"copy_f2g"}:
                    continue
                # Don't add another terminal
                if gen.kind in self.TERMINAL_OPS and has_terminal:
                    continue
                # Skip GEMM as epilogue (it's only for the first step here)
                if gen.kind == "gemm":
                    continue
                if gen.is_applicable(pool, working_params):
                    applicable.append(gen)

            if not applicable:
                break

            # Compute weights with diversity boost (like MLIRSmith's selectOpGeneratorDiverse)
            weights = []
            for gen in applicable:
                w = gen.base_weight
                if gen.kind not in covered_ops:
                    w += params.get("diversity_boost", 50.0)  # diversity boost for uncovered ops
                weights.append(w)

            chosen_gen = random.choices(applicable, weights=weights, k=1)[0]
            step = chosen_gen.apply(pool, working_params, counter)

            if step:
                steps.append(step)
                covered_ops.add(chosen_gen.kind)
                if chosen_gen.kind in self.TERMINAL_OPS:
                    has_terminal = True
                    # Reduce/softmax terminal ops write directly to C — mark as done
                    has_writeback = True
                    break
                if chosen_gen.kind == "copy_f2g":
                    has_writeback = True
                    break  # After writing back, we're done

        # Ensure write-back if we have fragments
        if not has_writeback and pool.fragment:
            frag = pool.fragment[-1]
            # Only add copy_f2g if not a reduce (reduce already writes to C directly)
            if len(frag.shape) == 2 and frag.shape == (block_M, block_N):
                writeback_gen = CopyF2GOpGen()
                writeback_step = writeback_gen._force_apply(frag, pool, working_params, counter)
                if writeback_step:
                    steps.append(writeback_step)
                    has_writeback = True

        loop_kind_str = params.get("loop_kind", "pipelined")
        if isinstance(loop_kind_str, LoopKind):
            loop_kind_str = loop_kind_str.value

        return DynamicSequence(
            steps=steps,
            pool=pool,
            M=params["M"],
            N=params["N"],
            K=params["K"],
            block_M=block_M,
            block_N=block_N,
            block_K=params["block_K"],
            threads=params["threads"],
            loop_kind=loop_kind_str,
            num_stages=params.get("num_stages", 2),
            dtype=dtype,
            name=params.get("name", "kernel_0"),
        )
