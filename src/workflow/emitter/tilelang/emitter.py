"""
TileLang Code Emitter — Translates TileProgram IR to TileLang executable code.
"""

from src.ir.ir import TileProgram, TileKernel, ComputeKind, LoopKind, DataType, REDUCE_OPS
from src.ops import OP_REGISTRY
from src.config import DEFAULT_CONFIG


def _torch_dtype(dtype: DataType) -> str:
    return {"float16": "float16", "float32": "float32"}.get(dtype.value, "float16")


class TileLangEmitter:
    """Emits TileLang code. Op-specific logic lives in ops.py."""

    def __init__(self, config=None):
        self.config = config or DEFAULT_CONFIG

    def emit(self, program: TileProgram) -> str:
        from src.workflow.emitter import _threshold_header
        lines = [
            "import tilelang",
            "import tilelang.language as T",
            "import torch",
            "",
            _threshold_header(self.config),
            "",
        ]
        for kernel in program.kernels:
            lines.append(self._emit_kernel(kernel))
            lines.append("")

        lines.append("if __name__ == '__main__':")
        for kernel in program.kernels:
            lines.append(f"    test_{kernel.name}()")
            lines.append(f"    print('{kernel.name} PASSED')")
        lines.append("    print('ALL PASSED')")
        return "\n".join(lines)

    def _emit_kernel(self, k: TileKernel) -> str:
        op_cls = OP_REGISTRY.get(k.compute_kind)
        if op_cls is None:
            raise ValueError(f"No op class registered for {k.compute_kind}")

        sp = "                "  # indentation inside with T.Kernel(...)
        body_lines = op_cls.tilelang_kernel_body(k, sp)
        test_lines = op_cls.tilelang_test_body(k)

        return self._wrap_tilelang(k, body_lines, test_lines)

    def _wrap_tilelang(self, k: TileKernel, body_lines: list, test_lines: list) -> str:
        """Wrap kernel body lines in the standard TileLang scaffolding."""
        kind = k.compute_kind
        torch_dtype = _torch_dtype(k.dtype)

        # ── kernel signature ──────────────────────────────────────────────
        if kind == ComputeKind.GEMM:
            jit_out_idx = "[2]"
            func_params = "M, N, K, block_M, block_N, block_K"
            buf_decls = [
                f"            A: T.Buffer((M, K), dtype),",
                f"            B: T.Buffer((K, N), dtype),",
                f"            C: T.Buffer((M, N), dtype),",
            ]
            grid = "T.ceildiv(N, block_N), T.ceildiv(M, block_M)"
            func_call = f"return kernel_func(M, N, K, block_M, block_N, block_K)"
            shape_decls = [
                f"    M, N, K = {k.M}, {k.N}, {k.K}",
                f"    block_M, block_N, block_K = {k.block_M}, {k.block_N}, {k.block_K}",
            ]
            extra_vars = [
                f'    dtype = "{k.dtype.value}"',
                f'    accum_dtype = "{k.acc_dtype}"',
            ]

        elif kind == ComputeKind.TRANSPOSE:
            jit_out_idx = "[1]"
            func_params = "M, N, block_M, block_N"
            buf_decls = [
                f"            A: T.Buffer((M, N), dtype),",
                f"            B: T.Buffer((N, M), dtype),",
            ]
            # grid: bx over M-axis tiles, by over N-axis tiles
            grid = "T.ceildiv(M, block_M), T.ceildiv(N, block_N)"
            func_call = f"return kernel_func(M, N, block_M, block_N)"
            shape_decls = [
                f"    M, N = {k.M}, {k.N}",
                f"    block_M, block_N = {k.block_M}, {k.block_N}",
            ]
            extra_vars = [f'    dtype = "{k.dtype.value}"']

        elif kind in REDUCE_OPS:
            jit_out_idx = "[1]"
            func_params = "M, N, block_M, block_N"
            buf_decls = [
                f"            A: T.Buffer((M, N), dtype),",
                f"            B: T.Buffer((M,), dtype),",
            ]
            grid = "T.ceildiv(N, block_N), T.ceildiv(M, block_M)"
            func_call = f"return kernel_func(M, N, block_M, block_N)"
            shape_decls = [
                f"    M, N = {k.M}, {k.N}",
                f"    block_M, block_N = {k.block_M}, {k.block_N}",
            ]
            extra_vars = [f'    dtype = "{k.dtype.value}"']

        elif kind == ComputeKind.SOFTMAX:
            # For softmax N must equal block_N
            jit_out_idx = "[1]"
            func_params = "M, N, block_M, block_N"
            buf_decls = [
                f"            A: T.Buffer((M, N), dtype),",
                f"            B: T.Buffer((M, N), dtype),",
            ]
            grid = "1, T.ceildiv(M, block_M)"
            func_call = f"return kernel_func(M, N, block_M, block_N)"
            shape_decls = [
                f"    M, N = {k.M}, {k.N}",
                f"    block_M, block_N = {k.block_M}, {k.block_N}",
            ]
            extra_vars = [f'    dtype = "{k.dtype.value}"']

        elif kind in (ComputeKind.ELEMWISE_ADD, ComputeKind.ELEMWISE_MUL,
                      ComputeKind.ELEMWISE_MAX, ComputeKind.ELEMWISE_SUB,
                      ComputeKind.WHERE):
            # Two-input, two-d output
            jit_out_idx = "[2]"
            func_params = "M, N, block_M, block_N"
            buf_decls = [
                f"            A: T.Buffer((M, N), dtype),",
                f"            B: T.Buffer((M, N), dtype),",
                f"            C: T.Buffer((M, N), dtype),",
            ]
            grid = "T.ceildiv(N, block_N), T.ceildiv(M, block_M)"
            func_call = f"return kernel_func(M, N, block_M, block_N)"
            shape_decls = [
                f"    M, N = {k.M}, {k.N}",
                f"    block_M, block_N = {k.block_M}, {k.block_N}",
            ]
            extra_vars = [f'    dtype = "{k.dtype.value}"']

        else:
            # Unary ops: COPY, SCALE, UNARY_EXP, UNARY_SQRT
            jit_out_idx = "[1]"
            func_params = "M, N, block_M, block_N"
            buf_decls = [
                f"            A: T.Buffer((M, N), dtype),",
                f"            B: T.Buffer((M, N), dtype),",
            ]
            grid = "T.ceildiv(N, block_N), T.ceildiv(M, block_M)"
            func_call = f"return kernel_func(M, N, block_M, block_N)"
            shape_decls = [
                f"    M, N = {k.M}, {k.N}",
                f"    block_M, block_N = {k.block_M}, {k.block_N}",
            ]
            extra_vars = [f'    dtype = "{k.dtype.value}"']

        # ── assemble lines ────────────────────────────────────────────────
        lines = []
        lines.append(f"def {k.name}():")
        lines.extend(shape_decls)
        lines.extend(extra_vars)
        lines.append(f"")
        lines.append(f"    @tilelang.jit(out_idx={jit_out_idx}, target=\"cuda\")")
        lines.append(f"    def kernel_func({func_params}):")
        lines.append(f"        @T.prim_func")
        lines.append(f"        def impl(")
        lines.extend(buf_decls)
        lines.append(f"        ):")
        lines.append(f"            with T.Kernel({grid}, threads={k.threads}) as (bx, by):")
        lines.extend(body_lines)
        lines.append(f"        return impl")
        lines.append(f"")
        lines.append(f"    {func_call}")
        lines.append(f"")
        lines.append(f"")
        lines.append(f"def test_{k.name}():")
        lines.extend(test_lines)
        return "\n".join(lines)

    def _loop_stmt(self, k: TileKernel) -> str:
        if k.loop_kind == LoopKind.PIPELINED:
            return f"for k in T.Pipelined(T.ceildiv(K, block_K), num_stages={k.num_stages}):"
        else:
            return f"for k in T.serial(T.ceildiv(K, block_K)):"
