"""
TileLang Dynamic Sequence Emitter — Translates DynamicSequence IR to TileLang executable code.
"""

from src.ir import DynamicSequence, TileBuffer, KernelStep
from src.config import DEFAULT_CONFIG


def _torch_dtype(dtype: str) -> str:
    return {"float16": "float16", "float32": "float32"}.get(dtype, "float16")


class TileLangDynamicEmitter:
    """Emits TileLang code for a DynamicSequence."""

    def __init__(self, config=None):
        self.config = config or DEFAULT_CONFIG

    def emit(self, seq: DynamicSequence) -> str:
        from src.workflow.emitter import _threshold_header
        lines = [
            "import tilelang",
            "import tilelang.language as T",
            "import torch",
            "",
            _threshold_header(self.config),
            "",
        ]
        lines.append(self._emit_kernel(seq))
        lines.append("")
        lines.append("if __name__ == '__main__':")
        lines.append(f"    test_{seq.name}()")
        lines.append(f"    print('{seq.name} PASSED')")
        lines.append("    print('ALL PASSED')")
        return "\n".join(lines)

    def _emit_kernel(self, seq: DynamicSequence) -> str:
        td = _torch_dtype(seq.dtype)

        # Determine output shape from last fragment
        ob = seq.output_buffer
        has_terminal_reduce = (
            ob is not None and len(ob.shape) == 1
        )
        has_terminal_softmax = any(s.op_kind == "softmax" for s in seq.steps)

        # Output buffer declaration
        if has_terminal_reduce:
            out_shape = "(M,)"
            out_dtype_str = "accum_dtype"
        else:
            out_shape = "(M, N)"
            out_dtype_str = "dtype"

        # Collect all global inputs: A, B, plus any extra D buffers
        all_global_in = seq.pool.global_in  # A, B, D2, D3, ...

        # Buffer declarations — A is always (M, K), B is always (K, N), Dx are (M, N)
        buf_decls = []
        for g in all_global_in:
            if g.name == "A":
                shape_str = "(M, K)"
            elif g.name == "B":
                shape_str = "(K, N)"
            else:
                shape_str = "(M, N)"
            buf_decls.append(f"            {g.name}: T.Buffer({shape_str}, dtype),")
        buf_decls.append(f"            C: T.Buffer({out_shape}, {out_dtype_str}),")

        n_bufs = len(all_global_in) + 1
        out_idx = n_bufs - 1

        # Collect all kernel body lines from steps
        body_lines = []
        for step in seq.steps:
            if step.tilelang_code:
                body_lines.extend(step.tilelang_code)

        # Grid
        if has_terminal_softmax:
            grid = "1, T.ceildiv(M, block_M)"
        elif has_terminal_reduce:
            grid = "T.ceildiv(N, block_N), T.ceildiv(M, block_M)"
        else:
            grid = "T.ceildiv(N, block_N), T.ceildiv(M, block_M)"

        # Shape declarations
        shape_decls = [
            f"    M, N, K = {seq.M}, {seq.N}, {seq.K}",
            f"    block_M, block_N, block_K = {seq.block_M}, {seq.block_N}, {seq.block_K}",
        ]
        extra_vars = [
            f'    dtype = "{seq.dtype}"',
            f'    accum_dtype = "{seq.acc_dtype}"',
        ]

        func_params = "M, N, K, block_M, block_N, block_K"

        # Test function
        test_lines = self._emit_test(seq, td, has_terminal_reduce, has_terminal_softmax)

        lines = []
        lines.append(f"def {seq.name}():")
        lines.extend(shape_decls)
        lines.extend(extra_vars)
        lines.append("")
        lines.append(f"    @tilelang.jit(out_idx=[{out_idx}], target=\"cuda\")")
        lines.append(f"    def kernel_func({func_params}):")
        lines.append(f"        @T.prim_func")
        lines.append(f"        def impl(")
        lines.extend(buf_decls)
        lines.append(f"        ):")
        lines.append(f"            with T.Kernel({grid}, threads={seq.threads}) as (bx, by):")
        lines.extend(body_lines)
        lines.append(f"        return impl")
        lines.append("")
        lines.append(f"    return kernel_func(M, N, K, block_M, block_N, block_K)")
        lines.append("")
        lines.append("")
        lines.append(f"def test_{seq.name}():")
        lines.extend(test_lines)
        return "\n".join(lines)

    def _emit_test(self, seq: DynamicSequence, td: str,
                   has_terminal_reduce: bool, has_terminal_softmax: bool) -> list:
        lines = []
        lines.append(f"    M, N, K = {seq.M}, {seq.N}, {seq.K}")
        lines.append(f"    kernel = {seq.name}()")

        # Create input tensors
        lines.append(f"    A = torch.randn(M, K, dtype=torch.{td}, device='cuda')")
        lines.append(f"    B = torch.randn(K, N, dtype=torch.{td}, device='cuda')")

        # Extra global inputs
        for g in seq.extra_inputs:
            lines.append(f"    {g.name} = torch.randn(M, N, dtype=torch.{td}, device='cuda')")

        # Call kernel
        extra_args = "".join(f", {g.name}" for g in seq.extra_inputs)
        lines.append(f"    C = kernel(A, B{extra_args})")

        # Reference computation
        lines.append(f"    # Reference computation")
        final_ref = seq.final_torch_ref
        lines.append(f"    ref = {final_ref}")

        # Comparison
        if has_terminal_reduce:
            lines.append(f"    max_diff, ref_norm, relative_err = _finite_compare(C, ref)")
            lines.append(f'    if relative_err > _THRESHOLDS["reduce"]:')
            lines.append(f'        raise RuntimeError(f"WRONG RESULT [dynamic_reduce]: max_diff={{max_diff:.6f}}, relative_err={{relative_err:.4f}}")')
        elif has_terminal_softmax:
            lines.append(f"    max_diff = (C.to(torch.float32) - ref.to(torch.float32)).abs().max().item()")
            lines.append(f'    if max_diff > _THRESHOLDS["softmax"]:')
            lines.append(f'        raise RuntimeError(f"WRONG RESULT [dynamic_softmax]: max_diff={{max_diff:.6f}}")')
        else:
            lines.append(f"    max_diff, ref_norm, relative_err = _finite_compare(C, ref)")
            lines.append(f'    threshold = _THRESHOLDS["pipeline_fp16"] if "{seq.dtype}" == "float16" else _THRESHOLDS["pipeline_fp32"]')
            lines.append(f"    if relative_err > threshold:")
            lines.append(f'        raise RuntimeError(f"WRONG RESULT [dynamic]: max_diff={{max_diff:.4f}}, relative_err={{relative_err:.4f}}")')

        return lines
