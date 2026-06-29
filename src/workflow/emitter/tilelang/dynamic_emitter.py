"""
TileLang Dynamic Sequence Emitter — Translates DynamicSequence IR to TileLang executable code.

Code generation is driven entirely by step.op_kind + step.attrs + step.inputs/outputs,
keeping the IR layer (src/ir/dynamic_seq.py) backend-agnostic.
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

    # ─────────────────────────────────────────────────────────────────────────
    # Step-level code generation
    # ─────────────────────────────────────────────────────────────────────────

    def _emit_step_code(self, step: KernelStep, seq: DynamicSequence) -> list:
        """Dispatch to per-op emitter based on step.op_kind."""
        sp = "                "
        kind = step.op_kind
        if kind == "gemm":
            return self._emit_gemm(step, sp)
        elif kind == "copy_g2s":
            return self._emit_copy_g2s(step, sp)
        elif kind == "copy_s2f":
            return self._emit_copy_s2f(step, sp)
        elif kind == "copy_f2g":
            return self._emit_copy_f2g(step, sp)
        elif kind == "scale":
            return self._emit_scale(step, sp)
        elif kind == "exp":
            return self._emit_exp(step, sp)
        elif kind == "sqrt":
            return self._emit_sqrt(step, sp)
        elif kind == "elemwise_add":
            return self._emit_elemwise_add(step, sp)
        elif kind == "elemwise_mul":
            return self._emit_elemwise_mul(step, sp)
        elif kind == "elemwise_max":
            return self._emit_elemwise_max(step, sp)
        elif kind == "softmax":
            return self._emit_softmax(step, sp)
        elif kind == "reduce_sum":
            return self._emit_reduce_sum(step, sp)
        elif kind == "reduce_max":
            return self._emit_reduce_max(step, sp)
        elif kind == "if_epilogue":
            return self._emit_if_epilogue(step, sp)
        elif kind == "double_pipeline":
            return self._emit_double_pipeline(step, sp)
        elif kind == "accumulate_reduce":
            return self._emit_accumulate_reduce(step, sp)
        else:
            return [f"{sp}# unknown op: {kind}"]

    def _emit_gemm(self, step: KernelStep, sp: str) -> list:
        a = step.attrs
        a_shared = a["a_shared"]
        b_shared = a["b_shared"]
        c_local = a["c_local"]
        loop_kind = a["loop_kind"]
        num_stages = a["num_stages"]

        lines = [
            f"{sp}{a_shared} = T.alloc_shared((block_M, block_K), dtype)",
            f"{sp}{b_shared} = T.alloc_shared((block_K, block_N), dtype)",
            f"{sp}{c_local} = T.alloc_fragment((block_M, block_N), accum_dtype)",
            f"{sp}T.clear({c_local})",
        ]
        if loop_kind == "pipelined":
            lines.append(f"{sp}for k in T.Pipelined(T.ceildiv(K, block_K), num_stages={num_stages}):")
        else:
            lines.append(f"{sp}for k in T.serial(T.ceildiv(K, block_K)):")
        lines += [
            f"{sp}    T.copy(A[by * block_M, k * block_K], {a_shared})",
            f"{sp}    T.copy(B[k * block_K, bx * block_N], {b_shared})",
            f"{sp}    T.gemm({a_shared}, {b_shared}, {c_local})",
        ]
        return lines

    def _emit_copy_g2s(self, step: KernelStep, sp: str) -> list:
        a = step.attrs
        shared_name = a["shared_name"]
        src_name = a["src_name"]
        return [
            f"{sp}{shared_name} = T.alloc_shared((block_M, block_N), dtype)",
            f"{sp}T.copy({src_name}[by * block_M, bx * block_N], {shared_name})",
        ]

    def _emit_copy_s2f(self, step: KernelStep, sp: str) -> list:
        a = step.attrs
        frag_name = a["frag_name"]
        src_name = a["src_name"]
        return [
            f"{sp}{frag_name} = T.alloc_fragment((block_M, block_N), dtype)",
            f"{sp}T.copy({src_name}, {frag_name})",
        ]

    def _emit_copy_f2g(self, step: KernelStep, sp: str) -> list:
        a = step.attrs
        frag_name = a["frag_name"]
        return [
            f"{sp}T.copy({frag_name}, C[by * block_M, bx * block_N])",
        ]

    def _emit_scale(self, step: KernelStep, sp: str) -> list:
        frag = step.attrs["frag_name"]
        alpha = step.attrs["alpha"]
        return [
            f"{sp}for i, j in T.Parallel(block_M, block_N):",
            f"{sp}    {frag}[i, j] = {frag}[i, j] * {alpha}",
        ]

    def _emit_exp(self, step: KernelStep, sp: str) -> list:
        frag = step.attrs["frag_name"]
        return [
            f"{sp}for i, j in T.Parallel(block_M, block_N):",
            f"{sp}    {frag}[i, j] = T.exp(T.cast({frag}[i, j], T.float32))",
        ]

    def _emit_sqrt(self, step: KernelStep, sp: str) -> list:
        frag = step.attrs["frag_name"]
        return [
            f"{sp}for i, j in T.Parallel(block_M, block_N):",
            f"{sp}    {frag}[i, j] = T.sqrt(T.abs({frag}[i, j]))",
        ]

    def _emit_elemwise_add(self, step: KernelStep, sp: str) -> list:
        a = step.attrs
        if not a.get("use_global", False):
            frag_a = a["frag_a_name"]
            frag_b = a["frag_b_name"]
            return [
                f"{sp}for i, j in T.Parallel(block_M, block_N):",
                f"{sp}    {frag_a}[i, j] = {frag_a}[i, j] + {frag_b}[i, j]",
            ]
        else:
            frag_a = a["frag_a_name"]
            d_name = a["d_name"]
            d_frag_name = a["d_frag_name"]
            return [
                f"{sp}{d_frag_name} = T.alloc_fragment((block_M, block_N), dtype)",
                f"{sp}T.copy({d_name}[by * block_M, bx * block_N], {d_frag_name})",
                f"{sp}for i, j in T.Parallel(block_M, block_N):",
                f"{sp}    {frag_a}[i, j] = {frag_a}[i, j] + {d_frag_name}[i, j]",
            ]

    def _emit_elemwise_mul(self, step: KernelStep, sp: str) -> list:
        a = step.attrs
        if not a.get("use_global", False):
            frag_a = a["frag_a_name"]
            frag_b = a["frag_b_name"]
            return [
                f"{sp}for i, j in T.Parallel(block_M, block_N):",
                f"{sp}    {frag_a}[i, j] = {frag_a}[i, j] * {frag_b}[i, j]",
            ]
        else:
            frag_a = a["frag_a_name"]
            d_name = a["d_name"]
            d_frag_name = a["d_frag_name"]
            return [
                f"{sp}{d_frag_name} = T.alloc_fragment((block_M, block_N), dtype)",
                f"{sp}T.copy({d_name}[by * block_M, bx * block_N], {d_frag_name})",
                f"{sp}for i, j in T.Parallel(block_M, block_N):",
                f"{sp}    {frag_a}[i, j] = {frag_a}[i, j] * {d_frag_name}[i, j]",
            ]

    def _emit_elemwise_max(self, step: KernelStep, sp: str) -> list:
        a = step.attrs
        frag_a = a["frag_a_name"]
        d_name = a["d_name"]
        d_frag_name = a["d_frag_name"]
        return [
            f"{sp}{d_frag_name} = T.alloc_fragment((block_M, block_N), dtype)",
            f"{sp}T.copy({d_name}[by * block_M, bx * block_N], {d_frag_name})",
            f"{sp}for i, j in T.Parallel(block_M, block_N):",
            f"{sp}    {frag_a}[i, j] = {frag_a}[i, j] if {frag_a}[i, j] > {d_frag_name}[i, j] else {d_frag_name}[i, j]",
        ]

    def _emit_softmax(self, step: KernelStep, sp: str) -> list:
        frag = step.attrs["frag_name"]
        return [
            f"{sp}max_local = T.alloc_fragment((block_M,), accum_dtype)",
            f"{sp}sum_local = T.alloc_fragment((block_M,), accum_dtype)",
            f"{sp}T.reduce_max({frag}, max_local, dim=1, clear=True)",
            f"{sp}for i, j in T.Parallel(block_M, block_N):",
            f"{sp}    {frag}[i, j] = T.exp({frag}[i, j] - max_local[i])",
            f"{sp}T.reduce_sum({frag}, sum_local, dim=1, clear=True)",
            f"{sp}for i, j in T.Parallel(block_M, block_N):",
            f"{sp}    {frag}[i, j] = {frag}[i, j] / sum_local[i]",
        ]

    def _emit_reduce_sum(self, step: KernelStep, sp: str) -> list:
        frag = step.attrs["frag_name"]
        reduce_name = step.attrs["reduce_name"]
        return [
            f"{sp}{reduce_name} = T.alloc_fragment((block_M,), accum_dtype)",
            f"{sp}T.reduce_sum({frag}, {reduce_name}, dim=1, clear=True)",
            f"{sp}T.copy({reduce_name}, C[by * block_M])",
        ]

    def _emit_reduce_max(self, step: KernelStep, sp: str) -> list:
        frag = step.attrs["frag_name"]
        reduce_name = step.attrs["reduce_name"]
        return [
            f"{sp}{reduce_name} = T.alloc_fragment((block_M,), accum_dtype)",
            f"{sp}T.reduce_max({frag}, {reduce_name}, dim=1, clear=True)",
            f"{sp}T.copy({reduce_name}, C[by * block_M])",
        ]

    def _emit_if_epilogue(self, step: KernelStep, sp: str) -> list:
        a = step.attrs
        threshold = a["threshold"]
        branch_a = a["branch_a"]
        branch_b = a["branch_b"]
        v = a["frag_name"]

        def tl_expr(op, val):
            if op == "exp":  return f"T.exp(T.cast({val}, T.float32))"
            if op == "sqrt": return f"T.sqrt(T.abs({val}))"
            if op == "neg":  return f"-{val}"
            if op == "scale": return f"{val} * 0.5"
            if op == "abs":  return f"T.abs({val})"
            return val

        a_desc = {"exp": "exp(x)", "sqrt": "sqrt(|x|)", "neg": "-x", "scale": "x*0.5", "abs": "|x|"}.get(branch_a, branch_a)
        b_desc = {"exp": "exp(x)", "sqrt": "sqrt(|x|)", "neg": "-x", "scale": "x*0.5", "abs": "|x|"}.get(branch_b, branch_b)

        return [
            f"{sp}# if_epilogue: {a_desc} if x>{threshold} else {b_desc}",
            f"{sp}for i, j in T.Parallel(block_M, block_N):",
            f"{sp}    if {v}[i, j] > {threshold}:",
            f"{sp}        {v}[i, j] = {tl_expr(branch_a, v + '[i, j]')}",
            f"{sp}    else:",
            f"{sp}        {v}[i, j] = {tl_expr(branch_b, v + '[i, j]')}",
        ]

    def _emit_double_pipeline(self, step: KernelStep, sp: str) -> list:
        a = step.attrs
        a2 = a["a2_name"]
        b2 = a["b2_name"]
        c2 = a["c2_name"]
        frag = a["frag_name"]
        num_stages = a["num_stages"]
        loop_kind = a.get("loop_kind", "pipelined")

        lines = [
            f"{sp}# double_pipeline: second K-loop over latter half of K",
            f"{sp}{a2} = T.alloc_shared((block_M, block_K), dtype)",
            f"{sp}{b2} = T.alloc_shared((block_K, block_N), dtype)",
            f"{sp}{c2} = T.alloc_fragment((block_M, block_N), accum_dtype)",
            f"{sp}T.clear({c2})",
        ]
        if loop_kind == "pipelined":
            lines.append(f"{sp}for k in T.Pipelined(T.ceildiv(K, block_K), num_stages={num_stages}):")
        else:
            lines.append(f"{sp}for k in T.serial(T.ceildiv(K, block_K)):")
        lines += [
            f"{sp}    T.copy(A[by * block_M, k * block_K], {a2})",
            f"{sp}    T.copy(B[k * block_K, bx * block_N], {b2})",
            f"{sp}    T.gemm({a2}, {b2}, {c2})",
            f"{sp}# add second pipeline result to first",
            f"{sp}for i, j in T.Parallel(block_M, block_N):",
            f"{sp}    {frag}[i, j] = {frag}[i, j] + {c2}[i, j]",
        ]
        return lines

    def _emit_accumulate_reduce(self, step: KernelStep, sp: str) -> list:
        a = step.attrs
        mode = a["mode"]
        row_stat = a["row_stat_name"]
        frag = a["frag_name"]

        if mode == "subtract_max":
            return [
                f"{sp}# accumulate_reduce: subtract row max (online softmax pattern)",
                f"{sp}{row_stat} = T.alloc_fragment((block_M,), accum_dtype)",
                f"{sp}T.reduce_max({frag}, {row_stat}, dim=1, clear=True)",
                f"{sp}for i, j in T.Parallel(block_M, block_N):",
                f"{sp}    {frag}[i, j] = {frag}[i, j] - {row_stat}[i]",
            ]
        else:  # divide_sum
            return [
                f"{sp}# accumulate_reduce: divide by row sum (normalization pattern)",
                f"{sp}{row_stat} = T.alloc_fragment((block_M,), accum_dtype)",
                f"{sp}T.reduce_sum({frag}, {row_stat}, dim=1, clear=True)",
                f"{sp}for i, j in T.Parallel(block_M, block_N):",
                f"{sp}    {frag}[i, j] = {frag}[i, j] / ({row_stat}[i] + 1e-6)",
            ]

    # ─────────────────────────────────────────────────────────────────────────
    # Kernel-level emission
    # ─────────────────────────────────────────────────────────────────────────

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
            body_lines.extend(self._emit_step_code(step, seq))

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
