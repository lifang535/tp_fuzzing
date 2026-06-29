"""
TileLang Pipeline Code Emitter — Translates TilePipeline IR to TileLang executable code.
"""

from src.ir import TilePipeline, PipelineStep
from src.ir.ir import ComputeKind, LoopKind, DataType, REDUCE_OPS
from src.config import DEFAULT_CONFIG


def _torch_dtype(dtype: DataType) -> str:
    return {"float16": "float16", "float32": "float32"}.get(dtype.value, "float16")


def _tl_dtype(dtype: DataType) -> str:
    return {"float16": "tl.float16", "float32": "tl.float32"}.get(dtype.value, "tl.float16")


class TileLangPipelineEmitter:
    """Emits TileLang code for a TilePipeline."""

    def __init__(self, config=None):
        self.config = config or DEFAULT_CONFIG

    def emit(self, pipeline: TilePipeline) -> str:
        from src.workflow.emitter import _threshold_header
        lines = [
            "import tilelang",
            "import tilelang.language as T",
            "import torch",
            "",
            _threshold_header(self.config),
            "",
        ]
        lines.append(self._emit_pipeline_kernel(pipeline))
        lines.append("")
        lines.append("if __name__ == '__main__':")
        lines.append(f"    test_{pipeline.name}()")
        lines.append(f"    print('{pipeline.name} PASSED')")
        lines.append("    print('ALL PASSED')")
        return "\n".join(lines)

    def _emit_pipeline_kernel(self, p: TilePipeline) -> str:
        if p.is_gemm_pipeline:
            return self._emit_gemm_pipeline(p)
        else:
            return self._emit_elemwise_chain(p)

    # ── GEMM-based pipeline ──────────────────────────────────────────────────

    def _emit_gemm_pipeline(self, p: TilePipeline) -> str:
        td = _torch_dtype(p.dtype)
        sp = "                "  # inside with T.Kernel(...)

        last_kind = p.last_kind
        has_terminal_reduce = last_kind in REDUCE_OPS
        has_terminal_softmax = last_kind == ComputeKind.SOFTMAX

        # Determine output buffer type and shape
        if has_terminal_reduce:
            out_shape = "(M,)"
            out_dtype_str = "accum_dtype"
        elif has_terminal_softmax:
            out_shape = "(M, N)"
            out_dtype_str = "dtype"
        else:
            out_shape = "(M, N)"
            out_dtype_str = "dtype"

        # Determine inputs needed beyond A, B (GEMM inputs)
        # Binary epilogue ops (ELEMWISE_ADD, ELEMWISE_MUL, WHERE) need an extra input
        extra_inputs = []
        for i, step in enumerate(p.steps[1:], start=1):
            if step.kind in (ComputeKind.ELEMWISE_ADD, ComputeKind.ELEMWISE_MUL,
                             ComputeKind.WHERE, ComputeKind.ELEMWISE_MAX,
                             ComputeKind.ELEMWISE_SUB):
                extra_inputs.append((i, step))

        # Build buffer declarations
        buf_decls = [
            f"            A: T.Buffer((M, K), dtype),",
            f"            B: T.Buffer((K, N), dtype),",
        ]
        # Add extra input buffers for binary ops
        for i, step in extra_inputs:
            buf_decls.append(f"            D{i}: T.Buffer((M, N), dtype),")
        # Output buffer
        buf_decls.append(f"            C: T.Buffer({out_shape}, {out_dtype_str}),")

        # Count outputs (last buffer is output)
        n_bufs = 2 + len(extra_inputs) + 1
        out_idx = n_bufs - 1  # 0-based

        # Func params
        func_params = "M, N, K, block_M, block_N, block_K"

        # Build func_call args
        if has_terminal_reduce:
            func_call = f"return kernel_func(M, N, K, block_M, block_N, block_K)"
        elif has_terminal_softmax:
            func_call = f"return kernel_func(M, N, K, block_M, block_N, block_K)"
        else:
            func_call = f"return kernel_func(M, N, K, block_M, block_N, block_K)"

        # Kernel body
        body_lines = self._gemm_pipeline_body(p, sp, extra_inputs, has_terminal_reduce, has_terminal_softmax)

        # Grid
        if has_terminal_softmax:
            # For softmax at end, N == block_N, so ceildiv(N,block_N)==1
            grid = "1, T.ceildiv(M, block_M)"
        elif has_terminal_reduce:
            grid = "T.ceildiv(N, block_N), T.ceildiv(M, block_M)"
        else:
            grid = "T.ceildiv(N, block_N), T.ceildiv(M, block_M)"

        # Test body
        test_lines = self._gemm_pipeline_test(p, td, extra_inputs, has_terminal_reduce, has_terminal_softmax)

        # Shape decls
        shape_decls = [
            f"    M, N, K = {p.M}, {p.N}, {p.K}",
            f"    block_M, block_N, block_K = {p.block_M}, {p.block_N}, {p.block_K}",
        ]
        extra_vars = [
            f'    dtype = "{p.dtype.value}"',
            f'    accum_dtype = "{p.acc_dtype}"',
        ]

        lines = []
        lines.append(f"def {p.name}():")
        lines.extend(shape_decls)
        lines.extend(extra_vars)
        lines.append("")
        lines.append(f"    @tilelang.jit(out_idx=[{out_idx}], target=\"cuda\")")
        lines.append(f"    def kernel_func({func_params}):")
        lines.append(f"        @T.prim_func")
        lines.append(f"        def impl(")
        lines.extend(buf_decls)
        lines.append(f"        ):")
        lines.append(f"            with T.Kernel({grid}, threads={p.threads}) as (bx, by):")
        lines.extend(body_lines)
        lines.append(f"        return impl")
        lines.append("")
        lines.append(f"    {func_call}")
        lines.append("")
        lines.append("")
        lines.append(f"def test_{p.name}():")
        lines.extend(test_lines)
        return "\n".join(lines)

    def _gemm_pipeline_body(self, p: TilePipeline, sp: str,
                             extra_inputs: list,
                             has_terminal_reduce: bool,
                             has_terminal_softmax: bool) -> list:
        loop_stmt = self._loop_stmt(p)
        lines = []

        # Allocate shared/fragment buffers
        lines.append(f"{sp}A_shared = T.alloc_shared((block_M, block_K), dtype)")
        lines.append(f"{sp}B_shared = T.alloc_shared((block_K, block_N), dtype)")
        lines.append(f"{sp}C_local = T.alloc_fragment((block_M, block_N), accum_dtype)")

        # Allocate extra fragment buffers for binary ops in epilogue
        for i, step in extra_inputs:
            lines.append(f"{sp}D{i}_local = T.alloc_fragment((block_M, block_N), dtype)")

        # GEMM accumulation
        lines.append(f"{sp}T.clear(C_local)")
        lines.append(f"{sp}{loop_stmt}")
        lines.append(f"{sp}    T.copy(A[by * block_M, k * block_K], A_shared)")
        lines.append(f"{sp}    T.copy(B[k * block_K, bx * block_N], B_shared)")
        lines.append(f"{sp}    T.gemm(A_shared, B_shared, C_local)")

        # Epilogue ops (all steps after GEMM, before terminal)
        steps_to_emit = p.steps[1:]
        if has_terminal_reduce or has_terminal_softmax:
            epilogue_steps = steps_to_emit[:-1]
            terminal_step = steps_to_emit[-1] if steps_to_emit else None
        else:
            epilogue_steps = steps_to_emit
            terminal_step = None

        for step in epilogue_steps:
            lines.extend(self._emit_epilogue_step(step, sp, extra_inputs, p.dtype))

        # Terminal step
        if terminal_step is not None:
            if terminal_step.kind == ComputeKind.SOFTMAX:
                lines.extend(self._emit_softmax_inplace(sp))
                lines.append(f"{sp}T.copy(C_local, C[by * block_M, 0])")
            elif terminal_step.kind in REDUCE_OPS:
                lines.extend(self._emit_reduce_terminal(terminal_step.kind, sp))
            else:
                # Should not happen, but fall back
                lines.append(f"{sp}T.copy(C_local, C[by * block_M, bx * block_N])")
        else:
            # Write back C_local (cast to output dtype if needed)
            lines.append(f"{sp}T.copy(C_local, C[by * block_M, bx * block_N])")

        return lines

    def _emit_epilogue_step(self, step: PipelineStep, sp: str,
                             extra_inputs: list, dtype: DataType) -> list:
        """Emit one epilogue op operating in-place on C_local."""
        lines = []
        kind = step.kind

        if kind == ComputeKind.SCALE:
            lines.append(f"{sp}for i, j in T.Parallel(block_M, block_N):")
            lines.append(f"{sp}    C_local[i, j] = C_local[i, j] * {step.alpha}")

        elif kind == ComputeKind.ELEMWISE_ADD:
            # Find which D buffer this is
            idx = next(i for i, s in extra_inputs if s is step)
            lines.append(f"{sp}T.copy(D{idx}[by * block_M, bx * block_N], D{idx}_local)")
            lines.append(f"{sp}for i, j in T.Parallel(block_M, block_N):")
            lines.append(f"{sp}    C_local[i, j] = C_local[i, j] + D{idx}_local[i, j]")

        elif kind == ComputeKind.ELEMWISE_MUL:
            idx = next(i for i, s in extra_inputs if s is step)
            lines.append(f"{sp}T.copy(D{idx}[by * block_M, bx * block_N], D{idx}_local)")
            lines.append(f"{sp}for i, j in T.Parallel(block_M, block_N):")
            lines.append(f"{sp}    C_local[i, j] = C_local[i, j] * D{idx}_local[i, j]")

        elif kind == ComputeKind.WHERE:
            idx = next(i for i, s in extra_inputs if s is step)
            lines.append(f"{sp}T.copy(D{idx}[by * block_M, bx * block_N], D{idx}_local)")
            lines.append(f"{sp}for i, j in T.Parallel(block_M, block_N):")
            lines.append(f"{sp}    C_local[i, j] = C_local[i, j] if C_local[i, j] > 0.0 else D{idx}_local[i, j]")

        elif kind == ComputeKind.UNARY_EXP:
            lines.append(f"{sp}for i, j in T.Parallel(block_M, block_N):")
            lines.append(f"{sp}    C_local[i, j] = T.exp(C_local[i, j])")

        elif kind == ComputeKind.UNARY_SQRT:
            lines.append(f"{sp}for i, j in T.Parallel(block_M, block_N):")
            lines.append(f"{sp}    C_local[i, j] = T.sqrt(T.abs(C_local[i, j]))")

        return lines

    def _emit_softmax_inplace(self, sp: str) -> list:
        """Emit softmax in-place on C_local."""
        return [
            f"{sp}max_local = T.alloc_fragment((block_M,), accum_dtype)",
            f"{sp}sum_local = T.alloc_fragment((block_M,), accum_dtype)",
            f"{sp}T.reduce_max(C_local, max_local, dim=1, clear=True)",
            f"{sp}for i, j in T.Parallel(block_M, block_N):",
            f"{sp}    C_local[i, j] = T.exp(C_local[i, j] - max_local[i])",
            f"{sp}T.reduce_sum(C_local, sum_local, dim=1, clear=True)",
            f"{sp}for i, j in T.Parallel(block_M, block_N):",
            f"{sp}    C_local[i, j] = C_local[i, j] / sum_local[i]",
        ]

    def _emit_reduce_terminal(self, kind: ComputeKind, sp: str) -> list:
        """Emit terminal reduce op on C_local, writing to B (1D output)."""
        reduce_fn = {
            ComputeKind.REDUCE_SUM: "T.reduce_sum",
            ComputeKind.REDUCE_MAX: "T.reduce_max",
            ComputeKind.REDUCE_MIN: "T.reduce_min",
        }[kind]
        return [
            f"{sp}C_reduce = T.alloc_fragment((block_M,), accum_dtype)",
            f"{sp}{reduce_fn}(C_local, C_reduce, dim=1, clear=True)",
            f"{sp}T.copy(C_reduce, C[by * block_M])",
        ]

    def _gemm_pipeline_test(self, p: TilePipeline, td: str,
                             extra_inputs: list,
                             has_terminal_reduce: bool,
                             has_terminal_softmax: bool) -> list:
        """Build the test function body for a GEMM pipeline."""
        lines = []
        lines.append(f"    M, N, K = {p.M}, {p.N}, {p.K}")

        # Create tensors
        lines.append(f"    kernel = {p.name}()")
        lines.append(f"    A = torch.randn(M, K, dtype=torch.{td}, device='cuda')")
        lines.append(f"    B = torch.randn(K, N, dtype=torch.{td}, device='cuda')")

        # Extra binary op inputs
        for i, step in extra_inputs:
            lines.append(f"    D{i} = torch.randn(M, N, dtype=torch.{td}, device='cuda')")

        # Call kernel
        extra_args = "".join(f", D{i}" for i, _ in extra_inputs)
        lines.append(f"    C = kernel(A, B{extra_args})")

        # Reference computation
        lines.append(f"    # Reference computation")
        lines.append(f"    ref = A.float() @ B.float()")

        # Apply each epilogue step to ref
        steps_to_emit = p.steps[1:]
        if has_terminal_reduce or has_terminal_softmax:
            epilogue_steps = steps_to_emit[:-1]
            terminal_step = steps_to_emit[-1] if steps_to_emit else None
        else:
            epilogue_steps = steps_to_emit
            terminal_step = None

        for step in epilogue_steps:
            kind = step.kind
            if kind == ComputeKind.SCALE:
                lines.append(f"    ref = ref * {step.alpha}")
            elif kind == ComputeKind.ELEMWISE_ADD:
                idx = next(i for i, s in extra_inputs if s is step)
                lines.append(f"    ref = ref + D{idx}.float()")
            elif kind == ComputeKind.ELEMWISE_MUL:
                idx = next(i for i, s in extra_inputs if s is step)
                lines.append(f"    ref = ref * D{idx}.float()")
            elif kind == ComputeKind.WHERE:
                idx = next(i for i, s in extra_inputs if s is step)
                lines.append(f"    ref = torch.where(ref > 0, ref, D{idx}.float())")
            elif kind == ComputeKind.UNARY_EXP:
                lines.append(f"    ref = torch.exp(ref)")
            elif kind == ComputeKind.UNARY_SQRT:
                lines.append(f"    ref = torch.sqrt(ref.abs())")

        if terminal_step is not None:
            if terminal_step.kind == ComputeKind.SOFTMAX:
                lines.append(f"    ref = torch.softmax(ref, dim=-1)")
            elif terminal_step.kind == ComputeKind.REDUCE_SUM:
                lines.append(f"    ref = ref.sum(dim=1)")
            elif terminal_step.kind == ComputeKind.REDUCE_MAX:
                lines.append(f"    ref = ref.max(dim=1).values")
            elif terminal_step.kind == ComputeKind.REDUCE_MIN:
                lines.append(f"    ref = ref.min(dim=1).values")

        # Comparison
        if has_terminal_reduce:
            lines.append(f"    max_diff, ref_norm, relative_err = _finite_compare(C, ref)")
            lines.append(f'    if relative_err > _THRESHOLDS["reduce"]:')
            lines.append(f'        raise RuntimeError(f"WRONG RESULT [pipeline_reduce]: max_diff={{max_diff:.6f}}, relative_err={{relative_err:.4f}}")')
        elif has_terminal_softmax:
            lines.append(f"    max_diff = (C.to(torch.float32) - ref.to(torch.float32)).abs().max().item()")
            lines.append(f'    if max_diff > _THRESHOLDS["softmax"]:')
            lines.append(f'        raise RuntimeError(f"WRONG RESULT [pipeline_softmax]: max_diff={{max_diff:.6f}}")')
        else:
            lines.append(f"    max_diff, ref_norm, relative_err = _finite_compare(C, ref)")
            lines.append(f'    threshold = _THRESHOLDS["pipeline_fp16"] if "{p.dtype.value}" == "float16" else _THRESHOLDS["pipeline_fp32"]')
            lines.append(f"    if relative_err > threshold:")
            lines.append(f'        raise RuntimeError(f"WRONG RESULT [pipeline]: max_diff={{max_diff:.4f}}, relative_err={{relative_err:.4f}}")')

        return lines

    # ── Elementwise chain pipeline ───────────────────────────────────────────

    def _emit_elemwise_chain(self, p: TilePipeline) -> str:
        td = _torch_dtype(p.dtype)
        sp = "                "  # inside with T.Kernel(...)

        last_kind = p.last_kind
        has_terminal_reduce = last_kind in REDUCE_OPS
        has_terminal_softmax = last_kind == ComputeKind.SOFTMAX

        # Collect binary ops that need extra inputs
        extra_inputs = []
        for i, step in enumerate(p.steps):
            if step.kind in (ComputeKind.ELEMWISE_ADD, ComputeKind.ELEMWISE_MUL,
                             ComputeKind.WHERE, ComputeKind.ELEMWISE_MAX,
                             ComputeKind.ELEMWISE_SUB):
                extra_inputs.append((i, step))

        # Output shape
        if has_terminal_reduce:
            out_shape = "(M,)"
            out_dtype_str = "dtype"
        else:
            out_shape = "(M, N)"
            out_dtype_str = "dtype"

        # Buffer declarations
        buf_decls = [f"            A: T.Buffer((M, N), dtype),"]
        for i, step in extra_inputs:
            buf_decls.append(f"            D{i}: T.Buffer((M, N), dtype),")
        buf_decls.append(f"            C: T.Buffer({out_shape}, {out_dtype_str}),")

        n_bufs = 1 + len(extra_inputs) + 1
        out_idx = n_bufs - 1

        # Grid
        if has_terminal_softmax:
            grid = "1, T.ceildiv(M, block_M)"
        else:
            grid = "T.ceildiv(N, block_N), T.ceildiv(M, block_M)"

        func_params = "M, N, block_M, block_N"
        func_call = f"return kernel_func(M, N, block_M, block_N)"

        # Kernel body
        body_lines = self._elemwise_chain_body(p, sp, extra_inputs, has_terminal_reduce, has_terminal_softmax)

        # Test body
        test_lines = self._elemwise_chain_test(p, td, extra_inputs, has_terminal_reduce, has_terminal_softmax)

        shape_decls = [
            f"    M, N = {p.M}, {p.N}",
            f"    block_M, block_N = {p.block_M}, {p.block_N}",
        ]
        extra_vars = [f'    dtype = "{p.dtype.value}"']

        lines = []
        lines.append(f"def {p.name}():")
        lines.extend(shape_decls)
        lines.extend(extra_vars)
        lines.append("")
        lines.append(f"    @tilelang.jit(out_idx=[{out_idx}], target=\"cuda\")")
        lines.append(f"    def kernel_func({func_params}):")
        lines.append(f"        @T.prim_func")
        lines.append(f"        def impl(")
        lines.extend(buf_decls)
        lines.append(f"        ):")
        lines.append(f"            with T.Kernel({grid}, threads={p.threads}) as (bx, by):")
        lines.extend(body_lines)
        lines.append(f"        return impl")
        lines.append("")
        lines.append(f"    {func_call}")
        lines.append("")
        lines.append("")
        lines.append(f"def test_{p.name}():")
        lines.extend(test_lines)
        return "\n".join(lines)

    def _elemwise_chain_body(self, p: TilePipeline, sp: str,
                              extra_inputs: list,
                              has_terminal_reduce: bool,
                              has_terminal_softmax: bool) -> list:
        """Emit body of elementwise chain kernel."""
        lines = []
        lines.append(f"{sp}acc = T.alloc_fragment((block_M, block_N), dtype)")

        # Allocate extra fragment buffers
        for i, step in extra_inputs:
            lines.append(f"{sp}D{i}_local = T.alloc_fragment((block_M, block_N), dtype)")

        # Load first input
        lines.append(f"{sp}T.copy(A[by * block_M, bx * block_N], acc)")

        # Apply each step
        steps_to_emit = p.steps
        if has_terminal_reduce or has_terminal_softmax:
            chain_steps = steps_to_emit[:-1]  # skip terminal
            terminal_step = steps_to_emit[-1]
        else:
            chain_steps = steps_to_emit
            terminal_step = None

        # First step is COPY — already done by loading into acc
        for step in chain_steps[1:]:
            kind = step.kind
            if kind == ComputeKind.SCALE:
                lines.append(f"{sp}for i, j in T.Parallel(block_M, block_N):")
                lines.append(f"{sp}    acc[i, j] = acc[i, j] * {step.alpha}")
            elif kind == ComputeKind.ELEMWISE_ADD:
                idx = next(i for i, s in extra_inputs if s is step)
                lines.append(f"{sp}T.copy(D{idx}[by * block_M, bx * block_N], D{idx}_local)")
                lines.append(f"{sp}for i, j in T.Parallel(block_M, block_N):")
                lines.append(f"{sp}    acc[i, j] = acc[i, j] + D{idx}_local[i, j]")
            elif kind == ComputeKind.ELEMWISE_MUL:
                idx = next(i for i, s in extra_inputs if s is step)
                lines.append(f"{sp}T.copy(D{idx}[by * block_M, bx * block_N], D{idx}_local)")
                lines.append(f"{sp}for i, j in T.Parallel(block_M, block_N):")
                lines.append(f"{sp}    acc[i, j] = acc[i, j] * D{idx}_local[i, j]")
            elif kind == ComputeKind.WHERE:
                idx = next(i for i, s in extra_inputs if s is step)
                lines.append(f"{sp}T.copy(D{idx}[by * block_M, bx * block_N], D{idx}_local)")
                lines.append(f"{sp}for i, j in T.Parallel(block_M, block_N):")
                lines.append(f"{sp}    acc[i, j] = acc[i, j] if acc[i, j] > 0.0 else D{idx}_local[i, j]")
            elif kind == ComputeKind.UNARY_EXP:
                lines.append(f"{sp}for i, j in T.Parallel(block_M, block_N):")
                lines.append(f"{sp}    acc[i, j] = T.exp(acc[i, j])")
            elif kind == ComputeKind.UNARY_SQRT:
                lines.append(f"{sp}for i, j in T.Parallel(block_M, block_N):")
                lines.append(f"{sp}    acc[i, j] = T.sqrt(T.abs(acc[i, j]))")

        # Terminal
        if terminal_step is not None:
            if terminal_step.kind == ComputeKind.SOFTMAX:
                lines.extend([
                    f"{sp}max_local = T.alloc_fragment((block_M,), dtype)",
                    f"{sp}sum_local = T.alloc_fragment((block_M,), dtype)",
                    f"{sp}T.reduce_max(acc, max_local, dim=1, clear=True)",
                    f"{sp}for i, j in T.Parallel(block_M, block_N):",
                    f"{sp}    acc[i, j] = T.exp(acc[i, j] - max_local[i])",
                    f"{sp}T.reduce_sum(acc, sum_local, dim=1, clear=True)",
                    f"{sp}for i, j in T.Parallel(block_M, block_N):",
                    f"{sp}    acc[i, j] = acc[i, j] / sum_local[i]",
                    f"{sp}T.copy(acc, C[by * block_M, 0])",
                ])
            elif terminal_step.kind in REDUCE_OPS:
                reduce_fn = {
                    ComputeKind.REDUCE_SUM: "T.reduce_sum",
                    ComputeKind.REDUCE_MAX: "T.reduce_max",
                    ComputeKind.REDUCE_MIN: "T.reduce_min",
                }[terminal_step.kind]
                lines.append(f"{sp}C_reduce = T.alloc_fragment((block_M,), dtype)")
                lines.append(f"{sp}{reduce_fn}(acc, C_reduce, dim=1, clear=True)")
                lines.append(f"{sp}T.copy(C_reduce, C[by * block_M])")
            else:
                lines.append(f"{sp}T.copy(acc, C[by * block_M, bx * block_N])")
        else:
            lines.append(f"{sp}T.copy(acc, C[by * block_M, bx * block_N])")

        return lines

    def _elemwise_chain_test(self, p: TilePipeline, td: str,
                              extra_inputs: list,
                              has_terminal_reduce: bool,
                              has_terminal_softmax: bool) -> list:
        lines = []
        lines.append(f"    M, N = {p.M}, {p.N}")
        lines.append(f"    kernel = {p.name}()")
        lines.append(f"    A = torch.randn(M, N, dtype=torch.{td}, device='cuda')")
        for i, step in extra_inputs:
            lines.append(f"    D{i} = torch.randn(M, N, dtype=torch.{td}, device='cuda')")

        extra_args = "".join(f", D{i}" for i, _ in extra_inputs)
        lines.append(f"    C = kernel(A{extra_args})")
        lines.append(f"    ref = A.float()")

        steps_to_emit = p.steps
        if has_terminal_reduce or has_terminal_softmax:
            chain_steps = steps_to_emit[:-1]
            terminal_step = steps_to_emit[-1]
        else:
            chain_steps = steps_to_emit
            terminal_step = None

        # First step is COPY — ref is already A.float()
        for step in chain_steps[1:]:
            kind = step.kind
            if kind == ComputeKind.SCALE:
                lines.append(f"    ref = ref * {step.alpha}")
            elif kind == ComputeKind.ELEMWISE_ADD:
                idx = next(i for i, s in extra_inputs if s is step)
                lines.append(f"    ref = ref + D{idx}.float()")
            elif kind == ComputeKind.ELEMWISE_MUL:
                idx = next(i for i, s in extra_inputs if s is step)
                lines.append(f"    ref = ref * D{idx}.float()")
            elif kind == ComputeKind.WHERE:
                idx = next(i for i, s in extra_inputs if s is step)
                lines.append(f"    ref = torch.where(ref > 0, ref, D{idx}.float())")
            elif kind == ComputeKind.UNARY_EXP:
                lines.append(f"    ref = torch.exp(ref)")
            elif kind == ComputeKind.UNARY_SQRT:
                lines.append(f"    ref = torch.sqrt(ref.abs())")

        if terminal_step is not None:
            if terminal_step.kind == ComputeKind.SOFTMAX:
                lines.append(f"    ref = torch.softmax(ref, dim=-1)")
            elif terminal_step.kind == ComputeKind.REDUCE_SUM:
                lines.append(f"    ref = ref.sum(dim=1)")
            elif terminal_step.kind == ComputeKind.REDUCE_MAX:
                lines.append(f"    ref = ref.max(dim=1).values")
            elif terminal_step.kind == ComputeKind.REDUCE_MIN:
                lines.append(f"    ref = ref.min(dim=1).values")

        if has_terminal_reduce:
            lines.append(f"    max_diff, ref_norm, relative_err = _finite_compare(C, ref)")
            lines.append(f'    if relative_err > _THRESHOLDS["reduce"]:')
            lines.append(f'        raise RuntimeError(f"WRONG RESULT [chain_reduce]: max_diff={{max_diff:.6f}}, relative_err={{relative_err:.4f}}")')
        elif has_terminal_softmax:
            lines.append(f"    max_diff = (C.to(torch.float32) - ref.to(torch.float32)).abs().max().item()")
            lines.append(f'    if max_diff > _THRESHOLDS["softmax"]:')
            lines.append(f'        raise RuntimeError(f"WRONG RESULT [chain_softmax]: max_diff={{max_diff:.6f}}")')
        else:
            lines.append(f"    max_diff, ref_norm, relative_err = _finite_compare(C, ref)")
            lines.append(f'    if relative_err > _THRESHOLDS["pipeline_fp16"]:')
            lines.append(f'        raise RuntimeError(f"WRONG RESULT [chain]: max_diff={{max_diff:.6f}}, relative_err={{relative_err:.4f}}")')
        return lines

    def _loop_stmt(self, p: TilePipeline) -> str:
        if p.loop_kind == LoopKind.PIPELINED:
            return f"for k in T.Pipelined(T.ceildiv(K, block_K), num_stages={p.num_stages}):"
        else:
            return f"for k in T.serial(T.ceildiv(K, block_K)):"
