"""
Triton Pipeline Code Emitter — Translates TilePipeline IR to Triton executable code.
"""

from src.ir import TilePipeline, PipelineStep
from src.ir.ir import ComputeKind, LoopKind, DataType, REDUCE_OPS
from src.config import DEFAULT_CONFIG


def _torch_dtype(dtype: DataType) -> str:
    return {"float16": "float16", "float32": "float32"}.get(dtype.value, "float16")


def _tl_dtype(dtype: DataType) -> str:
    return {"float16": "tl.float16", "float32": "tl.float32"}.get(dtype.value, "tl.float16")


class TritonPipelineEmitter:
    """Emits Triton code for a TilePipeline."""

    def __init__(self, config=None):
        self.config = config or DEFAULT_CONFIG

    def emit(self, pipeline: TilePipeline) -> str:
        from src.workflow.emitter import _threshold_header
        lines = [
            "import triton",
            "import triton.language as tl",
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

    def _emit_gemm_pipeline(self, p: TilePipeline) -> str:
        td = _torch_dtype(p.dtype)
        tld = _tl_dtype(p.dtype)
        sp = "    "

        last_kind = p.last_kind
        has_terminal_reduce = last_kind in REDUCE_OPS
        has_terminal_softmax = last_kind == ComputeKind.SOFTMAX

        # Collect binary epilogue inputs
        extra_inputs = []
        for i, step in enumerate(p.steps[1:], start=1):
            if step.kind in (ComputeKind.ELEMWISE_ADD, ComputeKind.ELEMWISE_MUL,
                             ComputeKind.WHERE, ComputeKind.ELEMWISE_MAX,
                             ComputeKind.ELEMWISE_SUB):
                extra_inputs.append((i, step))

        # Kernel args
        extra_arg_decls = "".join(f"\n    d{i}_ptr," for i, _ in extra_inputs)
        kernel_args = (
            f"    a_ptr, b_ptr,{extra_arg_decls} c_ptr,\n"
            f"    M, N, K,\n"
            f"    stride_am, stride_ak, stride_bk, stride_bn,\n"
            + ("".join(f"    stride_d{i}m, stride_d{i}n,\n" for i, _ in extra_inputs))
            + f"    stride_cm, stride_cn,\n"
            f"    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,"
        )

        # Kernel body
        loop_stmt = self._loop_stmt(p)
        body_lines = [
            f"{sp}pid_m = tl.program_id(0)",
            f"{sp}pid_n = tl.program_id(1)",
            f"{sp}offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)",
            f"{sp}offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)",
            f"{sp}offs_k = tl.arange(0, BLOCK_K)",
            f"{sp}acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)",
            f"{sp}{loop_stmt}",
            f"{sp}    a_ptrs = a_ptr + (offs_m[:, None] * stride_am + (k + offs_k[None, :]) * stride_ak)",
            f"{sp}    a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & ((k + offs_k[None, :]) < K), other=0.0).to({tld})",
            f"{sp}    b_ptrs = b_ptr + ((k + offs_k[:, None]) * stride_bk + offs_n[None, :] * stride_bn)",
            f"{sp}    b = tl.load(b_ptrs, mask=((k + offs_k[:, None]) < K) & (offs_n[None, :] < N), other=0.0).to({tld})",
            f"{sp}    acc += tl.dot(a, b)",
        ]

        # Epilogue steps
        steps_to_emit = p.steps[1:]
        if has_terminal_reduce or has_terminal_softmax:
            epilogue_steps = steps_to_emit[:-1]
            terminal_step = steps_to_emit[-1] if steps_to_emit else None
        else:
            epilogue_steps = steps_to_emit
            terminal_step = None

        for step in epilogue_steps:
            body_lines.extend(self._triton_epilogue_step(step, sp, extra_inputs, tld))

        # Terminal or write-back
        if terminal_step is not None:
            if terminal_step.kind == ComputeKind.SOFTMAX:
                body_lines.append(f"{sp}acc = tl.softmax(acc, 1)")
                body_lines.append(f"{sp}c_ptrs = c_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)")
                body_lines.append(f"{sp}tl.store(c_ptrs, acc.to({tld}), mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))")
            elif terminal_step.kind == ComputeKind.REDUCE_SUM:
                body_lines.append(f"{sp}result = tl.sum(acc, axis=1)")
                body_lines.append(f"{sp}c_ptrs = c_ptr + offs_m * stride_cm")
                body_lines.append(f"{sp}tl.atomic_add(c_ptrs, result, mask=offs_m < M)")
            elif terminal_step.kind == ComputeKind.REDUCE_MAX:
                body_lines.append(f"{sp}result = tl.max(acc, axis=1)")
                body_lines.append(f"{sp}c_ptrs = c_ptr + offs_m * stride_cm")
                body_lines.append(f"{sp}tl.store(c_ptrs, result, mask=offs_m < M)")
            elif terminal_step.kind == ComputeKind.REDUCE_MIN:
                body_lines.append(f"{sp}result = tl.min(acc, axis=1)")
                body_lines.append(f"{sp}c_ptrs = c_ptr + offs_m * stride_cm")
                body_lines.append(f"{sp}tl.store(c_ptrs, result, mask=offs_m < M)")
        else:
            body_lines.append(f"{sp}c_ptrs = c_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)")
            body_lines.append(f"{sp}tl.store(c_ptrs, acc.to({tld}), mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))")

        body_str = "\n".join(body_lines)

        # Launch + test function
        launch_lines = self._triton_gemm_launch_test(
            p, td, tld, extra_inputs, has_terminal_reduce, has_terminal_softmax
        )
        launch_str = "\n".join(launch_lines)

        return (
            f"@triton.jit\n"
            f"def {p.name}_kernel(\n"
            f"{kernel_args}\n"
            f"):\n"
            f"{body_str}\n"
            f"\n\n"
            f"{launch_str}"
        )

    def _triton_epilogue_step(self, step: PipelineStep, sp: str,
                               extra_inputs: list, tld: str) -> list:
        lines = []
        kind = step.kind
        if kind == ComputeKind.SCALE:
            lines.append(f"{sp}acc = acc * {step.alpha}")
        elif kind == ComputeKind.ELEMWISE_ADD:
            idx = next(i for i, s in extra_inputs if s is step)
            lines.append(f"{sp}d{idx}_ptrs = d{idx}_ptr + (offs_m[:, None] * stride_d{idx}m + offs_n[None, :] * stride_d{idx}n)")
            lines.append(f"{sp}d{idx} = tl.load(d{idx}_ptrs, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N), other=0.0).to(tl.float32)")
            lines.append(f"{sp}acc = acc + d{idx}")
        elif kind == ComputeKind.ELEMWISE_MUL:
            idx = next(i for i, s in extra_inputs if s is step)
            lines.append(f"{sp}d{idx}_ptrs = d{idx}_ptr + (offs_m[:, None] * stride_d{idx}m + offs_n[None, :] * stride_d{idx}n)")
            lines.append(f"{sp}d{idx} = tl.load(d{idx}_ptrs, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N), other=0.0).to(tl.float32)")
            lines.append(f"{sp}acc = acc * d{idx}")
        elif kind == ComputeKind.WHERE:
            idx = next(i for i, s in extra_inputs if s is step)
            lines.append(f"{sp}d{idx}_ptrs = d{idx}_ptr + (offs_m[:, None] * stride_d{idx}m + offs_n[None, :] * stride_d{idx}n)")
            lines.append(f"{sp}d{idx} = tl.load(d{idx}_ptrs, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N), other=0.0).to(tl.float32)")
            lines.append(f"{sp}acc = tl.where(acc > 0, acc, d{idx})")
        elif kind == ComputeKind.UNARY_EXP:
            lines.append(f"{sp}acc = tl.exp(acc)")
        elif kind == ComputeKind.UNARY_SQRT:
            lines.append(f"{sp}acc = tl.sqrt(tl.abs(acc.to(tl.float32)))")
        return lines

    def _triton_gemm_launch_test(self, p: TilePipeline, td: str, tld: str,
                                  extra_inputs: list,
                                  has_terminal_reduce: bool,
                                  has_terminal_softmax: bool) -> list:
        lines = []
        lines.append(f"def {p.name}():")
        lines.append(f"    M, N, K = {p.M}, {p.N}, {p.K}")
        lines.append(f"    A = torch.randn(M, K, dtype=torch.{td}, device='cuda')")
        lines.append(f"    B = torch.randn(K, N, dtype=torch.{td}, device='cuda')")
        for i, step in extra_inputs:
            lines.append(f"    D{i} = torch.randn(M, N, dtype=torch.{td}, device='cuda')")
        if has_terminal_reduce:
            lines.append(f"    C = torch.zeros(M, dtype=torch.float32, device='cuda')")
        elif has_terminal_softmax:
            lines.append(f"    C = torch.empty(M, N, dtype=torch.{td}, device='cuda')")
        else:
            lines.append(f"    C = torch.empty(M, N, dtype=torch.{td}, device='cuda')")
        lines.append(f"    grid = (triton.cdiv(M, {p.block_M}), triton.cdiv(N, {p.block_N}))")
        # Build kernel call
        d_args = "".join(f"\n        D{i}," for i, _ in extra_inputs)
        d_strides = "".join(f"\n        D{i}.stride(0), D{i}.stride(1)," for i, _ in extra_inputs)
        if has_terminal_reduce:
            c_strides = "C.stride(0), 1,"
        else:
            c_strides = "C.stride(0), C.stride(1),"
        lines.append(f"    {p.name}_kernel[grid](")
        lines.append(f"        A, B,{d_args} C,")
        lines.append(f"        M, N, K,")
        lines.append(f"        A.stride(0), A.stride(1), B.stride(0), B.stride(1),")
        if d_strides.strip():
            lines.append(f"       {d_strides}")
        lines.append(f"        {c_strides}")
        lines.append(f"        BLOCK_M={p.block_M}, BLOCK_N={p.block_N}, BLOCK_K={p.block_K},")
        lines.append(f"    )")
        extra_return = "".join(f", D{i}" for i, _ in extra_inputs)
        lines.append(f"    return A, B{extra_return}, C")
        lines.append("")
        lines.append("")
        lines.append(f"def test_{p.name}():")
        extra_unpack = "".join(f", D{i}" for i, _ in extra_inputs)
        lines.append(f"    A, B{extra_unpack}, C = {p.name}()")
        lines.append(f"    ref = A.float() @ B.float()")

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

        if has_terminal_reduce:
            lines.append(f"    max_diff, ref_norm, relative_err = _finite_compare(C, ref)")
            lines.append(f'    if relative_err > _THRESHOLDS["reduce"]:')
            lines.append(f'        raise RuntimeError(f"WRONG RESULT [triton_pipeline_reduce]: max_diff={{max_diff:.6f}}, relative_err={{relative_err:.4f}}")')
        elif has_terminal_softmax:
            lines.append(f"    max_diff = (C.to(torch.float32) - ref.to(torch.float32)).abs().max().item()")
            lines.append(f'    if max_diff > _THRESHOLDS["softmax"]:')
            lines.append(f'        raise RuntimeError(f"WRONG RESULT [triton_pipeline_softmax]: max_diff={{max_diff:.6f}}")')
        else:
            lines.append(f"    max_diff, ref_norm, relative_err = _finite_compare(C, ref)")
            lines.append(f'    threshold = _THRESHOLDS["pipeline_fp16"] if "{p.dtype.value}" == "float16" else _THRESHOLDS["pipeline_fp32"]')
            lines.append(f"    if relative_err > threshold:")
            lines.append(f'        raise RuntimeError(f"WRONG RESULT [triton_pipeline]: max_diff={{max_diff:.4f}}, relative_err={{relative_err:.4f}}")')
        return lines

    def _emit_elemwise_chain(self, p: TilePipeline) -> str:
        td = _torch_dtype(p.dtype)
        tld = _tl_dtype(p.dtype)
        sp = "    "

        last_kind = p.last_kind
        has_terminal_reduce = last_kind in REDUCE_OPS
        has_terminal_softmax = last_kind == ComputeKind.SOFTMAX

        extra_inputs = []
        for i, step in enumerate(p.steps):
            if step.kind in (ComputeKind.ELEMWISE_ADD, ComputeKind.ELEMWISE_MUL,
                             ComputeKind.WHERE, ComputeKind.ELEMWISE_MAX,
                             ComputeKind.ELEMWISE_SUB):
                extra_inputs.append((i, step))

        extra_arg_decls = "".join(f"\n    d{i}_ptr," for i, _ in extra_inputs)
        extra_stride_decls = "".join(f"\n    stride_d{i}m, stride_d{i}n," for i, _ in extra_inputs)
        kernel_args = (
            f"    a_ptr,{extra_arg_decls} c_ptr,\n"
            f"    M, N,\n"
            f"    stride_am, stride_an,{extra_stride_decls}\n"
            f"    stride_cm, stride_cn,\n"
            f"    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,"
        )

        body_lines = [
            f"{sp}pid_m = tl.program_id(0)",
            f"{sp}pid_n = tl.program_id(1)",
            f"{sp}offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)",
            f"{sp}offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)",
            f"{sp}a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_n[None, :] * stride_an",
            f"{sp}mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)",
            f"{sp}acc = tl.load(a_ptrs, mask=mask, other=0.0).to(tl.float32)",
        ]

        steps_to_emit = p.steps
        if has_terminal_reduce or has_terminal_softmax:
            chain_steps = steps_to_emit[:-1]
            terminal_step = steps_to_emit[-1]
        else:
            chain_steps = steps_to_emit
            terminal_step = None

        # First step is COPY — already loaded
        for step in chain_steps[1:]:
            body_lines.extend(self._triton_epilogue_step(step, sp, extra_inputs, tld))

        if terminal_step is not None:
            if terminal_step.kind == ComputeKind.SOFTMAX:
                body_lines.append(f"{sp}acc = tl.softmax(acc, 1)")
                body_lines.append(f"{sp}c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn")
                body_lines.append(f"{sp}tl.store(c_ptrs, acc.to({tld}), mask=mask)")
            elif terminal_step.kind == ComputeKind.REDUCE_SUM:
                body_lines.append(f"{sp}result = tl.sum(acc, axis=1)")
                body_lines.append(f"{sp}c_ptrs = c_ptr + offs_m * stride_cm")
                body_lines.append(f"{sp}tl.atomic_add(c_ptrs, result, mask=offs_m < M)")
            elif terminal_step.kind == ComputeKind.REDUCE_MAX:
                body_lines.append(f"{sp}result = tl.max(acc, axis=1)")
                body_lines.append(f"{sp}c_ptrs = c_ptr + offs_m * stride_cm")
                body_lines.append(f"{sp}tl.store(c_ptrs, result, mask=offs_m < M)")
            elif terminal_step.kind == ComputeKind.REDUCE_MIN:
                body_lines.append(f"{sp}result = tl.min(acc, axis=1)")
                body_lines.append(f"{sp}c_ptrs = c_ptr + offs_m * stride_cm")
                body_lines.append(f"{sp}tl.store(c_ptrs, result, mask=offs_m < M)")
        else:
            body_lines.append(f"{sp}c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn")
            body_lines.append(f"{sp}tl.store(c_ptrs, acc.to({tld}), mask=mask)")

        body_str = "\n".join(body_lines)
        launch_lines = self._triton_elemwise_launch_test(
            p, td, tld, extra_inputs, has_terminal_reduce, has_terminal_softmax
        )
        launch_str = "\n".join(launch_lines)

        return (
            f"@triton.jit\n"
            f"def {p.name}_kernel(\n"
            f"{kernel_args}\n"
            f"):\n"
            f"{body_str}\n"
            f"\n\n"
            f"{launch_str}"
        )

    def _triton_elemwise_launch_test(self, p: TilePipeline, td: str, tld: str,
                                      extra_inputs: list,
                                      has_terminal_reduce: bool,
                                      has_terminal_softmax: bool) -> list:
        lines = []
        lines.append(f"def {p.name}():")
        lines.append(f"    M, N = {p.M}, {p.N}")
        lines.append(f"    A = torch.randn(M, N, dtype=torch.{td}, device='cuda')")
        for i, step in extra_inputs:
            lines.append(f"    D{i} = torch.randn(M, N, dtype=torch.{td}, device='cuda')")
        if has_terminal_reduce:
            lines.append(f"    C = torch.zeros(M, dtype=torch.float32, device='cuda')")
        else:
            lines.append(f"    C = torch.empty(M, N, dtype=torch.{td}, device='cuda')")
        lines.append(f"    grid = (triton.cdiv(M, {p.block_M}), triton.cdiv(N, {p.block_N}))")
        d_args = "".join(f"\n        D{i}," for i, _ in extra_inputs)
        d_strides = "".join(f"\n        D{i}.stride(0), D{i}.stride(1)," for i, _ in extra_inputs)
        if has_terminal_reduce:
            c_strides = "C.stride(0), 1,"
        else:
            c_strides = "C.stride(0), C.stride(1),"
        lines.append(f"    {p.name}_kernel[grid](")
        lines.append(f"        A,{d_args} C,")
        lines.append(f"        M, N,")
        lines.append(f"        A.stride(0), A.stride(1),")
        if d_strides.strip():
            lines.append(f"       {d_strides}")
        lines.append(f"        {c_strides}")
        lines.append(f"        BLOCK_M={p.block_M}, BLOCK_N={p.block_N},")
        lines.append(f"    )")
        extra_return = "".join(f", D{i}" for i, _ in extra_inputs)
        lines.append(f"    return A{extra_return}, C")
        lines.append("")
        lines.append("")
        lines.append(f"def test_{p.name}():")
        extra_unpack = "".join(f", D{i}" for i, _ in extra_inputs)
        lines.append(f"    A{extra_unpack}, C = {p.name}()")
        lines.append(f"    ref = A.float()")

        steps_to_emit = p.steps
        if has_terminal_reduce or has_terminal_softmax:
            chain_steps = steps_to_emit[:-1]
            terminal_step = steps_to_emit[-1]
        else:
            chain_steps = steps_to_emit
            terminal_step = None

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
            lines.append(f'        raise RuntimeError(f"WRONG RESULT [triton_chain_reduce]: max_diff={{max_diff:.6f}}, relative_err={{relative_err:.4f}}")')
        elif has_terminal_softmax:
            lines.append(f"    max_diff = (C.to(torch.float32) - ref.to(torch.float32)).abs().max().item()")
            lines.append(f'    if max_diff > _THRESHOLDS["softmax"]:')
            lines.append(f'        raise RuntimeError(f"WRONG RESULT [triton_chain_softmax]: max_diff={{max_diff:.6f}}")')
        else:
            lines.append(f"    max_diff, ref_norm, relative_err = _finite_compare(C, ref)")
            lines.append(f'    if relative_err > _THRESHOLDS["pipeline_fp16"]:')
            lines.append(f'        raise RuntimeError(f"WRONG RESULT [triton_chain]: max_diff={{max_diff:.6f}}, relative_err={{relative_err:.4f}}")')
        return lines

    def _loop_stmt(self, p: TilePipeline) -> str:
        if p.loop_kind == LoopKind.PIPELINED:
            return f"for k in tl.range(0, K, BLOCK_K, num_stages={p.num_stages}):"
        else:
            return f"for k in range(0, K, BLOCK_K):"
