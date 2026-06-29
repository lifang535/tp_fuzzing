"""
Triton Dynamic Sequence Emitter — Translates DynamicSequence IR to Triton executable code.
"""

from src.ir import DynamicSequence, TileBuffer, KernelStep
from src.config import DEFAULT_CONFIG


def _torch_dtype(dtype: str) -> str:
    return {"float16": "float16", "float32": "float32"}.get(dtype, "float16")


class TritonDynamicEmitter:
    """
    Emits Triton code for a DynamicSequence.

    Triton uses a different execution model — each program instance handles one tile.
    We generate a simple Triton kernel that:
      1. Computes GEMM if the sequence starts with gemm
      2. Applies epilogue ops
      3. Writes result back
    """

    def __init__(self, config=None):
        self.config = config or DEFAULT_CONFIG

    def emit(self, seq: DynamicSequence) -> str:
        from src.workflow.emitter import _threshold_header
        lines = [
            "import triton",
            "import triton.language as tl",
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
        tld = f"tl.{'float16' if seq.dtype == 'float16' else 'float32'}"
        sp = "    "

        has_gemm = any(s.op_kind == "gemm" for s in seq.steps)
        has_terminal_reduce = seq.output_buffer is not None and len(seq.output_buffer.shape) == 1
        has_terminal_softmax = any(s.op_kind == "softmax" for s in seq.steps)

        # Collect extra global inputs
        extra_inputs = seq.extra_inputs  # D2, D3, ...

        # Loop kind
        if seq.loop_kind == "pipelined":
            loop_stmt = f"for k in tl.range(0, K, BLOCK_K, num_stages={seq.num_stages}):"
        else:
            loop_stmt = f"for k in range(0, K, BLOCK_K):"

        # Build kernel args
        extra_arg_decls = "".join(f"\n    {g.name.lower()}_ptr," for g in extra_inputs)
        extra_stride_decls = "".join(
            f"\n    stride_{g.name.lower()}m, stride_{g.name.lower()}n,"
            for g in extra_inputs
        )

        if has_terminal_reduce:
            kernel_args = (
                f"    a_ptr, b_ptr,{extra_arg_decls} c_ptr,\n"
                f"    M, N, K,\n"
                f"    stride_am, stride_ak, stride_bk, stride_bn,{extra_stride_decls}\n"
                f"    stride_cm,\n"
                f"    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,"
            )
        else:
            kernel_args = (
                f"    a_ptr, b_ptr,{extra_arg_decls} c_ptr,\n"
                f"    M, N, K,\n"
                f"    stride_am, stride_ak, stride_bk, stride_bn,{extra_stride_decls}\n"
                f"    stride_cm, stride_cn,\n"
                f"    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,"
            )

        # Build body
        body_lines = [
            f"{sp}pid_m = tl.program_id(0)",
            f"{sp}pid_n = tl.program_id(1)",
            f"{sp}offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)",
            f"{sp}offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)",
        ]

        if has_gemm:
            body_lines += [
                f"{sp}offs_k = tl.arange(0, BLOCK_K)",
                f"{sp}acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)",
                f"{sp}{loop_stmt}",
                f"{sp}    a_ptrs = a_ptr + (offs_m[:, None] * stride_am + (k + offs_k[None, :]) * stride_ak)",
                f"{sp}    a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & ((k + offs_k[None, :]) < K), other=0.0).to({tld})",
                f"{sp}    b_ptrs = b_ptr + ((k + offs_k[:, None]) * stride_bk + offs_n[None, :] * stride_bn)",
                f"{sp}    b = tl.load(b_ptrs, mask=((k + offs_k[:, None]) < K) & (offs_n[None, :] < N), other=0.0).to({tld})",
                f"{sp}    acc += tl.dot(a, b)",
            ]
        else:
            # Load A as starting point
            body_lines += [
                f"{sp}a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_n[None, :] * stride_bn",
                f"{sp}mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)",
                f"{sp}acc = tl.load(a_ptrs, mask=mask, other=0.0).to(tl.float32)",
            ]

        # Apply epilogue ops from steps
        body_lines_done = False
        for step in seq.steps:
            if step.op_kind == "scale":
                alpha = step.attrs.get("alpha", 1.0)
                body_lines.append(f"{sp}acc = acc * {alpha}")
            elif step.op_kind == "exp":
                # Clamp before exp to prevent inf: float16→[-10,10], float32→[-80,80]
                clamp_max = 10.0 if seq.dtype == "float16" else 80.0
                body_lines.append(f"{sp}acc = tl.exp(acc)")
            elif step.op_kind == "sqrt":
                body_lines.append(f"{sp}acc = tl.sqrt(tl.abs(acc.to(tl.float32)))")
            elif step.op_kind == "elemwise_add":
                if step.attrs.get("use_global", False):
                    d_name = step.inputs[1].name if len(step.inputs) > 1 else "D2"
                    d_ptr = d_name.lower() + "_ptr"
                    d_sm = f"stride_{d_name.lower()}m"
                    d_sn = f"stride_{d_name.lower()}n"
                    body_lines.append(f"{sp}d_ptrs = {d_ptr} + (offs_m[:, None] * {d_sm} + offs_n[None, :] * {d_sn})")
                    body_lines.append(f"{sp}d = tl.load(d_ptrs, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N), other=0.0).to(tl.float32)")
                    body_lines.append(f"{sp}acc = acc + d")
                else:
                    # Two-fragment add: skip (already accumulated)
                    pass
            elif step.op_kind == "elemwise_mul":
                if step.attrs.get("use_global", False):
                    d_name = step.inputs[1].name if len(step.inputs) > 1 else "D2"
                    d_ptr = d_name.lower() + "_ptr"
                    d_sm = f"stride_{d_name.lower()}m"
                    d_sn = f"stride_{d_name.lower()}n"
                    body_lines.append(f"{sp}d_ptrs = {d_ptr} + (offs_m[:, None] * {d_sm} + offs_n[None, :] * {d_sn})")
                    body_lines.append(f"{sp}d = tl.load(d_ptrs, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N), other=0.0).to(tl.float32)")
                    body_lines.append(f"{sp}acc = acc * d")
            elif step.op_kind == "elemwise_max":
                d_name = step.inputs[1].name if len(step.inputs) > 1 else "D2"
                d_ptr = d_name.lower() + "_ptr"
                d_sm = f"stride_{d_name.lower()}m"
                d_sn = f"stride_{d_name.lower()}n"
                body_lines.append(f"{sp}d_ptrs = {d_ptr} + (offs_m[:, None] * {d_sm} + offs_n[None, :] * {d_sn})")
                body_lines.append(f"{sp}d = tl.load(d_ptrs, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N), other=0.0).to(tl.float32)")
                body_lines.append(f"{sp}acc = tl.maximum(acc, d)")
            elif step.op_kind == "softmax":
                body_lines.append(f"{sp}acc = tl.softmax(acc, 1)")
            elif step.op_kind == "reduce_sum":
                body_lines.append(f"{sp}result = tl.sum(acc, axis=1)")
                body_lines.append(f"{sp}c_ptrs = c_ptr + offs_m * stride_cm")
                body_lines.append(f"{sp}tl.atomic_add(c_ptrs, result, mask=offs_m < M)")
                # Done — no more write-back needed
                body_lines_done = True
                break
            elif step.op_kind == "reduce_max":
                body_lines.append(f"{sp}result = tl.max(acc, axis=1)")
                body_lines.append(f"{sp}c_ptrs = c_ptr + offs_m * stride_cm")
                body_lines.append(f"{sp}tl.store(c_ptrs, result, mask=offs_m < M)")
                body_lines_done = True
                break

        # Write-back (if not already handled by reduce)
        if not body_lines_done:
            if has_terminal_softmax:
                body_lines.append(f"{sp}c_ptrs = c_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)")
                body_lines.append(f"{sp}tl.store(c_ptrs, acc.to({tld}), mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))")
            else:
                body_lines.append(f"{sp}c_ptrs = c_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)")
                body_lines.append(f"{sp}tl.store(c_ptrs, acc.to({tld}), mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))")

        body_str = "\n".join(body_lines)

        # Launch + test
        launch_lines = self._emit_launch_test(seq, td, tld, has_terminal_reduce, has_terminal_softmax, extra_inputs)
        launch_str = "\n".join(launch_lines)

        return (
            f"@triton.jit\n"
            f"def {seq.name}_kernel(\n"
            f"{kernel_args}\n"
            f"):\n"
            f"{body_str}\n"
            f"\n\n"
            f"{launch_str}"
        )

    def _emit_launch_test(self, seq: DynamicSequence, td: str, tld: str,
                           has_terminal_reduce: bool, has_terminal_softmax: bool,
                           extra_inputs: list) -> list:
        lines = []
        lines.append(f"def {seq.name}():")
        lines.append(f"    M, N, K = {seq.M}, {seq.N}, {seq.K}")
        lines.append(f"    A = torch.randn(M, K, dtype=torch.{td}, device='cuda')")
        lines.append(f"    B = torch.randn(K, N, dtype=torch.{td}, device='cuda')")
        for g in extra_inputs:
            lines.append(f"    {g.name} = torch.randn(M, N, dtype=torch.{td}, device='cuda')")

        if has_terminal_reduce:
            lines.append(f"    C = torch.zeros(M, dtype=torch.float32, device='cuda')")
        else:
            lines.append(f"    C = torch.empty(M, N, dtype=torch.{td}, device='cuda')")

        lines.append(f"    grid = (triton.cdiv(M, {seq.block_M}), triton.cdiv(N, {seq.block_N}))")

        d_args = "".join(f"\n        {g.name}," for g in extra_inputs)
        d_strides = "".join(f"\n        {g.name}.stride(0), {g.name}.stride(1)," for g in extra_inputs)

        if has_terminal_reduce:
            c_strides = "C.stride(0),"   # reduce output is 1D — only one stride
        else:
            c_strides = "C.stride(0), C.stride(1),"

        lines.append(f"    {seq.name}_kernel[grid](")
        lines.append(f"        A, B,{d_args} C,")
        lines.append(f"        M, N, K,")
        lines.append(f"        A.stride(0), A.stride(1), B.stride(0), B.stride(1),")
        if d_strides.strip():
            lines.append(f"       {d_strides}")
        lines.append(f"        {c_strides}")
        lines.append(f"        BLOCK_M={seq.block_M}, BLOCK_N={seq.block_N}, BLOCK_K={seq.block_K},")
        lines.append(f"    )")
        extra_return = "".join(f", {g.name}" for g in extra_inputs)
        lines.append(f"    return A, B{extra_return}, C")
        lines.append("")
        lines.append("")
        lines.append(f"def test_{seq.name}():")
        extra_unpack = "".join(f", {g.name}" for g in extra_inputs)
        lines.append(f"    A, B{extra_unpack}, C = {seq.name}()")
        lines.append(f"    # Reference computation")
        lines.append(f"    ref = {seq.final_torch_ref}")

        if has_terminal_reduce:
            lines.append(f"    max_diff, ref_norm, relative_err = _finite_compare(C, ref)")
            lines.append(f'    if relative_err > _THRESHOLDS["reduce"]:')
            lines.append(f'        raise RuntimeError(f"WRONG RESULT [triton_dynamic_reduce]: max_diff={{max_diff:.6f}}, relative_err={{relative_err:.4f}}")')
        elif has_terminal_softmax:
            lines.append(f"    max_diff = (C.to(torch.float32) - ref.to(torch.float32)).abs().max().item()")
            lines.append(f'    if max_diff > _THRESHOLDS["softmax"]:')
            lines.append(f'        raise RuntimeError(f"WRONG RESULT [triton_dynamic_softmax]: max_diff={{max_diff:.6f}}")')
        else:
            lines.append(f"    max_diff, ref_norm, relative_err = _finite_compare(C, ref)")
            lines.append(f'    threshold = _THRESHOLDS["pipeline_fp16"] if "{seq.dtype}" == "float16" else _THRESHOLDS["pipeline_fp32"]')
            lines.append(f"    if relative_err > threshold:")
            lines.append(f'        raise RuntimeError(f"WRONG RESULT [triton_dynamic]: max_diff={{max_diff:.4f}}, relative_err={{relative_err:.4f}}")')

        return lines
