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
            _threshold_header(self.config, dynamic=True),
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

        body_lines.extend(self._emit_steps(seq))

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

    def _emit_steps(self, seq):
        """Lower the actual named-buffer dataflow instead of one implicit accumulator."""
        lines = []

        def emit(code):
            lines.append('    ' + code)

        def dtype(buf):
            return f'tl.{buf.dtype}'

        def load_global(buf, target):
            prefix = buf.name.lower()
            strides = ('am', 'ak') if buf.name == 'A' else ('bk', 'bn') if buf.name == 'B' else (prefix + 'm', prefix + 'n')
            emit(f'{target}_ptrs = {prefix}_ptr + offs_m[:, None] * stride_{strides[0]} + offs_n[None, :] * stride_{strides[1]}')
            emit(f'{target} = tl.load({target}_ptrs, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N), other=0.0)')

        def branch(op, x):
            expressions = {'exp': f'tl.exp({x})', 'sqrt': f'tl.sqrt(tl.abs({x}))',
                           'neg': f'(-{x})', 'scale': f'({x} * 0.5)', 'abs': f'tl.abs({x})'}
            return expressions[op]

        for i, step in enumerate(seq.steps):
            kind, a = step.op_kind, step.attrs
            if kind == 'gemm':
                emit(f'{step.outputs[0].name} = acc')
                continue
            if kind == 'copy_g2s':
                load_global(step.inputs[0], step.outputs[0].name)
                continue
            if kind == 'copy_s2f':
                emit(f'{step.outputs[0].name} = {step.inputs[0].name}.to({dtype(step.outputs[0])})')
                continue
            x = step.inputs[0].name
            if kind == 'copy_f2g':
                emit('c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn')
                emit(f'tl.store(c_ptrs, {x}.to(tl.{seq.dtype}), mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))')
                continue
            out = step.outputs[0]
            xf = f'{x}.to(tl.float32)'
            if kind == 'scale':
                expr = f'{xf} * {a["alpha"]}'
            elif kind == 'exp':
                expr = f'tl.exp({xf})'
            elif kind == 'sqrt':
                expr = f'tl.sqrt(tl.abs({xf}))'
            elif kind in ('elemwise_add', 'elemwise_mul', 'elemwise_max'):
                rhs = step.inputs[1]
                y = rhs.name
                if rhs.scope == 'global':
                    y = f'input_{i}'
                    load_global(rhs, y)
                yf = f'{y}.to(tl.float32)'
                expr = f'{xf} + {yf}' if kind == 'elemwise_add' else f'{xf} * {yf}' if kind == 'elemwise_mul' else f'tl.maximum({xf}, {yf})'
            elif kind == 'if_epilogue':
                expr = f'tl.where({xf} > {a["threshold"]}, {branch(a["branch_a"], xf)}, {branch(a["branch_b"], xf)})'
            elif kind == 'double_pipeline':
                second = a['c2_name']
                emit(f'{second} = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)')
                loop = f'tl.range(0, K, BLOCK_K, num_stages={a["num_stages"]})' if a['loop_kind'] == 'pipelined' else 'range(0, K, BLOCK_K)'
                emit(f'for k2 in {loop}:')
                emit('    a2_ptrs = a_ptr + offs_m[:, None] * stride_am + (k2 + offs_k[None, :]) * stride_ak')
                emit('    b2_ptrs = b_ptr + (k2 + offs_k[:, None]) * stride_bk + offs_n[None, :] * stride_bn')
                emit('    a2 = tl.load(a2_ptrs, mask=(offs_m[:, None] < M) & (k2 + offs_k[None, :] < K), other=0.0)')
                emit('    b2 = tl.load(b2_ptrs, mask=(k2 + offs_k[:, None] < K) & (offs_n[None, :] < N), other=0.0)')
                emit(f'    {second} += tl.dot(a2, b2)')
                expr = f'{xf} + {second}'
            elif kind == 'accumulate_reduce':
                stat = a['row_stat_name']
                if a['mode'] == 'subtract_max':
                    emit(f'{stat} = tl.max({xf}, axis=1)[:, None]')
                    expr = f'{xf} - {stat}'
                elif a['mode'] == 'divide_sum':
                    emit(f'{stat} = tl.sum({xf}, axis=1)[:, None]')
                    expr = f'{xf} / ({stat} + 1e-6)'
                else:
                    raise ValueError(f'Unknown reduction mode: {a["mode"]}')
            elif kind == 'softmax':
                emit(f'soft_exp = tl.exp({xf} - tl.max({xf}, axis=1)[:, None]).to({dtype(out)})')
                expr = 'soft_exp.to(tl.float32) / tl.sum(soft_exp.to(tl.float32), axis=1)[:, None]'
            elif kind in ('reduce_sum', 'reduce_max'):
                fn = 'sum' if kind == 'reduce_sum' else 'max'
                emit(f'{out.name} = tl.{fn}({xf}, axis=1)')
                emit('c_ptrs = c_ptr + offs_m * stride_cm')
                emit(f'tl.store(c_ptrs, {out.name}, mask=offs_m < M)')
                continue
            else:
                raise ValueError(f'Unsupported dynamic op: {kind}')
            emit(f'{out.name} = ({expr}).to({dtype(out)})')
            if kind == 'softmax':
                emit('c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn')
                emit(f'tl.store(c_ptrs, {out.name}.to(tl.{seq.dtype}), mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))')
        return lines

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
        lines.append(f"        num_warps={seq.threads // 32}, BLOCK_M={seq.block_M}, BLOCK_N={seq.block_N}, BLOCK_K={seq.block_K},")
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
            lines.append(f"    max_diff = _max_diff(C, ref)")
            lines.append(f'    if max_diff > _THRESHOLDS["softmax"]:')
            lines.append(f'        raise RuntimeError(f"WRONG RESULT [triton_dynamic_softmax]: max_diff={{max_diff:.6f}}")')
        else:
            lines.append(f"    max_diff, ref_norm, relative_err = _finite_compare(C, ref)")
            lines.append(f'    threshold = _THRESHOLDS["pipeline_fp16"] if "{seq.dtype}" == "float16" else _THRESHOLDS["pipeline_fp32"]')
            lines.append(f"    if relative_err > threshold:")
            lines.append(f'        raise RuntimeError(f"WRONG RESULT [triton_dynamic]: max_diff={{max_diff:.4f}}, relative_err={{relative_err:.4f}}")')

        return lines
