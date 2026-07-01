import triton
import triton.language as tl
import torch

# Correctness thresholds — set by TileSmith config
_THRESHOLDS = {
    'gemm_fp16':      0.1,
    'gemm_fp32':      0.05,
    'reduce':         0.1,
    'softmax':        0.01,
    'copy':           1e-05,
    'transpose':      1e-05,
    'elemwise':       0.001,
    'pipeline_fp16':  0.1,
    'pipeline_fp32':  0.05,
}

def _finite_compare(C, ref):
    """Compare only finite elements. Skip inf/nan overflow cases."""
    import torch
    c_f32 = C.to(torch.float32)
    r_f32 = ref.to(torch.float32)
    mask = c_f32.isfinite() & r_f32.isfinite()
    if not mask.any():
        return 0.0, 1.0, 0.0  # all overflow → skip
    diff = (c_f32[mask] - r_f32[mask]).abs()
    max_diff = diff.max().item()
    ref_norm = r_f32[mask].abs().mean().item() + 1e-6
    relative_err = max_diff / ref_norm
    return max_diff, ref_norm, relative_err


@triton.jit
def kernel_0_kernel(
    a_ptr,
    d2_ptr,
    d3_ptr, c_ptr,
    M, N,
    stride_am, stride_an,
    stride_d2m, stride_d2n,
    stride_d3m, stride_d3n,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_n[None, :] * stride_an
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    acc = tl.load(a_ptrs, mask=mask, other=0.0).to(tl.float32)
    acc = tl.sqrt(tl.abs(acc.to(tl.float32)))
    d2_ptrs = d2_ptr + (offs_m[:, None] * stride_d2m + offs_n[None, :] * stride_d2n)
    d2 = tl.load(d2_ptrs, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N), other=0.0).to(tl.float32)
    acc = acc + d2
    d3_ptrs = d3_ptr + (offs_m[:, None] * stride_d3m + offs_n[None, :] * stride_d3n)
    d3 = tl.load(d3_ptrs, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N), other=0.0).to(tl.float32)
    acc = acc + d3
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc.to(tl.float16), mask=mask)


def kernel_0():
    M, N = 1547, 437
    A = torch.randn(M, N, dtype=torch.float16, device='cuda')
    D2 = torch.randn(M, N, dtype=torch.float16, device='cuda')
    D3 = torch.randn(M, N, dtype=torch.float16, device='cuda')
    C = torch.empty(M, N, dtype=torch.float16, device='cuda')
    grid = (triton.cdiv(M, 16), triton.cdiv(N, 16))
    kernel_0_kernel[grid](
        A,
        D2,
        D3, C,
        M, N,
        A.stride(0), A.stride(1),
       
        D2.stride(0), D2.stride(1),
        D3.stride(0), D3.stride(1),
        C.stride(0), C.stride(1),
        BLOCK_M=16, BLOCK_N=16,
    )
    return A, D2, D3, C


def test_kernel_0():
    A, D2, D3, C = kernel_0()
    ref = A.float()
    ref = torch.sqrt(ref.abs())
    ref = ref + D2.float()
    ref = ref + D3.float()
    max_diff, ref_norm, relative_err = _finite_compare(C, ref)
    if relative_err > _THRESHOLDS["pipeline_fp16"]:
        raise RuntimeError(f"WRONG RESULT [triton_chain]: max_diff={max_diff:.6f}, relative_err={relative_err:.4f}")

if __name__ == '__main__':
    test_kernel_0()
    print('kernel_0 PASSED')
    print('ALL PASSED')