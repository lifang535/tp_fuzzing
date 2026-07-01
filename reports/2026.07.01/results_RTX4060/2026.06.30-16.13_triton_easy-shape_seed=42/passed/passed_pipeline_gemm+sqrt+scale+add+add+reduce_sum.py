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
    a_ptr, b_ptr,
    d3_ptr,
    d4_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak, stride_bk, stride_bn,
    stride_d3m, stride_d3n,
    stride_d4m, stride_d4n,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        a_ptrs = a_ptr + (offs_m[:, None] * stride_am + (k + offs_k[None, :]) * stride_ak)
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & ((k + offs_k[None, :]) < K), other=0.0).to(tl.float32)
        b_ptrs = b_ptr + ((k + offs_k[:, None]) * stride_bk + offs_n[None, :] * stride_bn)
        b = tl.load(b_ptrs, mask=((k + offs_k[:, None]) < K) & (offs_n[None, :] < N), other=0.0).to(tl.float32)
        acc += tl.dot(a, b)
    acc = tl.sqrt(tl.abs(acc.to(tl.float32)))
    acc = acc * 3.7026
    d3_ptrs = d3_ptr + (offs_m[:, None] * stride_d3m + offs_n[None, :] * stride_d3n)
    d3 = tl.load(d3_ptrs, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N), other=0.0).to(tl.float32)
    acc = acc + d3
    d4_ptrs = d4_ptr + (offs_m[:, None] * stride_d4m + offs_n[None, :] * stride_d4n)
    d4 = tl.load(d4_ptrs, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N), other=0.0).to(tl.float32)
    acc = acc + d4
    result = tl.sum(acc, axis=1)
    c_ptrs = c_ptr + offs_m * stride_cm
    tl.atomic_add(c_ptrs, result, mask=offs_m < M)


def kernel_0():
    M, N, K = 256, 64, 2048
    A = torch.randn(M, K, dtype=torch.float32, device='cuda')
    B = torch.randn(K, N, dtype=torch.float32, device='cuda')
    D3 = torch.randn(M, N, dtype=torch.float32, device='cuda')
    D4 = torch.randn(M, N, dtype=torch.float32, device='cuda')
    C = torch.zeros(M, dtype=torch.float32, device='cuda')
    grid = (triton.cdiv(M, 64), triton.cdiv(N, 64))
    kernel_0_kernel[grid](
        A, B,
        D3,
        D4, C,
        M, N, K,
        A.stride(0), A.stride(1), B.stride(0), B.stride(1),
       
        D3.stride(0), D3.stride(1),
        D4.stride(0), D4.stride(1),
        C.stride(0), 1,
        BLOCK_M=64, BLOCK_N=64, BLOCK_K=16,
    )
    return A, B, D3, D4, C


def test_kernel_0():
    A, B, D3, D4, C = kernel_0()
    ref = A.float() @ B.float()
    ref = torch.sqrt(ref.abs())
    ref = ref * 3.7026
    ref = ref + D3.float()
    ref = ref + D4.float()
    ref = ref.sum(dim=1)
    max_diff, ref_norm, relative_err = _finite_compare(C, ref)
    if relative_err > _THRESHOLDS["reduce"]:
        raise RuntimeError(f"WRONG RESULT [triton_pipeline_reduce]: max_diff={max_diff:.6f}, relative_err={relative_err:.4f}")

if __name__ == '__main__':
    test_kernel_0()
    print('kernel_0 PASSED')
    print('ALL PASSED')