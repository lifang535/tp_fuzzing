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
    a_ptr, b_ptr, M, N,
    stride_am, stride_an, stride_bm, stride_bn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    # Load A tile: shape (BLOCK_M, BLOCK_N)
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_n[None, :] * stride_an
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    a = tl.load(a_ptrs, mask=mask, other=0.0)
    # Transpose: write B[offs_n, offs_m]
    b_ptrs = b_ptr + offs_n[:, None] * stride_bm + offs_m[None, :] * stride_bn
    t_mask = (offs_n[:, None] < N) & (offs_m[None, :] < M)
    tl.store(b_ptrs, tl.trans(a), mask=t_mask)


def kernel_0():
    M, N = 2048, 1024
    A = torch.randn(M, N, dtype=torch.float32, device='cuda')
    B = torch.empty(N, M, dtype=torch.float32, device='cuda')
    grid = (triton.cdiv(M, 16), triton.cdiv(N, 256))
    kernel_0_kernel[grid](
        A, B, M, N,
        A.stride(0), A.stride(1), B.stride(0), B.stride(1),
        BLOCK_M=16, BLOCK_N=256,
    )
    return A, B


def test_kernel_0():
    A, B = kernel_0()
    ref = A.T.contiguous()
    max_diff = (B.to(torch.float32) - ref.to(torch.float32)).abs().max().item()
    if max_diff > _THRESHOLDS["transpose"]:
        raise RuntimeError(f"WRONG RESULT [transpose]: max_diff={max_diff:.6f}")

if __name__ == '__main__':
    test_kernel_0()
    print('kernel_0 PASSED')
    print('ALL PASSED')