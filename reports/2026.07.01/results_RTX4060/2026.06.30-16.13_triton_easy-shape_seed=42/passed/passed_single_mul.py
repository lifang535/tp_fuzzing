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
    a_ptr, b_ptr, c_ptr, M, N, stride_m, stride_n, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    ptrs_a = a_ptr + offs_m[:, None] * stride_m + offs_n[None, :] * stride_n
    ptrs_b = b_ptr + offs_m[:, None] * stride_m + offs_n[None, :] * stride_n
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    a = tl.load(ptrs_a, mask=mask, other=0.0)
    b = tl.load(ptrs_b, mask=mask, other=0.0)
    c = a * b
    out_ptrs = c_ptr + offs_m[:, None] * stride_m + offs_n[None, :] * stride_n
    tl.store(out_ptrs, c, mask=mask)


def kernel_0():
    M, N = 128, 512
    A = torch.randn(M, N, dtype=torch.float16, device='cuda')
    B = torch.randn(M, N, dtype=torch.float16, device='cuda')
    C = torch.empty(M, N, dtype=torch.float16, device='cuda')
    grid = (triton.cdiv(M, 128), triton.cdiv(N, 16))
    kernel_0_kernel[grid](A, B, C, M, N, A.stride(0), A.stride(1), BLOCK_M=128, BLOCK_N=16)
    return A, B, C


def test_kernel_0():
    A, B, C = kernel_0()
    ref = A * B
    max_diff = (C.to(torch.float32) - ref.to(torch.float32)).abs().max().item()
    if max_diff > _THRESHOLDS["elemwise"]:
        raise RuntimeError(f"WRONG RESULT [elemwise_mul]: max_diff={max_diff:.6f}")

if __name__ == '__main__':
    test_kernel_0()
    print('kernel_0 PASSED')
    print('ALL PASSED')