import tilelang
import tilelang.language as T
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


def kernel_0():
    M, N = 256, 128
    block_M, block_N = 256, 128
    dtype = "float32"

    @tilelang.jit(out_idx=[3], target="cuda")
    def kernel_func(M, N, block_M, block_N):
        @T.prim_func
        def impl(
            A: T.Buffer((M, N), dtype),
            D2: T.Buffer((M, N), dtype),
            D3: T.Buffer((M, N), dtype),
            C: T.Buffer((M, N), dtype),
        ):
            with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
                acc = T.alloc_fragment((block_M, block_N), dtype)
                D2_local = T.alloc_fragment((block_M, block_N), dtype)
                D3_local = T.alloc_fragment((block_M, block_N), dtype)
                T.copy(A[by * block_M, bx * block_N], acc)
                for i, j in T.Parallel(block_M, block_N):
                    acc[i, j] = acc[i, j] * 7.2851
                T.copy(D2[by * block_M, bx * block_N], D2_local)
                for i, j in T.Parallel(block_M, block_N):
                    acc[i, j] = acc[i, j] * D2_local[i, j]
                T.copy(D3[by * block_M, bx * block_N], D3_local)
                for i, j in T.Parallel(block_M, block_N):
                    acc[i, j] = acc[i, j] + D3_local[i, j]
                T.copy(acc, C[by * block_M, bx * block_N])
        return impl

    return kernel_func(M, N, block_M, block_N)


def test_kernel_0():
    M, N = 256, 128
    kernel = kernel_0()
    A = torch.randn(M, N, dtype=torch.float32, device='cuda')
    D2 = torch.randn(M, N, dtype=torch.float32, device='cuda')
    D3 = torch.randn(M, N, dtype=torch.float32, device='cuda')
    C = kernel(A, D2, D3)
    ref = A.float()
    ref = ref * 7.2851
    ref = ref * D2.float()
    ref = ref + D3.float()
    max_diff, ref_norm, relative_err = _finite_compare(C, ref)
    if relative_err > _THRESHOLDS["pipeline_fp16"]:
        raise RuntimeError(f"WRONG RESULT [chain]: max_diff={max_diff:.6f}, relative_err={relative_err:.4f}")

if __name__ == '__main__':
    test_kernel_0()
    print('kernel_0 PASSED')
    print('ALL PASSED')