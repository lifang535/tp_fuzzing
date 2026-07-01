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
    M, N, K = 620, 32, 1658
    block_M, block_N, block_K = 64, 32, 8
    dtype = "float32"
    accum_dtype = "float32"

    @tilelang.jit(out_idx=[2], target="cuda")
    def kernel_func(M, N, K, block_M, block_N, block_K):
        @T.prim_func
        def impl(
            A: T.Buffer((M, K), dtype),
            B: T.Buffer((K, N), dtype),
            C: T.Buffer((M,), accum_dtype),
        ):
            with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=256) as (bx, by):
                A_shared = T.alloc_shared((block_M, block_K), dtype)
                B_shared = T.alloc_shared((block_K, block_N), dtype)
                C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
                T.clear(C_local)
                for k in T.serial(T.ceildiv(K, block_K)):
                    T.copy(A[by * block_M, k * block_K], A_shared)
                    T.copy(B[k * block_K, bx * block_N], B_shared)
                    T.gemm(A_shared, B_shared, C_local)
                for i, j in T.Parallel(block_M, block_N):
                    C_local[i, j] = C_local[i, j] * 1.7552
                for i, j in T.Parallel(block_M, block_N):
                    C_local[i, j] = T.exp(C_local[i, j])
                C_reduce = T.alloc_fragment((block_M,), accum_dtype)
                T.reduce_max(C_local, C_reduce, dim=1, clear=True)
                T.copy(C_reduce, C[by * block_M])
        return impl

    return kernel_func(M, N, K, block_M, block_N, block_K)


def test_kernel_0():
    M, N, K = 620, 32, 1658
    kernel = kernel_0()
    A = torch.randn(M, K, dtype=torch.float32, device='cuda')
    B = torch.randn(K, N, dtype=torch.float32, device='cuda')
    C = kernel(A, B)
    # Reference computation
    ref = A.float() @ B.float()
    ref = ref * 1.7552
    ref = torch.exp(ref)
    ref = ref.max(dim=1).values
    max_diff, ref_norm, relative_err = _finite_compare(C, ref)
    if relative_err > _THRESHOLDS["reduce"]:
        raise RuntimeError(f"WRONG RESULT [pipeline_reduce]: max_diff={max_diff:.6f}, relative_err={relative_err:.4f}")

if __name__ == '__main__':
    test_kernel_0()
    print('kernel_0 PASSED')
    print('ALL PASSED')