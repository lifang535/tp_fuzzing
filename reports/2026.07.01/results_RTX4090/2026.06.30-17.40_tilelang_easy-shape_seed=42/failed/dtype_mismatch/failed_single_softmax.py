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
    M, N = 128, 256
    block_M, block_N = 128, 256
    dtype = "float32"

    @tilelang.jit(out_idx=[1], target="cuda")
    def kernel_func(M, N, block_M, block_N):
        @T.prim_func
        def impl(
            A: T.Buffer((M, N), dtype),
            B: T.Buffer((M, N), dtype),
        ):
            with T.Kernel(1, T.ceildiv(M, block_M), threads=128) as (bx, by):
                A_local = T.alloc_fragment((block_M, block_N), dtype)
                max_local = T.alloc_fragment((block_M,), dtype)
                B_local = T.alloc_fragment((block_M, block_N), dtype)
                sum_local = T.alloc_fragment((block_M,), dtype)
                T.copy(A[by * block_M, 0], A_local)
                T.reduce_max(A_local, max_local, dim=1, clear=True)
                for i, j in T.Parallel(block_M, block_N):
                    B_local[i, j] = T.exp(A_local[i, j] - max_local[i])
                T.reduce_sum(B_local, sum_local, dim=1, clear=True)
                for i, j in T.Parallel(block_M, block_N):
                    B_local[i, j] = B_local[i, j] / sum_local[i]
                T.copy(B_local, B[by * block_M, 0])
        return impl

    return kernel_func(M, N, block_M, block_N)


def test_kernel_0():
    M, N = 128, 256
    kernel = kernel_0()
    A = torch.randn(M, N, dtype=torch.float32, device='cuda')
    B = kernel(A)
    ref = torch.softmax(A.float(), dim=-1).to(torch.float32)
    max_diff = (B.to(torch.float32) - ref.to(torch.float32)).abs().max().item()
    if max_diff > _THRESHOLDS["softmax"]:
        raise RuntimeError(f"WRONG RESULT [softmax]: max_diff={max_diff:.6f}")

if __name__ == '__main__':
    test_kernel_0()
    print('kernel_0 PASSED')
    print('ALL PASSED')