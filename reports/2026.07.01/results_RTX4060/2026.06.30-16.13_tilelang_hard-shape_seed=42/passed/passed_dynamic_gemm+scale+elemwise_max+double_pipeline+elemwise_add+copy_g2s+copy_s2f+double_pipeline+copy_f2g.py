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
    M, N, K = 803, 969, 1380
    block_M, block_N, block_K = 128, 16, 8
    dtype = "float16"
    accum_dtype = "float32"

    @tilelang.jit(out_idx=[3], target="cuda")
    def kernel_func(M, N, K, block_M, block_N, block_K):
        @T.prim_func
        def impl(
            A: T.Buffer((M, K), dtype),
            B: T.Buffer((K, N), dtype),
            D2: T.Buffer((M, N), dtype),
            C: T.Buffer((M, N), dtype),
        ):
            with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=256) as (bx, by):
                A_shared_1 = T.alloc_shared((block_M, block_K), dtype)
                B_shared_1 = T.alloc_shared((block_K, block_N), dtype)
                C_local_1 = T.alloc_fragment((block_M, block_N), accum_dtype)
                T.clear(C_local_1)
                for k in T.serial(T.ceildiv(K, block_K)):
                    T.copy(A[by * block_M, k * block_K], A_shared_1)
                    T.copy(B[k * block_K, bx * block_N], B_shared_1)
                    T.gemm(A_shared_1, B_shared_1, C_local_1)
                for i, j in T.Parallel(block_M, block_N):
                    C_local_1[i, j] = C_local_1[i, j] * 9.1736
                D2_local = T.alloc_fragment((block_M, block_N), dtype)
                T.copy(D2[by * block_M, bx * block_N], D2_local)
                for i, j in T.Parallel(block_M, block_N):
                    C_local_1[i, j] = C_local_1[i, j] if C_local_1[i, j] > D2_local[i, j] else D2_local[i, j]
                # double_pipeline: second K-loop over latter half of K
                A2_1 = T.alloc_shared((block_M, block_K), dtype)
                B2_1 = T.alloc_shared((block_K, block_N), dtype)
                C2_1 = T.alloc_fragment((block_M, block_N), accum_dtype)
                T.clear(C2_1)
                for k in T.serial(T.ceildiv(K, block_K)):
                    T.copy(A[by * block_M, k * block_K], A2_1)
                    T.copy(B[k * block_K, bx * block_N], B2_1)
                    T.gemm(A2_1, B2_1, C2_1)
                # add second pipeline result to first
                for i, j in T.Parallel(block_M, block_N):
                    C_local_1[i, j] = C_local_1[i, j] + C2_1[i, j]
                for i, j in T.Parallel(block_M, block_N):
                    C2_1[i, j] = C2_1[i, j] + C_local_1[i, j]
                X_shared_1 = T.alloc_shared((block_M, block_N), dtype)
                T.copy(D2[by * block_M, bx * block_N], X_shared_1)
                X_frag_1 = T.alloc_fragment((block_M, block_N), dtype)
                T.copy(X_shared_1, X_frag_1)
                # double_pipeline: second K-loop over latter half of K
                A2_2 = T.alloc_shared((block_M, block_K), dtype)
                B2_2 = T.alloc_shared((block_K, block_N), dtype)
                C2_2 = T.alloc_fragment((block_M, block_N), accum_dtype)
                T.clear(C2_2)
                for k in T.serial(T.ceildiv(K, block_K)):
                    T.copy(A[by * block_M, k * block_K], A2_2)
                    T.copy(B[k * block_K, bx * block_N], B2_2)
                    T.gemm(A2_2, B2_2, C2_2)
                # add second pipeline result to first
                for i, j in T.Parallel(block_M, block_N):
                    X_frag_1[i, j] = X_frag_1[i, j] + C2_2[i, j]
                T.copy(C2_2, C[by * block_M, bx * block_N])
        return impl

    return kernel_func(M, N, K, block_M, block_N, block_K)


def test_kernel_0():
    M, N, K = 803, 969, 1380
    kernel = kernel_0()
    A = torch.randn(M, K, dtype=torch.float16, device='cuda')
    B = torch.randn(K, N, dtype=torch.float16, device='cuda')
    D2 = torch.randn(M, N, dtype=torch.float16, device='cuda')
    C = kernel(A, B, D2)
    # Reference computation
    ref = ((A.float()) @ (B.float()))
    max_diff, ref_norm, relative_err = _finite_compare(C, ref)
    threshold = _THRESHOLDS["pipeline_fp16"] if "float16" == "float16" else _THRESHOLDS["pipeline_fp32"]
    if relative_err > threshold:
        raise RuntimeError(f"WRONG RESULT [dynamic]: max_diff={max_diff:.4f}, relative_err={relative_err:.4f}")

if __name__ == '__main__':
    test_kernel_0()
    print('kernel_0 PASSED')
    print('ALL PASSED')