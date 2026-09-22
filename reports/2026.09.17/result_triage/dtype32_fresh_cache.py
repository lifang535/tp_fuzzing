import tilelang
import tilelang.language as T
import torch
dtype = "float32"
@tilelang.jit
def triage_cache_probe():
    @T.prim_func
    def impl(A: T.Buffer((32, 32), dtype), C: T.Buffer((32, 32), dtype)):
        with T.Kernel(1, threads=128):
            for i, j in T.Parallel(32, 32):
                C[i, j] = A[i, j]
    return impl
a = torch.ones((32, 32), device='cuda', dtype=getattr(torch, dtype))
c = torch.empty_like(a)
triage_cache_probe()(a, c)
torch.cuda.synchronize()
torch.testing.assert_close(a, c)
print('PASSED', dtype)
