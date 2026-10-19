"""TileLang#22: bf16 pow rejection, with float32 pow and bf16 exp controls."""
import json
import torch
import tilelang
import tilelang.language as T


@tilelang.jit
def make(dtype, operation):
    @T.prim_func
    def kernel(X: T.Tensor((128,), dtype), Y: T.Tensor((128,), dtype), O: T.Tensor((128,), dtype)):
        with T.Kernel(1, threads=128):
            for i in T.Parallel(128):
                if operation == 'pow':
                    O[i] = T.pow(X[i], Y[i])
                else:
                    O[i] = T.exp(X[i])
    return kernel


rows = []
for dtype, operation in [('bfloat16', 'pow'), ('float32', 'pow'), ('bfloat16', 'exp')]:
    try:
        kernel = make(dtype, operation)
        x = torch.ones(128, device='cuda', dtype=getattr(torch, dtype))
        y = torch.full_like(x, 2)
        out = torch.empty_like(x)
        kernel(x, y, out)
        rows.append({'dtype': dtype, 'operation': operation, 'status': 'passed', 'first': out[0].item()})
    except Exception as exc:
        rows.append({'dtype': dtype, 'operation': operation, 'status': 'rejected',
                     'exception': type(exc).__name__, 'diagnostic': str(exc)})
print(json.dumps({'id': 'TileLang#22', 'results': rows}, indent=2))
