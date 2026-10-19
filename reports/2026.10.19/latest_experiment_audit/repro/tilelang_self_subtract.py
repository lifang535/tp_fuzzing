"""TileLang#24: x-x must preserve NaN for infinite/NaN inputs without fast math."""
import json
import math
import torch
import tilelang
import tilelang.language as T


@tilelang.jit
def make():
    @T.prim_func
    def kernel(X: T.Tensor((4,), 'float32'), O: T.Tensor((4,), 'float32')):
        with T.Kernel(1, threads=32):
            for i in T.Parallel(4):
                O[i] = X[i] - X[i]
    return kernel


kernel = make()
x = torch.tensor([float('inf'), -float('inf'), float('nan'), 1.5], device='cuda')
out = torch.empty_like(x)
kernel(x, out)
got = out.cpu().tolist()
print(json.dumps({'id': 'TileLang#24', 'inputs': ['+inf', '-inf', 'nan', 1.5],
                  'expected': ['nan', 'nan', 'nan', 0.0],
                  'actual': ['nan' if math.isnan(v) else v for v in got],
                  'mismatches': sum(not math.isnan(v) for v in got[:3]) + int(got[3] != 0),
                  'generated_cuda': kernel.get_kernel_source()}, indent=2))
