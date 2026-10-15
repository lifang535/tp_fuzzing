"""TileLang 0.1.14 (#20): a vectorized shared -> shared index remap drops the
column offset of a modular index.

For b1[i, j] = b0[(i + 3) % n, (j + 2) % n] at n = 16 the generated CUDA emits
the row index correctly (((threadIdx.x >> 3) + 3) & 15) * 16 but the column as
(threadIdx.x & 7) * 2, i.e. plain j: the `+ 2` and its `% 16` wrap are gone.
The result silently equals the row-only remap. At n = 32 the same program
emits ((threadIdx.x & 15) * 2 + 2) & 31 and is correct. Deterministic, and an
explicit T.sync_threads() does not change it, so it is not a race.

Found by the layout slice (2026-10-10 15:22 check), reduced from
failed_slice_layout_be9e5d5401c907ee.
"""
import re

import torch
import tilelang
import tilelang.language as T


@tilelang.jit
def kern(n: int, threads: int):
    @T.prim_func
    def main(X: T.Tensor((n, n), "float32"), O: T.Tensor((n, n), "float32")):
        with T.Kernel(1, threads=threads) as bx:
            b0 = T.alloc_shared((n, n), "float32")
            b1 = T.alloc_shared((n, n), "float32")
            T.copy(X, b0)
            for i, j in T.Parallel(n, n):
                b1[i, j] = b0[(i + 3) % n, (j + 2) % n]
            T.copy(b1, O)
    return main


for n, threads in ((16, 128), (32, 128)):
    g = torch.Generator().manual_seed(0)
    x = torch.randint(-4, 5, (n, n), generator=g).float().cuda()
    out = torch.zeros(n, n, device='cuda')
    k = kern(n, threads)
    k(x, out)
    torch.cuda.synchronize()
    rows = (torch.arange(n) + 3) % n
    cols = (torch.arange(n) + 2) % n
    want = x.cpu()[rows][:, cols]
    dropped = x.cpu()[rows]
    body = k.get_kernel_source()
    body = body[body.index('extern "C" __global__ void __launch_bounds__'):]
    index = re.search(r'\(\(float\*\)b1\)[^;]*b0\)[^;]*', body)
    print(f'{n}x{n} threads={threads}: wrong {int((out.cpu() != want).sum())}/{n * n}, '
          f'equals column-shift-dropped: {torch.equal(out.cpu(), dropped)}')
    print('   read:', (index.group(0) if index else 'see kernel source')[:150])
