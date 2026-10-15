"""TileLang 0.1.14 (#20, second form): the dropped column offset of a modular
index remap does not need vectorization.

b2[i, j] = b1[(i + 2) % 32, (j + 1) % 64] from a (32, 64) float16 shared buffer
into a fragment at threads=32 compiles to the scalar read
    b2[i_1] = b1[(((i_1 >> 1) + 2) & 31) * 64 + (i_1 & 1) * 32 + threadIdx.x]
without the "+ 1" and "& 63" of the column, so the result is the row-only remap.
With a zero row offset (i + 0, j + 1) at threads=32 the column offset is kept,
and threads=64/128 are correct: the trigger is a row and a column offset
together, under a (loop variable, thread id) split of the flattened index.
The first form (tl_remap_column_shift_dropped.py) is the 16x16 float32
shared -> shared case with float2 accesses. Found when checking the #20 tag of
bucket wrong_result:85fe6b1652 (vGPU, seed 42) at the 2026-10-11 06:20 check.
"""
import importlib.util
import re
import torch

SRC = '''
import tilelang
import tilelang.language as T

@tilelang.jit
def kern(threads: int, r: int, c: int):
    @T.prim_func
    def main(X: T.Tensor((32, 64), "float16"), O: T.Tensor((32, 64), "float16")):
        with T.Kernel(1, threads=threads) as bx:
            b1 = T.alloc_shared((32, 64), "float16")
            T.copy(X, b1)
            b2 = T.alloc_fragment((32, 64), "float16")
            for i, j in T.Parallel(32, 64):
                b2[i, j] = b1[(i + r) % 32, (j + c) % 64]
            T.copy(b2, O)
    return main
'''
open('/tmp/remap_fragment_kern.py', 'w').write(SRC)
spec = importlib.util.spec_from_file_location('remap_fragment_kern', '/tmp/remap_fragment_kern.py')
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
x = (torch.arange(32 * 64) % 251).to(torch.float16).view(32, 64).cuda()
rows, cols = torch.arange(32).view(-1, 1), torch.arange(64).view(1, -1)
for threads, r, c in ((32, 2, 1), (32, 0, 1), (32, 2, 0), (64, 2, 1), (128, 2, 1)):
    k = mod.kern(threads, r, c)
    out = torch.zeros(32, 64, dtype=torch.float16, device='cuda')
    k(x, out)
    torch.cuda.synchronize()
    want = x.cpu()[(rows + r) % 32, (cols + c) % 64]
    dropped = x.cpu()[(rows + r) % 32, cols.expand(32, 64)]
    body = k.get_kernel_source()
    read = [l.strip() for l in body.splitlines() if 'b2[' in l and 'b1' in l]
    print(f'threads={threads:3d} (i+{r}, j+{c}): wrong {int((out.cpu() != want).sum())}/2048, '
          f'equals column-dropped {torch.equal(out.cpu(), dropped)}')
    print('    ', (read[0] if read else '?')[:150])
