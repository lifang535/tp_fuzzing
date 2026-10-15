"""TileLang 0.1.14 (#23): T.copy of a shared-memory region with a column offset
into a register fragment, lowered to ldmatrix, drops the offset.

T.copy(b1[0:64, 8:16], b2) with b1 a (64, 16) float16 shared buffer and b2 a
(64, 8) fragment, at threads=256 (2 elements per thread), is lowered to
    tl::ptx_ldmatrix_x1(&b1[(threadIdx.x >> 5) * 128 + (threadIdx.x & 7) * 16], &b2[0])
with no "+ 8": the fragment silently receives columns 0:8. At threads=128 the
same copy becomes a plain uint2 load from b1 + ... + 8 and is correct; 4-byte
types (no ldmatrix) and a zero column offset are correct too. Related in effect
to #20 (a dropped column offset) but on a different lowering path (ldmatrix
copy selection here, vectorized T.Parallel index simplification there).

Found by the layout slice (reshape then region copy) at the 2026-10-11 00:19
check (vGPU, seed 42, failed_slice_layout_bdfd90df8bad62b0); T.reshape is not
needed.
"""
import torch
import tilelang
import tilelang.language as T


@tilelang.jit
def kern(dt: str, threads: int, c0: int):
    @T.prim_func
    def main(X: T.Tensor((64, 16), dt), O: T.Tensor((64, 8), dt)):
        with T.Kernel(1, threads=threads) as bx:
            b1 = T.alloc_shared((64, 16), dt)
            T.copy(X, b1)
            b2 = T.alloc_fragment((64, 8), dt)
            T.copy(b1[0:64, c0:c0 + 8], b2)
            T.copy(b2, O)
    return main


TORCH = {'float16': torch.float16, 'bfloat16': torch.bfloat16, 'int16': torch.int16, 'float32': torch.float32}
for dt, threads, c0 in (('float16', 256, 8), ('bfloat16', 256, 8), ('int16', 256, 8),
                        ('float16', 128, 8), ('float16', 256, 0), ('float32', 256, 8)):
    x = (torch.arange(64 * 16) % 97).to(TORCH[dt]).view(64, 16).cuda()
    out = torch.zeros(64, 8, dtype=TORCH[dt], device='cuda')
    k = kern(dt, threads, c0)
    k(x, out)
    torch.cuda.synchronize()
    want = x.cpu()[:, c0:c0 + 8]
    wrong = int((out.cpu() != want).sum())
    ldmatrix = 'ldmatrix' in k.get_kernel_source()
    print(f'{dt:8s} threads={threads} cols {c0}:{c0 + 8}: wrong {wrong}/512, '
          f'equals cols 0:8 {torch.equal(out.cpu(), x.cpu()[:, 0:8])}, ldmatrix {ldmatrix}')
