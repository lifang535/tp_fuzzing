"""Triton 3.8 (ptxas 12.9), sm_89: a 32x64 @ 64x128 f16 dot with num_warps=1
and the documented launch option maxnreg=128 raises "an illegal memory
access" at run time; without maxnreg (214 registers), with maxnreg=255 or
with num_warps=2 the same kernel is correct. maxnreg only caps registers, so
ptxas spills (138 spills, 552-byte stack frame) and the result must not
change. Rewriting the store as `acc * 1.0 + 0.0` (140 spills) also passes, so
the fault depends on the exact spill code. Found by TileSmith b1eded9f
(4090, gemm slice, 2026-10-12 17:28 check); the campaign failure ran on an
RTX 4090, this repro on an RTX 4060."""
import sys
import torch
import triton
import triton.language as tl


@triton.jit
def kernel(A, B, OUT):
    rm = tl.arange(0, 32)
    rn = tl.arange(0, 128)
    rk = tl.arange(0, 64)
    a = tl.load(A + (rm[:, None] * 47 + rk[None, :]), mask=(rk[None, :] < 47), other=0.0)
    b = tl.load(B + (rk[:, None] * 128 + rn[None, :]), mask=(rk[:, None] < 47), other=0.0)
    acc = tl.dot(a, b, out_dtype=tl.float32)
    tl.store(OUT + (rm[:, None] * 128 + rn[None, :]), acc)


maxnreg = int(sys.argv[1]) if len(sys.argv) > 1 else 128
a = torch.randint(-8, 8, (32, 47), device='cuda').half()
b = torch.randint(-8, 8, (47, 128), device='cuda').half()
out = torch.empty(32, 128, device='cuda')
k = kernel[(1,)](a, b, out, num_warps=1, **({'maxnreg': maxnreg} if maxnreg else {}))
print('registers', k.n_regs, 'spills', k.n_spills, flush=True)
torch.cuda.synchronize()  # maxnreg=128: RuntimeError: CUDA error: an illegal memory access was encountered
print('correct' if torch.equal(out, a.float() @ b.float()) else 'WRONG')
