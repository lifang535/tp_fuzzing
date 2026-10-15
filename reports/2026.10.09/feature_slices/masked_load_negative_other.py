"""Triton 3.8: a vectorized masked tl.load of int8/int16/uint8 with a negative
(or >= 128 unsigned) `other` fills masked lanes with -1/255 except the first
element of each packed 32-bit word.

Triton emits `mov.u32 %r, 0xfffffffffffffffe` for other=-2 (the sign-extended
scalar) where the packed register needs 0xfefefefe. ptxas 12.4 and 12.9 agree,
so the PTX is wrong. int32 and float types, and positive fills, are correct.
Found by the memory slice (2026-10-10).
"""
import torch
import triton
import triton.language as tl


@triton.jit
def kernel(X, OUT, ROWS: tl.constexpr, OTHER: tl.constexpr):
    r = tl.arange(0, 64)[:, None]
    c = tl.arange(0, 8)[None, :]
    v = tl.load(X + r * 8 + c, mask=r < ROWS, other=OTHER)
    tl.store(OUT + r * 8 + c, v)


for dtype, other in ((torch.int8, -2), (torch.int16, -300), (torch.uint8, 200), (torch.int32, -2)):
    x = torch.zeros(64, 8, dtype=dtype, device='cuda')
    out = torch.full((64, 8), 99, dtype=dtype, device='cuda')
    kernel[(1,)](x, out, 48, other, num_warps=1)
    masked = out[48:].cpu()
    print(f'{dtype}: other={other} masked row {masked[0].tolist()} wrong {int((masked != other).sum())}/{masked.numel()}')
