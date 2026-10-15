"""libdevice via Triton 3.8 (#24): signed libdevice.mul24 loses its documented
32-bit wraparound when its only use is a widening cast.

CUDA documents __mul24 / __nv_mul24 as "the least significant 32 bits of the
product of the least significant 24 bits of x and y". The libdevice bitcode
shipped with Triton (backends/nvidia/lib/libdevice.10.bc) implements it as

    %1 = shl i32 %x, 8 ; %2 = ashr i32 %1, 8      ; sign-extend the low 24 bits
    %3 = shl i32 %y, 8 ; %4 = ashr i32 %3, 8
    %5 = mul nsw i32 %2, %4

but two 24-bit values multiply to up to 2**46, so the `nsw` (no signed wrap)
claim is false whenever the product needs more than 31 bits. With
r.to(tl.int64) as the only use, the NVPTX backend legally (given nsw) turns
sext(mul nsw) into mul.wide.s32 and returns the full 48-bit product. The
unsigned __nv_umul24 (plain mul) and Triton's own x * y (plain mul) are
correct. Found by the math slice at the 2026-10-11 03:19 check (4090, seed 43,
failed_slice_math_5ab1471c92343be5): got -29231024119628 = expected
523296948 - 6806 * 2**32. Inspect the bitcode with
    llvm-dis libdevice.10.bc -o - | awk '/define .*@__nv_mul24\\(/,/^}/'
"""
import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


@triton.jit
def kernel(X, Y, OUT, OP: tl.constexpr):
    offs = tl.arange(0, 128)
    x = tl.load(X + offs)
    y = tl.load(Y + offs)
    if OP == 0:
        r = libdevice.mul24(x, y)
    elif OP == 1:
        r = libdevice.mul24(x.to(tl.uint32, bitcast=True), y.to(tl.uint32, bitcast=True)).to(tl.int32, bitcast=True)
    else:
        r = x * y
    tl.store(OUT + offs, r.to(tl.int64))


def low24(v, signed):
    v &= 0xffffff
    return v - (1 << 24) if signed and v >= 1 << 23 else v


def wrap32(v):
    v &= 0xffffffff
    return v - (1 << 32) if v >= 1 << 31 else v


g = torch.Generator().manual_seed(0)
x = torch.randint(-2**31, 2**31 - 1, (128,), generator=g, dtype=torch.int64).to(torch.int32).cuda()
y = torch.randint(-2**31, 2**31 - 1, (128,), generator=g, dtype=torch.int64).to(torch.int32).cuda()
xs, ys = x.cpu().tolist(), y.cpu().tolist()
for op, name in ((0, 'libdevice.mul24 (signed)'), (1, 'libdevice.mul24 (uint32)'), (2, 'x * y (control)')):
    out = torch.zeros(128, dtype=torch.int64, device='cuda')
    k = kernel[(1,)](x, y, out, op)
    exact = [a * b for a, b in zip(xs, ys)] if op == 2 else \
        [low24(a, op == 0) * low24(b, op == 0) for a, b in zip(xs, ys)]
    want = [wrap32(p) for p in exact]
    got = out.cpu().tolist()
    ptx = sorted({l.split()[0] for l in k.asm['ptx'].splitlines() if l.strip().startswith('mul.')})
    print(f'{name:26s} wrong {sum(a != b for a, b in zip(got, want)):3d}/128, '
          f'equals the unwrapped product {sum(a == b for a, b in zip(got, exact)):3d}/128, PTX {ptx}')
