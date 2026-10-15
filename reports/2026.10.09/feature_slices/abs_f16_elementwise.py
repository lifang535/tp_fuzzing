"""Triton 3.8 / ptxas 12.9: tl.abs on float16 values (no dot involved) keeps the sign in some lanes.

Generalizes mechanism #1: the PTX applies `and.b16 0x7fff` to every half; ptxas 12.9 masks only some of
the halves it packs into 32-bit registers. TRITON_PTXAS_PATH=<ptxas 12.4> gives the right result.
"""
import sys
import torch, triton, triton.language as tl

@triton.jit
def kernel(X, OUT, A: tl.constexpr, B: tl.constexpr, C: tl.constexpr, TO_INT: tl.constexpr, HALF: tl.constexpr):
    offs = tl.arange(0, A)[:, None, None] * (B * C) + tl.arange(0, B)[None, :, None] * C + tl.arange(0, C)[None, None, :]
    v = tl.load(X + offs).to(HALF)
    v = tl.abs(v)
    if TO_INT:
        tl.store(OUT + offs, v.to(tl.int32).to(tl.float64))
    else:
        tl.store(OUT + offs, v.to(tl.float64))

for half_name, half in (('f16', tl.float16), ('bf16', tl.bfloat16)):
    for shape in ((2, 4, 32), (1, 8, 32), (4, 4, 16), (1, 1, 256)):
        for warps in (1, 2, 4):
            for to_int in (1, 0):
                n = shape[0] * shape[1] * shape[2]
                x = -(torch.arange(n, device='cuda', dtype=torch.float32) % 7 + 1)
                out = torch.zeros(n, dtype=torch.float64, device='cuda')
                k = kernel[(1,)](x, out, *shape, to_int, half, num_warps=warps)
                bad = (out.cpu() < 0).sum().item()
                if bad:
                    print(f'{half_name} shape={shape} warps={warps} to_int={to_int}: {bad}/{n} negative outputs')
                if len(sys.argv) > 1 and bad and half_name == 'f16' and shape == (2, 4, 32) and warps == 2 and to_int:
                    sass_lines = k.asm.get('sass', '')
                    open('abs_f16_elementwise.cubin', 'wb').write(k.asm['cubin'])
print('done')
