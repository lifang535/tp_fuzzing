"""Triton 3.8 / bundled ptxas 12.9: a value folded to a constant, converted f32 -> fp8 and stored,
gets garbage in 2 of every 4 lanes (SASS: F2FP.SATFINITE.E5M2.F16.UNPACK_B_MERGE_C merges the high half
from an unrelated register). The PTX is correct; TRITON_PTXAS_PATH=<ptxas 12.4> gives the right result.
Run: python ptxas_const_fp8.py"""
import sys
import torch, triton, triton.language as tl

@triton.jit
def kernel(X, OUT, N: tl.constexpr, FOLD: tl.constexpr):
    offs = tl.arange(0, N)
    x = tl.load(X + offs)
    if FOLD:
        v = tl.where(tl.minimum(x, -1) > 1, x, 1).to(tl.float32)
    else:
        v = (x - x + 1).to(tl.float32) + (x > 1000).to(tl.float32)
    tl.store(OUT + offs, v.to(tl.float8e5))

for n in (256, 512, 1024, 2048, 4096):
    for warps in (1, 4):
        for fold in (1, 0):
            x = torch.arange(n, dtype=torch.int32, device='cuda') % 9
            out = torch.full((n,), 7.0, device='cuda').to(torch.float8_e5m2)
            k = kernel[(1,)](x, out, n, fold, num_warps=warps)
            got = out.float().cpu()
            wrong = (got != 1).nonzero().flatten()
            print(f'n={n:5d} warps={warps} fold={fold}: {len(wrong):5d} wrong; first wrong lanes {wrong[:6].tolist()} values {got[wrong[:3]].tolist()}')
            if fold and n == 4096 and warps == 4 and len(sys.argv) > 1:
                ptx = k.asm['ptx']
                lines = [l.strip() for l in ptx.splitlines() if 'cvt' in l and ('f16' in l or 'e5m2' in l) or 'st.global' in l or 'mov.b32' in l and '{' in l]
                print('\n'.join(lines[:24]))
if len(sys.argv) > 2:
    x = torch.arange(4096, dtype=torch.int32, device='cuda') % 9
    out = torch.empty(4096, device='cuda').to(torch.float8_e5m2)
    k = kernel[(1,)](x, out, 4096, 1, num_warps=4)
    ptx = k.asm['ptx']
    lines = ptx.splitlines()
    i = next(n for n, l in enumerate(lines) if 'cvt.rz.f16.f32' in l)
    print('\n'.join(lines[max(0, i - 12):i + 6]))
    open('const_fp8.ptx', 'w').write(ptx)
    open('const_fp8.cubin', 'wb').write(k.asm['cubin'])
