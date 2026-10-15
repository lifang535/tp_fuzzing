"""Compile-only: Triton tl.cumsum on a small fp16 tensor for AMD gfx1100 (RDNA3, wave32).
Usage: python tr_gfx1100_scan_lld.py [arch] [dtype] [reverse] [n] [num_warps]"""
import sys, triton, triton.language as tl
from triton.backends.compiler import GPUTarget

@triton.jit
def k(X, O, N: tl.constexpr, REV: tl.constexpr):
    o = tl.arange(0, N)
    tl.store(O + o, tl.cumsum(tl.load(X + o), axis=0, reverse=REV))

arch, dt, rev, n, w = (sys.argv[1:] + ['gfx1100', 'fp16', '1', '32', '4'][len(sys.argv) - 1:])[:5]
warp = 32 if arch.startswith('gfx11') or arch.startswith('gfx12') else 64
src = triton.compiler.ASTSource(fn=k, signature={'X': '*' + dt, 'O': '*' + dt, 'N': 'constexpr', 'REV': 'constexpr'},
                                constexprs={'N': int(n), 'REV': bool(int(rev))})
try:
    c = triton.compile(src, target=GPUTarget('hip', arch, warp), options={'num_warps': int(w)})
    print(f'{arch} {dt} reverse={rev} n={n} warps={w}: OK')
except Exception as e:
    print(f'{arch} {dt} reverse={rev} n={n} warps={w}: FAIL {type(e).__name__}: {str(e).splitlines()[-1][:160]}')
