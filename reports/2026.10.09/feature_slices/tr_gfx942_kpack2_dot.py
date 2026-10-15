"""Compile-only: one tl.dot (16x16 @ 16x64) on AMD gfx942 with the documented option kpack=2.
Usage: python tr_gfx942_kpack2_dot.py [dtype] [kpack] [num_warps] [K]"""
import sys, triton, triton.language as tl
from triton.backends.compiler import GPUTarget

@triton.jit
def k(X, Y, O, K: tl.constexpr):
    a = tl.load(X + tl.arange(0, 16)[:, None] * K + tl.arange(0, K)[None, :])
    b = tl.load(Y + tl.arange(0, K)[:, None] * 64 + tl.arange(0, 64)[None, :])
    tl.store(O + tl.arange(0, 16)[:, None] * 64 + tl.arange(0, 64)[None, :], tl.dot(a, b))

dt, kp, w, K = (sys.argv[1:] + ['fp32', '2', '1', '16'][len(sys.argv) - 1:])[:4]
src = triton.compiler.ASTSource(fn=k, signature={'X': '*' + dt, 'Y': '*' + dt, 'O': '*fp32', 'K': 'constexpr'},
                                constexprs={'K': int(K)})
try:
    triton.compile(src, target=GPUTarget('hip', 'gfx942', 64), options={'num_warps': int(w), 'kpack': int(kp)})
    print(f'gfx942 {dt} kpack={kp} warps={w} K={K}: OK')
except Exception as e:
    print(f'gfx942 {dt} kpack={kp} warps={w} K={K}: FAIL {str(e).splitlines()[-1][:150]}')
