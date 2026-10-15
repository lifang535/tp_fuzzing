"""Compile-only: tl.range(..., loop_unroll_factor=U, warp_specialize=True) over a block-pointer load,
for an NVIDIA target. Usage: python tr_blackwell_ws_unroll.py [arch] [unroll] [blockptr]"""
import sys, triton, triton.language as tl
from triton.backends.compiler import GPUTarget

@triton.jit
def k_bp(X, O, n, U: tl.constexpr):
    acc = tl.zeros((64,), tl.float32)
    px = tl.make_block_ptr(X, shape=(n * 64,), strides=(1,), offsets=(0,), block_shape=(64,), order=(0,))
    for t in tl.range(0, n, loop_unroll_factor=U, warp_specialize=True):
        acc += tl.load(px)
        px = tl.advance(px, (64,))
    tl.store(O + tl.arange(0, 64), acc)

@triton.jit
def k_ptr(X, O, n, U: tl.constexpr):
    acc = tl.zeros((64,), tl.float32)
    for t in tl.range(0, n, loop_unroll_factor=U, warp_specialize=True):
        acc += tl.load(X + t * 64 + tl.arange(0, 64))
    tl.store(O + tl.arange(0, 64), acc)

arch, u, bp = (sys.argv[1:] + ['100', '2', '1'][len(sys.argv) - 1:])[:3]
fn = k_bp if bp == '1' else k_ptr
src = triton.compiler.ASTSource(fn=fn, signature={'X': '*fp32', 'O': '*fp32', 'n': 'i32', 'U': 'constexpr'}, constexprs={'U': int(u)})
try:
    triton.compile(src, target=GPUTarget('cuda', int(arch), 32), options={'num_warps': 4})
    print(f'sm{arch} unroll={u} blockptr={bp}: OK')
except Exception as e:
    print(f'sm{arch} unroll={u} blockptr={bp}: FAIL {str(e).splitlines()[-1][:120]}')
