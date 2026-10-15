"""Triton 3.8, AMD backend: tl.atomic_xchg on int64 emits the LLVM
intrinsic llvm.amdgcn.raw.ptr.buffer.atomic.exch, which does not exist
(the buffer atomic is .swap), so LLVM translation fails. Compile only."""
import sys
import triton
import triton.language as tl
from triton.backends.compiler import GPUTarget


@triton.jit
def kernel(P, V):
    r = tl.arange(0, 64)
    tl.atomic_xchg(P + r, tl.load(V + r))


for dt in ('i64', 'i32', 'fp32', 'fp64'):
    src = triton.compiler.ASTSource(fn=kernel, signature={'P': f'*{dt}', 'V': f'*{dt}'}, constexprs={})
    for arch in ('gfx942', 'gfx950', 'gfx90a', 'gfx1100'):
        try:
            triton.compile(src, target=GPUTarget('hip', arch, 32 if arch.startswith('gfx11') else 64))
            print(f'{dt} {arch}: compiled')
        except Exception as exc:
            print(f'{dt} {arch}: {type(exc).__name__}: {str(exc).splitlines()[0][:100]}')
