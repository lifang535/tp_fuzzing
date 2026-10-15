"""Triton 3.8: an int8 tl.dot compiled for sm_75 (Turing) crashes in
TritonGPUAccelerateMatmul: the pass up-casts the i8 operands with
arith.extf, which only accepts floats. Compile only; no GPU needed."""
import triton
import triton.language as tl
from triton.backends.compiler import GPUTarget


@triton.jit
def kernel(A, B, C):
    r = tl.arange(0, 32)
    a = tl.load(A + r[:, None] * 32 + r[None, :])
    b = tl.load(B + r[:, None] * 32 + r[None, :])
    tl.store(C + r[:, None] * 32 + r[None, :], tl.dot(a, b, out_dtype=tl.int32))


src = triton.compiler.ASTSource(fn=kernel, signature={'A': '*i8', 'B': '*i8', 'C': '*i32'}, constexprs={})
for arch in (75, 80, 89):
    try:
        triton.compile(src, target=GPUTarget('cuda', arch, 32), options={'num_warps': 2})
        print(f'sm_{arch}: compiled')
    except Exception as exc:
        print(f'sm_{arch}: {type(exc).__name__}: {str(exc).splitlines()[0][:120]}')
