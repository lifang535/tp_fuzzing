"""TileLang 0.1.14: for sm_100/sm_120 the vectorizer widens accesses to 256
bits, and the CUDA printer has no type for the result: an int16 elementwise
loop becomes int16x16 and code generation fails with 'Cannot convert type
int16x16 to CUDA type'. The same kernel for sm_89/sm_90a uses 128-bit
int16x8 and builds. Lowering only; no GPU needed."""
import tilelang
import tilelang.language as T
from tilelang import tvm


def build(dtype):
    @T.prim_func
    def main(X: T.Tensor((512,), dtype), O: T.Tensor((512,), dtype)):
        with T.Kernel(1, threads=32) as bx:
            for i in T.Parallel(512):
                O[i] = X[i] + X[i]
    return main


for dtype in ('int16', 'int8', 'float16', 'int32', 'float32'):
    for arch in ('sm_89', 'sm_90a', 'sm_100a', 'sm_120'):
        target = {'kind': 'cuda', 'arch': arch}
        try:
            with tvm.target.Target(target):
                tilelang.lower(build(dtype), target=target)
            print(f'{dtype} {arch}: lowered')
        except Exception as exc:
            print(f'{dtype} {arch}: {type(exc).__name__}: {str(exc).strip().splitlines()[-1][:80]}')
