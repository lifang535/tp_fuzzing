"""TileLang layout inference: a gemm accumulator plus the row sums of an unrelated fragment.
Usage: python tl_layout_has_best.py [threads] [reduce source: other|acc] [run 0/1]"""
import sys, os, torch, tilelang
import tilelang.language as T
from tilelang import tvm
TH = int(sys.argv[1]) if len(sys.argv) > 1 else 256
SRC = sys.argv[2] if len(sys.argv) > 2 else 'other'

def typed_kernel_0():
    @T.prim_func
    def impl(A: T.Buffer((64, 8), "float16"), B: T.Buffer((8, 64), "float16"), H: T.Buffer((64, 64), "float16"), C: T.Buffer((64, 64), "float32")):
        with T.Kernel(1, threads=TH) as bx:
            As = T.alloc_shared((64, 8), "float16"); Bs = T.alloc_shared((8, 64), "float16")
            acc = T.alloc_fragment((64, 64), "float32")
            wide = T.alloc_fragment((64, 64), "float32")
            stat = T.alloc_fragment((64,), "float32")
            out = T.alloc_fragment((64, 64), "float32")
            T.copy(A, As); T.copy(B, Bs)
            T.clear(acc)
            T.gemm(As, Bs, acc)
            for i, j in T.Parallel(64, 64):
                wide[i, j] = T.cast(H[i, j], "float32") if SRC == 'other' else acc[i, j]
            T.reduce_sum(wide, stat, dim=1, clear=True)
            for i, j in T.Parallel(64, 64):
                out[i, j] = stat[i] + acc[i, j]
            T.copy(out, C)
    return impl

if __name__ == '__main__':
    try:
        with tvm.transform.PassContext(opt_level=3), tvm.target.Target('cuda'):
            tilelang.lower(typed_kernel_0(), target='cuda')
        print(f'threads={TH} reduce={SRC}: OK')
    except Exception as e:
        print(f'threads={TH} reduce={SRC}: FAIL {[l for l in str(e).splitlines() if l.strip()][-1][:100]}')
