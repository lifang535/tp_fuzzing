"""Lower-only: TileLang f16 T.gemm (64x32 @ 32xN) for sm_90a with T.Pipelined.
Usage: python tl_sm90_gmma_narrow_n.py [N] [threads] [arch]"""
import sys, tilelang
import tilelang.language as T
from tilelang import tvm

N, threads, arch = (sys.argv[1:] + ['16', '256', 'sm_90a'][len(sys.argv) - 1:])[:3]
N, threads = int(N), int(threads)

@T.prim_func
def main(A: T.Tensor((64, 32), "float16"), B: T.Tensor((32, N), "float16"), O: T.Tensor((64, N), "float32")):
    with T.Kernel(1, 1, threads=threads) as (bx, by):
        As = T.alloc_shared((64, 32), "float16")
        Bs = T.alloc_shared((32, N), "float16")
        C = T.alloc_fragment((64, N), "float32")
        T.clear(C)
        T.copy(A, As)
        T.copy(B, Bs)
        T.gemm(As, Bs, C)
        T.copy(C, O)

target = {'kind': 'cuda', 'arch': arch}
try:
    with tvm.transform.PassContext(opt_level=3), tvm.target.Target(target):
        tilelang.lower(main, target=target)
    print(f'{arch} N={N} threads={threads}: OK')
except Exception as e:
    print(f'{arch} N={N} threads={threads}: FAIL {type(e).__name__}: {[l for l in str(e).splitlines() if l.strip()][-1][:130]}')
