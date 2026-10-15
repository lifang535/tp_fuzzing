"""Lower-only: TileLang gemm with clear_accum=(k == 0) inside T.Pipelined, with the documented pass config
tl.loop_unswitching_allow_non_trivial_else. Usage: python tl_unswitch_else_ssa.py [arch] [dtype] [config 0/1]"""
import sys, tilelang
import tilelang.language as T
from tilelang import tvm

arch, dt, cfg = (sys.argv[1:] + ['sm_75', 'float8_e5m2', '1'][len(sys.argv) - 1:])[:3]

@T.prim_func
def main(A: T.Tensor((64, 96), dt), B: T.Tensor((96, 64), dt), O: T.Tensor((64, 64), "float32")):
    with T.Kernel(1, 1, threads=128) as (bx, by):
        As = T.alloc_shared((64, 32), dt)
        Bs = T.alloc_shared((32, 64), dt)
        C = T.alloc_fragment((64, 64), "float32")
        T.fill(C, 7)
        for k in T.Pipelined(3, num_stages=1):
            T.copy(A[0, k * 32], As)
            T.copy(B[k * 32, 0], Bs)
            T.gemm(As, Bs, C, clear_accum=(k == 0))
        T.copy(C, O)

target = {'kind': 'cuda', 'arch': arch}
configs = {'tl.loop_unswitching_allow_non_trivial_else': bool(int(cfg))}
try:
    with tvm.transform.PassContext(opt_level=3, config=configs), tvm.target.Target(target):
        tilelang.lower(main, target=target)
    print(f'{arch} {dt} unswitch_else={cfg}: OK')
except Exception as e:
    print(f'{arch} {dt} unswitch_else={cfg}: FAIL {[l for l in str(e).splitlines() if l.strip()][-1][:150]}')
