"""Lower-only: TileLang T.gemm on the HIP backend with k_pack, A taken from a fragment or shared.
Usage: python tl_amd_gemm_kpack.py <target> <dtype> <k_pack> <A: frag|shared> <transpose_A 0/1> [M N K threads]"""
import sys, tilelang
import tilelang.language as T
from tilelang import tvm

tgt, dt, kp, src, ta = sys.argv[1:6]
M, N, K, th = [int(v) for v in (sys.argv[6:10] or [128, 32, 32, 128])]
kp, ta = int(kp), bool(int(ta))
target = {'kind': 'hip', 'mcpu': tgt}
ashape = (K, M) if ta else (M, K)

@T.prim_func
def f(A: T.Tensor(ashape, dt), B: T.Tensor((K, N), dt), O: T.Tensor((M, N), "float32")):
    with T.Kernel(1, threads=th) as bx:
        As = T.alloc_shared(ashape, dt)
        Bs = T.alloc_shared((K, N), dt)
        C = T.alloc_fragment((M, N), "float32")
        T.clear(C)
        T.copy(A, As); T.copy(B, Bs)
        if src == 'frag':
            Af = T.alloc_fragment(ashape, dt)
            T.copy(As, Af)
            T.gemm(Af, Bs, C, transpose_A=ta, k_pack=kp)
        else:
            T.gemm(As, Bs, C, transpose_A=ta, k_pack=kp)
        T.copy(C, O)
try:
    with tvm.transform.PassContext(opt_level=3), tvm.target.Target(target):
        tilelang.lower(f, target=target)
    print(f'{sys.argv[1:]}: OK')
except Exception as e:
    lines = [l for l in str(e).splitlines() if l.strip()]
    print(f'{sys.argv[1:]}: FAIL {type(e).__name__}: {lines[-1][:110] if lines else ""}')
