"""Lower-only: TileLang narrow-integer select inside T.Parallel on the HIP backend.
Usage: python tl_amd_narrow_select.py <target> <dtype> <form>
form: select_xor | select | xor | reduce_only"""
import sys, tilelang
import tilelang.language as T
from tilelang import tvm

tgt, dt, form = sys.argv[1:4]
target = {'kind': 'cuda', 'arch': tgt} if tgt.startswith('sm') else {'kind': 'hip', 'mcpu': tgt}

@T.prim_func
def f(X: T.Tensor((128, 4), "float32"), O: T.Tensor((128, 4), dt)):
    with T.Kernel(1, threads=128) as bx:
        xf = T.alloc_fragment((128, 4), dt)
        for i, j in T.Parallel(128, 4):
            if form == 'select_xor':
                xf[i, j] = T.if_then_else(i < 128, T.cast(X[i, j], dt) ^ T.cast(3, dt), T.cast(0, dt))
            elif form == 'select':
                xf[i, j] = T.if_then_else(i < 128, T.cast(X[i, j], dt), T.cast(0, dt))
            else:
                xf[i, j] = T.cast(X[i, j], dt) ^ T.cast(3, dt)
        T.copy(xf, O)
try:
    with tvm.transform.PassContext(opt_level=3), tvm.target.Target(target):
        tilelang.lower(f, target=target)
    print(f'{tgt} {dt} {form}: OK')
except Exception as e:
    lines = [l for l in str(e).splitlines() if l.strip()]
    print(f'{tgt} {dt} {form}: FAIL {lines[-1][:120]}')
