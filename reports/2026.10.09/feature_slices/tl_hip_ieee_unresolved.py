"""TileLang 0.1.14, HIP backend: the IEEE rounding-mode intrinsics
(T.ieee_add/sub/mul/fmaf/fsqrt/frcp/fdiv) have CUDA lowerings but no HIP
lowering, so HIP code generation stops at 'Unresolved call tl.ieee_*'.
Lowering only; no AMD GPU or hipcc needed."""
import tilelang
import tilelang.language as T
from tilelang import tvm

CALLS = {
    'ieee_add': lambda a, b: T.ieee_add(a, b, 'rn'),
    'ieee_mul': lambda a, b: T.ieee_mul(a, b, 'rz'),
    'ieee_fmaf': lambda a, b: T.ieee_fmaf(a, b, a, 'rn'),
    'ieee_fsqrt': lambda a, b: T.ieee_fsqrt(a, 'rn'),
    'ieee_frcp': lambda a, b: T.ieee_frcp(a, 'rn'),
    'ieee_fdiv': lambda a, b: T.ieee_fdiv(a, b, 'rn'),
}


def build(call):
    @T.prim_func
    def main(X: T.Tensor((128,), 'float32'), Y: T.Tensor((128,), 'float32'), O: T.Tensor((128,), 'float32')):
        with T.Kernel(1, threads=128) as bx:
            for i in T.Parallel(128):
                O[i] = call(X[i], Y[i])
    return main


for name, call in CALLS.items():
    for target in ({'kind': 'cuda', 'arch': 'sm_89'}, {'kind': 'hip', 'mcpu': 'gfx942'}):
        try:
            with tvm.target.Target(target):
                tilelang.lower(build(call), target=target)
            print(f'{name} {target["kind"]}: lowered')
        except Exception as exc:
            print(f'{name} {target["kind"]}: {type(exc).__name__}: {str(exc).strip().splitlines()[-1][:90]}')
