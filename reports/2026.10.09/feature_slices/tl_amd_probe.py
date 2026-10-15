"""Lower-only probes of TileLang's HIP backend. Usage: python tl_amd_probe.py <case> <target> [dtype] [config]
target: gfx942 / gfx950 / sm_89 (CUDA, for contrast)."""
import sys, tilelang
import tilelang.language as T
from tilelang import tvm

case, tgt = sys.argv[1], sys.argv[2]
dt = sys.argv[3] if len(sys.argv) > 3 else None
cfg = {}
if len(sys.argv) > 4 and sys.argv[4] == 'index64':
    cfg = {'tl.config_index_bitwidth': 64}
target = {'kind': 'cuda', 'arch': tgt} if tgt.startswith('sm') else {'kind': 'hip', 'mcpu': tgt}

if case == 'fp8_const':        # select between a loaded fp8 value and an fp8 constant, then widen
    dt = dt or 'float8_e5m2'
    @T.prim_func
    def f(X: T.Tensor((128,), dt), O: T.Tensor((128,), "float32")):
        with T.Kernel(1, threads=128) as bx:
            for i in T.Parallel(128):
                O[i] = T.cast(T.if_then_else(i < 95, X[i], T.cast(0, dt)), "float32")
elif case == 'fp8_gemm':
    dt = dt or 'float8_e4m3'
    @T.prim_func
    def f(A: T.Tensor((64, 32), dt), B: T.Tensor((32, 64), dt), O: T.Tensor((64, 64), "float32")):
        with T.Kernel(1, 1, threads=128) as (bx, by):
            As = T.alloc_shared((64, 32), dt); Bs = T.alloc_shared((32, 64), dt)
            C = T.alloc_fragment((64, 64), "float32")
            T.clear(C); T.copy(A, As); T.copy(B, Bs); T.gemm(As, Bs, C); T.copy(C, O)
elif case == 'u8_reduce':      # narrow-integer max-reduction over a fragment
    dt = dt or 'uint8'
    @T.prim_func
    def f(X: T.Tensor((128, 4), dt), O: T.Tensor((4,), dt)):
        with T.Kernel(1, threads=128) as bx:
            xf = T.alloc_fragment((128, 4), dt); of = T.alloc_fragment((4,), dt)
            T.copy(X, xf)
            T.reduce_max(xf, of, dim=0, clear=True)
            T.copy(of, O)
elif case == 'i16_add':        # int16 elementwise add of a constant, 32 threads
    dt = dt or 'int16'
    @T.prim_func
    def f(X: T.Tensor((16, 16), dt), O: T.Tensor((16, 16), dt)):
        with T.Kernel(1, threads=32) as bx:
            b = T.alloc_fragment((16, 16), dt); o = T.alloc_fragment((16, 16), dt)
            T.copy(X, b)
            for i, j in T.Parallel(16, 16):
                o[i, j] = b[i, j] + T.cast(1, dt)
            T.copy(o, O)
try:
    with tvm.transform.PassContext(opt_level=3, config=cfg), tvm.target.Target(target):
        tilelang.lower(f, target=target)
    print(f'{case} {tgt} {dt} {cfg or ""}: OK')
except Exception as e:
    lines = [l for l in str(e).splitlines() if l.strip()]
    print(f'{case} {tgt} {dt} {cfg or ""}: FAIL {type(e).__name__}: {lines[-1][:140] if lines else ""}')
