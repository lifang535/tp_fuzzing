"""TileLang T.Pipelined accumulating 16x32 tiles; the copied tile is staged in a fragment or in shared memory.
Usage: python tl_pipelined_stage_fragment.py <fragment|shared> <auto|explicit> [num_stages]"""
import sys, torch, tilelang
import tilelang.language as T

where, sched = sys.argv[1], sys.argv[2]
ns = int(sys.argv[3]) if len(sys.argv) > 3 else 2
ORDER, STAGE = ([0, 1], [0, 1]) if sched == 'explicit' else (None, None)

@tilelang.jit
def make():
    @T.prim_func
    def k(X: T.Tensor((96, 32), "float32"), O: T.Tensor((16, 32), "float32")):
        with T.Kernel(1, threads=128) as bx:
            xs = T.alloc_fragment((16, 32), "float32") if where == 'fragment' else T.alloc_shared((16, 32), "float32")
            acc = T.alloc_fragment((16, 32), "float32")
            T.clear(acc)
            for t in T.Pipelined(6, num_stages=ns, order=ORDER, stage=STAGE):
                T.copy(X[t * 16, 0], xs)
                for i, j in T.Parallel(16, 32):
                    acc[i, j] = acc[i, j] + xs[i, j]
            T.copy(acc, O)
    return k
try:
    kern = make()
    x = torch.randn(96, 32, device='cuda'); o = torch.empty(16, 32, device='cuda'); kern(x, o)
    print(f'{where} {sched} num_stages={ns}: OK err {(o - x.view(6, 16, 32).sum(0)).abs().max().item():.1e}')
except Exception as e:
    lines = [l for l in str(e).splitlines() if l.strip()]
    print(f'{where} {sched} num_stages={ns}: FAIL {lines[-1][:130] if lines else type(e).__name__}')
