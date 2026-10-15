"""TileLang T.Pipelined whose body copies a tile into a fragment/shared buffer, row-reduces it, accumulates.
Usage: python tl_pipelined_stage_reduce.py <fragment|shared> <auto|explicit> [dtype]"""
import sys, torch, tilelang
import tilelang.language as T

where, sched = sys.argv[1], sys.argv[2]
dt = sys.argv[3] if len(sys.argv) > 3 else 'float32'
ORDER, STAGE = ([0, 1, 2], [0, 1, 1]) if sched == 'explicit' else (None, None)

@tilelang.jit
def make():
    @T.prim_func
    def k(X: T.Tensor((64, 64), dt), O: T.Tensor((32,), "float32")):
        with T.Kernel(1, threads=32) as bx:
            xs = T.alloc_fragment((32, 64), dt) if where == 'fragment' else T.alloc_shared((32, 64), dt)
            row = T.alloc_fragment((32,), dt)
            acc = T.alloc_fragment((32,), "float32")
            T.clear(acc)
            for t in T.Pipelined(2, num_stages=2, order=ORDER, stage=STAGE):
                T.copy(X[t * 32, 0], xs)
                T.reduce_sum(xs, row, dim=1)
                for i in T.Parallel(32):
                    acc[i] = acc[i] + T.cast(row[i], "float32")
            T.copy(acc, O)
    return k
tdt = getattr(torch, dt)
try:
    kern = make()
    x = torch.randint(-3, 4, (64, 64), device='cuda').to(tdt); o = torch.empty(32, device='cuda'); kern(x, o)
    print(f'{where} {sched} {dt}: OK err {(o - x.double().view(2, 32, 64).sum((0, 2)).float()).abs().max().item():.1e}')
except Exception as e:
    lines = [l for l in str(e).splitlines() if l.strip()]
    print(f'{where} {sched} {dt}: FAIL {lines[-1][:120] if lines else type(e).__name__}')
