"""T.atomic_add on fp16 destinations: does it write outside the destination?"""
import sys, torch, tilelang, tilelang.language as T

def build(n, slots, threads, dtype='float16', index='mod'):
    @tilelang.jit
    def kern():
        @T.prim_func
        def main(X: T.Tensor((n,), dtype), O: T.Tensor((slots,), dtype)):
            with T.Kernel(1, threads=threads) as bx:
                for i in T.Parallel(n):
                    if index == 'mod':
                        T.atomic_add(O[i % slots], X[i])
                    else:
                        T.atomic_add(O[i // (n // slots)], X[i])
        return main
    return kern()

for dtype in ('float16', 'bfloat16', 'float32'):
    for n, slots, threads, index in ((128, 1, 64, 'mod'), (128, 1, 128, 'mod'), (128, 2, 64, 'mod'), (256, 64, 64, 'block'),
                                     (256, 64, 256, 'block'), (256, 64, 64, 'mod'), (128, 16, 128, 'block')):
        k = build(n, slots, threads, dtype, index)
        guard = 8
        storage = torch.full((slots + 2 * guard,), 7.0, dtype=getattr(torch, dtype), device='cuda')
        out = storage[guard:guard + slots]
        out.fill_(0)
        x = torch.ones(n, dtype=getattr(torch, dtype), device='cuda')
        try:
            k(x, out)
            torch.cuda.synchronize()
            ok_guard = bool((storage[:guard] == 7).all() and (storage[-guard:] == 7).all())
            print(dtype, n, slots, threads, index, 'out', out[:4].tolist(), 'guard ok' if ok_guard else
                  f'GUARD CHANGED left={storage[:guard].tolist()} right={storage[-guard:].tolist()}', flush=True)
        except Exception as e:
            print(dtype, n, slots, threads, index, 'ERR', str(e).splitlines()[0][:100], flush=True)
            sys.exit(0)
