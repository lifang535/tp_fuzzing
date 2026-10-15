import sys
import torch
import tilelang
import tilelang.language as T


def build(kind, trips=3):
    @tilelang.jit
    def kern():
        @T.prim_func
        def main(X: T.Tensor((trips * 32, 32), "float16"), Y: T.Tensor((trips * 32, 32), "float16"),
                 O: T.Tensor((32, 32), "float32")):
            with T.Kernel(1, threads=128) as bx:
                xs = T.alloc_shared((32, 32), "float16")
                ws = T.alloc_shared((32, 32), "float16")
                acc = T.alloc_fragment((32, 32), "float32")
                T.clear(acc)
                if kind == 'while':
                    t = T.alloc_var("int32", init=0)
                    while t < trips:
                        T.copy(X[t * 32, 0], xs)
                        T.copy(Y[t * 32, 0], ws)
                        T.gemm(xs, ws, acc)
                        t = t + 1
                else:
                    for t in T.serial(trips):
                        T.copy(X[t * 32, 0], xs)
                        T.copy(Y[t * 32, 0], ws)
                        T.gemm(xs, ws, acc)
                T.copy(acc, O)
        return main
    return kern()


for kind in ('serial', 'while'):
    k = build(kind)
    g = torch.Generator().manual_seed(0)
    x = torch.randint(-2, 3, (96, 32), generator=g).half().cuda()
    y = torch.randint(-2, 3, (96, 32), generator=g).half().cuda()
    want = sum(x[t * 32:(t + 1) * 32].float() @ y[t * 32:(t + 1) * 32].float() for t in range(3))
    results = []
    for run in range(20):
        o = torch.zeros(32, 32, device='cuda')
        k(x, y, o)
        torch.cuda.synchronize()
        results.append(o.clone())
    wrong = sum(int(not torch.equal(r, want)) for r in results)
    distinct = len({tuple(r.flatten().tolist()) for r in results})
    print(f'{kind}: wrong runs {wrong}/20, distinct outputs {distinct}')
    if kind == 'while':
        src = k.get_kernel_source()
        lines = [l.strip() for l in src.splitlines() if 'sync' in l or 'while' in l or 'for (' in l or 'gemm' in l]
        print('\n'.join(lines[:20]))
