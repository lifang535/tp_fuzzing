"""TileLang 0.1.14 (#22): an unsound integer simplification of nested floordiv.

T.floordiv(T.floordiv(x, 3) * -1, 4) is compiled to floordiv(-x, 12): the
generated CUDA is
    int rmod = (x % 12); int rdiv = (x / -12);
    out = (rmod <= 0) ? rdiv : (rdiv - 1);
floordiv(floordiv(y, 3), 4) == floordiv(y, 12) holds for every integer y, so the
invalid step is moving the negation inside the inner floordiv:
-floor(x/3) is ceil(-x/3), not floor(-x/3). Values stay tiny, so no overflow is
involved. Found by the cast slice at the 2026-10-10 18:19 check (vGPU, seed 42,
failed_slice_cast_38626a55970a496e); a positive multiplier is a control.
"""
import re

import torch
import tilelang
import tilelang.language as T


@tilelang.jit
def kern(mul: int):
    @T.prim_func
    def main(X: T.Tensor((32,), "int32"), O: T.Tensor((32,), "int32")):
        with T.Kernel(1, threads=32) as bx:
            for i in T.Parallel(32):
                O[i] = T.floordiv(T.floordiv(X[i], T.cast(3, "int32")) * T.cast(mul, "int32"),
                                  T.cast(4, "int32"))
    return main


xs = list(range(-8, 9))
x = torch.tensor(xs + [0] * (32 - len(xs)), dtype=torch.int32).cuda()
for mul in (-1, -2, 2):
    out = torch.zeros(32, dtype=torch.int32, device='cuda')
    k = kern(mul)
    k(x, out)
    torch.cuda.synchronize()
    got = out.cpu().tolist()[:len(xs)]
    want = [((v // 3) * mul) // 4 for v in xs]       # Python // is floordiv
    bad = [(v, g, w) for v, g, w in zip(xs, got, want) if g != w]
    print(f'mul={mul:2d}: wrong {len(bad)}/{len(xs)}  (x, got, want): {bad}')
    body = k.get_kernel_source()
    body = body[body.index('extern "C" __global__ void __launch_bounds__'):]
    lines = [l.strip() for l in body.splitlines() if 'rmod' in l or 'rdiv' in l or 'O[' in l]
    print('    ', ' | '.join(lines)[:230])
