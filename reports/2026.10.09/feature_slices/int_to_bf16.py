"""TileLang T.cast from integer to bfloat16/float16 does not round to nearest-even.

Reference: integer -> float64 is exact; torch rounds float32 -> bf16/f16 to nearest-even correctly, and every
value below is exact in float32, so x.to(float32).to(dst) is the correctly rounded result.
"""
import sys
import torch, tilelang, tilelang.language as T

def build(src, dst, stage, n=128):
    @tilelang.jit
    def kern():
        @T.prim_func
        def main(X: T.Tensor((n,), src), O: T.Tensor((n,), dst)):
            with T.Kernel(1, threads=128) as bx:
                if stage == 'global':
                    for i in T.Parallel(n):
                        O[i] = T.cast(X[i], dst)
                else:
                    xs = T.alloc_fragment((n,), src)
                    of = T.alloc_fragment((n,), dst)
                    T.copy(X, xs)
                    for i in T.Parallel(n):
                        of[i] = T.cast(xs[i], dst)
                    T.copy(of, O)
        return main
    return kern()

# ties and near-ties of the destination grid: k * q + q/2 (+-1)
def values(src, dst):
    mant = 8 if dst == 'bfloat16' else 11
    hi = {'int16': 2 ** 15 - 1, 'int32': 2 ** 31 - 1, 'int64': 2 ** 40, 'uint16': 2 ** 16 - 1}[src]
    out = []
    for e in range(mant, min(hi.bit_length() - 1, 30)):
        q = 2 ** (e - mant + 1)
        for k in (2 ** (mant - 1) + 1, 2 ** (mant - 1) + 2, 2 ** mant - 3):
            base = k * q
            for d in (q // 2 - 1, q // 2, q // 2 + 1):
                if d > 0 and base + d <= hi:
                    out += [base + d, -(base + d)]
    out = out[:128] + [0] * (128 - len(out[:128]))
    return torch.tensor(out, dtype=torch.int64)

for src in ('int16', 'int32', 'int64'):
    for dst in ('bfloat16', 'float16'):
        if dst == 'float16' and src == 'int64':
            continue
        for stage in ('global', 'fragment'):
            x = values(src, dst)
            if dst == 'float16':
                x = x.clamp(-65504, 65504)
            ref = x.to(torch.float64).to(torch.float32).to(getattr(torch, dst))
            try:
                k = build(src, dst, stage)
                out = torch.empty(128, dtype=getattr(torch, dst), device='cuda')
                k(x.to(getattr(torch, src)).cuda(), out)
                torch.cuda.synchronize()
            except Exception as e:
                print(f'{src:5s}->{dst:8s} {stage:8s}: ERROR {str(e).strip().splitlines()[-1][:90]}')
                continue
            got = out.cpu()
            bad = (got.float() != ref.float()).nonzero().flatten()
            if len(bad):
                i = bad[0].item()
                exact = x[i].item()
                print(f'{src:5s}->{dst:8s} {stage:8s}: WRONG {len(bad):3d}/128  e.g. x={exact} got={got[i].item()} '
                      f'correct={ref[i].item()} (toward zero={"yes" if abs(got[i].item()) < abs(exact) else "no"})')
            else:
                print(f'{src:5s}->{dst:8s} {stage:8s}: ok')
if len(sys.argv) > 1:
    src = build('int16', 'bfloat16', sys.argv[1]).get_kernel_source()
    body = src[src.index('main_kernel('):]
    print(body[body.index('{'):][:700])
