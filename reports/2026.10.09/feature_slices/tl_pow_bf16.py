"""TileLang T.pow (and other float math) on bfloat16 / float16 / float32, built for sm_89 and run.
Usage: python tl_pow_bf16.py <dtype> <fn: pow|exp|sqrt|log|pow_const>"""
import sys, torch, tilelang
import tilelang.language as T

dt, fn = sys.argv[1], sys.argv[2]

@tilelang.jit
def make():
    @T.prim_func
    def k(X: T.Tensor((128,), dt), Y: T.Tensor((128,), dt), O: T.Tensor((128,), dt)):
        with T.Kernel(1, threads=128) as bx:
            for i in T.Parallel(128):
                if fn == 'pow':
                    O[i] = T.pow(X[i], Y[i])
                elif fn == 'pow_const':
                    O[i] = T.pow(X[i], T.cast(2.0, dt))
                elif fn == 'exp':
                    O[i] = T.exp(X[i])
                elif fn == 'sqrt':
                    O[i] = T.sqrt(X[i])
                else:
                    O[i] = T.log(X[i])
    return k
tdt = {'bfloat16': torch.bfloat16, 'float16': torch.float16, 'float32': torch.float32}[dt]
try:
    kern = make()
    x = torch.rand(128, device='cuda').add(0.5).to(tdt); y = torch.full((128,), 2.0, device='cuda').to(tdt)
    o = torch.empty_like(x); kern(x, y, o)
    ref = {'pow': x.float() ** y.float(), 'pow_const': x.float() ** 2, 'exp': x.float().exp(), 'sqrt': x.float().sqrt(), 'log': x.float().log()}[fn]
    print(f'{dt} {fn}: OK, max rel err {((o.float() - ref).abs() / ref.abs()).max().item():.2e}')
except Exception as e:
    lines = [l for l in str(e).splitlines() if l.strip()]
    if "-v" in sys.argv: print(str(e)[-1500:])
    print(f'{dt} {fn}: FAIL {lines[-1][:140] if lines else type(e).__name__}')
