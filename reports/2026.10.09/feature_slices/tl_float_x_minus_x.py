"""TileLang float identities on inf/NaN inputs (fast math is off by default).
Usage: python tl_float_x_minus_x.py <dtype>"""
import sys, re, torch, tilelang
import tilelang.language as T

dt = sys.argv[1] if len(sys.argv) > 1 else 'float16'
OPS = ['x - x', 'x * 0', 'x / x', 'x + 0 - x', '(x - x) * 1']

@tilelang.jit
def make():
    @T.prim_func
    def k(X: T.Tensor((4,), dt), O: T.Tensor((5, 4), dt)):
        with T.Kernel(1, threads=32) as bx:
            for i in T.Parallel(4):
                O[0, i] = X[i] - X[i]
                O[1, i] = X[i] * T.cast(0, dt)
                O[2, i] = X[i] / X[i]
                O[3, i] = X[i] + T.cast(0, dt) - X[i]
                O[4, i] = (X[i] - X[i]) * T.cast(1, dt)
    return k
kern = make()
tdt = getattr(torch, dt)
x = torch.tensor([float('inf'), float('-inf'), float('nan'), 1.5], device='cuda', dtype=tdt)
o = torch.zeros(5, 4, device='cuda', dtype=tdt); kern(x, o)
xd = x.double()
ref = torch.stack([xd - xd, xd * 0, xd / xd, xd + 0 - xd, (xd - xd) * 1])
src = kern.get_kernel_source()
for n, op in enumerate(OPS):
    got = o[n].tolist(); want = ref[n].tolist()
    same = all((a == b) or (a != a and b != b) for a, b in zip(got, want))
    print(f'{dt} {op:12s} x=[inf,-inf,nan,1.5] got {got}  ieee {want}  {"OK" if same else "WRONG"}')
print('  generated stores:', re.findall(r'O\[[^\]]*\] = [^;]*;', src)[:5])
