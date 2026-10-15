"""float32 -> fp8 (.to(tl.float8e4nv / tl.float8e5)) rounding on Triton 3.8 vs correct RTNE (torch, numpy agree)."""
import torch, triton, triton.language as tl

@triton.jit
def kernel(X, O, DST: tl.constexpr):
    offs = tl.arange(0, 16)
    tl.store(O + offs, tl.load(X + offs).to(DST))

for name, tdt, odt, values in (
        ('e4m3', tl.float8e4nv, torch.float8_e4m3fn, [0.53125006, 0.5312499, 0.53125, 0.5625, 240.0 - 2 ** -10, 232.0001, 3.0e-3, 0.0,
                                                      -0.53125006, 1.0625001, 17.000002, 0.0078125, 0.009765625, 448.0, 300.0, 0.25]),
        ('e5m2', tl.float8e5, torch.float8_e5m2, [36.000004, 35.999996, 36.0, 40.0, 0.15625001, 1.25000001, 3.0e-5, 0.0,
                                                  -36.000004, 7.0000005, 0.375, 1.0, 2.5, 57344.0, 3.5, 0.75])):
    x = torch.tensor(values, dtype=torch.float32)
    out = torch.empty(16, dtype=odt, device='cuda')
    kernel[(1,)](x.cuda(), out, tdt)
    got, ref = out.float().cpu(), x.to(odt).float()
    for v, g, r in zip(x.tolist(), got.tolist(), ref.tolist()):
        if g != r:
            print(f'{name}: x={v!r:>14}  triton={g:<10} correct={r}')
