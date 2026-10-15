"""tl.abs(tl.dot(a, b)) stored as bfloat16 keeps negative values on Triton 3.8."""
import torch, triton, triton.language as tl

@triton.jit
def kernel(A, B, OUT, OUT_DT: tl.constexpr, USE_DOT: tl.constexpr, ABS: tl.constexpr):
    r = tl.arange(0, 32)
    a = tl.load(A + r[:, None] * 32 + r[None, :])
    b = tl.load(B + r[:, None] * 32 + r[None, :])
    if USE_DOT:
        acc = tl.dot(a, b, out_dtype=tl.float32)
    else:
        acc = a.to(tl.float32) * 3.0 - b.to(tl.float32)
    if ABS == 1:
        acc = tl.abs(acc)
    elif ABS == 2:
        acc = tl.where(acc < 0, -acc, acc)
    tl.store(OUT + r[:, None] * 32 + r[None, :], acc.to(OUT_DT))

g = torch.Generator().manual_seed(0)
a = torch.randint(-1, 2, (32, 32), generator=g).half().cuda()
b = torch.randint(-1, 2, (32, 32), generator=g).half().cuda()
DT = {'bf16': (tl.bfloat16, torch.bfloat16), 'f16': (tl.float16, torch.float16), 'f32': (tl.float32, torch.float32)}
for name, (tdt, odt) in DT.items():
    for use_dot in (1, 0):
        for absmode in (1, 2):
            out = torch.empty(32, 32, dtype=odt, device='cuda')
            kernel[(1,)](a, b, out, tdt, use_dot, absmode, num_warps=4)
            ref = (a.float() @ b.float()) if use_dot else (a.float() * 3 - b.float())
            ref = ref.abs()
            bad = (out.float() != ref).sum().item()
            neg = (out.float() < 0).sum().item()
            print(f'out={name:4s} {"dot" if use_dot else "ew "} {"tl.abs" if absmode == 1 else "where"}: wrong {bad:4d}/1024, negative outputs {neg}')
