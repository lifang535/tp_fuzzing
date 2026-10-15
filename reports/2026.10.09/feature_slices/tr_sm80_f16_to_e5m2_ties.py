"""Triton f16 -> float8e5 (default rounding = rtne) on target sm_80, run on an sm_89 GPU.
Feeds every finite f16 whose low 8 mantissa bits are exactly 0x80 (a tie for e5m2)."""
import sys, re, torch, triton, triton.language as tl
from triton.backends.compiler import GPUTarget

@triton.jit
def k(X, O, n):
    o = tl.program_id(0) * 1024 + tl.arange(0, 1024)
    m = o < n
    tl.store(O + o, tl.load(X + o, mask=m).to(tl.float8e5), mask=m)

def rne_e5m2(bits16):
    # e5m2 is the high byte of f16; round-to-nearest-even on the dropped byte
    hi, lo = bits16 >> 8, bits16 & 0xff
    up = (lo > 0x80) | ((lo == 0x80) & ((hi & 1) == 1))
    return (hi + up.to(hi.dtype)) & 0xff

b = torch.arange(0, 1 << 16, dtype=torch.int32)
b = b[((b & 0xff) == 0x80) & (((b >> 10) & 0x1f) != 0x1f)]          # ties, finite
x = b.to(torch.int16).view(torch.float16).cuda()
want = rne_e5m2(b).to(torch.uint8)
arch = sys.argv[1] if len(sys.argv) > 1 else '80'
if arch != 'native':
    t = GPUTarget('cuda', int(arch), 32)
    triton.runtime.driver.active.get_current_target = lambda: t
out = torch.empty(x.numel(), dtype=torch.float8_e5m2, device='cuda')
h = k[(triton.cdiv(x.numel(), 1024),)](x, out, x.numel())
got = out.view(torch.uint8).cpu()
bad = (got != want).nonzero().flatten()
print(f'target={arch} triton {triton.__version__}: {len(bad)}/{x.numel()} ties rounded wrong')
for i in bad[:4].tolist():
    print(f'  f16 {x[i].item()!r:>12} -> got {out[i].float().item()}, rtne {want[i:i+1].view(torch.float8_e5m2).float().item()}')
print('  PTX cvt:', sorted(set(re.findall(r'cvt\.[\w.]+', h.asm['ptx']))))
