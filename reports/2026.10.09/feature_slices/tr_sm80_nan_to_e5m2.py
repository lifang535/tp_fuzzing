"""Triton f16 NaN -> float8e5 on target sm_80 (run on sm_89): which NaN payloads survive?"""
import sys, torch, triton, triton.language as tl
from triton.backends.compiler import GPUTarget

@triton.jit
def k(X, O):
    o = tl.arange(0, 1024)
    tl.store(O + o, tl.load(X + o).to(tl.float8e5))

arch = sys.argv[1]
if arch != 'native':
    t = GPUTarget('cuda', int(arch), 32)
    triton.runtime.driver.active.get_current_target = lambda: t
bits = torch.cat([torch.arange(0x7C01, 0x7C01 + 1023), torch.tensor([0x7FFF])]).to(torch.int16)  # NaN payloads
x = bits.view(torch.float16).cuda()
out = torch.empty(1024, dtype=torch.float8_e5m2, device='cuda')
k[(1,)](x, out)
f = out.float().cpu()
bad = (~f.isnan()).nonzero().flatten()
print(f'target={arch}: {len(bad)}/1024 f16 NaNs did not become NaN')
for i in bad[[0, len(bad) // 2, -1]].tolist() if len(bad) else []:
    print(f'   f16 bits 0x{int(bits[i]) & 0xffff:04x} -> {f[i].item()}')
