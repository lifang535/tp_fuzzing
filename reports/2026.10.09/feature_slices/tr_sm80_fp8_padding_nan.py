"""Triton block-pointer load with padding_option='nan' on fp8 tensors, compiled for a given target
and run on the local GPU. Out-of-bounds elements must read as NaN."""
import sys, re, torch, triton, triton.language as tl
from triton.backends.compiler import GPUTarget

@triton.jit
def k(X, OUT):
    p = tl.make_block_ptr(X, shape=(16, 11), strides=(11, 1), offsets=(0, 0), block_shape=(16, 16), order=(1, 0))
    v = tl.load(p, boundary_check=(0, 1), padding_option='nan')
    r = tl.arange(0, 16)[:, None]; c = tl.arange(0, 16)[None, :]
    tl.store(OUT + r * 16 + c, v)

arch = sys.argv[1]
if arch != 'native':
    t = GPUTarget('cuda', int(arch), 32)
    triton.runtime.driver.active.get_current_target = lambda: t
for name in sys.argv[2:]:
    dt = getattr(torch, name)
    x = torch.ones(16, 11, device='cuda').to(dt)
    out = torch.zeros(16, 16, device='cuda').to(dt)
    h = k[(1,)](x, out)
    pad = out[:, 11:].float()
    print(f'target={arch} {name:16s} padding values: {sorted(set(pad.flatten().tolist()), key=str)}  '
          f'(all NaN: {bool(pad.isnan().all())})')
    nanconst = sorted(set(re.findall(r'0x7[cCfF][0-9A-Fa-f]{2}|0x[fF][fF][fF][fF]|0f7FC00000|0x7E', h.asm['ptx'])))
    print('   NaN-like constants in PTX:', nanconst)
