"""Triton#1: compare half abs with an otherwise identical float32 control."""
import json
import torch
import triton
import triton.language as tl


@triton.jit
def kernel(X, OUT, HALF: tl.constexpr):
    offsets = (tl.arange(0, 2)[:, None, None] * 128
               + tl.arange(0, 4)[None, :, None] * 32
               + tl.arange(0, 32)[None, None, :])
    value = tl.load(X + offsets).to(HALF)
    value = tl.abs(value)
    tl.store(OUT + offsets, value.to(tl.int32).to(tl.float64))


rows = []
x = -(torch.arange(256, device='cuda', dtype=torch.float32) % 7 + 1)
for name, dtype in [('float16', tl.float16), ('float32_control', tl.float32)]:
    out = torch.empty(256, dtype=torch.float64, device='cuda')
    kernel[(1,)](x, out, dtype, num_warps=2)
    got = out.cpu()
    rows.append({'case': name, 'mismatches': int((got != -x.cpu()).sum()),
                 'negative_outputs': int((got < 0).sum()), 'first_16': got[:16].tolist()})
print(json.dumps({'id': 'Triton#1', 'results': rows}, indent=2))
