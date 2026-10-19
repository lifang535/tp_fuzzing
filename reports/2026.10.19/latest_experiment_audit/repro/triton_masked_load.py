"""Triton#8: negative masked fills for int8, with int32 and positive-fill controls."""
import json
import torch
import triton
import triton.language as tl


@triton.jit
def kernel(X, OUT, OTHER: tl.constexpr):
    r = tl.arange(0, 64)[:, None]
    c = tl.arange(0, 8)[None, :]
    value = tl.load(X + r * 8 + c, mask=r < 48, other=OTHER)
    tl.store(OUT + r * 8 + c, value)


rows = []
for dtype, other in [(torch.int8, -2), (torch.int32, -2), (torch.int8, 2)]:
    x = torch.zeros((64, 8), device='cuda', dtype=dtype)
    out = torch.full_like(x, 99)
    kernel[(1,)](x, out, other, num_warps=1)
    masked = out[48:].cpu()
    rows.append({'dtype': str(dtype), 'other': other, 'first_masked_row': masked[0].tolist(),
                 'mismatches': int((masked != other).sum()), 'elements': masked.numel()})
print(json.dumps({'id': 'Triton#8', 'results': rows}, indent=2))
