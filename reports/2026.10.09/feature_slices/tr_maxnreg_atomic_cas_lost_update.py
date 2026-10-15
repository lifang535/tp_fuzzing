"""Triton bf16 atomic_add (sem=acquire) loses updates when ptxas gets .maxnreg 32.

Found by TileSmith b1eded9f (4090, wrong_result:05e47ffef0, 2026-10-14 02:33 check).
bf16 atomic_add on sm_89 lowers to a 32-bit CAS loop plus fences. The PTX with and
without maxnreg differs only in the `.maxnreg 32` directive (0 spills either way), yet
the maxnreg=32 build drops two updates to slot 5 (got 10, expected 12), deterministically.
Passes with: no maxnreg, maxnreg>=40, num_warps>=2, or sem='relaxed'.
Toolchain: Triton 3.x bundled ptxas 12.9 (V12.9.86), RTX 4060 Laptop (sm_89).
"""
import torch
import triton
import triton.language as tl


@triton.jit
def kernel(X, Y, OUT):
    pid = tl.program_id(0)
    offs = pid * 1024 + tl.arange(0, 1024)
    v = tl.load(X + offs)
    w = tl.load(Y + offs)
    idx = ((offs) // 64) % 16
    tl.atomic_add(OUT + idx, v.to(tl.bfloat16), mask=(offs % 3) != 1, sem='acquire', scope='gpu')


def inputs(salt):
    g = torch.Generator().manual_seed(salt)
    return torch.randint(-1, 2, (1024,), generator=g).double()


def main():
    x, y = inputs(1), inputs(2)
    flat = torch.arange(1024)
    keep = (flat % 3) != 1
    expected = torch.full((16,), 7.0, dtype=torch.float64).index_add_(0, ((flat // 64) % 16)[keep], x[keep])
    X, Y = x.to(torch.bfloat16).cuda(), y.to(torch.bfloat16).cuda()
    for options in ({'num_warps': 1}, {'num_warps': 1, 'maxnreg': 32}):
        storage = torch.full((16 + 128,), 7.0, device='cuda', dtype=torch.bfloat16)
        out = storage[64:80]
        kernel[(1,)](X, Y, out, **options)
        torch.cuda.synchronize()
        bad = (out.double().cpu() != expected).nonzero().flatten().tolist()
        print(options, 'OK' if not bad else f'WRONG at {bad}: got {out[bad].tolist()} expected {expected[bad].tolist()}')


if __name__ == '__main__':
    main()
