"""Triton 3.8 (#21): tl.gather along axis 0 of a rank-3 tile whose index tile
is longer than the gathered axis aborts the compiler.

    python: lib/Dialect/TritonGPU/Transforms/OptimizeThreadLocality.cpp:220:
    setOptimizedGatherLayout(GatherOp, RewriterBase &):
    Assertion `GatherLoweringHelper(op).isWarpLocal()' failed.

The assertion kills the process (so the front end never reports an error).
Found independently by the layout slice on both servers at the 2026-10-10
15:22 check, reduced to source (2, 16, 4), index (4, 16, 4), axis 0,
num_warps=8 on 4090 (seed 43) and vGPU (seed 42); it also reproduces on a
laptop RTX 4060. Reduced from failed_slice_layout_25a8d38348f7454e and
failed_slice_layout_c8893af1ef12b289.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def kernel(X, IDX, OUT):
    src = (tl.arange(0, 2)[:, None, None] * 64 + tl.arange(0, 16)[None, :, None] * 4
           + tl.arange(0, 4)[None, None, :])
    out = (tl.arange(0, 4)[:, None, None] * 64 + tl.arange(0, 16)[None, :, None] * 4
           + tl.arange(0, 4)[None, None, :])
    x = tl.load(X + src)
    idx = tl.load(IDX + out)
    tl.store(OUT + out, tl.gather(x, idx, 0))


x = torch.arange(2 * 16 * 4, dtype=torch.int32, device='cuda').view(2, 16, 4)
idx = torch.randint(0, 2, (4, 16, 4), dtype=torch.int32, device='cuda')
out = torch.zeros(4, 16, 4, dtype=torch.int32, device='cuda')
print('compiling tl.gather(axis=0) with source (2, 16, 4) and index (4, 16, 4), num_warps=8 ...')
kernel[(1,)](x, idx, out, num_warps=8)
print('compiled; matches torch.gather:', torch.equal(out, torch.gather(x, 0, idx.long())))
