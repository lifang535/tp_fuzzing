"""Triton 3.8 (#16): tl.fdiv on fp16/bf16 aborts in ConvertTritonGPUToLLVM
(LLVM ERROR: Unsupported bitwidth); ieee_rounding=True on fp16/bf16/fp64 fails
LLVM translation. Usage: python tr_fdiv_half.py <f16|bf16|f64|f32> <ieee 0|1>"""
import sys
import torch
import triton
import triton.language as tl


@triton.jit
def k_fdiv(X, Y, O, IEEE: tl.constexpr):
    offs = tl.arange(0, 128)
    x = tl.load(X + offs)
    y = tl.load(Y + offs)
    tl.store(O + offs, tl.fdiv(x, y, ieee_rounding=IEEE))


dt = {'f16': torch.float16, 'bf16': torch.bfloat16, 'f64': torch.float64, 'f32': torch.float32}[sys.argv[1]]
ieee = sys.argv[2] == '1'
x = torch.full((128,), 3.0, dtype=dt, device='cuda')
y = torch.full((128,), 7.0, dtype=dt, device='cuda')
o = torch.zeros(128, dtype=dt, device='cuda')
k_fdiv[(1,)](x, y, o, ieee)
print(o[:2])
