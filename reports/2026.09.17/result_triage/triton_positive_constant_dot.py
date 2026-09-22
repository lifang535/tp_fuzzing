import torch
import triton
import triton.language as tl
@triton.jit
def dot_control(A,C):
    r=tl.arange(0,16)
    a=tl.load(A+r[:,None]*16+r[None,:])
    b=tl.full((16,16),0.125,tl.float16)
    acc=tl.full((16,16),0.125,tl.float32)
    c=tl.dot(a,b,acc,input_precision='ieee')
    tl.store(C+r[:,None]*16+r[None,:],c)
torch.manual_seed(0)
a=(torch.randn((16,16),device='cuda')*0.01).half()
c=torch.full((16,16),float('nan'),device='cuda')
dot_control[(1,)](a,c,num_warps=4,enable_fp_fusion=False)
ref=a.float()@torch.full((16,16),0.125,device='cuda')+0.125
print('finite',torch.isfinite(c).all().item(),'maxdiff',(c-ref).abs().max().item(),flush=True)
torch.testing.assert_close(c,ref)
print('ALL PASSED')
