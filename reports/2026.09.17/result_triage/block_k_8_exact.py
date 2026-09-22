import tilelang
import tilelang.language as T
import torch
@tilelang.jit
def control(threads):
    @T.prim_func
    def impl(A:T.Buffer((32,32),'float16'),B:T.Buffer((32,32),'float16'),C:T.Buffer((32,32),'float32')):
        with T.Kernel(1,threads=threads):
            aa=T.alloc_shared((32,8),'float16')
            bb=T.alloc_shared((8,32),'float16')
            cc=T.alloc_fragment((32,32),'float32')
            T.clear(cc)
            for ki in T.serial(4):
                T.copy(A[0,ki*8],aa)
                T.copy(B[ki*8,0],bb)
                T.gemm(aa,bb,cc)
            T.copy(cc,C)
    return impl
torch.manual_seed(0)
a=torch.ones((32,32),device='cuda',dtype=torch.float16)
b=torch.arange(32,device='cuda',dtype=torch.float16)[None,:].repeat(32,1)
ref=a.float()@b.float()
for threads in (128,256):
    c=torch.empty_like(ref)
    kernel=control(threads)
    for repeat in range(3):
        kernel(a,b,c)
        torch.cuda.synchronize()
        print('block_K',8,'threads',threads,'repeat',repeat,'maxdiff',(c-ref).abs().max().item(),flush=True)
    print('first_row_actual',c[0].tolist(),'first_row_reference',ref[0].tolist())
