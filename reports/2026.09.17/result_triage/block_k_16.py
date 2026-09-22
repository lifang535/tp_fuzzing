import tilelang
import tilelang.language as T
import torch
@tilelang.jit
def control(threads):
    @T.prim_func
    def impl(A:T.Buffer((32,32),'float16'),B:T.Buffer((32,32),'float16'),C:T.Buffer((32,32),'float32')):
        with T.Kernel(1,threads=threads):
            aa=T.alloc_shared((32,16),'float16')
            bb=T.alloc_shared((16,32),'float16')
            cc=T.alloc_fragment((32,32),'float32')
            T.clear(cc)
            for ki in T.serial(2):
                T.copy(A[0,ki*16],aa)
                T.copy(B[ki*16,0],bb)
                T.gemm(aa,bb,cc)
            T.copy(cc,C)
    return impl
torch.manual_seed(0)
a=torch.randn((32,32),device='cuda',dtype=torch.float16)
b=torch.randn_like(a)
ref=a.float()@b.float()
for threads in (128,256):
    c=torch.empty_like(ref)
    kernel=control(threads)
    for repeat in range(3):
        kernel(a,b,c)
        torch.cuda.synchronize()
        print('block_K',16,'threads',threads,'repeat',repeat,'maxdiff',(c-ref).abs().max().item(),flush=True)
    open(__file__+str(threads)+'.cuda','w').write(kernel.get_kernel_source())
