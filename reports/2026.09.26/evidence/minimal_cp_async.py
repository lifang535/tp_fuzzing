import tilelang
import tilelang.language as T
import json
@tilelang.jit
def matmul(stages):
    @T.prim_func
    def impl(A:T.Tensor((1,9157),'float16'),B:T.Tensor((9157,3709),'float16'),C:T.Tensor((1,3709),'float16')):
        with T.Kernel(1,58,threads=256) as (by,bx):
            As=T.alloc_shared((64,128),'float16')
            Bs=T.alloc_shared((128,64),'float16')
            acc=T.alloc_fragment((64,64),'float32')
            T.clear(acc)
            for k in T.Pipelined(72,num_stages=stages):
                T.copy(A[by*64,k*128],As)
                T.copy(B[k*128,bx*64],Bs)
                T.gemm(As,Bs,acc)
            T.copy(acc,C[by*64,bx*64])
    return impl
for stages in [4,0]:
    try:tilelang.compile(matmul.get_tir(stages),target='cuda');print(json.dumps({'stages':stages,'result':'PASS'}),flush=True)
    except Exception as e:print(json.dumps({'stages':stages,'result':'FAIL','error':str(e)}),flush=True)
