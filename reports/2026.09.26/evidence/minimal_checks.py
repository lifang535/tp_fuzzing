import json, traceback
import triton
import triton.language as tl
import tilelang
import tilelang.language as T
from triton.compiler import ASTSource
from triton.backends.compiler import GPUTarget

@triton.jit
def flip_default(A,B):
    i=tl.arange(0,128)
    x=tl.load(A+i)
    tl.store(B+i,tl.flip(x))

@triton.jit
def flip_explicit(A,B):
    i=tl.arange(0,128)
    x=tl.load(A+i)
    tl.store(B+i,tl.flip(x,dim=0))

@T.prim_func
def bool_output(A:T.Tensor((512,), 'float32'), B:T.Tensor((512,), 'bool')):
    with T.Kernel(1,threads=32):
        x=T.alloc_fragment((512,), 'float32')
        y=T.alloc_fragment((512,), 'bool')
        T.copy(A,x)
        for i in T.Parallel(512):
            y[i]=x[i]>0
        T.copy(y,B)

for name,fn in [
 ('flip_default',lambda:triton.compile(ASTSource(flip_default,signature={'A':'*fp32','B':'*fp32'}),target=GPUTarget('cuda',89,32))),
 ('flip_explicit',lambda:triton.compile(ASTSource(flip_explicit,signature={'A':'*fp32','B':'*fp32'}),target=GPUTarget('cuda',89,32))),
 ('bool_default',lambda:tilelang.compile(bool_output,target='cuda')),
 ('bool_no_vector',lambda:tilelang.compile(bool_output,target='cuda',pass_configs={'tirx.disable_vectorize':True}))]:
    try: fn(); print(json.dumps({'name':name,'result':'PASS'}),flush=True)
    except Exception as e:print(json.dumps({'name':name,'result':'FAIL','error':str(e),'trace':traceback.format_exc()}),flush=True)
