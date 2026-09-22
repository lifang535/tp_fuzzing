"""Small controls for cache keys, numerical conditioning and K tile constraints."""
from pathlib import Path
import tempfile
import json
import sys
from replay import ROOT, OUT, run

def save_run(label, code, cache=None):
    path = OUT/(label+'.py')
    path.write_text(code)
    return run(label, path, 90, cache=cache)

if sys.argv[1] == 'dtype':
    code = '''import tilelang
import tilelang.language as T
import torch
dtype = "DTYPE"
@tilelang.jit
def triage_cache_probe():
    @T.prim_func
    def impl(A: T.Buffer((32, 32), dtype), C: T.Buffer((32, 32), dtype)):
        with T.Kernel(1, threads=128):
            for i, j in T.Parallel(32, 32):
                C[i, j] = A[i, j]
    return impl
a = torch.ones((32, 32), device='cuda', dtype=getattr(torch, dtype))
c = torch.empty_like(a)
triage_cache_probe()(a, c)
torch.cuda.synchronize()
torch.testing.assert_close(a, c)
print('PASSED', dtype)
'''
    cache = tempfile.mkdtemp(prefix='triage_dtype_')
    save_run('dtype16_first', code.replace('DTYPE','float16'), cache)
    save_run('dtype32_same_cache', code.replace('DTYPE','float32'), cache)
    save_run('dtype32_fresh_cache', code.replace('DTYPE','float32'))
elif sys.argv[1] == 'block_k':
    for bk in (8,16):
        code = f'''import tilelang
import tilelang.language as T
import torch
@tilelang.jit
def control(threads):
    @T.prim_func
    def impl(A:T.Buffer((32,32),'float16'),B:T.Buffer((32,32),'float16'),C:T.Buffer((32,32),'float32')):
        with T.Kernel(1,threads=threads):
            aa=T.alloc_shared((32,{bk}),'float16')
            bb=T.alloc_shared(({bk},32),'float16')
            cc=T.alloc_fragment((32,32),'float32')
            T.clear(cc)
            for ki in T.serial({32//bk}):
                T.copy(A[0,ki*{bk}],aa)
                T.copy(B[ki*{bk},0],bb)
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
        print('block_K',{bk},'threads',threads,'repeat',repeat,'maxdiff',(c-ref).abs().max().item(),flush=True)
    open(__file__+str(threads)+'.cuda','w').write(kernel.get_kernel_source())
'''
        save_run('block_k_'+str(bk),code)
elif sys.argv[1] == 'numeric_exact':
    ident='119930b0d5c760cc'
    path=next(ROOT.glob('results/*/failed/*/*'+ident+'.py'))
    code=path.read_text().replace("scale=0.1", "scale=0.125")
    save_run(ident+'_binary_exact',code)
elif sys.argv[1] == 'block_k_exact':
    code = (OUT/'block_k_8.py').read_text().replace("a=torch.randn((32,32),device='cuda',dtype=torch.float16)", "a=torch.ones((32,32),device='cuda',dtype=torch.float16)").replace('b=torch.randn_like(a)', "b=torch.arange(32,device='cuda',dtype=torch.float16)[None,:].repeat(32,1)")
    code = code.replace("open(__file__+str(threads)+'.cuda','w').write(kernel.get_kernel_source())", "print('first_row_actual',c[0].tolist(),'first_row_reference',ref[0].tolist())")
    save_run('block_k_8_exact', code)
elif sys.argv[1] == 'extended':
    ident='25a869b78ca36590'
    source=next(ROOT.glob('results/*/failed/*/*'+ident+'.py'))
    hook = '''
_base_check=extended_check
def extended_check(actual, expected, label, matmul=False):
    print('OBS',label,'actual_range',actual.float().min().item(),actual.float().max().item(),'expected_range',expected.float().min().item(),expected.float().max().item(),flush=True)
    try: _base_check(actual,expected,label,matmul)
    except RuntimeError as e: print('OBSERVED_FAILURE',str(e),flush=True)
PROGRAM['runtime_cases']=[[0,1]]
'''
    code=source.read_text().replace('for seed in (input_seed, input_seed + 1):','for seed in (input_seed,):').replace('run_extended(PROGRAM, prepare_extended, 0, 3)','run_extended(PROGRAM, prepare_extended, 0, 1)').replace("if __name__ == '__main__':",hook+"\nif __name__ == '__main__':")
    save_run(ident+'_observed',code)
elif sys.argv[1] == 'extended_zero_loops':
    import ast
    ident='25a869b78ca36590'
    source=next(ROOT.glob('results/*/failed/*/*'+ident+'.py'))
    tree=ast.parse(source.read_text())
    removed=0
    for node in tree.body:
        if isinstance(node,ast.FunctionDef) and node.name.startswith('extended_') and node.name[-1:].isdigit():
            removed+=sum(isinstance(n,(ast.For,ast.While)) for n in node.body)
            node.body=[n for n in node.body if not isinstance(n,(ast.For,ast.While))]
    assert removed==8, removed
    code=ast.unparse(tree).replace("if __name__ == '__main__':", "PROGRAM['runtime_cases']=[[0,1]]\nif __name__ == '__main__':")
    save_run(ident+'_zero_loops_removed',code)
elif sys.argv[1] == 'nondeterminism_barrier':
    ident='28f7a4c7a8eda0b4'
    source=next(ROOT.glob('results/*/failed/*/*'+ident+'.py'))
    code=source.read_text().replace('        T.copy(arg1, v1_shared)', '        T.sync_threads()\n        T.copy(arg1, v1_shared)\n        T.sync_threads()').replace('ref, 3, 0.1, relative=True)', 'ref, 20, 0.1, relative=True)')
    save_run(ident+'_explicit_barrier',code)
elif sys.argv[1] == 'extended_more':
    run('extended_more_observations', OUT/'extended_more_observations.py',90)
elif sys.argv[1] == 'triton_dot':
    code='''import torch
import triton
import triton.language as tl
@triton.jit
def dot_control(A,C):
    r=tl.arange(0,16)
    a=tl.load(A+r[:,None]*16+r[None,:])
    b=tl.full((16,16),-0.125,tl.float16)
    acc=tl.full((16,16),0.125,tl.float32)
    c=tl.dot(a,b,acc,input_precision='ieee')
    tl.store(C+r[:,None]*16+r[None,:],c)
torch.manual_seed(0)
a=(torch.randn((16,16),device='cuda')*0.01).half()
c=torch.full((16,16),float('nan'),device='cuda')
dot_control[(1,)](a,c,num_warps=4,enable_fp_fusion=False)
ref=a.float()@torch.full((16,16),-0.125,device='cuda')+0.125
print('finite',torch.isfinite(c).all().item(),'maxdiff',(c-ref).abs().max().item(),flush=True)
torch.testing.assert_close(c,ref)
print('ALL PASSED')
'''
    save_run('triton_constant_dot',code)
elif sys.argv[1] == 'triton_dot_controls':
    code=(OUT/'triton_constant_dot.py').read_text()
    save_run('triton_positive_constant_dot',code.replace('-0.125','0.125'))
    code=code.replace('def dot_control(A,C):','def dot_control(A,B,C):').replace('b=tl.full((16,16),-0.125,tl.float16)','b=tl.load(B+r[:,None]*16+r[None,:])').replace("c=torch.full((16,16),float('nan'),device='cuda')", "b=torch.full((16,16),-0.125,device='cuda',dtype=torch.float16)\nc=torch.full((16,16),float('nan'),device='cuda')").replace('dot_control[(1,)](a,c,','dot_control[(1,)](a,b,c,')
    save_run('triton_memory_operand_dot',code)
