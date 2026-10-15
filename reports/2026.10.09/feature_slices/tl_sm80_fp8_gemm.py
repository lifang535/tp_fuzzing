"""TileLang fp8 (e4m3) T.gemm compiled for target sm_80 (which has no fp8 MMA), run on an sm_89 GPU.
Usage: python tl_sm80_fp8_gemm.py [arch] [dtype]"""
import sys, re, torch, tilelang
import tilelang.language as T

arch = sys.argv[1] if len(sys.argv) > 1 else 'sm_80'
dt = sys.argv[2] if len(sys.argv) > 2 else 'float8_e4m3'

@tilelang.jit(target={'kind': 'cuda', 'arch': arch})
def gemm():
    @T.prim_func
    def main(A: T.Tensor((64, 32), dt), B: T.Tensor((32, 64), dt), O: T.Tensor((64, 64), "float32")):
        with T.Kernel(1, 1, threads=128) as (bx, by):
            As = T.alloc_shared((64, 32), dt)
            Bs = T.alloc_shared((32, 64), dt)
            C = T.alloc_fragment((64, 64), "float32")
            T.clear(C)
            T.copy(A, As)
            T.copy(B, Bs)
            T.gemm(As, Bs, C)
            T.copy(C, O)
    return main

k = gemm()
src = k.get_kernel_source()
print(f'target {arch} {dt}: MMA used:', sorted(set(re.findall(r'SM\d+_\d+x\d+x\d+\w*|mma\.sync[\w.]*', src)))[:4])
tdt = {'float8_e4m3': torch.float8_e4m3fn, 'float16': torch.float16}[dt]
a = torch.randint(-2, 3, (64, 32), device='cuda').float().to(tdt)
b = torch.randint(-2, 3, (32, 64), device='cuda').float().to(tdt)
o = torch.empty(64, 64, device='cuda')
try:
    k(a, b, o); torch.cuda.synchronize()
    print('  ran, max error', (o - a.float() @ b.float()).abs().max().item())
except Exception as e:
    print('  run failed:', str(e).splitlines()[0])
