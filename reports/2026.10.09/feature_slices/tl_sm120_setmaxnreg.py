"""TileLang 0.1.14: a pipelined T.gemm compiled for sm_120 (consumer
Blackwell, RTX 50xx) is warp-specialized with register reallocation
(tl::warpgroup_reg_alloc/dealloc, PTX setmaxnreg), which sm_120 lacks; nvcc
stops at the templates' static_assert 'requires a target with CTA register
reconfiguration, such as sm_90a'. Needs nvcc >= 12.8, no GPU."""
import re
import tilelang
import tilelang.language as T
from tilelang import tvm
from tilelang.cuda.backend import tilelang_callback_cuda_compile

M = N = 64
K = 128


@T.prim_func
def main(A: T.Tensor((M, K), 'float16'), B: T.Tensor((K, N), 'float16'), C: T.Tensor((M, N), 'float32')):
    with T.Kernel(1, threads=128) as bx:
        As = T.alloc_shared((M, 32), 'float16')
        Bs = T.alloc_shared((32, N), 'float16')
        Cl = T.alloc_fragment((M, N), 'float32')
        T.clear(Cl)
        for k in T.Pipelined(K // 32, num_stages=2):
            T.copy(A[0, k * 32], As)
            T.copy(B[k * 32, 0], Bs)
            T.gemm(As, Bs, Cl)
        T.copy(Cl, C)


for arch in ('sm_89', 'sm_90a', 'sm_100a', 'sm_120'):
    target = {'kind': 'cuda', 'arch': arch}
    with tvm.target.Target(target):
        artifact = tilelang.lower(main, target=target)
    uses = sorted(set(re.findall(r'warpgroup_reg_\w+', artifact.kernel_source)))
    try:
        tilelang_callback_cuda_compile(artifact.kernel_source, artifact.target, {})
        print(f'{arch}: built; {uses}')
    except Exception as exc:
        found = re.search(r'static assertion failed with "([^"]+)"', str(exc))
        print(f'{arch}: nvcc failed; {uses}; {found.group(1) if found else str(exc)[-100:]}')
