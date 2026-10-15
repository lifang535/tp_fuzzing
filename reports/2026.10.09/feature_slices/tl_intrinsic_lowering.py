"""TileLang 0.1.14 (#17-#19): vectorized T.bitwise_not -> nvcc `no operator ~`
(not_i32_256), T.nearbyint/T.fmod on float16 -> undefined hnearbyint/hfmod,
T.popcount on int32 -> Unresolved call tirx.popcount. Usage: python
tl_intrinsic_lowering.py [case ...]"""
import re
import sys
import tilelang
import tilelang.language as T

CASES = {
    'not_i32': ('int32', 'T.bitwise_not(xf[i])'),
    'nearbyint_f16': ('float16', 'T.nearbyint(xf[i])'),
    'fmod_bf16': ('bfloat16', 'T.fmod(xf[i], xf[i])'),
    'popcount_i32': ('int32', 'T.popcount(xf[i])'),
    'not_i32_256': ('int32', 'T.bitwise_not(xf[i])', 256),
    'not_u8_256': ('uint8', 'T.bitwise_not(xf[i])', 256),
}


def build(dt, expr, n=128):
    src = f'''
@tilelang.jit
def kern():
    @T.prim_func
    def main(X: T.Tensor(({n},), "{dt}"), O: T.Tensor(({n},), "{dt}")):
        with T.Kernel(1, threads=128) as bx:
            xf = T.alloc_fragment(({n},), "{dt}")
            T.copy(X, xf)
            for i in T.Parallel({n}):
                O[i] = T.cast({expr}, "{dt}")
    return main
'''
    import hashlib, importlib.util, os
    path = os.path.abspath('k_' + hashlib.sha1(src.encode()).hexdigest()[:12] + '.py')
    with open(path, 'w') as f:
        f.write('import tilelang\nimport tilelang.language as T\n' + src)
    spec = importlib.util.spec_from_file_location(os.path.basename(path)[:-3], path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.kern()


for name in sys.argv[1:] or CASES:
    try:
        build(*CASES[name])
        print(name, 'built')
    except Exception as e:
        text = str(e)
        errs = re.findall(r'(tvm_kernels\.cu\(\d+\): error: [^\n]*)', text)
        print(name, type(e).__name__, errs[:2] if errs else text[-400:])
        lines = re.findall(r'\n(\s+[^\n]*\n\s+\^)', text)
        if lines:
            print('   ', lines[0].strip()[:300])
