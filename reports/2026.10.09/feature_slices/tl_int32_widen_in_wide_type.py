"""TileLang 0.1.14 (#15, int32 variant): integer arithmetic under a widening cast is
evaluated in the wide type. For T.cast(x * 3, "int64") with x int32 the generated CUDA
widens both operands first (make_longlong2) and multiplies in int64, so the product
never wraps at 32 bits: x=-1843722600 gives -5531167800 instead of -1236200504.
int16/int8 multiply inside short2/char fields and widen afterwards, which is correct.
Caveat: this is signed int32 overflow, which TVM's arithmetic may treat as undefined, so
upstream can call the program invalid; the unsigned uint8/uint16 cases of #15
(tl_narrow_int_widen.py), where wraparound is well defined, carry the claim.
Found by the math slice at the 2026-10-10 18:19 check."""
import importlib.util
import re
import torch

SRC = '''
import tilelang
import tilelang.language as T

@tilelang.jit
def kern(dt: str, n: int, threads: int, c: int):
    @T.prim_func
    def main(A: T.Tensor((n,), dt), O: T.Tensor((n,), "int64"), P: T.Tensor((n,), dt)):
        with T.Kernel(1, threads=threads) as bx:
            s = T.alloc_shared((n,), dt)
            T.copy(A, s)
            for i in T.Parallel(n):
                O[i] = T.cast(s[i] * T.cast(c, dt), "int64")
                P[i] = s[i] * T.cast(c, dt)
    return main
'''
open('/tmp/widen_sign_kern.py', 'w').write(SRC)
spec = importlib.util.spec_from_file_location('widen_sign_kern', '/tmp/widen_sign_kern.py')
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

TD = {'int32': torch.int32, 'int16': torch.int16, 'int8': torch.int8, 'int64': torch.int64}


def wrap(v, bits):
    v &= (1 << bits) - 1
    return v - (1 << bits) if v >= 1 << (bits - 1) else v


for dt, bits, n, threads in (('int32', 32, 512, 32), ('int32', 32, 512, 128), ('int16', 16, 512, 128),
                             ('int8', 8, 512, 128)):
    g = torch.Generator().manual_seed(3)
    lo, hi = -(1 << (bits - 1)), (1 << (bits - 1)) - 1
    a = torch.randint(lo, hi, (n,), generator=g, dtype=torch.int64)
    c = 3
    want_wide = [wrap(int(v) * c, bits) for v in a]          # TIR: mul is dt, then widen
    A = a.to(TD[dt]).cuda()
    o = torch.zeros(n, dtype=torch.int64, device='cuda')
    p = torch.zeros(n, dtype=TD[dt], device='cuda')
    k = mod.kern(dt, n, threads, c)
    k(A, o, p)
    torch.cuda.synchronize()
    got_wide = o.cpu().tolist()
    got_narrow = [wrap(int(v), bits) for v in p.cpu().to(torch.int64).tolist()]
    bad_w = [i for i in range(n) if got_wide[i] != want_wide[i]]
    bad_n = [i for i in range(n) if got_narrow[i] != want_wide[i]]
    zero_ext = sum(1 for i in bad_w if got_wide[i] == (want_wide[i] & ((1 << bits) - 1)))
    print(f'{dt} threads={threads}: widened wrong {len(bad_w)}/{n} (of which zero-extended: {zero_ext}), '
          f'narrow store wrong {len(bad_n)}/{n}')
    if bad_w:
        i = bad_w[0]
        print(f'   e.g. a={int(a[i])} * {c}: got {got_wide[i]}, want {want_wide[i]}, diff {got_wide[i] - want_wide[i]}')
        src = k.get_kernel_source()
        line = [l.strip() for l in src.splitlines() if 'O[' in l][:1]
        print('   ', line)
