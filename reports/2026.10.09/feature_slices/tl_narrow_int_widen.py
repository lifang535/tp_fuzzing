"""TileLang 0.1.14 (#15): narrow integer arithmetic followed by a widening T.cast
loses the wraparound: ((int64_t)((int)(a + b))) keeps the C-promoted value.
Run with TileLang; prints wrong counts per op (the narrow store is correct)."""
import importlib.util
import re
import torch

CASES = [('add_i8', 'int8', 'A[i] + B[i]', lambda a, b: a + b),
         ('add_u8', 'uint8', 'A[i] + B[i]', lambda a, b: a + b),
         ('mul_i16', 'int16', 'A[i] * B[i]', lambda a, b: a * b),
         ('shl_u8', 'uint8', 'T.shift_left(A[i], T.cast(3, "uint8"))', lambda a, b: a << 3),
         ('sub_u16', 'uint16', 'A[i] - B[i]', lambda a, b: a - b)]
for name, dt, expr, ref in CASES:
    src = f'''
import tilelang
import tilelang.language as T

@tilelang.jit
def kern():
    @T.prim_func
    def main(A: T.Tensor((128,), "{dt}"), B: T.Tensor((128,), "{dt}"), O: T.Tensor((128,), "int64"), P: T.Tensor((128,), "{dt}")):
        with T.Kernel(1, threads=128) as bx:
            for i in T.Parallel(128):
                O[i] = T.cast({expr}, "int64")
                P[i] = {expr}
    return main
'''
    path = f'/tmp/narrow_{name}.py'
    open(path, 'w').write(src)
    spec = importlib.util.spec_from_file_location(f'narrow_{name}', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    bits = int(re.sub(r'\D', '', dt))
    signed = dt.startswith('int')
    lo, hi = (-(1 << (bits - 1)), (1 << (bits - 1)) - 1) if signed else (0, (1 << bits) - 1)
    g = torch.Generator().manual_seed(1)
    a = torch.randint(lo, hi + 1, (128,), generator=g)
    b = torch.randint(lo, hi + 1, (128,), generator=g)
    td = getattr(torch, dt)
    o = torch.zeros(128, dtype=torch.int64, device='cuda')
    p = torch.zeros(128, dtype=td, device='cuda')
    k = mod.kern()
    k(a.to(td).cuda(), b.to(td).cuda(), o, p)

    def wrap(v):
        v &= (1 << bits) - 1
        return v - (1 << bits) if signed and v >= 1 << (bits - 1) else v
    want = [wrap(ref(int(x), int(y))) for x, y in zip(a, b)]
    got_o = o.cpu().tolist()
    got_p = [wrap(int(v)) for v in p.cpu().to(torch.int64).tolist()]
    bad_o = sum(g_ != w for g_, w in zip(got_o, want))
    bad_p = sum(g_ != w for g_, w in zip(got_p, want))
    src_cuda = k.get_kernel_source()
    line = [l.strip() for l in src_cuda.splitlines() if 'O[' in l][:1]
    print(f'{name}: cast-to-int64 wrong {bad_o}/128, narrow store wrong {bad_p}/128 | {line}')
