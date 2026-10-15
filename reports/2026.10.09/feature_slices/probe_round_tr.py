import itertools, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
from rounding_ref import round_to
from values import interesting, STORE
NAMES = ('f64', 'f32', 'f16', 'bf16', 'f8e4', 'f8e5')
CASES = [{'id': f'{s}->{d}/rtz{r}/b{b}', 'src': s, 'dst': d, 'rtz': r, 'block': b}
         for s, d in itertools.product(NAMES, NAMES) if s != d for r in (0, 1) for b in (128, 1024)]
_K = None
def kernel():
    global _K
    if _K is None:
        import triton, triton.language as tl
        @triton.jit
        def k(X, O, DST: tl.constexpr, RTZ: tl.constexpr, BLOCK: tl.constexpr):
            offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
            x = tl.load(X + offs)
            if RTZ:
                y = x.to(DST, fp_downcast_rounding='rtz')
            else:
                y = x.to(DST)
            tl.store(O + offs, y)
        _K = k
    return _K
def run(case):
    import triton.language as tl
    TT = {'f64': tl.float64, 'f32': tl.float32, 'f16': tl.float16, 'bf16': tl.bfloat16, 'f8e4': tl.float8e4nv, 'f8e5': tl.float8e5}
    src, dst, rtz, block = case['src'], case['dst'], case['rtz'], case['block']
    v = interesting(src, dst, 4096, seed=7)
    n = (len(v) // block) * block
    v = v[:n]
    X = v.to(STORE[src]).cuda()
    O = torch.empty(n, dtype=STORE[dst], device='cuda')
    kernel()[(n // block,)](X, O, TT[dst], rtz, block)
    torch.cuda.synchronize()
    ref = round_to(v, dst, 'rtz' if rtz else 'rtne')
    got = O.double().cpu()
    same = (got == ref) | (torch.isnan(got) & torch.isnan(ref))
    bad = (~same).nonzero().flatten()
    if len(bad) == 0:
        return 'ok'
    i = bad[0]
    return f'WRONG {len(bad)}/{n} e.g. x={v[i].item()!r} got={got[i].item()!r} ref={ref[i].item()!r}'
