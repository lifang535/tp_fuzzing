"""Triton 3.8: a scan inside a warp-specialized loop hits llvm_unreachable.

tl.range(..., warp_specialize=True) runs the NVGPUWarpSpecialization pass
(third_party/nvidia/hopper/lib/Transforms/WarpSpecialization). Its
TaskIdBackwardPropagation::propagateToParent walks from each op to the
enclosing loop and accepts scf.for, scf.if, tt.func and tt.reduce as
parents; anything else is "Other parent ops are not supported." and
llvm_unreachable. The combine region of tt.scan (tl.cumsum,
tl.associative_scan) is such a parent, so compilation aborts:

    Other parent ops are not supported.
    UNREACHABLE executed at .../WarpSpecialization/TaskIdPropagation.cpp:107!
    RuntimeError: PassManager::run failed

tl.sum and a custom tl.reduce in the same loop compile (tt.reduce is
accepted). Compiling for sm_80, sm_89 and sm_90 fails; sm_100 (the
Blackwell pipeline) compiles. Found by the TileSmith loop slice
(body=cumsum, loop=tl_range, ws=1) on 2026-10-11.
Usage: python tr_warp_specialize_scan_unreachable.py   (compile only, no GPU needed)
"""
import pathlib
import subprocess
import sys
import tempfile

TEMPLATE = '''import triton
import triton.language as tl
from triton.backends.compiler import GPUTarget


@triton.jit
def _add(a, b):
    return a + b


@triton.jit
def kernel(X, OUT, n: tl.constexpr):
    offs = tl.arange(0, 64)
    acc = tl.zeros((64,), tl.float32)
    for t in tl.range(0, n, warp_specialize=True):
        v = tl.load(X + t * 64 + offs)
        acc = acc + {body}
    tl.store(OUT + offs, acc)


src = triton.compiler.ASTSource(fn=kernel, signature={{'X': '*fp32', 'OUT': '*fp32', 'n': 'constexpr'}},
                                constexprs={{'n': 5}})
triton.compile(src, target=GPUTarget('cuda', {cap}, 32))
print('compiled')
'''

scratch = pathlib.Path(tempfile.mkdtemp())
for body in ('tl.cumsum(v, 0)', 'tl.associative_scan(v, 0, _add)', 'tl.sum(v, 0)', 'tl.reduce(v, 0, _add)'):
    for cap in (89, 90, 100):
        path = scratch / 'kernel.py'  # @triton.jit needs its source in a file
        path.write_text(TEMPLATE.format(body=body, cap=cap))
        r = subprocess.run([sys.executable, str(path)], capture_output=True, text=True)
        if 'compiled' in r.stdout:
            result = 'compiled'
        else:
            lines = [l for l in r.stderr.splitlines() if 'UNREACHABLE' in l or 'not supported' in l]
            result = ' / '.join(lines) or r.stderr.strip().splitlines()[-1]
        print(f'{body:34s} sm_{cap}: {result}')
