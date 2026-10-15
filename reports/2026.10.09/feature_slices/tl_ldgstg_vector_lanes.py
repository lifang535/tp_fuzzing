"""TileLang 0.1.14: LDG/STG lowering ignores the lanes of a vector element type.

With the pass configuration tl.enable_lower_ldgstg (documented as "converts
Ramp-based global buffer load/store to ldg/stg intrinsics"), a global access
whose buffer element is a vector type (T.view(A, ..., "float32x2"), as used by
T.add2 and the other packed x2 operations) fails to compile:

    Check failed: tgt_dtype.lanes() * tgt_dtype.bits() == src_dtype.lanes() *
    src_dtype.bits() (32 vs. 64) : reinterpret expects source and target to
    have the same number of bits

src/cuda/transform/lower_ldg_stg.cc picks the access width from
buffer->dtype.bits(), the width of one lane: a float32x2 element (64 bits) gets
a 32-bit ldg32/stg32 whose uint32 result is then reinterpreted as float32x2.
The width must be bits() * lanes() (float32x4 fails with 32 vs. 128,
int32x2 like float32x2). float16x2 compiles only because 16-bit lanes are not
lowered at all. Without the configuration, or with only the predicated
variant (tl.enable_lower_ldgstg_predicated), every case compiles and is
correct. Upstream main computes the width the same way.
Found by the TileSmith math slice (T.sub2 on float32 under cfg_pass=ldgstg)
on 2026-10-11.
"""
import torch
import tilelang
import tilelang.language as T


def build(scalar, lanes, configs):
    vector = f'{scalar}x{lanes}'
    n = 128 // lanes

    @tilelang.jit(pass_configs=configs)
    def kern():
        @T.prim_func
        def main(A: T.Tensor((128,), scalar), O: T.Tensor((128,), scalar)):
            with T.Kernel(1, threads=n) as bx:
                a = T.view(A, (n,), vector)
                o = T.view(O, (n,), vector)
                for i in T.Parallel(n):
                    o[i] = a[i]
        return main
    return kern()


for scalar, lanes, torch_t in (('float32', 2, torch.float32), ('int32', 2, torch.int32),
                               ('float32', 4, torch.float32), ('float16', 2, torch.float16)):
    for name, configs in (('default', {}), ('ldgstg', {'tl.enable_lower_ldgstg': True}),
                          ('predicated only', {'tl.enable_lower_ldgstg_predicated': True})):
        label = f'{scalar}x{lanes} copy, {name}'
        try:
            kernel = build(scalar, lanes, configs)
            a = torch.arange(128, device='cuda').to(torch_t)
            o = torch.zeros(128, device='cuda', dtype=torch_t)
            kernel(a, o)
            torch.cuda.synchronize()
            print(f'{label:38s}', 'correct' if torch.equal(o, a) else 'WRONG')
        except Exception as exc:  # TVM InternalError
            line = next((l for l in str(exc).splitlines() if 'Check failed' in l), str(exc).splitlines()[-1])
            print(f'{label:38s}', 'compile error:', line.strip()[:120])
