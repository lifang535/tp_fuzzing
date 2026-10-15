"""TileLang 0.1.14: tl.config_index_bitwidth=64 rewrites stored values, not
only indices.

The pass configuration tl.config_index_bitwidth ("Bitwidth for configuration
indices") runs ConfigIndexBitwidthRewriter (src/transform/config_index_bitwidth.cc).
To force 64-bit indices it sets is_enabled_ while visiting a whole BufferStore
or BufferLoad, and with is_enabled_ set it turns every signed integer IntImm
and every Cast to a signed integer type narrower than 64 bits into int64. The
stored value is visited too, so data arithmetic changes type:

1. Wrong results: an explicit narrowing cast in a stored value is dropped.
   O[i] = T.cast(T.cast(X[i], "int8"), "int32") returns X[i] unchanged
   (x = -640 gives -640 instead of -128); int16 likewise. uint8 is not
   affected (the rewriter only matches signed types).
2. Compile failure: int8/int16 element arithmetic in a vectorized loop
   (o = -b, o = b * T.cast(3, "int8")) becomes int64x16 / int64x8 vector
   arithmetic, which CodeGenTileLangCUDA::PrintType cannot print:
   "Cannot convert type int64x16 to CUDA type".

Every case is correct with the default configuration. Upstream main has the
same rewriter. Found by the TileSmith memory slice under cfg_pass=index64 on
2026-10-11.
"""
import torch
import tilelang
import tilelang.language as T

INDEX64 = {'tl.config_index_bitwidth': 64}


def narrowing(configs, narrow):
    @tilelang.jit(pass_configs=configs)
    def kern():
        @T.prim_func
        def main(X: T.Tensor((128,), "int32"), O: T.Tensor((128,), "int32")):
            with T.Kernel(1, threads=128) as bx:
                for i in T.Parallel(128):
                    O[i] = T.cast(T.cast(X[i], narrow), "int32")
        return main
    return kern()


def negate(configs, dt):
    @tilelang.jit(pass_configs=configs)
    def kern():
        @T.prim_func
        def main(X: T.Tensor((16, 32), dt), O: T.Tensor((16, 32), dt)):
            with T.Kernel(1, threads=32) as bx:
                b = T.alloc_fragment((16, 32), dt)
                T.copy(X, b)
                o = T.alloc_fragment((16, 32), dt)
                for i, j in T.Parallel(16, 32):
                    o[i, j] = -b[i, j]
                T.copy(o, O)
        return main
    return kern()


x = torch.arange(-640, 640, 10, dtype=torch.int32, device='cuda')
for narrow, torch_t in (('int8', torch.int8), ('int16', torch.int16), ('uint8', torch.uint8)):
    for name, configs in (('default', {}), ('index64', INDEX64)):
        o = torch.zeros(128, dtype=torch.int32, device='cuda')
        narrowing(configs, narrow)(x * 64 if narrow == 'int16' else x, o)
        want = (x * 64 if narrow == 'int16' else x).to(torch_t).to(torch.int32)
        bad = (o != want).nonzero()
        verdict = 'correct' if not len(bad) else (
            f'WRONG at {len(bad)}/128, e.g. x={int((x * 64 if narrow == "int16" else x)[bad[0]])}: '
            f'got {int(o[bad[0]])}, expected {int(want[bad[0]])}')
        print(f'cast through {narrow:5s} {name:8s} {verdict}')
for dt, torch_t in (('int8', torch.int8), ('int16', torch.int16)):
    for name, configs in (('default', {}), ('index64', INDEX64)):
        try:
            data = torch.randint(-20, 20, (16, 32), dtype=torch.int64).to(torch_t).cuda()
            out = torch.zeros(16, 32, dtype=torch_t, device='cuda')
            negate(configs, dt)(data, out)
            verdict = 'correct' if torch.equal(out, -data) else 'WRONG'
        except Exception as exc:  # TVM InternalError at codegen
            verdict = 'compile error: ' + next(l for l in str(exc).splitlines() if 'Cannot convert' in l).strip()[:80]
        print(f'negate {dt:5s} vector {name:8s} {verdict}')
