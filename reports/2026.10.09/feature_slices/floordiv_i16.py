"""Nested T.floordiv on int16 vectors: codegen 'Let cannot bind the same var to two different values'."""
import torch, tilelang, tilelang.language as T

def build(dtype, threads, nested):
    @tilelang.jit
    def kern():
        @T.prim_func
        def main(X: T.Tensor((64,), dtype), O: T.Tensor((64,), dtype)):
            with T.Kernel(1, threads=threads) as bx:
                for i in T.Parallel(64):
                    if nested:
                        O[i] = T.floordiv(T.floordiv(X[i], T.cast(-2, dtype)), T.cast(4, dtype))
                    else:
                        O[i] = T.floordiv(X[i], T.cast(-2, dtype))
        return main
    return kern()

for dtype in ('int16', 'int8', 'int32', 'int64'):
    for threads in (32, 64):
        for nested in (False, True):
            try:
                k = build(dtype, threads, nested)
                x = torch.arange(-32, 32, dtype=getattr(torch, dtype), device='cuda')
                o = torch.zeros_like(x)
                k(x, o); torch.cuda.synchronize()
                ref = torch.div(torch.div(x.cpu(), -2, rounding_mode='floor'), 4, rounding_mode='floor') if nested else torch.div(x.cpu(), -2, rounding_mode='floor')
                print(dtype, threads, 'nested' if nested else 'single', 'ok' if torch.equal(o.cpu(), ref) else 'WRONG', flush=True)
            except Exception as e:
                print(dtype, threads, 'nested' if nested else 'single', 'ERR', str(e).strip().splitlines()[-1][:110], flush=True)
