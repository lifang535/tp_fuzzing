from src.ir import LoopKind

def _tilelang(k, a, b, outsize):
    m, n, kk, bm, bn, bk = k.M, k.N, k.K, k.block_M, k.block_N, k.block_K
    am, an, ao, asize = a
    bs0, bs1, bo, bsize = b
    kind, dtype = k.compute_kind.value, k.dtype.value
    integer = kind in ('argmax', 'gemm_argmax')
    outdtype = 'int32' if integer else dtype
    # Bind dtype at module scope instead of inlining it into the impl source
    # so programs that differ only in dtype share one frontend-cache key and
    # hit the dtype_mismatch bug class (see src/backends/tilelang/typed.py).
    lines = [f'dtype = "{dtype}"', f'outdtype = "{outdtype}"',
             '@tilelang.jit', f'def make_{k.name}_probe(threads):',
             '    @T.prim_func',
             f'    def impl(A: T.Buffer(({asize},), dtype), B: T.Buffer(({bsize},), dtype), O: T.Buffer(({outsize + 32},), outdtype)):',
             f'        with T.Kernel({(m + bm - 1) // bm}, threads=threads) as by:']
    body = []
    if kind == 'copy':
        body += [f'for i, j in T.Parallel({bm}, {bn}):',
                 f'    if by * {bm} + i < {m} and j < {n}:',
                 f'        O[16 + (by * {bm} + i) * {n} + j] = A[{ao} + (by * {bm} + i) * {am} + j * {an}]']
    else:
        body += [f'acc = T.alloc_fragment(({bm}, {bn}), "float32")',
                 f'stat = T.alloc_fragment(({bm},), "float32")']
        if kind == 'gemm_argmax':
            loop = f'T.Pipelined({(kk + bk - 1) // bk}, num_stages={k.num_stages})' if k.loop_kind == LoopKind.PIPELINED else f'T.serial({(kk + bk - 1) // bk})'
            body += [f'As = T.alloc_shared(({bm}, {bk}), dtype)',
                     f'Bs = T.alloc_shared(({bk}, {bn}), dtype)', 'T.clear(acc)',
                     f'for ki in {loop}:',
                     f'    for i, j in T.Parallel({bm}, {bk}):',
                     f'        As[i, j] = T.if_then_else(by * {bm} + i < {m} and ki * {bk} + j < {kk}, A[{ao} + (by * {bm} + i) * {am} + (ki * {bk} + j) * {an}], T.cast(0, dtype))',
                     f'    for i, j in T.Parallel({bk}, {bn}):',
                     f'        Bs[i, j] = T.if_then_else(ki * {bk} + i < {kk} and j < {n}, B[{bo} + (ki * {bk} + i) * {bs0} + j * {bs1}], T.cast(0, dtype))',
                     '    T.gemm(As, Bs, acc)',
                     f'for i, j in T.Parallel({bm}, {bn}):',
                     f'    acc[i, j] = T.if_then_else(j < {n}, acc[i, j], -T.infinity("float32"))']
        else:
            neutral = 'T.infinity("float32")' if kind == 'reduce_min' else '-T.infinity("float32")' if kind in ('reduce_max', 'argmax', 'softmax') else 'T.float32(0)'
            body += [f'for i, j in T.Parallel({bm}, {bn}):',
                     f'    acc[i, j] = T.if_then_else(by * {bm} + i < {m} and j < {n}, T.cast(A[{ao} + (by * {bm} + i) * {am} + j * {an}], "float32"), {neutral})']
        if integer:
            body += [f'indices = T.alloc_fragment(({bm}, {bn}), "int32")',
                     f'index = T.alloc_fragment(({bm},), "int32")',
                     'T.reduce_max(acc, stat, dim=1)',
                     f'for i, j in T.Parallel({bm}, {bn}):',
                     f'    indices[i, j] = T.if_then_else(j < {n} and acc[i, j] == stat[i], j, 2147483647)',
                     'T.reduce_min(indices, index, dim=1)']
        elif kind == 'softmax':
            body += ['T.reduce_max(acc, stat, dim=1)',
                     f'for i, j in T.Parallel({bm}, {bn}):', '    acc[i, j] = T.exp(acc[i, j] - stat[i])',
                     'T.reduce_sum(acc, stat, dim=1)',
                     f'for i, j in T.Parallel({bm}, {bn}):',
                     f'    if by * {bm} + i < {m} and j < {n}:',
                     f'        O[16 + (by * {bm} + i) * {n} + j] = acc[i, j] / stat[i]']
        else:
            body += [f'T.{kind}(acc, stat, dim=1)']
        if kind != 'softmax':
            body += [f'for i in T.Parallel({bm}):', f'    if by * {bm} + i < {m}:',
                     f'        O[16 + by * {bm} + i] = {"index" if integer else "stat"}[i]']
    lines += ['            ' + line for line in body]
    lines += ['    return impl']
    return '\n'.join(lines)
