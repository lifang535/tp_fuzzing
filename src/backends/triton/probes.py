from src.ir import LoopKind

def _triton(k, a, b):
    m, n, kk, bm, bn, bk = k.M, k.N, k.K, k.block_M, k.block_N, k.block_K
    am, an, ao, _ = a
    bmstride, bnstride, bo, _ = b
    kind = k.compute_kind.value
    lines = ['@triton.jit', f'def {k.name}_probe(A, B, O):',
             f'    rows = tl.program_id(0) * {bm} + tl.arange(0, {bm})',
             f'    cols = tl.arange(0, {bn})']
    if kind == 'gemm_argmax':
        loop = f'tl.range(0, {((kk + bk - 1) // bk)}, num_stages={k.num_stages})' if k.loop_kind == LoopKind.PIPELINED else f'range({(kk + bk - 1) // bk})'
        lines += [f'    acc = tl.full(({bm}, {bn}), 0, tl.float32)',
                  f'    for ki in {loop}:',
                  f'        ks = ki * {bk} + tl.arange(0, {bk})',
                  f'        x = tl.load(A + {ao} + rows[:, None] * {am} + ks[None, :] * {an}, (rows[:, None] < {m}) & (ks[None, :] < {kk}), other=0)',
                  ]
        if k.input_layout == 'transposed':
            # Match the paper's dot(A, trans(B)) -> argmax composition explicitly.
            lines += [f'        y = tl.load(B + {bo} + cols[:, None] * {bnstride} + ks[None, :] * {bmstride}, (cols[:, None] < {n}) & (ks[None, :] < {kk}), other=0)',
                      '        acc += tl.dot(x, tl.trans(y))']
        else:
            lines += [f'        y = tl.load(B + {bo} + ks[:, None] * {bmstride} + cols[None, :] * {bnstride}, (ks[:, None] < {kk}) & (cols[None, :] < {n}), other=0)',
                      '        acc += tl.dot(x, y)']
        lines += [f'    acc = tl.where(cols[None, :] < {n}, acc, -float("inf"))']
    else:
        neutral = 'float("inf")' if kind == 'reduce_min' else '-float("inf")' if kind in ('reduce_max', 'argmax', 'softmax') else '0'
        lines += [f'    x = tl.load(A + {ao} + rows[:, None] * {am} + cols[None, :] * {an}, (rows[:, None] < {m}) & (cols[None, :] < {n}), other={neutral})',
                  '    acc = x.to(tl.float32)']
    if kind in ('argmax', 'gemm_argmax'):
        lines += ['    result = tl.argmax(acc, axis=1, tie_break_left=True)']
    elif kind.startswith('reduce_'):
        op = kind.removeprefix('reduce_')
        lines += [f'    result = tl.{op}(acc, axis=1)']
    elif kind == 'softmax':
        lines += ['    exp = tl.exp(acc - tl.max(acc, axis=1)[:, None])',
                  '    result = exp / tl.sum(exp, axis=1)[:, None]']
    else:
        lines += ['    result = x']
    if kind in ('copy', 'softmax'):
        lines += [f'    tl.store(O + 16 + rows[:, None] * {n} + cols[None, :], result, (rows[:, None] < {m}) & (cols[None, :] < {n}))']
    else:
        lines += [f'    tl.store(O + 16 + rows, result, rows < {m})']
    return '\n'.join(lines)
