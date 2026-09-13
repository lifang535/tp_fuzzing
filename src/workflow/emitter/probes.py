"""Emit explicit-stride kernels and paired/repeated execution checks.

TileLang uses flat physical buffers with explicit indexing: this exercises address
lowering, not the frontend's arbitrary-stride tensor descriptor support.
"""
import inspect
from . import probe_runtime
from src.ir import ComputeKind, LoopKind

_HELPERS = '\n'.join(inspect.getsource(fn) for fn in (
    probe_runtime._probe_input, probe_runtime._probe_exact, probe_runtime._run_probe))


def _layout(rows, cols, layout):
    if layout == 'contiguous':
        sm, sn, offset = cols, 1, 0
    elif layout == 'transposed':
        sm, sn, offset = 1, rows, 0
    elif layout == 'strided':
        sm, sn, offset = 2 * cols + 3, 2, 0
    elif layout == 'offset':
        sm, sn, offset = cols + 3, 1, 7
    else:
        raise ValueError(f'Unknown layout: {layout}')
    size = offset + (rows - 1) * sm + (cols - 1) * sn + 17
    return sm, sn, offset, size


def emit_probe(k, backend, config):
    kind = k.compute_kind.value
    if kind not in ('copy', 'reduce_sum', 'reduce_max', 'reduce_min', 'softmax', 'argmax', 'gemm_argmax'):
        raise ValueError(f'Unsupported probe: {kind}')
    if not (1 <= k.N <= k.block_N and k.block_N & (k.block_N - 1) == 0):
        raise ValueError('Probe row width must fit a power-of-two tile')
    if k.threads not in (128, 256) or k.repeat_count < 2:
        raise ValueError('Probes require 128/256 threads and at least two repetitions')
    gemm = kind == 'gemm_argmax'
    if gemm and k.input_pattern not in ('integer', 'ties'):
        raise ValueError('GEMM argmax requires exact integer/tie inputs')
    a = _layout(k.M, k.K if gemm else k.N, k.input_layout)
    b = _layout(k.K, k.N, k.input_layout) if gemm else a
    outsize = k.M * k.N if kind in ('copy', 'softmax') else k.M
    code = _triton(k, a, b) if backend == 'triton' else _tilelang(k, a, b, outsize)
    threads = [k.threads, 384 - k.threads] if k.schedule_pair else [k.threads]
    refs = {'copy': 'A', 'reduce_sum': 'A.float().sum(dim=1)',
            'reduce_max': 'A.float().amax(dim=1)', 'reduce_min': 'A.float().amin(dim=1)',
            'softmax': 'torch.softmax(A.float(), dim=1)', 'argmax': 'A.argmax(dim=1)',
            'gemm_argmax': '(A.float() @ B.float()).argmax(dim=1)'}
    test = [f'def test_{k.name}():',
            f'    torch.manual_seed({config.input_seed})',
            # Small integer GEMM values are exactly represented even by TF32.
            f'    a = _probe_input({k.M}, {k.K if gemm else k.N}, torch.{k.dtype.value}, {k.input_layout!r}, {k.input_pattern!r})',
            '    A = a[1]']
    if gemm:
        test += [f'    b = _probe_input({k.K}, {k.N}, torch.{k.dtype.value}, {k.input_layout!r}, {k.input_pattern!r})', '    B = b[1]']
    else:
        test += ['    b = a']
    test += [f'    reference = {refs[kind]}', '    launches = []']
    for t in threads:
        if backend == 'triton':
            test += [f'    def launch_{t}(out):',
                     f'        {k.name}_probe[({(k.M + k.block_M - 1) // k.block_M},)](a[0], b[0], out, num_warps={t // 32})',
                     f'    launches.append(launch_{t})']
        else:
            test += [f'    compiled_{t} = make_{k.name}_probe({t})',
                     f'    launches.append(lambda out: compiled_{t}(a[0], b[0], out))']
    threshold = config.softmax_atol if kind == 'softmax' else config.reduce_rtol
    test += [f'    _run_probe(launches, [a, b] if {gemm!r} else [a], reference, {kind!r}, {k.repeat_count}, {threshold!r})']
    return _HELPERS + '\n\n' + code + '\n\n' + '\n'.join(test)


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


def _tilelang(k, a, b, outsize):
    m, n, kk, bm, bn, bk = k.M, k.N, k.K, k.block_M, k.block_N, k.block_K
    am, an, ao, asize = a
    bs0, bs1, bo, bsize = b
    kind, dtype = k.compute_kind.value, k.dtype.value
    integer = kind in ('argmax', 'gemm_argmax')
    outdtype = 'int32' if integer else dtype
    lines = ['@tilelang.jit', f'def make_{k.name}_probe(threads):',
             '    @T.prim_func',
             f'    def impl(A: T.Buffer(({asize},), "{dtype}"), B: T.Buffer(({bsize},), "{dtype}"), O: T.Buffer(({outsize + 32},), "{outdtype}")):',
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
            body += [f'As = T.alloc_shared(({bm}, {bk}), "{dtype}")',
                     f'Bs = T.alloc_shared(({bk}, {bn}), "{dtype}")', 'T.clear(acc)',
                     f'for ki in {loop}:',
                     f'    for i, j in T.Parallel({bm}, {bk}):',
                     f'        As[i, j] = T.if_then_else(by * {bm} + i < {m} and ki * {bk} + j < {kk}, A[{ao} + (by * {bm} + i) * {am} + (ki * {bk} + j) * {an}], T.cast(0, "{dtype}"))',
                     f'    for i, j in T.Parallel({bk}, {bn}):',
                     f'        Bs[i, j] = T.if_then_else(ki * {bk} + i < {kk} and j < {n}, B[{bo} + (ki * {bk} + i) * {bs0} + j * {bs1}], T.cast(0, "{dtype}"))',
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
