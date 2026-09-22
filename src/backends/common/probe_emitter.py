"""Emit explicit-stride kernels and paired/repeated execution checks.

TileLang uses flat physical buffers with explicit indexing: this exercises address
lowering, not the frontend's arbitrary-stride tensor descriptor support.
"""
import inspect
from src.workflow.emitter import probe_runtime
from src.workflow.emitter import _threshold_header

_HELPERS = '\n'.join(inspect.getsource(fn) for fn in (
    probe_runtime._probe_input, probe_runtime._probe_exact, probe_runtime._run_probe))


def _layout(rows, cols, layout):
    if layout == 'contiguous':
        sm, sn, offset = cols, 1, 0
    elif layout == 'transposed':
        sm, sn, offset = 1, rows, 0
    elif layout == 'strided':
        sm, sn, offset = 2 * cols + 3, 2, 0
    elif layout == 'broadcast_rows':
        sm, sn, offset = 0, 1, 0
    elif layout == 'broadcast_cols':
        sm, sn, offset = 1, 0, 0
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
    from src.backends import get_backend
    adapter = get_backend(backend)
    code = adapter.probe_code(k, a, b, outsize)
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
    test += [f'    print(\'TILESMITH_STAGE=probe_reference\', file=sys.stderr, flush=True)',
             f'    reference = {refs[kind]}', '    launches = []']
    for t in threads:
        test += adapter.probe_launch(k, t)
    if k.cache_cycle:
        test += ['    launches.append(launches[0])  # Request A after executing B']
    threshold = config.softmax_atol if kind == 'softmax' else config.reduce_rtol
    test += [f'    _run_probe(launches, [a, b] if {gemm!r} else [a], reference, {kind!r}, {k.repeat_count}, {threshold!r}, instantiate=True)']
    main = f"\n\nif __name__ == '__main__':\n    test_{k.name}()\n    print('ALL PASSED')\n"
    return (adapter.imports + '\nimport torch\nimport sys\n' + _threshold_header(config)
            + '\n' + _HELPERS + '\n\n' + code + '\n\n' + '\n'.join(test) + main)
