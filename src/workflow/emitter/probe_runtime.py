"""Standalone runtime helpers for directed probes (also testable on CPU)."""


def _probe_input(rows, cols, dtype, layout, pattern, device='cuda'):
    import torch
    if layout == 'contiguous':
        strides, offset = (cols, 1), 0
    elif layout == 'transposed':
        strides, offset = (1, rows), 0
    elif layout == 'strided':
        strides, offset = (2 * cols + 3, 2), 0
    elif layout == 'offset':
        strides, offset = (cols + 3, 1), 7
    else:
        raise ValueError(f'Unknown layout: {layout}')
    size = offset + (rows - 1) * strides[0] + (cols - 1) * strides[1] + 1
    storage = torch.full((size + 16,), 19, device=device, dtype=dtype)
    view = storage.as_strided((rows, cols), strides, offset)
    if pattern == 'normal':
        values = torch.randn((rows, cols), device=device, dtype=dtype)
    elif pattern == 'integer':
        values = torch.randint(-2, 3, (rows, cols), device=device).to(dtype)
    elif pattern == 'negative':
        values = -torch.randint(1, 5, (rows, cols), device=device).to(dtype)
    elif pattern == 'indexed':
        values = ((torch.arange(rows * cols, device=device) % 1021) - 510).reshape(rows, cols).to(dtype)
    elif pattern in ('ties', 'zeros'):
        values = torch.ones((rows, cols), device=device, dtype=dtype) if pattern == 'ties' else torch.zeros((rows, cols), device=device, dtype=dtype)
    elif pattern in ('special', 'subnormal'):
        tiny = torch.finfo(dtype).tiny
        eps = torch.finfo(dtype).eps
        numbers = ([0., -0., float('nan'), float('inf'), -float('inf'), 1., -1.]
                   if pattern == 'special' else [tiny * eps, -tiny * eps, tiny / 2, -tiny / 2, tiny, -tiny, 0., -0.])
        values = torch.tensor(numbers, dtype=dtype, device=device)
        values = values.repeat((rows * cols + len(numbers) - 1) // len(numbers))[:rows * cols].reshape(rows, cols)
    else:
        raise ValueError(f'Unknown pattern: {pattern}')
    view.copy_(values)
    return storage, view, strides, offset


def _probe_exact(actual, expected, label):
    import torch
    # Compare bits, including signed zero and NaN payloads. Also works for int32.
    if actual.shape != expected.shape or actual.dtype != expected.dtype:
        raise RuntimeError(f'WRONG RESULT: {label}: shape/dtype mismatch')
    if not torch.equal(actual.contiguous().view(torch.uint8), expected.contiguous().view(torch.uint8)):
        raise RuntimeError(f'WRONG RESULT: {label}: bits differ')


def _run_probe(launches, inputs, reference, kind, repeats, threshold):
    import torch
    if repeats < 2:
        raise ValueError('Directed probes require at least two executions')
    integer = kind in ('argmax', 'gemm_argmax')
    dtype = torch.int32 if integer else inputs[0][1].dtype
    ref = reference.to(dtype)
    guard = 16
    # Deliberately initialize with an impossible index/NaN to detect missing stores.
    sentinel = -2147483647 if integer else float('nan')
    output = torch.full((ref.numel() + 2 * guard,), 23, dtype=dtype, device=ref.device)
    snapshots = [item[0].clone() for item in inputs]
    baseline = None
    for variant, launch in enumerate(launches):
        for repeat in range(repeats):
            output.fill_(23)
            output[guard:-guard].fill_(sentinel)
            if kind == 'copy':
                # Every unwritten byte differs, even when the expected value is NaN.
                output[guard:-guard].view(torch.uint8).copy_(
                    ref.contiguous().reshape(-1).view(torch.uint8).bitwise_not())
            launch(output)
            torch.cuda.synchronize()
            if not torch.all(output[:guard] == 23) or not torch.all(output[-guard:] == 23):
                raise RuntimeError(f'WRONG RESULT: output canary modified (schedule={variant}, repeat={repeat})')
            for item, snapshot in zip(inputs, snapshots):
                _probe_exact(item[0], snapshot, 'input storage modified')
            actual = output[guard:-guard].reshape(ref.shape).clone()
            if integer or kind == 'copy':
                _probe_exact(actual, ref, 'reference')
            else:
                # Local elementwise tolerance, rather than global mean normalization.
                torch.testing.assert_close(actual, ref, rtol=threshold, atol=threshold,
                                           equal_nan=False, msg=lambda msg: 'WRONG RESULT: reference: ' + msg)
            if repeat == 0:
                repeat_baseline = actual
                if baseline is None:
                    baseline = actual
                elif integer or kind == 'copy':
                    _probe_exact(actual, baseline, 'schedule invariance')
                else:
                    torch.testing.assert_close(actual, baseline, rtol=threshold, atol=threshold,
                                               msg=lambda msg: 'WRONG RESULT: schedule invariance: ' + msg)
            else:
                _probe_exact(actual, repeat_baseline, 'repeat determinism')
