"""Standalone runtime helpers for directed probes (also testable on CPU)."""


def _probe_input(rows, cols, dtype, layout, pattern, device='cuda'):
    import torch
    if layout == 'contiguous':
        strides, offset = (cols, 1), 0
    elif layout == 'transposed':
        strides, offset = (1, rows), 0
    elif layout == 'strided':
        strides, offset = (2 * cols + 3, 2), 0
    elif layout == 'broadcast_rows':
        strides, offset = (0, 1), 0
    elif layout == 'broadcast_cols':
        strides, offset = (1, 0), 0
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
    # Initialize only unique storage locations for overlapping broadcast views.
    if layout == 'broadcast_rows':
        view[0].copy_(values[0])
    elif layout == 'broadcast_cols':
        view[:, 0].copy_(values[:, 0])
    else:
        view.copy_(values)
    return storage, view, strides, offset


def _byte_view(tensor):
    """Byte view of a tensor's logical contents, valid for any shape/strides.

    `.contiguous()` is not enough: torch's contiguity check skips size-1
    dimensions, so a broadcast or transposed input whose last extent is 1
    (strides (1, 0), (1, rows), ...) stays "contiguous" with stride(-1) != 1,
    and `view(dtype)` then raises "self.stride(-1) must be 1 to view Half as
    Byte" instead of letting the probe compare bits. Materialize into a fresh
    buffer when the strides make the view illegal: same logical order, same bit
    comparison, ordinary strides.
    """
    import torch
    if tensor.dim() == 0:
        tensor = tensor.reshape(1)
    if tensor.stride(-1) != 1:
        buffer = torch.empty(tensor.shape, dtype=tensor.dtype, device=tensor.device)
        buffer.copy_(tensor)
        tensor = buffer
    return tensor.view(torch.uint8)


def _probe_exact(actual, expected, label):
    import torch
    # Compare bits, including signed zero and NaN payloads. Also works for int32.
    if actual.shape != expected.shape or actual.dtype != expected.dtype:
        raise RuntimeError(f'WRONG RESULT: {label}: shape/dtype mismatch')
    if not torch.equal(_byte_view(actual), _byte_view(expected)):
        raise RuntimeError(f'WRONG RESULT: {label}: bits differ')


def _run_probe(launches, inputs, reference, kind, repeats, threshold, instantiate=False):
    import torch
    import sys
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
        # Instantiate lazily: A must execute before requesting B and then A again.
        # This exercises frontend/JIT cache lookup, not just existing callable reuse.
        if instantiate:
            # stderr-only stage marker so timeout/crash location inference
            # names the probe variant (see _failure_location).
            print(f'TILESMITH_STAGE=probe_instantiate_{variant}', file=sys.stderr, flush=True)
            launch = launch()
        print(f'TILESMITH_STAGE=probe_variant_{variant}', file=sys.stderr, flush=True)
        for repeat in range(repeats):
            output.fill_(23)
            output[guard:-guard].fill_(sentinel)
            if kind == 'copy':
                # Every unwritten byte differs, even when the expected value is NaN.
                # _byte_view also covers a degenerate-strided reference view,
                # where .contiguous().reshape(-1).view(uint8) raised before.
                output[guard:-guard].view(torch.uint8).copy_(
                    _byte_view(ref).reshape(-1).bitwise_not())
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
