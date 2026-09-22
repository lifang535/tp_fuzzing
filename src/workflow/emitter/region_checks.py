"""CPU-testable checks embedded verbatim into standalone region reproducers."""
from .runtime import _finite_compare
from src.ir.layout import matrix_layout


def _region_input(shape, dtype, pattern, scale, device='cuda'):
    import torch
    if dtype == torch.int8:
        # int8 has no randn; small signed integers keep every product and
        # partial sum far inside the int32 accumulator range.
        if pattern == 'alternating':
            phase = torch.randint(0, 4, (), device=device)
            value = ((torch.arange(shape[0] * shape[1], device=device) + phase) % 4 - 2).reshape(shape).to(torch.int8)
        else:
            value = torch.randint(-8, 8, shape, device=device).to(torch.int8)
        return (value * scale).to(torch.int8)
    if pattern == 'normal':
        value = torch.randn(shape, dtype=dtype, device=device)
    elif pattern == 'integer':
        value = torch.randint(-3, 4, shape, device=device).to(dtype)
    elif pattern == 'alternating':
        # Signed, exactly representable values exercise cancellation and zeros.
        phase = torch.randint(0, 4, (), device=device)
        value = ((torch.arange(shape[0] * shape[1], device=device) + phase) % 4 - 2).reshape(shape).to(dtype)
    else:
        raise ValueError('Unknown region input pattern: ' + pattern)
    return value * scale


def _region_input_storage(shape, dtype, pattern, scale, layout, device='cuda'):
    """Initialize unique addresses; retain padding for input-integrity checks."""
    import torch
    rows, cols = shape
    stride_row, stride_col, offset, size = matrix_layout(rows, cols, layout)
    storage = torch.full((size,), 19, dtype=dtype, device=device)
    view = storage.as_strided(shape, (stride_row, stride_col), offset)
    # Broadcast views overlap: never copy an entire matrix into those views.
    if layout == 'broadcast_rows':
        view[0].copy_(_region_input((1, cols), dtype, pattern, scale, device)[0])
    elif layout == 'broadcast_cols':
        view[:, 0].copy_(_region_input((rows, 1), dtype, pattern, scale, device)[:, 0])
    else:
        view.copy_(_region_input(shape, dtype, pattern, scale, device))
    return storage, view


def _region_equal(actual, expected, label):
    import torch
    if not torch.equal(actual.contiguous().view(torch.uint8), expected.contiguous().view(torch.uint8)):
        raise RuntimeError('WRONG RESULT: ' + label)


def _region_check(actual, expected, relative, tolerance, label):
    try:
        maximum, _, normalized = _finite_compare(actual, expected)
    except RuntimeError as error:
        raise RuntimeError(f'WRONG RESULT: {label}: {error}') from error
    error = normalized if relative else maximum
    if error > tolerance:
        raise RuntimeError(f'WRONG RESULT: {label}: error={error}, tolerance={tolerance}')


def _ulp_jitter(x):
    """One-ulp perturbation of a storage tensor.

    The smallest difference any two implementations can possibly disagree by:
    the kernel and the reference read the same storage values, but arithmetic
    order and precision differ. A reference that changes far beyond tolerance
    under this perturbation cannot be reproduced by any kernel, so the
    numeric check is meaningless for it (see _reference_stable).
    """
    import torch
    if not x.dtype.is_floating_point:
        return x.clone()
    y = x.clone()
    y[y == 0] = torch.finfo(x.dtype).tiny
    # nextafter has no Half overload in its CUDA kernel, so perturb on CPU
    # (all floating dtypes supported there) and move back to the input device.
    y_cpu = y.cpu()
    return torch.nextafter(y_cpu, y_cpu.sign()).to(x.dtype).to(x.device)


def _reference_stable(reference, double_reference, jitter_reference=None):
    """Oracle trust gate for the fp32 reference interpreter.

    The numeric check is only meaningful when the reference reproduces its own
    double-precision copy: chaotic recurrences amplify arithmetic-order noise
    by orders of magnitude, so a disagreement means *no* implementation could
    pass the check and the program must not be reported as a wrong_result.
    The fixed 1e-2 self-error threshold sits between benign fp32 noise
    (<= 1e-4 relative, including the one-ulp fp16 output-rounding boundary)
    and chaotic divergence (>= 1) by several orders of magnitude.

    jitter_reference, when given, is the reference recomputed on one-ulp
    perturbed inputs: it catches the milder, order-sensitive amplifiers whose
    own rounding stays stable but whose output depends on reduction order —
    again unpassable by any real kernel. Either disagreement fails the gate.
    """
    alternates = [double_reference]
    if jitter_reference is not None:
        alternates.append(jitter_reference)
    for alternate in alternates:
        try:
            _, _, self_relative = _finite_compare(reference, alternate)
        except RuntimeError:
            self_relative = float('inf')
        if self_relative > 1e-2:
            return False
    return True


def _run_layout_case(launches, inputs, reference, repeats, tolerance, relative,
                     layout_a, layout_b, alternate, **kwargs):
    """One input layout pair inside the sweep: the primary pair reports
    failures with their historical labels, an alternate pair is re-raised as
    `layout invariance` so layout-dependent indexing bugs get their own
    root-cause class (layout_mismatch)."""
    if not alternate:
        return _run_region(launches, inputs, reference, repeats, tolerance, relative=relative, **kwargs)
    try:
        _run_region(launches, inputs, reference, repeats, tolerance, relative=relative, **kwargs)
    except RuntimeError as error:
        raise RuntimeError(f'layout invariance: {layout_a}/{layout_b}: {error}') from error


def _run_region(launches, inputs, reference, repeats, tolerance, relative=False,
                variant_kinds=None, references=None, prepare_markers=None,
                reference_verify=None, reference_jitter=None):
    """Compile every variant concurrently, then execute them in order.

    The caller computes reference before any target kernel can corrupt inputs.
    Tile geometry stays fixed across schedules, preserving tile-local semantics.
    The schedule sweep variants (threads, num_stages, loop_kind) all share one
    reference because the reference interpreter is schedule-independent;
    variant_kinds names each variant in cross-variant invariance diagnostics
    and references (if given) supplies one reference per variant.

    prepare_markers carries the stderr marker each variant's own prepare
    closure prints (`prepare_{p_index}_{i}` under a layout sweep); a compile
    failure re-prints the failing variant's marker before re-raising so the
    last-marker location inference still names the crashing variant even
    though the compiles themselves ran concurrently.

    reference_verify, when given, is a zero-argument closure returning the
    double-precision copy of the reference; reference_jitter likewise returns
    the reference recomputed on one-ulp-jittered inputs. A structured-reference
    failure is first checked against _reference_stable: when the fp32 reference
    disagrees with either copy, the failure is oracle noise (a chaotic or
    reduction-order-sensitive program), re-raised as ORACLE UNSTABLE instead of
    a wrong_result.
    """
    import torch
    import sys
    # Cold kernel compilation is the dominant harness cost (nvcc subprocesses,
    # CPU-bound; the GPU is idle until the first launch), so every variant is
    # compiled up front on at most this many threads instead of paying each
    # compile serially right before its execution. Execution below keeps the
    # historical serial order. This literal is self-contained: no module-level
    # name travels into the standalone reproducer. Kept in sync with
    # EXTENDED_COMPILE_THREADS in backends/common/knobs.py.
    compile_threads = 8
    if repeats < 1 or not launches:
        raise ValueError('Region execution requires a launch and positive repeats')
    if variant_kinds is None:
        variant_kinds = ['schedule'] * len(launches)
    if references is None:
        references = [reference] * len(launches)
    if prepare_markers is None:
        prepare_markers = [f'prepare_{variant}' for variant in range(len(launches))]
    if len(variant_kinds) != len(launches) or len(references) != len(launches) or len(prepare_markers) != len(launches):
        raise ValueError('Region variant metadata must match the launch count')
    guard = 16
    output = torch.full((reference.numel() + 2 * guard,), 23,
                        dtype=reference.dtype, device=reference.device)
    actual = output[guard:-guard].view_as(reference)
    snapshots = [tensor.clone() for tensor in inputs]
    baseline = None
    # Integer outputs (int8 GEMM -> int32 C) cannot hold NaN poison.
    floating = reference.dtype.is_floating_point
    prepared = [None] * len(launches)
    errors = [None] * len(launches)
    if len(launches) == 1:
        try:
            prepared[0] = launches[0]()
        except BaseException as error:
            errors[0] = error
    else:
        import threading
        def compile_variant(start):
            # Strided worker: with more variants than threads, one thread
            # compiles every compile_threads-th variant so no variant is
            # left for the (serialized) execution phase.
            for variant in range(start, len(launches), compile_threads):
                try:
                    prepared[variant] = launches[variant]()
                except BaseException as error:
                    errors[variant] = error
        threads = [threading.Thread(target=compile_variant, args=(start,))
                   for start in range(min(len(launches), compile_threads))]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
    for variant, launch in enumerate(prepared):
        if errors[variant] is not None:
            # Re-print the marker now: concurrent compiles interleave their
            # prepare markers, so the last one printed may not be the variant
            # that failed. Raising in variant order keeps the first-failure
            # contract of the historical serial loop.
            print(f'TILESMITH_STAGE={prepare_markers[variant]}', file=sys.stderr, flush=True)
            raise errors[variant]
        # stderr-only marker so a crash traceback names the failing variant
        # (location-aware root causes); the 'ALL PASSED' stdout line is untouched.
        print(f'TILESMITH_STAGE=execute_variant_{variant}', file=sys.stderr, flush=True)
        expected = references[variant]
        reference_nan = torch.isnan(expected) if floating else None
        for repeat in range(repeats):
            output.fill_(23)
            if floating:
                # Missing writes must fail even if the expected value itself is NaN.
                actual.fill_(float('nan'))
                actual.masked_fill_(reference_nan, 0.0)
            launch(actual)
            if reference.is_cuda:
                torch.cuda.synchronize(reference.device)
            if not torch.all(output[:guard] == 23) or not torch.all(output[-guard:] == 23):
                raise RuntimeError(f'WRONG RESULT: output canary modified (schedule={variant})')
            for tensor, snapshot in zip(inputs, snapshots):
                _region_equal(tensor, snapshot, 'input storage modified')
            if repeat == 0:
                if baseline is None:
                    baseline = actual.clone()
                    repeat_baseline = baseline
                else:
                    _region_check(actual, baseline, relative, tolerance,
                                  f'{variant_kinds[variant]} invariance')
                    repeat_baseline = actual.clone()
            else:
                _region_equal(actual, repeat_baseline, 'repeat determinism')
            try:
                _region_check(actual, expected, relative, tolerance, 'structured reference')
            except RuntimeError:
                # Suppress the chain: the chained WRONG RESULT text would leak
                # into stderr and skew root-cause classification.
                if reference_verify is not None and not _reference_stable(
                        expected, reference_verify(),
                        reference_jitter() if reference_jitter is not None else None):
                    raise RuntimeError('ORACLE UNSTABLE: reference disagrees with its '
                                       'fp64 or one-ulp-jittered copy; numeric check skipped') from None
                raise
