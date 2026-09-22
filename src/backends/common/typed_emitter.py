"""Lower typed regions, including real block-private global scratch traffic."""
import inspect
from dataclasses import asdict, replace
from src.ir.layout import matrix_layout
from src.workflow.emitter.typed_region_runtime import _typed_region_reference


def TypedLowering(program, backend, name, suffix='', options=None):
    """Return the registered backend's typed lowering."""
    from src.backends import get_backend
    return get_backend(backend).typed_lowering(program, name, suffix, options)


def _prepare_typed_launch(kernel, inputs, layouts):
    """Allocate per-block scratch and check every block's guards after a launch."""
    import torch
    scratch = [torch.full((blocks, elements + 32), 23, device=inputs[0].device, dtype=getattr(torch, dtype))
               for blocks, elements, dtype in layouts]
    def launch(output):
        # Poison contents each time: no dependency on a prior kernel invocation.
        for storage in scratch:
            storage.fill_(23)
            storage[:, 16:-16].fill_(float('nan'))
        kernel(*inputs, *(s.reshape(-1) for s in scratch), output)
        for storage in scratch:
            if not torch.all(storage[:, :16] == 23) or not torch.all(storage[:, -16:] == 23):
                raise RuntimeError('WRONG RESULT: scratch canary modified')
    return launch


def emit_typed_region(program, backend, config):
    from src.workflow.emitter import _threshold_header
    from .region import _region_layouts, _layout_sweep_pairs
    from .region_emitter import region_variant_pairs, _variant_kinds
    from src.workflow.emitter.region_checks import (_region_input, _region_input_storage, _region_equal,
                                                    _region_check, _ulp_jitter, _reference_stable,
                                                    _run_region, _run_layout_case)
    p, e = program.spec, program.execution
    gemm = program.body.operations[0].kind == 'gemm'
    # Div programs amplify magnitudes (clamped denominators chain by ~1e3 per
    # op), so absolute error is meaningless; use the relative mode for them.
    # int8 GEMMs (defensive: typed int8 bodies are not generated) compare
    # int32 outputs exactly with one-unit absolute slack.
    int8 = p.dtype.value == 'int8'
    output_dtype = 'int32' if int8 else p.dtype.value
    relative = False if int8 else gemm or any(op.kind == 'div' for op in program.all_operations())
    # Logical (all-contiguous) kernels declare 2D buffers and receive the
    # strided views; physical kernels receive flat storage.
    physical, _, _ = _region_layouts(program)
    launch_inputs = 'a_storage, b_storage' if physical else 'A, B'
    # Layout sweep (RC1/RC3): physical programs also run each input case
    # against an alternate layout pair. The relabel helper is only embedded
    # when it is used, keeping legacy emission byte-identical.
    sweep = e.layout_sweep and physical
    from src.backends import get_backend
    adapter = get_backend(backend)
    parts = [adapter.imports,
             'import torch', 'import sys', _threshold_header(config)]
    checks = (matrix_layout, _region_input, _region_input_storage, _region_equal,
              _region_check, _ulp_jitter, _reference_stable, _run_region)
    if sweep:
        checks += (_run_layout_case,)
    parts += [inspect.getsource(fn) for fn in checks + (_typed_region_reference, _prepare_typed_launch)]
    # fp64 copy of the typed interpreter, used by the oracle trust gate
    # (see _reference_stable in region_checks).
    from src.workflow.emitter.runtime import double_reference_source
    parts.append(double_reference_source(_typed_region_reference))
    tm, tn = (p.M+p.block_M-1)//p.block_M, (p.N+p.block_N-1)//p.block_N
    variant_pairs = region_variant_pairs(program, backend, config)
    variants = [variant for variant, _ in variant_pairs]
    variant_options = [options for _, options in variant_pairs]
    tolerance = 1.0 if int8 else (config.region_rtol_fp16 if p.dtype.value == 'float16' else config.region_rtol_fp32) if relative else config.elemwise_atol
    kinds = _variant_kinds(variants, variant_options, p)
    # Legacy programs (thread pair only) keep the plain historical call text.
    kind_arg = f', variant_kinds={kinds!r}' if any(k != 'schedule' for k in kinds) else ''
    if sweep:
        layout_pairs = _layout_sweep_pairs(program)
        # Layouts are baked into the kernel source, so every pair compiles its
        # own kernel set (global prepare index p * len(variants) + i).
        blocks = []
        for p_index, (la, lb) in enumerate(layout_pairs):
            runner_lines = []
            for i, variant in enumerate(variants):
                global_index = p_index * len(variants) + i
                pair_variant = replace(variant, execution=replace(
                    variant.execution, input_layout_a=la, input_layout_b=lb))
                name = f'typed_kernel_{global_index}'
                lower = TypedLowering(pair_variant, backend, name, f'_v{global_index}',
                                      options=variant_options[i])
                parts.append(lower.emit())
                scratch_layouts = [(tm*tn, t.dimensions(p)[0]*t.dimensions(p)[1], t.dtype) for t in lower.slots.values()]
                argv = ', '.join(lower.context + ['C'])
                prepare = adapter.typed_prepare(pair_variant, name, argv, options=variant_options[i])
                parts.append(f'def prepare_{global_index}(A, B):\n{prepare}\n    return _prepare_typed_launch(kernel, (A, B), {scratch_layouts!r})')
                runner_lines.append(f'''def run_variant_{p_index}_{i}():
    print('TILESMITH_STAGE=prepare_{p_index}_{i}', file=sys.stderr, flush=True)
    return prepare_{global_index}({launch_inputs})''')
            runner_code = '\n'.join('        ' + line for line in '\n'.join(runner_lines).split('\n'))
            # Concurrent prepares interleave markers, so _run_region re-prints
            # the failing variant's marker (this exact p_index form) before
            # re-raising a compile error.
            marker_arg = f', prepare_markers={[f"prepare_{p_index}_{i}" for i in range(len(variants))]!r}'
            blocks.append(f'''        print('TILESMITH_STAGE=layout:{la}/{lb}', file=sys.stderr, flush=True)
        a_storage, A = _region_input_storage(({p.M}, {p.K if gemm else p.N}), torch.{p.dtype.value}, {e.input_pattern!r}, {program.input_scale}, {la!r})
        b_storage, B = _region_input_storage(({p.K}, {p.N}), torch.{p.dtype.value}, {e.input_pattern!r}, {program.input_scale}, {lb!r})
{runner_code}
        print('TILESMITH_STAGE=reference', file=sys.stderr, flush=True)
        ref = _typed_region_reference(A, B, {asdict(program.body)!r}, {p.block_M}, {p.block_N}, {output_dtype!r}, {[fn.to_dict() for fn in program.functions]!r})
        _run_layout_case([{', '.join(f'run_variant_{p_index}_{i}' for i in range(len(variants)))}], (a_storage, b_storage), ref, {e.repeat_count}, {tolerance}, relative={relative!r}, layout_a={la!r}, layout_b={lb!r}, alternate={p_index != 0}{kind_arg}{marker_arg}, reference_verify=lambda: _typed_region_reference_double(A, B, {asdict(program.body)!r}, {p.block_M}, {p.block_N}, {output_dtype!r}, {[fn.to_dict() for fn in program.functions]!r}), reference_jitter=lambda: _typed_region_reference(_ulp_jitter(A), _ulp_jitter(B), {asdict(program.body)!r}, {p.block_M}, {p.block_N}, {output_dtype!r}, {[fn.to_dict() for fn in program.functions]!r}))''')
        run_block = '\n'.join(blocks)
        body = f'''def test_kernel_0():
    for input_case in range({e.input_seed_count}):
        torch.manual_seed({config.input_seed} + input_case)
{run_block}'''
    else:
        factories = []
        for i, variant in enumerate(variants):
            name = f'typed_kernel_{i}'
            lower = TypedLowering(variant, backend, name, f'_v{i}', options=variant_options[i])
            parts.append(lower.emit())
            scratch_layouts = [(tm*tn, t.dimensions(p)[0]*t.dimensions(p)[1], t.dtype) for t in lower.slots.values()]
            argv = ', '.join(lower.context + ['C'])
            prepare = adapter.typed_prepare(variant, name, argv, options=variant_options[i])
            parts.append(f'def prepare_{i}(A, B):\n{prepare}\n    return _prepare_typed_launch(kernel, (A, B), {scratch_layouts!r})')
            factories.append(f'''def run_variant_{i}():
    print('TILESMITH_STAGE=prepare_{i}', file=sys.stderr, flush=True)
    return prepare_{i}({launch_inputs})''')
        # The runner closures live inside test_kernel_0 so they can capture
        # the input_case locals (a_storage/b_storage or A/B) like the old
        # lambdas did. They sit at the for-body indentation level (8 spaces).
        factory_code = '\n'.join('        ' + line for line in '\n'.join(factories).split('\n'))
        body = f'''def test_kernel_0():
    for input_case in range({e.input_seed_count}):
        torch.manual_seed({config.input_seed} + input_case)
        a_storage, A = _region_input_storage(({p.M}, {p.K if gemm else p.N}), torch.{p.dtype.value}, {e.input_pattern!r}, {program.input_scale}, {e.input_layout_a!r})
        b_storage, B = _region_input_storage(({p.K}, {p.N}), torch.{p.dtype.value}, {e.input_pattern!r}, {program.input_scale}, {e.input_layout_b!r})
{factory_code}
        print('TILESMITH_STAGE=reference', file=sys.stderr, flush=True)
        ref = _typed_region_reference(A, B, {asdict(program.body)!r}, {p.block_M}, {p.block_N}, {output_dtype!r}, {[fn.to_dict() for fn in program.functions]!r})
        _run_region([{', '.join(f'run_variant_{i}' for i in range(len(variants)))}], (a_storage, b_storage), ref, {e.repeat_count}, {tolerance}, relative={relative!r}{kind_arg}, reference_verify=lambda: _typed_region_reference_double(A, B, {asdict(program.body)!r}, {p.block_M}, {p.block_N}, {output_dtype!r}, {[fn.to_dict() for fn in program.functions]!r}), reference_jitter=lambda: _typed_region_reference(_ulp_jitter(A), _ulp_jitter(B), {asdict(program.body)!r}, {p.block_M}, {p.block_N}, {output_dtype!r}, {[fn.to_dict() for fn in program.functions]!r}))'''
    parts.append(f'''
{body}

if __name__ == '__main__':
    test_kernel_0()
    print('ALL PASSED')
''')
    return '\n\n'.join(parts)
