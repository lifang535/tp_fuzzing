"""Recursive lowering of structured IR to real TileLang/Triton control flow."""
import inspect
from dataclasses import asdict, replace
from src.backends import get_backend
from src.backends.common.region import _region_layouts, _layout_sweep_pairs
from src.ir.layout import matrix_layout
from src.workflow.emitter.region_runtime import _region_reference
from src.workflow.emitter.runtime import double_reference_source

_REFERENCE_SOURCE = inspect.getsource(_region_reference)
# fp64 copy of the native interpreter, used by the oracle trust gate
# (see _reference_stable): only embedded where the fp32 copy is.
_REFERENCE_SOURCE_DOUBLE = double_reference_source(_region_reference)
from src.workflow.emitter.region_checks import _ulp_jitter
_ULP_JITTER_SOURCE = inspect.getsource(_ulp_jitter)


def emit_region(program, backend, config):
    get_backend(backend).validate_program(program)
    if program.typed:
        from .typed_emitter import emit_typed_region
        return emit_typed_region(program, backend, config)
    if program.body.operations[0].kind == 'probe':
        from .probe_emitter import emit_probe
        return emit_probe(program.spec, backend, config)
    if program.execution is not None:
        return _emit_checked_region(program, backend, config)
    from src.workflow.emitter import _threshold_header
    p = program.spec
    gemm = program.body.operations[0].kind == 'gemm'
    # int8 GEMMs emit int32 C from int8 inputs: the reference accumulates in
    # int32 too, and the comparison is exact with a one-unit absolute slack.
    int8 = p.dtype.value == 'int8'
    output_dtype = 'int32' if int8 else p.dtype.value
    relative = False if int8 else gemm or any(op.kind == 'div' for op in program.all_operations())
    adapter = get_backend(backend)
    imports = adapter.imports
    code, initialization, launch = adapter.native_region(program)
    if int8:
        tolerance_text = '1.0'
    else:
        tolerance_text = f'{config.region_rtol_fp16 if p.dtype.value == "float16" else config.region_rtol_fp32} if {relative!r} else {config.elemwise_atol}'
    test = f'''
def test_kernel_0():
    torch.manual_seed({config.input_seed})
    A = torch.randn(({p.M}, {p.K if gemm else p.N}), device='cuda', dtype=torch.{p.dtype.value}) * {program.input_scale}
    B = torch.randn(({p.K}, {p.N}), device='cuda', dtype=torch.{p.dtype.value}) * {program.input_scale}
    C = torch.empty(({p.M}, {p.N}), device='cuda', dtype=torch.{output_dtype})
    print('TILESMITH_STAGE=prepare', file=sys.stderr, flush=True)
    {initialization}
    print('TILESMITH_STAGE=execute', file=sys.stderr, flush=True)
    {launch}
    torch.cuda.synchronize()
    print('TILESMITH_STAGE=reference', file=sys.stderr, flush=True)
    ref = _region_reference(A, B, {asdict(program.body)!r}, {p.block_M}, {p.block_N}, {output_dtype!r}, {[asdict(fn) for fn in program.functions]!r})
    max_diff, _, relative_error = _finite_compare(C, ref)
    error = relative_error if {relative!r} else max_diff
    tolerance = {tolerance_text}
    if error > tolerance:
        # Oracle trust gate: the fp32 reference must reproduce its own fp64
        # copy and its one-ulp-jittered-input copy; otherwise no kernel could
        # pass this check and the failure is oracle noise, not a bug.
        self_relative = 0.0
        for alternate in (_region_reference_double(A, B, {asdict(program.body)!r}, {p.block_M}, {p.block_N}, {output_dtype!r}, {[asdict(fn) for fn in program.functions]!r}),
                          _region_reference(_ulp_jitter(A), _ulp_jitter(B), {asdict(program.body)!r}, {p.block_M}, {p.block_N}, {output_dtype!r}, {[asdict(fn) for fn in program.functions]!r})):
            try:
                _, _, candidate = _finite_compare(ref, alternate)
            except RuntimeError:
                candidate = float('inf')
            self_relative = max(self_relative, candidate)
        if self_relative > 1e-2:
            raise RuntimeError(f'ORACLE UNSTABLE: reference self-relative error={{self_relative:.3g}}; numeric check skipped')
        raise RuntimeError(f'WRONG RESULT: structured reference error={{error}}')

if __name__ == '__main__':
    test_kernel_0()
    print('ALL PASSED')
'''
    return (imports + '\nimport torch\nimport sys\n' + _threshold_header(config) + '\n'
            + _REFERENCE_SOURCE + '\n' + _REFERENCE_SOURCE_DOUBLE + '\n' + _ULP_JITTER_SOURCE + '\n' + code + test)


def region_variants(program, backend, config=None):
    """Threads, num_stages and loop_kind change the schedule, never the tile
    geometry: block shape is part of region semantics, so all variants share
    one reference."""
    return get_backend(backend).region_variants(program, config)


def region_variant_pairs(program, backend, config=None):
    """(variant, options) pairs; options carry compilation knobs that do not
    change the kernel source (pass_configs, enable_fp_fusion, swizzle)."""
    return get_backend(backend)._region_variant_pairs(program, config)


def _variant_kinds(variants, options, spec):
    """Cross-variant invariance label per sweep entry.

    Compilation-knob variants reuse the base program's spec, so their labels
    must come from options before the spec-diff fallback. The thread pair
    keeps the historical 'schedule' label so schedule_mismatch classification
    of past campaigns is preserved; the spec-diff knobs get their own labels
    routed to stage_mismatch / loop_kind_mismatch.
    """
    kinds = []
    for variant, variant_options in zip(variants, options):
        if variant_options.get('pass_configs') or variant_options.get('enable_fp_fusion'):
            kinds.append('pass-config')
        elif variant_options.get('swizzle'):
            kinds.append('swizzle')
        elif variant_options.get('warp_policy'):
            kinds.append('warp-policy')
        elif variant.spec.threads != spec.threads:
            kinds.append('schedule')
        elif variant.spec.loop_kind != spec.loop_kind:
            # The loop-kind variant also rewrites num_stages (SERIAL pins it
            # to 1), so loop_kind must win over num_stages here.
            kinds.append('loop-kind')
        elif variant.spec.num_stages != spec.num_stages:
            kinds.append('stage')
        else:
            kinds.append('schedule')
    return kinds


def _emit_checked_region(program, backend, config):
    from src.workflow.emitter import _threshold_header
    from src.workflow.emitter.region_checks import (_region_input, _region_input_storage, _region_equal,
                                                    _region_check, _ulp_jitter, _reference_stable,
                                                    _run_region, _run_layout_case)
    p, execution = program.spec, program.execution
    gemm = program.body.operations[0].kind == 'gemm'
    # Division amplifies magnitudes (the reference clamps the denominator to
    # 1e-3, so chained divs grow by ~1e3 per op): absolute error is
    # meaningless there, so div programs also use the relative mode. int8
    # GEMMs compare int32 outputs exactly with one-unit absolute slack.
    int8 = p.dtype.value == 'int8'
    output_dtype = 'int32' if int8 else p.dtype.value
    relative = False if int8 else gemm or any(op.kind == 'div' for op in program.all_operations())
    adapter = get_backend(backend)
    imports = adapter.imports
    parts = [imports, 'import torch', 'import sys', _threshold_header(config), _REFERENCE_SOURCE, _REFERENCE_SOURCE_DOUBLE]
    physical, _, _ = _region_layouts(program)
    # Layout sweep (RC1/RC3): physical programs also run each input case
    # against an alternate layout pair. The relabel helper is only embedded
    # when it is used, keeping legacy emission byte-identical.
    sweep = execution.layout_sweep and physical
    checks = (matrix_layout, _region_input, _region_input_storage, _region_equal, _region_check,
              _ulp_jitter, _reference_stable, _run_region)
    if sweep:
        checks += (_run_layout_case,)
    parts.extend(inspect.getsource(fn) for fn in checks)
    launch_inputs = 'a_storage, b_storage' if physical else 'A, B'
    variant_pairs = region_variant_pairs(program, backend, config)
    variants = [variant for variant, _ in variant_pairs]
    variant_options = [options for _, options in variant_pairs]
    tolerance = 1.0 if int8 else (config.region_rtol_fp16 if p.dtype.value == 'float16' else config.region_rtol_fp32) if relative else config.elemwise_atol
    kinds = _variant_kinds(variants, variant_options, p)
    # Legacy programs (thread pair only) keep the plain historical call text.
    kind_arg = f', variant_kinds={kinds!r}' if any(k != 'schedule' for k in kinds) else ''
    if sweep:
        layout_pairs = _layout_sweep_pairs(program)
        # Layouts are baked into the kernel source (strides, offset, storage
        # size), so every pair compiles its own kernel set: pair p uses the
        # global prepare index p * len(variants) + i, and its failure on p > 0
        # is re-raised as `layout invariance` -> layout_mismatch.
        blocks = []
        for p_index, (la, lb) in enumerate(layout_pairs):
            runner_lines = []
            for i, variant in enumerate(variants):
                pair_variant = replace(variant, execution=replace(
                    variant.execution, input_layout_a=la, input_layout_b=lb))
                code, prepare = adapter.checked_region(pair_variant, p_index * len(variants) + i,
                                                       variant_options[i])
                parts.append(code)
                parts.append(prepare)
                runner_lines.append(f'''def run_variant_{p_index}_{i}():
    print('TILESMITH_STAGE=prepare_{p_index}_{i}', file=sys.stderr, flush=True)
    return prepare_{p_index * len(variants) + i}({launch_inputs})''')
            runner_code = '\n'.join('        ' + line for line in '\n'.join(runner_lines).split('\n'))
            # Concurrent prepares interleave markers, so _run_region re-prints
            # the failing variant's marker (this exact p_index form) before
            # re-raising a compile error.
            marker_arg = f', prepare_markers={[f"prepare_{p_index}_{i}" for i in range(len(variants))]!r}'
            blocks.append(f'''        print('TILESMITH_STAGE=layout:{la}/{lb}', file=sys.stderr, flush=True)
        a_storage, A = _region_input_storage(({p.M}, {p.K if gemm else p.N}), dtype=torch.{p.dtype.value}, pattern={execution.input_pattern!r}, scale={program.input_scale}, layout={la!r})
        b_storage, B = _region_input_storage(({p.K}, {p.N}), dtype=torch.{p.dtype.value}, pattern={execution.input_pattern!r}, scale={program.input_scale}, layout={lb!r})
{runner_code}
        print('TILESMITH_STAGE=reference', file=sys.stderr, flush=True)
        ref = _region_reference(A, B, {asdict(program.body)!r}, {p.block_M}, {p.block_N}, {output_dtype!r}, {[asdict(fn) for fn in program.functions]!r})
        _run_layout_case([{', '.join(f'run_variant_{p_index}_{i}' for i in range(len(variants)))}], (a_storage, b_storage), ref, repeats={execution.repeat_count}, tolerance={tolerance}, relative={relative!r}, layout_a={la!r}, layout_b={lb!r}, alternate={p_index != 0}{kind_arg}{marker_arg}, reference_verify=lambda: _region_reference_double(A, B, {asdict(program.body)!r}, {p.block_M}, {p.block_N}, {output_dtype!r}, {[asdict(fn) for fn in program.functions]!r}), reference_jitter=lambda: _region_reference(_ulp_jitter(A), _ulp_jitter(B), {asdict(program.body)!r}, {p.block_M}, {p.block_N}, {output_dtype!r}, {[asdict(fn) for fn in program.functions]!r}))''')
        run_block = '\n'.join(blocks)
        body = f'''def test_kernel_0():
    for input_case in range({execution.input_seed_count}):
        torch.manual_seed({config.input_seed} + input_case)
{run_block}'''
    else:
        factories = []
        for i, variant in enumerate(variants):
            code, prepare = adapter.checked_region(variant, i, variant_options[i])
            parts.append(code)
            parts.append(prepare)
            factories.append(f'''def run_variant_{i}():
    print('TILESMITH_STAGE=prepare_{i}', file=sys.stderr, flush=True)
    return prepare_{i}({launch_inputs})''')
        # The runner closures live inside test_kernel_0 so they can capture
        # the input_case locals (a_storage/b_storage or A/B) like the old
        # lambdas did. They sit at the for-body indentation level (8 spaces).
        factory_code = '\n'.join('        ' + line for line in '\n'.join(factories).split('\n'))
        body = f'''def test_kernel_0():
    for input_case in range({execution.input_seed_count}):
        torch.manual_seed({config.input_seed} + input_case)
        a_storage, A = _region_input_storage(({p.M}, {p.K if gemm else p.N}), dtype=torch.{p.dtype.value}, pattern={execution.input_pattern!r}, scale={program.input_scale}, layout={execution.input_layout_a!r})
        b_storage, B = _region_input_storage(({p.K}, {p.N}), dtype=torch.{p.dtype.value}, pattern={execution.input_pattern!r}, scale={program.input_scale}, layout={execution.input_layout_b!r})
{factory_code}
        print('TILESMITH_STAGE=reference', file=sys.stderr, flush=True)
        ref = _region_reference(A, B, {asdict(program.body)!r}, {p.block_M}, {p.block_N}, {output_dtype!r}, {[asdict(fn) for fn in program.functions]!r})
        _run_region([{', '.join(f'run_variant_{i}' for i in range(len(variants)))}], (a_storage, b_storage), ref, repeats={execution.repeat_count}, tolerance={tolerance}, relative={relative!r}{kind_arg}, reference_verify=lambda: _region_reference_double(A, B, {asdict(program.body)!r}, {p.block_M}, {p.block_N}, {output_dtype!r}, {[asdict(fn) for fn in program.functions]!r}), reference_jitter=lambda: _region_reference(_ulp_jitter(A), _ulp_jitter(B), {asdict(program.body)!r}, {p.block_M}, {p.block_N}, {output_dtype!r}, {[asdict(fn) for fn in program.functions]!r}))'''
    parts.append(f'''
{body}

if __name__ == '__main__':
    test_kernel_0()
    print('ALL PASSED')
''')
    return '\n'.join(parts)
