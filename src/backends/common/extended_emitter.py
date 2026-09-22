"""Standalone reproducer assembly; target syntax stays in each backend."""
import inspect
from src.backends import get_backend
from src.workflow.emitter import extended_runtime as runtime
from src.workflow.generator.identities import (extended_variant_label, precision_program,
                                               identity_variant)

_RUNTIME = '\n\n'.join(inspect.getsource(getattr(runtime, name)) for name in (
    'extended_stage', 'record_extended_compilation', 'extended_inputs',
    'extended_reference', 'extended_check', 'extended_check_atomic',
    'extended_check_fma', '_atomic_kind', 'run_extended'))


def emit_extended(program, backend, config):
    adapter = get_backend(backend)
    adapter.validate_program(program)
    entries = adapter.extended_variants(program, config)
    # Precision and identity variants run transformed copies of the program
    # (fp16 accumulation / distributivity); ship them so the runtime can
    # compute their expected values.
    reference_programs = {}
    transformed = precision_program(program) if any(
        options.get('precision') == 'fp16' for _, options in entries) else None
    if transformed is not None:
        for index, (_, options) in enumerate(entries):
            if options.get('precision') == 'fp16':
                reference_programs[extended_variant_label(backend, index, options)] = transformed.to_dict()
    distributed = identity_variant(program) if any(
        options.get('identity') for _, options in entries) else None
    if distributed is not None:
        for index, (_, options) in enumerate(entries):
            if options.get('identity'):
                reference_programs[extended_variant_label(backend, index, options)] = distributed.to_dict()
    return '\n\n'.join([
        'import os\n' + adapter.imports,
        _RUNTIME,
        *(lower.emit() for lower, _ in entries),
        adapter.extended_compile_source(entries, program),
        f'PROGRAM = {program.to_dict()!r}',
        f'REFERENCE_PROGRAMS = {reference_programs!r}',
        f'''if __name__ == '__main__':
    if os.environ.get('TILESMITH_COMPILE_ONLY', {str(int(config.compile_only))!r}) == '1':
        prepare_extended(device_compile=False)
        extended_stage('complete', 'compile_only')
        print('COMPILE PASSED (no GPU execution)')
    else:
        run_extended(PROGRAM, prepare_extended, {config.input_seed}, {config.region_repeat_count}, reference_programs=REFERENCE_PROGRAMS)
        extended_stage('complete', 'execute')
        print('ALL PASSED')
'''])
