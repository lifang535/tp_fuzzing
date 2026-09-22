"""int8 x int8 region GEMM corpus: GPU oracle runs, or compile-only offline.

Default mode runs the campaign oracle (compile + launch + exact integer
reference) for each program on the given backends and reports block_K
coverage. --compile-only compiles the emitted native kernels without
launching: TileLang lowers to CUDA source, Triton builds PTX for sm_89.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import random
import sys
import tempfile
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.backends import get_backend
from src.config import Config
from src.workflow.generator.region_generator import RegionGenerator
from src.workflow.oracle import Oracle


def int8_config(compile_only=False):
    return Config(region_int8_prob=1, coverage_probe_prob=0, function_min_count=0,
                  region_typed_prob=0, region_layout_prob=0,
                  region_schedule_pair=False, region_stage_sweep=False,
                  region_loop_sweep=False, region_layout_sweep=False,
                  region_repeat_count=2, compile_only=compile_only)


def compile_only_program(program, backend, directory, label):
    """Compile the native kernel for sm_89 without launching."""
    if backend == 'triton':
        from src.backends.triton.region import triton_code
        path = directory / (label + '.py')
        # The oracle harness assembles the imports; compile-only emits the
        # bare kernel, so supply the module preamble here.
        path.write_text('import triton\nimport triton.language as tl\n' + triton_code(program))
        spec = importlib.util.spec_from_file_location(label, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        from triton.compiler import ASTSource, compile
        from triton.backends.compiler import GPUTarget
        signature = {'0': '*i8', '1': '*i8', '2': '*i32'}
        compiled = compile(ASTSource(module.kernel, signature),
                           target=GPUTarget('cuda', 89, 32),
                           options={'num_warps': program.spec.threads // 32})
        path.with_suffix('.ptx').write_text(compiled.asm['ptx'])
    else:
        from src.backends.tilelang.region import tilelang_code
        path = directory / (label + '.py')
        path.write_text('import tilelang\nimport tilelang.language as T\n' + tilelang_code(program))
        spec = importlib.util.spec_from_file_location(label, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        from tilelang import tvm
        from tilelang.engine import lower as lower_tilelang
        target = tvm.target.Target({'kind': 'cuda', 'arch': 'sm_89'})
        with target:
            ir = module.make_kernel.get_tir()
            path.with_suffix('.tir').write_text(str(ir))
            compiled = lower_tilelang(ir, target=target, enable_device_compile=False)
        path.with_suffix('.cu').write_text(compiled.kernel_source)
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--backend', choices=('both', 'triton', 'tilelang'), default='both')
    parser.add_argument('--seeds', type=int, default=8)
    parser.add_argument('--compile-only', action='store_true',
                        help='Compile kernels without launching (offline smoke)')
    args = parser.parse_args()
    if args.seeds < 1:
        parser.error('--seeds must be positive')
    directory = Path(tempfile.mkdtemp(prefix='tilesmith_region_int8_'))
    print('Artifacts:', directory, flush=True)
    summary = {'mode': 'compile_only' if args.compile_only else 'execute',
               'cases': [], 'block_k_covered': {}, 'complete': False}
    failures = []
    for backend in ('triton', 'tilelang'):
        if args.backend not in ('both', backend):
            continue
        covered = set()
        config = int8_config(args.compile_only)
        for seed in range(args.seeds):
            random.seed(seed)
            program = RegionGenerator(config, backend).generate()
            block_k = program.spec.block_K
            label = f'{backend}_int8_{seed}_bk{block_k}'
            case_dir = directory / label
            case_dir.mkdir()
            (case_dir / 'program.json').write_text(json.dumps(program.to_dict(), indent=2))
            started = time.monotonic()
            if args.compile_only:
                compile_only_program(program, backend, case_dir, 'kernel')
                passed = True
                note = 'compiled'
            else:
                report = Oracle(config, backend).test(program)
                passed = report is None
                note = 'bug: ' + report.bug_type.value if report else 'oracle pass'
            covered.add(block_k)
            entry = {'case': label, 'passed': passed, 'block_k': block_k, 'note': note,
                     'spec': program.spec.to_dict() if hasattr(program.spec, 'to_dict') else str(program.spec),
                     'seconds': round(time.monotonic() - started, 3)}
            summary['cases'].append(entry)
            summary['block_k_covered'][backend] = sorted(covered)
            (directory / 'summary.json').write_text(json.dumps(summary, indent=2))
            if passed:
                print(f'PASS {label} ({note})', flush=True)
            else:
                failures.append(label)
                print(f'FAIL {label} ({note})', flush=True)
    summary['complete'] = True
    (directory / 'summary.json').write_text(json.dumps(summary, indent=2))
    for backend, covered in summary['block_k_covered'].items():
        if not {32, 64} <= set(covered):
            failures.append(f'{backend}: block_K coverage {sorted(covered)} misses {{32, 64}}')
    if failures:
        print('FAILURES: ' + ', '.join(failures), flush=True)
        return 1
    print(f"All {len(summary['cases'])} int8 region kernels passed "
          f"({summary['mode']}); block_K 32/64 covered on both backends", flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
