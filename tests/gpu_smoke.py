"""Run small standalone reproducers on both GPU backends, without fuzz result writes.

Usage: python tests/gpu_smoke.py
Artifacts and logs are retained in the printed temporary directory on failure.
"""
import argparse
import os
import random
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import Config
from src.ir import TileKernel, ComputeKind, DataType
from src.backends.common.probes import probe_program
from src.workflow.oracle import Oracle


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--filter', default='', help='Run labels matching any comma-separated substring')
    parser.add_argument('--unaligned-region-strides', action='store_true',
                        help='Keep odd fp16 GEMM strides to diagnose cp.async lowering failures')
    parser.add_argument('--shared-cache', action='store_true', help='Exercise the user cache across cases')
    args = parser.parse_args()
    random.seed(10)
    directory = Path(tempfile.mkdtemp(prefix='tilesmith_gpu_smoke_'))
    print(f'Artifacts: {directory}', flush=True)
    failures = []
    total = 0
    for backend in ('triton', 'tilelang'):
        oracle = Oracle(Config(input_seed=7), backend)
        for dtype in ('float32', 'float16'):
            cases = {}
            from test_typed_regions import typed_program
            cases['region_typed_memory_load'] = typed_program(dtype, 'load')
            cases['region_typed_memory_gemm'] = typed_program(dtype, 'gemm')
            from test_regions import nested_program, loop_outer_program, migrated_ops_program
            cases["region_nested_load"] = nested_program(dtype, "load")
            cases["region_nested_gemm"] = nested_program(dtype, "gemm")
            cases["region_outer_load"] = loop_outer_program(dtype, "load")
            cases["region_outer_gemm"] = loop_outer_program(dtype, "gemm")
            cases["region_migrated_elementwise"] = migrated_ops_program(dtype)
            cases["region_migrated_reductions"] = migrated_ops_program(dtype, True)
            from test_functions import function_program
            cases['region_functions_load'] = function_program(dtype)
            cases['region_functions_gemm'] = function_program(dtype, 'gemm')
            cases['region_functions_reductions'] = function_program(dtype, reductions=True)
            from test_generation_diversity import arithmetic_program
            cases['region_div_minimum_functions'] = arithmetic_program(dtype)
            from test_region_coverage import coverage_program
            cases['region_coverage_load'] = coverage_program(dtype, 'load')
            cases['region_coverage_gemm'] = coverage_program(dtype, 'gemm')
            layout_cases = (
                ('offset_load', 'load', 'offset', 'contiguous'),
                ('broadcast_gemm', 'gemm', 'broadcast_rows', 'broadcast_cols'),
            ) if dtype == 'float16' else (
                ('transposed_load', 'load', 'transposed', 'contiguous'),
                ('strided_gemm', 'gemm', 'strided', 'contiguous'),
            )
            for label, initial, layout_a, layout_b in layout_cases:
                program = coverage_program(dtype, initial)
                program.input_scale = 0.125
                program.execution.input_pattern = 'integer'
                program.execution.input_seed_count = 1
                program.execution.repeat_count = 2
                program.execution.input_layout_a = layout_a
                program.execution.input_layout_b = layout_b
                program.execution.layout_sweep = True  # run the alternate layout pair too
                cases['region_layout_' + label] = program
            from src.ir.region import RegionProgram, Region, Operation
            spec = TileKernel('kernel_0', compute_kind=ComputeKind.COPY, M=33, N=33, K=32,
                              block_M=32, block_N=64, block_K=32, threads=128,
                              dtype=DataType(dtype), coverage_probe=True,
                              input_layout='strided', input_pattern='special', cache_cycle=True)
            cases['region_probe'] = RegionProgram(spec, Region([], [Operation('probe', 'v1')], 'v1'))
            for kind, pattern, layout, n in (
                (ComputeKind.COPY, 'indexed', 'broadcast_rows', 33),
                (ComputeKind.REDUCE_SUM, 'integer', 'broadcast_cols', 31),
                (ComputeKind.COPY, 'special', 'offset', 33),
                (ComputeKind.COPY, 'subnormal', 'transposed', 31),
                (ComputeKind.COPY, 'indexed', 'strided', 65),
                (ComputeKind.ARGMAX, 'negative', 'strided', 33),
                (ComputeKind.ARGMAX, 'ties', 'offset', 1),
                (ComputeKind.GEMM_ARGMAX, 'integer', 'transposed', 33),
                (ComputeKind.GEMM_ARGMAX, 'ties', 'offset', 31),
                (ComputeKind.REDUCE_SUM, 'integer', 'offset', 65),
                (ComputeKind.REDUCE_MAX, 'negative', 'strided', 33),
                (ComputeKind.REDUCE_MIN, 'integer', 'transposed', 31),
                (ComputeKind.SOFTMAX, 'negative', 'offset', 33),
            ):
                cases[f'probe_{kind.value}_{pattern}_{layout}'] = probe_program(TileKernel(
                    'kernel_0', compute_kind=kind, M=33, N=n, K=65,
                    block_M=32, block_N=max(32, 1 << (n - 1).bit_length()), block_K=32,
                    dtype=DataType(dtype), threads=128, coverage_probe=True,
                    input_pattern=pattern, input_layout=layout, cache_cycle=True))
            cases['probe_gemm_argmax_wide'] = probe_program(TileKernel(
                'kernel_0', compute_kind=ComputeKind.GEMM_ARGMAX, M=1, N=129, K=129,
                dtype=DataType(dtype), input_pattern='integer', input_layout='strided',
                num_stages=3))
            if args.unaligned_region_strides:
                for name in ('region_nested_gemm', 'region_outer_gemm'):
                    cases[name].spec.N, cases[name].spec.K = 35, 33
            for name, program in cases.items():
                label = f'{backend}_{dtype}_{name}'
                if not any(part in label for part in args.filter.split(',')):
                    continue
                total += 1
                path = directory / (label+'.py')
                path.write_text(oracle._emit_code(program))
                env = dict(os.environ)
                if not args.shared_cache:
                    env['TILELANG_CACHE_DIR'] = str(directory / (label + '_cache'))
                try:
                    result = subprocess.run([sys.executable, str(path)], env=env, capture_output=True, text=True, timeout=120)
                    log = result.stdout + result.stderr
                    failed = result.returncode != 0
                except subprocess.TimeoutExpired as exc:
                    failed = True
                    log = str(exc)
                if failed:
                    failures.append(label)
                    path.with_suffix('.log').write_text(log)
                    print('FAIL '+label+'\n'+log[-2200:], flush=True)
                else:
                    print('PASS '+label, flush=True)
    print(f'{total-len(failures)}/{total} passed; failures={failures}', flush=True)
    return bool(failures)

if __name__ == '__main__':
    sys.exit(main())
