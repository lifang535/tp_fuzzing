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
from src.ir import TileKernel, TileProgram, ComputeKind, DataType, TilePipeline, PipelineStep
from src.ir.dynamic_seq import (TileBuffer, TileValuePool, DynamicSequence, GemmOpGen,
    ScaleOpGen, AccumulateReduceOpGen, DoublePipelineOpGen, IfEpilogueOpGen,
    ElemwiseAddOpGen, CopyG2SOpGen, CopyS2FOpGen, ElemwiseMulOpGen,
    CopyF2GOpGen, SoftmaxOpGen)
from src.workflow.oracle import Oracle


def dynamic(ops, dtype, n=70):
    params = dict(M=33, N=n, K=32, block_M=32, block_N=32, block_K=32,
                  threads=128, loop_kind='pipelined', num_stages=2,
                  dtype=dtype, acc_dtype='float32')
    pool = TileValuePool(global_in=[TileBuffer('A', (33,32), dtype, 'global', 'A.float()'),
                                          TileBuffer('B', (32,n), dtype, 'global', 'B.float()')])
    counters = {}
    steps = [GemmOpGen().apply(pool, params, counters)]
    for gen in ops:
        steps.append(gen.apply(pool, params, counters))
    if not steps[-1].op_kind == 'softmax':
        steps.append(CopyF2GOpGen().apply(pool, params, counters))
    return DynamicSequence(steps, pool, M=33, N=n, K=32, block_M=32, block_N=32, block_K=32,
                           threads=128, loop_kind='pipelined', num_stages=2, dtype=dtype)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--filter', default='', help='Run labels matching any comma-separated substring')
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
            for kind, pattern, layout, n in (
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
                cases[f'probe_{kind.value}_{pattern}_{layout}'] = TileProgram([TileKernel(
                    'kernel_0', compute_kind=kind, M=33, N=n, K=65,
                    block_M=32, block_N=max(32, 1 << (n - 1).bit_length()), block_K=32,
                    dtype=DataType(dtype), threads=128, coverage_probe=True,
                    input_pattern=pattern, input_layout=layout)])
            for kind in (ComputeKind.COPY, ComputeKind.GEMM, ComputeKind.SOFTMAX):
                cases['single_'+kind.value] = TileProgram([TileKernel('kernel_0', compute_kind=kind,
                        M=33, N=64 if kind == ComputeKind.SOFTMAX else 70, K=32,
                        block_M=64, block_N=64, block_K=32, dtype=DataType(dtype), threads=256)])
            from src.workflow.generator.probes import repair_probe
            cases['probe_gemm_argmax_wide'] = TileProgram([repair_probe(TileKernel(
                'kernel_0', compute_kind=ComputeKind.GEMM_ARGMAX, M=1, N=129, K=129,
                dtype=DataType(dtype), input_pattern='integer', input_layout='strided',
                num_stages=3))])
            cases['local_reduce'] = dynamic([AccumulateReduceOpGen()], dtype)
            cases['double'] = dynamic([ScaleOpGen(), DoublePipelineOpGen()], dtype)
            cases['copy_mul'] = dynamic([ElemwiseAddOpGen(), CopyG2SOpGen(), CopyS2FOpGen(), ElemwiseMulOpGen()], dtype)
            cases['branch'] = dynamic([IfEpilogueOpGen()], dtype)
            cases['dynamic_softmax'] = dynamic([SoftmaxOpGen()], dtype, n=32)
            cases['pipeline'] = TilePipeline([PipelineStep(ComputeKind.GEMM), PipelineStep(ComputeKind.SCALE, alpha=0.5)],
                        M=33,N=70,K=32,block_M=64,block_N=64,block_K=32,dtype=DataType(dtype),threads=256)
            cases['pipeline_softmax'] = TilePipeline([PipelineStep(ComputeKind.GEMM), PipelineStep(ComputeKind.SOFTMAX)],
                        M=33,N=64,K=32,block_M=64,block_N=64,block_K=32,dtype=DataType(dtype),threads=256)
            cases['chain'] = TilePipeline([PipelineStep(ComputeKind.COPY), PipelineStep(ComputeKind.SCALE, alpha=5.0),
                                          PipelineStep(ComputeKind.UNARY_EXP), PipelineStep(ComputeKind.SCALE, alpha=0.001)],
                        M=33,N=70,K=32,block_M=32,block_N=32,block_K=32,dtype=DataType(dtype),threads=128)
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
