"""Compile typed-region kernels for sm_89 without launching or requiring CUDA.

Triton builds PTX/cubin; TileLang performs target lowering and CUDA source
generation. This does not replace tests/gpu_smoke.py --filter region_typed.
"""
import argparse
import importlib.util
from pathlib import Path
import random
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from test_typed_regions import typed_program
from src.ir.region import Operation as Op
from src.backends.common.typed_emitter import TypedLowering


def compile_program(p, backend, directory, label):
    p.validate()
    lower = TypedLowering(p, backend, 'typed_kernel')
    imports = ('import triton\nimport triton.language as tl\n' if backend == 'triton' else
               'import tilelang\nimport tilelang.language as T\n')
    path = directory / (label + '.py')
    path.write_text(imports + lower.emit())
    spec = importlib.util.spec_from_file_location(label, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if backend == 'triton':
        from triton.compiler import ASTSource, compile
        from triton.backends.compiler import GPUTarget
        dtype = p.spec.dtype.value
        dtypes = [dtype, dtype] + [t.dtype for t in lower.slots.values()] + [dtype]
        signature = {i: '*fp16' if t == 'float16' else '*fp32' for i, t in enumerate(dtypes)}
        compiled = compile(ASTSource(module.typed_kernel, signature), target=GPUTarget('cuda',89,32),
                           options={'num_warps':p.spec.threads//32, 'enable_fp_fusion':False})
        path.with_suffix('.ptx').write_text(compiled.asm['ptx'])
    else:
        from tilelang import tvm
        from tilelang.engine import lower as lower_tilelang
        target = tvm.target.Target({'kind':'cuda', 'arch':'sm_89'})
        with target:
            ir = module.typed_kernel.get_tir()
            path.with_suffix('.tir').write_text(str(ir))
            compiled = lower_tilelang(ir, target=target, enable_device_compile=False)
        path.with_suffix('.cu').write_text(compiled.kernel_source)
    print('PASS ' + label, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--backend', choices=('both', 'triton', 'tilelang'), default='both')
    parser.add_argument('--random-count', type=int, default=0, help='Additional generated programs per backend')
    args = parser.parse_args()
    if args.random_count < 0:
        parser.error('--random-count must be nonnegative')
    directory = Path(tempfile.mkdtemp(prefix='tilesmith_typed_offline_'))
    print(f'Artifacts: {directory}', flush=True)
    count = 0
    for backend in ('triton', 'tilelang'):
        if args.backend not in ('both', backend):
            continue
        for dtype in ('float16', 'float32'):
            for initial in ('load', 'gemm'):
                for compact in (False, True):
                    p = typed_program(dtype, initial)
                    if compact:
                        # Exercise column and scalar storage as well as row
                        # storage in the helper, with a second thread schedule.
                        p.spec.threads = 256
                        p.body.operations.extend([
                            Op('reduce_tile', 'column', ['answer'], {'axis':0, 'reduction':'max'}),
                            Op('reduce_tile', 'scalar', ['column'], {'axis':1, 'reduction':'sum'}),
                            Op('cast', 'small', ['scalar'], {'dtype':'float16'}),
                            Op('store_tile', 'small_buffer', ['small']),
                            Op('load_tile', 'small_read', ['small_buffer']),
                            Op('to_tile', 'final', ['small_read'])])
                        p.body.yield_value = 'final'
                    label = f'{backend}_{dtype}_{initial}_{"compact" if compact else "tile"}'
                    compile_program(p, backend, directory, label)
                    count += 1
        if args.random_count:
            from src.config import Config
            from src.workflow.generator.region_generator import RegionGenerator
            config = Config(coverage_probe_prob=0, region_typed_prob=0.7, dim_range=(1,80),
                            tile_size_choices=[32], block_k_choices=[32])
            generator = RegionGenerator(config, backend)
            for seed in range(args.random_count):
                random.seed(seed)
                p = generator.generate(initial='load' if seed % 2 == 0 else 'gemm')
                compile_program(p, backend, directory, f'{backend}_random_{seed}')
                count += 1
    print(f'{count} kernels compiled; no GPU execution', flush=True)


if __name__ == '__main__':
    main()
