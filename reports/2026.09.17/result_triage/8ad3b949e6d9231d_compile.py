import runpy
import triton
from triton.compiler import ASTSource
from triton.backends.compiler import GPUTarget
ns = runpy.run_path('/home/lifang535/fdu_lab/project/tile_program_fuzzing/tp_fuzzing/results/2026.09.17-20.03_triton_hard-shape_seed=42/failed/segfault/failed_calls_main(f0)__f0()_8ad3b949e6d9231d.py')
for name, layouts, options in [('typed_kernel_0', [], {'num_warps': 8, 'enable_fp_fusion': False}), ('typed_kernel_1', [], {'num_warps': 4, 'enable_fp_fusion': False})]:
    fn = ns[name]
    signature = {arg: '*fp32' for arg in fn.arg_names}
    for arg, layout in zip([x for x in fn.arg_names if x not in ('A', 'B', 'C')], layouts):
        signature[arg] = {'float16': '*fp16', 'float32': '*fp32'}[layout[2]]
    print('COMPILING', name, signature, options, flush=True)
    result = triton.compile(ASTSource(fn, signature, {}), target=GPUTarget('cuda', 89, 32), options=options)
    print('COMPILED', name, flush=True)
print('ALL COMPILED')
