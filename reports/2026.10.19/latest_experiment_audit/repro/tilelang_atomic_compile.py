"""TileLang#26: compile-only CUDA controls; no GPU work or installed-file changes.

Usage: NVCC=/usr/local/cuda/bin/nvcc python tilelang_atomic_compile.py
The corrected header exists only in stdin. All cubins go to /dev/null.
"""
import hashlib
import importlib.metadata
import json
import os
import pathlib
import shutil
import subprocess

base = pathlib.Path(importlib.metadata.distribution('tilelang').locate_file('tilelang'))
header = base / 'src/tl_templates/cuda/atomic.h'
original = header.read_text()
needle = 'std::is_same_v<NT1, __nv_bfloat16>)'
assert original.count(needle) >= 2, 'Header changed; inspect the branch before applying this control.'
control = original.replace(needle, 'std::is_same_v<NT1, __nv_bfloat16> || std::is_same_v<NT1, bfloat16_t>)', 4)
nvcc = os.environ.get('NVCC') or shutil.which('nvcc')
if not nvcc and pathlib.Path('/usr/local/cuda/bin/nvcc').is_file():
    nvcc = '/usr/local/cuda/bin/nvcc'
assert nvcc, 'nvcc is required'
rows = []
cases = [('original', 'bfloat16_t', 'Min', 75), ('original', 'bfloat16_t', 'Max', 75),
         ('original', 'half_t', 'Min', 75), ('original', 'bfloat16_t', 'Min', 80),
         ('control', 'bfloat16_t', 'Min', 75), ('control', 'bfloat16_t', 'Max', 75)]
for variant, dtype, operation, arch in cases:
    code = '#include "' + str(header) + '"\n' if variant == 'original' else control + '\n'
    code += 'extern "C" __global__ void test(cutlass::' + dtype + '* out, const cutlass::' + dtype + '* x) { Atomic' + operation + '(out, x[threadIdx.x]); }\n'
    args = [nvcc, '-x', 'cu', '--cubin', '-std=c++17', '-arch=sm_' + str(arch),
            '-I' + str(base / '3rdparty/cutlass/include'), '/dev/stdin', '-o', '/dev/null']
    p = subprocess.run(args, input=code, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=45)
    rows.append({'variant': variant, 'dtype': dtype, 'operation': operation, 'arch': arch,
                 'exit_code': p.returncode, 'diagnostic': p.stdout})
print(json.dumps({'id': 'TileLang#26', 'tilelang': importlib.metadata.version('tilelang'),
                  'nvcc_version': subprocess.check_output([nvcc, '--version'], text=True),
                  'header_sha256': hashlib.sha256(original.encode()).hexdigest(), 'results': rows}, indent=2))
