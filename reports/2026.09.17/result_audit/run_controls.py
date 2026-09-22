"""Bounded, sequential controls. Originals and campaign caches are untouched."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
OLD = OUT.parent / 'result_triage'
UPDATE = OUT.parent / 'result_triage_update'


def run(label, source, cache=None, timeout=60):
    cache = Path(cache or tempfile.mkdtemp(prefix='result_audit_'))
    env = dict(os.environ, OMP_NUM_THREADS='2', PYTHONFAULTHANDLER='1',
               TILELANG_CACHE_DIR=str(cache / 'tilelang'),
               TRITON_CACHE_DIR=str(cache / 'triton'))
    start = time.monotonic()
    with (OUT / (label + '.log')).open('w') as log:
        process = subprocess.Popen([sys.executable, '-u', str(source)], cwd=ROOT,
                                   env=env, stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=True)
        try:
            code = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
            code = 'timeout'
    record = dict(label=label, source=str(source), returncode=code,
                  seconds=round(time.monotonic() - start, 3), cache=str(cache))
    with (OUT / 'replays.jsonl').open('a') as f:
        f.write(json.dumps(record) + '\n')
    print(json.dumps(record), flush=True)


def save(label, code):
    path = OUT / (label + '.py')
    path.write_text(code)
    return path


if __name__ == '__main__':
    if sys.argv[1] == 'confirmed':
        cache = tempfile.mkdtemp(prefix='result_audit_dtype_')
        run('dtype16_first', OLD / 'dtype16_first.py', cache)
        run('dtype32_same_cache', OLD / 'dtype32_same_cache.py', cache)
        run('dtype32_fresh_cache', OLD / 'dtype32_fresh_cache.py')
        for label in ['block_k_8_exact', 'triton_constant_dot',
                      'triton_positive_constant_dot', 'triton_memory_operand_dot',
                      '4abdeda508af655d_compile',
                      '28f7a4c7a8eda0b4_explicit_barrier']:
            run(label, OLD / (label + '.py'))
        paths = json.loads((UPDATE / 'selected.json').read_text())
        run('cpasync_odd', ROOT / paths['cpasync_odd'].replace('.json', '.py'))
        source = next(ROOT.glob('results/*/failed/nondeterminism/*.py'))
        run('nondeterminism_original', source)
    elif sys.argv[1] == 'fuzzer':
        paths = json.loads((UPDATE / 'selected.json').read_text())
        for label in ['triton_softmax', 'triton_dynamic_gemm_softmax',
                      'tilelang_dynamic_gemm_softmax',
                      'tilelang_dynamic_gemm_accumulate_reduce_copy_f2g']:
            source = ROOT / paths[label].replace('.json', '.py')
            code = source.read_text().replace("if __name__ == '__main__':",
                                              "torch.manual_seed(42)\nif __name__ == '__main__':")
            run(label + '_original', save(label + '_original', code))
            if label.startswith('triton'):
                import re
                code = re.sub(r'(\s*)(\w+) = tl.softmax\((\w+), 1\)',
                              lambda m: m[1] + '_audit_exp = tl.exp(' + m[3] + ' - tl.max(' + m[3] + ', axis=1)[:, None])' + m[1] + m[2] + ' = _audit_exp / tl.sum(_audit_exp, axis=1)[:, None]', code)
            elif label.endswith('gemm_softmax'):
                code = code.replace('        return impl',
                                    '                T.copy(C_local_1, C[by * block_M, bx * block_N])\n        return impl')
            else:
                code = code.replace('                T.reduce_max(C_local_1, row_stat_1, dim=1, clear=True)',
                                    '                for i, j in T.Parallel(block_M, block_N):\n                    if bx * block_N + j >= N:\n                        C_local_1[i, j] = -T.infinity(accum_dtype)\n                T.reduce_max(C_local_1, row_stat_1, dim=1, clear=True)')
            run(label + '_corrected', save(label + '_corrected', code))
