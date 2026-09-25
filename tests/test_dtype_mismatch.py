"""dtype_mismatch regression: dtype-insensitive impl source collides in
tilelang's persistent frontend cache.

Historical runs (e.g. results/2026.09.14-18.37_tilelang_easy-shape_seed=42)
recorded `dtype_mismatch` failures: programs compiled expecting one dtype but
invoked with tensors of the other. The mechanism is tilelang's `@tilelang.jit`
frontend cache, keyed on `inspect.getsource(impl)` plus the jit call args.
Programs whose kernels differ only in a non-source binding (closure/global
dtype) share one cache entry, so the second program reuses the first dtype's
compiled kernel and the call fails with "kernel impl input A dtype mismatch".
The region emitter keeps that bug class reachable by binding dtype at module
scope instead of inlining it into the impl source.

That reachability is version-dependent, and the tests below assert whichever
behaviour the installed tilelang has (see _frontend_cache_collides): the class
is unreachable on 0.1.14, where the kernel cache is keyed on the parsed TIR.
"""
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest

import torch

from src.config import Config
from src.ir import DataType
from src.backends.common import versions
from src.backends.tilelang.region import tilelang_code
from src.workflow.oracle import Oracle


def _version_tuple(text):
    return tuple(int(part) for part in re.findall(r'\d+', text or '')[:3])


def _frontend_cache_collides() -> bool:
    """Does the installed tilelang still collide two dtypes onto one kernel?

    0.1.11 keys its frontend cache on ``inspect.getsource(impl)``
    (``jit/__init__.py:_frontend_cache_key_data``), which cannot see a dtype
    bound at module scope. 0.1.14 deleted that method and keys the kernel cache
    on a hash of ``func.script(show_meta=True)``
    (``cache/kernel_cache.py:_generate_key``) -- the parsed TIR, where the dtype
    is a concrete buffer type -- so the two dtypes no longer share an entry.
    """
    installed = versions.installed('tilelang')
    return installed is not None and _version_tuple(installed) < (0, 1, 14)


def _impl_source(code: str) -> str:
    """The jit-relevant source: everything from @tilelang.jit to return impl."""
    match = re.search(r'@tilelang\.jit.*?(?=\n    return impl)', code, re.S)
    assert match is not None, 'no @tilelang.jit block in emitted code'
    return match.group(0)


class DtypeMismatchTests(unittest.TestCase):
    def test_impl_source_is_dtype_insensitive(self):
        # CPU-only emitter check: two programs that differ only in dtype must
        # produce byte-identical jit source (the cache-collision precondition).
        from test_regions import nested_program
        for initial in ('load', 'gemm'):
            with self.subTest(initial=initial):
                code16 = tilelang_code(nested_program('float16', initial))
                code32 = tilelang_code(nested_program('float32', initial))
                self.assertEqual(_impl_source(code16), _impl_source(code32))
                # The dtype value still lands in the harness, just not in the
                # jit source, and the module-level binding stays valid Python.
                self.assertIn('dtype = "float16"', code16)
                full = Oracle(Config(coverage_probe_prob=0), 'tilelang')._emit_code(
                    nested_program('float16', initial))
                self.assertIn('dtype=torch.float16', full)

    @unittest.skipUnless(torch.cuda.is_available(), 'requires CUDA')
    def test_second_dtype_reuses_first_cached_kernel(self):
        from test_regions import nested_program
        config = Config(coverage_probe_prob=0)
        oracle = Oracle(config, backend='tilelang')
        cache_dir = tempfile.mkdtemp(prefix='tl_iso_cache_')
        self.addCleanup(shutil.rmtree, cache_dir, ignore_errors=True)
        env = dict(os.environ, TILELANG_CACHE_DIR=cache_dir)

        def run(program):
            code = oracle._emit_code(program)
            with tempfile.NamedTemporaryFile('w', suffix='.py', delete=False) as f:
                f.write(code)
                path = f.name
            self.addCleanup(os.unlink, path)
            return subprocess.run([sys.executable, path], capture_output=True,
                                  text=True, env=env, timeout=600)

        first = run(nested_program('float16', 'gemm'))
        self.assertEqual(first.returncode, 0, f'float16 baseline failed:\n{first.stderr}')
        second = run(nested_program('float32', 'gemm'))
        if _frontend_cache_collides():
            self.assertNotEqual(second.returncode, 0)
            self.assertIn('dtype mismatch, expected float16', second.stderr)
        else:
            # Upstream fixed the key: assert the fix, so that a reintroduction
            # fails here instead of leaving this test red on the target pair.
            # The precondition above still holds on 0.1.14 -- the dtype is
            # still absent from the jit source -- so what changed is the key.
            self.assertEqual(second.returncode, 0,
                             f'float32 rerun failed:\n{second.stderr}')
            self.assertNotIn('dtype mismatch', second.stderr)
