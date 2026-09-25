"""Version adaptation: both DSL pairs must run the same harness revision.

Upstream moved two things this harness depends on:

* TileLang 0.1.14 moved the CUDA compile callback from engine/lower.py to
  cuda/backend.py (same name, same signature) and stopped exporting it from
  engine/lower.py, so an unconditional import aborted every extended variant.
* Triton 3.8 rejects non-string ASTSource signature keys ("Signature keys must
  be string") and resolves them against the kernel's parameter names; 3.0
  accepted positional integer keys.

The generated harnesses resolve both at run time (try/except, parameter names),
so these tests drive the *emitted* code rather than the emitter's internals.
"""
import ast
import sys
import textwrap
import types
import unittest
from unittest.mock import patch

from src.backends import get_backend
from src.backends.common.knobs import (TILELANG_PASS_POOL, TILELANG_NUMERIC_POOL,
                                       TILELANG_REGION_PASS_POOL, missing_option_fields,
                                       missing_pool_keys, pool_keys)
from src.backends.common.versions import (DISTRIBUTIONS, LEGACY_TILELANG, LEGACY_TRITON,
                                          TARGET_TILELANG, TARGET_TRITON, describe,
                                          environment, installed, mismatches)
from src.config import Config
from src.workflow.generator.extended import ExtendedGenerator
from src.workflow.oracle import Oracle


class _Buffer:
    def __init__(self, name):
        self.name, self.base = name, None


class _Program:
    buffers = [_Buffer('A'), _Buffer('B')]


class _Lowering:
    name, watched = 'impl', ['r0']


def _emitted_import_block(source):
    """The callback import inside the emitted prepare_extended, dedented.

    Extracted by line number: ast.get_source_segment drops the first line's
    indentation, which would leave the body over-indented after dedent.
    """
    tree = ast.parse(source)
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                    and n.name == 'prepare_extended')
    statement = next(n for n in function.body if isinstance(n, ast.Try))
    return textwrap.dedent('\n'.join(source.splitlines()[statement.lineno - 1:statement.end_lineno]))


def _module(name, **attributes):
    module = types.ModuleType(name)
    for key, value in attributes.items():
        setattr(module, key, value)
    return module


def _emitted_ast_source(code):
    """The ASTSource signature literal the emitted triton harness compiles with."""
    for node in ast.walk(ast.parse(code)):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id == 'ASTSource'):
            return ast.literal_eval(node.args[1])
    return None


class TileLangCallbackImportTests(unittest.TestCase):
    """The emitted extended harness must import the callback from either pair."""

    def setUp(self):
        from src.backends.tilelang.extended import compile_source
        self.source = compile_source([(_Lowering(), {'threads': 128, 'stages': 1,
                                                     'pass_configs': {}})], _Program())
        self.block = _emitted_import_block(self.source)

    def test_callback_import_is_guarded(self):
        self.assertIn('from tilelang.cuda.backend import tilelang_callback_cuda_compile', self.block)
        self.assertIn('from tilelang.engine.lower import tilelang_callback_cuda_compile', self.block)

    def test_new_location_wins_when_both_exist(self):
        """TileLang >= 0.1.14: cuda/backend.py holds the callback."""
        new = _module('tilelang.cuda.backend', tilelang_callback_cuda_compile='new')
        old = _module('tilelang.engine.lower', tilelang_callback_cuda_compile='old')
        namespace = {}
        with patch.dict(sys.modules, {'tilelang.cuda.backend': new, 'tilelang.engine.lower': old}):
            exec(self.block, namespace)
        self.assertEqual(namespace['tilelang_callback_cuda_compile'], 'new')

    def test_old_location_is_the_fallback(self):
        """TileLang 0.1.11: engine/lower.py holds it and cuda/backend.py is absent."""
        old = _module('tilelang.engine.lower', tilelang_callback_cuda_compile='old')
        modules = {'tilelang.engine.lower': old, 'tilelang.cuda': _module('tilelang.cuda')}
        namespace = {}
        with patch.dict(sys.modules, modules):
            sys.modules.pop('tilelang.cuda.backend', None)
            exec(self.block, namespace)
        self.assertEqual(namespace['tilelang_callback_cuda_compile'], 'old')

    def test_a_missing_callback_still_fails_loudly(self):
        """Neither pair provides it: the ImportError must not be swallowed."""
        modules = {'tilelang.cuda': _module('tilelang.cuda'),
                   'tilelang.engine': _module('tilelang.engine'),
                   'tilelang.engine.lower': _module('tilelang.engine.lower')}
        with patch.dict(sys.modules, modules):
            sys.modules.pop('tilelang.cuda.backend', None)
            with self.assertRaises(ImportError):
                exec(self.block, {})


class TritonSignatureTests(unittest.TestCase):
    """ASTSource signatures are keyed by parameter name, on both pairs."""

    @classmethod
    def setUpClass(cls):
        config = Config(extended_prob=1, extended_precision_pair=False,
                        extended_identity_pair=False, random_config_count=0,
                        extended_int8_prob=0, extended_fma_prob=0,
                        extended_shape_op_prob=0, extended_atomic_prob=0)
        cls.program = ExtendedGenerator(config, 'triton').generate('shape_matmul')
        cls.code = Oracle(config, 'triton')._emit_code(cls.program)

    def test_signature_keys_are_parameter_names(self):
        from src.backends.triton.extended import ExtendedLowering
        entries = get_backend('triton').extended_variants(self.program)
        self.assertTrue(entries)
        for lowering, _ in entries:
            signature = lowering.signature()
            self.assertTrue(all(isinstance(key, str) for key in signature), signature)
            self.assertEqual(list(signature), lowering.parameters())
            self.assertTrue(all(value.startswith('*') for value in
                                list(signature.values())[:-2]), signature)
            self.assertEqual(list(signature.values())[-2:], ['i32', 'i32'])

    def test_signature_matches_the_emitted_kernel(self):
        from src.backends.triton.extended import ExtendedLowering
        lowering = ExtendedLowering(self.program, 'extended_0_0', stages=1)
        emitted = lowering.emit()
        kernel = next(node for node in ast.parse(emitted).body
                      if isinstance(node, ast.FunctionDef) and node.name == 'extended_0_0')
        self.assertEqual(list(lowering.signature()),
                         [argument.arg for argument in kernel.args.args])

    def test_emitted_harness_carries_named_keys(self):
        """3.8 rejects positional keys before anything else runs."""
        signature = _emitted_ast_source(self.code)
        self.assertTrue(signature)
        self.assertTrue(all(isinstance(key, str) for key in signature), signature)
        self.assertTrue(all(isinstance(value, str) for value in signature.values()))
        self.assertTrue(any(value.startswith('*') for value in signature.values()), signature)


class VersionMarkingTests(unittest.TestCase):
    def test_environment_records_every_distribution(self):
        record = environment()
        self.assertEqual(set(record), set(DISTRIBUTIONS) | {'python'})
        for name in DISTRIBUTIONS:
            self.assertTrue(record[name] is None or isinstance(record[name], str))

    def test_installed_reads_metadata_without_importing(self):
        self.assertIsNotNone(installed('tilelang'))
        self.assertIsNone(installed('definitely-not-a-package-xyz'))

    def test_target_and_legacy_pairs_are_recorded(self):
        self.assertEqual(TARGET_TILELANG, '0.1.14')
        self.assertEqual(TARGET_TRITON, '3.8.0')
        self.assertEqual((LEGACY_TILELANG, LEGACY_TRITON), ('0.1.11', '3.0.0'))

    def test_mismatch_is_reported_not_raised(self):
        """Fuzzing a newer DSL before the shims catch up must stay possible."""
        notes = mismatches()
        self.assertTrue(all(isinstance(note, str) for note in notes))
        banner = describe()
        for name in DISTRIBUTIONS:
            self.assertIn(name + '=', banner)

    def test_campaign_summary_records_the_environment(self):
        import inspect
        from src.workflow.fuzzer import fuzzer
        source = inspect.getsource(fuzzer)
        self.assertIn('"environment": environment()', source)
        self.assertIn('"target_versions"', source)


class PoolVersionTests(unittest.TestCase):
    """Pass-config pools must stay truthful about the installed release."""

    def test_missing_pool_keys_is_empty_on_supported_releases(self):
        missing = missing_pool_keys()
        if missing is None:
            self.skipTest('tilelang is not installed in this interpreter')
        self.assertEqual(missing, set(), 'pool keys absent from the installed tilelang')

    def test_missing_option_fields_is_empty_on_supported_releases(self):
        missing = missing_option_fields()
        if missing is None:
            self.skipTest('triton is not installed in this interpreter')
        self.assertEqual(missing, set(), 'triton options absent from the installed release')

    def test_pool_keys_helper_covers_every_pool(self):
        self.assertEqual(pool_keys(), set(TILELANG_PASS_POOL) | set(TILELANG_NUMERIC_POOL)
                         | set(TILELANG_REGION_PASS_POOL))

    def test_pools_do_not_depend_on_the_installed_version(self):
        """Static membership keeps sampling a pure function of (program, config)."""
        import src.backends.common.knobs as knobs
        with patch.object(knobs, 'missing_pool_keys', return_value=set()):
            self.assertEqual(knobs.TILELANG_PASS_POOL, TILELANG_PASS_POOL)


if __name__ == '__main__':
    unittest.main()
