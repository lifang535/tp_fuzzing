"""Random pass-pipeline sampling: deterministic compiler-config variants.

Sampling must be a pure function of (program, config): the evidence reader and
the timeout scaling re-derive extended_variants(program, config) and must see
the same list. Sampled configs are plain variants (baseline-checked at
runtime -> configuration_mismatch) and must not feed the precision/identity
sweeps, which iterate the deterministic configurations only.
"""
import ast
import hashlib
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.backends import get_backend
from src.backends.common.knobs import (TILELANG_PASS_POOL, TILELANG_NUMERIC_POOL, TRITON_MAXNREG,
                                       TRITON_STAGES, TRITON_WARPS, sample_configs)
from src.config import Config
from src.workflow.generator.extended import ExtendedGenerator
from src.workflow.generator.identities import extended_variant_label
from src.workflow.oracle import Oracle

# Keys the pool must never sample: numerics-changing, race-prone, safety-
# legalization removal, Hopper-only, debug noise, the deterministic tiers, and
# tl.config_index_bitwidth (universal MakePackedAPI breaker on tilelang 0.1.11).
EXCLUDED_TILELANG_KEYS = {
    'tl.enable_fast_math', 'tl.disable_thread_storage_sync', 'tl.disable_safe_memory_legalize',
    'tl.disable_wgmma', 'tl.disable_tma_lower', 'tl.device_compile_flags',
    'tl.ast_print_enable', 'tl.layout_visualization_enable', 'tl.layout_visualization_formats',
    'tl.enable_ptxas_verbose_output', 'tl.enable_dump_ir', 'tl.dump_ir_path',
    'tl.enable_vectorize_planner_verbose', 'tl.debug_merge_shared_memory_allocations',
    'tirx.disable_vectorize', 'tl.disable_loop_unswitching', 'tl.config_index_bitwidth',
}


def _program(backend, family='mixed'):
    config = Config(extended_prob=1, extended_precision_pair=False, extended_identity_pair=False)
    return ExtendedGenerator(config, backend).generate(family)


class SamplingTests(unittest.TestCase):
    def test_sampling_is_deterministic_per_program_and_seed(self):
        for backend in ('tilelang', 'triton'):
            with self.subTest(backend=backend):
                program = _program(backend)
                config = Config(extended_prob=1, random_config_count=2, seed=42)
                first = sample_configs(program, config, backend)
                second = sample_configs(program, config, backend)
                self.assertEqual(first, second)
                self.assertEqual(len(first), 2)
                other = sample_configs(program, Config(extended_prob=1, random_config_count=2, seed=7),
                                       backend)
                self.assertEqual(len(other), 2)

    def test_extended_variants_are_deterministic(self):
        for backend in ('tilelang', 'triton'):
            with self.subTest(backend=backend):
                adapter = get_backend(backend)
                program = _program(backend)
                config = Config(extended_prob=1, random_config_count=1, seed=3)
                self.assertEqual([(l.name, o) for l, o in adapter.extended_variants(program, config)],
                                 [(l.name, o) for l, o in adapter.extended_variants(program, config)])

    def test_label_uniqueness_and_observation_doubling(self):
        for backend in ('tilelang', 'triton'):
            adapter = get_backend(backend)
            config = Config(extended_prob=1, random_config_count=1,
                            extended_config_depth=1,
                            extended_precision_pair=False, extended_identity_pair=False)
            for pair in (True, False):
                with self.subTest(backend=backend, observation_pair=pair):
                    program = _program(backend)
                    program.observation_pair = pair
                    program.validate()
                    variants = adapter.extended_variants(program, config)
                    labels = [extended_variant_label(backend, i, options)
                              for i, (_, options) in enumerate(variants)]
                    self.assertEqual(len(set(labels)), len(labels))
                    # 2 deterministic configurations x observation modes, plus
                    # one sampled configuration in the same doubling scheme.
                    per_config = 2 if pair else 1
                    self.assertEqual(len(variants), 3 * per_config)
                    self.assertEqual(len({lower.name for lower, _ in variants}), len(variants))

    def test_pool_contents_stay_within_verified_safe_domains(self):
        self.assertFalse(EXCLUDED_TILELANG_KEYS & set(TILELANG_PASS_POOL))
        self.assertFalse(EXCLUDED_TILELANG_KEYS & set(TILELANG_NUMERIC_POOL))
        for backend in ('tilelang', 'triton'):
            with self.subTest(backend=backend):
                program = _program(backend)
                for options in sample_configs(program, Config(random_config_count=4, seed=11), backend):
                    if backend == 'tilelang':
                        self.assertIn(options['threads'], (32, 64, 128, 256))
                        self.assertIn(options['stages'], (1, 2, 3))
                        self.assertTrue(set(options['pass_configs']) <=
                                        set(TILELANG_PASS_POOL) | set(TILELANG_NUMERIC_POOL))
                        self.assertTrue(options['pass_configs'])
                    else:
                        self.assertIn(options['num_warps'], TRITON_WARPS)
                        self.assertIn(options['num_stages'], TRITON_STAGES)
                        self.assertFalse(options['enable_fp_fusion'])
                        self.assertIn(options['maxnreg'], TRITON_MAXNREG)

    def test_sampling_off_at_zero(self):
        for backend in ('tilelang', 'triton'):
            with self.subTest(backend=backend):
                adapter = get_backend(backend)
                program = _program(backend)
                config = Config(extended_prob=1, extended_config_depth=1,
                                extended_precision_pair=False, extended_identity_pair=False,
                                random_config_count=0)
                self.assertEqual(sample_configs(program, config, backend), [])
                # 2 deterministic configurations x observation modes.
                self.assertEqual(len(adapter.extended_variants(program, config)), 4)

    def test_sampled_configs_are_plain_variants(self):
        for backend in ('tilelang', 'triton'):
            with self.subTest(backend=backend):
                program = _program(backend)
                for options in sample_configs(program, Config(random_config_count=2, seed=5), backend):
                    self.assertNotIn('precision', options)
                    self.assertNotIn('identity', options)
                    self.assertNotIn('input_precision', options)

    def test_sampled_configs_do_not_feed_precision_sweep(self):
        from src.workflow.generator.identities import precision_program
        for backend in ('tilelang', 'triton'):
            with self.subTest(backend=backend):
                adapter = get_backend(backend)
                config = Config(extended_prob=1, extended_config_depth=1,
                                extended_precision_pair=True, extended_identity_pair=False,
                                random_config_count=2, extended_int8_prob=0,
                                extended_fma_prob=0, extended_shape_op_prob=0, extended_atomic_prob=0)
                program = ExtendedGenerator(config, backend).generate('shape_matmul')
                self.assertIsNotNone(precision_program(program))
                variants = adapter.extended_variants(program, config)
                prec = [v for v in variants if v[1].get('precision') == 'fp16']
                # One fp16-accumulation variant per deterministic configuration
                # only; sampled configs are not multiplied into the sweep.
                self.assertEqual(len(prec), 2)
                self.assertEqual(len({lower.name for lower, _ in variants}), len(variants))

    def test_evidence_count_consistency_with_sampling(self):
        config = Config(backends=['triton'], extended_prob=1, extended_config_depth=1,
                        extended_precision_pair=False, extended_identity_pair=False,
                        random_config_count=1)
        program = _program('triton')
        oracle = Oracle(config, 'triton')
        with tempfile.TemporaryDirectory() as directory:
            oracle.artifact_root = Path(directory) / 'artifacts'
            count = len(oracle.backend_impl.extended_variants(program, config))
            records = [dict(variant=f'triton_{i}', complete=True, features=[],
                            stages={'ptx': {'sha256': hashlib.sha256(b'fixture').hexdigest()}})
                       for i in range(count)]

            def launch(command, **options):
                artifact_dir = Path(options['env']['TILESMITH_ARTIFACT_DIR'])
                for record in records:
                    (artifact_dir / (record['variant'] + '.ptx')).write_text('fixture')
                (artifact_dir / 'compilation.json').write_text(json.dumps(records))
                (artifact_dir / 'progress.json').write_text(json.dumps(
                    {'stage': 'complete', 'variant': 'execute'}))
                return subprocess.CompletedProcess(command, 0, 'ALL PASSED\n', '')

            with patch('src.workflow.oracle.process.run_isolated', launch):
                self.assertIsNone(oracle.test(program))
            self.assertTrue(oracle.compilation_complete)

    def test_emission_ships_sampled_variants_without_reference_entries(self):
        config = Config(extended_prob=1, extended_config_depth=1,
                        extended_precision_pair=False, extended_identity_pair=False,
                        random_config_count=1)
        program = _program('triton')
        code = Oracle(config, 'triton')._emit_code(program)
        ast.parse(code)
        self.assertIn("'maxnreg':", code)
        self.assertIn('REFERENCE_PROGRAMS = {}', code)


if __name__ == '__main__':
    unittest.main()
