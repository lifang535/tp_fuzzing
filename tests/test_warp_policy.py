"""Warp-policy sweep (GemmWarpPolicy variants) + random-config-count bump.

The warp-level MLIRSmith knob: region gemm programs additionally execute
FullRow / FullCol GemmWarpPolicy variants of the base kernel (tilelang only —
its GemmWarpPolicy restructures the warp partition without touching the
per-tile math, so every variant shares the one reference). tilelang's warp
*specialization* itself is TMA-gated (sm_90+), so its differential rides the
pass-config pool via tl.disable_warp_specialized. Also guards the
--random-config-count default bump to 2.
"""
import unittest
from dataclasses import replace

from src.backends import get_backend
from src.backends.common import diagnostics, knobs
from src.backends.common.region_emitter import _variant_kinds
from src.backends.tilelang.params import check_warp_partition
from src.backends.tilelang.region import tilelang_code
from src.config import Config
from src.ir import ComputeKind, DataType, TileKernel
from src.ir.region import Operation, Region, RegionExecution, RegionProgram
from src.ir.serialization import program_from_dict, program_to_dict


def gemm_program(block_m=64, block_n=64, threads=128, warp_policy='square', **exec_overrides):
    spec = TileKernel('kernel_0', compute_kind=ComputeKind.GEMM, M=256, N=256, K=128,
                      block_M=block_m, block_N=block_n, block_K=32, threads=threads,
                      dtype=DataType.FLOAT16, warp_policy=warp_policy)
    body = Region(arguments=[], operations=[Operation('gemm', 'out', [])],
                  yield_value='out')
    execution = RegionExecution(**exec_overrides)
    return RegionProgram(spec=spec, body=body, execution=execution)


class WarpPolicyFieldTests(unittest.TestCase):
    def test_default_is_square_and_omitted_from_params(self):
        spec = TileKernel('kernel_0')
        self.assertEqual(spec.warp_policy, 'square')
        self.assertNotIn('warp_policy', spec.params_dict)

    def test_non_square_policy_round_trips_in_params(self):
        spec = TileKernel('kernel_0', warp_policy='full_row')
        self.assertEqual(spec.params_dict['warp_policy'], 'full_row')
        self.assertEqual(TileKernel('kernel_0', **spec.params_dict).warp_policy, 'full_row')

    def test_invalid_policy_rejected(self):
        with self.assertRaises(ValueError):
            TileKernel('kernel_0', warp_policy='diagonal')


class WarpPartitionPolicyTests(unittest.TestCase):
    def test_full_row_needs_m_tiles_per_warp(self):
        # 64x64 tile, 128 threads: 4 warps, m_tiles=4 -> feasible.
        self.assertTrue(check_warp_partition(64, 64, 128, 'full_row'))
        # 32x64 tile, 128 threads: m_tiles=2 < 4 warps -> infeasible.
        self.assertFalse(check_warp_partition(32, 64, 128, 'full_row'))

    def test_full_col_needs_n_tiles_per_warp(self):
        self.assertTrue(check_warp_partition(64, 64, 128, 'full_col'))
        # 64x16, 128 threads: n_tiles=2 < 4 warps -> infeasible.
        self.assertFalse(check_warp_partition(64, 16, 128, 'full_col'))

    def test_square_unchanged(self):
        # 32x16, 128 threads: 2x2 partition valid, both policies infeasible.
        self.assertTrue(check_warp_partition(32, 16, 128))
        self.assertFalse(check_warp_partition(32, 16, 128, 'full_row'))
        self.assertFalse(check_warp_partition(32, 16, 128, 'full_col'))


class VariantPairTests(unittest.TestCase):
    def test_tilelang_emits_policy_variants_when_feasible(self):
        program = gemm_program(64, 64, 128, warp_policy_sweep=True)
        backend = get_backend('tilelang')
        variants, options = zip(*backend._region_variant_pairs(program))
        policies = [(v.spec.warp_policy, o.get('warp_policy'))
                    for v, o in zip(variants, options) if o.get('warp_policy')]
        self.assertEqual([p for p, _ in policies], ['full_row', 'full_col'])
        self.assertEqual(policies, [('full_row', 'full_row'), ('full_col', 'full_col')])

    def test_infeasible_geometry_skips_policy_variants(self):
        program = gemm_program(32, 16, 128, warp_policy_sweep=True)
        backend = get_backend('tilelang')
        variants, options = zip(*backend._region_variant_pairs(program))
        self.assertFalse(any(o.get('warp_policy') for o in options))

    def test_sweep_flag_off_or_triton_emits_none(self):
        backend = get_backend('tilelang')
        program = gemm_program(64, 64, 128)  # warp_policy_sweep defaults False
        _, options = zip(*backend._region_variant_pairs(program))
        self.assertFalse(any(o.get('warp_policy') for o in options))
        triton = get_backend('triton')
        program = gemm_program(64, 64, 128, warp_policy_sweep=True)
        _, options = zip(*triton._region_variant_pairs(program))
        self.assertFalse(any(o.get('warp_policy') for o in options))

    def test_variant_kinds_labels_warp_policy(self):
        program = gemm_program(64, 64, 128, warp_policy_sweep=True)
        backend = get_backend('tilelang')
        variants, options = zip(*backend._region_variant_pairs(program))
        kinds = _variant_kinds(list(variants), list(options), program.spec)
        for o, kind in zip(options, kinds):
            if o.get('warp_policy'):
                self.assertEqual(kind, 'warp-policy')
        self.assertIn('warp-policy', kinds)

    def test_non_gemm_program_emits_no_policy_variants(self):
        program = gemm_program(64, 64, 128, warp_policy_sweep=True)
        program = replace(program, body=Region(arguments=['a', 'b'],
                                               operations=[Operation('mul', 'out', ['a', 'b'])],
                                               yield_value='out'))
        backend = get_backend('tilelang')
        _, options = zip(*backend._region_variant_pairs(program))
        self.assertFalse(any(o.get('warp_policy') for o in options))


class EmitterTests(unittest.TestCase):
    def test_gemm_line_carries_policy(self):
        code = tilelang_code(gemm_program(64, 64, 128, warp_policy='full_row'))
        self.assertIn('policy=T.GemmWarpPolicy.FullRow', code)
        code = tilelang_code(gemm_program(64, 64, 128, warp_policy='full_col'))
        self.assertIn('policy=T.GemmWarpPolicy.FullCol', code)

    def test_square_gemm_line_is_bare(self):
        code = tilelang_code(gemm_program(64, 64, 128))
        self.assertIn("T.gemm(As, Bs, out)", code)
        self.assertNotIn('GemmWarpPolicy', code)


class SerializationTests(unittest.TestCase):
    def test_policy_and_sweep_round_trip(self):
        program = gemm_program(64, 64, 128, warp_policy='full_col', warp_policy_sweep=True)
        clone = program_from_dict(program_to_dict(program))
        self.assertEqual(clone.spec.warp_policy, 'full_col')
        self.assertTrue(clone.execution.warp_policy_sweep)

    def test_defaults_stay_out_of_dict(self):
        data = program_to_dict(gemm_program(64, 64, 128))
        self.assertNotIn('warp_policy', data['spec'])
        self.assertNotIn('warp_policy_sweep', data['execution'])

    def test_invalid_sweep_type_rejected(self):
        with self.assertRaises(ValueError):
            gemm_program(64, 64, 128, warp_policy_sweep='yes').execution.validate()


class ClassificationTests(unittest.TestCase):
    def test_warp_policy_invariance_maps_to_mismatch(self):
        msg = ('wrong result: warp-policy invariance violated in variant 3: '
               'output differs from reference')
        self.assertEqual(diagnostics.classify_root_cause(msg), 'warp_policy_mismatch')


class KnobTests(unittest.TestCase):
    def test_disable_warp_specialized_in_region_pool(self):
        # The Hopper-active warp-specialization differential rides the region
        # pass-config pool (inert on this sm_89 machine).
        self.assertIn('tl.disable_warp_specialized', knobs.TILELANG_REGION_PASS_POOL)
        self.assertIn('tl.disable_warp_specialized', knobs.TILELANG_PASS_POOL)


class ConfigDefaultTests(unittest.TestCase):
    def test_random_config_count_defaults_to_two(self):
        self.assertEqual(Config().random_config_count, 2)
        self.assertTrue(Config().region_warp_policy_pair)


if __name__ == '__main__':
    unittest.main()
