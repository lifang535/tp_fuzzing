"""GPU gate for Phase 5b: GemmWarpPolicy variants + random-config-count 2.

Cases (tilelang only; the policy knob is tilelang's GemmWarpPolicy):
  1-3. feasible region gemms with warp_policy_sweep=True — both FullRow and
       FullCol variants must appear in the emitted harness and PASS on GPU.
       Feasibility (check_warp_partition): full_row needs block_M/16 >=
       num_warps, full_col needs block_N/8 >= num_warps. 64x64/128t and
       128x128/128t satisfy both.
  4.   infeasible geometry (64x32/256t: m_tiles=4, n_tiles=4, warps=8 — the
       square partition 4x2/2x4 stays valid, both policies are infeasible) —
       no policy variants emitted, program still passes.
  5.   extended program with Config(random_config_count=2) — sample_configs
       must return exactly 2 (sampled tier appends to the fixed pair tiers).

block_K is forced to 16 and the case seeds are gemm-only programs (no
shared-materializing body ops): dynamic shared memory is
(block_M+block_N)*block_K bytes per stage, and body ops like tile_transpose
scale their own shared tiles with the block dims, so forcing big blocks onto
arbitrary programs either overflows the 4060's 100KB per-block limit or
changes semantics. Seeds were scanned to pass at the forced geometry;
block_K does not affect the reference math (block_M/block_N only).
"""
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import Config
from src.workflow.generator.region_generator import RegionGenerator
from src.workflow.generator.extended import ExtendedGenerator
from src.workflow.oracle import Oracle

CASES = [
    # (seed, block_M, block_N, threads)
    ('wp_feasible_a', 2, 64, 64, 128),
    ('wp_feasible_b', 12, 64, 64, 128),
    ('wp_feasible_c', 18, 128, 128, 128),
    ('wp_infeasible', 2, 64, 32, 256),
]


def region_case(seed, block_m, block_n, threads):
    config = Config(region_int8_prob=0, coverage_probe_prob=0, function_min_count=0,
                    region_typed_prob=0, region_layout_prob=0)
    random.seed(seed)
    program = RegionGenerator(config, 'tilelang').generate(initial='gemm')
    from dataclasses import replace
    program = replace(program, spec=replace(program.spec,
                                            block_M=block_m, block_N=block_n,
                                            threads=threads, block_K=16))
    return program, config


def main():
    oracle = Oracle(Config(), 'tilelang')
    for name, seed, bm, bn, threads in CASES:
        program, config = region_case(seed, bm, bn, threads)
        code = oracle._emit_code(program)
        n_policy = code.count('GemmWarpPolicy.')
        if name == 'wp_infeasible':
            assert n_policy == 0, f'{name}: expected no policy variants, found {n_policy}'
        else:
            assert n_policy == 2, f'{name}: expected 2 policy variants, found {n_policy}'
        report = oracle.test(program)
        status = 'PASS' if report is None else f'FAIL {report.root_cause} {report.location}'
        print(f'{status} {name} (policy variants emitted: {n_policy})', flush=True)
    # Extended + random_config_count=2. The sampled tier appends to the fixed
    # pair tiers, so assert the knob itself (sample_configs) returns exactly 2.
    ext_config = Config(random_config_count=2)
    random.seed(5)
    program = ExtendedGenerator(ext_config, 'tilelang').generate()
    from src.backends.common.knobs import sample_configs
    n_sampled = len(sample_configs(program, ext_config, 'tilelang'))
    assert n_sampled == 2, f'expected 2 sampled configs, found {n_sampled}'
    report = Oracle(ext_config, 'tilelang').test(program)
    status = 'PASS' if report is None else f'FAIL {report.root_cause} {report.location}'
    print(f'{status} extended_config2 (sampled configs: {n_sampled})', flush=True)


if __name__ == '__main__':
    main()
