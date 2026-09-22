"""MLIRSmith-style random pass-pipeline sampling pools.

Keys are verified against the installed tilelang 0.1.11 (CUDA pipeline
consumers in cuda/pipeline.py, backend/pass_pipeline/pipeline_utils.py,
engine/lower.py, jit/adapter/libgen.py, and C++ src/transform) and triton
3.0.0 CUDAOptions. Sampling is a pure function of (program, config): a local
Random seeded from sha256(program sig + config.seed). Never the global random
state — evidence reads and timeout scaling re-derive extended_variants(
program, config) and must agree.

Excluded by policy: tl.enable_fast_math (changes numerics),
tl.disable_thread_storage_sync (race-prone), tl.disable_safe_memory_legalize
(removes safety legalization), tl.disable_wgmma / tl.disable_tma_lower
(Hopper-only; this machine is sm_89), every debug/dump key, and
tl.device_compile_flags (arbitrary injection). The deterministic sweep tiers
(tirx.disable_vectorize for depth 1, tl.disable_loop_unswitching for depth 2)
are also excluded so a sampled config always exercises a fresh combination.
"""
import hashlib
import random

# Concurrent device compiles per harness (region harnesses carry the same
# cap as the compile_threads local inside _run_region in
# workflow/emitter/region_checks.py). Cold
# kernel compilation dominates the oracle wall time and is CPU-bound
# (nvcc subprocesses), so variant kernels compile on a small thread pool
# while GPU execution stays serial.
EXTENDED_COMPILE_THREADS = 8

# Boolean pass_configs switches. Unknown keys can never error (every consumer
# uses config.get(key, default)), so pool membership is not a crash risk.
TILELANG_PASS_POOL: tuple = (
    'tl.disable_warp_specialized',
    'tl.disable_data_race_check',
    'tl.force_let_inline',
    'tl.enable_aggressive_shared_memory_merge',
    'tl.disable_shared_memory_reuse',
    'tl.loop_unswitching_allow_non_trivial_else',
    'tl.storage_rewrite_detect_inplace',
    'tl.enable_async_copy',
    'tl.disable_vectorize_256',
    'tl.disable_shuffle_elect',
    'tl.enable_lower_ldgstg',
    'tl.enable_lower_ldgstg_predicated',
    'tl.if_stmt_binding_inline_replayable_binds',
    'tl.disable_out_of_bound_warning',
)

# Numeric knobs with their verified-safe domains. tl.config_index_bitwidth is
# EXCLUDED: on tilelang 0.1.11 its very presence makes MakePackedAPI raise
# "impl variables (limit, steps) are used, but are not passed in as API
# arguments" on every extended program (any value) -- a universal breaker
# that would flood failed/ with duplicate signatures and skip execution.
TILELANG_NUMERIC_POOL: dict = {
    'tl.ptxas_register_usage_level': (0, 1, 2, 3),
}

# 1 warp is omitted: small internal matmuls need >= 2 warps on sm_89.
TRITON_WARPS: tuple = (2, 4, 8)
TRITON_STAGES: tuple = (1, 2, 3, 4)
# Below 128 registers ptxas fails complex kernels spuriously.
TRITON_MAXNREG: tuple = (128, 168)

# Region pass-config pairs keep a tighter pool than extended sampling: the
# checked harness compares every variant against one shared reference, so a
# key must be numerically neutral (pure compile-structure changes). The two
# proven extended sweep tiers plus allocation/pipelining rewrites qualify;
# tl.enable_fast_math and every storage-safety toggle do not.
# tl.disable_warp_specialized: the warp-specialization differential. Inert on
# this sm_89 machine (ws requires TMA -> sm_90+), but on Hopper-class GPUs it
# disables the automatic producer/consumer warp split of pipelined gemms —
# exactly the kind of warp-level structural probe MLIRSmith's tuning aims at,
# and still numerically neutral.
TILELANG_REGION_PASS_POOL: tuple = (
    'tirx.disable_vectorize',
    'tl.disable_loop_unswitching',
    'tl.enable_async_copy',
    'tl.enable_aggressive_shared_memory_merge',
    'tl.disable_shared_memory_reuse',
    'tl.disable_warp_specialized',
)


def region_pass_configs(program, config) -> dict:
    """Deterministic 1-3 key subset of TILELANG_REGION_PASS_POOL for the
    checked harness's pass-config invariance pair.

    Pure in (program, config) like sample_configs: evidence reads and timeout
    scaling re-derive region variants and must agree.
    """
    pool = TILELANG_REGION_PASS_POOL
    from src.ir import DataType
    if program.spec.dtype == DataType.INT8:
        # tirx.disable_vectorize de-vectorizes the shared-memory copy, and an
        # int8 cp_async transfer then falls to a 1-byte width, which tilelang's
        # CUDA codegen rejects ({4, 8, 16} required) -- a guaranteed compile
        # crash on every int8 draw, not an occasional bug.
        pool = tuple(k for k in pool if k != 'tirx.disable_vectorize')
    digest = hashlib.sha256(
        _program_sig(program).encode('utf-8') + b'|' + str(config.seed).encode('utf-8')).digest()
    rng = random.Random(digest)
    return {name: True for name in rng.sample(pool, rng.randint(1, min(3, len(pool))))}


def _program_sig(program):
    from src.workflow.fuzzer.fuzzer import TileSmith
    # _make_sig returns a (type, canonical-json) tuple; repr it like the
    # fuzzer's filename digests do.
    return repr(TileSmith._make_sig(program))


def sample_configs(program, config, backend: str) -> list:
    """Deterministic random compiler configurations for one extended program.

    Returns [] when config.random_config_count == 0. Pure in (program, config):
    the RNG seed derives only from the program signature and config.seed.
    """
    count = getattr(config, 'random_config_count', 0) or 0
    if not count:
        return []
    digest = hashlib.sha256(
        _program_sig(program).encode('utf-8') + b'|' + str(config.seed).encode('utf-8')).digest()
    rng = random.Random(digest)
    return [_sample_tilelang(program, rng) if backend == 'tilelang' else _sample_triton(rng)
            for _ in range(count)]


def _sample_tilelang(program, rng) -> dict:
    # Matmul thread floor mirrors backend.py: a warp needs a 16x16 output tile.
    products = [n.results[0].type for n in program.all_operations() if n.op == 'matmul']
    threads = min([128] + [max(32, t.size // 256 * 32) for t in products]) if products \
        else rng.choice((128, 256))
    # A random subset of the boolean pool (the pass-pipeline bug class lives in
    # combinations) plus at most one numeric knob.
    passes = {name: True for name in TILELANG_PASS_POOL if rng.random() < 0.5}
    if rng.random() < 0.5:
        name, values = rng.choice(tuple(TILELANG_NUMERIC_POOL.items()))
        passes[name] = rng.choice(values)
    if not passes:
        passes = {'tl.enable_async_copy': True}  # must differ from the base configuration
    return {'threads': threads, 'stages': rng.choice((1, 2, 3)), 'pass_configs': passes}


def _sample_triton(rng) -> dict:
    return {'num_warps': rng.choice(TRITON_WARPS),
            'num_stages': rng.choice(TRITON_STAGES),
            'enable_fp_fusion': False,  # pin, matching the base configurations
            'maxnreg': rng.choice(TRITON_MAXNREG)}
