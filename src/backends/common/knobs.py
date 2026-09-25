"""MLIRSmith-style random pass-pipeline sampling pools.

Version marking (see backends/common/versions.py): the pools are shared by the
target pair (tilelang 0.1.14 / triton 3.8.0 — every key below was re-checked
to still exist as a declared PassConfigKey and to still have consumers in
cuda/pipeline.py, backend/pass_pipeline/, jit/ and C++ src/transform) and the
legacy pair (tilelang 0.1.11 / triton 3.0.0). Pool membership stays static so
that sampling is a pure function of (program, config): a local Random seeded
from sha256(program sig + config.seed). Never the global random state —
evidence reads and timeout scaling re-derive extended_variants(program,
config) and must agree. `missing_pool_keys()` reports a key an installed
tilelang no longer declares; the campaign's startup banner surfaces it, and
tests/test_backend_versions.py fails on it.

Consumers are `config.get(key, default)` lookups, so a stale key is a silent
no-op (the variant compiles identically to the baseline) rather than a crash —
the failure mode to watch for is lost coverage, not false bugs.

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

# Exclusions re-checked on the target pair, so that a note from 0.1.11 is not
# mistaken for a fact about 0.1.14:
#   * tl.config_index_bitwidth (TILELANG_NUMERIC_POOL) — STILL BROKEN on
#     0.1.14. Forcing it into every sampled configuration makes
#     tilelang/extended/0 raise the same MakePackedAPI check as on 0.1.11 ("In
#     PrimFunc impl variables (limit,) are used, but are not passed in as API
#     arguments"; make_packed_api.cc line 577 on 0.1.11, line 1060 on 0.1.14),
#     while the same program passes with the knob absent. Keep it out.
#   * bfloat16 — NOT a pool edit: neither IR admits it (src/ir/ir.py's
#     DataType: float16/float32/int8; src/ir/extended.py's DTYPES:
#     float16/float32/int32/int8/bool; and the triton signature map has no
#     bf16 spelling), so it needs both whitelists, the emitters and the dtype
#     tolerances extended first. The 0.1.11 + sm_89 instability is why it was
#     left out, not the only obstacle.

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


def pool_keys() -> set:
    """Every pass-config key any pool can sample."""
    return set(TILELANG_PASS_POOL) | set(TILELANG_NUMERIC_POOL) | set(TILELANG_REGION_PASS_POOL)


def missing_pool_keys():
    """Pool keys the installed tilelang does not declare as a PassConfigKey.

    Empty on a supported release. A non-empty result means the key silently
    became a no-op — its consumer no longer reads it, so the sampled variant
    compiles exactly like the baseline and the coverage it was meant to add is
    gone. None when tilelang is not importable here (the harness machine need
    not have the DSL installed at all).
    """
    try:
        from tilelang import PassConfigKey
    except Exception:
        try:
            from tilelang.transform.pass_config import PassConfigKey
        except Exception:
            return None
    return pool_keys() - {member.value for member in PassConfigKey}


def missing_option_fields():
    """Triton compile options the installed CUDAOptions does not declare.

    The triton counterpart of missing_pool_keys(): CUDAOptions is a dataclass,
    so an unknown keyword is a TypeError at compile time rather than a silent
    no-op, but a field that vanished is just as lost as a stale pass key.
    None when triton is not importable here.
    """
    import dataclasses
    try:
        from triton.backends.nvidia.compiler import CUDAOptions
    except Exception:
        return None
    declared = {field.name for field in dataclasses.fields(CUDAOptions)}
    return {'num_warps', 'num_stages', 'maxnreg', 'enable_fp_fusion'} - declared


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
