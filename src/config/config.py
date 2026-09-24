"""
Configuration for TileSmith fuzzer.
All hyperparameters are centralized here with explanatory comments.
"""

from dataclasses import dataclass, field
from typing import List


def _default_dtypes():
    # NOTE: This function is called when Config() is first instantiated.
    # Do NOT call DataType here — use plain strings to avoid any import.
    # The generator (generator.py) resolves these to DataType at runtime.
    return ["float16", "float32"]


@dataclass
class Config:
    # ── Shape generation ────────────────────────────────────────────────
    # Size of the shared dimension pool. Larger pools increase diversity
    # but reduce the probability of compatible shapes across op chains.
    dim_pool_size: int = 20
    # Dimension value range [lo, hi] for M, N, K sampling.
    dim_range: tuple = (1, 16384)

    # ── Tile size choices ───────────────────────────────────────────────
    # Valid block_M/block_N sizes for TileLang (must be multiples of 16 for MMA).
    tile_size_choices: List[int] = field(default_factory=lambda: [16, 32, 64, 128, 256])
    # Valid block_K sizes (must be multiples of 8 for stride alignment).
    block_k_choices: List[int] = field(default_factory=lambda: [8, 16, 32, 64, 128])

    # ── Pipeline staging ────────────────────────────────────────────────
    # Supported num_stages values for T.Pipelined / tl.range pipelining.
    pipeline_stages_choices: List[int] = field(default_factory=lambda: [1, 2, 3, 4])
    # Thread counts per block. Must be multiples of warp_size=32.
    thread_choices: List[int] = field(default_factory=lambda: [128, 256])

    # ── Generation strategy ─────────────────────────────────────────────
    # One recursive function-template generator. Probes are restricted whole-function templates.
    region_max_depth: int = 2
    region_min_length: int = 3
    region_max_length: int = 8
    region_max_ops: int = 24
    region_control_prob: float = 0.25
    region_input_scale: float = 0.1
    region_input_seed_count: int = 2
    region_repeat_count: int = 3
    region_schedule_pair: bool = True
    # MLIRSmith-style schedule sweep: fresh native regions are also executed
    # across alternate num_stages (region_stage_sweep), loop_kind
    # (region_loop_sweep) and physical layout pairs (region_layout_sweep)
    # configurations sharing one reference.
    region_stage_sweep: bool = True
    region_loop_sweep: bool = True
    region_layout_sweep: bool = True
    region_layout_prob: float = 0.35  # Non-contiguous layout probability per used input.
    region_gemm_prob: float = 0.50  # Otherwise start a native region with load.
    region_typed_prob: float = 0.35  # New type/shape/memory ops within ordinary regions; 0 generates v3.
    region_scratch_max_bytes: int = 64 * 1024 * 1024  # Bound generated per-block global scratch.
    latest_value_prob: float = 0.60  # Other choices reuse visible SSA values.
    local_mutate_prob: float = 0.35  # Conditional on not selecting dtype mutation.
    function_min_count: int = 1  # Auxiliary functions; the entry kernel is additional.
    function_max_count: int = 3
    function_call_prob: float = 0.30
    structural_feedback: bool = True
    # MLIRSmith DiversityCriteria-style coverage-first weighting: a structural
    # feature that has never been attempted gets this fixed additive boost on
    # top of the passed-count decay (one-shot — the boost is lost once the
    # feature has been tried, even if it failed). 0 restores legacy weighting.
    uncovered_boost: float = 50.0
    coverage_probe_prob: float = 0.20
    # Extended exploration IR: disabled by default in the library;
    # the CLI opts new campaigns into a 25% mixture.
    extended_prob: float = 0.0
    extended_configuration_pair: bool = True
    extended_observation_pair: bool = True
    # MLIRSmith-style pass configuration sweep for extended programs:
    # 0 = a single configuration (no pairs at all, the historical
    # --no-extended-configurations behavior); 1 = the threads/stages pair;
    # 2 = additionally compile with a second pass configuration
    # (tl.disable_loop_unswitching on tilelang, enable_fp_fusion on triton).
    # tilelang's opt_level knob is not plumbable through tilelang.compile
    # (every s_tir pass declares opt_level=0), so pass_configs pairs are the
    # only way to reach the RC2 pass-pipeline bug class.
    extended_config_depth: int = 1
    extended_fast_math_pair: bool = False  # tl.enable_fast_math pair, changes numerics
    # MLIRSmith-style accumulator-width sweep (RC5): per base configuration an
    # fp16-accumulation copy of the program (tilelang T.gemm fp16 fragment,
    # triton tl.dot fp16 accumulator) checked against its own interpretation,
    # plus a triton ieee->tf32 input-precision variant. Enabled by default
    # after the GPU smoke verified fp16 accumulation on this machine; matmul-
    # less programs produce no variants.
    extended_precision_pair: bool = True
    # MLIRSmith-style algebraic-identity sweep (RC5): a distributivity copy of
    # each matmul-less extended program (mul(x, add/sub(y, z)) rewritten into
    # add/sub(mul(x, y), mul(x, z))) checked against its own interpretation.
    # Programs without a matching float pattern produce no variants.
    extended_identity_pair: bool = True
    # MLIRSmith-style random pass-pipeline sampling: each extended program is
    # additionally compiled with this many deterministic-random compiler
    # configurations (random subsets of the verified semantic-preserving
    # pass_configs pool, see backends/common/knobs.py), checked as plain
    # variants against the shared reference baseline (configuration_mismatch).
    # 0 disables. Sampling is a pure function of (program, seed). Raised to 2
    # (2026-09-21): one sample rarely covers the pool; the two samples are
    # deterministic per program, so campaign cost grows ~linearly.
    random_config_count: int = 2
    # ── New op surfaces ─────────────────────────────────────────────────
    # Bounded instance grids (Track C): per-(op, backend) round-robin cursors
    # make every corner of each new op's attribute domain appear exactly once
    # per round. When off, corners are sampled uniformly at random.
    instance_grid: bool = True
    # Global-memory atomics in extended programs (commutative scratch races).
    extended_atomic_prob: float = 0.25
    # Scalar fused multiply-add chains in extended programs.
    extended_fma_prob: float = 0.30
    # Triton shape primitives (flip everywhere; join/split/interleave triton).
    extended_shape_op_prob: float = 0.30
    # int8 x int8 matmul (int32 accumulator) in extended matmul programs.
    extended_int8_prob: float = 0.30
    # int8 + int8 GEMM-only native region programs (exact int32 reference).
    region_int8_prob: float = 0.15
    # Region pass-config variant pair (tilelang get_tir + tilelang.compile;
    # triton enable_fp_fusion), the tilelang T.use_swizzle variant pair and
    # the tilelang GemmWarpPolicy pair (FullRow / FullCol — the sm_89-active
    # warp-level knob; warp *specialization* itself is TMA-gated on sm_90+).
    region_pass_config: bool = True
    region_swizzle_pair: bool = True
    region_warp_policy_pair: bool = True
    compile_only: bool = False
    save_artifacts: bool = True  # Persist Extended evidence; otherwise use temporary files.
    probe_repeat_count: int = 3
    probe_schedule_pair: bool = True
    probe_cache_cycle: bool = True
    # With a non-empty seed pool, choose mutation and fresh generation equally.
    mutate_prob: float = 0.50
    # Within mutation, explicitly switch storage dtype while retaining the IR.
    # Used only when supported_dtypes contains a different dtype.
    dtype_mutate_prob: float = 0.25
    # Probability of adding a passing program to the seed pool.
    seed_add_prob: float = 0.30
    # Maximum seed pool size.
    seed_pool_max: int = 200

    # ── SCALE op scalar range ───────────────────────────────────────────
    # alpha is sampled uniformly from [scale_alpha_min, scale_alpha_max].
    scale_alpha_min: float = 0.1
    scale_alpha_max: float = 10.0

    # ── Dedup and pool rotation ─────────────────────────────────────────
    # Rotate dim_pool every N iterations to explore new parameter regions.
    # Without rotation, the 20-value pool gets exhausted quickly.
    pool_rotation_interval: int = 100

    # ── Bug deduplication ───────────────────────────────────────────────
    # Maximum reproducers saved per root_cause; 0 (the default) means no cap.
    # Every occurrence counts in summary.json root_causes either way — this
    # only gates *saving* (failed/{root_cause}/*.py,*.json). Coarse labels
    # merge distinct defects: unclassified compiler diagnostics all collapse
    # to 'other', so a per-label cap silently drops real bugs past the tenth
    # one. Set a positive value only to throttle a systematic front-end
    # rejection (a version-gapped DSL binding, a cache collision) that would
    # otherwise flood the output directory.
    max_same_root_cause: int = 0

    # Reproducers kept for the oracle trust gate; 0 (the default) means no cap,
    # the same convention as max_same_root_cause. These are not bugs — the
    # reference disagrees with its own fp64/jittered copy, so no kernel could
    # pass the numeric check — but each one is a distinct program whose numeric
    # check was skipped, and the mix of structures that reach the gate is only
    # auditable if the samples survive the run. The counter behind this is
    # global, not per root cause, so the two knobs never interact.
    max_oracle_unstable_saved: int = 0

    # ── Oracle timeouts ─────────────────────────────────────────────────
    compile_timeout: int = 60   # seconds for compilation
    execute_timeout: int = 60   # seconds for execution

    # ── Correctness thresholds ─────────────────────────────────────────
    # GEMM-entry regions compare relative error; load-entry regions compare
    # absolute error. Probes use operation-specific checks (copy is bitwise).
    # Reduce ops (relative):
    #   float16 reduce_sum/max/min over a tile — cumulative rounding can be
    #   larger than elementwise ops, but still small. 10% is conservative.
    reduce_rtol: float = 0.10

    # Softmax (absolute):
    #   Output is in [0, 1], so absolute error makes sense.
    #   float16 softmax has ~0.005 absolute error in normal conditions.
    softmax_atol: float = 1e-2

    # Elementwise ops (absolute): add, mul, max, sub, scale, exp, sqrt, where
    #   Single floating-point operation per element.
    #   float16: 1 ULP ≈ 0.001 near 1.0 — use 1e-3.
    #   float32: 1 ULP ≈ 1e-7 near 1.0 — use 1e-5.
    #   We use 1e-3 for both to handle float16 without special-casing.
    elemwise_atol: float = 1e-3

    # GEMM-entry Region programs (relative error).
    region_rtol_fp16: float = 0.10
    region_rtol_fp32: float = 0.05

    # ── Supported dtypes ────────────────────────────────────────────────
    # bfloat16 excluded: unstable on TileLang 0.1.11 + sm_89 (Ada Lovelace).
    supported_dtypes: List = field(default_factory=_default_dtypes)

    # ── Easy-shape mode ─────────────────────────────────────────────────
    # When enabled (--easy-shape), dim_pool is sampled from power-of-2 values
    # in 1..16384. Shapes below a tile still exercise boundary handling.
    easy_shape: bool = False
    # The pool of "nice" shapes used in easy-shape mode.
    easy_shape_values: List[int] = field(default_factory=lambda: [
        1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384,
    ])

    # ── Hardware constraint margin ─────────────────────────────────────
    # Fraction of GPU shared memory that is safe to use per thread block.
    # TileLang and Triton both use extra shared memory internally beyond
    # what A/B tiles and accumulators require (barrier metadata, alignment
    # padding, warp-level buffers). 0.5 = use at most 50% of hardware max,
    # which eliminates shared_memory_overflow false positives in practice.
    shmem_safety_fraction: float = 0.50

    # ── Historical trigger sampling ────────────────────────────────────
    # Probability that region spec sampling skips schedule/shared-memory
    # pre-validation and draws tile/thread/stage combinations directly from
    # the configured choices. The pre-validation filters are conservative:
    # they exclude exactly the combinations that historically reached the
    # warp_partition and shared_memory_overflow bug classes, so this mode
    # keeps those classes generable. 0.0 = always validate.
    unchecked_spec_prob: float = 0.30
    # Probability of forcing a boundary shape after tile selection. Random
    # dim pools almost never produce a shape smaller than its tile, and that
    # boundary is exactly where the historical ptx_async_boundary crashes
    # (tilelang cp.async byte-width) and other tail-handling bugs live.
    # GEMM-entry programs (which carry the cp.async trigger) get twice this
    # probability, and the bias branch completes the trigger shape: a tail of
    # exactly one element (M=1 or K=1), with a preference for fp16 + pipelined.
    boundary_shape_prob: float = 0.10

    # ── Runtime ────────────────────────────────────────────────────────
    backends: List[str] = field(default_factory=lambda: ["tilelang"])
    # Tensor inputs use an independent seed, embedded in every saved test.
    input_seed: int = 0
    seed: int | None = None
    output_dir: str = "results"


DEFAULT_CONFIG = Config()
