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
    dim_range: tuple = (1, 2048)

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
    # Probability of generating a pipeline (multi-step) kernel vs single-op.
    pipeline_prob: float = 0.40
    # Probability of generating a dynamic sequence (MLIRSmith-style) vs single-op.
    # Remaining probability (1 - pipeline_prob - dynamic_prob) → single op.
    dynamic_prob: float = 0.30
    # Probability of mutating a seed from the pool vs fresh generation.
    mutate_prob: float = 0.60
    # Probability of adding a passing program to the seed pool.
    seed_add_prob: float = 0.30
    # Maximum seed pool size.
    seed_pool_max: int = 200

    # ── Pipeline generation (template-based) ───────────────────────────
    # Probability of choosing GEMM-epilogue vs elementwise-chain strategy.
    pipeline_gemm_epilogue_prob: float = 0.60
    # Max number of epilogue ops after GEMM (0 to this value, inclusive).
    pipeline_max_epilogue_ops: int = 2
    # Max number of ops in an elementwise chain (1 to this value, inclusive).
    pipeline_max_elemwise_ops: int = 2
    # Probability of adding a terminal op (reduce/softmax) at the end.
    pipeline_terminal_prob: float = 0.40

    # ── Dynamic sequence (MLIRSmith-style) ──────────────────────────────
    # Max steps in a dynamically-generated op sequence (including GEMM).
    # Increasing this produces longer chains but slows compilation.
    dynamic_max_steps_min: int = 3
    dynamic_max_steps_max: int = 8
    # Diversity boost weight for uncovered op kinds (analogous to MLIRSmith's +5*priority_base).
    diversity_boost: float = 50.0

    # ── SCALE op scalar range ───────────────────────────────────────────
    # alpha is sampled uniformly from [scale_alpha_min, scale_alpha_max].
    scale_alpha_min: float = 0.1
    scale_alpha_max: float = 10.0

    # ── Dedup and pool rotation ─────────────────────────────────────────
    # Rotate dim_pool every N iterations to explore new parameter regions.
    # Without rotation, the 20-value pool gets exhausted quickly.
    pool_rotation_interval: int = 100

    # ── Bug deduplication ───────────────────────────────────────────────
    # Maximum times the same root_cause is reported before being marked "dup".
    max_same_root_cause: int = 1e6

    # ── Oracle timeouts ─────────────────────────────────────────────────
    compile_timeout: int = 60   # seconds for compilation
    execute_timeout: int = 60   # seconds for execution

    # ── Numerics ────────────────────────────────────────────────────────
    # exp() input clamp range to prevent inf overflow.
    # float16: exp(10) ≈ 22026 < 65504 (float16 max), exp(11.09) overflows.
    # float32: exp(80) ≈ 5.5e34 < 3.4e38 (float32 max), exp(88.7) overflows.
    exp_clamp_fp16: float = 10.0
    exp_clamp_fp32: float = 80.0

    # ── Correctness thresholds (epsilon) ───────────────────────────────
    #
    # Two comparison strategies are used depending on the op:
    #
    #   Relative error:  max_diff / (mean(|ref|) + 1e-6) > threshold
    #     Used for: GEMM, reduce, pipeline chains — where the output magnitude
    #     varies widely and absolute error would be meaningless.
    #
    #   Absolute error:  max_diff > threshold
    #     Used for: copy, transpose (should be exact), elementwise ops
    #     (output is bounded and comparable in magnitude).
    #
    # GEMM (relative):
    #   float16: MMA accumulates in float32, then casts back. With K=1024 and
    #            float16 inputs, normal round-off can reach ~0.5% relative error.
    #            We allow up to 10% — anything larger indicates boundary overflow.
    #   float32: tf32 MMA has ~0.1% round-off. 5% leaves ample room.
    gemm_rtol_fp16: float = 0.10
    gemm_rtol_fp32: float = 0.05

    # Reduce ops (relative):
    #   float16 reduce_sum/max/min over a tile — cumulative rounding can be
    #   larger than elementwise ops, but still small. 10% is conservative.
    reduce_rtol: float = 0.10

    # Softmax (absolute):
    #   Output is in [0, 1], so absolute error makes sense.
    #   float16 softmax has ~0.005 absolute error in normal conditions.
    softmax_atol: float = 1e-2

    # Copy / Transpose (absolute):
    #   No arithmetic — should be bitwise exact. Only floating-point
    #   layout/precision conversions can introduce tiny errors.
    copy_atol: float = 1e-5
    transpose_atol: float = 1e-5

    # Elementwise ops (absolute): add, mul, max, sub, scale, exp, sqrt, where
    #   Single floating-point operation per element.
    #   float16: 1 ULP ≈ 0.001 near 1.0 — use 1e-3.
    #   float32: 1 ULP ≈ 1e-7 near 1.0 — use 1e-5.
    #   We use 1e-3 for both to handle float16 without special-casing.
    elemwise_atol: float = 1e-3

    # Pipeline chains (relative):
    #   Multiple ops applied sequentially amplify round-off. Use the same
    #   looser threshold as GEMM since pipelines usually start with GEMM.
    pipeline_rtol_fp16: float = 0.10
    pipeline_rtol_fp32: float = 0.05

    # ── Supported dtypes ────────────────────────────────────────────────
    # bfloat16 excluded: unstable on TileLang 0.1.11 + sm_89 (Ada Lovelace).
    supported_dtypes: List = field(default_factory=_default_dtypes)

    # ── Easy-shape mode ─────────────────────────────────────────────────
    # When enabled (--easy-shape), dim_pool is sampled from power-of-2 values
    # (256, 512, 1024, 2048) instead of arbitrary integers in [1, 2048].
    # These "nice" shapes are divisible by all common block sizes, so fewer
    # kernels hit boundary-handling code paths. Expected effect: higher pass
    # rate, useful for verifying the fuzzer itself or as a warm-up corpus.
    easy_shape: bool = False
    # The pool of "nice" shapes used in easy-shape mode.
    easy_shape_values: List[int] = field(default_factory=lambda: [
        128, 256, 512, 1024, 2048,
    ])

    # ── Hardware constraint margin ─────────────────────────────────────
    # Fraction of GPU shared memory that is safe to use per thread block.
    # TileLang and Triton both use extra shared memory internally beyond
    # what A/B tiles and accumulators require (barrier metadata, alignment
    # padding, warp-level buffers). 0.5 = use at most 50% of hardware max,
    # which eliminates shared_memory_overflow false positives in practice.
    shmem_safety_fraction: float = 0.50

    # ── Runtime ────────────────────────────────────────────────────────
    backends: List[str] = field(default_factory=lambda: ["tilelang"])
    seed: int | None = None
    output_dir: str = "results"


DEFAULT_CONFIG = Config()
