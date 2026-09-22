# TileSmith

TileSmith generates and mutates GPU tile programs for TileLang and Triton, then checks compilation, execution, PyTorch references and repeat/configuration invariants.

The executable representations are **RegionProgram** and **ExtendedProgram**. Directed probes are whole-function Region programs. The historical TileProgram, TilePipeline and DynamicSequence implementations, operation registries, emitters and forwarding modules have been removed.

[中文](README.cn.md) · [Workflow](src/workflow/README.en.md)

## Source layout

| Location | Responsibility |
|---|---|
| `main.py` | CLI, backend loading, dump and campaign entry |
| `src/config/config.py` | Generation, mutation, limits and tolerances |
| `src/ir/ir.py` | TileKernel launch/probe parameters, dtype and scheduling enums |
| `src/ir/region.py` | Operations, lexical regions, functions and RegionProgram |
| `src/ir/region_ops.py`, `region_types.py` | Operation contracts, type inference and value pools |
| `src/ir/extended.py` | Extended types, operations, multi-result regions and validation |
| `src/ir/layout.py`, `serialization.py` | Physical layouts and current-format persistence |
| `src/backends/common/` | Shared policies, probe generation and standalone script assembly |
| `src/backends/tilelang/`, `triton/` | Target constraints, lowering and launch conventions |
| `src/workflow/generator/`, `mutator/` | Fresh generation and mutation |
| `src/workflow/emitter/` | Embedded reference interpreters and execution checks |
| `src/workflow/oracle/` | Isolated processes, timeouts, diagnostics and compiler evidence |
| `src/workflow/fuzzer/` | Campaign loop, deduplication, seeds, saving and resume |
| `src/workflow/feedback.py`, `extended_feedback.py` | Structural and compilation feedback |
| `src/workflow/coverage_audit.py` | Coverage evidence validation |
| `tests/` | Unit tests, offline compilation and GPU smoke tests |

TileKernel is a parameter object attached to a RegionProgram. Region operations and functions define its dataflow.

## Usage

Run from this directory. See [requirements.txt](requirements.txt) for dependency versions. Source generation and CPU unit tests do not require an available GPU; running kernels requires CUDA and the selected DSL.

```bash
python main.py --help
python main.py --list-kernels
python main.py --backend triton --seed 42 --dump
python main.py --backend tilelang --seed 42 -n 100 -o results
python main.py --backend triton --seed 42 -n 100 -o results

# Ordinary regions / probes / Extended only
python main.py --extended-prob 0 --probe-prob 0 -n 100
python main.py --extended-prob 0 --probe-prob 1 -n 100
python main.py --extended-prob 1 -n 100

# Extended compilation without kernel execution
python main.py --backend triton --compile-only -n 10

# Keep the original backend, seed, shape mode and generation settings
python main.py --backend triton --seed 42 --resume results/<campaign-directory> -n 100
```

`-n` counts newly executed tests, excluding deduplicated candidates. `--seed` controls generation/mutation; `--input-seed` controls tensor inputs. `--easy-shape` samples powers of two; sizes below a tile still exercise boundary handling.

## Selection and configuration

With an empty seed pool, candidates are generated fresh. Otherwise the default is **50% mutation / 50% fresh generation**, controlled by `Config.mutate_prob`.

Fresh generation selects Extended first. Remaining candidates enter the Region generator, where the probe probability is conditional. New CLI campaigns default to `extended_prob=0.25` and `coverage_probe_prob=0.20`: expected fresh-candidate proportions are 25% Extended, 15% probes and 60% ordinary regions. Deduplication and failures affect executed/passing proportions.

Library `Config()` defaults to `extended_prob=0`. CLI resume reads the saved Extended probability, falling back to 0 if absent.

| Option | Default | Effect |
|---|---:|---|
| `--backend` | tilelang | Select registered backend |
| `--extended-prob` | New CLI: 0.25 | Extended share of fresh candidates |
| `--probe-prob` | 0.20 | Conditional probe share within Region generation |
| `--gemm-prob` | 0.50 | GEMM entry versus load for ordinary regions |
| `--typed-op-prob` | 0.35 | Type/shape/memory operations; 0 generates v3 |
| `--function-min-count`, `--function-max-count` | 1, 3 | Auxiliary functions |
| `--function-call-prob` | 0.30 | Call selection when callees are available |
| `--dtype-mutate-prob` | 0.25 | Explicit storage dtype switch during mutation |
| `--local-mutate-prob` | 0.35 | Local mutation after skipping dtype mutation |
| `--region-input-seeds` | 2 | Input cases per ordinary region |
| `--region-repeat-count` | 3 | Executions per input/configuration |
| `--region-layout-prob` | 0.35 | Non-contiguous layout probability per used input |
| `--no-region-schedule-pair` | Off | Disable paired thread configurations |
| `--no-region-stage-sweep` | Off | Disable the alternate num_stages sweep for fresh regions |
| `--no-region-loop-sweep` | Off | Disable the alternate loop_kind sweep for fresh regions |
| `--no-region-layout-sweep` | Off | Disable the alternate layout pair sweep for fresh regions |
| `--extended-config-depth` | 1 | Extended configuration sweep depth: 0 = single configuration; 1 = threads/stages pair; 2 = additionally a second pass configuration |
| `--no-extended-configurations` | Off | Alias for `--extended-config-depth 0` |
| `--extended-fast-math` | Off | Additionally compile with `tl.enable_fast_math` (changes numerics) |
| `--no-extended-precision` | Off | Disable the accumulator-width sweep (fp16-accumulation copies + triton ieee→tf32) |
| `--no-extended-identities` | Off | Disable the algebraic-identity sweep (distributivity copies) |
| `--random-config-count` | 2 | Random pass-pipeline configurations sampled per extended program (deterministic per program + seed; 0 disables) |
| `--extended-atomic-prob` | 0.25 | Global-memory atomic add/max/min surface share in Extended programs |
| `--extended-fma-prob` | 0.30 | Scalar fused multiply-add chain probability in Extended programs |
| `--extended-shape-op-prob` | 0.30 | Shape primitive probability in Extended programs (flip on both DSLs; interleave/join/split on triton) |
| `--extended-int8-prob` | 0.30 | int8 x int8 matmul probability in Extended programs (int32 accumulator) |
| `--region-int8-prob` | 0.15 | Native int8 GEMM-only region probability (pre-validated spec grid) |
| `--no-region-pass-config` | Off | Disable the region pass-config invariance pair |
| `--no-region-swizzle` | Off | Disable the tilelang `T.use_swizzle` region variant pair |
| `--no-region-warp-policy` | Off | Disable the tilelang `GemmWarpPolicy` (FullRow/FullCol) region variant pair |
| `--no-instance-grids` | Off | Disable the per-(op, backend) round-robin instance grids for the new op surfaces |
| `--uncovered-boost` | 50.0 | Additive weight boost for never-attempted structural features (MLIRSmith-style diversity first; 0 restores legacy weighting) |
| `--no-structural-feedback` | Off | Disable structural feedback guidance |
| `--compile-only` | Off | Compile without execution; forces Extended probability to 1 |

See `--help` for other options and `src/config/config.py` for dimension pools, template limits, scratch budgets and tolerances.

## Generation and checking

Ordinary generation first builds operation trees and function signatures with `program_template()`. Instantiation binds operands from visible, compatible SSA values, supplies attributes and result names, and samples target-valid shapes and schedules. Helpers can call only earlier helpers. V3 uses full fp32 tiles; v4 adds fp16/fp32 values, tile/row/column/scalar shapes and tensor/buffer distinctions with scratch reads/writes.

Probes use a single `probe` node for copy, sum/max/min reduction, softmax, argmax or GEMM+argmax. They exercise physical strides, offsets, broadcasts, tails, exceptional values, repeated execution and cache reuse. Generation, mutation and persistence all use RegionProgram.

Extended has its own IR and five generation families: arithmetic, indexed_memory, shape_matmul, control_calls and mixed. Each skeleton instantiates typed operands, operations and attributes. It supports internal matmul, indexed memory, multiple region/function results, intermediate observations and paired compiler configurations. It is independent of the removed DynamicSequence implementation.

The MLIRSmith-style op-surface expansion adds new compiler code paths on top of both IRs. Extended programs gain global-memory atomics (commutative add/max/min over deliberately raced addresses), scalar FMA chains with data-dependent operands, triton shape primitives (flip/interleave/join/split) and int8 x int8 matmul with an int32 accumulator. Native regions gain transcendental elementwise ops (tanh/erf/log/log2/exp2/rsqrt/sin/cos/floor/ceil) and int8 GEMM-only programs whose spec comes from a pre-validated grid (block_K ∈ {32, 64}, int32 accumulator, exact integer reference). fp32 GEMMs are never combined with boundary step ops (ceil/floor/round/cast): TF32 tensor-core math flips rounding boundaries against the exact-fp32 reference and drowns the oracle in indistinguishable noise.

New-op attributes are sampled from bounded instance grids (`src/workflow/generator/grids.py`): per (op, backend) round-robin cursors make every corner cell appear exactly once per sweep (MLIRSmith exhaustive-instance philosophy applied to the new surfaces; legacy ops keep random sampling). Grid cursors persist in the campaign RNG state; `--no-instance-grids` restores pure random sampling.

## Diversity mechanisms and oracle dimensions

Following MLIRSmith's "one program × many configuration reuses, uncovered-feature-first, fine-grained localization", every program runs several paired checks beyond its base pipeline. The reference interpreter is schedule-independent (`_region_reference` is a pure IR interpreter), so schedule sweeps share one reference; sweeps that change numerics (accumulator width, algebraic identities) instead compute expectations per transformed program copy.

| Mechanism | Default | Checked content | Failure label (root_cause) |
|---|---|---|---|
| Uncovered-feature-first | On (`--uncovered-boost 50`) | Never-attempted structural features get a one-shot weight boost | — |
| Region schedule sweep | On (disabled by `--no-region-stage-sweep` / `--no-region-loop-sweep`) | Legal num_stages and loop_kind variants beyond the thread pair share the same reference | `stage_mismatch` / `loop_kind_mismatch` |
| Region layout sweep | On (disabled by `--no-region-layout-sweep`) | Layout programs run a second alternate layout pair; each pair compiles its own kernel set (layout constants are baked into kernel sources) | `layout_mismatch` |
| Region pass-config pair | On (disabled by `--no-region-pass-config`) | The region kernel additionally compiles through `@tilelang.jit(pass_configs=...)` with a deterministic sampled subset of the numerically-neutral region pool (knobs.py); triton launches `enable_fp_fusion=True`. The int8 region pool excludes `tirx.disable_vectorize` (de-vectorized int8 cp_async copies are rejected by tilelang codegen) | `pass_config_mismatch` |
| Region swizzle pair | On (disabled by `--no-region-swizzle`) | The tilelang gemm variant additionally annotates the kernel with `T.use_swizzle(panel_size=10)`; triton has no such knob and gets no pair | `swizzle_mismatch` |
| Region warp-policy pair | On (disabled by `--no-region-warp-policy`) | The tilelang gemm additionally compiles `GemmWarpPolicy.FullRow`/`FullCol` variants (all warps along M/N; per-tile math untouched, one shared reference), filtered per-policy by warp-partition feasibility; tilelang only | `warp_policy_mismatch` |
| Extended pass-configuration sweep | Depth 1 (`--extended-config-depth`) | Thread/stages pair (depth 1) + a second pass configuration `tl.disable_loop_unswitching` / `enable_fp_fusion` (depth 2) | `configuration_mismatch` (cross-variant consistency) |
| Random pass-pipeline sampling | 2 configs/program (`--random-config-count`, 0 disables) | Each extended program additionally compiles deterministic-random configurations: a random subset of 14 verified semantic-preserving tilelang `pass_configs` switches (plus optional numeric knobs, random threads/stages), or random `num_warps`/`num_stages`/`maxnreg` on triton; plain variants are baseline-checked | `configuration_mismatch` (cross-variant consistency) |
| Accumulator-width sweep | On (disabled by `--no-extended-precision`) | An fp16-accumulation copy of each base configuration (tilelang `T.gemm` fp16 fragment, triton `tl.dot` fp16 accumulator); the interpreter models MMA rounding per k=16; triton additionally gets an ieee→tf32 input-precision variant | `precision_mismatch` |
| Algebraic-identity sweep | On (disabled by `--no-extended-identities`) | In matmul-less programs, `mul(x, add/sub(y, z))` is rewritten into distributive form; expectations are computed on the transformed program | `algebraic_identity` |

tilelang's `opt_level` cannot penetrate `tilelang.compile` (all s_tir passes declare `opt_level=0`), so the RC2 pass-pipeline difference uses the verified `pass_configs` keys; `tl.enable_fast_math` changes numerics and stays off by default. The random-sampling pool (`src/backends/common/knobs.py`) only contains keys with verified consumers in the installed tilelang 0.1.11 (race-prone, safety-legalization-removal, Hopper-only and debug keys are excluded), and sampling is a pure function of (program signature, seed) so evidence reads and timeout scaling re-derive the same variant list.

Per-report localization is stored in summary.json's new `root_cause_locations` key (`root_cause → location → count`): locations come from the invariance label itself, the last `TILESMITH_STAGE` marker before a crash, a TVM pass name, or the reporting source file. The `root_causes` key keeps its `{str: int}` shape and `failed/` directory naming is unchanged.

Generated scripts embed inputs, references and checks. Ordinary regions check numerical results, repeated executions, paired schedules (threads, num_stages, loop_kind), layout pairs, input integrity and output guards; typed regions also guard scratch. Extended records compiler evidence and checks observations, pass configurations, randomly sampled pipelines, accumulator width, algebraic identities and scratch contents.

Feedback counts IR operations, dependencies, nesting, types, layouts and schedules, distinguishing attempts, successful executions and compilation. These are structural features, not compiler branch coverage. A failure category or numerical mismatch still requires triage before being called a compiler bug.

## Results and compatibility

Campaigns store `passed/`, `compiled/`, `failed/<root_cause>/`, Extended `artifacts/`, and summary, feedback, seed, dimension and RNG state files. Interrupted cases may also have `pending_program.pkl`. Passing and failing records include standalone Python reproducers. Region filenames use call structure plus a full-IR hash; Extended uses family plus hash. Compilation-only results are distinct from successfully executed results. summary.json additionally records `root_cause_locations` (`root_cause → location → count`) for fine-grained triage.

Region v1–v4 and Extended JSON records remain readable. Unused `legacy: null` and spec `alpha` fields in saved native records are normalized away. Nonempty legacy wrappers and historical single_op/pipeline/dynamic records are rejected; start a new campaign for those formats. Existing result/report directories and standalone reproducers are untouched.

The cleanup also removes redundant parameter sampling whose results were overwritten. Consequently the same seed need not produce the same future random stream across this change; saved native IR remains replayable. `pipeline_rtol_fp16/fp32` is renamed to `region_rtol_fp16/fp32` without changing tolerances. GEMM's `LoopKind.PIPELINED`, `num_stages` and `pipeline_stages_choices` remain active hardware scheduling options.

## Validation

```bash
python -B -m unittest discover -s tests -v
python -B tests/offline_typed_smoke.py
python -B tests/gpu_smoke.py
python -B tests/extended_smoke.py --seeds 1
python -B tests/region_int8_smoke.py           # int8 GEMM oracle runs on both backends
python -B tests/region_int8_smoke.py --compile-only
```

Unit tests exercise interpreters, types/scopes, layouts, mutation, feedback, persistence, backend dispatch and injected faults. `tests/fixtures/current_programs.json` contains 24 pre-cleanup current-format programs and function-AST digests, preserving emitted behavior through the cleanup.

Offline smoke compiles Triton PTX/cubin and performs TileLang lowering/CUDA source generation. This does not execute kernels; the three GPU smoke commands require CUDA.

To compare compilation stages of the same saved IR (requires CUDA):

```bash
python -B tests/profile_compilation.py --program <program.json> \
  --output reports/<new-experiment-directory> --repeats 3 --warm
```

Backends run sequentially with private DSL caches for each cold trial; `--warm` reuses the cache in a new process. The profiler records TileLang passes/NVCC, Triton compiler stages, imports, references and execution checks. `--single-variant` measures one compilation configuration/observation variant. Hooks are confined to experiment workers; normal campaigns are unchanged. Use `exclusive_seconds` when summing nested stages.

Measured results and bottleneck analysis are retained in the [compilation experiment report](reports/2026.09.17/compilation_profile/REPORT.md) (Chinese).
