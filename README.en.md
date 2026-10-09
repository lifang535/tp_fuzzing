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
| `src/ir/slice.py`, `src/workflow/slices/` | Feature-slice programs, their generators, exact references, reduction and scheduler |
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

Run from this directory. This branch targets **TileLang 0.1.14**, **Triton 3.8.0** and **PyTorch 2.4.0+cu124** (Python 3.11 in the two audited `tp_fuzzing_latest` server environments). The checked-in [requirements.txt](requirements.txt) still pins the *older* TileLang 0.1.11 / Triton 3.0.0 pair; do not use it unchanged to reproduce experiments on this branch. The `+cu124` PyTorch wheel also needs the appropriate PyTorch CUDA wheel index rather than a generic PyPI mirror. Check the active environment before running:

```bash
python -c "import sys; from importlib.metadata import version; print(sys.version.split()[0], {name: version(name) for name in ('torch', 'tilelang', 'triton')})"
```

The fuzzer records installed versions in each campaign's `summary.json` and warns on a changed environment when resuming. Source generation and CPU unit tests do not require an available GPU; executing kernels requires CUDA and the selected DSL.

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

# Avoid retaining large Extended compilation artifacts; results are still saved
python main.py --backend triton --seed 42 -n 100 --no-save-artifacts
```

`-n` counts newly executed tests, excluding deduplicated candidates. `--seed` controls generation/mutation; `--input-seed` controls tensor inputs. `--easy-shape` samples powers of two; sizes below a tile still exercise boundary handling.

Both built-in backends use the shared IR and campaign loop. TileLang and Triton differ in accepted configurations, lowering, launch conventions and diagnostic rules under `src/backends/`. To add another DSL, implement and register the interface in `src/backends/base.py`, then load its registration module with `--backend-plugin MODULE`; see the [workflow guide](src/workflow/README.en.md). Supporting a new IR format additionally requires generation, serialization and feedback work.

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
| `--extended-elementwise-prob` | 0.50 | Compose one common float32 math op into an Extended program; the `elementwise` family always selects three |
| `--extended-fma-prob` | 0.30 | Scalar fused multiply-add chain probability in Extended programs |
| `--extended-shape-op-prob` | 0.30 | Shared `flip` probability in fresh Extended seeds; Triton-only shape operations are generated by the separate extension stage |
| `--extended-int8-prob` | 0.30 | int8 x int8 matmul probability in Extended programs (int32 accumulator) |
| `--region-int8-prob` | 0.15 | Native int8 GEMM-only region probability (pre-validated spec grid) |
| `--no-region-pass-config` | Off | Disable the region pass-config invariance pair |
| `--no-region-swizzle` | Off | Disable the tilelang `T.use_swizzle` region variant pair |
| `--no-region-warp-policy` | Off | Disable the tilelang `GemmWarpPolicy` (FullRow/FullCol) region variant pair |
| `--no-instance-grids` | Off | Disable the per-(op, backend) round-robin instance grids for the new op surfaces |
| `--uncovered-boost` | 50.0 | Additive weight boost for never-attempted structural features (MLIRSmith-style diversity first; 0 restores legacy weighting) |
| `--no-structural-feedback` | Off | Disable structural feedback guidance |
| `--compile-only` | Off | Compile without execution; forces Extended probability to 1 |
| `--slice-prob` | New CLI: 0.4 | Probability that a test is a feature-slice program (0 when resuming a campaign without slices) |
| `--slices` | All | Comma-separated subset of `cast,reduce,scan,gemm,atomic` |
| `--no-slice-adaptive` | Off | Draw slices and knob assignments uniformly instead of by discovery estimate and uncovered knob pairs |

See `--help` for other options and `src/config/config.py` for dimension pools, template limits, scratch budgets and tolerances.

## Generation and checking

Ordinary generation first builds operation trees and function signatures with `program_template()`. Instantiation binds operands from visible, compatible SSA values, supplies attributes and result names, and samples target-valid shapes and schedules. Helpers can call only earlier helpers. V3 uses full fp32 tiles; v4 adds fp16/fp32 values, tile/row/column/scalar shapes and tensor/buffer distinctions with scratch reads/writes.

Probes use a single `probe` node for copy, sum/max/min reduction, softmax, argmax or GEMM+argmax. They exercise physical strides, offsets, broadcasts, tails, exceptional values, repeated execution and cache reuse. Generation, mutation and persistence all use RegionProgram.

Extended has its own IR and six generation families: arithmetic, indexed_memory, shape_matmul, control_calls, elementwise and mixed. Each skeleton instantiates typed operands, operations and attributes. The elementwise family cycles through 18 common float32 math operations on each backend. It supports internal matmul, indexed memory, multiple region/function results, intermediate observations and paired compiler configurations. It is independent of the removed DynamicSequence implementation. This is bounded operation-family coverage, not every TileLang or Triton API, dtype, shape, or hardware feature.

### Shared seeds and target-specific extension

Fresh Extended campaigns generate only shared IR structure. Triton-only `join/split/interleave` and pipelined `for` are reserved for the extension stage; `mixed` uses the same structure on both backends. `--legacy-extended-mix` restores the historical mixed generator for older experiment settings. In a new `main.py` campaign, `--dsl-extend-prob` defaults to 0.35: after at least one common Extended program passes execution, each new case has that probability of deriving a target-specific program from the bounded passing-source pool. Otherwise the normal Region/Extended generator runs (`--extended-prob` defaults to 0.25 within that route). The scheduler first covers untried operations and bounds each source/operation pair by `--dsl-source-variants`; adaptive scheduling then allocates attempts by recent outcomes, while the fixed policy continues balancing attempt counts; exhausted pools fall back to common generation. Among eligible parents it favors rare observed compiler-artifact stages, neighbouring IR operations, and compiler-setting/operation combinations. Only common passing programs enter the common mutation/source pools; target-specific compilation feedback has a separate ledger in `dsl_stage.json` and cannot steer common-IR generation. Repeated manually confirmed failure signatures reduce the weight of their parent seed, with a nonzero exploration floor; failures are still saved, and broad directory labels never trigger penalties. `--no-structural-feedback` disables this seed guidance. These artifact features are proxies, not measured compiler-pass or branch coverage; Region programs currently have structural feedback only. `--dsl-extend-prob 0` disables the integrated stage. The campaign summary records `dsl_extension.by_op` and `confirmed_failure_signatures`; restored baselines are revalidated before reuse.

Passing DSL derivatives now enter a separate target corpus. Within the DSL route, `--dsl-evolve-prob` (default 0.5) selects this corpus for another target operation, a local mutation that preserves target operations, or a bounded loop around a shape-preserving target operation. Composition follows the previous checked target output. `--dsl-max-depth` defaults to 3; terminal descendants remain saved but do not displace mutable ancestors. Restored target parents are revalidated. Their lineage records the immediate and original parent hashes, derivation depth, action and checked output. `--dsl-evolve-prob 0` restores one-step extension.

`--corpus-feedback` (default on) evicts redundant seeds before rare-feature representatives in bounded common and target pools; `--no-corpus-feedback` restores random replacement. Old CLI campaigns resume with evolution and corpus retention guidance disabled unless their saved settings or explicit flags enable them. Native and DSL candidates share quarantine admission and learning, with sampling and a forced final retry. Wrong-result diagnostic buckets never quarantine or demote structural features; unstable-oracle outcomes do not teach quarantine rules. `coverage_progress.json` and `coverage_progress.jsonl` record live counters every 100 tested programs and at shutdown. These counters measure structural/compiler-artifact features, not compiler branch coverage or independent bugs.

New CLI campaigns enable `--dsl-adaptive-schedule`. Older resumes preserve their saved policy; append that flag to the original resume command to opt in. `--no-dsl-adaptive-schedule` disables action adaptation, and `--no-structural-feedback` disables both seed and action guidance. The scheduler tracks recent passing source novelty, observed compiler-artifact novelty, oracle duration and repeated audited signatures per action, operation and immediate parent. It uses moving averages, bounded cost corrections and uniform exploration rising linearly from 10% to 50% over 256 consecutive unrewarded tests, resetting on useful novelty. Audited repeats receive zero reward even when they expose new lexical compiler artifacts. Unknown diagnostics and wrong-result buckets receive no audited-repeat penalty. Both common and target parents now apply the existing confirmed-failure seed penalty; failure saving is unchanged.

New CLI campaigns use `--dsl-source-variants 4`; older resumes default to 1. A passing common parent can produce bounded alternative derivatives even after its first derivative fails, exploring different axes, directions or shapes. Deterministic operations keep a single attempt. With `--dsl-attributes` and instance grids enabled, legal attributes rotate by operation, operand type and legal domain; `--no-instance-grids` retains random attribute sampling. Respelling excludes unchanged attributes and preserves result types and top-k sizes. Already-tested complete programs are rejected within bounded DSL generation retries, then generation falls back to the common route. `duplicate_derivatives` counts these rejections separately from executions and bugs. Variant attempt counts and attribute cursors survive resume.

The DSL ledger also records artifacts observed before a later failure, while `compiler_passed` separately counts evidence from passing executions. Failed programs never enter the passing corpus. A feature's first passing representative can retain a seed even if it previously appeared in a failed program. `dsl_stage.json` persists parent/action outcomes and reads older states. The summary's `dsl_extension.schedule` and progress log's `dsl_schedule` expose `tested`, `novel`, `known_repeats` and recent reward/cost. Schedule snapshots also persist `stagnant` and the current `exploration` probability for plateau recovery. `novel` counts tests adding these proxy features, not independent bugs. Action and `action:operation` aggregates overlap; use `total` for overall counts. `seconds` is a moving average of oracle time, excluding generation and parent revalidation.

A short matched-parent GPU ablation is available as `python tests/dsl_evolution_comparison.py --backend triton --passed-dir results/<campaign>/passed --output /tmp/dsl-comparison`. Both arms use the same revalidated parents and compiler/oracle settings, with alternating arm order across seeds. The default short check omits precision/identity/random-config sweeps; `--full-oracle` includes them. Results retain source/compiler feature sets, failures and lineage. Add `--comparison schedule` to compare fixed/adaptive scheduling with identical evolution, corpus retention and oracle settings. The default `evolution` comparison disables action adaptation in both arms. This short check uses equal test counts; use multiple seeds and controlled hardware/time budgets for claims about bug yield.

```bash
python main.py --backend triton --extended-prob 1 -n 10000
# Integrated common generation and target-specific derivatives:
python main.py --backend triton --extended-prob 0.5 --dsl-extend-prob 0.4 -n 10000
# Standalone extension of an existing passed/ corpus remains available:
python extend.py --backend triton --passed-dir results/<campaign>/passed -n 10000
python extend.py --backend tilelang --passed-dir results/<campaign>/passed -n 10000
```

The extension input is an **already instantiated ExtendedProgram JSON that passed execution**, not an IR template. Compile-only and Region records are ignored. The source is rerun for each input seed before a target-specific derivative is tested. A separate result directory contains `passed/`, `failed/`, and `summary.json`; derived JSON records the parent path/hash, target operation, and input seed. Use `--op` to focus on one entry in the target-operation table below. For large corpora, at most 5,000 source/operation pairs are reservoir-sampled into memory by default (`--max-sources 0` removes this limit). All failures are retained; by default only 20 passing reproducers per operation are saved (`--max-passed-saved 0` retains all), with complete counts in `summary.json`. An extension failure still requires manual confirmation before counting it as a DSL bug.

| Target | `--op` choices in the extend stage | Checked behavior |
| --- | --- | --- |
| Triton | `join`, `split`, `interleave`, `scan_sum`, `scan_product`, `sort`, `histogram`, `argmax`, `argmin`, `xor_sum`, `dsl_sigmoid`, `dsl_clamp`, `softmax`, `topk`, `gather`, `atomic_and`, `atomic_or`, `atomic_xor` | Shape transforms, scans/sort, integer reductions, elementwise math, bitwise atomic updates; `topk`/`gather` require an installed Triton exporting them |
| TileLang | `pipelined_for`, `scan_sum`, `scan_max`, `reduce_abssum`, `reduce_absmax`, `reduce_bitand`, `reduce_bitor`, `reduce_bitxor`, `dsl_sigmoid`, `dsl_clamp` | Pipelined loop, scans, reductions, elementwise math |

Numerical extensions keep the parent's checked outputs and add an independent output from the target primitive. Floating inputs are derived from a bounded parent result. Histogram and bitwise-reduction inputs use exact integer values from the parent when available, or runtime-parameter-dependent integer indices otherwise; initial histogram inputs stay in 0–15. Mutations may introduce out-of-range values; the reference ignores negative values and values >=16, matching Triton. Rank-2 Triton row softmax keeps the reduced dimension for correct broadcasting, including when re-emitting older saved programs. This avoids turning tolerated parent floating-point roundoff into a false discrete-result failure. These entries are a bounded subset of the DSL APIs. In particular, version- or hardware-specific asynchronous/TMA operations, arbitrary user-defined scan combiners, and unrestricted dtype/shape combinations are not claimed as covered.

To track the full installed CUDA language facade without conflating a name with execution coverage, run `python api_coverage.py --backend triton --summary results/<extension-run>/summary.json --passed-code-dir results/<common-run>/passed --output triton_api.json` (or use `tilelang`). The inventory records the installed version and each public language-owned callable. `executed_pass` requires at least one passing extension campaign case; `seen_in_passing_code` records a direct DSL call in a saved passing `.py`; `implemented_no_run` means code exists without campaign evidence; `unmapped` means the inventory has no evidence, which does not imply other fuzzer stages never call it. One passing case proves neither all parameter combinations nor hardware architectures. Run the inventory in the target Conda environment to audit Triton 3.8 and TileLang 0.1.14 rather than the local development versions.

The MLIRSmith-style op-surface expansion adds new compiler code paths on top of both IRs. Extended programs gain global-memory atomics (commutative add/max/min over deliberately raced addresses), scalar FMA chains with data-dependent operands, triton shape primitives (flip/interleave/join/split) and int8 x int8 matmul with an int32 accumulator. Native regions gain transcendental elementwise ops (tanh/erf/log/log2/exp2/rsqrt/sin/cos/floor/ceil) and int8 GEMM-only programs whose spec comes from a pre-validated grid (block_K ∈ {32, 64}, int32 accumulator, exact integer reference). fp32 GEMMs are never combined with boundary step ops (ceil/floor/round/cast): TF32 tensor-core math flips rounding boundaries against the exact-fp32 reference and drowns the oracle in indistinguishable noise.

New-op attributes are sampled from bounded instance grids (`src/workflow/generator/grids.py`): per (op, backend) round-robin cursors make every corner cell appear exactly once per sweep (MLIRSmith exhaustive-instance philosophy applied to the new surfaces; legacy ops keep random sampling). Grid cursors persist in the campaign RNG state; `--no-instance-grids` restores pure random sampling.

## Feature slices

After weeks of campaigns the Region/Extended routes kept re-finding the same few mechanisms (Chao1 ≈ observed buckets): their scheduler can only reorder programs the generators can express, and both IRs stop at fp16/fp32/int8/int32/bool, rank ≤ 2 and built-in combine functions. Following feature-focused test generation (FFTG, Zamudio Amaya et al., ASE'26) and the tile-program bug study (Rathnasuriya et al., ISSTA'26: type and operator handling are 49% of the 301 studied codegen bugs), `src/workflow/slices/` adds focused generators, one per bug-dense feature. Each owns a discrete knob space whose other dimensions stay simple:

| Slice | Focus | Knobs (abridged) |
|---|---|---|
| `cast` | conversion/elementwise chains | three steps of (op, dtype) over bf16, f16, f32, f64, fp8 e4m3/e5m2, i8–i64, u8–u32; rank 1–3; ragged tails; dynamic extents |
| `reduce` | reductions | sum/max/min/argmax/argmin/xor_sum and user `tl.reduce` combines incl. tuple (value, index) and (min, max) pairs on Triton; `T.reduce_*` with clear/batch/nan_propagate/shared sources on TileLang |
| `scan` | scans | cumsum/cumprod, user `associative_scan` combines incl. a non-commutative linear recurrence; reverse; `T.cumsum/T.cummax` in place or into a separate buffer |
| `gemm` | matrix multiply | MMA dtype (f16, bf16, fp8, i8, f32 with ieee/tf32/tf32x3), accumulator, transposed operands, batched (3-D) `tl.dot`, K loop and stages; TileLang register operand, k_pack, warp policy, clear_accum, serial/pipelined loop |
| `atomic` | global atomics | dtype × add/max/min/and/or/xor/xchg (Triton sem/scope) or addx2/addx4 (TileLang) × slot contention × masking |
| `round` | rounding conversions | float/integer source × destination, optionally through a wider format, on ties, near-ties, subnormals and range edges; Triton `fp_downcast_rounding='rtz'`; TileLang global/fragment/shared staging |

Inputs are dyadic rationals k·2^-f with small |k|, and every step transfers a static value domain; a knob value whose result would not be exactly representable is legalized away (YARPGen-style range tracking). The float64/int64 reference is therefore exact and outputs are compared bit for bit, so a mismatch is never rounding noise. The `round` slice instead feeds inexact values on purpose and compares with a correctly rounded reference implemented in the harness (`_slice_round`; it agrees with numpy, whereas torch rounds float64 conversions twice through float32). Each harness also checks a guarded output allocation, run-to-run determinism and a second launch configuration. Front-end rejections of features the installed DSL does not support are marked and classified `unsupported_feature`, not as failures.

The scheduler (`slices/scheduler.py`) draws slices by an incidence Good-Turing estimate of new failure buckets and new knob-pair cells per second (STADS, Böhme TOSEM'18) after a round-robin warm-up, and picks within a slice the candidate covering the most uncovered knob pairs (AETG-style 2-wise coverage) or a 1–2 knob mutation of an interesting earlier program. A failure is reduced knob by knob to its core (`slices/minimize.py`, at most 24 extra tests each and about a quarter of the slice route overall; at most two cores per crash bucket): a wrong result is bucketed by its reduced dtype path, since every wrong value shares the checker's message, and later candidates containing the core of a well-sampled bucket are skipped with probability 1 − max(0.02, 3/hits). The reduced program is saved next to the failure as `*.min.py`; its JSON records the core. `summary.json` reports per-slice tests, buckets, cores, Good-Turing/Chao1 and avoided candidates under `slices`, and `slice_state.json` restores the scheduler on resume. Failure keys now keep the failing MLIR pass with its first diagnostic and the first nvcc error, so `PassManager::run failed` and TileLang CUDA compilation failures split by mechanism. `route_stats` in `summary.json` and the progress files records tests and oracle seconds per generation route (fresh, mutate, dsl_extend, slice, slice_reduction), and every first bucket sighting records the route that found it, so discovery per route-hour can be compared.

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

tilelang's `opt_level` cannot penetrate `tilelang.compile` (all s_tir passes declare `opt_level=0`), so the RC2 pass-pipeline difference uses the verified `pass_configs` keys; `tl.enable_fast_math` changes numerics and stays off by default. The random-sampling pool (`src/backends/common/knobs.py`) only contains keys with consumers re-checked for the target TileLang 0.1.14 (race-prone, safety-legalization-removal, Hopper-only and debug keys are excluded), and sampling is a pure function of (program signature, seed) so evidence reads and timeout scaling re-derive the same variant list.

Per-report localization is stored in summary.json's new `root_cause_locations` key (`root_cause → location → count`): locations come from the invariance label itself, the last `TILESMITH_STAGE` marker before a crash, a TVM pass name, or the reporting source file. The `root_causes` key keeps its `{str: int}` shape and `failed/` directory naming is unchanged.

Generated scripts embed inputs, references and checks. Ordinary regions check numerical results, repeated executions, paired schedules (threads, num_stages, loop_kind), layout pairs, input integrity and output guards; typed regions also guard scratch. Extended records compiler evidence and checks observations, pass configurations, randomly sampled pipelines, accumulator width, algebraic identities and scratch contents.

Feedback counts IR operations, dependencies, nesting, types, layouts and schedules, distinguishing attempts, successful executions and compilation. These are structural features, not compiler branch coverage. A failure category or numerical mismatch still requires triage before being called a compiler bug.

## Results and compatibility

Campaigns store `passed/`, `compiled/`, `failed/<root_cause>/`, and summary, feedback, seed, dimension and RNG state files. Extended `artifacts/` are retained by default; `--no-save-artifacts` removes temporary compilation evidence after each test while retaining the result records. Interrupted cases may also have `pending_program.pkl`. Passing and failing records include standalone Python reproducers; `--no-save-passed-code` keeps only the IR JSON of passing programs (their reproducers take about 80% of `passed/` and are never read back), while failures always keep both. Region filenames use call structure plus a full-IR hash; Extended uses family plus hash. Compilation-only results are distinct from successfully executed results. summary.json additionally records `root_cause_locations` (`root_cause → location → count`) for fine-grained triage.

**Interpreting failures:** `failed/<root_cause>/` is an automated symptom label, not a confirmed DSL defect. Re-run representative `.py` reproducers in the recorded environment, inspect the adjacent `.json`, and isolate the generated kernel from the reference/checker before counting distinct bugs. Even a reproducible `WRONG RESULT` can come from the oracle or its tolerance: a confirmed `atomic_mismatch` false positive applied atomic-specific comparison to a buffer written by an ordinary store. Disk exhaustion, resource limits and unsupported configurations also need separate treatment. The [September 26 audit](reports/2026.09.26/REPORT.md) conservatively confirmed **four defect mechanisms** for its audited fourth-round campaigns (three TileLang, one Triton); its 534 saved matching records are not 534 independent bugs. That audit is a dated result, not an automatic classification of later campaigns.

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
