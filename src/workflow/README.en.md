# Workflow Modules

This directory contains the core fuzzing workflow components of TileSmith — a **backend-agnostic fuzzer for tile-based GPU programs**. It generates programs at the abstract IR level and translates them to specific backends (TileLang, Triton, etc.) via pluggable emitters.

**Design principle**: Only the emitter layer touches backend-specific APIs. The generator, mutator, oracle, and IR are entirely backend-independent. Adding a new backend requires only a new emitter + constraint file.

Supported backends:
- **TileLang**: tile-level DSL based on TVM (`T.gemm`, `T.copy`, `T.Parallel`)
- **Triton**: OpenAI's tile-level DSL (`tl.dot`, `tl.load`, `tl.store`)
- **(Extensible)**: CUTLASS, Hidet, or other tile compilers

---

## Module Structure

```
workflow/
├── generator/     Program generator
├── mutator/       Mutation engine
├── emitter/       Code emitter (translates IR → backend code)
│   ├── tilelang/  TileLang backend
│   └── triton/    Triton backend
├── oracle/        Test oracle (run + detect bugs)
└── fuzzer/        Main fuzzing loop
```

---

## Full Workflow

### Step 1: Program Generation (generator/)

`ProgramGenerator.generate()` selects one of three generation strategies:

**Strategy A — Single Op (30%)**
Picks one of 15 `ComputeKind` values (GEMM, COPY, SOFTMAX, etc.), generates hardware-valid parameters, produces a `TileKernel`.

**Strategy B — Template Pipeline (40%)**
Generates a `TilePipeline` from predefined structural templates:
- GEMM epilogue: `GEMM → [0-2 epilogue ops] → [optional terminal]`
- Elementwise chain: `COPY → [1-2 elementwise ops]`

**Strategy C — Dynamic Sequence (30%, MLIRSmith-style)**
`DynamicSequenceGenerator` uses a pool-driven approach analogous to MLIRSmith's `TypedValuePool`:

```
Initialize TileValuePool (contains input buffers A, B)
Loop 3-8 steps:
    1. Scan all OpGens, find those applicable given current pool state
    2. Select one by weighted random (uncovered ops get +50 diversity boost)
    3. Generate KernelStep, update pool and torch_ref
Always starts with GEMM
```

Each buffer carries a `torch_ref` (e.g., `"A.float() @ B.float()"`) that is updated in-place as each op transforms it, used for correctness verification.

**Supported Ops (16 total):**

| Category | Ops | Description |
|---|---|---|
| **Memory** | `gemm`, `copy_g2s`, `copy_s2f`, `copy_f2g` | GEMM accumulation, global↔shared↔fragment data movement |
| **Elementwise** | `scale`, `exp`, `sqrt`, `elemwise_add/mul/max` | Scalar multiply, exp/sqrt, binary add/mul/max |
| **Reduce** | `reduce_sum`, `reduce_max`, `softmax` | Row-wise reduction (terminal ops) |
| **Nested** | `if_epilogue`, `double_pipeline`, `accumulate_reduce` | Control flow and data flow nesting (analogous to MLIRSmith's scf.if / scf.for) |

**Three nested structures (analogous to MLIRSmith's scf.if / scf.for):**

These are backend-agnostic — defined at the IR layer, translated separately by each emitter:

- **`if_epilogue`** (analogous to `scf.if`): Per-element conditional branching (`x > threshold ? path_A : path_B`). Tests compiler's handling of predicated computation.
  - TileLang: if/else inside `T.Parallel`
  - Triton: `tl.where(condition, a, b)`

- **`double_pipeline`** (analogous to nested `affine.for`): Two independent K-dimension GEMM loops, each accumulating into separate fragments, results summed. Tests correctness when multiple pipelines write to the same output tile.
  - TileLang: two sets of `alloc_shared` + `Pipelined` loop + `gemm`
  - Triton: two K-loops + `tl.dot`, results `acc1 + acc2`

- **`accumulate_reduce`** (analogous to `scf.for` + reduce): Row-level reduce (max or sum) followed by broadcasting back to 2D fragment (e.g., `x[i,j] -= row_max[i]`). Core pattern for online softmax and layer normalization.
  - TileLang: `T.reduce_max/sum` + `T.Parallel` elementwise
  - Triton: `tl.max/sum(axis=1)` + broadcast subtract/divide

---

### Step 2: Mutation (mutator/)

60% probability of mutating a seed from `seed_pool`:

- **Parametric**: shape → 2^n / 2^n±1 / prime / extreme, tile size, dtype, threads
- **Structural**: loop_kind (pipelined ↔ serial), num_stages, compute_kind
- **Boundary**: `M = block_M * n + r` (r≠0) — forces boundary tile handling
- **Pipeline-specific**: add/remove/replace epilogue steps

All mutations are followed by `_enforce_constraints()` to ensure hardware validity.

---

### Step 3: Code Emission (emitter/)

Translates abstract IR to executable backend-specific Python code. The same IR can be emitted to multiple backends:

| IR Type | TileLang | Triton |
|---|---|---|
| `TileKernel` (single op) | `tilelang/emitter.py` | `triton/emitter.py` |
| `TilePipeline` (template) | `tilelang/pipeline_emitter.py` | `triton/pipeline_emitter.py` |
| `DynamicSequence` (dynamic) | `tilelang/dynamic_emitter.py` | `triton/dynamic_emitter.py` |

Each emitter produces a complete Python file containing:
1. Kernel function definition (calls compiler API)
2. Test function (creates GPU tensors, runs kernel, compares against reference)

**Reference consistency**: The `ref` computation is derived from IR semantics — not hand-written:
- Single op: each op class in `ops.py` defines its reference (GEMM: `ref = A @ B`)
- Pipeline: chained tracking (`ref = ref * alpha` → `ref = torch.exp(ref)` → ...)
- Dynamic: `TileBuffer.torch_ref` updated in-place per op step

---

### Step 4: Test Execution (oracle/)

Programs are executed in isolated subprocesses to contain crashes:

```python
subprocess.run([python3, tmp_file], timeout=compile_timeout + execute_timeout)
```

Error classification (generic + backend-specific):

**Generic (all tile compilers):**

| Category | Meaning | Real bug? |
|---|---|---|
| `wrong_result` | Kernel output disagrees with reference | ✅ |
| `dtype_mismatch` | Compiler's type inference contradicts declaration | ✅ |
| `shared_memory_overflow` | Tile params exceed GPU shared memory | ❌ Hardware limit |
| `gpu_oom` | GPU out-of-memory (transient) | ❌ Environment |
| `segfault` | Compiler segfault | ✅ |
| `ptx_async_boundary` | Async copy produces illegal byte width at tile boundary | ✅ |
| `tilelang_codegen_error` | Codegen internal assertion failure | ✅ |

**TileLang-specific:**

| Category | Meaning |
|---|---|
| `warp_partition` | MMA warp partition cannot match block dimensions |
| `layout_inference` | Cannot find valid memory layout |
| `alignment` | Block size violates MMA alignment |

**Triton-specific:**

| Category | Meaning |
|---|---|
| `dtype_unsupported_op` | Op doesn't support given dtype (e.g., tl.sqrt on fp16) |
| `triton_compile_error` | Triton compiler error |

---

### Step 5: Result Storage and Resume (fuzzer/)

```
results/{date-time}_{backend}_{easy/hard-shape}_seed={seed}/
├── passed/
│   ├── passed_single_{op}_{params}.py
│   ├── passed_pipeline_{op1}+{op2}+..._{params}.py
│   └── passed_dynamic_{op1}+{op2}+..._{params}.py
├── failed/
│   └── {root_cause}/
│       ├── failed_single_{op}_{params}.py
│       ├── failed_pipeline_{op1}+{op2}+..._{params}.py
│       └── failed_dynamic_{op1}+{op2}+..._{params}.py
└── summary.json          # cumulative across all sessions
```

Filename convention: `{passed/failed}_{type}_{ops}_{params}`

- Params format: `M{m},N{n},K{k},bM{block_M},bN{block_N},bK{block_K},t{threads},{loop_kind},s{num_stages},{dtype}`
- Example: `passed_dynamic_gemm+exp+copy_f2g_M256,N128,K64,bM32,bN64,bK16,t128,pipelined,s2,float16`

Only test cases with identical program structure AND all input parameters are considered duplicates (matching the dedup signature); different parameters produce distinct files.

**Resume** (`--resume <dir>`) continues an existing run. Steps:

1. **Config validation**: parse the directory name and verify backend / easy-shape / seed match the current CLI flags — mismatch raises an error immediately
2. **Rebuild `tested_configs`**: scan all `.json` files under `passed/` and `failed/`, reconstruct each sig via `_make_sig_from_dict` (same format as the runtime `_make_sig`), add to the dedup set
3. **Restore `known_root_causes`**: count files per root_cause directory, then overlay with `summary.json` trigger counts (`max(file_count, summary_count)`) so dup-bug counts lost between sessions are recovered
4. **Restore `total_tested`**: `max(total_file_count, summary["total_tested"])` — deduped-and-skipped configs are counted as tested but not saved to files
5. **Continue**: new tests are appended to the same directory; cumulative stats are written back to `summary.json` at session end

**`summary.json` field semantics**:
- `bugs_total` = `sum(root_causes.values())` — total trigger count
- `bugs_unique` = `len(root_causes)` — number of distinct root cause categories
- `root_causes` stores trigger counts (not file counts) and is the authoritative source for restoring `known_root_causes` on resume

---

## Deduplication and Pool Rotation

**Dedup**: Same `(compute_kind, M, N, K, block_M, block_N, block_K, dtype, loop, stages)` is tested only once.

**dim_pool rotation**: Every `pool_rotation_interval` (default 100) iterations, the dimension pool is re-randomized. This prevents the generator from exhausting its 20-value pool and significantly reduces duplicate rate (measured: 63% → 4%).

---

## Mapping to MLIRSmith

| MLIRSmith Component | TileSmith Equivalent |
|---|---|
| `TypedValuePool` | `TileValuePool` (`ir/dynamic_seq.py`) |
| `RegionGen.apply()` | `DynamicSequenceGenerator.generate()` |
| `OpGenerator` × 200+ | `OpGenBase` subclasses × 16 (`ir/dynamic_seq.py`) |
| `DiversityCriteria` | Diversity boost (uncovered ops weight +50) |
| `config.h` / `OpConf` | `config/config.py` (35+ hyperparameters) |
| Template JSON + Instantiation | `TilePipeline` + `PipelineGenerator` |
| Crash-only detection | Crash + differential testing + overflow detection |

---

## Known Limitations

1. **`dynamic_seq.py` generates TileLang code directly**: The `KernelStep.tilelang_code` field embeds TileLang syntax in the IR layer. Ideally, the IR should only record semantics, and the emitter should translate. This is a pragmatic shortcut — the Triton dynamic emitter independently generates Triton code from the step metadata, so correctness is not affected.

2. **Shared memory estimation is approximate**: Compilers internally allocate more shared memory than the tile formula predicts (barrier metadata, alignment padding, double-buffering). We use 50% safety margin, but occasional overflows still occur and are classified as `shared_memory_overflow`.
