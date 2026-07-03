# TileSmith — Structure-Aware Fuzzer for Tile Programs

TileSmith is a fuzzing tool designed for tile-based GPU program compilers (TileLang, Triton),
inspired by MLIRSmith's two-phase generation approach (structural template + parameter instantiation).

---

## Directory Structure

```
tp_fuzzing/
├── main.py                    # Entry point
├── src/
│   ├── config/                # Centralized hyperparameter configuration
│   │   └── config.py
│   ├── ir/                    # Intermediate Representation (IR)
│   │   ├── ir.py              # Core data structures (TileKernel, ComputeKind, etc.)
│   │   ├── pipeline.py        # Multi-step pipeline IR
│   │   └── dynamic_seq.py     # Dynamic sequence IR (MLIRSmith TypedValuePool style)
│   ├── constraints/           # Hardware constraint validation
│   │   └── constraints.py
│   ├── ops/                   # Operator registry (one class per ComputeKind)
│   │   └── ops.py
│   └── workflow/              # Fuzzing workflow
│       ├── generator/         # Program generator
│       ├── mutator/           # Mutation engine
│       ├── emitter/           # Code emitter (TileLang / Triton)
│       │   ├── tilelang/
│       │   └── triton/
│       ├── oracle/            # Test oracle (execution + bug detection)
│       └── fuzzer/            # Main fuzzing loop
```

---

## Quick Start

```bash
# Run with default settings (100 iterations, TileLang backend)
python main.py

# Specify iterations and random seed (reproducible)
python main.py -n 500 --seed 42

# Print generated code without executing
python main.py --dump --seed 42

# List all supported operator types
python main.py --list-kernels

# Use Triton backend
python main.py --backend triton -n 200

# Specify output directory
python main.py -o /tmp/fuzz_results -n 1000

# Use easy-shape mode (power-of-2 shapes only)
# Effect: ~14% higher pass rate; useful for validating the fuzzer itself or building a clean seed corpus
python main.py --easy-shape -n 200

# Compare pass rates between modes
python main.py --seed 42 -n 100 -o results/normal
python main.py --seed 42 -n 100 --easy-shape -o results/easy

# Resume a previous (possibly incomplete) run into the same result directory
python main.py --resume 2026.06.29-16.41_tilelang_easy-shape_seed=42 -n 200 --seed 42 --easy-shape
```

---

## Core Design

### Three Program Types

| Type | Probability | Description |
|------|-------------|-------------|
| `TilePipeline` | 40% | Template-based multi-step pipeline (GEMM + epilogue) |
| `DynamicSequence` | 30% | Pool-driven dynamic sequence (MLIRSmith-style) |
| `TileProgram` | 30% | Single-operator program |

### Supported Operators (15 kinds)

- Matrix multiplication: `gemm`
- Memory ops: `copy`
- Elementwise: `add`, `mul`, `max`, `sub`, `scale`, `exp`, `sqrt`, `where`
- Transpose: `transpose`
- Reduction: `reduce_sum`, `reduce_max`, `reduce_min`
- Composite: `softmax`

### Bug Classification

The tool automatically classifies discovered bugs into 10 categories:

| Category | Description |
|----------|-------------|
| `wrong_result` | Computed result differs from reference |
| `dtype_mismatch` | Compiler's internal type inference conflicts with declaration |
| `warp_partition` | Warp partitioning cannot satisfy block size |
| `shared_memory_overflow` | Shared memory exceeds hardware limit |
| `layout_inference` | TileLang layout inference finds no valid layout |
| `dtype_unsupported_op` | Operator does not support the given type (e.g. `tl.sqrt` with fp16) |
| `codegen_duplicate_arg` | Emitted kernel contains duplicate arguments |
| `triton_compile_error` | Triton compilation error |
| `segfault` | Compiler segmentation fault |
| `other` | Uncategorized errors |

> **Notes:**
> - `wrong_result` does not necessarily indicate a genuine compiler bug. In chained computations (multi-step pipelines or dynamic sequences), accumulated floating-point rounding errors can cause the output to diverge slightly from the reference, leading to false positives.
> - Hardware constraint failures such as `shared_memory_overflow` may be partially caused by inaccurate hardware introspection at the constraint-checking stage — if the actual GPU's shared memory capacity cannot be read reliably, the generator may produce kernels that exceed the true hardware limit.

---

## Output Structure

```
results/
└── 2026.06.26-10.30_tilelang_hard-shape_seed=42/
    ├── summary.json                              # Cumulative statistics (across all sessions)
    ├── passed/
    │   ├── passed_single_gemm_M128,N256,K64,bM64,bN128,bK32,t128,pipelined,s2,float16.py
    │   ├── passed_pipeline_gemm+scale+add_M512,N512,K128,bM64,bN64,bK32,t128,serial,s1,float16.py
    │   └── passed_dynamic_gemm+exp+copy_f2g_M256,N128,K64,bM32,bN64,bK16,t128,pipelined,s2,float16.py
    └── failed/
        └── {root_cause}/
            ├── failed_single_gemm_M128,N256,K64,bM64,bN128,bK32,t128,pipelined,s2,float16.py
            ├── failed_pipeline_gemm+where_M512,N512,K128,bM64,bN64,bK32,t128,serial,s1,float16.py
            └── failed_dynamic_gemm+sqrt+mul_M256,N128,K64,bM32,bN64,bK16,t128,pipelined,s2,float16.py
```

Filename convention: `{passed/failed}_{type}_{ops}_{params}`

- Params format: `M{m},N{n},K{k},bM{block_M},bN{block_N},bK{block_K},t{threads},{loop_kind},s{num_stages},{dtype}`
- Single op: `{passed/failed}_single_{op}_{params}`
- Template pipeline: `{passed/failed}_pipeline_{op1}+{op2}+..._{params}`
- Dynamic sequence: `{passed/failed}_dynamic_{op1}+{op2}+..._{params}`

Only test cases with identical program structure AND all input parameters are considered duplicates; different parameters produce distinct files.

`summary.json` format:

```json
{
  "backend": "tilelang",
  "total_tested": 2011,
  "bugs_total": 940,
  "bugs_unique": 4,
  "root_causes": {
    "wrong_result": 859,
    "shared_memory_overflow": 2,
    "warp_partition": 5,
    "dtype_mismatch": 74
  }
}
```

- `bugs_total`: sum of all root_cause trigger counts (`sum(root_causes.values())`)
- `bugs_unique`: number of distinct root cause categories (`len(root_causes)`)
- `root_causes`: trigger count per category, including duplicate bugs not saved to disk

## Resume

`--resume` continues a previous run into the same result directory:

```bash
python main.py --resume 2026.06.29-16.41_tilelang_easy-shape_seed=42 \
               -n 1000 --seed 42 --easy-shape
```

- `--backend`, `--easy-shape`, and `--seed` must match the directory name — mismatch raises an error
- Rebuilds the tested-config set from `passed/` and `failed/` files to skip already-tested programs
- Restores exact root_cause trigger counts from `summary.json` (including dup bugs not written to files)
- If `summary.json` is absent (run was interrupted before saving), falls back to file counts
- All statistics are written back to `summary.json` cumulatively at the end of each session

---

## Configuration

All hyperparameters are centralized in the `Config` dataclass in `src/config/config.py`.
Common options:

```python
Config(
    seed=42,              # Random seed (None = non-deterministic)
    backends=["tilelang"],# Target backend
    output_dir="results", # Output directory
    compile_timeout=60,   # Compilation timeout (seconds)
    execute_timeout=60,   # Execution timeout (seconds)
)
```

---

## Workflow

### Step 1 — Program Generation (`generator/`)

`ProgramGenerator.generate()` selects one of three strategies by probability:

**Strategy A — Single op (30%)**  
Randomly selects one of 15 `ComputeKind` values (weighted), validates hardware constraints, and emits a `TileKernel`.

**Strategy B — Template pipeline (40%)**  
Generates a `TilePipeline` from a predefined structural template:
- GEMM epilogue: `GEMM → [0–2 epilogue ops] → [optional terminal]`
- Elementwise chain: `COPY → [1–2 elementwise ops]`

**Strategy C — Dynamic sequence (30%, MLIRSmith-style)**  
`DynamicSequenceGenerator` maintains a `TileValuePool` (analogous to MLIRSmith's `TypedValuePool`) and makes incremental decisions:

```
Initialize TileValuePool (with input buffers A/B)
Loop 3–8 steps:
    1. Scan all OpGens, find ops available given current pool state
    2. Randomly select one by weight (uncovered ops get +50 diversity boost)
    3. Emit a KernelStep, update pool and torch_ref
Always starts with GEMM
```

Each buffer carries a `torch_ref` (e.g. `"A.float() @ B.float()"`) that is updated with every op step and used for correctness verification at the end.

### Step 2 — Mutation (`mutator/`)

60% of iterations mutate a passing program from `seed_pool`:

- **Parameter mutation**: shapes to 2^n / 2^n±1 / prime / extreme values; tile sizes to valid values; dtype switch; threads switch
- **Structural mutation**: toggle `loop_kind` (pipelined ↔ serial); adjust `num_stages`; replace `compute_kind`
- **Boundary mutation**: `M = block_M * n + r` (r ≠ 0) to trigger non-divisible boundary handling
- **Pipeline-specific**: add / remove / replace epilogue steps

After mutation, `_enforce_constraints()` repairs any invalid parameter combinations.

### Step 3 — Code Emission (`emitter/`)

Translates abstract IR into executable Python code strings. The same IR can target different backends:

| IR Type | TileLang | Triton |
|---------|----------|--------|
| `TileKernel` (single op) | `tilelang/emitter.py` | `triton/emitter.py` |
| `TilePipeline` (template) | `tilelang/pipeline_emitter.py` | `triton/pipeline_emitter.py` |
| `DynamicSequence` (dynamic) | `tilelang/dynamic_emitter.py` | `triton/dynamic_emitter.py` |

Each emitter produces a complete Python file with a kernel function and a test function (tensor creation, kernel execution, reference comparison).

### Step 4 — Test Execution (`oracle/`)

Runs generated code in an isolated subprocess to contain crashes:

```python
subprocess.run([python3, tmp_file], timeout=compile_timeout + execute_timeout)
```

### Step 5 — Result Saving (`fuzzer/`)

Passing programs are saved to `passed/` and optionally added to `seed_pool`. Failing programs are saved under `failed/{root_cause}/` with `.py` (reproducible code) and `.json` (metadata including full error output).

---

## Deduplication and Pool Rotation

**Deduplication**: programs with identical `(compute_kind, M, N, K, block, dtype, loop, stages)` signatures are tested only once.

**dim_pool rotation**: every `pool_rotation_interval` iterations (default 100), the dim_pool is re-randomized to prevent M/N/K values from being exhausted — measured to reduce duplicate rate from 63% to 4%.

---

## Correspondence with MLIRSmith

| MLIRSmith Component | TileSmith Equivalent |
|---------------------|----------------------|
| `TypedValuePool` | `TileValuePool` (`ir/dynamic_seq.py`) |
| `RegionGen.apply()` | `DynamicSequenceGenerator.generate()` |
| `OpGenerator` × 200+ | `OpGenBase` subclasses × 13 (`ir/dynamic_seq.py`) |
| `DiversityCriteria` | Diversity boost (uncovered op weight +50) |
| `config.h` / `OpConf` | `config/config.py` |
| Template JSON + instantiation | `TilePipeline` + `PipelineGenerator` |
| Crash detection only | Crash + correctness (differential testing) |
