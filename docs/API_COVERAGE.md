# CUDA DSL API coverage contract

Scope: the public callable names owned by `triton.language` and by TileLang's
CUDA `tilelang.language` facade in the **installed** versions. Other TileLang
backends (ROCm, CPU, Metal) and host-side compiler/autotuning APIs are separate
surfaces. An API name is not a semantic test case: each API also has an input,
dtype, shape, attribute, schedule, and GPU-architecture domain.

`api_coverage.py` inventories this facade dynamically and keeps three states:

- `unmapped`: the target-specific extend stage has no mapping. Some of these
  names may already be exercised in the shared generator or Region stage.
- `implemented_no_run`: an extend generator and lowering exist, but no passing
  extension campaign summary was supplied.
- `seen_in_passing_code`: a direct DSL call appears in at least one saved
  passing `.py` reproducer. This is code-generation evidence, not proof that
  every dynamic path of the API executed.
- `executed_pass`: at least one passing, baseline-revalidated extend case is
  recorded in a supplied summary. This proves one executable point, not the
  full argument domain or absence of bugs.

Run the inventory *inside each target environment* and retain the JSON with
the experiment evidence:

```bash
python api_coverage.py --backend triton --summary results/EXTENSION/summary.json --passed-code-dir results/COMMON/passed --output triton_api.json
python api_coverage.py --backend tilelang --summary results/EXTENSION/summary.json --passed-code-dir results/COMMON/passed --output tilelang_api.json
```

The first extension wave tests scans, reductions, sort/histogram, simple math,
and bitwise atomics. `topk` and `gather` are gated on exports from the installed
Triton package. Target-only features still requiring separate kernels and
oracles include tensor descriptors/TMA, low-precision and scaled dot, remaining
atomics and memory orders, inline assembly, random generators, user-defined
combiner functions, compiler hints, and device debug operations. Some features
require SM90+ or another accelerator; they cannot be counted as GPU-executed on
an RTX 4090 (SM89). Add such features as independent extend adapters so common
tile-based structure generation stays backend-neutral.

For an eventual full-API claim, define a frozen target-version manifest and
required architecture per entry, then require (1) legal generator, (2)
independent oracle or a documented metamorphic relation, (3) passing execution
evidence, and (4) a bounded attribute/dtype/shape grid. Report separately the
entries blocked by hardware, entries without a sound oracle, and entries not
yet implemented. Never use `implemented_no_run` as execution coverage.
