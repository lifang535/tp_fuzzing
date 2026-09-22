"""Phase 2 gate GPU repro: atomic / fma / shape-op / int8-dot extended programs.

Deterministic seed matrix over both backends and the families that carry the
four new op surfaces; every case runs through the full oracle (compile +
launch + extended reference checks) on the local GPU. Probabilities are
pinned to 1 so each case provably exercises its surface.
"""
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import Config
from src.workflow.generator.extended import ExtendedGenerator
from src.workflow.oracle import Oracle

# (family, seeds) — indexed_memory gets the most seeds so the random
# atomic fn/dtype draws cover add/max/min over int32 and float32.
MATRIX = (('indexed_memory', 8), ('arithmetic', 4), ('shape_matmul', 4), ('mixed', 4))


def config_for():
    return Config(extended_atomic_prob=1, extended_fma_prob=1,
                  extended_shape_op_prob=1, extended_int8_prob=1,
                  extended_config_depth=2)


def main():
    failures = []
    cases = 0
    started = time.monotonic()
    for backend in ('triton', 'tilelang'):
        config = config_for()
        for family, seeds in MATRIX:
            for seed in range(seeds):
                cases += 1
                label = f'{backend}/{family}/{seed}'
                random.seed(seed)
                program = ExtendedGenerator(config, backend).generate(family=family)
                began = time.monotonic()
                report = Oracle(config, backend).test(program)
                elapsed = time.monotonic() - began
                if report is None:
                    print(f'PASS {label} ({elapsed:.1f}s)', flush=True)
                else:
                    failures.append(label)
                    print(f'FAIL {label} ({elapsed:.1f}s): {report.bug_type.value} '
                          f'root_cause={report.root_cause} location={report.location}',
                          flush=True)
    print(f'{cases - len(failures)}/{cases} extended GPU repro cases passed '
          f'in {time.monotonic() - started:.0f}s', flush=True)
    if failures:
        print('FAILURES: ' + ', '.join(failures), flush=True)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
