"""Small, reproducible exploration corpus; compile-only or real GPU validation.

Each program runs in an isolated process and retains IR, logs and compiler
artifacts. Counts distinguish compilation from successful oracle execution.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.backends import get_backend
from src.config import Config
from src.workflow.generator.extended import ExtendedGenerator, FAMILIES, mutate_extended
from src.workflow.feedback import program_features
from src.workflow.oracle.process import run_isolated
from src.workflow.oracle.evidence import read_extended_evidence


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--backend', choices=('both', 'triton', 'tilelang'), default='both')
    parser.add_argument('--family', choices=FAMILIES)
    parser.add_argument('--regression', choices=('nan_reduction', 'nan_arithmetic'))
    parser.add_argument('--compile-only', action='store_true')
    parser.add_argument('--config-depth', type=int, choices=(0, 1, 2), default=2,
                        help='Extended configuration sweep depth (default 2: covers the pass-configuration pairs)')
    parser.add_argument('--no-extended-precision', action='store_true',
                        help='Disable the fp16-accumulation precision pair (and the triton tf32 variant)')
    parser.add_argument('--no-extended-identities', action='store_true',
                        help='Disable the distributivity identity pair on matmul-less programs')
    parser.add_argument('--seeds', type=int, default=1)
    parser.add_argument('--seed-start', type=int, default=0)
    parser.add_argument('--mutations', type=int, default=0)
    parser.add_argument('--timeout', type=int, default=300)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.seeds < 1 or args.mutations < 0 or args.timeout < 1:
        parser.error('Require positive seeds/timeout and nonnegative mutations')
    directory = args.output or Path(tempfile.mkdtemp(prefix='tilesmith_extended_'))
    directory.mkdir(parents=True, exist_ok=args.output is None)
    print('Artifacts:', directory, flush=True)
    summary = {'mode': 'compile_only' if args.compile_only else 'execute', 'cases': [], 'backends': {},
               'complete': False,
               'generation_config': {'extended_config_depth': args.config_depth,
                                     'extended_precision_pair': not args.no_extended_precision,
                                     'extended_identity_pair': not args.no_extended_identities},
               'expected_cases': (2 if args.backend == 'both' else 1) * args.seeds *
                                 (1 if args.regression or args.family else len(FAMILIES)) * (args.mutations + 1)}
    # Mark even a first-case interruption as an incomplete corpus.
    (directory / 'summary.json').write_text(json.dumps(summary, indent=2))
    for backend in ('triton', 'tilelang'):
        if args.backend not in ('both', backend):
            continue
        compiled_features, passed_features, compiler_features = set(), set(), set()
        config = Config(extended_prob=1, compile_only=args.compile_only, region_repeat_count=2,
                        extended_config_depth=args.config_depth,
                        extended_precision_pair=not args.no_extended_precision,
                        extended_identity_pair=not args.no_extended_identities)
        for seed in range(args.seed_start, args.seed_start + args.seeds):
            for family in ([args.regression] if args.regression else [args.family] if args.family else FAMILIES):
                random.seed(seed)
                if args.regression:
                    from test_extended import nan_reduction_program, nan_arithmetic_program
                    factory = nan_reduction_program if args.regression == 'nan_reduction' else nan_arithmetic_program
                    program = factory(config, backend)
                else:
                    program = ExtendedGenerator(config, backend).generate(family)
                for mutation in range(args.mutations + 1):
                    if mutation:
                        program = mutate_extended(program, config, backend)
                    label = f'{backend}_{family}_{seed}_{mutation}'
                    case_dir = directory / label
                    case_dir.mkdir(exist_ok=False)
                    (case_dir / 'program.json').write_text(json.dumps(program.to_dict(), indent=2))
                    path = case_dir / 'repro.py'
                    path.write_text(get_backend(backend).make_emitter(config).emit(program))
                    env = dict(os.environ, TILESMITH_ARTIFACT_DIR=str(case_dir),
                               TILELANG_CACHE_DIR=str(case_dir / 'tilelang_cache'),
                               TILESMITH_COMPILE_ONLY=str(int(args.compile_only)),
                               OMP_NUM_THREADS='1')
                    started = time.monotonic()
                    try:
                        result = run_isolated([sys.executable, '-B', str(path)], env=env, timeout=args.timeout)
                        output = result.stdout + result.stderr
                        stdout = result.stdout
                        returncode = result.returncode
                        if returncode:
                            output += f'\nProcess exited with return code {returncode}\n'
                    except subprocess.TimeoutExpired as exc:
                        def decode(value):
                            return value.decode(errors='replace') if isinstance(value, bytes) else value or ''
                        stdout = decode(exc.stdout)
                        output = stdout + decode(exc.stderr) + '\n' + str(exc)
                        returncode = None
                    evidence = read_extended_evidence(case_dir, len(get_backend(backend).extended_variants(program, config)))
                    records, progress, compiled = evidence.records, evidence.progress, evidence.compiled
                    success = returncode == 0 and evidence.completed(summary['mode'], stdout)
                    if not success:
                        output += '\nIncomplete or failed extended validation: ' + (evidence.diagnostic or 'oracle did not complete') + '\n'
                    (case_dir / 'run.log').write_text(output)
                    compiler_features.update(f for r in records for f in r['features'])
                    if compiled:
                        compiled_features.update(program_features(program))
                    if success and not args.compile_only:
                        passed_features.update(program_features(program))
                    entry = {'case': label, 'passed': success, 'compiled': compiled,
                             'returncode': returncode,
                             'backend': backend,
                             'source_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                             'program_sha256': hashlib.sha256((case_dir / 'program.json').read_bytes()).hexdigest(),
                             'variants': len(records), 'artifact_dir': str(case_dir),
                             'seconds': round(time.monotonic() - started, 3), 'progress': progress,
                             'operations': sorted({n.op for n in program.all_operations()}),
                             'dtypes': sorted({v.type.dtype for n in program.all_operations() for v in n.results})}
                    summary['cases'].append(entry)
                    # Preserve progress even when a later case is interrupted.
                    summary['backends'][backend] = {
                        'compiled_source_features': len(compiled_features),
                        'executed_source_features': len(passed_features),
                        'compiler_ir_features': len(compiler_features),
                        'executed_features': sorted(passed_features),
                        'compiler_features': sorted(compiler_features),
                    }
                    (directory / 'summary.json').write_text(json.dumps(summary, indent=2))
                    print(('PASS ' if success else 'FAIL ') + label, flush=True)
                    if not success:
                        print(output[-1800:], flush=True)
        summary['backends'][backend] = {
            'compiled_source_features': len(compiled_features),
            'executed_source_features': len(passed_features),
            'compiler_ir_features': len(compiler_features),
            'executed_features': sorted(passed_features),
            'compiler_features': sorted(compiler_features),
        }
        (directory / 'summary.json').write_text(json.dumps(summary, indent=2))
    summary['complete'] = True
    (directory / 'summary.json').write_text(json.dumps(summary, indent=2))
    failures = sum(not c['passed'] for c in summary['cases'])
    print(f"{len(summary['cases']) - failures}/{len(summary['cases'])} passed ({summary['mode']})", flush=True)
    return int(bool(failures))


if __name__ == '__main__':
    raise SystemExit(main())
