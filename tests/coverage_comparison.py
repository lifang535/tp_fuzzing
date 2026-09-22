"""Run a bounded v4 baseline or audit saved baseline/extended GPU corpora.

Example:
  python tests/coverage_comparison.py baseline --seeds 20 --output results/baseline
  python tests/coverage_comparison.py report --baseline results/baseline \
      --candidate results/extended_triton results/extended_tilelang --output results/comparison.json

The inventory measures source constructs in programs passing the oracle. It is
not compiler edge coverage, bug counts, or a throughput-controlled experiment.
"""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.backends import get_backend
from src.config import Config
from src.ir.serialization import program_from_dict
from src.ir.extended import ExtendedProgram
from src.workflow.coverage_audit import CAPABILITIES, program_capabilities
from src.workflow.generator.region_generator import RegionGenerator
from src.workflow.oracle.process import run_isolated
from src.workflow.oracle.evidence import read_extended_evidence


def baseline(args):
    args.output.mkdir(parents=True, exist_ok=False)
    summary = {'mode': 'execute', 'generator': 'region_v4', 'cases': [],
               'complete': False, 'expected_cases': args.seeds * (2 if args.backend == 'both' else 1)}
    config = Config(extended_prob=0, coverage_probe_prob=0, dim_range=(16, 80),
                    tile_size_choices=[32], block_k_choices=[32],
                    thread_choices=[128], region_input_seed_count=2, region_repeat_count=2)
    for backend in ('triton', 'tilelang'):
        if args.backend not in ('both', backend):
            continue
        for seed in range(args.seeds):
            random.seed(seed)
            p = RegionGenerator(config, backend).generate(initial='load' if seed % 2 == 0 else 'gemm')
            directory = args.output / f'{backend}_region_{seed}'
            directory.mkdir()
            (directory / 'program.json').write_text(json.dumps(p.to_dict(), indent=2))
            code = get_backend(backend).make_emitter(config).emit(p)
            path = directory / 'repro.py'
            path.write_text(code)
            start = time.monotonic()
            env = dict(os.environ, OMP_NUM_THREADS='1', TILELANG_CACHE_DIR=str(directory / 'tilelang_cache'))
            try:
                result = run_isolated([sys.executable, '-B', str(path)], env=env, timeout=args.timeout)
                log = result.stdout + result.stderr
                passed = result.returncode == 0 and 'ALL PASSED' in result.stdout.splitlines()
                returncode = result.returncode
                if returncode:
                    log += f'\nProcess exited with return code {returncode}\n'
            except subprocess.TimeoutExpired as exc:
                # communicate() can attach bytes even when text=True. A
                # timeout must remain a recorded failure, not abort the run.
                def decode(value):
                    return value.decode(errors='replace') if isinstance(value, bytes) else value or ''
                log = decode(exc.stdout) + decode(exc.stderr) + '\nTIMEOUT'
                passed = False
                returncode = None
            (directory / 'run.log').write_text(log)
            summary['cases'].append({'case': directory.name, 'backend': backend,
                'artifact_dir': str(directory.resolve()), 'compiled': passed, 'passed': passed,
                'returncode': returncode,
                'seconds': round(time.monotonic() - start, 3),
                'source_sha256': hashlib.sha256(code.encode()).hexdigest(),
                'program_sha256': hashlib.sha256((directory / 'program.json').read_bytes()).hexdigest()})
            (args.output / 'summary.json').write_text(json.dumps(summary, indent=2))
            print(('PASS ' if passed else 'FAIL ') + directory.name, flush=True)
    summary['complete'] = True
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2))
    return int(any(not c['passed'] for c in summary['cases']))


def audit(directories):
    result = {b: {'attempted': 0, 'compiled': 0, 'passed': 0, 'seconds': 0,
                  'capabilities': Counter(), 'compiler_ir_features': set(), 'failures': []}
              for b in ('triton', 'tilelang')}
    seen = set()
    for directory in directories:
        summary = json.loads((directory / 'summary.json').read_text())
        if summary.get('mode') != 'execute':
            raise ValueError('Execution coverage cannot include compile-only corpora: ' + str(directory))
        if not summary.get('complete', True) or len(summary['cases']) != summary.get('expected_cases', len(summary['cases'])):
            raise ValueError('Coverage corpus is incomplete: ' + str(directory))
        for c in summary['cases']:
            backend = c.get('backend') or c['case'].split('_')[0]
            bucket = result[backend]
            path = Path(c['artifact_dir'])
            if not path.is_dir():
                path = directory / c['case']
            path = path.resolve()
            if path in seen:
                continue
            seen.add(path)
            bucket['attempted'] += 1
            bucket['seconds'] += c.get('seconds', 0)
            if c.get('compiled'):
                bucket['compiled'] += 1
            if not c.get('passed') or not c.get('compiled'):
                bucket['failures'].append(str(path))
                continue
            log = (path / 'run.log').read_text()
            if 'ALL PASSED' not in log.splitlines():
                raise ValueError('Missing successful oracle log: ' + str(path))
            if c.get('returncode', 0) != 0:
                raise ValueError('Successful case has a failed process status: ' + str(path))
            if c.get('source_sha256') and hashlib.sha256((path / 'repro.py').read_bytes()).hexdigest() != c['source_sha256']:
                raise ValueError('Reproducer changed after validation: ' + str(path))
            if c.get('program_sha256') and hashlib.sha256((path / 'program.json').read_bytes()).hexdigest() != c['program_sha256']:
                raise ValueError('Program changed after validation: ' + str(path))
            program = program_from_dict(json.loads((path / 'program.json').read_text()))
            bucket['passed'] += 1
            bucket['capabilities'].update(program_capabilities(program))
            if isinstance(program, ExtendedProgram):
                if not (path / 'compilation.json').exists():
                    raise ValueError('Missing compilation evidence: ' + str(path))
                generation = summary.get('generation_config', {})
                # Campaigns saved before the precision/identity sweeps carry no
                # such keys: replay them with those sweeps off.
                config = Config(extended_prob=1,
                                extended_config_depth=generation.get('extended_config_depth', 1),
                                extended_precision_pair=generation.get('extended_precision_pair', False),
                                extended_identity_pair=generation.get('extended_identity_pair', False))
                expected = len(get_backend(backend).extended_variants(program, config))
                evidence = read_extended_evidence(path, expected)
                if not evidence.completed('execute', log):
                    raise ValueError('Incomplete compilation/execution evidence: ' + str(path)
                                     + ': ' + evidence.diagnostic)
                bucket['compiler_ir_features'].update(f for r in evidence.records for f in r['features'])
    for bucket in result.values():
        bucket['seconds'] = round(bucket['seconds'], 3)
        bucket['capabilities'] = dict(sorted(bucket['capabilities'].items()))
        bucket['compiler_ir_features'] = sorted(bucket['compiler_ir_features'])
    return result


def report(args):
    before, after = audit(args.baseline), audit(args.candidate)
    result = {'metric': 'source capability inventory in oracle-passing programs',
              'limitations': ['Not compiler edge coverage or confirmed bug counts',
                             'Not a throughput-controlled or statistical bug-yield experiment',
                             'Native source liveness is conservative; extended liveness is static',
                             'Newly observed constructs may already be expressible but absent from the sampled baseline',
                             'Compiler IR features are reported only for successful candidate programs'],
              'inventory': CAPABILITIES, 'baseline': before, 'candidate': after, 'new_capabilities': {}}
    for backend in before:
        new = sorted(after[backend]['capabilities'].keys() - before[backend]['capabilities'].keys())
        result['new_capabilities'][backend] = new
        print(f"{backend}: baseline={before[backend]['passed']}/{before[backend]['attempted']}, "
              f"candidate={after[backend]['passed']}/{after[backend]['attempted']}, new capabilities={len(new)}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    run = commands.add_parser('baseline')
    run.add_argument('--backend', choices=('both', 'triton', 'tilelang'), default='both')
    run.add_argument('--seeds', type=int, default=20)
    run.add_argument('--timeout', type=int, default=180)
    run.add_argument('--output', type=Path, required=True)
    compare = commands.add_parser('report')
    compare.add_argument('--baseline', nargs='+', type=Path, required=True)
    compare.add_argument('--candidate', nargs='+', type=Path, required=True)
    compare.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'baseline' and (args.seeds < 1 or args.timeout < 1):
        parser.error('seeds and timeout must be positive')
    return baseline(args) if args.command == 'baseline' else report(args)


if __name__ == '__main__':
    raise SystemExit(main())
