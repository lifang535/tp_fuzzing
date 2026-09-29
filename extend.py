"""Run DSL-specific extensions of previously passing common Extended IR.

Example:
  python extend.py --backend triton --passed-dir results/RUN/passed -n 10000

Only execute-mode Extended JSON records from passed/ are accepted. Every
source is rechecked for each input seed before its derivative is attributed to
the target-specific stage; stale baselines are counted separately.
"""
import argparse
from collections import Counter
from datetime import datetime
import hashlib
import json
from pathlib import Path
import random

from src.config import Config
from src.ir.serialization import program_from_dict
from src.workflow.generator.dsl_extend import DSL_OPS, eligible_ops, extend_passed
from src.workflow.generator.grids import GridState
from src.workflow.oracle import Oracle


def passing_sources(directory, backend, op_filter=None, max_sources=0, seed=0):
    if directory.name != 'passed' or not directory.is_dir():
        raise ValueError('--passed-dir must name an existing passed/ directory')
    sources, seen = [], 0
    sampler = random.Random(seed)
    for path in sorted(directory.glob('*.json')):
        try:
            data = json.loads(path.read_text())
            if data.get('validation_mode') != 'execute' or data.get('type') != 'extended':
                continue
            program = program_from_dict(data)
            digest = hashlib.sha256(json.dumps(program.to_dict(), sort_keys=True).encode()).hexdigest()
            for op in eligible_ops(program, backend):
                if op_filter and op != op_filter:
                    continue
                seen += 1
                candidate = (path, digest, program, op)
                if not max_sources or len(sources) < max_sources:
                    sources.append(candidate)
                else:
                    slot = sampler.randrange(seen)
                    if slot < max_sources:
                        sources[slot] = candidate
        except (ValueError, KeyError, TypeError, json.JSONDecodeError):
            continue
    return sources


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend', required=True, choices=tuple(DSL_OPS))
    parser.add_argument('--passed-dir', required=True, type=Path)
    parser.add_argument('--op', help='Target only one DSL extension operation')
    parser.add_argument('-n', type=int, default=100)
    parser.add_argument('--seed', type=int, default=42, help='Extension generation seed')
    parser.add_argument('--input-seed', type=int, default=0, help='First input seed; incremented every source/op round')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--save-artifacts', action='store_true')
    parser.add_argument('--max-passed-saved', type=int, default=20,
                        help='Passing reproducers retained per operation (0 keeps all; failures are always saved)')
    parser.add_argument('--max-sources', type=int, default=5000,
                        help='Maximum source/op pairs held in memory (0 keeps all; reservoir-sampled otherwise)')
    parser.add_argument('--log-every', type=int, default=100)
    args = parser.parse_args(argv)
    if args.n < 1 or args.input_seed < 0 or args.max_passed_saved < 0 or args.max_sources < 0 or args.log_every < 1:
        parser.error('-n and --log-every must be positive; seed and source/save limits nonnegative')
    if args.op and args.op not in DSL_OPS[args.backend]:
        parser.error(f'--op must be one of {DSL_OPS[args.backend]}')
    sources = passing_sources(args.passed_dir, args.backend, args.op, args.max_sources, args.seed)
    if not sources:
        parser.error('No execute-mode, common Extended passing seeds eligible for this backend/op')
    random.seed(args.seed)
    random.shuffle(sources)
    output = args.output or Path('results') / (datetime.now().strftime('%Y.%m.%d-%H.%M.%S') +
                                                    f'_extend_{args.backend}_seed={args.seed}')
    output.mkdir(parents=True, exist_ok=False)
    config = Config(backends=[args.backend], input_seed=args.input_seed,
                    extended_prob=1, extended_common_only=True,
                    extended_config_depth=0, extended_precision_pair=False,
                    extended_identity_pair=False, random_config_count=0,
                    save_artifacts=args.save_artifacts)
    oracle = Oracle(config, backend=args.backend)
    oracle.artifact_root = output / 'artifacts'
    from src.backends.common.versions import environment
    grids, baseline_cache, counts = GridState(), {}, Counter()
    summary = {'backend': args.backend, 'passed_dir': str(args.passed_dir.resolve()),
               'source_op_pairs': len(sources), 'max_sources': args.max_sources,
               'requested': args.n, 'tested': 0, 'passed': 0, 'failed': 0,
               'attempts': 0, 'baseline_rejected': 0, 'invalid_extension': 0,
               'saved_passed': 0, 'by_op': {},
               'environment': environment(), 'complete': False}
    (output / 'summary.json').write_text(json.dumps(summary, indent=2))
    attempts = 0
    while summary['tested'] < args.n and attempts < max(args.n * 50, len(sources) * 3):
        path, digest, parent, op = sources[attempts % len(sources)]
        config.input_seed = args.input_seed + attempts // len(sources)
        attempts += 1
        summary['attempts'] = attempts
        baseline_key = (digest, config.input_seed)
        if baseline_key not in baseline_cache:
            report = oracle.test(parent)
            baseline_cache[baseline_key] = report is None
            if report is not None:
                with (output / 'baseline_rejected.jsonl').open('a') as stream:
                    stream.write(json.dumps({'source_file': str(path.resolve()),
                                             'source_sha256': digest,
                                             'input_seed': config.input_seed,
                                             'failure': report.to_dict()}) + '\n')
        if not baseline_cache[baseline_key]:
            summary['baseline_rejected'] += 1
            (output / 'summary.json').write_text(json.dumps(summary, indent=2))
            continue
        try:
            program = extend_passed(parent, args.backend, op, config, grids)
        except ValueError as error:
            counts[op + ':invalid_extension'] += 1
            summary['invalid_extension'] += 1
            summary['by_op'] = dict(counts)
            with (output / 'invalid_extension.jsonl').open('a') as stream:
                stream.write(json.dumps({'source_file': str(path.resolve()),
                                         'extension_op': op, 'error': str(error)}) + '\n')
            (output / 'summary.json').write_text(json.dumps(summary, indent=2))
            continue
        bug = oracle.test(program)
        index = summary['tested']
        status = 'failed' if bug else 'passed'
        if bug or args.max_passed_saved == 0 or counts[op + ':saved_passed'] < args.max_passed_saved:
            directory = output / status / (bug.root_cause if bug else '')
            directory.mkdir(parents=True, exist_ok=True)
            name = f'{index:07d}_{op}_{digest[:12]}_input={config.input_seed}'
            record = program.to_dict()
            record.update({'source_file': str(path.resolve()), 'source_sha256': digest,
                           'source_family': parent.family, 'extension_op': op,
                           'backend': args.backend, 'input_seed': config.input_seed,
                           'validation_mode': 'execute', 'baseline_revalidated': True})
            (directory / (name + '.json')).write_text(json.dumps(record, indent=2))
            (directory / (name + '.py')).write_text(oracle._emit_code(program))
            if bug:
                (directory / (name + '.error.json')).write_text(json.dumps(bug.to_dict(), indent=2))
            else:
                counts[op + ':saved_passed'] += 1
                summary['saved_passed'] += 1
        summary['tested'] += 1
        summary[status] += 1
        counts[op + ':' + status] += 1
        summary['by_op'] = dict(counts)
        (output / 'summary.json').write_text(json.dumps(summary, indent=2))
        if bug or summary['tested'] == 1 or summary['tested'] % args.log_every == 0 or summary['tested'] == args.n:
            print(f"[{summary['tested']}/{args.n}] {op} {status} input_seed={config.input_seed}", flush=True)
    summary['complete'] = summary['tested'] == args.n
    (output / 'summary.json').write_text(json.dumps(summary, indent=2))
    print('Results:', output, flush=True)
    if not summary['complete']:
        raise RuntimeError('Could not reach requested count; inspect baseline_rejected and by_op in summary.json')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
