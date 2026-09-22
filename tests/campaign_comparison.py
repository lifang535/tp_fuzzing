"""Compare saved main.py campaigns without running kernels or changing results.

Counts saved outcomes, not confirmed compiler bugs. Artifact/variant files are
never counted as extra test cases. Time windows use the directory's local start
minute and result timestamps; copied files and resumed runs need extra care.
"""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
import statistics
import time


RUN_NAME = re.compile(r'^(\d{4}\.\d{2}\.\d{2}-\d{2}\.\d{2})_(.+)_(easy|hard)-shape_seed=(.+)$')


def program_domain(program):
    if program.get('type') == 'extended':
        return 'extended:' + program.get('family', 'unknown')
    if program.get('spec', {}).get('coverage_probe'):
        return 'probe'
    if program.get('legacy'):
        return 'legacy'
    return program.get('type', 'legacy')


def failure_bucket(report):
    """Triage hints, not a determination of compiler fault or root cause."""
    cause = report.get('root_cause', 'other')
    message = report.get('error_message', '').lower()
    if 'out of memory' in message or cause in ('gpu_oom', 'shared_memory_overflow'):
        return 'resource_limit'
    if cause == 'timeout':
        return 'timeout'
    if ('self.stride(-1) must be 1 to view' in message
            or 'incomplete extended compilation/execution evidence' in message):
        return 'harness_or_evidence'
    return 'needs_review'


def read_campaign(directory, snapshot):
    directory = Path(directory).resolve()
    match = RUN_NAME.fullmatch(directory.name)
    if match is None:
        raise ValueError('Expected a main.py campaign directory: ' + str(directory))
    stamp, backend, shape, seed = match.groups()
    start = datetime.strptime(stamp, '%Y.%m.%d-%H.%M').timestamp()
    rows, warnings = [], []
    for status in ('passed', 'failed'):
        for path in sorted((directory / status).rglob('*.json')):
            try:
                modified = path.stat().st_mtime
                if modified > snapshot:
                    continue
                record = json.loads(path.read_text())
                if not isinstance(record, dict):
                    raise ValueError('Expected a JSON object')
                if record.get('validation_mode') == 'compile_only':
                    warnings.append('Excluded compile-only outcome: ' + str(path))
                    continue
                if status == 'failed':
                    params = record.get('params', {})
                    program = params.get('extended_program', params.get('region_program', params))
                    timestamp = float(record.get('timestamp', modified))
                else:
                    program, timestamp = record, modified
                if not math.isfinite(timestamp) or timestamp < start:
                    raise ValueError('Invalid or pre-campaign result timestamp')
                if timestamp > snapshot:
                    continue
                rows.append({'timestamp': timestamp, 'status': status,
                             'domain': program_domain(program),
                             'category': record.get('root_cause', 'other') if status == 'failed' else None,
                             'triage': failure_bucket(record) if status == 'failed' else None,
                             'file': str(path.relative_to(directory))})
            except (OSError, ValueError, TypeError, AttributeError) as exc:
                warnings.append(str(path) + ': ' + str(exc))
    rows.sort(key=lambda row: (row['timestamp'], row['file']))
    summary = None
    if (directory / 'summary.json').exists():
        try:
            summary = json.loads((directory / 'summary.json').read_text())
            if not isinstance(summary, dict):
                raise ValueError('Expected a summary object')
        except (OSError, ValueError) as exc:
            warnings.append('Unreadable summary: ' + str(exc))
            summary = None
    if summary is None:
        warnings.append('No summary snapshot; live CLI configuration and unsaved outcomes are unknown.')
    elif summary.get('compile_only'):
        raise ValueError('Compile-only campaigns cannot be compared as execution campaigns: ' + str(directory))
    elif summary.get('total_tested') != len(rows):
        warnings.append('summary.total_tested differs from saved outcomes (snapshot, overwrites, or reporting limit).')
    return {'directory': str(directory), 'backend': backend, 'shape': shape, 'seed': seed,
            'start': start, 'rows': rows, 'summary': summary, 'warnings': warnings}


def summarize(campaign, minutes=None, cases=None):
    rows = campaign['rows']
    if minutes is not None:
        rows = [row for row in rows if row['timestamp'] <= campaign['start'] + minutes * 60]
    if cases is not None:
        rows = rows[:cases]
    # Full/n-case windows stop at the latest saved outcome. Fixed time windows
    # use their full budget only once a result at/after the boundary exists.
    observed = max(0, campaign['rows'][-1]['timestamp'] - campaign['start']) if campaign['rows'] else 0
    elapsed = min(minutes * 60, observed) if minutes is not None else (
        max(0, rows[-1]['timestamp'] - campaign['start']) if rows else 0)
    failed = [row for row in rows if row['status'] == 'failed']
    domains = defaultdict(Counter)
    for row in rows:
        domains[row['domain']][row['status']] += 1
    return {'saved_tested': len(rows), 'saved_passed': len(rows) - len(failed),
            'saved_failed': len(failed),
            'failure_rate_percent': round(100 * len(failed) / len(rows), 3) if rows else None,
            'observed_minutes_approx': round(elapsed / 60, 3),
            'saved_tests_per_hour_approx': round(3600 * len(rows) / elapsed, 2) if elapsed else None,
            'saved_failures_per_hour_approx': round(3600 * len(failed) / elapsed, 2) if elapsed else None,
            'window_reached': (observed >= minutes * 60 if minutes is not None else
                               len(rows) == cases if cases is not None else None),
            'failure_categories': dict(sorted(Counter(row['category'] for row in failed).items())),
            'triage': dict(sorted(Counter(row['triage'] for row in failed).items())),
            'domains': {key: dict(value) for key, value in sorted(domains.items())}}


def artifact_timings(directory, snapshot):
    """Approximate subprocess wall time from artifact mtimes, including failures."""
    durations = defaultdict(list)
    for path in (Path(directory) / 'artifacts').glob('*/program.json'):
        log = path.with_name('run.log')
        try:
            end, start = log.stat().st_mtime, path.stat().st_mtime
            if end > snapshot or end < start:
                continue
            program = json.loads(path.read_text())
            if program.get('type') == 'extended':
                durations[program['family']].append(end - start)
        except (OSError, ValueError, KeyError, AttributeError):
            continue
    return {key: {'artifacts_with_log': len(values), 'seconds_sum_approx': round(sum(values), 3),
                  'seconds_median_approx': round(statistics.median(values), 3)}
            for key, values in sorted(durations.items())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('campaigns', type=Path, nargs='+')
    parser.add_argument('--minutes', type=float, default=20)
    parser.add_argument('--cases', type=int, default=60)
    parser.add_argument('--output', type=Path, help='Write a new JSON report; refuses to overwrite')
    args = parser.parse_args()
    if not math.isfinite(args.minutes) or args.minutes <= 0 or args.cases <= 0:
        parser.error('minutes and cases must be positive')
    if args.output and any(args.output.resolve().is_relative_to(path.resolve()) for path in args.campaigns):
        parser.error('output must be outside the input campaign directories')
    snapshot = time.time()
    report = {'snapshot_utc': datetime.fromtimestamp(snapshot, timezone.utc).isoformat(),
              'metric': 'saved campaign outcomes, not confirmed unique compiler bugs',
              'limitations': [
                  'Backend, shape, configuration, compiler/GPU versions and runtime contention must match for A/B conclusions.',
                  'Saved outcomes can undercount attempts if filenames overwrite or reports are suppressed.',
                  'Passed timestamps and artifact durations use mtimes; copying or resuming can invalidate timing comparisons.',
                  'Directory start has minute precision; startup and cache warmup are included.',
                  'A failure category or needs_review count is not a unique compiler root cause.',
                  'No-summary live runs may have in-flight tests; partial JSON is skipped with a warning.',
                  'Triage buckets are hints; resource/time/harness failures may still need investigation.'],
              'first_minutes': args.minutes, 'first_cases': args.cases, 'campaigns': []}
    for directory in args.campaigns:
        campaign = read_campaign(directory, snapshot)
        report['campaigns'].append({
            **{key: campaign[key] for key in ('directory', 'backend', 'shape', 'seed', 'summary', 'warnings')},
            'full': summarize(campaign),
            'equal_time': summarize(campaign, minutes=args.minutes),
            'equal_cases': summarize(campaign, cases=args.cases),
            'extended_artifact_timings': artifact_timings(directory, snapshot)})
    print('Saved outcomes only; failure categories are not unique bugs.')
    for window in ('full', 'equal_time', 'equal_cases'):
        print('\n' + window + ':')
        print('campaign | tested | failed | failure % | minutes~ | categories')
        for campaign in report['campaigns']:
            result = campaign[window]
            print(f"{Path(campaign['directory']).name} | {result['saved_tested']} | "
                  f"{result['saved_failed']} | {result['failure_rate_percent']} | "
                  f"{result['observed_minutes_approx']} | {result['failure_categories']}"
                  + (' [window incomplete]' if result['window_reached'] is False else ''))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('x') as stream:
            json.dump(report, stream, indent=2, ensure_ascii=False)
            stream.write('\n')
    for campaign in report['campaigns']:
        for warning in campaign['warnings']:
            print('WARNING ' + Path(campaign['directory']).name + ': ' + warning)


if __name__ == '__main__':
    main()
