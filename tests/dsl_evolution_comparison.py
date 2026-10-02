"""Matched-parent GPU ablation of one-step versus evolving DSL exploration.

Both arms use the same revalidated common programs, seeds and compiler/oracle
settings. Reports measure executed source features and observed compiler IR
features, not compiler edge coverage or independently confirmed bug counts.
Existing background workloads are not stopped; throughput is descriptive.
"""
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import Config
from src.ir.serialization import program_from_dict
from src.workflow.feedback import program_digest, program_features
from src.workflow.fuzzer.fuzzer import TileSmith
from src.workflow.generator.dsl_extend import is_common_seed
from src.workflow.oracle import Oracle
from src.backends.common.versions import environment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend', required=True, choices=('triton', 'tilelang'))
    parser.add_argument('--passed-dir', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--iterations', type=int, default=24)
    parser.add_argument('--seeds', type=int, nargs='+', default=[101, 202])
    parser.add_argument('--parents', type=int, default=5)
    parser.add_argument('--full-oracle', action='store_true', help='Include precision/identity and random configuration sweeps')
    args = parser.parse_args()
    if min(args.iterations, args.parents) < 1:
        parser.error('iterations and parents must be positive')
    args.output.mkdir(parents=True, exist_ok=False)
    config = Config(backends=[args.backend], extended_prob=1, dsl_extend_prob=1,
                    save_artifacts=False, region_repeat_count=2,
                    random_config_count=2 if args.full_oracle else 0,
                    extended_precision_pair=args.full_oracle, extended_identity_pair=args.full_oracle)
    # Keep validation caches on the output filesystem, isolated from running campaigns.
    os.environ['TRITON_CACHE_DIR'] = str((args.output / 'triton_cache').resolve())
    os.environ['TILELANG_CACHE_DIR'] = str((args.output / 'tilelang_cache').resolve())
    report = {'backend': args.backend, 'environment': environment(), 'config': asdict(config),
              'commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
              'limitations': ['Short matched-parent ablation, not whole-campaign bug-yield evidence',
                              'IR/source feature counts are not compiler edge coverage',
                              'Background campaigns may contend for resources; wall time is descriptive'],
              'parents': [], 'rejected_parents': [], 'runs': [], 'complete': False}
    def save():
        path = args.output / 'comparison.json'
        temporary = path.with_suffix('.tmp')
        temporary.write_text(json.dumps(report, indent=2))
        temporary.replace(path)
    save()
    oracle = Oracle(config, args.backend)
    parents = []
    # Round-robin families, bounded attempts. Only saved executed common IR is eligible.
    families = ('arithmetic', 'indexed_memory', 'shape_matmul', 'control_calls', 'mixed')
    files = {family: iter(sorted(args.passed_dir.glob(f'passed_extended_{family}_*.json')))
             for family in families}
    for _ in range(8):
        for family in families:
            path = next(files[family], None)
            if path is None:
                continue
            record = json.loads(path.read_text())
            if record.get('validation_mode') != 'execute':
                continue
            program = program_from_dict(record)
            if not is_common_seed(program):
                continue
            bug = oracle.test(program)
            if bug is not None:
                report['rejected_parents'].append({'path': str(path), 'failure': bug.to_dict()})
                save()
                continue
            parents.append((program, path.resolve()))
            report['parents'].append({'path': str(path.resolve()), 'digest': program_digest(program), 'family': family})
            save()
            print('PARENT', len(parents), family, flush=True)
            if len(parents) >= args.parents:
                break
        if len(parents) >= args.parents:
            break
    if not parents:
        raise RuntimeError('No common parents passed revalidation')
    for index, seed in enumerate(args.seeds):
        # Alternate order to reduce a consistent first-arm cache/order advantage.
        for arm in (('baseline', 'candidate') if index % 2 == 0 else ('candidate', 'baseline')):
            arm_config = Config(**dict(asdict(config), seed=seed,
                                      dsl_evolve_prob=0 if arm == 'baseline' else 0.5,
                                      corpus_feedback=arm != 'baseline',
                                      output_dir=str(args.output / f'{arm}_{seed}')))
            fuzzer = TileSmith(arm_config)
            for program, path in parents:
                fuzzer.dsl_stage.add(program, path)
            observed_compiler, executed_compiler, executed_source = set(), set(), set()
            entry = {'arm': arm, 'seed': seed, 'path': str(fuzzer.output_dir), 'cases': [], 'complete': False}
            report['runs'].append(entry)
            execute = fuzzer.oracle.test
            def test(program):
                started = time.monotonic()
                bug = execute(program)
                features = {feature for record in fuzzer.oracle.last_compilation for feature in record.get('features', [])}
                observed_compiler.update(features)
                if bug is None:
                    executed_compiler.update(features)
                    executed_source.update(program_features(program))
                lineage = fuzzer._current_extension or {}
                row = {'digest': program_digest(program), 'passed': bug is None,
                       'compiled': fuzzer.oracle.compilation_complete,
                       'seconds': round(time.monotonic() - started, 3),
                       'depth': lineage.get('extension_depth', 0), 'action': lineage.get('extension_action'),
                       'op': lineage.get('extension_op'), 'category': bug.root_cause if bug else None,
                       'bucket': bug.failure_bucket if bug else None}
                entry['cases'].append(row)
                entry.update(observed_compiler_features=sorted(observed_compiler),
                             executed_compiler_features=sorted(executed_compiler),
                             executed_source_features=sorted(executed_source))
                save()
                print(args.backend, arm, seed, len(entry['cases']), 'PASS' if bug is None else bug.root_cause,
                      'depth', row['depth'], 'IR', len(executed_compiler), flush=True)
                return bug
            fuzzer.oracle.test = test
            started = time.monotonic()
            fuzzer.run(args.iterations, verbose=False)
            entry.update(complete=True, seconds=round(time.monotonic() - started, 3),
                         tested=fuzzer.stats.total_tested, passed=fuzzer.stats.programs_passed,
                         compiled=fuzzer.stats.programs_compiled,
                         failure_buckets=dict(fuzzer.failure_buckets))
            save()
    report['complete'] = True
    save()
    print('COMPLETE', args.output / 'comparison.json', flush=True)


if __name__ == '__main__':
    main()
