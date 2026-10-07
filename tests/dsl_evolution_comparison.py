"""Matched-parent GPU ablation of one-step versus evolving DSL exploration.

Both arms use the same revalidated common programs, seeds and compiler/oracle
settings. Reports measure executed source features and observed compiler IR
features, not compiler edge coverage or independently confirmed bug counts.
Existing background workloads are not stopped; throughput is descriptive.
"""
import argparse
from collections import Counter
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
    parser.add_argument('--source-variants', type=int, default=4,
                        help='Common-parent target variants in both arms (default 4)')
    parser.add_argument('--comparison', choices=('evolution', 'schedule'), default='evolution',
                        help='Compare one-step/evolving corpora, or fixed/adaptive DSL scheduling '
                             'with identical evolving corpora and oracle settings')
    parser.add_argument('--full-oracle', action='store_true', help='Include precision/identity and random configuration sweeps')
    args = parser.parse_args()
    if min(args.iterations, args.parents) < 1:
        parser.error('iterations and parents must be positive')
    if not 1 <= args.source_variants <= 32:
        parser.error('source-variants must be in [1, 32]')
    args.output.mkdir(parents=True, exist_ok=False)
    config = Config(backends=[args.backend], extended_prob=1, dsl_extend_prob=1,
                    dsl_source_variants=args.source_variants, dsl_matrix_prob=0.5, dsl_attributes=True,
                    save_artifacts=False, region_repeat_count=2,
                    random_config_count=2 if args.full_oracle else 0,
                    extended_precision_pair=args.full_oracle, extended_identity_pair=args.full_oracle)
    # TileLang 0.1.14 restores executable kernels without their lowered TIR.
    # A warm-cache pass therefore exposes fewer features for the SAME input,
    # changing both measurement and subsequent feedback-guided generation.
    # Disable that cache only in this comparison process and its children.
    if args.backend == 'tilelang':
        os.environ['TILELANG_DISABLE_CACHE'] = '1'

    def isolate_caches(name):
        directory = (args.output / 'caches' / name).resolve()
        os.environ['TRITON_CACHE_DIR'] = str(directory / 'triton')
        os.environ['TILELANG_CACHE_DIR'] = str(directory / 'tilelang')

    isolate_caches('parent_validation')
    report = {'backend': args.backend, 'comparison': args.comparison,
              'environment': environment(), 'config': asdict(config),
              'commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
              'cache_policy': {'isolated_per_arm_and_seed': True,
                               'tilelang_cache_disabled': args.backend == 'tilelang'},
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
            isolate_caches(f'{arm}_{seed}')
            arm_config = Config(**dict(asdict(config), seed=seed,
                                      dsl_evolve_prob=0 if args.comparison == 'evolution' and arm == 'baseline' else 0.5,
                                      corpus_feedback=args.comparison == 'schedule' or arm != 'baseline',
                                      dsl_adaptive_schedule=args.comparison == 'schedule' and arm == 'candidate',
                                      output_dir=str(args.output / f'{arm}_{seed}')))
            fuzzer = TileSmith(arm_config)
            for program, path in parents:
                fuzzer.dsl_stage.add(program, path)
            observed_compiler, executed_compiler, executed_source = set(), set(), set()
            entry = {'arm': arm, 'seed': seed, 'config': asdict(arm_config),
                     'path': str(fuzzer.output_dir), 'cases': [], 'complete': False,
                     'compiler_feature_measurement_complete': True}
            report['runs'].append(entry)
            execute = fuzzer.oracle.test
            def test(program):
                started = time.monotonic()
                bug = execute(program)
                records = fuzzer.oracle.last_compilation
                features = {feature for record in fuzzer.oracle.last_compilation for feature in record.get('features', [])}
                # Keep acquisition completeness separate from successful GPU
                # execution: missing IR must never look like a coverage loss.
                required_stages = ({'lowered_tir', 'cuda'} if args.backend == 'tilelang'
                                   else {'ttir', 'ttgir', 'llir', 'ptx'})
                missing_stages = {record['variant']: sorted(required_stages - record.get('stages', {}).keys())
                                  for record in records
                                  if required_stages - record.get('stages', {}).keys()}
                measurement_complete = bool(records) and not missing_stages
                if bug is None and not measurement_complete:
                    entry['compiler_feature_measurement_complete'] = False
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
                       'bucket': bug.failure_bucket if bug else None,
                       'compiler_feature_measurement_complete': measurement_complete,
                       'missing_compiler_stages': missing_stages,
                       'compiler_stages': {record['variant']: sorted(record.get('stages', {})) for record in records},
                       'compiler_feature_count': len(features),
                       'compiler_feature_kinds': dict(Counter(json.loads(feature)[0] for feature in features))}
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
                         dsl_schedule=fuzzer.dsl_stage.schedule.snapshot(include_parents=False),
                         failure_buckets=dict(fuzzer.failure_buckets))
            save()
    report['complete'] = True
    save()
    print('COMPLETE', args.output / 'comparison.json', flush=True)


if __name__ == '__main__':
    main()
