"""Prevent failed/compile-only programs and dead source from inflating reports."""
import copy
import hashlib
import json
from pathlib import Path
import random
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from src.config import Config
from src.ir.extended import Node, Value, TensorType
from src.workflow.coverage_audit import CAPABILITIES, program_capabilities
from src.workflow.generator.extended import ExtendedGenerator
from src.workflow.generator.region_generator import RegionGenerator
from coverage_comparison import audit, baseline


class CoverageAuditTests(unittest.TestCase):
    def setUp(self):
        state = random.getstate()
        self.addCleanup(random.setstate, state)
        random.seed(1)

    def test_shared_inventory_detects_increment_without_family_labels(self):
        # The int8 flavor produces gemm-only programs whose internal_matmul
        # would land in the old program too; this test compares the shared
        # inventory of the float domain.
        config = Config(coverage_probe_prob=0, dim_range=(16, 80), region_int8_prob=0)
        old = RegionGenerator(config, 'triton').generate()
        new = ExtendedGenerator(config, 'triton').generate('mixed')
        old_features, new_features = program_capabilities(old), program_capabilities(new)
        self.assertTrue(old_features <= CAPABILITIES.keys())
        self.assertTrue(new_features <= CAPABILITIES.keys())
        self.assertTrue({'internal_matmul', 'dependent_matmul', 'computed_memory_mask',
                         'bounded_while', 'multiple_function_results', 'scratch_contents_checked'} <= new_features - old_features)
        new.family = 'arithmetic'
        self.assertEqual(program_capabilities(new), new_features)

    def test_dead_arithmetic_does_not_count_and_observations_are_optional(self):
        p = ExtendedGenerator(Config(extended_int8_prob=0, extended_fma_prob=0,
                           extended_shape_op_prob=0, extended_atomic_prob=0),
                       'triton').generate('shape_matmul')
        p.body.operations.extend([
            Node('constant', [Value('unused_i', TensorType('int32'))], attrs={'value': 1}),
            Node('bitxor', [Value('unused_xor', TensorType('int32'))], ['unused_i', 'unused_i'])])
        self.assertNotIn('integer_dataflow', program_capabilities(p))
        p.observations.append('unused_xor')
        self.assertIn('integer_dataflow', program_capabilities(p))
        p.observation_pair = False
        self.assertNotIn('integer_dataflow', program_capabilities(p))
        self.assertNotIn('intermediate_observations', program_capabilities(p))

    def test_report_excludes_failed_cases_and_rejects_compile_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            p = RegionGenerator(Config(coverage_probe_prob=0, dim_range=(16, 80)), 'triton').generate()
            for label in ('triton_pass', 'triton_fail'):
                case = path / label
                case.mkdir()
                (case / 'program.json').write_text(json.dumps(p.to_dict()))
                (case / 'run.log').write_text('ALL PASSED' if label.endswith('pass') else 'failure')
            summary = {'mode': 'execute', 'cases': [
                {'case': label, 'artifact_dir': str(path / label), 'compiled': True, 'passed': label.endswith('pass')}
                for label in ('triton_pass', 'triton_fail')]}
            (path / 'summary.json').write_text(json.dumps(summary))
            result = audit([path])['triton']
            self.assertEqual((result['attempted'], result['compiled'], result['passed']), (2, 2, 1))
            self.assertTrue(all(count == 1 for count in result['capabilities'].values()))
            summary['mode'] = 'compile_only'
            (path / 'summary.json').write_text(json.dumps(summary))
            with self.assertRaisesRegex(ValueError, 'compile-only'):
                audit([path])

    def test_report_rejects_changed_program_and_missing_compiler_evidence(self):
        from src.backends import get_backend
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            case = path / 'triton_checked'
            case.mkdir()
            program = ExtendedGenerator(Config(), 'triton').generate('arithmetic')
            files = {'program.json': json.dumps(program.to_dict()), 'repro.py': '# checked source',
                     'run.log': 'ALL PASSED',
                     'progress.json': json.dumps({'stage': 'complete', 'variant': 'execute'})}
            # Pin the sweep knobs in the fixture so the audit replays the same
            # variant count the records were built from.
            fixture = Config(extended_precision_pair=False, extended_identity_pair=False)
            records = []
            for i, _ in enumerate(get_backend('triton').extended_variants(program, fixture)):
                label = f'triton_{i}'
                files[label + '.ptx'] = 'checked compiler output'
                records.append({'variant': label, 'complete': True, 'features': [], 'stages': {
                    'ptx': {'sha256': hashlib.sha256(files[label + '.ptx'].encode()).hexdigest()}}})
            files['compilation.json'] = json.dumps(records)
            for name, contents in files.items():
                (case / name).write_text(contents)
            entry = {'case': case.name, 'artifact_dir': str(case), 'compiled': True, 'passed': True,
                     'source_sha256': hashlib.sha256(files['repro.py'].encode()).hexdigest(),
                     'program_sha256': hashlib.sha256(files['program.json'].encode()).hexdigest()}
            (path / 'summary.json').write_text(json.dumps({'mode': 'execute', 'cases': [entry],
                'generation_config': {'extended_config_depth': 1, 'extended_precision_pair': False,
                                      'extended_identity_pair': False}}))
            self.assertEqual(audit([path])['triton']['passed'], 1)
            for name, message in [('program.json', 'Program changed'), ('repro.py', 'Reproducer changed'),
                                  ('triton_0.ptx', 'Compiler artifact changed')]:
                (case / name).write_text(files[name] + ' ')
                with self.assertRaisesRegex(ValueError, message):
                    audit([path])
                (case / name).write_text(files[name])
            for malformed in ('{', 'null', json.dumps([records[0]] * len(records)),
                              json.dumps([dict(r, stages={}) for r in records])):
                with self.subTest(manifest=malformed):
                    (case / 'compilation.json').write_text(malformed)
                    with self.assertRaisesRegex(ValueError, 'Incomplete compilation'):
                        audit([path])
            (case / 'compilation.json').write_text(files['compilation.json'])
            for fault in ('failed_exit', 'false_marker', 'compile_only'):
                with self.subTest(fault=fault):
                    bad_entry = dict(entry, returncode=7 if fault == 'failed_exit' else 0)
                    (path / 'summary.json').write_text(json.dumps({'mode': 'execute', 'cases': [bad_entry]}))
                    (case / 'run.log').write_text('prefix ALL PASSED suffix' if fault == 'false_marker' else 'ALL PASSED')
                    (case / 'progress.json').write_text(json.dumps({
                        'stage': 'complete', 'variant': 'compile_only' if fault == 'compile_only' else 'execute'}))
                    with self.assertRaises(ValueError):
                        audit([path])
            (path / 'summary.json').write_text(json.dumps({'mode': 'execute', 'cases': [entry]}))
            (case / 'run.log').write_text(files['run.log'])
            (case / 'progress.json').write_text(files['progress.json'])
            (case / 'compilation.json').unlink()
            with self.assertRaisesRegex(ValueError, 'Missing compilation evidence'):
                audit([path])

    def test_timeout_keeps_partial_output_and_continues_corpus(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'baseline'
            args = SimpleNamespace(output=path, seeds=2, backend='triton', timeout=1)
            timeout = subprocess.TimeoutExpired('repro.py', 1, output=b'compile started\n', stderr=b'partial diagnostic\n')
            success = subprocess.CompletedProcess('repro.py', 0, 'ALL PASSED', '')
            with patch('coverage_comparison.run_isolated', side_effect=[timeout, success]):
                self.assertEqual(baseline(args), 1)
            summary = json.loads((path / 'summary.json').read_text())
            self.assertTrue(summary['complete'])
            self.assertEqual([c['passed'] for c in summary['cases']], [False, True])
            self.assertIn('partial diagnostic\n\nTIMEOUT', (path / 'triton_region_0' / 'run.log').read_text())
            self.assertEqual(audit([path])['triton']['passed'], 1)
            summary['complete'] = False
            (path / 'summary.json').write_text(json.dumps(summary))
            with self.assertRaisesRegex(ValueError, 'incomplete'):
                audit([path])
