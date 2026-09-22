"""Fault injection for interrupted/early-exiting extended compiler processes."""
import contextlib
import copy
import hashlib
import io
import json
import random
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.config import Config
from src.workflow.fuzzer.fuzzer import TileSmith
from src.workflow.generator.extended import ExtendedGenerator
from src.workflow.oracle import Oracle, BugType


class ExtendedEvidenceTests(unittest.TestCase):
    def setUp(self):
        state = random.getstate()
        self.addCleanup(random.setstate, state)
        random.seed(0)
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.directory = Path(directory.name)
        # Pin the numeric sweeps off: the fixture records are built for the
        # plain variant count, and identity variants are pattern-dependent.
        self.config = Config(backends=['triton'], extended_prob=1, extended_config_depth=2,
                             extended_precision_pair=False, extended_identity_pair=False,
                             output_dir=str(self.directory))
        self.program = ExtendedGenerator(self.config, 'triton').generate('arithmetic')
        self.oracle = Oracle(self.config, 'triton')
        self.oracle.artifact_root = self.directory / 'artifacts'
        self.records = [dict(variant=f'triton_{i}', complete=True, features=['compiler_feature'],
                             stages={'ptx': {'sha256': hashlib.sha256(b'fixture').hexdigest()}})
                        for i in range(len(self.oracle.backend_impl.extended_variants(self.program, self.config)))]

    def launcher(self, *, records=True, progress=True, stdout='ALL PASSED\n',
                 returncode=0, timeout=False):
        def launch(command, **options):
            directory = Path(options['env']['TILESMITH_ARTIFACT_DIR'])
            for record in self.records:
                (directory / (record['variant'] + '.ptx')).write_text('fixture')
            for name, value, default in (
                    ('compilation.json', records, self.records),
                    ('progress.json', progress, {'stage': 'complete', 'variant': 'execute'})):
                if value is None:
                    continue
                value = default if value is True else value
                (directory / name).write_text(value if isinstance(value, str) else json.dumps(value))
            if timeout:
                raise subprocess.TimeoutExpired(command, 1, output=b'partial stdout', stderr=b'partial stderr')
            return subprocess.CompletedProcess(command, returncode, stdout, '')
        return launch

    def test_success_requires_manifest_completion_and_oracle_marker(self):
        for fault in ({'records': None}, {'progress': None}, {'stdout': ''},
                      {'stdout': 'prefix ALL PASSED suffix\n'},
                      {'progress': {'stage': 'execute', 'variant': 'triton_0'}},
                      {'progress': {'stage': 'complete', 'variant': 'compile_only'}}):
            with self.subTest(fault=fault), patch('src.workflow.oracle.process.run_isolated', self.launcher(**fault)):
                report = self.oracle.test(self.program)
                self.assertIsNotNone(report)
                self.assertIn('Incomplete extended', report.error_message)
        with patch('src.workflow.oracle.process.run_isolated', self.launcher()):
            self.assertIsNone(self.oracle.test(self.program))
        self.assertTrue(self.oracle.compilation_complete)

    def test_no_save_artifacts_validates_evidence_and_cleans_up(self):
        self.config.save_artifacts = False
        for fault in ({}, {'records': None}, {'returncode': 7}, {'timeout': True}):
            with self.subTest(fault=fault), patch(
                    'src.workflow.oracle.process.run_isolated', self.launcher(**fault)):
                report = self.oracle.test(self.program)
            self.assertEqual(report is None, not fault)
            if not fault:
                self.assertTrue(self.oracle.compilation_complete)
                self.assertEqual(self.oracle.last_compilation, self.records)
            self.assertFalse(self.oracle.last_artifact_dir.exists())
            self.assertFalse(self.oracle.artifact_root.exists())

    def test_truncated_or_malformed_evidence_does_not_abort(self):
        for field in ('records', 'progress'):
            for malformed in ('{"unfinished":', 'null', '42', '[]', '{"unexpected":true}'):
                with self.subTest(field=field, malformed=malformed), patch(
                        'src.workflow.oracle.process.run_isolated', self.launcher(**{field: malformed})):
                    self.assertIsNotNone(self.oracle.test(self.program))

    def test_duplicate_empty_or_unfinished_compilation_is_not_complete(self):
        duplicate = [self.records[0]] * len(self.records)
        incomplete = copy.deepcopy(self.records)
        incomplete[-1]['complete'] = False
        empty_stages = copy.deepcopy(self.records)
        empty_stages[-1]['stages'] = {}
        for records in (duplicate, incomplete, empty_stages, [None], {'unexpected': []}):
            with self.subTest(records=records), patch(
                    'src.workflow.oracle.process.run_isolated', self.launcher(records=records)):
                self.assertIsNotNone(self.oracle.test(self.program))
                self.assertFalse(self.oracle.compilation_complete)

    def test_failure_and_timeout_preserve_diagnostics_with_truncated_metadata(self):
        with patch('src.workflow.oracle.process.run_isolated', self.launcher(
                returncode=7, stdout='', records='{', progress='{')):
            report = self.oracle.test(self.program)
        self.assertIn('return code 7', report.error_message)
        self.assertIn('return code 7', (self.oracle.last_artifact_dir / 'run.log').read_text())
        with patch('src.workflow.oracle.process.run_isolated', self.launcher(
                records='{', progress='{', timeout=True)):
            report = self.oracle.test(self.program)
        self.assertEqual(report.bug_type, BugType.TIMEOUT)
        self.assertFalse(self.oracle.compilation_complete)
        self.assertEqual(self.oracle.last_compilation, [])
        log = (self.oracle.last_artifact_dir / 'run.log').read_text()
        self.assertIn('partial stdout', log)
        self.assertIn('partial stderr', log)

    def test_changed_compiler_artifact_cannot_count_as_success(self):
        def launch(command, **options):
            result = self.launcher()(command, **options)
            (Path(options['env']['TILESMITH_ARTIFACT_DIR']) / 'triton_0.ptx').write_text('changed')
            return result
        with patch('src.workflow.oracle.process.run_isolated', launch):
            report = self.oracle.test(self.program)
        self.assertIn('Compiler artifact changed', report.error_message)
        self.assertFalse(self.oracle.compilation_complete)
        self.assertEqual(self.oracle.last_compilation, [])

    def test_smoke_continues_after_truncated_or_false_success_evidence(self):
        import extended_smoke
        faults = [dict(records='{', progress='{', timeout=True),
                  dict(records=[self.records[0]] * len(self.records)),
                  dict(progress={'stage': 'complete', 'variant': 'compile_only'}),
                  dict(stdout='prefix ALL PASSED suffix\n'), dict(stdout=''), {}]
        launchers = iter([self.launcher(**fault) for fault in faults])
        output = self.directory / 'smoke'
        with contextlib.redirect_stdout(io.StringIO()), patch('sys.argv', [
                'extended_smoke.py', '--backend', 'triton', '--family', 'arithmetic',
                '--no-extended-precision', '--no-extended-identities',
                '--seeds', str(len(faults)), '--output', str(output)]), patch(
                'extended_smoke.run_isolated', side_effect=lambda *a, **kw: next(launchers)(*a, **kw)):
            self.assertEqual(extended_smoke.main(), 1)
        summary = json.loads((output / 'summary.json').read_text())
        self.assertTrue(summary['complete'])
        self.assertEqual([c['passed'] for c in summary['cases']], [False] * 5 + [True])
        self.assertIn('partial stderr', (output / summary['cases'][0]['case'] / 'run.log').read_text())
        from coverage_comparison import audit
        counts = audit([output])['triton']
        self.assertEqual((counts['attempted'], counts['passed']), (6, 1))

    def test_smoke_marks_first_case_interruption_incomplete(self):
        import extended_smoke
        output = self.directory / 'interrupted'
        with contextlib.redirect_stdout(io.StringIO()), patch('sys.argv', [
                'extended_smoke.py', '--backend', 'triton', '--family', 'arithmetic',
                '--output', str(output)]), patch('extended_smoke.run_isolated', side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):
                extended_smoke.main()
        summary = json.loads((output / 'summary.json').read_text())
        self.assertFalse(summary['complete'])
        from coverage_comparison import audit
        with self.assertRaisesRegex(ValueError, 'incomplete'):
            audit([output])

    def test_compile_only_requires_its_own_completion_evidence(self):
        self.config.compile_only = True
        with patch('src.workflow.oracle.process.run_isolated', self.launcher()):
            self.assertIsNotNone(self.oracle.test(self.program))
        with patch('src.workflow.oracle.process.run_isolated', self.launcher(
                progress={'stage': 'complete', 'variant': 'compile_only'},
                stdout='COMPILE PASSED (no GPU execution)\n')):
            self.assertIsNone(self.oracle.test(self.program))
        self.assertTrue(self.oracle.compilation_complete)

    def test_campaign_continues_and_excludes_incomplete_execution_from_coverage(self):
        other = copy.deepcopy(self.program)
        other.blocks = 4
        launchers = iter([self.launcher(records='{', progress='{'), self.launcher()])
        with contextlib.redirect_stdout(io.StringIO()):
            fuzzer = TileSmith(self.config)
            with patch.object(fuzzer, '_generate_test_case', side_effect=[self.program, other]), patch(
                    'src.workflow.oracle.process.run_isolated',
                    side_effect=lambda *a, **kw: next(launchers)(*a, **kw)):
                stats = fuzzer.run(2, verbose=False)
        self.assertEqual(stats.total_tested, 2)
        self.assertEqual(stats.programs_passed, 1)
        self.assertEqual(stats.programs_compiled, 1)
        self.assertEqual(len(stats.bugs_found), 1)
        self.assertTrue(fuzzer.feedback.passed)
        self.assertEqual(set(fuzzer.feedback.passed.values()), {1})

    def test_extended_crash_location_comes_from_progress_manifest(self):
        """Location-aware root causes: the progress.json stage:variant beats
        message inference; without a manifest the location is inferred (here
        empty for a bare exit-code failure)."""
        with patch('src.workflow.oracle.process.run_isolated', self.launcher(
                returncode=7, stdout='', progress={'stage': 'execute', 'variant': 'triton_2'})):
            report = self.oracle.test(self.program)
        self.assertEqual(report.location, 'execute:triton_2')
        self.assertIn('location', report.to_dict())
        with patch('src.workflow.oracle.process.run_isolated', self.launcher(
                returncode=7, stdout='', progress=None)):
            report = self.oracle.test(self.program)
        self.assertEqual(report.location, '')

    def test_campaign_summary_records_root_cause_locations(self):
        from src.workflow.oracle import BugReport

        def buggy(program):
            report = BugReport(BugType.RUNTIME_CRASH, 'RuntimeError: CUDA error: out of memory\n')
            report.classify_root_cause('triton')
            report.location = 'execute:triton_2'
            return report

        with contextlib.redirect_stdout(io.StringIO()):
            fuzzer = TileSmith(self.config)
            with patch.object(fuzzer, '_generate_test_case', return_value=self.program), \
                    patch.object(fuzzer.oracle, 'test', side_effect=buggy):
                fuzzer.run(1, verbose=False)
        summary = json.loads((fuzzer.output_dir / 'summary.json').read_text())
        self.assertEqual(summary['root_causes'], {'gpu_oom': 1})
        self.assertEqual(summary['root_cause_locations'], {'gpu_oom': {'execute:triton_2': 1}})


if __name__ == '__main__':
    unittest.main()
