import json
import os
from pathlib import Path
import tempfile
import unittest

from campaign_comparison import failure_bucket, read_campaign, summarize


class CampaignComparisonTests(unittest.TestCase):
    def test_live_results_exclude_artifacts_partial_and_compile_only(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / '2026.09.17-18.45_tilelang_hard-shape_seed=42'
            for folder in ('passed', 'failed/wrong_result', 'compiled', 'artifacts/a'):
                (root / folder).mkdir(parents=True)
            start = read_campaign(root, float('inf'))['start']
            records = {
                'passed/one.json': {'type': 'region', 'spec': {'coverage_probe': True}},
                'failed/wrong_result/two.json': {'root_cause': 'wrong_result', 'timestamp': start + 20,
                    'params': {'extended_program': {'type': 'extended', 'family': 'mixed'}}},
                'artifacts/a/program.json': {'type': 'extended', 'family': 'mixed'},
                'compiled/three.json': {'type': 'extended', 'validation_mode': 'compile_only'},
                'passed/four.json': {'type': 'extended', 'validation_mode': 'compile_only'},
            }
            for name, record in records.items():
                path = root / name
                path.write_text(json.dumps(record))
                os.utime(path, (start + 10, start + 10))
            partial = root / 'passed/partial.json'
            partial.write_text('{')
            os.utime(partial, (start + 10, start + 10))
            campaign = read_campaign(root, start + 30)
            result = summarize(campaign)
            self.assertEqual((result['saved_tested'], result['saved_failed']), (2, 1))
            self.assertEqual(result['domains'], {'probe': {'passed': 1}, 'extended:mixed': {'failed': 1}})
            self.assertTrue(any('partial.json' in warning for warning in campaign['warnings']))
            self.assertEqual(summarize(read_campaign(root, start + 15))['saved_tested'], 1)

    def test_time_and_case_windows_use_chronological_outcomes(self):
        campaign = {'start': 100, 'rows': [
            {'timestamp': stamp, 'status': 'failed' if stamp == 160 else 'passed',
             'domain': 'region', 'category': 'wrong_result', 'triage': 'needs_review'}
            for stamp in (110, 160, 221)]}
        window = summarize(campaign, minutes=1)
        self.assertEqual((window['saved_tested'], window['saved_failed']), (2, 1))
        self.assertEqual(window['observed_minutes_approx'], 1)
        self.assertTrue(window['window_reached'])
        self.assertEqual(summarize(campaign, cases=1)['saved_failed'], 0)
        self.assertFalse(summarize(campaign, cases=4)['window_reached'])
        self.assertFalse(summarize(campaign, minutes=3)['window_reached'])

    def test_triage_reads_error_despite_misleading_legacy_category(self):
        self.assertEqual(failure_bucket({'root_cause': 'assertion_failure',
            'error_message': 'CUDA error: out of memory; enable device-side assertions'}), 'resource_limit')
        self.assertEqual(failure_bucket({'root_cause': 'other',
            'error_message': 'self.stride(-1) must be 1 to view Float as Byte'}), 'harness_or_evidence')
        self.assertEqual(failure_bucket({'root_cause': 'timeout'}), 'timeout')
        self.assertEqual(failure_bucket({'root_cause': 'wrong_result'}), 'needs_review')

    def test_empty_campaign_does_not_claim_completed_window(self):
        result = summarize({'start': 100, 'rows': []}, minutes=20)
        self.assertEqual(result['saved_tested'], 0)
        self.assertIsNone(result['failure_rate_percent'])
        self.assertFalse(result['window_reached'])

    def test_compile_only_campaign_is_not_an_execution_baseline(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / '2026.09.17-18.45_triton_hard-shape_seed=42'
            root.mkdir()
            (root / 'summary.json').write_text(json.dumps({'compile_only': True}))
            with self.assertRaisesRegex(ValueError, 'Compile-only campaigns'):
                read_campaign(root, float('inf'))


if __name__ == '__main__':
    unittest.main()
