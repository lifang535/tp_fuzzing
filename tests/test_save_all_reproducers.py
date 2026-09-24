"""Every failing program reaches disk, whatever its label says.

Both throttles default to 0 = no cap. The root-cause label is coarse — every
unclassified compiler diagnostic collapses into 'other', every front-end
rejection shares one name — so a per-label cap discards distinct defects and
the program is gone: summary.json keeps the occurrence count but the
reproducer cannot be recovered afterwards. These tests drive the real campaign
loop with a stubbed oracle verdict to pin the *saving* contract (and the
opposite direction, so the assertion stays sensitive to the cap).
"""
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from src.config import Config
from src.workflow.fuzzer.fuzzer import TileSmith
from src.workflow.oracle import BugReport, BugType


def _run_campaign(directory, iterations, verdict, verbose=False, **overrides):
    """Run a real campaign whose oracle verdict comes from `verdict(call_index)`.

    Generation, dedup, feedback and the finally-block bookkeeping are the
    production ones; only the subprocess verdict is replaced, so the loop's
    saving decisions are exercised unchanged and stay CPU-only.
    """
    settings = dict(seed=42, output_dir=directory, dim_range=(1, 64),
                    coverage_probe_prob=0, region_typed_prob=0, extended_prob=0,
                    structural_feedback=False)
    settings.update(overrides)
    fuzzer = TileSmith(Config(**settings))
    calls = []

    def fake_test(program):
        calls.append(program)
        return verdict(len(calls) - 1)

    fuzzer.oracle.test = fake_test
    log = io.StringIO()
    with contextlib.redirect_stdout(log):
        fuzzer.run(iterations, verbose=verbose)
    return fuzzer, calls, log.getvalue()


def _saved(fuzzer, root_cause):
    """The campaign writes into its own timestamped folder under output_dir."""
    folder = fuzzer.output_dir / 'failed' / root_cause
    return sorted(path.stem for path in folder.glob('*.py'))


def _summary(fuzzer):
    return json.loads((fuzzer.output_dir / 'summary.json').read_text())


class DefaultThresholdsTests(unittest.TestCase):
    def test_both_throttles_default_to_unlimited(self):
        config = Config()
        self.assertEqual(config.max_same_root_cause, 0)
        self.assertEqual(config.max_oracle_unstable_saved, 0)


class BugSavingTests(unittest.TestCase):
    def test_one_coarse_label_saves_every_distinct_program(self):
        # Every program fails the same way: a coarse label merging distinct
        # defects is exactly the case a per-label cap would silently truncate.
        def verdict(index):
            return BugReport(bug_type=BugType.WRONG_RESULT,
                             error_message=f'WRONG RESULT: case {index}',
                             root_cause='other',
                             generated_code=f'# reproducer {index}\n')

        with tempfile.TemporaryDirectory() as directory:
            fuzzer, calls, log = _run_campaign(directory, 12, verdict)
            saved = _saved(fuzzer, 'other')
            self.assertEqual(fuzzer.stats.total_tested, 12)
            self.assertEqual(len(saved), 12)
            self.assertEqual(len(set(saved)), 12)  # distinct programs, distinct names
            self.assertEqual(len(fuzzer.stats.unique_bugs), 12)
            # Both halves of each reproducer, and the code that failed in it.
            folder = fuzzer.output_dir / 'failed' / 'other'
            for stem in saved:
                self.assertTrue((folder / f'{stem}.json').is_file())
            bodies = {path.read_text() for path in folder.glob('*.py')}
            self.assertEqual(bodies, {f'# reproducer {i}\n' for i in range(12)})

            summary = _summary(fuzzer)
            self.assertEqual(summary['max_same_root_cause'], 0)
            self.assertEqual(summary['bugs_total'], 12)
            self.assertEqual(summary['root_causes']['other'], 12)

    def test_a_positive_cap_still_throttles_saving(self):
        def verdict(index):
            return BugReport(bug_type=BugType.WRONG_RESULT,
                             error_message=f'WRONG RESULT: case {index}',
                             root_cause='other',
                             generated_code=f'# reproducer {index}\n')

        with tempfile.TemporaryDirectory() as directory:
            fuzzer, _, log = _run_campaign(directory, 8, verdict, max_same_root_cause=3)
            self.assertEqual(len(_saved(fuzzer, 'other')), 3)
            # Throttled saving never throttles counting.
            self.assertEqual(len(fuzzer.stats.bugs_found), 8)
            self.assertEqual(_summary(fuzzer)['root_causes']['other'], 8)


class OracleUnstableSavingTests(unittest.TestCase):
    def test_every_unstable_sample_is_saved_and_stays_out_of_the_bug_list(self):
        def verdict(index):
            return BugReport(bug_type=BugType.ORACLE_UNSTABLE,
                             error_message=f'ORACLE UNSTABLE: reference disagrees (case {index})',
                             root_cause='oracle_unstable',
                             generated_code=f'# unstable sample {index}\n')

        with tempfile.TemporaryDirectory() as directory:
            fuzzer, _, log = _run_campaign(directory, 7, verdict)
            self.assertEqual(len(_saved(fuzzer, 'oracle_unstable')), 7)
            self.assertEqual(fuzzer.stats.oracle_unstable, 7)
            # Audit samples, not bugs: they stay out of the bug lists entirely.
            self.assertEqual(fuzzer.stats.bugs_found, [])
            self.assertEqual(fuzzer.stats.unique_bugs, [])
            summary = _summary(fuzzer)
            self.assertEqual(summary['oracle_unstable'], 7)
            self.assertEqual(summary['bugs_total'], 0)
            self.assertEqual(summary['max_oracle_unstable_saved'], 0)

    def test_a_positive_cap_still_throttles_unstable_samples(self):
        def verdict(index):
            return BugReport(bug_type=BugType.ORACLE_UNSTABLE,
                             error_message=f'ORACLE UNSTABLE: reference disagrees (case {index})',
                             root_cause='oracle_unstable',
                             generated_code=f'# unstable sample {index}\n')

        with tempfile.TemporaryDirectory() as directory:
            fuzzer, _, log = _run_campaign(directory, 5, verdict, max_oracle_unstable_saved=2)
            self.assertEqual(len(_saved(fuzzer, 'oracle_unstable')), 2)
            self.assertEqual(fuzzer.stats.oracle_unstable, 5)


class LogMarkerTests(unittest.TestCase):
    def test_markers_report_what_reached_disk(self):
        # Operators read the campaign log, not the directory: the marker is the
        # only in-flight evidence that a reproducer was written.
        def verdict(index):
            if index % 2:
                return BugReport(bug_type=BugType.ORACLE_UNSTABLE, error_message='ORACLE UNSTABLE: x',
                                 root_cause='oracle_unstable', generated_code='# unstable\n')
            return BugReport(bug_type=BugType.WRONG_RESULT, error_message='WRONG RESULT: x',
                             root_cause='other', generated_code='# bug\n')

        # Capped: the second bug and the second sample are dropped, and the log
        # says so. NEW marks the first occurrence, saved the ones after it.
        with tempfile.TemporaryDirectory() as directory:
            _, _, log = _run_campaign(directory, 4, verdict, verbose=True,
                                      max_oracle_unstable_saved=1, max_same_root_cause=2)
            self.assertEqual(log.count('[FAILED] (NEW / other)'), 1)
            self.assertEqual(log.count('[FAILED] (saved / other)'), 1)
            self.assertEqual(log.count('[ORACLE UNSTABLE] (saved)'), 1)
            self.assertEqual(log.count('[ORACLE UNSTABLE] (dup)'), 1)
            self.assertNotIn('(dup / other)', log)

        # Uncapped (the shipped defaults): nothing is dropped, so `dup` cannot
        # appear anywhere — this is the marker that proves the new code runs.
        with tempfile.TemporaryDirectory() as directory:
            _, _, log = _run_campaign(directory, 4, verdict, verbose=True)
            self.assertEqual(log.count('[FAILED] (NEW / other)'), 1)
            self.assertEqual(log.count('[FAILED] (saved / other)'), 1)
            self.assertEqual(log.count('[ORACLE UNSTABLE] (saved)'), 2)
            self.assertNotIn('dup', log)


class IndependentCounterTests(unittest.TestCase):
    def test_an_uncapped_bug_run_does_not_exhaust_the_unstable_budget(self):
        # The unstable samples share one global counter, not a per-label one;
        # a run full of ordinary failures must not consume its budget.
        def verdict(index):
            if index < 6:
                return BugReport(bug_type=BugType.WRONG_RESULT, error_message='WRONG RESULT: x',
                                 root_cause='other', generated_code='# bug\n')
            return BugReport(bug_type=BugType.ORACLE_UNSTABLE, error_message='ORACLE UNSTABLE: x',
                             root_cause='oracle_unstable', generated_code='# unstable\n')

        with tempfile.TemporaryDirectory() as directory:
            fuzzer, _, log = _run_campaign(directory, 10, verdict)
            self.assertEqual(len(_saved(fuzzer, 'other')), 6)
            self.assertEqual(len(_saved(fuzzer, 'oracle_unstable')), 4)
            self.assertEqual(fuzzer.stats.oracle_unstable, 4)


if __name__ == '__main__':
    unittest.main()
