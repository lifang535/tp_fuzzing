"""Regressions for escaping audited repeats without discarding new failures."""
from collections import Counter
import copy
import json
import random
import tempfile
import unittest
from unittest.mock import patch

from src.config import Config
from src.workflow.feedback import StructuralFeedback, key
from src.workflow.fuzzer.dsl_schedule import DSLSchedule
from src.workflow.fuzzer.dsl_stage import DSLStage
from src.workflow.fuzzer.fuzzer import TileSmith
from src.workflow.generator.dsl_extend import extend_passed
from src.workflow.generator.extended import ExtendedGenerator
from src.workflow.oracle import BugReport, BugType


class DSLScheduleTests(unittest.TestCase):
    def setUp(self):
        state = random.getstate()
        self.addCleanup(random.setstate, state)
        random.seed(91)

    def observe(self, schedule, parent, action, *, novel=False, repeat=False, seconds=1):
        schedule.observe(parent, action, 'scan_sum', structural_novelty=novel,
                         compiler_novelty=novel, seconds=seconds, known_repeat=repeat)

    def test_budget_moves_to_productive_actions_with_exploration_remaining(self):
        schedule = DSLSchedule()
        for _ in range(40):
            self.observe(schedule, 'seed', 'compose', repeat=True)
            self.observe(schedule, 'seed', 'mutate', novel=True)
        actions = ['compose', 'mutate']
        weights = [schedule.weight(action, digest='seed') for action in actions]
        draws = Counter(schedule.choose(actions, weights) for _ in range(4000))
        self.assertGreater(draws['mutate'], 4 * draws['compose'])
        self.assertGreater(draws['compose'], 100)
        # Disabled actions cannot be selected by the exploration mixture.
        self.assertEqual({schedule.choose(actions, [0, 1]) for _ in range(100)}, {'mutate'})

    def test_parent_specific_outcomes_and_recent_recovery(self):
        schedule = DSLSchedule()
        for _ in range(40):
            self.observe(schedule, 'bad', 'compose', repeat=True)
            self.observe(schedule, 'good', 'compose', novel=True)
        before = schedule.weight('compose', 'scan_sum', 'bad')
        self.assertGreater(schedule.weight('compose', 'scan_sum', 'good'), 3 * before)
        self.assertGreater(schedule.parent_weight('good'), 3 * schedule.parent_weight('bad'))
        for _ in range(40):
            self.observe(schedule, 'bad', 'compose', novel=True)
        self.assertGreater(schedule.weight('compose', 'scan_sum', 'bad'), 3 * before)

    def test_cost_bias_is_bounded_and_unknown_failures_have_no_repeat_penalty(self):
        slow, fast, known = DSLSchedule(), DSLSchedule(), DSLSchedule()
        for _ in range(30):
            self.observe(slow, 'seed', 'compose', seconds=100)
            self.observe(fast, 'seed', 'compose', seconds=0.001)
            self.observe(known, 'seed', 'compose', seconds=0.001, repeat=True)
        self.assertGreater(fast.weight('compose'), known.weight('compose'))
        self.assertEqual(fast.total.known_repeats, 0)
        # A campaign of cheap failures does not gain an unbounded advantage.
        self.assertAlmostEqual(fast.weight('compose'), slow.weight('compose'))

    def test_schedule_round_trip_pruning_and_invalid_measurements(self):
        schedule = DSLSchedule()
        self.observe(schedule, 'kept', 'compose', novel=True)
        self.observe(schedule, 'evicted', 'mutate', repeat=True)
        state = json.loads(json.dumps(schedule.snapshot()))
        restored = DSLSchedule()
        restored.restore(state)
        self.assertEqual(restored.snapshot(), state)
        self.assertEqual(restored.weight('compose', digest='kept'),
                         schedule.weight('compose', digest='kept'))
        restored.prune({'kept'})
        self.assertEqual(set(restored.parents), {'kept'})
        before = restored.snapshot()
        for value in (float('nan'), float('inf'), -1):
            broken = copy.deepcopy(state)
            broken['total']['seconds'] = value
            with self.assertRaises(ValueError):
                restored.restore(broken)
            self.assertEqual(restored.snapshot(), before)


class DSLFeedbackIntegrationTests(unittest.TestCase):
    def setUp(self):
        state = random.getstate()
        self.addCleanup(random.setstate, state)
        random.seed(91)
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.config = Config(seed=91, backends=['tilelang'], output_dir=directory.name,
                             extended_prob=1, dsl_extend_prob=1, dsl_evolve_prob=1,
                             extended_atomic_prob=0, extended_fma_prob=0,
                             extended_shape_op_prob=0, extended_int8_prob=0,
                             extended_elementwise_prob=0)
        self.parent = ExtendedGenerator(self.config, 'tilelang').generate('arithmetic')

    def test_target_seed_known_failure_penalty_and_partial_compilation_credit(self):
        feedback = StructuralFeedback()
        stage = DSLStage(self.config, 'tilelang', feedback=feedback)
        stage.add(self.parent, 'parent.json')
        child = extend_passed(self.parent, 'tilelang', 'scan_sum', self.config)
        lineage = {'extension_op': 'scan_sum', 'extension_action': 'extend',
                   'extension_depth': 1, 'source_sha256': stage.digest(self.parent)}
        stage.retain_target(child, 'child.json', lineage, [])
        digest = stage.digest(child)
        initial = stage.source_weight(child, digest)
        for _ in range(5):
            feedback.observe_known_failure(digest, 'tilelang_reduce_layout')
        self.assertGreater(stage.source_weight(child, digest), 0)
        self.assertLess(stage.source_weight(child, digest), initial / 2)

        feature = key('compiler_stage', 'lowered_tir')
        records = [{'features': [feature]}]
        self.assertEqual(stage.observe_compilation(digest, records, passed=False), 1)
        self.assertEqual(stage.compiler[feature], 1)
        self.assertFalse(stage.compiler_passed)
        self.assertNotIn(feature, stage.source_compiler.get(digest, []))
        self.assertFalse(feedback.compiler)
        self.assertEqual(stage.observe_compilation(digest, records, passed=True), 1)
        self.assertEqual(stage.observe_compilation(digest, records, passed=True), 0)
        self.assertEqual(stage.compiler_passed[feature], 2)
        restored = DSLStage(self.config, 'tilelang', feedback=feedback)
        restored.restore(json.loads(json.dumps(stage.snapshot())))
        self.assertEqual(restored.source_weight(child, digest), stage.source_weight(child, digest))

    def test_old_snapshot_restores_passing_features_and_empty_schedule(self):
        stage = DSLStage(self.config, 'tilelang')
        stage.add(self.parent, 'parent.json')
        state = stage.snapshot()
        state.update(version=2, compiler={'old_feature': 3})
        del state['compiler_passed'], state['schedule']
        restored = DSLStage(self.config, 'tilelang')
        restored.restore(state)
        self.assertEqual(restored.compiler_passed, Counter(old_feature=3))
        self.assertEqual(restored.schedule.total.tested, 0)

    def test_one_step_campaign_does_not_reward_the_same_source_features_forever(self):
        self.config.dsl_evolve_prob = 0
        stage = DSLStage(self.config, 'tilelang')
        child = extend_passed(self.parent, 'tilelang', 'scan_sum', self.config)
        lineage = {'extension_op': 'scan_sum', 'source_sha256': stage.digest(self.parent)}
        for _ in range(5):
            stage.observe_outcome(child, lineage, [], passed=True, seconds=1)
            self.assertFalse(stage.retain_target(child, 'child.json', lineage, []))
        self.assertEqual(stage.schedule.total.novel, 1)
        self.assertTrue(stage.target_structural)

    def test_campaign_records_audited_repeats_saves_all_failures_and_resumes(self):
        fuzzer = TileSmith(self.config)
        calls = 0
        def oracle(program):
            nonlocal calls
            calls += 1
            fuzzer.oracle.last_compilation = [{'features': [key('compiler_stage', 'lowered_tir')]}]
            fuzzer.oracle.compilation_complete = calls <= 2
            if calls <= 2:
                return None
            return BugReport(BugType.COMPILE_CRASH, 'Cannot convert type boolx8 to CUDA type',
                             root_cause='tilelang_codegen_error', params=program.params_dict)
        with patch.object(fuzzer.generator, 'generate', return_value=self.parent), \
                patch.object(fuzzer.oracle, 'test', side_effect=oracle):
            fuzzer.run(6, verbose=False)
        self.assertEqual(fuzzer.dsl_stage.schedule.total.tested, 5)
        self.assertEqual(fuzzer.dsl_stage.schedule.total.known_repeats, 3)
        self.assertEqual(len(list((fuzzer.output_dir / 'failed/tilelang_codegen_error').glob('*.json'))), 4)
        progress = json.loads((fuzzer.output_dir / 'coverage_progress.json').read_text())
        self.assertEqual(progress['dsl_schedule']['total']['known_repeats'], 3)
        self.assertNotIn('parents', progress['dsl_schedule'])
        restored = TileSmith(self.config, resume_dir=str(fuzzer.output_dir))
        self.assertEqual(restored.dsl_stage.schedule.snapshot(), fuzzer.dsl_stage.schedule.snapshot())

    def test_wrong_result_candidates_are_saved_without_known_repeat_penalty(self):
        fuzzer = TileSmith(self.config)
        calls = 0
        def oracle(program):
            nonlocal calls
            calls += 1
            if calls <= 2:
                return None
            return BugReport(BugType.WRONG_RESULT, 'WRONG RESULT: same broad message',
                             root_cause='wrong_result', params=program.params_dict)
        with patch.object(fuzzer.generator, 'generate', return_value=self.parent), \
                patch.object(fuzzer.oracle, 'test', side_effect=oracle):
            fuzzer.run(5, verbose=False)
        self.assertEqual(fuzzer.dsl_stage.schedule.total.tested, 4)
        self.assertEqual(fuzzer.dsl_stage.schedule.total.known_repeats, 0)
        self.assertFalse(fuzzer.feedback.known_seed_failures)
        self.assertEqual(len(list((fuzzer.output_dir / 'failed/wrong_result').glob('*.json'))), 3)

    def test_disabled_adaptation_does_not_consume_schedule_choices(self):
        stage = DSLStage(self.config, 'tilelang')
        for structural, adaptive in ((False, True), (True, False)):
            self.config.structural_feedback = structural
            self.config.dsl_adaptive_schedule = adaptive
            with patch.object(stage.schedule, 'choose', side_effect=AssertionError('disabled')):
                self.assertIn(stage._choose(['a', 'b'], [1, 1]), ('a', 'b'))


if __name__ == '__main__':
    unittest.main()
