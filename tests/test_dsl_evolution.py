"""Contracts for a separate, bounded, recursively explored target corpus."""
import ast
from collections import Counter
import copy
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

import torch

from src.backends import get_backend
from src.config import Config
from src.workflow.emitter.extended_runtime import extended_inputs, extended_reference
from src.workflow.feedback import StructuralFeedback, corpus_eviction, program_features
from src.workflow.fuzzer.dsl_stage import DSLStage
from src.workflow.fuzzer.fuzzer import TileSmith
from src.workflow.generator.dsl_extend import eligible_ops, extend_passed, loop_target, is_common_seed
from src.workflow.generator.extended import ExtendedGenerator
from src.workflow.oracle import BugReport, BugType


class DSLEvolutionTests(unittest.TestCase):
    def setUp(self):
        state = random.getstate()
        self.addCleanup(random.setstate, state)
        random.seed(73)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.config = Config(seed=73, backends=['tilelang'], output_dir=self.temp.name,
                             extended_prob=1, dsl_extend_prob=1, dsl_evolve_prob=1,
                             extended_atomic_prob=0, extended_fma_prob=0,
                             extended_shape_op_prob=0, extended_int8_prob=0,
                             extended_elementwise_prob=0)
        self.parent = ExtendedGenerator(self.config, 'tilelang').generate('arithmetic')

    def test_composition_consumes_previous_checked_target_output(self):
        first = extend_passed(self.parent, 'tilelang', 'scan_sum', self.config)
        previous = first.body.returns[-1]
        original = first.to_dict()
        second = extend_passed(first, 'tilelang', 'scan_max', self.config,
                               allow_target=True, input_name=previous)
        producers = {v.name: n for n in second.body.operations for v in n.results}
        def ancestors(name):
            return {name} | set().union(*(ancestors(v) for v in producers[name].operands))
        self.assertIn(previous, ancestors(second.body.returns[-1]))
        self.assertEqual(first.to_dict(), original)
        self.assertFalse(eligible_ops(first, 'tilelang'))
        self.assertTrue(eligible_ops(first, 'tilelang', allow_target=True))
        ast.parse(get_backend('tilelang').make_emitter(self.config).emit(second))

    def test_loop_zero_one_and_two_iterations_match_independent_tensor_operations(self):
        first = extend_passed(self.parent, 'tilelang', 'scan_sum', self.config)
        scan = first.body.operations[-1]
        # Expose the pre-scan value to compute a separate expected value.
        first.observations = [scan.operands[0]]
        wrapped = loop_target(first, 'tilelang')
        self.assertEqual(wrapped.body.returns, first.body.returns)
        memory = extended_inputs(first.to_dict(), seed=4)
        original, _ = extended_reference(first.to_dict(), memory, 0, 1)
        current = original[scan.operands[0]]
        for steps in (0, 1, 2, 4):
            expected = current
            for _ in range(min(steps, 2)):
                expected = expected.cumsum(dim=-1)
            actual, _ = extended_reference(wrapped.to_dict(), memory, steps, 1)
            torch.testing.assert_close(actual[scan.results[0].name], expected)
        self.assertNotEqual(program_features(first), program_features(wrapped))

    def test_evolution_retains_lineage_and_revalidates_restored_target(self):
        stage = DSLStage(self.config, 'tilelang', feedback=StructuralFeedback())
        stage.add(self.parent, 'parent.json')
        class PassingOracle:
            def test(self, program):
                return None
        first, lineage = stage.generate(PassingOracle())
        stage.retain_target(first, 'first.json', lineage, [])
        second, next_lineage = stage.generate(PassingOracle())
        self.assertEqual(next_lineage['extension_depth'], 2)
        self.assertEqual(next_lineage['source_sha256'], stage.digest(first))
        self.assertEqual(next_lineage['root_source_sha256'], stage.digest(self.parent))
        self.assertNotEqual(second.to_dict(), first.to_dict())
        self.assertTrue(all(is_common_seed(p) for p, *_ in stage.sources))
        restored = DSLStage(self.config, 'tilelang')
        restored.restore(json.loads(json.dumps(stage.snapshot())))
        self.assertEqual(len(restored.targets), 1)
        self.assertFalse(restored.targets[0][2])
        class StaleOracle:
            def test(self, program):
                return object()
        self.assertIsNone(restored.generate(StaleOracle()))
        self.assertEqual(restored.baseline_rejected, 2)
        self.assertFalse(restored.targets)

    def test_depth_limit_and_old_snapshot_fallback(self):
        config = copy.copy(self.config)
        config.dsl_max_depth = 1
        stage = DSLStage(config, 'tilelang')
        stage.add(self.parent, 'parent.json')
        class Oracle:
            def test(self, program):
                return None
        first, lineage = stage.generate(Oracle())
        stage.retain_target(first, 'first.json', lineage, [])
        _, next_lineage = stage.generate(Oracle())
        self.assertEqual(next_lineage['extension_depth'], 1)
        old = stage.snapshot()
        for name in ('targets', 'target_structural', 'evolution_counts', 'version'):
            old.pop(name, None)
        restored = DSLStage(config, 'tilelang')
        restored.restore(old)
        self.assertFalse(restored.targets)
        self.assertEqual(len(restored.sources), 1)

    def test_target_failures_do_not_enter_corpus_or_common_feedback(self):
        fuzzer = TileSmith(self.config)
        calls = 0
        def oracle(program):
            nonlocal calls
            calls += 1
            fuzzer.oracle.compilation_complete = True
            if calls == 3:
                return BugReport(BugType.WRONG_RESULT, 'WRONG RESULT: candidate',
                                 root_cause='wrong_result', params=program.params_dict)
            return None
        with patch.object(fuzzer.generator, 'generate', return_value=self.parent), \
                patch.object(fuzzer.oracle, 'test', side_effect=oracle):
            fuzzer.run(3, verbose=False)
        self.assertEqual(len(fuzzer.dsl_stage.targets), 1)
        self.assertEqual(set(fuzzer.feedback.passed), program_features(self.parent))
        self.assertTrue(all(is_common_seed(p) for p in fuzzer.seed_pool))
        progress = json.loads((fuzzer.output_dir / 'coverage_progress.json').read_text())
        self.assertEqual(progress['total_tested'], 3)
        failed = list((fuzzer.output_dir / 'failed/wrong_result').glob('*.json'))
        self.assertEqual(len(failed), 1)
        self.assertEqual(json.loads(failed[0].read_text())['extension_depth'], 2)

    def test_corpus_keeps_unique_features_and_can_reject_redundant_candidate(self):
        sets = [{'common', 'rare'}, {'common'}, {'common', 'new'}]
        self.assertEqual(corpus_eviction(sets), 1)
        sets = [{'common', 'rare'}, {'common', 'new'}, {'common'}]
        self.assertEqual(corpus_eviction(sets), 2)

    def test_wrong_result_buckets_do_not_demote_common_features(self):
        config = copy.copy(self.config)
        config.dsl_extend_prob = 0
        config.explained_feedback = True
        fuzzer = TileSmith(config)
        other = copy.deepcopy(self.parent)
        other.input_pattern = 'boundary' if other.input_pattern != 'boundary' else 'integer'
        bugs = [BugReport(BugType.WRONG_RESULT, 'WRONG RESULT: shared template',
                          root_cause='wrong_result', params=p.params_dict) for p in (self.parent, other)]
        with patch.object(fuzzer.generator, 'generate', side_effect=[self.parent, other]), \
                patch.object(fuzzer.oracle, 'test', side_effect=bugs):
            fuzzer.run(2, verbose=False)
        self.assertEqual(fuzzer.feedback.explained, Counter())
        self.assertEqual(sum(fuzzer.failure_buckets.values()), 2)


if __name__ == '__main__':
    unittest.main()
