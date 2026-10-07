"""The integrated campaign alternates common generation and DSL derivatives."""
import hashlib
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

from src.config import Config
from src.workflow.fuzzer.dsl_stage import DSLStage
from src.workflow.fuzzer.fuzzer import TileSmith
from src.workflow.feedback import StructuralFeedback, key
from src.workflow.generator.dsl_extend import eligible_ops, is_common_seed
from src.workflow.generator.extended import ExtendedGenerator
from src.workflow.generator.grids import GridState
from api_coverage import campaign_passes


class IntegratedDSLStageTests(unittest.TestCase):
    def setUp(self):
        state = random.getstate()
        self.addCleanup(random.setstate, state)
        random.seed(53)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.config = Config(backends=['tilelang'], output_dir=self.temp.name,
                             seed=53, extended_prob=1, dsl_extend_prob=1,
                             structural_feedback=False, seed_add_prob=0,
                             extended_atomic_prob=0, extended_fma_prob=0,
                             extended_shape_op_prob=0, extended_int8_prob=0)

    def test_campaign_derives_only_after_common_parent_passes(self):
        parent = ExtendedGenerator(self.config, 'tilelang').generate('arithmetic')
        fuzzer = TileSmith(self.config)
        with patch.object(fuzzer.generator, 'generate', return_value=parent), \
                patch.object(fuzzer.oracle, 'test', return_value=None):
            fuzzer.run(2, verbose=False)
        records = [json.loads(p.read_text()) for p in (fuzzer.output_dir/'passed').glob('*.json')]
        self.assertEqual(len(records), 2)
        common = next(record for record in records if 'extension_op' not in record)
        derived = next(record for record in records if 'extension_op' in record)
        self.assertTrue(is_common_seed(fuzzer._dict_to_program(common)))
        self.assertEqual(derived['source_sha256'],
                         hashlib.sha256(json.dumps(parent.to_dict(), sort_keys=True).encode()).hexdigest())
        self.assertTrue(Path(derived['source_file']).exists())
        self.assertEqual(len(fuzzer.dsl_stage.sources), 1)
        self.assertEqual(sum(v for k,v in fuzzer.dsl_stage.counts.items() if k.endswith(':passed')), 1)
        summary = json.loads((fuzzer.output_dir/'summary.json').read_text())
        self.assertEqual(summary['dsl_extension']['source_pool'], 1)
        self.assertEqual(summary['generation_config']['dsl_extend_prob'], 1)
        self.assertEqual(campaign_passes([fuzzer.output_dir/'summary.json'], 'tilelang')[
                         derived['extension_op']], 1)
        restored = TileSmith(self.config, resume_dir=str(fuzzer.output_dir))
        self.assertEqual(len(restored.dsl_stage.sources), 1)
        self.assertFalse(restored.dsl_stage.sources[0][2])

    def test_op_selection_covers_each_eligible_operation_before_repeat(self):
        parent = ExtendedGenerator(self.config, 'tilelang').generate('arithmetic')
        stage = DSLStage(self.config, 'tilelang')
        stage.add(parent, 'parent.json')
        class PassingOracle:
            def test(self, program):
                return None
        seen = []
        for _ in eligible_ops(parent, 'tilelang'):
            child, origin = stage.generate(PassingOracle())
            self.assertEqual(child.family, 'extend_tilelang_' + origin['extension_op'])
            seen.append(origin['extension_op'])
            stage.record(origin['extension_op'], True)
        self.assertEqual(set(seen), set(eligible_ops(parent, 'tilelang')))
        self.assertIsNone(stage.generate(PassingOracle()))

    def test_dsl_compiler_feedback_is_separate_from_common_generation(self):
        parent = ExtendedGenerator(self.config, 'tilelang').generate('arithmetic')
        feedback = StructuralFeedback()
        stage = DSLStage(self.config, 'tilelang', feedback=feedback)
        stage.add(parent, 'parent.json')
        digest = stage.sources[0][3]
        feature = key('compiler_pair', 'lowered_tir', 'T.copy', 'T.gemm')
        stage.observe_compilation(digest, [{'features': [feature]}], passed=True)
        self.assertEqual(stage.compiler[feature], 1)
        self.assertFalse(feedback.compiler)
        initial_weight = stage.source_weight(parent, digest)
        self.assertGreater(initial_weight, feedback.seed_weight(parent))
        for _ in range(3):
            feedback.observe_known_failure(digest, 'tilelang_bool_cuda_type')
        self.assertLess(stage.source_weight(parent, digest), initial_weight)
        restored = DSLStage(self.config, 'tilelang', feedback=feedback)
        restored.restore(stage.snapshot())
        self.assertEqual(restored.source_compiler[digest], [feature])

    def test_bounded_variants_continue_after_failure_and_restore_attempts(self):
        self.config.dsl_source_variants = 4
        self.config.dsl_attributes = True
        parent = ExtendedGenerator(self.config, 'tilelang').generate('arithmetic')
        grids = GridState()
        stage = DSLStage(self.config, 'tilelang', grids=grids)
        stage.add(parent, 'parent.json')
        class Oracle:
            def test(self, program):
                return None
        with patch('src.workflow.fuzzer.dsl_stage.eligible_ops', return_value=('scan_sum',)):
            first, lineage = stage.generate(Oracle())
            stage.observe_outcome(first, lineage, [], passed=False, seconds=1, known_repeat=True)
            restored_grids = GridState()
            restored_grids.load(grids.save())
            restored = DSLStage(self.config, 'tilelang', grids=restored_grids)
            restored.restore(json.loads(json.dumps(stage.snapshot())))
            children = [first]
            for trial in (2, 3, 4):
                child, lineage = restored.generate(Oracle())
                self.assertEqual(lineage['extension_variant'], trial)
                children.append(child)
            self.assertEqual(len({restored.digest(child) for child in children}), 4)
            self.assertIsNone(restored.generate(Oracle()))
            # Old snapshots recorded one attempt for every tried pair.
            old = stage.snapshot()
            old.pop('source_attempts')
            legacy = DSLStage(self.config, 'tilelang')
            legacy.restore(old)
            self.assertEqual(legacy.generate(Oracle())[1]['extension_variant'], 2)

    def test_duplicate_derivatives_are_rejected_with_a_bounded_retry(self):
        self.config.dsl_source_variants = 4
        self.config.dsl_attributes = True
        parent = ExtendedGenerator(self.config, 'tilelang').generate('arithmetic')
        stage = DSLStage(self.config, 'tilelang', grids=GridState())
        stage.add(parent, 'parent.json')
        with patch('src.workflow.fuzzer.dsl_stage.eligible_ops', return_value=('scan_sum',)):
            self.assertIsNone(stage.generate(None, accept=lambda p: False))
            self.assertEqual(stage.duplicate_derivatives, 2)
            self.assertIsNone(stage.generate(None, accept=lambda p: False))
            self.assertEqual(stage.duplicate_derivatives, 4)
            self.assertIsNone(stage.generate(None, accept=lambda p: False))
            self.assertEqual(stage.schedule.total.tested, 0)

    def test_restored_source_is_revalidated_and_rejected_if_stale(self):
        parent = ExtendedGenerator(self.config, 'tilelang').generate('arithmetic')
        stage = DSLStage(self.config, 'tilelang')
        stage.add(parent, 'parent.json')
        restored = DSLStage(self.config, 'tilelang')
        restored.restore(stage.snapshot())
        class FailingOracle:
            def test(self, program):
                return object()
        self.assertIsNone(restored.generate(FailingOracle()))
        self.assertEqual(restored.baseline_rejected, 1)
        self.assertFalse(restored.sources)

    def test_interrupted_derivative_keeps_lineage_on_resume(self):
        parent = ExtendedGenerator(self.config, 'tilelang').generate('arithmetic')
        fuzzer = TileSmith(self.config)
        calls = 0
        def stop_on_derivative(program):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise KeyboardInterrupt
            return None
        with patch.object(fuzzer.generator, 'generate', return_value=parent), \
                patch.object(fuzzer.oracle, 'test', side_effect=stop_on_derivative):
            with self.assertRaises(KeyboardInterrupt):
                fuzzer.run(2, verbose=False)
        restored = TileSmith(self.config, resume_dir=str(fuzzer.output_dir))
        self.assertIsNotNone(restored._resume_pending_extension)
        with patch.object(restored.oracle, 'test', return_value=None):
            restored.run(1, verbose=False)
        self.assertEqual(len(restored.dsl_stage.sources), 1)
        self.assertEqual(sum(v for k, v in restored.dsl_stage.counts.items()
                             if k.endswith(':passed')), 1)


if __name__ == '__main__':
    unittest.main()
