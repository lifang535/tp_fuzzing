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
from src.workflow.generator.dsl_extend import eligible_ops, is_common_seed
from src.workflow.generator.extended import ExtendedGenerator
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
