"""SSA dependency feedback, persistence and successful campaign novelty."""
import ast
import contextlib
import copy
from dataclasses import asdict
import io
import json
import random
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import torch
from src.config import Config
from src.ir import TileKernel, ComputeKind, DataType
from src.ir.region import RegionProgram, Region, Operation as Op
from src.workflow.feedback import (StructuralFeedback, confirmed_failure, key,
                                   program_digest, program_features)
from src.workflow.fuzzer.fuzzer import TileSmith
from src.workflow.emitter.region_runtime import _region_reference
from src.workflow.oracle import Oracle
from src.workflow.oracle.oracle import BugReport, BugType


def dataflow_program():
    spec = TileKernel('kernel_0', compute_kind=ComputeKind.COPY, M=2, N=4, K=4,
                      block_M=16, block_N=16, block_K=16, dtype=DataType.FLOAT32)
    return RegionProgram(spec, Region([], [Op('load', 'entry'),
        Op('copy', 'saved', ['entry']), Op('scale', 'scaled', ['entry'], {'alpha': 2.0}),
        Op('add', 'out', ['scaled', 'saved'])], 'out'))


class FeedbackTests(unittest.TestCase):
    def test_campaign_attributes_known_repeats_to_mutation_parent(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            config = Config(output_dir=directory, seed=23, mutate_prob=1,
                            seed_add_prob=1, backends=['tilelang'])
            fuzzer = TileSmith(config)
            parent = dataflow_program()
            children = []
            for size in (3, 4):
                child = copy.deepcopy(parent)
                child.spec.M = size
                children.append(child)
            bugs = [BugReport(BugType.COMPILE_CRASH,
                              'Cannot convert type boolx8 to CUDA type',
                              params=child.params_dict,
                              root_cause='tilelang_codegen_error') for child in children]
            with patch.object(fuzzer.generator, 'generate', return_value=parent), \
                    patch.object(fuzzer.mutator, 'mutate', side_effect=children), \
                    patch.object(fuzzer.oracle, 'test', side_effect=[None, *bugs]):
                fuzzer.run(3, verbose=False)
            digest = program_digest(parent)
            self.assertEqual(fuzzer.feedback.known_seed_failures[digest], 2)
            self.assertEqual(fuzzer.feedback.known_signatures['tilelang_bool_cuda_type'], 2)
            saved = list((fuzzer.output_dir / 'failed' / 'tilelang_codegen_error').glob('*.json'))
            self.assertEqual(len(saved), 2)
            self.assertTrue(all(json.loads(path.read_text())['generation_origin']['seed_digest'] == digest
                                for path in saved))
            restored = TileSmith(config, resume_dir=str(fuzzer.output_dir))
            self.assertEqual(restored.feedback.known_seed_failures[digest], 2)

    def test_compiler_features_and_confirmed_failures_change_seed_weight(self):
        program = dataflow_program()
        feedback = StructuralFeedback()
        digest = program_digest(program)
        features = [key('compiler_stage', 'ttgir'),
                    key('compiler_pair', 'ttgir', 'tt.load', 'tt.add')]
        records = [{'features': features}]
        feedback.observe(program, True)
        feedback.observe_compilation(program, records, complete=True)
        structural_only = feedback.seed_weight(program)
        feedback.register_seed(program, records)
        self.assertGreater(feedback.seed_weight(program), structural_only)
        known = 'tilelang_bool_cuda_type'
        feedback.observe_known_failure(digest, known)
        first_failure_weight = feedback.seed_weight(program)
        self.assertGreater(first_failure_weight, structural_only)
        feedback.observe_known_failure(digest, known)
        self.assertLess(feedback.seed_weight(program), first_failure_weight)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'feedback.json'
            feedback.save(path)
            restored = StructuralFeedback()
            restored.restore(path)
            self.assertEqual(restored.seed_weight(program), feedback.seed_weight(program))
            self.assertEqual(restored.known_signatures[known], 2)
            restored.retain_seeds(set())
            self.assertNotIn(digest, restored.seed_compiler)

    def test_failure_matching_does_not_use_coarse_category(self):
        def bug(message):
            return BugReport(BugType.COMPILE_CRASH, message, root_cause='triton_compile_error')
        self.assertIsNone(confirmed_failure(bug('different internal assertion'), 'triton'))
        self.assertEqual(confirmed_failure(bug(
            'triton.compiler.errors.CompilationError: at 2:7:\n'
            '    x = (tl.flip(value)).to(tl.float32)\n           ^\n'
            'Process exited with return code 1'), 'triton'), 'triton_flip_default_axis')

    def test_compilation_records_stage_and_ir_neighbours(self):
        from src.workflow.emitter.extended_runtime import record_extended_compilation
        import json
        import os
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, TILESMITH_ARTIFACT_DIR=directory):
            record_extended_compilation('variant', {'ttgir': 'tt.load tt.add tt.store'},
                                        {'enable_fp_fusion': True})
            record = json.loads((Path(directory) / 'compilation.json').read_text())[0]
            self.assertIn(key('compiler_stage', 'ttgir'), record['features'])
            self.assertIn(key('compiler_pair', 'ttgir', 'tt.load', 'tt.add'), record['features'])
            self.assertIn(key('compiler_setting_op', 'ttgir', 'enable_fp_fusion', True, 'tt.load'),
                          record['features'])

    def test_previous_feedback_version_remains_resumable(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'feedback.json'
            path.write_text(json.dumps({'version': 1, 'attempted': {}, 'passed': {},
                                        'compiled': {}, 'compiler': {key('compiler', 'ttir', 'tt.load'): 2}}))
            feedback = StructuralFeedback()
            feedback.restore(path)
            self.assertEqual(feedback.compiler[key('compiler', 'ttir', 'tt.load')], 2)
            self.assertFalse(feedback.seed_compiler)

    def test_data_edges_follow_operands_not_text_adjacency(self):
        features = program_features(dataflow_program())
        self.assertIn(key('data', 'scale', 'add'), features)
        self.assertIn(key('data', 'copy', 'add'), features)
        self.assertIn(key('data', 'load', 'scale'), features)
        self.assertNotIn(key('data', 'copy', 'scale'), features)

    def test_snapshot_preserves_old_value_and_round_trips(self):
        p = dataflow_program()
        p.validate()
        a = torch.arange(8, dtype=torch.float32).reshape(2, 4)
        result = _region_reference(a, a, asdict(p.body), 16, 16, 'float32')
        torch.testing.assert_close(result, a * 3)
        fuzzer = TileSmith.__new__(TileSmith)
        restored = fuzzer._dict_to_program(fuzzer._program_to_dict(p))
        self.assertEqual(restored.to_dict(), p.to_dict())
        for backend in ('tilelang', 'triton'):
            ast.parse(Oracle(Config(), backend)._emit_code(restored))

    def test_failed_program_does_not_reduce_success_novelty(self):
        feedback = StructuralFeedback()
        p = dataflow_program()
        before = feedback.seed_weight(p)
        self.assertEqual(feedback.observe(p, False), 0)
        self.assertEqual(feedback.seed_weight(p), before)
        self.assertGreater(feedback.observe(p, True), 0)
        self.assertLess(feedback.seed_weight(p), before)
        self.assertEqual(feedback.observe(p, True), 0)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'feedback.json'
            feedback.save(path)
            restored = StructuralFeedback()
            restored.restore(path)
            self.assertEqual(restored.passed, feedback.passed)
            self.assertEqual(restored.seed_weight(p), feedback.seed_weight(p))


    def test_weight_with_zero_boost_matches_legacy_formula(self):
        feedback = StructuralFeedback()
        feature = key('op', 'gemm')
        feedback.passed[feature] = 3
        feedback.attempted[feature] = 7
        # Legacy formula was 1 + passed_decay / (1 + passed[f]) — boost=0 must
        # reproduce it exactly, regardless of the attempted count.
        self.assertEqual(feedback.weight(feature, passed_decay=2.0, uncovered_boost=0.0),
                         1 + 2 / (1 + 3))
        self.assertEqual(feedback.weight(feature, passed_decay=4.0, uncovered_boost=0.0),
                         1 + 4 / (1 + 3))

    def test_uncovered_boost_applies_only_to_never_attempted_features(self):
        feedback = StructuralFeedback()
        tried = key('op', 'tried')
        fresh = key('op', 'fresh')
        feedback.attempted[tried] = 1          # tried but never passed
        feedback.passed[tried] = 0
        # Attempted-but-never-passed features are demoted below base (0.6):
        # they only produced rejections, so they must not out-compete
        # untried or proven combinations (MLIRSmith wasted-effort rule).
        self.assertEqual(feedback.weight(tried, passed_decay=2.0, uncovered_boost=50.0),
                         1.0 * 0.6)
        self.assertEqual(feedback.weight(fresh, passed_decay=2.0, uncovered_boost=50.0),
                         1 + 2 / 1 + 50.0)
        # The boost is one-shot: observing the feature (even as a failure)
        # removes it, and the failed feature stays demoted below base.
        feedback.observe(dataflow_program(), False)
        for op_kind in ('load', 'copy', 'scale', 'add'):
            self.assertEqual(feedback.weight(key('op', op_kind), passed_decay=2.0, uncovered_boost=50.0),
                             1.0 * 0.6)


    def test_generated_dataflow_is_interpretable(self):
        from dataclasses import asdict
        from src.workflow.generator.region_generator import RegionGenerator
        from src.workflow.emitter.region_runtime import _region_reference
        random.seed(37)
        gen = RegionGenerator(Config(coverage_probe_prob=0))
        gen.feedback = StructuralFeedback()
        for _ in range(30):
            p = gen.generate()
            p.spec.M, p.spec.N, p.spec.K = 3, 5, 4
            gemm = p.body.operations[0].kind == 'gemm'
            a = torch.ones(3, 4 if gemm else 5) * .1
            b = torch.ones(4, 5) * .1
            from src.workflow.emitter.typed_region_runtime import _typed_region_reference
            reference = _typed_region_reference if p.typed else _region_reference
            output = reference(a, b, asdict(p.body), 32, 32, p.spec.dtype.value,
                                       [asdict(fn) for fn in p.functions])
            self.assertEqual(tuple(output.shape), (3,5))
            gen.feedback.observe(p, True)


    def test_campaign_saves_feedback_and_restores_it(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            config = Config(output_dir=directory, seed=19)
            fuzzer = TileSmith(config)
            with patch.object(fuzzer, '_generate_test_case', return_value=dataflow_program()), patch.object(fuzzer.oracle, 'test', return_value=None):
                fuzzer.run(1, verbose=False)
            self.assertTrue(fuzzer.feedback.passed)
            self.assertEqual(len(fuzzer.seed_pool), 1)
            restored = TileSmith(config, resume_dir=str(fuzzer.output_dir))
            self.assertEqual(restored.feedback.passed, fuzzer.feedback.passed)
            self.assertIs(restored.generator.region_gen.feedback, restored.feedback)
            self.assertIs(restored.mutator.feedback, restored.feedback)



if __name__ == '__main__':
    unittest.main()
