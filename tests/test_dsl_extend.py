"""The common generator and the post-pass DSL extension are distinct stages."""
import ast
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

from src.backends import get_backend
from src.config import Config
from src.workflow.generator.dsl_extend import eligible_ops, extend_passed, is_common_seed
from src.workflow.generator.extended import ExtendedGenerator
from src.workflow.generator.grids import GridState
from src.workflow.emitter.extended_runtime import extended_inputs, extended_reference
from src.workflow.oracle import BugReport, BugType
from extend import main, passing_sources
from api_coverage import audit, campaign_passes, passing_code_calls


class DslExtensionTests(unittest.TestCase):
    def setUp(self):
        state = random.getstate()
        self.addCleanup(random.setstate, state)
        random.seed(12)
        self.config = Config(extended_prob=1, extended_atomic_prob=0,
                             extended_fma_prob=0, extended_shape_op_prob=1,
                             extended_int8_prob=0, extended_elementwise_prob=0)

    def test_common_seed_excludes_target_only_operations(self):
        for backend in ('triton', 'tilelang'):
            for family in ('arithmetic', 'control_calls', 'mixed'):
                with self.subTest(backend=backend, family=family):
                    program = ExtendedGenerator(self.config, backend).generate(family)
                    self.assertTrue(is_common_seed(program))
                    self.assertFalse({'join', 'split', 'interleave'} &
                                     {node.op for node in program.all_operations()})
                    self.assertTrue(all(not node.attrs.get('pipelined') for node in program.all_operations()
                                        if node.op == 'for'))

    def test_triton_shape_extension_preserves_parent(self):
        parent = ExtendedGenerator(self.config, 'triton').generate('arithmetic')
        original = parent.to_dict()
        for op in ('join', 'split', 'interleave'):
            with self.subTest(op=op):
                child = extend_passed(parent, 'triton', op, self.config, GridState())
                self.assertEqual(parent.to_dict(), original)
                self.assertIn(op, {node.op for node in child.all_operations()})
                self.assertEqual(child.family, 'extend_triton_' + op)
                ast.parse(get_backend('triton').make_emitter(self.config).emit(child))

    def test_tilelang_pipelined_extension_and_input_filter(self):
        parent = ExtendedGenerator(self.config, 'tilelang').generate('control_calls')
        self.assertIn('pipelined_for', eligible_ops(parent, 'tilelang'))
        child = extend_passed(parent, 'tilelang', 'pipelined_for', self.config)
        self.assertTrue(any(node.op == 'for' and node.attrs['pipelined']
                            for node in child.all_operations()))
        self.assertTrue(is_common_seed(parent))
        self.assertFalse(is_common_seed(child))
        ast.parse(get_backend('tilelang').make_emitter(self.config).emit(child))

    def test_numeric_target_operations_have_checked_outputs(self):
        for backend, ops in (('triton', ('scan_sum', 'scan_product', 'sort', 'histogram',
                                       'argmax', 'argmin', 'xor_sum', 'dsl_sigmoid',
                                       'dsl_clamp', 'softmax')),
                             ('tilelang', ('scan_sum', 'scan_max', 'reduce_abssum', 'reduce_absmax',
                                           'reduce_bitand', 'reduce_bitor', 'reduce_bitxor',
                                           'dsl_sigmoid', 'dsl_clamp'))):
            parent = ExtendedGenerator(self.config, backend).generate('arithmetic')
            for op in ops:
                with self.subTest(backend=backend, op=op):
                    child = extend_passed(parent, backend, op, self.config)
                    self.assertEqual(parent.body.returns[0], child.body.returns[0])
                    self.assertEqual(len(child.body.returns), len(parent.body.returns) + 1)
                    self.assertEqual(child.body.operations[-1].op, op)
                    self.assertFalse(is_common_seed(child))
                    ast.parse(get_backend(backend).make_emitter(self.config).emit(child))

    def test_histogram_binning_uses_integer_dataflow(self):
        parent = ExtendedGenerator(self.config, 'triton').generate('indexed_memory')
        child = extend_passed(parent, 'triton', 'histogram', self.config)
        added = child.body.operations[len(parent.body.operations):]
        histogram = added[-1]
        self.assertEqual(histogram.op, 'histogram')
        producer = next(n for n in added if any(v.name == histogram.operands[0] for v in n.results))
        self.assertEqual(producer.op, 'bitand')
        self.assertFalse(any(n.op == 'cast' and n.results[0].type.dtype == 'int32' for n in added))

    def test_newer_triton_primitives_are_version_gated(self):
        parent = ExtendedGenerator(self.config, 'triton').generate('arithmetic')
        with patch('src.workflow.generator.dsl_extend._available', return_value=False):
            self.assertFalse({'topk', 'gather'} & set(eligible_ops(parent, 'triton')))
        with patch('src.workflow.generator.dsl_extend._available', return_value=True):
            for op in ('topk', 'gather'):
                with self.subTest(op=op):
                    child = extend_passed(parent, 'triton', op, self.config)
                    self.assertEqual(child.body.operations[-1].op, op)
                    ast.parse(get_backend('triton').make_emitter(self.config).emit(child))

    def test_triton_bitwise_atomic_extension_checks_scratch(self):
        parent = ExtendedGenerator(self.config, 'triton').generate('arithmetic')
        for op in ('atomic_and', 'atomic_or', 'atomic_xor'):
            with self.subTest(op=op):
                child = extend_passed(parent, 'triton', op, self.config)
                self.assertEqual(child.body.operations[-1].op, op)
                self.assertEqual(child.buffers[-1].role, 'scratch')
                self.assertEqual(parent.body.returns, child.body.returns)
                memory = extended_inputs(child.to_dict())
                _, final = extended_reference(child.to_dict(), memory, 5, 15)
                self.assertFalse(final[child.buffers[-1].name].equal(memory[child.buffers[-1].name]))
                ast.parse(get_backend('triton').make_emitter(self.config).emit(child))

    def test_api_audit_requires_execution_evidence(self):
        names = {'histogram': {'module': 'triton.language.standard', 'kind': 'function'},
                 'load': {'module': 'triton.language.core', 'kind': 'function'}}
        result = audit('triton', names, {'histogram': 0})
        rows = {row['api']: row for row in result['entries']}
        self.assertEqual(rows['histogram']['status'], 'implemented_no_run')
        self.assertEqual(rows['load']['status'], 'unmapped')
        result = audit('triton', names, {'histogram': 2})
        self.assertEqual(result['entries'][0]['status'], 'executed_pass')
        self.assertEqual(result['entries'][0]['passing_cases'], 2)
        result = audit('triton', names, {}, {'load': 1})
        self.assertEqual(result['entries'][1]['status'], 'seen_in_passing_code')
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'summary.json'
            path.write_text(json.dumps({'backend': 'triton', 'by_op': {'histogram:passed': 2}}))
            self.assertEqual(campaign_passes([path], 'triton')['histogram'], 2)
            passed = Path(temp) / 'passed'
            passed.mkdir()
            (passed / 'case.py').write_text('x = tl.load(ptr)\ny = tl.store(ptr, x)\n')
            self.assertEqual(passing_code_calls([passed], 'triton')['load'], 1)

    def test_passed_records_only_and_stage_provenance(self):
        parent = ExtendedGenerator(self.config, 'triton').generate('arithmetic')
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            passed = root / 'passed'
            passed.mkdir()
            good = dict(parent.to_dict(), validation_mode='execute')
            (passed / 'good.json').write_text(json.dumps(good))
            (passed / 'compile_only.json').write_text(json.dumps(dict(good, validation_mode='compile_only')))
            self.assertEqual(len(passing_sources(passed, 'triton')), len(eligible_ops(parent, 'triton')))
            self.assertEqual(len(passing_sources(passed, 'triton', max_sources=2)), 2)
            self.assertEqual([entry[3] for entry in passing_sources(passed, 'triton', op_filter='join')], ['join'])

            class StubOracle:
                def __init__(self, config, backend):
                    self.config = config
                def test(self, program):
                    return None
                def _emit_code(self, program):
                    return 'pass\n'

            with patch('extend.Oracle', StubOracle):
                self.assertEqual(main(['--backend', 'triton', '--passed-dir', str(passed),
                                       '--op', 'join', '-n', '1', '--output', str(root / 'extended')]), 0)
            records = list((root / 'extended' / 'passed').glob('*.json'))
            self.assertEqual(len(records), 1)
            record = json.loads(records[0].read_text())
            self.assertEqual(record['extension_op'], 'join')
            self.assertTrue(record['baseline_revalidated'])
            self.assertEqual(record['source_file'], str((passed / 'good.json').resolve()))

            class FailingBaseline(StubOracle):
                def test(self, program):
                    return BugReport(BugType.WRONG_RESULT, 'baseline mismatch',
                                     root_cause='wrong_result')

            with patch('extend.Oracle', FailingBaseline):
                with self.assertRaisesRegex(RuntimeError, 'Could not reach requested count'):
                    main(['--backend', 'triton', '--passed-dir', str(passed),
                          '--op', 'join', '-n', '1', '--output', str(root / 'rejected')])
            rejected = json.loads((root / 'rejected' / 'summary.json').read_text())
            self.assertEqual(rejected['tested'], 0)
            self.assertGreater(rejected['baseline_rejected'], 0)
            self.assertFalse((root / 'rejected' / 'failed').exists())
