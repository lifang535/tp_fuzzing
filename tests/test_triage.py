"""Failure triage: diagnostic buckets, the known-bug quarantine, duplicate
damping in structural feedback, swarm generation and the named flip axis.

The campaign tests drive the real loop with a stubbed oracle verdict, so
bucketing, logging, persistence and resume run unchanged on the CPU.
"""
from collections import Counter
import contextlib
import hashlib
import io
import json
import random
import re
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from src.config import Config
from src.workflow import triage
from src.workflow.extended_feedback import extended_features
from src.workflow.feedback import StructuralFeedback, confirmed_failure, key, program_features
from src.workflow.fuzzer.fuzzer import TileSmith
from src.workflow.generator.dsl_extend import extend_passed
from src.workflow.generator.extended import Builder, ExtendedGenerator, flip_axes, mutate_extended
from src.workflow.generator.region_generator import RegionGenerator
from src.workflow.oracle.oracle import BugReport, BugType
from src.workflow.triage import Quarantine, failure_bucket, failure_key, species
from test_extended_ops import emit, reference, validated
from test_feedback import dataflow_program


def tvm_check(index):
    # Paths, SSA names and values differ between reproducers of one check.
    return ('Traceback (most recent call last):\n'
            f'  File "/tmp/run{index}/tilelang/engine/lower.py", line 88, in lower\n'
            '    mod = transform(mod)\n'
            f'  File "/src/runtime/logging.cc", line {index + 300}, in tvm::runtime::detail::LogFatal::~LogFatal()\n'
            f'RuntimeError: Check failed: (e{index} == e{index + 1}) is false: {8 * index} vs {index + 64}\n')


def bool_codegen(index):
    return f'tvm.error.InternalError: Cannot convert type boolx{4 << index % 3} to CUDA type'


FLIP_DEFAULT = ('Traceback (most recent call last):\n'
                '  File "/root/tp/kernel.py", line 40, in compile\n'
                '    kernel[grid](x)\n'
                'triton.compiler.errors.CompilationError: at 12:8:\n'
                '    y = tl.flip(x)\n'
                '        ^\n'
                "TypeError(\"'<=' not supported between instances of 'int' and 'NoneType'\")\n")


class FailureKeyTests(unittest.TestCase):
    def test_tvm_check_keeps_its_condition_and_innermost_frame(self):
        expected = 'RuntimeError: Check failed: (e# == e#) is false | in lower'
        self.assertEqual(failure_key(tvm_check(12)), expected)
        # The values after the condition, the temporary path and the logging
        # frame that throws are not part of the mechanism.
        self.assertEqual(failure_key(tvm_check(3)), expected)

    def test_dtype_names_survive_normalization(self):
        self.assertNotEqual(failure_key('TypeError: unsupported dtype float16'),
                            failure_key('TypeError: unsupported dtype float32'))
        self.assertEqual(failure_key(bool_codegen(0)),
                         'tvm.error.InternalError: Cannot convert type boolx# to CUDA type')
        self.assertEqual(failure_key(bool_codegen(1)), failure_key(bool_codegen(2)))

    def test_paths_and_numbers_are_masked(self):
        self.assertEqual(failure_key("FileNotFoundError: [Errno 2] No such file or directory: '/tmp/tmpab12/k.cubin'"),
                         "FileNotFoundError: [Errno #] No such file or directory: '<path>'")

    def test_triton_compile_error_keeps_the_call_under_the_caret(self):
        self.assertEqual(failure_key(FLIP_DEFAULT),
                         'triton.compiler.errors.CompilationError: at #:#: | in compile | at tl.flip | '
                         "TypeError(\"'<=' not supported between instances of 'int' and 'NoneType'\")")
        named = FLIP_DEFAULT.replace('tl.flip(x)', 'tl.flip(x, 0)')
        self.assertEqual(confirmed_failure(FLIP_DEFAULT, 'triton'), 'triton_flip_default_axis')
        self.assertIsNone(confirmed_failure(named, 'triton'))
        moved = FLIP_DEFAULT.replace('    y = tl.flip(x)\n        ^', '    y = 2 * tl.where(m, x, 0)\n            ^')
        self.assertNotEqual(failure_key(moved), failure_key(FLIP_DEFAULT))

    def test_chained_traceback_keeps_its_root(self):
        message = ('Traceback (most recent call last):\n'
                   '  File "/src/a.py", line 3, in parse\n'
                   'ValueError: inner detail 42\n'
                   '\n'
                   'The above exception was the direct cause of the following exception:\n'
                   '\n'
                   'Traceback (most recent call last):\n'
                   '  File "/src/b.py", line 9, in build\n'
                   'RuntimeError: build failed\n')
        self.assertEqual(failure_key(message),
                         'RuntimeError: build failed | in build | from ValueError: inner detail #')

    def test_fatal_signal_is_appended_once(self):
        crash = 'RuntimeError: CUDA error: an illegal memory access was encountered\n'
        self.assertEqual(failure_key(crash + 'Process terminated by signal 6 (SIGABRT)'),
                         'RuntimeError: CUDA error: an illegal memory access was encountered | signal SIGABRT')
        self.assertEqual(failure_key('RuntimeError: worker died with SIGSEGV\n'
                                     'Process terminated by signal 11 (SIGSEGV)'),
                         'RuntimeError: worker died with SIGSEGV')

    def test_location_only_names_failures_without_an_exception(self):
        self.assertEqual(failure_key('Execution timed out', 'execute'), 'Execution timed out | at execute')
        self.assertEqual(failure_key('Execution timed out', 'compile:triton_8_prec'),
                         'Execution timed out | at compile:triton_#_prec')
        self.assertEqual(failure_key('ValueError: bad shape', 'compile:triton_8_prec'), 'ValueError: bad shape')


class FailureBucketTests(unittest.TestCase):
    def test_audited_signature_is_its_own_bucket(self):
        bucket, bucket_key = failure_bucket(bool_codegen(0), 'tilelang_codegen_error', 'tilelang_bool_cuda_type')
        self.assertEqual(bucket, 'tilelang_bool_cuda_type')
        self.assertEqual(bucket_key, failure_key(bool_codegen(0)))

    def test_other_failures_are_root_cause_and_digest(self):
        bucket, bucket_key = failure_bucket(tvm_check(1), 'other')
        self.assertRegex(bucket, r'^other:[0-9a-f]{10}$')
        self.assertEqual(bucket, 'other:' + hashlib.sha1(bucket_key.encode()).hexdigest()[:10])
        self.assertEqual(failure_bucket(tvm_check(7), 'other')[0], bucket)
        self.assertTrue(failure_bucket(tvm_check(1), '')[0].startswith('other:'))
        self.assertNotEqual(failure_bucket(tvm_check(1), 'tilelang_layout')[0], bucket)


class SpeciesTests(unittest.TestCase):
    def test_chao1_and_good_turing(self):
        self.assertEqual(species(Counter(a=1, b=1, c=2, d=5), samples=100),
                         {'observed': 4, 'singletons': 2, 'doubletons': 1, 'chao1': 6.0,
                          'samples': 100, 'unseen_probability': 0.02})
        # Without doubletons the bias-corrected form keeps the bound finite.
        self.assertEqual(species(Counter(a=1, b=1, c=3))['chao1'], 4.0)
        self.assertNotIn('samples', species(Counter(a=1)))
        self.assertEqual(species(Counter(), samples=0)['unseen_probability'], 1.0)


def scenario(passes=30, failures=5, window=64):
    """`passes` tests without the trigger, then `failures` that all carry it."""
    quarantine = Quarantine(window=window)
    for i in range(passes):
        quarantine.observe({'op:add', f'size:{i % 4}'})
    for i in range(failures):
        quarantine.observe({'op:select', 'dtype:bool', f'size:{i % 4}'}, 'bool')
    return quarantine


class QuarantineTests(unittest.TestCase):
    def test_learns_the_discriminating_feature(self):
        quarantine = scenario()
        stats = quarantine.stats()
        # op:select is just as precise; the smaller (first sorted) id wins.
        self.assertEqual(stats['rules'], {'bool': [['dtype:bool']]})
        self.assertEqual(stats['learned'], 1)
        self.assertEqual(quarantine.matches({'dtype:bool', 'op:add'}), ['bool'])
        self.assertEqual(quarantine.matches({'op:select'}), [])

    def test_admission_draws_against_the_bucket_epsilon(self):
        quarantine = scenario()
        self.assertAlmostEqual(quarantine.epsilon('bool'), 0.6)  # K = 3 over 5 hits
        candidate = {'dtype:bool', 'op:add'}
        with patch.object(triage.random, 'random', return_value=0.99):
            self.assertFalse(quarantine.admit(candidate))
            self.assertTrue(quarantine.admit(candidate, force=True))
            self.assertTrue(quarantine.admit({'op:add'}))
        with patch.object(triage.random, 'random', return_value=0.0):
            self.assertTrue(quarantine.admit(candidate))
        stats = quarantine.stats()
        self.assertEqual((stats['rejected'], stats['explored'], stats['forced']), ({'bool': 1}, {'bool': 1}, 1))

    def test_epsilon_shrinks_with_hits_down_to_the_floor(self):
        quarantine = Quarantine()
        self.assertEqual(quarantine.epsilon('bool'), 1.0)
        quarantine.hits.update({'bool': 30, 'other:0123456789': 30})
        self.assertAlmostEqual(quarantine.epsilon('bool'), 3 / 30)
        # An automatic bucket may merge mechanisms: it is revisited more.
        self.assertAlmostEqual(quarantine.epsilon('other:0123456789'), 8 / 30)
        quarantine.hits['bool'] = 10 ** 6
        self.assertEqual(quarantine.epsilon('bool'), 0.01)
        custom = Quarantine(min_explore=0.05)
        custom.hits['bool'] = 10 ** 6
        self.assertEqual(custom.epsilon('bool'), 0.05)

    def test_rule_retires_when_matching_tests_pass(self):
        quarantine = scenario()
        for _ in range(2):
            quarantine.observe({'dtype:bool', 'op:add'})
        self.assertIn('bool', quarantine.rules)  # 5 of 7 matching tests failed
        quarantine.observe({'dtype:bool', 'op:add'})
        self.assertEqual(quarantine.rules, {})  # 5 of 8 is below 0.9 - 0.2
        self.assertEqual(quarantine.retired, 1)

    def test_rules_accumulate_per_bucket(self):
        quarantine = scenario()
        for i in range(5):
            quarantine.observe({'op:flip', f'size:{i % 4}'}, 'bool')
        self.assertEqual(quarantine.stats()['rules'], {'bool': [['dtype:bool'], ['op:flip']]})
        self.assertEqual(quarantine.learned, 2)
        self.assertEqual(quarantine.matches({'op:flip'}), ['bool'])

    def test_no_rule_without_discriminating_evidence(self):
        quarantine = Quarantine(window=64)
        for i in range(30):
            quarantine.observe({'op:add', f'size:{i % 4}'})
        for i in range(5):
            quarantine.observe({'op:add', f'size:{i % 4}'}, 'x')
        self.assertEqual((quarantine.rules, quarantine.learned), ({}, 0))
        # Few negatives make any conjunction look precise.
        quarantine = scenario(passes=10)
        self.assertEqual(quarantine.rules, {})
        for i in range(10):
            quarantine.observe({'op:add', f'size:{i % 4}'})
        quarantine.observe({'op:select', 'dtype:bool'}, 'bool')
        self.assertEqual(quarantine.stats()['rules'], {'bool': [['dtype:bool']]})

    def test_unlearnable_failures_are_counted_without_a_rule(self):
        quarantine = scenario(failures=0)
        for i in range(8):
            quarantine.observe({'op:select', 'dtype:bool', f'size:{i % 4}'}, 'wrong', learn=False)
        self.assertEqual((quarantine.rules, quarantine.learned, quarantine.hits['wrong']), ({}, 0, 8))
        self.assertEqual([label for _, label in quarantine.samples].count('wrong'), 8)

    def test_snapshot_round_trip(self):
        quarantine = scenario()
        quarantine.observe({'op:flip'}, 'other:0123456789')
        with patch.object(triage.random, 'random', return_value=0.99):
            quarantine.admit({'dtype:bool'})
        state = json.loads(json.dumps(quarantine.snapshot()))
        restored = Quarantine(window=64)
        restored.restore(state)
        self.assertEqual(restored.stats(), quarantine.stats())
        self.assertEqual(restored.snapshot(), quarantine.snapshot())
        self.assertEqual(restored.matches({'dtype:bool', 'op:add'}), ['bool'])
        self.assertEqual(restored.hits, quarantine.hits)

    def test_invalid_state_is_rejected_without_partial_restore(self):
        quarantine = scenario()
        before = quarantine.snapshot()
        valid = {'version': 1, 'vocabulary': ['a']}
        for state in ({'version': 2, 'vocabulary': []},
                      {'version': 1, 'vocabulary': 'a'},
                      {'version': 1, 'vocabulary': [1]},
                      dict(valid, samples=[[[1], None]]),
                      dict(valid, samples=[[[True], None]]),
                      dict(valid, samples=[[[0], 3]]),
                      dict(valid, samples=[[[0]]]),
                      dict(valid, rules={'b': [[0, 1]]}),
                      dict(valid, rules={'b': [0]}),
                      dict(valid, hits={'b': -1}),
                      dict(valid, explored={'b': 1.5}),
                      dict(valid, forced=-1),
                      dict(valid, learned='3')):
            with self.subTest(state=state), self.assertRaises(ValueError):
                quarantine.restore(state)
            self.assertEqual(quarantine.snapshot(), before)

    def test_window_is_trimmed_on_observe_and_restore(self):
        def names(quarantine):
            reverse = {i: f for f, i in quarantine.vocabulary.items()}
            return [reverse[next(iter(ids))] for ids, _ in quarantine.samples]

        large = Quarantine(window=128)
        small = Quarantine(window=64)
        for i in range(100):
            large.observe({f'f:{i}'})
            small.observe({f'f:{i}'})
        self.assertEqual(names(large), [f'f:{i}' for i in range(100)])
        self.assertEqual(names(small), [f'f:{i}' for i in range(36, 100)])
        restored = Quarantine(window=64)
        restored.restore(large.snapshot())
        self.assertEqual(names(restored), names(small))

    def test_constructor_validation_and_sorted_ids(self):
        for settings in ({'window': 63}, {'precision': 0.5}, {'precision': 1.01},
                         {'min_explore': 0}, {'min_explore': 1.5}):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                Quarantine(**settings)
        Quarantine(window=64, precision=1.0, min_explore=1.0)
        quarantine = Quarantine()
        quarantine.ids({'b', 'c', 'a'})
        self.assertEqual(quarantine.vocabulary, {'a': 0, 'b': 1, 'c': 2})


class ExplainedFeedbackTests(unittest.TestCase):
    def test_explained_features_lose_their_rarity_boost(self):
        feedback = StructuralFeedback()
        feature = key('op', 'exp')
        feedback.attempted[feature] = feedback.passed[feature] = 1
        self.assertEqual(feedback.weight(feature), 2.0)
        feedback.explain({feature})
        self.assertAlmostEqual(feedback.weight(feature), 1 + 2 / 3)

        program = dataflow_program()
        feedback.observe(program, True)
        before = feedback.seed_weight(program)
        feedback.explain(program_features(program))
        self.assertLess(feedback.seed_weight(program), before)

    def test_explained_counts_persist_and_old_files_restore_empty(self):
        feedback = StructuralFeedback()
        feedback.explain({key('op', 'exp')})
        feedback.explain({key('op', 'exp'), key('op', 'add')})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'structural_feedback.json'
            feedback.save(path)
            restored = StructuralFeedback()
            restored.restore(path)
            self.assertEqual(restored.explained, Counter({key('op', 'exp'): 2, key('op', 'add'): 1}))
            data = json.loads(path.read_text())
            del data['explained']
            path.write_text(json.dumps(data))
            restored = StructuralFeedback()
            restored.restore(path)
            self.assertEqual(restored.explained, Counter())
            data['explained'] = {key('op', 'exp'): -1}
            path.write_text(json.dumps(data))
            with self.assertRaises(ValueError):
                StructuralFeedback().restore(path)


def template_kinds(nodes):
    """Template kinds chosen by the swarm draw (carry merges are fixed)."""
    for node in nodes:
        if not node.carry_merge:
            yield node.kind
        for child in node.regions:
            yield from template_kinds(child)


class SwarmTests(unittest.TestCase):
    def setUp(self):
        state = random.getstate()
        self.addCleanup(random.setstate, state)
        random.seed(5)

    def kinds(self, swarm, **overrides):
        generator = RegionGenerator(Config(coverage_probe_prob=0, region_typed_prob=0, **overrides))
        generator._swarm = set(swarm)
        return {kind for _ in range(40) for kind in template_kinds(generator.template())}

    def test_templates_draw_from_the_swarm(self):
        self.assertEqual(self.kinds({'exp', 'add', 'for'}), {'exp', 'add', 'for'})
        # Controls outside the swarm are never chosen, even when forced.
        self.assertEqual(self.kinds({'exp'}, region_control_prob=1), {'exp'})

    def test_an_empty_choice_falls_back_to_all_candidates(self):
        kinds = self.kinds({'cast'})  # only typed operations, which are off
        self.assertLessEqual(kinds, set(RegionGenerator.LEAVES))
        self.assertGreater(len(kinds), 1)

    def test_program_template_draws_one_mask_per_program(self):
        generator = RegionGenerator(Config(coverage_probe_prob=0, region_typed_prob=0, swarm_prob=1))
        masks = []
        for _ in range(10):
            template = generator.program_template()
            mask = generator._swarm
            self.assertLessEqual(mask, set(RegionGenerator.SWARM_KINDS))
            chosen = set(template_kinds(template.entry[1:]))
            for function in template.functions:
                chosen |= set(template_kinds(function.operations))
            self.assertLessEqual(chosen - {'call'}, mask)
            masks.append(frozenset(mask))
        self.assertGreater(len(set(masks)), 1)

    def test_mask_is_reset_for_unmasked_and_probe_templates(self):
        generator = RegionGenerator(Config(coverage_probe_prob=0, swarm_prob=0))
        generator._swarm = {'exp'}
        generator.program_template()
        self.assertIsNone(generator._swarm)
        probe = RegionGenerator(Config(coverage_probe_prob=1, swarm_prob=1))
        probe._swarm = {'exp'}
        self.assertEqual(probe.program_template().entry[0].kind, 'probe')
        self.assertIsNone(probe._swarm)

    def test_swarm_kinds_and_validation(self):
        kinds = RegionGenerator.SWARM_KINDS
        self.assertEqual(len(kinds), len(set(kinds)))
        self.assertTrue({'for', 'if', 'exp', 'cast', 'store_tile'} <= set(kinds))
        self.assertNotIn('to_tile', kinds)
        with self.assertRaises(ValueError):
            RegionGenerator(Config(swarm_prob=1.5))


def flip_program(backend, **attrs):
    generator = ExtendedGenerator(Config(extended_prob=1), backend)
    builder = Builder(generator)
    buffer = generator.buffer('float32', 64)
    loaded = builder.load(buffer, (8, 8), builder.indices((8, 8), shuffled=False))
    flipped = builder.emit('flip', [loaded], [loaded.type], **attrs)
    builder.block.returns = [flipped.name]
    return validated(generator, builder), buffer, flipped


class FlipAxisTests(unittest.TestCase):
    def test_axis_must_be_an_in_range_integer(self):
        for axis in (0, 1, -1, -2):
            flip_program('triton', axis=axis)
        for axis in (2, -3, '0', True, 0.0, None):
            with self.subTest(axis=axis), self.assertRaisesRegex(ValueError, 'Invalid flip axis'):
                flip_program('triton', axis=axis)

    def test_lowerings_reverse_the_named_axis(self):
        for attrs, axis in (({}, 1), ({'axis': 0}, 0), ({'axis': -2}, 0), ({'axis': 1}, 1)):
            with self.subTest(attrs=attrs):
                program = flip_program('triton', **attrs)[0]
                code = emit('triton', program)
                self.assertRegex(code, rf'tl\.flip\(e\d+, {axis}\)')
                # The one-argument call is the confirmed default-axis bug.
                self.assertNotRegex(code, r'tl\.flip\(\s*\w+\s*\)')
                tilelang = emit('tilelang', flip_program('tilelang', **attrs)[0])
                self.assertIn('_flip_shared[7 - i, j]' if axis == 0 else '_flip_shared[i, 7 - j]', tilelang)

    def test_reference_and_feature_follow_the_axis(self):
        for attrs, axis in (({}, 1), ({'axis': 0}, 0), ({'axis': -2}, 0)):
            with self.subTest(attrs=attrs):
                program, buffer, flipped = flip_program('tilelang', **attrs)
                outputs, memory = reference(program)
                expected = torch.flip(memory[buffer.name][0, 16:80].view(8, 8), [axis])
                self.assertTrue(torch.equal(outputs[flipped.name][0], expected))
                self.assertIn(key('attribute', 'flip', axis), extended_features(program))

    def test_flip_axes_follow_the_installed_triton(self):
        self.assertEqual(flip_axes('tilelang', 3), (0, 1, 2))
        try:
            import triton.language as tl
        except ImportError:
            self.skipTest('triton is not installed')
        final_only = 'only final dimension' in (tl.flip.__doc__ or '')
        self.assertEqual(flip_axes('triton', 2), (1,) if final_only else (0, 1))

    def test_generation_and_mutation_name_supported_axes(self):
        state = random.getstate()
        self.addCleanup(random.setstate, state)
        config = Config(extended_prob=1, extended_shape_op_prob=1)
        for backend in ('triton', 'tilelang'):
            seen = set()
            for seed in range(12):
                random.seed(seed)
                program = ExtendedGenerator(config, backend).generate('arithmetic')
                for candidate in [program] + [mutate_extended(program, config, backend) for _ in range(4)]:
                    candidate.validate()
                    for node in candidate.all_operations():
                        if node.op == 'flip':
                            rank = len(node.results[0].type.shape)
                            self.assertIn(node.attrs['axis'], flip_axes(backend, rank))
                            seen.add((rank, node.attrs['axis']))
            self.assertEqual({axis for rank, axis in seen if rank == 2}, set(flip_axes(backend, 2)))


class DslInputFilterTests(unittest.TestCase):
    def test_vector_input_filter_is_a_magnitude_bound(self):
        # TVM folds `x == x` to true, which let NaN reach the clamp.
        state = random.getstate()
        self.addCleanup(random.setstate, state)
        random.seed(12)
        config = Config(extended_prob=1, extended_atomic_prob=0, extended_fma_prob=0,
                        extended_shape_op_prob=1, extended_int8_prob=0, extended_elementwise_prob=0)
        for backend in ('tilelang', 'triton'):
            parent = ExtendedGenerator(config, backend).generate('arithmetic')
            child = extend_passed(parent, backend, 'dsl_sigmoid', config)
            added = child.body.operations[len(parent.body.operations):]
            producers = {value.name: node for node in added for value in node.results}
            self.assertFalse([node for node in child.all_operations()
                              if node.op == 'eq' and node.operands[0] == node.operands[1]])
            bound = next(node for node in added if node.op == 'lt')
            magnitude, limit = (producers[name] for name in bound.operands)
            self.assertEqual(magnitude.op, 'abs')
            self.assertEqual((limit.op, limit.attrs['value']), ('constant', 1e30))
            guard = next(node for node in added if node.op == 'select')
            self.assertEqual(guard.operands[:2], [bound.results[0].name, magnitude.operands[0]])


def campaign(directory, iterations, verdict, resume=None, verbose=True, **overrides):
    """A real campaign whose oracle verdict is `verdict(call_index, program)`."""
    settings = dict(seed=42, output_dir=directory, dim_range=(1, 64), coverage_probe_prob=0,
                    region_typed_prob=0, extended_prob=0, structural_feedback=False,
                    backends=['tilelang'], quarantine=True, explained_feedback=True)
    settings.update(overrides)
    calls = []

    def fake_test(program):
        calls.append(program)
        return verdict(len(calls) - 1, program)

    log = io.StringIO()
    with contextlib.redirect_stdout(log):
        fuzzer = TileSmith(Config(**settings), resume_dir=resume)
        fuzzer.oracle.test = fake_test
        fuzzer.run(iterations, verbose=verbose)
    return fuzzer, calls, log.getvalue()


def resume(fuzzer, **overrides):
    settings = dict(seed=42, output_dir=str(fuzzer.output_dir.parent), dim_range=(1, 64), coverage_probe_prob=0,
                    region_typed_prob=0, extended_prob=0, structural_feedback=False,
                    backends=['tilelang'], quarantine=True, explained_feedback=True)
    settings.update(overrides)
    with contextlib.redirect_stdout(io.StringIO()):
        return TileSmith(Config(**settings), resume_dir=str(fuzzer.output_dir))


def cycle(index, program):
    """pass, a TVM check, the audited bool failure, oracle noise, ..."""
    kind = index % 4
    if kind == 0:
        return None
    if kind == 1:
        return BugReport(BugType.COMPILE_CRASH, tvm_check(index), params=program.params_dict,
                         root_cause='other', generated_code='# check\n')
    if kind == 2:
        return BugReport(BugType.COMPILE_CRASH, bool_codegen(index), params=program.params_dict,
                         root_cause='tilelang_codegen_error', generated_code='# bool\n')
    return BugReport(BugType.ORACLE_UNSTABLE, f'ORACLE UNSTABLE: reference disagrees (case {index})',
                     params=program.params_dict, root_cause='oracle_unstable', generated_code='# unstable\n')


class CampaignBucketTests(unittest.TestCase):
    def test_buckets_reach_log_summary_and_resume(self):
        other = failure_bucket(tvm_check(1), 'other')[0]
        unstable = failure_bucket('ORACLE UNSTABLE: reference disagrees (case 3)', 'oracle_unstable')[0]
        with tempfile.TemporaryDirectory() as directory:
            fuzzer, calls, log = campaign(directory, 12, cycle)
            self.assertEqual(len(calls), 12)
            expected = {other: 3, 'tilelang_bool_cuda_type': 3, unstable: 3}
            self.assertEqual(fuzzer.failure_buckets, Counter(expected))

            lines = log.splitlines()
            def marked(pattern):
                return [line for line in lines if re.search(pattern, line)]
            self.assertEqual(len(marked(rf'\[FAILED\] \(NEW / other\) .*\[{other} NEW\]$')), 1)
            self.assertEqual(len(marked(rf'\[FAILED\] \(saved / other\) .*\[{other} x2\]$')), 1)
            self.assertEqual(len(marked(rf'\[{other} x3\]$')), 1)
            self.assertEqual(len(marked(r'\[FAILED\] \(NEW / tilelang_codegen_error\) .*'
                                        r'\[tilelang_bool_cuda_type NEW\]$')), 1)
            self.assertEqual(len(marked(rf'\[ORACLE UNSTABLE\] \(saved\) .*\[{unstable} NEW\]$')), 1)

            summary = json.loads((fuzzer.output_dir / 'summary.json').read_text())
            self.assertEqual(summary['failure_buckets'], expected)
            self.assertEqual(summary['failure_bucket_keys'][other],
                             'RuntimeError: Check failed: (e# == e#) is false | in lower')
            # Oracle noise is not a failure mechanism.
            self.assertEqual(summary['failure_species'],
                             {'observed': 2, 'singletons': 0, 'doubletons': 0, 'chao1': 2.0,
                              'samples': 12, 'unseen_probability': 0.0})
            self.assertEqual(summary['structural_species']['observed'], len(fuzzer.feedback.passed))
            self.assertEqual(summary['quarantine'], json.loads(json.dumps(fuzzer.quarantine.stats())))
            self.assertLessEqual({'quarantine': True, 'quarantine_window': 1024, 'quarantine_precision': 0.9,
                                  'quarantine_retries': 8, 'quarantine_min_explore': 0.01,
                                  'explained_feedback': True, 'swarm_prob': 0.0}.items(),
                                 summary['generation_config'].items())
            saved = json.loads(next((fuzzer.output_dir / 'failed' / 'other').glob('*.json')).read_text())
            self.assertEqual(saved['failure_bucket'], other)

            state = json.loads((fuzzer.output_dir / 'quarantine.json').read_text())
            self.assertEqual(state, json.loads(json.dumps(fuzzer.quarantine.snapshot())))
            restored = resume(fuzzer)
            self.assertEqual(json.loads(json.dumps(restored.quarantine.snapshot())), state)
            self.assertEqual(restored.failure_buckets, fuzzer.failure_buckets)

    def test_history_without_buckets_is_rebucketed(self):
        with tempfile.TemporaryDirectory() as directory:
            fuzzer, _, _ = campaign(directory, 12, cycle, verbose=False)
            for path in (fuzzer.output_dir / 'failed').rglob('*.json'):
                report = json.loads(path.read_text())
                del report['failure_bucket'], report['failure_key']
                path.write_text(json.dumps(report))
            path = fuzzer.output_dir / 'summary.json'
            summary = json.loads(path.read_text())
            del summary['failure_buckets'], summary['failure_bucket_keys']
            path.write_text(json.dumps(summary))
            restored = resume(fuzzer)
            self.assertEqual(restored.failure_buckets, fuzzer.failure_buckets)
            self.assertEqual(restored.failure_bucket_keys, fuzzer.failure_bucket_keys)

    def test_quarantine_off_writes_no_state(self):
        with tempfile.TemporaryDirectory() as directory:
            fuzzer, _, _ = campaign(directory, 8, cycle, verbose=False, quarantine=False)
            self.assertIsNone(fuzzer.quarantine)
            self.assertFalse((fuzzer.output_dir / 'quarantine.json').exists())
            summary = json.loads((fuzzer.output_dir / 'summary.json').read_text())
            self.assertIsNone(summary['quarantine'])
            self.assertEqual(sum(summary['failure_buckets'].values()), 6)

    def test_duplicate_failures_explain_their_features(self):
        def same_failure(index, program):
            return BugReport(BugType.COMPILE_CRASH, bool_codegen(index), params=program.params_dict,
                             root_cause='tilelang_codegen_error', generated_code='# bool\n')

        with tempfile.TemporaryDirectory() as directory:
            fuzzer, calls, _ = campaign(directory, 6, same_failure, verbose=False, quarantine=False)
            # The first failure of a bucket is news; its repeats are not.
            self.assertEqual(fuzzer.feedback.explained,
                             Counter(f for program in calls[1:] for f in program_features(program)))
        with tempfile.TemporaryDirectory() as directory:
            fuzzer, _, _ = campaign(directory, 6, same_failure, verbose=False, explained_feedback=False)
            self.assertEqual(fuzzer.feedback.explained, Counter())

    def test_quarantine_retries_bound_the_redraws(self):
        programs = []
        for size in (1, 2, 16):  # singleton, tail and full M: distinct features
            program = dataflow_program()
            program.spec.M = size
            programs.append(program)
        for retries, verdicts, chosen, forced in ((2, [False, False, True], 2, [False, False, True]),
                                                  (0, [True], 0, [True]),
                                                  (8, [True], 0, [False])):
            with self.subTest(retries=retries), tempfile.TemporaryDirectory() as directory, \
                    contextlib.redirect_stdout(io.StringIO()):
                fuzzer = TileSmith(Config(seed=42, output_dir=directory, backends=['tilelang'],
                                          quarantine=True, quarantine_retries=retries))
                with patch.object(fuzzer, '_generate_native', side_effect=programs), \
                        patch.object(fuzzer.quarantine, 'admit', side_effect=verdicts) as admit:
                    self.assertIs(fuzzer._generate_test_case(), programs[chosen])
                self.assertEqual([call.kwargs['force'] for call in admit.call_args_list], forced)
                self.assertEqual([call.args[0] for call in admit.call_args_list],
                                 [program_features(program) for program in programs[:len(forced)]])
                self.assertEqual(fuzzer._current_features, program_features(programs[chosen]))
        with self.assertRaises(ValueError), tempfile.TemporaryDirectory() as directory:
            TileSmith(Config(output_dir=directory, quarantine_retries=-1))

    def test_learned_rule_cuts_repeats_of_a_recurring_bucket(self):
        # A stub compiler bug that every program containing exp triggers.
        exp = key('op', 'exp')
        for message, root_cause, bucket in (
                ('tvm.error.InternalError: exp lowering failed', 'tilelang_codegen_error', None),
                (bool_codegen(0), 'tilelang_codegen_error', 'tilelang_bool_cuda_type')):
            bucket = bucket or failure_bucket(message, root_cause)[0]

            def verdict(index, program):
                if exp not in program_features(program):
                    return None
                return BugReport(BugType.COMPILE_CRASH, message, params=program.params_dict,
                                 root_cause=root_cause, generated_code='# exp\n')

            with self.subTest(bucket=bucket), tempfile.TemporaryDirectory() as first, \
                    tempfile.TemporaryDirectory() as second:
                baseline, _, _ = campaign(first, 300, verdict, verbose=False, quarantine=False)
                guarded, _, _ = campaign(second, 300, verdict, verbose=False)
                self.assertEqual(guarded.quarantine.stats()['rules'], {bucket: [[exp]]})
                self.assertGreater(baseline.failure_buckets[bucket], 30)
                self.assertLess(guarded.failure_buckets[bucket], 0.7 * baseline.failure_buckets[bucket])
                # Every quarantined bucket is still revisited.
                self.assertGreater(guarded.quarantine.explored[bucket], 0)

    def test_wrong_results_are_never_quarantined(self):
        # Distinct miscompilations share this message: one rule could hide all.
        exp = key('op', 'exp')
        message = 'RuntimeError: WRONG RESULT: structured reference: error=0.5, tolerance=0.01'

        def verdict(index, program):
            if exp not in program_features(program):
                return None
            return BugReport(BugType.WRONG_RESULT, message, params=program.params_dict,
                             root_cause='wrong_result', generated_code='# exp\n')

        with tempfile.TemporaryDirectory() as directory:
            fuzzer, _, _ = campaign(directory, 150, verdict, verbose=False)
        bucket = failure_bucket(message, 'wrong_result')[0]
        self.assertGreater(fuzzer.failure_buckets[bucket], 10)
        self.assertEqual(fuzzer.quarantine.hits[bucket], fuzzer.failure_buckets[bucket])
        self.assertEqual((fuzzer.quarantine.rules, fuzzer.quarantine.learned), ({}, 0))


NOVEL = 'ValueError: a lowering check no earlier test failed'


def later(index, program):
    """cycle, except that the sixth test of this segment fails anew."""
    if index == 5:
        return BugReport(BugType.COMPILE_CRASH, NOVEL, params=program.params_dict,
                         root_cause='other', generated_code='# novel\n')
    return cycle(index, program)


class DiscoveryTimelineTests(unittest.TestCase):
    def summary(self, fuzzer):
        return json.loads((fuzzer.output_dir / 'summary.json').read_text())

    def test_discovery_keeps_triggers_buckets_and_confirmed_root_causes_apart(self):
        other = failure_bucket(tvm_check(1), 'other')[0]
        unstable = failure_bucket('ORACLE UNSTABLE: reference disagrees (case 3)', 'oracle_unstable')[0]
        with tempfile.TemporaryDirectory() as directory:
            fuzzer, _, _ = campaign(directory, 12, cycle, verbose=False)
            summary = self.summary(fuzzer)
            first = summary['failure_bucket_first_seen']
            self.assertEqual([(bucket, seen['tested']) for bucket, seen in first.items()],
                             [(other, 2), ('tilelang_bool_cuda_type', 3), (unstable, 4)])
            self.assertEqual(summary['timeline_origin']['tested'], 0)
            discovery = summary['discovery']
            self.assertLessEqual({'failure_triggers': 6, 'oracle_unstable': 3, 'diagnostic_buckets': 2,
                                  'confirmed_root_causes': ['tilelang_bool_cuda_type'],
                                  'tests_since_new_bucket': 9, 'new_buckets_last_hour': 2,
                                  'new_confirmed_last_hour': 1}.items(), discovery.items())
            # Oracle noise seen later is not a discovery.
            self.assertEqual(discovery['last_new_bucket'], dict(first['tilelang_bool_cuda_type'],
                                                                bucket='tilelang_bool_cuda_type'))
            self.assertGreater(discovery['new_confirmed_per_hour'], 0)
            progress = json.loads((fuzzer.output_dir / 'coverage_progress.json').read_text())
            self.assertEqual(progress['discovery']['diagnostic_buckets'], 2)

    def test_closing_stats_and_features_name_what_they_count(self):
        """Lowered IR of programs that then fail is counted apart from that of
        passing programs, and the closing stats keep triggers, buckets and
        confirmed root causes apart from each other and from oracle noise."""
        with tempfile.TemporaryDirectory() as directory:
            log = io.StringIO()
            with contextlib.redirect_stdout(log):
                fuzzer = TileSmith(Config(seed=42, output_dir=directory, dim_range=(1, 64), coverage_probe_prob=0,
                                          region_typed_prob=0, extended_prob=0, structural_feedback=False,
                                          backends=['tilelang'], quarantine=True, explained_feedback=True))
                calls = []

                def lowered_then_verdict(program):
                    calls.append(program)
                    # Every program lowers to an IR feature of its own.
                    fuzzer.oracle.last_compilation = [{'features': [key('compiler_stage', f'test{len(calls)}')]}]
                    fuzzer.oracle.compilation_complete = True
                    return cycle(len(calls) - 1, program)
                fuzzer.oracle.test = lowered_then_verdict
                fuzzer.run(12, verbose=True)
            summary = self.summary(fuzzer)
            # Of twelve lowered programs, the first of every four passes.
            self.assertEqual((summary['compiler_ir_features'], summary['compiler_ir_features_passed']), (12, 3))
            progress = json.loads((fuzzer.output_dir / 'coverage_progress.json').read_text())
            self.assertEqual(progress['compiler_ir_features_passed'], 3)
            stats = log.getvalue().split('=== TileSmith Fuzzing Stats ===')[1]
            for line in ('Failure triggers: 6', 'Failure categories: 2', 'Diagnostic buckets: 2',
                         'Confirmed root causes: 1', 'Oracle unstable (not failures): 3'):
                self.assertIn(line + '\n', stats)
            self.assertNotIn('Bugs', stats)

    def test_resumed_campaigns_continue_the_timeline(self):
        with tempfile.TemporaryDirectory() as directory:
            fuzzer, _, _ = campaign(directory, 12, cycle, verbose=False)
            before = self.summary(fuzzer)
            before['campaign_seconds'] = 1000.0  # the first segment ran that long
            (fuzzer.output_dir / 'summary.json').write_text(json.dumps(before))
            resumed, _, _ = campaign(directory, 8, later, resume=str(fuzzer.output_dir), verbose=False)
            after = self.summary(resumed)
            novel = failure_bucket(NOVEL, 'other')[0]
            # Old buckets seen again keep their sighting; the new one is test 18.
            first = after['failure_bucket_first_seen']
            self.assertEqual(list(first), list(before['failure_bucket_first_seen']) + [novel])
            self.assertEqual({bucket: first[bucket] for bucket in before['failure_bucket_first_seen']},
                             before['failure_bucket_first_seen'])
            self.assertEqual(first[novel]['tested'], 18)
            self.assertGreaterEqual(first[novel]['seconds'], 1000)
            self.assertGreaterEqual(after['campaign_seconds'], first[novel]['seconds'])
            self.assertEqual(after['timeline_origin'], before['timeline_origin'])
            self.assertEqual(after['discovery']['last_new_bucket']['bucket'], novel)
            self.assertEqual(after['discovery']['tests_since_new_bucket'], 2)
            # Oracle noise stays out of the bug counts across the resume.
            self.assertEqual(after['oracle_unstable'], 5)
            self.assertNotIn('oracle_unstable', after['root_causes'])
            self.assertEqual(after['bugs_total'], sum(after['root_causes'].values()))
            self.assertEqual(after['discovery']['failure_triggers'], after['bugs_total'])

    def test_campaigns_from_before_the_timeline_start_it_when_resumed(self):
        with tempfile.TemporaryDirectory() as directory:
            fuzzer, _, _ = campaign(directory, 12, cycle, verbose=False)
            path = fuzzer.output_dir / 'summary.json'
            summary = json.loads(path.read_text())
            for name in ('failure_bucket_first_seen', 'campaign_seconds', 'timeline_origin', 'discovery'):
                del summary[name]
            # Resuming used to count the saved noise as a root cause.
            summary['root_causes']['oracle_unstable'] = 3
            path.write_text(json.dumps(summary))
            (fuzzer.output_dir / 'coverage_progress.jsonl').unlink()
            restored = resume(fuzzer)
            self.assertEqual(restored.timeline_origin['tested'], 12)
            times = {}
            for report in (fuzzer.output_dir / 'failed').rglob('*.json'):
                report = json.loads(report.read_text())
                times[report['failure_bucket']] = min(times.get(report['failure_bucket'], report['timestamp']),
                                                      report['timestamp'])
            self.assertEqual(restored.failure_bucket_first_seen,
                             {bucket: {'tested': None, 'seconds': None, 'time': time}
                              for bucket, time in times.items()})
            discovery = restored._discovery()
            self.assertIsNone(discovery['last_new_bucket'])
            self.assertEqual((discovery['diagnostic_buckets'], discovery['tests_since_new_bucket'],
                              discovery['new_buckets_per_hour']), (2, 0, 0.0))
            self.assertEqual(restored.known_root_causes, {'other': 3, 'tilelang_codegen_error': 3})
            self.assertEqual(restored.stats.oracle_unstable, 3)
            self.assertNotIn('oracle_unstable', restored.root_cause_locations)

    def test_a_killed_segment_is_recovered_from_the_progress_log(self):
        with tempfile.TemporaryDirectory() as directory:
            fuzzer, _, _ = campaign(directory, 12, cycle, verbose=False)
            path = fuzzer.output_dir / 'summary.json'
            stale = path.read_text()
            campaign(directory, 8, later, resume=str(fuzzer.output_dir), verbose=False)
            seconds = json.loads(path.read_text())['campaign_seconds']
            path.write_text(stale)  # as if the second segment died before its summary
            restored = resume(fuzzer)
            # Progress is written every 100 tests and at exit: an upper bound.
            self.assertEqual(restored.failure_bucket_first_seen[failure_bucket(NOVEL, 'other')[0]]['tested'], 20)
            self.assertAlmostEqual(restored._historical_seconds, seconds, delta=1)
            self.assertEqual(restored.timeline_origin, json.loads(stale)['timeline_origin'])


class MainDefaultsTests(unittest.TestCase):
    def configs(self, *argv):
        import main
        configs = []

        class Recorder:
            def __init__(self, config, resume_dir=None):
                configs.append(config)

            def run(self, num_iterations, verbose):
                pass

        with patch.object(main, 'TileSmith', Recorder), patch.object(sys, 'argv', ['main.py', *argv]):
            self.assertEqual(main.main(), 0)
        return configs[0]

    def triage_settings(self, *argv):
        config = self.configs(*argv)
        return config.quarantine, config.explained_feedback, config.swarm_prob, config.quarantine_retries

    def test_new_campaigns_enable_triage_and_resumes_keep_their_setting(self):
        with tempfile.TemporaryDirectory() as directory:
            self.assertEqual(self.triage_settings('-o', directory), (True, True, 0.5, 8))
            self.assertEqual(self.triage_settings('-o', directory, '--no-quarantine', '--swarm-prob', '0',
                                                  '--quarantine-retries', '3'), (False, True, 0.0, 3))
            old = Path(directory) / 'old'
            old.mkdir()
            (old / 'summary.json').write_text(json.dumps({'generation_config': {'extended_prob': 0.0}}))
            self.assertEqual(self.triage_settings('-o', directory, '--resume', 'old'), (False, False, 0.0, 8))
            new = Path(directory) / 'new'
            new.mkdir()
            (new / 'summary.json').write_text(json.dumps({'generation_config': {
                'extended_prob': 0.0, 'quarantine': True, 'explained_feedback': True, 'swarm_prob': 0.5}}))
            self.assertEqual(self.triage_settings('-o', directory, '--resume', 'new'), (True, True, 0.5, 8))
            self.assertEqual(self.triage_settings('-o', directory, '--resume', 'old', '--quarantine'),
                             (True, False, 0.0, 8))

    def test_invalid_triage_flags_are_rejected(self):
        for argv in (('--swarm-prob', '1.5'), ('--quarantine-retries', '-1')):
            with self.subTest(argv=argv), self.assertRaises(SystemExit), \
                    contextlib.redirect_stderr(io.StringIO()):
                self.configs(*argv)


if __name__ == '__main__':
    unittest.main()
