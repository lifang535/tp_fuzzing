"""The extended oracle trust gate: perturbed references decide which
mismatches the reference itself cannot settle."""
import ast
import builtins
import json
import random
import symtable
import traceback
import unittest

import torch

from src.backends import get_backend
from src.backends.common.diagnostics import classify_root_cause
from src.config import Config
from src.ir.extended import ExtendedProgram, TensorType as Ty
from src.workflow.emitter.extended_runtime import (extended_check, extended_check_atomic,
                                                   extended_check_fma, extended_inputs,
                                                   extended_reference, run_extended)
from src.workflow.generator.extended import Builder, ExtendedGenerator
from src.workflow.triage import failure_key, wrong_result_origin


def rounding_program():
    """floor(cos(x)) and cos(x) < 1 on inputs in {0, +-0.125, +-0.25}: exact
    at x != 0, but at x == 0 one rounding of cos decides between 1 and 0."""
    gen = ExtendedGenerator(Config(extended_prob=1), 'triton')
    b = Builder(gen)
    buf = gen.buffer('float32', 64)
    x = b.load(buf, (64,), b.indices((64,), shuffled=False))
    cosine = b.emit('cos', [x], [Ty('float32', (64,))])
    floor = b.emit('floor', [cosine], [Ty('float32', (64,))])
    below = b.binary('lt', cosine, b.constant(Ty('float32', (64,)), 1.0))
    b.block.returns = [floor.name, below.name, cosine.name]
    program = ExtendedProgram(b.block, gen.buffers, input_pattern='integer', family='hand')
    program.validate()
    return json.loads(json.dumps(program.to_dict()))


def simulated(program, perturb=None, fault=None):
    """prepare() for run_extended: one variant that interprets the program,
    optionally with a rounding pattern, then applies `fault` to its outputs."""
    watched = list(dict.fromkeys(program['body']['returns'] + program['observations']))

    def launch(memory, outputs, steps, limit):
        values, final = extended_reference(program, memory, steps, limit, perturb)
        for name, value in final.items():
            memory[name].copy_(value)
        for name, storage in outputs.items():
            value = values[name]
            if fault is not None:
                value = fault(name, value)
            storage[:, 16:-16].copy_(value.reshape(program['blocks'], -1))
    return lambda: [('simulated', watched, launch)]


def failure(call):
    """The formatted traceback of a RuntimeError raised by `call`."""
    try:
        call()
    except RuntimeError as error:
        return ''.join(traceback.format_exception(type(error), error, error.__traceback__))
    raise AssertionError('no failure')


class EnvelopeGateTests(unittest.TestCase):
    def setUp(self):
        state = random.getstate()
        self.addCleanup(random.setstate, state)
        random.seed(3)
        self.program = rounding_program()
        self.floor, self.below, self.cosine = self.program['body']['returns']
        self.zeros = extended_inputs(self.program)[self.program['buffers'][0]['name']][:, 16:-16] == 0

    def test_rounding_decided_values_are_oracle_unstable(self):
        # A kernel whose cos rounds down: floor and the comparison flip
        # exactly where x == 0 and nowhere else.
        actual, _ = extended_reference(self.program, extended_inputs(self.program), 0, 1, 1)
        expected, _ = extended_reference(self.program, extended_inputs(self.program), 0, 1)
        self.assertTrue(self.zeros.any())
        self.assertTrue(torch.equal(actual[self.floor] != expected[self.floor], self.zeros))
        self.assertTrue(torch.equal(actual[self.below] != expected[self.below], self.zeros))
        text = failure(lambda: run_extended(self.program, simulated(self.program, 1), device='cpu'))
        self.assertIn('ORACLE UNSTABLE: simulated:' + self.floor, text)
        self.assertEqual(classify_root_cause(text), 'oracle_unstable')
        # The faithful kernel passes without consulting the envelope.
        run_extended(self.program, simulated(self.program), device='cpu')

    def test_stable_mismatches_stay_wrong_results_and_name_every_value(self):
        def fault(name, value):
            return value + 1 if name == self.floor else value + 0.5 if name == self.cosine else value
        text = failure(lambda: run_extended(self.program, simulated(self.program, fault=fault), device='cpu'))
        self.assertEqual(classify_root_cause(text), 'wrong_result')
        self.assertIn(f'WRONG VALUES: {self.floor}, {self.cosine}\n', text + '\n')
        # The check names floor; cos, which floor reads, is the earlier value.
        self.assertEqual(wrong_result_origin(self.program, text), 'cos float32 r1')
        self.assertNotIn('During handling', text)
        # The key keeps the first line and innermost frame of the old check.
        self.assertEqual(failure_key(text), 'RuntimeError: WRONG RESULT: simulated:e#:seed=#:steps=#:limit=#; '
                                            'max_abs=#; index=#; actual=#; expected=# | in extended_check')

    def test_an_explained_value_does_not_hide_a_wrong_one(self):
        # floor flips only where the reference is undecided, cos is wrong
        # everywhere: the run fails on cos alone.
        def fault(name, value):
            if name == self.floor:
                return torch.where(self.zeros, value - 1, value)
            return value * 2 if name == self.cosine else value
        text = failure(lambda: run_extended(self.program, simulated(self.program, fault=fault), device='cpu'))
        self.assertEqual(classify_root_cause(text), 'wrong_result')
        self.assertIn(f'simulated:{self.cosine}:seed=', text)
        self.assertIn(f'WRONG VALUES: {self.cosine}\n', text + '\n')
        # A flip where the reference is decided is a wrong result as well.
        def stable_flip(name, value):
            return torch.where(~self.zeros, value + 1, value) if name == self.floor else value
        text = failure(lambda: run_extended(self.program, simulated(self.program, fault=stable_flip), device='cpu'))
        self.assertIn(f'simulated:{self.floor}:seed=', text)
        self.assertEqual(classify_root_cause(text), 'wrong_result')

    def test_canaries_and_input_corruption_are_never_explained(self):
        def corrupt(memory, outputs, steps, limit):
            values, _ = extended_reference(self.program, memory, steps, limit, 1)
            for name, storage in outputs.items():
                storage[:, 16:-16].copy_(values[name].reshape(self.program['blocks'], -1))
            outputs[self.floor][:, 0] = 0
        text = failure(lambda: run_extended(self.program, lambda: [('simulated', list(self.program['body']['returns']),
                                                                    corrupt)], device='cpu'))
        self.assertIn('output canary', text)
        self.assertEqual(classify_root_cause(text), 'output_out_of_bounds')

        def overwrite(memory, outputs, steps, limit):
            values, _ = extended_reference(self.program, memory, steps, limit, 1)
            for name, storage in outputs.items():
                storage[:, 16:-16].copy_(values[name].reshape(self.program['blocks'], -1))
            memory[self.program['buffers'][0]['name']][:, 16] += 1
        text = failure(lambda: run_extended(self.program, lambda: [('simulated', list(self.program['body']['returns']),
                                                                    overwrite)], device='cpu'))
        self.assertIn('input storage modified', text)
        self.assertEqual(classify_root_cause(text), 'input_corruption')


class CheckEnvelopeTests(unittest.TestCase):
    def test_checks_excuse_only_disagreements_inside_an_unstable_envelope(self):
        expected = torch.tensor([1., 1., 0.5, 2.])
        # Index 0: the evaluations span [0, 1]; index 1: also unstable but
        # the kernel left the span; index 2: stable; index 3: NaN only valid
        # if an evaluation is NaN.
        envelope = torch.tensor([[1., 1., 0.5, 2.], [0., 1.5, 0.5, 2.], [1., 1., 0.5, float('nan')]])
        self.assertEqual(extended_check(torch.tensor([0., 1., 0.5, float('nan')]), expected, 'ok',
                                        envelope=envelope), 2)
        self.assertEqual(extended_check(expected.clone(), expected, 'equal', envelope=lambda: 1 / 0), 0)
        for actual, index in ((torch.tensor([0., 2., 0.5, 2.]), 1), (torch.tensor([1., 1., 0.25, 2.]), 2)):
            with self.subTest(index=index), self.assertRaisesRegex(RuntimeError, f'index={index};'):
                extended_check(actual, expected, 'outside', envelope=envelope)
        # A NaN where every evaluation is finite stays a wrong result.
        with self.assertRaisesRegex(RuntimeError, 'index=0; actual=nan'):
            extended_check(torch.tensor([float('nan'), 1., 0.5, 2.]), expected, 'nan', envelope=envelope)
        # Without an envelope, or with a mismatched one, nothing is excused.
        with self.assertRaisesRegex(RuntimeError, r'mismatched=2/4'):
            extended_check(torch.tensor([0., 1., 0.5, float('nan')]), expected, 'plain')
        with self.assertRaisesRegex(RuntimeError, r'mismatched=2/4'):
            extended_check(torch.tensor([0., 1., 0.5, float('nan')]), expected, 'shape', envelope=envelope[:, :2])

    def test_integer_atomic_and_fma_checks_take_the_envelope(self):
        expected = torch.tensor([1, 4], dtype=torch.int32)
        envelope = torch.tensor([[1, 4], [0, 4], [2, 4]], dtype=torch.int32)
        self.assertEqual(extended_check(torch.tensor([2, 4], dtype=torch.int32), expected, 'int', envelope=envelope), 1)
        with self.assertRaisesRegex(RuntimeError, 'index=1'):
            extended_check(torch.tensor([1, 5], dtype=torch.int32), expected, 'int', envelope=envelope)
        self.assertEqual(extended_check_atomic(torch.tensor([0, 4], dtype=torch.int32), expected,
                                               'atomic:x', 'add', envelope), 1)
        one = torch.tensor([1.], dtype=torch.float32)
        spread = torch.stack([one, one + 1e-3, one - 1e-3])
        self.assertEqual(extended_check_fma(one + 5e-4, one, 'fma:x', envelope=spread), 1)
        with self.assertRaisesRegex(RuntimeError, 'WRONG RESULT: fma:x'):
            extended_check_fma(one + 5e-4, one, 'fma:x', envelope=torch.stack([one, one, one]))
        # 0-d values keep working.
        scalar = torch.tensor(3.)
        self.assertEqual(extended_check(torch.tensor(2.), scalar, 'scalar',
                                        envelope=torch.tensor([3., 2., 3.])), 1)


class EmittedHarnessTests(unittest.TestCase):
    def test_emitted_extended_harnesses_are_self_contained(self):
        builtins_names = set(dir(builtins)) | {'__name__', '__file__'}
        for backend in ('triton', 'tilelang'):
            random.seed(7)
            program = ExtendedGenerator(Config(extended_prob=1), backend).generate('mixed')
            code = get_backend(backend).make_emitter(Config()).emit(program)
            ast.parse(code)
            table = symtable.symtable(code, '<harness>', 'exec')
            defined = set(table.get_identifiers()) | builtins_names
            unresolved = []

            def walk(scope):
                unresolved.extend(n for n in scope.get_globals() if n not in defined)
                for child in scope.get_children():
                    walk(child)
            for child in table.get_children():
                walk(child)
            with self.subTest(backend=backend):
                self.assertEqual(sorted(set(unresolved)), [])
                for helper in ('_reference_nudge', 'extended_envelopes', 'extended_explained', 'extended_verdict'):
                    self.assertIn(f'def {helper}(', code)


if __name__ == '__main__':
    unittest.main()
