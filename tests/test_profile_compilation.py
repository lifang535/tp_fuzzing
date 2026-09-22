"""Keep the profiling experiment's accounting and instrumentation honest."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from profile_compilation import Recorder, instrument_harness


class ProfilingTests(unittest.TestCase):
    def test_nested_times_are_additive_without_double_counting(self):
        with tempfile.TemporaryDirectory() as directory, patch(
                'profile_compilation.time.perf_counter', side_effect=[1000, 1001, 1003, 1005, 1009]):
            recorder = Recorder(Path(directory) / 'events.jsonl')
            with recorder.span('total'):
                with recorder.span('compiler'):
                    pass
            recorder.stream.close()
            child, total = recorder.events
            self.assertEqual(child['seconds'], 2)
            self.assertEqual(total['seconds'], 8)
            self.assertEqual(total['exclusive_seconds'], 6)
            self.assertEqual(child['parent'], total['id'])
            self.assertEqual(sum(e['exclusive_seconds'] for e in recorder.events), 8)
            saved = [json.loads(line) for line in (Path(directory) / 'events.jsonl').read_text().splitlines()]
            self.assertEqual(saved, recorder.events)

    def test_exception_keeps_balanced_spans_and_original_error(self):
        with tempfile.TemporaryDirectory() as directory:
            recorder = Recorder(Path(directory) / 'events.jsonl')
            with self.assertRaisesRegex(ValueError, 'compile failed'):
                with recorder.span('compiler'):
                    raise ValueError('compile failed')
            recorder.stream.close()
            self.assertEqual(recorder.stack, [])
            self.assertEqual(recorder.events[0]['name'], 'compiler')

    def test_static_method_hook_preserves_calling_convention(self):
        class Backend:
            @staticmethod
            def stage(value, factor):
                return value * factor
        with tempfile.TemporaryDirectory() as directory:
            recorder = Recorder(Path(directory) / 'events.jsonl')
            recorder.patch(Backend, 'stage', 'backend.stage')
            self.assertEqual(Backend().stage(3, 7), 21)
            recorder.stream.close()
            self.assertEqual(len(recorder.events), 1)

    def test_harness_hooks_update_function_globals_not_just_runpy_copy(self):
        scope = {'__file__': '<profile-test>'}
        exec(compile('def _region_reference(x):\n    return x + 1\n'
                     'def test_kernel_0():\n    return _region_reference(6)\n',
                     '<profile-test>', 'exec'), scope)
        namespace = dict(scope)
        with tempfile.TemporaryDirectory() as directory:
            recorder = Recorder(Path(directory) / 'events.jsonl')
            instrument_harness(namespace, recorder)
            self.assertEqual(namespace['test_kernel_0'](), 7)
            recorder.stream.close()
            self.assertEqual([e['name'] for e in recorder.events], ['runtime.reference'])


if __name__ == '__main__':
    unittest.main()
