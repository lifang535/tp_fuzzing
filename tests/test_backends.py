"""Exercise a third registered backend through the unchanged campaign."""
from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from src.backends import Backend, backend_names, get_backend, register_backend
from src.backends.triton.backend import TritonBackend
from src.config import Config
from src.ir import DataType
from src.workflow.generator import ProgramGenerator
from src.workflow.mutator import Mutator
from src.workflow.oracle import Oracle, BugReport, BugType


class RecordingBackend(TritonBackend):
    """Reuse the existing IR target profile, supply an independent CPU harness.

    This is a dispatch test fixture, not an implementation of another GPU DSL.
    """
    name = 'recording'

    def __init__(self):
        self.events = []
        self.fail = False

    def sample_region_spec(self, *args, **kwargs):
        self.events.append('sample')
        return super().sample_region_spec(*args, **kwargs)

    def dtype_parameters_valid(self, spec):
        self.events.append('repair')
        return super().dtype_parameters_valid(spec)

    def validate_program(self, program):
        self.events.append('validate')
        super().validate_program(program)

    def generate_probe(self, config):
        self.events.append('probe')
        return super().generate_probe(config)

    def mutate_probe(self, program, config):
        self.events.append('mutate_probe')
        return super().mutate_probe(program, config)

    def make_emitter(self, config=None):
        owner = self
        class Emitter:
            def emit(self, program):
                owner.events.append('emit')
                return ("import os\nassert os.environ['TILESMITH_PLUGIN_TEST'] == 'active'\n" +
                        ("raise RuntimeError('plugin failure')\n" if owner.fail else "print('PLUGIN PASSED')\n"))
        return Emitter()


    def execution_command(self, path):
        self.events.append('command')
        return [sys.executable, '-I', path]

    def execution_options(self, config):
        self.events.append('environment')
        return {'env': {**os.environ, 'TILESMITH_PLUGIN_TEST': 'active'}}

    def classify_error(self, message):
        self.events.append('error')
        return BugType.RUNTIME_CRASH

    def classify_root_cause(self, message):
        self.events.append('root_cause')
        return 'plugin_failure'


class BackendTests(unittest.TestCase):
    def setUp(self):
        state = random.getstate()
        self.addCleanup(random.setstate, state)
        self.registry = patch.dict('src.backends._backends')
        self.registry.start()
        self.addCleanup(self.registry.stop)
        self.backend = RecordingBackend()
        register_backend(self.backend)
        random.seed(42)

    def test_registry_validation(self):
        self.assertIn('recording', backend_names())
        self.assertIs(get_backend('recording'), self.backend)
        with self.assertRaisesRegex(ValueError, 'already registered'):
            register_backend(RecordingBackend())
        with self.assertRaisesRegex(ValueError, 'Unknown backend'):
            get_backend('missing-backend')
        with self.assertRaises(TypeError):
            register_backend(object())
        invalid = RecordingBackend()
        invalid.name = 'invalid_name'
        with self.assertRaises(ValueError):
            register_backend(invalid)
    def test_registered_generation_mutation_execution_and_diagnostics(self):
        config = Config(coverage_probe_prob=0, dtype_mutate_prob=1, dim_range=(1,64))
        program = ProgramGenerator(config, 'recording').generate()
        changed = Mutator(config, 'recording').mutate(program)
        from src.workflow.generator.region_generator import gemm_step_op_program, step_op_noise_trap
        dtype_branch_skipped = gemm_step_op_program(program) and program.spec.dtype != DataType.FLOAT32
        if dtype_branch_skipped:
            # The TF32-noise trap guard blocks the fp32 flip for gemm+step
            # programs; mutation falls through to a structure-changing branch,
            # which must still never introduce a trap program.
            self.assertFalse(step_op_noise_trap(changed))
        else:
            self.assertNotEqual(program.spec.dtype, changed.spec.dtype)
        oracle = Oracle(config, 'recording')
        self.assertIsNone(oracle.test(changed))
        self.backend.fail = True
        report = oracle.test(changed)
        self.assertEqual(report.bug_type, BugType.RUNTIME_CRASH)
        self.assertEqual(report.root_cause, 'plugin_failure')
        expected = {'sample', 'validate', 'emit', 'command', 'environment', 'error', 'root_cause'}
        if not dtype_branch_skipped:
            expected.add('repair')
        self.assertTrue(expected <= set(self.backend.events))

    def test_probe_policy_dispatch(self):
        config = Config(coverage_probe_prob=1)
        program = ProgramGenerator(config, 'recording').generate()
        Mutator(config, 'recording').mutate(program)
        self.assertIn('probe', self.backend.events)
        self.assertIn('mutate_probe', self.backend.events)

    def test_derived_backend_keeps_registered_identity_during_lowering(self):
        class DerivedBackend(TritonBackend):
            name = 'derived-triton'

            def typed_prepare(self, program, name, argv, options=None):
                return super().typed_prepare(program, name, argv, options) + '\n    # derived backend hook'

        register_backend(DerivedBackend())
        config = Config(coverage_probe_prob=0, region_typed_prob=1, region_int8_prob=0)
        program = ProgramGenerator(config, 'derived-triton').generate()
        code = Oracle(config, 'derived-triton')._emit_code(program)
        self.assertIn('# derived backend hook', code)
        self.assertNotIn('# derived backend hook', Oracle(config, 'triton')._emit_code(program))

    def test_campaign_and_resume_with_registered_backend(self):
        from src.workflow.fuzzer import TileSmith
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            config = Config(backends=['recording'], coverage_probe_prob=0,
                            output_dir=directory, seed=42, dim_range=(1,64))
            campaign = TileSmith(config)
            campaign.run(2, verbose=False)
            resumed = TileSmith(config, resume_dir=str(campaign.output_dir))
            resumed.run(1, verbose=False)
            summary = json.loads((campaign.output_dir/'summary.json').read_text())
            self.assertEqual(summary['backend'], 'recording')
            self.assertEqual(summary['total_tested'], 3)
            self.assertEqual(summary['bugs_total'], 0)

    def test_cli_loads_external_backend_module(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, 'external_backend.py').write_text(
                'from test_backends import RecordingBackend\n'
                'from src.backends import register_backend\n'
                'register_backend(RecordingBackend())\n')
            environment = {**os.environ, 'PYTHONPATH': os.pathsep.join((directory, str(root/'tests'), str(root)))}
            command = [sys.executable, '-B', 'main.py', '--backend-plugin', 'external_backend',
                       '--backend', 'recording', '--probe-prob', '0', '--seed', '42']
            dump = subprocess.run(command+['--dump'], cwd=root, env=environment,
                                  capture_output=True, text=True, timeout=30)
            self.assertEqual(dump.returncode, 0, dump.stderr)
            self.assertIn('PLUGIN PASSED', dump.stdout)
            run = subprocess.run(command+['-n','1','-o',directory,'-q'], cwd=root, env=environment,
                                 capture_output=True, text=True, timeout=30)
            self.assertEqual(run.returncode, 0, run.stderr)
            summary = json.loads(next(Path(directory).glob('*/summary.json')).read_text())
            self.assertEqual(summary['backend'], 'recording')
            self.assertEqual(summary['bugs_total'], 0)

    def test_historical_diagnostic_precedence(self):
        cases = [('WRONG RESULT: scratch canary modified', 'scratch_out_of_bounds'),
                 ('WRONG RESULT: output canary modified', 'output_out_of_bounds'),
                 ('internalerror codegen triton.compiler', 'tilelang_codegen_error'),
                 ('dtype mismatch has no attribute', 'dtype_mismatch'),
                 ('no available layout', 'layout_inference'),
                 ('cp_async assertion failed', 'ptx_async_boundary'),
                 ('Triton CompilationError', 'triton_compile_error'),
                 ('CUDA error: out of memory', 'gpu_oom'),
                 ('Process terminated by signal 11 (SIGSEGV)', 'segfault')]
        for message, expected in cases:
            for backend in (None, 'tilelang', 'triton'):
                report = BugReport(BugType.COMPILE_CRASH, message)
                report.classify_root_cause(backend)
                self.assertEqual(report.root_cause, expected)


if __name__ == '__main__':
    unittest.main()
