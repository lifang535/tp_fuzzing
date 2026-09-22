"""Extension points between the campaign and a DSL implementation.

The shared Region IR defines logical semantics. Backends own accepted target
parameters, lowering, executable harnesses and process configuration. Custom
generators/mutators retain the shared IR and the campaign's state protocol;
introducing a new IR format also requires serialization and feedback support.
"""
from abc import ABC, abstractmethod
import sys


class Backend(ABC):
    name = ''
    supports_extended = False

    def extended_variants(self, program):
        raise NotImplementedError(f'{self.name} does not support extended programs')

    def extended_compile_source(self, entries, program):
        raise NotImplementedError(f'{self.name} does not support extended compilation')

    def make_generator(self, config):
        from src.workflow.generator import ProgramGenerator
        return ProgramGenerator(config, self.name)

    def make_mutator(self, config):
        from src.workflow.mutator import Mutator
        return Mutator(config, self.name)

    @abstractmethod
    def make_emitter(self, config=None):
        """Return an object with emit(program), supporting this backend's IR."""


    def validate_program(self, program):
        """Validate the shared IR, then optionally add DSL-specific constraints."""
        program.validate()

    def sample_region_spec(self, generator, body, initial, functions=(), dtype=None):
        raise NotImplementedError(f'{self.name} does not support shared region generation')

    def dtype_parameters_valid(self, spec):
        raise NotImplementedError(f'{self.name} does not support shared region mutation')

    def generate_probe(self, config):
        raise NotImplementedError(f'{self.name} does not support directed probes; set --probe-prob 0')

    def mutate_probe(self, program, config):
        raise NotImplementedError(f'{self.name} does not support directed probes')


    def execution_command(self, path):
        return [sys.executable, path]

    def execution_options(self, config):
        """Extra subprocess.run options, e.g. env or cwd; inherit defaults."""
        return {}

    def classify_error(self, message):
        from src.workflow.oracle.oracle import BugType
        return BugType.WRONG_RESULT if 'wrong result' in message.lower() else BugType.COMPILE_CRASH

    def classify_root_cause(self, message):
        from .common.diagnostics import classify_root_cause
        return classify_root_cause(message)
