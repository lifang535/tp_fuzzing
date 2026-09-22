"""Shared dispatch for the two executable program representations."""
from src.config import DEFAULT_CONFIG
from src.ir.region import RegionProgram
from src.ir.extended import ExtendedProgram


class ProgramEmitter:
    def __init__(self, config, backend):
        self.config = config or DEFAULT_CONFIG
        self.backend = backend

    def emit(self, program):
        if isinstance(program, RegionProgram):
            from .region_emitter import emit_region
            return emit_region(program, self.backend, self.config)
        if isinstance(program, ExtendedProgram):
            from .extended_emitter import emit_extended
            return emit_extended(program, self.backend, self.config)
        raise TypeError(f'Unsupported program: {type(program).__name__}')
