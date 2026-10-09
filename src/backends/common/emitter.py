"""Shared dispatch for the executable program representations."""
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
        from src.ir.slice import SliceProgram
        if isinstance(program, SliceProgram):
            if program.backend != self.backend:
                raise ValueError(f'Slice program for {program.backend!r} emitted for {self.backend!r}')
            from src.workflow.slices import emit_slice
            return emit_slice(program, self.config)
        raise TypeError(f'Unsupported program: {type(program).__name__}')
