"""Feature-slice programs: one kernel whose focus feature varies richly.

A slice names a bug-dense feature dimension of tile DSLs (dtype conversion,
reduction, scan, GEMM, memory access, atomics, loops). Its parameters are a
flat assignment of discrete knobs, so interaction coverage, quarantine rules
and failure attribution all speak the same feature vocabulary. The kernel
source, inputs and exact reference are derived from (slice, params, backend)
by src/workflow/slices; the IR stores no generated code.
"""
from dataclasses import dataclass, field


@dataclass
class SliceProgram:
    slice: str
    params: dict = field(default_factory=dict)
    backend: str = ''

    def to_dict(self):
        return {'type': 'slice', 'version': 1, 'slice': self.slice,
                'backend': self.backend, 'params': dict(sorted(self.params.items()))}

    @classmethod
    def from_dict(cls, data):
        """Restore a saved program, legalized under the current knob space.

        A knob added since the program was saved takes its default and a
        removed one is dropped, so campaigns written by an earlier slice
        version still resume; the executed program may then differ from the
        saved one, and its signature with it."""
        if data.get('version') != 1:
            raise ValueError('Unsupported slice IR version')
        from src.workflow.slices import SLICES, make_program
        if data['slice'] not in SLICES:
            raise ValueError('Unknown slice: ' + str(data['slice']))
        return make_program(data['slice'], dict(data['params']), data.get('backend', ''))

    @property
    def params_dict(self):
        return {'slice_program': self.to_dict()}

    def validate(self):
        from src.workflow.slices import validate_program
        validate_program(self)

    def features(self):
        """Every knob assignment, prefixed by the slice name."""
        return {f'slice={self.slice}'} | {f'{self.slice}.{k}={v}' for k, v in self.params.items()}
