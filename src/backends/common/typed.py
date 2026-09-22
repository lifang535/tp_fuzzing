"""Scoped value types and buffer provenance shared by typed lowerings."""
from src.ir.region_types import infer_program
from src.ir.layout import matrix_layout

class TypedLoweringBase:

    def __init__(self, program, backend, name, suffix=''):
        (self.program, self.p, self.backend) = (program, program.spec, backend)
        (self.name, self.suffix) = (name, suffix)
        (self.types, self.aliases, self.slots) = infer_program(program)
        self.buffers = {slot: f'scratch_{i}' for (i, slot) in enumerate(self.slots)}
        self.context = ['A', 'B'] + list(self.buffers.values())
        self.lines = []

    def add(self, indent, text):
        self.lines.append('    ' * indent + text)

    def ty(self, scope, value):
        return self.types[scope, value]

    def index(self, axis, iteration):
        return {'row': 'by', 'column': 'bx', 'checkerboard': '(by + bx)', 'iteration': iteration}[axis]

    def input_layout(self, source):
        (p, e) = (self.p, self.program.execution)
        gemm = self.program.body.operations[0].kind == 'gemm'
        (rows, cols) = (p.M, p.K if gemm else p.N) if source == 'A' else (p.K, p.N)
        layout = e.input_layout_a if source == 'A' else e.input_layout_b
        return (rows, cols, matrix_layout(rows, cols, layout))
