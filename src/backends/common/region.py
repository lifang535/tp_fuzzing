"""Logical layout/index expressions shared by native lowerings."""
from src.ir.layout import matrix_layout

def _region_layouts(program):
    p = program.spec
    gemm = program.body.operations[0].kind == 'gemm'
    a_layout = program.execution.input_layout_a if program.execution else 'contiguous'
    b_layout = program.execution.input_layout_b if program.execution else 'contiguous'
    physical = a_layout != 'contiguous' or b_layout != 'contiguous'
    return (physical, matrix_layout(p.M, p.K if gemm else p.N, a_layout),
            matrix_layout(p.K, p.N, b_layout))


def _layout_sweep_pairs(program):
    """The input layout pairs the checked/typed emitters compile per input
    case: the primary pair plus (for physical layout-sweep programs) one
    alternate. Empty when the sweep is off; the oracle uses this to scale
    the per-program timeout with the compiled kernel count."""
    execution = program.execution
    if execution is None or not execution.layout_sweep:
        return []
    physical, _, _ = _region_layouts(program)
    if not physical:
        return []
    gemm = program.body.operations[0].kind == 'gemm'
    primary = (execution.input_layout_a, execution.input_layout_b)
    alternate = (('offset', 'strided') if gemm
                 else ('strided' if execution.input_layout_a == 'offset' else 'offset', 'contiguous'))
    return [primary] + ([alternate] if alternate != primary else [])

def _index_expression(axis, iteration):
    return {'row': 'by', 'column': 'bx', 'checkerboard': '(by + bx)', 'iteration': iteration}[axis]

def _predicate_expression(attrs, iteration):
    return f'{_index_expression(attrs.get("predicate", "row"), iteration)} % {attrs.get("modulus", 2)} == {attrs["parity"]}'
