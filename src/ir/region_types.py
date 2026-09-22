"""Type/shape inference and buffer provenance for region v4.

Shapes are relative to the block: tile=(M,N), row=(M,1), column=(1,N),
scalar=(1,1). Buffers are mutable, block-private global scratch. They may be
captured by nested regions but cannot escape through yield or function calls.
"""
from dataclasses import dataclass, asdict


@dataclass(frozen=True)
class ValueType:
    dtype: str = 'float32'
    shape: str = 'tile'
    kind: str = 'tensor'

    def __post_init__(self):
        if self.dtype not in ('float16', 'float32') or self.shape not in ('tile', 'row', 'column', 'scalar') or self.kind not in ('tensor', 'buffer'):
            raise ValueError('Unsupported region value type')

    def dimensions(self, spec):
        return {'tile': (spec.block_M, spec.block_N), 'row': (spec.block_M, 1),
                'column': (1, spec.block_N), 'scalar': (1, 1)}[self.shape]

    def to_dict(self):
        return asdict(self)


FULL = ValueType()


def argument_types(function):
    types = ([ValueType(**t) for t in function.argument_types] if function.argument_types is not None
             else [FULL] * len(function.body.arguments))
    if len(types) != len(function.body.arguments) or any(t.kind != 'tensor' for t in types):
        raise ValueError('Function arguments require matching tensor types')
    return types


def result_type(kind, args, attrs):
    """Infer a leaf operation. No implicit broadcasting or mixed-dtype arithmetic."""
    if kind in ('load', 'gemm', 'to_tile'):
        if kind == 'to_tile' and args[0].kind != 'tensor':
            raise ValueError('to_tile requires a tensor')
        return FULL
    if kind == 'load_input':
        if attrs.get('source') not in ('A', 'B'):
            raise ValueError('Invalid input source')
        for name in ('row_offset', 'col_offset'):
            if type(attrs.get(name, 0)) is not int or not 0 <= attrs.get(name, 0) <= 3:
                raise ValueError('Invalid input offset')
        return ValueType(attrs.get('dtype', 'float32'))
    if kind == 'store_tile':
        if args[0].kind != 'tensor':
            raise ValueError('store_tile requires a tensor')
        return ValueType(args[0].dtype, args[0].shape, 'buffer')
    if kind in ('load_tile', 'write_tile'):
        buffer = args[0]
        if buffer.kind != 'buffer':
            raise ValueError(kind + ' requires a buffer')
        tensor = ValueType(buffer.dtype, buffer.shape)
        if kind == 'write_tile' and args[1] != tensor:
            raise ValueError('Buffer write type mismatch')
        return tensor if kind == 'load_tile' else buffer
    if not args or any(t.kind != 'tensor' for t in args):
        raise ValueError('Arithmetic requires tensor operands')
    if any(t != args[0] for t in args):
        raise ValueError('Operand type/shape mismatch: ' + kind)
    t = args[0]
    if kind == 'cast':
        return ValueType(attrs.get('dtype'), t.shape)
    if kind == 'reduce_tile':
        if attrs.get('axis') not in (0, 1) or attrs.get('reduction') not in ('sum', 'max', 'min'):
            raise ValueError('Invalid compact reduction')
        shape = ('row' if t.shape in ('tile', 'row') else 'scalar') if attrs['axis'] == 1 else ('column' if t.shape in ('tile', 'column') else 'scalar')
        return ValueType('float32', shape)
    if kind == 'broadcast_tile':
        return ValueType(t.dtype)
    if kind == 'tile_transpose':
        return ValueType(t.dtype, {'tile':'tile', 'row':'column', 'column':'row', 'scalar':'scalar'}[t.shape])
    if kind.startswith('row_'):
        return ValueType('float32', t.shape)
    return t


def infer_program(program):
    """Return scoped value types, buffer aliases, and physical scratch slots."""
    types, aliases, slots, signatures = {}, {}, {}, {}

    def visit(region, inherited, scope, arg_types):
        visible = dict(inherited)
        visible.update(zip(region.arguments, arg_types))
        for name, t in zip(region.arguments, arg_types):
            types[scope, name] = t
        for op in region.operations:
            args = [visible[v] for v in op.operands]
            if op.kind == 'call':
                expected, out = signatures[op.attrs['callee']]
                if args != expected:
                    raise ValueError('Call argument type mismatch')
            elif op.kind in ('if', 'for'):
                if args[0].kind != 'tensor':
                    raise ValueError('Control flow requires a tensor carry')
                for child in op.regions:
                    if visit(child, visible, scope, [args[0]]) != args[0]:
                        raise ValueError('Region yield type must match carried/input type')
                out = args[0]
            else:
                out = result_type(op.kind, args, op.attrs)
            if op.kind == 'load_input' and op.attrs['source'] == 'B' and program.body.operations[0].kind != 'gemm':
                raise ValueError('Load-entry programs only expose input A')
            if op.kind == 'store_tile':
                slot = (scope, op.result)
                slots[slot] = out
                aliases[scope, op.result] = slot
            elif op.kind == 'write_tile':
                aliases[scope, op.result] = aliases[scope, op.operands[0]]
            types[scope, op.result] = out
            visible[op.result] = out
        out = visible[region.yield_value]
        if out.kind != 'tensor':
            raise ValueError('Buffers cannot escape through region yield')
        return out

    for fn in program.functions:
        args = argument_types(fn)
        signatures[fn.name] = (args, visit(fn.body, {}, fn.name, args))
    if visit(program.body, {}, 'main', []) != FULL:
        raise ValueError('Entry output must be a full float32 tile; insert to_tile')
    return types, aliases, slots


def scratch_bytes(program):
    _, _, slots = infer_program(program)
    p = program.spec
    blocks = ((p.M + p.block_M - 1) // p.block_M) * ((p.N + p.block_N - 1) // p.block_N)
    return blocks * sum((t.dimensions(p)[0] * t.dimensions(p)[1] + 32) * (2 if t.dtype == 'float16' else 4)
                        for t in slots.values())
