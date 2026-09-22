"""Versioned, typed exploration IR with explicit memory and region interfaces.

The old Region v1-v4 domain stays unchanged. This domain has concrete scalar,
vector and matrix shapes, multiple results, and block-private aliased buffers.
It is shared semantics, not a spelling of either target language.
"""
from dataclasses import dataclass, field, asdict
import math
import keyword


DTYPES = ('float16', 'float32', 'int32', 'int8', 'bool')


@dataclass(frozen=True)
class TensorType:
    dtype: str
    shape: tuple = ()

    def __post_init__(self):
        object.__setattr__(self, 'shape', tuple(self.shape))
        if self.dtype not in DTYPES or len(self.shape) > 2 or any(
                type(n) is not int or n < 1 or n > 64 or n & (n - 1) for n in self.shape):
            raise ValueError('Unsupported extended tensor type')

    @property
    def size(self):
        return math.prod(self.shape)


@dataclass
class Value:
    name: str
    type: TensorType


@dataclass
class Node:
    op: str
    results: list[Value] = field(default_factory=list)
    operands: list[str] = field(default_factory=list)
    attrs: dict = field(default_factory=dict)
    regions: list = field(default_factory=list)


@dataclass
class Block:
    arguments: list[Value] = field(default_factory=list)
    operations: list[Node] = field(default_factory=list)
    returns: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data):
        def value(v):
            return Value(v['name'], TensorType(**v['type']))
        return cls([value(v) for v in data['arguments']], [
            Node(n['op'], [value(v) for v in n['results']], n['operands'], n['attrs'],
                 [cls.from_dict(r) for r in n['regions']]) for n in data['operations']], data['returns'])


@dataclass
class Helper:
    name: str
    body: Block


@dataclass
class Buffer:
    name: str
    dtype: str
    size: int
    role: str = 'input'
    base: str | None = None
    offset: int = 0
    stride: int = 1


def walk(block):
    for node in block.operations:
        yield node
        for region in node.regions:
            yield from walk(region)


@dataclass
class ExtendedProgram:
    body: Block
    buffers: list[Buffer]
    functions: list[Helper] = field(default_factory=list)
    observations: list[str] = field(default_factory=list)
    blocks: int = 2
    input_pattern: str = 'integer'
    # Execute the same compiled signature at several runtime bounds, including 0.
    runtime_cases: list[tuple] = field(default_factory=lambda: [(0, 1), (1, 15), (3, 31)])
    configuration_pair: bool = True
    observation_pair: bool = True
    pass_config_pair: bool = True
    fast_math_pair: bool = False
    precision_pair: bool = True
    identity_pair: bool = True
    family: str = 'mixed'

    def to_dict(self):
        return {'type': 'extended', 'version': 1, **asdict(self)}

    @classmethod
    def from_dict(cls, data):
        if data.get('version') != 1:
            raise ValueError('Unsupported extended IR version')
        # data.get with defaults keeps programs saved before the pass
        # configuration sweep restorable.
        fields = {k: data.get(k, default) for k, default in (
            ('observations', []), ('blocks', 2), ('input_pattern', 'integer'),
            ('runtime_cases', [(0, 1), (1, 15), (3, 31)]), ('configuration_pair', True),
            ('observation_pair', True), ('pass_config_pair', True),
            ('fast_math_pair', False), ('precision_pair', True), ('identity_pair', True),
            ('family', 'mixed'))}
        fields['runtime_cases'] = [tuple(c) for c in fields['runtime_cases']]
        program = cls(Block.from_dict(data['body']), [Buffer(**b) for b in data['buffers']],
                      [Helper(f['name'], Block.from_dict(f['body'])) for f in data['functions']], **fields)
        program.validate()
        return program

    @property
    def params_dict(self):
        return {'extended_program': self.to_dict()}

    def all_operations(self):
        for fn in self.functions:
            yield from walk(fn.body)
        yield from walk(self.body)

    def validate(self):
        return analyze(self)


def broadcast_shape(a, b):
    a, b = tuple(a), tuple(b)
    result = []
    for x, y in zip((1,) * (max(len(a), len(b)) - len(a)) + a,
                    (1,) * (max(len(a), len(b)) - len(b)) + b):
        if x != y and x != 1 and y != 1:
            raise ValueError('Incompatible broadcast shapes')
        result.append(max(x, y))
    return tuple(result)


def analyze(program):
    """Validate dominance, signatures, effects and unique-address writes.

    Returns scoped types and the set of statically observable operation IDs.
    Stores are observable because the oracle compares all scratch contents.
    Memory bounds are also guarded by both lowerings, independently of masks.
    """
    if type(program.blocks) is not int or not 1 <= program.blocks <= 8:
        raise ValueError('Invalid block count')
    if program.input_pattern not in ('integer', 'normal', 'boundary', 'special'):
        raise ValueError('Invalid input distribution')
    if not 1 <= len(program.runtime_cases) <= 8 or any(
            len(c) != 2 or any(type(x) is not int for x in c) or not 0 <= c[0] <= 8 or not 0 <= c[1] <= 4096
            for c in program.runtime_cases):
        raise ValueError('Invalid runtime cases')
    if type(program.configuration_pair) is not bool or type(program.observation_pair) is not bool:
        raise ValueError('Invalid execution flags')
    if type(program.pass_config_pair) is not bool or type(program.fast_math_pair) is not bool \
            or type(program.precision_pair) is not bool or type(program.identity_pair) is not bool:
        raise ValueError('Invalid pass configuration flags')
    names, buffers = set(), {}
    def name(n):
        if not isinstance(n, str) or not n.isidentifier() or keyword.iskeyword(n) or n in names:
            raise ValueError('Invalid or duplicate name: ' + str(n))
        names.add(n)
    for buf in program.buffers:
        name(buf.name)
        if buf.dtype not in DTYPES or buf.role not in ('input', 'scratch') or type(buf.size) is not int or not 1 <= buf.size <= 4096:
            raise ValueError('Invalid buffer')
        if type(buf.offset) is not int or type(buf.stride) is not int or buf.offset < 0 or not 1 <= buf.stride <= 4:
            raise ValueError('Invalid buffer view')
        if buf.base is not None:
            base = buffers.get(buf.base)
            if base is None or base.base is not None or (buf.role, buf.dtype) != (base.role, base.dtype):
                raise ValueError('Invalid buffer alias')
            if buf.offset + (buf.size - 1) * buf.stride >= base.size:
                raise ValueError('Buffer view exceeds allocation')
        elif buf.offset or buf.stride != 1:
            raise ValueError('Root buffers must be dense allocations')
        buffers[buf.name] = buf
    if not buffers or len(buffers) > 32 or len(program.functions) > 8:
        raise ValueError('Buffer/function budget exceeded')
    types, signatures, producers, dependencies, effects = {}, {}, {}, {}, set()
    argument_sources, functions, calls = {}, {f.name: f for f in program.functions}, {}
    count = 0

    def visit(block, inherited, scope, depth=0):
        nonlocal count
        if depth > 4:
            raise ValueError('Region depth exceeded')
        visible = dict(inherited)
        unique = {}  # A proven permutation of 0..size-1, for race-free stores.
        for arg in block.arguments:
            name(arg.name)
            visible[arg.name] = arg.type
            types[scope, arg.name] = arg.type
        for node in block.operations:
            count += 1
            if count > 256:
                raise ValueError('Operation budget exceeded')
            if any(v not in visible for v in node.operands):
                raise ValueError('Non-dominating operand: ' + node.op)
            args = [visible[v] for v in node.operands]
            outs = [v.type for v in node.results]
            a, op = node.attrs, node.op
            expected = []
            if op in ('constant', 'index', 'parameter'):
                if args or len(outs) != 1:
                    raise ValueError('Invalid source operation')
                expected = outs
                if op == 'constant' and (not isinstance(a.get('value'), (int, float, bool)) or not math.isfinite(a['value'])):
                    raise ValueError('Invalid constant')
                if op == 'index':
                    if outs[0].dtype != 'int32' or type(a.get('shift', 0)) is not int or a.get('reverse', False) not in (True, False):
                        raise ValueError('Invalid index permutation')
                    unique[node.results[0].name] = outs[0].size
                if op == 'parameter' and (a.get('name') not in ('steps', 'limit', 'block') or outs != [TensorType('int32')]):
                    raise ValueError('Invalid runtime parameter')
            elif op == 'cast':
                if len(args) != 1 or len(outs) != 1:
                    raise ValueError('Invalid cast')
                expected = [TensorType(outs[0].dtype, args[0].shape)]
            elif op in ('reshape', 'broadcast', 'transpose', 'slice'):
                if len(args) != 1 or len(outs) != 1:
                    raise ValueError('Invalid shape operation')
                src, dst = args[0], outs[0]
                if op == 'reshape' and src.size != dst.size:
                    raise ValueError('Reshape changes element count')
                if op == 'broadcast' and broadcast_shape(src.shape, dst.shape) != dst.shape:
                    raise ValueError('Invalid broadcast')
                if op == 'transpose' and (len(src.shape) != 2 or dst.shape != src.shape[::-1]):
                    raise ValueError('Invalid transpose')
                if op == 'slice':
                    offsets = a.get('offsets', [])
                    if len(offsets) != len(src.shape) or len(dst.shape) != len(src.shape) or any(
                            type(off) is not int or off < 0 or off + n > old for off, n, old in zip(offsets, dst.shape, src.shape)):
                        raise ValueError('Invalid slice')
                expected = [TensorType(src.dtype, dst.shape)]
                if op == 'reshape' and node.operands[0] in unique:
                    unique[node.results[0].name] = unique[node.operands[0]]
            elif op == 'flip':
                if len(args) != 1 or len(outs) != 1:
                    raise ValueError('Invalid flip')
                expected = [TensorType(args[0].dtype, args[0].shape)]
                # Reversing a permutation is still a permutation.
                if node.operands[0] in unique:
                    unique[node.results[0].name] = unique[node.operands[0]]
            elif op in ('interleave', 'join'):
                if len(args) != 2 or args[0] != args[1] or len(outs) != 1:
                    raise ValueError('Invalid ' + op)
                if op == 'interleave' and len(args[0].shape) not in (1, 2):
                    raise ValueError('Interleave requires 1-D or 2-D operands')
                if op == 'join' and len(args[0].shape) != 1:
                    raise ValueError('Join requires 1-D operands')
                shape = list(args[0].shape)
                if op == 'interleave':
                    shape[-1] *= 2
                else:
                    shape.append(2)
                expected = [TensorType(args[0].dtype, tuple(shape))]
            elif op == 'split':
                if (len(args) != 1 or len(outs) != 2 or len(args[0].shape) not in (1, 2)
                        or args[0].shape[-1] != 2):
                    raise ValueError('Invalid split')
                expected = [TensorType(args[0].dtype, args[0].shape[:-1]),
                            TensorType(args[0].dtype, args[0].shape[:-1])]
            elif op in ('add', 'sub', 'mul', 'bitand', 'bitxor', 'mod', 'lt', 'eq', 'and', 'or'):
                if len(args) != 2 or args[0].dtype != args[1].dtype:
                    raise ValueError('Binary operation dtype/arity mismatch')
                dtype = args[0].dtype
                if dtype == 'int8':
                    # int8 only enters through casts, shape ops and matmul
                    # operands; i8 arithmetic promotion is unverified.
                    raise ValueError('int8 arithmetic is unsupported')
                if op in ('bitand', 'bitxor', 'mod') and dtype != 'int32':
                    raise ValueError('Integer operation requires int32')
                if op in ('and', 'or') and dtype != 'bool':
                    raise ValueError('Boolean operation requires bool')
                if op in ('add', 'sub', 'mul') and dtype == 'bool':
                    raise ValueError('Boolean arithmetic is unsupported')
                if op == 'mod':
                    # Divisor is sanitized by both backends and the interpreter.
                    pass
                expected = [TensorType('bool' if op in ('lt', 'eq') else dtype,
                                       broadcast_shape(args[0].shape, args[1].shape))]
            elif op == 'select':
                if len(args) != 3 or args[0].dtype != 'bool' or args[1].dtype != args[2].dtype:
                    raise ValueError('Invalid select')
                expected = [TensorType(args[1].dtype, broadcast_shape(args[0].shape, broadcast_shape(args[1].shape, args[2].shape)))]
            elif op == 'reduce':
                if len(args) != 1 or a.get('kind') not in ('sum', 'max', 'min') or a.get('axis') not in range(len(args[0].shape)) or args[0].dtype in ('bool', 'int8'):
                    raise ValueError('Invalid reduction')
                shape = list(args[0].shape)
                del shape[a['axis']]
                expected = [TensorType('float32' if args[0].dtype.startswith('float') else 'int32', shape)]
            elif op == 'fma':
                if (len(args) != 3 or args[0].dtype != args[1].dtype or args[1].dtype != args[2].dtype
                        or args[0].dtype not in ('float16', 'float32')
                        or any(t.shape != () for t in args)):
                    raise ValueError('Invalid fma')
                expected = [TensorType(args[0].dtype)]
            elif op in ('atomic_add', 'atomic_max', 'atomic_min'):
                buf = buffers.get(a.get('buffer'))
                if (buf is None or buf.role != 'scratch' or len(args) != 3
                        or args[0].dtype != 'int32' or args[1] != TensorType('bool', args[0].shape)
                        or args[2] != TensorType(buf.dtype, args[0].shape)
                        or buf.dtype not in ('int32', 'float16', 'float32') or len(outs) != 0):
                    raise ValueError('Invalid atomic operation')
                # The race is the point: a commutative reduction over possibly
                # duplicate addresses, so indices are deliberately not unique.
            elif op == 'matmul':
                # The accumulator may be float16 (precision sweep); the matmul
                # result then carries the accumulation dtype. int8 operands
                # accumulate in int32 and require K >= 32 for the pipelined
                # shared-memory s8 path.
                if len(args) != 3 or any(len(t.shape) != 2 for t in args) or args[0].dtype != args[1].dtype:
                    raise ValueError('Invalid matmul operands')
                if args[0].dtype == 'int8':
                    if (args[2].dtype != 'int32' or args[0].shape[1] != args[1].shape[0]
                            or args[2].shape != (args[0].shape[0], args[1].shape[1])
                            or any(n < 16 for t in args[:2] for n in t.shape) or args[0].shape[1] < 32):
                        raise ValueError('Invalid int8 matmul')
                elif (args[0].dtype not in ('float16', 'float32') or args[2].dtype not in ('float16', 'float32')
                        or args[0].shape[1] != args[1].shape[0] or args[2].shape != (args[0].shape[0], args[1].shape[1])
                        or any(n < 16 for t in args for n in t.shape)):
                    raise ValueError('Invalid matmul operands/accumulator')
                expected = [args[2]]
            elif op in ('load', 'store'):
                buf = buffers.get(a.get('buffer'))
                if buf is None or len(args) != (2 if op == 'load' else 3) or args[0].dtype != 'int32' or args[1] != TensorType('bool', args[0].shape):
                    raise ValueError('Invalid indexed memory operation')
                if op == 'store':
                    if buf.role != 'scratch' or args[2] != TensorType(buf.dtype, args[0].shape):
                        raise ValueError('Invalid store type or destination')
                    if node.operands[0] not in unique:
                        raise ValueError('Store indices must be a proven unique permutation')
                expected = [TensorType(buf.dtype, args[0].shape)] if op == 'load' else []
            elif op == 'call':
                signature = signatures.get(a.get('callee'))
                if signature is None or args != signature[0]:
                    raise ValueError('Invalid call signature, forward or recursive call')
                expected = signature[1]
            elif op in ('for', 'while', 'if'):
                if not args or args[0] != TensorType('bool' if op == 'if' else 'int32'):
                    raise ValueError('Control requires a scalar predicate/bound')
                if len(node.regions) != (2 if op == 'if' else 1):
                    raise ValueError('Invalid region count')
                if op != 'if' and (type(a.get('max_steps')) is not int or not 0 <= a['max_steps'] <= 8):
                    raise ValueError('Loops must have a finite upper bound')
                carried = args[1:]
                for child in node.regions:
                    expected_args = carried if op == 'if' else [TensorType('int32')] + carried
                    if [v.type for v in child.arguments] != expected_args or visit(child, visible, scope, depth + 1) != carried:
                        raise ValueError('Region argument/yield signature mismatch')
                expected = carried
            elif op == 'barrier':
                if args:
                    raise ValueError('Barrier has no operands')
            else:
                raise ValueError('Unknown extended operation: ' + op)
            if op not in ('for', 'while', 'if') and node.regions:
                raise ValueError('Unexpected nested region')
            if expected != outs:
                raise ValueError(f'Incorrect result types for {op}: {outs} != {expected}')
            ident = (scope, id(node))
            dependencies[ident] = {(scope, v) for v in node.operands}
            if op in ('for', 'while', 'if'):
                for child in node.regions:
                    dependencies[ident].update((scope, v) for v in child.returns)
                    arguments = child.arguments if op == 'if' else child.arguments[1:]
                    for arg, source in zip(arguments, node.operands[1:]):
                        argument_sources.setdefault((scope, arg.name), set()).add((scope, source))
                    if op != 'if':
                        argument_sources[scope, child.arguments[0].name] = {(scope, node.operands[0])}
                        for arg, source in zip(arguments, child.returns):
                            argument_sources[scope, arg.name].add((scope, source))
            elif op == 'call':
                callee = a['callee']
                calls.setdefault(callee, []).append(ident)
                dependencies[ident].update((callee, v) for v in functions[callee].body.returns)
                for arg, source in zip(functions[callee].body.arguments, node.operands):
                    argument_sources.setdefault((callee, arg.name), set()).add((scope, source))
            if op in ('store', 'barrier', 'atomic_add', 'atomic_max', 'atomic_min'):
                effects.add(ident)
            for result in node.results:
                name(result.name)
                visible[result.name] = result.type
                types[scope, result.name] = result.type
                producers[scope, result.name] = ident
        if any(v not in visible for v in block.returns):
            raise ValueError('Return value is not visible')
        return [visible[v] for v in block.returns]

    for fn in program.functions:
        name(fn.name)
        signatures[fn.name] = ([v.type for v in fn.body.arguments], visit(fn.body, {}, fn.name))
    if program.body.arguments or not program.body.returns:
        raise ValueError('Entry must have outputs and no SSA arguments')
    visit(program.body, {}, 'main')
    if len(set(program.observations)) != len(program.observations) or len(program.observations) > 16:
        raise ValueError('Invalid observation set')
    top = {v.name for n in program.body.operations for v in n.results}
    if any(v not in top for v in program.observations):
        raise ValueError('Observation must dominate function exit')
    # Mark only reachable effects and values that can reach a checked output.
    # Call sites share a conservative argument union; this is potential source
    # liveness, not proof that a compiler retained an operation.
    reachable = {'main'}
    while True:
        expanded = reachable | {fn for fn, sites in calls.items() if any(s[0] in reachable for s in sites)}
        if expanded == reachable:
            break
        reachable = expanded
    live = {ident for ident in effects if ident[0] in reachable}
    watched = program.observations if program.observation_pair else []
    pending = [('main', v) for v in program.body.returns + watched]
    # Keep control predicates and loop bounds for observable nested effects.
    for scope, block in [('main', program.body)] + [(f.name, f.body) for f in program.functions if f.name in reachable]:
        for node in walk(block):
            if node.regions and any((scope, id(n)) in live for child in node.regions for n in walk(child)):
                live.add((scope, id(node)))
    for fn in reachable - {'main'}:
        if any(scope == fn for scope, _ in live):
            live.update(site for site in calls[fn] if site[0] in reachable)
    for ident in live:
        pending.extend(dependencies[ident])
    seen = set()
    while pending:
        value = pending.pop()
        if value in seen:
            continue
        seen.add(value)
        pending.extend(argument_sources.get(value, ()))
        if value in producers:
            ident = producers[value]
            live.add(ident)
            pending.extend(dependencies[ident])
    return types, live
