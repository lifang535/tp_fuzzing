"""Versioned, typed exploration IR with explicit memory and region interfaces.

The old Region v1-v4 domain stays unchanged. This domain has concrete scalar,
vector and matrix shapes, multiple results, and block-private aliased buffers.
It is shared semantics, not a spelling of either target language.
"""
from dataclasses import dataclass, field, asdict
import math
import keyword


DTYPES = ('float16', 'float32', 'int32', 'int8', 'bool')

# Shared contract for Extended math operations supported by both lowerings
# and the reference interpreter.
FLOAT_UNARY_OPS = ('neg', 'abs', 'sqrt', 'exp', 'log', 'log2', 'exp2',
                   'rsqrt', 'sin', 'cos', 'floor', 'ceil', 'tanh', 'erf', 'round')
FLOAT_BINARY_OPS = ('minimum', 'maximum', 'div')
# Target calls whose attributes choose an axis, a direction, an order or a
# count; absent attributes keep the historical 1-D spelling.
TARGET_ATTRIBUTE_OPS = frozenset(('scan_sum', 'scan_product', 'scan_max', 'sort', 'softmax',
                                  'topk', 'argmax', 'argmin', 'xor_sum', 'reduce_abssum',
                                  'reduce_absmax', 'reduce_bitand', 'reduce_bitor',
                                  'reduce_bitxor'))


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


def target_axis(attrs, ty):
    """The normalized axis of a target scan, softmax or reduction.

    An absent axis keeps old 1-D programs valid: the minor dimension.
    """
    axis = attrs.get('axis', -1)
    if type(axis) is not int or not -len(ty.shape) <= axis < len(ty.shape):
        raise ValueError('Invalid target operation axis')
    return axis % len(ty.shape)


def spelled_axis(attrs, ty):
    """The validated axis as the program spells it, possibly negative."""
    return attrs.get('axis', target_axis(attrs, ty))


def reduced_shape(ty, axis):
    return ty.shape[:axis] + ty.shape[axis + 1:]


ATOMIC_OPS = ('atomic_add', 'atomic_max', 'atomic_min', 'atomic_and', 'atomic_or', 'atomic_xor')
CONTROL_OPS = ('for', 'while', 'if')
# Operations with a side effect the oracle observes (scratch contents).
EFFECT_OPS = ('store', 'barrier') + ATOMIC_OPS


def analyze(program):
    """Validate dominance, signatures, effects and unique-address writes.

    Returns scoped types and the set of statically observable operation IDs.
    Stores are observable because the oracle compares all scratch contents.
    Memory bounds are also guarded by both lowerings, independently of masks.
    """
    analysis = _Analysis(program)
    for fn in program.functions:
        analysis.name(fn.name)
        analysis.signatures[fn.name] = ([v.type for v in fn.body.arguments], analysis.visit(fn.body, {}, fn.name))
    if program.body.arguments or not program.body.returns:
        raise ValueError('Entry must have outputs and no SSA arguments')
    analysis.visit(program.body, {}, 'main')
    if len(set(program.observations)) != len(program.observations) or len(program.observations) > 16:
        raise ValueError('Invalid observation set')
    top = {v.name for n in program.body.operations for v in n.results}
    if any(v not in top for v in program.observations):
        raise ValueError('Observation must dominate function exit')
    return analysis.types, analysis.live()


class _Analysis:
    """State of one analyze() run: names, buffers, types and the dataflow graph."""

    def __init__(self, program):
        self.program = program
        self.names, self.buffers = set(), {}
        self.types, self.signatures, self.producers, self.dependencies = {}, {}, {}, {}
        self.effects, self.argument_sources, self.calls = set(), {}, {}
        self.functions = {f.name: f for f in program.functions}
        self.count = 0
        self._check_settings()
        self._declare_buffers()

    def name(self, n):
        if not isinstance(n, str) or not n.isidentifier() or keyword.iskeyword(n) or n in self.names:
            raise ValueError('Invalid or duplicate name: ' + str(n))
        self.names.add(n)

    def _check_settings(self):
        program = self.program
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

    def _declare_buffers(self):
        for buf in self.program.buffers:
            self.name(buf.name)
            if buf.dtype not in DTYPES or buf.role not in ('input', 'scratch') or type(buf.size) is not int \
                    or not 1 <= buf.size <= 4096:
                raise ValueError('Invalid buffer')
            if type(buf.offset) is not int or type(buf.stride) is not int or buf.offset < 0 or not 1 <= buf.stride <= 4:
                raise ValueError('Invalid buffer view')
            if buf.base is not None:
                base = self.buffers.get(buf.base)
                if base is None or base.base is not None or (buf.role, buf.dtype) != (base.role, base.dtype):
                    raise ValueError('Invalid buffer alias')
                if buf.offset + (buf.size - 1) * buf.stride >= base.size:
                    raise ValueError('Buffer view exceeds allocation')
            elif buf.offset or buf.stride != 1:
                raise ValueError('Root buffers must be dense allocations')
            self.buffers[buf.name] = buf
        if not self.buffers or len(self.buffers) > 32 or len(self.program.functions) > 8:
            raise ValueError('Buffer/function budget exceeded')

    def visit(self, block, inherited, scope, depth=0):
        """Check a block; returns the types it yields."""
        if depth > 4:
            raise ValueError('Region depth exceeded')
        visible = dict(inherited)
        unique = {}  # A proven permutation of 0..size-1, for race-free stores.
        for arg in block.arguments:
            self.name(arg.name)
            visible[arg.name] = arg.type
            self.types[scope, arg.name] = arg.type
        for node in block.operations:
            self.count += 1
            if self.count > 256:
                raise ValueError('Operation budget exceeded')
            if any(v not in visible for v in node.operands):
                raise ValueError('Non-dominating operand: ' + node.op)
            op = node.op
            step = _Step(node, [visible[v] for v in node.operands], unique, visible, scope, depth)
            rule = _RULES.get(op)
            if rule is None:
                raise ValueError('Unknown extended operation: ' + op)
            expected = rule(self, step)
            if op not in CONTROL_OPS and node.regions:
                raise ValueError('Unexpected nested region')
            if expected != step.outs:
                raise ValueError(f'Incorrect result types for {op}: {step.outs} != {expected}')
            self._record(node, scope)
            for result in node.results:
                self.name(result.name)
                visible[result.name] = result.type
                self.types[scope, result.name] = result.type
                self.producers[scope, result.name] = (scope, id(node))
        if any(v not in visible for v in block.returns):
            raise ValueError('Return value is not visible')
        return [visible[v] for v in block.returns]

    def _record(self, node, scope):
        """Dataflow edges of one operation: operands, region yields, call returns."""
        ident = (scope, id(node))
        self.dependencies[ident] = {(scope, v) for v in node.operands}
        if node.op in CONTROL_OPS:
            for child in node.regions:
                self.dependencies[ident].update((scope, v) for v in child.returns)
                arguments = child.arguments if node.op == 'if' else child.arguments[1:]
                for arg, source in zip(arguments, node.operands[1:]):
                    self.argument_sources.setdefault((scope, arg.name), set()).add((scope, source))
                if node.op != 'if':
                    self.argument_sources[scope, child.arguments[0].name] = {(scope, node.operands[0])}
                    for arg, source in zip(arguments, child.returns):
                        self.argument_sources[scope, arg.name].add((scope, source))
        elif node.op == 'call':
            callee = node.attrs['callee']
            self.calls.setdefault(callee, []).append(ident)
            self.dependencies[ident].update((callee, v) for v in self.functions[callee].body.returns)
            for arg, source in zip(self.functions[callee].body.arguments, node.operands):
                self.argument_sources.setdefault((callee, arg.name), set()).add((scope, source))
        if node.op in EFFECT_OPS:
            self.effects.add(ident)

    def live(self):
        """Operations that reach a checked output or an observable effect.

        Only reachable effects and values that can reach a checked output are
        marked. Call sites share a conservative argument union; this is
        potential source liveness, not proof that a compiler retained an
        operation.
        """
        program, calls = self.program, self.calls
        reachable = {'main'}
        while True:
            expanded = reachable | {fn for fn, sites in calls.items() if any(s[0] in reachable for s in sites)}
            if expanded == reachable:
                break
            reachable = expanded
        live = {ident for ident in self.effects if ident[0] in reachable}
        watched = program.observations if program.observation_pair else []
        pending = [('main', v) for v in program.body.returns + watched]
        # Keep control predicates and loop bounds for observable nested effects.
        for scope, block in [('main', program.body)] + [(f.name, f.body) for f in program.functions
                                                         if f.name in reachable]:
            for node in walk(block):
                if node.regions and any((scope, id(n)) in live for child in node.regions for n in walk(child)):
                    live.add((scope, id(node)))
        for fn in reachable - {'main'}:
            if any(scope == fn for scope, _ in live):
                live.update(site for site in calls[fn] if site[0] in reachable)
        for ident in live:
            pending.extend(self.dependencies[ident])
        seen = set()
        while pending:
            value = pending.pop()
            if value in seen:
                continue
            seen.add(value)
            pending.extend(self.argument_sources.get(value, ()))
            if value in self.producers:
                ident = self.producers[value]
                live.add(ident)
                pending.extend(self.dependencies[ident])
        return live

    # Type rules: each checks one operation and returns its expected result types.

    def _source(self, s):
        a, op, outs = s.attrs, s.op, s.outs
        if s.args or len(outs) != 1:
            raise ValueError('Invalid source operation')
        if op == 'constant' and (not isinstance(a.get('value'), (int, float, bool)) or not math.isfinite(a['value'])):
            raise ValueError('Invalid constant')
        if op == 'index':
            if outs[0].dtype != 'int32' or type(a.get('shift', 0)) is not int or a.get('reverse', False) not in (True, False):
                raise ValueError('Invalid index permutation')
            s.unique[s.node.results[0].name] = outs[0].size
        if op == 'parameter' and (a.get('name') not in ('steps', 'limit', 'block') or outs != [TensorType('int32')]):
            raise ValueError('Invalid runtime parameter')
        return outs

    def _cast(self, s):
        if len(s.args) != 1 or len(s.outs) != 1:
            raise ValueError('Invalid cast')
        return [TensorType(s.outs[0].dtype, s.args[0].shape)]

    def _float_unary(self, s):
        if len(s.args) != 1 or s.args[0].dtype != 'float32':
            raise ValueError('Extended math unary requires one float32 operand')
        if s.op == 'round' and s.attrs.get('dtype') not in ('float16', 'float32'):
            raise ValueError('Extended round requires a floating point target dtype')
        return [s.args[0]]

    def _float_binary(self, s):
        args = s.args
        if len(args) != 2 or args[0] != args[1] or args[0].dtype != 'float32':
            raise ValueError('Extended math binary requires matching float32 operands')
        return [args[0]]

    def _shape(self, s):
        if len(s.args) != 1 or len(s.outs) != 1:
            raise ValueError('Invalid shape operation')
        op, src, dst = s.op, s.args[0], s.outs[0]
        if op == 'reshape' and src.size != dst.size:
            raise ValueError('Reshape changes element count')
        if op == 'broadcast' and broadcast_shape(src.shape, dst.shape) != dst.shape:
            raise ValueError('Invalid broadcast')
        if op == 'transpose' and (len(src.shape) != 2 or dst.shape != src.shape[::-1]):
            raise ValueError('Invalid transpose')
        if op == 'slice':
            offsets = s.attrs.get('offsets', [])
            if len(offsets) != len(src.shape) or len(dst.shape) != len(src.shape) or any(
                    type(off) is not int or off < 0 or off + n > old for off, n, old in zip(offsets, dst.shape, src.shape)):
                raise ValueError('Invalid slice')
        if op == 'reshape':
            s.keep_permutation()
        return [TensorType(src.dtype, dst.shape)]

    def _flip(self, s):
        args = s.args
        if len(args) != 1 or len(s.outs) != 1 or not args[0].shape:
            raise ValueError('Invalid flip')
        # Absent axis keeps old programs valid: the minor dimension.
        axis = s.attrs.get('axis', -1)
        if type(axis) is not int or not -len(args[0].shape) <= axis < len(args[0].shape):
            raise ValueError('Invalid flip axis')
        # Reversing a permutation is still a permutation.
        s.keep_permutation()
        return [TensorType(args[0].dtype, args[0].shape)]

    def _pair(self, s):
        args, op = s.args, s.op
        if len(args) != 2 or args[0] != args[1] or len(s.outs) != 1:
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
        return [TensorType(args[0].dtype, tuple(shape))]

    def _split(self, s):
        args = s.args
        if len(args) != 1 or len(s.outs) != 2 or len(args[0].shape) not in (1, 2) or args[0].shape[-1] != 2:
            raise ValueError('Invalid split')
        return [TensorType(args[0].dtype, args[0].shape[:-1]), TensorType(args[0].dtype, args[0].shape[:-1])]

    # Spelling flags: sort/topk reorder only the minor dimension.
    _FLAGS = {'scan_sum': ('reverse',), 'scan_product': ('reverse',), 'scan_max': ('reverse',),
              'sort': ('descending',), 'softmax': ('keep_dims',)}

    def _float_target(self, s):
        args, op, a = s.args, s.op, s.attrs
        if len(args) != 1 or len(args[0].shape) not in (1, 2) or args[0].dtype != 'float32' or len(s.outs) != 1:
            raise ValueError('Invalid scan or sort operand')
        if op.startswith('scan_') or op == 'softmax':
            target_axis(a, args[0])
        for flag in self._FLAGS.get(op, ()):
            if type(a.get(flag, False)) is not bool:
                raise ValueError(f'Invalid {op} {flag} flag')
        return [args[0]]

    def _reduction(self, s, dtype, message, result='int32'):
        args = s.args
        if len(args) != 1 or len(args[0].shape) not in (1, 2) or args[0].dtype != dtype or len(s.outs) != 1:
            raise ValueError(message)
        return [TensorType(result, reduced_shape(args[0], target_axis(s.attrs, args[0])))]

    def _integer_reduction(self, s):
        return self._reduction(s, 'int32', 'Invalid integer reduction operand')

    def _absolute_reduction(self, s):
        return self._reduction(s, 'float32', 'Invalid absolute reduction operand', result='float32')

    def _bitwise_reduction(self, s):
        return self._reduction(s, 'int32', 'Invalid bitwise reduction operand')

    def _topk(self, s):
        args, a = s.args, s.attrs
        k = a.get('k')
        if (len(args) != 1 or len(args[0].shape) not in (1, 2) or args[0].dtype != 'float32'
                or type(k) is not int or k < 1 or k > args[0].shape[-1] or k & (k - 1)
                or type(a.get('descending', True)) is not bool or len(s.outs) != 1):
            raise ValueError('Invalid topk operand or k')
        return [TensorType('float32', args[0].shape[:-1] + (k,))]

    def _gather(self, s):
        args = s.args
        if (len(args) != 2 or len(args[0].shape) != 1 or args[0].dtype != 'float32'
                or args[1] != TensorType('int32', args[0].shape) or len(s.outs) != 1):
            raise ValueError('Invalid gather operands')
        return [args[0]]

    def _histogram(self, s):
        args = s.args
        if len(args) != 1 or len(args[0].shape) != 1 or args[0].dtype != 'int32' or len(s.outs) != 1:
            raise ValueError('Invalid histogram operand')
        return [TensorType('int32', (16,))]

    def _binary(self, s):
        args, op = s.args, s.op
        if len(args) != 2 or args[0].dtype != args[1].dtype:
            raise ValueError('Binary operation dtype/arity mismatch')
        dtype = args[0].dtype
        if dtype == 'int8':
            # int8 only enters through casts, shape ops and matmul operands;
            # i8 arithmetic promotion is unverified.
            raise ValueError('int8 arithmetic is unsupported')
        if op in ('bitand', 'bitxor', 'mod') and dtype != 'int32':
            raise ValueError('Integer operation requires int32')
        if op in ('and', 'or') and dtype != 'bool':
            raise ValueError('Boolean operation requires bool')
        if op in ('add', 'sub', 'mul') and dtype == 'bool':
            raise ValueError('Boolean arithmetic is unsupported')
        # A mod divisor is sanitized by both backends and the interpreter.
        return [TensorType('bool' if op in ('lt', 'eq') else dtype, broadcast_shape(args[0].shape, args[1].shape))]

    def _select(self, s):
        args = s.args
        if len(args) != 3 or args[0].dtype != 'bool' or args[1].dtype != args[2].dtype:
            raise ValueError('Invalid select')
        return [TensorType(args[1].dtype, broadcast_shape(args[0].shape, broadcast_shape(args[1].shape, args[2].shape)))]

    def _reduce(self, s):
        args, a = s.args, s.attrs
        if (len(args) != 1 or a.get('kind') not in ('sum', 'max', 'min') or a.get('axis') not in range(len(args[0].shape))
                or args[0].dtype in ('bool', 'int8')):
            raise ValueError('Invalid reduction')
        shape = list(args[0].shape)
        del shape[a['axis']]
        return [TensorType('float32' if args[0].dtype.startswith('float') else 'int32', shape)]

    def _fma(self, s):
        args = s.args
        if (len(args) != 3 or args[0].dtype != args[1].dtype or args[1].dtype != args[2].dtype
                or args[0].dtype not in ('float16', 'float32') or any(t.shape != () for t in args)):
            raise ValueError('Invalid fma')
        return [TensorType(args[0].dtype)]

    def _atomic(self, s):
        args = s.args
        buf = self.buffers.get(s.attrs.get('buffer'))
        dtypes = ('int32',) if s.op in ('atomic_and', 'atomic_or', 'atomic_xor') else ('int32', 'float16', 'float32')
        if (buf is None or buf.role != 'scratch' or len(args) != 3
                or args[0].dtype != 'int32' or args[1] != TensorType('bool', args[0].shape)
                or args[2] != TensorType(buf.dtype, args[0].shape) or buf.dtype not in dtypes or len(s.outs) != 0):
            raise ValueError('Invalid atomic operation')
        # The race is the point: a commutative reduction over possibly
        # duplicate addresses, so indices are deliberately not unique.
        return []

    def _matmul(self, s):
        # The accumulator may be float16 (precision sweep); the matmul result
        # then carries the accumulation dtype. int8 operands accumulate in
        # int32 and require K >= 32 for the pipelined shared-memory s8 path.
        args = s.args
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
        return [args[2]]

    def _memory(self, s):
        args, op = s.args, s.op
        buf = self.buffers.get(s.attrs.get('buffer'))
        if (buf is None or len(args) != (2 if op == 'load' else 3) or args[0].dtype != 'int32'
                or args[1] != TensorType('bool', args[0].shape)):
            raise ValueError('Invalid indexed memory operation')
        if op == 'store':
            if buf.role != 'scratch' or args[2] != TensorType(buf.dtype, args[0].shape):
                raise ValueError('Invalid store type or destination')
            if s.node.operands[0] not in s.unique:
                raise ValueError('Store indices must be a proven unique permutation')
            return []
        return [TensorType(buf.dtype, args[0].shape)]

    def _call(self, s):
        signature = self.signatures.get(s.attrs.get('callee'))
        if signature is None or s.args != signature[0]:
            raise ValueError('Invalid call signature, forward or recursive call')
        return signature[1]

    def _control(self, s):
        args, op, a, node = s.args, s.op, s.attrs, s.node
        if not args or args[0] != TensorType('bool' if op == 'if' else 'int32'):
            raise ValueError('Control requires a scalar predicate/bound')
        if len(node.regions) != (2 if op == 'if' else 1):
            raise ValueError('Invalid region count')
        if op != 'if' and (type(a.get('max_steps')) is not int or not 0 <= a['max_steps'] <= 8):
            raise ValueError('Loops must have a finite upper bound')
        carried = args[1:]
        for child in node.regions:
            expected_args = carried if op == 'if' else [TensorType('int32')] + carried
            if ([v.type for v in child.arguments] != expected_args
                    or self.visit(child, s.visible, s.scope, s.depth + 1) != carried):
                raise ValueError('Region argument/yield signature mismatch')
        return carried

    def _barrier(self, s):
        if s.args:
            raise ValueError('Barrier has no operands')
        return []


class _Step:
    """One operation under check, with the block state its rule may update."""

    def __init__(self, node, args, unique, visible, scope, depth):
        self.node, self.op, self.attrs = node, node.op, node.attrs
        self.args, self.outs = args, [v.type for v in node.results]
        self.unique, self.visible, self.scope, self.depth = unique, visible, scope, depth

    def keep_permutation(self):
        """The result is a permutation of 0..size-1 whenever the operand is."""
        if self.node.operands[0] in self.unique:
            self.unique[self.node.results[0].name] = self.unique[self.node.operands[0]]


_RULES = {op: rule for ops, rule in (
    (('constant', 'index', 'parameter'), _Analysis._source),
    (('cast',), _Analysis._cast),
    (FLOAT_UNARY_OPS, _Analysis._float_unary),
    (FLOAT_BINARY_OPS, _Analysis._float_binary),
    (('reshape', 'broadcast', 'transpose', 'slice'), _Analysis._shape),
    (('flip',), _Analysis._flip),
    (('interleave', 'join'), _Analysis._pair),
    (('split',), _Analysis._split),
    (('scan_sum', 'scan_product', 'scan_max', 'sort', 'dsl_sigmoid', 'dsl_clamp', 'softmax'), _Analysis._float_target),
    (('argmax', 'argmin', 'xor_sum'), _Analysis._integer_reduction),
    (('topk',), _Analysis._topk),
    (('gather',), _Analysis._gather),
    (('reduce_abssum', 'reduce_absmax'), _Analysis._absolute_reduction),
    (('histogram',), _Analysis._histogram),
    (('reduce_bitand', 'reduce_bitor', 'reduce_bitxor'), _Analysis._bitwise_reduction),
    (('add', 'sub', 'mul', 'bitand', 'bitxor', 'mod', 'lt', 'eq', 'and', 'or'), _Analysis._binary),
    (('select',), _Analysis._select),
    (('reduce',), _Analysis._reduce),
    (('fma',), _Analysis._fma),
    (ATOMIC_OPS, _Analysis._atomic),
    (('matmul',), _Analysis._matmul),
    (('load', 'store'), _Analysis._memory),
    (('call',), _Analysis._call),
    (CONTROL_OPS, _Analysis._control),
    (('barrier',), _Analysis._barrier),
) for op in ops}
