"""Target-specific mutations of already instantiated, passing common IR.

The common generator never calls this module. A campaign may feed a saved
``passed/*.json`` program here after rechecking it on the target environment.
The derived program is a new, fully instantiated ExtendedProgram; no template
is substituted at execution time.
"""
import copy
import inspect
import random
from functools import lru_cache

from src.ir.extended import (Block, Buffer, ExtendedProgram, Node, TensorType, Value,
                             TARGET_ATTRIBUTE_OPS, reduced_shape, target_axis)
from .extended import Builder, ExtendedGenerator


DSL_OPS = {
    'triton': ('join', 'split', 'interleave', 'scan_sum', 'scan_product', 'sort', 'histogram',
               'argmax', 'argmin', 'xor_sum', 'dsl_sigmoid', 'dsl_clamp', 'softmax',
               'topk', 'gather', 'atomic_and', 'atomic_or', 'atomic_xor'),
    'tilelang': ('pipelined_for', 'scan_sum', 'scan_max', 'reduce_abssum', 'reduce_absmax',
                 'reduce_bitand', 'reduce_bitor', 'reduce_bitxor', 'dsl_sigmoid', 'dsl_clamp'),
}
TARGET_ONLY_OPS = frozenset(('join', 'split', 'interleave', 'scan_sum',
                             'scan_product', 'scan_max', 'sort',
                             'reduce_abssum', 'reduce_absmax', 'histogram',
                             'reduce_bitand', 'reduce_bitor', 'reduce_bitxor',
                             'argmax', 'argmin', 'xor_sum', 'dsl_sigmoid', 'dsl_clamp',
                             'softmax', 'topk', 'gather', 'atomic_and', 'atomic_or', 'atomic_xor'))
INTEGER_OPS = ('histogram', 'argmax', 'argmin', 'xor_sum',
               'reduce_bitand', 'reduce_bitor', 'reduce_bitxor')
SCAN_OPS = ('scan_sum', 'scan_product', 'scan_max')
# Results drop the reduced axis.
REDUCTION_OPS = ('argmax', 'argmin', 'xor_sum', 'reduce_abssum', 'reduce_absmax',
                 'reduce_bitand', 'reduce_bitor', 'reduce_bitxor')
# Target calls that also take a rank-2 operand; histogram and gather keep
# their 1-D contracts.
MATRIX_OPS = TARGET_ATTRIBUTE_OPS | {'dsl_sigmoid', 'dsl_clamp'}
TRITON_API = {'scan_sum': 'cumsum', 'scan_product': 'cumprod', 'scan_max': 'cummax'}


@lru_cache(maxsize=None)
def _available(backend, api):
    if backend != 'triton':
        return True
    import triton.language as tl
    return hasattr(tl, api)


@lru_cache(maxsize=None)
def _accepts(backend, api, parameter):
    """Whether the installed front end names `parameter`; TileLang always does.

    Triton 3.0's softmax has no dim or keep_dims (it normalizes axis 0) and
    3.8 names both. An unknown keyword would be a harness error, not a bug.
    """
    if backend != 'triton':
        return True
    try:
        import triton.language as tl
        fn = getattr(tl, api, None)
        return parameter in inspect.signature(getattr(fn, 'fn', fn)).parameters
    except (ImportError, TypeError, ValueError):
        return False


def target_attributes(op, ty, backend, axes=None):
    """Draw how target call `op` spells its attributes for operand type `ty`.

    `axes` restricts the normalized axis, so that a mutation keeps the result
    type of a reduction. Both spellings of an axis name the same dimension.
    """
    rank = len(ty.shape)
    attrs = {}
    if op in SCAN_OPS or op in REDUCTION_OPS or op == 'softmax':
        spelled = op != 'softmax' or _accepts(backend, 'softmax', 'dim')
        choices = [axis for axis in (range(rank) if axes is None else axes) if spelled or axis == 0]
        if not choices:
            raise ValueError(f'No {op} axis keeps the result type')
        axis = random.choice(choices)
        attrs['axis'] = axis - rank if spelled and random.random() < .25 else axis
    if op in SCAN_OPS and _accepts(backend, TRITON_API[op], 'reverse'):
        attrs['reverse'] = random.random() < .5
    if op == 'sort' or (op == 'topk' and _accepts(backend, 'topk', 'descending')):
        attrs['descending'] = random.random() < .5
    if op == 'softmax' and _accepts(backend, 'softmax', 'keep_dims'):
        # Triton 3.8 subtracts a dim=1 row maximum without keep_dims along
        # the wrong axis (main ignores keep_dims); keep that spelling rare.
        known = rank == 2 and attrs['axis'] % rank == 1
        attrs['keep_dims'] = random.random() < (.875 if known else .5)
    if op == 'topk':
        # Every power of two from 2 to the extent: k = extent is a full sort,
        # smaller k adds the bitonic top-k reductions.
        sizes = [1 << i for i in range(1, ty.shape[-1].bit_length())]
        if not sizes:
            raise ValueError('topk needs at least two candidates')
        attrs['k'] = random.choice(sizes)
    return attrs


def result_type(op, ty, attrs):
    if op == 'histogram':
        return TensorType('int32', (16,))
    if op == 'topk':
        return TensorType('float32', ty.shape[:-1] + (attrs['k'],))
    if op in REDUCTION_OPS:
        return TensorType(ty.dtype, reduced_shape(ty, target_axis(attrs, ty)))
    return ty


def _fold(builder, value):
    """Reshape a vector into a matrix whose extents are both at least two."""
    size = value.type.size
    rows = random.choice([1 << i for i in range(1, size.bit_length() - 1)])
    return builder.reshape(value, (rows, size // rows))


def _bounded(builder, value):
    """Finite fp32 values in [-1, 1] of any shape, from any dtype."""
    if value.type.dtype != 'float32':
        value = builder.cast(value, 'float32')
    # Not `value == value`: TVM's simplifier folds self-comparison to true,
    # so NaN reached the clamp below (-1 on CUDA, 0 in the reference) and
    # every downstream target op reported a false wrong_result. A magnitude
    # bound cannot be folded and also rejects +/-Inf.
    magnitude = builder.emit('abs', [value], [value.type])
    finite = builder.binary('lt', magnitude, builder.constant(value.type, 1e30))
    value = builder.emit('select', [finite, value,
                                   builder.constant(value.type, 0)], [value.type])
    # Restrict scan products and absolute reductions to finite, bounded
    # values even when the passing parent computes a large magnitude.
    low = builder.constant(value.type, -1)
    high = builder.constant(value.type, 1)
    value = builder.emit('maximum', [value, low], [value.type])
    return builder.emit('minimum', [value, high], [value.type])


def _vector_input(builder, answer):
    """Make an input-dependent, finite 1-D fp32 value for target primitives."""
    value = answer
    if len(value.type.shape) == 2:
        if value.type.dtype in ('bool', 'int8'):
            value = builder.cast(value, 'int32')
        value = builder.reduce(value, axis=0)
    elif not value.type.shape:
        value = builder.emit('broadcast', [value], [TensorType(value.type.dtype, (16,))])
    return _bounded(builder, value)


def _matrix_input(builder, answer, checked=(), minor=1):
    """A finite rank-2 fp32 operand derived from checked values, or None.

    The answer itself is preferred; another checked matrix may replace a
    vector answer, which otherwise folds into a matrix.
    """
    value = None
    if len(answer.type.shape) == 2:
        if answer.type.shape[1] >= minor:
            value = answer
    elif checked and (len(answer.type.shape) != 1 or answer.type.size < 4 or random.random() < .5):
        value = random.choice(checked)
    elif len(answer.type.shape) == 1 and answer.type.size >= 4:
        value = _fold(builder, answer)
    return None if value is None else _bounded(builder, value)


def _integer_matrix(builder, values, preferred=None):
    """An exact rank-2 int32 operand for an axis reduction, or None.

    Like `_integer_vector`, a composed integer answer is followed; float
    values are never binned. No mask: an axis reduction has no bins.
    """
    exact = ('int32', 'int8', 'bool')
    if preferred is not None and preferred.type.dtype == 'int32':
        candidates = [preferred]
    else:
        candidates = [v for v in values if v.type.dtype in exact and len(v.type.shape) == 2]
        candidates = candidates or [v for v in values if v.type.dtype in exact
                                    and len(v.type.shape) == 1 and v.type.size >= 4]
    if not candidates:
        return None
    value = random.choice(candidates)
    if len(value.type.shape) != 2:
        if len(value.type.shape) != 1 or value.type.size < 4:
            return None
        value = _fold(builder, value)
    return value if value.type.dtype == 'int32' else builder.cast(value, 'int32')


def _integer_vector(builder, values, preferred=None):
    """Use exact integer dataflow; float-to-int binning amplifies roundoff."""
    candidates = [v for v in values if v.type.dtype == 'int32' and v.type.shape]
    if preferred is not None and preferred.type.dtype == 'int32':
        if not preferred.type.shape:
            preferred = builder.emit('broadcast', [preferred], [TensorType('int32', (16,))])
        candidates.append(preferred)
    if candidates:
        value = candidates[-1]
        if len(value.type.shape) == 2:
            value = builder.binary('bitand', value, builder.constant(value.type, 15))
            value = builder.reduce(value, axis=0)
    else:
        value = builder.indices((16,), shuffled=False)
        steps = builder.emit('parameter', types=[TensorType('int32')], name='steps')
        mask = builder.binary('lt', value, steps)
        value = builder.emit('select', [mask, value,
                                        builder.constant(value.type, 0)], [value.type])
    return builder.binary('bitand', value, builder.constant(value.type, 15))


def is_common_seed(program):
    """Reject old mixed-stage seeds whose structure already names a target op."""
    return (isinstance(program, ExtendedProgram)
            and not any(n.op in TARGET_ONLY_OPS or (n.op == 'for' and n.attrs.get('pipelined'))
                        for n in program.all_operations()))


def eligible_ops(program, backend, *, allow_target=False):
    if (backend not in DSL_OPS or not isinstance(program, ExtendedProgram)
            or (not allow_target and not is_common_seed(program))):
        return ()
    operations = list(program.all_operations())
    result = []
    for op in DSL_OPS[backend]:
        if op == 'pipelined_for' and not any(n.op == 'for' and not n.attrs.get('pipelined')
                                           for n in operations):
            continue
        if op in ('topk', 'gather') and not _available(backend, op):
            continue
        result.append(op)
    return tuple(result)


def extend_passed(program, backend, op, config, grids=None, *, allow_target=False, input_name=None):
    """Return a validated derivative without modifying the passing parent."""
    if op not in eligible_ops(program, backend, allow_target=allow_target):
        raise ValueError(f'{op} is not eligible for this common {backend} seed')
    result = copy.deepcopy(program)
    if op == 'pipelined_for':
        next(n for n in result.all_operations() if n.op == 'for' and not n.attrs.get('pipelined')).attrs['pipelined'] = True
    else:
        generator = ExtendedGenerator(config, backend, grids=grids)
        generator.buffers = result.buffers
        def value_names(block):
            yield from (v.name for v in block.arguments)
            for node in block.operations:
                yield from (v.name for v in node.results)
                for region in node.regions:
                    yield from value_names(region)
        names = list(value_names(result.body))
        for function in result.functions:
            names.extend(value_names(function.body))
        generator.serial = max((int(name[1:]) for name in names
                                if name.startswith('e') and name[1:].isdigit()), default=0)
        top_values = [v for n in result.body.operations for v in n.results]
        builder = Builder(generator, result.body, top_values)
        answer_name = input_name or result.body.returns[0]
        answer = next(v for v in top_values if v.name == answer_name)
        if backend == 'triton' and op in ('join', 'split', 'interleave'):
            if answer.type.dtype in ('bool', 'int8'):
                answer = builder.cast(answer, 'int32')
            updated, observed = generator.shape_ops(builder, answer, op=op)
            result.body.returns[result.body.returns.index(answer_name)] = updated.name
            result.observations = list(dict.fromkeys([v.name for v in observed] + result.observations))[:8]
        elif op in ('atomic_and', 'atomic_or', 'atomic_xor'):
            values = _integer_vector(builder, top_values, answer if allow_target else None)
            # Every active value has bit 2 set. Three lanes contend per
            # address, so AND/OR/XOR all change the initial scratch value 11.
            values = builder.binary('add', builder.binary('bitand', values,
                                    builder.constant(values.type, 3)),
                                    builder.constant(values.type, 4))
            lanes = builder.indices(values.type.shape, shuffled=False)
            indices = builder.binary('bitand', lanes, builder.constant(values.type, 3))
            mask = builder.binary('lt', lanes, builder.constant(values.type, 12))
            existing = {b.name for b in result.buffers}
            name = 'extend_atomic_scratch'
            while name in existing:
                name += '_next'
            result.buffers.append(Buffer(name, 'int32', 16, role='scratch'))
            builder.emit(op, [indices, mask, values], buffer=name)
        else:
            # A vector has one axis; a matrix operand also reaches the axis,
            # layout and warp-mapping choices of the target lowering.
            matrix = (op in MATRIX_OPS and config.dsl_matrix_prob > 0
                      and random.random() < config.dsl_matrix_prob)
            preferred = answer if allow_target else None
            operand = None
            if op in INTEGER_OPS:
                if matrix:
                    operand = _integer_matrix(builder, top_values, preferred)
                if operand is None:
                    operand = _integer_vector(builder, top_values, preferred)
            else:
                if matrix:
                    # A fresh derivative may read any checked matrix; a
                    # composed one follows the previous target output.
                    observed = set(result.body.returns + result.observations) - {answer.name}
                    checked = [v for v in top_values if v.name in observed
                               and len(v.type.shape) == 2 and min(v.type.shape) >= 2] if input_name is None else []
                    operand = _matrix_input(builder, answer, checked, minor=2 if op == 'topk' else 1)
                if operand is None:
                    operand = _vector_input(builder, answer)
            if config.dsl_attributes:
                attrs = target_attributes(op, operand.type, backend)
            else:
                # The historical spelling: minor axis, forward, ascending
                # sort, descending top-4, and tl.softmax(x, 0), which every
                # Triton version reads as axis 0 (an absent axis is minor).
                attrs = {'k': min(4, operand.type.shape[-1])} if op == 'topk' else {}
                if op == 'softmax' and len(operand.type.shape) == 2:
                    attrs = {'axis': 0}
            inputs = [operand]
            if op == 'gather':
                inputs.append(builder.indices(operand.type.shape))
            added = builder.emit(op, inputs, [result_type(op, operand.type, attrs)], **attrs)
            result.body.returns.append(added.name)
        result.observation_pair = True
    result.family = f'extend_{backend}_{op}'
    from src.backends import get_backend
    get_backend(backend).validate_program(result)
    return result


def respell_target(program, backend):
    """Re-draw the attributes of one target call, keeping its result type.

    A reduction keeps an axis of the same extent and topk keeps k, so every
    use of the result stays valid; the interpreter follows the new spelling.
    """
    result = copy.deepcopy(program)
    types, _ = result.validate()
    candidates = [n for n in result.all_operations() if n.op in TARGET_ATTRIBUTE_OPS]
    if not candidates:
        raise ValueError('No target operation to respell')
    node = random.choice(candidates)
    # SSA names are program-unique, so the scoped type table resolves them.
    ty = next(t for (_, name), t in types.items() if name == node.operands[0])
    axes = None
    if node.op in REDUCTION_OPS:
        axes = [axis for axis in range(len(ty.shape))
                if reduced_shape(ty, axis) == node.results[0].type.shape]
    attrs = target_attributes(node.op, ty, backend, axes)
    if node.op == 'topk':
        attrs['k'] = node.attrs['k']
    node.attrs = attrs
    from src.backends import get_backend
    get_backend(backend).validate_program(result)
    return result


def loop_target(program, backend):
    """Move one shape-preserving target op into a checked runtime-bounded loop.

    The result keeps its original SSA name, so existing uses and observations
    remain live. This changes semantics; the independent interpreter computes
    the expected result, including the zero-iteration case.
    """
    result = copy.deepcopy(program)
    types, _ = result.validate()
    candidates = [n for n in result.body.operations
                  if n.op in ('scan_sum', 'scan_product', 'scan_max', 'sort',
                              'dsl_sigmoid', 'dsl_clamp', 'softmax')
                  and len(n.results) == 1 and n.results[0].type == types['main', n.operands[0]]]
    if not candidates:
        raise ValueError('No shape-preserving target operation to wrap')
    node = random.choice(candidates)
    names = {v.name for n in result.all_operations() for v in n.results}
    names.update(v.name for n in result.all_operations() for r in n.regions for v in r.arguments)
    names.update(v.name for fn in result.functions for v in fn.body.arguments)
    serial = max((int(n[1:]) for n in names if n.startswith('e') and n[1:].isdigit()), default=0)
    def fresh(ty):
        nonlocal serial
        serial += 1
        return Value(f'e{serial}', ty)
    bound = fresh(TensorType('int32'))
    iv, carried = fresh(TensorType('int32')), fresh(node.results[0].type)
    inner = copy.deepcopy(node)
    inner.results = [fresh(node.results[0].type)]
    inner.operands[0] = carried.name
    body = Block([iv, carried], [inner], [inner.results[0].name])
    loop = Node('for', node.results, [bound.name, node.operands[0]],
                {'max_steps': 2, 'pipelined': False}, [body])
    index = result.body.operations.index(node)
    result.body.operations[index:index + 1] = [Node('parameter', [bound], [], {'name': 'steps'}), loop]
    from src.backends import get_backend
    get_backend(backend).validate_program(result)
    return result
