"""Target-specific mutations of already instantiated, passing common IR.

The common generator never calls this module. A campaign may feed a saved
``passed/*.json`` program here after rechecking it on the target environment.
The derived program is a new, fully instantiated ExtendedProgram; no template
is substituted at execution time.
"""
import copy
from functools import lru_cache

from src.ir.extended import Buffer, ExtendedProgram, TensorType
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


@lru_cache(maxsize=None)
def _available(backend, api):
    if backend != 'triton':
        return True
    import triton.language as tl
    return hasattr(tl, api)


def _vector_input(builder, answer):
    """Make an input-dependent, finite 1-D fp32 value for target primitives."""
    value = answer
    if len(value.type.shape) == 2:
        if value.type.dtype in ('bool', 'int8'):
            value = builder.cast(value, 'int32')
        value = builder.reduce(value, axis=0)
    elif not value.type.shape:
        value = builder.emit('broadcast', [value], [TensorType(value.type.dtype, (16,))])
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


def _integer_vector(builder, values):
    """Use exact integer dataflow; float-to-int binning amplifies roundoff."""
    candidates = [v for v in values if v.type.dtype == 'int32' and v.type.shape]
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


def eligible_ops(program, backend):
    if backend not in DSL_OPS or not is_common_seed(program):
        return ()
    operations = list(program.all_operations())
    result = []
    for op in DSL_OPS[backend]:
        if op == 'pipelined_for' and not any(n.op == 'for' for n in operations):
            continue
        if op in ('topk', 'gather') and not _available(backend, op):
            continue
        result.append(op)
    return tuple(result)


def extend_passed(program, backend, op, config, grids=None):
    """Return a validated derivative without modifying the passing parent."""
    if op not in eligible_ops(program, backend):
        raise ValueError(f'{op} is not eligible for this common {backend} seed')
    result = copy.deepcopy(program)
    if op == 'pipelined_for':
        next(n for n in result.all_operations() if n.op == 'for').attrs['pipelined'] = True
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
        answer = next(v for v in top_values if v.name == result.body.returns[0])
        if backend == 'triton' and op in ('join', 'split', 'interleave'):
            updated, observed = generator.shape_ops(builder, answer, op=op)
            result.body.returns[0] = updated.name
            result.observations = list(dict.fromkeys([v.name for v in observed] + result.observations))[:8]
        elif op in ('atomic_and', 'atomic_or', 'atomic_xor'):
            values = _integer_vector(builder, top_values)
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
            if op in ('histogram', 'argmax', 'argmin', 'xor_sum') or op.startswith('reduce_bit'):
                bins = _integer_vector(builder, top_values)
                added = builder.emit(op, [bins], [TensorType('int32', (16,) if op == 'histogram' else ())])
            else:
                vector = _vector_input(builder, answer)
                if op == 'topk':
                    k = min(4, vector.type.shape[0])
                    added = builder.emit(op, [vector], [TensorType('float32', (k,))], k=k)
                elif op == 'gather':
                    indices = builder.indices(vector.type.shape)
                    added = builder.emit(op, [vector, indices], [vector.type])
                else:
                    ty = TensorType('float32', () if op.startswith('reduce_') else vector.type.shape)
                    added = builder.emit(op, [vector], [ty])
            result.body.returns.append(added.name)
        result.observation_pair = True
    result.family = f'extend_{backend}_{op}'
    from src.backends import get_backend
    get_backend(backend).validate_program(result)
    return result
