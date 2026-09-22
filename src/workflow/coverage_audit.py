"""A fixed semantic inventory shared by native regions and exploration IR.

These tags describe constructs in checked source programs, not compiler edges
or confirmed bugs. Extended nodes must be potentially observable. Native-region
counts are conservative (they may include dead source operations).
"""
from src.ir.extended import ExtendedProgram, analyze, walk
from src.ir.region import RegionProgram, walk as walk_region


CAPABILITIES = {
    'mixed_float_values': 'fp16 and fp32 intermediate values',
    'compact_reduction': 'Reduction to a smaller tensor',
    'function_calls': 'Calls to generated functions',
    'nested_control': 'Nested control regions',
    'scratch_read_write': 'Mutable scratch reads and writes',
    'native_half_arithmetic': 'Arithmetic computed in fp16',
    'integer_dataflow': 'int32 arithmetic or bitwise dataflow',
    'boolean_dataflow': 'Computed bool values and logical operations',
    'computed_memory_mask': 'An SSA comparison/logical value masks memory',
    'indexed_gather': 'Computed indices select loaded elements',
    'injective_scatter': 'Proven unique SSA indices select stored elements',
    'overlapping_strided_views': 'Accessed strided views overlap one allocation',
    'internal_matmul': 'Matmul consumes explicit SSA operands and accumulator',
    'dependent_matmul': 'One matmul result feeds a later matmul',
    'non_square_transpose': 'Transpose with unequal input dimensions',
    'tensor_slice': 'A tensor subview participates in computation',
    'heterogeneous_function_arguments': 'Function arguments have different types/shapes',
    'multiple_function_results': 'A call produces more than one result',
    'heterogeneous_loop_state': 'Loop-carried values have different types/shapes',
    'multiple_loop_results': 'A loop produces more than one result',
    'runtime_for_bound': 'A generated for loop uses a runtime SSA bound',
    'bounded_while': 'A generated while loop has an enforced upper bound',
    'data_dependent_if': 'A scalar SSA value controls a branch',
    'intermediate_observations': 'An additional variant checks intermediate values',
    'scratch_contents_checked': 'The oracle compares all scratch contents',
    'transcendental_elementwise': 'tanh/erf/log/exp2/rsqrt/sin/cos/floor/ceil elementwise ops',
    'global_atomics': 'Global-memory atomic add/max/min over raced addresses',
    'scalar_fma': 'Scalar fused multiply-add chains',
    'shape_join_split': 'Join/split shape primitives',
    'shape_flip_interleave': 'Flip/interleave shape primitives',
    'int8_matmul': 'int8-by-int8 matmul with an int32 accumulator',
}

TRANSCENDENTAL_OPS = frozenset(('tanh', 'erf', 'log', 'log2', 'exp2', 'rsqrt',
                                'sin', 'cos', 'floor', 'ceil'))


def program_capabilities(program):
    program.validate()
    found = set()
    if isinstance(program, RegionProgram):
        if program.body.operations[0].kind == 'probe':
            raise ValueError('Coverage audit expects a native region or extended program')
        operations = list(program.all_operations())
        kinds = {n.kind for n in operations}
        if 'call' in kinds:
            found.add('function_calls')
        if {'load_tile', 'store_tile'} <= kinds:
            found.add('scratch_read_write')
        if kinds & TRANSCENDENTAL_OPS:
            found.add('transcendental_elementwise')
        if 'reduce_tile' in kinds:
            found.add('compact_reduction')
        if any(n.regions and any(x.regions for r in n.regions for x in walk_region(r)) for n in operations):
            found.add('nested_control')
        if program.typed:
            from src.ir.region_types import infer_program, argument_types
            types, _, _ = infer_program(program)
            if {'float16', 'float32'} <= {t.dtype for t in types.values()}:
                found.add('mixed_float_values')
            if any(len(set(argument_types(fn))) > 1 for fn in program.functions):
                found.add('heterogeneous_function_arguments')
        return found
    if not isinstance(program, ExtendedProgram):
        raise ValueError('Unsupported coverage audit IR')

    types, live = analyze(program)
    nodes = [(scope, n) for scope, body in [('main', program.body)] +
             [(fn.name, fn.body) for fn in program.functions] for n in walk(body)
             if (scope, id(n)) in live]
    dtypes = {v.type.dtype for _, n in nodes for v in n.results}
    if {'float16', 'float32'} <= dtypes:
        found.add('mixed_float_values')
    if program.observation_pair and set(program.observations) - set(program.body.returns):
        found.add('intermediate_observations')
    producers = {(scope, v.name): n for scope, n in nodes for v in n.results}
    def upstream(scope, value, operation, seen=None):
        seen = set() if seen is None else seen
        if value in seen:
            return False
        seen.add(value)
        n = producers.get((scope, value))
        return n is not None and (n.op == operation or any(upstream(scope, a, operation, seen) for a in n.operands))

    accessed = set()
    for scope, n in nodes:
        op = n.op
        if op == 'reduce':
            found.add('compact_reduction')
        if op in ('add', 'sub', 'mul') and n.results[0].type.dtype == 'float16':
            found.add('native_half_arithmetic')
        if op in ('add', 'sub', 'mul', 'bitand', 'bitxor', 'mod') and n.results[0].type.dtype == 'int32':
            found.add('integer_dataflow')
        if op in ('lt', 'eq', 'and', 'or'):
            found.add('boolean_dataflow')
        if op in ('load', 'store'):
            accessed.add(n.attrs['buffer'])
            mask = producers.get((scope, n.operands[1]))
            if mask is not None and mask.op in ('lt', 'eq', 'and', 'or'):
                found.add('computed_memory_mask')
            index = producers.get((scope, n.operands[0]))
            if op == 'load' and index is not None and index.op not in ('constant', 'index'):
                found.add('indexed_gather')
            if op == 'store':
                found.update(('injective_scatter', 'scratch_contents_checked'))
        if op == 'matmul':
            found.add('internal_matmul')
            if types[scope, n.operands[0]].dtype == 'int8':
                found.add('int8_matmul')
            if any(upstream(scope, a, 'matmul') for a in n.operands):
                found.add('dependent_matmul')
        if op == 'fma':
            found.add('scalar_fma')
        if op.startswith('atomic_'):
            found.add('global_atomics')
        if op in ('join', 'split'):
            found.add('shape_join_split')
        if op in ('flip', 'interleave'):
            found.add('shape_flip_interleave')
        if op == 'transpose' and len(set(types[scope, n.operands[0]].shape)) > 1:
            found.add('non_square_transpose')
        if op == 'slice':
            found.add('tensor_slice')
        if op == 'call':
            found.add('function_calls')
            if len({types[scope, a] for a in n.operands}) > 1:
                found.add('heterogeneous_function_arguments')
            if len(n.results) > 1:
                found.add('multiple_function_results')
        if op in ('for', 'while'):
            found.add('runtime_for_bound' if op == 'for' else 'bounded_while')
            if len(n.results) > 1:
                found.add('multiple_loop_results')
            if len({v.type for v in n.results}) > 1:
                found.add('heterogeneous_loop_state')
        if op == 'if':
            found.add('data_dependent_if')
        if n.regions and any(x.regions for r in n.regions for x in walk(r)):
            found.add('nested_control')
    buffers = {b.name: b for b in program.buffers}
    if (any(n.op == 'load' and buffers[n.attrs['buffer']].role == 'scratch' for _, n in nodes)
            and any(n.op == 'store' for _, n in nodes)):
        found.add('scratch_read_write')
    views = [b for b in program.buffers if b.base and b.name in accessed]
    for i, a in enumerate(views):
        for b in views[i + 1:]:
            if a.base == b.base and (a.stride > 1 or b.stride > 1):
                if set(range(a.offset, a.offset + a.size * a.stride, a.stride)).intersection(
                        range(b.offset, b.offset + b.size * b.stride, b.stride)):
                    found.add('overlapping_strided_views')
    return found
