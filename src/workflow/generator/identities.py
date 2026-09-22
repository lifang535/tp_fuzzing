"""Program transformations shared by the precision and identity oracle pairs.

Each transformation rewrites a generated ExtendedProgram into a variant whose
own interpretation provides its expected values; the emitter ships these
alongside the original PROGRAM so the runtime can compute per-variant
references instead of reusing one shared expectation.
"""
import copy

from src.ir.extended import TensorType as Ty, Node, Value, broadcast_shape


def extended_variant_label(backend, index, options):
    """Label recorded in compilation evidence and used to match precision and
    identity variants to their transformed reference programs."""
    suffix = '_prec' if options.get('precision') == 'fp16' else '_ident' if options.get('identity') else ''
    return f'{backend}_{index}{suffix}'


def _consumer_dtype(program, node, position, types):
    """Fixed dtype a consumer expects at an operand position, or None when the
    consumer accepts the operand's own dtype (dtype-agnostic operations)."""
    op = node.op
    if op in ('add', 'sub', 'mul'):
        return types[node.operands[1 - position]].dtype
    if op == 'select' and position in (1, 2):
        return types[node.operands[3 - position]].dtype
    if op == 'store' and position == 2:
        return next(b.dtype for b in program.buffers if b.name == node.attrs['buffer'])
    if op == 'call':
        callee = next(f for f in program.functions if f.name == node.attrs['callee'])
        return callee.body.arguments[position].type.dtype
    return None


def _followed_dtype(node, types):
    """Result dtype for ops that follow their operands; None when the declared
    result type stands on its own (cast, reduce, matmul, call, ...)."""
    op = node.op
    if op in ('reshape', 'broadcast', 'transpose', 'slice', 'flip'):
        return types[node.operands[0]].dtype
    if op in ('add', 'sub', 'mul') and types[node.operands[0]].dtype == types[node.operands[1]].dtype:
        return types[node.operands[0]].dtype
    if op == 'select' and types[node.operands[1]].dtype == types[node.operands[2]].dtype:
        return types[node.operands[1]].dtype
    return None


def precision_program(program):
    """fp16-accumulation copy of an extended program.

    Every matmul with a constant accumulator becomes a half-precision
    accumulation: the accumulator constant and the matmul result are retyped
    to float16. Dtype-agnostic consumers (reshape/transpose/slice/broadcast)
    and dtype-following arithmetic inherit the new dtype, and consumers whose
    rules fix an operand dtype (mixed fp16/fp32 arithmetic, calls, stores)
    get an explicit cast, so the IR stays type-consistent under analyze().
    Hardware MMA rounds the accumulator once per k-step, which the
    interpreter models, so precision variants compare like against like.

    Returns None when no constant-accumulator matmul exists (no precision
    variant is generated for that program) or when the rewritten program no
    longer validates (unknown program shape; the pair is skipped there).
    """
    program = copy.deepcopy(program)
    # Depth-first program order; each node is paired with the block whose
    # operations list holds it (region blocks included), so casts can be
    # inserted immediately before their consumer.
    ordered = []

    def collect(block):
        for node in block.operations:
            ordered.append((block, node))
            for region in node.regions:
                collect(region)

    collect(program.body)
    for fn in program.functions:
        collect(fn.body)
    constants = {node.results[0].name: node for _, node in ordered if node.op == 'constant'}
    retyped = set()
    for _, node in ordered:
        if node.op != 'matmul':
            continue
        accumulator = constants.get(node.operands[2])
        if accumulator is None:
            continue
        # A constant may accumulate several matmuls; retype it once and retype
        # every matmul that uses it.
        if accumulator.results[0].type.dtype == 'float32':
            accumulator.results[0].type = Ty('float16', accumulator.results[0].type.shape)
        node.results[0].type = Ty('float16', node.results[0].type.shape)
        retyped.add(node.results[0].name)
    if not retyped:
        return None
    # Typed names include block and region arguments, not only SSA results.
    types = {}
    for block in [program.body] + [f.body for f in program.functions]:
        for value in block.arguments:
            types[value.name] = value.type
    for _, node in ordered:
        for region in node.regions:
            for value in region.arguments:
                types[value.name] = value.type
        for value in node.results:
            types[value.name] = value.type
    # Repair operand mismatches and retype dtype-following results until the
    # types stabilise (one shared cast per operand/dtype pair, inserted before
    # the earliest consumer).
    casts = {}
    for _ in range(len(ordered) + 1):
        changed = False
        for block, node in ordered:
            for position, operand in enumerate(node.operands):
                target = _consumer_dtype(program, node, position, types)
                if target is None or types[operand].dtype == target:
                    continue
                cast = casts.get((operand, target))
                if cast is None:
                    cast = Node('cast', [Value(operand + '_prec', Ty(target, types[operand].shape))],
                                [operand], {})
                    block.operations.insert(block.operations.index(node), cast)
                    types[cast.results[0].name] = cast.results[0].type
                    casts[operand, target] = cast
                node.operands[position] = cast.results[0].name
                changed = True
            if node.results:
                dtype = _followed_dtype(node, types)
                if dtype is not None and node.results[0].type.dtype != dtype:
                    node.results[0].type = Ty(dtype, node.results[0].type.shape)
                    types[node.results[0].name] = node.results[0].type
                    changed = True
        if not changed:
            break
    try:
        program.validate()
    except ValueError:
        return None
    return program


def _producer_map(program):
    """SSA result name → (block, node), over the body and every helper."""
    producers = {}
    ordered = []

    def collect(block):
        for node in block.operations:
            ordered.append((block, node))
            for value in node.results:
                producers[value.name] = (block, node)
            for region in node.regions:
                collect(region)

    collect(program.body)
    for fn in program.functions:
        collect(fn.body)
    return producers, ordered


def identity_variant(program):
    """Distributivity copy of an extended program (RC5 algebraic identities).

    The first `mul(x, add(y, z))` / `mul(x, sub(y, z))` over floats is
    rewritten into `add(mul(x, y), mul(x, z))` / `sub(mul(x, y), mul(x, z))`:
    two multiply nodes with fresh SSA names are inserted before the original
    node, whose op becomes the distributed add/sub while keeping its result
    name and type, so every consumer and observation stays dominated.
    Broadcasting is associative, so the declared result shape is unchanged;
    fp16/fp32 distributivity differs by ~1 ulp, far below the 0.002 element
    tolerance. Matmul programs are left to the precision pair, and programs
    without a matching float pattern return None (no variant).
    """
    program = copy.deepcopy(program)
    if any(n.op == 'matmul' for n in program.all_operations()):
        return None
    producers, ordered = _producer_map(program)
    # Operand types come from the declared Value objects (names are globally
    # unique per analyze()); the taken-name set mirrors analyze()'s namespace:
    # buffers, functions, block/region arguments and every SSA result.
    types = {}
    taken = {b.name for b in program.buffers} | {f.name for f in program.functions}
    for block, node in ordered:
        for value in block.arguments:
            types[value.name] = value.type
            taken.add(value.name)
        for value in node.results:
            types[value.name] = value.type
            taken.add(value.name)
    for block, node in ordered:
        if node.op != 'mul':
            continue
        dtype = node.results[0].type.dtype
        if dtype not in ('float16', 'float32'):
            continue
        # The distributed operand may sit in either position; the other
        # operand is the factor x.
        for factor_position, distributed_position in ((0, 1), (1, 0)):
            x = node.operands[factor_position]
            inner = producers.get(node.operands[distributed_position])
            if inner is None or inner[1].op not in ('add', 'sub'):
                continue
            y, z = inner[1].operands
            # Fresh names derived from the rewritten result; extend until
            # they are unique in the whole namespace.
            base = node.results[0].name
            candidate = (base + '_l', base + '_r')
            while any(name in taken for name in candidate):
                candidate = tuple(name + 'i' for name in candidate)
            name_l, name_r = candidate
            taken.update(candidate)
            left = Node('mul', [Value(name_l, Ty(dtype, broadcast_shape(
                types[x].shape, types[y].shape)))], [x, y], {})
            right = Node('mul', [Value(name_r, Ty(dtype, broadcast_shape(
                types[x].shape, types[z].shape)))], [x, z], {})
            position = block.operations.index(node)
            block.operations[position:position] = [left, right]
            node.op = inner[1].op
            node.operands = [name_l, name_r]
            try:
                program.validate()
            except ValueError:
                return None
            return program
    return None
