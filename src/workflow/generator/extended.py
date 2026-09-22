"""Type-directed generation with bounded operand synthesis and coverage anchors."""
import copy
import random
from src.ir.extended import (TensorType as Ty, Value, Node, Block, Helper, Buffer,
                             ExtendedProgram, broadcast_shape)
from src.workflow.generator.grids import (ATOMIC_GRID, FMA_GRID, INT8_MATMUL_GRID,
                                          SHAPE_OP_GRID)


FAMILIES = ('arithmetic', 'indexed_memory', 'shape_matmul', 'control_calls', 'mixed')

# Per-backend shape-op availability; flip is expressible in both DSLs while
# join/split/interleave are triton-only primitives.
EXTENDED_SHAPE_CAPS = {
    'flip': {'tilelang', 'triton'},
    'interleave': {'triton'},
    'join': {'triton'},
    'split': {'triton'},
}


class Builder:
    def __init__(self, owner, block=None, inherited=()):
        self.owner = owner
        self.block = block or Block()
        self.pool = list(inherited) + list(self.block.arguments)

    def value(self, ty):
        self.owner.serial += 1
        return Value('e' + str(self.owner.serial), ty)

    def emit(self, op, inputs=(), types=(), **attrs):
        results = [self.value(t) for t in types]
        self.block.operations.append(Node(op, results, [v.name for v in inputs], attrs))
        self.pool.extend(results)
        return results[0] if len(results) == 1 else results

    def constant(self, ty, value=None):
        if value is None:
            value = random.choice((0.0, 0.125, -0.125, 0.5)) if ty.dtype.startswith('float') else random.choice((0, 1, 3))
        if ty.dtype == 'bool':
            value = bool(value)
        return self.emit('constant', types=[ty], value=value)

    def cast(self, value, dtype):
        return self.emit('cast', [value], [Ty(dtype, value.type.shape)])

    def binary(self, op, x, y):
        dtype = 'bool' if op in ('lt', 'eq') else x.type.dtype
        return self.emit(op, [x, y], [Ty(dtype, broadcast_shape(x.type.shape, y.type.shape))])

    def reshape(self, value, shape):
        return self.emit('reshape', [value], [Ty(value.type.dtype, shape)])

    def reduce(self, value, axis, kind='sum'):
        shape = list(value.type.shape)
        del shape[axis]
        dtype = 'float32' if value.type.dtype.startswith('float') else 'int32'
        return self.emit('reduce', [value], [Ty(dtype, shape)], axis=axis, kind=kind)

    def indices(self, shape, shuffled=True):
        ty = Ty('int32', shape)
        return self.emit('index', types=[ty], shift=random.randrange(ty.size) if shuffled else 0,
                         reverse=random.choice((False, True)) if shuffled else False)

    def load(self, buf, shape, indices=None, mask=None):
        indices = indices or self.indices(shape)
        mask = mask or self.constant(Ty('bool', shape), True)
        return self.emit('load', [indices, mask], [Ty(buf.dtype, shape)], buffer=buf.name)

    def get_or_create(self, ty, depth=0):
        candidates = [v for v in self.pool if v.type == ty]
        if candidates and random.random() < .7:
            return random.choice(candidates)
        # Synthesis is bounded, and every inserted producer enters the pool.
        if depth >= 2 or not ty.shape:
            return self.constant(ty)
        if ty.dtype == 'int8':
            # Bounded cast from int32: [-8, 7] has identical wrap semantics in
            # torch, TIR and Triton narrowing.
            source = self.get_or_create(Ty('int32', ty.shape), depth + 1)
            source = self.binary('bitand', source, self.constant(source.type, 15))
            source = self.binary('sub', source, self.constant(source.type, 8))
            return self.cast(source, 'int8')
        # Do not synthesize undefined float->int conversions from arbitrary
        # arithmetic (NaN/Inf/out-of-range). Integer->float remains composable,
        # but scale indices first to avoid drowning later matmuls in overflow.
        same_shape = [v for v in self.pool if v.type.shape == ty.shape and
                      v.type.dtype != 'bool' and ty.dtype.startswith('float')]
        if same_shape and random.random() < .35:
            value = random.choice(same_shape)
            if value.type.dtype == 'int32':
                value = self.binary('bitand', value, self.constant(value.type, 7))
                value = self.cast(value, ty.dtype)
                return self.binary('mul', value, self.constant(ty, 0.125))
            return self.cast(value, ty.dtype)
        if ty.dtype == 'bool':
            idx = self.indices(ty.shape)
            bound = self.emit('parameter', types=[Ty('int32')], name='limit')
            return self.binary('lt', idx, bound)
        buf = self.owner.buffer(ty.dtype, ty.size)
        return self.load(buf, ty.shape)

    def arithmetic(self, value, length=3):
        current = value
        for _ in range(length):
            dtype = current.type.dtype
            rhs = self.get_or_create(current.type)
            choices = ('add', 'sub', 'mul') if dtype.startswith('float') else ('add', 'bitxor', 'bitand')
            current = self.binary(random.choice(choices), current, rhs)
            if random.random() < .3 and dtype.startswith('float'):
                current = self.cast(current, 'float32' if dtype == 'float16' else 'float16')
        return current


class ExtendedGenerator:
    def __init__(self, config, backend, feedback=None, grids=None):
        self.config, self.backend, self.feedback = config, backend, feedback
        self.grids = grids  # None for the mutation path: fall back to random.
        self.serial = 0
        self.buffers = []

    def buffer(self, dtype, size, role='input', **view):
        buf = Buffer('mem' + str(len(self.buffers)), dtype, size, role, **view)
        self.buffers.append(buf)
        return buf

    def grid_cell(self, op, grid):
        if self.grids is not None:
            return self.grids.next_cell(op, self.backend, grid)
        return random.choice(grid)

    def generate(self, family=None):
        self.serial, self.buffers = 0, []
        if family is None:
            if self.feedback is None:
                family = random.choice(FAMILIES)
            else:
                from src.workflow.feedback import key
                family = random.choices(FAMILIES, weights=[
                    self.feedback.weight(key('extended_family', f), passed_decay=4.0,
                                         uncovered_boost=self.config.uncovered_boost) for f in FAMILIES])[0]
        if family not in FAMILIES:
            raise ValueError('Unknown extended generation family')
        builder = Builder(self)
        shape = random.choice(((16, 16), (16, 32), (32, 16)))
        dtype = random.choice(('float16', 'float32'))
        start = builder.get_or_create(Ty(dtype, shape))
        answer = builder.arithmetic(start, random.randint(1, 3))
        watched, functions, extras = [start], [], []

        # The mixed family composes several feature blocks. tilelang's
        # LayoutInference turns superlinear on large fragment graphs
        # (measured: ~62s for the four-block mixed kernel vs ~7s per kernel
        # for a single-block family), so the tilelang mixed case keeps just
        # the arithmetic x matmul composition. Every dropped block is still
        # exercised by its own family case on both backends, and the matmul
        # keeps the precision/identity pair surface alive. Triton compiles
        # the full composition in seconds.
        blocks = []
        if family in ('arithmetic', 'mixed'):
            blocks.append('arithmetic')
        if family in ('indexed_memory', 'mixed'):
            blocks.append('indexed_memory')
        if family in ('shape_matmul', 'mixed'):
            blocks.append('shape_matmul')
        if family in ('control_calls', 'mixed'):
            blocks.append('control_calls')
        if family == 'mixed' and self.backend == 'tilelang':
            blocks = ['arithmetic', 'shape_matmul']

        if 'arithmetic' in blocks:
            # Native half arithmetic is preserved as half; no implicit f32 lift.
            half = builder.cast(answer, 'float16')
            half = builder.binary('mul', half, builder.constant(half.type, 0.5))
            integer = builder.get_or_create(Ty('int32', shape))
            integer = builder.binary(random.choice(('bitand', 'bitxor')), integer,
                                     builder.constant(integer.type, random.choice((1, 3, 7))))
            predicate = builder.binary('lt', integer, builder.constant(integer.type, 3))
            predicate = builder.binary('and', predicate, builder.get_or_create(predicate.type))
            answer = builder.emit('select', [predicate, half, builder.constant(half.type, -0.125)], [half.type])
            extras += [integer, predicate]
            watched += [half]

        if 'indexed_memory' in blocks:
            answer, observed = self.memory(builder, answer)
            watched.extend(observed)

        if 'shape_matmul' in blocks:
            answer, observed = self.matmul(builder, answer)
            watched.extend(observed)

        if 'control_calls' in blocks:
            answer, functions, scalar_outputs = self.control(builder, answer)
            extras.extend(scalar_outputs)

        # New op surfaces: global-memory atomics (effects only), scalar FMA
        # chains with data-dependent operands, and triton shape primitives.
        # Each is its own compiler code path and consumes one grid corner.
        if random.random() < self.config.extended_atomic_prob:
            self.atomic(builder, answer)
        if random.random() < self.config.extended_fma_prob:
            observed = self.fma(builder, answer)
            watched.extend(observed)
        if random.random() < self.config.extended_shape_op_prob:
            answer, observed = self.shape_ops(builder, answer)
            watched.extend(observed)

        # More producers/consumers around the anchors make them compositional.
        answer = builder.arithmetic(answer, random.randint(1, 3))
        if random.random() < .5:
            compact = builder.reduce(answer, random.randrange(len(answer.type.shape)), random.choice(('sum', 'max', 'min')))
            extras.append(compact)
        builder.block.returns = [answer.name] + [v.name for v in extras]
        # Keep the ordinary variant and an independently checked observation
        # variant. Observations select actual typed values, not lossy checksums.
        selected = watched + random.sample([v for v in builder.pool if v.type.dtype in ('bool', 'int32')],
                                          min(2, sum(v.type.dtype in ('bool', 'int32') for v in builder.pool)))
        observations = list(dict.fromkeys(v.name for v in selected if v.name not in builder.block.returns))[:8]
        max_size = max(b.size for b in self.buffers if b.role == 'input')
        program = ExtendedProgram(builder.block, self.buffers, functions, observations,
                                  blocks=random.choice((2, 3)),
                                  input_pattern=random.choice(('integer', 'normal', 'boundary')),
                                  runtime_cases=[(0, 1), (1, max(1, max_size - 1)), (random.randint(2, 4), max_size)],
                                  configuration_pair=self.config.extended_configuration_pair,
                                  observation_pair=self.config.extended_observation_pair,
                                  pass_config_pair=self.config.extended_config_depth >= 2,
                                  fast_math_pair=self.config.extended_fast_math_pair, family=family)
        from src.backends import get_backend
        get_backend(self.backend).validate_program(program)
        return program

    def memory(self, b, value):
        # Two overlapping views of one allocation. Every individual scatter is
        # injective, and barriers order writes across the two views.
        ty, n = value.type, value.type.size
        root = self.buffer(ty.dtype, 2 * n + 4, 'scratch')
        left = self.buffer(ty.dtype, n, 'scratch', base=root.name, offset=1, stride=2)
        right = self.buffer(ty.dtype, n, 'scratch', base=root.name, offset=2, stride=1)
        idx = b.indices(ty.shape)
        limit = b.emit('parameter', types=[Ty('int32')], name='limit')
        mask = b.binary('lt', idx, limit)
        b.emit('store', [idx, mask, value], buffer=left.name)
        b.emit('barrier')
        read = b.load(left, ty.shape, idx, mask)
        updated = b.binary('add', read, b.constant(ty, 0.125))
        # Every lane must finish its snapshot before another lane overwrites
        # the overlapping view. A write->read barrier alone is insufficient.
        b.emit('barrier')
        b.emit('store', [idx, mask, updated], buffer=right.name)
        b.emit('barrier')
        # Re-reading through the original alias tests an actual read-after-write
        # dependence. Unwritten scratch starts at a known finite sentinel.
        reread = b.load(left, ty.shape, idx, b.constant(Ty('bool', ty.shape), True))
        gather_idx = b.binary('bitxor', idx, b.constant(Ty('int32', ty.shape), random.choice((1, 3, 7))))
        gathered = b.load(right, ty.shape, gather_idx, mask)
        return b.binary('add', reread, gathered), [read, reread]

    def matmul(self, b, value):
        if random.random() < self.config.extended_int8_prob:
            # int8 x int8 with an int32 accumulator (exact oracle). Shapes
            # come from the pre-validated grid so K >= 32 always holds.
            cell = self.grid_cell('int8_matmul', INT8_MATMUL_GRID)
            m, n, k = cell['m'], cell['n'], cell['k']
            left = b.get_or_create(Ty('int8', (m, k)))
            right = b.get_or_create(Ty('int8', (k, n)))
            accumulator = b.constant(Ty('int32', (m, n)), 0)
            product = b.emit('matmul', [left, right, accumulator], [accumulator.type])
            # Fold the exact int32 result into the checked float chain.
            total = b.reduce(product, 0)
            total = b.reduce(total, 0)
            scaled = b.binary('mul', b.cast(total, 'float32'), b.constant(Ty('float32'), 0.0078125))
            expanded = b.emit('broadcast', [scaled], [Ty('float32', value.type.shape)])
            if value.type.dtype == 'float16':
                expanded = b.cast(expanded, 'float16')
            return b.binary('add', value, expanded), [product]
        # A non-square transpose and slice participate in a first matmul;
        # its computed result becomes an operand of a second matmul.
        source = b.cast(value, 'float16')
        transposed = b.emit('transpose', [source], [Ty('float16', source.type.shape[::-1])])
        if transposed.type.shape[0] == 32:
            transposed = b.emit('slice', [transposed], [Ty('float16', (16, transposed.type.shape[1]))],
                                offsets=[random.choice((0, 8, 16)), 0])
        m, k = transposed.type.shape
        rhs = b.get_or_create(Ty('float16', (k, 16)))
        accumulator = b.constant(Ty('float32', (m, 16)), random.choice((0.0, 0.125)))
        product = b.emit('matmul', [transposed, rhs, accumulator], [accumulator.type])
        row = b.reduce(product, 1, 'max')
        column = b.reshape(row, (m, 1))
        adjusted = b.binary('sub', product, column)
        adjusted = b.binary('mul', adjusted, b.constant(adjusted.type, 0.125))
        second = b.get_or_create(Ty('float16', (16, 16)))
        result = b.emit('matmul', [b.cast(adjusted, 'float16'), second, accumulator], [accumulator.type])
        # Reshape round trips are emitted as actual DSL shape operations.
        flat = b.reshape(result, (result.type.size,)) if result.type.size <= 64 else b.reshape(result, (16, m))
        result = b.reshape(flat, (m, 16))
        return result, [product, adjusted]

    def atomic(self, b, value, fn=None):
        """Commutative global-memory reduction over possibly duplicate
        addresses. The race is the point, so indices are deliberately not
        proven unique, and the oracle compares the whole scratch memory."""
        cell = self.grid_cell('atomic', ATOMIC_GRID)
        fn = fn or random.choice(('add', 'max', 'min'))
        dtype, shape = cell['dtype'], value.type.shape
        root = self.buffer(dtype, max(16, value.type.size), 'scratch')
        # Bounded payload synthesis: constants and constant arithmetic only,
        # never loads (the 'special' input pattern holds NaNs, and NaN ordering
        # is not commutative under max/min).
        pick = {'positive': lambda: random.choice((1, 2, 3) if dtype == 'int32' else (0.125, 0.25, 0.5)),
                'negative': lambda: -random.choice((1, 2, 3) if dtype == 'int32' else (0.125, 0.25, 0.5)),
                'mixed': lambda: random.choice((-2, -1, 1, 2) if dtype == 'int32' else (-0.25, -0.125, 0.125, 0.25)),
                'zero': lambda: 0}[cell['value']]
        ty = Ty(dtype, shape)
        if cell['value'] == 'zero':
            payload = b.emit('broadcast', [b.constant(Ty(dtype), 0)], [ty])
        else:
            first = b.emit('broadcast', [b.constant(Ty(dtype), pick())], [ty])
            second = b.emit('broadcast', [b.constant(Ty(dtype), pick())], [ty])
            payload = b.binary(random.choice(('add', 'sub')), first, second)
        idx = b.indices(shape, shuffled=True)
        if cell['index'] == 'paired':
            # i and i + size/2 collide, so every raced address is hit twice.
            idx = b.binary('bitand', idx, b.constant(Ty('int32', shape), value.type.size // 2 - 1))
        elif cell['index'] == 'uniform':
            # Heavy collisions over a small address window.
            idx = b.binary('bitand', idx, b.constant(Ty('int32', shape), 7))
        if cell['mask'] == 'half':
            probe = b.indices(shape, shuffled=True)
            mask = b.binary('lt', probe, b.constant(Ty('int32', shape), value.type.size // 2))
        else:
            mask = b.constant(Ty('bool', shape), True)
        b.emit(f'atomic_{fn}', [idx, mask, payload], buffer=root.name)

    def fma(self, b, value):
        """Scalar fused multiply-add chains with data-dependent operands; the
        grid corner pins the sign/zero pattern so constant folding cannot
        erase the fused operation."""
        cell = self.grid_cell('fma', FMA_GRID)
        dtype = cell['dtype']
        # Load three scalars from input memory (never constant-foldable), then
        # scale them into the corner's sign pattern.
        source = self.buffer(dtype, 4)
        loaded = [b.load(source, (), b.constant(Ty('int32'), i), b.constant(Ty('bool'), True))
                  for i in range(3)]
        scaled = [b.binary('mul', loaded[i], b.constant(Ty(dtype), cell[axis]))
                  for i, axis in enumerate(('x', 'y', 'z'))]
        first = b.emit('fma', scaled, [Ty(dtype)])
        chained = b.emit('fma', [first, scaled[1], scaled[2]], [Ty(dtype)])
        return [first, chained]

    def shape_ops(self, b, value):
        """Triton shape primitives folded into the checked chain. flip is
        shape-preserving and applied in place; the others synthesize bounded
        sources per grid corner, and the results are both watched (exact
        order check) and reduced back into the answer."""
        available = [op for op, backends in EXTENDED_SHAPE_CAPS.items() if self.backend in backends]
        op = random.choice(available)
        cell = self.grid_cell('shape_' + op, SHAPE_OP_GRID[op])
        dtype = value.type.dtype
        if op == 'flip':
            flipped = b.emit('flip', [value], [value.type])
            return flipped, [flipped]
        shape = cell['shape']
        left = b.get_or_create(Ty(dtype, shape))
        if op == 'split':
            first, second = b.emit('split', [left], [Ty(dtype, shape[:-1]), Ty(dtype, shape[:-1])])
            parts, watched = [first, second], [first, second]
        elif op == 'join':
            right = b.get_or_create(Ty(dtype, shape))
            joined = b.emit('join', [left, right], [Ty(dtype, shape + (2,))])
            # A join/split round trip: the compiler must reproduce both
            # sources exactly through the new minor dimension.
            first, second = b.emit('split', [joined], [Ty(dtype, shape), Ty(dtype, shape)])
            parts, watched = [first, second], [joined]
        else:  # interleave
            right = b.get_or_create(Ty(dtype, shape))
            mixed = b.emit('interleave', [left, right], [Ty(dtype, shape[:-1] + (shape[-1] * 2,))])
            parts, watched = [mixed], [mixed]
        for part in parts:
            scalar = part
            while scalar.type.shape:
                scalar = b.reduce(scalar, 0)  # promotes float16 -> float32
            expanded = b.emit('broadcast', [scalar], [Ty(scalar.type.dtype, value.type.shape)])
            if expanded.type.dtype != value.type.dtype:
                expanded = b.cast(expanded, value.type.dtype)
            value = b.binary('add', value, expanded)
        return value, watched

    def control(self, b, value):
        value = b.cast(value, 'float16')
        m, _ = value.type.shape
        tile_arg, row_arg = b.value(value.type), b.value(Ty('float32', (m,)))
        helper = Builder(self, Block([tile_arg, row_arg]))
        row = helper.reduce(tile_arg, 1, random.choice(('sum', 'max')))
        row = helper.binary('add', row, row_arg)
        scalar = helper.reduce(row, 0)
        expanded = helper.reshape(row, (m, 1))
        expanded = helper.cast(expanded, 'float16')
        tile = helper.binary('add', tile_arg, expanded)
        helper.block.returns = [tile.name, scalar.name]
        fn = Helper('mixed_helper', helper.block)
        row_input = b.get_or_create(Ty('float32', (m,)))
        first, stat = b.emit('call', [value, row_input], [value.type, Ty('float32')], callee=fn.name)
        bound = b.emit('parameter', types=[Ty('int32')], name='steps')
        counter = b.constant(Ty('int32'), 0)
        # Read a scalar from input memory, so the uniform branch predicate is
        # data-dependent rather than selected by the generator.
        control_input = self.buffer('int32', 1)
        control_value = b.load(control_input, (), b.constant(Ty('int32'), 0), b.constant(Ty('bool'), True))
        predicate = b.binary('lt', control_value, b.constant(Ty('int32'), 0))
        carried = [first, stat, counter]
        for loop in ('for', 'while'):
            arguments = [b.value(Ty('int32'))] + [b.value(v.type) for v in carried]
            child = Builder(self, Block(arguments), b.pool)
            iv, tile, total, count = arguments
            branches = []
            for operation in ('add', 'sub'):
                arg = b.value(tile.type)
                branch = Builder(self, Block([arg]), child.pool)
                result = branch.binary(operation, arg, branch.constant(arg.type, 0.125))
                branch.block.returns = [result.name]
                branches.append(branch.block)
            merged = child.value(tile.type)
            child.block.operations.append(Node('if', [merged], [predicate.name, tile.name], {}, branches))
            child.pool.append(merged)
            new_tile, new_stat = child.emit('call', [merged, row_input], [tile.type, Ty('float32')], callee=fn.name)
            # Keep half values bounded even for multiple runtime iterations.
            new_tile = child.binary('mul', new_tile, child.constant(tile.type, 0.125))
            total = child.binary('add', total, new_stat)
            count = child.binary('add', count, child.binary('add', iv, child.constant(Ty('int32'), 1)))
            child.block.returns = [new_tile.name, total.name, count.name]
            results = [b.value(v.type) for v in carried]
            b.block.operations.append(Node(loop, results, [bound.name] + [v.name for v in carried],
                                           {'max_steps': 4, 'pipelined': loop == 'for' and random.choice((True, False))}, [child.block]))
            b.pool.extend(results)
            carried = results
        return carried[0], [fn], carried[1:]


def mutate_extended(program, config, backend):
    if random.random() < .3:
        return ExtendedGenerator(config, backend).generate(program.family)
    result = copy.deepcopy(program)
    candidates = [n for n in result.all_operations() if n.op in ('constant', 'index', 'add', 'sub', 'lt', 'eq')]
    node = random.choice(candidates)
    if node.op == 'constant':
        dtype = node.results[0].type.dtype
        node.attrs['value'] = (random.choice((False, True)) if dtype == 'bool' else
                               random.choice((-1, 0, 1, 3)) if dtype == 'int32' else random.choice((-0.125, 0., 0.125, 0.5)))
    elif node.op == 'index':
        node.attrs['shift'] = random.randrange(node.results[0].type.size)
        node.attrs['reverse'] = not node.attrs.get('reverse', False)
    else:
        node.op = {'add':'sub', 'sub':'add', 'lt':'eq', 'eq':'lt'}[node.op]
    if random.random() < .3:
        result.input_pattern = random.choice(('integer', 'normal', 'boundary'))
    from src.backends import get_backend
    get_backend(backend).validate_program(result)
    return result
