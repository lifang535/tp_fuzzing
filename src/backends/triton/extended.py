"""Triton lowering of typed exploration programs; no TileLang conditionals."""
from src.ir.extended import analyze, spelled_axis, FLOAT_UNARY_OPS, FLOAT_BINARY_OPS
from src.workflow.generator.identities import extended_variant_label
from .ops import elementwise_expr


class ExtendedLowering:
    def __init__(self, program, name, observe=False, stages=1, input_precision='ieee'):
        self.program, self.name, self.observe, self.stages = program, name, observe, stages
        self.input_precision = input_precision
        self.types, _ = analyze(program)
        self.buffers = {b.name: b for b in program.buffers}
        self.roots = [b for b in program.buffers if b.base is None]
        self.watched = list(dict.fromkeys(program.body.returns + (program.observations if observe else [])))
        self.context = [b.name for b in self.roots] + ['steps', 'limit', 'bid']
        self.lines = []

    def add(self, indent, text):
        self.lines.append('    ' * indent + text)

    def ty(self, scope, name):
        return self.types[scope, name]

    def indices(self, shape):
        if not shape:
            return 'tl.full((), 0, tl.int32)'
        if len(shape) == 1:
            return f'tl.arange(0, {shape[0]})'
        return f'(tl.arange(0, {shape[0]})[:, None] * {shape[1]} + tl.arange(0, {shape[1]})[None, :])'

    def slice_axis(self, source, shape, axis, offset, size, prefix, indent):
        """Bit-preserving slices using reshape/permute/split (also Triton 3.0).

        Unaligned slices join two smaller contiguous pieces. Unlike a reduction
        based gather emulation this preserves NaNs and signed zero exactly.
        """
        extent = shape[axis]
        if extent == size and offset == 0:
            return source
        if offset % size:
            parts = [self.slice_axis(source, shape, axis, offset + i * (size // 2),
                                     size // 2, prefix + str(i), indent) for i in range(2)]
            joined_shape = list(shape)
            joined_shape[axis] = size
            order = list(range(len(shape)))
            order.insert(axis, len(shape))
            self.add(indent, f'{prefix} = tl.reshape(tl.permute(tl.join({parts[0]}, {parts[1]}), {tuple(order)!r}), {tuple(joined_shape)!r})')
            return prefix
        half = extent // 2
        expanded = list(shape[:axis]) + [2, half] + list(shape[axis + 1:])
        order = [i for i in range(len(expanded)) if i != axis] + [axis]
        self.add(indent, f'{prefix}_lo, {prefix}_hi = tl.split(tl.permute(tl.reshape({source}, {tuple(expanded)!r}), {tuple(order)!r}))')
        shape = list(shape)
        shape[axis] = half
        return self.slice_axis(prefix + ('_hi' if offset >= half else '_lo'), shape,
                               axis, offset % half, size, prefix + '_next', indent)

    # Operations lowered by one expression template over the operand names
    # ({0}, {1}, ...) and the result type t.
    TEMPLATES = {
        'cast': '{0}.to(tl.{t.dtype})',
        'reshape': 'tl.reshape({0}, {t.shape!r})',
        'broadcast': 'tl.broadcast_to({0}, {t.shape!r})',
        'transpose': 'tl.trans({0})',
        'select': 'tl.where({0}, {1}, {2})',
        'fma': 'tl.fma({0}, {1}, {2})',
        'interleave': 'tl.interleave({0}, {1})',
        'join': 'tl.join({0}, {1})',
        'histogram': 'tl.histogram({0}, 16)',
        'dsl_sigmoid': 'tl.sigmoid({0})',
        'dsl_clamp': 'tl.clamp({0}, -0.5, 0.5)',
        'gather': 'tl.gather({0}, {1}, 0)',
        **{op: f'({{0}} {symbol} {{1}})' for op, symbol in (
            ('add', '+'), ('sub', '-'), ('mul', '*'), ('bitand', '&'), ('bitxor', '^'), ('lt', '<'),
            ('eq', '=='), ('and', '&'), ('or', '|'))},
    }

    def lower(self, block, scope, indent):
        for node in block.operations:
            expression = self.lower_operation(node, scope, indent)
            if expression is not None:
                t = node.results[0].type
                self.add(indent, f'{node.results[0].name} = ({expression}).to(tl.{t.dtype})')

    def lower_operation(self, node, scope, indent):
        """The expression of a single-result operation; other operations emit statements."""
        op, args, a = node.op, node.operands, node.attrs
        names = [v.name for v in node.results]
        t = node.results[0].type if names else None
        if op in self.TEMPLATES:
            return self.TEMPLATES[op].format(*args, t=t)
        if op == 'constant':
            return f"tl.full({t.shape!r}, {a['value']!r}, tl.{t.dtype})"
        if op == 'index':
            expression = self.indices(t.shape)
            if a.get('reverse'):
                expression = f'({t.size - 1} - {expression})'
            return f"(({expression} + {a.get('shift', 0)}) % {t.size})"
        if op == 'parameter':
            return {'steps': 'steps', 'limit': 'limit', 'block': 'bid'}[a['name']]
        if op == 'slice':
            source, shape = args[0], list(self.ty(scope, args[0]).shape)
            for axis, (offset, size) in enumerate(zip(a['offsets'], t.shape)):
                if offset == 0 and size == shape[axis]:
                    continue
                source = self.slice_axis(source, shape, axis, offset, size, names[0] + f'_slice{axis}', indent)
                shape[axis] = size
            return source
        if op in FLOAT_UNARY_OPS + FLOAT_BINARY_OPS:
            return elementwise_expr(op, args[0], args[1] if len(args) > 1 else '', '', a, '')
        if op == 'mod':
            # A non-negative remainder by a divisor of magnitude >= 1.
            divisor = f'tl.maximum(tl.abs({args[1]}), 1)'
            remainder = f'({args[0]} % {divisor})'
            return f'tl.where({remainder} < 0, {remainder} + {divisor}, {remainder})'
        if op == 'reduce':
            expression = f"tl.{a['kind']}({args[0]}.to(tl.{t.dtype}), {a['axis']})"
            # Triton's default max/min prefer numbers over NaNs, whereas the
            # shared IR propagates any NaN, like the CPU interpreter.
            if t.dtype.startswith('float') and a['kind'] in ('max', 'min'):
                expression = (f"tl.where(tl.sum(({args[0]} != {args[0]}).to(tl.int32), {a['axis']}) > 0, "
                              f"float('nan'), {expression})")
            return expression
        if op == 'matmul':
            # The accumulator dtype selects the accumulation width; the tf32
            # sweep swaps the input precision on the ieee baseline.
            if t.dtype == 'int32':
                # int8 operands accumulate in int32; int inputs reject the
                # input_precision argument entirely.
                return f'tl.dot({args[0]}, {args[1]}, {args[2]}, out_dtype=tl.int32)'
            # Triton defaults out_dtype to float32, which rejects an fp16
            # accumulator, so name it explicitly for the precision pair.
            out_dtype = f', out_dtype=tl.{t.dtype}' if t.dtype == 'float16' else ''
            return f"tl.dot({args[0]}, {args[1]}, {args[2]}, input_precision='{self.input_precision}'{out_dtype})"
        if op == 'flip':
            # Always name the axis: the dim=None default is a confirmed Triton
            # 3.8 front-end bug (triton_flip_default_axis) that would mask the
            # rest of every program containing a flip.
            return f"tl.flip({args[0]}, {a.get('axis', -1) % len(t.shape)})"
        if op in ('scan_sum', 'scan_product', 'scan_max'):
            fn = {'scan_sum': 'cumsum', 'scan_product': 'cumprod', 'scan_max': 'cummax'}[op]
            # A negative axis is printed as spelled, for the front end to normalize.
            reverse = ', reverse=True' if a.get('reverse') else ''
            return f'tl.{fn}({args[0]}, {spelled_axis(a, t)}{reverse})'
        if op == 'sort':
            # Triton sorts only the minor dimension.
            return f"tl.sort({args[0]}, descending={a.get('descending', False)})"
        if op in ('argmax', 'argmin', 'xor_sum'):
            return f'tl.{op}({args[0]}, {spelled_axis(a, self.ty(scope, args[0]))})'
        if op == 'softmax':
            axis = spelled_axis(a, t)
            if axis == 0 and not a.get('keep_dims'):
                # Also valid on Triton 3.0, whose only softmax positional is
                # ieee_rounding and which always normalizes axis 0.
                return f'tl.softmax({args[0]}, 0)'
            # Keep the reduced dimension when broadcasting a row reduction;
            # this also repairs saved programs that requested keep_dims=False.
            keep = a.get('keep_dims') or (len(t.shape) == 2 and axis % 2 == 1)
            return f"tl.softmax({args[0]}, dim={axis}{', keep_dims=True' if keep else ''})"
        if op == 'topk':
            # Only the minor dimension; descending=True is the default.
            descending = '' if a.get('descending', True) else ', descending=False'
            return f'tl.topk({args[0]}, {a["k"]}, dim={len(t.shape) - 1}{descending})'
        if op == 'load':
            address, mask = self.global_access(a['buffer'], args[0], args[1])
            return f'tl.load({address}, {mask}, other=0)'
        if op == 'store':
            address, mask = self.global_access(a['buffer'], args[0], args[1])
            self.add(indent, f'tl.store({address}, {args[2]}, {mask})')
        elif op.startswith('atomic_'):
            address, mask = self.global_access(a['buffer'], args[0], args[1])
            self.add(indent, f'tl.atomic_{op.split("_", 1)[1]}({address}, {args[2]}, {mask})')
        elif op == 'split':
            # Two results; the elements keep their dtype.
            self.add(indent, f'{names[0]}, {names[1]} = tl.split({args[0]})')
        elif op == 'barrier':
            self.add(indent, 'tl.debug_barrier()')
        elif op == 'call':
            self.add(indent, f"{', '.join(names)} = {a['callee']}_{self.name}({', '.join(args + self.context)})")
        elif op in ('for', 'while', 'if'):
            self.lower_control(node, scope, indent)
        else:
            raise ValueError('Unsupported Triton operation: ' + op)
        return None

    def global_access(self, buffer, index, mask):
        """Address and bounds mask of an indexed access to a (possibly aliased) buffer."""
        buf = self.buffers[buffer]
        root = self.buffers[buf.base or buf.name]
        address = f'{root.name} + bid * {root.size + 32} + {16 + buf.offset} + {index} * {buf.stride}'
        return address, f'({mask} & ({index} >= 0) & ({index} < {buf.size}))'

    def lower_control(self, node, scope, indent):
        """if/for/while: carried values are plain variables copied in and out."""
        op, args, a = node.op, node.operands, node.attrs
        names = [v.name for v in node.results]
        if op == 'if':
            self.add(indent, f'if {args[0]}:')
        else:
            for name, source in zip(names, args[1:]):
                self.add(indent, f'{name} = {source}')
            bound = f'tl.minimum(tl.maximum({args[0]}, 0), {a["max_steps"]})'
            iv = node.regions[0].arguments[0].name
            if op == 'while':
                self.add(indent, f'{iv} = tl.full((), 0, tl.int32)')
                self.add(indent, f'while {iv} < {bound}:')
            else:
                iterator = f'tl.range(0, {bound}, num_stages={self.stages})' if a.get('pipelined') else f'range({bound})'
                self.add(indent, f'for {iv} in {iterator}:')
        for i, child in enumerate(node.regions):
            if i:
                self.add(indent, 'else:')
            child_args = child.arguments if op == 'if' else child.arguments[1:]
            for arg, source in zip(child_args, args[1:] if op == 'if' else names):
                self.add(indent + 1, f'{arg.name} = {source}')
            self.lower(child, scope, indent + 1)
            for name, source in zip(names, child.returns):
                self.add(indent + 1, f'{name} = {source}')
            if op == 'while':
                self.add(indent + 1, f'{iv} += 1')

    def parameters(self):
        """Kernel parameter names, in launch order (see `signature`)."""
        return [b.name for b in self.roots] + ['out_' + v for v in self.watched] + ['steps', 'limit']

    def emit(self):
        for fn in self.program.functions:
            self.add(0, '@triton.jit')
            self.add(0, f"def {fn.name}_{self.name}({', '.join([v.name for v in fn.body.arguments] + self.context)}):")
            self.lower(fn.body, fn.name, 1)
            self.add(1, f"return {', '.join(fn.body.returns)}")
        self.add(0, '@triton.jit')
        self.add(0, f"def {self.name}({', '.join(self.parameters())}):")
        self.add(1, 'bid = tl.program_id(0)')
        self.lower(self.program.body, 'main', 1)
        for name in self.watched:
            t = self.ty('main', name)
            self.add(1, f'tl.store(out_{name} + bid * {t.size + 32} + 16 + {self.indices(t.shape)}, {name})')
        return '\n'.join(self.lines).replace('tl.bool', 'tl.int1')

    def signature(self):
        """ASTSource signature keyed by parameter name.

        Triton 3.0 accepted positional keys ({0: '*fp16', ...}); 3.8 rejects
        anything else with "Signature keys must be string" and resolves each
        key against the kernel's parameter list, so the keys are the emitted
        parameter names of `parameters()`. Name-keyed signatures compile on
        3.0 as well, which keeps one harness running on both pairs.
        """
        dtype = {'float16':'fp16', 'float32':'fp32', 'int32':'i32', 'int8':'i8', 'bool':'i1'}
        pointers = [b.dtype for b in self.roots] + [self.ty('main', v).dtype for v in self.watched]
        types = ['*' + dtype[t] for t in pointers] + ['i32', 'i32']
        return dict(zip(self.parameters(), types))


def compile_source(entries, program):
    lines = ['def prepare_extended(device_compile=True):',
             '    import torch', '    from triton.compiler import ASTSource, compile',
             '    from triton.backends.compiler import GPUTarget',
             "    arch = int(os.environ.get('TILESMITH_CUDA_ARCH', '0'))",
             '    if not arch:',
             '        major, minor = torch.cuda.get_device_capability() if torch.cuda.is_available() else (8, 9)',
             '        arch = major * 10 + minor', '    variants = []']
    roots = [b.name for b in program.buffers if b.base is None]
    for index, (lower, options) in enumerate(entries):
        label = extended_variant_label('triton', index, options)
        # The precision/input_precision markers are lowering knobs and
        # evidence metadata, not triton.compile options.
        compile_options = {k: v for k, v in options.items() if k not in ('precision', 'identity', 'input_precision')}
        lines += [f'    extended_stage("compile", {label!r})',
                  f'    compiled_{index} = compile(ASTSource({lower.name}, {lower.signature()!r}), target=GPUTarget("cuda", arch, 32), options={compile_options!r})',
                  f'    record_extended_compilation({label!r}, compiled_{index}.asm, {options!r})',
                  f'    def launch_{index}(memories, outputs, steps, limit):',
                  f'        arguments = [memories[n] for n in {roots!r}] + [outputs[n] for n in {lower.watched!r}]',
                  f'        compiled_{index}[({program.blocks}, 1, 1)](*arguments, steps, limit)',
                  f'    variants.append(({label!r}, {lower.watched!r}, launch_{index}))']
    lines.append('    return variants')
    return '\n'.join(lines)
