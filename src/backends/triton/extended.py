"""Triton lowering of typed exploration programs; no TileLang conditionals."""
from src.ir.extended import analyze
from src.workflow.generator.identities import extended_variant_label


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

    def lower(self, block, scope, indent):
        for node in block.operations:
            op, args, a = node.op, node.operands, node.attrs
            names = [v.name for v in node.results]
            out = names[0] if names else None
            t = node.results[0].type if names else None
            expression = None
            if op == 'constant':
                expression = f"tl.full({t.shape!r}, {a['value']!r}, tl.{t.dtype})"
            elif op == 'index':
                expression = self.indices(t.shape)
                if a.get('reverse'):
                    expression = f'({t.size - 1} - {expression})'
                expression = f"(({expression} + {a.get('shift', 0)}) % {t.size})"
            elif op == 'parameter':
                expression = {'steps':'steps', 'limit':'limit', 'block':'bid'}[a['name']]
            elif op == 'cast':
                expression = f'{args[0]}.to(tl.{t.dtype})'
            elif op == 'reshape':
                expression = f'tl.reshape({args[0]}, {t.shape!r})'
            elif op == 'broadcast':
                expression = f'tl.broadcast_to({args[0]}, {t.shape!r})'
            elif op == 'transpose':
                expression = f'tl.trans({args[0]})'
            elif op == 'slice':
                source = args[0]
                shape = list(self.ty(scope, source).shape)
                for axis, (offset, size) in enumerate(zip(a['offsets'], t.shape)):
                    if offset == 0 and size == shape[axis]:
                        continue
                    source = self.slice_axis(source, shape, axis, offset, size,
                                             out + f'_slice{axis}', indent)
                    shape[axis] = size
                expression = source
            elif op in ('add', 'sub', 'mul', 'bitand', 'bitxor', 'lt', 'eq', 'and', 'or'):
                symbol = {'add':'+', 'sub':'-', 'mul':'*', 'bitand':'&', 'bitxor':'^', 'lt':'<', 'eq':'==', 'and':'&', 'or':'|'}[op]
                expression = f'({args[0]} {symbol} {args[1]})'
            elif op == 'mod':
                rem = f'({args[0]} % tl.maximum(tl.abs({args[1]}), 1))'
                expression = f'tl.where({rem} < 0, {rem} + tl.maximum(tl.abs({args[1]}), 1), {rem})'
            elif op == 'select':
                expression = f"tl.where({', '.join(args)})"
            elif op == 'reduce':
                dtype = t.dtype
                expression = f"tl.{a['kind']}({args[0]}.to(tl.{dtype}), {a['axis']})"
                # Triton's default max/min prefer numbers over NaNs, whereas
                # the shared IR propagates any NaN, like the CPU interpreter.
                if dtype.startswith('float') and a['kind'] in ('max', 'min'):
                    expression = f"tl.where(tl.sum(({args[0]} != {args[0]}).to(tl.int32), {a['axis']}) > 0, float('nan'), {expression})"
            elif op == 'matmul':
                # The accumulator dtype selects the accumulation width; the
                # tf32 sweep swaps the input precision on the ieee baseline.
                # Triton defaults out_dtype to float32, which rejects an fp16
                # accumulator, so name it explicitly for the precision pair.
                if t.dtype == 'int32':
                    # int8 operands accumulate in int32; int inputs reject the
                    # input_precision argument entirely.
                    expression = f'tl.dot({args[0]}, {args[1]}, {args[2]}, out_dtype=tl.int32)'
                else:
                    out_dtype = f', out_dtype=tl.{t.dtype}' if t.dtype == 'float16' else ''
                    expression = f"tl.dot({args[0]}, {args[1]}, {args[2]}, input_precision='{self.input_precision}'{out_dtype})"
            elif op == 'fma':
                expression = f'tl.fma({args[0]}, {args[1]}, {args[2]})'
            elif op == 'flip':
                expression = f'tl.flip({args[0]})'
            elif op == 'interleave':
                expression = f'tl.interleave({args[0]}, {args[1]})'
            elif op == 'join':
                expression = f'tl.join({args[0]}, {args[1]})'
            elif op == 'split':
                # Two results bypass the single-expression tail; the elements
                # keep their dtype.
                self.add(indent, f'{names[0]}, {names[1]} = tl.split({args[0]})')
            elif op in ('atomic_add', 'atomic_max', 'atomic_min'):
                buf = self.buffers[a['buffer']]
                root = self.buffers[buf.base or buf.name]
                address = f'{root.name} + bid * {root.size + 32} + {16 + buf.offset} + {args[0]} * {buf.stride}'
                mask = f'({args[1]} & ({args[0]} >= 0) & ({args[0]} < {buf.size}))'
                self.add(indent, f'tl.atomic_{op.split("_", 1)[1]}({address}, {args[2]}, {mask})')
            elif op in ('load', 'store'):
                buf = self.buffers[a['buffer']]
                root = self.buffers[buf.base or buf.name]
                address = f'{root.name} + bid * {root.size + 32} + {16 + buf.offset} + {args[0]} * {buf.stride}'
                mask = f'({args[1]} & ({args[0]} >= 0) & ({args[0]} < {buf.size}))'
                if op == 'load':
                    expression = f'tl.load({address}, {mask}, other=0)'
                else:
                    self.add(indent, f'tl.store({address}, {args[2]}, {mask})')
            elif op == 'barrier':
                self.add(indent, 'tl.debug_barrier()')
            elif op == 'call':
                self.add(indent, f"{', '.join(names)} = {a['callee']}_{self.name}({', '.join(args + self.context)})")
            elif op in ('for', 'while', 'if'):
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
            else:
                raise ValueError('Unsupported Triton operation: ' + op)
            if expression is not None:
                self.add(indent, f'{out} = ({expression}).to(tl.{t.dtype})')

    def emit(self):
        for fn in self.program.functions:
            self.add(0, '@triton.jit')
            self.add(0, f"def {fn.name}_{self.name}({', '.join([v.name for v in fn.body.arguments] + self.context)}):")
            self.lower(fn.body, fn.name, 1)
            self.add(1, f"return {', '.join(fn.body.returns)}")
        self.add(0, '@triton.jit')
        self.add(0, f"def {self.name}({', '.join([b.name for b in self.roots] + ['out_' + v for v in self.watched] + ['steps', 'limit'])}):")
        self.add(1, 'bid = tl.program_id(0)')
        self.lower(self.program.body, 'main', 1)
        for name in self.watched:
            t = self.ty('main', name)
            self.add(1, f'tl.store(out_{name} + bid * {t.size + 32} + 16 + {self.indices(t.shape)}, {name})')
        return '\n'.join(self.lines).replace('tl.bool', 'tl.int1')

    def signature(self):
        dtype = {'float16':'fp16', 'float32':'fp32', 'int32':'i32', 'int8':'i8', 'bool':'i1'}
        pointers = [b.dtype for b in self.roots] + [self.ty('main', v).dtype for v in self.watched]
        return {i: '*' + dtype[t] for i, t in enumerate(pointers)} | {len(pointers):'i32', len(pointers) + 1:'i32'}


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
