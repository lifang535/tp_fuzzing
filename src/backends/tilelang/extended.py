"""TileLang exploration lowering, with explicit fragments and GEMM staging."""
from src.ir.extended import analyze, walk
from src.workflow.generator.identities import extended_variant_label


class ExtendedLowering:
    def __init__(self, program, name, observe=False, threads=128, stages=1):
        self.program, self.name, self.observe = program, name, observe
        self.threads, self.stages = threads, stages
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

    def ref(self, scope, name, indices=()):
        shape = self.ty(scope, name).shape
        if not shape:
            return name + '[0]'
        indices = list(indices)[-len(shape):]
        if len(indices) != len(shape):
            raise ValueError('Missing indices in TileLang lowering')
        return name + '[' + ', '.join('0' if size == 1 else index for size, index in zip(shape, indices)) + ']'

    def loop(self, indent, shape):
        if not shape:
            return indent, ()
        indices = ('i', 'j')[:len(shape)]
        self.add(indent, f"for {', '.join(indices)} in T.Parallel({', '.join(map(str, shape))}):")
        return indent + 1, indices

    def allocate(self, block, indent):
        args = {v.name for v in block.arguments}
        values = {v.name: v.type for node in walk(block) for v in node.results}
        for node in walk(block):
            for child in node.regions:
                values.update((v.name, v.type) for v in child.arguments)
        for name, ty in values.items():
            if name not in args:
                self.add(indent, f'{name} = T.alloc_fragment({ty.shape or (1,)!r}, "{ty.dtype}")')
        for node in walk(block):
            if node.op == 'slice':
                ty = next(t for (scope, name), t in self.types.items() if name == node.operands[0])
                self.add(indent, f'{node.results[0].name}_view_shared = T.alloc_shared({ty.shape!r}, "{ty.dtype}")')
            elif node.op == 'flip':
                # A reversed fragment read is not a bijective layout the
                # LayoutInferencer can express ("no available layout found");
                # stage the source in shared so the reversal happens on a
                # freely indexable buffer (same trick as slice/reduce).
                ty = next(t for (scope, name), t in self.types.items() if name == node.operands[0])
                self.add(indent, f'{node.results[0].name}_flip_shared = T.alloc_shared({ty.shape!r}, "{ty.dtype}")')
            elif node.op == 'matmul':
                out = node.results[0].name
                for label, arg in zip(('a', 'b'), node.operands):
                    # Scoped type lookup is resolved by the unique SSA names.
                    ty = next(t for (scope, name), t in self.types.items() if name == arg)
                    self.add(indent, f'{out}_{label}_shared = T.alloc_shared({ty.shape!r}, "{ty.dtype}")')
            elif node.op == 'reduce':
                out = node.results[0].name
                ty = next(t for (scope, name), t in self.types.items() if name == node.operands[0])
                self.add(indent, f'{out}_wide = T.alloc_fragment({ty.shape!r}, "{node.results[0].type.dtype}")')
                if len(ty.shape) == 2 and node.attrs['axis'] == 0:
                    self.add(indent, f'{out}_reduce_shared = T.alloc_shared({ty.shape!r}, "{ty.dtype}")')
                if (ty.dtype.startswith('float') and node.attrs['kind'] in ('max', 'min')
                        and not (len(ty.shape) == 2 and node.attrs['axis'] == 0)):
                    self.add(indent, f'{out}_nan = T.alloc_fragment({ty.shape!r}, "int32")')
                    self.add(indent, f'{out}_nan_count = T.alloc_fragment({node.results[0].type.shape or (1,)!r}, "int32")')

    def copy(self, scope, indent, src, dst):
        ty = self.ty(scope, dst)
        level, indices = self.loop(indent, ty.shape)
        self.add(level, f'{self.ref(scope, dst, indices)} = {self.ref(scope, src, indices)}')

    def lower(self, block, scope, indent):
        for node in block.operations:
            op, args, a = node.op, node.operands, node.attrs
            names = [v.name for v in node.results]
            out = names[0] if names else None
            ty = node.results[0].type if names else None
            if op in ('for', 'while', 'if'):
                if op == 'if':
                    self.add(indent, f'if {self.ref(scope, args[0])}:')
                else:
                    for name, src in zip(names, args[1:]):
                        self.copy(scope, indent, src, name)
                    bound = f'T.min(T.max({self.ref(scope, args[0])}, 0), {a["max_steps"]})'
                    iv = node.regions[0].arguments[0].name
                    iterator = out + '_iteration'
                    if op == 'while':
                        # LayoutInference turns superlinear on python-while
                        # bodies carrying fragment traffic (measured: ~24s of
                        # a 62s compile on the mixed family). The bounded
                        # while has no breaks and one trailing increment, so
                        # a serial for-loop is semantically identical and
                        # takes the normal layout path.
                        loop = f'T.serial({bound})'
                    else:
                        loop = f'T.Pipelined({bound}, num_stages={self.stages})' if a.get('pipelined') else f'T.serial({bound})'
                    self.add(indent, f'for {iterator} in {loop}:')
                    self.add(indent + 1, f'{iv}[0] = {iterator}')
                for i, child in enumerate(node.regions):
                    if i:
                        self.add(indent, 'else:')
                    child_args = child.arguments if op == 'if' else child.arguments[1:]
                    for arg, src in zip(child_args, args[1:] if op == 'if' else names):
                        self.copy(scope, indent + 1, src, arg.name)
                    self.lower(child, scope, indent + 1)
                    for name, src in zip(names, child.returns):
                        self.copy(scope, indent + 1, src, name)
                continue
            if op == 'call':
                self.add(indent, f"{a['callee']}_{self.name}({', '.join(args + self.context + names)})")
                continue
            if op == 'barrier':
                self.add(indent, 'T.sync_threads()')
                continue
            if op == 'slice':
                # A proper subset of a fragment is not a bijective layout
                # transform. Materialize it so consumers may choose their own
                # thread mapping (in particular when feeding an MMA operand).
                self.add(indent, f'T.copy({args[0]}, {out}_view_shared)')
                self.add(indent, 'T.sync_threads()')
            if op == 'flip':
                # Same contract as slice: the reversed read below happens on
                # the staged shared copy, never on the source fragment.
                self.add(indent, f'T.copy({args[0]}, {out}_flip_shared)')
                self.add(indent, 'T.sync_threads()')
            if op == 'matmul':
                self.add(indent, f'T.copy({args[0]}, {out}_a_shared)')
                self.add(indent, f'T.copy({args[1]}, {out}_b_shared)')
                self.add(indent, f'T.copy({args[2]}, {out})')
                self.add(indent, f'T.gemm({out}_a_shared, {out}_b_shared, {out})')
                continue
            if op == 'reduce':
                src_ty = self.ty(scope, args[0])
                staged = len(src_ty.shape) == 2 and a['axis'] == 0
                if staged:
                    # The column reduction may require a different fragment
                    # layout from an MMA accumulator and its row consumers.
                    self.add(indent, f'T.copy({args[0]}, {out}_reduce_shared)')
                    self.add(indent, 'T.sync_threads()')
                level, indices = self.loop(indent, src_ty.shape)
                src = f"{out}_reduce_shared[{', '.join(indices)}]" if staged else self.ref(scope, args[0], indices)
                self.add(level, f"{out}_wide[{', '.join(indices)}] = T.cast({src}, '{ty.dtype}')")
                self.add(indent, f"T.reduce_{a['kind']}({out}_wide, {out}, dim={a['axis']}, clear=True)")
                if src_ty.dtype.startswith('float') and a['kind'] in ('max', 'min'):
                    if staged:
                        # A second reduction connected to both the source and
                        # output overconstrains column layouts after MMA. Scan
                        # the already staged input per output lane for NaNs;
                        # the actual numeric operation stays a tile reduction.
                        level, indices = self.loop(indent, ty.shape)
                        self.add(level, f'for nan_row in T.serial({src_ty.shape[0]}):')
                        src = f"{out}_reduce_shared[nan_row, {indices[0]}]"
                        # TIR simplifies self-comparisons algebraically. Use
                        # its NaN intrinsic so this check survives lowering.
                        self.add(level + 1, f'if T.isnan({src}):')
                        self.add(level + 2, f"{self.ref(scope, out, indices)} = T.cast(float('nan'), '{ty.dtype}')")
                        continue
                    level, indices = self.loop(indent, src_ty.shape)
                    src = f"{out}_wide[{', '.join(indices)}]"
                    self.add(level, f"{out}_nan[{', '.join(indices)}] = T.if_then_else(T.isnan({src}), 1, 0)")
                    self.add(indent, f"T.reduce_sum({out}_nan, {out}_nan_count, dim={a['axis']}, clear=True)")
                    level, indices = self.loop(indent, ty.shape)
                    ref = self.ref(scope, out, indices)
                    self.add(level, f"{ref} = T.if_then_else({out}_nan_count[{', '.join(indices) or '0'}] > 0, T.cast(float('nan'), '{ty.dtype}'), {ref})")
                continue
            if op in ('atomic_add', 'atomic_max', 'atomic_min'):
                buf = self.buffers[a['buffer']]
                root = self.buffers[buf.base or buf.name]
                shape = self.ty(scope, args[0]).shape
                level, indices = self.loop(indent, shape)
                refs = [self.ref(scope, arg, indices) for arg in args]
                address = f'{root.name}[bid, {16 + buf.offset} + {refs[0]} * {buf.stride}]'
                fn = op.split('_', 1)[1]
                # Bounds are guarded by the branch, like stores; masked lanes
                # never reach the atomic at all.
                mask = f'({refs[1]} and {refs[0]} >= 0 and {refs[0]} < {buf.size})'
                self.add(level, f'if {mask}:')
                self.add(level + 1, f'T.atomic_{fn}({address}, {refs[2]}, memory_order="relaxed")')
                continue
            shape = ty.shape if ty is not None else self.ty(scope, args[0]).shape
            level, indices = self.loop(indent, shape)
            references = [self.ref(scope, arg, indices) for arg in args] if op not in ('reshape', 'transpose', 'slice', 'flip') else []
            linear = '0' if not shape else indices[0] if len(shape) == 1 else f'({indices[0]} * {shape[1]} + {indices[1]})'
            if op == 'constant':
                expression = repr(a['value'])
            elif op == 'index':
                expression = linear
                if a.get('reverse'):
                    expression = f'({ty.size - 1} - {expression})'
                expression = f"(({expression} + {a.get('shift', 0)}) % {ty.size})"
            elif op == 'parameter':
                expression = {'steps':'steps', 'limit':'limit', 'block':'bid'}[a['name']]
            elif op in ('cast', 'broadcast'):
                expression = references[0]
            elif op == 'reshape':
                old = self.ty(scope, args[0]).shape
                ix = () if not old else (linear,) if len(old) == 1 else (f'({linear}) // {old[1]}', f'({linear}) % {old[1]}')
                expression = self.ref(scope, args[0], ix)
            elif op == 'transpose':
                expression = self.ref(scope, args[0], indices[::-1])
            elif op == 'flip':
                # Reverse the minor dimension; a copied permutation remains
                # a permutation for later store/unique checks. The source was
                # staged in shared memory (fragment reversal has no layout).
                reversed_indices = indices[:-1] + (f'{shape[-1] - 1} - {indices[-1]}',)
                expression = f"{out}_flip_shared[{', '.join('0' if size == 1 else index for size, index in zip(shape, reversed_indices))}]"
            elif op == 'slice':
                positions = ', '.join(f'({i} + {offset})' for i, offset in zip(indices, a['offsets']))
                expression = f'{out}_view_shared[{positions}]'
            elif op in ('add', 'sub', 'mul', 'bitand', 'bitxor', 'lt', 'eq', 'and', 'or'):
                symbol = {'add':'+', 'sub':'-', 'mul':'*', 'bitand':'&', 'bitxor':'^', 'lt':'<', 'eq':'==', 'and':'and', 'or':'or'}[op]
                expression = f'({references[0]} {symbol} {references[1]})'
            elif op == 'fma':
                # TileLang has no scalar fma (T.fma2 is packed x2 only); the
                # plain contraction may or may not fuse under the compiler's
                # pass pipeline, and the fma checker accepts both outcomes.
                expression = f'({references[0]} * {references[1]} + {references[2]})'
            elif op == 'mod':
                expression = f'T.floormod({references[0]}, T.max(T.abs({references[1]}), 1))'
            elif op == 'select':
                expression = f"T.if_then_else({', '.join(references)})"
            elif op in ('load', 'store'):
                buf = self.buffers[a['buffer']]
                root = self.buffers[buf.base or buf.name]
                address = f'{root.name}[bid, {16 + buf.offset} + {references[0]} * {buf.stride}]'
                mask = f'({references[1]} and {references[0]} >= 0 and {references[0]} < {buf.size})'
                if op == 'store':
                    self.add(level, f'if {mask}:')
                    self.add(level + 1, f'{address} = {references[2]}')
                    continue
                expression = f'T.if_then_else({mask}, {address}, T.cast(0, "{ty.dtype}"))'
            else:
                raise ValueError('Unsupported TileLang operation: ' + op)
            self.add(level, f'{self.ref(scope, out, indices)} = T.cast({expression}, "{ty.dtype}")')

    def emit(self):
        self.add(0, '@tilelang.jit')
        self.add(0, f'def {self.name}():')
        for fn in self.program.functions:
            outs = ['result_' + v for v in fn.body.returns]
            self.add(1, '@T.macro')
            self.add(1, f"def {fn.name}_{self.name}({', '.join([v.name for v in fn.body.arguments] + self.context + outs)}):")
            self.allocate(fn.body, 2)
            self.lower(fn.body, fn.name, 2)
            for name, src in zip(outs, fn.body.returns):
                ty = self.ty(fn.name, src)
                level, indices = self.loop(2, ty.shape)
                self.add(level, f"{name}[{', '.join(indices) or '0'}] = {self.ref(fn.name, src, indices)}")
        self.add(1, '@T.prim_func')
        params = [f'{b.name}: T.Buffer(({self.program.blocks}, {b.size + 32}), "{b.dtype}")' for b in self.roots]
        params += [f'out_{name}: T.Buffer(({self.program.blocks}, {self.ty("main", name).size + 32}), "{self.ty("main", name).dtype}")' for name in self.watched]
        params += ['steps: T.int32', 'limit: T.int32']
        self.add(1, f"def impl({', '.join(params)}):")
        self.add(2, f'with T.Kernel({self.program.blocks}, threads={self.threads}) as bid:')
        self.allocate(self.program.body, 3)
        self.lower(self.program.body, 'main', 3)
        for name in self.watched:
            ty = self.ty('main', name)
            level, indices = self.loop(3, ty.shape)
            linear = '0' if not indices else indices[0] if len(indices) == 1 else f'({indices[0]} * {ty.shape[1]} + {indices[1]})'
            self.add(level, f'out_{name}[bid, 16 + {linear}] = {self.ref("main", name, indices)}')
        self.add(1, 'return impl')
        return '\n'.join(self.lines)


def compile_source(entries, program):
    """Emit prepare_extended: sequential lowering, concurrent device compiles.

    Cold device compiles (nvcc subprocesses) dominate the extended harness
    wall time, so they run on a small thread pool while evidence writes and
    launch factories stay sequential in variant order. A compile failure is
    re-raised in variant order after re-writing the progress marker, keeping
    the first-failure and location-inference contracts of the serial version.
    """
    from src.backends.common.knobs import EXTENDED_COMPILE_THREADS
    labels = [extended_variant_label('tilelang', index, options) for index, (lower, options) in enumerate(entries)]
    roots = [b.name for b in program.buffers if b.base is None]
    watched = [lower.watched for lower, _ in entries]
    options = [options for _, options in entries]
    lines = ['def prepare_extended(device_compile=True):', '    import torch', '    import threading',
             '    from tilelang import tvm', '    from tilelang.engine import lower as tilelang_lower',
             "    arch = int(os.environ.get('TILESMITH_CUDA_ARCH', '0'))",
             '    if not arch:',
             '        major, minor = torch.cuda.get_device_capability() if torch.cuda.is_available() else (8, 9)',
             '        arch = major * 10 + minor',
             '    target = tvm.target.Target({"kind": "cuda", "arch": "sm_" + str(arch)})', '    variants = []',
             f'    labels = {labels!r}', f'    compile_options = {options!r}',
             f'    watched = {watched!r}', f'    roots = {roots!r}', '    irs = []']
    # Phase 1 — lowering, sequential: a lowering crash names its variant
    # deterministically through the progress marker it just wrote.
    for index, (lower, _) in enumerate(entries):
        lines += [f'    extended_stage("lowering", labels[{index}])',
                  '    with target:',
                  f'        irs.append({lower.name}.get_tir())',
                  f'    record_extended_compilation(labels[{index}], {{"tir": str(irs[{index}])}}, compile_options[{index}], complete=False)']
    # Phase 2 — device compiles, concurrent: CPU-bound nvcc work overlaps
    # instead of paying each serial compile before the next lowering.
    lines += ['    compiled = [None] * len(irs)', '    lowered = [None] * len(irs)',
              '    errors = [None] * len(irs)',
              '    def compile_variant(start):',
              f'        for index in range(start, len(irs), {EXTENDED_COMPILE_THREADS}):',
              '            try:',
              '                if device_compile:',
              '                    compiled[index] = tilelang.compile(irs[index], target=target, pass_configs=compile_options[index]["pass_configs"])',
              '                else:',
              '                    with target, tvm.transform.PassContext(opt_level=3, config=compile_options[index]["pass_configs"]):',
              '                        lowered[index] = tilelang_lower(irs[index], target=target, enable_device_compile=False)',
              '            except BaseException as error:',
              '                errors[index] = error',
              f'    threads = [threading.Thread(target=compile_variant, args=(start,)) for start in range(min(len(irs), {EXTENDED_COMPILE_THREADS}))]',
              '    for thread in threads:',
              '        thread.start()',
              '    for thread in threads:',
              '        thread.join()',
              '    for index in range(len(irs)):',
              '        if errors[index] is not None:',
              '            extended_stage("device_compile", labels[index])',
              '            raise errors[index]',
              # TileLang moved the CUDA compile callback out of engine/lower.py
              # (0.1.14 keeps it in tilelang/cuda/backend.py, same name and
              # signature) and 0.1.14's engine/lower.py no longer exports it at
              # all. Resolve it at run time so one harness runs on both pairs;
              # an unconditional import here would abort every extended
              # variant, which is the whole track.
              '    try:',
              '        from tilelang.cuda.backend import tilelang_callback_cuda_compile',
              '    except ImportError:',
              '        from tilelang.engine.lower import tilelang_callback_cuda_compile',
              '    def make_launch(index):',
              '        def launch(memories, outputs, steps, limit):',
              '            arguments = [memories[n] for n in roots] + [outputs[n] for n in watched[index]]',
              '            compiled[index](*arguments, steps, limit)',
              '        return launch']
    # Phase 3 — evidence records and launch factories, sequential in variant
    # order so compilation.json and variants keep the historical ordering.
    lines += ['    for index in range(len(irs)):',
              '        if device_compile:',
              '            compiled_index = compiled[index]',
              '            artifact_index = compiled_index.artifact',
              '            artifacts_index = {"cuda": compiled_index.get_kernel_source()}',
              '            if artifact_index is not None:',
              '                artifacts_index["lowered_tir"] = str(artifact_index.device_mod)',
              '            record_extended_compilation(labels[index], artifacts_index, compile_options[index])',
              '            variants.append((labels[index], watched[index], make_launch(index)))',
              '        else:',
              '            lowered_index = lowered[index]',
              '            record_extended_compilation(labels[index], {"lowered_tir": str(lowered_index.device_mod), "cuda": lowered_index.kernel_source}, compile_options[index], complete=False)',
              '            cubin = tilelang_callback_cuda_compile(lowered_index.kernel_source, target, compile_options[index]["pass_configs"])',
              '            record_extended_compilation(labels[index], {"cubin": cubin}, compile_options[index])']
    lines.append('    return variants')
    return '\n'.join(lines)
