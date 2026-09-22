"""Backend-specific typed region lowering."""
from src.ir import LoopKind
from src.ir.region import walk
from src.backends.common.typed import TypedLoweringBase
from .ops import elementwise_expr

class TypedLowering(TypedLoweringBase):

    def __init__(self, program, backend, name, suffix='', decorator='@tilelang.jit', swizzle=False):
        super().__init__(program, backend, name, suffix)
        self.decorator = decorator
        self.swizzle = swizzle

    def address(self, scope, value):
        slot = self.aliases[scope, value]
        (h, w) = self.slots[slot].dimensions(self.p)
        tn = (self.p.N + self.p.block_N - 1) // self.p.block_N
        base = f'(by * {tn} + bx) * {h * w + 32} + 16'
        return f'{self.buffers[slot]}[{base} + i * {w} + j]'

    def allocate(self, body, scope, indent):
        entries = [(arg, self.ty(scope, arg)) for arg in body.arguments]
        for op in walk(body):
            entries.append((op.result, self.ty(scope, op.result)))
            for child in op.regions:
                entries.extend(((arg, self.ty(scope, arg)) for arg in child.arguments))
            if op.kind.startswith('row_') or op.kind == 'reduce_tile':
                (h, w) = self.ty(scope, op.operands[0]).dimensions(self.p)
                axis = op.attrs['axis'] if op.kind == 'reduce_tile' else 1
                self.add(indent, f'{op.result}_stat = T.alloc_fragment(({(h if axis == 1 else w)},), "float32")')
                self.add(indent, f'{op.result}_wide = T.alloc_fragment(({h}, {w}), "float32")')
            if op.kind == 'tile_transpose':
                t = self.ty(scope, op.operands[0])
                self.add(indent, f'{op.result}_shared = T.alloc_shared({t.dimensions(self.p)}, "{t.dtype}")')
        arguments = set(body.arguments)
        for (value, t) in entries:
            if t.kind == 'tensor' and value not in arguments:
                self.add(indent, f'{value} = T.alloc_fragment({t.dimensions(self.p)}, "{t.dtype}")')

    def lower(self, body, scope, indent, iteration='0'):
        p = self.p
        for op in body.operations:
            (k, out, args, a) = (op.kind, op.result, op.operands, op.attrs)
            t = self.ty(scope, out)
            (h, w) = t.dimensions(p)
            if k == 'call':
                argv = args + self.context + ['by', 'bx', iteration]
                self.add(indent, f"{a['callee']}({', '.join(argv + [out])})")
            elif k in ('if', 'for'):
                if k == 'for':
                    self.add(indent, f'T.copy({args[0]}, {out})')
                    self.add(indent, f"for iter_{out} in T.serial({a['trip_count']}):")
                else:
                    condition = f"{self.index(a.get('predicate', 'row'), iteration)} % {a.get('modulus', 2)} == {a['parity']}"
                    self.add(indent, f'if {condition}:')
                for (i, child) in enumerate(op.regions):
                    if i:
                        self.add(indent, 'else:')
                    source = out if k == 'for' else args[0]
                    arg = child.arguments[0]
                    self.add(indent + 1, f'T.copy({source}, {arg})')
                    iv = f"({a.get('start', 0)} + iter_{out} * {a.get('step', 1)})" if k == 'for' else iteration
                    self.lower(child, scope, indent + 1, iv)
                    self.add(indent + 1, f'T.copy({child.yield_value}, {out})')
            elif k in ('load', 'load_input'):
                source = a.get('source', 'A')
                (nr, nc, (s0, s1, offset, _)) = self.input_layout(source)
                r = f'by * {p.block_M} + ' + 'i' + f" + {a.get('row_offset', 0)}"
                c = f'bx * {p.block_N} + ' + 'j' + f" + {a.get('col_offset', 0)}"
                index = f'{source}[{r}, {c}]' if self.logical else f'{source}[{offset} + ({r}) * {s0} + ({c}) * {s1}]'
                self.add(indent, f'for i, j in T.Parallel({h}, {w}):')
                self.add(indent + 1, f'{out}[i, j] = T.if_then_else(({r}) < {nr} and ({c}) < {nc}, T.cast({index}, "{t.dtype}"), T.cast(0, "{t.dtype}"))')
            elif k == 'gemm':
                (_, _, (s0, s1, offset, _)) = self.input_layout('A')
                (_, _, (b0, b1, bo, _)) = self.input_layout('B')
                self.add(indent, f'T.clear({out})')
                iterator = f'T.Pipelined({(p.K + p.block_K - 1) // p.block_K}, num_stages={p.num_stages})' if p.loop_kind == LoopKind.PIPELINED else f'T.serial({(p.K + p.block_K - 1) // p.block_K})'
                self.add(indent, f'for ki in {iterator}:')
                if self.logical:
                    # Logical T.copy loads take the cp.async lowering path, which
                    # crashes for fp16 + pipelined + K % block_K != 0 (the
                    # historical ptx_async_boundary trigger). The physical
                    # guarded loads below never emit cp.async.
                    self.add(indent + 1, f'T.copy(A[by * {p.block_M}, ki * {p.block_K}], As)')
                    self.add(indent + 1, f'T.copy(B[ki * {p.block_K}, bx * {p.block_N}], Bs)')
                else:
                    self.add(indent + 1, f'for i, j in T.Parallel({p.block_M}, {p.block_K}):')
                    self.add(indent + 2, f'As[i, j] = T.if_then_else(by*{p.block_M}+i < {p.M} and ki*{p.block_K}+j < {p.K}, A[{offset}+(by*{p.block_M}+i)*{s0}+(ki*{p.block_K}+j)*{s1}], T.cast(0, dtype))')
                    self.add(indent + 1, f'for i, j in T.Parallel({p.block_K}, {p.block_N}):')
                    self.add(indent + 2, f'Bs[i, j] = T.if_then_else(ki*{p.block_K}+i < {p.K} and bx*{p.block_N}+j < {p.N}, B[{bo}+(ki*{p.block_K}+i)*{b0}+(bx*{p.block_N}+j)*{b1}], T.cast(0, dtype))')
                self.add(indent + 1, f'T.gemm(As, Bs, {out})')
            elif k in ('store_tile', 'write_tile', 'load_tile'):
                handle = args[0] if k == 'load_tile' else out
                address = self.address(scope, handle)
                self.add(indent, 'T.sync_threads()')
                self.add(indent, f'for i, j in T.Parallel({h}, {w}):')
                self.add(indent + 1, f'{out}[i, j] = {address}' if k == 'load_tile' else f'{address} = {args[-1]}[i, j]')
                self.add(indent, 'T.sync_threads()')
            else:
                src = self.ty(scope, args[0])
                (sh, sw) = src.dimensions(p)
                if k == 'tile_transpose':
                    self.add(indent, f'T.copy({args[0]}, {out}_shared)')
                    expr = f'{out}_shared[j, i]'
                elif k.startswith('row_') or k == 'reduce_tile':
                    axis = a['axis'] if k == 'reduce_tile' else 1
                    reduction = a['reduction'] if k == 'reduce_tile' else k[4:]
                    self.add(indent, f'for i, j in T.Parallel({sh}, {sw}):')
                    self.add(indent + 1, f'{out}_wide[i, j] = T.cast({args[0]}[i, j], "float32")')
                    self.add(indent, f"T.reduce_{('max' if reduction == 'softmax' else reduction)}({out}_wide, {out}_stat, dim={axis}, clear=True)")
                    if reduction == 'softmax':
                        self.add(indent, f'for i, j in T.Parallel({h}, {w}):')
                        self.add(indent + 1, f'{out}[i, j] = T.exp({out}_wide[i, j] - {out}_stat[i])')
                        self.add(indent, f'T.reduce_sum({out}, {out}_stat, dim=1, clear=True)')
                        expr = f'{out}[i, j] / {out}_stat[i]'
                    else:
                        expr = f"{out}_stat[{('i' if axis == 1 else 'j')}]"
                else:
                    (ix, jx) = ('0' if sh == 1 else 'i', '0' if sw == 1 else 'j')
                    x = f'T.cast({args[0]}[{ix}, {jx}], "float32")'
                    y = f'T.cast({args[1]}[{ix}, {jx}], "float32")' if len(args) > 1 else ''
                    z = f'T.cast({args[2]}[{ix}, {jx}], "float32")' if len(args) > 2 else ''
                    if k in ('cast', 'to_tile', 'broadcast_tile'):
                        expr = x
                    else:
                        index_term = self.index(a['axis'], iteration) if k == 'index_add' else ''
                        expr = elementwise_expr(k, x, y, z, a, index_term)
                self.add(indent, f'for i, j in T.Parallel({h}, {w}):')
                self.add(indent + 1, f'{out}[i, j] = T.cast({expr}, "{t.dtype}")')

    def emit(self):
        p = self.p
        # Logical mode when both inputs are contiguous: buffers are declared
        # 2D and gemm loads use T.copy, so tilelang lowers them with cp.async
        # (the ptx_async_boundary trigger). Any non-contiguous layout needs
        # flat storage plus manual stride arithmetic instead.
        self.logical = (self.program.execution.input_layout_a == 'contiguous'
                        and self.program.execution.input_layout_b == 'contiguous')
        gemm = self.program.body.operations[0].kind == 'gemm'
        # Bind dtype at module scope instead of inlining it into the impl
        # source. tilelang's frontend cache keys on inspect.getsource(impl),
        # so programs that differ only in dtype then share one cache entry:
        # the second compile reuses the first dtype's kernel and the call
        # fails with "kernel impl input A dtype mismatch". This deliberately
        # keeps the dtype_mismatch bug class reachable.
        self.add(0, f'dtype = "{p.dtype.value}"')
        (tm, tn) = ((p.M + p.block_M - 1) // p.block_M, (p.N + p.block_N - 1) // p.block_N)
        self.add(0, self.decorator)
        self.add(0, f'def {self.name}():')
        for fn in self.program.functions:
            self.add(1, '@T.macro')
            self.add(1, f"def {fn.name}({', '.join(fn.body.arguments + self.context + ['by', 'bx', 'iv', 'fn_out'])}):")
            self.allocate(fn.body, fn.name, 2)
            self.lower(fn.body, fn.name, 2, 'iv')
            self.add(2, f'T.copy({fn.body.yield_value}, fn_out)')
        a_shape = f'({p.M}, {p.K if gemm else p.N})' if self.logical else f'({self.input_layout("A")[2][3]},)'
        b_shape = f'({p.K}, {p.N})' if self.logical else f'({self.input_layout("B")[2][3]},)'
        params = [f'A: T.Buffer({a_shape}, dtype)', f'B: T.Buffer({b_shape}, dtype)']
        for (slot, t) in self.slots.items():
            (h, w) = t.dimensions(p)
            params.append(f'{self.buffers[slot]}: T.Buffer(({tm * tn * (h * w + 32)},), "{t.dtype}")')
        params.append(f'C: T.Buffer(({p.M}, {p.N}), dtype)')
        self.add(1, '@T.prim_func')
        self.add(1, f"def impl({', '.join(params)}):")
        self.add(2, f'with T.Kernel({tm}, {tn}, threads={p.threads}) as (by, bx):')
        self.allocate(self.program.body, 'main', 3)
        if self.program.body.operations[0].kind == 'gemm':
            self.add(3, f'As = T.alloc_shared(({p.block_M}, {p.block_K}), dtype)')
            self.add(3, f'Bs = T.alloc_shared(({p.block_K}, {p.block_N}), dtype)')
            if self.swizzle:
                # Threadblock rasterization swizzle: reorders which block runs
                # which tile without touching the per-tile math, so the variant
                # stays reference-invariant while exercising a different codegen.
                self.add(3, 'T.use_swizzle(panel_size=10, order="row")')
        self.lower(self.program.body, 'main', 3)
        self.add(3, f'for i, j in T.Parallel({p.block_M}, {p.block_N}):')
        self.add(4, f'if by*{p.block_M}+i < {p.M} and bx*{p.block_N}+j < {p.N}:')
        self.add(5, f'C[by*{p.block_M}+i, bx*{p.block_N}+j] = {self.program.body.yield_value}[i, j]')
        self.add(1, 'return impl')
        return '\n'.join(self.lines)
