"""Backend-specific typed region lowering."""
from src.ir import LoopKind
from src.ir.region import walk
from src.backends.common.typed import TypedLoweringBase
from .ops import elementwise_expr

class TypedLowering(TypedLoweringBase):

    def address(self, scope, value):
        slot = self.aliases[scope, value]
        (h, w) = self.slots[slot].dimensions(self.p)
        tn = (self.p.N + self.p.block_N - 1) // self.p.block_N
        base = f'(by * {tn} + bx) * {h * w + 32} + 16'
        return f'{self.buffers[slot]} + {base} + tl.arange(0, {h})[:, None] * {w} + tl.arange(0, {w})[None, :]'

    def lower(self, body, scope, indent, iteration='0'):
        p = self.p
        for op in body.operations:
            (k, out, args, a) = (op.kind, op.result, op.operands, op.attrs)
            t = self.ty(scope, out)
            (h, w) = t.dimensions(p)
            if k == 'call':
                argv = args + self.context + ['by', 'bx', iteration]
                self.add(indent, f"{out} = {a['callee']}{self.suffix}({', '.join(argv)})")
            elif k in ('if', 'for'):
                if k == 'for':
                    self.add(indent, f'{out} = {args[0]}')
                    self.add(indent, f"for iter_{out} in range({a['trip_count']}):")
                else:
                    condition = f"{self.index(a.get('predicate', 'row'), iteration)} % {a.get('modulus', 2)} == {a['parity']}"
                    self.add(indent, f'if {condition}:')
                for (i, child) in enumerate(op.regions):
                    if i:
                        self.add(indent, 'else:')
                    source = out if k == 'for' else args[0]
                    arg = child.arguments[0]
                    self.add(indent + 1, f'{arg} = {source}')
                    iv = f"({a.get('start', 0)} + iter_{out} * {a.get('step', 1)})" if k == 'for' else iteration
                    self.lower(child, scope, indent + 1, iv)
                    self.add(indent + 1, f'{out} = {child.yield_value}')
            elif k in ('load', 'load_input'):
                source = a.get('source', 'A')
                (nr, nc, (s0, s1, offset, _)) = self.input_layout(source)
                r = f'by * {p.block_M} + ' + f'tl.arange(0, {h})[:, None]' + f" + {a.get('row_offset', 0)}"
                c = f'bx * {p.block_N} + ' + f'tl.arange(0, {w})[None, :]' + f" + {a.get('col_offset', 0)}"
                addr = f'{offset} + ({r}) * {s0} + ({c}) * {s1}'
                self.add(indent, f'{out} = tl.load({source} + {addr}, (({r}) < {nr}) & (({c}) < {nc}), other=0).to(tl.{t.dtype})')
            elif k == 'gemm':
                (_, _, (s0, s1, offset, _)) = self.input_layout('A')
                (_, _, (b0, b1, bo, _)) = self.input_layout('B')
                self.add(indent, f'{out} = tl.full(({h}, {w}), 0, tl.float32)')
                iterator = f'tl.range(0, {(p.K + p.block_K - 1) // p.block_K}, num_stages={p.num_stages})' if p.loop_kind == LoopKind.PIPELINED else f'range({(p.K + p.block_K - 1) // p.block_K})'
                self.add(indent, f'for ki in {iterator}:')
                self.add(indent + 1, f'ks = ki * {p.block_K} + tl.arange(0, {p.block_K})')
                self.add(indent + 1, f'aa = tl.load(A + {offset} + rows[:, None]*{s0} + ks[None, :]*{s1}, (rows[:, None] < {p.M}) & (ks[None, :] < {p.K}), other=0)')
                self.add(indent + 1, f'bb = tl.load(B + {bo} + ks[:, None]*{b0} + cols[None, :]*{b1}, (ks[:, None] < {p.K}) & (cols[None, :] < {p.N}), other=0)')
                self.add(indent + 1, f'{out} = tl.dot(aa, bb, {out})')
            elif k in ('store_tile', 'write_tile', 'load_tile'):
                handle = args[0] if k == 'load_tile' else out
                address = self.address(scope, handle)
                self.add(indent, 'tl.debug_barrier()')
                if k == 'load_tile':
                    self.add(indent, f'{out} = tl.load({address}, volatile=True)')
                else:
                    self.add(indent, f'tl.store({address}, {args[-1]})')
                self.add(indent, 'tl.debug_barrier()')
            else:
                x = f'{args[0]}.to(tl.float32)'
                y = f'{args[1]}.to(tl.float32)' if len(args) > 1 else ''
                z = f'{args[2]}.to(tl.float32)' if len(args) > 2 else ''
                if k == 'cast':
                    expr = f'{args[0]}.to(tl.{t.dtype})'
                elif k in ('to_tile', 'broadcast_tile'):
                    expr = f'tl.broadcast_to({args[0]}, ({h}, {w}))'
                elif k == 'reduce_tile':
                    expr = f"tl.{a['reduction']}({x}, {a['axis']}, keep_dims=True)"
                elif k == 'tile_transpose':
                    expr = f'tl.trans({args[0]})'
                elif k.startswith('row_'):
                    if k == 'row_softmax':
                        self.add(indent, f'{out}_exp = tl.exp({x} - tl.max({x}, 1, keep_dims=True))')
                        expr = f'{out}_exp / tl.sum({out}_exp, 1, keep_dims=True)'
                    else:
                        expr = f'tl.broadcast_to(tl.{k[4:]}({x}, 1, keep_dims=True), ({h}, {w}))'
                else:
                    index_term = self.index(a['axis'], iteration) if k == 'index_add' else ''
                    expr = elementwise_expr(k, x, y, z, a, index_term)
                self.add(indent, f'{out} = ({expr}).to(tl.{t.dtype})')

    def emit(self):
        p = self.p
        (tm, tn) = ((p.M + p.block_M - 1) // p.block_M, (p.N + p.block_N - 1) // p.block_N)
        for fn in self.program.functions:
            self.add(0, '@triton.jit')
            self.add(0, f"def {fn.name}{self.suffix}({', '.join(fn.body.arguments + self.context + ['by', 'bx', 'iv'])}):")
            self.lower(fn.body, fn.name, 1, 'iv')
            self.add(1, f'return {fn.body.yield_value}')
        self.add(0, '@triton.jit')
        self.add(0, f"def {self.name}({', '.join(self.context + ['C'])}):")
        self.add(1, 'by = tl.program_id(0)')
        self.add(1, 'bx = tl.program_id(1)')
        self.add(1, f'rows = by * {p.block_M} + tl.arange(0, {p.block_M})')
        self.add(1, f'cols = bx * {p.block_N} + tl.arange(0, {p.block_N})')
        self.lower(self.program.body, 'main', 1)
        self.add(1, f'tl.store(C + rows[:, None]*{p.N} + cols[None, :], {self.program.body.yield_value}, (rows[:, None] < {p.M}) & (cols[None, :] < {p.N}))')
        return '\n'.join(self.lines)
