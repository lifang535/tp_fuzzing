from src.ir import LoopKind
from src.ir.region import walk
from src.backends.common.region import _region_layouts, _index_expression, _predicate_expression
from .ops import elementwise_expr

def triton_code(program, name='kernel', suffix=''):
    p = program.spec
    physical, (as0, as1, ao, _), (bs0, bs1, bo, _) = _region_layouts(program)
    lines = ['@triton.jit', f'def {name}(A, B, C):',
             '    by = tl.program_id(0)', '    bx = tl.program_id(1)',
             f'    rows = by * {p.block_M} + tl.arange(0, {p.block_M})',
             f'    cols = bx * {p.block_N} + tl.arange(0, {p.block_N})']
    def add(indent, line):
        lines.append('    ' * indent + line)
    def lower(region, indent, iteration='0'):
        for op in region.operations:
            k, out, args, attrs = op.kind, op.result, op.operands, op.attrs
            if k == 'call':
                add(indent, f'{out} = {attrs["callee"]}{suffix}({", ".join(args + ["by", "bx", iteration])})')
            elif k == 'load':
                address = (f'A + {ao} + rows[:, None] * {as0} + cols[None, :] * {as1}' if physical
                           else f'A + rows[:, None] * {p.N} + cols[None, :]')
                add(indent, f'{out} = tl.load({address}, (rows[:, None] < {p.M}) & (cols[None, :] < {p.N}), other=0).to(tl.float32)')
            elif k == 'gemm':
                # int8 x int8: keep the raw i8 tiles and accumulate in int32
                # (tl.dot requires out_dtype=tl.int32 for integer inputs).
                int8 = p.dtype.value == 'int8'
                acc = 'tl.int32' if int8 else 'tl.float32'
                add(indent, f'{out} = tl.full(({p.block_M}, {p.block_N}), 0, {acc})')
                loop = f'tl.range(0, {(p.K+p.block_K-1)//p.block_K}, num_stages={p.num_stages})' if p.loop_kind == LoopKind.PIPELINED else f'range({(p.K+p.block_K-1)//p.block_K})'
                add(indent, 'for ki in ' + loop + ':')
                add(indent+1, f'ks = ki * {p.block_K} + tl.arange(0, {p.block_K})')
                a_address = (f'A + {ao} + rows[:, None] * {as0} + ks[None, :] * {as1}' if physical
                             else f'A + rows[:, None] * {p.K} + ks[None, :]')
                b_address = (f'B + {bo} + ks[:, None] * {bs0} + cols[None, :] * {bs1}' if physical
                             else f'B + ks[:, None] * {p.N} + cols[None, :]')
                add(indent+1, f'a = tl.load({a_address}, (rows[:, None] < {p.M}) & (ks[None, :] < {p.K}), other=0)')
                add(indent+1, f'b = tl.load({b_address}, (ks[:, None] < {p.K}) & (cols[None, :] < {p.N}), other=0)')
                if int8:
                    add(indent+1, f'{out} = tl.dot(a, b, {out}, out_dtype=tl.int32)')
                else:
                    add(indent+1, f'{out} = tl.dot(a, b, {out})')
            elif k == 'for':
                child = op.regions[0]
                add(indent, f'{out} = {args[0]}')
                add(indent, f'for iter_{out} in range({attrs["trip_count"]}):')
                add(indent+1, f'{child.arguments[0]} = {out}')
                index = f'({attrs.get("start", 0)} + iter_{out} * {attrs.get("step", 1)})'
                lower(child, indent+1, index)
                add(indent+1, f'{out} = {child.yield_value}')
            elif k == 'if':
                add(indent, f'if {_predicate_expression(attrs, iteration)}:')
                for i, child in enumerate(op.regions):
                    if i:
                        add(indent, 'else:')
                    add(indent+1, f'{child.arguments[0]} = {args[0]}')
                    lower(child, indent+1, iteration)
                    add(indent+1, f'{out} = {child.yield_value}')
            elif k.startswith('row_'):
                x = args[0]
                if k == 'row_softmax':
                    add(indent, f'{out}_exp = tl.exp({x} - tl.max({x}, 1)[:, None])')
                    add(indent, f'{out} = {out}_exp / tl.sum({out}_exp, 1)[:, None]')
                else:
                    fn = k.removeprefix('row_')
                    add(indent, f'{out} = tl.broadcast_to(tl.{fn}({x}, 1)[:, None], ({p.block_M}, {p.block_N}))')
            else:
                x = args[0]
                y = args[1] if len(args) > 1 else ''
                z = args[2] if len(args) > 2 else ''
                index_term = _index_expression(attrs["axis"], iteration) if k == 'index_add' else ''
                expr = elementwise_expr(k, x, y, z, attrs, index_term)
                add(indent, f'{out} = {expr}')
    entry = lines[:]
    lines.clear()
    for fn in program.functions:
        lines.extend(['@triton.jit', f'def {fn.name}{suffix}({", ".join(fn.body.arguments + ["by", "bx", "iv"])}):'])
        lower(fn.body, 1, 'iv')
        add(1, f'return {fn.body.yield_value}')
        lines.append('')
    lines.extend(entry)
    lower(program.body, 1)
    add(1, f'tl.store(C + rows[:, None] * {p.N} + cols[None, :], {program.body.yield_value}, (rows[:, None] < {p.M}) & (cols[None, :] < {p.N}))')
    return '\n'.join(lines) + '\n'
