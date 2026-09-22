from src.ir import LoopKind
from src.ir.region import walk
from src.backends.common.region import _region_layouts, _index_expression, _predicate_expression
from .ops import elementwise_expr

def tilelang_code(program, name='make_kernel', decorator='@tilelang.jit', swizzle=False):
    p = program.spec
    gemm = program.body.operations[0].kind == 'gemm'
    physical, (as0, as1, ao, asize), (bs0, bs1, bo, bsize) = _region_layouts(program)
    a_shape = f'({asize},)' if physical else f'({p.M}, {p.K if gemm else p.N})'
    b_shape = f'({bsize},)' if physical else f'({p.K}, {p.N})'
    # Bind dtype at module scope instead of inlining it into the impl source.
    # tilelang's frontend cache keys on inspect.getsource(impl) plus the jit
    # call args, so programs that differ only in dtype then share one cache
    # entry: the second compile reuses the first dtype's kernel and the call
    # fails with "kernel impl input A dtype mismatch". This deliberately keeps
    # the dtype_mismatch bug class reachable (historical single/pipeline
    # emitters relied on the same closure-insensitive pattern).
    lines = [f'dtype = "{p.dtype.value}"',
             # int8 GEMMs accumulate in int32: C and the gemm fragment are
             # int32 while A/B (and their shared tiles) stay int8. Float
             # programs keep their historical C dtype and float32 fragments.
             'c_dtype = "int32" if dtype == "int8" else dtype',
             'frag_dtype = "int32" if dtype == "int8" else "float32"',
             decorator, f'def {name}():', '    @T.prim_func',
             f'    def impl(A: T.Buffer({a_shape}, dtype), B: T.Buffer({b_shape}, dtype), C: T.Buffer(({p.M}, {p.N}), c_dtype)):',
             f'        with T.Kernel({(p.M+p.block_M-1)//p.block_M}, {(p.N+p.block_N-1)//p.block_N}, threads={p.threads}) as (by, bx):']
    def add(indent, line):
        lines.append('    ' * indent + line)
    # Allocate physical storage for SSA values; lexical visibility stays in the IR.
    def allocate(body, indent):
        for op in walk(body):
            add(indent, f'{op.result} = T.alloc_fragment(({p.block_M}, {p.block_N}), frag_dtype)')
            if op.kind == 'tile_transpose':
                add(indent, f'{op.result}_shared = T.alloc_shared(({p.block_M}, {p.block_N}), "float32")')
            if op.kind.startswith('row_'):
                add(indent, f'{op.result}_stat = T.alloc_fragment(({p.block_M},), "float32")')
            for child in op.regions:
                add(indent, f'{child.arguments[0]} = T.alloc_fragment(({p.block_M}, {p.block_N}), "float32")')
    allocate(program.body, 3)
    if gemm:
        add(3, f'As = T.alloc_shared(({p.block_M}, {p.block_K}), dtype)')
        add(3, f'Bs = T.alloc_shared(({p.block_K}, {p.block_N}), dtype)')
        if swizzle:
            # Threadblock rasterization swizzle: reorders which block runs
            # which tile without touching the per-tile math, so the variant
            # stays reference-invariant while exercising a different codegen.
            add(3, 'T.use_swizzle(panel_size=10, order="row")')
    # GemmWarpPolicy variant knob: a non-square policy restructures the warp
    # partition (all warps along M or N) without touching the per-tile math.
    warp_policy_arg = {
        'full_row': ', policy=T.GemmWarpPolicy.FullRow',
        'full_col': ', policy=T.GemmWarpPolicy.FullCol',
    }.get(p.warp_policy, '')
    def lower(region, indent, iteration='0'):
        for op in region.operations:
            k, out, args, attrs = op.kind, op.result, op.operands, op.attrs
            if k == 'call':
                add(indent, f'{attrs["callee"]}({", ".join(args + ["by", "bx", iteration, out])})')
            elif k == 'load':
                add(indent, f'for i, j in T.Parallel({p.block_M}, {p.block_N}):')
                source = (f'A[{ao} + (by * {p.block_M} + i) * {as0} + (bx * {p.block_N} + j) * {as1}]' if physical
                          else f'A[by * {p.block_M} + i, bx * {p.block_N} + j]')
                add(indent+1, f'{out}[i, j] = T.if_then_else(by * {p.block_M} + i < {p.M} and bx * {p.block_N} + j < {p.N}, T.cast({source}, "float32"), T.float32(0))')
            elif k == 'gemm':
                add(indent, f'T.clear({out})')
                loop = f'T.Pipelined({(p.K+p.block_K-1)//p.block_K}, num_stages={p.num_stages})' if p.loop_kind == LoopKind.PIPELINED else f'T.serial({(p.K+p.block_K-1)//p.block_K})'
                add(indent, f'for ki in {loop}:')
                if physical:
                    add(indent+1, f'for i, j in T.Parallel({p.block_M}, {p.block_K}):')
                    add(indent+2, f'As[i, j] = T.if_then_else(by * {p.block_M} + i < {p.M} and ki * {p.block_K} + j < {p.K}, A[{ao} + (by * {p.block_M} + i) * {as0} + (ki * {p.block_K} + j) * {as1}], T.cast(0, dtype))')
                    add(indent+1, f'for i, j in T.Parallel({p.block_K}, {p.block_N}):')
                    add(indent+2, f'Bs[i, j] = T.if_then_else(ki * {p.block_K} + i < {p.K} and bx * {p.block_N} + j < {p.N}, B[{bo} + (ki * {p.block_K} + i) * {bs0} + (bx * {p.block_N} + j) * {bs1}], T.cast(0, dtype))')
                else:
                    add(indent+1, f'T.copy(A[by * {p.block_M}, ki * {p.block_K}], As)')
                    add(indent+1, f'T.copy(B[ki * {p.block_K}, bx * {p.block_N}], Bs)')
                add(indent+1, f'T.gemm(As, Bs, {out}{warp_policy_arg})')
            elif k == 'for':
                child = op.regions[0]
                add(indent, f'T.copy({args[0]}, {out})')
                add(indent, f'for iter_{out} in T.serial({attrs["trip_count"]}):')
                add(indent+1, f'T.copy({out}, {child.arguments[0]})')
                index = f'({attrs.get("start", 0)} + iter_{out} * {attrs.get("step", 1)})'
                lower(child, indent+1, index)
                add(indent+1, f'T.copy({child.yield_value}, {out})')
            elif k == 'if':
                add(indent, f'if {_predicate_expression(attrs, iteration)}:')
                for i, child in enumerate(op.regions):
                    if i:
                        add(indent, 'else:')
                    add(indent+1, f'T.copy({args[0]}, {child.arguments[0]})')
                    lower(child, indent+1, iteration)
                    add(indent+1, f'T.copy({child.yield_value}, {out})')
            elif k == 'tile_transpose':
                # A shared staging tile decouples source/destination fragment
                # layouts; direct fragment indexing can overconstrain reductions.
                add(indent, f'T.copy({args[0]}, {out}_shared)')
                add(indent, f'for i, j in T.Parallel({p.block_M}, {p.block_N}):')
                add(indent+1, f'{out}[i, j] = {out}_shared[j, i]')
            elif k.startswith('row_'):
                x, stat = args[0], out + '_stat'
                if k == 'row_softmax':
                    add(indent, f'T.reduce_max({x}, {stat}, dim=1, clear=True)')
                    add(indent, f'for i, j in T.Parallel({p.block_M}, {p.block_N}):')
                    add(indent+1, f'{out}[i, j] = T.exp({x}[i, j] - {stat}[i])')
                    add(indent, f'T.reduce_sum({out}, {stat}, dim=1, clear=True)')
                    add(indent, f'for i, j in T.Parallel({p.block_M}, {p.block_N}):')
                    add(indent+1, f'{out}[i, j] = {out}[i, j] / {stat}[i]')
                else:
                    fn = k.removeprefix('row_')
                    add(indent, f'T.reduce_{fn}({x}, {stat}, dim=1, clear=True)')
                    add(indent, f'for i, j in T.Parallel({p.block_M}, {p.block_N}):')
                    add(indent+1, f'{out}[i, j] = {stat}[i]')
            else:
                x = f'{args[0]}[i, j]'
                y = f'{args[1]}[i, j]' if len(args) > 1 else ''
                z = f'{args[2]}[i, j]' if len(args) > 2 else ''
                index_term = _index_expression(attrs["axis"], iteration) if k == 'index_add' else ''
                expr = elementwise_expr(k, x, y, z, attrs, index_term)
                add(indent, f'for i, j in T.Parallel({p.block_M}, {p.block_N}):')
                add(indent+1, f'{out}[i, j] = {expr}')
    # Cut the impl's decorator together with its def: the @T.prim_func line
    # sits in its own list item and must travel with def impl(...), or it
    # decorates the first inserted @T.macro (a Macro object, which the eager
    # builder rejects with "got Macro").
    entry_start = next(i for i, line in enumerate(lines) if line.strip() == '@T.prim_func')
    entry = lines[entry_start:]
    del lines[entry_start:]
    for fn in program.functions:
        add(1, '@T.macro')
        add(1, f'def {fn.name}({", ".join(fn.body.arguments + ["by", "bx", "iv", "fn_out"])}):')
        allocate(fn.body, 2)
        lower(fn.body, 2, 'iv')
        add(2, f'T.copy({fn.body.yield_value}, fn_out)')
    lines.extend(entry)
    lower(program.body, 3)
    add(3, f'T.copy({program.body.yield_value}, C[by * {p.block_M}, bx * {p.block_N}])')
    lines.append('    return impl')
    return '\n'.join(lines) + '\n'
