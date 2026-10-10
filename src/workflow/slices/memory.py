"""Memory access forms: strides, masks, block pointers, descriptors, hints.

Vectorization, coalescing and alignment analyses specialize a load or store
on what they can prove about its addresses and masks: contiguity and
divisibility hints, strides (transposed or padded views), a ragged or
irregular mask, the fill value of masked lanes, block-pointer boundary
checks and padding, tensor descriptors, cache modifiers and eviction
policies, and atomic-free scatters. A program copies a strided window of a
2-D source tensor into a 2-D output through one access form, optionally
transforming the value, and the reference is the same indexing in torch:
every masked-off output lane holds the fill value and every lane outside
the window keeps the canary.
"""
from .base import Slice
from .dtypes import DTYPES, Domain

MEMORY_DT = ('f32', 'f16', 'bf16', 'f64', 'f8e4', 'f8e5', 'i8', 'i16', 'i32', 'i64', 'u8')
TILES = ((16, 16), (16, 32), (32, 16), (32, 32), (8, 64), (64, 8), (32, 64), (64, 32), (16, 128))
FILLS = (0, 1, -2, 7)


def tile_name(tile):
    return f'{tile[0]}x{tile[1]}'


class MemorySlice(Slice):
    name = 'memory'
    SIMPLEST = {'dt': 'f32', 'tile': '16x16', 'origin': 0, 'row_pad': 0, 'col_extent': 'full',
                'row_extent': 'full', 'view': 'row', 'fill': 0, 'transform': 'none', 'grid': 1,
                'access': {'triton': 'ptr', 'tilelang': 'copy'}, 'mask': 'none', 'hint': 'none',
                'cache': 'none', 'evict': 'none', 'volatile': 0, 'scache': 'none', 'sevict': 'none',
                'padding': 'zero', 'dyn': 0, 'coalesced': 0, 'disable_tma': 0, 'stage': 'fragment'}

    def space(self, backend):
        space = {'dt': MEMORY_DT, 'tile': tuple(tile_name(t) for t in TILES), 'origin': (0, 1, 3, 8),
                 'row_pad': (0, 1, 4, 16), 'col_extent': ('full', 'cut'), 'row_extent': ('full', 'cut'),
                 'view': ('row', 'col'), 'fill': FILLS, 'transform': ('none', 'neg', 'add', 'swap'),
                 'grid': (1, 2, 4), 'pair': (0, 1)}
        if backend == 'triton':
            space.update(access=('ptr', 'bptr', 'desc'), mask=('none', 'bound', 'ragged', 'parity'),
                         hint=('none', 'multiple', 'contig', 'both', 'assume'),
                         cache=('none', '.ca', '.cg', '.cv'), evict=('none', 'evict_first', 'evict_last'),
                         volatile=(0, 1), scache=('none', '.wb', '.cg', '.cs', '.wt'),
                         sevict=('none', 'evict_first', 'evict_last'), padding=('zero', 'nan'),
                         dyn=(0, 1), warps=(1, 2, 4, 8), warps2=(1, 2, 4, 8))
        else:
            space.update(access=('copy', 'loop', 'region'), mask=('none', 'bound', 'ragged'),
                         coalesced=(0, 1, 2, 4, 8), disable_tma=(0, 1),
                         evict=('none', 'evict_first', 'evict_last'), stage=('fragment', 'shared'),
                         threads=(32, 64, 128, 256), threads2=(32, 64, 128, 256))
        return space

    # ---- legalization -------------------------------------------------------
    def legalize(self, params, backend):
        space = self.space(backend)
        for knob, values in space.items():
            if params.get(knob) not in values:
                params[knob] = values[0]
        for knob in list(params):
            if knob not in space:
                del params[knob]
        dt = params['dt']
        d = DTYPES[dt]
        tile = tuple(int(v) for v in params['tile'].split('x'))
        if d.is_fp8 and params['transform'] in ('neg', 'add'):
            params['transform'] = 'none'
        if d.kind == 'uint' and params['transform'] == 'neg':
            params['transform'] = 'add'
        if d.kind == 'uint' and params['fill'] < 0:
            params['fill'] = 1
        if params['transform'] == 'swap' and (tile[0] != tile[1] or params['grid'] != 1):
            params['transform'] = 'none'
        if backend == 'triton':
            self.legalize_triton(params, tile)
        else:
            self.legalize_tilelang(params, tile)
        rows, cols = tile[0] * params['grid'], tile[1]
        cut_rows = rows - rows // 4 - 1 if params['row_extent'] == 'cut' else rows
        cut_cols = cols - cols // 4 - 1 if params['col_extent'] == 'cut' else cols
        if params['view'] == 'col':
            # the source is the transpose of a row-major tensor
            src_shape = (cut_cols + params['origin'], cut_rows + params['origin'] + params['row_pad'])
        else:
            src_shape = (cut_rows + params['origin'], cut_cols + params['origin'] + params['row_pad'])
        if not params['pair']:
            params['warps2' if backend == 'triton' else 'threads2'] = params['warps' if backend == 'triton'
                                                                            else 'threads']
        nonneg = d.kind == 'uint'
        return {'backend': backend, 'tile': tile, 'rows': rows, 'cols': cols, 'cut_rows': cut_rows,
                'cut_cols': cut_cols, 'src_shape': src_shape, 'dt': dt, 'in_dt': dt, 'out_dt': dt,
                'input_domain': Domain(0, 6) if nonneg else Domain(-4, 4)}

    @staticmethod
    def legalize_triton(params, tile):
        access = params['access']
        if access == 'desc':
            # descriptors need 16-byte aligned blocks and row strides
            size = DTYPES[params['dt']].bits // 8
            if (tile[1] * size) % 16 or params['view'] == 'col':
                params['access'] = access = 'bptr'
            else:
                params['origin'] = 0
                cols = tile[1] - tile[1] // 4 - 1 if params['col_extent'] == 'cut' else tile[1]
                pads = [params['row_pad']] + [v for v in (0, 1, 4, 16) if v != params['row_pad']]
                aligned = [v for v in pads if ((cols + v) * size) % 16 == 0]
                if aligned:
                    params['row_pad'] = aligned[0]
                else:
                    params['col_extent'] = 'full'
                    params['row_pad'] = next(v for v in pads if ((tile[1] + v) * size) % 16 == 0)
        if access in ('bptr', 'desc'):
            params['hint'] = 'none'
            params['mask'] = 'bound'
        else:
            params['padding'] = 'zero'
        if access == 'desc':
            params['padding'] = 'zero'
            params['cache'], params['evict'], params['volatile'] = 'none', 'none', 0
            params['scache'], params['sevict'] = 'none', 'none'
        if params['mask'] == 'none':
            params['row_extent'] = params['col_extent'] = 'full'
        if params['mask'] in ('none', 'bound') and params['padding'] == 'zero':
            pass
        if not DTYPES[params['dt']].is_float:
            params['padding'] = 'zero'
        # PTX rejects these combinations outright (ptxas: modifier ... cannot
        # be combined with ...): one cache control per access
        if params['volatile']:
            params['cache'], params['evict'] = 'none', 'none'
        if params['cache'] != 'none':
            params['evict'] = 'none'
        if params['scache'] != 'none':
            params['sevict'] = 'none'
        if params['hint'] in ('multiple', 'both') and params['origin'] % 8:
            params['hint'] = 'contig'
        if params['hint'] == 'assume' and not params['dyn']:
            params['hint'] = 'none'
        if params['fill'] != 0 and params['mask'] == 'none':
            params['fill'] = 0
        if DTYPES[params['dt']].is_fp8 and params['fill'] not in (0, 1):
            params['fill'] = 1

    @staticmethod
    def legalize_tilelang(params, tile):
        access = params['access']
        if params['view'] == 'col':
            params['access'] = access = 'loop'
        if access in ('copy', 'region'):
            params['mask'] = 'bound'
            params['fill'] = 0
        if access == 'loop':
            params['coalesced'], params['disable_tma'], params['evict'] = 0, 0, 'none'
        if params['mask'] == 'none':
            params['row_extent'] = params['col_extent'] = 'full'
        if params['fill'] != 0 and params['mask'] == 'none':
            params['fill'] = 0
        if DTYPES[params['dt']].is_fp8:
            params['transform'] = 'none' if params['transform'] in ('neg', 'add') else params['transform']
            if params['fill'] not in (0, 1):
                params['fill'] = 1
        size = DTYPES[params['dt']].bits // 8
        # a coalesced width must divide the vector the copy can prove aligned
        cols = tile[1] - tile[1] // 4 - 1 if params['col_extent'] == 'cut' else tile[1]
        if params['coalesced'] and (tile[1] % params['coalesced'] or params['coalesced'] * size > 16
                                    or params['origin'] or ((cols + params['row_pad']) * size) % 16
                                    or params['access'] == 'loop'):
            params['coalesced'] = 0
        for knob in ('threads', 'threads2'):
            while params[knob] > 32 and (tile[0] * tile[1]) % params[knob]:
                params[knob] //= 2

    # ---- harness ------------------------------------------------------------
    def input_shapes(self, plan):
        return plan['src_shape'], (1,)

    def plan_data(self, params, plan):
        return {}

    def mask_expr(self, params, r, c, plan, lang):
        """The element mask over global (row, col) of the output tile."""
        mask = params['mask']
        terms = []
        if mask in ('bound', 'ragged', 'parity'):
            terms += [f'({r} < {plan["cut_rows"]})', f'({c} < {plan["cut_cols"]})']
        if mask == 'ragged':
            terms.append(f'({c} <= {r} % {plan["cols"]} + {plan["cols"] // 3})')
        if mask == 'parity':
            terms.append(f'(({r} + {c}) % 3 != 1)')
        joiner = ' & ' if lang == 'triton' else ' and '
        return joiner.join(terms)

    def triton_kernel(self, params, plan):
        rows_tile, cols = plan['tile']
        origin = params['origin']
        src_rows, src_cols = plan['src_shape']
        if params['view'] == 'col':
            row_stride, col_stride = 1, src_cols
        else:
            row_stride, col_stride = src_cols, 1
        fill = params['fill']
        other = f'{float(fill)!r}' if DTYPES[plan['dt']].is_float else f'{fill}'
        lines = ['    pid = tl.program_id(0)', f'    r = pid * {rows_tile} + tl.arange(0, {rows_tile})[:, None]',
                 f'    c = tl.arange(0, {cols})[None, :]']
        access = params['access']
        load_opts = []
        if params['cache'] != 'none':
            load_opts.append(f"cache_modifier='{params['cache']}'")
        if params['evict'] != 'none':
            load_opts.append(f"eviction_policy='{params['evict']}'")
        if params['volatile']:
            load_opts.append('volatile=True')
        store_opts = []
        if params['scache'] != 'none':
            store_opts.append(f"cache_modifier='{params['scache']}'")
        if params['sevict'] != 'none':
            store_opts.append(f"eviction_policy='{params['sevict']}'")
        if access == 'ptr':
            strides = ('rs', 'cs') if params['dyn'] else (row_stride, col_stride)
            lines += [f'    ri = pid * {rows_tile} + tl.arange(0, {rows_tile}) + {origin}',
                      f'    ci = tl.arange(0, {cols}) + {origin}']
            hint = params['hint']
            if hint in ('multiple', 'both'):
                lines += [f'    ri = tl.multiple_of(ri, [{min(rows_tile, origin) if origin else rows_tile}])',
                          f'    ci = tl.multiple_of(ci, [{min(cols, origin) if origin else cols}])']
            if hint in ('contig', 'both'):
                lines += [f'    ri = tl.max_contiguous(ri, [{rows_tile}])', f'    ci = tl.max_contiguous(ci, [{cols}])']
            if hint == 'assume':
                lines += ['    tl.assume(rs > 0)', '    tl.assume(cs > 0)']
            lines.append(f'    offs = ri[:, None] * {strides[0]} + ci[None, :] * {strides[1]}')
            mask = self.mask_expr(params, 'r', 'c', plan, 'triton')
            masking = f', mask={mask}, other={other}' if mask else ''
            opts = ''.join(', ' + o for o in load_opts)
            lines.append(f'    v = tl.load(X + offs{masking}{opts})')
        elif access == 'bptr':
            order = '(1, 0)' if params['view'] == 'row' else '(0, 1)'
            lines += [f'    p = tl.make_block_ptr(X, shape=({plan["cut_rows"] + origin}, {plan["cut_cols"] + origin}), '
                      f'strides=({row_stride}, {col_stride}), offsets=(pid * {rows_tile} + {origin}, {origin}), '
                      f'block_shape=({rows_tile}, {cols}), order={order})']
            padding = f", padding_option='{params['padding']}'"
            opts = ''.join(', ' + o for o in load_opts)
            lines.append(f'    v = tl.load(p, boundary_check=(0, 1){padding}{opts})')
        else:
            lines += [f'    d = tl.make_tensor_descriptor(X, shape=[{plan["cut_rows"]}, {plan["cut_cols"]}], '
                      f'strides=[{row_stride}, 1], block_shape=[{rows_tile}, {cols}])',
                      f'    v = d.load([pid * {rows_tile}, 0])']
        lines += self.triton_transform(params, plan)
        out_offs = f'r * {cols} + c'
        store_mask = f'(r < {plan["rows"]}) & (c < {cols})'
        opts = ''.join(', ' + o for o in store_opts)
        lines.append(f'    tl.store(OUT + {out_offs}, v, mask={store_mask}{opts})')
        signature = 'X, Y, OUT, rs, cs' if params['dyn'] and access == 'ptr' else 'X, Y, OUT'
        launch_args = f', {row_stride}, {col_stride}' if params['dyn'] and access == 'ptr' else ''
        setup = []
        if access == 'desc':
            setup = ['    triton.set_allocator(lambda size, align, stream: torch.empty(size, dtype=torch.int8, '
                     "device='cuda'))"]
        return '\n'.join(['@triton.jit', f'def kernel({signature}):'] + lines + [
            '', 'def launch(X, Y, out, options):'] + setup + [
            f"    kernel[({params['grid']},)](X, Y, out{launch_args}, **options)"])

    @staticmethod
    def triton_transform(params, plan):
        transform = params['transform']
        if transform == 'neg':
            return ['    v = -v']
        if transform == 'add':
            return [f"    v = v + {'1.0' if DTYPES[plan['dt']].is_float else '1'}"]
        if transform == 'swap':
            return ['    v = tl.trans(v)']
        return []

    def tilelang_kernel(self, params, plan):
        from .cast import tilelang_launch
        rows_tile, cols = plan['tile']
        t = DTYPES[plan['dt']].tilelang
        origin = params['origin']
        src_rows, src_cols = plan['src_shape']
        col_view = params['view'] == 'col'
        fill = float(params['fill']) if DTYPES[plan['dt']].is_float else params['fill']
        one = '1.0' if DTYPES[plan['dt']].is_float else '1'
        access = params['access']
        alloc = 'T.alloc_shared' if params['stage'] == 'shared' else 'T.alloc_fragment'
        lines = [f'b = {alloc}(({rows_tile}, {cols}), "{t}")']
        index = (f'X[{origin} + j, bx * {rows_tile} + i + {origin}]' if col_view
                 else f'X[bx * {rows_tile} + i + {origin}, {origin} + j]')
        if access in ('copy', 'region'):
            copy_opts = []
            if params['coalesced']:
                copy_opts.append(f"coalesced_width={params['coalesced']}")
            if params['disable_tma']:
                copy_opts.append('disable_tma=True')
            if params['evict'] != 'none':
                copy_opts.append(f"eviction_policy='{params['evict']}'")
            opts = ''.join(', ' + o for o in copy_opts)
            if col_view:
                # a strided source is gathered element by element
                lines += [f'for i, j in T.Parallel({rows_tile}, {cols}):', f'    b[i, j] = {index}']
            elif access == 'copy':
                lines.append(f'T.copy(X[bx * {rows_tile} + {origin}, {origin}], b{opts})')
            else:
                lines.append(f'T.copy(X[bx * {rows_tile} + {origin}:bx * {rows_tile} + {origin + rows_tile}, '
                             f'{origin}:{origin + cols}], b{opts})')
        else:
            mask = self.mask_expr(params, f'(bx * {rows_tile} + i)', 'j', plan, 'tilelang')
            value = index if not mask else f'T.if_then_else({mask}, {index}, T.cast({fill}, "{t}"))'
            if mask:
                # clamp the address so that masked lanes never read out of bounds
                safe = (f'X[T.min({origin} + j, {src_rows - 1}), T.min(bx * {rows_tile} + i + {origin}, {src_cols - 1})]'
                        if col_view else
                        f'X[T.min(bx * {rows_tile} + i + {origin}, {src_rows - 1}), T.min({origin} + j, {src_cols - 1})]')
                value = f'T.if_then_else({mask}, {safe}, T.cast({fill}, "{t}"))'
            lines += [f'for i, j in T.Parallel({rows_tile}, {cols}):', f'    b[i, j] = {value}']
        transform = params['transform']
        out = 'b'
        if transform != 'none':
            lines.append(f'o = T.alloc_fragment(({rows_tile}, {cols}), "{t}")')
            if transform == 'swap':
                expr = 'b[j, i]'
            elif transform == 'neg':
                expr = '(-b[i, j])'
            elif transform == 'add':
                expr = f'(b[i, j] + T.cast({one}, "{t}"))'
            else:
                expr = 'b[i, j]'
            lines += [f'for i, j in T.Parallel({rows_tile}, {cols}):', f'    o[i, j] = {expr}']
            out = 'o'
        lines.append(f'T.copy({out}, O[bx * {rows_tile}, 0])')
        signature = (f'X: T.Tensor({tuple(plan["src_shape"])!r}, "{t}"), Y: T.Tensor((1,), "{t}"), '
                     f'O: T.Tensor(({plan["rows"]}, {cols}), "{t}")')
        return tilelang_launch(signature, lines, grid=str(params['grid']))

    def reference(self, params, plan):
        rows, cols = plan['rows'], plan['cols']
        rows_tile = plan['tile'][0]
        origin = params['origin']
        backend = plan['backend']
        fill = float(params['fill'])
        lines = ['def reference(x, y):',
                 f'    src = x.t() if {params["view"] == "col"!r} else x',
                 f'    r = torch.arange({rows}).view(-1, 1)',
                 f'    c = torch.arange({cols}).view(1, -1)',
                 f'    rr = torch.clamp(r + {origin}, max=src.shape[0] - 1)',
                 f'    cc = torch.clamp(c + {origin}, max=src.shape[1] - 1)',
                 '    v = src[rr, cc]']
        access = params['access']
        cut = f'(r < {plan["cut_rows"]}) & (c < {plan["cut_cols"]})'
        if backend == 'triton' and access in ('bptr', 'desc'):
            pad = 'float("nan")' if params['padding'] == 'nan' else '0.0'
            lines.append(f'    v = torch.where({cut}, v, torch.full_like(v, {pad}))')
        elif backend == 'tilelang' and access in ('copy', 'region'):
            lines.append(f'    inside = (r + {origin} < src.shape[0]) & (c + {origin} < src.shape[1])')
            lines.append('    v = torch.where(inside, v, torch.zeros_like(v))')
        else:
            mask = self.mask_expr(params, 'r', 'c', plan, 'triton')
            if mask:
                lines.append(f'    v = torch.where({mask}, v, torch.full_like(v, {fill!r}))')
        transform = params['transform']
        if transform == 'neg':
            lines.append('    v = -v')
        elif transform == 'add':
            lines.append('    v = v + 1')
        elif transform == 'swap':
            lines.append(f'    v = v.view({rows_tile}, {cols}).t().contiguous().view({rows}, {cols})')
        lines.append('    return v.contiguous()')
        return '\n'.join(lines)
