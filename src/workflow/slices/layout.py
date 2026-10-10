"""Shape operations chained over a tile in some layout.

Triton lowers every shape operation (reshape, permute, trans, expand_dims /
broadcast_to, join / split / interleave, cat, flip, sort, gather, ravel)
through layout conversions whose correctness depends on the layout the
value arrives in: a load's blocked layout, a value computed from aranges,
an MMA accumulator, a scan, or a reduction broadcast back. TileLang's
equivalents are buffer views and copies between shared memory and register
fragments: T.reshape, T.view, T.transpose, region copies, index remapping,
swizzled shared layouts and reductions broadcast back. A program produces a
tile one way, applies up to three shape operations and stores the result.
Values are small integers whose growth is bounded during legalization, so
the reference (the same operations in torch) is exact.
"""
import itertools
import math

from .base import Slice, parse_shape
from .dtypes import DTYPES, Domain
from .gemm import warp_partition

MAX_NUMEL = 8192
# Largest integer magnitude a dtype holds exactly.
EXACT = {'f16': 2048, 'bf16': 256, 'f32': 2 ** 24, 'f64': 2 ** 53, 'i8': 127, 'i16': 32767, 'i32': 2 ** 31 - 1,
         'i64': 2 ** 63 - 1, 'u8': 255}
TRITON_SOURCES = ('load', 'arange', 'dot', 'cumsum', 'bcast')
TRITON_OPS = ('none', 'reshape', 'permute', 'trans', 'expand', 'join', 'split', 'interleave', 'flip', 'sort',
              'gather', 'cat', 'sumkeep', 'convert', 'ravel')
TRITON_DT = ('f32', 'f16', 'bf16', 'f64', 'i8', 'i16', 'i32', 'i64', 'u8')
TRITON_LAYOUT_SHAPES = ('128', '256', '1024', '16x16', '16x32', '32x16', '64x8', '8x64', '2x128', '32x32', '64x64',
                        '4x8x8', '2x16x4', '2x4x32', '8x2x16')
TILELANG_SOURCES = ('shared', 'fragment', 'gemm', 'reduce')
TILELANG_OPS = ('none', 'transpose', 'reshape', 'view', 'region', 'shift', 'scale', 'rbcast', 'swizzle', 'copy')
TILELANG_DT = ('f16', 'bf16', 'f32', 'f64', 'i8', 'i16', 'i32', 'i64', 'u8')
TILELANG_LAYOUT_SHAPES = ('16x16', '16x32', '32x16', '32x32', '32x64', '64x32', '64x64', '16x128', '128x16', '64x128')
TWIN = {'f16': 'i16', 'bf16': 'i16', 'f32': 'i32', 'f64': 'i64', 'i16': 'f16', 'i32': 'f32', 'i64': 'f64'}
STEPS = 3


def fits(bound, nonneg, dt):
    return bound <= EXACT[dt] and (nonneg or dt != 'u8')


def reshapes(shape, max_rank):
    """Power-of-two shapes of the same size, other than shape itself."""
    numel = math.prod(shape)
    log = numel.bit_length() - 1
    found = []
    for rank in range(1, max_rank + 1):
        for parts in itertools.product(range(1, log + 1), repeat=rank):
            if sum(parts) == log:
                candidate = tuple(2 ** p for p in parts)
                if candidate != tuple(shape):
                    found.append(candidate)
    return found


def nth_permutation(rank, k):
    perms = [p for p in itertools.permutations(range(rank)) if list(p) != list(range(rank))]
    return perms[k % len(perms)]


def arange(dim, rank, size):
    index = ['None'] * rank
    index[dim] = ':'
    suffix = '' if rank == 1 else '[' + ', '.join(index) + ']'
    return f'tl.arange(0, {size}){suffix}'


def nd_offsets(shape):
    strides = [math.prod(shape[i + 1:]) for i in range(len(shape))]
    return ' + '.join(f'{arange(d, len(shape), n)} * {s}' for d, (n, s) in enumerate(zip(shape, strides)))


def torch_index(dim, rank, size):
    view = [1] * rank
    view[dim] = size
    return f'torch.arange({size}).view({tuple(view)!r})'


class LayoutSlice(Slice):
    name = 'layout'
    SIMPLEST = {'source': {'triton': 'load', 'tilelang': 'shared'}, 'dt': 'f32', 'store': 'native',
                'out_dt': 'same', 'op1': 'none', 'op2': 'none', 'op3': 'none', 'arg1': 0, 'arg2': 0,
                'arg3': 0, 'scope1': 'shared', 'scope2': 'shared', 'scope3': 'shared', 'k': 16}

    def space(self, backend):
        if backend == 'triton':
            space = {'source': TRITON_SOURCES, 'dt': TRITON_DT, 'shape': TRITON_LAYOUT_SHAPES, 'k': (16, 32, 64),
                     'store': ('native', 'flat'), 'out_dt': ('same',) + TRITON_DT, 'pair': (0, 1),
                     'warps': (1, 2, 4, 8), 'warps2': (1, 2, 4, 8)}
            ops = TRITON_OPS
        else:
            space = {'source': TILELANG_SOURCES, 'dt': TILELANG_DT, 'shape': TILELANG_LAYOUT_SHAPES,
                     'k': (16, 32, 64), 'out_dt': ('same',) + TILELANG_DT, 'pair': (0, 1),
                     'threads': (32, 64, 128, 256), 'threads2': (32, 64, 128, 256)}
            ops = TILELANG_OPS
        for i in range(1, STEPS + 1):
            space[f'op{i}'] = ops
            space[f'arg{i}'] = tuple(range(8))
            if backend == 'tilelang':
                space[f'scope{i}'] = ('shared', 'fragment')
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
        plan = (self.legalize_triton(params) if backend == 'triton' else self.legalize_tilelang(params))
        plan['backend'] = backend
        plan['input_domain'] = Domain(0 if plan['in_dt'] == 'u8' else -3, 3)
        if not params['pair']:
            params['warps2' if backend == 'triton' else 'threads2'] = params['warps' if backend == 'triton'
                                                                            else 'threads']
        return plan

    def legalize_triton(self, params):
        dt = params['dt']
        source = params['source']
        shape = parse_shape(params['shape'])
        if source == 'dot':
            if dt not in ('f16', 'bf16', 'f32', 'i8'):
                params['dt'] = dt = 'f16'
            if len(shape) != 2 or min(shape) < 16:
                params['shape'] = '32x32'
                shape = (32, 32)
            if dt == 'i8':
                params['k'] = max(params['k'], 32)
            cur, bound = ('i32' if dt == 'i8' else 'f32'), 9 * params['k']
        else:
            params['k'] = 16
            cur, bound = dt, 3
            if source in ('cumsum', 'bcast'):
                axis = params['arg1'] % len(shape)
                grown = 3 * shape[axis] if source == 'cumsum' else 3 * (shape[axis] + 1)
                promoted = {'bf16': 'f32' if source == 'cumsum' else 'bf16', 'i8': 'i32', 'i16': 'i32'}.get(dt, dt)
                if dt == 'u8' or not fits(grown, False, promoted):
                    params['source'] = source = 'load'
                else:
                    cur, bound = promoted, grown
        nonneg = cur == 'u8'
        steps = []
        for i in range(1, STEPS + 1):
            op, arg = params[f'op{i}'], params[f'arg{i}']
            step = self.triton_step(op, arg, shape, cur, bound, nonneg)
            if step is None:
                params[f'op{i}'] = op = 'none'
            if op == 'none':
                params[f'arg{i}'] = 0
                continue
            steps.append(step)
            shape, cur, bound = step['shape'], step['dt'], step['bound']
        out = params['out_dt']
        if out != 'same' and (out == cur or not fits(bound, nonneg, out)):
            params['out_dt'] = out = 'same'
        if len(shape) == 1:
            params['store'] = 'native'
        return {'in_dt': dt, 'source': source, 'shape0': parse_shape(params['shape']), 'steps': steps,
                'final_shape': shape, 'final_dt': cur, 'out_dt': cur if out == 'same' else out,
                'k': params['k'], 'axis0': params['arg1'] % len(parse_shape(params['shape']))}

    def triton_step(self, op, arg, shape, dt, bound, nonneg):
        """One legal step on (shape, dt, bound) or None."""
        rank, numel = len(shape), math.prod(shape)
        step = {'op': op, 'shape': shape, 'dt': dt, 'bound': bound}
        if op == 'none':
            return step
        if op == 'reshape':
            options = reshapes(shape, 4)
            step['shape'] = options[arg % len(options)]
        elif op == 'permute':
            if rank < 2:
                return None
            step['perm'] = nth_permutation(rank, arg)
            step['shape'] = tuple(shape[p] for p in step['perm'])
        elif op == 'trans':
            if rank != 2:
                return None
            step['shape'] = shape[::-1]
        elif op == 'expand':
            if rank > 3 or numel * 2 > MAX_NUMEL:
                return None
            step['axis'] = arg % (rank + 1)
            step['size'] = 4 if arg >= 4 and numel * 4 <= MAX_NUMEL else 2
            step['shape'] = shape[:step['axis']] + (step['size'],) + shape[step['axis']:]
        elif op in ('join', 'interleave', 'cat'):
            if numel * 2 > MAX_NUMEL or (op == 'join' and rank > 3) or not fits(bound + 2, nonneg, dt):
                return None
            step['bound'] = bound + 2
            if op == 'join':
                step['shape'] = shape + (2,)
            elif op == 'interleave':
                step['shape'] = shape[:-1] + (2 * shape[-1],)
            else:
                step['axis'] = arg % rank
                step['shape'] = tuple(2 * n if d == step['axis'] else n for d, n in enumerate(shape))
        elif op == 'split':
            if shape[-1] < 2 or (shape[-1] > 2 and rank > 3):
                return None
            step['pick'] = arg % 3
            if step['pick'] == 2:
                if not fits(2 * bound, nonneg, dt):
                    step['pick'] = 0
                else:
                    step['bound'] = 2 * bound
            step['shape'] = shape[:-1] if shape[-1] == 2 else shape[:-1] + (shape[-1] // 2,)
        elif op == 'flip':
            step['axis'] = arg % rank
        elif op == 'sort':
            step['descending'] = arg % 2
        elif op == 'gather':
            axis = arg % rank
            n = shape[axis]
            choice = (arg // rank) % 3
            m = {0: n, 1: n // 2 if n >= 4 else n, 2: 2 * n}[choice]
            if m * numel // n > MAX_NUMEL:
                m = n
            step['axis'], step['index_len'] = axis, m
            step['coefs'] = tuple(((arg + 3 * d) % 4) or 1 if d == axis else (arg + d) % 3 for d in range(rank))
            step['offset'] = arg % 5
            step['shape'] = tuple(m if d == axis else s for d, s in enumerate(shape))
        elif op == 'sumkeep':
            if nonneg:
                return None
            axis = arg % rank
            result = {'i8': 'i32', 'i16': 'i32'}.get(dt, dt)
            grown = bound * (shape[axis] + 1)
            if not fits(grown, nonneg, result):
                return None
            step.update(axis=axis, dt=result, bound=grown)
        elif op == 'convert':
            targets = [t for t in TRITON_DT if t != dt and fits(bound, nonneg, t)]
            if not targets:
                return None
            step['dt'] = targets[arg % len(targets)]
        elif op == 'ravel':
            if rank == 1:
                return None
            step['shape'] = (numel,)
        return step

    def legalize_tilelang(self, params):
        dt = params['dt']
        source = params['source']
        shape = parse_shape(params['shape'])
        if source == 'gemm':
            if dt not in ('f16', 'bf16', 'i8'):
                params['dt'] = dt = 'f16'
            if dt == 'i8':
                params['k'] = max(params['k'], 32)
            if min(shape) < 32:
                params['shape'] = '32x32'
                shape = (32, 32)
            cur, bound, scope = ('i32' if dt == 'i8' else 'f32'), 9 * params['k'], 'fragment'
        else:
            params['k'] = 16
            cur, bound = dt, 3
            scope = 'fragment' if source in ('fragment', 'reduce') else 'shared'
            if source == 'reduce':
                grown = 3 * (shape[1] + 1)
                if dt == 'u8' or not fits(grown, False, dt):
                    params['source'] = source = 'fragment'
                else:
                    bound = grown
        nonneg = cur == 'u8'
        bits = False
        steps = []
        fragments = [math.prod(shape)] if scope == 'fragment' else []
        for i in range(1, STEPS + 1):
            op, arg, target = params[f'op{i}'], params[f'arg{i}'], params[f'scope{i}']
            if op in ('transpose', 'shift', 'region') and scope == 'fragment':
                # a remap reads shared memory: move the fragment there instead
                params[f'op{i}'], params[f'scope{i}'] = op, target = 'copy', 'shared'
            step = self.tilelang_step(op, arg, target, shape, cur, bound, nonneg, bits, scope)
            if step is None:
                params[f'op{i}'] = op = 'none'
            if op == 'none' or op not in ('transpose', 'region', 'shift', 'scale', 'rbcast', 'copy'):
                params[f'scope{i}'] = 'shared'
            if op == 'none':
                params[f'arg{i}'] = 0
                continue
            if op in ('transpose', 'region', 'shift', 'scale', 'rbcast', 'copy'):
                step['scope'] = params[f'scope{i}']
            steps.append(step)
            shape, cur, bound, bits, scope = step['shape'], step['dt'], step['bound'], step['bits'], step['scope']
            if scope == 'fragment':
                fragments.append(math.prod(shape))
            fragments += step.get('fragments', [])
        out = params['out_dt']
        if out != 'same' and (out == cur or bits or not fits(bound, nonneg, out)):
            params['out_dt'] = out = 'same'
        if out != 'same':
            fragments.append(math.prod(shape))
        smallest = min(fragments) if fragments else 256
        m, n = parse_shape(params['shape'])
        for knob in ('threads', 'threads2'):
            while params[knob] > 32 and (params[knob] > smallest or smallest % params[knob]
                                         or (source == 'gemm' and not warp_partition(m, n, params[knob] // 32,
                                                                                     'Square'))):
                params[knob] //= 2
        return {'in_dt': dt, 'source': source, 'shape0': parse_shape(params['shape']), 'steps': steps,
                'final_shape': shape, 'final_dt': cur, 'out_dt': cur if out == 'same' else out, 'k': params['k'],
                'final_scope': scope}

    def tilelang_step(self, op, arg, target, shape, dt, bound, nonneg, bits, scope):
        rank, numel = len(shape), math.prod(shape)
        step = {'op': op, 'shape': shape, 'dt': dt, 'bound': bound, 'bits': bits, 'scope': scope}
        if op == 'none':
            return step
        # A fragment is distributed over threads: TileLang cannot infer a
        # layout for reading one at remapped indices, so remaps read shared
        # memory (a copy step moves a fragment there first).
        if op == 'transpose':
            if rank != 2 or scope != 'shared':
                return None
            step['shape'] = shape[::-1]
            step['method'] = 'transpose' if target == 'shared' and arg % 2 else 'index'
        elif op == 'reshape':
            options = [s for s in reshapes(shape, 2) if min(s) >= 4 or len(s) == 1]
            if not options:
                return None
            step['shape'] = options[arg % len(options)]
        elif op == 'view':
            if dt not in TWIN or (bits and arg % 2):
                return None
            step['dt'] = TWIN[dt]
            step['bits'] = True
        elif op == 'region':
            if rank != 2 or min(shape) < 8 or scope != 'shared':
                return None
            axis = arg % 2
            step['axis'], step['start'] = axis, (shape[axis] // 2) * ((arg // 2) % 2)
            step['shape'] = (shape[0] // 2, shape[1]) if axis == 0 else (shape[0], shape[1] // 2)
        elif op == 'shift':
            if rank != 2 or scope != 'shared':
                return None
            step['rows'] = (arg % 3) + 1
            step['cols'] = (2 * (arg // 3) + 1, arg % 4)
        elif op == 'scale':
            if bits or not fits(2 * bound + 1, nonneg, dt):
                return None
            step['bound'] = 2 * bound + 1
        elif op == 'rbcast':
            if rank != 2 or bits or nonneg:
                return None
            axis = arg % 2
            grown = bound * (shape[axis] + 1)
            if not fits(grown, nonneg, dt):
                return None
            step['axis'], step['bound'] = axis, grown
            step['fragments'] = [numel, shape[1 - axis]]
            if scope != 'fragment':
                step['fragments'].append(numel)
        elif op == 'swizzle':
            if rank != 2 or dt not in ('f16', 'bf16') or shape[1] < 32 or shape[0] < 8:
                return None
            step['scope'] = 'shared'
        elif op == 'copy':
            if target == scope:
                return None
        return step

    # ---- harness ------------------------------------------------------------
    def input_shapes(self, plan):
        if plan['source'] == 'dot' or plan['source'] == 'gemm':
            m, n = plan['shape0']
            return (m, plan['k']), (plan['k'], n)
        if plan['source'] == 'arange':
            return (1,), (1,)
        return plan['shape0'], (1,)

    def plan_data(self, params, plan):
        return {}

    def triton_kernel(self, params, plan):
        shape0, dt = plan['shape0'], plan['in_dt']
        body = []
        source = plan['source']
        if source == 'dot':
            m, n = shape0
            k = plan['k']
            body += [f'    a = tl.load(X + {nd_offsets((m, k))})', f'    b = tl.load(Y + {nd_offsets((k, n))})',
                     '    x = tl.dot(a, b)']
        elif source == 'arange':
            rank = len(shape0)
            terms = ' + '.join(f'{arange(d, rank, s)} * {d + 2}' for d, s in enumerate(shape0))
            modulus, shift = (4, 0) if dt == 'u8' else (7, 3)
            body.append(f'    x = (tl.broadcast_to({terms}, {shape0!r}) % {modulus} - {shift}).to('
                        f'{DTYPES[dt].triton})')
        else:
            body.append(f'    x = tl.load(X + {nd_offsets(shape0)})')
            if source == 'cumsum':
                body.append(f"    x = tl.cumsum(x, axis={plan['axis0']})")
            elif source == 'bcast':
                body.append(f"    x = x + tl.sum(x, axis={plan['axis0']}, keep_dims=True)")
        shape = shape0
        for step in plan['steps']:
            body += ['    ' + line for line in self.triton_lines(step, shape)]
            shape = step['shape']
        if plan['out_dt'] != plan['final_dt']:
            body.append(f"    x = x.to({DTYPES[plan['out_dt']].triton})")
        if params['store'] == 'flat':
            numel = math.prod(shape)
            body += [f'    x = tl.reshape(x, ({numel},))', f'    tl.store(OUT + tl.arange(0, {numel}), x)']
        else:
            body.append(f'    tl.store(OUT + {nd_offsets(shape)}, x)')
        return '\n'.join(['@triton.jit', 'def kernel(X, Y, OUT):'] + body + [
            '', 'def launch(X, Y, out, options):', '    kernel[(1,)](X, Y, out, **options)'])

    @staticmethod
    def triton_lines(step, shape):
        op, rank = step['op'], len(shape)
        if op == 'reshape':
            return [f"x = tl.reshape(x, {step['shape']!r})"]
        if op == 'permute':
            return [f"x = tl.permute(x, {step['perm']!r})"]
        if op == 'trans':
            return ['x = tl.trans(x)']
        if op == 'expand':
            return [f"x = tl.broadcast_to(tl.expand_dims(x, {step['axis']}), {step['shape']!r})"]
        if op == 'join':
            return ['x = tl.join(x, x + 1)']
        if op == 'interleave':
            return ['x = tl.interleave(x, x + 2)']
        if op == 'cat':
            return [f"x = tl.cat(x, x + 2, dim={step['axis']})"]
        if op == 'split':
            lines = [] if shape[-1] == 2 else [f"x = tl.reshape(x, {shape[:-1] + (shape[-1] // 2, 2)!r})"]
            lines.append('a, b = tl.split(x)')
            lines.append('x = ' + ('a', 'b', 'a + b')[step['pick']])
            return lines
        if op == 'flip':
            return [f"x = tl.flip(x, {step['axis']})"]
        if op == 'sort':
            return [f"x = tl.sort(x, dim={rank - 1}, descending={bool(step['descending'])})"]
        if op == 'gather':
            axis = step['axis']
            index_shape = step['shape']
            terms = ' + '.join(f'{arange(d, rank, s)} * {c}' for d, (s, c) in enumerate(zip(index_shape,
                                                                                          step['coefs'])))
            return [f"idx = (tl.broadcast_to({terms}, {index_shape!r}) + {step['offset']}) % {shape[axis]}",
                    f'x = tl.gather(x, idx, {axis})']
        if op == 'sumkeep':
            return [f"x = x + tl.sum(x, axis={step['axis']}, keep_dims=True)"]
        if op == 'convert':
            return [f"x = x.to({DTYPES[step['dt']].triton})"]
        if op == 'ravel':
            return ['x = tl.ravel(x)']
        raise ValueError(op)

    def reference(self, params, plan):
        lines = ['def reference(x, y):']
        source = plan['source']
        shape = plan['shape0']
        if plan['backend'] == 'tilelang':
            return self.tilelang_reference(params, plan)
        if source == 'dot':
            lines.append('    v = x @ y')
        elif source == 'arange':
            rank = len(shape)
            terms = ' + '.join(f'{torch_index(d, rank, s)} * {d + 2}' for d, s in enumerate(shape))
            modulus, shift = (4, 0) if plan['in_dt'] == 'u8' else (7, 3)
            lines.append(f'    v = (({terms}).expand({shape!r}) % {modulus} - {shift}).double()')
        else:
            lines.append('    v = x.clone()')
            if source == 'cumsum':
                lines.append(f"    v = torch.cumsum(v, {plan['axis0']})")
            elif source == 'bcast':
                lines.append(f"    v = v + v.sum({plan['axis0']}, keepdim=True)")
        for step in plan['steps']:
            lines += ['    ' + line for line in self.torch_lines(step, shape)]
            shape = step['shape']
        if params['store'] == 'flat':
            lines.append('    v = v.reshape(-1)')
        lines.append('    return v.contiguous()')
        return '\n'.join(lines)

    @staticmethod
    def torch_lines(step, shape):
        op, rank = step['op'], len(shape)
        if op in ('reshape', 'ravel'):
            return [f"v = v.reshape({step['shape']!r})"]
        if op == 'permute':
            return [f"v = v.permute({step['perm']!r})"]
        if op == 'trans':
            return ['v = v.t()']
        if op == 'expand':
            return [f"v = v.unsqueeze({step['axis']}).expand({step['shape']!r})"]
        if op == 'join':
            return ['v = torch.stack([v, v + 1], -1)']
        if op == 'interleave':
            return [f"v = torch.stack([v, v + 2], -1).reshape({step['shape']!r})"]
        if op == 'cat':
            return [f"v = torch.cat([v, v + 2], {step['axis']})"]
        if op == 'split':
            lines = [] if shape[-1] == 2 else [f"v = v.reshape({shape[:-1] + (shape[-1] // 2, 2)!r})"]
            return lines + ['v = ' + ('v[..., 0]', 'v[..., 1]', 'v[..., 0] + v[..., 1]')[step['pick']]]
        if op == 'flip':
            return [f"v = v.flip({step['axis']})"]
        if op == 'sort':
            return [f"v = torch.sort(v, dim=-1, descending={bool(step['descending'])}).values"]
        if op == 'gather':
            axis = step['axis']
            index_shape = step['shape']
            terms = ' + '.join(f'{torch_index(d, rank, s)} * {c}' for d, (s, c) in enumerate(zip(index_shape,
                                                                                               step['coefs'])))
            return [f"idx = (({terms}).expand({index_shape!r}) + {step['offset']}) % {shape[axis]}",
                    f'v = torch.gather(v, {axis}, idx)']
        if op == 'sumkeep':
            return [f"v = v + v.sum({step['axis']}, keepdim=True)"]
        if op == 'convert':
            return []
        raise ValueError(op)

    # ---- TileLang -----------------------------------------------------------
    def tilelang_kernel(self, params, plan):
        from .cast import tilelang_launch
        dt = plan['in_dt']
        t = DTYPES[dt].tilelang
        shape = plan['shape0']
        lines = []
        source = plan['source']
        if source == 'gemm':
            m, n = shape
            k = plan['k']
            acc = DTYPES['i32' if dt == 'i8' else 'f32'].tilelang
            lines += [f'As = T.alloc_shared(({m}, {k}), "{t}")', f'Bs = T.alloc_shared(({k}, {n}), "{t}")',
                      'T.copy(X, As)', 'T.copy(Y, Bs)', f'b0 = T.alloc_fragment({shape!r}, "{acc}")', 'T.clear(b0)',
                      'T.gemm(As, Bs, b0)']
            signature = (f'X: T.Tensor(({m}, {k}), "{t}"), Y: T.Tensor(({k}, {n}), "{t}"), ')
        else:
            alloc = 'T.alloc_shared' if source == 'shared' else 'T.alloc_fragment'
            lines += [f'b0 = {alloc}({shape!r}, "{t}")', 'T.copy(X, b0)']
            if source == 'reduce':
                lines += [f'r0 = T.alloc_fragment(({shape[0]},), "{t}")', 'T.reduce_sum(b0, r0, dim=1)',
                          f'for i, j in T.Parallel({shape[0]}, {shape[1]}):', '    b0[i, j] = b0[i, j] + r0[i]']
            signature = f'X: T.Tensor({shape!r}, "{t}"), Y: T.Tensor((1,), "{t}"), '
        current = 'b0'
        scope = 'fragment' if source in ('gemm', 'fragment', 'reduce') else 'shared'
        cur_dt = 'i32' if source == 'gemm' and dt == 'i8' else 'f32' if source == 'gemm' else dt
        for index, step in enumerate(plan['steps'], 1):
            name = f'b{index}'
            lines += self.tilelang_lines(step, current, name, shape, scope, cur_dt)
            current, shape, scope, cur_dt = name, step['shape'], step['scope'], step['dt']
        out_t = DTYPES[plan['out_dt']].tilelang
        if plan['out_dt'] != plan['final_dt']:
            idx = ['i', 'j'][:len(shape)]
            lines += [f'oc = T.alloc_fragment({shape!r}, "{out_t}")',
                      f'for {", ".join(idx)} in T.Parallel({", ".join(map(str, shape))}):',
                      f'    oc[{", ".join(idx)}] = T.cast({current}[{", ".join(idx)}], "{out_t}")']
            current = 'oc'
        lines.append(f'T.copy({current}, O)')
        signature += f'O: T.Tensor({tuple(shape)!r}, "{out_t}")'
        return tilelang_launch(signature, lines)

    @staticmethod
    def tilelang_lines(step, src, dst, shape, scope, dt):
        op = step['op']
        t = DTYPES[step['dt']].tilelang
        alloc = 'T.alloc_shared' if step['scope'] == 'shared' else 'T.alloc_fragment'
        new = f'{dst} = {alloc}({step["shape"]!r}, "{t}")'
        if op == 'transpose':
            if step['method'] == 'transpose':
                return [new, f'T.transpose({src}, {dst})']
            r, c = step['shape']
            return [new, f'for i, j in T.Parallel({r}, {c}):', f'    {dst}[i, j] = {src}[j, i]']
        if op == 'reshape':
            return [f'{dst} = T.reshape({src}, {step["shape"]!r})']
        if op == 'view':
            return [f'{dst} = T.view({src}, {step["shape"]!r}, "{t}")']
        if op == 'region':
            r, c = step['shape']
            r0, c0 = (step['start'], 0) if step['axis'] == 0 else (0, step['start'])
            return [new, f'T.copy({src}[{r0}:{r0 + r}, {c0}:{c0 + c}], {dst})']
        if op == 'shift':
            r, c = shape
            mul, add = step['cols']
            return [new, f'for i, j in T.Parallel({r}, {c}):',
                    f"    {dst}[i, j] = {src}[(i + {step['rows']}) % {r}, (j * {mul} + {add}) % {c}]"]
        if op == 'scale':
            idx = ['i', 'j'][:len(shape)]
            at = '[' + ', '.join(idx) + ']'
            offset = '+' if DTYPES[step['dt']].kind == 'uint' else '-'
            return [new, f'for {", ".join(idx)} in T.Parallel({", ".join(map(str, shape))}):',
                    f'    {dst}{at} = {src}{at} * T.cast(2, "{t}") {offset} T.cast(1, "{t}")']
        if op == 'rbcast':
            r, c = shape
            axis = step['axis']
            lines = []
            if scope != 'fragment':
                lines += [f'{dst}f = T.alloc_fragment({shape!r}, "{t}")', f'T.copy({src}, {dst}f)']
                src = f'{dst}f'
            reduced = (r,) if axis == 1 else (c,)
            lines += [f'{dst}r = T.alloc_fragment({reduced!r}, "{t}")', f'T.reduce_sum({src}, {dst}r, dim={axis})',
                      new, f'for i, j in T.Parallel({r}, {c}):',
                      f"    {dst}[i, j] = {src}[i, j] + {dst}r[{'i' if axis == 1 else 'j'}]"]
            return lines
        if op == 'swizzle':
            return [new, f'T.annotate_layout({{{dst}: tilelang.layout.make_swizzled_layout({dst})}})',
                    f'T.copy({src}, {dst})']
        if op == 'copy':
            return [new, f'T.copy({src}, {dst})']
        raise ValueError(op)

    def tilelang_reference(self, params, plan):
        torch_t = {d: DTYPES[d].torch for d in DTYPES}
        lines = ['def reference(x, y):']
        source, shape, dt = plan['source'], plan['shape0'], plan['in_dt']
        if source == 'gemm':
            acc = 'i32' if dt == 'i8' else 'f32'
            lines.append(f'    v = (x @ y).to(torch.{torch_t[acc]})')
        else:
            lines.append(f'    v = x.to(torch.{torch_t[dt]})')
            if source == 'reduce':
                lines.append(f'    v = (v.double() + v.double().sum(1, keepdim=True)).to(torch.{torch_t[dt]})')
        for step in plan['steps']:
            op = step['op']
            to = f"torch.{torch_t[step['dt']]}"
            if op == 'transpose':
                lines.append('    v = v.t().contiguous()')
            elif op == 'reshape':
                lines.append(f"    v = v.reshape({step['shape']!r})")
            elif op == 'view':
                lines.append(f'    v = v.contiguous().view({to})')
            elif op == 'region':
                r, c = step['shape']
                r0, c0 = (step['start'], 0) if step['axis'] == 0 else (0, step['start'])
                lines.append(f'    v = v[{r0}:{r0 + r}, {c0}:{c0 + c}].contiguous()')
            elif op == 'shift':
                r, c = shape
                mul, add = step['cols']
                lines += [f"    rows = (torch.arange({r}) + {step['rows']}) % {r}",
                          f'    cols = (torch.arange({c}) * {mul} + {add}) % {c}',
                          '    v = v[rows][:, cols].contiguous()']
            elif op == 'scale':
                offset = '+' if DTYPES[step['dt']].kind == 'uint' else '-'
                lines.append(f'    v = (v.double() * 2 {offset} 1).to({to})')
            elif op == 'rbcast':
                lines.append(f"    v = (v.double() + v.double().sum({step['axis']}, keepdim=True)).to({to})")
            shape = step['shape']
        if plan['out_dt'] != plan['final_dt']:
            lines.append(f"    v = v.double().to(torch.{torch_t[plan['out_dt']]})")
        lines.append('    return v.contiguous()')
        return '\n'.join(lines)
