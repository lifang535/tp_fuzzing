"""Independent, JSON-driven reference and checks embedded in reproducers."""


def extended_stage(stage, variant):
    import json
    import os
    from pathlib import Path
    directory = os.environ.get('TILESMITH_ARTIFACT_DIR')
    if directory:
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        (path / 'progress.json').write_text(json.dumps({'stage': stage, 'variant': variant}))


def record_extended_compilation(label, artifacts, options, complete=True):
    import hashlib
    import json
    import os
    import re
    from pathlib import Path
    features, stages = set(), {}
    directory = os.environ.get('TILESMITH_ARTIFACT_DIR')
    settings = []
    if isinstance(options, dict):
        for name, value in options.items():
            if name == 'pass_configs' and isinstance(value, dict):
                settings.extend((key, item) for key, item in value.items()
                                if isinstance(item, (bool, int, str)))
            elif name in ('enable_fp_fusion', 'enable_fast_math', 'num_warps', 'num_stages') \
                    and isinstance(value, (bool, int, str)):
                settings.append((name, value))
    features.update(json.dumps(['compiler_setting', name, value], separators=(',', ':'))
                    for name, value in settings)
    for stage, source in artifacts.items():
        if not isinstance(source, (str, bytes, bytearray)):
            continue
        raw = source.encode() if isinstance(source, str) else source
        stages[stage] = {'sha256': hashlib.sha256(raw).hexdigest(), 'bytes': len(raw)}
        # This is an observed compiler artifact, not a claim about individual
        # pass coverage. Keep stage reachability even when the IR has no tokens.
        features.add(json.dumps(['compiler_stage', stage], separators=(',', ':')))
        if isinstance(source, str) and stage in ('ttir', 'ttgir', 'llir', 'ptx', 'lowered_tir', 'cuda'):
            # Post-lowering vocabulary. The saved input TIR is evidence of the
            # attempted program, not a retained compiler operation.
            tokens = re.findall(r'\b(?:arith|math|scf|cf|tt|ttg|ttng|llvm|nvvm|T)\.[A-Za-z_][\w.]*', source)
            if stage in ('ptx', 'cuda'):
                tokens += re.findall(r'\b(?:mma|ld|st|bar|shfl|cvt|add|mul|setp|selp)(?:\.[A-Za-z0-9_]+)+', source)
            features.update(json.dumps(['compiler', stage, op], separators=(',', ':')) for op in tokens)
            # Lexical neighbours are a bounded IR-combination proxy. They are
            # deliberately not called def-use edges or executed pass coverage.
            features.update(json.dumps(['compiler_pair', stage, left, right], separators=(',', ':'))
                            for left, right in zip(tokens, tokens[1:]) if left != right)
            features.update(json.dumps(['compiler_setting_op', stage, name, value, op], separators=(',', ':'))
                            for name, value in settings for op in set(tokens))
        if directory:
            path = Path(directory)
            path.mkdir(parents=True, exist_ok=True)
            (path / (label + '.' + stage)).write_bytes(raw)
    if directory:
        path = Path(directory) / 'compilation.json'
        records = json.loads(path.read_text()) if path.exists() else []
        previous = next((r for r in records if r['variant'] == label), None)
        if previous is not None:
            stages = dict(previous['stages'], **stages)
            features.update(previous['features'])
            records.remove(previous)
        records.append({'variant': label, 'options': options, 'stages': stages,
                        'features': sorted(features), 'complete': complete})
        path.write_text(json.dumps(records, indent=2))


def extended_inputs(program, seed=0, device='cpu'):
    import torch
    torch.manual_seed(seed)
    result = {}
    for buffer in program['buffers']:
        if buffer['base'] is not None:
            continue
        dtype = getattr(torch, buffer['dtype'])
        size, blocks = buffer['size'], program['blocks']
        storage = torch.full((blocks, size + 32), 19, dtype=dtype, device=device)
        payload = storage[:, 16:-16]
        if buffer['role'] == 'scratch':
            payload.fill_(False if dtype == torch.bool else 0.25 if dtype.is_floating_point else 11)
        elif dtype == torch.bool:
            payload.copy_(torch.randint(0, 2, payload.shape, device=device).bool())
        elif dtype == torch.int32:
            payload.copy_(torch.randint(-7, 8, payload.shape, dtype=dtype, device=device))
            if size == 1:
                payload[:, 0] = torch.arange(blocks, device=device) % 2 * 2 - 1
        elif dtype == torch.int8:
            # torch.randint rejects int8; generate int32 and narrow.
            payload.copy_(torch.randint(-8, 8, payload.shape, dtype=torch.int32, device=device).to(torch.int8))
        elif program['input_pattern'] == 'normal':
            payload.copy_(torch.randn(payload.shape, device=device) * 0.125)
        elif program['input_pattern'] == 'special':
            # Explicit opt-in for exceptional-value regressions. Keep the
            # values in input memory so constant folding cannot erase them.
            table = torch.tensor([float('nan'), float('inf'), -float('inf'),
                                  0., -0., 0.125, -0.125, 0.00006103515625], device=device)
            indices = torch.arange(blocks * size, device=device).reshape(blocks, size)
            payload.copy_(table[indices % len(table)])
        elif program['input_pattern'] == 'boundary':
            # Finite values around rounding boundaries, signed zero and tiny
            # normals. These remain useful through cast and native arithmetic.
            table = torch.tensor([0., -0., 0.125, -0.125, 0.12493896484375,
                                  0.1251220703125, 0.00006103515625, -0.00006103515625], device=device)
            indices = torch.arange(blocks * size, device=device).reshape(blocks, size)
            payload.copy_(table[(indices + seed) % len(table)])
        else:
            payload.copy_(torch.randint(-2, 3, payload.shape, device=device) * 0.125)
        result[buffer['name']] = storage
    return result


def _reference_nudge(op, attrs, args, value, direction):
    """Move an inexact floating-point result by the error of a valid evaluation.

    A kernel evaluates `op` in fp32 (or exactly) and rounds to the result
    dtype, so a valid result differs from the reference by the op's fp32
    error and at most one unit in the last place of the result: contracted
    products, approximate division, roots and transcendentals (absolute
    error for sin, cos and log), and reordered or TF32 accumulation.
    Correctly rounded add/sub and exact operations (comparisons, selection,
    data movement, floor/ceil/round, min/max, casts, integers) are not
    moved; they only propagate the moves of their operands.
    """
    import math
    import torch
    if not value.dtype.is_floating_point:
        return value
    single = 2.0 ** -23
    v = value.double()
    if op in ('mul', 'div', 'sqrt', 'rsqrt'):
        scale = 2 * single * v.abs()
    elif op == 'fma':
        # Unfused evaluation rounds the product first.
        scale = torch.finfo(value.dtype).eps * (args[0].double() * args[1].double()).abs()
    elif op in ('exp', 'exp2', 'tanh', 'erf', 'dsl_sigmoid'):
        scale = 16 * single * v.abs()
    elif op in ('sin', 'cos', 'log', 'log2'):
        scale = 16 * single * v.abs() + 2.0 ** -19 * (args[0] != 0).double()
    elif op == 'softmax':
        scale = (2 * math.log2(max(args[0].shape[attrs.get('axis', -1)], 2)) + 16) * single * v.abs()
    elif op == 'reduce_abssum' or (op == 'reduce' and attrs['kind'] == 'sum'):
        axis = attrs.get('axis', -1)
        terms = args[0].shape[axis]
        scale = (2 * math.log2(max(terms, 2)) + 4) * single * args[0].double().abs().sum(axis)
    elif op == 'scan_sum':
        # Each prefix accumulates the magnitudes it has seen so far.
        axis, reverse = attrs.get('axis', -1), attrs.get('reverse', False)
        magnitude = args[0].double().abs()
        magnitude = magnitude.flip(axis).cumsum(axis).flip(axis) if reverse else magnitude.cumsum(axis)
        scale = (2 * math.log2(max(args[0].shape[axis], 2)) + 4) * single * magnitude
    elif op == 'scan_product':
        scale = (args[0].shape[attrs.get('axis', -1)] + 4) * single * v.abs()
    elif op == 'matmul':
        # fp16 accumulators round every step; the TF32 input-precision
        # sweep rounds fp32 operands to 11 bits.
        unit = max(torch.finfo(value.dtype).eps, 2.0 ** -11 if args[0].dtype == torch.float32 else 0.)
        magnitude = args[0].double().abs() @ args[1].double().abs() + args[2].double().abs()
        scale = (2 * math.log2(max(args[0].shape[-1], 2)) + 4) * unit * magnitude
    else:
        return value
    info = torch.finfo(value.dtype)
    # ldexp computes its power of two in fp32, so the shift is split to
    # reach 2^104 (an fp32 value near the maximum).
    _, exponent = torch.frexp(v)
    half = exponent // 2
    ulp = torch.ldexp(torch.ldexp(torch.full_like(v, info.eps / 2), half), exponent - half)
    ulp = ulp.clamp_min(info.tiny * info.eps)
    step = torch.maximum(scale.reshape(v.shape), ulp * (v != 0))
    moved = v + direction(v.shape) * step
    return torch.where(torch.isfinite(v), moved, v).to(value.dtype)


def extended_reference(program, inputs, steps, limit, perturb=None):
    """Interpret the program for every block; return watched values and memory.

    `perturb` selects a pattern of rounding moves (_reference_nudge) for the
    inexact results: 0 moves every element up, 1 down, larger patterns draw
    seeded per-element signs. Unperturbed evaluation is the reference.
    """
    import torch
    memory = {name: value.clone() for name, value in inputs.items()}
    buffers = {b['name']: b for b in program['buffers']}
    functions = {f['name']: f['body'] for f in program['functions']}
    device = next(iter(inputs.values())).device
    outputs = {}
    signs = torch.Generator().manual_seed(perturb) if perturb is not None else None

    def direction(shape):
        if perturb in (0, 1):
            return 1 - 2 * perturb
        return (torch.randint(0, 2, tuple(shape), generator=signs) * 2 - 1).to(device)

    def run(body, inherited, arguments, bid):
        values = dict(inherited)
        values.update(zip((v['name'] for v in body['arguments']), arguments))
        for node in body['operations']:
            op, a = node['op'], node['attrs']
            args = [values[v] for v in node['operands']]
            types = [v['type'] for v in node['results']]
            out = []
            if op == 'constant':
                out = [torch.full(types[0]['shape'], a['value'], dtype=getattr(torch, types[0]['dtype']), device=device)]
            elif op == 'index':
                import math
                size = math.prod(types[0]['shape'])
                index = torch.arange(size, dtype=torch.int32, device=device)
                if a.get('reverse', False):
                    index = size - 1 - index
                out = [((index + a.get('shift', 0)) % size).reshape(types[0]['shape'])]
            elif op == 'parameter':
                out = [torch.tensor({'steps': steps, 'limit': limit, 'block': bid}[a['name']], dtype=torch.int32, device=device)]
            elif op == 'cast':
                out = [args[0].to(getattr(torch, types[0]['dtype']))]
            elif op in ('neg', 'abs', 'sqrt', 'exp', 'log', 'log2', 'exp2',
                        'rsqrt', 'sin', 'cos', 'floor', 'ceil', 'tanh', 'erf', 'round'):
                x = args[0].float()
                if op == 'neg': value = -x
                elif op == 'abs': value = x.abs()
                elif op == 'sqrt': value = x.abs().sqrt()
                elif op == 'exp': value = x.clamp(-10, 10).exp()
                elif op == 'log': value = x.abs().clamp_min(0.001).log()
                elif op == 'log2': value = x.abs().clamp_min(0.001).log2()
                elif op == 'exp2': value = x.clamp(-10, 10).exp2()
                elif op == 'rsqrt': value = x.abs().clamp_min(1e-6).rsqrt()
                elif op == 'sin': value = x.sin()
                elif op == 'cos': value = x.cos()
                elif op == 'floor': value = x.floor()
                elif op == 'ceil': value = x.ceil()
                elif op == 'tanh': value = x.tanh()
                elif op == 'round': value = x.to(getattr(torch, a['dtype'])).float()
                else: value = torch.erf(x)
                out = [value]
            elif op in ('minimum', 'maximum', 'div'):
                x, y = (value.float() for value in args)
                if op == 'minimum': value = torch.minimum(x, y)
                elif op == 'maximum': value = torch.maximum(x, y)
                else: value = x / y.abs().clamp_min(0.001)
                out = [value]
            elif op == 'reshape':
                out = [args[0].reshape(types[0]['shape'])]
            elif op == 'broadcast':
                out = [args[0].expand(types[0]['shape'])]
            elif op == 'transpose':
                out = [args[0].T]
            elif op == 'slice':
                out = [args[0][tuple(slice(off, off + n) for off, n in zip(a['offsets'], types[0]['shape']))]]
            elif op in ('add', 'sub', 'mul', 'bitand', 'bitxor', 'mod', 'lt', 'eq', 'and', 'or'):
                x, y = args
                if op == 'add': value = x + y
                elif op == 'sub': value = x - y
                elif op == 'mul': value = x * y
                elif op in ('bitand', 'and'): value = x & y
                elif op in ('bitxor',): value = x ^ y
                elif op == 'or': value = x | y
                elif op == 'mod': value = torch.remainder(x, y.abs().clamp_min(1))
                elif op == 'lt': value = x < y
                else: value = x == y
                out = [value]
            elif op == 'select':
                out = [torch.where(*args)]
            elif op == 'reduce':
                value = args[0].float() if args[0].dtype.is_floating_point else args[0]
                if a['kind'] == 'sum': value = value.sum(a['axis'])
                elif a['kind'] == 'max': value = value.amax(a['axis'])
                else: value = value.amin(a['axis'])
                out = [value]
            elif op == 'matmul':
                # Accumulation semantics do not reuse the generated kernel.
                if args[0].dtype == torch.int8:
                    # int8 by int8 with an int32 accumulator is exact.
                    # torch.matmul does not implement integer dtypes; fp64
                    # keeps the 8-bit products and sums exact for the
                    # narrowing cast back to int32.
                    out = [((args[0].to(torch.float64) @ args[1].to(torch.float64))
                            + args[2].to(torch.float64)).to(torch.int32)]
                elif args[2].dtype == torch.float16:
                    # fp16 accumulation: hardware MMA rounds the accumulator
                    # once per k-step, so model per-16 partial products.
                    out = args[2].float()
                    for start in range(0, args[0].shape[1], 16):
                        out = (out + args[0][:, start:start + 16].float()
                               @ args[1][start:start + 16, :].float()).to(torch.float16)
                    out = [out]
                else:
                    out = [args[0].float() @ args[1].float() + args[2]]
            elif op == 'fma':
                # One double-precision reference serves fused and unfused
                # fp16/fp32 evaluation; the fma checker accepts both.
                out = [(args[0].double() * args[1].double() + args[2].double()).to(args[0].dtype)]
            elif op == 'flip':
                out = [torch.flip(args[0], [a.get('axis', -1)])]
            elif op == 'interleave':
                # stack along the minor axis then reshape interleaves the
                # elements: [a0, b0, a1, b1, ...].
                out = [torch.stack((args[0], args[1]), dim=-1).reshape(
                    args[0].shape[:-1] + (args[0].shape[-1] * 2,))]
            elif op == 'join':
                out = [torch.stack((args[0], args[1]), dim=-1)]
            elif op == 'split':
                out = [args[0][..., 0], args[0][..., 1]]
            elif op in ('scan_sum', 'scan_product', 'scan_max'):
                # An absent axis is the minor one; a reverse scan accumulates
                # from the far end, i.e. a forward scan of the flipped axis.
                axis, reverse = a.get('axis', -1), a.get('reverse', False)
                value = torch.flip(args[0], [axis]) if reverse else args[0]
                if op == 'scan_sum': value = torch.cumsum(value, dim=axis)
                elif op == 'scan_product': value = torch.cumprod(value, dim=axis)
                else: value = torch.cummax(value, dim=axis).values
                out = [torch.flip(value, [axis]) if reverse else value]
            elif op == 'sort':
                out = [torch.sort(args[0], dim=-1, descending=a.get('descending', False)).values]
            elif op in ('reduce_abssum', 'reduce_absmax'):
                axis = a.get('axis', -1)
                out = [args[0].abs().sum(axis) if op == 'reduce_abssum' else args[0].abs().amax(axis)]
            elif op == 'histogram':
                out = [torch.bincount(args[0].long(), minlength=16).to(torch.int32)]
            elif op in ('reduce_bitand', 'reduce_bitor', 'reduce_bitxor', 'xor_sum'):
                items = args[0].unbind(a.get('axis', -1))
                acc = items[0]
                for item in items[1:]:
                    if op == 'reduce_bitand': acc = torch.bitwise_and(acc, item)
                    elif op == 'reduce_bitor': acc = torch.bitwise_or(acc, item)
                    else: acc = torch.bitwise_xor(acc, item)
                out = [acc]
            elif op in ('argmax', 'argmin'):
                # Ties resolve to the first index, like tie_break_left=True.
                fn = torch.argmax if op == 'argmax' else torch.argmin
                out = [fn(args[0], dim=a.get('axis', -1)).to(torch.int32)]
            elif op == 'dsl_sigmoid':
                out = [torch.sigmoid(args[0])]
            elif op == 'dsl_clamp':
                out = [torch.clamp(args[0], -0.5, 0.5)]
            elif op == 'softmax':
                # keep_dims only spells the target call; the result is the same.
                out = [torch.softmax(args[0], dim=a.get('axis', -1))]
            elif op == 'topk':
                out = [torch.topk(args[0], a['k'], dim=-1, largest=a.get('descending', True)).values]
            elif op == 'gather':
                out = [torch.gather(args[0], 0, args[1].long())]
            elif op in ('atomic_add', 'atomic_max', 'atomic_min',
                        'atomic_and', 'atomic_or', 'atomic_xor'):
                buf = buffers[a['buffer']]
                root = buf['base'] or buf['name']
                index = args[0].long()
                valid = args[1] & (index >= 0) & (index < buf['size'])
                address = 16 + buf['offset'] + index.clamp(0, buf['size'] - 1) * buf['stride']
                # A masked scatter-reduce models any interleaving of the
                # commutative race; include_self folds the initial scratch
                # value into the reduction exactly like the hardware. The
                # scatter addresses are the payload offsets inside the root
                # row, mirroring the kernels' 16 + offset + index * stride.
                slot = memory[root][bid]
                if op in ('atomic_and', 'atomic_or', 'atomic_xor'):
                    # Integer bitwise updates commute, so this serial order
                    # has the same final memory as any GPU interleaving.
                    fn = {'atomic_and': torch.bitwise_and, 'atomic_or': torch.bitwise_or,
                          'atomic_xor': torch.bitwise_xor}[op]
                    for location, enabled, value in zip(address.reshape(-1), valid.reshape(-1), args[2].reshape(-1)):
                        if bool(enabled):
                            slot[location] = fn(slot[location], value)
                else:
                    kind = {'atomic_add': 'sum', 'atomic_max': 'amax', 'atomic_min': 'amin'}[op]
                    target = slot.double() if slot.dtype.is_floating_point else slot.long()
                    source = args[2][valid].double() if slot.dtype.is_floating_point else args[2][valid].long()
                    target.scatter_reduce_(0, address[valid], source, reduce=kind, include_self=True)
                    slot.copy_(target.to(slot.dtype))
            elif op in ('load', 'store'):
                buf = buffers[a['buffer']]
                root = buf['base'] or buf['name']
                index = args[0].long()
                valid = args[1] & (index >= 0) & (index < buf['size'])
                address = 16 + buf['offset'] + index.clamp(0, buf['size'] - 1) * buf['stride']
                if op == 'load':
                    out = [torch.where(valid, memory[root][bid, address], 0)]
                else:
                    memory[root][bid, address[valid]] = args[2][valid]
            elif op == 'call':
                out, _ = run(functions[a['callee']], {}, args, bid)
            elif op == 'if':
                branch = node['regions'][0 if bool(args[0]) else 1]
                out, _ = run(branch, values, args[1:], bid)
            elif op in ('for', 'while'):
                out = args[1:]
                for iteration in range(max(0, min(int(args[0]), a['max_steps']))):
                    iv = torch.tensor(iteration, dtype=torch.int32, device=device)
                    out, _ = run(node['regions'][0], values, [iv] + out, bid)
            elif op != 'barrier':
                raise ValueError('Unsupported reference operation: ' + op)
            for result, value in zip(node['results'], out):
                value = value.to(getattr(torch, result['type']['dtype'])).reshape(result['type']['shape'])
                if perturb is not None:
                    value = _reference_nudge(op, a, args, value, direction)
                values[result['name']] = value
        return [values[v] for v in body['returns']], values

    wanted = list(dict.fromkeys(program['body']['returns'] + program['observations']))
    for bid in range(program['blocks']):
        _, values = run(program['body'], {}, [], bid)
        for name in wanted:
            outputs.setdefault(name, []).append(values[name].clone())
    return {name: torch.stack(value) for name, value in outputs.items()}, memory


def extended_envelopes(program, inputs, steps, limit, reference, patterns=8):
    """Stack each reference value with its perturbed evaluations.

    `reference` maps watched names and 'memory:'-prefixed buffers to the
    unperturbed interpretation of `program`, which is index 0 of every stack.
    A pattern whose interpretation fails (a perturbation can move an index
    or a trip count) is left out.
    """
    import torch
    stacks = {name: [value] for name, value in reference.items()}
    for pattern in range(patterns):
        try:
            outputs, memory = extended_reference(program, inputs, steps, limit, pattern)
        except Exception:
            continue
        outputs.update(('memory:' + name, value) for name, value in memory.items())
        for name, values in stacks.items():
            values.append(outputs[name].to(values[0].device))
    return {name: torch.stack(values) for name, values in stacks.items()}


def extended_explained(actual, expected, envelope, close):
    """Mismatched elements that the reference itself cannot decide.

    `envelope` stacks the reference with its perturbed evaluations. An
    element is explained when a perturbation moves the reference beyond
    `close` and both compared values lie in the range the evaluations span;
    a NaN or an infinity must be one of the evaluations.
    """
    import torch
    unstable = ~close(envelope[1:], envelope[:1]).all(0)
    if not actual.dtype.is_floating_point:
        low, high = envelope.long().amin(0), envelope.long().amax(0)

        def inside(value):
            return (low <= value.long()) & (value.long() <= high)
    else:
        finite = torch.isfinite(envelope)
        low = torch.where(finite, envelope, float('inf')).amin(0)
        high = torch.where(finite, envelope, -float('inf')).amax(0)

        def inside(value):
            within = finite.any(0) & torch.isfinite(value) & close(value, torch.minimum(torch.maximum(value, low), high))
            special = ((envelope == value) | (torch.isnan(envelope) & torch.isnan(value))).any(0)
            return within | (~torch.isfinite(value) & special)
    return unstable & inside(actual) & inside(expected)


def extended_verdict(actual, expected, label, close, envelope=None):
    """(WRONG RESULT message or None, number of explained mismatches).

    `close` decides each element. When some element mismatches, the
    perturbed reference `envelope` (or a callable computing it) excuses the
    elements extended_explained accepts; the message reports the largest
    remaining difference.
    """
    import torch
    mismatch = ~close(actual, expected)
    if not bool(mismatch.any()):
        return None, 0
    if callable(envelope):
        envelope = envelope()
    real = mismatch
    if envelope is not None and envelope.shape[1:] == actual.shape:
        real = mismatch & ~extended_explained(actual, expected, envelope.to(actual.device), close)
        if not bool(real.any()):
            return None, int(mismatch.sum())
    difference = (actual.to(torch.float64) - expected.to(torch.float64)).abs()
    index = int(torch.where(real, torch.nan_to_num(difference, nan=float('inf')), -1.).reshape(-1).argmax())
    return (f'WRONG RESULT: {label}; max_abs={difference.reshape(-1)[index].item()}; '
            f'index={index}; actual={actual.reshape(-1)[index].item()}; '
            f'expected={expected.reshape(-1)[index].item()}\n'
            f'mismatched={int(real.sum())}/{real.numel()}'), 0


def extended_check(actual, expected, label, matmul=False, envelope=None):
    """Integers and booleans compare exactly, floats within a relative
    tolerance (2% when the program has a matmul) with matching NaN and
    infinities. Returns how many mismatches `envelope` explains
    (extended_verdict); any other mismatch raises."""
    import torch
    if actual.shape != expected.shape or actual.dtype != expected.dtype:
        raise RuntimeError('WRONG RESULT: type/shape mismatch: ' + label)
    tolerance = 0.02 if matmul else 0.002
    if actual.dtype.is_floating_point:
        def close(a, b):
            return torch.isclose(a, b, rtol=tolerance, atol=tolerance * 0.125, equal_nan=True)
    else:
        def close(a, b):
            return a == b
    message, explained = extended_verdict(actual, expected, label, close, envelope)
    if message:
        raise RuntimeError(message)
    return explained


def extended_check_atomic(actual, expected, label, kind, envelope=None):
    """Race-aware scratch comparison. add races accumulate in an arbitrary
    order, so the fp64 reference differs by a few rounding ulps (tolerance),
    while max/min races are order-independent and compare exactly except for
    signed zero (which both orderings may pick). `envelope` as in
    extended_check."""
    import torch
    if actual.shape != expected.shape or actual.dtype != expected.dtype:
        raise RuntimeError('WRONG RESULT: type/shape mismatch: ' + label)
    if not actual.dtype.is_floating_point:
        def close(a, b):
            return a == b
    elif kind in ('max', 'min'):
        def close(a, b):
            return (a == b) | (torch.isnan(a) & torch.isnan(b))
    else:
        # n <= 64 raced contributions per address differ by a few ulps of
        # ordering; losing a contribution moves by an order of magnitude.
        def close(a, b):
            return torch.isclose(a, b, rtol=0.02, atol=0.02 * 0.125, equal_nan=True)
    message, explained = extended_verdict(actual, expected, label, close, envelope)
    if message:
        raise RuntimeError(message)
    return explained


def extended_check_fma(actual, expected, label, matmul=False, envelope=None):
    """A fused and an unfused evaluation both round within ~1 ulp of the exact
    product-add; 4 ulps accepts either while rejecting operand mixups and
    exponent bugs. Exceptional values compare exactly like extended_check.
    The matmul flag is accepted for call-site uniformity and ignored."""
    import torch
    if actual.shape != expected.shape or actual.dtype != expected.dtype:
        raise RuntimeError('WRONG RESULT: type/shape mismatch: ' + label)
    if not actual.dtype.is_floating_point:
        raise RuntimeError('WRONG RESULT: fma on non-float: ' + label)
    info = torch.finfo(actual.dtype)

    def close(a, b):
        tolerance = 4 * torch.where(b == 0, info.tiny, b.abs()) * info.eps
        same = (torch.isnan(a) & torch.isnan(b)) | (torch.isinf(b) & (a == b))
        return same | (torch.isfinite(b) & ((a - b).abs() <= tolerance))
    message, explained = extended_verdict(actual, expected, label, close, envelope)
    if message:
        raise RuntimeError(message)
    return explained


def _atomic_kind(program):
    for node in program['body']['operations'] + [n for f in program['functions'] for n in f['body']['operations']]:
        if node['op'].startswith('atomic_'):
            return node['op'].split('_', 1)[1]
    return None


def run_extended(program, prepare, input_seed=0, repeats=2, device='cuda', reference_programs=None):
    import torch
    import math
    types = {v['name']: v['type'] for node in program['body']['operations'] for v in node['results']}
    has_matmul = any(n['op'] == 'matmul' for n in program['body']['operations'])
    atomic_kind = _atomic_kind(program)
    fma_names = {v['name'] for node in program['body']['operations']
                 + [n for f in program['functions'] for n in f['body']['operations']]
                 if node['op'] == 'fma' for v in node['results']}
    # Fused or unfused evaluation may differ per compile configuration, so the
    # checked fma outputs and the invariance baseline both use the fma check.
    check = lambda name: extended_check_fma if name in fma_names else extended_check
    # Every check of one launch runs, so the report names all mismatching
    # values; the first failure in check order is raised. Mismatches that
    # the perturbed reference explains are reported only if nothing fails.
    failures, unstable, unstable_label = [], 0, None

    def judge(name, check_fn, *args):
        nonlocal unstable, unstable_label
        try:
            explained = check_fn(*args)
        except RuntimeError as error:
            if not str(error).startswith('WRONG RESULT'):
                raise
            failures.append((name, error))
            return
        if explained and unstable_label is None:
            unstable_label = args[2]
        unstable += explained
    # Compile once, reuse each signature for all runtime shapes/bounds and seeds.
    variants = prepare()
    extended_stage('reference', 'cpu')
    for seed in (input_seed, input_seed + 1):
        # The reference always runs on CPU, independently of GPU codegen/TF32.
        host_original = extended_inputs(program, seed, 'cpu')
        original = {name: value.to(device) for name, value in host_original.items()}
        for steps, limit in program['runtime_cases']:
            expected, expected_memory = extended_reference(program, host_original, steps, limit)
            expected = {name: value.to(device) for name, value in expected.items()}
            expected_memory = {name: value.to(device) for name, value in expected_memory.items()}
            # Precision variants run an fp16-accumulation copy of the program;
            # their expected values, output dtypes and scratch memories come
            # from that copy and never feed the cross-variant invariance
            # baseline (a different accumulation width legitimately differs).
            alternates = {}
            for label, reference in (reference_programs or {}).items():
                reference_expected, reference_memory = extended_reference(reference, host_original, steps, limit)
                alternates[label] = (
                    {name: value.to(device) for name, value in reference_expected.items()},
                    {name: value.to(device) for name, value in reference_memory.items()},
                    {v['name']: v['type'] for node in reference['body']['operations'] for v in node['results']})
            # Perturbed references of the program (None) or of an alternate,
            # computed for the first mismatch of this case that needs them.
            envelopes = {}

            def envelope(key, name):
                if key not in envelopes:
                    reference, outputs, memory = ((program, expected, expected_memory) if key is None
                                                  else (reference_programs[key],) + alternates[key][:2])
                    outputs = dict(outputs)
                    outputs.update(('memory:' + n, value) for n, value in memory.items())
                    envelopes[key] = extended_envelopes(reference, host_original, steps, limit, outputs)
                return envelopes[key].get(name)
            baselines = {}
            for label, watched, launch in variants:
                extended_stage('execute', label)
                alternate = alternates.get(label)
                if alternate is not None:
                    variant_expected, variant_expected_memory, variant_types = alternate
                    # The label suffix tells which transformed reference this
                    # variant runs: `_prec` -> fp16 accumulation, `_ident` ->
                    # distributivity (checked against its own interpretation,
                    # never against the cross-variant invariance baseline).
                    prefix = 'identity:' if label.endswith('_ident') else 'precision:'
                    key = label
                else:
                    variant_expected, variant_expected_memory, variant_types = expected, expected_memory, types
                    prefix, key = '', None
                previous = None
                for repeat in range(repeats):
                    memories = {name: value.clone() for name, value in original.items()}
                    output_storage = {name: torch.full((program['blocks'], math.prod(variant_types[name]['shape']) + 32),
                                      23, dtype=getattr(torch, variant_types[name]['dtype']), device=device) for name in watched}
                    # For bool/int too, a missing write must differ from reference.
                    for name, storage in output_storage.items():
                        ref = variant_expected[name].reshape(program['blocks'], -1)
                        poison = ~ref if ref.dtype == torch.bool else ref + 1 if not ref.dtype.is_floating_point else torch.where(torch.isnan(ref), 0., float('nan'))
                        storage[:, 16:-16].copy_(poison)
                    launch(memories, output_storage, steps, limit)
                    if device != 'cpu':
                        torch.cuda.synchronize()
                    values, failures = {}, []
                    for name, storage in output_storage.items():
                        guard = True if storage.dtype == torch.bool else 23
                        if not torch.all(storage[:, :16] == guard) or not torch.all(storage[:, -16:] == guard):
                            failures.append((name, RuntimeError('WRONG RESULT: output canary: ' + label)))
                        actual = storage[:, 16:-16].reshape(variant_expected[name].shape)
                        perturbed = lambda: envelope(key, name)
                        if name in fma_names:
                            judge(name, extended_check_fma, actual, variant_expected[name],
                                  f'{prefix}fma:{label}:{name}:seed={seed}:steps={steps}:limit={limit}', False, perturbed)
                        else:
                            judge(name, extended_check, actual, variant_expected[name],
                                  f'{prefix}{label}:{name}:seed={seed}:steps={steps}:limit={limit}', has_matmul, perturbed)
                        if alternate is None:
                            if name in baselines:
                                judge(name, check(name), actual, baselines[name], 'configuration/observation invariance:' + name,
                                      has_matmul, lambda: envelope(None, name))
                            else:
                                baselines[name] = actual.clone()
                        values[name] = actual.clone()
                    for buf in program['buffers']:
                        if buf['base'] is not None:
                            continue
                        name = buf['name']
                        if buf['role'] == 'input':
                            if not torch.equal(memories[name].view(torch.uint8), original[name].view(torch.uint8)):
                                failures.append(('memory:' + name, RuntimeError('WRONG RESULT: input storage modified: ' + name)))
                        else:
                            if not torch.equal(memories[name][:, :16], original[name][:, :16]) or not torch.equal(memories[name][:, -16:], original[name][:, -16:]):
                                failures.append(('memory:' + name, RuntimeError('WRONG RESULT: scratch canary: ' + name)))
                            # Full memory comparison includes aliases, untouched
                            # elements and both guards, not only the final output.
                            perturbed = lambda: envelope(key, 'memory:' + name)
                            if atomic_kind is not None:
                                judge('memory:' + name, extended_check_atomic, memories[name], variant_expected_memory[name],
                                      prefix + 'atomic:' + name, atomic_kind, perturbed)
                            else:
                                judge('memory:' + name, extended_check, memories[name], variant_expected_memory[name],
                                      prefix + 'scratch:' + name, has_matmul, perturbed)
                        values['memory:' + name] = memories[name].clone()
                    if failures:
                        failure = failures[0][1]
                        failure.args = (f'{failure}\nWRONG VALUES: ' + ', '.join(dict.fromkeys(n for n, _ in failures)),)
                        raise failure
                    if previous is not None:
                        for name, actual in values.items():
                            if name.startswith('memory:') and atomic_kind is not None:
                                # Same kernel, same input, but the hardware
                                # does not promise a fixed racing order; the
                                # commutative tolerance absorbs reorderings.
                                extended_check_atomic(actual, previous[name],
                                                      'repeat determinism:' + name, atomic_kind)
                            elif not torch.equal(actual.contiguous().view(torch.uint8), previous[name].contiguous().view(torch.uint8)):
                                raise RuntimeError('WRONG RESULT: repeat determinism:' + name)
                    previous = values
    if unstable:
        raise RuntimeError(f'ORACLE UNSTABLE: {unstable_label}; {unstable} mismatched elements lie within '
                           'the perturbed reference; numeric check skipped')
