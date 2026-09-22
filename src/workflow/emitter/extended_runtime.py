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
    for stage, source in artifacts.items():
        if not isinstance(source, (str, bytes, bytearray)):
            continue
        raw = source.encode() if isinstance(source, str) else source
        stages[stage] = {'sha256': hashlib.sha256(raw).hexdigest(), 'bytes': len(raw)}
        if isinstance(source, str) and stage in ('ttir', 'ttgir', 'llir', 'ptx', 'lowered_tir', 'cuda'):
            # Post-lowering vocabulary. The saved input TIR is evidence of the
            # attempted program, not a retained compiler operation.
            tokens = re.findall(r'\b(?:arith|math|scf|cf|tt|ttg|ttng|llvm|nvvm|T)\.[A-Za-z_][\w.]*', source)
            if stage in ('ptx', 'cuda'):
                tokens += re.findall(r'\b(?:mma|ld|st|bar|shfl|cvt|add|mul|setp|selp)(?:\.[A-Za-z0-9_]+)+', source)
            features.update(json.dumps(['compiler', stage, op], separators=(',', ':')) for op in tokens)
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


def extended_reference(program, inputs, steps, limit):
    import torch
    memory = {name: value.clone() for name, value in inputs.items()}
    buffers = {b['name']: b for b in program['buffers']}
    functions = {f['name']: f['body'] for f in program['functions']}
    device = next(iter(inputs.values())).device
    outputs = {}

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
                out = [torch.flip(args[0], [-1])]
            elif op == 'interleave':
                # stack along the minor axis then reshape interleaves the
                # elements: [a0, b0, a1, b1, ...].
                out = [torch.stack((args[0], args[1]), dim=-1).reshape(
                    args[0].shape[:-1] + (args[0].shape[-1] * 2,))]
            elif op == 'join':
                out = [torch.stack((args[0], args[1]), dim=-1)]
            elif op == 'split':
                out = [args[0][..., 0], args[0][..., 1]]
            elif op in ('atomic_add', 'atomic_max', 'atomic_min'):
                buf = buffers[a['buffer']]
                root = buf['base'] or buf['name']
                index = args[0].long()
                valid = args[1] & (index >= 0) & (index < buf['size'])
                address = 16 + buf['offset'] + index.clamp(0, buf['size'] - 1) * buf['stride']
                kind = {'atomic_add': 'sum', 'atomic_max': 'amax', 'atomic_min': 'amin'}[op]
                # A masked scatter-reduce models any interleaving of the
                # commutative race; include_self folds the initial scratch
                # value into the reduction exactly like the hardware. The
                # scatter addresses are the payload offsets inside the root
                # row, mirroring the kernels' 16 + offset + index * stride.
                slot = memory[root][bid]
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
                values[result['name']] = value.to(getattr(torch, result['type']['dtype'])).reshape(result['type']['shape'])
        return [values[v] for v in body['returns']], values

    wanted = list(dict.fromkeys(program['body']['returns'] + program['observations']))
    for bid in range(program['blocks']):
        _, values = run(program['body'], {}, [], bid)
        for name in wanted:
            outputs.setdefault(name, []).append(values[name].clone())
    return {name: torch.stack(value) for name, value in outputs.items()}, memory


def extended_check(actual, expected, label, matmul=False):
    import torch
    if actual.shape != expected.shape or actual.dtype != expected.dtype:
        raise RuntimeError('WRONG RESULT: type/shape mismatch: ' + label)
    if not actual.dtype.is_floating_point:
        equal = torch.equal(actual, expected)
    else:
        # Check exceptional values separately; never let NaNs hide a mismatch.
        special = torch.isnan(expected) | torch.isinf(expected)
        equal = (torch.equal(torch.isnan(actual), torch.isnan(expected)) and
                 torch.equal(torch.isposinf(actual), torch.isposinf(expected)) and
                 torch.equal(torch.isneginf(actual), torch.isneginf(expected)))
        if equal:
            finite_actual, finite_expected = actual[~special], expected[~special]
            tolerance = 0.02 if matmul else 0.002
            equal = torch.allclose(finite_actual, finite_expected, rtol=tolerance, atol=tolerance * 0.125)
    if not equal:
        difference = (actual.to(torch.float64) - expected.to(torch.float64)).abs()
        if actual.dtype.is_floating_point:
            matching_special = ((torch.isnan(actual) & torch.isnan(expected)) |
                                (torch.isinf(actual) & (actual == expected)))
            difference = torch.where(matching_special, 0., difference)
        index = int(torch.nan_to_num(difference, nan=float('inf')).reshape(-1).argmax())
        raise RuntimeError(f'WRONG RESULT: {label}; max_abs={difference.reshape(-1)[index].item()}; '
                           f'index={index}; actual={actual.reshape(-1)[index].item()}; '
                           f'expected={expected.reshape(-1)[index].item()}')


def extended_check_atomic(actual, expected, label, kind):
    """Race-aware scratch comparison. add races accumulate in an arbitrary
    order, so the fp64 reference differs by a few rounding ulps (tolerance),
    while max/min races are order-independent and compare bitwise except for
    signed zero (which both orderings may pick)."""
    import torch
    if actual.shape != expected.shape or actual.dtype != expected.dtype:
        raise RuntimeError('WRONG RESULT: type/shape mismatch: ' + label)
    if not actual.dtype.is_floating_point:
        equal = torch.equal(actual, expected)
    else:
        special = torch.isnan(expected) | torch.isinf(expected)
        equal = (torch.equal(torch.isnan(actual), torch.isnan(expected)) and
                 torch.equal(torch.isposinf(actual), torch.isposinf(expected)) and
                 torch.equal(torch.isneginf(actual), torch.isneginf(expected)))
        if equal and kind in ('max', 'min'):
            zeros = lambda t: torch.where(t == 0, torch.zeros_like(t), t)
            equal = torch.equal(zeros(actual), zeros(expected))
        elif equal:
            finite_actual, finite_expected = actual[~special], expected[~special]
            # n <= 64 raced contributions per address differ by a few ulps of
            # ordering; losing a contribution moves by an order of magnitude.
            equal = torch.allclose(finite_actual, finite_expected, rtol=0.02, atol=0.02 * 0.125)
    if not equal:
        difference = (actual.to(torch.float64) - expected.to(torch.float64)).abs()
        index = int(torch.nan_to_num(difference, nan=float('inf')).reshape(-1).argmax())
        raise RuntimeError(f'WRONG RESULT: {label}; max_abs={difference.reshape(-1)[index].item()}; '
                           f'index={index}; actual={actual.reshape(-1)[index].item()}; '
                           f'expected={expected.reshape(-1)[index].item()}')


def extended_check_fma(actual, expected, label, matmul=False):
    """A fused and an unfused evaluation both round within ~1 ulp of the exact
    product-add; 4 ulps accepts either while rejecting operand mixups and
    exponent bugs. Exceptional values compare exactly like extended_check.
    The matmul flag is accepted for call-site uniformity and ignored."""
    import torch
    if actual.shape != expected.shape or actual.dtype != expected.dtype:
        raise RuntimeError('WRONG RESULT: type/shape mismatch: ' + label)
    if not actual.dtype.is_floating_point:
        raise RuntimeError('WRONG RESULT: fma on non-float: ' + label)
    same_special = (torch.equal(torch.isnan(actual), torch.isnan(expected)) and
                    torch.equal(torch.isposinf(actual), torch.isposinf(expected)) and
                    torch.equal(torch.isneginf(actual), torch.isneginf(expected)))
    if same_special:
        info = torch.finfo(actual.dtype)
        special = torch.isnan(expected) | torch.isinf(expected)
        tolerance = 4 * torch.where(expected == 0, info.tiny, expected.abs()) * info.eps
        equal = bool((((actual - expected).abs().le(tolerance)) | special).all())
    else:
        equal = False
    if not equal:
        difference = (actual.to(torch.float64) - expected.to(torch.float64)).abs()
        matching_special = ((torch.isnan(actual) & torch.isnan(expected)) |
                            (torch.isinf(actual) & (actual == expected)))
        difference = torch.where(matching_special, 0., difference)
        index = int(torch.nan_to_num(difference, nan=float('inf')).reshape(-1).argmax())
        raise RuntimeError(f'WRONG RESULT: {label}; max_abs={difference.reshape(-1)[index].item()}; '
                           f'index={index}; actual={actual.reshape(-1)[index].item()}; '
                           f'expected={expected.reshape(-1)[index].item()}')


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
                else:
                    variant_expected, variant_expected_memory, variant_types = expected, expected_memory, types
                    prefix = ''
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
                    values = {}
                    for name, storage in output_storage.items():
                        guard = True if storage.dtype == torch.bool else 23
                        if not torch.all(storage[:, :16] == guard) or not torch.all(storage[:, -16:] == guard):
                            raise RuntimeError('WRONG RESULT: output canary: ' + label)
                        actual = storage[:, 16:-16].reshape(variant_expected[name].shape)
                        if name in fma_names:
                            extended_check_fma(actual, variant_expected[name],
                                               f'{prefix}fma:{label}:{name}:seed={seed}:steps={steps}:limit={limit}')
                        else:
                            extended_check(actual, variant_expected[name],
                                           f'{prefix}{label}:{name}:seed={seed}:steps={steps}:limit={limit}', has_matmul)
                        if alternate is None:
                            if name in baselines:
                                check(name)(actual, baselines[name], 'configuration/observation invariance:' + name, has_matmul)
                            else:
                                baselines[name] = actual.clone()
                        values[name] = actual.clone()
                    for buf in program['buffers']:
                        if buf['base'] is not None:
                            continue
                        name = buf['name']
                        if buf['role'] == 'input':
                            if not torch.equal(memories[name].view(torch.uint8), original[name].view(torch.uint8)):
                                raise RuntimeError('WRONG RESULT: input storage modified: ' + name)
                        else:
                            if not torch.equal(memories[name][:, :16], original[name][:, :16]) or not torch.equal(memories[name][:, -16:], original[name][:, -16:]):
                                raise RuntimeError('WRONG RESULT: scratch canary: ' + name)
                            # Full memory comparison includes aliases, untouched
                            # elements and both guards, not only the final output.
                            if atomic_kind is not None:
                                extended_check_atomic(memories[name], variant_expected_memory[name],
                                                      prefix + 'atomic:' + name, atomic_kind)
                            else:
                                extended_check(memories[name], variant_expected_memory[name], prefix + 'scratch:' + name, has_matmul)
                        values['memory:' + name] = memories[name].clone()
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
