import os
import triton
import triton.language as tl

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


@triton.jit
def mixed_helper_extended_0_0(e55, e56, mem0, mem1, mem2, mem3, mem6, mem7, mem8, mem9, steps, limit, bid):
    e57 = (tl.sum(e55.to(tl.float32), 1)).to(tl.float32)
    e58 = ((e57 + e56)).to(tl.float32)
    e59 = (tl.sum(e58.to(tl.float32), 0)).to(tl.float32)
    e60 = (tl.reshape(e58, (16, 1))).to(tl.float32)
    e61 = (e60.to(tl.float16)).to(tl.float16)
    e62 = ((e55 + e61)).to(tl.float16)
    return e62, e59
@triton.jit
def extended_0_0(mem0, mem1, mem2, mem3, mem6, mem7, mem8, mem9, out_e131, out_e16, out_e22, out_e115, out_e116, out_e132, steps, limit):
    bid = tl.program_id(0)
    e1 = ((((tl.arange(0, 16)[:, None] * 32 + tl.arange(0, 32)[None, :]) + 10) % 512)).to(tl.int32)
    e2 = (tl.full((16, 32), True, tl.int1)).to(tl.int1)
    e3 = (tl.load(mem0 + bid * 544 + 16 + e1 * 1, (e2 & (e1 >= 0) & (e1 < 512)), other=0)).to(tl.float16)
    e4 = ((((tl.arange(0, 16)[:, None] * 32 + tl.arange(0, 32)[None, :]) + 17) % 512)).to(tl.int32)
    e5 = (tl.full((16, 32), True, tl.int1)).to(tl.int1)
    e6 = (tl.load(mem1 + bid * 544 + 16 + e4 * 1, (e5 & (e4 >= 0) & (e4 < 512)), other=0)).to(tl.float16)
    e7 = ((e3 * e6)).to(tl.float16)
    e8 = (e7.to(tl.float32)).to(tl.float32)
    e9 = (e8.to(tl.float16)).to(tl.float16)
    e10 = (tl.full((16, 32), 0.5, tl.float16)).to(tl.float16)
    e11 = ((e9 * e10)).to(tl.float16)
    e12 = ((((tl.arange(0, 16)[:, None] * 32 + tl.arange(0, 32)[None, :]) + 71) % 512)).to(tl.int32)
    e13 = (tl.full((16, 32), True, tl.int1)).to(tl.int1)
    e14 = (tl.load(mem2 + bid * 544 + 16 + e12 * 1, (e13 & (e12 >= 0) & (e12 < 512)), other=0)).to(tl.int32)
    e15 = (tl.full((16, 32), 3, tl.int32)).to(tl.int32)
    e16 = ((e14 & e15)).to(tl.int32)
    e17 = (tl.full((16, 32), 3, tl.int32)).to(tl.int32)
    e18 = ((e16 < e17)).to(tl.int1)
    e19 = ((((511 - (tl.arange(0, 16)[:, None] * 32 + tl.arange(0, 32)[None, :])) + 252) % 512)).to(tl.int32)
    e20 = (limit).to(tl.int32)
    e21 = ((e19 < e20)).to(tl.int1)
    e22 = ((e18 & e21)).to(tl.int1)
    e23 = (tl.full((16, 32), -0.125, tl.float16)).to(tl.float16)
    e24 = (tl.where(e22, e11, e23)).to(tl.float16)
    e25 = ((((511 - (tl.arange(0, 16)[:, None] * 32 + tl.arange(0, 32)[None, :])) + 305) % 512)).to(tl.int32)
    e26 = (limit).to(tl.int32)
    e27 = ((e25 < e26)).to(tl.int1)
    tl.store(mem3 + bid * 1060 + 17 + e25 * 2, e24, (e27 & (e25 >= 0) & (e25 < 512)))
    tl.debug_barrier()
    e28 = (tl.load(mem3 + bid * 1060 + 17 + e25 * 2, (e27 & (e25 >= 0) & (e25 < 512)), other=0)).to(tl.float16)
    e29 = (tl.full((16, 32), 0.125, tl.float16)).to(tl.float16)
    e30 = ((e28 + e29)).to(tl.float16)
    tl.debug_barrier()
    tl.store(mem3 + bid * 1060 + 18 + e25 * 1, e30, (e27 & (e25 >= 0) & (e25 < 512)))
    tl.debug_barrier()
    e31 = (tl.full((16, 32), True, tl.int1)).to(tl.int1)
    e32 = (tl.load(mem3 + bid * 1060 + 17 + e25 * 2, (e31 & (e25 >= 0) & (e25 < 512)), other=0)).to(tl.float16)
    e33 = (tl.full((16, 32), 3, tl.int32)).to(tl.int32)
    e34 = ((e25 ^ e33)).to(tl.int32)
    e35 = (tl.load(mem3 + bid * 1060 + 18 + e34 * 1, (e27 & (e34 >= 0) & (e34 < 512)), other=0)).to(tl.float16)
    e36 = ((e32 + e35)).to(tl.float16)
    e37 = (e36.to(tl.float16)).to(tl.float16)
    e38 = (tl.trans(e37)).to(tl.float16)
    e39_slice00_lo, e39_slice00_hi = tl.split(tl.permute(tl.reshape(e38, (2, 16, 16)), (1, 2, 0)))
    e39_slice00_next_lo, e39_slice00_next_hi = tl.split(tl.permute(tl.reshape(e39_slice00_lo, (2, 8, 16)), (1, 2, 0)))
    e39_slice01_lo, e39_slice01_hi = tl.split(tl.permute(tl.reshape(e38, (2, 16, 16)), (1, 2, 0)))
    e39_slice01_next_lo, e39_slice01_next_hi = tl.split(tl.permute(tl.reshape(e39_slice01_hi, (2, 8, 16)), (1, 2, 0)))
    e39_slice0 = tl.reshape(tl.permute(tl.join(e39_slice00_next_hi, e39_slice01_next_lo), (2, 0, 1)), (16, 16))
    e39 = (e39_slice0).to(tl.float16)
    e40 = (tl.full((16, 16), 0.0, tl.float32)).to(tl.float32)
    e41 = (tl.dot(e39, e39, e40, input_precision='ieee')).to(tl.float32)
    e42 = (tl.where(tl.sum((e41 != e41).to(tl.int32), 1) > 0, float('nan'), tl.max(e41.to(tl.float32), 1))).to(tl.float32)
    e43 = (tl.reshape(e42, (16, 1))).to(tl.float32)
    e44 = ((e41 + e43)).to(tl.float32)
    e45 = (tl.full((16, 16), 0.125, tl.float32)).to(tl.float32)
    e46 = ((e44 * e45)).to(tl.float32)
    e47 = ((((tl.arange(0, 16)[:, None] * 16 + tl.arange(0, 16)[None, :]) + 44) % 256)).to(tl.int32)
    e48 = (tl.full((16, 16), True, tl.int1)).to(tl.int1)
    e49 = (tl.load(mem6 + bid * 288 + 16 + e47 * 1, (e48 & (e47 >= 0) & (e47 < 256)), other=0)).to(tl.float16)
    e50 = (e46.to(tl.float16)).to(tl.float16)
    e51 = (tl.dot(e50, e49, e40, input_precision='ieee')).to(tl.float32)
    e52 = (tl.reshape(e51, (16, 16))).to(tl.float32)
    e53 = (tl.reshape(e52, (16, 16))).to(tl.float32)
    e54 = (e53.to(tl.float16)).to(tl.float16)
    e63 = (e42.to(tl.float32)).to(tl.float32)
    e64, e65 = mixed_helper_extended_0_0(e54, e63, mem0, mem1, mem2, mem3, mem6, mem7, mem8, mem9, steps, limit, bid)
    e66 = (steps).to(tl.int32)
    e67 = (tl.full((), 0, tl.int32)).to(tl.int32)
    e68 = (tl.full((), 0, tl.int32)).to(tl.int32)
    e69 = (tl.full((), True, tl.int1)).to(tl.int1)
    e70 = (tl.load(mem7 + bid * 33 + 16 + e68 * 1, (e69 & (e68 >= 0) & (e68 < 1)), other=0)).to(tl.int32)
    e71 = (tl.full((), 0, tl.int32)).to(tl.int32)
    e72 = ((e70 < e71)).to(tl.int1)
    e92 = e64
    e93 = e65
    e94 = e67
    for e73 in range(tl.minimum(tl.maximum(e66, 0), 4)):
        e74 = e92
        e75 = e93
        e76 = e94
        if e72:
            e77 = e74
            e78 = (tl.full((16, 16), 0.125, tl.float16)).to(tl.float16)
            e79 = ((e77 + e78)).to(tl.float16)
            e83 = e79
        else:
            e80 = e74
            e81 = (tl.full((16, 16), 0.125, tl.float16)).to(tl.float16)
            e82 = ((e80 - e81)).to(tl.float16)
            e83 = e82
        e84, e85 = mixed_helper_extended_0_0(e83, e63, mem0, mem1, mem2, mem3, mem6, mem7, mem8, mem9, steps, limit, bid)
        e86 = (tl.full((16, 16), 0.125, tl.float16)).to(tl.float16)
        e87 = ((e84 * e86)).to(tl.float16)
        e88 = ((e75 + e85)).to(tl.float32)
        e89 = (tl.full((), 1, tl.int32)).to(tl.int32)
        e90 = ((e73 + e89)).to(tl.int32)
        e91 = ((e76 + e90)).to(tl.int32)
        e92 = e87
        e93 = e88
        e94 = e91
    e114 = e92
    e115 = e93
    e116 = e94
    e95 = tl.full((), 0, tl.int32)
    while e95 < tl.minimum(tl.maximum(e66, 0), 4):
        e96 = e114
        e97 = e115
        e98 = e116
        if e72:
            e99 = e96
            e100 = (tl.full((16, 16), 0.125, tl.float16)).to(tl.float16)
            e101 = ((e99 + e100)).to(tl.float16)
            e105 = e101
        else:
            e102 = e96
            e103 = (tl.full((16, 16), 0.125, tl.float16)).to(tl.float16)
            e104 = ((e102 - e103)).to(tl.float16)
            e105 = e104
        e106, e107 = mixed_helper_extended_0_0(e105, e63, mem0, mem1, mem2, mem3, mem6, mem7, mem8, mem9, steps, limit, bid)
        e108 = (tl.full((16, 16), 0.125, tl.float16)).to(tl.float16)
        e109 = ((e106 * e108)).to(tl.float16)
        e110 = ((e97 + e107)).to(tl.float32)
        e111 = (tl.full((), 1, tl.int32)).to(tl.int32)
        e112 = ((e95 + e111)).to(tl.int32)
        e113 = ((e98 + e112)).to(tl.int32)
        e114 = e109
        e115 = e110
        e116 = e113
        e95 += 1
    e117 = (tl.full((), -0.125, tl.float32)).to(tl.float32)
    e118 = (tl.broadcast_to(e117, (16, 16))).to(tl.float32)
    e119 = (tl.full((), -0.25, tl.float32)).to(tl.float32)
    e120 = (tl.broadcast_to(e119, (16, 16))).to(tl.float32)
    e121 = ((e118 - e120)).to(tl.float32)
    e122 = ((((255 - (tl.arange(0, 16)[:, None] * 16 + tl.arange(0, 16)[None, :])) + 28) % 256)).to(tl.int32)
    e123 = (tl.full((16, 16), True, tl.int1)).to(tl.int1)
    tl.atomic_max(mem8 + bid * 288 + 16 + e122 * 1, e121, (e123 & (e122 >= 0) & (e122 < 256)))
    e124 = ((e114 - e92)).to(tl.float16)
    e125 = (e124.to(tl.float32)).to(tl.float32)
    e126 = ((((255 - (tl.arange(0, 16)[:, None] * 16 + tl.arange(0, 16)[None, :])) + 194) % 256)).to(tl.int32)
    e127 = (tl.full((16, 16), True, tl.int1)).to(tl.int1)
    e128 = (tl.load(mem9 + bid * 288 + 16 + e126 * 1, (e127 & (e126 >= 0) & (e126 < 256)), other=0)).to(tl.float32)
    e129 = ((e125 - e128)).to(tl.float32)
    e130 = (e129.to(tl.float16)).to(tl.float16)
    e131 = ((e130 + e49)).to(tl.float16)
    e132 = (tl.where(tl.sum((e131 != e131).to(tl.int32), 0) > 0, float('nan'), tl.max(e131.to(tl.float32), 0))).to(tl.float32)
    tl.store(out_e131 + bid * 288 + 16 + (tl.arange(0, 16)[:, None] * 16 + tl.arange(0, 16)[None, :]), e131)
    tl.store(out_e16 + bid * 544 + 16 + (tl.arange(0, 16)[:, None] * 32 + tl.arange(0, 32)[None, :]), e16)
    tl.store(out_e22 + bid * 544 + 16 + (tl.arange(0, 16)[:, None] * 32 + tl.arange(0, 32)[None, :]), e22)
    tl.store(out_e115 + bid * 33 + 16 + tl.full((), 0, tl.int32), e115)
    tl.store(out_e116 + bid * 33 + 16 + tl.full((), 0, tl.int32), e116)
    tl.store(out_e132 + bid * 48 + 16 + tl.arange(0, 16), e132)

@triton.jit
def mixed_helper_extended_0_1(e55, e56, mem0, mem1, mem2, mem3, mem6, mem7, mem8, mem9, steps, limit, bid):
    e57 = (tl.sum(e55.to(tl.float32), 1)).to(tl.float32)
    e58 = ((e57 + e56)).to(tl.float32)
    e59 = (tl.sum(e58.to(tl.float32), 0)).to(tl.float32)
    e60 = (tl.reshape(e58, (16, 1))).to(tl.float32)
    e61 = (e60.to(tl.float16)).to(tl.float16)
    e62 = ((e55 + e61)).to(tl.float16)
    return e62, e59
@triton.jit
def extended_0_1(mem0, mem1, mem2, mem3, mem6, mem7, mem8, mem9, out_e131, out_e16, out_e22, out_e115, out_e116, out_e132, out_e3, out_e11, out_e28, out_e32, out_e41, out_e46, out_e21, out_e4, steps, limit):
    bid = tl.program_id(0)
    e1 = ((((tl.arange(0, 16)[:, None] * 32 + tl.arange(0, 32)[None, :]) + 10) % 512)).to(tl.int32)
    e2 = (tl.full((16, 32), True, tl.int1)).to(tl.int1)
    e3 = (tl.load(mem0 + bid * 544 + 16 + e1 * 1, (e2 & (e1 >= 0) & (e1 < 512)), other=0)).to(tl.float16)
    e4 = ((((tl.arange(0, 16)[:, None] * 32 + tl.arange(0, 32)[None, :]) + 17) % 512)).to(tl.int32)
    e5 = (tl.full((16, 32), True, tl.int1)).to(tl.int1)
    e6 = (tl.load(mem1 + bid * 544 + 16 + e4 * 1, (e5 & (e4 >= 0) & (e4 < 512)), other=0)).to(tl.float16)
    e7 = ((e3 * e6)).to(tl.float16)
    e8 = (e7.to(tl.float32)).to(tl.float32)
    e9 = (e8.to(tl.float16)).to(tl.float16)
    e10 = (tl.full((16, 32), 0.5, tl.float16)).to(tl.float16)
    e11 = ((e9 * e10)).to(tl.float16)
    e12 = ((((tl.arange(0, 16)[:, None] * 32 + tl.arange(0, 32)[None, :]) + 71) % 512)).to(tl.int32)
    e13 = (tl.full((16, 32), True, tl.int1)).to(tl.int1)
    e14 = (tl.load(mem2 + bid * 544 + 16 + e12 * 1, (e13 & (e12 >= 0) & (e12 < 512)), other=0)).to(tl.int32)
    e15 = (tl.full((16, 32), 3, tl.int32)).to(tl.int32)
    e16 = ((e14 & e15)).to(tl.int32)
    e17 = (tl.full((16, 32), 3, tl.int32)).to(tl.int32)
    e18 = ((e16 < e17)).to(tl.int1)
    e19 = ((((511 - (tl.arange(0, 16)[:, None] * 32 + tl.arange(0, 32)[None, :])) + 252) % 512)).to(tl.int32)
    e20 = (limit).to(tl.int32)
    e21 = ((e19 < e20)).to(tl.int1)
    e22 = ((e18 & e21)).to(tl.int1)
    e23 = (tl.full((16, 32), -0.125, tl.float16)).to(tl.float16)
    e24 = (tl.where(e22, e11, e23)).to(tl.float16)
    e25 = ((((511 - (tl.arange(0, 16)[:, None] * 32 + tl.arange(0, 32)[None, :])) + 305) % 512)).to(tl.int32)
    e26 = (limit).to(tl.int32)
    e27 = ((e25 < e26)).to(tl.int1)
    tl.store(mem3 + bid * 1060 + 17 + e25 * 2, e24, (e27 & (e25 >= 0) & (e25 < 512)))
    tl.debug_barrier()
    e28 = (tl.load(mem3 + bid * 1060 + 17 + e25 * 2, (e27 & (e25 >= 0) & (e25 < 512)), other=0)).to(tl.float16)
    e29 = (tl.full((16, 32), 0.125, tl.float16)).to(tl.float16)
    e30 = ((e28 + e29)).to(tl.float16)
    tl.debug_barrier()
    tl.store(mem3 + bid * 1060 + 18 + e25 * 1, e30, (e27 & (e25 >= 0) & (e25 < 512)))
    tl.debug_barrier()
    e31 = (tl.full((16, 32), True, tl.int1)).to(tl.int1)
    e32 = (tl.load(mem3 + bid * 1060 + 17 + e25 * 2, (e31 & (e25 >= 0) & (e25 < 512)), other=0)).to(tl.float16)
    e33 = (tl.full((16, 32), 3, tl.int32)).to(tl.int32)
    e34 = ((e25 ^ e33)).to(tl.int32)
    e35 = (tl.load(mem3 + bid * 1060 + 18 + e34 * 1, (e27 & (e34 >= 0) & (e34 < 512)), other=0)).to(tl.float16)
    e36 = ((e32 + e35)).to(tl.float16)
    e37 = (e36.to(tl.float16)).to(tl.float16)
    e38 = (tl.trans(e37)).to(tl.float16)
    e39_slice00_lo, e39_slice00_hi = tl.split(tl.permute(tl.reshape(e38, (2, 16, 16)), (1, 2, 0)))
    e39_slice00_next_lo, e39_slice00_next_hi = tl.split(tl.permute(tl.reshape(e39_slice00_lo, (2, 8, 16)), (1, 2, 0)))
    e39_slice01_lo, e39_slice01_hi = tl.split(tl.permute(tl.reshape(e38, (2, 16, 16)), (1, 2, 0)))
    e39_slice01_next_lo, e39_slice01_next_hi = tl.split(tl.permute(tl.reshape(e39_slice01_hi, (2, 8, 16)), (1, 2, 0)))
    e39_slice0 = tl.reshape(tl.permute(tl.join(e39_slice00_next_hi, e39_slice01_next_lo), (2, 0, 1)), (16, 16))
    e39 = (e39_slice0).to(tl.float16)
    e40 = (tl.full((16, 16), 0.0, tl.float32)).to(tl.float32)
    e41 = (tl.dot(e39, e39, e40, input_precision='ieee')).to(tl.float32)
    e42 = (tl.where(tl.sum((e41 != e41).to(tl.int32), 1) > 0, float('nan'), tl.max(e41.to(tl.float32), 1))).to(tl.float32)
    e43 = (tl.reshape(e42, (16, 1))).to(tl.float32)
    e44 = ((e41 + e43)).to(tl.float32)
    e45 = (tl.full((16, 16), 0.125, tl.float32)).to(tl.float32)
    e46 = ((e44 * e45)).to(tl.float32)
    e47 = ((((tl.arange(0, 16)[:, None] * 16 + tl.arange(0, 16)[None, :]) + 44) % 256)).to(tl.int32)
    e48 = (tl.full((16, 16), True, tl.int1)).to(tl.int1)
    e49 = (tl.load(mem6 + bid * 288 + 16 + e47 * 1, (e48 & (e47 >= 0) & (e47 < 256)), other=0)).to(tl.float16)
    e50 = (e46.to(tl.float16)).to(tl.float16)
    e51 = (tl.dot(e50, e49, e40, input_precision='ieee')).to(tl.float32)
    e52 = (tl.reshape(e51, (16, 16))).to(tl.float32)
    e53 = (tl.reshape(e52, (16, 16))).to(tl.float32)
    e54 = (e53.to(tl.float16)).to(tl.float16)
    e63 = (e42.to(tl.float32)).to(tl.float32)
    e64, e65 = mixed_helper_extended_0_1(e54, e63, mem0, mem1, mem2, mem3, mem6, mem7, mem8, mem9, steps, limit, bid)
    e66 = (steps).to(tl.int32)
    e67 = (tl.full((), 0, tl.int32)).to(tl.int32)
    e68 = (tl.full((), 0, tl.int32)).to(tl.int32)
    e69 = (tl.full((), True, tl.int1)).to(tl.int1)
    e70 = (tl.load(mem7 + bid * 33 + 16 + e68 * 1, (e69 & (e68 >= 0) & (e68 < 1)), other=0)).to(tl.int32)
    e71 = (tl.full((), 0, tl.int32)).to(tl.int32)
    e72 = ((e70 < e71)).to(tl.int1)
    e92 = e64
    e93 = e65
    e94 = e67
    for e73 in range(tl.minimum(tl.maximum(e66, 0), 4)):
        e74 = e92
        e75 = e93
        e76 = e94
        if e72:
            e77 = e74
            e78 = (tl.full((16, 16), 0.125, tl.float16)).to(tl.float16)
            e79 = ((e77 + e78)).to(tl.float16)
            e83 = e79
        else:
            e80 = e74
            e81 = (tl.full((16, 16), 0.125, tl.float16)).to(tl.float16)
            e82 = ((e80 - e81)).to(tl.float16)
            e83 = e82
        e84, e85 = mixed_helper_extended_0_1(e83, e63, mem0, mem1, mem2, mem3, mem6, mem7, mem8, mem9, steps, limit, bid)
        e86 = (tl.full((16, 16), 0.125, tl.float16)).to(tl.float16)
        e87 = ((e84 * e86)).to(tl.float16)
        e88 = ((e75 + e85)).to(tl.float32)
        e89 = (tl.full((), 1, tl.int32)).to(tl.int32)
        e90 = ((e73 + e89)).to(tl.int32)
        e91 = ((e76 + e90)).to(tl.int32)
        e92 = e87
        e93 = e88
        e94 = e91
    e114 = e92
    e115 = e93
    e116 = e94
    e95 = tl.full((), 0, tl.int32)
    while e95 < tl.minimum(tl.maximum(e66, 0), 4):
        e96 = e114
        e97 = e115
        e98 = e116
        if e72:
            e99 = e96
            e100 = (tl.full((16, 16), 0.125, tl.float16)).to(tl.float16)
            e101 = ((e99 + e100)).to(tl.float16)
            e105 = e101
        else:
            e102 = e96
            e103 = (tl.full((16, 16), 0.125, tl.float16)).to(tl.float16)
            e104 = ((e102 - e103)).to(tl.float16)
            e105 = e104
        e106, e107 = mixed_helper_extended_0_1(e105, e63, mem0, mem1, mem2, mem3, mem6, mem7, mem8, mem9, steps, limit, bid)
        e108 = (tl.full((16, 16), 0.125, tl.float16)).to(tl.float16)
        e109 = ((e106 * e108)).to(tl.float16)
        e110 = ((e97 + e107)).to(tl.float32)
        e111 = (tl.full((), 1, tl.int32)).to(tl.int32)
        e112 = ((e95 + e111)).to(tl.int32)
        e113 = ((e98 + e112)).to(tl.int32)
        e114 = e109
        e115 = e110
        e116 = e113
        e95 += 1
    e117 = (tl.full((), -0.125, tl.float32)).to(tl.float32)
    e118 = (tl.broadcast_to(e117, (16, 16))).to(tl.float32)
    e119 = (tl.full((), -0.25, tl.float32)).to(tl.float32)
    e120 = (tl.broadcast_to(e119, (16, 16))).to(tl.float32)
    e121 = ((e118 - e120)).to(tl.float32)
    e122 = ((((255 - (tl.arange(0, 16)[:, None] * 16 + tl.arange(0, 16)[None, :])) + 28) % 256)).to(tl.int32)
    e123 = (tl.full((16, 16), True, tl.int1)).to(tl.int1)
    tl.atomic_max(mem8 + bid * 288 + 16 + e122 * 1, e121, (e123 & (e122 >= 0) & (e122 < 256)))
    e124 = ((e114 - e92)).to(tl.float16)
    e125 = (e124.to(tl.float32)).to(tl.float32)
    e126 = ((((255 - (tl.arange(0, 16)[:, None] * 16 + tl.arange(0, 16)[None, :])) + 194) % 256)).to(tl.int32)
    e127 = (tl.full((16, 16), True, tl.int1)).to(tl.int1)
    e128 = (tl.load(mem9 + bid * 288 + 16 + e126 * 1, (e127 & (e126 >= 0) & (e126 < 256)), other=0)).to(tl.float32)
    e129 = ((e125 - e128)).to(tl.float32)
    e130 = (e129.to(tl.float16)).to(tl.float16)
    e131 = ((e130 + e49)).to(tl.float16)
    e132 = (tl.where(tl.sum((e131 != e131).to(tl.int32), 0) > 0, float('nan'), tl.max(e131.to(tl.float32), 0))).to(tl.float32)
    tl.store(out_e131 + bid * 288 + 16 + (tl.arange(0, 16)[:, None] * 16 + tl.arange(0, 16)[None, :]), e131)
    tl.store(out_e16 + bid * 544 + 16 + (tl.arange(0, 16)[:, None] * 32 + tl.arange(0, 32)[None, :]), e16)
    tl.store(out_e22 + bid * 544 + 16 + (tl.arange(0, 16)[:, None] * 32 + tl.arange(0, 32)[None, :]), e22)
    tl.store(out_e115 + bid * 33 + 16 + tl.full((), 0, tl.int32), e115)
    tl.store(out_e116 + bid * 33 + 16 + tl.full((), 0, tl.int32), e116)
    tl.store(out_e132 + bid * 48 + 16 + tl.arange(0, 16), e132)
    tl.store(out_e3 + bid * 544 + 16 + (tl.arange(0, 16)[:, None] * 32 + tl.arange(0, 32)[None, :]), e3)
    tl.store(out_e11 + bid * 544 + 16 + (tl.arange(0, 16)[:, None] * 32 + tl.arange(0, 32)[None, :]), e11)
    tl.store(out_e28 + bid * 544 + 16 + (tl.arange(0, 16)[:, None] * 32 + tl.arange(0, 32)[None, :]), e28)
    tl.store(out_e32 + bid * 544 + 16 + (tl.arange(0, 16)[:, None] * 32 + tl.arange(0, 32)[None, :]), e32)
    tl.store(out_e41 + bid * 288 + 16 + (tl.arange(0, 16)[:, None] * 16 + tl.arange(0, 16)[None, :]), e41)
    tl.store(out_e46 + bid * 288 + 16 + (tl.arange(0, 16)[:, None] * 16 + tl.arange(0, 16)[None, :]), e46)
    tl.store(out_e21 + bid * 544 + 16 + (tl.arange(0, 16)[:, None] * 32 + tl.arange(0, 32)[None, :]), e21)
    tl.store(out_e4 + bid * 544 + 16 + (tl.arange(0, 16)[:, None] * 32 + tl.arange(0, 32)[None, :]), e4)

@triton.jit
def mixed_helper_extended_1_0(e55, e56, mem0, mem1, mem2, mem3, mem6, mem7, mem8, mem9, steps, limit, bid):
    e57 = (tl.sum(e55.to(tl.float32), 1)).to(tl.float32)
    e58 = ((e57 + e56)).to(tl.float32)
    e59 = (tl.sum(e58.to(tl.float32), 0)).to(tl.float32)
    e60 = (tl.reshape(e58, (16, 1))).to(tl.float32)
    e61 = (e60.to(tl.float16)).to(tl.float16)
    e62 = ((e55 + e61)).to(tl.float16)
    return e62, e59
@triton.jit
def extended_1_0(mem0, mem1, mem2, mem3, mem6, mem7, mem8, mem9, out_e131, out_e16, out_e22, out_e115, out_e116, out_e132, steps, limit):
    bid = tl.program_id(0)
    e1 = ((((tl.arange(0, 16)[:, None] * 32 + tl.arange(0, 32)[None, :]) + 10) % 512)).to(tl.int32)
    e2 = (tl.full((16, 32), True, tl.int1)).to(tl.int1)
    e3 = (tl.load(mem0 + bid * 544 + 16 + e1 * 1, (e2 & (e1 >= 0) & (e1 < 512)), other=0)).to(tl.float16)
    e4 = ((((tl.arange(0, 16)[:, None] * 32 + tl.arange(0, 32)[None, :]) + 17) % 512)).to(tl.int32)
    e5 = (tl.full((16, 32), True, tl.int1)).to(tl.int1)
    e6 = (tl.load(mem1 + bid * 544 + 16 + e4 * 1, (e5 & (e4 >= 0) & (e4 < 512)), other=0)).to(tl.float16)
    e7 = ((e3 * e6)).to(tl.float16)
    e8 = (e7.to(tl.float32)).to(tl.float32)
    e9 = (e8.to(tl.float16)).to(tl.float16)
    e10 = (tl.full((16, 32), 0.5, tl.float16)).to(tl.float16)
    e11 = ((e9 * e10)).to(tl.float16)
    e12 = ((((tl.arange(0, 16)[:, None] * 32 + tl.arange(0, 32)[None, :]) + 71) % 512)).to(tl.int32)
    e13 = (tl.full((16, 32), True, tl.int1)).to(tl.int1)
    e14 = (tl.load(mem2 + bid * 544 + 16 + e12 * 1, (e13 & (e12 >= 0) & (e12 < 512)), other=0)).to(tl.int32)
    e15 = (tl.full((16, 32), 3, tl.int32)).to(tl.int32)
    e16 = ((e14 & e15)).to(tl.int32)
    e17 = (tl.full((16, 32), 3, tl.int32)).to(tl.int32)
    e18 = ((e16 < e17)).to(tl.int1)
    e19 = ((((511 - (tl.arange(0, 16)[:, None] * 32 + tl.arange(0, 32)[None, :])) + 252) % 512)).to(tl.int32)
    e20 = (limit).to(tl.int32)
    e21 = ((e19 < e20)).to(tl.int1)
    e22 = ((e18 & e21)).to(tl.int1)
    e23 = (tl.full((16, 32), -0.125, tl.float16)).to(tl.float16)
    e24 = (tl.where(e22, e11, e23)).to(tl.float16)
    e25 = ((((511 - (tl.arange(0, 16)[:, None] * 32 + tl.arange(0, 32)[None, :])) + 305) % 512)).to(tl.int32)
    e26 = (limit).to(tl.int32)
    e27 = ((e25 < e26)).to(tl.int1)
    tl.store(mem3 + bid * 1060 + 17 + e25 * 2, e24, (e27 & (e25 >= 0) & (e25 < 512)))
    tl.debug_barrier()
    e28 = (tl.load(mem3 + bid * 1060 + 17 + e25 * 2, (e27 & (e25 >= 0) & (e25 < 512)), other=0)).to(tl.float16)
    e29 = (tl.full((16, 32), 0.125, tl.float16)).to(tl.float16)
    e30 = ((e28 + e29)).to(tl.float16)
    tl.debug_barrier()
    tl.store(mem3 + bid * 1060 + 18 + e25 * 1, e30, (e27 & (e25 >= 0) & (e25 < 512)))
    tl.debug_barrier()
    e31 = (tl.full((16, 32), True, tl.int1)).to(tl.int1)
    e32 = (tl.load(mem3 + bid * 1060 + 17 + e25 * 2, (e31 & (e25 >= 0) & (e25 < 512)), other=0)).to(tl.float16)
    e33 = (tl.full((16, 32), 3, tl.int32)).to(tl.int32)
    e34 = ((e25 ^ e33)).to(tl.int32)
    e35 = (tl.load(mem3 + bid * 1060 + 18 + e34 * 1, (e27 & (e34 >= 0) & (e34 < 512)), other=0)).to(tl.float16)
    e36 = ((e32 + e35)).to(tl.float16)
    e37 = (e36.to(tl.float16)).to(tl.float16)
    e38 = (tl.trans(e37)).to(tl.float16)
    e39_slice00_lo, e39_slice00_hi = tl.split(tl.permute(tl.reshape(e38, (2, 16, 16)), (1, 2, 0)))
    e39_slice00_next_lo, e39_slice00_next_hi = tl.split(tl.permute(tl.reshape(e39_slice00_lo, (2, 8, 16)), (1, 2, 0)))
    e39_slice01_lo, e39_slice01_hi = tl.split(tl.permute(tl.reshape(e38, (2, 16, 16)), (1, 2, 0)))
    e39_slice01_next_lo, e39_slice01_next_hi = tl.split(tl.permute(tl.reshape(e39_slice01_hi, (2, 8, 16)), (1, 2, 0)))
    e39_slice0 = tl.reshape(tl.permute(tl.join(e39_slice00_next_hi, e39_slice01_next_lo), (2, 0, 1)), (16, 16))
    e39 = (e39_slice0).to(tl.float16)
    e40 = (tl.full((16, 16), 0.0, tl.float32)).to(tl.float32)
    e41 = (tl.dot(e39, e39, e40, input_precision='ieee')).to(tl.float32)
    e42 = (tl.where(tl.sum((e41 != e41).to(tl.int32), 1) > 0, float('nan'), tl.max(e41.to(tl.float32), 1))).to(tl.float32)
    e43 = (tl.reshape(e42, (16, 1))).to(tl.float32)
    e44 = ((e41 + e43)).to(tl.float32)
    e45 = (tl.full((16, 16), 0.125, tl.float32)).to(tl.float32)
    e46 = ((e44 * e45)).to(tl.float32)
    e47 = ((((tl.arange(0, 16)[:, None] * 16 + tl.arange(0, 16)[None, :]) + 44) % 256)).to(tl.int32)
    e48 = (tl.full((16, 16), True, tl.int1)).to(tl.int1)
    e49 = (tl.load(mem6 + bid * 288 + 16 + e47 * 1, (e48 & (e47 >= 0) & (e47 < 256)), other=0)).to(tl.float16)
    e50 = (e46.to(tl.float16)).to(tl.float16)
    e51 = (tl.dot(e50, e49, e40, input_precision='ieee')).to(tl.float32)
    e52 = (tl.reshape(e51, (16, 16))).to(tl.float32)
    e53 = (tl.reshape(e52, (16, 16))).to(tl.float32)
    e54 = (e53.to(tl.float16)).to(tl.float16)
    e63 = (e42.to(tl.float32)).to(tl.float32)
    e64, e65 = mixed_helper_extended_1_0(e54, e63, mem0, mem1, mem2, mem3, mem6, mem7, mem8, mem9, steps, limit, bid)
    e66 = (steps).to(tl.int32)
    e67 = (tl.full((), 0, tl.int32)).to(tl.int32)
    e68 = (tl.full((), 0, tl.int32)).to(tl.int32)
    e69 = (tl.full((), True, tl.int1)).to(tl.int1)
    e70 = (tl.load(mem7 + bid * 33 + 16 + e68 * 1, (e69 & (e68 >= 0) & (e68 < 1)), other=0)).to(tl.int32)
    e71 = (tl.full((), 0, tl.int32)).to(tl.int32)
    e72 = ((e70 < e71)).to(tl.int1)
    e92 = e64
    e93 = e65
    e94 = e67
    for e73 in range(tl.minimum(tl.maximum(e66, 0), 4)):
        e74 = e92
        e75 = e93
        e76 = e94
        if e72:
            e77 = e74
            e78 = (tl.full((16, 16), 0.125, tl.float16)).to(tl.float16)
            e79 = ((e77 + e78)).to(tl.float16)
            e83 = e79
        else:
            e80 = e74
            e81 = (tl.full((16, 16), 0.125, tl.float16)).to(tl.float16)
            e82 = ((e80 - e81)).to(tl.float16)
            e83 = e82
        e84, e85 = mixed_helper_extended_1_0(e83, e63, mem0, mem1, mem2, mem3, mem6, mem7, mem8, mem9, steps, limit, bid)
        e86 = (tl.full((16, 16), 0.125, tl.float16)).to(tl.float16)
        e87 = ((e84 * e86)).to(tl.float16)
        e88 = ((e75 + e85)).to(tl.float32)
        e89 = (tl.full((), 1, tl.int32)).to(tl.int32)
        e90 = ((e73 + e89)).to(tl.int32)
        e91 = ((e76 + e90)).to(tl.int32)
        e92 = e87
        e93 = e88
        e94 = e91
    e114 = e92
    e115 = e93
    e116 = e94
    e95 = tl.full((), 0, tl.int32)
    while e95 < tl.minimum(tl.maximum(e66, 0), 4):
        e96 = e114
        e97 = e115
        e98 = e116
        if e72:
            e99 = e96
            e100 = (tl.full((16, 16), 0.125, tl.float16)).to(tl.float16)
            e101 = ((e99 + e100)).to(tl.float16)
            e105 = e101
        else:
            e102 = e96
            e103 = (tl.full((16, 16), 0.125, tl.float16)).to(tl.float16)
            e104 = ((e102 - e103)).to(tl.float16)
            e105 = e104
        e106, e107 = mixed_helper_extended_1_0(e105, e63, mem0, mem1, mem2, mem3, mem6, mem7, mem8, mem9, steps, limit, bid)
        e108 = (tl.full((16, 16), 0.125, tl.float16)).to(tl.float16)
        e109 = ((e106 * e108)).to(tl.float16)
        e110 = ((e97 + e107)).to(tl.float32)
        e111 = (tl.full((), 1, tl.int32)).to(tl.int32)
        e112 = ((e95 + e111)).to(tl.int32)
        e113 = ((e98 + e112)).to(tl.int32)
        e114 = e109
        e115 = e110
        e116 = e113
        e95 += 1
    e117 = (tl.full((), -0.125, tl.float32)).to(tl.float32)
    e118 = (tl.broadcast_to(e117, (16, 16))).to(tl.float32)
    e119 = (tl.full((), -0.25, tl.float32)).to(tl.float32)
    e120 = (tl.broadcast_to(e119, (16, 16))).to(tl.float32)
    e121 = ((e118 - e120)).to(tl.float32)
    e122 = ((((255 - (tl.arange(0, 16)[:, None] * 16 + tl.arange(0, 16)[None, :])) + 28) % 256)).to(tl.int32)
    e123 = (tl.full((16, 16), True, tl.int1)).to(tl.int1)
    tl.atomic_max(mem8 + bid * 288 + 16 + e122 * 1, e121, (e123 & (e122 >= 0) & (e122 < 256)))
    e124 = ((e114 - e92)).to(tl.float16)
    e125 = (e124.to(tl.float32)).to(tl.float32)
    e126 = ((((255 - (tl.arange(0, 16)[:, None] * 16 + tl.arange(0, 16)[None, :])) + 194) % 256)).to(tl.int32)
    e127 = (tl.full((16, 16), True, tl.int1)).to(tl.int1)
    e128 = (tl.load(mem9 + bid * 288 + 16 + e126 * 1, (e127 & (e126 >= 0) & (e126 < 256)), other=0)).to(tl.float32)
    e129 = ((e125 - e128)).to(tl.float32)
    e130 = (e129.to(tl.float16)).to(tl.float16)
    e131 = ((e130 + e49)).to(tl.float16)
    e132 = (tl.where(tl.sum((e131 != e131).to(tl.int32), 0) > 0, float('nan'), tl.max(e131.to(tl.float32), 0))).to(tl.float32)
    tl.store(out_e131 + bid * 288 + 16 + (tl.arange(0, 16)[:, None] * 16 + tl.arange(0, 16)[None, :]), e131)
    tl.store(out_e16 + bid * 544 + 16 + (tl.arange(0, 16)[:, None] * 32 + tl.arange(0, 32)[None, :]), e16)
    tl.store(out_e22 + bid * 544 + 16 + (tl.arange(0, 16)[:, None] * 32 + tl.arange(0, 32)[None, :]), e22)
    tl.store(out_e115 + bid * 33 + 16 + tl.full((), 0, tl.int32), e115)
    tl.store(out_e116 + bid * 33 + 16 + tl.full((), 0, tl.int32), e116)
    tl.store(out_e132 + bid * 48 + 16 + tl.arange(0, 16), e132)

@triton.jit
def mixed_helper_extended_1_1(e55, e56, mem0, mem1, mem2, mem3, mem6, mem7, mem8, mem9, steps, limit, bid):
    e57 = (tl.sum(e55.to(tl.float32), 1)).to(tl.float32)
    e58 = ((e57 + e56)).to(tl.float32)
    e59 = (tl.sum(e58.to(tl.float32), 0)).to(tl.float32)
    e60 = (tl.reshape(e58, (16, 1))).to(tl.float32)
    e61 = (e60.to(tl.float16)).to(tl.float16)
    e62 = ((e55 + e61)).to(tl.float16)
    return e62, e59
@triton.jit
def extended_1_1(mem0, mem1, mem2, mem3, mem6, mem7, mem8, mem9, out_e131, out_e16, out_e22, out_e115, out_e116, out_e132, out_e3, out_e11, out_e28, out_e32, out_e41, out_e46, out_e21, out_e4, steps, limit):
    bid = tl.program_id(0)
    e1 = ((((tl.arange(0, 16)[:, None] * 32 + tl.arange(0, 32)[None, :]) + 10) % 512)).to(tl.int32)
    e2 = (tl.full((16, 32), True, tl.int1)).to(tl.int1)
    e3 = (tl.load(mem0 + bid * 544 + 16 + e1 * 1, (e2 & (e1 >= 0) & (e1 < 512)), other=0)).to(tl.float16)
    e4 = ((((tl.arange(0, 16)[:, None] * 32 + tl.arange(0, 32)[None, :]) + 17) % 512)).to(tl.int32)
    e5 = (tl.full((16, 32), True, tl.int1)).to(tl.int1)
    e6 = (tl.load(mem1 + bid * 544 + 16 + e4 * 1, (e5 & (e4 >= 0) & (e4 < 512)), other=0)).to(tl.float16)
    e7 = ((e3 * e6)).to(tl.float16)
    e8 = (e7.to(tl.float32)).to(tl.float32)
    e9 = (e8.to(tl.float16)).to(tl.float16)
    e10 = (tl.full((16, 32), 0.5, tl.float16)).to(tl.float16)
    e11 = ((e9 * e10)).to(tl.float16)
    e12 = ((((tl.arange(0, 16)[:, None] * 32 + tl.arange(0, 32)[None, :]) + 71) % 512)).to(tl.int32)
    e13 = (tl.full((16, 32), True, tl.int1)).to(tl.int1)
    e14 = (tl.load(mem2 + bid * 544 + 16 + e12 * 1, (e13 & (e12 >= 0) & (e12 < 512)), other=0)).to(tl.int32)
    e15 = (tl.full((16, 32), 3, tl.int32)).to(tl.int32)
    e16 = ((e14 & e15)).to(tl.int32)
    e17 = (tl.full((16, 32), 3, tl.int32)).to(tl.int32)
    e18 = ((e16 < e17)).to(tl.int1)
    e19 = ((((511 - (tl.arange(0, 16)[:, None] * 32 + tl.arange(0, 32)[None, :])) + 252) % 512)).to(tl.int32)
    e20 = (limit).to(tl.int32)
    e21 = ((e19 < e20)).to(tl.int1)
    e22 = ((e18 & e21)).to(tl.int1)
    e23 = (tl.full((16, 32), -0.125, tl.float16)).to(tl.float16)
    e24 = (tl.where(e22, e11, e23)).to(tl.float16)
    e25 = ((((511 - (tl.arange(0, 16)[:, None] * 32 + tl.arange(0, 32)[None, :])) + 305) % 512)).to(tl.int32)
    e26 = (limit).to(tl.int32)
    e27 = ((e25 < e26)).to(tl.int1)
    tl.store(mem3 + bid * 1060 + 17 + e25 * 2, e24, (e27 & (e25 >= 0) & (e25 < 512)))
    tl.debug_barrier()
    e28 = (tl.load(mem3 + bid * 1060 + 17 + e25 * 2, (e27 & (e25 >= 0) & (e25 < 512)), other=0)).to(tl.float16)
    e29 = (tl.full((16, 32), 0.125, tl.float16)).to(tl.float16)
    e30 = ((e28 + e29)).to(tl.float16)
    tl.debug_barrier()
    tl.store(mem3 + bid * 1060 + 18 + e25 * 1, e30, (e27 & (e25 >= 0) & (e25 < 512)))
    tl.debug_barrier()
    e31 = (tl.full((16, 32), True, tl.int1)).to(tl.int1)
    e32 = (tl.load(mem3 + bid * 1060 + 17 + e25 * 2, (e31 & (e25 >= 0) & (e25 < 512)), other=0)).to(tl.float16)
    e33 = (tl.full((16, 32), 3, tl.int32)).to(tl.int32)
    e34 = ((e25 ^ e33)).to(tl.int32)
    e35 = (tl.load(mem3 + bid * 1060 + 18 + e34 * 1, (e27 & (e34 >= 0) & (e34 < 512)), other=0)).to(tl.float16)
    e36 = ((e32 + e35)).to(tl.float16)
    e37 = (e36.to(tl.float16)).to(tl.float16)
    e38 = (tl.trans(e37)).to(tl.float16)
    e39_slice00_lo, e39_slice00_hi = tl.split(tl.permute(tl.reshape(e38, (2, 16, 16)), (1, 2, 0)))
    e39_slice00_next_lo, e39_slice00_next_hi = tl.split(tl.permute(tl.reshape(e39_slice00_lo, (2, 8, 16)), (1, 2, 0)))
    e39_slice01_lo, e39_slice01_hi = tl.split(tl.permute(tl.reshape(e38, (2, 16, 16)), (1, 2, 0)))
    e39_slice01_next_lo, e39_slice01_next_hi = tl.split(tl.permute(tl.reshape(e39_slice01_hi, (2, 8, 16)), (1, 2, 0)))
    e39_slice0 = tl.reshape(tl.permute(tl.join(e39_slice00_next_hi, e39_slice01_next_lo), (2, 0, 1)), (16, 16))
    e39 = (e39_slice0).to(tl.float16)
    e40 = (tl.full((16, 16), 0.0, tl.float32)).to(tl.float32)
    e41 = (tl.dot(e39, e39, e40, input_precision='ieee')).to(tl.float32)
    e42 = (tl.where(tl.sum((e41 != e41).to(tl.int32), 1) > 0, float('nan'), tl.max(e41.to(tl.float32), 1))).to(tl.float32)
    e43 = (tl.reshape(e42, (16, 1))).to(tl.float32)
    e44 = ((e41 + e43)).to(tl.float32)
    e45 = (tl.full((16, 16), 0.125, tl.float32)).to(tl.float32)
    e46 = ((e44 * e45)).to(tl.float32)
    e47 = ((((tl.arange(0, 16)[:, None] * 16 + tl.arange(0, 16)[None, :]) + 44) % 256)).to(tl.int32)
    e48 = (tl.full((16, 16), True, tl.int1)).to(tl.int1)
    e49 = (tl.load(mem6 + bid * 288 + 16 + e47 * 1, (e48 & (e47 >= 0) & (e47 < 256)), other=0)).to(tl.float16)
    e50 = (e46.to(tl.float16)).to(tl.float16)
    e51 = (tl.dot(e50, e49, e40, input_precision='ieee')).to(tl.float32)
    e52 = (tl.reshape(e51, (16, 16))).to(tl.float32)
    e53 = (tl.reshape(e52, (16, 16))).to(tl.float32)
    e54 = (e53.to(tl.float16)).to(tl.float16)
    e63 = (e42.to(tl.float32)).to(tl.float32)
    e64, e65 = mixed_helper_extended_1_1(e54, e63, mem0, mem1, mem2, mem3, mem6, mem7, mem8, mem9, steps, limit, bid)
    e66 = (steps).to(tl.int32)
    e67 = (tl.full((), 0, tl.int32)).to(tl.int32)
    e68 = (tl.full((), 0, tl.int32)).to(tl.int32)
    e69 = (tl.full((), True, tl.int1)).to(tl.int1)
    e70 = (tl.load(mem7 + bid * 33 + 16 + e68 * 1, (e69 & (e68 >= 0) & (e68 < 1)), other=0)).to(tl.int32)
    e71 = (tl.full((), 0, tl.int32)).to(tl.int32)
    e72 = ((e70 < e71)).to(tl.int1)
    e92 = e64
    e93 = e65
    e94 = e67
    for e73 in range(tl.minimum(tl.maximum(e66, 0), 4)):
        e74 = e92
        e75 = e93
        e76 = e94
        if e72:
            e77 = e74
            e78 = (tl.full((16, 16), 0.125, tl.float16)).to(tl.float16)
            e79 = ((e77 + e78)).to(tl.float16)
            e83 = e79
        else:
            e80 = e74
            e81 = (tl.full((16, 16), 0.125, tl.float16)).to(tl.float16)
            e82 = ((e80 - e81)).to(tl.float16)
            e83 = e82
        e84, e85 = mixed_helper_extended_1_1(e83, e63, mem0, mem1, mem2, mem3, mem6, mem7, mem8, mem9, steps, limit, bid)
        e86 = (tl.full((16, 16), 0.125, tl.float16)).to(tl.float16)
        e87 = ((e84 * e86)).to(tl.float16)
        e88 = ((e75 + e85)).to(tl.float32)
        e89 = (tl.full((), 1, tl.int32)).to(tl.int32)
        e90 = ((e73 + e89)).to(tl.int32)
        e91 = ((e76 + e90)).to(tl.int32)
        e92 = e87
        e93 = e88
        e94 = e91
    e114 = e92
    e115 = e93
    e116 = e94
    e95 = tl.full((), 0, tl.int32)
    while e95 < tl.minimum(tl.maximum(e66, 0), 4):
        e96 = e114
        e97 = e115
        e98 = e116
        if e72:
            e99 = e96
            e100 = (tl.full((16, 16), 0.125, tl.float16)).to(tl.float16)
            e101 = ((e99 + e100)).to(tl.float16)
            e105 = e101
        else:
            e102 = e96
            e103 = (tl.full((16, 16), 0.125, tl.float16)).to(tl.float16)
            e104 = ((e102 - e103)).to(tl.float16)
            e105 = e104
        e106, e107 = mixed_helper_extended_1_1(e105, e63, mem0, mem1, mem2, mem3, mem6, mem7, mem8, mem9, steps, limit, bid)
        e108 = (tl.full((16, 16), 0.125, tl.float16)).to(tl.float16)
        e109 = ((e106 * e108)).to(tl.float16)
        e110 = ((e97 + e107)).to(tl.float32)
        e111 = (tl.full((), 1, tl.int32)).to(tl.int32)
        e112 = ((e95 + e111)).to(tl.int32)
        e113 = ((e98 + e112)).to(tl.int32)
        e114 = e109
        e115 = e110
        e116 = e113
        e95 += 1
    e117 = (tl.full((), -0.125, tl.float32)).to(tl.float32)
    e118 = (tl.broadcast_to(e117, (16, 16))).to(tl.float32)
    e119 = (tl.full((), -0.25, tl.float32)).to(tl.float32)
    e120 = (tl.broadcast_to(e119, (16, 16))).to(tl.float32)
    e121 = ((e118 - e120)).to(tl.float32)
    e122 = ((((255 - (tl.arange(0, 16)[:, None] * 16 + tl.arange(0, 16)[None, :])) + 28) % 256)).to(tl.int32)
    e123 = (tl.full((16, 16), True, tl.int1)).to(tl.int1)
    tl.atomic_max(mem8 + bid * 288 + 16 + e122 * 1, e121, (e123 & (e122 >= 0) & (e122 < 256)))
    e124 = ((e114 - e92)).to(tl.float16)
    e125 = (e124.to(tl.float32)).to(tl.float32)
    e126 = ((((255 - (tl.arange(0, 16)[:, None] * 16 + tl.arange(0, 16)[None, :])) + 194) % 256)).to(tl.int32)
    e127 = (tl.full((16, 16), True, tl.int1)).to(tl.int1)
    e128 = (tl.load(mem9 + bid * 288 + 16 + e126 * 1, (e127 & (e126 >= 0) & (e126 < 256)), other=0)).to(tl.float32)
    e129 = ((e125 - e128)).to(tl.float32)
    e130 = (e129.to(tl.float16)).to(tl.float16)
    e131 = ((e130 + e49)).to(tl.float16)
    e132 = (tl.where(tl.sum((e131 != e131).to(tl.int32), 0) > 0, float('nan'), tl.max(e131.to(tl.float32), 0))).to(tl.float32)
    tl.store(out_e131 + bid * 288 + 16 + (tl.arange(0, 16)[:, None] * 16 + tl.arange(0, 16)[None, :]), e131)
    tl.store(out_e16 + bid * 544 + 16 + (tl.arange(0, 16)[:, None] * 32 + tl.arange(0, 32)[None, :]), e16)
    tl.store(out_e22 + bid * 544 + 16 + (tl.arange(0, 16)[:, None] * 32 + tl.arange(0, 32)[None, :]), e22)
    tl.store(out_e115 + bid * 33 + 16 + tl.full((), 0, tl.int32), e115)
    tl.store(out_e116 + bid * 33 + 16 + tl.full((), 0, tl.int32), e116)
    tl.store(out_e132 + bid * 48 + 16 + tl.arange(0, 16), e132)
    tl.store(out_e3 + bid * 544 + 16 + (tl.arange(0, 16)[:, None] * 32 + tl.arange(0, 32)[None, :]), e3)
    tl.store(out_e11 + bid * 544 + 16 + (tl.arange(0, 16)[:, None] * 32 + tl.arange(0, 32)[None, :]), e11)
    tl.store(out_e28 + bid * 544 + 16 + (tl.arange(0, 16)[:, None] * 32 + tl.arange(0, 32)[None, :]), e28)
    tl.store(out_e32 + bid * 544 + 16 + (tl.arange(0, 16)[:, None] * 32 + tl.arange(0, 32)[None, :]), e32)
    tl.store(out_e41 + bid * 288 + 16 + (tl.arange(0, 16)[:, None] * 16 + tl.arange(0, 16)[None, :]), e41)
    tl.store(out_e46 + bid * 288 + 16 + (tl.arange(0, 16)[:, None] * 16 + tl.arange(0, 16)[None, :]), e46)
    tl.store(out_e21 + bid * 544 + 16 + (tl.arange(0, 16)[:, None] * 32 + tl.arange(0, 32)[None, :]), e21)
    tl.store(out_e4 + bid * 544 + 16 + (tl.arange(0, 16)[:, None] * 32 + tl.arange(0, 32)[None, :]), e4)

@triton.jit
def mixed_helper_extended_2_0(e55, e56, mem0, mem1, mem2, mem3, mem6, mem7, mem8, mem9, steps, limit, bid):
    e57 = (tl.sum(e55.to(tl.float32), 1)).to(tl.float32)
    e58 = ((e57 + e56)).to(tl.float32)
    e59 = (tl.sum(e58.to(tl.float32), 0)).to(tl.float32)
    e60 = (tl.reshape(e58, (16, 1))).to(tl.float32)
    e61 = (e60.to(tl.float16)).to(tl.float16)
    e62 = ((e55 + e61)).to(tl.float16)
    return e62, e59
@triton.jit
def extended_2_0(mem0, mem1, mem2, mem3, mem6, mem7, mem8, mem9, out_e131, out_e16, out_e22, out_e115, out_e116, out_e132, steps, limit):
    bid = tl.program_id(0)
    e1 = ((((tl.arange(0, 16)[:, None] * 32 + tl.arange(0, 32)[None, :]) + 10) % 512)).to(tl.int32)
    e2 = (tl.full((16, 32), True, tl.int1)).to(tl.int1)
    e3 = (tl.load(mem0 + bid * 544 + 16 + e1 * 1, (e2 & (e1 >= 0) & (e1 < 512)), other=0)).to(tl.float16)
    e4 = ((((tl.arange(0, 16)[:, None] * 32 + tl.arange(0, 32)[None, :]) + 17) % 512)).to(tl.int32)
    e5 = (tl.full((16, 32), True, tl.int1)).to(tl.int1)
    e6 = (tl.load(mem1 + bid * 544 + 16 + e4 * 1, (e5 & (e4 >= 0) & (e4 < 512)), other=0)).to(tl.float16)
    e7 = ((e3 * e6)).to(tl.float16)
    e8 = (e7.to(tl.float32)).to(tl.float32)
    e9 = (e8.to(tl.float16)).to(tl.float16)
    e10 = (tl.full((16, 32), 0.5, tl.float16)).to(tl.float16)
    e11 = ((e9 * e10)).to(tl.float16)
    e12 = ((((tl.arange(0, 16)[:, None] * 32 + tl.arange(0, 32)[None, :]) + 71) % 512)).to(tl.int32)
    e13 = (tl.full((16, 32), True, tl.int1)).to(tl.int1)
    e14 = (tl.load(mem2 + bid * 544 + 16 + e12 * 1, (e13 & (e12 >= 0) & (e12 < 512)), other=0)).to(tl.int32)
    e15 = (tl.full((16, 32), 3, tl.int32)).to(tl.int32)
    e16 = ((e14 & e15)).to(tl.int32)
    e17 = (tl.full((16, 32), 3, tl.int32)).to(tl.int32)
    e18 = ((e16 < e17)).to(tl.int1)
    e19 = ((((511 - (tl.arange(0, 16)[:, None] * 32 + tl.arange(0, 32)[None, :])) + 252) % 512)).to(tl.int32)
    e20 = (limit).to(tl.int32)
    e21 = ((e19 < e20)).to(tl.int1)
    e22 = ((e18 & e21)).to(tl.int1)
    e23 = (tl.full((16, 32), -0.125, tl.float16)).to(tl.float16)
    e24 = (tl.where(e22, e11, e23)).to(tl.float16)
    e25 = ((((511 - (tl.arange(0, 16)[:, None] * 32 + tl.arange(0, 32)[None, :])) + 305) % 512)).to(tl.int32)
    e26 = (limit).to(tl.int32)
    e27 = ((e25 < e26)).to(tl.int1)
    tl.store(mem3 + bid * 1060 + 17 + e25 * 2, e24, (e27 & (e25 >= 0) & (e25 < 512)))
    tl.debug_barrier()
    e28 = (tl.load(mem3 + bid * 1060 + 17 + e25 * 2, (e27 & (e25 >= 0) & (e25 < 512)), other=0)).to(tl.float16)
    e29 = (tl.full((16, 32), 0.125, tl.float16)).to(tl.float16)
    e30 = ((e28 + e29)).to(tl.float16)
    tl.debug_barrier()
    tl.store(mem3 + bid * 1060 + 18 + e25 * 1, e30, (e27 & (e25 >= 0) & (e25 < 512)))
    tl.debug_barrier()
    e31 = (tl.full((16, 32), True, tl.int1)).to(tl.int1)
    e32 = (tl.load(mem3 + bid * 1060 + 17 + e25 * 2, (e31 & (e25 >= 0) & (e25 < 512)), other=0)).to(tl.float16)
    e33 = (tl.full((16, 32), 3, tl.int32)).to(tl.int32)
    e34 = ((e25 ^ e33)).to(tl.int32)
    e35 = (tl.load(mem3 + bid * 1060 + 18 + e34 * 1, (e27 & (e34 >= 0) & (e34 < 512)), other=0)).to(tl.float16)
    e36 = ((e32 + e35)).to(tl.float16)
    e37 = (e36.to(tl.float16)).to(tl.float16)
    e38 = (tl.trans(e37)).to(tl.float16)
    e39_slice00_lo, e39_slice00_hi = tl.split(tl.permute(tl.reshape(e38, (2, 16, 16)), (1, 2, 0)))
    e39_slice00_next_lo, e39_slice00_next_hi = tl.split(tl.permute(tl.reshape(e39_slice00_lo, (2, 8, 16)), (1, 2, 0)))
    e39_slice01_lo, e39_slice01_hi = tl.split(tl.permute(tl.reshape(e38, (2, 16, 16)), (1, 2, 0)))
    e39_slice01_next_lo, e39_slice01_next_hi = tl.split(tl.permute(tl.reshape(e39_slice01_hi, (2, 8, 16)), (1, 2, 0)))
    e39_slice0 = tl.reshape(tl.permute(tl.join(e39_slice00_next_hi, e39_slice01_next_lo), (2, 0, 1)), (16, 16))
    e39 = (e39_slice0).to(tl.float16)
    e40 = (tl.full((16, 16), 0.0, tl.float32)).to(tl.float32)
    e41 = (tl.dot(e39, e39, e40, input_precision='ieee')).to(tl.float32)
    e42 = (tl.where(tl.sum((e41 != e41).to(tl.int32), 1) > 0, float('nan'), tl.max(e41.to(tl.float32), 1))).to(tl.float32)
    e43 = (tl.reshape(e42, (16, 1))).to(tl.float32)
    e44 = ((e41 + e43)).to(tl.float32)
    e45 = (tl.full((16, 16), 0.125, tl.float32)).to(tl.float32)
    e46 = ((e44 * e45)).to(tl.float32)
    e47 = ((((tl.arange(0, 16)[:, None] * 16 + tl.arange(0, 16)[None, :]) + 44) % 256)).to(tl.int32)
    e48 = (tl.full((16, 16), True, tl.int1)).to(tl.int1)
    e49 = (tl.load(mem6 + bid * 288 + 16 + e47 * 1, (e48 & (e47 >= 0) & (e47 < 256)), other=0)).to(tl.float16)
    e50 = (e46.to(tl.float16)).to(tl.float16)
    e51 = (tl.dot(e50, e49, e40, input_precision='ieee')).to(tl.float32)
    e52 = (tl.reshape(e51, (16, 16))).to(tl.float32)
    e53 = (tl.reshape(e52, (16, 16))).to(tl.float32)
    e54 = (e53.to(tl.float16)).to(tl.float16)
    e63 = (e42.to(tl.float32)).to(tl.float32)
    e64, e65 = mixed_helper_extended_2_0(e54, e63, mem0, mem1, mem2, mem3, mem6, mem7, mem8, mem9, steps, limit, bid)
    e66 = (steps).to(tl.int32)
    e67 = (tl.full((), 0, tl.int32)).to(tl.int32)
    e68 = (tl.full((), 0, tl.int32)).to(tl.int32)
    e69 = (tl.full((), True, tl.int1)).to(tl.int1)
    e70 = (tl.load(mem7 + bid * 33 + 16 + e68 * 1, (e69 & (e68 >= 0) & (e68 < 1)), other=0)).to(tl.int32)
    e71 = (tl.full((), 0, tl.int32)).to(tl.int32)
    e72 = ((e70 < e71)).to(tl.int1)
    e92 = e64
    e93 = e65
    e94 = e67
    for e73 in range(tl.minimum(tl.maximum(e66, 0), 4)):
        e74 = e92
        e75 = e93
        e76 = e94
        if e72:
            e77 = e74
            e78 = (tl.full((16, 16), 0.125, tl.float16)).to(tl.float16)
            e79 = ((e77 + e78)).to(tl.float16)
            e83 = e79
        else:
            e80 = e74
            e81 = (tl.full((16, 16), 0.125, tl.float16)).to(tl.float16)
            e82 = ((e80 - e81)).to(tl.float16)
            e83 = e82
        e84, e85 = mixed_helper_extended_2_0(e83, e63, mem0, mem1, mem2, mem3, mem6, mem7, mem8, mem9, steps, limit, bid)
        e86 = (tl.full((16, 16), 0.125, tl.float16)).to(tl.float16)
        e87 = ((e84 * e86)).to(tl.float16)
        e88 = ((e75 + e85)).to(tl.float32)
        e89 = (tl.full((), 1, tl.int32)).to(tl.int32)
        e90 = ((e73 + e89)).to(tl.int32)
        e91 = ((e76 + e90)).to(tl.int32)
        e92 = e87
        e93 = e88
        e94 = e91
    e114 = e92
    e115 = e93
    e116 = e94
    e95 = tl.full((), 0, tl.int32)
    while e95 < tl.minimum(tl.maximum(e66, 0), 4):
        e96 = e114
        e97 = e115
        e98 = e116
        if e72:
            e99 = e96
            e100 = (tl.full((16, 16), 0.125, tl.float16)).to(tl.float16)
            e101 = ((e99 + e100)).to(tl.float16)
            e105 = e101
        else:
            e102 = e96
            e103 = (tl.full((16, 16), 0.125, tl.float16)).to(tl.float16)
            e104 = ((e102 - e103)).to(tl.float16)
            e105 = e104
        e106, e107 = mixed_helper_extended_2_0(e105, e63, mem0, mem1, mem2, mem3, mem6, mem7, mem8, mem9, steps, limit, bid)
        e108 = (tl.full((16, 16), 0.125, tl.float16)).to(tl.float16)
        e109 = ((e106 * e108)).to(tl.float16)
        e110 = ((e97 + e107)).to(tl.float32)
        e111 = (tl.full((), 1, tl.int32)).to(tl.int32)
        e112 = ((e95 + e111)).to(tl.int32)
        e113 = ((e98 + e112)).to(tl.int32)
        e114 = e109
        e115 = e110
        e116 = e113
        e95 += 1
    e117 = (tl.full((), -0.125, tl.float32)).to(tl.float32)
    e118 = (tl.broadcast_to(e117, (16, 16))).to(tl.float32)
    e119 = (tl.full((), -0.25, tl.float32)).to(tl.float32)
    e120 = (tl.broadcast_to(e119, (16, 16))).to(tl.float32)
    e121 = ((e118 - e120)).to(tl.float32)
    e122 = ((((255 - (tl.arange(0, 16)[:, None] * 16 + tl.arange(0, 16)[None, :])) + 28) % 256)).to(tl.int32)
    e123 = (tl.full((16, 16), True, tl.int1)).to(tl.int1)
    tl.atomic_max(mem8 + bid * 288 + 16 + e122 * 1, e121, (e123 & (e122 >= 0) & (e122 < 256)))
    e124 = ((e114 - e92)).to(tl.float16)
    e125 = (e124.to(tl.float32)).to(tl.float32)
    e126 = ((((255 - (tl.arange(0, 16)[:, None] * 16 + tl.arange(0, 16)[None, :])) + 194) % 256)).to(tl.int32)
    e127 = (tl.full((16, 16), True, tl.int1)).to(tl.int1)
    e128 = (tl.load(mem9 + bid * 288 + 16 + e126 * 1, (e127 & (e126 >= 0) & (e126 < 256)), other=0)).to(tl.float32)
    e129 = ((e125 - e128)).to(tl.float32)
    e130 = (e129.to(tl.float16)).to(tl.float16)
    e131 = ((e130 + e49)).to(tl.float16)
    e132 = (tl.where(tl.sum((e131 != e131).to(tl.int32), 0) > 0, float('nan'), tl.max(e131.to(tl.float32), 0))).to(tl.float32)
    tl.store(out_e131 + bid * 288 + 16 + (tl.arange(0, 16)[:, None] * 16 + tl.arange(0, 16)[None, :]), e131)
    tl.store(out_e16 + bid * 544 + 16 + (tl.arange(0, 16)[:, None] * 32 + tl.arange(0, 32)[None, :]), e16)
    tl.store(out_e22 + bid * 544 + 16 + (tl.arange(0, 16)[:, None] * 32 + tl.arange(0, 32)[None, :]), e22)
    tl.store(out_e115 + bid * 33 + 16 + tl.full((), 0, tl.int32), e115)
    tl.store(out_e116 + bid * 33 + 16 + tl.full((), 0, tl.int32), e116)
    tl.store(out_e132 + bid * 48 + 16 + tl.arange(0, 16), e132)

@triton.jit
def mixed_helper_extended_2_1(e55, e56, mem0, mem1, mem2, mem3, mem6, mem7, mem8, mem9, steps, limit, bid):
    e57 = (tl.sum(e55.to(tl.float32), 1)).to(tl.float32)
    e58 = ((e57 + e56)).to(tl.float32)
    e59 = (tl.sum(e58.to(tl.float32), 0)).to(tl.float32)
    e60 = (tl.reshape(e58, (16, 1))).to(tl.float32)
    e61 = (e60.to(tl.float16)).to(tl.float16)
    e62 = ((e55 + e61)).to(tl.float16)
    return e62, e59
@triton.jit
def extended_2_1(mem0, mem1, mem2, mem3, mem6, mem7, mem8, mem9, out_e131, out_e16, out_e22, out_e115, out_e116, out_e132, out_e3, out_e11, out_e28, out_e32, out_e41, out_e46, out_e21, out_e4, steps, limit):
    bid = tl.program_id(0)
    e1 = ((((tl.arange(0, 16)[:, None] * 32 + tl.arange(0, 32)[None, :]) + 10) % 512)).to(tl.int32)
    e2 = (tl.full((16, 32), True, tl.int1)).to(tl.int1)
    e3 = (tl.load(mem0 + bid * 544 + 16 + e1 * 1, (e2 & (e1 >= 0) & (e1 < 512)), other=0)).to(tl.float16)
    e4 = ((((tl.arange(0, 16)[:, None] * 32 + tl.arange(0, 32)[None, :]) + 17) % 512)).to(tl.int32)
    e5 = (tl.full((16, 32), True, tl.int1)).to(tl.int1)
    e6 = (tl.load(mem1 + bid * 544 + 16 + e4 * 1, (e5 & (e4 >= 0) & (e4 < 512)), other=0)).to(tl.float16)
    e7 = ((e3 * e6)).to(tl.float16)
    e8 = (e7.to(tl.float32)).to(tl.float32)
    e9 = (e8.to(tl.float16)).to(tl.float16)
    e10 = (tl.full((16, 32), 0.5, tl.float16)).to(tl.float16)
    e11 = ((e9 * e10)).to(tl.float16)
    e12 = ((((tl.arange(0, 16)[:, None] * 32 + tl.arange(0, 32)[None, :]) + 71) % 512)).to(tl.int32)
    e13 = (tl.full((16, 32), True, tl.int1)).to(tl.int1)
    e14 = (tl.load(mem2 + bid * 544 + 16 + e12 * 1, (e13 & (e12 >= 0) & (e12 < 512)), other=0)).to(tl.int32)
    e15 = (tl.full((16, 32), 3, tl.int32)).to(tl.int32)
    e16 = ((e14 & e15)).to(tl.int32)
    e17 = (tl.full((16, 32), 3, tl.int32)).to(tl.int32)
    e18 = ((e16 < e17)).to(tl.int1)
    e19 = ((((511 - (tl.arange(0, 16)[:, None] * 32 + tl.arange(0, 32)[None, :])) + 252) % 512)).to(tl.int32)
    e20 = (limit).to(tl.int32)
    e21 = ((e19 < e20)).to(tl.int1)
    e22 = ((e18 & e21)).to(tl.int1)
    e23 = (tl.full((16, 32), -0.125, tl.float16)).to(tl.float16)
    e24 = (tl.where(e22, e11, e23)).to(tl.float16)
    e25 = ((((511 - (tl.arange(0, 16)[:, None] * 32 + tl.arange(0, 32)[None, :])) + 305) % 512)).to(tl.int32)
    e26 = (limit).to(tl.int32)
    e27 = ((e25 < e26)).to(tl.int1)
    tl.store(mem3 + bid * 1060 + 17 + e25 * 2, e24, (e27 & (e25 >= 0) & (e25 < 512)))
    tl.debug_barrier()
    e28 = (tl.load(mem3 + bid * 1060 + 17 + e25 * 2, (e27 & (e25 >= 0) & (e25 < 512)), other=0)).to(tl.float16)
    e29 = (tl.full((16, 32), 0.125, tl.float16)).to(tl.float16)
    e30 = ((e28 + e29)).to(tl.float16)
    tl.debug_barrier()
    tl.store(mem3 + bid * 1060 + 18 + e25 * 1, e30, (e27 & (e25 >= 0) & (e25 < 512)))
    tl.debug_barrier()
    e31 = (tl.full((16, 32), True, tl.int1)).to(tl.int1)
    e32 = (tl.load(mem3 + bid * 1060 + 17 + e25 * 2, (e31 & (e25 >= 0) & (e25 < 512)), other=0)).to(tl.float16)
    e33 = (tl.full((16, 32), 3, tl.int32)).to(tl.int32)
    e34 = ((e25 ^ e33)).to(tl.int32)
    e35 = (tl.load(mem3 + bid * 1060 + 18 + e34 * 1, (e27 & (e34 >= 0) & (e34 < 512)), other=0)).to(tl.float16)
    e36 = ((e32 + e35)).to(tl.float16)
    e37 = (e36.to(tl.float16)).to(tl.float16)
    e38 = (tl.trans(e37)).to(tl.float16)
    e39_slice00_lo, e39_slice00_hi = tl.split(tl.permute(tl.reshape(e38, (2, 16, 16)), (1, 2, 0)))
    e39_slice00_next_lo, e39_slice00_next_hi = tl.split(tl.permute(tl.reshape(e39_slice00_lo, (2, 8, 16)), (1, 2, 0)))
    e39_slice01_lo, e39_slice01_hi = tl.split(tl.permute(tl.reshape(e38, (2, 16, 16)), (1, 2, 0)))
    e39_slice01_next_lo, e39_slice01_next_hi = tl.split(tl.permute(tl.reshape(e39_slice01_hi, (2, 8, 16)), (1, 2, 0)))
    e39_slice0 = tl.reshape(tl.permute(tl.join(e39_slice00_next_hi, e39_slice01_next_lo), (2, 0, 1)), (16, 16))
    e39 = (e39_slice0).to(tl.float16)
    e40 = (tl.full((16, 16), 0.0, tl.float32)).to(tl.float32)
    e41 = (tl.dot(e39, e39, e40, input_precision='ieee')).to(tl.float32)
    e42 = (tl.where(tl.sum((e41 != e41).to(tl.int32), 1) > 0, float('nan'), tl.max(e41.to(tl.float32), 1))).to(tl.float32)
    e43 = (tl.reshape(e42, (16, 1))).to(tl.float32)
    e44 = ((e41 + e43)).to(tl.float32)
    e45 = (tl.full((16, 16), 0.125, tl.float32)).to(tl.float32)
    e46 = ((e44 * e45)).to(tl.float32)
    e47 = ((((tl.arange(0, 16)[:, None] * 16 + tl.arange(0, 16)[None, :]) + 44) % 256)).to(tl.int32)
    e48 = (tl.full((16, 16), True, tl.int1)).to(tl.int1)
    e49 = (tl.load(mem6 + bid * 288 + 16 + e47 * 1, (e48 & (e47 >= 0) & (e47 < 256)), other=0)).to(tl.float16)
    e50 = (e46.to(tl.float16)).to(tl.float16)
    e51 = (tl.dot(e50, e49, e40, input_precision='ieee')).to(tl.float32)
    e52 = (tl.reshape(e51, (16, 16))).to(tl.float32)
    e53 = (tl.reshape(e52, (16, 16))).to(tl.float32)
    e54 = (e53.to(tl.float16)).to(tl.float16)
    e63 = (e42.to(tl.float32)).to(tl.float32)
    e64, e65 = mixed_helper_extended_2_1(e54, e63, mem0, mem1, mem2, mem3, mem6, mem7, mem8, mem9, steps, limit, bid)
    e66 = (steps).to(tl.int32)
    e67 = (tl.full((), 0, tl.int32)).to(tl.int32)
    e68 = (tl.full((), 0, tl.int32)).to(tl.int32)
    e69 = (tl.full((), True, tl.int1)).to(tl.int1)
    e70 = (tl.load(mem7 + bid * 33 + 16 + e68 * 1, (e69 & (e68 >= 0) & (e68 < 1)), other=0)).to(tl.int32)
    e71 = (tl.full((), 0, tl.int32)).to(tl.int32)
    e72 = ((e70 < e71)).to(tl.int1)
    e92 = e64
    e93 = e65
    e94 = e67
    for e73 in range(tl.minimum(tl.maximum(e66, 0), 4)):
        e74 = e92
        e75 = e93
        e76 = e94
        if e72:
            e77 = e74
            e78 = (tl.full((16, 16), 0.125, tl.float16)).to(tl.float16)
            e79 = ((e77 + e78)).to(tl.float16)
            e83 = e79
        else:
            e80 = e74
            e81 = (tl.full((16, 16), 0.125, tl.float16)).to(tl.float16)
            e82 = ((e80 - e81)).to(tl.float16)
            e83 = e82
        e84, e85 = mixed_helper_extended_2_1(e83, e63, mem0, mem1, mem2, mem3, mem6, mem7, mem8, mem9, steps, limit, bid)
        e86 = (tl.full((16, 16), 0.125, tl.float16)).to(tl.float16)
        e87 = ((e84 * e86)).to(tl.float16)
        e88 = ((e75 + e85)).to(tl.float32)
        e89 = (tl.full((), 1, tl.int32)).to(tl.int32)
        e90 = ((e73 + e89)).to(tl.int32)
        e91 = ((e76 + e90)).to(tl.int32)
        e92 = e87
        e93 = e88
        e94 = e91
    e114 = e92
    e115 = e93
    e116 = e94
    e95 = tl.full((), 0, tl.int32)
    while e95 < tl.minimum(tl.maximum(e66, 0), 4):
        e96 = e114
        e97 = e115
        e98 = e116
        if e72:
            e99 = e96
            e100 = (tl.full((16, 16), 0.125, tl.float16)).to(tl.float16)
            e101 = ((e99 + e100)).to(tl.float16)
            e105 = e101
        else:
            e102 = e96
            e103 = (tl.full((16, 16), 0.125, tl.float16)).to(tl.float16)
            e104 = ((e102 - e103)).to(tl.float16)
            e105 = e104
        e106, e107 = mixed_helper_extended_2_1(e105, e63, mem0, mem1, mem2, mem3, mem6, mem7, mem8, mem9, steps, limit, bid)
        e108 = (tl.full((16, 16), 0.125, tl.float16)).to(tl.float16)
        e109 = ((e106 * e108)).to(tl.float16)
        e110 = ((e97 + e107)).to(tl.float32)
        e111 = (tl.full((), 1, tl.int32)).to(tl.int32)
        e112 = ((e95 + e111)).to(tl.int32)
        e113 = ((e98 + e112)).to(tl.int32)
        e114 = e109
        e115 = e110
        e116 = e113
        e95 += 1
    e117 = (tl.full((), -0.125, tl.float32)).to(tl.float32)
    e118 = (tl.broadcast_to(e117, (16, 16))).to(tl.float32)
    e119 = (tl.full((), -0.25, tl.float32)).to(tl.float32)
    e120 = (tl.broadcast_to(e119, (16, 16))).to(tl.float32)
    e121 = ((e118 - e120)).to(tl.float32)
    e122 = ((((255 - (tl.arange(0, 16)[:, None] * 16 + tl.arange(0, 16)[None, :])) + 28) % 256)).to(tl.int32)
    e123 = (tl.full((16, 16), True, tl.int1)).to(tl.int1)
    tl.atomic_max(mem8 + bid * 288 + 16 + e122 * 1, e121, (e123 & (e122 >= 0) & (e122 < 256)))
    e124 = ((e114 - e92)).to(tl.float16)
    e125 = (e124.to(tl.float32)).to(tl.float32)
    e126 = ((((255 - (tl.arange(0, 16)[:, None] * 16 + tl.arange(0, 16)[None, :])) + 194) % 256)).to(tl.int32)
    e127 = (tl.full((16, 16), True, tl.int1)).to(tl.int1)
    e128 = (tl.load(mem9 + bid * 288 + 16 + e126 * 1, (e127 & (e126 >= 0) & (e126 < 256)), other=0)).to(tl.float32)
    e129 = ((e125 - e128)).to(tl.float32)
    e130 = (e129.to(tl.float16)).to(tl.float16)
    e131 = ((e130 + e49)).to(tl.float16)
    e132 = (tl.where(tl.sum((e131 != e131).to(tl.int32), 0) > 0, float('nan'), tl.max(e131.to(tl.float32), 0))).to(tl.float32)
    tl.store(out_e131 + bid * 288 + 16 + (tl.arange(0, 16)[:, None] * 16 + tl.arange(0, 16)[None, :]), e131)
    tl.store(out_e16 + bid * 544 + 16 + (tl.arange(0, 16)[:, None] * 32 + tl.arange(0, 32)[None, :]), e16)
    tl.store(out_e22 + bid * 544 + 16 + (tl.arange(0, 16)[:, None] * 32 + tl.arange(0, 32)[None, :]), e22)
    tl.store(out_e115 + bid * 33 + 16 + tl.full((), 0, tl.int32), e115)
    tl.store(out_e116 + bid * 33 + 16 + tl.full((), 0, tl.int32), e116)
    tl.store(out_e132 + bid * 48 + 16 + tl.arange(0, 16), e132)
    tl.store(out_e3 + bid * 544 + 16 + (tl.arange(0, 16)[:, None] * 32 + tl.arange(0, 32)[None, :]), e3)
    tl.store(out_e11 + bid * 544 + 16 + (tl.arange(0, 16)[:, None] * 32 + tl.arange(0, 32)[None, :]), e11)
    tl.store(out_e28 + bid * 544 + 16 + (tl.arange(0, 16)[:, None] * 32 + tl.arange(0, 32)[None, :]), e28)
    tl.store(out_e32 + bid * 544 + 16 + (tl.arange(0, 16)[:, None] * 32 + tl.arange(0, 32)[None, :]), e32)
    tl.store(out_e41 + bid * 288 + 16 + (tl.arange(0, 16)[:, None] * 16 + tl.arange(0, 16)[None, :]), e41)
    tl.store(out_e46 + bid * 288 + 16 + (tl.arange(0, 16)[:, None] * 16 + tl.arange(0, 16)[None, :]), e46)
    tl.store(out_e21 + bid * 544 + 16 + (tl.arange(0, 16)[:, None] * 32 + tl.arange(0, 32)[None, :]), e21)
    tl.store(out_e4 + bid * 544 + 16 + (tl.arange(0, 16)[:, None] * 32 + tl.arange(0, 32)[None, :]), e4)

@triton.jit
def mixed_helper_extended_3_0(e55, e56, mem0, mem1, mem2, mem3, mem6, mem7, mem8, mem9, steps, limit, bid):
    e57 = (tl.sum(e55.to(tl.float32), 1)).to(tl.float32)
    e58 = ((e57 + e56)).to(tl.float32)
    e59 = (tl.sum(e58.to(tl.float32), 0)).to(tl.float32)
    e60 = (tl.reshape(e58, (16, 1))).to(tl.float32)
    e61 = (e60.to(tl.float16)).to(tl.float16)
    e62 = ((e55 + e61)).to(tl.float16)
    return e62, e59
@triton.jit
def extended_3_0(mem0, mem1, mem2, mem3, mem6, mem7, mem8, mem9, out_e131, out_e16, out_e22, out_e115, out_e116, out_e132, steps, limit):
    bid = tl.program_id(0)
    e1 = ((((tl.arange(0, 16)[:, None] * 32 + tl.arange(0, 32)[None, :]) + 10) % 512)).to(tl.int32)
    e2 = (tl.full((16, 32), True, tl.int1)).to(tl.int1)
    e3 = (tl.load(mem0 + bid * 544 + 16 + e1 * 1, (e2 & (e1 >= 0) & (e1 < 512)), other=0)).to(tl.float16)
    e4 = ((((tl.arange(0, 16)[:, None] * 32 + tl.arange(0, 32)[None, :]) + 17) % 512)).to(tl.int32)
    e5 = (tl.full((16, 32), True, tl.int1)).to(tl.int1)
    e6 = (tl.load(mem1 + bid * 544 + 16 + e4 * 1, (e5 & (e4 >= 0) & (e4 < 512)), other=0)).to(tl.float16)
    e7 = ((e3 * e6)).to(tl.float16)
    e8 = (e7.to(tl.float32)).to(tl.float32)
    e9 = (e8.to(tl.float16)).to(tl.float16)
    e10 = (tl.full((16, 32), 0.5, tl.float16)).to(tl.float16)
    e11 = ((e9 * e10)).to(tl.float16)
    e12 = ((((tl.arange(0, 16)[:, None] * 32 + tl.arange(0, 32)[None, :]) + 71) % 512)).to(tl.int32)
    e13 = (tl.full((16, 32), True, tl.int1)).to(tl.int1)
    e14 = (tl.load(mem2 + bid * 544 + 16 + e12 * 1, (e13 & (e12 >= 0) & (e12 < 512)), other=0)).to(tl.int32)
    e15 = (tl.full((16, 32), 3, tl.int32)).to(tl.int32)
    e16 = ((e14 & e15)).to(tl.int32)
    e17 = (tl.full((16, 32), 3, tl.int32)).to(tl.int32)
    e18 = ((e16 < e17)).to(tl.int1)
    e19 = ((((511 - (tl.arange(0, 16)[:, None] * 32 + tl.arange(0, 32)[None, :])) + 252) % 512)).to(tl.int32)
    e20 = (limit).to(tl.int32)
    e21 = ((e19 < e20)).to(tl.int1)
    e22 = ((e18 & e21)).to(tl.int1)
    e23 = (tl.full((16, 32), -0.125, tl.float16)).to(tl.float16)
    e24 = (tl.where(e22, e11, e23)).to(tl.float16)
    e25 = ((((511 - (tl.arange(0, 16)[:, None] * 32 + tl.arange(0, 32)[None, :])) + 305) % 512)).to(tl.int32)
    e26 = (limit).to(tl.int32)
    e27 = ((e25 < e26)).to(tl.int1)
    tl.store(mem3 + bid * 1060 + 17 + e25 * 2, e24, (e27 & (e25 >= 0) & (e25 < 512)))
    tl.debug_barrier()
    e28 = (tl.load(mem3 + bid * 1060 + 17 + e25 * 2, (e27 & (e25 >= 0) & (e25 < 512)), other=0)).to(tl.float16)
    e29 = (tl.full((16, 32), 0.125, tl.float16)).to(tl.float16)
    e30 = ((e28 + e29)).to(tl.float16)
    tl.debug_barrier()
    tl.store(mem3 + bid * 1060 + 18 + e25 * 1, e30, (e27 & (e25 >= 0) & (e25 < 512)))
    tl.debug_barrier()
    e31 = (tl.full((16, 32), True, tl.int1)).to(tl.int1)
    e32 = (tl.load(mem3 + bid * 1060 + 17 + e25 * 2, (e31 & (e25 >= 0) & (e25 < 512)), other=0)).to(tl.float16)
    e33 = (tl.full((16, 32), 3, tl.int32)).to(tl.int32)
    e34 = ((e25 ^ e33)).to(tl.int32)
    e35 = (tl.load(mem3 + bid * 1060 + 18 + e34 * 1, (e27 & (e34 >= 0) & (e34 < 512)), other=0)).to(tl.float16)
    e36 = ((e32 + e35)).to(tl.float16)
    e37 = (e36.to(tl.float16)).to(tl.float16)
    e38 = (tl.trans(e37)).to(tl.float16)
    e39_slice00_lo, e39_slice00_hi = tl.split(tl.permute(tl.reshape(e38, (2, 16, 16)), (1, 2, 0)))
    e39_slice00_next_lo, e39_slice00_next_hi = tl.split(tl.permute(tl.reshape(e39_slice00_lo, (2, 8, 16)), (1, 2, 0)))
    e39_slice01_lo, e39_slice01_hi = tl.split(tl.permute(tl.reshape(e38, (2, 16, 16)), (1, 2, 0)))
    e39_slice01_next_lo, e39_slice01_next_hi = tl.split(tl.permute(tl.reshape(e39_slice01_hi, (2, 8, 16)), (1, 2, 0)))
    e39_slice0 = tl.reshape(tl.permute(tl.join(e39_slice00_next_hi, e39_slice01_next_lo), (2, 0, 1)), (16, 16))
    e39 = (e39_slice0).to(tl.float16)
    e40 = (tl.full((16, 16), 0.0, tl.float32)).to(tl.float32)
    e41 = (tl.dot(e39, e39, e40, input_precision='ieee')).to(tl.float32)
    e42 = (tl.where(tl.sum((e41 != e41).to(tl.int32), 1) > 0, float('nan'), tl.max(e41.to(tl.float32), 1))).to(tl.float32)
    e43 = (tl.reshape(e42, (16, 1))).to(tl.float32)
    e44 = ((e41 + e43)).to(tl.float32)
    e45 = (tl.full((16, 16), 0.125, tl.float32)).to(tl.float32)
    e46 = ((e44 * e45)).to(tl.float32)
    e47 = ((((tl.arange(0, 16)[:, None] * 16 + tl.arange(0, 16)[None, :]) + 44) % 256)).to(tl.int32)
    e48 = (tl.full((16, 16), True, tl.int1)).to(tl.int1)
    e49 = (tl.load(mem6 + bid * 288 + 16 + e47 * 1, (e48 & (e47 >= 0) & (e47 < 256)), other=0)).to(tl.float16)
    e50 = (e46.to(tl.float16)).to(tl.float16)
    e51 = (tl.dot(e50, e49, e40, input_precision='ieee')).to(tl.float32)
    e52 = (tl.reshape(e51, (16, 16))).to(tl.float32)
    e53 = (tl.reshape(e52, (16, 16))).to(tl.float32)
    e54 = (e53.to(tl.float16)).to(tl.float16)
    e63 = (e42.to(tl.float32)).to(tl.float32)
    e64, e65 = mixed_helper_extended_3_0(e54, e63, mem0, mem1, mem2, mem3, mem6, mem7, mem8, mem9, steps, limit, bid)
    e66 = (steps).to(tl.int32)
    e67 = (tl.full((), 0, tl.int32)).to(tl.int32)
    e68 = (tl.full((), 0, tl.int32)).to(tl.int32)
    e69 = (tl.full((), True, tl.int1)).to(tl.int1)
    e70 = (tl.load(mem7 + bid * 33 + 16 + e68 * 1, (e69 & (e68 >= 0) & (e68 < 1)), other=0)).to(tl.int32)
    e71 = (tl.full((), 0, tl.int32)).to(tl.int32)
    e72 = ((e70 < e71)).to(tl.int1)
    e92 = e64
    e93 = e65
    e94 = e67
    for e73 in range(tl.minimum(tl.maximum(e66, 0), 4)):
        e74 = e92
        e75 = e93
        e76 = e94
        if e72:
            e77 = e74
            e78 = (tl.full((16, 16), 0.125, tl.float16)).to(tl.float16)
            e79 = ((e77 + e78)).to(tl.float16)
            e83 = e79
        else:
            e80 = e74
            e81 = (tl.full((16, 16), 0.125, tl.float16)).to(tl.float16)
            e82 = ((e80 - e81)).to(tl.float16)
            e83 = e82
        e84, e85 = mixed_helper_extended_3_0(e83, e63, mem0, mem1, mem2, mem3, mem6, mem7, mem8, mem9, steps, limit, bid)
        e86 = (tl.full((16, 16), 0.125, tl.float16)).to(tl.float16)
        e87 = ((e84 * e86)).to(tl.float16)
        e88 = ((e75 + e85)).to(tl.float32)
        e89 = (tl.full((), 1, tl.int32)).to(tl.int32)
        e90 = ((e73 + e89)).to(tl.int32)
        e91 = ((e76 + e90)).to(tl.int32)
        e92 = e87
        e93 = e88
        e94 = e91
    e114 = e92
    e115 = e93
    e116 = e94
    e95 = tl.full((), 0, tl.int32)
    while e95 < tl.minimum(tl.maximum(e66, 0), 4):
        e96 = e114
        e97 = e115
        e98 = e116
        if e72:
            e99 = e96
            e100 = (tl.full((16, 16), 0.125, tl.float16)).to(tl.float16)
            e101 = ((e99 + e100)).to(tl.float16)
            e105 = e101
        else:
            e102 = e96
            e103 = (tl.full((16, 16), 0.125, tl.float16)).to(tl.float16)
            e104 = ((e102 - e103)).to(tl.float16)
            e105 = e104
        e106, e107 = mixed_helper_extended_3_0(e105, e63, mem0, mem1, mem2, mem3, mem6, mem7, mem8, mem9, steps, limit, bid)
        e108 = (tl.full((16, 16), 0.125, tl.float16)).to(tl.float16)
        e109 = ((e106 * e108)).to(tl.float16)
        e110 = ((e97 + e107)).to(tl.float32)
        e111 = (tl.full((), 1, tl.int32)).to(tl.int32)
        e112 = ((e95 + e111)).to(tl.int32)
        e113 = ((e98 + e112)).to(tl.int32)
        e114 = e109
        e115 = e110
        e116 = e113
        e95 += 1
    e117 = (tl.full((), -0.125, tl.float32)).to(tl.float32)
    e118 = (tl.broadcast_to(e117, (16, 16))).to(tl.float32)
    e119 = (tl.full((), -0.25, tl.float32)).to(tl.float32)
    e120 = (tl.broadcast_to(e119, (16, 16))).to(tl.float32)
    e121 = ((e118 - e120)).to(tl.float32)
    e122 = ((((255 - (tl.arange(0, 16)[:, None] * 16 + tl.arange(0, 16)[None, :])) + 28) % 256)).to(tl.int32)
    e123 = (tl.full((16, 16), True, tl.int1)).to(tl.int1)
    tl.atomic_max(mem8 + bid * 288 + 16 + e122 * 1, e121, (e123 & (e122 >= 0) & (e122 < 256)))
    e124 = ((e114 - e92)).to(tl.float16)
    e125 = (e124.to(tl.float32)).to(tl.float32)
    e126 = ((((255 - (tl.arange(0, 16)[:, None] * 16 + tl.arange(0, 16)[None, :])) + 194) % 256)).to(tl.int32)
    e127 = (tl.full((16, 16), True, tl.int1)).to(tl.int1)
    e128 = (tl.load(mem9 + bid * 288 + 16 + e126 * 1, (e127 & (e126 >= 0) & (e126 < 256)), other=0)).to(tl.float32)
    e129 = ((e125 - e128)).to(tl.float32)
    e130 = (e129.to(tl.float16)).to(tl.float16)
    e131 = ((e130 + e49)).to(tl.float16)
    e132 = (tl.where(tl.sum((e131 != e131).to(tl.int32), 0) > 0, float('nan'), tl.max(e131.to(tl.float32), 0))).to(tl.float32)
    tl.store(out_e131 + bid * 288 + 16 + (tl.arange(0, 16)[:, None] * 16 + tl.arange(0, 16)[None, :]), e131)
    tl.store(out_e16 + bid * 544 + 16 + (tl.arange(0, 16)[:, None] * 32 + tl.arange(0, 32)[None, :]), e16)
    tl.store(out_e22 + bid * 544 + 16 + (tl.arange(0, 16)[:, None] * 32 + tl.arange(0, 32)[None, :]), e22)
    tl.store(out_e115 + bid * 33 + 16 + tl.full((), 0, tl.int32), e115)
    tl.store(out_e116 + bid * 33 + 16 + tl.full((), 0, tl.int32), e116)
    tl.store(out_e132 + bid * 48 + 16 + tl.arange(0, 16), e132)

@triton.jit
def mixed_helper_extended_3_1(e55, e56, mem0, mem1, mem2, mem3, mem6, mem7, mem8, mem9, steps, limit, bid):
    e57 = (tl.sum(e55.to(tl.float32), 1)).to(tl.float32)
    e58 = ((e57 + e56)).to(tl.float32)
    e59 = (tl.sum(e58.to(tl.float32), 0)).to(tl.float32)
    e60 = (tl.reshape(e58, (16, 1))).to(tl.float32)
    e61 = (e60.to(tl.float16)).to(tl.float16)
    e62 = ((e55 + e61)).to(tl.float16)
    return e62, e59
@triton.jit
def extended_3_1(mem0, mem1, mem2, mem3, mem6, mem7, mem8, mem9, out_e131, out_e16, out_e22, out_e115, out_e116, out_e132, out_e3, out_e11, out_e28, out_e32, out_e41, out_e46, out_e21, out_e4, steps, limit):
    bid = tl.program_id(0)
    e1 = ((((tl.arange(0, 16)[:, None] * 32 + tl.arange(0, 32)[None, :]) + 10) % 512)).to(tl.int32)
    e2 = (tl.full((16, 32), True, tl.int1)).to(tl.int1)
    e3 = (tl.load(mem0 + bid * 544 + 16 + e1 * 1, (e2 & (e1 >= 0) & (e1 < 512)), other=0)).to(tl.float16)
    e4 = ((((tl.arange(0, 16)[:, None] * 32 + tl.arange(0, 32)[None, :]) + 17) % 512)).to(tl.int32)
    e5 = (tl.full((16, 32), True, tl.int1)).to(tl.int1)
    e6 = (tl.load(mem1 + bid * 544 + 16 + e4 * 1, (e5 & (e4 >= 0) & (e4 < 512)), other=0)).to(tl.float16)
    e7 = ((e3 * e6)).to(tl.float16)
    e8 = (e7.to(tl.float32)).to(tl.float32)
    e9 = (e8.to(tl.float16)).to(tl.float16)
    e10 = (tl.full((16, 32), 0.5, tl.float16)).to(tl.float16)
    e11 = ((e9 * e10)).to(tl.float16)
    e12 = ((((tl.arange(0, 16)[:, None] * 32 + tl.arange(0, 32)[None, :]) + 71) % 512)).to(tl.int32)
    e13 = (tl.full((16, 32), True, tl.int1)).to(tl.int1)
    e14 = (tl.load(mem2 + bid * 544 + 16 + e12 * 1, (e13 & (e12 >= 0) & (e12 < 512)), other=0)).to(tl.int32)
    e15 = (tl.full((16, 32), 3, tl.int32)).to(tl.int32)
    e16 = ((e14 & e15)).to(tl.int32)
    e17 = (tl.full((16, 32), 3, tl.int32)).to(tl.int32)
    e18 = ((e16 < e17)).to(tl.int1)
    e19 = ((((511 - (tl.arange(0, 16)[:, None] * 32 + tl.arange(0, 32)[None, :])) + 252) % 512)).to(tl.int32)
    e20 = (limit).to(tl.int32)
    e21 = ((e19 < e20)).to(tl.int1)
    e22 = ((e18 & e21)).to(tl.int1)
    e23 = (tl.full((16, 32), -0.125, tl.float16)).to(tl.float16)
    e24 = (tl.where(e22, e11, e23)).to(tl.float16)
    e25 = ((((511 - (tl.arange(0, 16)[:, None] * 32 + tl.arange(0, 32)[None, :])) + 305) % 512)).to(tl.int32)
    e26 = (limit).to(tl.int32)
    e27 = ((e25 < e26)).to(tl.int1)
    tl.store(mem3 + bid * 1060 + 17 + e25 * 2, e24, (e27 & (e25 >= 0) & (e25 < 512)))
    tl.debug_barrier()
    e28 = (tl.load(mem3 + bid * 1060 + 17 + e25 * 2, (e27 & (e25 >= 0) & (e25 < 512)), other=0)).to(tl.float16)
    e29 = (tl.full((16, 32), 0.125, tl.float16)).to(tl.float16)
    e30 = ((e28 + e29)).to(tl.float16)
    tl.debug_barrier()
    tl.store(mem3 + bid * 1060 + 18 + e25 * 1, e30, (e27 & (e25 >= 0) & (e25 < 512)))
    tl.debug_barrier()
    e31 = (tl.full((16, 32), True, tl.int1)).to(tl.int1)
    e32 = (tl.load(mem3 + bid * 1060 + 17 + e25 * 2, (e31 & (e25 >= 0) & (e25 < 512)), other=0)).to(tl.float16)
    e33 = (tl.full((16, 32), 3, tl.int32)).to(tl.int32)
    e34 = ((e25 ^ e33)).to(tl.int32)
    e35 = (tl.load(mem3 + bid * 1060 + 18 + e34 * 1, (e27 & (e34 >= 0) & (e34 < 512)), other=0)).to(tl.float16)
    e36 = ((e32 + e35)).to(tl.float16)
    e37 = (e36.to(tl.float16)).to(tl.float16)
    e38 = (tl.trans(e37)).to(tl.float16)
    e39_slice00_lo, e39_slice00_hi = tl.split(tl.permute(tl.reshape(e38, (2, 16, 16)), (1, 2, 0)))
    e39_slice00_next_lo, e39_slice00_next_hi = tl.split(tl.permute(tl.reshape(e39_slice00_lo, (2, 8, 16)), (1, 2, 0)))
    e39_slice01_lo, e39_slice01_hi = tl.split(tl.permute(tl.reshape(e38, (2, 16, 16)), (1, 2, 0)))
    e39_slice01_next_lo, e39_slice01_next_hi = tl.split(tl.permute(tl.reshape(e39_slice01_hi, (2, 8, 16)), (1, 2, 0)))
    e39_slice0 = tl.reshape(tl.permute(tl.join(e39_slice00_next_hi, e39_slice01_next_lo), (2, 0, 1)), (16, 16))
    e39 = (e39_slice0).to(tl.float16)
    e40 = (tl.full((16, 16), 0.0, tl.float32)).to(tl.float32)
    e41 = (tl.dot(e39, e39, e40, input_precision='ieee')).to(tl.float32)
    e42 = (tl.where(tl.sum((e41 != e41).to(tl.int32), 1) > 0, float('nan'), tl.max(e41.to(tl.float32), 1))).to(tl.float32)
    e43 = (tl.reshape(e42, (16, 1))).to(tl.float32)
    e44 = ((e41 + e43)).to(tl.float32)
    e45 = (tl.full((16, 16), 0.125, tl.float32)).to(tl.float32)
    e46 = ((e44 * e45)).to(tl.float32)
    e47 = ((((tl.arange(0, 16)[:, None] * 16 + tl.arange(0, 16)[None, :]) + 44) % 256)).to(tl.int32)
    e48 = (tl.full((16, 16), True, tl.int1)).to(tl.int1)
    e49 = (tl.load(mem6 + bid * 288 + 16 + e47 * 1, (e48 & (e47 >= 0) & (e47 < 256)), other=0)).to(tl.float16)
    e50 = (e46.to(tl.float16)).to(tl.float16)
    e51 = (tl.dot(e50, e49, e40, input_precision='ieee')).to(tl.float32)
    e52 = (tl.reshape(e51, (16, 16))).to(tl.float32)
    e53 = (tl.reshape(e52, (16, 16))).to(tl.float32)
    e54 = (e53.to(tl.float16)).to(tl.float16)
    e63 = (e42.to(tl.float32)).to(tl.float32)
    e64, e65 = mixed_helper_extended_3_1(e54, e63, mem0, mem1, mem2, mem3, mem6, mem7, mem8, mem9, steps, limit, bid)
    e66 = (steps).to(tl.int32)
    e67 = (tl.full((), 0, tl.int32)).to(tl.int32)
    e68 = (tl.full((), 0, tl.int32)).to(tl.int32)
    e69 = (tl.full((), True, tl.int1)).to(tl.int1)
    e70 = (tl.load(mem7 + bid * 33 + 16 + e68 * 1, (e69 & (e68 >= 0) & (e68 < 1)), other=0)).to(tl.int32)
    e71 = (tl.full((), 0, tl.int32)).to(tl.int32)
    e72 = ((e70 < e71)).to(tl.int1)
    e92 = e64
    e93 = e65
    e94 = e67
    for e73 in range(tl.minimum(tl.maximum(e66, 0), 4)):
        e74 = e92
        e75 = e93
        e76 = e94
        if e72:
            e77 = e74
            e78 = (tl.full((16, 16), 0.125, tl.float16)).to(tl.float16)
            e79 = ((e77 + e78)).to(tl.float16)
            e83 = e79
        else:
            e80 = e74
            e81 = (tl.full((16, 16), 0.125, tl.float16)).to(tl.float16)
            e82 = ((e80 - e81)).to(tl.float16)
            e83 = e82
        e84, e85 = mixed_helper_extended_3_1(e83, e63, mem0, mem1, mem2, mem3, mem6, mem7, mem8, mem9, steps, limit, bid)
        e86 = (tl.full((16, 16), 0.125, tl.float16)).to(tl.float16)
        e87 = ((e84 * e86)).to(tl.float16)
        e88 = ((e75 + e85)).to(tl.float32)
        e89 = (tl.full((), 1, tl.int32)).to(tl.int32)
        e90 = ((e73 + e89)).to(tl.int32)
        e91 = ((e76 + e90)).to(tl.int32)
        e92 = e87
        e93 = e88
        e94 = e91
    e114 = e92
    e115 = e93
    e116 = e94
    e95 = tl.full((), 0, tl.int32)
    while e95 < tl.minimum(tl.maximum(e66, 0), 4):
        e96 = e114
        e97 = e115
        e98 = e116
        if e72:
            e99 = e96
            e100 = (tl.full((16, 16), 0.125, tl.float16)).to(tl.float16)
            e101 = ((e99 + e100)).to(tl.float16)
            e105 = e101
        else:
            e102 = e96
            e103 = (tl.full((16, 16), 0.125, tl.float16)).to(tl.float16)
            e104 = ((e102 - e103)).to(tl.float16)
            e105 = e104
        e106, e107 = mixed_helper_extended_3_1(e105, e63, mem0, mem1, mem2, mem3, mem6, mem7, mem8, mem9, steps, limit, bid)
        e108 = (tl.full((16, 16), 0.125, tl.float16)).to(tl.float16)
        e109 = ((e106 * e108)).to(tl.float16)
        e110 = ((e97 + e107)).to(tl.float32)
        e111 = (tl.full((), 1, tl.int32)).to(tl.int32)
        e112 = ((e95 + e111)).to(tl.int32)
        e113 = ((e98 + e112)).to(tl.int32)
        e114 = e109
        e115 = e110
        e116 = e113
        e95 += 1
    e117 = (tl.full((), -0.125, tl.float32)).to(tl.float32)
    e118 = (tl.broadcast_to(e117, (16, 16))).to(tl.float32)
    e119 = (tl.full((), -0.25, tl.float32)).to(tl.float32)
    e120 = (tl.broadcast_to(e119, (16, 16))).to(tl.float32)
    e121 = ((e118 - e120)).to(tl.float32)
    e122 = ((((255 - (tl.arange(0, 16)[:, None] * 16 + tl.arange(0, 16)[None, :])) + 28) % 256)).to(tl.int32)
    e123 = (tl.full((16, 16), True, tl.int1)).to(tl.int1)
    tl.atomic_max(mem8 + bid * 288 + 16 + e122 * 1, e121, (e123 & (e122 >= 0) & (e122 < 256)))
    e124 = ((e114 - e92)).to(tl.float16)
    e125 = (e124.to(tl.float32)).to(tl.float32)
    e126 = ((((255 - (tl.arange(0, 16)[:, None] * 16 + tl.arange(0, 16)[None, :])) + 194) % 256)).to(tl.int32)
    e127 = (tl.full((16, 16), True, tl.int1)).to(tl.int1)
    e128 = (tl.load(mem9 + bid * 288 + 16 + e126 * 1, (e127 & (e126 >= 0) & (e126 < 256)), other=0)).to(tl.float32)
    e129 = ((e125 - e128)).to(tl.float32)
    e130 = (e129.to(tl.float16)).to(tl.float16)
    e131 = ((e130 + e49)).to(tl.float16)
    e132 = (tl.where(tl.sum((e131 != e131).to(tl.int32), 0) > 0, float('nan'), tl.max(e131.to(tl.float32), 0))).to(tl.float32)
    tl.store(out_e131 + bid * 288 + 16 + (tl.arange(0, 16)[:, None] * 16 + tl.arange(0, 16)[None, :]), e131)
    tl.store(out_e16 + bid * 544 + 16 + (tl.arange(0, 16)[:, None] * 32 + tl.arange(0, 32)[None, :]), e16)
    tl.store(out_e22 + bid * 544 + 16 + (tl.arange(0, 16)[:, None] * 32 + tl.arange(0, 32)[None, :]), e22)
    tl.store(out_e115 + bid * 33 + 16 + tl.full((), 0, tl.int32), e115)
    tl.store(out_e116 + bid * 33 + 16 + tl.full((), 0, tl.int32), e116)
    tl.store(out_e132 + bid * 48 + 16 + tl.arange(0, 16), e132)
    tl.store(out_e3 + bid * 544 + 16 + (tl.arange(0, 16)[:, None] * 32 + tl.arange(0, 32)[None, :]), e3)
    tl.store(out_e11 + bid * 544 + 16 + (tl.arange(0, 16)[:, None] * 32 + tl.arange(0, 32)[None, :]), e11)
    tl.store(out_e28 + bid * 544 + 16 + (tl.arange(0, 16)[:, None] * 32 + tl.arange(0, 32)[None, :]), e28)
    tl.store(out_e32 + bid * 544 + 16 + (tl.arange(0, 16)[:, None] * 32 + tl.arange(0, 32)[None, :]), e32)
    tl.store(out_e41 + bid * 288 + 16 + (tl.arange(0, 16)[:, None] * 16 + tl.arange(0, 16)[None, :]), e41)
    tl.store(out_e46 + bid * 288 + 16 + (tl.arange(0, 16)[:, None] * 16 + tl.arange(0, 16)[None, :]), e46)
    tl.store(out_e21 + bid * 544 + 16 + (tl.arange(0, 16)[:, None] * 32 + tl.arange(0, 32)[None, :]), e21)
    tl.store(out_e4 + bid * 544 + 16 + (tl.arange(0, 16)[:, None] * 32 + tl.arange(0, 32)[None, :]), e4)

@triton.jit
def mixed_helper_extended_0_prec(e55, e56, mem0, mem1, mem2, mem3, mem6, mem7, mem8, mem9, steps, limit, bid):
    e57 = (tl.sum(e55.to(tl.float32), 1)).to(tl.float32)
    e58 = ((e57 + e56)).to(tl.float32)
    e59 = (tl.sum(e58.to(tl.float32), 0)).to(tl.float32)
    e60 = (tl.reshape(e58, (16, 1))).to(tl.float32)
    e61 = (e60.to(tl.float16)).to(tl.float16)
    e62 = ((e55 + e61)).to(tl.float16)
    return e62, e59
@triton.jit
def extended_0_prec(mem0, mem1, mem2, mem3, mem6, mem7, mem8, mem9, out_e131, out_e16, out_e22, out_e115, out_e116, out_e132, out_e3, out_e11, out_e28, out_e32, out_e41, out_e46, out_e21, out_e4, steps, limit):
    bid = tl.program_id(0)
    e1 = ((((tl.arange(0, 16)[:, None] * 32 + tl.arange(0, 32)[None, :]) + 10) % 512)).to(tl.int32)
    e2 = (tl.full((16, 32), True, tl.int1)).to(tl.int1)
    e3 = (tl.load(mem0 + bid * 544 + 16 + e1 * 1, (e2 & (e1 >= 0) & (e1 < 512)), other=0)).to(tl.float16)
    e4 = ((((tl.arange(0, 16)[:, None] * 32 + tl.arange(0, 32)[None, :]) + 17) % 512)).to(tl.int32)
    e5 = (tl.full((16, 32), True, tl.int1)).to(tl.int1)
    e6 = (tl.load(mem1 + bid * 544 + 16 + e4 * 1, (e5 & (e4 >= 0) & (e4 < 512)), other=0)).to(tl.float16)
    e7 = ((e3 * e6)).to(tl.float16)
    e8 = (e7.to(tl.float32)).to(tl.float32)
    e9 = (e8.to(tl.float16)).to(tl.float16)
    e10 = (tl.full((16, 32), 0.5, tl.float16)).to(tl.float16)
    e11 = ((e9 * e10)).to(tl.float16)
    e12 = ((((tl.arange(0, 16)[:, None] * 32 + tl.arange(0, 32)[None, :]) + 71) % 512)).to(tl.int32)
    e13 = (tl.full((16, 32), True, tl.int1)).to(tl.int1)
    e14 = (tl.load(mem2 + bid * 544 + 16 + e12 * 1, (e13 & (e12 >= 0) & (e12 < 512)), other=0)).to(tl.int32)
    e15 = (tl.full((16, 32), 3, tl.int32)).to(tl.int32)
    e16 = ((e14 & e15)).to(tl.int32)
    e17 = (tl.full((16, 32), 3, tl.int32)).to(tl.int32)
    e18 = ((e16 < e17)).to(tl.int1)
    e19 = ((((511 - (tl.arange(0, 16)[:, None] * 32 + tl.arange(0, 32)[None, :])) + 252) % 512)).to(tl.int32)
    e20 = (limit).to(tl.int32)
    e21 = ((e19 < e20)).to(tl.int1)
    e22 = ((e18 & e21)).to(tl.int1)
    e23 = (tl.full((16, 32), -0.125, tl.float16)).to(tl.float16)
    e24 = (tl.where(e22, e11, e23)).to(tl.float16)
    e25 = ((((511 - (tl.arange(0, 16)[:, None] * 32 + tl.arange(0, 32)[None, :])) + 305) % 512)).to(tl.int32)
    e26 = (limit).to(tl.int32)
    e27 = ((e25 < e26)).to(tl.int1)
    tl.store(mem3 + bid * 1060 + 17 + e25 * 2, e24, (e27 & (e25 >= 0) & (e25 < 512)))
    tl.debug_barrier()
    e28 = (tl.load(mem3 + bid * 1060 + 17 + e25 * 2, (e27 & (e25 >= 0) & (e25 < 512)), other=0)).to(tl.float16)
    e29 = (tl.full((16, 32), 0.125, tl.float16)).to(tl.float16)
    e30 = ((e28 + e29)).to(tl.float16)
    tl.debug_barrier()
    tl.store(mem3 + bid * 1060 + 18 + e25 * 1, e30, (e27 & (e25 >= 0) & (e25 < 512)))
    tl.debug_barrier()
    e31 = (tl.full((16, 32), True, tl.int1)).to(tl.int1)
    e32 = (tl.load(mem3 + bid * 1060 + 17 + e25 * 2, (e31 & (e25 >= 0) & (e25 < 512)), other=0)).to(tl.float16)
    e33 = (tl.full((16, 32), 3, tl.int32)).to(tl.int32)
    e34 = ((e25 ^ e33)).to(tl.int32)
    e35 = (tl.load(mem3 + bid * 1060 + 18 + e34 * 1, (e27 & (e34 >= 0) & (e34 < 512)), other=0)).to(tl.float16)
    e36 = ((e32 + e35)).to(tl.float16)
    e37 = (e36.to(tl.float16)).to(tl.float16)
    e38 = (tl.trans(e37)).to(tl.float16)
    e39_slice00_lo, e39_slice00_hi = tl.split(tl.permute(tl.reshape(e38, (2, 16, 16)), (1, 2, 0)))
    e39_slice00_next_lo, e39_slice00_next_hi = tl.split(tl.permute(tl.reshape(e39_slice00_lo, (2, 8, 16)), (1, 2, 0)))
    e39_slice01_lo, e39_slice01_hi = tl.split(tl.permute(tl.reshape(e38, (2, 16, 16)), (1, 2, 0)))
    e39_slice01_next_lo, e39_slice01_next_hi = tl.split(tl.permute(tl.reshape(e39_slice01_hi, (2, 8, 16)), (1, 2, 0)))
    e39_slice0 = tl.reshape(tl.permute(tl.join(e39_slice00_next_hi, e39_slice01_next_lo), (2, 0, 1)), (16, 16))
    e39 = (e39_slice0).to(tl.float16)
    e40 = (tl.full((16, 16), 0.0, tl.float16)).to(tl.float16)
    e41 = (tl.dot(e39, e39, e40, input_precision='ieee', out_dtype=tl.float16)).to(tl.float16)
    e42 = (tl.where(tl.sum((e41 != e41).to(tl.int32), 1) > 0, float('nan'), tl.max(e41.to(tl.float32), 1))).to(tl.float32)
    e43 = (tl.reshape(e42, (16, 1))).to(tl.float32)
    e41_prec = (e41.to(tl.float32)).to(tl.float32)
    e44 = ((e41_prec + e43)).to(tl.float32)
    e45 = (tl.full((16, 16), 0.125, tl.float32)).to(tl.float32)
    e46 = ((e44 * e45)).to(tl.float32)
    e47 = ((((tl.arange(0, 16)[:, None] * 16 + tl.arange(0, 16)[None, :]) + 44) % 256)).to(tl.int32)
    e48 = (tl.full((16, 16), True, tl.int1)).to(tl.int1)
    e49 = (tl.load(mem6 + bid * 288 + 16 + e47 * 1, (e48 & (e47 >= 0) & (e47 < 256)), other=0)).to(tl.float16)
    e50 = (e46.to(tl.float16)).to(tl.float16)
    e51 = (tl.dot(e50, e49, e40, input_precision='ieee', out_dtype=tl.float16)).to(tl.float16)
    e52 = (tl.reshape(e51, (16, 16))).to(tl.float16)
    e53 = (tl.reshape(e52, (16, 16))).to(tl.float16)
    e54 = (e53.to(tl.float16)).to(tl.float16)
    e63 = (e42.to(tl.float32)).to(tl.float32)
    e64, e65 = mixed_helper_extended_0_prec(e54, e63, mem0, mem1, mem2, mem3, mem6, mem7, mem8, mem9, steps, limit, bid)
    e66 = (steps).to(tl.int32)
    e67 = (tl.full((), 0, tl.int32)).to(tl.int32)
    e68 = (tl.full((), 0, tl.int32)).to(tl.int32)
    e69 = (tl.full((), True, tl.int1)).to(tl.int1)
    e70 = (tl.load(mem7 + bid * 33 + 16 + e68 * 1, (e69 & (e68 >= 0) & (e68 < 1)), other=0)).to(tl.int32)
    e71 = (tl.full((), 0, tl.int32)).to(tl.int32)
    e72 = ((e70 < e71)).to(tl.int1)
    e92 = e64
    e93 = e65
    e94 = e67
    for e73 in range(tl.minimum(tl.maximum(e66, 0), 4)):
        e74 = e92
        e75 = e93
        e76 = e94
        if e72:
            e77 = e74
            e78 = (tl.full((16, 16), 0.125, tl.float16)).to(tl.float16)
            e79 = ((e77 + e78)).to(tl.float16)
            e83 = e79
        else:
            e80 = e74
            e81 = (tl.full((16, 16), 0.125, tl.float16)).to(tl.float16)
            e82 = ((e80 - e81)).to(tl.float16)
            e83 = e82
        e84, e85 = mixed_helper_extended_0_prec(e83, e63, mem0, mem1, mem2, mem3, mem6, mem7, mem8, mem9, steps, limit, bid)
        e86 = (tl.full((16, 16), 0.125, tl.float16)).to(tl.float16)
        e87 = ((e84 * e86)).to(tl.float16)
        e88 = ((e75 + e85)).to(tl.float32)
        e89 = (tl.full((), 1, tl.int32)).to(tl.int32)
        e90 = ((e73 + e89)).to(tl.int32)
        e91 = ((e76 + e90)).to(tl.int32)
        e92 = e87
        e93 = e88
        e94 = e91
    e114 = e92
    e115 = e93
    e116 = e94
    e95 = tl.full((), 0, tl.int32)
    while e95 < tl.minimum(tl.maximum(e66, 0), 4):
        e96 = e114
        e97 = e115
        e98 = e116
        if e72:
            e99 = e96
            e100 = (tl.full((16, 16), 0.125, tl.float16)).to(tl.float16)
            e101 = ((e99 + e100)).to(tl.float16)
            e105 = e101
        else:
            e102 = e96
            e103 = (tl.full((16, 16), 0.125, tl.float16)).to(tl.float16)
            e104 = ((e102 - e103)).to(tl.float16)
            e105 = e104
        e106, e107 = mixed_helper_extended_0_prec(e105, e63, mem0, mem1, mem2, mem3, mem6, mem7, mem8, mem9, steps, limit, bid)
        e108 = (tl.full((16, 16), 0.125, tl.float16)).to(tl.float16)
        e109 = ((e106 * e108)).to(tl.float16)
        e110 = ((e97 + e107)).to(tl.float32)
        e111 = (tl.full((), 1, tl.int32)).to(tl.int32)
        e112 = ((e95 + e111)).to(tl.int32)
        e113 = ((e98 + e112)).to(tl.int32)
        e114 = e109
        e115 = e110
        e116 = e113
        e95 += 1
    e117 = (tl.full((), -0.125, tl.float32)).to(tl.float32)
    e118 = (tl.broadcast_to(e117, (16, 16))).to(tl.float32)
    e119 = (tl.full((), -0.25, tl.float32)).to(tl.float32)
    e120 = (tl.broadcast_to(e119, (16, 16))).to(tl.float32)
    e121 = ((e118 - e120)).to(tl.float32)
    e122 = ((((255 - (tl.arange(0, 16)[:, None] * 16 + tl.arange(0, 16)[None, :])) + 28) % 256)).to(tl.int32)
    e123 = (tl.full((16, 16), True, tl.int1)).to(tl.int1)
    tl.atomic_max(mem8 + bid * 288 + 16 + e122 * 1, e121, (e123 & (e122 >= 0) & (e122 < 256)))
    e124 = ((e114 - e92)).to(tl.float16)
    e125 = (e124.to(tl.float32)).to(tl.float32)
    e126 = ((((255 - (tl.arange(0, 16)[:, None] * 16 + tl.arange(0, 16)[None, :])) + 194) % 256)).to(tl.int32)
    e127 = (tl.full((16, 16), True, tl.int1)).to(tl.int1)
    e128 = (tl.load(mem9 + bid * 288 + 16 + e126 * 1, (e127 & (e126 >= 0) & (e126 < 256)), other=0)).to(tl.float32)
    e129 = ((e125 - e128)).to(tl.float32)
    e130 = (e129.to(tl.float16)).to(tl.float16)
    e131 = ((e130 + e49)).to(tl.float16)
    e132 = (tl.where(tl.sum((e131 != e131).to(tl.int32), 0) > 0, float('nan'), tl.max(e131.to(tl.float32), 0))).to(tl.float32)
    tl.store(out_e131 + bid * 288 + 16 + (tl.arange(0, 16)[:, None] * 16 + tl.arange(0, 16)[None, :]), e131)
    tl.store(out_e16 + bid * 544 + 16 + (tl.arange(0, 16)[:, None] * 32 + tl.arange(0, 32)[None, :]), e16)
    tl.store(out_e22 + bid * 544 + 16 + (tl.arange(0, 16)[:, None] * 32 + tl.arange(0, 32)[None, :]), e22)
    tl.store(out_e115 + bid * 33 + 16 + tl.full((), 0, tl.int32), e115)
    tl.store(out_e116 + bid * 33 + 16 + tl.full((), 0, tl.int32), e116)
    tl.store(out_e132 + bid * 48 + 16 + tl.arange(0, 16), e132)
    tl.store(out_e3 + bid * 544 + 16 + (tl.arange(0, 16)[:, None] * 32 + tl.arange(0, 32)[None, :]), e3)
    tl.store(out_e11 + bid * 544 + 16 + (tl.arange(0, 16)[:, None] * 32 + tl.arange(0, 32)[None, :]), e11)
    tl.store(out_e28 + bid * 544 + 16 + (tl.arange(0, 16)[:, None] * 32 + tl.arange(0, 32)[None, :]), e28)
    tl.store(out_e32 + bid * 544 + 16 + (tl.arange(0, 16)[:, None] * 32 + tl.arange(0, 32)[None, :]), e32)
    tl.store(out_e41 + bid * 288 + 16 + (tl.arange(0, 16)[:, None] * 16 + tl.arange(0, 16)[None, :]), e41)
    tl.store(out_e46 + bid * 288 + 16 + (tl.arange(0, 16)[:, None] * 16 + tl.arange(0, 16)[None, :]), e46)
    tl.store(out_e21 + bid * 544 + 16 + (tl.arange(0, 16)[:, None] * 32 + tl.arange(0, 32)[None, :]), e21)
    tl.store(out_e4 + bid * 544 + 16 + (tl.arange(0, 16)[:, None] * 32 + tl.arange(0, 32)[None, :]), e4)

@triton.jit
def mixed_helper_extended_1_prec(e55, e56, mem0, mem1, mem2, mem3, mem6, mem7, mem8, mem9, steps, limit, bid):
    e57 = (tl.sum(e55.to(tl.float32), 1)).to(tl.float32)
    e58 = ((e57 + e56)).to(tl.float32)
    e59 = (tl.sum(e58.to(tl.float32), 0)).to(tl.float32)
    e60 = (tl.reshape(e58, (16, 1))).to(tl.float32)
    e61 = (e60.to(tl.float16)).to(tl.float16)
    e62 = ((e55 + e61)).to(tl.float16)
    return e62, e59
@triton.jit
def extended_1_prec(mem0, mem1, mem2, mem3, mem6, mem7, mem8, mem9, out_e131, out_e16, out_e22, out_e115, out_e116, out_e132, out_e3, out_e11, out_e28, out_e32, out_e41, out_e46, out_e21, out_e4, steps, limit):
    bid = tl.program_id(0)
    e1 = ((((tl.arange(0, 16)[:, None] * 32 + tl.arange(0, 32)[None, :]) + 10) % 512)).to(tl.int32)
    e2 = (tl.full((16, 32), True, tl.int1)).to(tl.int1)
    e3 = (tl.load(mem0 + bid * 544 + 16 + e1 * 1, (e2 & (e1 >= 0) & (e1 < 512)), other=0)).to(tl.float16)
    e4 = ((((tl.arange(0, 16)[:, None] * 32 + tl.arange(0, 32)[None, :]) + 17) % 512)).to(tl.int32)
    e5 = (tl.full((16, 32), True, tl.int1)).to(tl.int1)
    e6 = (tl.load(mem1 + bid * 544 + 16 + e4 * 1, (e5 & (e4 >= 0) & (e4 < 512)), other=0)).to(tl.float16)
    e7 = ((e3 * e6)).to(tl.float16)
    e8 = (e7.to(tl.float32)).to(tl.float32)
    e9 = (e8.to(tl.float16)).to(tl.float16)
    e10 = (tl.full((16, 32), 0.5, tl.float16)).to(tl.float16)
    e11 = ((e9 * e10)).to(tl.float16)
    e12 = ((((tl.arange(0, 16)[:, None] * 32 + tl.arange(0, 32)[None, :]) + 71) % 512)).to(tl.int32)
    e13 = (tl.full((16, 32), True, tl.int1)).to(tl.int1)
    e14 = (tl.load(mem2 + bid * 544 + 16 + e12 * 1, (e13 & (e12 >= 0) & (e12 < 512)), other=0)).to(tl.int32)
    e15 = (tl.full((16, 32), 3, tl.int32)).to(tl.int32)
    e16 = ((e14 & e15)).to(tl.int32)
    e17 = (tl.full((16, 32), 3, tl.int32)).to(tl.int32)
    e18 = ((e16 < e17)).to(tl.int1)
    e19 = ((((511 - (tl.arange(0, 16)[:, None] * 32 + tl.arange(0, 32)[None, :])) + 252) % 512)).to(tl.int32)
    e20 = (limit).to(tl.int32)
    e21 = ((e19 < e20)).to(tl.int1)
    e22 = ((e18 & e21)).to(tl.int1)
    e23 = (tl.full((16, 32), -0.125, tl.float16)).to(tl.float16)
    e24 = (tl.where(e22, e11, e23)).to(tl.float16)
    e25 = ((((511 - (tl.arange(0, 16)[:, None] * 32 + tl.arange(0, 32)[None, :])) + 305) % 512)).to(tl.int32)
    e26 = (limit).to(tl.int32)
    e27 = ((e25 < e26)).to(tl.int1)
    tl.store(mem3 + bid * 1060 + 17 + e25 * 2, e24, (e27 & (e25 >= 0) & (e25 < 512)))
    tl.debug_barrier()
    e28 = (tl.load(mem3 + bid * 1060 + 17 + e25 * 2, (e27 & (e25 >= 0) & (e25 < 512)), other=0)).to(tl.float16)
    e29 = (tl.full((16, 32), 0.125, tl.float16)).to(tl.float16)
    e30 = ((e28 + e29)).to(tl.float16)
    tl.debug_barrier()
    tl.store(mem3 + bid * 1060 + 18 + e25 * 1, e30, (e27 & (e25 >= 0) & (e25 < 512)))
    tl.debug_barrier()
    e31 = (tl.full((16, 32), True, tl.int1)).to(tl.int1)
    e32 = (tl.load(mem3 + bid * 1060 + 17 + e25 * 2, (e31 & (e25 >= 0) & (e25 < 512)), other=0)).to(tl.float16)
    e33 = (tl.full((16, 32), 3, tl.int32)).to(tl.int32)
    e34 = ((e25 ^ e33)).to(tl.int32)
    e35 = (tl.load(mem3 + bid * 1060 + 18 + e34 * 1, (e27 & (e34 >= 0) & (e34 < 512)), other=0)).to(tl.float16)
    e36 = ((e32 + e35)).to(tl.float16)
    e37 = (e36.to(tl.float16)).to(tl.float16)
    e38 = (tl.trans(e37)).to(tl.float16)
    e39_slice00_lo, e39_slice00_hi = tl.split(tl.permute(tl.reshape(e38, (2, 16, 16)), (1, 2, 0)))
    e39_slice00_next_lo, e39_slice00_next_hi = tl.split(tl.permute(tl.reshape(e39_slice00_lo, (2, 8, 16)), (1, 2, 0)))
    e39_slice01_lo, e39_slice01_hi = tl.split(tl.permute(tl.reshape(e38, (2, 16, 16)), (1, 2, 0)))
    e39_slice01_next_lo, e39_slice01_next_hi = tl.split(tl.permute(tl.reshape(e39_slice01_hi, (2, 8, 16)), (1, 2, 0)))
    e39_slice0 = tl.reshape(tl.permute(tl.join(e39_slice00_next_hi, e39_slice01_next_lo), (2, 0, 1)), (16, 16))
    e39 = (e39_slice0).to(tl.float16)
    e40 = (tl.full((16, 16), 0.0, tl.float16)).to(tl.float16)
    e41 = (tl.dot(e39, e39, e40, input_precision='ieee', out_dtype=tl.float16)).to(tl.float16)
    e42 = (tl.where(tl.sum((e41 != e41).to(tl.int32), 1) > 0, float('nan'), tl.max(e41.to(tl.float32), 1))).to(tl.float32)
    e43 = (tl.reshape(e42, (16, 1))).to(tl.float32)
    e41_prec = (e41.to(tl.float32)).to(tl.float32)
    e44 = ((e41_prec + e43)).to(tl.float32)
    e45 = (tl.full((16, 16), 0.125, tl.float32)).to(tl.float32)
    e46 = ((e44 * e45)).to(tl.float32)
    e47 = ((((tl.arange(0, 16)[:, None] * 16 + tl.arange(0, 16)[None, :]) + 44) % 256)).to(tl.int32)
    e48 = (tl.full((16, 16), True, tl.int1)).to(tl.int1)
    e49 = (tl.load(mem6 + bid * 288 + 16 + e47 * 1, (e48 & (e47 >= 0) & (e47 < 256)), other=0)).to(tl.float16)
    e50 = (e46.to(tl.float16)).to(tl.float16)
    e51 = (tl.dot(e50, e49, e40, input_precision='ieee', out_dtype=tl.float16)).to(tl.float16)
    e52 = (tl.reshape(e51, (16, 16))).to(tl.float16)
    e53 = (tl.reshape(e52, (16, 16))).to(tl.float16)
    e54 = (e53.to(tl.float16)).to(tl.float16)
    e63 = (e42.to(tl.float32)).to(tl.float32)
    e64, e65 = mixed_helper_extended_1_prec(e54, e63, mem0, mem1, mem2, mem3, mem6, mem7, mem8, mem9, steps, limit, bid)
    e66 = (steps).to(tl.int32)
    e67 = (tl.full((), 0, tl.int32)).to(tl.int32)
    e68 = (tl.full((), 0, tl.int32)).to(tl.int32)
    e69 = (tl.full((), True, tl.int1)).to(tl.int1)
    e70 = (tl.load(mem7 + bid * 33 + 16 + e68 * 1, (e69 & (e68 >= 0) & (e68 < 1)), other=0)).to(tl.int32)
    e71 = (tl.full((), 0, tl.int32)).to(tl.int32)
    e72 = ((e70 < e71)).to(tl.int1)
    e92 = e64
    e93 = e65
    e94 = e67
    for e73 in range(tl.minimum(tl.maximum(e66, 0), 4)):
        e74 = e92
        e75 = e93
        e76 = e94
        if e72:
            e77 = e74
            e78 = (tl.full((16, 16), 0.125, tl.float16)).to(tl.float16)
            e79 = ((e77 + e78)).to(tl.float16)
            e83 = e79
        else:
            e80 = e74
            e81 = (tl.full((16, 16), 0.125, tl.float16)).to(tl.float16)
            e82 = ((e80 - e81)).to(tl.float16)
            e83 = e82
        e84, e85 = mixed_helper_extended_1_prec(e83, e63, mem0, mem1, mem2, mem3, mem6, mem7, mem8, mem9, steps, limit, bid)
        e86 = (tl.full((16, 16), 0.125, tl.float16)).to(tl.float16)
        e87 = ((e84 * e86)).to(tl.float16)
        e88 = ((e75 + e85)).to(tl.float32)
        e89 = (tl.full((), 1, tl.int32)).to(tl.int32)
        e90 = ((e73 + e89)).to(tl.int32)
        e91 = ((e76 + e90)).to(tl.int32)
        e92 = e87
        e93 = e88
        e94 = e91
    e114 = e92
    e115 = e93
    e116 = e94
    e95 = tl.full((), 0, tl.int32)
    while e95 < tl.minimum(tl.maximum(e66, 0), 4):
        e96 = e114
        e97 = e115
        e98 = e116
        if e72:
            e99 = e96
            e100 = (tl.full((16, 16), 0.125, tl.float16)).to(tl.float16)
            e101 = ((e99 + e100)).to(tl.float16)
            e105 = e101
        else:
            e102 = e96
            e103 = (tl.full((16, 16), 0.125, tl.float16)).to(tl.float16)
            e104 = ((e102 - e103)).to(tl.float16)
            e105 = e104
        e106, e107 = mixed_helper_extended_1_prec(e105, e63, mem0, mem1, mem2, mem3, mem6, mem7, mem8, mem9, steps, limit, bid)
        e108 = (tl.full((16, 16), 0.125, tl.float16)).to(tl.float16)
        e109 = ((e106 * e108)).to(tl.float16)
        e110 = ((e97 + e107)).to(tl.float32)
        e111 = (tl.full((), 1, tl.int32)).to(tl.int32)
        e112 = ((e95 + e111)).to(tl.int32)
        e113 = ((e98 + e112)).to(tl.int32)
        e114 = e109
        e115 = e110
        e116 = e113
        e95 += 1
    e117 = (tl.full((), -0.125, tl.float32)).to(tl.float32)
    e118 = (tl.broadcast_to(e117, (16, 16))).to(tl.float32)
    e119 = (tl.full((), -0.25, tl.float32)).to(tl.float32)
    e120 = (tl.broadcast_to(e119, (16, 16))).to(tl.float32)
    e121 = ((e118 - e120)).to(tl.float32)
    e122 = ((((255 - (tl.arange(0, 16)[:, None] * 16 + tl.arange(0, 16)[None, :])) + 28) % 256)).to(tl.int32)
    e123 = (tl.full((16, 16), True, tl.int1)).to(tl.int1)
    tl.atomic_max(mem8 + bid * 288 + 16 + e122 * 1, e121, (e123 & (e122 >= 0) & (e122 < 256)))
    e124 = ((e114 - e92)).to(tl.float16)
    e125 = (e124.to(tl.float32)).to(tl.float32)
    e126 = ((((255 - (tl.arange(0, 16)[:, None] * 16 + tl.arange(0, 16)[None, :])) + 194) % 256)).to(tl.int32)
    e127 = (tl.full((16, 16), True, tl.int1)).to(tl.int1)
    e128 = (tl.load(mem9 + bid * 288 + 16 + e126 * 1, (e127 & (e126 >= 0) & (e126 < 256)), other=0)).to(tl.float32)
    e129 = ((e125 - e128)).to(tl.float32)
    e130 = (e129.to(tl.float16)).to(tl.float16)
    e131 = ((e130 + e49)).to(tl.float16)
    e132 = (tl.where(tl.sum((e131 != e131).to(tl.int32), 0) > 0, float('nan'), tl.max(e131.to(tl.float32), 0))).to(tl.float32)
    tl.store(out_e131 + bid * 288 + 16 + (tl.arange(0, 16)[:, None] * 16 + tl.arange(0, 16)[None, :]), e131)
    tl.store(out_e16 + bid * 544 + 16 + (tl.arange(0, 16)[:, None] * 32 + tl.arange(0, 32)[None, :]), e16)
    tl.store(out_e22 + bid * 544 + 16 + (tl.arange(0, 16)[:, None] * 32 + tl.arange(0, 32)[None, :]), e22)
    tl.store(out_e115 + bid * 33 + 16 + tl.full((), 0, tl.int32), e115)
    tl.store(out_e116 + bid * 33 + 16 + tl.full((), 0, tl.int32), e116)
    tl.store(out_e132 + bid * 48 + 16 + tl.arange(0, 16), e132)
    tl.store(out_e3 + bid * 544 + 16 + (tl.arange(0, 16)[:, None] * 32 + tl.arange(0, 32)[None, :]), e3)
    tl.store(out_e11 + bid * 544 + 16 + (tl.arange(0, 16)[:, None] * 32 + tl.arange(0, 32)[None, :]), e11)
    tl.store(out_e28 + bid * 544 + 16 + (tl.arange(0, 16)[:, None] * 32 + tl.arange(0, 32)[None, :]), e28)
    tl.store(out_e32 + bid * 544 + 16 + (tl.arange(0, 16)[:, None] * 32 + tl.arange(0, 32)[None, :]), e32)
    tl.store(out_e41 + bid * 288 + 16 + (tl.arange(0, 16)[:, None] * 16 + tl.arange(0, 16)[None, :]), e41)
    tl.store(out_e46 + bid * 288 + 16 + (tl.arange(0, 16)[:, None] * 16 + tl.arange(0, 16)[None, :]), e46)
    tl.store(out_e21 + bid * 544 + 16 + (tl.arange(0, 16)[:, None] * 32 + tl.arange(0, 32)[None, :]), e21)
    tl.store(out_e4 + bid * 544 + 16 + (tl.arange(0, 16)[:, None] * 32 + tl.arange(0, 32)[None, :]), e4)

@triton.jit
def mixed_helper_extended_tf32(e55, e56, mem0, mem1, mem2, mem3, mem6, mem7, mem8, mem9, steps, limit, bid):
    e57 = (tl.sum(e55.to(tl.float32), 1)).to(tl.float32)
    e58 = ((e57 + e56)).to(tl.float32)
    e59 = (tl.sum(e58.to(tl.float32), 0)).to(tl.float32)
    e60 = (tl.reshape(e58, (16, 1))).to(tl.float32)
    e61 = (e60.to(tl.float16)).to(tl.float16)
    e62 = ((e55 + e61)).to(tl.float16)
    return e62, e59
@triton.jit
def extended_tf32(mem0, mem1, mem2, mem3, mem6, mem7, mem8, mem9, out_e131, out_e16, out_e22, out_e115, out_e116, out_e132, steps, limit):
    bid = tl.program_id(0)
    e1 = ((((tl.arange(0, 16)[:, None] * 32 + tl.arange(0, 32)[None, :]) + 10) % 512)).to(tl.int32)
    e2 = (tl.full((16, 32), True, tl.int1)).to(tl.int1)
    e3 = (tl.load(mem0 + bid * 544 + 16 + e1 * 1, (e2 & (e1 >= 0) & (e1 < 512)), other=0)).to(tl.float16)
    e4 = ((((tl.arange(0, 16)[:, None] * 32 + tl.arange(0, 32)[None, :]) + 17) % 512)).to(tl.int32)
    e5 = (tl.full((16, 32), True, tl.int1)).to(tl.int1)
    e6 = (tl.load(mem1 + bid * 544 + 16 + e4 * 1, (e5 & (e4 >= 0) & (e4 < 512)), other=0)).to(tl.float16)
    e7 = ((e3 * e6)).to(tl.float16)
    e8 = (e7.to(tl.float32)).to(tl.float32)
    e9 = (e8.to(tl.float16)).to(tl.float16)
    e10 = (tl.full((16, 32), 0.5, tl.float16)).to(tl.float16)
    e11 = ((e9 * e10)).to(tl.float16)
    e12 = ((((tl.arange(0, 16)[:, None] * 32 + tl.arange(0, 32)[None, :]) + 71) % 512)).to(tl.int32)
    e13 = (tl.full((16, 32), True, tl.int1)).to(tl.int1)
    e14 = (tl.load(mem2 + bid * 544 + 16 + e12 * 1, (e13 & (e12 >= 0) & (e12 < 512)), other=0)).to(tl.int32)
    e15 = (tl.full((16, 32), 3, tl.int32)).to(tl.int32)
    e16 = ((e14 & e15)).to(tl.int32)
    e17 = (tl.full((16, 32), 3, tl.int32)).to(tl.int32)
    e18 = ((e16 < e17)).to(tl.int1)
    e19 = ((((511 - (tl.arange(0, 16)[:, None] * 32 + tl.arange(0, 32)[None, :])) + 252) % 512)).to(tl.int32)
    e20 = (limit).to(tl.int32)
    e21 = ((e19 < e20)).to(tl.int1)
    e22 = ((e18 & e21)).to(tl.int1)
    e23 = (tl.full((16, 32), -0.125, tl.float16)).to(tl.float16)
    e24 = (tl.where(e22, e11, e23)).to(tl.float16)
    e25 = ((((511 - (tl.arange(0, 16)[:, None] * 32 + tl.arange(0, 32)[None, :])) + 305) % 512)).to(tl.int32)
    e26 = (limit).to(tl.int32)
    e27 = ((e25 < e26)).to(tl.int1)
    tl.store(mem3 + bid * 1060 + 17 + e25 * 2, e24, (e27 & (e25 >= 0) & (e25 < 512)))
    tl.debug_barrier()
    e28 = (tl.load(mem3 + bid * 1060 + 17 + e25 * 2, (e27 & (e25 >= 0) & (e25 < 512)), other=0)).to(tl.float16)
    e29 = (tl.full((16, 32), 0.125, tl.float16)).to(tl.float16)
    e30 = ((e28 + e29)).to(tl.float16)
    tl.debug_barrier()
    tl.store(mem3 + bid * 1060 + 18 + e25 * 1, e30, (e27 & (e25 >= 0) & (e25 < 512)))
    tl.debug_barrier()
    e31 = (tl.full((16, 32), True, tl.int1)).to(tl.int1)
    e32 = (tl.load(mem3 + bid * 1060 + 17 + e25 * 2, (e31 & (e25 >= 0) & (e25 < 512)), other=0)).to(tl.float16)
    e33 = (tl.full((16, 32), 3, tl.int32)).to(tl.int32)
    e34 = ((e25 ^ e33)).to(tl.int32)
    e35 = (tl.load(mem3 + bid * 1060 + 18 + e34 * 1, (e27 & (e34 >= 0) & (e34 < 512)), other=0)).to(tl.float16)
    e36 = ((e32 + e35)).to(tl.float16)
    e37 = (e36.to(tl.float16)).to(tl.float16)
    e38 = (tl.trans(e37)).to(tl.float16)
    e39_slice00_lo, e39_slice00_hi = tl.split(tl.permute(tl.reshape(e38, (2, 16, 16)), (1, 2, 0)))
    e39_slice00_next_lo, e39_slice00_next_hi = tl.split(tl.permute(tl.reshape(e39_slice00_lo, (2, 8, 16)), (1, 2, 0)))
    e39_slice01_lo, e39_slice01_hi = tl.split(tl.permute(tl.reshape(e38, (2, 16, 16)), (1, 2, 0)))
    e39_slice01_next_lo, e39_slice01_next_hi = tl.split(tl.permute(tl.reshape(e39_slice01_hi, (2, 8, 16)), (1, 2, 0)))
    e39_slice0 = tl.reshape(tl.permute(tl.join(e39_slice00_next_hi, e39_slice01_next_lo), (2, 0, 1)), (16, 16))
    e39 = (e39_slice0).to(tl.float16)
    e40 = (tl.full((16, 16), 0.0, tl.float32)).to(tl.float32)
    e41 = (tl.dot(e39, e39, e40, input_precision='tf32')).to(tl.float32)
    e42 = (tl.where(tl.sum((e41 != e41).to(tl.int32), 1) > 0, float('nan'), tl.max(e41.to(tl.float32), 1))).to(tl.float32)
    e43 = (tl.reshape(e42, (16, 1))).to(tl.float32)
    e44 = ((e41 + e43)).to(tl.float32)
    e45 = (tl.full((16, 16), 0.125, tl.float32)).to(tl.float32)
    e46 = ((e44 * e45)).to(tl.float32)
    e47 = ((((tl.arange(0, 16)[:, None] * 16 + tl.arange(0, 16)[None, :]) + 44) % 256)).to(tl.int32)
    e48 = (tl.full((16, 16), True, tl.int1)).to(tl.int1)
    e49 = (tl.load(mem6 + bid * 288 + 16 + e47 * 1, (e48 & (e47 >= 0) & (e47 < 256)), other=0)).to(tl.float16)
    e50 = (e46.to(tl.float16)).to(tl.float16)
    e51 = (tl.dot(e50, e49, e40, input_precision='tf32')).to(tl.float32)
    e52 = (tl.reshape(e51, (16, 16))).to(tl.float32)
    e53 = (tl.reshape(e52, (16, 16))).to(tl.float32)
    e54 = (e53.to(tl.float16)).to(tl.float16)
    e63 = (e42.to(tl.float32)).to(tl.float32)
    e64, e65 = mixed_helper_extended_tf32(e54, e63, mem0, mem1, mem2, mem3, mem6, mem7, mem8, mem9, steps, limit, bid)
    e66 = (steps).to(tl.int32)
    e67 = (tl.full((), 0, tl.int32)).to(tl.int32)
    e68 = (tl.full((), 0, tl.int32)).to(tl.int32)
    e69 = (tl.full((), True, tl.int1)).to(tl.int1)
    e70 = (tl.load(mem7 + bid * 33 + 16 + e68 * 1, (e69 & (e68 >= 0) & (e68 < 1)), other=0)).to(tl.int32)
    e71 = (tl.full((), 0, tl.int32)).to(tl.int32)
    e72 = ((e70 < e71)).to(tl.int1)
    e92 = e64
    e93 = e65
    e94 = e67
    for e73 in range(tl.minimum(tl.maximum(e66, 0), 4)):
        e74 = e92
        e75 = e93
        e76 = e94
        if e72:
            e77 = e74
            e78 = (tl.full((16, 16), 0.125, tl.float16)).to(tl.float16)
            e79 = ((e77 + e78)).to(tl.float16)
            e83 = e79
        else:
            e80 = e74
            e81 = (tl.full((16, 16), 0.125, tl.float16)).to(tl.float16)
            e82 = ((e80 - e81)).to(tl.float16)
            e83 = e82
        e84, e85 = mixed_helper_extended_tf32(e83, e63, mem0, mem1, mem2, mem3, mem6, mem7, mem8, mem9, steps, limit, bid)
        e86 = (tl.full((16, 16), 0.125, tl.float16)).to(tl.float16)
        e87 = ((e84 * e86)).to(tl.float16)
        e88 = ((e75 + e85)).to(tl.float32)
        e89 = (tl.full((), 1, tl.int32)).to(tl.int32)
        e90 = ((e73 + e89)).to(tl.int32)
        e91 = ((e76 + e90)).to(tl.int32)
        e92 = e87
        e93 = e88
        e94 = e91
    e114 = e92
    e115 = e93
    e116 = e94
    e95 = tl.full((), 0, tl.int32)
    while e95 < tl.minimum(tl.maximum(e66, 0), 4):
        e96 = e114
        e97 = e115
        e98 = e116
        if e72:
            e99 = e96
            e100 = (tl.full((16, 16), 0.125, tl.float16)).to(tl.float16)
            e101 = ((e99 + e100)).to(tl.float16)
            e105 = e101
        else:
            e102 = e96
            e103 = (tl.full((16, 16), 0.125, tl.float16)).to(tl.float16)
            e104 = ((e102 - e103)).to(tl.float16)
            e105 = e104
        e106, e107 = mixed_helper_extended_tf32(e105, e63, mem0, mem1, mem2, mem3, mem6, mem7, mem8, mem9, steps, limit, bid)
        e108 = (tl.full((16, 16), 0.125, tl.float16)).to(tl.float16)
        e109 = ((e106 * e108)).to(tl.float16)
        e110 = ((e97 + e107)).to(tl.float32)
        e111 = (tl.full((), 1, tl.int32)).to(tl.int32)
        e112 = ((e95 + e111)).to(tl.int32)
        e113 = ((e98 + e112)).to(tl.int32)
        e114 = e109
        e115 = e110
        e116 = e113
        e95 += 1
    e117 = (tl.full((), -0.125, tl.float32)).to(tl.float32)
    e118 = (tl.broadcast_to(e117, (16, 16))).to(tl.float32)
    e119 = (tl.full((), -0.25, tl.float32)).to(tl.float32)
    e120 = (tl.broadcast_to(e119, (16, 16))).to(tl.float32)
    e121 = ((e118 - e120)).to(tl.float32)
    e122 = ((((255 - (tl.arange(0, 16)[:, None] * 16 + tl.arange(0, 16)[None, :])) + 28) % 256)).to(tl.int32)
    e123 = (tl.full((16, 16), True, tl.int1)).to(tl.int1)
    tl.atomic_max(mem8 + bid * 288 + 16 + e122 * 1, e121, (e123 & (e122 >= 0) & (e122 < 256)))
    e124 = ((e114 - e92)).to(tl.float16)
    e125 = (e124.to(tl.float32)).to(tl.float32)
    e126 = ((((255 - (tl.arange(0, 16)[:, None] * 16 + tl.arange(0, 16)[None, :])) + 194) % 256)).to(tl.int32)
    e127 = (tl.full((16, 16), True, tl.int1)).to(tl.int1)
    e128 = (tl.load(mem9 + bid * 288 + 16 + e126 * 1, (e127 & (e126 >= 0) & (e126 < 256)), other=0)).to(tl.float32)
    e129 = ((e125 - e128)).to(tl.float32)
    e130 = (e129.to(tl.float16)).to(tl.float16)
    e131 = ((e130 + e49)).to(tl.float16)
    e132 = (tl.where(tl.sum((e131 != e131).to(tl.int32), 0) > 0, float('nan'), tl.max(e131.to(tl.float32), 0))).to(tl.float32)
    tl.store(out_e131 + bid * 288 + 16 + (tl.arange(0, 16)[:, None] * 16 + tl.arange(0, 16)[None, :]), e131)
    tl.store(out_e16 + bid * 544 + 16 + (tl.arange(0, 16)[:, None] * 32 + tl.arange(0, 32)[None, :]), e16)
    tl.store(out_e22 + bid * 544 + 16 + (tl.arange(0, 16)[:, None] * 32 + tl.arange(0, 32)[None, :]), e22)
    tl.store(out_e115 + bid * 33 + 16 + tl.full((), 0, tl.int32), e115)
    tl.store(out_e116 + bid * 33 + 16 + tl.full((), 0, tl.int32), e116)
    tl.store(out_e132 + bid * 48 + 16 + tl.arange(0, 16), e132)

def prepare_extended(device_compile=True):
    import torch
    from triton.compiler import ASTSource, compile
    from triton.backends.compiler import GPUTarget
    arch = int(os.environ.get('TILESMITH_CUDA_ARCH', '0'))
    if not arch:
        major, minor = torch.cuda.get_device_capability() if torch.cuda.is_available() else (8, 9)
        arch = major * 10 + minor
    variants = []
    extended_stage("compile", 'triton_0')
    compiled_0 = compile(ASTSource(extended_0_0, {'mem0': '*fp16', 'mem1': '*fp16', 'mem2': '*i32', 'mem3': '*fp16', 'mem6': '*fp16', 'mem7': '*i32', 'mem8': '*fp32', 'mem9': '*fp32', 'out_e131': '*fp16', 'out_e16': '*i32', 'out_e22': '*i1', 'out_e115': '*fp32', 'out_e116': '*i32', 'out_e132': '*fp32', 'steps': 'i32', 'limit': 'i32'}), target=GPUTarget("cuda", arch, 32), options={'num_warps': 4, 'num_stages': 1, 'enable_fp_fusion': False})
    record_extended_compilation('triton_0', compiled_0.asm, {'num_warps': 4, 'num_stages': 1, 'enable_fp_fusion': False})
    def launch_0(memories, outputs, steps, limit):
        arguments = [memories[n] for n in ['mem0', 'mem1', 'mem2', 'mem3', 'mem6', 'mem7', 'mem8', 'mem9']] + [outputs[n] for n in ['e131', 'e16', 'e22', 'e115', 'e116', 'e132']]
        compiled_0[(2, 1, 1)](*arguments, steps, limit)
    variants.append(('triton_0', ['e131', 'e16', 'e22', 'e115', 'e116', 'e132'], launch_0))
    extended_stage("compile", 'triton_1')
    compiled_1 = compile(ASTSource(extended_0_1, {'mem0': '*fp16', 'mem1': '*fp16', 'mem2': '*i32', 'mem3': '*fp16', 'mem6': '*fp16', 'mem7': '*i32', 'mem8': '*fp32', 'mem9': '*fp32', 'out_e131': '*fp16', 'out_e16': '*i32', 'out_e22': '*i1', 'out_e115': '*fp32', 'out_e116': '*i32', 'out_e132': '*fp32', 'out_e3': '*fp16', 'out_e11': '*fp16', 'out_e28': '*fp16', 'out_e32': '*fp16', 'out_e41': '*fp32', 'out_e46': '*fp32', 'out_e21': '*i1', 'out_e4': '*i32', 'steps': 'i32', 'limit': 'i32'}), target=GPUTarget("cuda", arch, 32), options={'num_warps': 4, 'num_stages': 1, 'enable_fp_fusion': False})
    record_extended_compilation('triton_1', compiled_1.asm, {'num_warps': 4, 'num_stages': 1, 'enable_fp_fusion': False})
    def launch_1(memories, outputs, steps, limit):
        arguments = [memories[n] for n in ['mem0', 'mem1', 'mem2', 'mem3', 'mem6', 'mem7', 'mem8', 'mem9']] + [outputs[n] for n in ['e131', 'e16', 'e22', 'e115', 'e116', 'e132', 'e3', 'e11', 'e28', 'e32', 'e41', 'e46', 'e21', 'e4']]
        compiled_1[(2, 1, 1)](*arguments, steps, limit)
    variants.append(('triton_1', ['e131', 'e16', 'e22', 'e115', 'e116', 'e132', 'e3', 'e11', 'e28', 'e32', 'e41', 'e46', 'e21', 'e4'], launch_1))
    extended_stage("compile", 'triton_2')
    compiled_2 = compile(ASTSource(extended_1_0, {'mem0': '*fp16', 'mem1': '*fp16', 'mem2': '*i32', 'mem3': '*fp16', 'mem6': '*fp16', 'mem7': '*i32', 'mem8': '*fp32', 'mem9': '*fp32', 'out_e131': '*fp16', 'out_e16': '*i32', 'out_e22': '*i1', 'out_e115': '*fp32', 'out_e116': '*i32', 'out_e132': '*fp32', 'steps': 'i32', 'limit': 'i32'}), target=GPUTarget("cuda", arch, 32), options={'num_warps': 8, 'num_stages': 2, 'enable_fp_fusion': False})
    record_extended_compilation('triton_2', compiled_2.asm, {'num_warps': 8, 'num_stages': 2, 'enable_fp_fusion': False})
    def launch_2(memories, outputs, steps, limit):
        arguments = [memories[n] for n in ['mem0', 'mem1', 'mem2', 'mem3', 'mem6', 'mem7', 'mem8', 'mem9']] + [outputs[n] for n in ['e131', 'e16', 'e22', 'e115', 'e116', 'e132']]
        compiled_2[(2, 1, 1)](*arguments, steps, limit)
    variants.append(('triton_2', ['e131', 'e16', 'e22', 'e115', 'e116', 'e132'], launch_2))
    extended_stage("compile", 'triton_3')
    compiled_3 = compile(ASTSource(extended_1_1, {'mem0': '*fp16', 'mem1': '*fp16', 'mem2': '*i32', 'mem3': '*fp16', 'mem6': '*fp16', 'mem7': '*i32', 'mem8': '*fp32', 'mem9': '*fp32', 'out_e131': '*fp16', 'out_e16': '*i32', 'out_e22': '*i1', 'out_e115': '*fp32', 'out_e116': '*i32', 'out_e132': '*fp32', 'out_e3': '*fp16', 'out_e11': '*fp16', 'out_e28': '*fp16', 'out_e32': '*fp16', 'out_e41': '*fp32', 'out_e46': '*fp32', 'out_e21': '*i1', 'out_e4': '*i32', 'steps': 'i32', 'limit': 'i32'}), target=GPUTarget("cuda", arch, 32), options={'num_warps': 8, 'num_stages': 2, 'enable_fp_fusion': False})
    record_extended_compilation('triton_3', compiled_3.asm, {'num_warps': 8, 'num_stages': 2, 'enable_fp_fusion': False})
    def launch_3(memories, outputs, steps, limit):
        arguments = [memories[n] for n in ['mem0', 'mem1', 'mem2', 'mem3', 'mem6', 'mem7', 'mem8', 'mem9']] + [outputs[n] for n in ['e131', 'e16', 'e22', 'e115', 'e116', 'e132', 'e3', 'e11', 'e28', 'e32', 'e41', 'e46', 'e21', 'e4']]
        compiled_3[(2, 1, 1)](*arguments, steps, limit)
    variants.append(('triton_3', ['e131', 'e16', 'e22', 'e115', 'e116', 'e132', 'e3', 'e11', 'e28', 'e32', 'e41', 'e46', 'e21', 'e4'], launch_3))
    extended_stage("compile", 'triton_4')
    compiled_4 = compile(ASTSource(extended_2_0, {'mem0': '*fp16', 'mem1': '*fp16', 'mem2': '*i32', 'mem3': '*fp16', 'mem6': '*fp16', 'mem7': '*i32', 'mem8': '*fp32', 'mem9': '*fp32', 'out_e131': '*fp16', 'out_e16': '*i32', 'out_e22': '*i1', 'out_e115': '*fp32', 'out_e116': '*i32', 'out_e132': '*fp32', 'steps': 'i32', 'limit': 'i32'}), target=GPUTarget("cuda", arch, 32), options={'num_warps': 2, 'num_stages': 2, 'enable_fp_fusion': False, 'maxnreg': 168})
    record_extended_compilation('triton_4', compiled_4.asm, {'num_warps': 2, 'num_stages': 2, 'enable_fp_fusion': False, 'maxnreg': 168})
    def launch_4(memories, outputs, steps, limit):
        arguments = [memories[n] for n in ['mem0', 'mem1', 'mem2', 'mem3', 'mem6', 'mem7', 'mem8', 'mem9']] + [outputs[n] for n in ['e131', 'e16', 'e22', 'e115', 'e116', 'e132']]
        compiled_4[(2, 1, 1)](*arguments, steps, limit)
    variants.append(('triton_4', ['e131', 'e16', 'e22', 'e115', 'e116', 'e132'], launch_4))
    extended_stage("compile", 'triton_5')
    compiled_5 = compile(ASTSource(extended_2_1, {'mem0': '*fp16', 'mem1': '*fp16', 'mem2': '*i32', 'mem3': '*fp16', 'mem6': '*fp16', 'mem7': '*i32', 'mem8': '*fp32', 'mem9': '*fp32', 'out_e131': '*fp16', 'out_e16': '*i32', 'out_e22': '*i1', 'out_e115': '*fp32', 'out_e116': '*i32', 'out_e132': '*fp32', 'out_e3': '*fp16', 'out_e11': '*fp16', 'out_e28': '*fp16', 'out_e32': '*fp16', 'out_e41': '*fp32', 'out_e46': '*fp32', 'out_e21': '*i1', 'out_e4': '*i32', 'steps': 'i32', 'limit': 'i32'}), target=GPUTarget("cuda", arch, 32), options={'num_warps': 2, 'num_stages': 2, 'enable_fp_fusion': False, 'maxnreg': 168})
    record_extended_compilation('triton_5', compiled_5.asm, {'num_warps': 2, 'num_stages': 2, 'enable_fp_fusion': False, 'maxnreg': 168})
    def launch_5(memories, outputs, steps, limit):
        arguments = [memories[n] for n in ['mem0', 'mem1', 'mem2', 'mem3', 'mem6', 'mem7', 'mem8', 'mem9']] + [outputs[n] for n in ['e131', 'e16', 'e22', 'e115', 'e116', 'e132', 'e3', 'e11', 'e28', 'e32', 'e41', 'e46', 'e21', 'e4']]
        compiled_5[(2, 1, 1)](*arguments, steps, limit)
    variants.append(('triton_5', ['e131', 'e16', 'e22', 'e115', 'e116', 'e132', 'e3', 'e11', 'e28', 'e32', 'e41', 'e46', 'e21', 'e4'], launch_5))
    extended_stage("compile", 'triton_6')
    compiled_6 = compile(ASTSource(extended_3_0, {'mem0': '*fp16', 'mem1': '*fp16', 'mem2': '*i32', 'mem3': '*fp16', 'mem6': '*fp16', 'mem7': '*i32', 'mem8': '*fp32', 'mem9': '*fp32', 'out_e131': '*fp16', 'out_e16': '*i32', 'out_e22': '*i1', 'out_e115': '*fp32', 'out_e116': '*i32', 'out_e132': '*fp32', 'steps': 'i32', 'limit': 'i32'}), target=GPUTarget("cuda", arch, 32), options={'num_warps': 8, 'num_stages': 3, 'enable_fp_fusion': False, 'maxnreg': 128})
    record_extended_compilation('triton_6', compiled_6.asm, {'num_warps': 8, 'num_stages': 3, 'enable_fp_fusion': False, 'maxnreg': 128})
    def launch_6(memories, outputs, steps, limit):
        arguments = [memories[n] for n in ['mem0', 'mem1', 'mem2', 'mem3', 'mem6', 'mem7', 'mem8', 'mem9']] + [outputs[n] for n in ['e131', 'e16', 'e22', 'e115', 'e116', 'e132']]
        compiled_6[(2, 1, 1)](*arguments, steps, limit)
    variants.append(('triton_6', ['e131', 'e16', 'e22', 'e115', 'e116', 'e132'], launch_6))
    extended_stage("compile", 'triton_7')
    compiled_7 = compile(ASTSource(extended_3_1, {'mem0': '*fp16', 'mem1': '*fp16', 'mem2': '*i32', 'mem3': '*fp16', 'mem6': '*fp16', 'mem7': '*i32', 'mem8': '*fp32', 'mem9': '*fp32', 'out_e131': '*fp16', 'out_e16': '*i32', 'out_e22': '*i1', 'out_e115': '*fp32', 'out_e116': '*i32', 'out_e132': '*fp32', 'out_e3': '*fp16', 'out_e11': '*fp16', 'out_e28': '*fp16', 'out_e32': '*fp16', 'out_e41': '*fp32', 'out_e46': '*fp32', 'out_e21': '*i1', 'out_e4': '*i32', 'steps': 'i32', 'limit': 'i32'}), target=GPUTarget("cuda", arch, 32), options={'num_warps': 8, 'num_stages': 3, 'enable_fp_fusion': False, 'maxnreg': 128})
    record_extended_compilation('triton_7', compiled_7.asm, {'num_warps': 8, 'num_stages': 3, 'enable_fp_fusion': False, 'maxnreg': 128})
    def launch_7(memories, outputs, steps, limit):
        arguments = [memories[n] for n in ['mem0', 'mem1', 'mem2', 'mem3', 'mem6', 'mem7', 'mem8', 'mem9']] + [outputs[n] for n in ['e131', 'e16', 'e22', 'e115', 'e116', 'e132', 'e3', 'e11', 'e28', 'e32', 'e41', 'e46', 'e21', 'e4']]
        compiled_7[(2, 1, 1)](*arguments, steps, limit)
    variants.append(('triton_7', ['e131', 'e16', 'e22', 'e115', 'e116', 'e132', 'e3', 'e11', 'e28', 'e32', 'e41', 'e46', 'e21', 'e4'], launch_7))
    extended_stage("compile", 'triton_8_prec')
    compiled_8 = compile(ASTSource(extended_0_prec, {'mem0': '*fp16', 'mem1': '*fp16', 'mem2': '*i32', 'mem3': '*fp16', 'mem6': '*fp16', 'mem7': '*i32', 'mem8': '*fp32', 'mem9': '*fp32', 'out_e131': '*fp16', 'out_e16': '*i32', 'out_e22': '*i1', 'out_e115': '*fp32', 'out_e116': '*i32', 'out_e132': '*fp32', 'out_e3': '*fp16', 'out_e11': '*fp16', 'out_e28': '*fp16', 'out_e32': '*fp16', 'out_e41': '*fp16', 'out_e46': '*fp32', 'out_e21': '*i1', 'out_e4': '*i32', 'steps': 'i32', 'limit': 'i32'}), target=GPUTarget("cuda", arch, 32), options={'num_warps': 4, 'num_stages': 1, 'enable_fp_fusion': False})
    record_extended_compilation('triton_8_prec', compiled_8.asm, {'num_warps': 4, 'num_stages': 1, 'enable_fp_fusion': False, 'precision': 'fp16'})
    def launch_8(memories, outputs, steps, limit):
        arguments = [memories[n] for n in ['mem0', 'mem1', 'mem2', 'mem3', 'mem6', 'mem7', 'mem8', 'mem9']] + [outputs[n] for n in ['e131', 'e16', 'e22', 'e115', 'e116', 'e132', 'e3', 'e11', 'e28', 'e32', 'e41', 'e46', 'e21', 'e4']]
        compiled_8[(2, 1, 1)](*arguments, steps, limit)
    variants.append(('triton_8_prec', ['e131', 'e16', 'e22', 'e115', 'e116', 'e132', 'e3', 'e11', 'e28', 'e32', 'e41', 'e46', 'e21', 'e4'], launch_8))
    extended_stage("compile", 'triton_9_prec')
    compiled_9 = compile(ASTSource(extended_1_prec, {'mem0': '*fp16', 'mem1': '*fp16', 'mem2': '*i32', 'mem3': '*fp16', 'mem6': '*fp16', 'mem7': '*i32', 'mem8': '*fp32', 'mem9': '*fp32', 'out_e131': '*fp16', 'out_e16': '*i32', 'out_e22': '*i1', 'out_e115': '*fp32', 'out_e116': '*i32', 'out_e132': '*fp32', 'out_e3': '*fp16', 'out_e11': '*fp16', 'out_e28': '*fp16', 'out_e32': '*fp16', 'out_e41': '*fp16', 'out_e46': '*fp32', 'out_e21': '*i1', 'out_e4': '*i32', 'steps': 'i32', 'limit': 'i32'}), target=GPUTarget("cuda", arch, 32), options={'num_warps': 8, 'num_stages': 2, 'enable_fp_fusion': False})
    record_extended_compilation('triton_9_prec', compiled_9.asm, {'num_warps': 8, 'num_stages': 2, 'enable_fp_fusion': False, 'precision': 'fp16'})
    def launch_9(memories, outputs, steps, limit):
        arguments = [memories[n] for n in ['mem0', 'mem1', 'mem2', 'mem3', 'mem6', 'mem7', 'mem8', 'mem9']] + [outputs[n] for n in ['e131', 'e16', 'e22', 'e115', 'e116', 'e132', 'e3', 'e11', 'e28', 'e32', 'e41', 'e46', 'e21', 'e4']]
        compiled_9[(2, 1, 1)](*arguments, steps, limit)
    variants.append(('triton_9_prec', ['e131', 'e16', 'e22', 'e115', 'e116', 'e132', 'e3', 'e11', 'e28', 'e32', 'e41', 'e46', 'e21', 'e4'], launch_9))
    extended_stage("compile", 'triton_10')
    compiled_10 = compile(ASTSource(extended_tf32, {'mem0': '*fp16', 'mem1': '*fp16', 'mem2': '*i32', 'mem3': '*fp16', 'mem6': '*fp16', 'mem7': '*i32', 'mem8': '*fp32', 'mem9': '*fp32', 'out_e131': '*fp16', 'out_e16': '*i32', 'out_e22': '*i1', 'out_e115': '*fp32', 'out_e116': '*i32', 'out_e132': '*fp32', 'steps': 'i32', 'limit': 'i32'}), target=GPUTarget("cuda", arch, 32), options={'num_warps': 4, 'num_stages': 1, 'enable_fp_fusion': False})
    record_extended_compilation('triton_10', compiled_10.asm, {'num_warps': 4, 'num_stages': 1, 'enable_fp_fusion': False, 'input_precision': 'tf32'})
    def launch_10(memories, outputs, steps, limit):
        arguments = [memories[n] for n in ['mem0', 'mem1', 'mem2', 'mem3', 'mem6', 'mem7', 'mem8', 'mem9']] + [outputs[n] for n in ['e131', 'e16', 'e22', 'e115', 'e116', 'e132']]
        compiled_10[(2, 1, 1)](*arguments, steps, limit)
    variants.append(('triton_10', ['e131', 'e16', 'e22', 'e115', 'e116', 'e132'], launch_10))
    return variants

PROGRAM = {'type': 'extended', 'version': 1, 'body': {'arguments': [], 'operations': [{'op': 'index', 'results': [{'name': 'e1', 'type': {'dtype': 'int32', 'shape': (16, 32)}}], 'operands': [], 'attrs': {'shift': 10, 'reverse': False}, 'regions': []}, {'op': 'constant', 'results': [{'name': 'e2', 'type': {'dtype': 'bool', 'shape': (16, 32)}}], 'operands': [], 'attrs': {'value': True}, 'regions': []}, {'op': 'load', 'results': [{'name': 'e3', 'type': {'dtype': 'float16', 'shape': (16, 32)}}], 'operands': ['e1', 'e2'], 'attrs': {'buffer': 'mem0'}, 'regions': []}, {'op': 'index', 'results': [{'name': 'e4', 'type': {'dtype': 'int32', 'shape': (16, 32)}}], 'operands': [], 'attrs': {'shift': 17, 'reverse': False}, 'regions': []}, {'op': 'constant', 'results': [{'name': 'e5', 'type': {'dtype': 'bool', 'shape': (16, 32)}}], 'operands': [], 'attrs': {'value': True}, 'regions': []}, {'op': 'load', 'results': [{'name': 'e6', 'type': {'dtype': 'float16', 'shape': (16, 32)}}], 'operands': ['e4', 'e5'], 'attrs': {'buffer': 'mem1'}, 'regions': []}, {'op': 'mul', 'results': [{'name': 'e7', 'type': {'dtype': 'float16', 'shape': (16, 32)}}], 'operands': ['e3', 'e6'], 'attrs': {}, 'regions': []}, {'op': 'cast', 'results': [{'name': 'e8', 'type': {'dtype': 'float32', 'shape': (16, 32)}}], 'operands': ['e7'], 'attrs': {}, 'regions': []}, {'op': 'cast', 'results': [{'name': 'e9', 'type': {'dtype': 'float16', 'shape': (16, 32)}}], 'operands': ['e8'], 'attrs': {}, 'regions': []}, {'op': 'constant', 'results': [{'name': 'e10', 'type': {'dtype': 'float16', 'shape': (16, 32)}}], 'operands': [], 'attrs': {'value': 0.5}, 'regions': []}, {'op': 'mul', 'results': [{'name': 'e11', 'type': {'dtype': 'float16', 'shape': (16, 32)}}], 'operands': ['e9', 'e10'], 'attrs': {}, 'regions': []}, {'op': 'index', 'results': [{'name': 'e12', 'type': {'dtype': 'int32', 'shape': (16, 32)}}], 'operands': [], 'attrs': {'shift': 71, 'reverse': False}, 'regions': []}, {'op': 'constant', 'results': [{'name': 'e13', 'type': {'dtype': 'bool', 'shape': (16, 32)}}], 'operands': [], 'attrs': {'value': True}, 'regions': []}, {'op': 'load', 'results': [{'name': 'e14', 'type': {'dtype': 'int32', 'shape': (16, 32)}}], 'operands': ['e12', 'e13'], 'attrs': {'buffer': 'mem2'}, 'regions': []}, {'op': 'constant', 'results': [{'name': 'e15', 'type': {'dtype': 'int32', 'shape': (16, 32)}}], 'operands': [], 'attrs': {'value': 3}, 'regions': []}, {'op': 'bitand', 'results': [{'name': 'e16', 'type': {'dtype': 'int32', 'shape': (16, 32)}}], 'operands': ['e14', 'e15'], 'attrs': {}, 'regions': []}, {'op': 'constant', 'results': [{'name': 'e17', 'type': {'dtype': 'int32', 'shape': (16, 32)}}], 'operands': [], 'attrs': {'value': 3}, 'regions': []}, {'op': 'lt', 'results': [{'name': 'e18', 'type': {'dtype': 'bool', 'shape': (16, 32)}}], 'operands': ['e16', 'e17'], 'attrs': {}, 'regions': []}, {'op': 'index', 'results': [{'name': 'e19', 'type': {'dtype': 'int32', 'shape': (16, 32)}}], 'operands': [], 'attrs': {'shift': 252, 'reverse': True}, 'regions': []}, {'op': 'parameter', 'results': [{'name': 'e20', 'type': {'dtype': 'int32', 'shape': ()}}], 'operands': [], 'attrs': {'name': 'limit'}, 'regions': []}, {'op': 'lt', 'results': [{'name': 'e21', 'type': {'dtype': 'bool', 'shape': (16, 32)}}], 'operands': ['e19', 'e20'], 'attrs': {}, 'regions': []}, {'op': 'and', 'results': [{'name': 'e22', 'type': {'dtype': 'bool', 'shape': (16, 32)}}], 'operands': ['e18', 'e21'], 'attrs': {}, 'regions': []}, {'op': 'constant', 'results': [{'name': 'e23', 'type': {'dtype': 'float16', 'shape': (16, 32)}}], 'operands': [], 'attrs': {'value': -0.125}, 'regions': []}, {'op': 'select', 'results': [{'name': 'e24', 'type': {'dtype': 'float16', 'shape': (16, 32)}}], 'operands': ['e22', 'e11', 'e23'], 'attrs': {}, 'regions': []}, {'op': 'index', 'results': [{'name': 'e25', 'type': {'dtype': 'int32', 'shape': (16, 32)}}], 'operands': [], 'attrs': {'shift': 305, 'reverse': True}, 'regions': []}, {'op': 'parameter', 'results': [{'name': 'e26', 'type': {'dtype': 'int32', 'shape': ()}}], 'operands': [], 'attrs': {'name': 'limit'}, 'regions': []}, {'op': 'lt', 'results': [{'name': 'e27', 'type': {'dtype': 'bool', 'shape': (16, 32)}}], 'operands': ['e25', 'e26'], 'attrs': {}, 'regions': []}, {'op': 'store', 'results': [], 'operands': ['e25', 'e27', 'e24'], 'attrs': {'buffer': 'mem4'}, 'regions': []}, {'op': 'barrier', 'results': [], 'operands': [], 'attrs': {}, 'regions': []}, {'op': 'load', 'results': [{'name': 'e28', 'type': {'dtype': 'float16', 'shape': (16, 32)}}], 'operands': ['e25', 'e27'], 'attrs': {'buffer': 'mem4'}, 'regions': []}, {'op': 'constant', 'results': [{'name': 'e29', 'type': {'dtype': 'float16', 'shape': (16, 32)}}], 'operands': [], 'attrs': {'value': 0.125}, 'regions': []}, {'op': 'add', 'results': [{'name': 'e30', 'type': {'dtype': 'float16', 'shape': (16, 32)}}], 'operands': ['e28', 'e29'], 'attrs': {}, 'regions': []}, {'op': 'barrier', 'results': [], 'operands': [], 'attrs': {}, 'regions': []}, {'op': 'store', 'results': [], 'operands': ['e25', 'e27', 'e30'], 'attrs': {'buffer': 'mem5'}, 'regions': []}, {'op': 'barrier', 'results': [], 'operands': [], 'attrs': {}, 'regions': []}, {'op': 'constant', 'results': [{'name': 'e31', 'type': {'dtype': 'bool', 'shape': (16, 32)}}], 'operands': [], 'attrs': {'value': True}, 'regions': []}, {'op': 'load', 'results': [{'name': 'e32', 'type': {'dtype': 'float16', 'shape': (16, 32)}}], 'operands': ['e25', 'e31'], 'attrs': {'buffer': 'mem4'}, 'regions': []}, {'op': 'constant', 'results': [{'name': 'e33', 'type': {'dtype': 'int32', 'shape': (16, 32)}}], 'operands': [], 'attrs': {'value': 3}, 'regions': []}, {'op': 'bitxor', 'results': [{'name': 'e34', 'type': {'dtype': 'int32', 'shape': (16, 32)}}], 'operands': ['e25', 'e33'], 'attrs': {}, 'regions': []}, {'op': 'load', 'results': [{'name': 'e35', 'type': {'dtype': 'float16', 'shape': (16, 32)}}], 'operands': ['e34', 'e27'], 'attrs': {'buffer': 'mem5'}, 'regions': []}, {'op': 'add', 'results': [{'name': 'e36', 'type': {'dtype': 'float16', 'shape': (16, 32)}}], 'operands': ['e32', 'e35'], 'attrs': {}, 'regions': []}, {'op': 'cast', 'results': [{'name': 'e37', 'type': {'dtype': 'float16', 'shape': (16, 32)}}], 'operands': ['e36'], 'attrs': {}, 'regions': []}, {'op': 'transpose', 'results': [{'name': 'e38', 'type': {'dtype': 'float16', 'shape': (32, 16)}}], 'operands': ['e37'], 'attrs': {}, 'regions': []}, {'op': 'slice', 'results': [{'name': 'e39', 'type': {'dtype': 'float16', 'shape': (16, 16)}}], 'operands': ['e38'], 'attrs': {'offsets': [8, 0]}, 'regions': []}, {'op': 'constant', 'results': [{'name': 'e40', 'type': {'dtype': 'float32', 'shape': (16, 16)}}], 'operands': [], 'attrs': {'value': 0.0}, 'regions': []}, {'op': 'matmul', 'results': [{'name': 'e41', 'type': {'dtype': 'float32', 'shape': (16, 16)}}], 'operands': ['e39', 'e39', 'e40'], 'attrs': {}, 'regions': []}, {'op': 'reduce', 'results': [{'name': 'e42', 'type': {'dtype': 'float32', 'shape': (16,)}}], 'operands': ['e41'], 'attrs': {'axis': 1, 'kind': 'max'}, 'regions': []}, {'op': 'reshape', 'results': [{'name': 'e43', 'type': {'dtype': 'float32', 'shape': (16, 1)}}], 'operands': ['e42'], 'attrs': {}, 'regions': []}, {'op': 'add', 'results': [{'name': 'e44', 'type': {'dtype': 'float32', 'shape': (16, 16)}}], 'operands': ['e41', 'e43'], 'attrs': {}, 'regions': []}, {'op': 'constant', 'results': [{'name': 'e45', 'type': {'dtype': 'float32', 'shape': (16, 16)}}], 'operands': [], 'attrs': {'value': 0.125}, 'regions': []}, {'op': 'mul', 'results': [{'name': 'e46', 'type': {'dtype': 'float32', 'shape': (16, 16)}}], 'operands': ['e44', 'e45'], 'attrs': {}, 'regions': []}, {'op': 'index', 'results': [{'name': 'e47', 'type': {'dtype': 'int32', 'shape': (16, 16)}}], 'operands': [], 'attrs': {'shift': 44, 'reverse': False}, 'regions': []}, {'op': 'constant', 'results': [{'name': 'e48', 'type': {'dtype': 'bool', 'shape': (16, 16)}}], 'operands': [], 'attrs': {'value': True}, 'regions': []}, {'op': 'load', 'results': [{'name': 'e49', 'type': {'dtype': 'float16', 'shape': (16, 16)}}], 'operands': ['e47', 'e48'], 'attrs': {'buffer': 'mem6'}, 'regions': []}, {'op': 'cast', 'results': [{'name': 'e50', 'type': {'dtype': 'float16', 'shape': (16, 16)}}], 'operands': ['e46'], 'attrs': {}, 'regions': []}, {'op': 'matmul', 'results': [{'name': 'e51', 'type': {'dtype': 'float32', 'shape': (16, 16)}}], 'operands': ['e50', 'e49', 'e40'], 'attrs': {}, 'regions': []}, {'op': 'reshape', 'results': [{'name': 'e52', 'type': {'dtype': 'float32', 'shape': (16, 16)}}], 'operands': ['e51'], 'attrs': {}, 'regions': []}, {'op': 'reshape', 'results': [{'name': 'e53', 'type': {'dtype': 'float32', 'shape': (16, 16)}}], 'operands': ['e52'], 'attrs': {}, 'regions': []}, {'op': 'cast', 'results': [{'name': 'e54', 'type': {'dtype': 'float16', 'shape': (16, 16)}}], 'operands': ['e53'], 'attrs': {}, 'regions': []}, {'op': 'cast', 'results': [{'name': 'e63', 'type': {'dtype': 'float32', 'shape': (16,)}}], 'operands': ['e42'], 'attrs': {}, 'regions': []}, {'op': 'call', 'results': [{'name': 'e64', 'type': {'dtype': 'float16', 'shape': (16, 16)}}, {'name': 'e65', 'type': {'dtype': 'float32', 'shape': ()}}], 'operands': ['e54', 'e63'], 'attrs': {'callee': 'mixed_helper'}, 'regions': []}, {'op': 'parameter', 'results': [{'name': 'e66', 'type': {'dtype': 'int32', 'shape': ()}}], 'operands': [], 'attrs': {'name': 'steps'}, 'regions': []}, {'op': 'constant', 'results': [{'name': 'e67', 'type': {'dtype': 'int32', 'shape': ()}}], 'operands': [], 'attrs': {'value': 0}, 'regions': []}, {'op': 'constant', 'results': [{'name': 'e68', 'type': {'dtype': 'int32', 'shape': ()}}], 'operands': [], 'attrs': {'value': 0}, 'regions': []}, {'op': 'constant', 'results': [{'name': 'e69', 'type': {'dtype': 'bool', 'shape': ()}}], 'operands': [], 'attrs': {'value': True}, 'regions': []}, {'op': 'load', 'results': [{'name': 'e70', 'type': {'dtype': 'int32', 'shape': ()}}], 'operands': ['e68', 'e69'], 'attrs': {'buffer': 'mem7'}, 'regions': []}, {'op': 'constant', 'results': [{'name': 'e71', 'type': {'dtype': 'int32', 'shape': ()}}], 'operands': [], 'attrs': {'value': 0}, 'regions': []}, {'op': 'lt', 'results': [{'name': 'e72', 'type': {'dtype': 'bool', 'shape': ()}}], 'operands': ['e70', 'e71'], 'attrs': {}, 'regions': []}, {'op': 'for', 'results': [{'name': 'e92', 'type': {'dtype': 'float16', 'shape': (16, 16)}}, {'name': 'e93', 'type': {'dtype': 'float32', 'shape': ()}}, {'name': 'e94', 'type': {'dtype': 'int32', 'shape': ()}}], 'operands': ['e66', 'e64', 'e65', 'e67'], 'attrs': {'max_steps': 4, 'pipelined': False}, 'regions': [{'arguments': [{'name': 'e73', 'type': {'dtype': 'int32', 'shape': ()}}, {'name': 'e74', 'type': {'dtype': 'float16', 'shape': (16, 16)}}, {'name': 'e75', 'type': {'dtype': 'float32', 'shape': ()}}, {'name': 'e76', 'type': {'dtype': 'int32', 'shape': ()}}], 'operations': [{'op': 'if', 'results': [{'name': 'e83', 'type': {'dtype': 'float16', 'shape': (16, 16)}}], 'operands': ['e72', 'e74'], 'attrs': {}, 'regions': [{'arguments': [{'name': 'e77', 'type': {'dtype': 'float16', 'shape': (16, 16)}}], 'operations': [{'op': 'constant', 'results': [{'name': 'e78', 'type': {'dtype': 'float16', 'shape': (16, 16)}}], 'operands': [], 'attrs': {'value': 0.125}, 'regions': []}, {'op': 'add', 'results': [{'name': 'e79', 'type': {'dtype': 'float16', 'shape': (16, 16)}}], 'operands': ['e77', 'e78'], 'attrs': {}, 'regions': []}], 'returns': ['e79']}, {'arguments': [{'name': 'e80', 'type': {'dtype': 'float16', 'shape': (16, 16)}}], 'operations': [{'op': 'constant', 'results': [{'name': 'e81', 'type': {'dtype': 'float16', 'shape': (16, 16)}}], 'operands': [], 'attrs': {'value': 0.125}, 'regions': []}, {'op': 'sub', 'results': [{'name': 'e82', 'type': {'dtype': 'float16', 'shape': (16, 16)}}], 'operands': ['e80', 'e81'], 'attrs': {}, 'regions': []}], 'returns': ['e82']}]}, {'op': 'call', 'results': [{'name': 'e84', 'type': {'dtype': 'float16', 'shape': (16, 16)}}, {'name': 'e85', 'type': {'dtype': 'float32', 'shape': ()}}], 'operands': ['e83', 'e63'], 'attrs': {'callee': 'mixed_helper'}, 'regions': []}, {'op': 'constant', 'results': [{'name': 'e86', 'type': {'dtype': 'float16', 'shape': (16, 16)}}], 'operands': [], 'attrs': {'value': 0.125}, 'regions': []}, {'op': 'mul', 'results': [{'name': 'e87', 'type': {'dtype': 'float16', 'shape': (16, 16)}}], 'operands': ['e84', 'e86'], 'attrs': {}, 'regions': []}, {'op': 'add', 'results': [{'name': 'e88', 'type': {'dtype': 'float32', 'shape': ()}}], 'operands': ['e75', 'e85'], 'attrs': {}, 'regions': []}, {'op': 'constant', 'results': [{'name': 'e89', 'type': {'dtype': 'int32', 'shape': ()}}], 'operands': [], 'attrs': {'value': 1}, 'regions': []}, {'op': 'add', 'results': [{'name': 'e90', 'type': {'dtype': 'int32', 'shape': ()}}], 'operands': ['e73', 'e89'], 'attrs': {}, 'regions': []}, {'op': 'add', 'results': [{'name': 'e91', 'type': {'dtype': 'int32', 'shape': ()}}], 'operands': ['e76', 'e90'], 'attrs': {}, 'regions': []}], 'returns': ['e87', 'e88', 'e91']}]}, {'op': 'while', 'results': [{'name': 'e114', 'type': {'dtype': 'float16', 'shape': (16, 16)}}, {'name': 'e115', 'type': {'dtype': 'float32', 'shape': ()}}, {'name': 'e116', 'type': {'dtype': 'int32', 'shape': ()}}], 'operands': ['e66', 'e92', 'e93', 'e94'], 'attrs': {'max_steps': 4, 'pipelined': False}, 'regions': [{'arguments': [{'name': 'e95', 'type': {'dtype': 'int32', 'shape': ()}}, {'name': 'e96', 'type': {'dtype': 'float16', 'shape': (16, 16)}}, {'name': 'e97', 'type': {'dtype': 'float32', 'shape': ()}}, {'name': 'e98', 'type': {'dtype': 'int32', 'shape': ()}}], 'operations': [{'op': 'if', 'results': [{'name': 'e105', 'type': {'dtype': 'float16', 'shape': (16, 16)}}], 'operands': ['e72', 'e96'], 'attrs': {}, 'regions': [{'arguments': [{'name': 'e99', 'type': {'dtype': 'float16', 'shape': (16, 16)}}], 'operations': [{'op': 'constant', 'results': [{'name': 'e100', 'type': {'dtype': 'float16', 'shape': (16, 16)}}], 'operands': [], 'attrs': {'value': 0.125}, 'regions': []}, {'op': 'add', 'results': [{'name': 'e101', 'type': {'dtype': 'float16', 'shape': (16, 16)}}], 'operands': ['e99', 'e100'], 'attrs': {}, 'regions': []}], 'returns': ['e101']}, {'arguments': [{'name': 'e102', 'type': {'dtype': 'float16', 'shape': (16, 16)}}], 'operations': [{'op': 'constant', 'results': [{'name': 'e103', 'type': {'dtype': 'float16', 'shape': (16, 16)}}], 'operands': [], 'attrs': {'value': 0.125}, 'regions': []}, {'op': 'sub', 'results': [{'name': 'e104', 'type': {'dtype': 'float16', 'shape': (16, 16)}}], 'operands': ['e102', 'e103'], 'attrs': {}, 'regions': []}], 'returns': ['e104']}]}, {'op': 'call', 'results': [{'name': 'e106', 'type': {'dtype': 'float16', 'shape': (16, 16)}}, {'name': 'e107', 'type': {'dtype': 'float32', 'shape': ()}}], 'operands': ['e105', 'e63'], 'attrs': {'callee': 'mixed_helper'}, 'regions': []}, {'op': 'constant', 'results': [{'name': 'e108', 'type': {'dtype': 'float16', 'shape': (16, 16)}}], 'operands': [], 'attrs': {'value': 0.125}, 'regions': []}, {'op': 'mul', 'results': [{'name': 'e109', 'type': {'dtype': 'float16', 'shape': (16, 16)}}], 'operands': ['e106', 'e108'], 'attrs': {}, 'regions': []}, {'op': 'add', 'results': [{'name': 'e110', 'type': {'dtype': 'float32', 'shape': ()}}], 'operands': ['e97', 'e107'], 'attrs': {}, 'regions': []}, {'op': 'constant', 'results': [{'name': 'e111', 'type': {'dtype': 'int32', 'shape': ()}}], 'operands': [], 'attrs': {'value': 1}, 'regions': []}, {'op': 'add', 'results': [{'name': 'e112', 'type': {'dtype': 'int32', 'shape': ()}}], 'operands': ['e95', 'e111'], 'attrs': {}, 'regions': []}, {'op': 'add', 'results': [{'name': 'e113', 'type': {'dtype': 'int32', 'shape': ()}}], 'operands': ['e98', 'e112'], 'attrs': {}, 'regions': []}], 'returns': ['e109', 'e110', 'e113']}]}, {'op': 'constant', 'results': [{'name': 'e117', 'type': {'dtype': 'float32', 'shape': ()}}], 'operands': [], 'attrs': {'value': -0.125}, 'regions': []}, {'op': 'broadcast', 'results': [{'name': 'e118', 'type': {'dtype': 'float32', 'shape': (16, 16)}}], 'operands': ['e117'], 'attrs': {}, 'regions': []}, {'op': 'constant', 'results': [{'name': 'e119', 'type': {'dtype': 'float32', 'shape': ()}}], 'operands': [], 'attrs': {'value': -0.25}, 'regions': []}, {'op': 'broadcast', 'results': [{'name': 'e120', 'type': {'dtype': 'float32', 'shape': (16, 16)}}], 'operands': ['e119'], 'attrs': {}, 'regions': []}, {'op': 'sub', 'results': [{'name': 'e121', 'type': {'dtype': 'float32', 'shape': (16, 16)}}], 'operands': ['e118', 'e120'], 'attrs': {}, 'regions': []}, {'op': 'index', 'results': [{'name': 'e122', 'type': {'dtype': 'int32', 'shape': (16, 16)}}], 'operands': [], 'attrs': {'shift': 28, 'reverse': True}, 'regions': []}, {'op': 'constant', 'results': [{'name': 'e123', 'type': {'dtype': 'bool', 'shape': (16, 16)}}], 'operands': [], 'attrs': {'value': True}, 'regions': []}, {'op': 'atomic_max', 'results': [], 'operands': ['e122', 'e123', 'e121'], 'attrs': {'buffer': 'mem8'}, 'regions': []}, {'op': 'sub', 'results': [{'name': 'e124', 'type': {'dtype': 'float16', 'shape': (16, 16)}}], 'operands': ['e114', 'e92'], 'attrs': {}, 'regions': []}, {'op': 'cast', 'results': [{'name': 'e125', 'type': {'dtype': 'float32', 'shape': (16, 16)}}], 'operands': ['e124'], 'attrs': {}, 'regions': []}, {'op': 'index', 'results': [{'name': 'e126', 'type': {'dtype': 'int32', 'shape': (16, 16)}}], 'operands': [], 'attrs': {'shift': 194, 'reverse': True}, 'regions': []}, {'op': 'constant', 'results': [{'name': 'e127', 'type': {'dtype': 'bool', 'shape': (16, 16)}}], 'operands': [], 'attrs': {'value': True}, 'regions': []}, {'op': 'load', 'results': [{'name': 'e128', 'type': {'dtype': 'float32', 'shape': (16, 16)}}], 'operands': ['e126', 'e127'], 'attrs': {'buffer': 'mem9'}, 'regions': []}, {'op': 'sub', 'results': [{'name': 'e129', 'type': {'dtype': 'float32', 'shape': (16, 16)}}], 'operands': ['e125', 'e128'], 'attrs': {}, 'regions': []}, {'op': 'cast', 'results': [{'name': 'e130', 'type': {'dtype': 'float16', 'shape': (16, 16)}}], 'operands': ['e129'], 'attrs': {}, 'regions': []}, {'op': 'add', 'results': [{'name': 'e131', 'type': {'dtype': 'float16', 'shape': (16, 16)}}], 'operands': ['e130', 'e49'], 'attrs': {}, 'regions': []}, {'op': 'reduce', 'results': [{'name': 'e132', 'type': {'dtype': 'float32', 'shape': (16,)}}], 'operands': ['e131'], 'attrs': {'axis': 0, 'kind': 'max'}, 'regions': []}], 'returns': ['e131', 'e16', 'e22', 'e115', 'e116', 'e132']}, 'buffers': [{'name': 'mem0', 'dtype': 'float16', 'size': 512, 'role': 'input', 'base': None, 'offset': 0, 'stride': 1}, {'name': 'mem1', 'dtype': 'float16', 'size': 512, 'role': 'input', 'base': None, 'offset': 0, 'stride': 1}, {'name': 'mem2', 'dtype': 'int32', 'size': 512, 'role': 'input', 'base': None, 'offset': 0, 'stride': 1}, {'name': 'mem3', 'dtype': 'float16', 'size': 1028, 'role': 'scratch', 'base': None, 'offset': 0, 'stride': 1}, {'name': 'mem4', 'dtype': 'float16', 'size': 512, 'role': 'scratch', 'base': 'mem3', 'offset': 1, 'stride': 2}, {'name': 'mem5', 'dtype': 'float16', 'size': 512, 'role': 'scratch', 'base': 'mem3', 'offset': 2, 'stride': 1}, {'name': 'mem6', 'dtype': 'float16', 'size': 256, 'role': 'input', 'base': None, 'offset': 0, 'stride': 1}, {'name': 'mem7', 'dtype': 'int32', 'size': 1, 'role': 'input', 'base': None, 'offset': 0, 'stride': 1}, {'name': 'mem8', 'dtype': 'float32', 'size': 256, 'role': 'scratch', 'base': None, 'offset': 0, 'stride': 1}, {'name': 'mem9', 'dtype': 'float32', 'size': 256, 'role': 'input', 'base': None, 'offset': 0, 'stride': 1}], 'functions': [{'name': 'mixed_helper', 'body': {'arguments': [{'name': 'e55', 'type': {'dtype': 'float16', 'shape': (16, 16)}}, {'name': 'e56', 'type': {'dtype': 'float32', 'shape': (16,)}}], 'operations': [{'op': 'reduce', 'results': [{'name': 'e57', 'type': {'dtype': 'float32', 'shape': (16,)}}], 'operands': ['e55'], 'attrs': {'axis': 1, 'kind': 'sum'}, 'regions': []}, {'op': 'add', 'results': [{'name': 'e58', 'type': {'dtype': 'float32', 'shape': (16,)}}], 'operands': ['e57', 'e56'], 'attrs': {}, 'regions': []}, {'op': 'reduce', 'results': [{'name': 'e59', 'type': {'dtype': 'float32', 'shape': ()}}], 'operands': ['e58'], 'attrs': {'axis': 0, 'kind': 'sum'}, 'regions': []}, {'op': 'reshape', 'results': [{'name': 'e60', 'type': {'dtype': 'float32', 'shape': (16, 1)}}], 'operands': ['e58'], 'attrs': {}, 'regions': []}, {'op': 'cast', 'results': [{'name': 'e61', 'type': {'dtype': 'float16', 'shape': (16, 1)}}], 'operands': ['e60'], 'attrs': {}, 'regions': []}, {'op': 'add', 'results': [{'name': 'e62', 'type': {'dtype': 'float16', 'shape': (16, 16)}}], 'operands': ['e55', 'e61'], 'attrs': {}, 'regions': []}], 'returns': ['e62', 'e59']}}], 'observations': ['e3', 'e11', 'e28', 'e32', 'e41', 'e46', 'e21', 'e4'], 'blocks': 2, 'input_pattern': 'integer', 'runtime_cases': [(0, 1), (1, 511), (3, 512)], 'configuration_pair': True, 'observation_pair': True, 'pass_config_pair': False, 'fast_math_pair': False, 'precision_pair': True, 'identity_pair': True, 'family': 'mixed'}

REFERENCE_PROGRAMS = {'triton_8_prec': {'type': 'extended', 'version': 1, 'body': {'arguments': [], 'operations': [{'op': 'index', 'results': [{'name': 'e1', 'type': {'dtype': 'int32', 'shape': (16, 32)}}], 'operands': [], 'attrs': {'shift': 10, 'reverse': False}, 'regions': []}, {'op': 'constant', 'results': [{'name': 'e2', 'type': {'dtype': 'bool', 'shape': (16, 32)}}], 'operands': [], 'attrs': {'value': True}, 'regions': []}, {'op': 'load', 'results': [{'name': 'e3', 'type': {'dtype': 'float16', 'shape': (16, 32)}}], 'operands': ['e1', 'e2'], 'attrs': {'buffer': 'mem0'}, 'regions': []}, {'op': 'index', 'results': [{'name': 'e4', 'type': {'dtype': 'int32', 'shape': (16, 32)}}], 'operands': [], 'attrs': {'shift': 17, 'reverse': False}, 'regions': []}, {'op': 'constant', 'results': [{'name': 'e5', 'type': {'dtype': 'bool', 'shape': (16, 32)}}], 'operands': [], 'attrs': {'value': True}, 'regions': []}, {'op': 'load', 'results': [{'name': 'e6', 'type': {'dtype': 'float16', 'shape': (16, 32)}}], 'operands': ['e4', 'e5'], 'attrs': {'buffer': 'mem1'}, 'regions': []}, {'op': 'mul', 'results': [{'name': 'e7', 'type': {'dtype': 'float16', 'shape': (16, 32)}}], 'operands': ['e3', 'e6'], 'attrs': {}, 'regions': []}, {'op': 'cast', 'results': [{'name': 'e8', 'type': {'dtype': 'float32', 'shape': (16, 32)}}], 'operands': ['e7'], 'attrs': {}, 'regions': []}, {'op': 'cast', 'results': [{'name': 'e9', 'type': {'dtype': 'float16', 'shape': (16, 32)}}], 'operands': ['e8'], 'attrs': {}, 'regions': []}, {'op': 'constant', 'results': [{'name': 'e10', 'type': {'dtype': 'float16', 'shape': (16, 32)}}], 'operands': [], 'attrs': {'value': 0.5}, 'regions': []}, {'op': 'mul', 'results': [{'name': 'e11', 'type': {'dtype': 'float16', 'shape': (16, 32)}}], 'operands': ['e9', 'e10'], 'attrs': {}, 'regions': []}, {'op': 'index', 'results': [{'name': 'e12', 'type': {'dtype': 'int32', 'shape': (16, 32)}}], 'operands': [], 'attrs': {'shift': 71, 'reverse': False}, 'regions': []}, {'op': 'constant', 'results': [{'name': 'e13', 'type': {'dtype': 'bool', 'shape': (16, 32)}}], 'operands': [], 'attrs': {'value': True}, 'regions': []}, {'op': 'load', 'results': [{'name': 'e14', 'type': {'dtype': 'int32', 'shape': (16, 32)}}], 'operands': ['e12', 'e13'], 'attrs': {'buffer': 'mem2'}, 'regions': []}, {'op': 'constant', 'results': [{'name': 'e15', 'type': {'dtype': 'int32', 'shape': (16, 32)}}], 'operands': [], 'attrs': {'value': 3}, 'regions': []}, {'op': 'bitand', 'results': [{'name': 'e16', 'type': {'dtype': 'int32', 'shape': (16, 32)}}], 'operands': ['e14', 'e15'], 'attrs': {}, 'regions': []}, {'op': 'constant', 'results': [{'name': 'e17', 'type': {'dtype': 'int32', 'shape': (16, 32)}}], 'operands': [], 'attrs': {'value': 3}, 'regions': []}, {'op': 'lt', 'results': [{'name': 'e18', 'type': {'dtype': 'bool', 'shape': (16, 32)}}], 'operands': ['e16', 'e17'], 'attrs': {}, 'regions': []}, {'op': 'index', 'results': [{'name': 'e19', 'type': {'dtype': 'int32', 'shape': (16, 32)}}], 'operands': [], 'attrs': {'shift': 252, 'reverse': True}, 'regions': []}, {'op': 'parameter', 'results': [{'name': 'e20', 'type': {'dtype': 'int32', 'shape': ()}}], 'operands': [], 'attrs': {'name': 'limit'}, 'regions': []}, {'op': 'lt', 'results': [{'name': 'e21', 'type': {'dtype': 'bool', 'shape': (16, 32)}}], 'operands': ['e19', 'e20'], 'attrs': {}, 'regions': []}, {'op': 'and', 'results': [{'name': 'e22', 'type': {'dtype': 'bool', 'shape': (16, 32)}}], 'operands': ['e18', 'e21'], 'attrs': {}, 'regions': []}, {'op': 'constant', 'results': [{'name': 'e23', 'type': {'dtype': 'float16', 'shape': (16, 32)}}], 'operands': [], 'attrs': {'value': -0.125}, 'regions': []}, {'op': 'select', 'results': [{'name': 'e24', 'type': {'dtype': 'float16', 'shape': (16, 32)}}], 'operands': ['e22', 'e11', 'e23'], 'attrs': {}, 'regions': []}, {'op': 'index', 'results': [{'name': 'e25', 'type': {'dtype': 'int32', 'shape': (16, 32)}}], 'operands': [], 'attrs': {'shift': 305, 'reverse': True}, 'regions': []}, {'op': 'parameter', 'results': [{'name': 'e26', 'type': {'dtype': 'int32', 'shape': ()}}], 'operands': [], 'attrs': {'name': 'limit'}, 'regions': []}, {'op': 'lt', 'results': [{'name': 'e27', 'type': {'dtype': 'bool', 'shape': (16, 32)}}], 'operands': ['e25', 'e26'], 'attrs': {}, 'regions': []}, {'op': 'store', 'results': [], 'operands': ['e25', 'e27', 'e24'], 'attrs': {'buffer': 'mem4'}, 'regions': []}, {'op': 'barrier', 'results': [], 'operands': [], 'attrs': {}, 'regions': []}, {'op': 'load', 'results': [{'name': 'e28', 'type': {'dtype': 'float16', 'shape': (16, 32)}}], 'operands': ['e25', 'e27'], 'attrs': {'buffer': 'mem4'}, 'regions': []}, {'op': 'constant', 'results': [{'name': 'e29', 'type': {'dtype': 'float16', 'shape': (16, 32)}}], 'operands': [], 'attrs': {'value': 0.125}, 'regions': []}, {'op': 'add', 'results': [{'name': 'e30', 'type': {'dtype': 'float16', 'shape': (16, 32)}}], 'operands': ['e28', 'e29'], 'attrs': {}, 'regions': []}, {'op': 'barrier', 'results': [], 'operands': [], 'attrs': {}, 'regions': []}, {'op': 'store', 'results': [], 'operands': ['e25', 'e27', 'e30'], 'attrs': {'buffer': 'mem5'}, 'regions': []}, {'op': 'barrier', 'results': [], 'operands': [], 'attrs': {}, 'regions': []}, {'op': 'constant', 'results': [{'name': 'e31', 'type': {'dtype': 'bool', 'shape': (16, 32)}}], 'operands': [], 'attrs': {'value': True}, 'regions': []}, {'op': 'load', 'results': [{'name': 'e32', 'type': {'dtype': 'float16', 'shape': (16, 32)}}], 'operands': ['e25', 'e31'], 'attrs': {'buffer': 'mem4'}, 'regions': []}, {'op': 'constant', 'results': [{'name': 'e33', 'type': {'dtype': 'int32', 'shape': (16, 32)}}], 'operands': [], 'attrs': {'value': 3}, 'regions': []}, {'op': 'bitxor', 'results': [{'name': 'e34', 'type': {'dtype': 'int32', 'shape': (16, 32)}}], 'operands': ['e25', 'e33'], 'attrs': {}, 'regions': []}, {'op': 'load', 'results': [{'name': 'e35', 'type': {'dtype': 'float16', 'shape': (16, 32)}}], 'operands': ['e34', 'e27'], 'attrs': {'buffer': 'mem5'}, 'regions': []}, {'op': 'add', 'results': [{'name': 'e36', 'type': {'dtype': 'float16', 'shape': (16, 32)}}], 'operands': ['e32', 'e35'], 'attrs': {}, 'regions': []}, {'op': 'cast', 'results': [{'name': 'e37', 'type': {'dtype': 'float16', 'shape': (16, 32)}}], 'operands': ['e36'], 'attrs': {}, 'regions': []}, {'op': 'transpose', 'results': [{'name': 'e38', 'type': {'dtype': 'float16', 'shape': (32, 16)}}], 'operands': ['e37'], 'attrs': {}, 'regions': []}, {'op': 'slice', 'results': [{'name': 'e39', 'type': {'dtype': 'float16', 'shape': (16, 16)}}], 'operands': ['e38'], 'attrs': {'offsets': [8, 0]}, 'regions': []}, {'op': 'constant', 'results': [{'name': 'e40', 'type': {'dtype': 'float16', 'shape': (16, 16)}}], 'operands': [], 'attrs': {'value': 0.0}, 'regions': []}, {'op': 'matmul', 'results': [{'name': 'e41', 'type': {'dtype': 'float16', 'shape': (16, 16)}}], 'operands': ['e39', 'e39', 'e40'], 'attrs': {}, 'regions': []}, {'op': 'reduce', 'results': [{'name': 'e42', 'type': {'dtype': 'float32', 'shape': (16,)}}], 'operands': ['e41'], 'attrs': {'axis': 1, 'kind': 'max'}, 'regions': []}, {'op': 'reshape', 'results': [{'name': 'e43', 'type': {'dtype': 'float32', 'shape': (16, 1)}}], 'operands': ['e42'], 'attrs': {}, 'regions': []}, {'op': 'cast', 'results': [{'name': 'e41_prec', 'type': {'dtype': 'float32', 'shape': (16, 16)}}], 'operands': ['e41'], 'attrs': {}, 'regions': []}, {'op': 'add', 'results': [{'name': 'e44', 'type': {'dtype': 'float32', 'shape': (16, 16)}}], 'operands': ['e41_prec', 'e43'], 'attrs': {}, 'regions': []}, {'op': 'constant', 'results': [{'name': 'e45', 'type': {'dtype': 'float32', 'shape': (16, 16)}}], 'operands': [], 'attrs': {'value': 0.125}, 'regions': []}, {'op': 'mul', 'results': [{'name': 'e46', 'type': {'dtype': 'float32', 'shape': (16, 16)}}], 'operands': ['e44', 'e45'], 'attrs': {}, 'regions': []}, {'op': 'index', 'results': [{'name': 'e47', 'type': {'dtype': 'int32', 'shape': (16, 16)}}], 'operands': [], 'attrs': {'shift': 44, 'reverse': False}, 'regions': []}, {'op': 'constant', 'results': [{'name': 'e48', 'type': {'dtype': 'bool', 'shape': (16, 16)}}], 'operands': [], 'attrs': {'value': True}, 'regions': []}, {'op': 'load', 'results': [{'name': 'e49', 'type': {'dtype': 'float16', 'shape': (16, 16)}}], 'operands': ['e47', 'e48'], 'attrs': {'buffer': 'mem6'}, 'regions': []}, {'op': 'cast', 'results': [{'name': 'e50', 'type': {'dtype': 'float16', 'shape': (16, 16)}}], 'operands': ['e46'], 'attrs': {}, 'regions': []}, {'op': 'matmul', 'results': [{'name': 'e51', 'type': {'dtype': 'float16', 'shape': (16, 16)}}], 'operands': ['e50', 'e49', 'e40'], 'attrs': {}, 'regions': []}, {'op': 'reshape', 'results': [{'name': 'e52', 'type': {'dtype': 'float16', 'shape': (16, 16)}}], 'operands': ['e51'], 'attrs': {}, 'regions': []}, {'op': 'reshape', 'results': [{'name': 'e53', 'type': {'dtype': 'float16', 'shape': (16, 16)}}], 'operands': ['e52'], 'attrs': {}, 'regions': []}, {'op': 'cast', 'results': [{'name': 'e54', 'type': {'dtype': 'float16', 'shape': (16, 16)}}], 'operands': ['e53'], 'attrs': {}, 'regions': []}, {'op': 'cast', 'results': [{'name': 'e63', 'type': {'dtype': 'float32', 'shape': (16,)}}], 'operands': ['e42'], 'attrs': {}, 'regions': []}, {'op': 'call', 'results': [{'name': 'e64', 'type': {'dtype': 'float16', 'shape': (16, 16)}}, {'name': 'e65', 'type': {'dtype': 'float32', 'shape': ()}}], 'operands': ['e54', 'e63'], 'attrs': {'callee': 'mixed_helper'}, 'regions': []}, {'op': 'parameter', 'results': [{'name': 'e66', 'type': {'dtype': 'int32', 'shape': ()}}], 'operands': [], 'attrs': {'name': 'steps'}, 'regions': []}, {'op': 'constant', 'results': [{'name': 'e67', 'type': {'dtype': 'int32', 'shape': ()}}], 'operands': [], 'attrs': {'value': 0}, 'regions': []}, {'op': 'constant', 'results': [{'name': 'e68', 'type': {'dtype': 'int32', 'shape': ()}}], 'operands': [], 'attrs': {'value': 0}, 'regions': []}, {'op': 'constant', 'results': [{'name': 'e69', 'type': {'dtype': 'bool', 'shape': ()}}], 'operands': [], 'attrs': {'value': True}, 'regions': []}, {'op': 'load', 'results': [{'name': 'e70', 'type': {'dtype': 'int32', 'shape': ()}}], 'operands': ['e68', 'e69'], 'attrs': {'buffer': 'mem7'}, 'regions': []}, {'op': 'constant', 'results': [{'name': 'e71', 'type': {'dtype': 'int32', 'shape': ()}}], 'operands': [], 'attrs': {'value': 0}, 'regions': []}, {'op': 'lt', 'results': [{'name': 'e72', 'type': {'dtype': 'bool', 'shape': ()}}], 'operands': ['e70', 'e71'], 'attrs': {}, 'regions': []}, {'op': 'for', 'results': [{'name': 'e92', 'type': {'dtype': 'float16', 'shape': (16, 16)}}, {'name': 'e93', 'type': {'dtype': 'float32', 'shape': ()}}, {'name': 'e94', 'type': {'dtype': 'int32', 'shape': ()}}], 'operands': ['e66', 'e64', 'e65', 'e67'], 'attrs': {'max_steps': 4, 'pipelined': False}, 'regions': [{'arguments': [{'name': 'e73', 'type': {'dtype': 'int32', 'shape': ()}}, {'name': 'e74', 'type': {'dtype': 'float16', 'shape': (16, 16)}}, {'name': 'e75', 'type': {'dtype': 'float32', 'shape': ()}}, {'name': 'e76', 'type': {'dtype': 'int32', 'shape': ()}}], 'operations': [{'op': 'if', 'results': [{'name': 'e83', 'type': {'dtype': 'float16', 'shape': (16, 16)}}], 'operands': ['e72', 'e74'], 'attrs': {}, 'regions': [{'arguments': [{'name': 'e77', 'type': {'dtype': 'float16', 'shape': (16, 16)}}], 'operations': [{'op': 'constant', 'results': [{'name': 'e78', 'type': {'dtype': 'float16', 'shape': (16, 16)}}], 'operands': [], 'attrs': {'value': 0.125}, 'regions': []}, {'op': 'add', 'results': [{'name': 'e79', 'type': {'dtype': 'float16', 'shape': (16, 16)}}], 'operands': ['e77', 'e78'], 'attrs': {}, 'regions': []}], 'returns': ['e79']}, {'arguments': [{'name': 'e80', 'type': {'dtype': 'float16', 'shape': (16, 16)}}], 'operations': [{'op': 'constant', 'results': [{'name': 'e81', 'type': {'dtype': 'float16', 'shape': (16, 16)}}], 'operands': [], 'attrs': {'value': 0.125}, 'regions': []}, {'op': 'sub', 'results': [{'name': 'e82', 'type': {'dtype': 'float16', 'shape': (16, 16)}}], 'operands': ['e80', 'e81'], 'attrs': {}, 'regions': []}], 'returns': ['e82']}]}, {'op': 'call', 'results': [{'name': 'e84', 'type': {'dtype': 'float16', 'shape': (16, 16)}}, {'name': 'e85', 'type': {'dtype': 'float32', 'shape': ()}}], 'operands': ['e83', 'e63'], 'attrs': {'callee': 'mixed_helper'}, 'regions': []}, {'op': 'constant', 'results': [{'name': 'e86', 'type': {'dtype': 'float16', 'shape': (16, 16)}}], 'operands': [], 'attrs': {'value': 0.125}, 'regions': []}, {'op': 'mul', 'results': [{'name': 'e87', 'type': {'dtype': 'float16', 'shape': (16, 16)}}], 'operands': ['e84', 'e86'], 'attrs': {}, 'regions': []}, {'op': 'add', 'results': [{'name': 'e88', 'type': {'dtype': 'float32', 'shape': ()}}], 'operands': ['e75', 'e85'], 'attrs': {}, 'regions': []}, {'op': 'constant', 'results': [{'name': 'e89', 'type': {'dtype': 'int32', 'shape': ()}}], 'operands': [], 'attrs': {'value': 1}, 'regions': []}, {'op': 'add', 'results': [{'name': 'e90', 'type': {'dtype': 'int32', 'shape': ()}}], 'operands': ['e73', 'e89'], 'attrs': {}, 'regions': []}, {'op': 'add', 'results': [{'name': 'e91', 'type': {'dtype': 'int32', 'shape': ()}}], 'operands': ['e76', 'e90'], 'attrs': {}, 'regions': []}], 'returns': ['e87', 'e88', 'e91']}]}, {'op': 'while', 'results': [{'name': 'e114', 'type': {'dtype': 'float16', 'shape': (16, 16)}}, {'name': 'e115', 'type': {'dtype': 'float32', 'shape': ()}}, {'name': 'e116', 'type': {'dtype': 'int32', 'shape': ()}}], 'operands': ['e66', 'e92', 'e93', 'e94'], 'attrs': {'max_steps': 4, 'pipelined': False}, 'regions': [{'arguments': [{'name': 'e95', 'type': {'dtype': 'int32', 'shape': ()}}, {'name': 'e96', 'type': {'dtype': 'float16', 'shape': (16, 16)}}, {'name': 'e97', 'type': {'dtype': 'float32', 'shape': ()}}, {'name': 'e98', 'type': {'dtype': 'int32', 'shape': ()}}], 'operations': [{'op': 'if', 'results': [{'name': 'e105', 'type': {'dtype': 'float16', 'shape': (16, 16)}}], 'operands': ['e72', 'e96'], 'attrs': {}, 'regions': [{'arguments': [{'name': 'e99', 'type': {'dtype': 'float16', 'shape': (16, 16)}}], 'operations': [{'op': 'constant', 'results': [{'name': 'e100', 'type': {'dtype': 'float16', 'shape': (16, 16)}}], 'operands': [], 'attrs': {'value': 0.125}, 'regions': []}, {'op': 'add', 'results': [{'name': 'e101', 'type': {'dtype': 'float16', 'shape': (16, 16)}}], 'operands': ['e99', 'e100'], 'attrs': {}, 'regions': []}], 'returns': ['e101']}, {'arguments': [{'name': 'e102', 'type': {'dtype': 'float16', 'shape': (16, 16)}}], 'operations': [{'op': 'constant', 'results': [{'name': 'e103', 'type': {'dtype': 'float16', 'shape': (16, 16)}}], 'operands': [], 'attrs': {'value': 0.125}, 'regions': []}, {'op': 'sub', 'results': [{'name': 'e104', 'type': {'dtype': 'float16', 'shape': (16, 16)}}], 'operands': ['e102', 'e103'], 'attrs': {}, 'regions': []}], 'returns': ['e104']}]}, {'op': 'call', 'results': [{'name': 'e106', 'type': {'dtype': 'float16', 'shape': (16, 16)}}, {'name': 'e107', 'type': {'dtype': 'float32', 'shape': ()}}], 'operands': ['e105', 'e63'], 'attrs': {'callee': 'mixed_helper'}, 'regions': []}, {'op': 'constant', 'results': [{'name': 'e108', 'type': {'dtype': 'float16', 'shape': (16, 16)}}], 'operands': [], 'attrs': {'value': 0.125}, 'regions': []}, {'op': 'mul', 'results': [{'name': 'e109', 'type': {'dtype': 'float16', 'shape': (16, 16)}}], 'operands': ['e106', 'e108'], 'attrs': {}, 'regions': []}, {'op': 'add', 'results': [{'name': 'e110', 'type': {'dtype': 'float32', 'shape': ()}}], 'operands': ['e97', 'e107'], 'attrs': {}, 'regions': []}, {'op': 'constant', 'results': [{'name': 'e111', 'type': {'dtype': 'int32', 'shape': ()}}], 'operands': [], 'attrs': {'value': 1}, 'regions': []}, {'op': 'add', 'results': [{'name': 'e112', 'type': {'dtype': 'int32', 'shape': ()}}], 'operands': ['e95', 'e111'], 'attrs': {}, 'regions': []}, {'op': 'add', 'results': [{'name': 'e113', 'type': {'dtype': 'int32', 'shape': ()}}], 'operands': ['e98', 'e112'], 'attrs': {}, 'regions': []}], 'returns': ['e109', 'e110', 'e113']}]}, {'op': 'constant', 'results': [{'name': 'e117', 'type': {'dtype': 'float32', 'shape': ()}}], 'operands': [], 'attrs': {'value': -0.125}, 'regions': []}, {'op': 'broadcast', 'results': [{'name': 'e118', 'type': {'dtype': 'float32', 'shape': (16, 16)}}], 'operands': ['e117'], 'attrs': {}, 'regions': []}, {'op': 'constant', 'results': [{'name': 'e119', 'type': {'dtype': 'float32', 'shape': ()}}], 'operands': [], 'attrs': {'value': -0.25}, 'regions': []}, {'op': 'broadcast', 'results': [{'name': 'e120', 'type': {'dtype': 'float32', 'shape': (16, 16)}}], 'operands': ['e119'], 'attrs': {}, 'regions': []}, {'op': 'sub', 'results': [{'name': 'e121', 'type': {'dtype': 'float32', 'shape': (16, 16)}}], 'operands': ['e118', 'e120'], 'attrs': {}, 'regions': []}, {'op': 'index', 'results': [{'name': 'e122', 'type': {'dtype': 'int32', 'shape': (16, 16)}}], 'operands': [], 'attrs': {'shift': 28, 'reverse': True}, 'regions': []}, {'op': 'constant', 'results': [{'name': 'e123', 'type': {'dtype': 'bool', 'shape': (16, 16)}}], 'operands': [], 'attrs': {'value': True}, 'regions': []}, {'op': 'atomic_max', 'results': [], 'operands': ['e122', 'e123', 'e121'], 'attrs': {'buffer': 'mem8'}, 'regions': []}, {'op': 'sub', 'results': [{'name': 'e124', 'type': {'dtype': 'float16', 'shape': (16, 16)}}], 'operands': ['e114', 'e92'], 'attrs': {}, 'regions': []}, {'op': 'cast', 'results': [{'name': 'e125', 'type': {'dtype': 'float32', 'shape': (16, 16)}}], 'operands': ['e124'], 'attrs': {}, 'regions': []}, {'op': 'index', 'results': [{'name': 'e126', 'type': {'dtype': 'int32', 'shape': (16, 16)}}], 'operands': [], 'attrs': {'shift': 194, 'reverse': True}, 'regions': []}, {'op': 'constant', 'results': [{'name': 'e127', 'type': {'dtype': 'bool', 'shape': (16, 16)}}], 'operands': [], 'attrs': {'value': True}, 'regions': []}, {'op': 'load', 'results': [{'name': 'e128', 'type': {'dtype': 'float32', 'shape': (16, 16)}}], 'operands': ['e126', 'e127'], 'attrs': {'buffer': 'mem9'}, 'regions': []}, {'op': 'sub', 'results': [{'name': 'e129', 'type': {'dtype': 'float32', 'shape': (16, 16)}}], 'operands': ['e125', 'e128'], 'attrs': {}, 'regions': []}, {'op': 'cast', 'results': [{'name': 'e130', 'type': {'dtype': 'float16', 'shape': (16, 16)}}], 'operands': ['e129'], 'attrs': {}, 'regions': []}, {'op': 'add', 'results': [{'name': 'e131', 'type': {'dtype': 'float16', 'shape': (16, 16)}}], 'operands': ['e130', 'e49'], 'attrs': {}, 'regions': []}, {'op': 'reduce', 'results': [{'name': 'e132', 'type': {'dtype': 'float32', 'shape': (16,)}}], 'operands': ['e131'], 'attrs': {'axis': 0, 'kind': 'max'}, 'regions': []}], 'returns': ['e131', 'e16', 'e22', 'e115', 'e116', 'e132']}, 'buffers': [{'name': 'mem0', 'dtype': 'float16', 'size': 512, 'role': 'input', 'base': None, 'offset': 0, 'stride': 1}, {'name': 'mem1', 'dtype': 'float16', 'size': 512, 'role': 'input', 'base': None, 'offset': 0, 'stride': 1}, {'name': 'mem2', 'dtype': 'int32', 'size': 512, 'role': 'input', 'base': None, 'offset': 0, 'stride': 1}, {'name': 'mem3', 'dtype': 'float16', 'size': 1028, 'role': 'scratch', 'base': None, 'offset': 0, 'stride': 1}, {'name': 'mem4', 'dtype': 'float16', 'size': 512, 'role': 'scratch', 'base': 'mem3', 'offset': 1, 'stride': 2}, {'name': 'mem5', 'dtype': 'float16', 'size': 512, 'role': 'scratch', 'base': 'mem3', 'offset': 2, 'stride': 1}, {'name': 'mem6', 'dtype': 'float16', 'size': 256, 'role': 'input', 'base': None, 'offset': 0, 'stride': 1}, {'name': 'mem7', 'dtype': 'int32', 'size': 1, 'role': 'input', 'base': None, 'offset': 0, 'stride': 1}, {'name': 'mem8', 'dtype': 'float32', 'size': 256, 'role': 'scratch', 'base': None, 'offset': 0, 'stride': 1}, {'name': 'mem9', 'dtype': 'float32', 'size': 256, 'role': 'input', 'base': None, 'offset': 0, 'stride': 1}], 'functions': [{'name': 'mixed_helper', 'body': {'arguments': [{'name': 'e55', 'type': {'dtype': 'float16', 'shape': (16, 16)}}, {'name': 'e56', 'type': {'dtype': 'float32', 'shape': (16,)}}], 'operations': [{'op': 'reduce', 'results': [{'name': 'e57', 'type': {'dtype': 'float32', 'shape': (16,)}}], 'operands': ['e55'], 'attrs': {'axis': 1, 'kind': 'sum'}, 'regions': []}, {'op': 'add', 'results': [{'name': 'e58', 'type': {'dtype': 'float32', 'shape': (16,)}}], 'operands': ['e57', 'e56'], 'attrs': {}, 'regions': []}, {'op': 'reduce', 'results': [{'name': 'e59', 'type': {'dtype': 'float32', 'shape': ()}}], 'operands': ['e58'], 'attrs': {'axis': 0, 'kind': 'sum'}, 'regions': []}, {'op': 'reshape', 'results': [{'name': 'e60', 'type': {'dtype': 'float32', 'shape': (16, 1)}}], 'operands': ['e58'], 'attrs': {}, 'regions': []}, {'op': 'cast', 'results': [{'name': 'e61', 'type': {'dtype': 'float16', 'shape': (16, 1)}}], 'operands': ['e60'], 'attrs': {}, 'regions': []}, {'op': 'add', 'results': [{'name': 'e62', 'type': {'dtype': 'float16', 'shape': (16, 16)}}], 'operands': ['e55', 'e61'], 'attrs': {}, 'regions': []}], 'returns': ['e62', 'e59']}}], 'observations': ['e3', 'e11', 'e28', 'e32', 'e41', 'e46', 'e21', 'e4'], 'blocks': 2, 'input_pattern': 'integer', 'runtime_cases': [(0, 1), (1, 511), (3, 512)], 'configuration_pair': True, 'observation_pair': True, 'pass_config_pair': False, 'fast_math_pair': False, 'precision_pair': True, 'identity_pair': True, 'family': 'mixed'}, 'triton_9_prec': {'type': 'extended', 'version': 1, 'body': {'arguments': [], 'operations': [{'op': 'index', 'results': [{'name': 'e1', 'type': {'dtype': 'int32', 'shape': (16, 32)}}], 'operands': [], 'attrs': {'shift': 10, 'reverse': False}, 'regions': []}, {'op': 'constant', 'results': [{'name': 'e2', 'type': {'dtype': 'bool', 'shape': (16, 32)}}], 'operands': [], 'attrs': {'value': True}, 'regions': []}, {'op': 'load', 'results': [{'name': 'e3', 'type': {'dtype': 'float16', 'shape': (16, 32)}}], 'operands': ['e1', 'e2'], 'attrs': {'buffer': 'mem0'}, 'regions': []}, {'op': 'index', 'results': [{'name': 'e4', 'type': {'dtype': 'int32', 'shape': (16, 32)}}], 'operands': [], 'attrs': {'shift': 17, 'reverse': False}, 'regions': []}, {'op': 'constant', 'results': [{'name': 'e5', 'type': {'dtype': 'bool', 'shape': (16, 32)}}], 'operands': [], 'attrs': {'value': True}, 'regions': []}, {'op': 'load', 'results': [{'name': 'e6', 'type': {'dtype': 'float16', 'shape': (16, 32)}}], 'operands': ['e4', 'e5'], 'attrs': {'buffer': 'mem1'}, 'regions': []}, {'op': 'mul', 'results': [{'name': 'e7', 'type': {'dtype': 'float16', 'shape': (16, 32)}}], 'operands': ['e3', 'e6'], 'attrs': {}, 'regions': []}, {'op': 'cast', 'results': [{'name': 'e8', 'type': {'dtype': 'float32', 'shape': (16, 32)}}], 'operands': ['e7'], 'attrs': {}, 'regions': []}, {'op': 'cast', 'results': [{'name': 'e9', 'type': {'dtype': 'float16', 'shape': (16, 32)}}], 'operands': ['e8'], 'attrs': {}, 'regions': []}, {'op': 'constant', 'results': [{'name': 'e10', 'type': {'dtype': 'float16', 'shape': (16, 32)}}], 'operands': [], 'attrs': {'value': 0.5}, 'regions': []}, {'op': 'mul', 'results': [{'name': 'e11', 'type': {'dtype': 'float16', 'shape': (16, 32)}}], 'operands': ['e9', 'e10'], 'attrs': {}, 'regions': []}, {'op': 'index', 'results': [{'name': 'e12', 'type': {'dtype': 'int32', 'shape': (16, 32)}}], 'operands': [], 'attrs': {'shift': 71, 'reverse': False}, 'regions': []}, {'op': 'constant', 'results': [{'name': 'e13', 'type': {'dtype': 'bool', 'shape': (16, 32)}}], 'operands': [], 'attrs': {'value': True}, 'regions': []}, {'op': 'load', 'results': [{'name': 'e14', 'type': {'dtype': 'int32', 'shape': (16, 32)}}], 'operands': ['e12', 'e13'], 'attrs': {'buffer': 'mem2'}, 'regions': []}, {'op': 'constant', 'results': [{'name': 'e15', 'type': {'dtype': 'int32', 'shape': (16, 32)}}], 'operands': [], 'attrs': {'value': 3}, 'regions': []}, {'op': 'bitand', 'results': [{'name': 'e16', 'type': {'dtype': 'int32', 'shape': (16, 32)}}], 'operands': ['e14', 'e15'], 'attrs': {}, 'regions': []}, {'op': 'constant', 'results': [{'name': 'e17', 'type': {'dtype': 'int32', 'shape': (16, 32)}}], 'operands': [], 'attrs': {'value': 3}, 'regions': []}, {'op': 'lt', 'results': [{'name': 'e18', 'type': {'dtype': 'bool', 'shape': (16, 32)}}], 'operands': ['e16', 'e17'], 'attrs': {}, 'regions': []}, {'op': 'index', 'results': [{'name': 'e19', 'type': {'dtype': 'int32', 'shape': (16, 32)}}], 'operands': [], 'attrs': {'shift': 252, 'reverse': True}, 'regions': []}, {'op': 'parameter', 'results': [{'name': 'e20', 'type': {'dtype': 'int32', 'shape': ()}}], 'operands': [], 'attrs': {'name': 'limit'}, 'regions': []}, {'op': 'lt', 'results': [{'name': 'e21', 'type': {'dtype': 'bool', 'shape': (16, 32)}}], 'operands': ['e19', 'e20'], 'attrs': {}, 'regions': []}, {'op': 'and', 'results': [{'name': 'e22', 'type': {'dtype': 'bool', 'shape': (16, 32)}}], 'operands': ['e18', 'e21'], 'attrs': {}, 'regions': []}, {'op': 'constant', 'results': [{'name': 'e23', 'type': {'dtype': 'float16', 'shape': (16, 32)}}], 'operands': [], 'attrs': {'value': -0.125}, 'regions': []}, {'op': 'select', 'results': [{'name': 'e24', 'type': {'dtype': 'float16', 'shape': (16, 32)}}], 'operands': ['e22', 'e11', 'e23'], 'attrs': {}, 'regions': []}, {'op': 'index', 'results': [{'name': 'e25', 'type': {'dtype': 'int32', 'shape': (16, 32)}}], 'operands': [], 'attrs': {'shift': 305, 'reverse': True}, 'regions': []}, {'op': 'parameter', 'results': [{'name': 'e26', 'type': {'dtype': 'int32', 'shape': ()}}], 'operands': [], 'attrs': {'name': 'limit'}, 'regions': []}, {'op': 'lt', 'results': [{'name': 'e27', 'type': {'dtype': 'bool', 'shape': (16, 32)}}], 'operands': ['e25', 'e26'], 'attrs': {}, 'regions': []}, {'op': 'store', 'results': [], 'operands': ['e25', 'e27', 'e24'], 'attrs': {'buffer': 'mem4'}, 'regions': []}, {'op': 'barrier', 'results': [], 'operands': [], 'attrs': {}, 'regions': []}, {'op': 'load', 'results': [{'name': 'e28', 'type': {'dtype': 'float16', 'shape': (16, 32)}}], 'operands': ['e25', 'e27'], 'attrs': {'buffer': 'mem4'}, 'regions': []}, {'op': 'constant', 'results': [{'name': 'e29', 'type': {'dtype': 'float16', 'shape': (16, 32)}}], 'operands': [], 'attrs': {'value': 0.125}, 'regions': []}, {'op': 'add', 'results': [{'name': 'e30', 'type': {'dtype': 'float16', 'shape': (16, 32)}}], 'operands': ['e28', 'e29'], 'attrs': {}, 'regions': []}, {'op': 'barrier', 'results': [], 'operands': [], 'attrs': {}, 'regions': []}, {'op': 'store', 'results': [], 'operands': ['e25', 'e27', 'e30'], 'attrs': {'buffer': 'mem5'}, 'regions': []}, {'op': 'barrier', 'results': [], 'operands': [], 'attrs': {}, 'regions': []}, {'op': 'constant', 'results': [{'name': 'e31', 'type': {'dtype': 'bool', 'shape': (16, 32)}}], 'operands': [], 'attrs': {'value': True}, 'regions': []}, {'op': 'load', 'results': [{'name': 'e32', 'type': {'dtype': 'float16', 'shape': (16, 32)}}], 'operands': ['e25', 'e31'], 'attrs': {'buffer': 'mem4'}, 'regions': []}, {'op': 'constant', 'results': [{'name': 'e33', 'type': {'dtype': 'int32', 'shape': (16, 32)}}], 'operands': [], 'attrs': {'value': 3}, 'regions': []}, {'op': 'bitxor', 'results': [{'name': 'e34', 'type': {'dtype': 'int32', 'shape': (16, 32)}}], 'operands': ['e25', 'e33'], 'attrs': {}, 'regions': []}, {'op': 'load', 'results': [{'name': 'e35', 'type': {'dtype': 'float16', 'shape': (16, 32)}}], 'operands': ['e34', 'e27'], 'attrs': {'buffer': 'mem5'}, 'regions': []}, {'op': 'add', 'results': [{'name': 'e36', 'type': {'dtype': 'float16', 'shape': (16, 32)}}], 'operands': ['e32', 'e35'], 'attrs': {}, 'regions': []}, {'op': 'cast', 'results': [{'name': 'e37', 'type': {'dtype': 'float16', 'shape': (16, 32)}}], 'operands': ['e36'], 'attrs': {}, 'regions': []}, {'op': 'transpose', 'results': [{'name': 'e38', 'type': {'dtype': 'float16', 'shape': (32, 16)}}], 'operands': ['e37'], 'attrs': {}, 'regions': []}, {'op': 'slice', 'results': [{'name': 'e39', 'type': {'dtype': 'float16', 'shape': (16, 16)}}], 'operands': ['e38'], 'attrs': {'offsets': [8, 0]}, 'regions': []}, {'op': 'constant', 'results': [{'name': 'e40', 'type': {'dtype': 'float16', 'shape': (16, 16)}}], 'operands': [], 'attrs': {'value': 0.0}, 'regions': []}, {'op': 'matmul', 'results': [{'name': 'e41', 'type': {'dtype': 'float16', 'shape': (16, 16)}}], 'operands': ['e39', 'e39', 'e40'], 'attrs': {}, 'regions': []}, {'op': 'reduce', 'results': [{'name': 'e42', 'type': {'dtype': 'float32', 'shape': (16,)}}], 'operands': ['e41'], 'attrs': {'axis': 1, 'kind': 'max'}, 'regions': []}, {'op': 'reshape', 'results': [{'name': 'e43', 'type': {'dtype': 'float32', 'shape': (16, 1)}}], 'operands': ['e42'], 'attrs': {}, 'regions': []}, {'op': 'cast', 'results': [{'name': 'e41_prec', 'type': {'dtype': 'float32', 'shape': (16, 16)}}], 'operands': ['e41'], 'attrs': {}, 'regions': []}, {'op': 'add', 'results': [{'name': 'e44', 'type': {'dtype': 'float32', 'shape': (16, 16)}}], 'operands': ['e41_prec', 'e43'], 'attrs': {}, 'regions': []}, {'op': 'constant', 'results': [{'name': 'e45', 'type': {'dtype': 'float32', 'shape': (16, 16)}}], 'operands': [], 'attrs': {'value': 0.125}, 'regions': []}, {'op': 'mul', 'results': [{'name': 'e46', 'type': {'dtype': 'float32', 'shape': (16, 16)}}], 'operands': ['e44', 'e45'], 'attrs': {}, 'regions': []}, {'op': 'index', 'results': [{'name': 'e47', 'type': {'dtype': 'int32', 'shape': (16, 16)}}], 'operands': [], 'attrs': {'shift': 44, 'reverse': False}, 'regions': []}, {'op': 'constant', 'results': [{'name': 'e48', 'type': {'dtype': 'bool', 'shape': (16, 16)}}], 'operands': [], 'attrs': {'value': True}, 'regions': []}, {'op': 'load', 'results': [{'name': 'e49', 'type': {'dtype': 'float16', 'shape': (16, 16)}}], 'operands': ['e47', 'e48'], 'attrs': {'buffer': 'mem6'}, 'regions': []}, {'op': 'cast', 'results': [{'name': 'e50', 'type': {'dtype': 'float16', 'shape': (16, 16)}}], 'operands': ['e46'], 'attrs': {}, 'regions': []}, {'op': 'matmul', 'results': [{'name': 'e51', 'type': {'dtype': 'float16', 'shape': (16, 16)}}], 'operands': ['e50', 'e49', 'e40'], 'attrs': {}, 'regions': []}, {'op': 'reshape', 'results': [{'name': 'e52', 'type': {'dtype': 'float16', 'shape': (16, 16)}}], 'operands': ['e51'], 'attrs': {}, 'regions': []}, {'op': 'reshape', 'results': [{'name': 'e53', 'type': {'dtype': 'float16', 'shape': (16, 16)}}], 'operands': ['e52'], 'attrs': {}, 'regions': []}, {'op': 'cast', 'results': [{'name': 'e54', 'type': {'dtype': 'float16', 'shape': (16, 16)}}], 'operands': ['e53'], 'attrs': {}, 'regions': []}, {'op': 'cast', 'results': [{'name': 'e63', 'type': {'dtype': 'float32', 'shape': (16,)}}], 'operands': ['e42'], 'attrs': {}, 'regions': []}, {'op': 'call', 'results': [{'name': 'e64', 'type': {'dtype': 'float16', 'shape': (16, 16)}}, {'name': 'e65', 'type': {'dtype': 'float32', 'shape': ()}}], 'operands': ['e54', 'e63'], 'attrs': {'callee': 'mixed_helper'}, 'regions': []}, {'op': 'parameter', 'results': [{'name': 'e66', 'type': {'dtype': 'int32', 'shape': ()}}], 'operands': [], 'attrs': {'name': 'steps'}, 'regions': []}, {'op': 'constant', 'results': [{'name': 'e67', 'type': {'dtype': 'int32', 'shape': ()}}], 'operands': [], 'attrs': {'value': 0}, 'regions': []}, {'op': 'constant', 'results': [{'name': 'e68', 'type': {'dtype': 'int32', 'shape': ()}}], 'operands': [], 'attrs': {'value': 0}, 'regions': []}, {'op': 'constant', 'results': [{'name': 'e69', 'type': {'dtype': 'bool', 'shape': ()}}], 'operands': [], 'attrs': {'value': True}, 'regions': []}, {'op': 'load', 'results': [{'name': 'e70', 'type': {'dtype': 'int32', 'shape': ()}}], 'operands': ['e68', 'e69'], 'attrs': {'buffer': 'mem7'}, 'regions': []}, {'op': 'constant', 'results': [{'name': 'e71', 'type': {'dtype': 'int32', 'shape': ()}}], 'operands': [], 'attrs': {'value': 0}, 'regions': []}, {'op': 'lt', 'results': [{'name': 'e72', 'type': {'dtype': 'bool', 'shape': ()}}], 'operands': ['e70', 'e71'], 'attrs': {}, 'regions': []}, {'op': 'for', 'results': [{'name': 'e92', 'type': {'dtype': 'float16', 'shape': (16, 16)}}, {'name': 'e93', 'type': {'dtype': 'float32', 'shape': ()}}, {'name': 'e94', 'type': {'dtype': 'int32', 'shape': ()}}], 'operands': ['e66', 'e64', 'e65', 'e67'], 'attrs': {'max_steps': 4, 'pipelined': False}, 'regions': [{'arguments': [{'name': 'e73', 'type': {'dtype': 'int32', 'shape': ()}}, {'name': 'e74', 'type': {'dtype': 'float16', 'shape': (16, 16)}}, {'name': 'e75', 'type': {'dtype': 'float32', 'shape': ()}}, {'name': 'e76', 'type': {'dtype': 'int32', 'shape': ()}}], 'operations': [{'op': 'if', 'results': [{'name': 'e83', 'type': {'dtype': 'float16', 'shape': (16, 16)}}], 'operands': ['e72', 'e74'], 'attrs': {}, 'regions': [{'arguments': [{'name': 'e77', 'type': {'dtype': 'float16', 'shape': (16, 16)}}], 'operations': [{'op': 'constant', 'results': [{'name': 'e78', 'type': {'dtype': 'float16', 'shape': (16, 16)}}], 'operands': [], 'attrs': {'value': 0.125}, 'regions': []}, {'op': 'add', 'results': [{'name': 'e79', 'type': {'dtype': 'float16', 'shape': (16, 16)}}], 'operands': ['e77', 'e78'], 'attrs': {}, 'regions': []}], 'returns': ['e79']}, {'arguments': [{'name': 'e80', 'type': {'dtype': 'float16', 'shape': (16, 16)}}], 'operations': [{'op': 'constant', 'results': [{'name': 'e81', 'type': {'dtype': 'float16', 'shape': (16, 16)}}], 'operands': [], 'attrs': {'value': 0.125}, 'regions': []}, {'op': 'sub', 'results': [{'name': 'e82', 'type': {'dtype': 'float16', 'shape': (16, 16)}}], 'operands': ['e80', 'e81'], 'attrs': {}, 'regions': []}], 'returns': ['e82']}]}, {'op': 'call', 'results': [{'name': 'e84', 'type': {'dtype': 'float16', 'shape': (16, 16)}}, {'name': 'e85', 'type': {'dtype': 'float32', 'shape': ()}}], 'operands': ['e83', 'e63'], 'attrs': {'callee': 'mixed_helper'}, 'regions': []}, {'op': 'constant', 'results': [{'name': 'e86', 'type': {'dtype': 'float16', 'shape': (16, 16)}}], 'operands': [], 'attrs': {'value': 0.125}, 'regions': []}, {'op': 'mul', 'results': [{'name': 'e87', 'type': {'dtype': 'float16', 'shape': (16, 16)}}], 'operands': ['e84', 'e86'], 'attrs': {}, 'regions': []}, {'op': 'add', 'results': [{'name': 'e88', 'type': {'dtype': 'float32', 'shape': ()}}], 'operands': ['e75', 'e85'], 'attrs': {}, 'regions': []}, {'op': 'constant', 'results': [{'name': 'e89', 'type': {'dtype': 'int32', 'shape': ()}}], 'operands': [], 'attrs': {'value': 1}, 'regions': []}, {'op': 'add', 'results': [{'name': 'e90', 'type': {'dtype': 'int32', 'shape': ()}}], 'operands': ['e73', 'e89'], 'attrs': {}, 'regions': []}, {'op': 'add', 'results': [{'name': 'e91', 'type': {'dtype': 'int32', 'shape': ()}}], 'operands': ['e76', 'e90'], 'attrs': {}, 'regions': []}], 'returns': ['e87', 'e88', 'e91']}]}, {'op': 'while', 'results': [{'name': 'e114', 'type': {'dtype': 'float16', 'shape': (16, 16)}}, {'name': 'e115', 'type': {'dtype': 'float32', 'shape': ()}}, {'name': 'e116', 'type': {'dtype': 'int32', 'shape': ()}}], 'operands': ['e66', 'e92', 'e93', 'e94'], 'attrs': {'max_steps': 4, 'pipelined': False}, 'regions': [{'arguments': [{'name': 'e95', 'type': {'dtype': 'int32', 'shape': ()}}, {'name': 'e96', 'type': {'dtype': 'float16', 'shape': (16, 16)}}, {'name': 'e97', 'type': {'dtype': 'float32', 'shape': ()}}, {'name': 'e98', 'type': {'dtype': 'int32', 'shape': ()}}], 'operations': [{'op': 'if', 'results': [{'name': 'e105', 'type': {'dtype': 'float16', 'shape': (16, 16)}}], 'operands': ['e72', 'e96'], 'attrs': {}, 'regions': [{'arguments': [{'name': 'e99', 'type': {'dtype': 'float16', 'shape': (16, 16)}}], 'operations': [{'op': 'constant', 'results': [{'name': 'e100', 'type': {'dtype': 'float16', 'shape': (16, 16)}}], 'operands': [], 'attrs': {'value': 0.125}, 'regions': []}, {'op': 'add', 'results': [{'name': 'e101', 'type': {'dtype': 'float16', 'shape': (16, 16)}}], 'operands': ['e99', 'e100'], 'attrs': {}, 'regions': []}], 'returns': ['e101']}, {'arguments': [{'name': 'e102', 'type': {'dtype': 'float16', 'shape': (16, 16)}}], 'operations': [{'op': 'constant', 'results': [{'name': 'e103', 'type': {'dtype': 'float16', 'shape': (16, 16)}}], 'operands': [], 'attrs': {'value': 0.125}, 'regions': []}, {'op': 'sub', 'results': [{'name': 'e104', 'type': {'dtype': 'float16', 'shape': (16, 16)}}], 'operands': ['e102', 'e103'], 'attrs': {}, 'regions': []}], 'returns': ['e104']}]}, {'op': 'call', 'results': [{'name': 'e106', 'type': {'dtype': 'float16', 'shape': (16, 16)}}, {'name': 'e107', 'type': {'dtype': 'float32', 'shape': ()}}], 'operands': ['e105', 'e63'], 'attrs': {'callee': 'mixed_helper'}, 'regions': []}, {'op': 'constant', 'results': [{'name': 'e108', 'type': {'dtype': 'float16', 'shape': (16, 16)}}], 'operands': [], 'attrs': {'value': 0.125}, 'regions': []}, {'op': 'mul', 'results': [{'name': 'e109', 'type': {'dtype': 'float16', 'shape': (16, 16)}}], 'operands': ['e106', 'e108'], 'attrs': {}, 'regions': []}, {'op': 'add', 'results': [{'name': 'e110', 'type': {'dtype': 'float32', 'shape': ()}}], 'operands': ['e97', 'e107'], 'attrs': {}, 'regions': []}, {'op': 'constant', 'results': [{'name': 'e111', 'type': {'dtype': 'int32', 'shape': ()}}], 'operands': [], 'attrs': {'value': 1}, 'regions': []}, {'op': 'add', 'results': [{'name': 'e112', 'type': {'dtype': 'int32', 'shape': ()}}], 'operands': ['e95', 'e111'], 'attrs': {}, 'regions': []}, {'op': 'add', 'results': [{'name': 'e113', 'type': {'dtype': 'int32', 'shape': ()}}], 'operands': ['e98', 'e112'], 'attrs': {}, 'regions': []}], 'returns': ['e109', 'e110', 'e113']}]}, {'op': 'constant', 'results': [{'name': 'e117', 'type': {'dtype': 'float32', 'shape': ()}}], 'operands': [], 'attrs': {'value': -0.125}, 'regions': []}, {'op': 'broadcast', 'results': [{'name': 'e118', 'type': {'dtype': 'float32', 'shape': (16, 16)}}], 'operands': ['e117'], 'attrs': {}, 'regions': []}, {'op': 'constant', 'results': [{'name': 'e119', 'type': {'dtype': 'float32', 'shape': ()}}], 'operands': [], 'attrs': {'value': -0.25}, 'regions': []}, {'op': 'broadcast', 'results': [{'name': 'e120', 'type': {'dtype': 'float32', 'shape': (16, 16)}}], 'operands': ['e119'], 'attrs': {}, 'regions': []}, {'op': 'sub', 'results': [{'name': 'e121', 'type': {'dtype': 'float32', 'shape': (16, 16)}}], 'operands': ['e118', 'e120'], 'attrs': {}, 'regions': []}, {'op': 'index', 'results': [{'name': 'e122', 'type': {'dtype': 'int32', 'shape': (16, 16)}}], 'operands': [], 'attrs': {'shift': 28, 'reverse': True}, 'regions': []}, {'op': 'constant', 'results': [{'name': 'e123', 'type': {'dtype': 'bool', 'shape': (16, 16)}}], 'operands': [], 'attrs': {'value': True}, 'regions': []}, {'op': 'atomic_max', 'results': [], 'operands': ['e122', 'e123', 'e121'], 'attrs': {'buffer': 'mem8'}, 'regions': []}, {'op': 'sub', 'results': [{'name': 'e124', 'type': {'dtype': 'float16', 'shape': (16, 16)}}], 'operands': ['e114', 'e92'], 'attrs': {}, 'regions': []}, {'op': 'cast', 'results': [{'name': 'e125', 'type': {'dtype': 'float32', 'shape': (16, 16)}}], 'operands': ['e124'], 'attrs': {}, 'regions': []}, {'op': 'index', 'results': [{'name': 'e126', 'type': {'dtype': 'int32', 'shape': (16, 16)}}], 'operands': [], 'attrs': {'shift': 194, 'reverse': True}, 'regions': []}, {'op': 'constant', 'results': [{'name': 'e127', 'type': {'dtype': 'bool', 'shape': (16, 16)}}], 'operands': [], 'attrs': {'value': True}, 'regions': []}, {'op': 'load', 'results': [{'name': 'e128', 'type': {'dtype': 'float32', 'shape': (16, 16)}}], 'operands': ['e126', 'e127'], 'attrs': {'buffer': 'mem9'}, 'regions': []}, {'op': 'sub', 'results': [{'name': 'e129', 'type': {'dtype': 'float32', 'shape': (16, 16)}}], 'operands': ['e125', 'e128'], 'attrs': {}, 'regions': []}, {'op': 'cast', 'results': [{'name': 'e130', 'type': {'dtype': 'float16', 'shape': (16, 16)}}], 'operands': ['e129'], 'attrs': {}, 'regions': []}, {'op': 'add', 'results': [{'name': 'e131', 'type': {'dtype': 'float16', 'shape': (16, 16)}}], 'operands': ['e130', 'e49'], 'attrs': {}, 'regions': []}, {'op': 'reduce', 'results': [{'name': 'e132', 'type': {'dtype': 'float32', 'shape': (16,)}}], 'operands': ['e131'], 'attrs': {'axis': 0, 'kind': 'max'}, 'regions': []}], 'returns': ['e131', 'e16', 'e22', 'e115', 'e116', 'e132']}, 'buffers': [{'name': 'mem0', 'dtype': 'float16', 'size': 512, 'role': 'input', 'base': None, 'offset': 0, 'stride': 1}, {'name': 'mem1', 'dtype': 'float16', 'size': 512, 'role': 'input', 'base': None, 'offset': 0, 'stride': 1}, {'name': 'mem2', 'dtype': 'int32', 'size': 512, 'role': 'input', 'base': None, 'offset': 0, 'stride': 1}, {'name': 'mem3', 'dtype': 'float16', 'size': 1028, 'role': 'scratch', 'base': None, 'offset': 0, 'stride': 1}, {'name': 'mem4', 'dtype': 'float16', 'size': 512, 'role': 'scratch', 'base': 'mem3', 'offset': 1, 'stride': 2}, {'name': 'mem5', 'dtype': 'float16', 'size': 512, 'role': 'scratch', 'base': 'mem3', 'offset': 2, 'stride': 1}, {'name': 'mem6', 'dtype': 'float16', 'size': 256, 'role': 'input', 'base': None, 'offset': 0, 'stride': 1}, {'name': 'mem7', 'dtype': 'int32', 'size': 1, 'role': 'input', 'base': None, 'offset': 0, 'stride': 1}, {'name': 'mem8', 'dtype': 'float32', 'size': 256, 'role': 'scratch', 'base': None, 'offset': 0, 'stride': 1}, {'name': 'mem9', 'dtype': 'float32', 'size': 256, 'role': 'input', 'base': None, 'offset': 0, 'stride': 1}], 'functions': [{'name': 'mixed_helper', 'body': {'arguments': [{'name': 'e55', 'type': {'dtype': 'float16', 'shape': (16, 16)}}, {'name': 'e56', 'type': {'dtype': 'float32', 'shape': (16,)}}], 'operations': [{'op': 'reduce', 'results': [{'name': 'e57', 'type': {'dtype': 'float32', 'shape': (16,)}}], 'operands': ['e55'], 'attrs': {'axis': 1, 'kind': 'sum'}, 'regions': []}, {'op': 'add', 'results': [{'name': 'e58', 'type': {'dtype': 'float32', 'shape': (16,)}}], 'operands': ['e57', 'e56'], 'attrs': {}, 'regions': []}, {'op': 'reduce', 'results': [{'name': 'e59', 'type': {'dtype': 'float32', 'shape': ()}}], 'operands': ['e58'], 'attrs': {'axis': 0, 'kind': 'sum'}, 'regions': []}, {'op': 'reshape', 'results': [{'name': 'e60', 'type': {'dtype': 'float32', 'shape': (16, 1)}}], 'operands': ['e58'], 'attrs': {}, 'regions': []}, {'op': 'cast', 'results': [{'name': 'e61', 'type': {'dtype': 'float16', 'shape': (16, 1)}}], 'operands': ['e60'], 'attrs': {}, 'regions': []}, {'op': 'add', 'results': [{'name': 'e62', 'type': {'dtype': 'float16', 'shape': (16, 16)}}], 'operands': ['e55', 'e61'], 'attrs': {}, 'regions': []}], 'returns': ['e62', 'e59']}}], 'observations': ['e3', 'e11', 'e28', 'e32', 'e41', 'e46', 'e21', 'e4'], 'blocks': 2, 'input_pattern': 'integer', 'runtime_cases': [(0, 1), (1, 511), (3, 512)], 'configuration_pair': True, 'observation_pair': True, 'pass_config_pair': False, 'fast_math_pair': False, 'precision_pair': True, 'identity_pair': True, 'family': 'mixed'}}

if __name__ == '__main__':
    if os.environ.get('TILESMITH_COMPILE_ONLY', '0') == '1':
        prepare_extended(device_compile=False)
        extended_stage('complete', 'compile_only')
        print('COMPILE PASSED (no GPU execution)')
    else:
        run_extended(PROGRAM, prepare_extended, 0, 3, reference_programs=REFERENCE_PROGRAMS)
        extended_stage('complete', 'execute')
        print('ALL PASSED')
