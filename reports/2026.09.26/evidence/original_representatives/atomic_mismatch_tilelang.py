import os
import tilelang
import tilelang.language as T

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


@tilelang.jit
def extended_0_0():
    @T.prim_func
    def impl(mem0: T.Buffer((3, 544), "float32"), mem1: T.Buffer((3, 544), "float32"), mem2: T.Buffer((3, 1060), "float32"), mem5: T.Buffer((3, 544), "float32"), out_e30: T.Buffer((3, 544), "float16"), steps: T.int32, limit: T.int32):
        with T.Kernel(3, threads=128) as bid:
            e1 = T.alloc_fragment((32, 16), "int32")
            e2 = T.alloc_fragment((32, 16), "bool")
            e3 = T.alloc_fragment((32, 16), "float32")
            e4 = T.alloc_fragment((32, 16), "float32")
            e5 = T.alloc_fragment((32, 16), "float32")
            e6 = T.alloc_fragment((32, 16), "int32")
            e7 = T.alloc_fragment((32, 16), "bool")
            e8 = T.alloc_fragment((32, 16), "float32")
            e9 = T.alloc_fragment((32, 16), "float32")
            e10 = T.alloc_fragment((32, 16), "int32")
            e11 = T.alloc_fragment((1,), "int32")
            e12 = T.alloc_fragment((32, 16), "bool")
            e13 = T.alloc_fragment((32, 16), "float32")
            e14 = T.alloc_fragment((32, 16), "float32")
            e15 = T.alloc_fragment((32, 16), "float32")
            e16 = T.alloc_fragment((32, 16), "bool")
            e17 = T.alloc_fragment((32, 16), "float32")
            e18 = T.alloc_fragment((32, 16), "int32")
            e19 = T.alloc_fragment((32, 16), "int32")
            e20 = T.alloc_fragment((32, 16), "float32")
            e21 = T.alloc_fragment((32, 16), "float32")
            e22 = T.alloc_fragment((1,), "float32")
            e23 = T.alloc_fragment((32, 16), "float32")
            e24 = T.alloc_fragment((32, 16), "int32")
            e25 = T.alloc_fragment((32, 16), "int32")
            e26 = T.alloc_fragment((32, 16), "int32")
            e27 = T.alloc_fragment((32, 16), "bool")
            e28 = T.alloc_fragment((32, 16), "float32")
            e29 = T.alloc_fragment((32, 16), "float16")
            e30 = T.alloc_fragment((32, 16), "float16")
            for i, j in T.Parallel(32, 16):
                e1[i, j] = T.cast((((i * 16 + j) + 119) % 512), "int32")
            for i, j in T.Parallel(32, 16):
                e2[i, j] = T.cast(True, "bool")
            for i, j in T.Parallel(32, 16):
                e3[i, j] = T.cast(T.if_then_else((e2[i, j] and e1[i, j] >= 0 and e1[i, j] < 512), mem0[bid, 16 + e1[i, j] * 1], T.cast(0, "float32")), "float32")
            for i, j in T.Parallel(32, 16):
                e4[i, j] = T.cast((e3[i, j] + e3[i, j]), "float32")
            for i, j in T.Parallel(32, 16):
                e5[i, j] = T.cast((e4[i, j] * e4[i, j]), "float32")
            for i, j in T.Parallel(32, 16):
                e6[i, j] = T.cast((((i * 16 + j) + 400) % 512), "int32")
            for i, j in T.Parallel(32, 16):
                e7[i, j] = T.cast(True, "bool")
            for i, j in T.Parallel(32, 16):
                e8[i, j] = T.cast(T.if_then_else((e7[i, j] and e6[i, j] >= 0 and e6[i, j] < 512), mem1[bid, 16 + e6[i, j] * 1], T.cast(0, "float32")), "float32")
            for i, j in T.Parallel(32, 16):
                e9[i, j] = T.cast((e5[i, j] + e8[i, j]), "float32")
            for i, j in T.Parallel(32, 16):
                e10[i, j] = T.cast((((511 - (i * 16 + j)) + 163) % 512), "int32")
            e11[0] = T.cast(limit, "int32")
            for i, j in T.Parallel(32, 16):
                e12[i, j] = T.cast((e10[i, j] < e11[0]), "bool")
            for i, j in T.Parallel(32, 16):
                if (e12[i, j] and e10[i, j] >= 0 and e10[i, j] < 512):
                    mem2[bid, 17 + e10[i, j] * 2] = e9[i, j]
            T.sync_threads()
            for i, j in T.Parallel(32, 16):
                e13[i, j] = T.cast(T.if_then_else((e12[i, j] and e10[i, j] >= 0 and e10[i, j] < 512), mem2[bid, 17 + e10[i, j] * 2], T.cast(0, "float32")), "float32")
            for i, j in T.Parallel(32, 16):
                e14[i, j] = T.cast(0.125, "float32")
            for i, j in T.Parallel(32, 16):
                e15[i, j] = T.cast((e13[i, j] + e14[i, j]), "float32")
            T.sync_threads()
            for i, j in T.Parallel(32, 16):
                if (e12[i, j] and e10[i, j] >= 0 and e10[i, j] < 512):
                    mem2[bid, 18 + e10[i, j] * 1] = e15[i, j]
            T.sync_threads()
            for i, j in T.Parallel(32, 16):
                e16[i, j] = T.cast(True, "bool")
            for i, j in T.Parallel(32, 16):
                e17[i, j] = T.cast(T.if_then_else((e16[i, j] and e10[i, j] >= 0 and e10[i, j] < 512), mem2[bid, 17 + e10[i, j] * 2], T.cast(0, "float32")), "float32")
            for i, j in T.Parallel(32, 16):
                e18[i, j] = T.cast(3, "int32")
            for i, j in T.Parallel(32, 16):
                e19[i, j] = T.cast((e10[i, j] ^ e18[i, j]), "int32")
            for i, j in T.Parallel(32, 16):
                e20[i, j] = T.cast(T.if_then_else((e12[i, j] and e19[i, j] >= 0 and e19[i, j] < 512), mem2[bid, 18 + e19[i, j] * 1], T.cast(0, "float32")), "float32")
            for i, j in T.Parallel(32, 16):
                e21[i, j] = T.cast((e17[i, j] + e20[i, j]), "float32")
            e22[0] = T.cast(0, "float32")
            for i, j in T.Parallel(32, 16):
                e23[i, j] = T.cast(e22[0], "float32")
            for i, j in T.Parallel(32, 16):
                e24[i, j] = T.cast((((i * 16 + j) + 150) % 512), "int32")
            for i, j in T.Parallel(32, 16):
                e25[i, j] = T.cast(7, "int32")
            for i, j in T.Parallel(32, 16):
                e26[i, j] = T.cast((e24[i, j] & e25[i, j]), "int32")
            for i, j in T.Parallel(32, 16):
                e27[i, j] = T.cast(True, "bool")
            for i, j in T.Parallel(32, 16):
                if (e27[i, j] and e26[i, j] >= 0 and e26[i, j] < 512):
                    T.atomic_min(mem5[bid, 16 + e26[i, j] * 1], e23[i, j], memory_order="relaxed")
            for i, j in T.Parallel(32, 16):
                e28[i, j] = T.cast((e21[i, j] + e21[i, j]), "float32")
            for i, j in T.Parallel(32, 16):
                e29[i, j] = T.cast(e28[i, j], "float16")
            for i, j in T.Parallel(32, 16):
                e30[i, j] = T.cast((e29[i, j] * e29[i, j]), "float16")
            for i, j in T.Parallel(32, 16):
                out_e30[bid, 16 + (i * 16 + j)] = e30[i, j]
    return impl

@tilelang.jit
def extended_0_1():
    @T.prim_func
    def impl(mem0: T.Buffer((3, 544), "float32"), mem1: T.Buffer((3, 544), "float32"), mem2: T.Buffer((3, 1060), "float32"), mem5: T.Buffer((3, 544), "float32"), out_e30: T.Buffer((3, 544), "float16"), out_e3: T.Buffer((3, 544), "float32"), out_e13: T.Buffer((3, 544), "float32"), out_e17: T.Buffer((3, 544), "float32"), out_e27: T.Buffer((3, 544), "bool"), out_e7: T.Buffer((3, 544), "bool"), steps: T.int32, limit: T.int32):
        with T.Kernel(3, threads=128) as bid:
            e1 = T.alloc_fragment((32, 16), "int32")
            e2 = T.alloc_fragment((32, 16), "bool")
            e3 = T.alloc_fragment((32, 16), "float32")
            e4 = T.alloc_fragment((32, 16), "float32")
            e5 = T.alloc_fragment((32, 16), "float32")
            e6 = T.alloc_fragment((32, 16), "int32")
            e7 = T.alloc_fragment((32, 16), "bool")
            e8 = T.alloc_fragment((32, 16), "float32")
            e9 = T.alloc_fragment((32, 16), "float32")
            e10 = T.alloc_fragment((32, 16), "int32")
            e11 = T.alloc_fragment((1,), "int32")
            e12 = T.alloc_fragment((32, 16), "bool")
            e13 = T.alloc_fragment((32, 16), "float32")
            e14 = T.alloc_fragment((32, 16), "float32")
            e15 = T.alloc_fragment((32, 16), "float32")
            e16 = T.alloc_fragment((32, 16), "bool")
            e17 = T.alloc_fragment((32, 16), "float32")
            e18 = T.alloc_fragment((32, 16), "int32")
            e19 = T.alloc_fragment((32, 16), "int32")
            e20 = T.alloc_fragment((32, 16), "float32")
            e21 = T.alloc_fragment((32, 16), "float32")
            e22 = T.alloc_fragment((1,), "float32")
            e23 = T.alloc_fragment((32, 16), "float32")
            e24 = T.alloc_fragment((32, 16), "int32")
            e25 = T.alloc_fragment((32, 16), "int32")
            e26 = T.alloc_fragment((32, 16), "int32")
            e27 = T.alloc_fragment((32, 16), "bool")
            e28 = T.alloc_fragment((32, 16), "float32")
            e29 = T.alloc_fragment((32, 16), "float16")
            e30 = T.alloc_fragment((32, 16), "float16")
            for i, j in T.Parallel(32, 16):
                e1[i, j] = T.cast((((i * 16 + j) + 119) % 512), "int32")
            for i, j in T.Parallel(32, 16):
                e2[i, j] = T.cast(True, "bool")
            for i, j in T.Parallel(32, 16):
                e3[i, j] = T.cast(T.if_then_else((e2[i, j] and e1[i, j] >= 0 and e1[i, j] < 512), mem0[bid, 16 + e1[i, j] * 1], T.cast(0, "float32")), "float32")
            for i, j in T.Parallel(32, 16):
                e4[i, j] = T.cast((e3[i, j] + e3[i, j]), "float32")
            for i, j in T.Parallel(32, 16):
                e5[i, j] = T.cast((e4[i, j] * e4[i, j]), "float32")
            for i, j in T.Parallel(32, 16):
                e6[i, j] = T.cast((((i * 16 + j) + 400) % 512), "int32")
            for i, j in T.Parallel(32, 16):
                e7[i, j] = T.cast(True, "bool")
            for i, j in T.Parallel(32, 16):
                e8[i, j] = T.cast(T.if_then_else((e7[i, j] and e6[i, j] >= 0 and e6[i, j] < 512), mem1[bid, 16 + e6[i, j] * 1], T.cast(0, "float32")), "float32")
            for i, j in T.Parallel(32, 16):
                e9[i, j] = T.cast((e5[i, j] + e8[i, j]), "float32")
            for i, j in T.Parallel(32, 16):
                e10[i, j] = T.cast((((511 - (i * 16 + j)) + 163) % 512), "int32")
            e11[0] = T.cast(limit, "int32")
            for i, j in T.Parallel(32, 16):
                e12[i, j] = T.cast((e10[i, j] < e11[0]), "bool")
            for i, j in T.Parallel(32, 16):
                if (e12[i, j] and e10[i, j] >= 0 and e10[i, j] < 512):
                    mem2[bid, 17 + e10[i, j] * 2] = e9[i, j]
            T.sync_threads()
            for i, j in T.Parallel(32, 16):
                e13[i, j] = T.cast(T.if_then_else((e12[i, j] and e10[i, j] >= 0 and e10[i, j] < 512), mem2[bid, 17 + e10[i, j] * 2], T.cast(0, "float32")), "float32")
            for i, j in T.Parallel(32, 16):
                e14[i, j] = T.cast(0.125, "float32")
            for i, j in T.Parallel(32, 16):
                e15[i, j] = T.cast((e13[i, j] + e14[i, j]), "float32")
            T.sync_threads()
            for i, j in T.Parallel(32, 16):
                if (e12[i, j] and e10[i, j] >= 0 and e10[i, j] < 512):
                    mem2[bid, 18 + e10[i, j] * 1] = e15[i, j]
            T.sync_threads()
            for i, j in T.Parallel(32, 16):
                e16[i, j] = T.cast(True, "bool")
            for i, j in T.Parallel(32, 16):
                e17[i, j] = T.cast(T.if_then_else((e16[i, j] and e10[i, j] >= 0 and e10[i, j] < 512), mem2[bid, 17 + e10[i, j] * 2], T.cast(0, "float32")), "float32")
            for i, j in T.Parallel(32, 16):
                e18[i, j] = T.cast(3, "int32")
            for i, j in T.Parallel(32, 16):
                e19[i, j] = T.cast((e10[i, j] ^ e18[i, j]), "int32")
            for i, j in T.Parallel(32, 16):
                e20[i, j] = T.cast(T.if_then_else((e12[i, j] and e19[i, j] >= 0 and e19[i, j] < 512), mem2[bid, 18 + e19[i, j] * 1], T.cast(0, "float32")), "float32")
            for i, j in T.Parallel(32, 16):
                e21[i, j] = T.cast((e17[i, j] + e20[i, j]), "float32")
            e22[0] = T.cast(0, "float32")
            for i, j in T.Parallel(32, 16):
                e23[i, j] = T.cast(e22[0], "float32")
            for i, j in T.Parallel(32, 16):
                e24[i, j] = T.cast((((i * 16 + j) + 150) % 512), "int32")
            for i, j in T.Parallel(32, 16):
                e25[i, j] = T.cast(7, "int32")
            for i, j in T.Parallel(32, 16):
                e26[i, j] = T.cast((e24[i, j] & e25[i, j]), "int32")
            for i, j in T.Parallel(32, 16):
                e27[i, j] = T.cast(True, "bool")
            for i, j in T.Parallel(32, 16):
                if (e27[i, j] and e26[i, j] >= 0 and e26[i, j] < 512):
                    T.atomic_min(mem5[bid, 16 + e26[i, j] * 1], e23[i, j], memory_order="relaxed")
            for i, j in T.Parallel(32, 16):
                e28[i, j] = T.cast((e21[i, j] + e21[i, j]), "float32")
            for i, j in T.Parallel(32, 16):
                e29[i, j] = T.cast(e28[i, j], "float16")
            for i, j in T.Parallel(32, 16):
                e30[i, j] = T.cast((e29[i, j] * e29[i, j]), "float16")
            for i, j in T.Parallel(32, 16):
                out_e30[bid, 16 + (i * 16 + j)] = e30[i, j]
            for i, j in T.Parallel(32, 16):
                out_e3[bid, 16 + (i * 16 + j)] = e3[i, j]
            for i, j in T.Parallel(32, 16):
                out_e13[bid, 16 + (i * 16 + j)] = e13[i, j]
            for i, j in T.Parallel(32, 16):
                out_e17[bid, 16 + (i * 16 + j)] = e17[i, j]
            for i, j in T.Parallel(32, 16):
                out_e27[bid, 16 + (i * 16 + j)] = e27[i, j]
            for i, j in T.Parallel(32, 16):
                out_e7[bid, 16 + (i * 16 + j)] = e7[i, j]
    return impl

@tilelang.jit
def extended_1_0():
    @T.prim_func
    def impl(mem0: T.Buffer((3, 544), "float32"), mem1: T.Buffer((3, 544), "float32"), mem2: T.Buffer((3, 1060), "float32"), mem5: T.Buffer((3, 544), "float32"), out_e30: T.Buffer((3, 544), "float16"), steps: T.int32, limit: T.int32):
        with T.Kernel(3, threads=256) as bid:
            e1 = T.alloc_fragment((32, 16), "int32")
            e2 = T.alloc_fragment((32, 16), "bool")
            e3 = T.alloc_fragment((32, 16), "float32")
            e4 = T.alloc_fragment((32, 16), "float32")
            e5 = T.alloc_fragment((32, 16), "float32")
            e6 = T.alloc_fragment((32, 16), "int32")
            e7 = T.alloc_fragment((32, 16), "bool")
            e8 = T.alloc_fragment((32, 16), "float32")
            e9 = T.alloc_fragment((32, 16), "float32")
            e10 = T.alloc_fragment((32, 16), "int32")
            e11 = T.alloc_fragment((1,), "int32")
            e12 = T.alloc_fragment((32, 16), "bool")
            e13 = T.alloc_fragment((32, 16), "float32")
            e14 = T.alloc_fragment((32, 16), "float32")
            e15 = T.alloc_fragment((32, 16), "float32")
            e16 = T.alloc_fragment((32, 16), "bool")
            e17 = T.alloc_fragment((32, 16), "float32")
            e18 = T.alloc_fragment((32, 16), "int32")
            e19 = T.alloc_fragment((32, 16), "int32")
            e20 = T.alloc_fragment((32, 16), "float32")
            e21 = T.alloc_fragment((32, 16), "float32")
            e22 = T.alloc_fragment((1,), "float32")
            e23 = T.alloc_fragment((32, 16), "float32")
            e24 = T.alloc_fragment((32, 16), "int32")
            e25 = T.alloc_fragment((32, 16), "int32")
            e26 = T.alloc_fragment((32, 16), "int32")
            e27 = T.alloc_fragment((32, 16), "bool")
            e28 = T.alloc_fragment((32, 16), "float32")
            e29 = T.alloc_fragment((32, 16), "float16")
            e30 = T.alloc_fragment((32, 16), "float16")
            for i, j in T.Parallel(32, 16):
                e1[i, j] = T.cast((((i * 16 + j) + 119) % 512), "int32")
            for i, j in T.Parallel(32, 16):
                e2[i, j] = T.cast(True, "bool")
            for i, j in T.Parallel(32, 16):
                e3[i, j] = T.cast(T.if_then_else((e2[i, j] and e1[i, j] >= 0 and e1[i, j] < 512), mem0[bid, 16 + e1[i, j] * 1], T.cast(0, "float32")), "float32")
            for i, j in T.Parallel(32, 16):
                e4[i, j] = T.cast((e3[i, j] + e3[i, j]), "float32")
            for i, j in T.Parallel(32, 16):
                e5[i, j] = T.cast((e4[i, j] * e4[i, j]), "float32")
            for i, j in T.Parallel(32, 16):
                e6[i, j] = T.cast((((i * 16 + j) + 400) % 512), "int32")
            for i, j in T.Parallel(32, 16):
                e7[i, j] = T.cast(True, "bool")
            for i, j in T.Parallel(32, 16):
                e8[i, j] = T.cast(T.if_then_else((e7[i, j] and e6[i, j] >= 0 and e6[i, j] < 512), mem1[bid, 16 + e6[i, j] * 1], T.cast(0, "float32")), "float32")
            for i, j in T.Parallel(32, 16):
                e9[i, j] = T.cast((e5[i, j] + e8[i, j]), "float32")
            for i, j in T.Parallel(32, 16):
                e10[i, j] = T.cast((((511 - (i * 16 + j)) + 163) % 512), "int32")
            e11[0] = T.cast(limit, "int32")
            for i, j in T.Parallel(32, 16):
                e12[i, j] = T.cast((e10[i, j] < e11[0]), "bool")
            for i, j in T.Parallel(32, 16):
                if (e12[i, j] and e10[i, j] >= 0 and e10[i, j] < 512):
                    mem2[bid, 17 + e10[i, j] * 2] = e9[i, j]
            T.sync_threads()
            for i, j in T.Parallel(32, 16):
                e13[i, j] = T.cast(T.if_then_else((e12[i, j] and e10[i, j] >= 0 and e10[i, j] < 512), mem2[bid, 17 + e10[i, j] * 2], T.cast(0, "float32")), "float32")
            for i, j in T.Parallel(32, 16):
                e14[i, j] = T.cast(0.125, "float32")
            for i, j in T.Parallel(32, 16):
                e15[i, j] = T.cast((e13[i, j] + e14[i, j]), "float32")
            T.sync_threads()
            for i, j in T.Parallel(32, 16):
                if (e12[i, j] and e10[i, j] >= 0 and e10[i, j] < 512):
                    mem2[bid, 18 + e10[i, j] * 1] = e15[i, j]
            T.sync_threads()
            for i, j in T.Parallel(32, 16):
                e16[i, j] = T.cast(True, "bool")
            for i, j in T.Parallel(32, 16):
                e17[i, j] = T.cast(T.if_then_else((e16[i, j] and e10[i, j] >= 0 and e10[i, j] < 512), mem2[bid, 17 + e10[i, j] * 2], T.cast(0, "float32")), "float32")
            for i, j in T.Parallel(32, 16):
                e18[i, j] = T.cast(3, "int32")
            for i, j in T.Parallel(32, 16):
                e19[i, j] = T.cast((e10[i, j] ^ e18[i, j]), "int32")
            for i, j in T.Parallel(32, 16):
                e20[i, j] = T.cast(T.if_then_else((e12[i, j] and e19[i, j] >= 0 and e19[i, j] < 512), mem2[bid, 18 + e19[i, j] * 1], T.cast(0, "float32")), "float32")
            for i, j in T.Parallel(32, 16):
                e21[i, j] = T.cast((e17[i, j] + e20[i, j]), "float32")
            e22[0] = T.cast(0, "float32")
            for i, j in T.Parallel(32, 16):
                e23[i, j] = T.cast(e22[0], "float32")
            for i, j in T.Parallel(32, 16):
                e24[i, j] = T.cast((((i * 16 + j) + 150) % 512), "int32")
            for i, j in T.Parallel(32, 16):
                e25[i, j] = T.cast(7, "int32")
            for i, j in T.Parallel(32, 16):
                e26[i, j] = T.cast((e24[i, j] & e25[i, j]), "int32")
            for i, j in T.Parallel(32, 16):
                e27[i, j] = T.cast(True, "bool")
            for i, j in T.Parallel(32, 16):
                if (e27[i, j] and e26[i, j] >= 0 and e26[i, j] < 512):
                    T.atomic_min(mem5[bid, 16 + e26[i, j] * 1], e23[i, j], memory_order="relaxed")
            for i, j in T.Parallel(32, 16):
                e28[i, j] = T.cast((e21[i, j] + e21[i, j]), "float32")
            for i, j in T.Parallel(32, 16):
                e29[i, j] = T.cast(e28[i, j], "float16")
            for i, j in T.Parallel(32, 16):
                e30[i, j] = T.cast((e29[i, j] * e29[i, j]), "float16")
            for i, j in T.Parallel(32, 16):
                out_e30[bid, 16 + (i * 16 + j)] = e30[i, j]
    return impl

@tilelang.jit
def extended_1_1():
    @T.prim_func
    def impl(mem0: T.Buffer((3, 544), "float32"), mem1: T.Buffer((3, 544), "float32"), mem2: T.Buffer((3, 1060), "float32"), mem5: T.Buffer((3, 544), "float32"), out_e30: T.Buffer((3, 544), "float16"), out_e3: T.Buffer((3, 544), "float32"), out_e13: T.Buffer((3, 544), "float32"), out_e17: T.Buffer((3, 544), "float32"), out_e27: T.Buffer((3, 544), "bool"), out_e7: T.Buffer((3, 544), "bool"), steps: T.int32, limit: T.int32):
        with T.Kernel(3, threads=256) as bid:
            e1 = T.alloc_fragment((32, 16), "int32")
            e2 = T.alloc_fragment((32, 16), "bool")
            e3 = T.alloc_fragment((32, 16), "float32")
            e4 = T.alloc_fragment((32, 16), "float32")
            e5 = T.alloc_fragment((32, 16), "float32")
            e6 = T.alloc_fragment((32, 16), "int32")
            e7 = T.alloc_fragment((32, 16), "bool")
            e8 = T.alloc_fragment((32, 16), "float32")
            e9 = T.alloc_fragment((32, 16), "float32")
            e10 = T.alloc_fragment((32, 16), "int32")
            e11 = T.alloc_fragment((1,), "int32")
            e12 = T.alloc_fragment((32, 16), "bool")
            e13 = T.alloc_fragment((32, 16), "float32")
            e14 = T.alloc_fragment((32, 16), "float32")
            e15 = T.alloc_fragment((32, 16), "float32")
            e16 = T.alloc_fragment((32, 16), "bool")
            e17 = T.alloc_fragment((32, 16), "float32")
            e18 = T.alloc_fragment((32, 16), "int32")
            e19 = T.alloc_fragment((32, 16), "int32")
            e20 = T.alloc_fragment((32, 16), "float32")
            e21 = T.alloc_fragment((32, 16), "float32")
            e22 = T.alloc_fragment((1,), "float32")
            e23 = T.alloc_fragment((32, 16), "float32")
            e24 = T.alloc_fragment((32, 16), "int32")
            e25 = T.alloc_fragment((32, 16), "int32")
            e26 = T.alloc_fragment((32, 16), "int32")
            e27 = T.alloc_fragment((32, 16), "bool")
            e28 = T.alloc_fragment((32, 16), "float32")
            e29 = T.alloc_fragment((32, 16), "float16")
            e30 = T.alloc_fragment((32, 16), "float16")
            for i, j in T.Parallel(32, 16):
                e1[i, j] = T.cast((((i * 16 + j) + 119) % 512), "int32")
            for i, j in T.Parallel(32, 16):
                e2[i, j] = T.cast(True, "bool")
            for i, j in T.Parallel(32, 16):
                e3[i, j] = T.cast(T.if_then_else((e2[i, j] and e1[i, j] >= 0 and e1[i, j] < 512), mem0[bid, 16 + e1[i, j] * 1], T.cast(0, "float32")), "float32")
            for i, j in T.Parallel(32, 16):
                e4[i, j] = T.cast((e3[i, j] + e3[i, j]), "float32")
            for i, j in T.Parallel(32, 16):
                e5[i, j] = T.cast((e4[i, j] * e4[i, j]), "float32")
            for i, j in T.Parallel(32, 16):
                e6[i, j] = T.cast((((i * 16 + j) + 400) % 512), "int32")
            for i, j in T.Parallel(32, 16):
                e7[i, j] = T.cast(True, "bool")
            for i, j in T.Parallel(32, 16):
                e8[i, j] = T.cast(T.if_then_else((e7[i, j] and e6[i, j] >= 0 and e6[i, j] < 512), mem1[bid, 16 + e6[i, j] * 1], T.cast(0, "float32")), "float32")
            for i, j in T.Parallel(32, 16):
                e9[i, j] = T.cast((e5[i, j] + e8[i, j]), "float32")
            for i, j in T.Parallel(32, 16):
                e10[i, j] = T.cast((((511 - (i * 16 + j)) + 163) % 512), "int32")
            e11[0] = T.cast(limit, "int32")
            for i, j in T.Parallel(32, 16):
                e12[i, j] = T.cast((e10[i, j] < e11[0]), "bool")
            for i, j in T.Parallel(32, 16):
                if (e12[i, j] and e10[i, j] >= 0 and e10[i, j] < 512):
                    mem2[bid, 17 + e10[i, j] * 2] = e9[i, j]
            T.sync_threads()
            for i, j in T.Parallel(32, 16):
                e13[i, j] = T.cast(T.if_then_else((e12[i, j] and e10[i, j] >= 0 and e10[i, j] < 512), mem2[bid, 17 + e10[i, j] * 2], T.cast(0, "float32")), "float32")
            for i, j in T.Parallel(32, 16):
                e14[i, j] = T.cast(0.125, "float32")
            for i, j in T.Parallel(32, 16):
                e15[i, j] = T.cast((e13[i, j] + e14[i, j]), "float32")
            T.sync_threads()
            for i, j in T.Parallel(32, 16):
                if (e12[i, j] and e10[i, j] >= 0 and e10[i, j] < 512):
                    mem2[bid, 18 + e10[i, j] * 1] = e15[i, j]
            T.sync_threads()
            for i, j in T.Parallel(32, 16):
                e16[i, j] = T.cast(True, "bool")
            for i, j in T.Parallel(32, 16):
                e17[i, j] = T.cast(T.if_then_else((e16[i, j] and e10[i, j] >= 0 and e10[i, j] < 512), mem2[bid, 17 + e10[i, j] * 2], T.cast(0, "float32")), "float32")
            for i, j in T.Parallel(32, 16):
                e18[i, j] = T.cast(3, "int32")
            for i, j in T.Parallel(32, 16):
                e19[i, j] = T.cast((e10[i, j] ^ e18[i, j]), "int32")
            for i, j in T.Parallel(32, 16):
                e20[i, j] = T.cast(T.if_then_else((e12[i, j] and e19[i, j] >= 0 and e19[i, j] < 512), mem2[bid, 18 + e19[i, j] * 1], T.cast(0, "float32")), "float32")
            for i, j in T.Parallel(32, 16):
                e21[i, j] = T.cast((e17[i, j] + e20[i, j]), "float32")
            e22[0] = T.cast(0, "float32")
            for i, j in T.Parallel(32, 16):
                e23[i, j] = T.cast(e22[0], "float32")
            for i, j in T.Parallel(32, 16):
                e24[i, j] = T.cast((((i * 16 + j) + 150) % 512), "int32")
            for i, j in T.Parallel(32, 16):
                e25[i, j] = T.cast(7, "int32")
            for i, j in T.Parallel(32, 16):
                e26[i, j] = T.cast((e24[i, j] & e25[i, j]), "int32")
            for i, j in T.Parallel(32, 16):
                e27[i, j] = T.cast(True, "bool")
            for i, j in T.Parallel(32, 16):
                if (e27[i, j] and e26[i, j] >= 0 and e26[i, j] < 512):
                    T.atomic_min(mem5[bid, 16 + e26[i, j] * 1], e23[i, j], memory_order="relaxed")
            for i, j in T.Parallel(32, 16):
                e28[i, j] = T.cast((e21[i, j] + e21[i, j]), "float32")
            for i, j in T.Parallel(32, 16):
                e29[i, j] = T.cast(e28[i, j], "float16")
            for i, j in T.Parallel(32, 16):
                e30[i, j] = T.cast((e29[i, j] * e29[i, j]), "float16")
            for i, j in T.Parallel(32, 16):
                out_e30[bid, 16 + (i * 16 + j)] = e30[i, j]
            for i, j in T.Parallel(32, 16):
                out_e3[bid, 16 + (i * 16 + j)] = e3[i, j]
            for i, j in T.Parallel(32, 16):
                out_e13[bid, 16 + (i * 16 + j)] = e13[i, j]
            for i, j in T.Parallel(32, 16):
                out_e17[bid, 16 + (i * 16 + j)] = e17[i, j]
            for i, j in T.Parallel(32, 16):
                out_e27[bid, 16 + (i * 16 + j)] = e27[i, j]
            for i, j in T.Parallel(32, 16):
                out_e7[bid, 16 + (i * 16 + j)] = e7[i, j]
    return impl

@tilelang.jit
def extended_2_0():
    @T.prim_func
    def impl(mem0: T.Buffer((3, 544), "float32"), mem1: T.Buffer((3, 544), "float32"), mem2: T.Buffer((3, 1060), "float32"), mem5: T.Buffer((3, 544), "float32"), out_e30: T.Buffer((3, 544), "float16"), steps: T.int32, limit: T.int32):
        with T.Kernel(3, threads=128) as bid:
            e1 = T.alloc_fragment((32, 16), "int32")
            e2 = T.alloc_fragment((32, 16), "bool")
            e3 = T.alloc_fragment((32, 16), "float32")
            e4 = T.alloc_fragment((32, 16), "float32")
            e5 = T.alloc_fragment((32, 16), "float32")
            e6 = T.alloc_fragment((32, 16), "int32")
            e7 = T.alloc_fragment((32, 16), "bool")
            e8 = T.alloc_fragment((32, 16), "float32")
            e9 = T.alloc_fragment((32, 16), "float32")
            e10 = T.alloc_fragment((32, 16), "int32")
            e11 = T.alloc_fragment((1,), "int32")
            e12 = T.alloc_fragment((32, 16), "bool")
            e13 = T.alloc_fragment((32, 16), "float32")
            e14 = T.alloc_fragment((32, 16), "float32")
            e15 = T.alloc_fragment((32, 16), "float32")
            e16 = T.alloc_fragment((32, 16), "bool")
            e17 = T.alloc_fragment((32, 16), "float32")
            e18 = T.alloc_fragment((32, 16), "int32")
            e19 = T.alloc_fragment((32, 16), "int32")
            e20 = T.alloc_fragment((32, 16), "float32")
            e21 = T.alloc_fragment((32, 16), "float32")
            e22 = T.alloc_fragment((1,), "float32")
            e23 = T.alloc_fragment((32, 16), "float32")
            e24 = T.alloc_fragment((32, 16), "int32")
            e25 = T.alloc_fragment((32, 16), "int32")
            e26 = T.alloc_fragment((32, 16), "int32")
            e27 = T.alloc_fragment((32, 16), "bool")
            e28 = T.alloc_fragment((32, 16), "float32")
            e29 = T.alloc_fragment((32, 16), "float16")
            e30 = T.alloc_fragment((32, 16), "float16")
            for i, j in T.Parallel(32, 16):
                e1[i, j] = T.cast((((i * 16 + j) + 119) % 512), "int32")
            for i, j in T.Parallel(32, 16):
                e2[i, j] = T.cast(True, "bool")
            for i, j in T.Parallel(32, 16):
                e3[i, j] = T.cast(T.if_then_else((e2[i, j] and e1[i, j] >= 0 and e1[i, j] < 512), mem0[bid, 16 + e1[i, j] * 1], T.cast(0, "float32")), "float32")
            for i, j in T.Parallel(32, 16):
                e4[i, j] = T.cast((e3[i, j] + e3[i, j]), "float32")
            for i, j in T.Parallel(32, 16):
                e5[i, j] = T.cast((e4[i, j] * e4[i, j]), "float32")
            for i, j in T.Parallel(32, 16):
                e6[i, j] = T.cast((((i * 16 + j) + 400) % 512), "int32")
            for i, j in T.Parallel(32, 16):
                e7[i, j] = T.cast(True, "bool")
            for i, j in T.Parallel(32, 16):
                e8[i, j] = T.cast(T.if_then_else((e7[i, j] and e6[i, j] >= 0 and e6[i, j] < 512), mem1[bid, 16 + e6[i, j] * 1], T.cast(0, "float32")), "float32")
            for i, j in T.Parallel(32, 16):
                e9[i, j] = T.cast((e5[i, j] + e8[i, j]), "float32")
            for i, j in T.Parallel(32, 16):
                e10[i, j] = T.cast((((511 - (i * 16 + j)) + 163) % 512), "int32")
            e11[0] = T.cast(limit, "int32")
            for i, j in T.Parallel(32, 16):
                e12[i, j] = T.cast((e10[i, j] < e11[0]), "bool")
            for i, j in T.Parallel(32, 16):
                if (e12[i, j] and e10[i, j] >= 0 and e10[i, j] < 512):
                    mem2[bid, 17 + e10[i, j] * 2] = e9[i, j]
            T.sync_threads()
            for i, j in T.Parallel(32, 16):
                e13[i, j] = T.cast(T.if_then_else((e12[i, j] and e10[i, j] >= 0 and e10[i, j] < 512), mem2[bid, 17 + e10[i, j] * 2], T.cast(0, "float32")), "float32")
            for i, j in T.Parallel(32, 16):
                e14[i, j] = T.cast(0.125, "float32")
            for i, j in T.Parallel(32, 16):
                e15[i, j] = T.cast((e13[i, j] + e14[i, j]), "float32")
            T.sync_threads()
            for i, j in T.Parallel(32, 16):
                if (e12[i, j] and e10[i, j] >= 0 and e10[i, j] < 512):
                    mem2[bid, 18 + e10[i, j] * 1] = e15[i, j]
            T.sync_threads()
            for i, j in T.Parallel(32, 16):
                e16[i, j] = T.cast(True, "bool")
            for i, j in T.Parallel(32, 16):
                e17[i, j] = T.cast(T.if_then_else((e16[i, j] and e10[i, j] >= 0 and e10[i, j] < 512), mem2[bid, 17 + e10[i, j] * 2], T.cast(0, "float32")), "float32")
            for i, j in T.Parallel(32, 16):
                e18[i, j] = T.cast(3, "int32")
            for i, j in T.Parallel(32, 16):
                e19[i, j] = T.cast((e10[i, j] ^ e18[i, j]), "int32")
            for i, j in T.Parallel(32, 16):
                e20[i, j] = T.cast(T.if_then_else((e12[i, j] and e19[i, j] >= 0 and e19[i, j] < 512), mem2[bid, 18 + e19[i, j] * 1], T.cast(0, "float32")), "float32")
            for i, j in T.Parallel(32, 16):
                e21[i, j] = T.cast((e17[i, j] + e20[i, j]), "float32")
            e22[0] = T.cast(0, "float32")
            for i, j in T.Parallel(32, 16):
                e23[i, j] = T.cast(e22[0], "float32")
            for i, j in T.Parallel(32, 16):
                e24[i, j] = T.cast((((i * 16 + j) + 150) % 512), "int32")
            for i, j in T.Parallel(32, 16):
                e25[i, j] = T.cast(7, "int32")
            for i, j in T.Parallel(32, 16):
                e26[i, j] = T.cast((e24[i, j] & e25[i, j]), "int32")
            for i, j in T.Parallel(32, 16):
                e27[i, j] = T.cast(True, "bool")
            for i, j in T.Parallel(32, 16):
                if (e27[i, j] and e26[i, j] >= 0 and e26[i, j] < 512):
                    T.atomic_min(mem5[bid, 16 + e26[i, j] * 1], e23[i, j], memory_order="relaxed")
            for i, j in T.Parallel(32, 16):
                e28[i, j] = T.cast((e21[i, j] + e21[i, j]), "float32")
            for i, j in T.Parallel(32, 16):
                e29[i, j] = T.cast(e28[i, j], "float16")
            for i, j in T.Parallel(32, 16):
                e30[i, j] = T.cast((e29[i, j] * e29[i, j]), "float16")
            for i, j in T.Parallel(32, 16):
                out_e30[bid, 16 + (i * 16 + j)] = e30[i, j]
    return impl

@tilelang.jit
def extended_2_1():
    @T.prim_func
    def impl(mem0: T.Buffer((3, 544), "float32"), mem1: T.Buffer((3, 544), "float32"), mem2: T.Buffer((3, 1060), "float32"), mem5: T.Buffer((3, 544), "float32"), out_e30: T.Buffer((3, 544), "float16"), out_e3: T.Buffer((3, 544), "float32"), out_e13: T.Buffer((3, 544), "float32"), out_e17: T.Buffer((3, 544), "float32"), out_e27: T.Buffer((3, 544), "bool"), out_e7: T.Buffer((3, 544), "bool"), steps: T.int32, limit: T.int32):
        with T.Kernel(3, threads=128) as bid:
            e1 = T.alloc_fragment((32, 16), "int32")
            e2 = T.alloc_fragment((32, 16), "bool")
            e3 = T.alloc_fragment((32, 16), "float32")
            e4 = T.alloc_fragment((32, 16), "float32")
            e5 = T.alloc_fragment((32, 16), "float32")
            e6 = T.alloc_fragment((32, 16), "int32")
            e7 = T.alloc_fragment((32, 16), "bool")
            e8 = T.alloc_fragment((32, 16), "float32")
            e9 = T.alloc_fragment((32, 16), "float32")
            e10 = T.alloc_fragment((32, 16), "int32")
            e11 = T.alloc_fragment((1,), "int32")
            e12 = T.alloc_fragment((32, 16), "bool")
            e13 = T.alloc_fragment((32, 16), "float32")
            e14 = T.alloc_fragment((32, 16), "float32")
            e15 = T.alloc_fragment((32, 16), "float32")
            e16 = T.alloc_fragment((32, 16), "bool")
            e17 = T.alloc_fragment((32, 16), "float32")
            e18 = T.alloc_fragment((32, 16), "int32")
            e19 = T.alloc_fragment((32, 16), "int32")
            e20 = T.alloc_fragment((32, 16), "float32")
            e21 = T.alloc_fragment((32, 16), "float32")
            e22 = T.alloc_fragment((1,), "float32")
            e23 = T.alloc_fragment((32, 16), "float32")
            e24 = T.alloc_fragment((32, 16), "int32")
            e25 = T.alloc_fragment((32, 16), "int32")
            e26 = T.alloc_fragment((32, 16), "int32")
            e27 = T.alloc_fragment((32, 16), "bool")
            e28 = T.alloc_fragment((32, 16), "float32")
            e29 = T.alloc_fragment((32, 16), "float16")
            e30 = T.alloc_fragment((32, 16), "float16")
            for i, j in T.Parallel(32, 16):
                e1[i, j] = T.cast((((i * 16 + j) + 119) % 512), "int32")
            for i, j in T.Parallel(32, 16):
                e2[i, j] = T.cast(True, "bool")
            for i, j in T.Parallel(32, 16):
                e3[i, j] = T.cast(T.if_then_else((e2[i, j] and e1[i, j] >= 0 and e1[i, j] < 512), mem0[bid, 16 + e1[i, j] * 1], T.cast(0, "float32")), "float32")
            for i, j in T.Parallel(32, 16):
                e4[i, j] = T.cast((e3[i, j] + e3[i, j]), "float32")
            for i, j in T.Parallel(32, 16):
                e5[i, j] = T.cast((e4[i, j] * e4[i, j]), "float32")
            for i, j in T.Parallel(32, 16):
                e6[i, j] = T.cast((((i * 16 + j) + 400) % 512), "int32")
            for i, j in T.Parallel(32, 16):
                e7[i, j] = T.cast(True, "bool")
            for i, j in T.Parallel(32, 16):
                e8[i, j] = T.cast(T.if_then_else((e7[i, j] and e6[i, j] >= 0 and e6[i, j] < 512), mem1[bid, 16 + e6[i, j] * 1], T.cast(0, "float32")), "float32")
            for i, j in T.Parallel(32, 16):
                e9[i, j] = T.cast((e5[i, j] + e8[i, j]), "float32")
            for i, j in T.Parallel(32, 16):
                e10[i, j] = T.cast((((511 - (i * 16 + j)) + 163) % 512), "int32")
            e11[0] = T.cast(limit, "int32")
            for i, j in T.Parallel(32, 16):
                e12[i, j] = T.cast((e10[i, j] < e11[0]), "bool")
            for i, j in T.Parallel(32, 16):
                if (e12[i, j] and e10[i, j] >= 0 and e10[i, j] < 512):
                    mem2[bid, 17 + e10[i, j] * 2] = e9[i, j]
            T.sync_threads()
            for i, j in T.Parallel(32, 16):
                e13[i, j] = T.cast(T.if_then_else((e12[i, j] and e10[i, j] >= 0 and e10[i, j] < 512), mem2[bid, 17 + e10[i, j] * 2], T.cast(0, "float32")), "float32")
            for i, j in T.Parallel(32, 16):
                e14[i, j] = T.cast(0.125, "float32")
            for i, j in T.Parallel(32, 16):
                e15[i, j] = T.cast((e13[i, j] + e14[i, j]), "float32")
            T.sync_threads()
            for i, j in T.Parallel(32, 16):
                if (e12[i, j] and e10[i, j] >= 0 and e10[i, j] < 512):
                    mem2[bid, 18 + e10[i, j] * 1] = e15[i, j]
            T.sync_threads()
            for i, j in T.Parallel(32, 16):
                e16[i, j] = T.cast(True, "bool")
            for i, j in T.Parallel(32, 16):
                e17[i, j] = T.cast(T.if_then_else((e16[i, j] and e10[i, j] >= 0 and e10[i, j] < 512), mem2[bid, 17 + e10[i, j] * 2], T.cast(0, "float32")), "float32")
            for i, j in T.Parallel(32, 16):
                e18[i, j] = T.cast(3, "int32")
            for i, j in T.Parallel(32, 16):
                e19[i, j] = T.cast((e10[i, j] ^ e18[i, j]), "int32")
            for i, j in T.Parallel(32, 16):
                e20[i, j] = T.cast(T.if_then_else((e12[i, j] and e19[i, j] >= 0 and e19[i, j] < 512), mem2[bid, 18 + e19[i, j] * 1], T.cast(0, "float32")), "float32")
            for i, j in T.Parallel(32, 16):
                e21[i, j] = T.cast((e17[i, j] + e20[i, j]), "float32")
            e22[0] = T.cast(0, "float32")
            for i, j in T.Parallel(32, 16):
                e23[i, j] = T.cast(e22[0], "float32")
            for i, j in T.Parallel(32, 16):
                e24[i, j] = T.cast((((i * 16 + j) + 150) % 512), "int32")
            for i, j in T.Parallel(32, 16):
                e25[i, j] = T.cast(7, "int32")
            for i, j in T.Parallel(32, 16):
                e26[i, j] = T.cast((e24[i, j] & e25[i, j]), "int32")
            for i, j in T.Parallel(32, 16):
                e27[i, j] = T.cast(True, "bool")
            for i, j in T.Parallel(32, 16):
                if (e27[i, j] and e26[i, j] >= 0 and e26[i, j] < 512):
                    T.atomic_min(mem5[bid, 16 + e26[i, j] * 1], e23[i, j], memory_order="relaxed")
            for i, j in T.Parallel(32, 16):
                e28[i, j] = T.cast((e21[i, j] + e21[i, j]), "float32")
            for i, j in T.Parallel(32, 16):
                e29[i, j] = T.cast(e28[i, j], "float16")
            for i, j in T.Parallel(32, 16):
                e30[i, j] = T.cast((e29[i, j] * e29[i, j]), "float16")
            for i, j in T.Parallel(32, 16):
                out_e30[bid, 16 + (i * 16 + j)] = e30[i, j]
            for i, j in T.Parallel(32, 16):
                out_e3[bid, 16 + (i * 16 + j)] = e3[i, j]
            for i, j in T.Parallel(32, 16):
                out_e13[bid, 16 + (i * 16 + j)] = e13[i, j]
            for i, j in T.Parallel(32, 16):
                out_e17[bid, 16 + (i * 16 + j)] = e17[i, j]
            for i, j in T.Parallel(32, 16):
                out_e27[bid, 16 + (i * 16 + j)] = e27[i, j]
            for i, j in T.Parallel(32, 16):
                out_e7[bid, 16 + (i * 16 + j)] = e7[i, j]
    return impl

@tilelang.jit
def extended_3_0():
    @T.prim_func
    def impl(mem0: T.Buffer((3, 544), "float32"), mem1: T.Buffer((3, 544), "float32"), mem2: T.Buffer((3, 1060), "float32"), mem5: T.Buffer((3, 544), "float32"), out_e30: T.Buffer((3, 544), "float16"), steps: T.int32, limit: T.int32):
        with T.Kernel(3, threads=256) as bid:
            e1 = T.alloc_fragment((32, 16), "int32")
            e2 = T.alloc_fragment((32, 16), "bool")
            e3 = T.alloc_fragment((32, 16), "float32")
            e4 = T.alloc_fragment((32, 16), "float32")
            e5 = T.alloc_fragment((32, 16), "float32")
            e6 = T.alloc_fragment((32, 16), "int32")
            e7 = T.alloc_fragment((32, 16), "bool")
            e8 = T.alloc_fragment((32, 16), "float32")
            e9 = T.alloc_fragment((32, 16), "float32")
            e10 = T.alloc_fragment((32, 16), "int32")
            e11 = T.alloc_fragment((1,), "int32")
            e12 = T.alloc_fragment((32, 16), "bool")
            e13 = T.alloc_fragment((32, 16), "float32")
            e14 = T.alloc_fragment((32, 16), "float32")
            e15 = T.alloc_fragment((32, 16), "float32")
            e16 = T.alloc_fragment((32, 16), "bool")
            e17 = T.alloc_fragment((32, 16), "float32")
            e18 = T.alloc_fragment((32, 16), "int32")
            e19 = T.alloc_fragment((32, 16), "int32")
            e20 = T.alloc_fragment((32, 16), "float32")
            e21 = T.alloc_fragment((32, 16), "float32")
            e22 = T.alloc_fragment((1,), "float32")
            e23 = T.alloc_fragment((32, 16), "float32")
            e24 = T.alloc_fragment((32, 16), "int32")
            e25 = T.alloc_fragment((32, 16), "int32")
            e26 = T.alloc_fragment((32, 16), "int32")
            e27 = T.alloc_fragment((32, 16), "bool")
            e28 = T.alloc_fragment((32, 16), "float32")
            e29 = T.alloc_fragment((32, 16), "float16")
            e30 = T.alloc_fragment((32, 16), "float16")
            for i, j in T.Parallel(32, 16):
                e1[i, j] = T.cast((((i * 16 + j) + 119) % 512), "int32")
            for i, j in T.Parallel(32, 16):
                e2[i, j] = T.cast(True, "bool")
            for i, j in T.Parallel(32, 16):
                e3[i, j] = T.cast(T.if_then_else((e2[i, j] and e1[i, j] >= 0 and e1[i, j] < 512), mem0[bid, 16 + e1[i, j] * 1], T.cast(0, "float32")), "float32")
            for i, j in T.Parallel(32, 16):
                e4[i, j] = T.cast((e3[i, j] + e3[i, j]), "float32")
            for i, j in T.Parallel(32, 16):
                e5[i, j] = T.cast((e4[i, j] * e4[i, j]), "float32")
            for i, j in T.Parallel(32, 16):
                e6[i, j] = T.cast((((i * 16 + j) + 400) % 512), "int32")
            for i, j in T.Parallel(32, 16):
                e7[i, j] = T.cast(True, "bool")
            for i, j in T.Parallel(32, 16):
                e8[i, j] = T.cast(T.if_then_else((e7[i, j] and e6[i, j] >= 0 and e6[i, j] < 512), mem1[bid, 16 + e6[i, j] * 1], T.cast(0, "float32")), "float32")
            for i, j in T.Parallel(32, 16):
                e9[i, j] = T.cast((e5[i, j] + e8[i, j]), "float32")
            for i, j in T.Parallel(32, 16):
                e10[i, j] = T.cast((((511 - (i * 16 + j)) + 163) % 512), "int32")
            e11[0] = T.cast(limit, "int32")
            for i, j in T.Parallel(32, 16):
                e12[i, j] = T.cast((e10[i, j] < e11[0]), "bool")
            for i, j in T.Parallel(32, 16):
                if (e12[i, j] and e10[i, j] >= 0 and e10[i, j] < 512):
                    mem2[bid, 17 + e10[i, j] * 2] = e9[i, j]
            T.sync_threads()
            for i, j in T.Parallel(32, 16):
                e13[i, j] = T.cast(T.if_then_else((e12[i, j] and e10[i, j] >= 0 and e10[i, j] < 512), mem2[bid, 17 + e10[i, j] * 2], T.cast(0, "float32")), "float32")
            for i, j in T.Parallel(32, 16):
                e14[i, j] = T.cast(0.125, "float32")
            for i, j in T.Parallel(32, 16):
                e15[i, j] = T.cast((e13[i, j] + e14[i, j]), "float32")
            T.sync_threads()
            for i, j in T.Parallel(32, 16):
                if (e12[i, j] and e10[i, j] >= 0 and e10[i, j] < 512):
                    mem2[bid, 18 + e10[i, j] * 1] = e15[i, j]
            T.sync_threads()
            for i, j in T.Parallel(32, 16):
                e16[i, j] = T.cast(True, "bool")
            for i, j in T.Parallel(32, 16):
                e17[i, j] = T.cast(T.if_then_else((e16[i, j] and e10[i, j] >= 0 and e10[i, j] < 512), mem2[bid, 17 + e10[i, j] * 2], T.cast(0, "float32")), "float32")
            for i, j in T.Parallel(32, 16):
                e18[i, j] = T.cast(3, "int32")
            for i, j in T.Parallel(32, 16):
                e19[i, j] = T.cast((e10[i, j] ^ e18[i, j]), "int32")
            for i, j in T.Parallel(32, 16):
                e20[i, j] = T.cast(T.if_then_else((e12[i, j] and e19[i, j] >= 0 and e19[i, j] < 512), mem2[bid, 18 + e19[i, j] * 1], T.cast(0, "float32")), "float32")
            for i, j in T.Parallel(32, 16):
                e21[i, j] = T.cast((e17[i, j] + e20[i, j]), "float32")
            e22[0] = T.cast(0, "float32")
            for i, j in T.Parallel(32, 16):
                e23[i, j] = T.cast(e22[0], "float32")
            for i, j in T.Parallel(32, 16):
                e24[i, j] = T.cast((((i * 16 + j) + 150) % 512), "int32")
            for i, j in T.Parallel(32, 16):
                e25[i, j] = T.cast(7, "int32")
            for i, j in T.Parallel(32, 16):
                e26[i, j] = T.cast((e24[i, j] & e25[i, j]), "int32")
            for i, j in T.Parallel(32, 16):
                e27[i, j] = T.cast(True, "bool")
            for i, j in T.Parallel(32, 16):
                if (e27[i, j] and e26[i, j] >= 0 and e26[i, j] < 512):
                    T.atomic_min(mem5[bid, 16 + e26[i, j] * 1], e23[i, j], memory_order="relaxed")
            for i, j in T.Parallel(32, 16):
                e28[i, j] = T.cast((e21[i, j] + e21[i, j]), "float32")
            for i, j in T.Parallel(32, 16):
                e29[i, j] = T.cast(e28[i, j], "float16")
            for i, j in T.Parallel(32, 16):
                e30[i, j] = T.cast((e29[i, j] * e29[i, j]), "float16")
            for i, j in T.Parallel(32, 16):
                out_e30[bid, 16 + (i * 16 + j)] = e30[i, j]
    return impl

@tilelang.jit
def extended_3_1():
    @T.prim_func
    def impl(mem0: T.Buffer((3, 544), "float32"), mem1: T.Buffer((3, 544), "float32"), mem2: T.Buffer((3, 1060), "float32"), mem5: T.Buffer((3, 544), "float32"), out_e30: T.Buffer((3, 544), "float16"), out_e3: T.Buffer((3, 544), "float32"), out_e13: T.Buffer((3, 544), "float32"), out_e17: T.Buffer((3, 544), "float32"), out_e27: T.Buffer((3, 544), "bool"), out_e7: T.Buffer((3, 544), "bool"), steps: T.int32, limit: T.int32):
        with T.Kernel(3, threads=256) as bid:
            e1 = T.alloc_fragment((32, 16), "int32")
            e2 = T.alloc_fragment((32, 16), "bool")
            e3 = T.alloc_fragment((32, 16), "float32")
            e4 = T.alloc_fragment((32, 16), "float32")
            e5 = T.alloc_fragment((32, 16), "float32")
            e6 = T.alloc_fragment((32, 16), "int32")
            e7 = T.alloc_fragment((32, 16), "bool")
            e8 = T.alloc_fragment((32, 16), "float32")
            e9 = T.alloc_fragment((32, 16), "float32")
            e10 = T.alloc_fragment((32, 16), "int32")
            e11 = T.alloc_fragment((1,), "int32")
            e12 = T.alloc_fragment((32, 16), "bool")
            e13 = T.alloc_fragment((32, 16), "float32")
            e14 = T.alloc_fragment((32, 16), "float32")
            e15 = T.alloc_fragment((32, 16), "float32")
            e16 = T.alloc_fragment((32, 16), "bool")
            e17 = T.alloc_fragment((32, 16), "float32")
            e18 = T.alloc_fragment((32, 16), "int32")
            e19 = T.alloc_fragment((32, 16), "int32")
            e20 = T.alloc_fragment((32, 16), "float32")
            e21 = T.alloc_fragment((32, 16), "float32")
            e22 = T.alloc_fragment((1,), "float32")
            e23 = T.alloc_fragment((32, 16), "float32")
            e24 = T.alloc_fragment((32, 16), "int32")
            e25 = T.alloc_fragment((32, 16), "int32")
            e26 = T.alloc_fragment((32, 16), "int32")
            e27 = T.alloc_fragment((32, 16), "bool")
            e28 = T.alloc_fragment((32, 16), "float32")
            e29 = T.alloc_fragment((32, 16), "float16")
            e30 = T.alloc_fragment((32, 16), "float16")
            for i, j in T.Parallel(32, 16):
                e1[i, j] = T.cast((((i * 16 + j) + 119) % 512), "int32")
            for i, j in T.Parallel(32, 16):
                e2[i, j] = T.cast(True, "bool")
            for i, j in T.Parallel(32, 16):
                e3[i, j] = T.cast(T.if_then_else((e2[i, j] and e1[i, j] >= 0 and e1[i, j] < 512), mem0[bid, 16 + e1[i, j] * 1], T.cast(0, "float32")), "float32")
            for i, j in T.Parallel(32, 16):
                e4[i, j] = T.cast((e3[i, j] + e3[i, j]), "float32")
            for i, j in T.Parallel(32, 16):
                e5[i, j] = T.cast((e4[i, j] * e4[i, j]), "float32")
            for i, j in T.Parallel(32, 16):
                e6[i, j] = T.cast((((i * 16 + j) + 400) % 512), "int32")
            for i, j in T.Parallel(32, 16):
                e7[i, j] = T.cast(True, "bool")
            for i, j in T.Parallel(32, 16):
                e8[i, j] = T.cast(T.if_then_else((e7[i, j] and e6[i, j] >= 0 and e6[i, j] < 512), mem1[bid, 16 + e6[i, j] * 1], T.cast(0, "float32")), "float32")
            for i, j in T.Parallel(32, 16):
                e9[i, j] = T.cast((e5[i, j] + e8[i, j]), "float32")
            for i, j in T.Parallel(32, 16):
                e10[i, j] = T.cast((((511 - (i * 16 + j)) + 163) % 512), "int32")
            e11[0] = T.cast(limit, "int32")
            for i, j in T.Parallel(32, 16):
                e12[i, j] = T.cast((e10[i, j] < e11[0]), "bool")
            for i, j in T.Parallel(32, 16):
                if (e12[i, j] and e10[i, j] >= 0 and e10[i, j] < 512):
                    mem2[bid, 17 + e10[i, j] * 2] = e9[i, j]
            T.sync_threads()
            for i, j in T.Parallel(32, 16):
                e13[i, j] = T.cast(T.if_then_else((e12[i, j] and e10[i, j] >= 0 and e10[i, j] < 512), mem2[bid, 17 + e10[i, j] * 2], T.cast(0, "float32")), "float32")
            for i, j in T.Parallel(32, 16):
                e14[i, j] = T.cast(0.125, "float32")
            for i, j in T.Parallel(32, 16):
                e15[i, j] = T.cast((e13[i, j] + e14[i, j]), "float32")
            T.sync_threads()
            for i, j in T.Parallel(32, 16):
                if (e12[i, j] and e10[i, j] >= 0 and e10[i, j] < 512):
                    mem2[bid, 18 + e10[i, j] * 1] = e15[i, j]
            T.sync_threads()
            for i, j in T.Parallel(32, 16):
                e16[i, j] = T.cast(True, "bool")
            for i, j in T.Parallel(32, 16):
                e17[i, j] = T.cast(T.if_then_else((e16[i, j] and e10[i, j] >= 0 and e10[i, j] < 512), mem2[bid, 17 + e10[i, j] * 2], T.cast(0, "float32")), "float32")
            for i, j in T.Parallel(32, 16):
                e18[i, j] = T.cast(3, "int32")
            for i, j in T.Parallel(32, 16):
                e19[i, j] = T.cast((e10[i, j] ^ e18[i, j]), "int32")
            for i, j in T.Parallel(32, 16):
                e20[i, j] = T.cast(T.if_then_else((e12[i, j] and e19[i, j] >= 0 and e19[i, j] < 512), mem2[bid, 18 + e19[i, j] * 1], T.cast(0, "float32")), "float32")
            for i, j in T.Parallel(32, 16):
                e21[i, j] = T.cast((e17[i, j] + e20[i, j]), "float32")
            e22[0] = T.cast(0, "float32")
            for i, j in T.Parallel(32, 16):
                e23[i, j] = T.cast(e22[0], "float32")
            for i, j in T.Parallel(32, 16):
                e24[i, j] = T.cast((((i * 16 + j) + 150) % 512), "int32")
            for i, j in T.Parallel(32, 16):
                e25[i, j] = T.cast(7, "int32")
            for i, j in T.Parallel(32, 16):
                e26[i, j] = T.cast((e24[i, j] & e25[i, j]), "int32")
            for i, j in T.Parallel(32, 16):
                e27[i, j] = T.cast(True, "bool")
            for i, j in T.Parallel(32, 16):
                if (e27[i, j] and e26[i, j] >= 0 and e26[i, j] < 512):
                    T.atomic_min(mem5[bid, 16 + e26[i, j] * 1], e23[i, j], memory_order="relaxed")
            for i, j in T.Parallel(32, 16):
                e28[i, j] = T.cast((e21[i, j] + e21[i, j]), "float32")
            for i, j in T.Parallel(32, 16):
                e29[i, j] = T.cast(e28[i, j], "float16")
            for i, j in T.Parallel(32, 16):
                e30[i, j] = T.cast((e29[i, j] * e29[i, j]), "float16")
            for i, j in T.Parallel(32, 16):
                out_e30[bid, 16 + (i * 16 + j)] = e30[i, j]
            for i, j in T.Parallel(32, 16):
                out_e3[bid, 16 + (i * 16 + j)] = e3[i, j]
            for i, j in T.Parallel(32, 16):
                out_e13[bid, 16 + (i * 16 + j)] = e13[i, j]
            for i, j in T.Parallel(32, 16):
                out_e17[bid, 16 + (i * 16 + j)] = e17[i, j]
            for i, j in T.Parallel(32, 16):
                out_e27[bid, 16 + (i * 16 + j)] = e27[i, j]
            for i, j in T.Parallel(32, 16):
                out_e7[bid, 16 + (i * 16 + j)] = e7[i, j]
    return impl

@tilelang.jit
def extended_0_ident():
    @T.prim_func
    def impl(mem0: T.Buffer((3, 544), "float32"), mem1: T.Buffer((3, 544), "float32"), mem2: T.Buffer((3, 1060), "float32"), mem5: T.Buffer((3, 544), "float32"), out_e30: T.Buffer((3, 544), "float16"), out_e3: T.Buffer((3, 544), "float32"), out_e13: T.Buffer((3, 544), "float32"), out_e17: T.Buffer((3, 544), "float32"), out_e27: T.Buffer((3, 544), "bool"), out_e7: T.Buffer((3, 544), "bool"), steps: T.int32, limit: T.int32):
        with T.Kernel(3, threads=128) as bid:
            e1 = T.alloc_fragment((32, 16), "int32")
            e2 = T.alloc_fragment((32, 16), "bool")
            e3 = T.alloc_fragment((32, 16), "float32")
            e4 = T.alloc_fragment((32, 16), "float32")
            e5_l = T.alloc_fragment((32, 16), "float32")
            e5_r = T.alloc_fragment((32, 16), "float32")
            e5 = T.alloc_fragment((32, 16), "float32")
            e6 = T.alloc_fragment((32, 16), "int32")
            e7 = T.alloc_fragment((32, 16), "bool")
            e8 = T.alloc_fragment((32, 16), "float32")
            e9 = T.alloc_fragment((32, 16), "float32")
            e10 = T.alloc_fragment((32, 16), "int32")
            e11 = T.alloc_fragment((1,), "int32")
            e12 = T.alloc_fragment((32, 16), "bool")
            e13 = T.alloc_fragment((32, 16), "float32")
            e14 = T.alloc_fragment((32, 16), "float32")
            e15 = T.alloc_fragment((32, 16), "float32")
            e16 = T.alloc_fragment((32, 16), "bool")
            e17 = T.alloc_fragment((32, 16), "float32")
            e18 = T.alloc_fragment((32, 16), "int32")
            e19 = T.alloc_fragment((32, 16), "int32")
            e20 = T.alloc_fragment((32, 16), "float32")
            e21 = T.alloc_fragment((32, 16), "float32")
            e22 = T.alloc_fragment((1,), "float32")
            e23 = T.alloc_fragment((32, 16), "float32")
            e24 = T.alloc_fragment((32, 16), "int32")
            e25 = T.alloc_fragment((32, 16), "int32")
            e26 = T.alloc_fragment((32, 16), "int32")
            e27 = T.alloc_fragment((32, 16), "bool")
            e28 = T.alloc_fragment((32, 16), "float32")
            e29 = T.alloc_fragment((32, 16), "float16")
            e30 = T.alloc_fragment((32, 16), "float16")
            for i, j in T.Parallel(32, 16):
                e1[i, j] = T.cast((((i * 16 + j) + 119) % 512), "int32")
            for i, j in T.Parallel(32, 16):
                e2[i, j] = T.cast(True, "bool")
            for i, j in T.Parallel(32, 16):
                e3[i, j] = T.cast(T.if_then_else((e2[i, j] and e1[i, j] >= 0 and e1[i, j] < 512), mem0[bid, 16 + e1[i, j] * 1], T.cast(0, "float32")), "float32")
            for i, j in T.Parallel(32, 16):
                e4[i, j] = T.cast((e3[i, j] + e3[i, j]), "float32")
            for i, j in T.Parallel(32, 16):
                e5_l[i, j] = T.cast((e4[i, j] * e3[i, j]), "float32")
            for i, j in T.Parallel(32, 16):
                e5_r[i, j] = T.cast((e4[i, j] * e3[i, j]), "float32")
            for i, j in T.Parallel(32, 16):
                e5[i, j] = T.cast((e5_l[i, j] + e5_r[i, j]), "float32")
            for i, j in T.Parallel(32, 16):
                e6[i, j] = T.cast((((i * 16 + j) + 400) % 512), "int32")
            for i, j in T.Parallel(32, 16):
                e7[i, j] = T.cast(True, "bool")
            for i, j in T.Parallel(32, 16):
                e8[i, j] = T.cast(T.if_then_else((e7[i, j] and e6[i, j] >= 0 and e6[i, j] < 512), mem1[bid, 16 + e6[i, j] * 1], T.cast(0, "float32")), "float32")
            for i, j in T.Parallel(32, 16):
                e9[i, j] = T.cast((e5[i, j] + e8[i, j]), "float32")
            for i, j in T.Parallel(32, 16):
                e10[i, j] = T.cast((((511 - (i * 16 + j)) + 163) % 512), "int32")
            e11[0] = T.cast(limit, "int32")
            for i, j in T.Parallel(32, 16):
                e12[i, j] = T.cast((e10[i, j] < e11[0]), "bool")
            for i, j in T.Parallel(32, 16):
                if (e12[i, j] and e10[i, j] >= 0 and e10[i, j] < 512):
                    mem2[bid, 17 + e10[i, j] * 2] = e9[i, j]
            T.sync_threads()
            for i, j in T.Parallel(32, 16):
                e13[i, j] = T.cast(T.if_then_else((e12[i, j] and e10[i, j] >= 0 and e10[i, j] < 512), mem2[bid, 17 + e10[i, j] * 2], T.cast(0, "float32")), "float32")
            for i, j in T.Parallel(32, 16):
                e14[i, j] = T.cast(0.125, "float32")
            for i, j in T.Parallel(32, 16):
                e15[i, j] = T.cast((e13[i, j] + e14[i, j]), "float32")
            T.sync_threads()
            for i, j in T.Parallel(32, 16):
                if (e12[i, j] and e10[i, j] >= 0 and e10[i, j] < 512):
                    mem2[bid, 18 + e10[i, j] * 1] = e15[i, j]
            T.sync_threads()
            for i, j in T.Parallel(32, 16):
                e16[i, j] = T.cast(True, "bool")
            for i, j in T.Parallel(32, 16):
                e17[i, j] = T.cast(T.if_then_else((e16[i, j] and e10[i, j] >= 0 and e10[i, j] < 512), mem2[bid, 17 + e10[i, j] * 2], T.cast(0, "float32")), "float32")
            for i, j in T.Parallel(32, 16):
                e18[i, j] = T.cast(3, "int32")
            for i, j in T.Parallel(32, 16):
                e19[i, j] = T.cast((e10[i, j] ^ e18[i, j]), "int32")
            for i, j in T.Parallel(32, 16):
                e20[i, j] = T.cast(T.if_then_else((e12[i, j] and e19[i, j] >= 0 and e19[i, j] < 512), mem2[bid, 18 + e19[i, j] * 1], T.cast(0, "float32")), "float32")
            for i, j in T.Parallel(32, 16):
                e21[i, j] = T.cast((e17[i, j] + e20[i, j]), "float32")
            e22[0] = T.cast(0, "float32")
            for i, j in T.Parallel(32, 16):
                e23[i, j] = T.cast(e22[0], "float32")
            for i, j in T.Parallel(32, 16):
                e24[i, j] = T.cast((((i * 16 + j) + 150) % 512), "int32")
            for i, j in T.Parallel(32, 16):
                e25[i, j] = T.cast(7, "int32")
            for i, j in T.Parallel(32, 16):
                e26[i, j] = T.cast((e24[i, j] & e25[i, j]), "int32")
            for i, j in T.Parallel(32, 16):
                e27[i, j] = T.cast(True, "bool")
            for i, j in T.Parallel(32, 16):
                if (e27[i, j] and e26[i, j] >= 0 and e26[i, j] < 512):
                    T.atomic_min(mem5[bid, 16 + e26[i, j] * 1], e23[i, j], memory_order="relaxed")
            for i, j in T.Parallel(32, 16):
                e28[i, j] = T.cast((e21[i, j] + e21[i, j]), "float32")
            for i, j in T.Parallel(32, 16):
                e29[i, j] = T.cast(e28[i, j], "float16")
            for i, j in T.Parallel(32, 16):
                e30[i, j] = T.cast((e29[i, j] * e29[i, j]), "float16")
            for i, j in T.Parallel(32, 16):
                out_e30[bid, 16 + (i * 16 + j)] = e30[i, j]
            for i, j in T.Parallel(32, 16):
                out_e3[bid, 16 + (i * 16 + j)] = e3[i, j]
            for i, j in T.Parallel(32, 16):
                out_e13[bid, 16 + (i * 16 + j)] = e13[i, j]
            for i, j in T.Parallel(32, 16):
                out_e17[bid, 16 + (i * 16 + j)] = e17[i, j]
            for i, j in T.Parallel(32, 16):
                out_e27[bid, 16 + (i * 16 + j)] = e27[i, j]
            for i, j in T.Parallel(32, 16):
                out_e7[bid, 16 + (i * 16 + j)] = e7[i, j]
    return impl

def prepare_extended(device_compile=True):
    import torch
    import threading
    from tilelang import tvm
    from tilelang.engine import lower as tilelang_lower
    arch = int(os.environ.get('TILESMITH_CUDA_ARCH', '0'))
    if not arch:
        major, minor = torch.cuda.get_device_capability() if torch.cuda.is_available() else (8, 9)
        arch = major * 10 + minor
    target = tvm.target.Target({"kind": "cuda", "arch": "sm_" + str(arch)})
    variants = []
    labels = ['tilelang_0', 'tilelang_1', 'tilelang_2', 'tilelang_3', 'tilelang_4', 'tilelang_5', 'tilelang_6', 'tilelang_7', 'tilelang_8_ident']
    compile_options = [{'threads': 128, 'stages': 1, 'pass_configs': {}}, {'threads': 128, 'stages': 1, 'pass_configs': {}}, {'threads': 256, 'stages': 2, 'pass_configs': {'tirx.disable_vectorize': True}}, {'threads': 256, 'stages': 2, 'pass_configs': {'tirx.disable_vectorize': True}}, {'threads': 128, 'stages': 1, 'pass_configs': {'tl.force_let_inline': True, 'tl.enable_aggressive_shared_memory_merge': True, 'tl.disable_shared_memory_reuse': True, 'tl.loop_unswitching_allow_non_trivial_else': True, 'tl.enable_lower_ldgstg': True, 'tl.enable_lower_ldgstg_predicated': True, 'tl.if_stmt_binding_inline_replayable_binds': True}}, {'threads': 128, 'stages': 1, 'pass_configs': {'tl.force_let_inline': True, 'tl.enable_aggressive_shared_memory_merge': True, 'tl.disable_shared_memory_reuse': True, 'tl.loop_unswitching_allow_non_trivial_else': True, 'tl.enable_lower_ldgstg': True, 'tl.enable_lower_ldgstg_predicated': True, 'tl.if_stmt_binding_inline_replayable_binds': True}}, {'threads': 256, 'stages': 3, 'pass_configs': {'tl.disable_warp_specialized': True, 'tl.disable_data_race_check': True, 'tl.force_let_inline': True, 'tl.storage_rewrite_detect_inplace': True, 'tl.disable_vectorize_256': True, 'tl.disable_shuffle_elect': True, 'tl.if_stmt_binding_inline_replayable_binds': True, 'tl.disable_out_of_bound_warning': True, 'tl.ptxas_register_usage_level': 0}}, {'threads': 256, 'stages': 3, 'pass_configs': {'tl.disable_warp_specialized': True, 'tl.disable_data_race_check': True, 'tl.force_let_inline': True, 'tl.storage_rewrite_detect_inplace': True, 'tl.disable_vectorize_256': True, 'tl.disable_shuffle_elect': True, 'tl.if_stmt_binding_inline_replayable_binds': True, 'tl.disable_out_of_bound_warning': True, 'tl.ptxas_register_usage_level': 0}}, {'threads': 128, 'stages': 1, 'pass_configs': {}, 'identity': True}]
    watched = [['e30'], ['e30', 'e3', 'e13', 'e17', 'e27', 'e7'], ['e30'], ['e30', 'e3', 'e13', 'e17', 'e27', 'e7'], ['e30'], ['e30', 'e3', 'e13', 'e17', 'e27', 'e7'], ['e30'], ['e30', 'e3', 'e13', 'e17', 'e27', 'e7'], ['e30', 'e3', 'e13', 'e17', 'e27', 'e7']]
    roots = ['mem0', 'mem1', 'mem2', 'mem5']
    irs = []
    extended_stage("lowering", labels[0])
    with target:
        irs.append(extended_0_0.get_tir())
    record_extended_compilation(labels[0], {"tir": str(irs[0])}, compile_options[0], complete=False)
    extended_stage("lowering", labels[1])
    with target:
        irs.append(extended_0_1.get_tir())
    record_extended_compilation(labels[1], {"tir": str(irs[1])}, compile_options[1], complete=False)
    extended_stage("lowering", labels[2])
    with target:
        irs.append(extended_1_0.get_tir())
    record_extended_compilation(labels[2], {"tir": str(irs[2])}, compile_options[2], complete=False)
    extended_stage("lowering", labels[3])
    with target:
        irs.append(extended_1_1.get_tir())
    record_extended_compilation(labels[3], {"tir": str(irs[3])}, compile_options[3], complete=False)
    extended_stage("lowering", labels[4])
    with target:
        irs.append(extended_2_0.get_tir())
    record_extended_compilation(labels[4], {"tir": str(irs[4])}, compile_options[4], complete=False)
    extended_stage("lowering", labels[5])
    with target:
        irs.append(extended_2_1.get_tir())
    record_extended_compilation(labels[5], {"tir": str(irs[5])}, compile_options[5], complete=False)
    extended_stage("lowering", labels[6])
    with target:
        irs.append(extended_3_0.get_tir())
    record_extended_compilation(labels[6], {"tir": str(irs[6])}, compile_options[6], complete=False)
    extended_stage("lowering", labels[7])
    with target:
        irs.append(extended_3_1.get_tir())
    record_extended_compilation(labels[7], {"tir": str(irs[7])}, compile_options[7], complete=False)
    extended_stage("lowering", labels[8])
    with target:
        irs.append(extended_0_ident.get_tir())
    record_extended_compilation(labels[8], {"tir": str(irs[8])}, compile_options[8], complete=False)
    compiled = [None] * len(irs)
    lowered = [None] * len(irs)
    errors = [None] * len(irs)
    def compile_variant(start):
        for index in range(start, len(irs), 8):
            try:
                if device_compile:
                    compiled[index] = tilelang.compile(irs[index], target=target, pass_configs=compile_options[index]["pass_configs"])
                else:
                    with target, tvm.transform.PassContext(opt_level=3, config=compile_options[index]["pass_configs"]):
                        lowered[index] = tilelang_lower(irs[index], target=target, enable_device_compile=False)
            except BaseException as error:
                errors[index] = error
    threads = [threading.Thread(target=compile_variant, args=(start,)) for start in range(min(len(irs), 8))]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    for index in range(len(irs)):
        if errors[index] is not None:
            extended_stage("device_compile", labels[index])
            raise errors[index]
    try:
        from tilelang.cuda.backend import tilelang_callback_cuda_compile
    except ImportError:
        from tilelang.engine.lower import tilelang_callback_cuda_compile
    def make_launch(index):
        def launch(memories, outputs, steps, limit):
            arguments = [memories[n] for n in roots] + [outputs[n] for n in watched[index]]
            compiled[index](*arguments, steps, limit)
        return launch
    for index in range(len(irs)):
        if device_compile:
            compiled_index = compiled[index]
            artifact_index = compiled_index.artifact
            artifacts_index = {"cuda": compiled_index.get_kernel_source()}
            if artifact_index is not None:
                artifacts_index["lowered_tir"] = str(artifact_index.device_mod)
            record_extended_compilation(labels[index], artifacts_index, compile_options[index])
            variants.append((labels[index], watched[index], make_launch(index)))
        else:
            lowered_index = lowered[index]
            record_extended_compilation(labels[index], {"lowered_tir": str(lowered_index.device_mod), "cuda": lowered_index.kernel_source}, compile_options[index], complete=False)
            cubin = tilelang_callback_cuda_compile(lowered_index.kernel_source, target, compile_options[index]["pass_configs"])
            record_extended_compilation(labels[index], {"cubin": cubin}, compile_options[index])
    return variants

PROGRAM = {'type': 'extended', 'version': 1, 'body': {'arguments': [], 'operations': [{'op': 'index', 'results': [{'name': 'e1', 'type': {'dtype': 'int32', 'shape': (32, 16)}}], 'operands': [], 'attrs': {'shift': 119, 'reverse': False}, 'regions': []}, {'op': 'constant', 'results': [{'name': 'e2', 'type': {'dtype': 'bool', 'shape': (32, 16)}}], 'operands': [], 'attrs': {'value': True}, 'regions': []}, {'op': 'load', 'results': [{'name': 'e3', 'type': {'dtype': 'float32', 'shape': (32, 16)}}], 'operands': ['e1', 'e2'], 'attrs': {'buffer': 'mem0'}, 'regions': []}, {'op': 'add', 'results': [{'name': 'e4', 'type': {'dtype': 'float32', 'shape': (32, 16)}}], 'operands': ['e3', 'e3'], 'attrs': {}, 'regions': []}, {'op': 'mul', 'results': [{'name': 'e5', 'type': {'dtype': 'float32', 'shape': (32, 16)}}], 'operands': ['e4', 'e4'], 'attrs': {}, 'regions': []}, {'op': 'index', 'results': [{'name': 'e6', 'type': {'dtype': 'int32', 'shape': (32, 16)}}], 'operands': [], 'attrs': {'shift': 400, 'reverse': False}, 'regions': []}, {'op': 'constant', 'results': [{'name': 'e7', 'type': {'dtype': 'bool', 'shape': (32, 16)}}], 'operands': [], 'attrs': {'value': True}, 'regions': []}, {'op': 'load', 'results': [{'name': 'e8', 'type': {'dtype': 'float32', 'shape': (32, 16)}}], 'operands': ['e6', 'e7'], 'attrs': {'buffer': 'mem1'}, 'regions': []}, {'op': 'add', 'results': [{'name': 'e9', 'type': {'dtype': 'float32', 'shape': (32, 16)}}], 'operands': ['e5', 'e8'], 'attrs': {}, 'regions': []}, {'op': 'index', 'results': [{'name': 'e10', 'type': {'dtype': 'int32', 'shape': (32, 16)}}], 'operands': [], 'attrs': {'shift': 163, 'reverse': True}, 'regions': []}, {'op': 'parameter', 'results': [{'name': 'e11', 'type': {'dtype': 'int32', 'shape': ()}}], 'operands': [], 'attrs': {'name': 'limit'}, 'regions': []}, {'op': 'lt', 'results': [{'name': 'e12', 'type': {'dtype': 'bool', 'shape': (32, 16)}}], 'operands': ['e10', 'e11'], 'attrs': {}, 'regions': []}, {'op': 'store', 'results': [], 'operands': ['e10', 'e12', 'e9'], 'attrs': {'buffer': 'mem3'}, 'regions': []}, {'op': 'barrier', 'results': [], 'operands': [], 'attrs': {}, 'regions': []}, {'op': 'load', 'results': [{'name': 'e13', 'type': {'dtype': 'float32', 'shape': (32, 16)}}], 'operands': ['e10', 'e12'], 'attrs': {'buffer': 'mem3'}, 'regions': []}, {'op': 'constant', 'results': [{'name': 'e14', 'type': {'dtype': 'float32', 'shape': (32, 16)}}], 'operands': [], 'attrs': {'value': 0.125}, 'regions': []}, {'op': 'add', 'results': [{'name': 'e15', 'type': {'dtype': 'float32', 'shape': (32, 16)}}], 'operands': ['e13', 'e14'], 'attrs': {}, 'regions': []}, {'op': 'barrier', 'results': [], 'operands': [], 'attrs': {}, 'regions': []}, {'op': 'store', 'results': [], 'operands': ['e10', 'e12', 'e15'], 'attrs': {'buffer': 'mem4'}, 'regions': []}, {'op': 'barrier', 'results': [], 'operands': [], 'attrs': {}, 'regions': []}, {'op': 'constant', 'results': [{'name': 'e16', 'type': {'dtype': 'bool', 'shape': (32, 16)}}], 'operands': [], 'attrs': {'value': True}, 'regions': []}, {'op': 'load', 'results': [{'name': 'e17', 'type': {'dtype': 'float32', 'shape': (32, 16)}}], 'operands': ['e10', 'e16'], 'attrs': {'buffer': 'mem3'}, 'regions': []}, {'op': 'constant', 'results': [{'name': 'e18', 'type': {'dtype': 'int32', 'shape': (32, 16)}}], 'operands': [], 'attrs': {'value': 3}, 'regions': []}, {'op': 'bitxor', 'results': [{'name': 'e19', 'type': {'dtype': 'int32', 'shape': (32, 16)}}], 'operands': ['e10', 'e18'], 'attrs': {}, 'regions': []}, {'op': 'load', 'results': [{'name': 'e20', 'type': {'dtype': 'float32', 'shape': (32, 16)}}], 'operands': ['e19', 'e12'], 'attrs': {'buffer': 'mem4'}, 'regions': []}, {'op': 'add', 'results': [{'name': 'e21', 'type': {'dtype': 'float32', 'shape': (32, 16)}}], 'operands': ['e17', 'e20'], 'attrs': {}, 'regions': []}, {'op': 'constant', 'results': [{'name': 'e22', 'type': {'dtype': 'float32', 'shape': ()}}], 'operands': [], 'attrs': {'value': 0}, 'regions': []}, {'op': 'broadcast', 'results': [{'name': 'e23', 'type': {'dtype': 'float32', 'shape': (32, 16)}}], 'operands': ['e22'], 'attrs': {}, 'regions': []}, {'op': 'index', 'results': [{'name': 'e24', 'type': {'dtype': 'int32', 'shape': (32, 16)}}], 'operands': [], 'attrs': {'shift': 150, 'reverse': False}, 'regions': []}, {'op': 'constant', 'results': [{'name': 'e25', 'type': {'dtype': 'int32', 'shape': (32, 16)}}], 'operands': [], 'attrs': {'value': 7}, 'regions': []}, {'op': 'bitand', 'results': [{'name': 'e26', 'type': {'dtype': 'int32', 'shape': (32, 16)}}], 'operands': ['e24', 'e25'], 'attrs': {}, 'regions': []}, {'op': 'constant', 'results': [{'name': 'e27', 'type': {'dtype': 'bool', 'shape': (32, 16)}}], 'operands': [], 'attrs': {'value': True}, 'regions': []}, {'op': 'atomic_min', 'results': [], 'operands': ['e26', 'e27', 'e23'], 'attrs': {'buffer': 'mem5'}, 'regions': []}, {'op': 'add', 'results': [{'name': 'e28', 'type': {'dtype': 'float32', 'shape': (32, 16)}}], 'operands': ['e21', 'e21'], 'attrs': {}, 'regions': []}, {'op': 'cast', 'results': [{'name': 'e29', 'type': {'dtype': 'float16', 'shape': (32, 16)}}], 'operands': ['e28'], 'attrs': {}, 'regions': []}, {'op': 'mul', 'results': [{'name': 'e30', 'type': {'dtype': 'float16', 'shape': (32, 16)}}], 'operands': ['e29', 'e29'], 'attrs': {}, 'regions': []}], 'returns': ['e30']}, 'buffers': [{'name': 'mem0', 'dtype': 'float32', 'size': 512, 'role': 'input', 'base': None, 'offset': 0, 'stride': 1}, {'name': 'mem1', 'dtype': 'float32', 'size': 512, 'role': 'input', 'base': None, 'offset': 0, 'stride': 1}, {'name': 'mem2', 'dtype': 'float32', 'size': 1028, 'role': 'scratch', 'base': None, 'offset': 0, 'stride': 1}, {'name': 'mem3', 'dtype': 'float32', 'size': 512, 'role': 'scratch', 'base': 'mem2', 'offset': 1, 'stride': 2}, {'name': 'mem4', 'dtype': 'float32', 'size': 512, 'role': 'scratch', 'base': 'mem2', 'offset': 2, 'stride': 1}, {'name': 'mem5', 'dtype': 'float32', 'size': 512, 'role': 'scratch', 'base': None, 'offset': 0, 'stride': 1}], 'functions': [], 'observations': ['e3', 'e13', 'e17', 'e27', 'e7'], 'blocks': 3, 'input_pattern': 'normal', 'runtime_cases': [(0, 1), (1, 511), (3, 512)], 'configuration_pair': True, 'observation_pair': True, 'pass_config_pair': False, 'fast_math_pair': False, 'precision_pair': True, 'identity_pair': True, 'family': 'indexed_memory'}

REFERENCE_PROGRAMS = {'tilelang_8_ident': {'type': 'extended', 'version': 1, 'body': {'arguments': [], 'operations': [{'op': 'index', 'results': [{'name': 'e1', 'type': {'dtype': 'int32', 'shape': (32, 16)}}], 'operands': [], 'attrs': {'shift': 119, 'reverse': False}, 'regions': []}, {'op': 'constant', 'results': [{'name': 'e2', 'type': {'dtype': 'bool', 'shape': (32, 16)}}], 'operands': [], 'attrs': {'value': True}, 'regions': []}, {'op': 'load', 'results': [{'name': 'e3', 'type': {'dtype': 'float32', 'shape': (32, 16)}}], 'operands': ['e1', 'e2'], 'attrs': {'buffer': 'mem0'}, 'regions': []}, {'op': 'add', 'results': [{'name': 'e4', 'type': {'dtype': 'float32', 'shape': (32, 16)}}], 'operands': ['e3', 'e3'], 'attrs': {}, 'regions': []}, {'op': 'mul', 'results': [{'name': 'e5_l', 'type': {'dtype': 'float32', 'shape': (32, 16)}}], 'operands': ['e4', 'e3'], 'attrs': {}, 'regions': []}, {'op': 'mul', 'results': [{'name': 'e5_r', 'type': {'dtype': 'float32', 'shape': (32, 16)}}], 'operands': ['e4', 'e3'], 'attrs': {}, 'regions': []}, {'op': 'add', 'results': [{'name': 'e5', 'type': {'dtype': 'float32', 'shape': (32, 16)}}], 'operands': ['e5_l', 'e5_r'], 'attrs': {}, 'regions': []}, {'op': 'index', 'results': [{'name': 'e6', 'type': {'dtype': 'int32', 'shape': (32, 16)}}], 'operands': [], 'attrs': {'shift': 400, 'reverse': False}, 'regions': []}, {'op': 'constant', 'results': [{'name': 'e7', 'type': {'dtype': 'bool', 'shape': (32, 16)}}], 'operands': [], 'attrs': {'value': True}, 'regions': []}, {'op': 'load', 'results': [{'name': 'e8', 'type': {'dtype': 'float32', 'shape': (32, 16)}}], 'operands': ['e6', 'e7'], 'attrs': {'buffer': 'mem1'}, 'regions': []}, {'op': 'add', 'results': [{'name': 'e9', 'type': {'dtype': 'float32', 'shape': (32, 16)}}], 'operands': ['e5', 'e8'], 'attrs': {}, 'regions': []}, {'op': 'index', 'results': [{'name': 'e10', 'type': {'dtype': 'int32', 'shape': (32, 16)}}], 'operands': [], 'attrs': {'shift': 163, 'reverse': True}, 'regions': []}, {'op': 'parameter', 'results': [{'name': 'e11', 'type': {'dtype': 'int32', 'shape': ()}}], 'operands': [], 'attrs': {'name': 'limit'}, 'regions': []}, {'op': 'lt', 'results': [{'name': 'e12', 'type': {'dtype': 'bool', 'shape': (32, 16)}}], 'operands': ['e10', 'e11'], 'attrs': {}, 'regions': []}, {'op': 'store', 'results': [], 'operands': ['e10', 'e12', 'e9'], 'attrs': {'buffer': 'mem3'}, 'regions': []}, {'op': 'barrier', 'results': [], 'operands': [], 'attrs': {}, 'regions': []}, {'op': 'load', 'results': [{'name': 'e13', 'type': {'dtype': 'float32', 'shape': (32, 16)}}], 'operands': ['e10', 'e12'], 'attrs': {'buffer': 'mem3'}, 'regions': []}, {'op': 'constant', 'results': [{'name': 'e14', 'type': {'dtype': 'float32', 'shape': (32, 16)}}], 'operands': [], 'attrs': {'value': 0.125}, 'regions': []}, {'op': 'add', 'results': [{'name': 'e15', 'type': {'dtype': 'float32', 'shape': (32, 16)}}], 'operands': ['e13', 'e14'], 'attrs': {}, 'regions': []}, {'op': 'barrier', 'results': [], 'operands': [], 'attrs': {}, 'regions': []}, {'op': 'store', 'results': [], 'operands': ['e10', 'e12', 'e15'], 'attrs': {'buffer': 'mem4'}, 'regions': []}, {'op': 'barrier', 'results': [], 'operands': [], 'attrs': {}, 'regions': []}, {'op': 'constant', 'results': [{'name': 'e16', 'type': {'dtype': 'bool', 'shape': (32, 16)}}], 'operands': [], 'attrs': {'value': True}, 'regions': []}, {'op': 'load', 'results': [{'name': 'e17', 'type': {'dtype': 'float32', 'shape': (32, 16)}}], 'operands': ['e10', 'e16'], 'attrs': {'buffer': 'mem3'}, 'regions': []}, {'op': 'constant', 'results': [{'name': 'e18', 'type': {'dtype': 'int32', 'shape': (32, 16)}}], 'operands': [], 'attrs': {'value': 3}, 'regions': []}, {'op': 'bitxor', 'results': [{'name': 'e19', 'type': {'dtype': 'int32', 'shape': (32, 16)}}], 'operands': ['e10', 'e18'], 'attrs': {}, 'regions': []}, {'op': 'load', 'results': [{'name': 'e20', 'type': {'dtype': 'float32', 'shape': (32, 16)}}], 'operands': ['e19', 'e12'], 'attrs': {'buffer': 'mem4'}, 'regions': []}, {'op': 'add', 'results': [{'name': 'e21', 'type': {'dtype': 'float32', 'shape': (32, 16)}}], 'operands': ['e17', 'e20'], 'attrs': {}, 'regions': []}, {'op': 'constant', 'results': [{'name': 'e22', 'type': {'dtype': 'float32', 'shape': ()}}], 'operands': [], 'attrs': {'value': 0}, 'regions': []}, {'op': 'broadcast', 'results': [{'name': 'e23', 'type': {'dtype': 'float32', 'shape': (32, 16)}}], 'operands': ['e22'], 'attrs': {}, 'regions': []}, {'op': 'index', 'results': [{'name': 'e24', 'type': {'dtype': 'int32', 'shape': (32, 16)}}], 'operands': [], 'attrs': {'shift': 150, 'reverse': False}, 'regions': []}, {'op': 'constant', 'results': [{'name': 'e25', 'type': {'dtype': 'int32', 'shape': (32, 16)}}], 'operands': [], 'attrs': {'value': 7}, 'regions': []}, {'op': 'bitand', 'results': [{'name': 'e26', 'type': {'dtype': 'int32', 'shape': (32, 16)}}], 'operands': ['e24', 'e25'], 'attrs': {}, 'regions': []}, {'op': 'constant', 'results': [{'name': 'e27', 'type': {'dtype': 'bool', 'shape': (32, 16)}}], 'operands': [], 'attrs': {'value': True}, 'regions': []}, {'op': 'atomic_min', 'results': [], 'operands': ['e26', 'e27', 'e23'], 'attrs': {'buffer': 'mem5'}, 'regions': []}, {'op': 'add', 'results': [{'name': 'e28', 'type': {'dtype': 'float32', 'shape': (32, 16)}}], 'operands': ['e21', 'e21'], 'attrs': {}, 'regions': []}, {'op': 'cast', 'results': [{'name': 'e29', 'type': {'dtype': 'float16', 'shape': (32, 16)}}], 'operands': ['e28'], 'attrs': {}, 'regions': []}, {'op': 'mul', 'results': [{'name': 'e30', 'type': {'dtype': 'float16', 'shape': (32, 16)}}], 'operands': ['e29', 'e29'], 'attrs': {}, 'regions': []}], 'returns': ['e30']}, 'buffers': [{'name': 'mem0', 'dtype': 'float32', 'size': 512, 'role': 'input', 'base': None, 'offset': 0, 'stride': 1}, {'name': 'mem1', 'dtype': 'float32', 'size': 512, 'role': 'input', 'base': None, 'offset': 0, 'stride': 1}, {'name': 'mem2', 'dtype': 'float32', 'size': 1028, 'role': 'scratch', 'base': None, 'offset': 0, 'stride': 1}, {'name': 'mem3', 'dtype': 'float32', 'size': 512, 'role': 'scratch', 'base': 'mem2', 'offset': 1, 'stride': 2}, {'name': 'mem4', 'dtype': 'float32', 'size': 512, 'role': 'scratch', 'base': 'mem2', 'offset': 2, 'stride': 1}, {'name': 'mem5', 'dtype': 'float32', 'size': 512, 'role': 'scratch', 'base': None, 'offset': 0, 'stride': 1}], 'functions': [], 'observations': ['e3', 'e13', 'e17', 'e27', 'e7'], 'blocks': 3, 'input_pattern': 'normal', 'runtime_cases': [(0, 1), (1, 511), (3, 512)], 'configuration_pair': True, 'observation_pair': True, 'pass_config_pair': False, 'fast_math_pair': False, 'precision_pair': True, 'identity_pair': True, 'family': 'indexed_memory'}}

if __name__ == '__main__':
    if os.environ.get('TILESMITH_COMPILE_ONLY', '0') == '1':
        prepare_extended(device_compile=False)
        extended_stage('complete', 'compile_only')
        print('COMPILE PASSED (no GPU execution)')
    else:
        run_extended(PROGRAM, prepare_extended, 0, 3, reference_programs=REFERENCE_PROGRAMS)
        extended_stage('complete', 'execute')
        print('ALL PASSED')
