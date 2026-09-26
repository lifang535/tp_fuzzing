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
    @T.macro
    def mixed_helper_extended_0_0(e6, e7, mem0, mem1, mem2, steps, limit, bid, result_e13, result_e10):
        e8 = T.alloc_fragment((16,), "float32")
        e9 = T.alloc_fragment((16,), "float32")
        e10 = T.alloc_fragment((1,), "float32")
        e11 = T.alloc_fragment((16, 1), "float32")
        e12 = T.alloc_fragment((16, 1), "float16")
        e13 = T.alloc_fragment((16, 32), "float16")
        e8_wide = T.alloc_fragment((16, 32), "float32")
        e10_wide = T.alloc_fragment((16,), "float32")
        for i, j in T.Parallel(16, 32):
            e8_wide[i, j] = T.cast(e6[i, j], 'float32')
        T.reduce_sum(e8_wide, e8, dim=1, clear=True)
        for i in T.Parallel(16):
            e9[i] = T.cast((e8[i] + e7[i]), "float32")
        for i in T.Parallel(16):
            e10_wide[i] = T.cast(e9[i], 'float32')
        T.reduce_sum(e10_wide, e10, dim=0, clear=True)
        for i, j in T.Parallel(16, 1):
            e11[i, 0] = T.cast(e9[(i * 1 + j)], "float32")
        for i, j in T.Parallel(16, 1):
            e12[i, 0] = T.cast(e11[i, 0], "float16")
        for i, j in T.Parallel(16, 32):
            e13[i, j] = T.cast((e6[i, j] + e12[i, 0]), "float16")
        for i, j in T.Parallel(16, 32):
            result_e13[i, j] = e13[i, j]
        result_e10[0] = e10[0]
    @T.prim_func
    def impl(mem0: T.Buffer((2, 544), "float16"), mem1: T.Buffer((2, 48), "float32"), mem2: T.Buffer((2, 33), "int32"), out_e73: T.Buffer((2, 544), "float16"), out_e68: T.Buffer((2, 33), "float32"), out_e69: T.Buffer((2, 33), "int32"), out_e74: T.Buffer((2, 48), "float32"), steps: T.int32, limit: T.int32):
        with T.Kernel(2, threads=128) as bid:
            e1 = T.alloc_fragment((16, 32), "int32")
            e2 = T.alloc_fragment((16, 32), "bool")
            e3 = T.alloc_fragment((16, 32), "float16")
            e4 = T.alloc_fragment((16, 32), "float16")
            e5 = T.alloc_fragment((16, 32), "float16")
            e14 = T.alloc_fragment((16,), "int32")
            e15 = T.alloc_fragment((16,), "bool")
            e16 = T.alloc_fragment((16,), "float32")
            e17 = T.alloc_fragment((16, 32), "float16")
            e18 = T.alloc_fragment((1,), "float32")
            e19 = T.alloc_fragment((1,), "int32")
            e20 = T.alloc_fragment((1,), "int32")
            e21 = T.alloc_fragment((1,), "int32")
            e22 = T.alloc_fragment((1,), "bool")
            e23 = T.alloc_fragment((1,), "int32")
            e24 = T.alloc_fragment((1,), "int32")
            e25 = T.alloc_fragment((1,), "bool")
            e45 = T.alloc_fragment((16, 32), "float16")
            e46 = T.alloc_fragment((1,), "float32")
            e47 = T.alloc_fragment((1,), "int32")
            e36 = T.alloc_fragment((16, 32), "float16")
            e31 = T.alloc_fragment((16, 32), "float16")
            e32 = T.alloc_fragment((16, 32), "float16")
            e34 = T.alloc_fragment((16, 32), "float16")
            e35 = T.alloc_fragment((16, 32), "float16")
            e37 = T.alloc_fragment((16, 32), "float16")
            e38 = T.alloc_fragment((1,), "float32")
            e39 = T.alloc_fragment((16, 32), "float16")
            e40 = T.alloc_fragment((16, 32), "float16")
            e41 = T.alloc_fragment((1,), "float32")
            e42 = T.alloc_fragment((1,), "int32")
            e43 = T.alloc_fragment((1,), "int32")
            e44 = T.alloc_fragment((1,), "int32")
            e67 = T.alloc_fragment((16, 32), "float16")
            e68 = T.alloc_fragment((1,), "float32")
            e69 = T.alloc_fragment((1,), "int32")
            e58 = T.alloc_fragment((16, 32), "float16")
            e53 = T.alloc_fragment((16, 32), "float16")
            e54 = T.alloc_fragment((16, 32), "float16")
            e56 = T.alloc_fragment((16, 32), "float16")
            e57 = T.alloc_fragment((16, 32), "float16")
            e59 = T.alloc_fragment((16, 32), "float16")
            e60 = T.alloc_fragment((1,), "float32")
            e61 = T.alloc_fragment((16, 32), "float16")
            e62 = T.alloc_fragment((16, 32), "float16")
            e63 = T.alloc_fragment((1,), "float32")
            e64 = T.alloc_fragment((1,), "int32")
            e65 = T.alloc_fragment((1,), "int32")
            e66 = T.alloc_fragment((1,), "int32")
            e70 = T.alloc_fragment((16, 32), "float16")
            e71 = T.alloc_fragment((16, 32), "float16")
            e72 = T.alloc_fragment((16, 32), "float16")
            e73 = T.alloc_fragment((16, 32), "float16")
            e74 = T.alloc_fragment((16,), "float32")
            e26 = T.alloc_fragment((1,), "int32")
            e27 = T.alloc_fragment((16, 32), "float16")
            e28 = T.alloc_fragment((1,), "float32")
            e29 = T.alloc_fragment((1,), "int32")
            e30 = T.alloc_fragment((16, 32), "float16")
            e33 = T.alloc_fragment((16, 32), "float16")
            e48 = T.alloc_fragment((1,), "int32")
            e49 = T.alloc_fragment((16, 32), "float16")
            e50 = T.alloc_fragment((1,), "float32")
            e51 = T.alloc_fragment((1,), "int32")
            e52 = T.alloc_fragment((16, 32), "float16")
            e55 = T.alloc_fragment((16, 32), "float16")
            e70_flip_shared = T.alloc_shared((16, 32), "float16")
            e74_wide = T.alloc_fragment((16, 32), "float32")
            for i, j in T.Parallel(16, 32):
                e1[i, j] = T.cast((((i * 32 + j) + 63) % 512), "int32")
            for i, j in T.Parallel(16, 32):
                e2[i, j] = T.cast(True, "bool")
            for i, j in T.Parallel(16, 32):
                e3[i, j] = T.cast(T.if_then_else((e2[i, j] and e1[i, j] >= 0 and e1[i, j] < 512), mem0[bid, 16 + e1[i, j] * 1], T.cast(0, "float16")), "float16")
            for i, j in T.Parallel(16, 32):
                e4[i, j] = T.cast((e3[i, j] * e3[i, j]), "float16")
            for i, j in T.Parallel(16, 32):
                e5[i, j] = T.cast(e4[i, j], "float16")
            for i in T.Parallel(16):
                e14[i] = T.cast(((i + 11) % 16), "int32")
            for i in T.Parallel(16):
                e15[i] = T.cast(True, "bool")
            for i in T.Parallel(16):
                e16[i] = T.cast(T.if_then_else((e15[i] and e14[i] >= 0 and e14[i] < 16), mem1[bid, 16 + e14[i] * 1], T.cast(0, "float32")), "float32")
            mixed_helper_extended_0_0(e5, e16, mem0, mem1, mem2, steps, limit, bid, e17, e18)
            e19[0] = T.cast(steps, "int32")
            e20[0] = T.cast(0, "int32")
            e21[0] = T.cast(0, "int32")
            e22[0] = T.cast(True, "bool")
            e23[0] = T.cast(T.if_then_else((e22[0] and e21[0] >= 0 and e21[0] < 1), mem2[bid, 16 + e21[0] * 1], T.cast(0, "int32")), "int32")
            e24[0] = T.cast(0, "int32")
            e25[0] = T.cast((e23[0] < e24[0]), "bool")
            for i, j in T.Parallel(16, 32):
                e45[i, j] = e17[i, j]
            e46[0] = e18[0]
            e47[0] = e20[0]
            for e45_iteration in T.serial(T.min(T.max(e19[0], 0), 4)):
                e26[0] = e45_iteration
                for i, j in T.Parallel(16, 32):
                    e27[i, j] = e45[i, j]
                e28[0] = e46[0]
                e29[0] = e47[0]
                if e25[0]:
                    for i, j in T.Parallel(16, 32):
                        e30[i, j] = e27[i, j]
                    for i, j in T.Parallel(16, 32):
                        e31[i, j] = T.cast(0.125, "float16")
                    for i, j in T.Parallel(16, 32):
                        e32[i, j] = T.cast((e30[i, j] + e31[i, j]), "float16")
                    for i, j in T.Parallel(16, 32):
                        e36[i, j] = e32[i, j]
                else:
                    for i, j in T.Parallel(16, 32):
                        e33[i, j] = e27[i, j]
                    for i, j in T.Parallel(16, 32):
                        e34[i, j] = T.cast(0.125, "float16")
                    for i, j in T.Parallel(16, 32):
                        e35[i, j] = T.cast((e33[i, j] - e34[i, j]), "float16")
                    for i, j in T.Parallel(16, 32):
                        e36[i, j] = e35[i, j]
                mixed_helper_extended_0_0(e36, e16, mem0, mem1, mem2, steps, limit, bid, e37, e38)
                for i, j in T.Parallel(16, 32):
                    e39[i, j] = T.cast(0.125, "float16")
                for i, j in T.Parallel(16, 32):
                    e40[i, j] = T.cast((e37[i, j] * e39[i, j]), "float16")
                e41[0] = T.cast((e28[0] + e38[0]), "float32")
                e42[0] = T.cast(1, "int32")
                e43[0] = T.cast((e26[0] + e42[0]), "int32")
                e44[0] = T.cast((e29[0] + e43[0]), "int32")
                for i, j in T.Parallel(16, 32):
                    e45[i, j] = e40[i, j]
                e46[0] = e41[0]
                e47[0] = e44[0]
            for i, j in T.Parallel(16, 32):
                e67[i, j] = e45[i, j]
            e68[0] = e46[0]
            e69[0] = e47[0]
            for e67_iteration in T.serial(T.min(T.max(e19[0], 0), 4)):
                e48[0] = e67_iteration
                for i, j in T.Parallel(16, 32):
                    e49[i, j] = e67[i, j]
                e50[0] = e68[0]
                e51[0] = e69[0]
                if e25[0]:
                    for i, j in T.Parallel(16, 32):
                        e52[i, j] = e49[i, j]
                    for i, j in T.Parallel(16, 32):
                        e53[i, j] = T.cast(0.125, "float16")
                    for i, j in T.Parallel(16, 32):
                        e54[i, j] = T.cast((e52[i, j] + e53[i, j]), "float16")
                    for i, j in T.Parallel(16, 32):
                        e58[i, j] = e54[i, j]
                else:
                    for i, j in T.Parallel(16, 32):
                        e55[i, j] = e49[i, j]
                    for i, j in T.Parallel(16, 32):
                        e56[i, j] = T.cast(0.125, "float16")
                    for i, j in T.Parallel(16, 32):
                        e57[i, j] = T.cast((e55[i, j] - e56[i, j]), "float16")
                    for i, j in T.Parallel(16, 32):
                        e58[i, j] = e57[i, j]
                mixed_helper_extended_0_0(e58, e16, mem0, mem1, mem2, steps, limit, bid, e59, e60)
                for i, j in T.Parallel(16, 32):
                    e61[i, j] = T.cast(0.125, "float16")
                for i, j in T.Parallel(16, 32):
                    e62[i, j] = T.cast((e59[i, j] * e61[i, j]), "float16")
                e63[0] = T.cast((e50[0] + e60[0]), "float32")
                e64[0] = T.cast(1, "int32")
                e65[0] = T.cast((e48[0] + e64[0]), "int32")
                e66[0] = T.cast((e51[0] + e65[0]), "int32")
                for i, j in T.Parallel(16, 32):
                    e67[i, j] = e62[i, j]
                e68[0] = e63[0]
                e69[0] = e66[0]
            T.copy(e67, e70_flip_shared)
            T.sync_threads()
            for i, j in T.Parallel(16, 32):
                e70[i, j] = T.cast(e70_flip_shared[i, 31 - j], "float16")
            for i, j in T.Parallel(16, 32):
                e71[i, j] = T.cast((e70[i, j] - e3[i, j]), "float16")
            for i, j in T.Parallel(16, 32):
                e72[i, j] = T.cast((e71[i, j] - e71[i, j]), "float16")
            for i, j in T.Parallel(16, 32):
                e73[i, j] = T.cast((e72[i, j] * e3[i, j]), "float16")
            for i, j in T.Parallel(16, 32):
                e74_wide[i, j] = T.cast(e73[i, j], 'float32')
            T.reduce_sum(e74_wide, e74, dim=1, clear=True)
            for i, j in T.Parallel(16, 32):
                out_e73[bid, 16 + (i * 32 + j)] = e73[i, j]
            out_e68[bid, 16 + 0] = e68[0]
            out_e69[bid, 16 + 0] = e69[0]
            for i in T.Parallel(16):
                out_e74[bid, 16 + i] = e74[i]
    return impl

@tilelang.jit
def extended_0_1():
    @T.macro
    def mixed_helper_extended_0_1(e6, e7, mem0, mem1, mem2, steps, limit, bid, result_e13, result_e10):
        e8 = T.alloc_fragment((16,), "float32")
        e9 = T.alloc_fragment((16,), "float32")
        e10 = T.alloc_fragment((1,), "float32")
        e11 = T.alloc_fragment((16, 1), "float32")
        e12 = T.alloc_fragment((16, 1), "float16")
        e13 = T.alloc_fragment((16, 32), "float16")
        e8_wide = T.alloc_fragment((16, 32), "float32")
        e10_wide = T.alloc_fragment((16,), "float32")
        for i, j in T.Parallel(16, 32):
            e8_wide[i, j] = T.cast(e6[i, j], 'float32')
        T.reduce_sum(e8_wide, e8, dim=1, clear=True)
        for i in T.Parallel(16):
            e9[i] = T.cast((e8[i] + e7[i]), "float32")
        for i in T.Parallel(16):
            e10_wide[i] = T.cast(e9[i], 'float32')
        T.reduce_sum(e10_wide, e10, dim=0, clear=True)
        for i, j in T.Parallel(16, 1):
            e11[i, 0] = T.cast(e9[(i * 1 + j)], "float32")
        for i, j in T.Parallel(16, 1):
            e12[i, 0] = T.cast(e11[i, 0], "float16")
        for i, j in T.Parallel(16, 32):
            e13[i, j] = T.cast((e6[i, j] + e12[i, 0]), "float16")
        for i, j in T.Parallel(16, 32):
            result_e13[i, j] = e13[i, j]
        result_e10[0] = e10[0]
    @T.prim_func
    def impl(mem0: T.Buffer((2, 544), "float16"), mem1: T.Buffer((2, 48), "float32"), mem2: T.Buffer((2, 33), "int32"), out_e73: T.Buffer((2, 544), "float16"), out_e68: T.Buffer((2, 33), "float32"), out_e69: T.Buffer((2, 33), "int32"), out_e74: T.Buffer((2, 48), "float32"), out_e3: T.Buffer((2, 544), "float16"), out_e70: T.Buffer((2, 544), "float16"), out_e19: T.Buffer((2, 33), "int32"), out_e22: T.Buffer((2, 33), "bool"), steps: T.int32, limit: T.int32):
        with T.Kernel(2, threads=128) as bid:
            e1 = T.alloc_fragment((16, 32), "int32")
            e2 = T.alloc_fragment((16, 32), "bool")
            e3 = T.alloc_fragment((16, 32), "float16")
            e4 = T.alloc_fragment((16, 32), "float16")
            e5 = T.alloc_fragment((16, 32), "float16")
            e14 = T.alloc_fragment((16,), "int32")
            e15 = T.alloc_fragment((16,), "bool")
            e16 = T.alloc_fragment((16,), "float32")
            e17 = T.alloc_fragment((16, 32), "float16")
            e18 = T.alloc_fragment((1,), "float32")
            e19 = T.alloc_fragment((1,), "int32")
            e20 = T.alloc_fragment((1,), "int32")
            e21 = T.alloc_fragment((1,), "int32")
            e22 = T.alloc_fragment((1,), "bool")
            e23 = T.alloc_fragment((1,), "int32")
            e24 = T.alloc_fragment((1,), "int32")
            e25 = T.alloc_fragment((1,), "bool")
            e45 = T.alloc_fragment((16, 32), "float16")
            e46 = T.alloc_fragment((1,), "float32")
            e47 = T.alloc_fragment((1,), "int32")
            e36 = T.alloc_fragment((16, 32), "float16")
            e31 = T.alloc_fragment((16, 32), "float16")
            e32 = T.alloc_fragment((16, 32), "float16")
            e34 = T.alloc_fragment((16, 32), "float16")
            e35 = T.alloc_fragment((16, 32), "float16")
            e37 = T.alloc_fragment((16, 32), "float16")
            e38 = T.alloc_fragment((1,), "float32")
            e39 = T.alloc_fragment((16, 32), "float16")
            e40 = T.alloc_fragment((16, 32), "float16")
            e41 = T.alloc_fragment((1,), "float32")
            e42 = T.alloc_fragment((1,), "int32")
            e43 = T.alloc_fragment((1,), "int32")
            e44 = T.alloc_fragment((1,), "int32")
            e67 = T.alloc_fragment((16, 32), "float16")
            e68 = T.alloc_fragment((1,), "float32")
            e69 = T.alloc_fragment((1,), "int32")
            e58 = T.alloc_fragment((16, 32), "float16")
            e53 = T.alloc_fragment((16, 32), "float16")
            e54 = T.alloc_fragment((16, 32), "float16")
            e56 = T.alloc_fragment((16, 32), "float16")
            e57 = T.alloc_fragment((16, 32), "float16")
            e59 = T.alloc_fragment((16, 32), "float16")
            e60 = T.alloc_fragment((1,), "float32")
            e61 = T.alloc_fragment((16, 32), "float16")
            e62 = T.alloc_fragment((16, 32), "float16")
            e63 = T.alloc_fragment((1,), "float32")
            e64 = T.alloc_fragment((1,), "int32")
            e65 = T.alloc_fragment((1,), "int32")
            e66 = T.alloc_fragment((1,), "int32")
            e70 = T.alloc_fragment((16, 32), "float16")
            e71 = T.alloc_fragment((16, 32), "float16")
            e72 = T.alloc_fragment((16, 32), "float16")
            e73 = T.alloc_fragment((16, 32), "float16")
            e74 = T.alloc_fragment((16,), "float32")
            e26 = T.alloc_fragment((1,), "int32")
            e27 = T.alloc_fragment((16, 32), "float16")
            e28 = T.alloc_fragment((1,), "float32")
            e29 = T.alloc_fragment((1,), "int32")
            e30 = T.alloc_fragment((16, 32), "float16")
            e33 = T.alloc_fragment((16, 32), "float16")
            e48 = T.alloc_fragment((1,), "int32")
            e49 = T.alloc_fragment((16, 32), "float16")
            e50 = T.alloc_fragment((1,), "float32")
            e51 = T.alloc_fragment((1,), "int32")
            e52 = T.alloc_fragment((16, 32), "float16")
            e55 = T.alloc_fragment((16, 32), "float16")
            e70_flip_shared = T.alloc_shared((16, 32), "float16")
            e74_wide = T.alloc_fragment((16, 32), "float32")
            for i, j in T.Parallel(16, 32):
                e1[i, j] = T.cast((((i * 32 + j) + 63) % 512), "int32")
            for i, j in T.Parallel(16, 32):
                e2[i, j] = T.cast(True, "bool")
            for i, j in T.Parallel(16, 32):
                e3[i, j] = T.cast(T.if_then_else((e2[i, j] and e1[i, j] >= 0 and e1[i, j] < 512), mem0[bid, 16 + e1[i, j] * 1], T.cast(0, "float16")), "float16")
            for i, j in T.Parallel(16, 32):
                e4[i, j] = T.cast((e3[i, j] * e3[i, j]), "float16")
            for i, j in T.Parallel(16, 32):
                e5[i, j] = T.cast(e4[i, j], "float16")
            for i in T.Parallel(16):
                e14[i] = T.cast(((i + 11) % 16), "int32")
            for i in T.Parallel(16):
                e15[i] = T.cast(True, "bool")
            for i in T.Parallel(16):
                e16[i] = T.cast(T.if_then_else((e15[i] and e14[i] >= 0 and e14[i] < 16), mem1[bid, 16 + e14[i] * 1], T.cast(0, "float32")), "float32")
            mixed_helper_extended_0_1(e5, e16, mem0, mem1, mem2, steps, limit, bid, e17, e18)
            e19[0] = T.cast(steps, "int32")
            e20[0] = T.cast(0, "int32")
            e21[0] = T.cast(0, "int32")
            e22[0] = T.cast(True, "bool")
            e23[0] = T.cast(T.if_then_else((e22[0] and e21[0] >= 0 and e21[0] < 1), mem2[bid, 16 + e21[0] * 1], T.cast(0, "int32")), "int32")
            e24[0] = T.cast(0, "int32")
            e25[0] = T.cast((e23[0] < e24[0]), "bool")
            for i, j in T.Parallel(16, 32):
                e45[i, j] = e17[i, j]
            e46[0] = e18[0]
            e47[0] = e20[0]
            for e45_iteration in T.serial(T.min(T.max(e19[0], 0), 4)):
                e26[0] = e45_iteration
                for i, j in T.Parallel(16, 32):
                    e27[i, j] = e45[i, j]
                e28[0] = e46[0]
                e29[0] = e47[0]
                if e25[0]:
                    for i, j in T.Parallel(16, 32):
                        e30[i, j] = e27[i, j]
                    for i, j in T.Parallel(16, 32):
                        e31[i, j] = T.cast(0.125, "float16")
                    for i, j in T.Parallel(16, 32):
                        e32[i, j] = T.cast((e30[i, j] + e31[i, j]), "float16")
                    for i, j in T.Parallel(16, 32):
                        e36[i, j] = e32[i, j]
                else:
                    for i, j in T.Parallel(16, 32):
                        e33[i, j] = e27[i, j]
                    for i, j in T.Parallel(16, 32):
                        e34[i, j] = T.cast(0.125, "float16")
                    for i, j in T.Parallel(16, 32):
                        e35[i, j] = T.cast((e33[i, j] - e34[i, j]), "float16")
                    for i, j in T.Parallel(16, 32):
                        e36[i, j] = e35[i, j]
                mixed_helper_extended_0_1(e36, e16, mem0, mem1, mem2, steps, limit, bid, e37, e38)
                for i, j in T.Parallel(16, 32):
                    e39[i, j] = T.cast(0.125, "float16")
                for i, j in T.Parallel(16, 32):
                    e40[i, j] = T.cast((e37[i, j] * e39[i, j]), "float16")
                e41[0] = T.cast((e28[0] + e38[0]), "float32")
                e42[0] = T.cast(1, "int32")
                e43[0] = T.cast((e26[0] + e42[0]), "int32")
                e44[0] = T.cast((e29[0] + e43[0]), "int32")
                for i, j in T.Parallel(16, 32):
                    e45[i, j] = e40[i, j]
                e46[0] = e41[0]
                e47[0] = e44[0]
            for i, j in T.Parallel(16, 32):
                e67[i, j] = e45[i, j]
            e68[0] = e46[0]
            e69[0] = e47[0]
            for e67_iteration in T.serial(T.min(T.max(e19[0], 0), 4)):
                e48[0] = e67_iteration
                for i, j in T.Parallel(16, 32):
                    e49[i, j] = e67[i, j]
                e50[0] = e68[0]
                e51[0] = e69[0]
                if e25[0]:
                    for i, j in T.Parallel(16, 32):
                        e52[i, j] = e49[i, j]
                    for i, j in T.Parallel(16, 32):
                        e53[i, j] = T.cast(0.125, "float16")
                    for i, j in T.Parallel(16, 32):
                        e54[i, j] = T.cast((e52[i, j] + e53[i, j]), "float16")
                    for i, j in T.Parallel(16, 32):
                        e58[i, j] = e54[i, j]
                else:
                    for i, j in T.Parallel(16, 32):
                        e55[i, j] = e49[i, j]
                    for i, j in T.Parallel(16, 32):
                        e56[i, j] = T.cast(0.125, "float16")
                    for i, j in T.Parallel(16, 32):
                        e57[i, j] = T.cast((e55[i, j] - e56[i, j]), "float16")
                    for i, j in T.Parallel(16, 32):
                        e58[i, j] = e57[i, j]
                mixed_helper_extended_0_1(e58, e16, mem0, mem1, mem2, steps, limit, bid, e59, e60)
                for i, j in T.Parallel(16, 32):
                    e61[i, j] = T.cast(0.125, "float16")
                for i, j in T.Parallel(16, 32):
                    e62[i, j] = T.cast((e59[i, j] * e61[i, j]), "float16")
                e63[0] = T.cast((e50[0] + e60[0]), "float32")
                e64[0] = T.cast(1, "int32")
                e65[0] = T.cast((e48[0] + e64[0]), "int32")
                e66[0] = T.cast((e51[0] + e65[0]), "int32")
                for i, j in T.Parallel(16, 32):
                    e67[i, j] = e62[i, j]
                e68[0] = e63[0]
                e69[0] = e66[0]
            T.copy(e67, e70_flip_shared)
            T.sync_threads()
            for i, j in T.Parallel(16, 32):
                e70[i, j] = T.cast(e70_flip_shared[i, 31 - j], "float16")
            for i, j in T.Parallel(16, 32):
                e71[i, j] = T.cast((e70[i, j] - e3[i, j]), "float16")
            for i, j in T.Parallel(16, 32):
                e72[i, j] = T.cast((e71[i, j] - e71[i, j]), "float16")
            for i, j in T.Parallel(16, 32):
                e73[i, j] = T.cast((e72[i, j] * e3[i, j]), "float16")
            for i, j in T.Parallel(16, 32):
                e74_wide[i, j] = T.cast(e73[i, j], 'float32')
            T.reduce_sum(e74_wide, e74, dim=1, clear=True)
            for i, j in T.Parallel(16, 32):
                out_e73[bid, 16 + (i * 32 + j)] = e73[i, j]
            out_e68[bid, 16 + 0] = e68[0]
            out_e69[bid, 16 + 0] = e69[0]
            for i in T.Parallel(16):
                out_e74[bid, 16 + i] = e74[i]
            for i, j in T.Parallel(16, 32):
                out_e3[bid, 16 + (i * 32 + j)] = e3[i, j]
            for i, j in T.Parallel(16, 32):
                out_e70[bid, 16 + (i * 32 + j)] = e70[i, j]
            out_e19[bid, 16 + 0] = e19[0]
            out_e22[bid, 16 + 0] = e22[0]
    return impl

@tilelang.jit
def extended_1_0():
    @T.macro
    def mixed_helper_extended_1_0(e6, e7, mem0, mem1, mem2, steps, limit, bid, result_e13, result_e10):
        e8 = T.alloc_fragment((16,), "float32")
        e9 = T.alloc_fragment((16,), "float32")
        e10 = T.alloc_fragment((1,), "float32")
        e11 = T.alloc_fragment((16, 1), "float32")
        e12 = T.alloc_fragment((16, 1), "float16")
        e13 = T.alloc_fragment((16, 32), "float16")
        e8_wide = T.alloc_fragment((16, 32), "float32")
        e10_wide = T.alloc_fragment((16,), "float32")
        for i, j in T.Parallel(16, 32):
            e8_wide[i, j] = T.cast(e6[i, j], 'float32')
        T.reduce_sum(e8_wide, e8, dim=1, clear=True)
        for i in T.Parallel(16):
            e9[i] = T.cast((e8[i] + e7[i]), "float32")
        for i in T.Parallel(16):
            e10_wide[i] = T.cast(e9[i], 'float32')
        T.reduce_sum(e10_wide, e10, dim=0, clear=True)
        for i, j in T.Parallel(16, 1):
            e11[i, 0] = T.cast(e9[(i * 1 + j)], "float32")
        for i, j in T.Parallel(16, 1):
            e12[i, 0] = T.cast(e11[i, 0], "float16")
        for i, j in T.Parallel(16, 32):
            e13[i, j] = T.cast((e6[i, j] + e12[i, 0]), "float16")
        for i, j in T.Parallel(16, 32):
            result_e13[i, j] = e13[i, j]
        result_e10[0] = e10[0]
    @T.prim_func
    def impl(mem0: T.Buffer((2, 544), "float16"), mem1: T.Buffer((2, 48), "float32"), mem2: T.Buffer((2, 33), "int32"), out_e73: T.Buffer((2, 544), "float16"), out_e68: T.Buffer((2, 33), "float32"), out_e69: T.Buffer((2, 33), "int32"), out_e74: T.Buffer((2, 48), "float32"), steps: T.int32, limit: T.int32):
        with T.Kernel(2, threads=256) as bid:
            e1 = T.alloc_fragment((16, 32), "int32")
            e2 = T.alloc_fragment((16, 32), "bool")
            e3 = T.alloc_fragment((16, 32), "float16")
            e4 = T.alloc_fragment((16, 32), "float16")
            e5 = T.alloc_fragment((16, 32), "float16")
            e14 = T.alloc_fragment((16,), "int32")
            e15 = T.alloc_fragment((16,), "bool")
            e16 = T.alloc_fragment((16,), "float32")
            e17 = T.alloc_fragment((16, 32), "float16")
            e18 = T.alloc_fragment((1,), "float32")
            e19 = T.alloc_fragment((1,), "int32")
            e20 = T.alloc_fragment((1,), "int32")
            e21 = T.alloc_fragment((1,), "int32")
            e22 = T.alloc_fragment((1,), "bool")
            e23 = T.alloc_fragment((1,), "int32")
            e24 = T.alloc_fragment((1,), "int32")
            e25 = T.alloc_fragment((1,), "bool")
            e45 = T.alloc_fragment((16, 32), "float16")
            e46 = T.alloc_fragment((1,), "float32")
            e47 = T.alloc_fragment((1,), "int32")
            e36 = T.alloc_fragment((16, 32), "float16")
            e31 = T.alloc_fragment((16, 32), "float16")
            e32 = T.alloc_fragment((16, 32), "float16")
            e34 = T.alloc_fragment((16, 32), "float16")
            e35 = T.alloc_fragment((16, 32), "float16")
            e37 = T.alloc_fragment((16, 32), "float16")
            e38 = T.alloc_fragment((1,), "float32")
            e39 = T.alloc_fragment((16, 32), "float16")
            e40 = T.alloc_fragment((16, 32), "float16")
            e41 = T.alloc_fragment((1,), "float32")
            e42 = T.alloc_fragment((1,), "int32")
            e43 = T.alloc_fragment((1,), "int32")
            e44 = T.alloc_fragment((1,), "int32")
            e67 = T.alloc_fragment((16, 32), "float16")
            e68 = T.alloc_fragment((1,), "float32")
            e69 = T.alloc_fragment((1,), "int32")
            e58 = T.alloc_fragment((16, 32), "float16")
            e53 = T.alloc_fragment((16, 32), "float16")
            e54 = T.alloc_fragment((16, 32), "float16")
            e56 = T.alloc_fragment((16, 32), "float16")
            e57 = T.alloc_fragment((16, 32), "float16")
            e59 = T.alloc_fragment((16, 32), "float16")
            e60 = T.alloc_fragment((1,), "float32")
            e61 = T.alloc_fragment((16, 32), "float16")
            e62 = T.alloc_fragment((16, 32), "float16")
            e63 = T.alloc_fragment((1,), "float32")
            e64 = T.alloc_fragment((1,), "int32")
            e65 = T.alloc_fragment((1,), "int32")
            e66 = T.alloc_fragment((1,), "int32")
            e70 = T.alloc_fragment((16, 32), "float16")
            e71 = T.alloc_fragment((16, 32), "float16")
            e72 = T.alloc_fragment((16, 32), "float16")
            e73 = T.alloc_fragment((16, 32), "float16")
            e74 = T.alloc_fragment((16,), "float32")
            e26 = T.alloc_fragment((1,), "int32")
            e27 = T.alloc_fragment((16, 32), "float16")
            e28 = T.alloc_fragment((1,), "float32")
            e29 = T.alloc_fragment((1,), "int32")
            e30 = T.alloc_fragment((16, 32), "float16")
            e33 = T.alloc_fragment((16, 32), "float16")
            e48 = T.alloc_fragment((1,), "int32")
            e49 = T.alloc_fragment((16, 32), "float16")
            e50 = T.alloc_fragment((1,), "float32")
            e51 = T.alloc_fragment((1,), "int32")
            e52 = T.alloc_fragment((16, 32), "float16")
            e55 = T.alloc_fragment((16, 32), "float16")
            e70_flip_shared = T.alloc_shared((16, 32), "float16")
            e74_wide = T.alloc_fragment((16, 32), "float32")
            for i, j in T.Parallel(16, 32):
                e1[i, j] = T.cast((((i * 32 + j) + 63) % 512), "int32")
            for i, j in T.Parallel(16, 32):
                e2[i, j] = T.cast(True, "bool")
            for i, j in T.Parallel(16, 32):
                e3[i, j] = T.cast(T.if_then_else((e2[i, j] and e1[i, j] >= 0 and e1[i, j] < 512), mem0[bid, 16 + e1[i, j] * 1], T.cast(0, "float16")), "float16")
            for i, j in T.Parallel(16, 32):
                e4[i, j] = T.cast((e3[i, j] * e3[i, j]), "float16")
            for i, j in T.Parallel(16, 32):
                e5[i, j] = T.cast(e4[i, j], "float16")
            for i in T.Parallel(16):
                e14[i] = T.cast(((i + 11) % 16), "int32")
            for i in T.Parallel(16):
                e15[i] = T.cast(True, "bool")
            for i in T.Parallel(16):
                e16[i] = T.cast(T.if_then_else((e15[i] and e14[i] >= 0 and e14[i] < 16), mem1[bid, 16 + e14[i] * 1], T.cast(0, "float32")), "float32")
            mixed_helper_extended_1_0(e5, e16, mem0, mem1, mem2, steps, limit, bid, e17, e18)
            e19[0] = T.cast(steps, "int32")
            e20[0] = T.cast(0, "int32")
            e21[0] = T.cast(0, "int32")
            e22[0] = T.cast(True, "bool")
            e23[0] = T.cast(T.if_then_else((e22[0] and e21[0] >= 0 and e21[0] < 1), mem2[bid, 16 + e21[0] * 1], T.cast(0, "int32")), "int32")
            e24[0] = T.cast(0, "int32")
            e25[0] = T.cast((e23[0] < e24[0]), "bool")
            for i, j in T.Parallel(16, 32):
                e45[i, j] = e17[i, j]
            e46[0] = e18[0]
            e47[0] = e20[0]
            for e45_iteration in T.serial(T.min(T.max(e19[0], 0), 4)):
                e26[0] = e45_iteration
                for i, j in T.Parallel(16, 32):
                    e27[i, j] = e45[i, j]
                e28[0] = e46[0]
                e29[0] = e47[0]
                if e25[0]:
                    for i, j in T.Parallel(16, 32):
                        e30[i, j] = e27[i, j]
                    for i, j in T.Parallel(16, 32):
                        e31[i, j] = T.cast(0.125, "float16")
                    for i, j in T.Parallel(16, 32):
                        e32[i, j] = T.cast((e30[i, j] + e31[i, j]), "float16")
                    for i, j in T.Parallel(16, 32):
                        e36[i, j] = e32[i, j]
                else:
                    for i, j in T.Parallel(16, 32):
                        e33[i, j] = e27[i, j]
                    for i, j in T.Parallel(16, 32):
                        e34[i, j] = T.cast(0.125, "float16")
                    for i, j in T.Parallel(16, 32):
                        e35[i, j] = T.cast((e33[i, j] - e34[i, j]), "float16")
                    for i, j in T.Parallel(16, 32):
                        e36[i, j] = e35[i, j]
                mixed_helper_extended_1_0(e36, e16, mem0, mem1, mem2, steps, limit, bid, e37, e38)
                for i, j in T.Parallel(16, 32):
                    e39[i, j] = T.cast(0.125, "float16")
                for i, j in T.Parallel(16, 32):
                    e40[i, j] = T.cast((e37[i, j] * e39[i, j]), "float16")
                e41[0] = T.cast((e28[0] + e38[0]), "float32")
                e42[0] = T.cast(1, "int32")
                e43[0] = T.cast((e26[0] + e42[0]), "int32")
                e44[0] = T.cast((e29[0] + e43[0]), "int32")
                for i, j in T.Parallel(16, 32):
                    e45[i, j] = e40[i, j]
                e46[0] = e41[0]
                e47[0] = e44[0]
            for i, j in T.Parallel(16, 32):
                e67[i, j] = e45[i, j]
            e68[0] = e46[0]
            e69[0] = e47[0]
            for e67_iteration in T.serial(T.min(T.max(e19[0], 0), 4)):
                e48[0] = e67_iteration
                for i, j in T.Parallel(16, 32):
                    e49[i, j] = e67[i, j]
                e50[0] = e68[0]
                e51[0] = e69[0]
                if e25[0]:
                    for i, j in T.Parallel(16, 32):
                        e52[i, j] = e49[i, j]
                    for i, j in T.Parallel(16, 32):
                        e53[i, j] = T.cast(0.125, "float16")
                    for i, j in T.Parallel(16, 32):
                        e54[i, j] = T.cast((e52[i, j] + e53[i, j]), "float16")
                    for i, j in T.Parallel(16, 32):
                        e58[i, j] = e54[i, j]
                else:
                    for i, j in T.Parallel(16, 32):
                        e55[i, j] = e49[i, j]
                    for i, j in T.Parallel(16, 32):
                        e56[i, j] = T.cast(0.125, "float16")
                    for i, j in T.Parallel(16, 32):
                        e57[i, j] = T.cast((e55[i, j] - e56[i, j]), "float16")
                    for i, j in T.Parallel(16, 32):
                        e58[i, j] = e57[i, j]
                mixed_helper_extended_1_0(e58, e16, mem0, mem1, mem2, steps, limit, bid, e59, e60)
                for i, j in T.Parallel(16, 32):
                    e61[i, j] = T.cast(0.125, "float16")
                for i, j in T.Parallel(16, 32):
                    e62[i, j] = T.cast((e59[i, j] * e61[i, j]), "float16")
                e63[0] = T.cast((e50[0] + e60[0]), "float32")
                e64[0] = T.cast(1, "int32")
                e65[0] = T.cast((e48[0] + e64[0]), "int32")
                e66[0] = T.cast((e51[0] + e65[0]), "int32")
                for i, j in T.Parallel(16, 32):
                    e67[i, j] = e62[i, j]
                e68[0] = e63[0]
                e69[0] = e66[0]
            T.copy(e67, e70_flip_shared)
            T.sync_threads()
            for i, j in T.Parallel(16, 32):
                e70[i, j] = T.cast(e70_flip_shared[i, 31 - j], "float16")
            for i, j in T.Parallel(16, 32):
                e71[i, j] = T.cast((e70[i, j] - e3[i, j]), "float16")
            for i, j in T.Parallel(16, 32):
                e72[i, j] = T.cast((e71[i, j] - e71[i, j]), "float16")
            for i, j in T.Parallel(16, 32):
                e73[i, j] = T.cast((e72[i, j] * e3[i, j]), "float16")
            for i, j in T.Parallel(16, 32):
                e74_wide[i, j] = T.cast(e73[i, j], 'float32')
            T.reduce_sum(e74_wide, e74, dim=1, clear=True)
            for i, j in T.Parallel(16, 32):
                out_e73[bid, 16 + (i * 32 + j)] = e73[i, j]
            out_e68[bid, 16 + 0] = e68[0]
            out_e69[bid, 16 + 0] = e69[0]
            for i in T.Parallel(16):
                out_e74[bid, 16 + i] = e74[i]
    return impl

@tilelang.jit
def extended_1_1():
    @T.macro
    def mixed_helper_extended_1_1(e6, e7, mem0, mem1, mem2, steps, limit, bid, result_e13, result_e10):
        e8 = T.alloc_fragment((16,), "float32")
        e9 = T.alloc_fragment((16,), "float32")
        e10 = T.alloc_fragment((1,), "float32")
        e11 = T.alloc_fragment((16, 1), "float32")
        e12 = T.alloc_fragment((16, 1), "float16")
        e13 = T.alloc_fragment((16, 32), "float16")
        e8_wide = T.alloc_fragment((16, 32), "float32")
        e10_wide = T.alloc_fragment((16,), "float32")
        for i, j in T.Parallel(16, 32):
            e8_wide[i, j] = T.cast(e6[i, j], 'float32')
        T.reduce_sum(e8_wide, e8, dim=1, clear=True)
        for i in T.Parallel(16):
            e9[i] = T.cast((e8[i] + e7[i]), "float32")
        for i in T.Parallel(16):
            e10_wide[i] = T.cast(e9[i], 'float32')
        T.reduce_sum(e10_wide, e10, dim=0, clear=True)
        for i, j in T.Parallel(16, 1):
            e11[i, 0] = T.cast(e9[(i * 1 + j)], "float32")
        for i, j in T.Parallel(16, 1):
            e12[i, 0] = T.cast(e11[i, 0], "float16")
        for i, j in T.Parallel(16, 32):
            e13[i, j] = T.cast((e6[i, j] + e12[i, 0]), "float16")
        for i, j in T.Parallel(16, 32):
            result_e13[i, j] = e13[i, j]
        result_e10[0] = e10[0]
    @T.prim_func
    def impl(mem0: T.Buffer((2, 544), "float16"), mem1: T.Buffer((2, 48), "float32"), mem2: T.Buffer((2, 33), "int32"), out_e73: T.Buffer((2, 544), "float16"), out_e68: T.Buffer((2, 33), "float32"), out_e69: T.Buffer((2, 33), "int32"), out_e74: T.Buffer((2, 48), "float32"), out_e3: T.Buffer((2, 544), "float16"), out_e70: T.Buffer((2, 544), "float16"), out_e19: T.Buffer((2, 33), "int32"), out_e22: T.Buffer((2, 33), "bool"), steps: T.int32, limit: T.int32):
        with T.Kernel(2, threads=256) as bid:
            e1 = T.alloc_fragment((16, 32), "int32")
            e2 = T.alloc_fragment((16, 32), "bool")
            e3 = T.alloc_fragment((16, 32), "float16")
            e4 = T.alloc_fragment((16, 32), "float16")
            e5 = T.alloc_fragment((16, 32), "float16")
            e14 = T.alloc_fragment((16,), "int32")
            e15 = T.alloc_fragment((16,), "bool")
            e16 = T.alloc_fragment((16,), "float32")
            e17 = T.alloc_fragment((16, 32), "float16")
            e18 = T.alloc_fragment((1,), "float32")
            e19 = T.alloc_fragment((1,), "int32")
            e20 = T.alloc_fragment((1,), "int32")
            e21 = T.alloc_fragment((1,), "int32")
            e22 = T.alloc_fragment((1,), "bool")
            e23 = T.alloc_fragment((1,), "int32")
            e24 = T.alloc_fragment((1,), "int32")
            e25 = T.alloc_fragment((1,), "bool")
            e45 = T.alloc_fragment((16, 32), "float16")
            e46 = T.alloc_fragment((1,), "float32")
            e47 = T.alloc_fragment((1,), "int32")
            e36 = T.alloc_fragment((16, 32), "float16")
            e31 = T.alloc_fragment((16, 32), "float16")
            e32 = T.alloc_fragment((16, 32), "float16")
            e34 = T.alloc_fragment((16, 32), "float16")
            e35 = T.alloc_fragment((16, 32), "float16")
            e37 = T.alloc_fragment((16, 32), "float16")
            e38 = T.alloc_fragment((1,), "float32")
            e39 = T.alloc_fragment((16, 32), "float16")
            e40 = T.alloc_fragment((16, 32), "float16")
            e41 = T.alloc_fragment((1,), "float32")
            e42 = T.alloc_fragment((1,), "int32")
            e43 = T.alloc_fragment((1,), "int32")
            e44 = T.alloc_fragment((1,), "int32")
            e67 = T.alloc_fragment((16, 32), "float16")
            e68 = T.alloc_fragment((1,), "float32")
            e69 = T.alloc_fragment((1,), "int32")
            e58 = T.alloc_fragment((16, 32), "float16")
            e53 = T.alloc_fragment((16, 32), "float16")
            e54 = T.alloc_fragment((16, 32), "float16")
            e56 = T.alloc_fragment((16, 32), "float16")
            e57 = T.alloc_fragment((16, 32), "float16")
            e59 = T.alloc_fragment((16, 32), "float16")
            e60 = T.alloc_fragment((1,), "float32")
            e61 = T.alloc_fragment((16, 32), "float16")
            e62 = T.alloc_fragment((16, 32), "float16")
            e63 = T.alloc_fragment((1,), "float32")
            e64 = T.alloc_fragment((1,), "int32")
            e65 = T.alloc_fragment((1,), "int32")
            e66 = T.alloc_fragment((1,), "int32")
            e70 = T.alloc_fragment((16, 32), "float16")
            e71 = T.alloc_fragment((16, 32), "float16")
            e72 = T.alloc_fragment((16, 32), "float16")
            e73 = T.alloc_fragment((16, 32), "float16")
            e74 = T.alloc_fragment((16,), "float32")
            e26 = T.alloc_fragment((1,), "int32")
            e27 = T.alloc_fragment((16, 32), "float16")
            e28 = T.alloc_fragment((1,), "float32")
            e29 = T.alloc_fragment((1,), "int32")
            e30 = T.alloc_fragment((16, 32), "float16")
            e33 = T.alloc_fragment((16, 32), "float16")
            e48 = T.alloc_fragment((1,), "int32")
            e49 = T.alloc_fragment((16, 32), "float16")
            e50 = T.alloc_fragment((1,), "float32")
            e51 = T.alloc_fragment((1,), "int32")
            e52 = T.alloc_fragment((16, 32), "float16")
            e55 = T.alloc_fragment((16, 32), "float16")
            e70_flip_shared = T.alloc_shared((16, 32), "float16")
            e74_wide = T.alloc_fragment((16, 32), "float32")
            for i, j in T.Parallel(16, 32):
                e1[i, j] = T.cast((((i * 32 + j) + 63) % 512), "int32")
            for i, j in T.Parallel(16, 32):
                e2[i, j] = T.cast(True, "bool")
            for i, j in T.Parallel(16, 32):
                e3[i, j] = T.cast(T.if_then_else((e2[i, j] and e1[i, j] >= 0 and e1[i, j] < 512), mem0[bid, 16 + e1[i, j] * 1], T.cast(0, "float16")), "float16")
            for i, j in T.Parallel(16, 32):
                e4[i, j] = T.cast((e3[i, j] * e3[i, j]), "float16")
            for i, j in T.Parallel(16, 32):
                e5[i, j] = T.cast(e4[i, j], "float16")
            for i in T.Parallel(16):
                e14[i] = T.cast(((i + 11) % 16), "int32")
            for i in T.Parallel(16):
                e15[i] = T.cast(True, "bool")
            for i in T.Parallel(16):
                e16[i] = T.cast(T.if_then_else((e15[i] and e14[i] >= 0 and e14[i] < 16), mem1[bid, 16 + e14[i] * 1], T.cast(0, "float32")), "float32")
            mixed_helper_extended_1_1(e5, e16, mem0, mem1, mem2, steps, limit, bid, e17, e18)
            e19[0] = T.cast(steps, "int32")
            e20[0] = T.cast(0, "int32")
            e21[0] = T.cast(0, "int32")
            e22[0] = T.cast(True, "bool")
            e23[0] = T.cast(T.if_then_else((e22[0] and e21[0] >= 0 and e21[0] < 1), mem2[bid, 16 + e21[0] * 1], T.cast(0, "int32")), "int32")
            e24[0] = T.cast(0, "int32")
            e25[0] = T.cast((e23[0] < e24[0]), "bool")
            for i, j in T.Parallel(16, 32):
                e45[i, j] = e17[i, j]
            e46[0] = e18[0]
            e47[0] = e20[0]
            for e45_iteration in T.serial(T.min(T.max(e19[0], 0), 4)):
                e26[0] = e45_iteration
                for i, j in T.Parallel(16, 32):
                    e27[i, j] = e45[i, j]
                e28[0] = e46[0]
                e29[0] = e47[0]
                if e25[0]:
                    for i, j in T.Parallel(16, 32):
                        e30[i, j] = e27[i, j]
                    for i, j in T.Parallel(16, 32):
                        e31[i, j] = T.cast(0.125, "float16")
                    for i, j in T.Parallel(16, 32):
                        e32[i, j] = T.cast((e30[i, j] + e31[i, j]), "float16")
                    for i, j in T.Parallel(16, 32):
                        e36[i, j] = e32[i, j]
                else:
                    for i, j in T.Parallel(16, 32):
                        e33[i, j] = e27[i, j]
                    for i, j in T.Parallel(16, 32):
                        e34[i, j] = T.cast(0.125, "float16")
                    for i, j in T.Parallel(16, 32):
                        e35[i, j] = T.cast((e33[i, j] - e34[i, j]), "float16")
                    for i, j in T.Parallel(16, 32):
                        e36[i, j] = e35[i, j]
                mixed_helper_extended_1_1(e36, e16, mem0, mem1, mem2, steps, limit, bid, e37, e38)
                for i, j in T.Parallel(16, 32):
                    e39[i, j] = T.cast(0.125, "float16")
                for i, j in T.Parallel(16, 32):
                    e40[i, j] = T.cast((e37[i, j] * e39[i, j]), "float16")
                e41[0] = T.cast((e28[0] + e38[0]), "float32")
                e42[0] = T.cast(1, "int32")
                e43[0] = T.cast((e26[0] + e42[0]), "int32")
                e44[0] = T.cast((e29[0] + e43[0]), "int32")
                for i, j in T.Parallel(16, 32):
                    e45[i, j] = e40[i, j]
                e46[0] = e41[0]
                e47[0] = e44[0]
            for i, j in T.Parallel(16, 32):
                e67[i, j] = e45[i, j]
            e68[0] = e46[0]
            e69[0] = e47[0]
            for e67_iteration in T.serial(T.min(T.max(e19[0], 0), 4)):
                e48[0] = e67_iteration
                for i, j in T.Parallel(16, 32):
                    e49[i, j] = e67[i, j]
                e50[0] = e68[0]
                e51[0] = e69[0]
                if e25[0]:
                    for i, j in T.Parallel(16, 32):
                        e52[i, j] = e49[i, j]
                    for i, j in T.Parallel(16, 32):
                        e53[i, j] = T.cast(0.125, "float16")
                    for i, j in T.Parallel(16, 32):
                        e54[i, j] = T.cast((e52[i, j] + e53[i, j]), "float16")
                    for i, j in T.Parallel(16, 32):
                        e58[i, j] = e54[i, j]
                else:
                    for i, j in T.Parallel(16, 32):
                        e55[i, j] = e49[i, j]
                    for i, j in T.Parallel(16, 32):
                        e56[i, j] = T.cast(0.125, "float16")
                    for i, j in T.Parallel(16, 32):
                        e57[i, j] = T.cast((e55[i, j] - e56[i, j]), "float16")
                    for i, j in T.Parallel(16, 32):
                        e58[i, j] = e57[i, j]
                mixed_helper_extended_1_1(e58, e16, mem0, mem1, mem2, steps, limit, bid, e59, e60)
                for i, j in T.Parallel(16, 32):
                    e61[i, j] = T.cast(0.125, "float16")
                for i, j in T.Parallel(16, 32):
                    e62[i, j] = T.cast((e59[i, j] * e61[i, j]), "float16")
                e63[0] = T.cast((e50[0] + e60[0]), "float32")
                e64[0] = T.cast(1, "int32")
                e65[0] = T.cast((e48[0] + e64[0]), "int32")
                e66[0] = T.cast((e51[0] + e65[0]), "int32")
                for i, j in T.Parallel(16, 32):
                    e67[i, j] = e62[i, j]
                e68[0] = e63[0]
                e69[0] = e66[0]
            T.copy(e67, e70_flip_shared)
            T.sync_threads()
            for i, j in T.Parallel(16, 32):
                e70[i, j] = T.cast(e70_flip_shared[i, 31 - j], "float16")
            for i, j in T.Parallel(16, 32):
                e71[i, j] = T.cast((e70[i, j] - e3[i, j]), "float16")
            for i, j in T.Parallel(16, 32):
                e72[i, j] = T.cast((e71[i, j] - e71[i, j]), "float16")
            for i, j in T.Parallel(16, 32):
                e73[i, j] = T.cast((e72[i, j] * e3[i, j]), "float16")
            for i, j in T.Parallel(16, 32):
                e74_wide[i, j] = T.cast(e73[i, j], 'float32')
            T.reduce_sum(e74_wide, e74, dim=1, clear=True)
            for i, j in T.Parallel(16, 32):
                out_e73[bid, 16 + (i * 32 + j)] = e73[i, j]
            out_e68[bid, 16 + 0] = e68[0]
            out_e69[bid, 16 + 0] = e69[0]
            for i in T.Parallel(16):
                out_e74[bid, 16 + i] = e74[i]
            for i, j in T.Parallel(16, 32):
                out_e3[bid, 16 + (i * 32 + j)] = e3[i, j]
            for i, j in T.Parallel(16, 32):
                out_e70[bid, 16 + (i * 32 + j)] = e70[i, j]
            out_e19[bid, 16 + 0] = e19[0]
            out_e22[bid, 16 + 0] = e22[0]
    return impl

@tilelang.jit
def extended_2_0():
    @T.macro
    def mixed_helper_extended_2_0(e6, e7, mem0, mem1, mem2, steps, limit, bid, result_e13, result_e10):
        e8 = T.alloc_fragment((16,), "float32")
        e9 = T.alloc_fragment((16,), "float32")
        e10 = T.alloc_fragment((1,), "float32")
        e11 = T.alloc_fragment((16, 1), "float32")
        e12 = T.alloc_fragment((16, 1), "float16")
        e13 = T.alloc_fragment((16, 32), "float16")
        e8_wide = T.alloc_fragment((16, 32), "float32")
        e10_wide = T.alloc_fragment((16,), "float32")
        for i, j in T.Parallel(16, 32):
            e8_wide[i, j] = T.cast(e6[i, j], 'float32')
        T.reduce_sum(e8_wide, e8, dim=1, clear=True)
        for i in T.Parallel(16):
            e9[i] = T.cast((e8[i] + e7[i]), "float32")
        for i in T.Parallel(16):
            e10_wide[i] = T.cast(e9[i], 'float32')
        T.reduce_sum(e10_wide, e10, dim=0, clear=True)
        for i, j in T.Parallel(16, 1):
            e11[i, 0] = T.cast(e9[(i * 1 + j)], "float32")
        for i, j in T.Parallel(16, 1):
            e12[i, 0] = T.cast(e11[i, 0], "float16")
        for i, j in T.Parallel(16, 32):
            e13[i, j] = T.cast((e6[i, j] + e12[i, 0]), "float16")
        for i, j in T.Parallel(16, 32):
            result_e13[i, j] = e13[i, j]
        result_e10[0] = e10[0]
    @T.prim_func
    def impl(mem0: T.Buffer((2, 544), "float16"), mem1: T.Buffer((2, 48), "float32"), mem2: T.Buffer((2, 33), "int32"), out_e73: T.Buffer((2, 544), "float16"), out_e68: T.Buffer((2, 33), "float32"), out_e69: T.Buffer((2, 33), "int32"), out_e74: T.Buffer((2, 48), "float32"), steps: T.int32, limit: T.int32):
        with T.Kernel(2, threads=256) as bid:
            e1 = T.alloc_fragment((16, 32), "int32")
            e2 = T.alloc_fragment((16, 32), "bool")
            e3 = T.alloc_fragment((16, 32), "float16")
            e4 = T.alloc_fragment((16, 32), "float16")
            e5 = T.alloc_fragment((16, 32), "float16")
            e14 = T.alloc_fragment((16,), "int32")
            e15 = T.alloc_fragment((16,), "bool")
            e16 = T.alloc_fragment((16,), "float32")
            e17 = T.alloc_fragment((16, 32), "float16")
            e18 = T.alloc_fragment((1,), "float32")
            e19 = T.alloc_fragment((1,), "int32")
            e20 = T.alloc_fragment((1,), "int32")
            e21 = T.alloc_fragment((1,), "int32")
            e22 = T.alloc_fragment((1,), "bool")
            e23 = T.alloc_fragment((1,), "int32")
            e24 = T.alloc_fragment((1,), "int32")
            e25 = T.alloc_fragment((1,), "bool")
            e45 = T.alloc_fragment((16, 32), "float16")
            e46 = T.alloc_fragment((1,), "float32")
            e47 = T.alloc_fragment((1,), "int32")
            e36 = T.alloc_fragment((16, 32), "float16")
            e31 = T.alloc_fragment((16, 32), "float16")
            e32 = T.alloc_fragment((16, 32), "float16")
            e34 = T.alloc_fragment((16, 32), "float16")
            e35 = T.alloc_fragment((16, 32), "float16")
            e37 = T.alloc_fragment((16, 32), "float16")
            e38 = T.alloc_fragment((1,), "float32")
            e39 = T.alloc_fragment((16, 32), "float16")
            e40 = T.alloc_fragment((16, 32), "float16")
            e41 = T.alloc_fragment((1,), "float32")
            e42 = T.alloc_fragment((1,), "int32")
            e43 = T.alloc_fragment((1,), "int32")
            e44 = T.alloc_fragment((1,), "int32")
            e67 = T.alloc_fragment((16, 32), "float16")
            e68 = T.alloc_fragment((1,), "float32")
            e69 = T.alloc_fragment((1,), "int32")
            e58 = T.alloc_fragment((16, 32), "float16")
            e53 = T.alloc_fragment((16, 32), "float16")
            e54 = T.alloc_fragment((16, 32), "float16")
            e56 = T.alloc_fragment((16, 32), "float16")
            e57 = T.alloc_fragment((16, 32), "float16")
            e59 = T.alloc_fragment((16, 32), "float16")
            e60 = T.alloc_fragment((1,), "float32")
            e61 = T.alloc_fragment((16, 32), "float16")
            e62 = T.alloc_fragment((16, 32), "float16")
            e63 = T.alloc_fragment((1,), "float32")
            e64 = T.alloc_fragment((1,), "int32")
            e65 = T.alloc_fragment((1,), "int32")
            e66 = T.alloc_fragment((1,), "int32")
            e70 = T.alloc_fragment((16, 32), "float16")
            e71 = T.alloc_fragment((16, 32), "float16")
            e72 = T.alloc_fragment((16, 32), "float16")
            e73 = T.alloc_fragment((16, 32), "float16")
            e74 = T.alloc_fragment((16,), "float32")
            e26 = T.alloc_fragment((1,), "int32")
            e27 = T.alloc_fragment((16, 32), "float16")
            e28 = T.alloc_fragment((1,), "float32")
            e29 = T.alloc_fragment((1,), "int32")
            e30 = T.alloc_fragment((16, 32), "float16")
            e33 = T.alloc_fragment((16, 32), "float16")
            e48 = T.alloc_fragment((1,), "int32")
            e49 = T.alloc_fragment((16, 32), "float16")
            e50 = T.alloc_fragment((1,), "float32")
            e51 = T.alloc_fragment((1,), "int32")
            e52 = T.alloc_fragment((16, 32), "float16")
            e55 = T.alloc_fragment((16, 32), "float16")
            e70_flip_shared = T.alloc_shared((16, 32), "float16")
            e74_wide = T.alloc_fragment((16, 32), "float32")
            for i, j in T.Parallel(16, 32):
                e1[i, j] = T.cast((((i * 32 + j) + 63) % 512), "int32")
            for i, j in T.Parallel(16, 32):
                e2[i, j] = T.cast(True, "bool")
            for i, j in T.Parallel(16, 32):
                e3[i, j] = T.cast(T.if_then_else((e2[i, j] and e1[i, j] >= 0 and e1[i, j] < 512), mem0[bid, 16 + e1[i, j] * 1], T.cast(0, "float16")), "float16")
            for i, j in T.Parallel(16, 32):
                e4[i, j] = T.cast((e3[i, j] * e3[i, j]), "float16")
            for i, j in T.Parallel(16, 32):
                e5[i, j] = T.cast(e4[i, j], "float16")
            for i in T.Parallel(16):
                e14[i] = T.cast(((i + 11) % 16), "int32")
            for i in T.Parallel(16):
                e15[i] = T.cast(True, "bool")
            for i in T.Parallel(16):
                e16[i] = T.cast(T.if_then_else((e15[i] and e14[i] >= 0 and e14[i] < 16), mem1[bid, 16 + e14[i] * 1], T.cast(0, "float32")), "float32")
            mixed_helper_extended_2_0(e5, e16, mem0, mem1, mem2, steps, limit, bid, e17, e18)
            e19[0] = T.cast(steps, "int32")
            e20[0] = T.cast(0, "int32")
            e21[0] = T.cast(0, "int32")
            e22[0] = T.cast(True, "bool")
            e23[0] = T.cast(T.if_then_else((e22[0] and e21[0] >= 0 and e21[0] < 1), mem2[bid, 16 + e21[0] * 1], T.cast(0, "int32")), "int32")
            e24[0] = T.cast(0, "int32")
            e25[0] = T.cast((e23[0] < e24[0]), "bool")
            for i, j in T.Parallel(16, 32):
                e45[i, j] = e17[i, j]
            e46[0] = e18[0]
            e47[0] = e20[0]
            for e45_iteration in T.serial(T.min(T.max(e19[0], 0), 4)):
                e26[0] = e45_iteration
                for i, j in T.Parallel(16, 32):
                    e27[i, j] = e45[i, j]
                e28[0] = e46[0]
                e29[0] = e47[0]
                if e25[0]:
                    for i, j in T.Parallel(16, 32):
                        e30[i, j] = e27[i, j]
                    for i, j in T.Parallel(16, 32):
                        e31[i, j] = T.cast(0.125, "float16")
                    for i, j in T.Parallel(16, 32):
                        e32[i, j] = T.cast((e30[i, j] + e31[i, j]), "float16")
                    for i, j in T.Parallel(16, 32):
                        e36[i, j] = e32[i, j]
                else:
                    for i, j in T.Parallel(16, 32):
                        e33[i, j] = e27[i, j]
                    for i, j in T.Parallel(16, 32):
                        e34[i, j] = T.cast(0.125, "float16")
                    for i, j in T.Parallel(16, 32):
                        e35[i, j] = T.cast((e33[i, j] - e34[i, j]), "float16")
                    for i, j in T.Parallel(16, 32):
                        e36[i, j] = e35[i, j]
                mixed_helper_extended_2_0(e36, e16, mem0, mem1, mem2, steps, limit, bid, e37, e38)
                for i, j in T.Parallel(16, 32):
                    e39[i, j] = T.cast(0.125, "float16")
                for i, j in T.Parallel(16, 32):
                    e40[i, j] = T.cast((e37[i, j] * e39[i, j]), "float16")
                e41[0] = T.cast((e28[0] + e38[0]), "float32")
                e42[0] = T.cast(1, "int32")
                e43[0] = T.cast((e26[0] + e42[0]), "int32")
                e44[0] = T.cast((e29[0] + e43[0]), "int32")
                for i, j in T.Parallel(16, 32):
                    e45[i, j] = e40[i, j]
                e46[0] = e41[0]
                e47[0] = e44[0]
            for i, j in T.Parallel(16, 32):
                e67[i, j] = e45[i, j]
            e68[0] = e46[0]
            e69[0] = e47[0]
            for e67_iteration in T.serial(T.min(T.max(e19[0], 0), 4)):
                e48[0] = e67_iteration
                for i, j in T.Parallel(16, 32):
                    e49[i, j] = e67[i, j]
                e50[0] = e68[0]
                e51[0] = e69[0]
                if e25[0]:
                    for i, j in T.Parallel(16, 32):
                        e52[i, j] = e49[i, j]
                    for i, j in T.Parallel(16, 32):
                        e53[i, j] = T.cast(0.125, "float16")
                    for i, j in T.Parallel(16, 32):
                        e54[i, j] = T.cast((e52[i, j] + e53[i, j]), "float16")
                    for i, j in T.Parallel(16, 32):
                        e58[i, j] = e54[i, j]
                else:
                    for i, j in T.Parallel(16, 32):
                        e55[i, j] = e49[i, j]
                    for i, j in T.Parallel(16, 32):
                        e56[i, j] = T.cast(0.125, "float16")
                    for i, j in T.Parallel(16, 32):
                        e57[i, j] = T.cast((e55[i, j] - e56[i, j]), "float16")
                    for i, j in T.Parallel(16, 32):
                        e58[i, j] = e57[i, j]
                mixed_helper_extended_2_0(e58, e16, mem0, mem1, mem2, steps, limit, bid, e59, e60)
                for i, j in T.Parallel(16, 32):
                    e61[i, j] = T.cast(0.125, "float16")
                for i, j in T.Parallel(16, 32):
                    e62[i, j] = T.cast((e59[i, j] * e61[i, j]), "float16")
                e63[0] = T.cast((e50[0] + e60[0]), "float32")
                e64[0] = T.cast(1, "int32")
                e65[0] = T.cast((e48[0] + e64[0]), "int32")
                e66[0] = T.cast((e51[0] + e65[0]), "int32")
                for i, j in T.Parallel(16, 32):
                    e67[i, j] = e62[i, j]
                e68[0] = e63[0]
                e69[0] = e66[0]
            T.copy(e67, e70_flip_shared)
            T.sync_threads()
            for i, j in T.Parallel(16, 32):
                e70[i, j] = T.cast(e70_flip_shared[i, 31 - j], "float16")
            for i, j in T.Parallel(16, 32):
                e71[i, j] = T.cast((e70[i, j] - e3[i, j]), "float16")
            for i, j in T.Parallel(16, 32):
                e72[i, j] = T.cast((e71[i, j] - e71[i, j]), "float16")
            for i, j in T.Parallel(16, 32):
                e73[i, j] = T.cast((e72[i, j] * e3[i, j]), "float16")
            for i, j in T.Parallel(16, 32):
                e74_wide[i, j] = T.cast(e73[i, j], 'float32')
            T.reduce_sum(e74_wide, e74, dim=1, clear=True)
            for i, j in T.Parallel(16, 32):
                out_e73[bid, 16 + (i * 32 + j)] = e73[i, j]
            out_e68[bid, 16 + 0] = e68[0]
            out_e69[bid, 16 + 0] = e69[0]
            for i in T.Parallel(16):
                out_e74[bid, 16 + i] = e74[i]
    return impl

@tilelang.jit
def extended_2_1():
    @T.macro
    def mixed_helper_extended_2_1(e6, e7, mem0, mem1, mem2, steps, limit, bid, result_e13, result_e10):
        e8 = T.alloc_fragment((16,), "float32")
        e9 = T.alloc_fragment((16,), "float32")
        e10 = T.alloc_fragment((1,), "float32")
        e11 = T.alloc_fragment((16, 1), "float32")
        e12 = T.alloc_fragment((16, 1), "float16")
        e13 = T.alloc_fragment((16, 32), "float16")
        e8_wide = T.alloc_fragment((16, 32), "float32")
        e10_wide = T.alloc_fragment((16,), "float32")
        for i, j in T.Parallel(16, 32):
            e8_wide[i, j] = T.cast(e6[i, j], 'float32')
        T.reduce_sum(e8_wide, e8, dim=1, clear=True)
        for i in T.Parallel(16):
            e9[i] = T.cast((e8[i] + e7[i]), "float32")
        for i in T.Parallel(16):
            e10_wide[i] = T.cast(e9[i], 'float32')
        T.reduce_sum(e10_wide, e10, dim=0, clear=True)
        for i, j in T.Parallel(16, 1):
            e11[i, 0] = T.cast(e9[(i * 1 + j)], "float32")
        for i, j in T.Parallel(16, 1):
            e12[i, 0] = T.cast(e11[i, 0], "float16")
        for i, j in T.Parallel(16, 32):
            e13[i, j] = T.cast((e6[i, j] + e12[i, 0]), "float16")
        for i, j in T.Parallel(16, 32):
            result_e13[i, j] = e13[i, j]
        result_e10[0] = e10[0]
    @T.prim_func
    def impl(mem0: T.Buffer((2, 544), "float16"), mem1: T.Buffer((2, 48), "float32"), mem2: T.Buffer((2, 33), "int32"), out_e73: T.Buffer((2, 544), "float16"), out_e68: T.Buffer((2, 33), "float32"), out_e69: T.Buffer((2, 33), "int32"), out_e74: T.Buffer((2, 48), "float32"), out_e3: T.Buffer((2, 544), "float16"), out_e70: T.Buffer((2, 544), "float16"), out_e19: T.Buffer((2, 33), "int32"), out_e22: T.Buffer((2, 33), "bool"), steps: T.int32, limit: T.int32):
        with T.Kernel(2, threads=256) as bid:
            e1 = T.alloc_fragment((16, 32), "int32")
            e2 = T.alloc_fragment((16, 32), "bool")
            e3 = T.alloc_fragment((16, 32), "float16")
            e4 = T.alloc_fragment((16, 32), "float16")
            e5 = T.alloc_fragment((16, 32), "float16")
            e14 = T.alloc_fragment((16,), "int32")
            e15 = T.alloc_fragment((16,), "bool")
            e16 = T.alloc_fragment((16,), "float32")
            e17 = T.alloc_fragment((16, 32), "float16")
            e18 = T.alloc_fragment((1,), "float32")
            e19 = T.alloc_fragment((1,), "int32")
            e20 = T.alloc_fragment((1,), "int32")
            e21 = T.alloc_fragment((1,), "int32")
            e22 = T.alloc_fragment((1,), "bool")
            e23 = T.alloc_fragment((1,), "int32")
            e24 = T.alloc_fragment((1,), "int32")
            e25 = T.alloc_fragment((1,), "bool")
            e45 = T.alloc_fragment((16, 32), "float16")
            e46 = T.alloc_fragment((1,), "float32")
            e47 = T.alloc_fragment((1,), "int32")
            e36 = T.alloc_fragment((16, 32), "float16")
            e31 = T.alloc_fragment((16, 32), "float16")
            e32 = T.alloc_fragment((16, 32), "float16")
            e34 = T.alloc_fragment((16, 32), "float16")
            e35 = T.alloc_fragment((16, 32), "float16")
            e37 = T.alloc_fragment((16, 32), "float16")
            e38 = T.alloc_fragment((1,), "float32")
            e39 = T.alloc_fragment((16, 32), "float16")
            e40 = T.alloc_fragment((16, 32), "float16")
            e41 = T.alloc_fragment((1,), "float32")
            e42 = T.alloc_fragment((1,), "int32")
            e43 = T.alloc_fragment((1,), "int32")
            e44 = T.alloc_fragment((1,), "int32")
            e67 = T.alloc_fragment((16, 32), "float16")
            e68 = T.alloc_fragment((1,), "float32")
            e69 = T.alloc_fragment((1,), "int32")
            e58 = T.alloc_fragment((16, 32), "float16")
            e53 = T.alloc_fragment((16, 32), "float16")
            e54 = T.alloc_fragment((16, 32), "float16")
            e56 = T.alloc_fragment((16, 32), "float16")
            e57 = T.alloc_fragment((16, 32), "float16")
            e59 = T.alloc_fragment((16, 32), "float16")
            e60 = T.alloc_fragment((1,), "float32")
            e61 = T.alloc_fragment((16, 32), "float16")
            e62 = T.alloc_fragment((16, 32), "float16")
            e63 = T.alloc_fragment((1,), "float32")
            e64 = T.alloc_fragment((1,), "int32")
            e65 = T.alloc_fragment((1,), "int32")
            e66 = T.alloc_fragment((1,), "int32")
            e70 = T.alloc_fragment((16, 32), "float16")
            e71 = T.alloc_fragment((16, 32), "float16")
            e72 = T.alloc_fragment((16, 32), "float16")
            e73 = T.alloc_fragment((16, 32), "float16")
            e74 = T.alloc_fragment((16,), "float32")
            e26 = T.alloc_fragment((1,), "int32")
            e27 = T.alloc_fragment((16, 32), "float16")
            e28 = T.alloc_fragment((1,), "float32")
            e29 = T.alloc_fragment((1,), "int32")
            e30 = T.alloc_fragment((16, 32), "float16")
            e33 = T.alloc_fragment((16, 32), "float16")
            e48 = T.alloc_fragment((1,), "int32")
            e49 = T.alloc_fragment((16, 32), "float16")
            e50 = T.alloc_fragment((1,), "float32")
            e51 = T.alloc_fragment((1,), "int32")
            e52 = T.alloc_fragment((16, 32), "float16")
            e55 = T.alloc_fragment((16, 32), "float16")
            e70_flip_shared = T.alloc_shared((16, 32), "float16")
            e74_wide = T.alloc_fragment((16, 32), "float32")
            for i, j in T.Parallel(16, 32):
                e1[i, j] = T.cast((((i * 32 + j) + 63) % 512), "int32")
            for i, j in T.Parallel(16, 32):
                e2[i, j] = T.cast(True, "bool")
            for i, j in T.Parallel(16, 32):
                e3[i, j] = T.cast(T.if_then_else((e2[i, j] and e1[i, j] >= 0 and e1[i, j] < 512), mem0[bid, 16 + e1[i, j] * 1], T.cast(0, "float16")), "float16")
            for i, j in T.Parallel(16, 32):
                e4[i, j] = T.cast((e3[i, j] * e3[i, j]), "float16")
            for i, j in T.Parallel(16, 32):
                e5[i, j] = T.cast(e4[i, j], "float16")
            for i in T.Parallel(16):
                e14[i] = T.cast(((i + 11) % 16), "int32")
            for i in T.Parallel(16):
                e15[i] = T.cast(True, "bool")
            for i in T.Parallel(16):
                e16[i] = T.cast(T.if_then_else((e15[i] and e14[i] >= 0 and e14[i] < 16), mem1[bid, 16 + e14[i] * 1], T.cast(0, "float32")), "float32")
            mixed_helper_extended_2_1(e5, e16, mem0, mem1, mem2, steps, limit, bid, e17, e18)
            e19[0] = T.cast(steps, "int32")
            e20[0] = T.cast(0, "int32")
            e21[0] = T.cast(0, "int32")
            e22[0] = T.cast(True, "bool")
            e23[0] = T.cast(T.if_then_else((e22[0] and e21[0] >= 0 and e21[0] < 1), mem2[bid, 16 + e21[0] * 1], T.cast(0, "int32")), "int32")
            e24[0] = T.cast(0, "int32")
            e25[0] = T.cast((e23[0] < e24[0]), "bool")
            for i, j in T.Parallel(16, 32):
                e45[i, j] = e17[i, j]
            e46[0] = e18[0]
            e47[0] = e20[0]
            for e45_iteration in T.serial(T.min(T.max(e19[0], 0), 4)):
                e26[0] = e45_iteration
                for i, j in T.Parallel(16, 32):
                    e27[i, j] = e45[i, j]
                e28[0] = e46[0]
                e29[0] = e47[0]
                if e25[0]:
                    for i, j in T.Parallel(16, 32):
                        e30[i, j] = e27[i, j]
                    for i, j in T.Parallel(16, 32):
                        e31[i, j] = T.cast(0.125, "float16")
                    for i, j in T.Parallel(16, 32):
                        e32[i, j] = T.cast((e30[i, j] + e31[i, j]), "float16")
                    for i, j in T.Parallel(16, 32):
                        e36[i, j] = e32[i, j]
                else:
                    for i, j in T.Parallel(16, 32):
                        e33[i, j] = e27[i, j]
                    for i, j in T.Parallel(16, 32):
                        e34[i, j] = T.cast(0.125, "float16")
                    for i, j in T.Parallel(16, 32):
                        e35[i, j] = T.cast((e33[i, j] - e34[i, j]), "float16")
                    for i, j in T.Parallel(16, 32):
                        e36[i, j] = e35[i, j]
                mixed_helper_extended_2_1(e36, e16, mem0, mem1, mem2, steps, limit, bid, e37, e38)
                for i, j in T.Parallel(16, 32):
                    e39[i, j] = T.cast(0.125, "float16")
                for i, j in T.Parallel(16, 32):
                    e40[i, j] = T.cast((e37[i, j] * e39[i, j]), "float16")
                e41[0] = T.cast((e28[0] + e38[0]), "float32")
                e42[0] = T.cast(1, "int32")
                e43[0] = T.cast((e26[0] + e42[0]), "int32")
                e44[0] = T.cast((e29[0] + e43[0]), "int32")
                for i, j in T.Parallel(16, 32):
                    e45[i, j] = e40[i, j]
                e46[0] = e41[0]
                e47[0] = e44[0]
            for i, j in T.Parallel(16, 32):
                e67[i, j] = e45[i, j]
            e68[0] = e46[0]
            e69[0] = e47[0]
            for e67_iteration in T.serial(T.min(T.max(e19[0], 0), 4)):
                e48[0] = e67_iteration
                for i, j in T.Parallel(16, 32):
                    e49[i, j] = e67[i, j]
                e50[0] = e68[0]
                e51[0] = e69[0]
                if e25[0]:
                    for i, j in T.Parallel(16, 32):
                        e52[i, j] = e49[i, j]
                    for i, j in T.Parallel(16, 32):
                        e53[i, j] = T.cast(0.125, "float16")
                    for i, j in T.Parallel(16, 32):
                        e54[i, j] = T.cast((e52[i, j] + e53[i, j]), "float16")
                    for i, j in T.Parallel(16, 32):
                        e58[i, j] = e54[i, j]
                else:
                    for i, j in T.Parallel(16, 32):
                        e55[i, j] = e49[i, j]
                    for i, j in T.Parallel(16, 32):
                        e56[i, j] = T.cast(0.125, "float16")
                    for i, j in T.Parallel(16, 32):
                        e57[i, j] = T.cast((e55[i, j] - e56[i, j]), "float16")
                    for i, j in T.Parallel(16, 32):
                        e58[i, j] = e57[i, j]
                mixed_helper_extended_2_1(e58, e16, mem0, mem1, mem2, steps, limit, bid, e59, e60)
                for i, j in T.Parallel(16, 32):
                    e61[i, j] = T.cast(0.125, "float16")
                for i, j in T.Parallel(16, 32):
                    e62[i, j] = T.cast((e59[i, j] * e61[i, j]), "float16")
                e63[0] = T.cast((e50[0] + e60[0]), "float32")
                e64[0] = T.cast(1, "int32")
                e65[0] = T.cast((e48[0] + e64[0]), "int32")
                e66[0] = T.cast((e51[0] + e65[0]), "int32")
                for i, j in T.Parallel(16, 32):
                    e67[i, j] = e62[i, j]
                e68[0] = e63[0]
                e69[0] = e66[0]
            T.copy(e67, e70_flip_shared)
            T.sync_threads()
            for i, j in T.Parallel(16, 32):
                e70[i, j] = T.cast(e70_flip_shared[i, 31 - j], "float16")
            for i, j in T.Parallel(16, 32):
                e71[i, j] = T.cast((e70[i, j] - e3[i, j]), "float16")
            for i, j in T.Parallel(16, 32):
                e72[i, j] = T.cast((e71[i, j] - e71[i, j]), "float16")
            for i, j in T.Parallel(16, 32):
                e73[i, j] = T.cast((e72[i, j] * e3[i, j]), "float16")
            for i, j in T.Parallel(16, 32):
                e74_wide[i, j] = T.cast(e73[i, j], 'float32')
            T.reduce_sum(e74_wide, e74, dim=1, clear=True)
            for i, j in T.Parallel(16, 32):
                out_e73[bid, 16 + (i * 32 + j)] = e73[i, j]
            out_e68[bid, 16 + 0] = e68[0]
            out_e69[bid, 16 + 0] = e69[0]
            for i in T.Parallel(16):
                out_e74[bid, 16 + i] = e74[i]
            for i, j in T.Parallel(16, 32):
                out_e3[bid, 16 + (i * 32 + j)] = e3[i, j]
            for i, j in T.Parallel(16, 32):
                out_e70[bid, 16 + (i * 32 + j)] = e70[i, j]
            out_e19[bid, 16 + 0] = e19[0]
            out_e22[bid, 16 + 0] = e22[0]
    return impl

@tilelang.jit
def extended_3_0():
    @T.macro
    def mixed_helper_extended_3_0(e6, e7, mem0, mem1, mem2, steps, limit, bid, result_e13, result_e10):
        e8 = T.alloc_fragment((16,), "float32")
        e9 = T.alloc_fragment((16,), "float32")
        e10 = T.alloc_fragment((1,), "float32")
        e11 = T.alloc_fragment((16, 1), "float32")
        e12 = T.alloc_fragment((16, 1), "float16")
        e13 = T.alloc_fragment((16, 32), "float16")
        e8_wide = T.alloc_fragment((16, 32), "float32")
        e10_wide = T.alloc_fragment((16,), "float32")
        for i, j in T.Parallel(16, 32):
            e8_wide[i, j] = T.cast(e6[i, j], 'float32')
        T.reduce_sum(e8_wide, e8, dim=1, clear=True)
        for i in T.Parallel(16):
            e9[i] = T.cast((e8[i] + e7[i]), "float32")
        for i in T.Parallel(16):
            e10_wide[i] = T.cast(e9[i], 'float32')
        T.reduce_sum(e10_wide, e10, dim=0, clear=True)
        for i, j in T.Parallel(16, 1):
            e11[i, 0] = T.cast(e9[(i * 1 + j)], "float32")
        for i, j in T.Parallel(16, 1):
            e12[i, 0] = T.cast(e11[i, 0], "float16")
        for i, j in T.Parallel(16, 32):
            e13[i, j] = T.cast((e6[i, j] + e12[i, 0]), "float16")
        for i, j in T.Parallel(16, 32):
            result_e13[i, j] = e13[i, j]
        result_e10[0] = e10[0]
    @T.prim_func
    def impl(mem0: T.Buffer((2, 544), "float16"), mem1: T.Buffer((2, 48), "float32"), mem2: T.Buffer((2, 33), "int32"), out_e73: T.Buffer((2, 544), "float16"), out_e68: T.Buffer((2, 33), "float32"), out_e69: T.Buffer((2, 33), "int32"), out_e74: T.Buffer((2, 48), "float32"), steps: T.int32, limit: T.int32):
        with T.Kernel(2, threads=256) as bid:
            e1 = T.alloc_fragment((16, 32), "int32")
            e2 = T.alloc_fragment((16, 32), "bool")
            e3 = T.alloc_fragment((16, 32), "float16")
            e4 = T.alloc_fragment((16, 32), "float16")
            e5 = T.alloc_fragment((16, 32), "float16")
            e14 = T.alloc_fragment((16,), "int32")
            e15 = T.alloc_fragment((16,), "bool")
            e16 = T.alloc_fragment((16,), "float32")
            e17 = T.alloc_fragment((16, 32), "float16")
            e18 = T.alloc_fragment((1,), "float32")
            e19 = T.alloc_fragment((1,), "int32")
            e20 = T.alloc_fragment((1,), "int32")
            e21 = T.alloc_fragment((1,), "int32")
            e22 = T.alloc_fragment((1,), "bool")
            e23 = T.alloc_fragment((1,), "int32")
            e24 = T.alloc_fragment((1,), "int32")
            e25 = T.alloc_fragment((1,), "bool")
            e45 = T.alloc_fragment((16, 32), "float16")
            e46 = T.alloc_fragment((1,), "float32")
            e47 = T.alloc_fragment((1,), "int32")
            e36 = T.alloc_fragment((16, 32), "float16")
            e31 = T.alloc_fragment((16, 32), "float16")
            e32 = T.alloc_fragment((16, 32), "float16")
            e34 = T.alloc_fragment((16, 32), "float16")
            e35 = T.alloc_fragment((16, 32), "float16")
            e37 = T.alloc_fragment((16, 32), "float16")
            e38 = T.alloc_fragment((1,), "float32")
            e39 = T.alloc_fragment((16, 32), "float16")
            e40 = T.alloc_fragment((16, 32), "float16")
            e41 = T.alloc_fragment((1,), "float32")
            e42 = T.alloc_fragment((1,), "int32")
            e43 = T.alloc_fragment((1,), "int32")
            e44 = T.alloc_fragment((1,), "int32")
            e67 = T.alloc_fragment((16, 32), "float16")
            e68 = T.alloc_fragment((1,), "float32")
            e69 = T.alloc_fragment((1,), "int32")
            e58 = T.alloc_fragment((16, 32), "float16")
            e53 = T.alloc_fragment((16, 32), "float16")
            e54 = T.alloc_fragment((16, 32), "float16")
            e56 = T.alloc_fragment((16, 32), "float16")
            e57 = T.alloc_fragment((16, 32), "float16")
            e59 = T.alloc_fragment((16, 32), "float16")
            e60 = T.alloc_fragment((1,), "float32")
            e61 = T.alloc_fragment((16, 32), "float16")
            e62 = T.alloc_fragment((16, 32), "float16")
            e63 = T.alloc_fragment((1,), "float32")
            e64 = T.alloc_fragment((1,), "int32")
            e65 = T.alloc_fragment((1,), "int32")
            e66 = T.alloc_fragment((1,), "int32")
            e70 = T.alloc_fragment((16, 32), "float16")
            e71 = T.alloc_fragment((16, 32), "float16")
            e72 = T.alloc_fragment((16, 32), "float16")
            e73 = T.alloc_fragment((16, 32), "float16")
            e74 = T.alloc_fragment((16,), "float32")
            e26 = T.alloc_fragment((1,), "int32")
            e27 = T.alloc_fragment((16, 32), "float16")
            e28 = T.alloc_fragment((1,), "float32")
            e29 = T.alloc_fragment((1,), "int32")
            e30 = T.alloc_fragment((16, 32), "float16")
            e33 = T.alloc_fragment((16, 32), "float16")
            e48 = T.alloc_fragment((1,), "int32")
            e49 = T.alloc_fragment((16, 32), "float16")
            e50 = T.alloc_fragment((1,), "float32")
            e51 = T.alloc_fragment((1,), "int32")
            e52 = T.alloc_fragment((16, 32), "float16")
            e55 = T.alloc_fragment((16, 32), "float16")
            e70_flip_shared = T.alloc_shared((16, 32), "float16")
            e74_wide = T.alloc_fragment((16, 32), "float32")
            for i, j in T.Parallel(16, 32):
                e1[i, j] = T.cast((((i * 32 + j) + 63) % 512), "int32")
            for i, j in T.Parallel(16, 32):
                e2[i, j] = T.cast(True, "bool")
            for i, j in T.Parallel(16, 32):
                e3[i, j] = T.cast(T.if_then_else((e2[i, j] and e1[i, j] >= 0 and e1[i, j] < 512), mem0[bid, 16 + e1[i, j] * 1], T.cast(0, "float16")), "float16")
            for i, j in T.Parallel(16, 32):
                e4[i, j] = T.cast((e3[i, j] * e3[i, j]), "float16")
            for i, j in T.Parallel(16, 32):
                e5[i, j] = T.cast(e4[i, j], "float16")
            for i in T.Parallel(16):
                e14[i] = T.cast(((i + 11) % 16), "int32")
            for i in T.Parallel(16):
                e15[i] = T.cast(True, "bool")
            for i in T.Parallel(16):
                e16[i] = T.cast(T.if_then_else((e15[i] and e14[i] >= 0 and e14[i] < 16), mem1[bid, 16 + e14[i] * 1], T.cast(0, "float32")), "float32")
            mixed_helper_extended_3_0(e5, e16, mem0, mem1, mem2, steps, limit, bid, e17, e18)
            e19[0] = T.cast(steps, "int32")
            e20[0] = T.cast(0, "int32")
            e21[0] = T.cast(0, "int32")
            e22[0] = T.cast(True, "bool")
            e23[0] = T.cast(T.if_then_else((e22[0] and e21[0] >= 0 and e21[0] < 1), mem2[bid, 16 + e21[0] * 1], T.cast(0, "int32")), "int32")
            e24[0] = T.cast(0, "int32")
            e25[0] = T.cast((e23[0] < e24[0]), "bool")
            for i, j in T.Parallel(16, 32):
                e45[i, j] = e17[i, j]
            e46[0] = e18[0]
            e47[0] = e20[0]
            for e45_iteration in T.serial(T.min(T.max(e19[0], 0), 4)):
                e26[0] = e45_iteration
                for i, j in T.Parallel(16, 32):
                    e27[i, j] = e45[i, j]
                e28[0] = e46[0]
                e29[0] = e47[0]
                if e25[0]:
                    for i, j in T.Parallel(16, 32):
                        e30[i, j] = e27[i, j]
                    for i, j in T.Parallel(16, 32):
                        e31[i, j] = T.cast(0.125, "float16")
                    for i, j in T.Parallel(16, 32):
                        e32[i, j] = T.cast((e30[i, j] + e31[i, j]), "float16")
                    for i, j in T.Parallel(16, 32):
                        e36[i, j] = e32[i, j]
                else:
                    for i, j in T.Parallel(16, 32):
                        e33[i, j] = e27[i, j]
                    for i, j in T.Parallel(16, 32):
                        e34[i, j] = T.cast(0.125, "float16")
                    for i, j in T.Parallel(16, 32):
                        e35[i, j] = T.cast((e33[i, j] - e34[i, j]), "float16")
                    for i, j in T.Parallel(16, 32):
                        e36[i, j] = e35[i, j]
                mixed_helper_extended_3_0(e36, e16, mem0, mem1, mem2, steps, limit, bid, e37, e38)
                for i, j in T.Parallel(16, 32):
                    e39[i, j] = T.cast(0.125, "float16")
                for i, j in T.Parallel(16, 32):
                    e40[i, j] = T.cast((e37[i, j] * e39[i, j]), "float16")
                e41[0] = T.cast((e28[0] + e38[0]), "float32")
                e42[0] = T.cast(1, "int32")
                e43[0] = T.cast((e26[0] + e42[0]), "int32")
                e44[0] = T.cast((e29[0] + e43[0]), "int32")
                for i, j in T.Parallel(16, 32):
                    e45[i, j] = e40[i, j]
                e46[0] = e41[0]
                e47[0] = e44[0]
            for i, j in T.Parallel(16, 32):
                e67[i, j] = e45[i, j]
            e68[0] = e46[0]
            e69[0] = e47[0]
            for e67_iteration in T.serial(T.min(T.max(e19[0], 0), 4)):
                e48[0] = e67_iteration
                for i, j in T.Parallel(16, 32):
                    e49[i, j] = e67[i, j]
                e50[0] = e68[0]
                e51[0] = e69[0]
                if e25[0]:
                    for i, j in T.Parallel(16, 32):
                        e52[i, j] = e49[i, j]
                    for i, j in T.Parallel(16, 32):
                        e53[i, j] = T.cast(0.125, "float16")
                    for i, j in T.Parallel(16, 32):
                        e54[i, j] = T.cast((e52[i, j] + e53[i, j]), "float16")
                    for i, j in T.Parallel(16, 32):
                        e58[i, j] = e54[i, j]
                else:
                    for i, j in T.Parallel(16, 32):
                        e55[i, j] = e49[i, j]
                    for i, j in T.Parallel(16, 32):
                        e56[i, j] = T.cast(0.125, "float16")
                    for i, j in T.Parallel(16, 32):
                        e57[i, j] = T.cast((e55[i, j] - e56[i, j]), "float16")
                    for i, j in T.Parallel(16, 32):
                        e58[i, j] = e57[i, j]
                mixed_helper_extended_3_0(e58, e16, mem0, mem1, mem2, steps, limit, bid, e59, e60)
                for i, j in T.Parallel(16, 32):
                    e61[i, j] = T.cast(0.125, "float16")
                for i, j in T.Parallel(16, 32):
                    e62[i, j] = T.cast((e59[i, j] * e61[i, j]), "float16")
                e63[0] = T.cast((e50[0] + e60[0]), "float32")
                e64[0] = T.cast(1, "int32")
                e65[0] = T.cast((e48[0] + e64[0]), "int32")
                e66[0] = T.cast((e51[0] + e65[0]), "int32")
                for i, j in T.Parallel(16, 32):
                    e67[i, j] = e62[i, j]
                e68[0] = e63[0]
                e69[0] = e66[0]
            T.copy(e67, e70_flip_shared)
            T.sync_threads()
            for i, j in T.Parallel(16, 32):
                e70[i, j] = T.cast(e70_flip_shared[i, 31 - j], "float16")
            for i, j in T.Parallel(16, 32):
                e71[i, j] = T.cast((e70[i, j] - e3[i, j]), "float16")
            for i, j in T.Parallel(16, 32):
                e72[i, j] = T.cast((e71[i, j] - e71[i, j]), "float16")
            for i, j in T.Parallel(16, 32):
                e73[i, j] = T.cast((e72[i, j] * e3[i, j]), "float16")
            for i, j in T.Parallel(16, 32):
                e74_wide[i, j] = T.cast(e73[i, j], 'float32')
            T.reduce_sum(e74_wide, e74, dim=1, clear=True)
            for i, j in T.Parallel(16, 32):
                out_e73[bid, 16 + (i * 32 + j)] = e73[i, j]
            out_e68[bid, 16 + 0] = e68[0]
            out_e69[bid, 16 + 0] = e69[0]
            for i in T.Parallel(16):
                out_e74[bid, 16 + i] = e74[i]
    return impl

@tilelang.jit
def extended_3_1():
    @T.macro
    def mixed_helper_extended_3_1(e6, e7, mem0, mem1, mem2, steps, limit, bid, result_e13, result_e10):
        e8 = T.alloc_fragment((16,), "float32")
        e9 = T.alloc_fragment((16,), "float32")
        e10 = T.alloc_fragment((1,), "float32")
        e11 = T.alloc_fragment((16, 1), "float32")
        e12 = T.alloc_fragment((16, 1), "float16")
        e13 = T.alloc_fragment((16, 32), "float16")
        e8_wide = T.alloc_fragment((16, 32), "float32")
        e10_wide = T.alloc_fragment((16,), "float32")
        for i, j in T.Parallel(16, 32):
            e8_wide[i, j] = T.cast(e6[i, j], 'float32')
        T.reduce_sum(e8_wide, e8, dim=1, clear=True)
        for i in T.Parallel(16):
            e9[i] = T.cast((e8[i] + e7[i]), "float32")
        for i in T.Parallel(16):
            e10_wide[i] = T.cast(e9[i], 'float32')
        T.reduce_sum(e10_wide, e10, dim=0, clear=True)
        for i, j in T.Parallel(16, 1):
            e11[i, 0] = T.cast(e9[(i * 1 + j)], "float32")
        for i, j in T.Parallel(16, 1):
            e12[i, 0] = T.cast(e11[i, 0], "float16")
        for i, j in T.Parallel(16, 32):
            e13[i, j] = T.cast((e6[i, j] + e12[i, 0]), "float16")
        for i, j in T.Parallel(16, 32):
            result_e13[i, j] = e13[i, j]
        result_e10[0] = e10[0]
    @T.prim_func
    def impl(mem0: T.Buffer((2, 544), "float16"), mem1: T.Buffer((2, 48), "float32"), mem2: T.Buffer((2, 33), "int32"), out_e73: T.Buffer((2, 544), "float16"), out_e68: T.Buffer((2, 33), "float32"), out_e69: T.Buffer((2, 33), "int32"), out_e74: T.Buffer((2, 48), "float32"), out_e3: T.Buffer((2, 544), "float16"), out_e70: T.Buffer((2, 544), "float16"), out_e19: T.Buffer((2, 33), "int32"), out_e22: T.Buffer((2, 33), "bool"), steps: T.int32, limit: T.int32):
        with T.Kernel(2, threads=256) as bid:
            e1 = T.alloc_fragment((16, 32), "int32")
            e2 = T.alloc_fragment((16, 32), "bool")
            e3 = T.alloc_fragment((16, 32), "float16")
            e4 = T.alloc_fragment((16, 32), "float16")
            e5 = T.alloc_fragment((16, 32), "float16")
            e14 = T.alloc_fragment((16,), "int32")
            e15 = T.alloc_fragment((16,), "bool")
            e16 = T.alloc_fragment((16,), "float32")
            e17 = T.alloc_fragment((16, 32), "float16")
            e18 = T.alloc_fragment((1,), "float32")
            e19 = T.alloc_fragment((1,), "int32")
            e20 = T.alloc_fragment((1,), "int32")
            e21 = T.alloc_fragment((1,), "int32")
            e22 = T.alloc_fragment((1,), "bool")
            e23 = T.alloc_fragment((1,), "int32")
            e24 = T.alloc_fragment((1,), "int32")
            e25 = T.alloc_fragment((1,), "bool")
            e45 = T.alloc_fragment((16, 32), "float16")
            e46 = T.alloc_fragment((1,), "float32")
            e47 = T.alloc_fragment((1,), "int32")
            e36 = T.alloc_fragment((16, 32), "float16")
            e31 = T.alloc_fragment((16, 32), "float16")
            e32 = T.alloc_fragment((16, 32), "float16")
            e34 = T.alloc_fragment((16, 32), "float16")
            e35 = T.alloc_fragment((16, 32), "float16")
            e37 = T.alloc_fragment((16, 32), "float16")
            e38 = T.alloc_fragment((1,), "float32")
            e39 = T.alloc_fragment((16, 32), "float16")
            e40 = T.alloc_fragment((16, 32), "float16")
            e41 = T.alloc_fragment((1,), "float32")
            e42 = T.alloc_fragment((1,), "int32")
            e43 = T.alloc_fragment((1,), "int32")
            e44 = T.alloc_fragment((1,), "int32")
            e67 = T.alloc_fragment((16, 32), "float16")
            e68 = T.alloc_fragment((1,), "float32")
            e69 = T.alloc_fragment((1,), "int32")
            e58 = T.alloc_fragment((16, 32), "float16")
            e53 = T.alloc_fragment((16, 32), "float16")
            e54 = T.alloc_fragment((16, 32), "float16")
            e56 = T.alloc_fragment((16, 32), "float16")
            e57 = T.alloc_fragment((16, 32), "float16")
            e59 = T.alloc_fragment((16, 32), "float16")
            e60 = T.alloc_fragment((1,), "float32")
            e61 = T.alloc_fragment((16, 32), "float16")
            e62 = T.alloc_fragment((16, 32), "float16")
            e63 = T.alloc_fragment((1,), "float32")
            e64 = T.alloc_fragment((1,), "int32")
            e65 = T.alloc_fragment((1,), "int32")
            e66 = T.alloc_fragment((1,), "int32")
            e70 = T.alloc_fragment((16, 32), "float16")
            e71 = T.alloc_fragment((16, 32), "float16")
            e72 = T.alloc_fragment((16, 32), "float16")
            e73 = T.alloc_fragment((16, 32), "float16")
            e74 = T.alloc_fragment((16,), "float32")
            e26 = T.alloc_fragment((1,), "int32")
            e27 = T.alloc_fragment((16, 32), "float16")
            e28 = T.alloc_fragment((1,), "float32")
            e29 = T.alloc_fragment((1,), "int32")
            e30 = T.alloc_fragment((16, 32), "float16")
            e33 = T.alloc_fragment((16, 32), "float16")
            e48 = T.alloc_fragment((1,), "int32")
            e49 = T.alloc_fragment((16, 32), "float16")
            e50 = T.alloc_fragment((1,), "float32")
            e51 = T.alloc_fragment((1,), "int32")
            e52 = T.alloc_fragment((16, 32), "float16")
            e55 = T.alloc_fragment((16, 32), "float16")
            e70_flip_shared = T.alloc_shared((16, 32), "float16")
            e74_wide = T.alloc_fragment((16, 32), "float32")
            for i, j in T.Parallel(16, 32):
                e1[i, j] = T.cast((((i * 32 + j) + 63) % 512), "int32")
            for i, j in T.Parallel(16, 32):
                e2[i, j] = T.cast(True, "bool")
            for i, j in T.Parallel(16, 32):
                e3[i, j] = T.cast(T.if_then_else((e2[i, j] and e1[i, j] >= 0 and e1[i, j] < 512), mem0[bid, 16 + e1[i, j] * 1], T.cast(0, "float16")), "float16")
            for i, j in T.Parallel(16, 32):
                e4[i, j] = T.cast((e3[i, j] * e3[i, j]), "float16")
            for i, j in T.Parallel(16, 32):
                e5[i, j] = T.cast(e4[i, j], "float16")
            for i in T.Parallel(16):
                e14[i] = T.cast(((i + 11) % 16), "int32")
            for i in T.Parallel(16):
                e15[i] = T.cast(True, "bool")
            for i in T.Parallel(16):
                e16[i] = T.cast(T.if_then_else((e15[i] and e14[i] >= 0 and e14[i] < 16), mem1[bid, 16 + e14[i] * 1], T.cast(0, "float32")), "float32")
            mixed_helper_extended_3_1(e5, e16, mem0, mem1, mem2, steps, limit, bid, e17, e18)
            e19[0] = T.cast(steps, "int32")
            e20[0] = T.cast(0, "int32")
            e21[0] = T.cast(0, "int32")
            e22[0] = T.cast(True, "bool")
            e23[0] = T.cast(T.if_then_else((e22[0] and e21[0] >= 0 and e21[0] < 1), mem2[bid, 16 + e21[0] * 1], T.cast(0, "int32")), "int32")
            e24[0] = T.cast(0, "int32")
            e25[0] = T.cast((e23[0] < e24[0]), "bool")
            for i, j in T.Parallel(16, 32):
                e45[i, j] = e17[i, j]
            e46[0] = e18[0]
            e47[0] = e20[0]
            for e45_iteration in T.serial(T.min(T.max(e19[0], 0), 4)):
                e26[0] = e45_iteration
                for i, j in T.Parallel(16, 32):
                    e27[i, j] = e45[i, j]
                e28[0] = e46[0]
                e29[0] = e47[0]
                if e25[0]:
                    for i, j in T.Parallel(16, 32):
                        e30[i, j] = e27[i, j]
                    for i, j in T.Parallel(16, 32):
                        e31[i, j] = T.cast(0.125, "float16")
                    for i, j in T.Parallel(16, 32):
                        e32[i, j] = T.cast((e30[i, j] + e31[i, j]), "float16")
                    for i, j in T.Parallel(16, 32):
                        e36[i, j] = e32[i, j]
                else:
                    for i, j in T.Parallel(16, 32):
                        e33[i, j] = e27[i, j]
                    for i, j in T.Parallel(16, 32):
                        e34[i, j] = T.cast(0.125, "float16")
                    for i, j in T.Parallel(16, 32):
                        e35[i, j] = T.cast((e33[i, j] - e34[i, j]), "float16")
                    for i, j in T.Parallel(16, 32):
                        e36[i, j] = e35[i, j]
                mixed_helper_extended_3_1(e36, e16, mem0, mem1, mem2, steps, limit, bid, e37, e38)
                for i, j in T.Parallel(16, 32):
                    e39[i, j] = T.cast(0.125, "float16")
                for i, j in T.Parallel(16, 32):
                    e40[i, j] = T.cast((e37[i, j] * e39[i, j]), "float16")
                e41[0] = T.cast((e28[0] + e38[0]), "float32")
                e42[0] = T.cast(1, "int32")
                e43[0] = T.cast((e26[0] + e42[0]), "int32")
                e44[0] = T.cast((e29[0] + e43[0]), "int32")
                for i, j in T.Parallel(16, 32):
                    e45[i, j] = e40[i, j]
                e46[0] = e41[0]
                e47[0] = e44[0]
            for i, j in T.Parallel(16, 32):
                e67[i, j] = e45[i, j]
            e68[0] = e46[0]
            e69[0] = e47[0]
            for e67_iteration in T.serial(T.min(T.max(e19[0], 0), 4)):
                e48[0] = e67_iteration
                for i, j in T.Parallel(16, 32):
                    e49[i, j] = e67[i, j]
                e50[0] = e68[0]
                e51[0] = e69[0]
                if e25[0]:
                    for i, j in T.Parallel(16, 32):
                        e52[i, j] = e49[i, j]
                    for i, j in T.Parallel(16, 32):
                        e53[i, j] = T.cast(0.125, "float16")
                    for i, j in T.Parallel(16, 32):
                        e54[i, j] = T.cast((e52[i, j] + e53[i, j]), "float16")
                    for i, j in T.Parallel(16, 32):
                        e58[i, j] = e54[i, j]
                else:
                    for i, j in T.Parallel(16, 32):
                        e55[i, j] = e49[i, j]
                    for i, j in T.Parallel(16, 32):
                        e56[i, j] = T.cast(0.125, "float16")
                    for i, j in T.Parallel(16, 32):
                        e57[i, j] = T.cast((e55[i, j] - e56[i, j]), "float16")
                    for i, j in T.Parallel(16, 32):
                        e58[i, j] = e57[i, j]
                mixed_helper_extended_3_1(e58, e16, mem0, mem1, mem2, steps, limit, bid, e59, e60)
                for i, j in T.Parallel(16, 32):
                    e61[i, j] = T.cast(0.125, "float16")
                for i, j in T.Parallel(16, 32):
                    e62[i, j] = T.cast((e59[i, j] * e61[i, j]), "float16")
                e63[0] = T.cast((e50[0] + e60[0]), "float32")
                e64[0] = T.cast(1, "int32")
                e65[0] = T.cast((e48[0] + e64[0]), "int32")
                e66[0] = T.cast((e51[0] + e65[0]), "int32")
                for i, j in T.Parallel(16, 32):
                    e67[i, j] = e62[i, j]
                e68[0] = e63[0]
                e69[0] = e66[0]
            T.copy(e67, e70_flip_shared)
            T.sync_threads()
            for i, j in T.Parallel(16, 32):
                e70[i, j] = T.cast(e70_flip_shared[i, 31 - j], "float16")
            for i, j in T.Parallel(16, 32):
                e71[i, j] = T.cast((e70[i, j] - e3[i, j]), "float16")
            for i, j in T.Parallel(16, 32):
                e72[i, j] = T.cast((e71[i, j] - e71[i, j]), "float16")
            for i, j in T.Parallel(16, 32):
                e73[i, j] = T.cast((e72[i, j] * e3[i, j]), "float16")
            for i, j in T.Parallel(16, 32):
                e74_wide[i, j] = T.cast(e73[i, j], 'float32')
            T.reduce_sum(e74_wide, e74, dim=1, clear=True)
            for i, j in T.Parallel(16, 32):
                out_e73[bid, 16 + (i * 32 + j)] = e73[i, j]
            out_e68[bid, 16 + 0] = e68[0]
            out_e69[bid, 16 + 0] = e69[0]
            for i in T.Parallel(16):
                out_e74[bid, 16 + i] = e74[i]
            for i, j in T.Parallel(16, 32):
                out_e3[bid, 16 + (i * 32 + j)] = e3[i, j]
            for i, j in T.Parallel(16, 32):
                out_e70[bid, 16 + (i * 32 + j)] = e70[i, j]
            out_e19[bid, 16 + 0] = e19[0]
            out_e22[bid, 16 + 0] = e22[0]
    return impl

@tilelang.jit
def extended_0_ident():
    @T.macro
    def mixed_helper_extended_0_ident(e6, e7, mem0, mem1, mem2, steps, limit, bid, result_e13, result_e10):
        e8 = T.alloc_fragment((16,), "float32")
        e9 = T.alloc_fragment((16,), "float32")
        e10 = T.alloc_fragment((1,), "float32")
        e11 = T.alloc_fragment((16, 1), "float32")
        e12 = T.alloc_fragment((16, 1), "float16")
        e13 = T.alloc_fragment((16, 32), "float16")
        e8_wide = T.alloc_fragment((16, 32), "float32")
        e10_wide = T.alloc_fragment((16,), "float32")
        for i, j in T.Parallel(16, 32):
            e8_wide[i, j] = T.cast(e6[i, j], 'float32')
        T.reduce_sum(e8_wide, e8, dim=1, clear=True)
        for i in T.Parallel(16):
            e9[i] = T.cast((e8[i] + e7[i]), "float32")
        for i in T.Parallel(16):
            e10_wide[i] = T.cast(e9[i], 'float32')
        T.reduce_sum(e10_wide, e10, dim=0, clear=True)
        for i, j in T.Parallel(16, 1):
            e11[i, 0] = T.cast(e9[(i * 1 + j)], "float32")
        for i, j in T.Parallel(16, 1):
            e12[i, 0] = T.cast(e11[i, 0], "float16")
        for i, j in T.Parallel(16, 32):
            e13[i, j] = T.cast((e6[i, j] + e12[i, 0]), "float16")
        for i, j in T.Parallel(16, 32):
            result_e13[i, j] = e13[i, j]
        result_e10[0] = e10[0]
    @T.prim_func
    def impl(mem0: T.Buffer((2, 544), "float16"), mem1: T.Buffer((2, 48), "float32"), mem2: T.Buffer((2, 33), "int32"), out_e73: T.Buffer((2, 544), "float16"), out_e68: T.Buffer((2, 33), "float32"), out_e69: T.Buffer((2, 33), "int32"), out_e74: T.Buffer((2, 48), "float32"), out_e3: T.Buffer((2, 544), "float16"), out_e70: T.Buffer((2, 544), "float16"), out_e19: T.Buffer((2, 33), "int32"), out_e22: T.Buffer((2, 33), "bool"), steps: T.int32, limit: T.int32):
        with T.Kernel(2, threads=128) as bid:
            e1 = T.alloc_fragment((16, 32), "int32")
            e2 = T.alloc_fragment((16, 32), "bool")
            e3 = T.alloc_fragment((16, 32), "float16")
            e4 = T.alloc_fragment((16, 32), "float16")
            e5 = T.alloc_fragment((16, 32), "float16")
            e14 = T.alloc_fragment((16,), "int32")
            e15 = T.alloc_fragment((16,), "bool")
            e16 = T.alloc_fragment((16,), "float32")
            e17 = T.alloc_fragment((16, 32), "float16")
            e18 = T.alloc_fragment((1,), "float32")
            e19 = T.alloc_fragment((1,), "int32")
            e20 = T.alloc_fragment((1,), "int32")
            e21 = T.alloc_fragment((1,), "int32")
            e22 = T.alloc_fragment((1,), "bool")
            e23 = T.alloc_fragment((1,), "int32")
            e24 = T.alloc_fragment((1,), "int32")
            e25 = T.alloc_fragment((1,), "bool")
            e45 = T.alloc_fragment((16, 32), "float16")
            e46 = T.alloc_fragment((1,), "float32")
            e47 = T.alloc_fragment((1,), "int32")
            e36 = T.alloc_fragment((16, 32), "float16")
            e31 = T.alloc_fragment((16, 32), "float16")
            e32 = T.alloc_fragment((16, 32), "float16")
            e34 = T.alloc_fragment((16, 32), "float16")
            e35 = T.alloc_fragment((16, 32), "float16")
            e37 = T.alloc_fragment((16, 32), "float16")
            e38 = T.alloc_fragment((1,), "float32")
            e39 = T.alloc_fragment((16, 32), "float16")
            e40 = T.alloc_fragment((16, 32), "float16")
            e41 = T.alloc_fragment((1,), "float32")
            e42 = T.alloc_fragment((1,), "int32")
            e43 = T.alloc_fragment((1,), "int32")
            e44 = T.alloc_fragment((1,), "int32")
            e67 = T.alloc_fragment((16, 32), "float16")
            e68 = T.alloc_fragment((1,), "float32")
            e69 = T.alloc_fragment((1,), "int32")
            e58 = T.alloc_fragment((16, 32), "float16")
            e53 = T.alloc_fragment((16, 32), "float16")
            e54 = T.alloc_fragment((16, 32), "float16")
            e56 = T.alloc_fragment((16, 32), "float16")
            e57 = T.alloc_fragment((16, 32), "float16")
            e59 = T.alloc_fragment((16, 32), "float16")
            e60 = T.alloc_fragment((1,), "float32")
            e61 = T.alloc_fragment((16, 32), "float16")
            e62 = T.alloc_fragment((16, 32), "float16")
            e63 = T.alloc_fragment((1,), "float32")
            e64 = T.alloc_fragment((1,), "int32")
            e65 = T.alloc_fragment((1,), "int32")
            e66 = T.alloc_fragment((1,), "int32")
            e70 = T.alloc_fragment((16, 32), "float16")
            e71 = T.alloc_fragment((16, 32), "float16")
            e72 = T.alloc_fragment((16, 32), "float16")
            e73_l = T.alloc_fragment((16, 32), "float16")
            e73_r = T.alloc_fragment((16, 32), "float16")
            e73 = T.alloc_fragment((16, 32), "float16")
            e74 = T.alloc_fragment((16,), "float32")
            e26 = T.alloc_fragment((1,), "int32")
            e27 = T.alloc_fragment((16, 32), "float16")
            e28 = T.alloc_fragment((1,), "float32")
            e29 = T.alloc_fragment((1,), "int32")
            e30 = T.alloc_fragment((16, 32), "float16")
            e33 = T.alloc_fragment((16, 32), "float16")
            e48 = T.alloc_fragment((1,), "int32")
            e49 = T.alloc_fragment((16, 32), "float16")
            e50 = T.alloc_fragment((1,), "float32")
            e51 = T.alloc_fragment((1,), "int32")
            e52 = T.alloc_fragment((16, 32), "float16")
            e55 = T.alloc_fragment((16, 32), "float16")
            e70_flip_shared = T.alloc_shared((16, 32), "float16")
            e74_wide = T.alloc_fragment((16, 32), "float32")
            for i, j in T.Parallel(16, 32):
                e1[i, j] = T.cast((((i * 32 + j) + 63) % 512), "int32")
            for i, j in T.Parallel(16, 32):
                e2[i, j] = T.cast(True, "bool")
            for i, j in T.Parallel(16, 32):
                e3[i, j] = T.cast(T.if_then_else((e2[i, j] and e1[i, j] >= 0 and e1[i, j] < 512), mem0[bid, 16 + e1[i, j] * 1], T.cast(0, "float16")), "float16")
            for i, j in T.Parallel(16, 32):
                e4[i, j] = T.cast((e3[i, j] * e3[i, j]), "float16")
            for i, j in T.Parallel(16, 32):
                e5[i, j] = T.cast(e4[i, j], "float16")
            for i in T.Parallel(16):
                e14[i] = T.cast(((i + 11) % 16), "int32")
            for i in T.Parallel(16):
                e15[i] = T.cast(True, "bool")
            for i in T.Parallel(16):
                e16[i] = T.cast(T.if_then_else((e15[i] and e14[i] >= 0 and e14[i] < 16), mem1[bid, 16 + e14[i] * 1], T.cast(0, "float32")), "float32")
            mixed_helper_extended_0_ident(e5, e16, mem0, mem1, mem2, steps, limit, bid, e17, e18)
            e19[0] = T.cast(steps, "int32")
            e20[0] = T.cast(0, "int32")
            e21[0] = T.cast(0, "int32")
            e22[0] = T.cast(True, "bool")
            e23[0] = T.cast(T.if_then_else((e22[0] and e21[0] >= 0 and e21[0] < 1), mem2[bid, 16 + e21[0] * 1], T.cast(0, "int32")), "int32")
            e24[0] = T.cast(0, "int32")
            e25[0] = T.cast((e23[0] < e24[0]), "bool")
            for i, j in T.Parallel(16, 32):
                e45[i, j] = e17[i, j]
            e46[0] = e18[0]
            e47[0] = e20[0]
            for e45_iteration in T.serial(T.min(T.max(e19[0], 0), 4)):
                e26[0] = e45_iteration
                for i, j in T.Parallel(16, 32):
                    e27[i, j] = e45[i, j]
                e28[0] = e46[0]
                e29[0] = e47[0]
                if e25[0]:
                    for i, j in T.Parallel(16, 32):
                        e30[i, j] = e27[i, j]
                    for i, j in T.Parallel(16, 32):
                        e31[i, j] = T.cast(0.125, "float16")
                    for i, j in T.Parallel(16, 32):
                        e32[i, j] = T.cast((e30[i, j] + e31[i, j]), "float16")
                    for i, j in T.Parallel(16, 32):
                        e36[i, j] = e32[i, j]
                else:
                    for i, j in T.Parallel(16, 32):
                        e33[i, j] = e27[i, j]
                    for i, j in T.Parallel(16, 32):
                        e34[i, j] = T.cast(0.125, "float16")
                    for i, j in T.Parallel(16, 32):
                        e35[i, j] = T.cast((e33[i, j] - e34[i, j]), "float16")
                    for i, j in T.Parallel(16, 32):
                        e36[i, j] = e35[i, j]
                mixed_helper_extended_0_ident(e36, e16, mem0, mem1, mem2, steps, limit, bid, e37, e38)
                for i, j in T.Parallel(16, 32):
                    e39[i, j] = T.cast(0.125, "float16")
                for i, j in T.Parallel(16, 32):
                    e40[i, j] = T.cast((e37[i, j] * e39[i, j]), "float16")
                e41[0] = T.cast((e28[0] + e38[0]), "float32")
                e42[0] = T.cast(1, "int32")
                e43[0] = T.cast((e26[0] + e42[0]), "int32")
                e44[0] = T.cast((e29[0] + e43[0]), "int32")
                for i, j in T.Parallel(16, 32):
                    e45[i, j] = e40[i, j]
                e46[0] = e41[0]
                e47[0] = e44[0]
            for i, j in T.Parallel(16, 32):
                e67[i, j] = e45[i, j]
            e68[0] = e46[0]
            e69[0] = e47[0]
            for e67_iteration in T.serial(T.min(T.max(e19[0], 0), 4)):
                e48[0] = e67_iteration
                for i, j in T.Parallel(16, 32):
                    e49[i, j] = e67[i, j]
                e50[0] = e68[0]
                e51[0] = e69[0]
                if e25[0]:
                    for i, j in T.Parallel(16, 32):
                        e52[i, j] = e49[i, j]
                    for i, j in T.Parallel(16, 32):
                        e53[i, j] = T.cast(0.125, "float16")
                    for i, j in T.Parallel(16, 32):
                        e54[i, j] = T.cast((e52[i, j] + e53[i, j]), "float16")
                    for i, j in T.Parallel(16, 32):
                        e58[i, j] = e54[i, j]
                else:
                    for i, j in T.Parallel(16, 32):
                        e55[i, j] = e49[i, j]
                    for i, j in T.Parallel(16, 32):
                        e56[i, j] = T.cast(0.125, "float16")
                    for i, j in T.Parallel(16, 32):
                        e57[i, j] = T.cast((e55[i, j] - e56[i, j]), "float16")
                    for i, j in T.Parallel(16, 32):
                        e58[i, j] = e57[i, j]
                mixed_helper_extended_0_ident(e58, e16, mem0, mem1, mem2, steps, limit, bid, e59, e60)
                for i, j in T.Parallel(16, 32):
                    e61[i, j] = T.cast(0.125, "float16")
                for i, j in T.Parallel(16, 32):
                    e62[i, j] = T.cast((e59[i, j] * e61[i, j]), "float16")
                e63[0] = T.cast((e50[0] + e60[0]), "float32")
                e64[0] = T.cast(1, "int32")
                e65[0] = T.cast((e48[0] + e64[0]), "int32")
                e66[0] = T.cast((e51[0] + e65[0]), "int32")
                for i, j in T.Parallel(16, 32):
                    e67[i, j] = e62[i, j]
                e68[0] = e63[0]
                e69[0] = e66[0]
            T.copy(e67, e70_flip_shared)
            T.sync_threads()
            for i, j in T.Parallel(16, 32):
                e70[i, j] = T.cast(e70_flip_shared[i, 31 - j], "float16")
            for i, j in T.Parallel(16, 32):
                e71[i, j] = T.cast((e70[i, j] - e3[i, j]), "float16")
            for i, j in T.Parallel(16, 32):
                e72[i, j] = T.cast((e71[i, j] - e71[i, j]), "float16")
            for i, j in T.Parallel(16, 32):
                e73_l[i, j] = T.cast((e3[i, j] * e71[i, j]), "float16")
            for i, j in T.Parallel(16, 32):
                e73_r[i, j] = T.cast((e3[i, j] * e71[i, j]), "float16")
            for i, j in T.Parallel(16, 32):
                e73[i, j] = T.cast((e73_l[i, j] - e73_r[i, j]), "float16")
            for i, j in T.Parallel(16, 32):
                e74_wide[i, j] = T.cast(e73[i, j], 'float32')
            T.reduce_sum(e74_wide, e74, dim=1, clear=True)
            for i, j in T.Parallel(16, 32):
                out_e73[bid, 16 + (i * 32 + j)] = e73[i, j]
            out_e68[bid, 16 + 0] = e68[0]
            out_e69[bid, 16 + 0] = e69[0]
            for i in T.Parallel(16):
                out_e74[bid, 16 + i] = e74[i]
            for i, j in T.Parallel(16, 32):
                out_e3[bid, 16 + (i * 32 + j)] = e3[i, j]
            for i, j in T.Parallel(16, 32):
                out_e70[bid, 16 + (i * 32 + j)] = e70[i, j]
            out_e19[bid, 16 + 0] = e19[0]
            out_e22[bid, 16 + 0] = e22[0]
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
    compile_options = [{'threads': 128, 'stages': 1, 'pass_configs': {}}, {'threads': 128, 'stages': 1, 'pass_configs': {}}, {'threads': 256, 'stages': 2, 'pass_configs': {'tirx.disable_vectorize': True}}, {'threads': 256, 'stages': 2, 'pass_configs': {'tirx.disable_vectorize': True}}, {'threads': 256, 'stages': 2, 'pass_configs': {'tl.disable_warp_specialized': True, 'tl.enable_aggressive_shared_memory_merge': True, 'tl.loop_unswitching_allow_non_trivial_else': True, 'tl.storage_rewrite_detect_inplace': True, 'tl.if_stmt_binding_inline_replayable_binds': True, 'tl.disable_out_of_bound_warning': True}}, {'threads': 256, 'stages': 2, 'pass_configs': {'tl.disable_warp_specialized': True, 'tl.enable_aggressive_shared_memory_merge': True, 'tl.loop_unswitching_allow_non_trivial_else': True, 'tl.storage_rewrite_detect_inplace': True, 'tl.if_stmt_binding_inline_replayable_binds': True, 'tl.disable_out_of_bound_warning': True}}, {'threads': 256, 'stages': 3, 'pass_configs': {'tl.disable_data_race_check': True, 'tl.disable_shared_memory_reuse': True, 'tl.loop_unswitching_allow_non_trivial_else': True, 'tl.storage_rewrite_detect_inplace': True, 'tl.enable_async_copy': True, 'tl.enable_lower_ldgstg': True, 'tl.enable_lower_ldgstg_predicated': True, 'tl.disable_out_of_bound_warning': True, 'tl.ptxas_register_usage_level': 3}}, {'threads': 256, 'stages': 3, 'pass_configs': {'tl.disable_data_race_check': True, 'tl.disable_shared_memory_reuse': True, 'tl.loop_unswitching_allow_non_trivial_else': True, 'tl.storage_rewrite_detect_inplace': True, 'tl.enable_async_copy': True, 'tl.enable_lower_ldgstg': True, 'tl.enable_lower_ldgstg_predicated': True, 'tl.disable_out_of_bound_warning': True, 'tl.ptxas_register_usage_level': 3}}, {'threads': 128, 'stages': 1, 'pass_configs': {}, 'identity': True}]
    watched = [['e73', 'e68', 'e69', 'e74'], ['e73', 'e68', 'e69', 'e74', 'e3', 'e70', 'e19', 'e22'], ['e73', 'e68', 'e69', 'e74'], ['e73', 'e68', 'e69', 'e74', 'e3', 'e70', 'e19', 'e22'], ['e73', 'e68', 'e69', 'e74'], ['e73', 'e68', 'e69', 'e74', 'e3', 'e70', 'e19', 'e22'], ['e73', 'e68', 'e69', 'e74'], ['e73', 'e68', 'e69', 'e74', 'e3', 'e70', 'e19', 'e22'], ['e73', 'e68', 'e69', 'e74', 'e3', 'e70', 'e19', 'e22']]
    roots = ['mem0', 'mem1', 'mem2']
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

PROGRAM = {'type': 'extended', 'version': 1, 'body': {'arguments': [], 'operations': [{'op': 'index', 'results': [{'name': 'e1', 'type': {'dtype': 'int32', 'shape': (16, 32)}}], 'operands': [], 'attrs': {'shift': 63, 'reverse': False}, 'regions': []}, {'op': 'constant', 'results': [{'name': 'e2', 'type': {'dtype': 'bool', 'shape': (16, 32)}}], 'operands': [], 'attrs': {'value': True}, 'regions': []}, {'op': 'load', 'results': [{'name': 'e3', 'type': {'dtype': 'float16', 'shape': (16, 32)}}], 'operands': ['e1', 'e2'], 'attrs': {'buffer': 'mem0'}, 'regions': []}, {'op': 'mul', 'results': [{'name': 'e4', 'type': {'dtype': 'float16', 'shape': (16, 32)}}], 'operands': ['e3', 'e3'], 'attrs': {}, 'regions': []}, {'op': 'cast', 'results': [{'name': 'e5', 'type': {'dtype': 'float16', 'shape': (16, 32)}}], 'operands': ['e4'], 'attrs': {}, 'regions': []}, {'op': 'index', 'results': [{'name': 'e14', 'type': {'dtype': 'int32', 'shape': (16,)}}], 'operands': [], 'attrs': {'shift': 11, 'reverse': False}, 'regions': []}, {'op': 'constant', 'results': [{'name': 'e15', 'type': {'dtype': 'bool', 'shape': (16,)}}], 'operands': [], 'attrs': {'value': True}, 'regions': []}, {'op': 'load', 'results': [{'name': 'e16', 'type': {'dtype': 'float32', 'shape': (16,)}}], 'operands': ['e14', 'e15'], 'attrs': {'buffer': 'mem1'}, 'regions': []}, {'op': 'call', 'results': [{'name': 'e17', 'type': {'dtype': 'float16', 'shape': (16, 32)}}, {'name': 'e18', 'type': {'dtype': 'float32', 'shape': ()}}], 'operands': ['e5', 'e16'], 'attrs': {'callee': 'mixed_helper'}, 'regions': []}, {'op': 'parameter', 'results': [{'name': 'e19', 'type': {'dtype': 'int32', 'shape': ()}}], 'operands': [], 'attrs': {'name': 'steps'}, 'regions': []}, {'op': 'constant', 'results': [{'name': 'e20', 'type': {'dtype': 'int32', 'shape': ()}}], 'operands': [], 'attrs': {'value': 0}, 'regions': []}, {'op': 'constant', 'results': [{'name': 'e21', 'type': {'dtype': 'int32', 'shape': ()}}], 'operands': [], 'attrs': {'value': 0}, 'regions': []}, {'op': 'constant', 'results': [{'name': 'e22', 'type': {'dtype': 'bool', 'shape': ()}}], 'operands': [], 'attrs': {'value': True}, 'regions': []}, {'op': 'load', 'results': [{'name': 'e23', 'type': {'dtype': 'int32', 'shape': ()}}], 'operands': ['e21', 'e22'], 'attrs': {'buffer': 'mem2'}, 'regions': []}, {'op': 'constant', 'results': [{'name': 'e24', 'type': {'dtype': 'int32', 'shape': ()}}], 'operands': [], 'attrs': {'value': 0}, 'regions': []}, {'op': 'lt', 'results': [{'name': 'e25', 'type': {'dtype': 'bool', 'shape': ()}}], 'operands': ['e23', 'e24'], 'attrs': {}, 'regions': []}, {'op': 'for', 'results': [{'name': 'e45', 'type': {'dtype': 'float16', 'shape': (16, 32)}}, {'name': 'e46', 'type': {'dtype': 'float32', 'shape': ()}}, {'name': 'e47', 'type': {'dtype': 'int32', 'shape': ()}}], 'operands': ['e19', 'e17', 'e18', 'e20'], 'attrs': {'max_steps': 4, 'pipelined': False}, 'regions': [{'arguments': [{'name': 'e26', 'type': {'dtype': 'int32', 'shape': ()}}, {'name': 'e27', 'type': {'dtype': 'float16', 'shape': (16, 32)}}, {'name': 'e28', 'type': {'dtype': 'float32', 'shape': ()}}, {'name': 'e29', 'type': {'dtype': 'int32', 'shape': ()}}], 'operations': [{'op': 'if', 'results': [{'name': 'e36', 'type': {'dtype': 'float16', 'shape': (16, 32)}}], 'operands': ['e25', 'e27'], 'attrs': {}, 'regions': [{'arguments': [{'name': 'e30', 'type': {'dtype': 'float16', 'shape': (16, 32)}}], 'operations': [{'op': 'constant', 'results': [{'name': 'e31', 'type': {'dtype': 'float16', 'shape': (16, 32)}}], 'operands': [], 'attrs': {'value': 0.125}, 'regions': []}, {'op': 'add', 'results': [{'name': 'e32', 'type': {'dtype': 'float16', 'shape': (16, 32)}}], 'operands': ['e30', 'e31'], 'attrs': {}, 'regions': []}], 'returns': ['e32']}, {'arguments': [{'name': 'e33', 'type': {'dtype': 'float16', 'shape': (16, 32)}}], 'operations': [{'op': 'constant', 'results': [{'name': 'e34', 'type': {'dtype': 'float16', 'shape': (16, 32)}}], 'operands': [], 'attrs': {'value': 0.125}, 'regions': []}, {'op': 'sub', 'results': [{'name': 'e35', 'type': {'dtype': 'float16', 'shape': (16, 32)}}], 'operands': ['e33', 'e34'], 'attrs': {}, 'regions': []}], 'returns': ['e35']}]}, {'op': 'call', 'results': [{'name': 'e37', 'type': {'dtype': 'float16', 'shape': (16, 32)}}, {'name': 'e38', 'type': {'dtype': 'float32', 'shape': ()}}], 'operands': ['e36', 'e16'], 'attrs': {'callee': 'mixed_helper'}, 'regions': []}, {'op': 'constant', 'results': [{'name': 'e39', 'type': {'dtype': 'float16', 'shape': (16, 32)}}], 'operands': [], 'attrs': {'value': 0.125}, 'regions': []}, {'op': 'mul', 'results': [{'name': 'e40', 'type': {'dtype': 'float16', 'shape': (16, 32)}}], 'operands': ['e37', 'e39'], 'attrs': {}, 'regions': []}, {'op': 'add', 'results': [{'name': 'e41', 'type': {'dtype': 'float32', 'shape': ()}}], 'operands': ['e28', 'e38'], 'attrs': {}, 'regions': []}, {'op': 'constant', 'results': [{'name': 'e42', 'type': {'dtype': 'int32', 'shape': ()}}], 'operands': [], 'attrs': {'value': 1}, 'regions': []}, {'op': 'add', 'results': [{'name': 'e43', 'type': {'dtype': 'int32', 'shape': ()}}], 'operands': ['e26', 'e42'], 'attrs': {}, 'regions': []}, {'op': 'add', 'results': [{'name': 'e44', 'type': {'dtype': 'int32', 'shape': ()}}], 'operands': ['e29', 'e43'], 'attrs': {}, 'regions': []}], 'returns': ['e40', 'e41', 'e44']}]}, {'op': 'while', 'results': [{'name': 'e67', 'type': {'dtype': 'float16', 'shape': (16, 32)}}, {'name': 'e68', 'type': {'dtype': 'float32', 'shape': ()}}, {'name': 'e69', 'type': {'dtype': 'int32', 'shape': ()}}], 'operands': ['e19', 'e45', 'e46', 'e47'], 'attrs': {'max_steps': 4, 'pipelined': False}, 'regions': [{'arguments': [{'name': 'e48', 'type': {'dtype': 'int32', 'shape': ()}}, {'name': 'e49', 'type': {'dtype': 'float16', 'shape': (16, 32)}}, {'name': 'e50', 'type': {'dtype': 'float32', 'shape': ()}}, {'name': 'e51', 'type': {'dtype': 'int32', 'shape': ()}}], 'operations': [{'op': 'if', 'results': [{'name': 'e58', 'type': {'dtype': 'float16', 'shape': (16, 32)}}], 'operands': ['e25', 'e49'], 'attrs': {}, 'regions': [{'arguments': [{'name': 'e52', 'type': {'dtype': 'float16', 'shape': (16, 32)}}], 'operations': [{'op': 'constant', 'results': [{'name': 'e53', 'type': {'dtype': 'float16', 'shape': (16, 32)}}], 'operands': [], 'attrs': {'value': 0.125}, 'regions': []}, {'op': 'add', 'results': [{'name': 'e54', 'type': {'dtype': 'float16', 'shape': (16, 32)}}], 'operands': ['e52', 'e53'], 'attrs': {}, 'regions': []}], 'returns': ['e54']}, {'arguments': [{'name': 'e55', 'type': {'dtype': 'float16', 'shape': (16, 32)}}], 'operations': [{'op': 'constant', 'results': [{'name': 'e56', 'type': {'dtype': 'float16', 'shape': (16, 32)}}], 'operands': [], 'attrs': {'value': 0.125}, 'regions': []}, {'op': 'sub', 'results': [{'name': 'e57', 'type': {'dtype': 'float16', 'shape': (16, 32)}}], 'operands': ['e55', 'e56'], 'attrs': {}, 'regions': []}], 'returns': ['e57']}]}, {'op': 'call', 'results': [{'name': 'e59', 'type': {'dtype': 'float16', 'shape': (16, 32)}}, {'name': 'e60', 'type': {'dtype': 'float32', 'shape': ()}}], 'operands': ['e58', 'e16'], 'attrs': {'callee': 'mixed_helper'}, 'regions': []}, {'op': 'constant', 'results': [{'name': 'e61', 'type': {'dtype': 'float16', 'shape': (16, 32)}}], 'operands': [], 'attrs': {'value': 0.125}, 'regions': []}, {'op': 'mul', 'results': [{'name': 'e62', 'type': {'dtype': 'float16', 'shape': (16, 32)}}], 'operands': ['e59', 'e61'], 'attrs': {}, 'regions': []}, {'op': 'add', 'results': [{'name': 'e63', 'type': {'dtype': 'float32', 'shape': ()}}], 'operands': ['e50', 'e60'], 'attrs': {}, 'regions': []}, {'op': 'constant', 'results': [{'name': 'e64', 'type': {'dtype': 'int32', 'shape': ()}}], 'operands': [], 'attrs': {'value': 1}, 'regions': []}, {'op': 'add', 'results': [{'name': 'e65', 'type': {'dtype': 'int32', 'shape': ()}}], 'operands': ['e48', 'e64'], 'attrs': {}, 'regions': []}, {'op': 'add', 'results': [{'name': 'e66', 'type': {'dtype': 'int32', 'shape': ()}}], 'operands': ['e51', 'e65'], 'attrs': {}, 'regions': []}], 'returns': ['e62', 'e63', 'e66']}]}, {'op': 'flip', 'results': [{'name': 'e70', 'type': {'dtype': 'float16', 'shape': (16, 32)}}], 'operands': ['e67'], 'attrs': {}, 'regions': []}, {'op': 'sub', 'results': [{'name': 'e71', 'type': {'dtype': 'float16', 'shape': (16, 32)}}], 'operands': ['e70', 'e3'], 'attrs': {}, 'regions': []}, {'op': 'sub', 'results': [{'name': 'e72', 'type': {'dtype': 'float16', 'shape': (16, 32)}}], 'operands': ['e71', 'e71'], 'attrs': {}, 'regions': []}, {'op': 'mul', 'results': [{'name': 'e73', 'type': {'dtype': 'float16', 'shape': (16, 32)}}], 'operands': ['e72', 'e3'], 'attrs': {}, 'regions': []}, {'op': 'reduce', 'results': [{'name': 'e74', 'type': {'dtype': 'float32', 'shape': (16,)}}], 'operands': ['e73'], 'attrs': {'axis': 1, 'kind': 'sum'}, 'regions': []}], 'returns': ['e73', 'e68', 'e69', 'e74']}, 'buffers': [{'name': 'mem0', 'dtype': 'float16', 'size': 512, 'role': 'input', 'base': None, 'offset': 0, 'stride': 1}, {'name': 'mem1', 'dtype': 'float32', 'size': 16, 'role': 'input', 'base': None, 'offset': 0, 'stride': 1}, {'name': 'mem2', 'dtype': 'int32', 'size': 1, 'role': 'input', 'base': None, 'offset': 0, 'stride': 1}], 'functions': [{'name': 'mixed_helper', 'body': {'arguments': [{'name': 'e6', 'type': {'dtype': 'float16', 'shape': (16, 32)}}, {'name': 'e7', 'type': {'dtype': 'float32', 'shape': (16,)}}], 'operations': [{'op': 'reduce', 'results': [{'name': 'e8', 'type': {'dtype': 'float32', 'shape': (16,)}}], 'operands': ['e6'], 'attrs': {'axis': 1, 'kind': 'sum'}, 'regions': []}, {'op': 'add', 'results': [{'name': 'e9', 'type': {'dtype': 'float32', 'shape': (16,)}}], 'operands': ['e8', 'e7'], 'attrs': {}, 'regions': []}, {'op': 'reduce', 'results': [{'name': 'e10', 'type': {'dtype': 'float32', 'shape': ()}}], 'operands': ['e9'], 'attrs': {'axis': 0, 'kind': 'sum'}, 'regions': []}, {'op': 'reshape', 'results': [{'name': 'e11', 'type': {'dtype': 'float32', 'shape': (16, 1)}}], 'operands': ['e9'], 'attrs': {}, 'regions': []}, {'op': 'cast', 'results': [{'name': 'e12', 'type': {'dtype': 'float16', 'shape': (16, 1)}}], 'operands': ['e11'], 'attrs': {}, 'regions': []}, {'op': 'add', 'results': [{'name': 'e13', 'type': {'dtype': 'float16', 'shape': (16, 32)}}], 'operands': ['e6', 'e12'], 'attrs': {}, 'regions': []}], 'returns': ['e13', 'e10']}}], 'observations': ['e3', 'e70', 'e19', 'e22'], 'blocks': 2, 'input_pattern': 'normal', 'runtime_cases': [(0, 1), (1, 511), (2, 512)], 'configuration_pair': True, 'observation_pair': True, 'pass_config_pair': False, 'fast_math_pair': False, 'precision_pair': True, 'identity_pair': True, 'family': 'control_calls'}

REFERENCE_PROGRAMS = {'tilelang_8_ident': {'type': 'extended', 'version': 1, 'body': {'arguments': [], 'operations': [{'op': 'index', 'results': [{'name': 'e1', 'type': {'dtype': 'int32', 'shape': (16, 32)}}], 'operands': [], 'attrs': {'shift': 63, 'reverse': False}, 'regions': []}, {'op': 'constant', 'results': [{'name': 'e2', 'type': {'dtype': 'bool', 'shape': (16, 32)}}], 'operands': [], 'attrs': {'value': True}, 'regions': []}, {'op': 'load', 'results': [{'name': 'e3', 'type': {'dtype': 'float16', 'shape': (16, 32)}}], 'operands': ['e1', 'e2'], 'attrs': {'buffer': 'mem0'}, 'regions': []}, {'op': 'mul', 'results': [{'name': 'e4', 'type': {'dtype': 'float16', 'shape': (16, 32)}}], 'operands': ['e3', 'e3'], 'attrs': {}, 'regions': []}, {'op': 'cast', 'results': [{'name': 'e5', 'type': {'dtype': 'float16', 'shape': (16, 32)}}], 'operands': ['e4'], 'attrs': {}, 'regions': []}, {'op': 'index', 'results': [{'name': 'e14', 'type': {'dtype': 'int32', 'shape': (16,)}}], 'operands': [], 'attrs': {'shift': 11, 'reverse': False}, 'regions': []}, {'op': 'constant', 'results': [{'name': 'e15', 'type': {'dtype': 'bool', 'shape': (16,)}}], 'operands': [], 'attrs': {'value': True}, 'regions': []}, {'op': 'load', 'results': [{'name': 'e16', 'type': {'dtype': 'float32', 'shape': (16,)}}], 'operands': ['e14', 'e15'], 'attrs': {'buffer': 'mem1'}, 'regions': []}, {'op': 'call', 'results': [{'name': 'e17', 'type': {'dtype': 'float16', 'shape': (16, 32)}}, {'name': 'e18', 'type': {'dtype': 'float32', 'shape': ()}}], 'operands': ['e5', 'e16'], 'attrs': {'callee': 'mixed_helper'}, 'regions': []}, {'op': 'parameter', 'results': [{'name': 'e19', 'type': {'dtype': 'int32', 'shape': ()}}], 'operands': [], 'attrs': {'name': 'steps'}, 'regions': []}, {'op': 'constant', 'results': [{'name': 'e20', 'type': {'dtype': 'int32', 'shape': ()}}], 'operands': [], 'attrs': {'value': 0}, 'regions': []}, {'op': 'constant', 'results': [{'name': 'e21', 'type': {'dtype': 'int32', 'shape': ()}}], 'operands': [], 'attrs': {'value': 0}, 'regions': []}, {'op': 'constant', 'results': [{'name': 'e22', 'type': {'dtype': 'bool', 'shape': ()}}], 'operands': [], 'attrs': {'value': True}, 'regions': []}, {'op': 'load', 'results': [{'name': 'e23', 'type': {'dtype': 'int32', 'shape': ()}}], 'operands': ['e21', 'e22'], 'attrs': {'buffer': 'mem2'}, 'regions': []}, {'op': 'constant', 'results': [{'name': 'e24', 'type': {'dtype': 'int32', 'shape': ()}}], 'operands': [], 'attrs': {'value': 0}, 'regions': []}, {'op': 'lt', 'results': [{'name': 'e25', 'type': {'dtype': 'bool', 'shape': ()}}], 'operands': ['e23', 'e24'], 'attrs': {}, 'regions': []}, {'op': 'for', 'results': [{'name': 'e45', 'type': {'dtype': 'float16', 'shape': (16, 32)}}, {'name': 'e46', 'type': {'dtype': 'float32', 'shape': ()}}, {'name': 'e47', 'type': {'dtype': 'int32', 'shape': ()}}], 'operands': ['e19', 'e17', 'e18', 'e20'], 'attrs': {'max_steps': 4, 'pipelined': False}, 'regions': [{'arguments': [{'name': 'e26', 'type': {'dtype': 'int32', 'shape': ()}}, {'name': 'e27', 'type': {'dtype': 'float16', 'shape': (16, 32)}}, {'name': 'e28', 'type': {'dtype': 'float32', 'shape': ()}}, {'name': 'e29', 'type': {'dtype': 'int32', 'shape': ()}}], 'operations': [{'op': 'if', 'results': [{'name': 'e36', 'type': {'dtype': 'float16', 'shape': (16, 32)}}], 'operands': ['e25', 'e27'], 'attrs': {}, 'regions': [{'arguments': [{'name': 'e30', 'type': {'dtype': 'float16', 'shape': (16, 32)}}], 'operations': [{'op': 'constant', 'results': [{'name': 'e31', 'type': {'dtype': 'float16', 'shape': (16, 32)}}], 'operands': [], 'attrs': {'value': 0.125}, 'regions': []}, {'op': 'add', 'results': [{'name': 'e32', 'type': {'dtype': 'float16', 'shape': (16, 32)}}], 'operands': ['e30', 'e31'], 'attrs': {}, 'regions': []}], 'returns': ['e32']}, {'arguments': [{'name': 'e33', 'type': {'dtype': 'float16', 'shape': (16, 32)}}], 'operations': [{'op': 'constant', 'results': [{'name': 'e34', 'type': {'dtype': 'float16', 'shape': (16, 32)}}], 'operands': [], 'attrs': {'value': 0.125}, 'regions': []}, {'op': 'sub', 'results': [{'name': 'e35', 'type': {'dtype': 'float16', 'shape': (16, 32)}}], 'operands': ['e33', 'e34'], 'attrs': {}, 'regions': []}], 'returns': ['e35']}]}, {'op': 'call', 'results': [{'name': 'e37', 'type': {'dtype': 'float16', 'shape': (16, 32)}}, {'name': 'e38', 'type': {'dtype': 'float32', 'shape': ()}}], 'operands': ['e36', 'e16'], 'attrs': {'callee': 'mixed_helper'}, 'regions': []}, {'op': 'constant', 'results': [{'name': 'e39', 'type': {'dtype': 'float16', 'shape': (16, 32)}}], 'operands': [], 'attrs': {'value': 0.125}, 'regions': []}, {'op': 'mul', 'results': [{'name': 'e40', 'type': {'dtype': 'float16', 'shape': (16, 32)}}], 'operands': ['e37', 'e39'], 'attrs': {}, 'regions': []}, {'op': 'add', 'results': [{'name': 'e41', 'type': {'dtype': 'float32', 'shape': ()}}], 'operands': ['e28', 'e38'], 'attrs': {}, 'regions': []}, {'op': 'constant', 'results': [{'name': 'e42', 'type': {'dtype': 'int32', 'shape': ()}}], 'operands': [], 'attrs': {'value': 1}, 'regions': []}, {'op': 'add', 'results': [{'name': 'e43', 'type': {'dtype': 'int32', 'shape': ()}}], 'operands': ['e26', 'e42'], 'attrs': {}, 'regions': []}, {'op': 'add', 'results': [{'name': 'e44', 'type': {'dtype': 'int32', 'shape': ()}}], 'operands': ['e29', 'e43'], 'attrs': {}, 'regions': []}], 'returns': ['e40', 'e41', 'e44']}]}, {'op': 'while', 'results': [{'name': 'e67', 'type': {'dtype': 'float16', 'shape': (16, 32)}}, {'name': 'e68', 'type': {'dtype': 'float32', 'shape': ()}}, {'name': 'e69', 'type': {'dtype': 'int32', 'shape': ()}}], 'operands': ['e19', 'e45', 'e46', 'e47'], 'attrs': {'max_steps': 4, 'pipelined': False}, 'regions': [{'arguments': [{'name': 'e48', 'type': {'dtype': 'int32', 'shape': ()}}, {'name': 'e49', 'type': {'dtype': 'float16', 'shape': (16, 32)}}, {'name': 'e50', 'type': {'dtype': 'float32', 'shape': ()}}, {'name': 'e51', 'type': {'dtype': 'int32', 'shape': ()}}], 'operations': [{'op': 'if', 'results': [{'name': 'e58', 'type': {'dtype': 'float16', 'shape': (16, 32)}}], 'operands': ['e25', 'e49'], 'attrs': {}, 'regions': [{'arguments': [{'name': 'e52', 'type': {'dtype': 'float16', 'shape': (16, 32)}}], 'operations': [{'op': 'constant', 'results': [{'name': 'e53', 'type': {'dtype': 'float16', 'shape': (16, 32)}}], 'operands': [], 'attrs': {'value': 0.125}, 'regions': []}, {'op': 'add', 'results': [{'name': 'e54', 'type': {'dtype': 'float16', 'shape': (16, 32)}}], 'operands': ['e52', 'e53'], 'attrs': {}, 'regions': []}], 'returns': ['e54']}, {'arguments': [{'name': 'e55', 'type': {'dtype': 'float16', 'shape': (16, 32)}}], 'operations': [{'op': 'constant', 'results': [{'name': 'e56', 'type': {'dtype': 'float16', 'shape': (16, 32)}}], 'operands': [], 'attrs': {'value': 0.125}, 'regions': []}, {'op': 'sub', 'results': [{'name': 'e57', 'type': {'dtype': 'float16', 'shape': (16, 32)}}], 'operands': ['e55', 'e56'], 'attrs': {}, 'regions': []}], 'returns': ['e57']}]}, {'op': 'call', 'results': [{'name': 'e59', 'type': {'dtype': 'float16', 'shape': (16, 32)}}, {'name': 'e60', 'type': {'dtype': 'float32', 'shape': ()}}], 'operands': ['e58', 'e16'], 'attrs': {'callee': 'mixed_helper'}, 'regions': []}, {'op': 'constant', 'results': [{'name': 'e61', 'type': {'dtype': 'float16', 'shape': (16, 32)}}], 'operands': [], 'attrs': {'value': 0.125}, 'regions': []}, {'op': 'mul', 'results': [{'name': 'e62', 'type': {'dtype': 'float16', 'shape': (16, 32)}}], 'operands': ['e59', 'e61'], 'attrs': {}, 'regions': []}, {'op': 'add', 'results': [{'name': 'e63', 'type': {'dtype': 'float32', 'shape': ()}}], 'operands': ['e50', 'e60'], 'attrs': {}, 'regions': []}, {'op': 'constant', 'results': [{'name': 'e64', 'type': {'dtype': 'int32', 'shape': ()}}], 'operands': [], 'attrs': {'value': 1}, 'regions': []}, {'op': 'add', 'results': [{'name': 'e65', 'type': {'dtype': 'int32', 'shape': ()}}], 'operands': ['e48', 'e64'], 'attrs': {}, 'regions': []}, {'op': 'add', 'results': [{'name': 'e66', 'type': {'dtype': 'int32', 'shape': ()}}], 'operands': ['e51', 'e65'], 'attrs': {}, 'regions': []}], 'returns': ['e62', 'e63', 'e66']}]}, {'op': 'flip', 'results': [{'name': 'e70', 'type': {'dtype': 'float16', 'shape': (16, 32)}}], 'operands': ['e67'], 'attrs': {}, 'regions': []}, {'op': 'sub', 'results': [{'name': 'e71', 'type': {'dtype': 'float16', 'shape': (16, 32)}}], 'operands': ['e70', 'e3'], 'attrs': {}, 'regions': []}, {'op': 'sub', 'results': [{'name': 'e72', 'type': {'dtype': 'float16', 'shape': (16, 32)}}], 'operands': ['e71', 'e71'], 'attrs': {}, 'regions': []}, {'op': 'mul', 'results': [{'name': 'e73_l', 'type': {'dtype': 'float16', 'shape': (16, 32)}}], 'operands': ['e3', 'e71'], 'attrs': {}, 'regions': []}, {'op': 'mul', 'results': [{'name': 'e73_r', 'type': {'dtype': 'float16', 'shape': (16, 32)}}], 'operands': ['e3', 'e71'], 'attrs': {}, 'regions': []}, {'op': 'sub', 'results': [{'name': 'e73', 'type': {'dtype': 'float16', 'shape': (16, 32)}}], 'operands': ['e73_l', 'e73_r'], 'attrs': {}, 'regions': []}, {'op': 'reduce', 'results': [{'name': 'e74', 'type': {'dtype': 'float32', 'shape': (16,)}}], 'operands': ['e73'], 'attrs': {'axis': 1, 'kind': 'sum'}, 'regions': []}], 'returns': ['e73', 'e68', 'e69', 'e74']}, 'buffers': [{'name': 'mem0', 'dtype': 'float16', 'size': 512, 'role': 'input', 'base': None, 'offset': 0, 'stride': 1}, {'name': 'mem1', 'dtype': 'float32', 'size': 16, 'role': 'input', 'base': None, 'offset': 0, 'stride': 1}, {'name': 'mem2', 'dtype': 'int32', 'size': 1, 'role': 'input', 'base': None, 'offset': 0, 'stride': 1}], 'functions': [{'name': 'mixed_helper', 'body': {'arguments': [{'name': 'e6', 'type': {'dtype': 'float16', 'shape': (16, 32)}}, {'name': 'e7', 'type': {'dtype': 'float32', 'shape': (16,)}}], 'operations': [{'op': 'reduce', 'results': [{'name': 'e8', 'type': {'dtype': 'float32', 'shape': (16,)}}], 'operands': ['e6'], 'attrs': {'axis': 1, 'kind': 'sum'}, 'regions': []}, {'op': 'add', 'results': [{'name': 'e9', 'type': {'dtype': 'float32', 'shape': (16,)}}], 'operands': ['e8', 'e7'], 'attrs': {}, 'regions': []}, {'op': 'reduce', 'results': [{'name': 'e10', 'type': {'dtype': 'float32', 'shape': ()}}], 'operands': ['e9'], 'attrs': {'axis': 0, 'kind': 'sum'}, 'regions': []}, {'op': 'reshape', 'results': [{'name': 'e11', 'type': {'dtype': 'float32', 'shape': (16, 1)}}], 'operands': ['e9'], 'attrs': {}, 'regions': []}, {'op': 'cast', 'results': [{'name': 'e12', 'type': {'dtype': 'float16', 'shape': (16, 1)}}], 'operands': ['e11'], 'attrs': {}, 'regions': []}, {'op': 'add', 'results': [{'name': 'e13', 'type': {'dtype': 'float16', 'shape': (16, 32)}}], 'operands': ['e6', 'e12'], 'attrs': {}, 'regions': []}], 'returns': ['e13', 'e10']}}], 'observations': ['e3', 'e70', 'e19', 'e22'], 'blocks': 2, 'input_pattern': 'normal', 'runtime_cases': [(0, 1), (1, 511), (2, 512)], 'configuration_pair': True, 'observation_pair': True, 'pass_config_pair': False, 'fast_math_pair': False, 'precision_pair': True, 'identity_pair': True, 'family': 'control_calls'}}

if __name__ == '__main__':
    if os.environ.get('TILESMITH_COMPILE_ONLY', '0') == '1':
        prepare_extended(device_compile=False)
        extended_stage('complete', 'compile_only')
        print('COMPILE PASSED (no GPU execution)')
    else:
        run_extended(PROGRAM, prepare_extended, 0, 3, reference_programs=REFERENCE_PROGRAMS)
        extended_stage('complete', 'execute')
        print('ALL PASSED')
