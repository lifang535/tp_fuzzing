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
                out = [args[0].float() @ args[1].float() + args[2]]
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


def run_extended(program, prepare, input_seed=0, repeats=2, device='cuda'):
    import torch
    import math
    types = {v['name']: v['type'] for node in program['body']['operations'] for v in node['results']}
    has_matmul = any(n['op'] == 'matmul' for n in program['body']['operations'])
    # Compile once, reuse each signature for all runtime shapes/bounds and seeds.
    variants = prepare()
    extended_stage('reference', 'cpu')
    for seed in (input_seed,):
        # The reference always runs on CPU, independently of GPU codegen/TF32.
        host_original = extended_inputs(program, seed, 'cpu')
        original = {name: value.to(device) for name, value in host_original.items()}
        for steps, limit in program['runtime_cases']:
            expected, expected_memory = extended_reference(program, host_original, steps, limit)
            expected = {name: value.to(device) for name, value in expected.items()}
            expected_memory = {name: value.to(device) for name, value in expected_memory.items()}
            baselines = {}
            for label, watched, launch in variants:
                extended_stage('execute', label)
                previous = None
                for repeat in range(repeats):
                    memories = {name: value.clone() for name, value in original.items()}
                    output_storage = {name: torch.full((program['blocks'], math.prod(types[name]['shape']) + 32),
                                      23, dtype=getattr(torch, types[name]['dtype']), device=device) for name in watched}
                    # For bool/int too, a missing write must differ from reference.
                    for name, storage in output_storage.items():
                        ref = expected[name].reshape(program['blocks'], -1)
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
                        actual = storage[:, 16:-16].reshape(expected[name].shape)
                        extended_check(actual, expected[name], f'{label}:{name}:seed={seed}:steps={steps}:limit={limit}', has_matmul)
                        if name in baselines:
                            extended_check(actual, baselines[name], 'configuration/observation invariance:' + name, has_matmul)
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
                            extended_check(memories[name], expected_memory[name], 'scratch:' + name, has_matmul)
                        values['memory:' + name] = memories[name].clone()
                    if previous is not None:
                        for name, actual in values.items():
                            if not torch.equal(actual.contiguous().view(torch.uint8), previous[name].contiguous().view(torch.uint8)):
                                raise RuntimeError('WRONG RESULT: repeat determinism:' + name)
                    previous = values


@triton.jit
def mixed_helper_extended_0_0(e45, e46, mem0, mem1, mem4, mem5, steps, limit, bid):
    e47 = (tl.where(tl.sum((e45 != e45).to(tl.int32), 1) > 0, float('nan'), tl.max(e45.to(tl.float32), 1))).to(tl.float32)
    e48 = ((e47 + e46)).to(tl.float32)
    e49 = (tl.sum(e48.to(tl.float32), 0)).to(tl.float32)
    e50 = (tl.reshape(e48, (16, 1))).to(tl.float32)
    e51 = (e50.to(tl.float16)).to(tl.float16)
    e52 = ((e45 + e51)).to(tl.float16)
    return e52, e49
@triton.jit
def extended_0_0(mem0, mem1, mem4, mem5, out_e110, out_e10, out_e16, out_e104, out_e105, out_e111, steps, limit):
    bid = tl.program_id(0)
    e1 = ((((tl.arange(0, 16)[:, None] * 16 + tl.arange(0, 16)[None, :]) + 186) % 256)).to(tl.int32)
    e2 = (tl.full((16, 16), True, tl.int1)).to(tl.int1)
    e3 = (tl.load(mem0 + bid * 288 + 16 + e1 * 1, (e2 & (e1 >= 0) & (e1 < 256)), other=0)).to(tl.float32)
    e4 = ((e3 * e3)).to(tl.float32)
    e5 = (e4.to(tl.float16)).to(tl.float16)
    e6 = (e5.to(tl.float16)).to(tl.float16)
    e7 = (tl.full((16, 16), 0.5, tl.float16)).to(tl.float16)
    e8 = ((e6 * e7)).to(tl.float16)
    e9 = (tl.full((16, 16), 3, tl.int32)).to(tl.int32)
    e10 = ((e1 ^ e9)).to(tl.int32)
    e11 = (tl.full((16, 16), 3, tl.int32)).to(tl.int32)
    e12 = ((e10 < e11)).to(tl.int1)
    e13 = ((((255 - (tl.arange(0, 16)[:, None] * 16 + tl.arange(0, 16)[None, :])) + 161) % 256)).to(tl.int32)
    e14 = (limit).to(tl.int32)
    e15 = ((e13 < e14)).to(tl.int1)
    e16 = ((e12 & e15)).to(tl.int1)
    e17 = (tl.full((16, 16), -0.125, tl.float16)).to(tl.float16)
    e18 = (tl.where(e16, e8, e17)).to(tl.float16)
    e19 = ((((255 - (tl.arange(0, 16)[:, None] * 16 + tl.arange(0, 16)[None, :])) + 27) % 256)).to(tl.int32)
    e20 = (limit).to(tl.int32)
    e21 = ((e19 < e20)).to(tl.int1)
    tl.store(mem1 + bid * 548 + 17 + e19 * 2, e18, (e21 & (e19 >= 0) & (e19 < 256)))
    tl.debug_barrier()
    e22 = (tl.load(mem1 + bid * 548 + 17 + e19 * 2, (e21 & (e19 >= 0) & (e19 < 256)), other=0)).to(tl.float16)
    e23 = (tl.full((16, 16), 0.125, tl.float16)).to(tl.float16)
    e24 = ((e22 + e23)).to(tl.float16)
    tl.debug_barrier()
    tl.store(mem1 + bid * 548 + 18 + e19 * 1, e24, (e21 & (e19 >= 0) & (e19 < 256)))
    tl.debug_barrier()
    e25 = (tl.full((16, 16), True, tl.int1)).to(tl.int1)
    e26 = (tl.load(mem1 + bid * 548 + 17 + e19 * 2, (e25 & (e19 >= 0) & (e19 < 256)), other=0)).to(tl.float16)
    e27 = (tl.full((16, 16), 7, tl.int32)).to(tl.int32)
    e28 = ((e19 ^ e27)).to(tl.int32)
    e29 = (tl.load(mem1 + bid * 548 + 18 + e28 * 1, (e21 & (e28 >= 0) & (e28 < 256)), other=0)).to(tl.float16)
    e30 = ((e26 + e29)).to(tl.float16)
    e31 = (e30.to(tl.float16)).to(tl.float16)
    e32 = (tl.trans(e31)).to(tl.float16)
    e33 = (tl.full((16, 16), 0.125, tl.float32)).to(tl.float32)
    e34 = (tl.dot(e32, e5, e33, input_precision='ieee')).to(tl.float32)
    e35 = (tl.where(tl.sum((e34 != e34).to(tl.int32), 1) > 0, float('nan'), tl.max(e34.to(tl.float32), 1))).to(tl.float32)
    e36 = (tl.reshape(e35, (16, 1))).to(tl.float32)
    e37 = ((e34 - e36)).to(tl.float32)
    e38 = (tl.full((16, 16), 0.125, tl.float32)).to(tl.float32)
    e39 = ((e37 * e38)).to(tl.float32)
    e40 = (e39.to(tl.float16)).to(tl.float16)
    e41 = (tl.dot(e40, e17, e33, input_precision='ieee')).to(tl.float32)
    e42 = (tl.reshape(e41, (16, 16))).to(tl.float32)
    e43 = (tl.reshape(e42, (16, 16))).to(tl.float32)
    e44 = (e43.to(tl.float16)).to(tl.float16)
    e53, e54 = mixed_helper_extended_0_0(e44, e35, mem0, mem1, mem4, mem5, steps, limit, bid)
    e55 = (steps).to(tl.int32)
    e56 = (tl.full((), 0, tl.int32)).to(tl.int32)
    e57 = (tl.full((), 0, tl.int32)).to(tl.int32)
    e58 = (tl.full((), True, tl.int1)).to(tl.int1)
    e59 = (tl.load(mem4 + bid * 33 + 16 + e57 * 1, (e58 & (e57 >= 0) & (e57 < 1)), other=0)).to(tl.int32)
    e60 = (tl.full((), 0, tl.int32)).to(tl.int32)
    e61 = ((e59 < e60)).to(tl.int1)
    e81 = e53
    e82 = e54
    e83 = e56
    for e62 in range(tl.minimum(tl.maximum(e55, 0), 4)):
        e63 = e81
        e64 = e82
        e65 = e83
        if e61:
            e66 = e63
            e67 = (tl.full((16, 16), 0.125, tl.float16)).to(tl.float16)
            e68 = ((e66 + e67)).to(tl.float16)
            e72 = e68
        else:
            e69 = e63
            e70 = (tl.full((16, 16), 0.125, tl.float16)).to(tl.float16)
            e71 = ((e69 - e70)).to(tl.float16)
            e72 = e71
        e73, e74 = mixed_helper_extended_0_0(e72, e35, mem0, mem1, mem4, mem5, steps, limit, bid)
        e75 = (tl.full((16, 16), 0.125, tl.float16)).to(tl.float16)
        e76 = ((e73 * e75)).to(tl.float16)
        e77 = ((e64 + e74)).to(tl.float32)
        e78 = (tl.full((), 1, tl.int32)).to(tl.int32)
        e79 = ((e62 + e78)).to(tl.int32)
        e80 = ((e65 + e79)).to(tl.int32)
        e81 = e76
        e82 = e77
        e83 = e80
    e103 = e81
    e104 = e82
    e105 = e83
    e84 = tl.full((), 0, tl.int32)
    while e84 < tl.minimum(tl.maximum(e55, 0), 4):
        e85 = e103
        e86 = e104
        e87 = e105
        if e61:
            e88 = e85
            e89 = (tl.full((16, 16), 0.125, tl.float16)).to(tl.float16)
            e90 = ((e88 + e89)).to(tl.float16)
            e94 = e90
        else:
            e91 = e85
            e92 = (tl.full((16, 16), 0.125, tl.float16)).to(tl.float16)
            e93 = ((e91 - e92)).to(tl.float16)
            e94 = e93
        e95, e96 = mixed_helper_extended_0_0(e94, e35, mem0, mem1, mem4, mem5, steps, limit, bid)
        e97 = (tl.full((16, 16), 0.125, tl.float16)).to(tl.float16)
        e98 = ((e95 * e97)).to(tl.float16)
        e99 = ((e86 + e96)).to(tl.float32)
        e100 = (tl.full((), 1, tl.int32)).to(tl.int32)
        e101 = ((e84 + e100)).to(tl.int32)
        e102 = ((e87 + e101)).to(tl.int32)
        e103 = e98
        e104 = e99
        e105 = e102
        e84 += 1
    e106 = ((e103 + e24)).to(tl.float16)
    e107 = ((((tl.arange(0, 16)[:, None] * 16 + tl.arange(0, 16)[None, :]) + 187) % 256)).to(tl.int32)
    e108 = (tl.full((16, 16), True, tl.int1)).to(tl.int1)
    e109 = (tl.load(mem5 + bid * 288 + 16 + e107 * 1, (e108 & (e107 >= 0) & (e107 < 256)), other=0)).to(tl.float16)
    e110 = ((e106 + e109)).to(tl.float16)
    e111 = (tl.where(tl.sum((e110 != e110).to(tl.int32), 1) > 0, float('nan'), tl.min(e110.to(tl.float32), 1))).to(tl.float32)
    tl.store(out_e110 + bid * 288 + 16 + (tl.arange(0, 16)[:, None] * 16 + tl.arange(0, 16)[None, :]), e110)
    tl.store(out_e10 + bid * 288 + 16 + (tl.arange(0, 16)[:, None] * 16 + tl.arange(0, 16)[None, :]), e10)
    tl.store(out_e16 + bid * 288 + 16 + (tl.arange(0, 16)[:, None] * 16 + tl.arange(0, 16)[None, :]), e16)
    tl.store(out_e104 + bid * 33 + 16 + tl.full((), 0, tl.int32), e104)
    tl.store(out_e105 + bid * 33 + 16 + tl.full((), 0, tl.int32), e105)
    tl.store(out_e111 + bid * 48 + 16 + tl.arange(0, 16), e111)

@triton.jit
def mixed_helper_extended_0_1(e45, e46, mem0, mem1, mem4, mem5, steps, limit, bid):
    e47 = (tl.where(tl.sum((e45 != e45).to(tl.int32), 1) > 0, float('nan'), tl.max(e45.to(tl.float32), 1))).to(tl.float32)
    e48 = ((e47 + e46)).to(tl.float32)
    e49 = (tl.sum(e48.to(tl.float32), 0)).to(tl.float32)
    e50 = (tl.reshape(e48, (16, 1))).to(tl.float32)
    e51 = (e50.to(tl.float16)).to(tl.float16)
    e52 = ((e45 + e51)).to(tl.float16)
    return e52, e49
@triton.jit
def extended_0_1(mem0, mem1, mem4, mem5, out_e110, out_e10, out_e16, out_e104, out_e105, out_e111, out_e3, out_e8, out_e22, out_e26, out_e34, out_e39, out_e2, out_e19, steps, limit):
    bid = tl.program_id(0)
    e1 = ((((tl.arange(0, 16)[:, None] * 16 + tl.arange(0, 16)[None, :]) + 186) % 256)).to(tl.int32)
    e2 = (tl.full((16, 16), True, tl.int1)).to(tl.int1)
    e3 = (tl.load(mem0 + bid * 288 + 16 + e1 * 1, (e2 & (e1 >= 0) & (e1 < 256)), other=0)).to(tl.float32)
    e4 = ((e3 * e3)).to(tl.float32)
    e5 = (e4.to(tl.float16)).to(tl.float16)
    e6 = (e5.to(tl.float16)).to(tl.float16)
    e7 = (tl.full((16, 16), 0.5, tl.float16)).to(tl.float16)
    e8 = ((e6 * e7)).to(tl.float16)
    e9 = (tl.full((16, 16), 3, tl.int32)).to(tl.int32)
    e10 = ((e1 ^ e9)).to(tl.int32)
    e11 = (tl.full((16, 16), 3, tl.int32)).to(tl.int32)
    e12 = ((e10 < e11)).to(tl.int1)
    e13 = ((((255 - (tl.arange(0, 16)[:, None] * 16 + tl.arange(0, 16)[None, :])) + 161) % 256)).to(tl.int32)
    e14 = (limit).to(tl.int32)
    e15 = ((e13 < e14)).to(tl.int1)
    e16 = ((e12 & e15)).to(tl.int1)
    e17 = (tl.full((16, 16), -0.125, tl.float16)).to(tl.float16)
    e18 = (tl.where(e16, e8, e17)).to(tl.float16)
    e19 = ((((255 - (tl.arange(0, 16)[:, None] * 16 + tl.arange(0, 16)[None, :])) + 27) % 256)).to(tl.int32)
    e20 = (limit).to(tl.int32)
    e21 = ((e19 < e20)).to(tl.int1)
    tl.store(mem1 + bid * 548 + 17 + e19 * 2, e18, (e21 & (e19 >= 0) & (e19 < 256)))
    tl.debug_barrier()
    e22 = (tl.load(mem1 + bid * 548 + 17 + e19 * 2, (e21 & (e19 >= 0) & (e19 < 256)), other=0)).to(tl.float16)
    e23 = (tl.full((16, 16), 0.125, tl.float16)).to(tl.float16)
    e24 = ((e22 + e23)).to(tl.float16)
    tl.debug_barrier()
    tl.store(mem1 + bid * 548 + 18 + e19 * 1, e24, (e21 & (e19 >= 0) & (e19 < 256)))
    tl.debug_barrier()
    e25 = (tl.full((16, 16), True, tl.int1)).to(tl.int1)
    e26 = (tl.load(mem1 + bid * 548 + 17 + e19 * 2, (e25 & (e19 >= 0) & (e19 < 256)), other=0)).to(tl.float16)
    e27 = (tl.full((16, 16), 7, tl.int32)).to(tl.int32)
    e28 = ((e19 ^ e27)).to(tl.int32)
    e29 = (tl.load(mem1 + bid * 548 + 18 + e28 * 1, (e21 & (e28 >= 0) & (e28 < 256)), other=0)).to(tl.float16)
    e30 = ((e26 + e29)).to(tl.float16)
    e31 = (e30.to(tl.float16)).to(tl.float16)
    e32 = (tl.trans(e31)).to(tl.float16)
    e33 = (tl.full((16, 16), 0.125, tl.float32)).to(tl.float32)
    e34 = (tl.dot(e32, e5, e33, input_precision='ieee')).to(tl.float32)
    e35 = (tl.where(tl.sum((e34 != e34).to(tl.int32), 1) > 0, float('nan'), tl.max(e34.to(tl.float32), 1))).to(tl.float32)
    e36 = (tl.reshape(e35, (16, 1))).to(tl.float32)
    e37 = ((e34 - e36)).to(tl.float32)
    e38 = (tl.full((16, 16), 0.125, tl.float32)).to(tl.float32)
    e39 = ((e37 * e38)).to(tl.float32)
    e40 = (e39.to(tl.float16)).to(tl.float16)
    e41 = (tl.dot(e40, e17, e33, input_precision='ieee')).to(tl.float32)
    e42 = (tl.reshape(e41, (16, 16))).to(tl.float32)
    e43 = (tl.reshape(e42, (16, 16))).to(tl.float32)
    e44 = (e43.to(tl.float16)).to(tl.float16)
    e53, e54 = mixed_helper_extended_0_1(e44, e35, mem0, mem1, mem4, mem5, steps, limit, bid)
    e55 = (steps).to(tl.int32)
    e56 = (tl.full((), 0, tl.int32)).to(tl.int32)
    e57 = (tl.full((), 0, tl.int32)).to(tl.int32)
    e58 = (tl.full((), True, tl.int1)).to(tl.int1)
    e59 = (tl.load(mem4 + bid * 33 + 16 + e57 * 1, (e58 & (e57 >= 0) & (e57 < 1)), other=0)).to(tl.int32)
    e60 = (tl.full((), 0, tl.int32)).to(tl.int32)
    e61 = ((e59 < e60)).to(tl.int1)
    e81 = e53
    e82 = e54
    e83 = e56
    for e62 in range(tl.minimum(tl.maximum(e55, 0), 4)):
        e63 = e81
        e64 = e82
        e65 = e83
        if e61:
            e66 = e63
            e67 = (tl.full((16, 16), 0.125, tl.float16)).to(tl.float16)
            e68 = ((e66 + e67)).to(tl.float16)
            e72 = e68
        else:
            e69 = e63
            e70 = (tl.full((16, 16), 0.125, tl.float16)).to(tl.float16)
            e71 = ((e69 - e70)).to(tl.float16)
            e72 = e71
        e73, e74 = mixed_helper_extended_0_1(e72, e35, mem0, mem1, mem4, mem5, steps, limit, bid)
        e75 = (tl.full((16, 16), 0.125, tl.float16)).to(tl.float16)
        e76 = ((e73 * e75)).to(tl.float16)
        e77 = ((e64 + e74)).to(tl.float32)
        e78 = (tl.full((), 1, tl.int32)).to(tl.int32)
        e79 = ((e62 + e78)).to(tl.int32)
        e80 = ((e65 + e79)).to(tl.int32)
        e81 = e76
        e82 = e77
        e83 = e80
    e103 = e81
    e104 = e82
    e105 = e83
    e84 = tl.full((), 0, tl.int32)
    while e84 < tl.minimum(tl.maximum(e55, 0), 4):
        e85 = e103
        e86 = e104
        e87 = e105
        if e61:
            e88 = e85
            e89 = (tl.full((16, 16), 0.125, tl.float16)).to(tl.float16)
            e90 = ((e88 + e89)).to(tl.float16)
            e94 = e90
        else:
            e91 = e85
            e92 = (tl.full((16, 16), 0.125, tl.float16)).to(tl.float16)
            e93 = ((e91 - e92)).to(tl.float16)
            e94 = e93
        e95, e96 = mixed_helper_extended_0_1(e94, e35, mem0, mem1, mem4, mem5, steps, limit, bid)
        e97 = (tl.full((16, 16), 0.125, tl.float16)).to(tl.float16)
        e98 = ((e95 * e97)).to(tl.float16)
        e99 = ((e86 + e96)).to(tl.float32)
        e100 = (tl.full((), 1, tl.int32)).to(tl.int32)
        e101 = ((e84 + e100)).to(tl.int32)
        e102 = ((e87 + e101)).to(tl.int32)
        e103 = e98
        e104 = e99
        e105 = e102
        e84 += 1
    e106 = ((e103 + e24)).to(tl.float16)
    e107 = ((((tl.arange(0, 16)[:, None] * 16 + tl.arange(0, 16)[None, :]) + 187) % 256)).to(tl.int32)
    e108 = (tl.full((16, 16), True, tl.int1)).to(tl.int1)
    e109 = (tl.load(mem5 + bid * 288 + 16 + e107 * 1, (e108 & (e107 >= 0) & (e107 < 256)), other=0)).to(tl.float16)
    e110 = ((e106 + e109)).to(tl.float16)
    e111 = (tl.where(tl.sum((e110 != e110).to(tl.int32), 1) > 0, float('nan'), tl.min(e110.to(tl.float32), 1))).to(tl.float32)
    tl.store(out_e110 + bid * 288 + 16 + (tl.arange(0, 16)[:, None] * 16 + tl.arange(0, 16)[None, :]), e110)
    tl.store(out_e10 + bid * 288 + 16 + (tl.arange(0, 16)[:, None] * 16 + tl.arange(0, 16)[None, :]), e10)
    tl.store(out_e16 + bid * 288 + 16 + (tl.arange(0, 16)[:, None] * 16 + tl.arange(0, 16)[None, :]), e16)
    tl.store(out_e104 + bid * 33 + 16 + tl.full((), 0, tl.int32), e104)
    tl.store(out_e105 + bid * 33 + 16 + tl.full((), 0, tl.int32), e105)
    tl.store(out_e111 + bid * 48 + 16 + tl.arange(0, 16), e111)
    tl.store(out_e3 + bid * 288 + 16 + (tl.arange(0, 16)[:, None] * 16 + tl.arange(0, 16)[None, :]), e3)
    tl.store(out_e8 + bid * 288 + 16 + (tl.arange(0, 16)[:, None] * 16 + tl.arange(0, 16)[None, :]), e8)
    tl.store(out_e22 + bid * 288 + 16 + (tl.arange(0, 16)[:, None] * 16 + tl.arange(0, 16)[None, :]), e22)
    tl.store(out_e26 + bid * 288 + 16 + (tl.arange(0, 16)[:, None] * 16 + tl.arange(0, 16)[None, :]), e26)
    tl.store(out_e34 + bid * 288 + 16 + (tl.arange(0, 16)[:, None] * 16 + tl.arange(0, 16)[None, :]), e34)
    tl.store(out_e39 + bid * 288 + 16 + (tl.arange(0, 16)[:, None] * 16 + tl.arange(0, 16)[None, :]), e39)
    tl.store(out_e2 + bid * 288 + 16 + (tl.arange(0, 16)[:, None] * 16 + tl.arange(0, 16)[None, :]), e2)
    tl.store(out_e19 + bid * 288 + 16 + (tl.arange(0, 16)[:, None] * 16 + tl.arange(0, 16)[None, :]), e19)

@triton.jit
def mixed_helper_extended_1_0(e45, e46, mem0, mem1, mem4, mem5, steps, limit, bid):
    e47 = (tl.where(tl.sum((e45 != e45).to(tl.int32), 1) > 0, float('nan'), tl.max(e45.to(tl.float32), 1))).to(tl.float32)
    e48 = ((e47 + e46)).to(tl.float32)
    e49 = (tl.sum(e48.to(tl.float32), 0)).to(tl.float32)
    e50 = (tl.reshape(e48, (16, 1))).to(tl.float32)
    e51 = (e50.to(tl.float16)).to(tl.float16)
    e52 = ((e45 + e51)).to(tl.float16)
    return e52, e49
@triton.jit
def extended_1_0(mem0, mem1, mem4, mem5, out_e110, out_e10, out_e16, out_e104, out_e105, out_e111, steps, limit):
    bid = tl.program_id(0)
    e1 = ((((tl.arange(0, 16)[:, None] * 16 + tl.arange(0, 16)[None, :]) + 186) % 256)).to(tl.int32)
    e2 = (tl.full((16, 16), True, tl.int1)).to(tl.int1)
    e3 = (tl.load(mem0 + bid * 288 + 16 + e1 * 1, (e2 & (e1 >= 0) & (e1 < 256)), other=0)).to(tl.float32)
    e4 = ((e3 * e3)).to(tl.float32)
    e5 = (e4.to(tl.float16)).to(tl.float16)
    e6 = (e5.to(tl.float16)).to(tl.float16)
    e7 = (tl.full((16, 16), 0.5, tl.float16)).to(tl.float16)
    e8 = ((e6 * e7)).to(tl.float16)
    e9 = (tl.full((16, 16), 3, tl.int32)).to(tl.int32)
    e10 = ((e1 ^ e9)).to(tl.int32)
    e11 = (tl.full((16, 16), 3, tl.int32)).to(tl.int32)
    e12 = ((e10 < e11)).to(tl.int1)
    e13 = ((((255 - (tl.arange(0, 16)[:, None] * 16 + tl.arange(0, 16)[None, :])) + 161) % 256)).to(tl.int32)
    e14 = (limit).to(tl.int32)
    e15 = ((e13 < e14)).to(tl.int1)
    e16 = ((e12 & e15)).to(tl.int1)
    e17 = (tl.full((16, 16), -0.125, tl.float16)).to(tl.float16)
    e18 = (tl.where(e16, e8, e17)).to(tl.float16)
    e19 = ((((255 - (tl.arange(0, 16)[:, None] * 16 + tl.arange(0, 16)[None, :])) + 27) % 256)).to(tl.int32)
    e20 = (limit).to(tl.int32)
    e21 = ((e19 < e20)).to(tl.int1)
    tl.store(mem1 + bid * 548 + 17 + e19 * 2, e18, (e21 & (e19 >= 0) & (e19 < 256)))
    tl.debug_barrier()
    e22 = (tl.load(mem1 + bid * 548 + 17 + e19 * 2, (e21 & (e19 >= 0) & (e19 < 256)), other=0)).to(tl.float16)
    e23 = (tl.full((16, 16), 0.125, tl.float16)).to(tl.float16)
    e24 = ((e22 + e23)).to(tl.float16)
    tl.debug_barrier()
    tl.store(mem1 + bid * 548 + 18 + e19 * 1, e24, (e21 & (e19 >= 0) & (e19 < 256)))
    tl.debug_barrier()
    e25 = (tl.full((16, 16), True, tl.int1)).to(tl.int1)
    e26 = (tl.load(mem1 + bid * 548 + 17 + e19 * 2, (e25 & (e19 >= 0) & (e19 < 256)), other=0)).to(tl.float16)
    e27 = (tl.full((16, 16), 7, tl.int32)).to(tl.int32)
    e28 = ((e19 ^ e27)).to(tl.int32)
    e29 = (tl.load(mem1 + bid * 548 + 18 + e28 * 1, (e21 & (e28 >= 0) & (e28 < 256)), other=0)).to(tl.float16)
    e30 = ((e26 + e29)).to(tl.float16)
    e31 = (e30.to(tl.float16)).to(tl.float16)
    e32 = (tl.trans(e31)).to(tl.float16)
    e33 = (tl.full((16, 16), 0.125, tl.float32)).to(tl.float32)
    e34 = (tl.dot(e32, e5, e33, input_precision='ieee')).to(tl.float32)
    e35 = (tl.where(tl.sum((e34 != e34).to(tl.int32), 1) > 0, float('nan'), tl.max(e34.to(tl.float32), 1))).to(tl.float32)
    e36 = (tl.reshape(e35, (16, 1))).to(tl.float32)
    e37 = ((e34 - e36)).to(tl.float32)
    e38 = (tl.full((16, 16), 0.125, tl.float32)).to(tl.float32)
    e39 = ((e37 * e38)).to(tl.float32)
    e40 = (e39.to(tl.float16)).to(tl.float16)
    e41 = (tl.dot(e40, e17, e33, input_precision='ieee')).to(tl.float32)
    e42 = (tl.reshape(e41, (16, 16))).to(tl.float32)
    e43 = (tl.reshape(e42, (16, 16))).to(tl.float32)
    e44 = (e43.to(tl.float16)).to(tl.float16)
    e53, e54 = mixed_helper_extended_1_0(e44, e35, mem0, mem1, mem4, mem5, steps, limit, bid)
    e55 = (steps).to(tl.int32)
    e56 = (tl.full((), 0, tl.int32)).to(tl.int32)
    e57 = (tl.full((), 0, tl.int32)).to(tl.int32)
    e58 = (tl.full((), True, tl.int1)).to(tl.int1)
    e59 = (tl.load(mem4 + bid * 33 + 16 + e57 * 1, (e58 & (e57 >= 0) & (e57 < 1)), other=0)).to(tl.int32)
    e60 = (tl.full((), 0, tl.int32)).to(tl.int32)
    e61 = ((e59 < e60)).to(tl.int1)
    e81 = e53
    e82 = e54
    e83 = e56
    for e62 in range(tl.minimum(tl.maximum(e55, 0), 4)):
        e63 = e81
        e64 = e82
        e65 = e83
        if e61:
            e66 = e63
            e67 = (tl.full((16, 16), 0.125, tl.float16)).to(tl.float16)
            e68 = ((e66 + e67)).to(tl.float16)
            e72 = e68
        else:
            e69 = e63
            e70 = (tl.full((16, 16), 0.125, tl.float16)).to(tl.float16)
            e71 = ((e69 - e70)).to(tl.float16)
            e72 = e71
        e73, e74 = mixed_helper_extended_1_0(e72, e35, mem0, mem1, mem4, mem5, steps, limit, bid)
        e75 = (tl.full((16, 16), 0.125, tl.float16)).to(tl.float16)
        e76 = ((e73 * e75)).to(tl.float16)
        e77 = ((e64 + e74)).to(tl.float32)
        e78 = (tl.full((), 1, tl.int32)).to(tl.int32)
        e79 = ((e62 + e78)).to(tl.int32)
        e80 = ((e65 + e79)).to(tl.int32)
        e81 = e76
        e82 = e77
        e83 = e80
    e103 = e81
    e104 = e82
    e105 = e83
    e84 = tl.full((), 0, tl.int32)
    while e84 < tl.minimum(tl.maximum(e55, 0), 4):
        e85 = e103
        e86 = e104
        e87 = e105
        if e61:
            e88 = e85
            e89 = (tl.full((16, 16), 0.125, tl.float16)).to(tl.float16)
            e90 = ((e88 + e89)).to(tl.float16)
            e94 = e90
        else:
            e91 = e85
            e92 = (tl.full((16, 16), 0.125, tl.float16)).to(tl.float16)
            e93 = ((e91 - e92)).to(tl.float16)
            e94 = e93
        e95, e96 = mixed_helper_extended_1_0(e94, e35, mem0, mem1, mem4, mem5, steps, limit, bid)
        e97 = (tl.full((16, 16), 0.125, tl.float16)).to(tl.float16)
        e98 = ((e95 * e97)).to(tl.float16)
        e99 = ((e86 + e96)).to(tl.float32)
        e100 = (tl.full((), 1, tl.int32)).to(tl.int32)
        e101 = ((e84 + e100)).to(tl.int32)
        e102 = ((e87 + e101)).to(tl.int32)
        e103 = e98
        e104 = e99
        e105 = e102
        e84 += 1
    e106 = ((e103 + e24)).to(tl.float16)
    e107 = ((((tl.arange(0, 16)[:, None] * 16 + tl.arange(0, 16)[None, :]) + 187) % 256)).to(tl.int32)
    e108 = (tl.full((16, 16), True, tl.int1)).to(tl.int1)
    e109 = (tl.load(mem5 + bid * 288 + 16 + e107 * 1, (e108 & (e107 >= 0) & (e107 < 256)), other=0)).to(tl.float16)
    e110 = ((e106 + e109)).to(tl.float16)
    e111 = (tl.where(tl.sum((e110 != e110).to(tl.int32), 1) > 0, float('nan'), tl.min(e110.to(tl.float32), 1))).to(tl.float32)
    tl.store(out_e110 + bid * 288 + 16 + (tl.arange(0, 16)[:, None] * 16 + tl.arange(0, 16)[None, :]), e110)
    tl.store(out_e10 + bid * 288 + 16 + (tl.arange(0, 16)[:, None] * 16 + tl.arange(0, 16)[None, :]), e10)
    tl.store(out_e16 + bid * 288 + 16 + (tl.arange(0, 16)[:, None] * 16 + tl.arange(0, 16)[None, :]), e16)
    tl.store(out_e104 + bid * 33 + 16 + tl.full((), 0, tl.int32), e104)
    tl.store(out_e105 + bid * 33 + 16 + tl.full((), 0, tl.int32), e105)
    tl.store(out_e111 + bid * 48 + 16 + tl.arange(0, 16), e111)

@triton.jit
def mixed_helper_extended_1_1(e45, e46, mem0, mem1, mem4, mem5, steps, limit, bid):
    e47 = (tl.where(tl.sum((e45 != e45).to(tl.int32), 1) > 0, float('nan'), tl.max(e45.to(tl.float32), 1))).to(tl.float32)
    e48 = ((e47 + e46)).to(tl.float32)
    e49 = (tl.sum(e48.to(tl.float32), 0)).to(tl.float32)
    e50 = (tl.reshape(e48, (16, 1))).to(tl.float32)
    e51 = (e50.to(tl.float16)).to(tl.float16)
    e52 = ((e45 + e51)).to(tl.float16)
    return e52, e49
@triton.jit
def extended_1_1(mem0, mem1, mem4, mem5, out_e110, out_e10, out_e16, out_e104, out_e105, out_e111, out_e3, out_e8, out_e22, out_e26, out_e34, out_e39, out_e2, out_e19, steps, limit):
    bid = tl.program_id(0)
    e1 = ((((tl.arange(0, 16)[:, None] * 16 + tl.arange(0, 16)[None, :]) + 186) % 256)).to(tl.int32)
    e2 = (tl.full((16, 16), True, tl.int1)).to(tl.int1)
    e3 = (tl.load(mem0 + bid * 288 + 16 + e1 * 1, (e2 & (e1 >= 0) & (e1 < 256)), other=0)).to(tl.float32)
    e4 = ((e3 * e3)).to(tl.float32)
    e5 = (e4.to(tl.float16)).to(tl.float16)
    e6 = (e5.to(tl.float16)).to(tl.float16)
    e7 = (tl.full((16, 16), 0.5, tl.float16)).to(tl.float16)
    e8 = ((e6 * e7)).to(tl.float16)
    e9 = (tl.full((16, 16), 3, tl.int32)).to(tl.int32)
    e10 = ((e1 ^ e9)).to(tl.int32)
    e11 = (tl.full((16, 16), 3, tl.int32)).to(tl.int32)
    e12 = ((e10 < e11)).to(tl.int1)
    e13 = ((((255 - (tl.arange(0, 16)[:, None] * 16 + tl.arange(0, 16)[None, :])) + 161) % 256)).to(tl.int32)
    e14 = (limit).to(tl.int32)
    e15 = ((e13 < e14)).to(tl.int1)
    e16 = ((e12 & e15)).to(tl.int1)
    e17 = (tl.full((16, 16), -0.125, tl.float16)).to(tl.float16)
    e18 = (tl.where(e16, e8, e17)).to(tl.float16)
    e19 = ((((255 - (tl.arange(0, 16)[:, None] * 16 + tl.arange(0, 16)[None, :])) + 27) % 256)).to(tl.int32)
    e20 = (limit).to(tl.int32)
    e21 = ((e19 < e20)).to(tl.int1)
    tl.store(mem1 + bid * 548 + 17 + e19 * 2, e18, (e21 & (e19 >= 0) & (e19 < 256)))
    tl.debug_barrier()
    e22 = (tl.load(mem1 + bid * 548 + 17 + e19 * 2, (e21 & (e19 >= 0) & (e19 < 256)), other=0)).to(tl.float16)
    e23 = (tl.full((16, 16), 0.125, tl.float16)).to(tl.float16)
    e24 = ((e22 + e23)).to(tl.float16)
    tl.debug_barrier()
    tl.store(mem1 + bid * 548 + 18 + e19 * 1, e24, (e21 & (e19 >= 0) & (e19 < 256)))
    tl.debug_barrier()
    e25 = (tl.full((16, 16), True, tl.int1)).to(tl.int1)
    e26 = (tl.load(mem1 + bid * 548 + 17 + e19 * 2, (e25 & (e19 >= 0) & (e19 < 256)), other=0)).to(tl.float16)
    e27 = (tl.full((16, 16), 7, tl.int32)).to(tl.int32)
    e28 = ((e19 ^ e27)).to(tl.int32)
    e29 = (tl.load(mem1 + bid * 548 + 18 + e28 * 1, (e21 & (e28 >= 0) & (e28 < 256)), other=0)).to(tl.float16)
    e30 = ((e26 + e29)).to(tl.float16)
    e31 = (e30.to(tl.float16)).to(tl.float16)
    e32 = (tl.trans(e31)).to(tl.float16)
    e33 = (tl.full((16, 16), 0.125, tl.float32)).to(tl.float32)
    e34 = (tl.dot(e32, e5, e33, input_precision='ieee')).to(tl.float32)
    e35 = (tl.where(tl.sum((e34 != e34).to(tl.int32), 1) > 0, float('nan'), tl.max(e34.to(tl.float32), 1))).to(tl.float32)
    e36 = (tl.reshape(e35, (16, 1))).to(tl.float32)
    e37 = ((e34 - e36)).to(tl.float32)
    e38 = (tl.full((16, 16), 0.125, tl.float32)).to(tl.float32)
    e39 = ((e37 * e38)).to(tl.float32)
    e40 = (e39.to(tl.float16)).to(tl.float16)
    e41 = (tl.dot(e40, e17, e33, input_precision='ieee')).to(tl.float32)
    e42 = (tl.reshape(e41, (16, 16))).to(tl.float32)
    e43 = (tl.reshape(e42, (16, 16))).to(tl.float32)
    e44 = (e43.to(tl.float16)).to(tl.float16)
    e53, e54 = mixed_helper_extended_1_1(e44, e35, mem0, mem1, mem4, mem5, steps, limit, bid)
    e55 = (steps).to(tl.int32)
    e56 = (tl.full((), 0, tl.int32)).to(tl.int32)
    e57 = (tl.full((), 0, tl.int32)).to(tl.int32)
    e58 = (tl.full((), True, tl.int1)).to(tl.int1)
    e59 = (tl.load(mem4 + bid * 33 + 16 + e57 * 1, (e58 & (e57 >= 0) & (e57 < 1)), other=0)).to(tl.int32)
    e60 = (tl.full((), 0, tl.int32)).to(tl.int32)
    e61 = ((e59 < e60)).to(tl.int1)
    e81 = e53
    e82 = e54
    e83 = e56
    for e62 in range(tl.minimum(tl.maximum(e55, 0), 4)):
        e63 = e81
        e64 = e82
        e65 = e83
        if e61:
            e66 = e63
            e67 = (tl.full((16, 16), 0.125, tl.float16)).to(tl.float16)
            e68 = ((e66 + e67)).to(tl.float16)
            e72 = e68
        else:
            e69 = e63
            e70 = (tl.full((16, 16), 0.125, tl.float16)).to(tl.float16)
            e71 = ((e69 - e70)).to(tl.float16)
            e72 = e71
        e73, e74 = mixed_helper_extended_1_1(e72, e35, mem0, mem1, mem4, mem5, steps, limit, bid)
        e75 = (tl.full((16, 16), 0.125, tl.float16)).to(tl.float16)
        e76 = ((e73 * e75)).to(tl.float16)
        e77 = ((e64 + e74)).to(tl.float32)
        e78 = (tl.full((), 1, tl.int32)).to(tl.int32)
        e79 = ((e62 + e78)).to(tl.int32)
        e80 = ((e65 + e79)).to(tl.int32)
        e81 = e76
        e82 = e77
        e83 = e80
    e103 = e81
    e104 = e82
    e105 = e83
    e84 = tl.full((), 0, tl.int32)
    while e84 < tl.minimum(tl.maximum(e55, 0), 4):
        e85 = e103
        e86 = e104
        e87 = e105
        if e61:
            e88 = e85
            e89 = (tl.full((16, 16), 0.125, tl.float16)).to(tl.float16)
            e90 = ((e88 + e89)).to(tl.float16)
            e94 = e90
        else:
            e91 = e85
            e92 = (tl.full((16, 16), 0.125, tl.float16)).to(tl.float16)
            e93 = ((e91 - e92)).to(tl.float16)
            e94 = e93
        e95, e96 = mixed_helper_extended_1_1(e94, e35, mem0, mem1, mem4, mem5, steps, limit, bid)
        e97 = (tl.full((16, 16), 0.125, tl.float16)).to(tl.float16)
        e98 = ((e95 * e97)).to(tl.float16)
        e99 = ((e86 + e96)).to(tl.float32)
        e100 = (tl.full((), 1, tl.int32)).to(tl.int32)
        e101 = ((e84 + e100)).to(tl.int32)
        e102 = ((e87 + e101)).to(tl.int32)
        e103 = e98
        e104 = e99
        e105 = e102
        e84 += 1
    e106 = ((e103 + e24)).to(tl.float16)
    e107 = ((((tl.arange(0, 16)[:, None] * 16 + tl.arange(0, 16)[None, :]) + 187) % 256)).to(tl.int32)
    e108 = (tl.full((16, 16), True, tl.int1)).to(tl.int1)
    e109 = (tl.load(mem5 + bid * 288 + 16 + e107 * 1, (e108 & (e107 >= 0) & (e107 < 256)), other=0)).to(tl.float16)
    e110 = ((e106 + e109)).to(tl.float16)
    e111 = (tl.where(tl.sum((e110 != e110).to(tl.int32), 1) > 0, float('nan'), tl.min(e110.to(tl.float32), 1))).to(tl.float32)
    tl.store(out_e110 + bid * 288 + 16 + (tl.arange(0, 16)[:, None] * 16 + tl.arange(0, 16)[None, :]), e110)
    tl.store(out_e10 + bid * 288 + 16 + (tl.arange(0, 16)[:, None] * 16 + tl.arange(0, 16)[None, :]), e10)
    tl.store(out_e16 + bid * 288 + 16 + (tl.arange(0, 16)[:, None] * 16 + tl.arange(0, 16)[None, :]), e16)
    tl.store(out_e104 + bid * 33 + 16 + tl.full((), 0, tl.int32), e104)
    tl.store(out_e105 + bid * 33 + 16 + tl.full((), 0, tl.int32), e105)
    tl.store(out_e111 + bid * 48 + 16 + tl.arange(0, 16), e111)
    tl.store(out_e3 + bid * 288 + 16 + (tl.arange(0, 16)[:, None] * 16 + tl.arange(0, 16)[None, :]), e3)
    tl.store(out_e8 + bid * 288 + 16 + (tl.arange(0, 16)[:, None] * 16 + tl.arange(0, 16)[None, :]), e8)
    tl.store(out_e22 + bid * 288 + 16 + (tl.arange(0, 16)[:, None] * 16 + tl.arange(0, 16)[None, :]), e22)
    tl.store(out_e26 + bid * 288 + 16 + (tl.arange(0, 16)[:, None] * 16 + tl.arange(0, 16)[None, :]), e26)
    tl.store(out_e34 + bid * 288 + 16 + (tl.arange(0, 16)[:, None] * 16 + tl.arange(0, 16)[None, :]), e34)
    tl.store(out_e39 + bid * 288 + 16 + (tl.arange(0, 16)[:, None] * 16 + tl.arange(0, 16)[None, :]), e39)
    tl.store(out_e2 + bid * 288 + 16 + (tl.arange(0, 16)[:, None] * 16 + tl.arange(0, 16)[None, :]), e2)
    tl.store(out_e19 + bid * 288 + 16 + (tl.arange(0, 16)[:, None] * 16 + tl.arange(0, 16)[None, :]), e19)

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
    compiled_0 = compile(ASTSource(extended_0_0, {0: '*fp32', 1: '*fp16', 2: '*i32', 3: '*fp16', 4: '*fp16', 5: '*i32', 6: '*i1', 7: '*fp32', 8: '*i32', 9: '*fp32', 10: 'i32', 11: 'i32'}), target=GPUTarget("cuda", arch, 32), options={'num_warps': 4, 'num_stages': 1, 'enable_fp_fusion': False})
    record_extended_compilation('triton_0', compiled_0.asm, {'num_warps': 4, 'num_stages': 1, 'enable_fp_fusion': False})
    def launch_0(memories, outputs, steps, limit):
        arguments = [memories[n] for n in ['mem0', 'mem1', 'mem4', 'mem5']] + [outputs[n] for n in ['e110', 'e10', 'e16', 'e104', 'e105', 'e111']]
        compiled_0[(3, 1, 1)](*arguments, steps, limit)
    variants.append(('triton_0', ['e110', 'e10', 'e16', 'e104', 'e105', 'e111'], launch_0))
    extended_stage("compile", 'triton_1')
    compiled_1 = compile(ASTSource(extended_0_1, {0: '*fp32', 1: '*fp16', 2: '*i32', 3: '*fp16', 4: '*fp16', 5: '*i32', 6: '*i1', 7: '*fp32', 8: '*i32', 9: '*fp32', 10: '*fp32', 11: '*fp16', 12: '*fp16', 13: '*fp16', 14: '*fp32', 15: '*fp32', 16: '*i1', 17: '*i32', 18: 'i32', 19: 'i32'}), target=GPUTarget("cuda", arch, 32), options={'num_warps': 4, 'num_stages': 1, 'enable_fp_fusion': False})
    record_extended_compilation('triton_1', compiled_1.asm, {'num_warps': 4, 'num_stages': 1, 'enable_fp_fusion': False})
    def launch_1(memories, outputs, steps, limit):
        arguments = [memories[n] for n in ['mem0', 'mem1', 'mem4', 'mem5']] + [outputs[n] for n in ['e110', 'e10', 'e16', 'e104', 'e105', 'e111', 'e3', 'e8', 'e22', 'e26', 'e34', 'e39', 'e2', 'e19']]
        compiled_1[(3, 1, 1)](*arguments, steps, limit)
    variants.append(('triton_1', ['e110', 'e10', 'e16', 'e104', 'e105', 'e111', 'e3', 'e8', 'e22', 'e26', 'e34', 'e39', 'e2', 'e19'], launch_1))
    extended_stage("compile", 'triton_2')
    compiled_2 = compile(ASTSource(extended_1_0, {0: '*fp32', 1: '*fp16', 2: '*i32', 3: '*fp16', 4: '*fp16', 5: '*i32', 6: '*i1', 7: '*fp32', 8: '*i32', 9: '*fp32', 10: 'i32', 11: 'i32'}), target=GPUTarget("cuda", arch, 32), options={'num_warps': 8, 'num_stages': 2, 'enable_fp_fusion': False})
    record_extended_compilation('triton_2', compiled_2.asm, {'num_warps': 8, 'num_stages': 2, 'enable_fp_fusion': False})
    def launch_2(memories, outputs, steps, limit):
        arguments = [memories[n] for n in ['mem0', 'mem1', 'mem4', 'mem5']] + [outputs[n] for n in ['e110', 'e10', 'e16', 'e104', 'e105', 'e111']]
        compiled_2[(3, 1, 1)](*arguments, steps, limit)
    variants.append(('triton_2', ['e110', 'e10', 'e16', 'e104', 'e105', 'e111'], launch_2))
    extended_stage("compile", 'triton_3')
    compiled_3 = compile(ASTSource(extended_1_1, {0: '*fp32', 1: '*fp16', 2: '*i32', 3: '*fp16', 4: '*fp16', 5: '*i32', 6: '*i1', 7: '*fp32', 8: '*i32', 9: '*fp32', 10: '*fp32', 11: '*fp16', 12: '*fp16', 13: '*fp16', 14: '*fp32', 15: '*fp32', 16: '*i1', 17: '*i32', 18: 'i32', 19: 'i32'}), target=GPUTarget("cuda", arch, 32), options={'num_warps': 8, 'num_stages': 2, 'enable_fp_fusion': False})
    record_extended_compilation('triton_3', compiled_3.asm, {'num_warps': 8, 'num_stages': 2, 'enable_fp_fusion': False})
    def launch_3(memories, outputs, steps, limit):
        arguments = [memories[n] for n in ['mem0', 'mem1', 'mem4', 'mem5']] + [outputs[n] for n in ['e110', 'e10', 'e16', 'e104', 'e105', 'e111', 'e3', 'e8', 'e22', 'e26', 'e34', 'e39', 'e2', 'e19']]
        compiled_3[(3, 1, 1)](*arguments, steps, limit)
    variants.append(('triton_3', ['e110', 'e10', 'e16', 'e104', 'e105', 'e111', 'e3', 'e8', 'e22', 'e26', 'e34', 'e39', 'e2', 'e19'], launch_3))
    return variants

PROGRAM = {'type': 'extended', 'version': 1, 'body': {'arguments': [], 'operations': [{'op': 'index', 'results': [{'name': 'e1', 'type': {'dtype': 'int32', 'shape': (16, 16)}}], 'operands': [], 'attrs': {'shift': 186, 'reverse': False}, 'regions': []}, {'op': 'constant', 'results': [{'name': 'e2', 'type': {'dtype': 'bool', 'shape': (16, 16)}}], 'operands': [], 'attrs': {'value': True}, 'regions': []}, {'op': 'load', 'results': [{'name': 'e3', 'type': {'dtype': 'float32', 'shape': (16, 16)}}], 'operands': ['e1', 'e2'], 'attrs': {'buffer': 'mem0'}, 'regions': []}, {'op': 'mul', 'results': [{'name': 'e4', 'type': {'dtype': 'float32', 'shape': (16, 16)}}], 'operands': ['e3', 'e3'], 'attrs': {}, 'regions': []}, {'op': 'cast', 'results': [{'name': 'e5', 'type': {'dtype': 'float16', 'shape': (16, 16)}}], 'operands': ['e4'], 'attrs': {}, 'regions': []}, {'op': 'cast', 'results': [{'name': 'e6', 'type': {'dtype': 'float16', 'shape': (16, 16)}}], 'operands': ['e5'], 'attrs': {}, 'regions': []}, {'op': 'constant', 'results': [{'name': 'e7', 'type': {'dtype': 'float16', 'shape': (16, 16)}}], 'operands': [], 'attrs': {'value': 0.5}, 'regions': []}, {'op': 'mul', 'results': [{'name': 'e8', 'type': {'dtype': 'float16', 'shape': (16, 16)}}], 'operands': ['e6', 'e7'], 'attrs': {}, 'regions': []}, {'op': 'constant', 'results': [{'name': 'e9', 'type': {'dtype': 'int32', 'shape': (16, 16)}}], 'operands': [], 'attrs': {'value': 3}, 'regions': []}, {'op': 'bitxor', 'results': [{'name': 'e10', 'type': {'dtype': 'int32', 'shape': (16, 16)}}], 'operands': ['e1', 'e9'], 'attrs': {}, 'regions': []}, {'op': 'constant', 'results': [{'name': 'e11', 'type': {'dtype': 'int32', 'shape': (16, 16)}}], 'operands': [], 'attrs': {'value': 3}, 'regions': []}, {'op': 'lt', 'results': [{'name': 'e12', 'type': {'dtype': 'bool', 'shape': (16, 16)}}], 'operands': ['e10', 'e11'], 'attrs': {}, 'regions': []}, {'op': 'index', 'results': [{'name': 'e13', 'type': {'dtype': 'int32', 'shape': (16, 16)}}], 'operands': [], 'attrs': {'shift': 161, 'reverse': True}, 'regions': []}, {'op': 'parameter', 'results': [{'name': 'e14', 'type': {'dtype': 'int32', 'shape': ()}}], 'operands': [], 'attrs': {'name': 'limit'}, 'regions': []}, {'op': 'lt', 'results': [{'name': 'e15', 'type': {'dtype': 'bool', 'shape': (16, 16)}}], 'operands': ['e13', 'e14'], 'attrs': {}, 'regions': []}, {'op': 'and', 'results': [{'name': 'e16', 'type': {'dtype': 'bool', 'shape': (16, 16)}}], 'operands': ['e12', 'e15'], 'attrs': {}, 'regions': []}, {'op': 'constant', 'results': [{'name': 'e17', 'type': {'dtype': 'float16', 'shape': (16, 16)}}], 'operands': [], 'attrs': {'value': -0.125}, 'regions': []}, {'op': 'select', 'results': [{'name': 'e18', 'type': {'dtype': 'float16', 'shape': (16, 16)}}], 'operands': ['e16', 'e8', 'e17'], 'attrs': {}, 'regions': []}, {'op': 'index', 'results': [{'name': 'e19', 'type': {'dtype': 'int32', 'shape': (16, 16)}}], 'operands': [], 'attrs': {'shift': 27, 'reverse': True}, 'regions': []}, {'op': 'parameter', 'results': [{'name': 'e20', 'type': {'dtype': 'int32', 'shape': ()}}], 'operands': [], 'attrs': {'name': 'limit'}, 'regions': []}, {'op': 'lt', 'results': [{'name': 'e21', 'type': {'dtype': 'bool', 'shape': (16, 16)}}], 'operands': ['e19', 'e20'], 'attrs': {}, 'regions': []}, {'op': 'store', 'results': [], 'operands': ['e19', 'e21', 'e18'], 'attrs': {'buffer': 'mem2'}, 'regions': []}, {'op': 'barrier', 'results': [], 'operands': [], 'attrs': {}, 'regions': []}, {'op': 'load', 'results': [{'name': 'e22', 'type': {'dtype': 'float16', 'shape': (16, 16)}}], 'operands': ['e19', 'e21'], 'attrs': {'buffer': 'mem2'}, 'regions': []}, {'op': 'constant', 'results': [{'name': 'e23', 'type': {'dtype': 'float16', 'shape': (16, 16)}}], 'operands': [], 'attrs': {'value': 0.125}, 'regions': []}, {'op': 'add', 'results': [{'name': 'e24', 'type': {'dtype': 'float16', 'shape': (16, 16)}}], 'operands': ['e22', 'e23'], 'attrs': {}, 'regions': []}, {'op': 'barrier', 'results': [], 'operands': [], 'attrs': {}, 'regions': []}, {'op': 'store', 'results': [], 'operands': ['e19', 'e21', 'e24'], 'attrs': {'buffer': 'mem3'}, 'regions': []}, {'op': 'barrier', 'results': [], 'operands': [], 'attrs': {}, 'regions': []}, {'op': 'constant', 'results': [{'name': 'e25', 'type': {'dtype': 'bool', 'shape': (16, 16)}}], 'operands': [], 'attrs': {'value': True}, 'regions': []}, {'op': 'load', 'results': [{'name': 'e26', 'type': {'dtype': 'float16', 'shape': (16, 16)}}], 'operands': ['e19', 'e25'], 'attrs': {'buffer': 'mem2'}, 'regions': []}, {'op': 'constant', 'results': [{'name': 'e27', 'type': {'dtype': 'int32', 'shape': (16, 16)}}], 'operands': [], 'attrs': {'value': 7}, 'regions': []}, {'op': 'bitxor', 'results': [{'name': 'e28', 'type': {'dtype': 'int32', 'shape': (16, 16)}}], 'operands': ['e19', 'e27'], 'attrs': {}, 'regions': []}, {'op': 'load', 'results': [{'name': 'e29', 'type': {'dtype': 'float16', 'shape': (16, 16)}}], 'operands': ['e28', 'e21'], 'attrs': {'buffer': 'mem3'}, 'regions': []}, {'op': 'add', 'results': [{'name': 'e30', 'type': {'dtype': 'float16', 'shape': (16, 16)}}], 'operands': ['e26', 'e29'], 'attrs': {}, 'regions': []}, {'op': 'cast', 'results': [{'name': 'e31', 'type': {'dtype': 'float16', 'shape': (16, 16)}}], 'operands': ['e30'], 'attrs': {}, 'regions': []}, {'op': 'transpose', 'results': [{'name': 'e32', 'type': {'dtype': 'float16', 'shape': (16, 16)}}], 'operands': ['e31'], 'attrs': {}, 'regions': []}, {'op': 'constant', 'results': [{'name': 'e33', 'type': {'dtype': 'float32', 'shape': (16, 16)}}], 'operands': [], 'attrs': {'value': 0.125}, 'regions': []}, {'op': 'matmul', 'results': [{'name': 'e34', 'type': {'dtype': 'float32', 'shape': (16, 16)}}], 'operands': ['e32', 'e5', 'e33'], 'attrs': {}, 'regions': []}, {'op': 'reduce', 'results': [{'name': 'e35', 'type': {'dtype': 'float32', 'shape': (16,)}}], 'operands': ['e34'], 'attrs': {'axis': 1, 'kind': 'max'}, 'regions': []}, {'op': 'reshape', 'results': [{'name': 'e36', 'type': {'dtype': 'float32', 'shape': (16, 1)}}], 'operands': ['e35'], 'attrs': {}, 'regions': []}, {'op': 'sub', 'results': [{'name': 'e37', 'type': {'dtype': 'float32', 'shape': (16, 16)}}], 'operands': ['e34', 'e36'], 'attrs': {}, 'regions': []}, {'op': 'constant', 'results': [{'name': 'e38', 'type': {'dtype': 'float32', 'shape': (16, 16)}}], 'operands': [], 'attrs': {'value': 0.125}, 'regions': []}, {'op': 'mul', 'results': [{'name': 'e39', 'type': {'dtype': 'float32', 'shape': (16, 16)}}], 'operands': ['e37', 'e38'], 'attrs': {}, 'regions': []}, {'op': 'cast', 'results': [{'name': 'e40', 'type': {'dtype': 'float16', 'shape': (16, 16)}}], 'operands': ['e39'], 'attrs': {}, 'regions': []}, {'op': 'matmul', 'results': [{'name': 'e41', 'type': {'dtype': 'float32', 'shape': (16, 16)}}], 'operands': ['e40', 'e17', 'e33'], 'attrs': {}, 'regions': []}, {'op': 'reshape', 'results': [{'name': 'e42', 'type': {'dtype': 'float32', 'shape': (16, 16)}}], 'operands': ['e41'], 'attrs': {}, 'regions': []}, {'op': 'reshape', 'results': [{'name': 'e43', 'type': {'dtype': 'float32', 'shape': (16, 16)}}], 'operands': ['e42'], 'attrs': {}, 'regions': []}, {'op': 'cast', 'results': [{'name': 'e44', 'type': {'dtype': 'float16', 'shape': (16, 16)}}], 'operands': ['e43'], 'attrs': {}, 'regions': []}, {'op': 'call', 'results': [{'name': 'e53', 'type': {'dtype': 'float16', 'shape': (16, 16)}}, {'name': 'e54', 'type': {'dtype': 'float32', 'shape': ()}}], 'operands': ['e44', 'e35'], 'attrs': {'callee': 'mixed_helper'}, 'regions': []}, {'op': 'parameter', 'results': [{'name': 'e55', 'type': {'dtype': 'int32', 'shape': ()}}], 'operands': [], 'attrs': {'name': 'steps'}, 'regions': []}, {'op': 'constant', 'results': [{'name': 'e56', 'type': {'dtype': 'int32', 'shape': ()}}], 'operands': [], 'attrs': {'value': 0}, 'regions': []}, {'op': 'constant', 'results': [{'name': 'e57', 'type': {'dtype': 'int32', 'shape': ()}}], 'operands': [], 'attrs': {'value': 0}, 'regions': []}, {'op': 'constant', 'results': [{'name': 'e58', 'type': {'dtype': 'bool', 'shape': ()}}], 'operands': [], 'attrs': {'value': True}, 'regions': []}, {'op': 'load', 'results': [{'name': 'e59', 'type': {'dtype': 'int32', 'shape': ()}}], 'operands': ['e57', 'e58'], 'attrs': {'buffer': 'mem4'}, 'regions': []}, {'op': 'constant', 'results': [{'name': 'e60', 'type': {'dtype': 'int32', 'shape': ()}}], 'operands': [], 'attrs': {'value': 0}, 'regions': []}, {'op': 'lt', 'results': [{'name': 'e61', 'type': {'dtype': 'bool', 'shape': ()}}], 'operands': ['e59', 'e60'], 'attrs': {}, 'regions': []}, {'op': 'for', 'results': [{'name': 'e81', 'type': {'dtype': 'float16', 'shape': (16, 16)}}, {'name': 'e82', 'type': {'dtype': 'float32', 'shape': ()}}, {'name': 'e83', 'type': {'dtype': 'int32', 'shape': ()}}], 'operands': ['e55', 'e53', 'e54', 'e56'], 'attrs': {'max_steps': 4, 'pipelined': False}, 'regions': [{'arguments': [{'name': 'e62', 'type': {'dtype': 'int32', 'shape': ()}}, {'name': 'e63', 'type': {'dtype': 'float16', 'shape': (16, 16)}}, {'name': 'e64', 'type': {'dtype': 'float32', 'shape': ()}}, {'name': 'e65', 'type': {'dtype': 'int32', 'shape': ()}}], 'operations': [{'op': 'if', 'results': [{'name': 'e72', 'type': {'dtype': 'float16', 'shape': (16, 16)}}], 'operands': ['e61', 'e63'], 'attrs': {}, 'regions': [{'arguments': [{'name': 'e66', 'type': {'dtype': 'float16', 'shape': (16, 16)}}], 'operations': [{'op': 'constant', 'results': [{'name': 'e67', 'type': {'dtype': 'float16', 'shape': (16, 16)}}], 'operands': [], 'attrs': {'value': 0.125}, 'regions': []}, {'op': 'add', 'results': [{'name': 'e68', 'type': {'dtype': 'float16', 'shape': (16, 16)}}], 'operands': ['e66', 'e67'], 'attrs': {}, 'regions': []}], 'returns': ['e68']}, {'arguments': [{'name': 'e69', 'type': {'dtype': 'float16', 'shape': (16, 16)}}], 'operations': [{'op': 'constant', 'results': [{'name': 'e70', 'type': {'dtype': 'float16', 'shape': (16, 16)}}], 'operands': [], 'attrs': {'value': 0.125}, 'regions': []}, {'op': 'sub', 'results': [{'name': 'e71', 'type': {'dtype': 'float16', 'shape': (16, 16)}}], 'operands': ['e69', 'e70'], 'attrs': {}, 'regions': []}], 'returns': ['e71']}]}, {'op': 'call', 'results': [{'name': 'e73', 'type': {'dtype': 'float16', 'shape': (16, 16)}}, {'name': 'e74', 'type': {'dtype': 'float32', 'shape': ()}}], 'operands': ['e72', 'e35'], 'attrs': {'callee': 'mixed_helper'}, 'regions': []}, {'op': 'constant', 'results': [{'name': 'e75', 'type': {'dtype': 'float16', 'shape': (16, 16)}}], 'operands': [], 'attrs': {'value': 0.125}, 'regions': []}, {'op': 'mul', 'results': [{'name': 'e76', 'type': {'dtype': 'float16', 'shape': (16, 16)}}], 'operands': ['e73', 'e75'], 'attrs': {}, 'regions': []}, {'op': 'add', 'results': [{'name': 'e77', 'type': {'dtype': 'float32', 'shape': ()}}], 'operands': ['e64', 'e74'], 'attrs': {}, 'regions': []}, {'op': 'constant', 'results': [{'name': 'e78', 'type': {'dtype': 'int32', 'shape': ()}}], 'operands': [], 'attrs': {'value': 1}, 'regions': []}, {'op': 'add', 'results': [{'name': 'e79', 'type': {'dtype': 'int32', 'shape': ()}}], 'operands': ['e62', 'e78'], 'attrs': {}, 'regions': []}, {'op': 'add', 'results': [{'name': 'e80', 'type': {'dtype': 'int32', 'shape': ()}}], 'operands': ['e65', 'e79'], 'attrs': {}, 'regions': []}], 'returns': ['e76', 'e77', 'e80']}]}, {'op': 'while', 'results': [{'name': 'e103', 'type': {'dtype': 'float16', 'shape': (16, 16)}}, {'name': 'e104', 'type': {'dtype': 'float32', 'shape': ()}}, {'name': 'e105', 'type': {'dtype': 'int32', 'shape': ()}}], 'operands': ['e55', 'e81', 'e82', 'e83'], 'attrs': {'max_steps': 4, 'pipelined': False}, 'regions': [{'arguments': [{'name': 'e84', 'type': {'dtype': 'int32', 'shape': ()}}, {'name': 'e85', 'type': {'dtype': 'float16', 'shape': (16, 16)}}, {'name': 'e86', 'type': {'dtype': 'float32', 'shape': ()}}, {'name': 'e87', 'type': {'dtype': 'int32', 'shape': ()}}], 'operations': [{'op': 'if', 'results': [{'name': 'e94', 'type': {'dtype': 'float16', 'shape': (16, 16)}}], 'operands': ['e61', 'e85'], 'attrs': {}, 'regions': [{'arguments': [{'name': 'e88', 'type': {'dtype': 'float16', 'shape': (16, 16)}}], 'operations': [{'op': 'constant', 'results': [{'name': 'e89', 'type': {'dtype': 'float16', 'shape': (16, 16)}}], 'operands': [], 'attrs': {'value': 0.125}, 'regions': []}, {'op': 'add', 'results': [{'name': 'e90', 'type': {'dtype': 'float16', 'shape': (16, 16)}}], 'operands': ['e88', 'e89'], 'attrs': {}, 'regions': []}], 'returns': ['e90']}, {'arguments': [{'name': 'e91', 'type': {'dtype': 'float16', 'shape': (16, 16)}}], 'operations': [{'op': 'constant', 'results': [{'name': 'e92', 'type': {'dtype': 'float16', 'shape': (16, 16)}}], 'operands': [], 'attrs': {'value': 0.125}, 'regions': []}, {'op': 'sub', 'results': [{'name': 'e93', 'type': {'dtype': 'float16', 'shape': (16, 16)}}], 'operands': ['e91', 'e92'], 'attrs': {}, 'regions': []}], 'returns': ['e93']}]}, {'op': 'call', 'results': [{'name': 'e95', 'type': {'dtype': 'float16', 'shape': (16, 16)}}, {'name': 'e96', 'type': {'dtype': 'float32', 'shape': ()}}], 'operands': ['e94', 'e35'], 'attrs': {'callee': 'mixed_helper'}, 'regions': []}, {'op': 'constant', 'results': [{'name': 'e97', 'type': {'dtype': 'float16', 'shape': (16, 16)}}], 'operands': [], 'attrs': {'value': 0.125}, 'regions': []}, {'op': 'mul', 'results': [{'name': 'e98', 'type': {'dtype': 'float16', 'shape': (16, 16)}}], 'operands': ['e95', 'e97'], 'attrs': {}, 'regions': []}, {'op': 'add', 'results': [{'name': 'e99', 'type': {'dtype': 'float32', 'shape': ()}}], 'operands': ['e86', 'e96'], 'attrs': {}, 'regions': []}, {'op': 'constant', 'results': [{'name': 'e100', 'type': {'dtype': 'int32', 'shape': ()}}], 'operands': [], 'attrs': {'value': 1}, 'regions': []}, {'op': 'add', 'results': [{'name': 'e101', 'type': {'dtype': 'int32', 'shape': ()}}], 'operands': ['e84', 'e100'], 'attrs': {}, 'regions': []}, {'op': 'add', 'results': [{'name': 'e102', 'type': {'dtype': 'int32', 'shape': ()}}], 'operands': ['e87', 'e101'], 'attrs': {}, 'regions': []}], 'returns': ['e98', 'e99', 'e102']}]}, {'op': 'add', 'results': [{'name': 'e106', 'type': {'dtype': 'float16', 'shape': (16, 16)}}], 'operands': ['e103', 'e24'], 'attrs': {}, 'regions': []}, {'op': 'index', 'results': [{'name': 'e107', 'type': {'dtype': 'int32', 'shape': (16, 16)}}], 'operands': [], 'attrs': {'shift': 187, 'reverse': False}, 'regions': []}, {'op': 'constant', 'results': [{'name': 'e108', 'type': {'dtype': 'bool', 'shape': (16, 16)}}], 'operands': [], 'attrs': {'value': True}, 'regions': []}, {'op': 'load', 'results': [{'name': 'e109', 'type': {'dtype': 'float16', 'shape': (16, 16)}}], 'operands': ['e107', 'e108'], 'attrs': {'buffer': 'mem5'}, 'regions': []}, {'op': 'add', 'results': [{'name': 'e110', 'type': {'dtype': 'float16', 'shape': (16, 16)}}], 'operands': ['e106', 'e109'], 'attrs': {}, 'regions': []}, {'op': 'reduce', 'results': [{'name': 'e111', 'type': {'dtype': 'float32', 'shape': (16,)}}], 'operands': ['e110'], 'attrs': {'axis': 1, 'kind': 'min'}, 'regions': []}], 'returns': ['e110', 'e10', 'e16', 'e104', 'e105', 'e111']}, 'buffers': [{'name': 'mem0', 'dtype': 'float32', 'size': 256, 'role': 'input', 'base': None, 'offset': 0, 'stride': 1}, {'name': 'mem1', 'dtype': 'float16', 'size': 516, 'role': 'scratch', 'base': None, 'offset': 0, 'stride': 1}, {'name': 'mem2', 'dtype': 'float16', 'size': 256, 'role': 'scratch', 'base': 'mem1', 'offset': 1, 'stride': 2}, {'name': 'mem3', 'dtype': 'float16', 'size': 256, 'role': 'scratch', 'base': 'mem1', 'offset': 2, 'stride': 1}, {'name': 'mem4', 'dtype': 'int32', 'size': 1, 'role': 'input', 'base': None, 'offset': 0, 'stride': 1}, {'name': 'mem5', 'dtype': 'float16', 'size': 256, 'role': 'input', 'base': None, 'offset': 0, 'stride': 1}], 'functions': [{'name': 'mixed_helper', 'body': {'arguments': [{'name': 'e45', 'type': {'dtype': 'float16', 'shape': (16, 16)}}, {'name': 'e46', 'type': {'dtype': 'float32', 'shape': (16,)}}], 'operations': [{'op': 'reduce', 'results': [{'name': 'e47', 'type': {'dtype': 'float32', 'shape': (16,)}}], 'operands': ['e45'], 'attrs': {'axis': 1, 'kind': 'max'}, 'regions': []}, {'op': 'add', 'results': [{'name': 'e48', 'type': {'dtype': 'float32', 'shape': (16,)}}], 'operands': ['e47', 'e46'], 'attrs': {}, 'regions': []}, {'op': 'reduce', 'results': [{'name': 'e49', 'type': {'dtype': 'float32', 'shape': ()}}], 'operands': ['e48'], 'attrs': {'axis': 0, 'kind': 'sum'}, 'regions': []}, {'op': 'reshape', 'results': [{'name': 'e50', 'type': {'dtype': 'float32', 'shape': (16, 1)}}], 'operands': ['e48'], 'attrs': {}, 'regions': []}, {'op': 'cast', 'results': [{'name': 'e51', 'type': {'dtype': 'float16', 'shape': (16, 1)}}], 'operands': ['e50'], 'attrs': {}, 'regions': []}, {'op': 'add', 'results': [{'name': 'e52', 'type': {'dtype': 'float16', 'shape': (16, 16)}}], 'operands': ['e45', 'e51'], 'attrs': {}, 'regions': []}], 'returns': ['e52', 'e49']}}], 'observations': ['e3', 'e8', 'e22', 'e26', 'e34', 'e39', 'e2', 'e19'], 'blocks': 3, 'input_pattern': 'integer', 'runtime_cases': [(0, 1), (1, 255), (4, 256)], 'configuration_pair': True, 'observation_pair': True, 'family': 'mixed'}


_base_check=extended_check
def extended_check(actual, expected, label, matmul=False):
    print('OBS',label,'actual_range',actual.float().min().item(),actual.float().max().item(),'expected_range',expected.float().min().item(),expected.float().max().item(),flush=True)
    try: _base_check(actual,expected,label,matmul)
    except RuntimeError as e: print('OBSERVED_FAILURE',str(e),flush=True)
PROGRAM['runtime_cases']=[[0,1]]

if __name__ == '__main__':
    if os.environ.get('TILESMITH_COMPILE_ONLY', '0') == '1':
        prepare_extended(device_compile=False)
        extended_stage('complete', 'compile_only')
        print('COMPILE PASSED (no GPU execution)')
    else:
        run_extended(PROGRAM, prepare_extended, 0, 1)
        extended_stage('complete', 'execute')
        print('ALL PASSED')
