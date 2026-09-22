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
    for seed in (input_seed, input_seed + 1):
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
def mixed_helper_extended_0_0(e56, e57, mem0, mem1, mem2, mem3, mem6, mem7, mem8, steps, limit, bid):
    e58 = (tl.sum(e56.to(tl.float32), 1)).to(tl.float32)
    e59 = ((e58 + e57)).to(tl.float32)
    e60 = (tl.sum(e59.to(tl.float32), 0)).to(tl.float32)
    e61 = (tl.reshape(e59, (16, 1))).to(tl.float32)
    e62 = (e61.to(tl.float16)).to(tl.float16)
    e63 = ((e56 + e62)).to(tl.float16)
    return e63, e60
@triton.jit
def extended_0_0(mem0, mem1, mem2, mem3, mem6, mem7, mem8, out_e122, out_e18, out_e24, out_e115, out_e116, steps, limit):
    bid = tl.program_id(0)
    e1 = ((((511 - (tl.arange(0, 32)[:, None] * 16 + tl.arange(0, 16)[None, :])) + 44) % 512)).to(tl.int32)
    e2 = (tl.full((32, 16), True, tl.int1)).to(tl.int1)
    e3 = (tl.load(mem0 + bid * 544 + 16 + e1 * 1, (e2 & (e1 >= 0) & (e1 < 512)), other=0)).to(tl.float32)
    e4 = ((e3 + e3)).to(tl.float32)
    e5 = ((((tl.arange(0, 32)[:, None] * 16 + tl.arange(0, 16)[None, :]) + 370) % 512)).to(tl.int32)
    e6 = (tl.full((32, 16), True, tl.int1)).to(tl.int1)
    e7 = (tl.load(mem1 + bid * 544 + 16 + e5 * 1, (e6 & (e5 >= 0) & (e5 < 512)), other=0)).to(tl.float32)
    e8 = ((e4 * e7)).to(tl.float32)
    e9 = (e8.to(tl.float16)).to(tl.float16)
    e10 = ((e9 + e9)).to(tl.float16)
    e11 = (e10.to(tl.float16)).to(tl.float16)
    e12 = (tl.full((32, 16), 0.5, tl.float16)).to(tl.float16)
    e13 = ((e11 * e12)).to(tl.float16)
    e14 = ((((511 - (tl.arange(0, 32)[:, None] * 16 + tl.arange(0, 16)[None, :])) + 389) % 512)).to(tl.int32)
    e15 = (tl.full((32, 16), True, tl.int1)).to(tl.int1)
    e16 = (tl.load(mem2 + bid * 544 + 16 + e14 * 1, (e15 & (e14 >= 0) & (e14 < 512)), other=0)).to(tl.int32)
    e17 = (tl.full((32, 16), 7, tl.int32)).to(tl.int32)
    e18 = ((e16 ^ e17)).to(tl.int32)
    e19 = (tl.full((32, 16), 3, tl.int32)).to(tl.int32)
    e20 = ((e18 < e19)).to(tl.int1)
    e21 = ((((511 - (tl.arange(0, 32)[:, None] * 16 + tl.arange(0, 16)[None, :])) + 166) % 512)).to(tl.int32)
    e22 = (limit).to(tl.int32)
    e23 = ((e21 < e22)).to(tl.int1)
    e24 = ((e20 & e23)).to(tl.int1)
    e25 = (tl.full((32, 16), -0.125, tl.float16)).to(tl.float16)
    e26 = (tl.where(e24, e13, e25)).to(tl.float16)
    e27 = ((((tl.arange(0, 32)[:, None] * 16 + tl.arange(0, 16)[None, :]) + 363) % 512)).to(tl.int32)
    e28 = (limit).to(tl.int32)
    e29 = ((e27 < e28)).to(tl.int1)
    tl.store(mem3 + bid * 1060 + 17 + e27 * 2, e26, (e29 & (e27 >= 0) & (e27 < 512)))
    tl.debug_barrier()
    e30 = (tl.load(mem3 + bid * 1060 + 17 + e27 * 2, (e29 & (e27 >= 0) & (e27 < 512)), other=0)).to(tl.float16)
    e31 = (tl.full((32, 16), 0.125, tl.float16)).to(tl.float16)
    e32 = ((e30 + e31)).to(tl.float16)
    tl.debug_barrier()
    tl.store(mem3 + bid * 1060 + 18 + e27 * 1, e32, (e29 & (e27 >= 0) & (e27 < 512)))
    tl.debug_barrier()
    e33 = (tl.full((32, 16), True, tl.int1)).to(tl.int1)
    e34 = (tl.load(mem3 + bid * 1060 + 17 + e27 * 2, (e33 & (e27 >= 0) & (e27 < 512)), other=0)).to(tl.float16)
    e35 = (tl.full((32, 16), 7, tl.int32)).to(tl.int32)
    e36 = ((e27 ^ e35)).to(tl.int32)
    e37 = (tl.load(mem3 + bid * 1060 + 18 + e36 * 1, (e29 & (e36 >= 0) & (e36 < 512)), other=0)).to(tl.float16)
    e38 = ((e34 + e37)).to(tl.float16)
    e39 = (e38.to(tl.float16)).to(tl.float16)
    e40 = (tl.trans(e39)).to(tl.float16)
    e41 = (tl.full((16, 16), 0.0, tl.float32)).to(tl.float32)
    e42 = (tl.dot(e40, e34, e41, input_precision='ieee')).to(tl.float32)
    e43 = (tl.where(tl.sum((e42 != e42).to(tl.int32), 1) > 0, float('nan'), tl.max(e42.to(tl.float32), 1))).to(tl.float32)
    e44 = (tl.reshape(e43, (16, 1))).to(tl.float32)
    e45 = ((e42 - e44)).to(tl.float32)
    e46 = (tl.full((16, 16), 0.125, tl.float32)).to(tl.float32)
    e47 = ((e45 * e46)).to(tl.float32)
    e48 = ((((tl.arange(0, 16)[:, None] * 16 + tl.arange(0, 16)[None, :]) + 87) % 256)).to(tl.int32)
    e49 = (tl.full((16, 16), True, tl.int1)).to(tl.int1)
    e50 = (tl.load(mem6 + bid * 288 + 16 + e48 * 1, (e49 & (e48 >= 0) & (e48 < 256)), other=0)).to(tl.float16)
    e51 = (e47.to(tl.float16)).to(tl.float16)
    e52 = (tl.dot(e51, e50, e41, input_precision='ieee')).to(tl.float32)
    e53 = (tl.reshape(e52, (16, 16))).to(tl.float32)
    e54 = (tl.reshape(e53, (16, 16))).to(tl.float32)
    e55 = (e54.to(tl.float16)).to(tl.float16)
    e64, e65 = mixed_helper_extended_0_0(e55, e43, mem0, mem1, mem2, mem3, mem6, mem7, mem8, steps, limit, bid)
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
    for e73 in tl.range(0, tl.minimum(tl.maximum(e66, 0), 4), num_stages=1):
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
        e84, e85 = mixed_helper_extended_0_0(e83, e43, mem0, mem1, mem2, mem3, mem6, mem7, mem8, steps, limit, bid)
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
        e106, e107 = mixed_helper_extended_0_0(e105, e43, mem0, mem1, mem2, mem3, mem6, mem7, mem8, steps, limit, bid)
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
    e117 = ((e114 + e50)).to(tl.float16)
    e118 = ((((tl.arange(0, 16)[:, None] * 16 + tl.arange(0, 16)[None, :]) + 33) % 256)).to(tl.int32)
    e119 = (tl.full((16, 16), True, tl.int1)).to(tl.int1)
    e120 = (tl.load(mem8 + bid * 288 + 16 + e118 * 1, (e119 & (e118 >= 0) & (e118 < 256)), other=0)).to(tl.float16)
    e121 = ((e117 * e120)).to(tl.float16)
    e122 = ((e121 - e120)).to(tl.float16)
    tl.store(out_e122 + bid * 288 + 16 + (tl.arange(0, 16)[:, None] * 16 + tl.arange(0, 16)[None, :]), e122)
    tl.store(out_e18 + bid * 544 + 16 + (tl.arange(0, 32)[:, None] * 16 + tl.arange(0, 16)[None, :]), e18)
    tl.store(out_e24 + bid * 544 + 16 + (tl.arange(0, 32)[:, None] * 16 + tl.arange(0, 16)[None, :]), e24)
    tl.store(out_e115 + bid * 33 + 16 + tl.full((), 0, tl.int32), e115)
    tl.store(out_e116 + bid * 33 + 16 + tl.full((), 0, tl.int32), e116)

@triton.jit
def mixed_helper_extended_0_1(e56, e57, mem0, mem1, mem2, mem3, mem6, mem7, mem8, steps, limit, bid):
    e58 = (tl.sum(e56.to(tl.float32), 1)).to(tl.float32)
    e59 = ((e58 + e57)).to(tl.float32)
    e60 = (tl.sum(e59.to(tl.float32), 0)).to(tl.float32)
    e61 = (tl.reshape(e59, (16, 1))).to(tl.float32)
    e62 = (e61.to(tl.float16)).to(tl.float16)
    e63 = ((e56 + e62)).to(tl.float16)
    return e63, e60
@triton.jit
def extended_0_1(mem0, mem1, mem2, mem3, mem6, mem7, mem8, out_e122, out_e18, out_e24, out_e115, out_e116, out_e3, out_e13, out_e30, out_e34, out_e42, out_e47, out_e19, out_e28, steps, limit):
    bid = tl.program_id(0)
    e1 = ((((511 - (tl.arange(0, 32)[:, None] * 16 + tl.arange(0, 16)[None, :])) + 44) % 512)).to(tl.int32)
    e2 = (tl.full((32, 16), True, tl.int1)).to(tl.int1)
    e3 = (tl.load(mem0 + bid * 544 + 16 + e1 * 1, (e2 & (e1 >= 0) & (e1 < 512)), other=0)).to(tl.float32)
    e4 = ((e3 + e3)).to(tl.float32)
    e5 = ((((tl.arange(0, 32)[:, None] * 16 + tl.arange(0, 16)[None, :]) + 370) % 512)).to(tl.int32)
    e6 = (tl.full((32, 16), True, tl.int1)).to(tl.int1)
    e7 = (tl.load(mem1 + bid * 544 + 16 + e5 * 1, (e6 & (e5 >= 0) & (e5 < 512)), other=0)).to(tl.float32)
    e8 = ((e4 * e7)).to(tl.float32)
    e9 = (e8.to(tl.float16)).to(tl.float16)
    e10 = ((e9 + e9)).to(tl.float16)
    e11 = (e10.to(tl.float16)).to(tl.float16)
    e12 = (tl.full((32, 16), 0.5, tl.float16)).to(tl.float16)
    e13 = ((e11 * e12)).to(tl.float16)
    e14 = ((((511 - (tl.arange(0, 32)[:, None] * 16 + tl.arange(0, 16)[None, :])) + 389) % 512)).to(tl.int32)
    e15 = (tl.full((32, 16), True, tl.int1)).to(tl.int1)
    e16 = (tl.load(mem2 + bid * 544 + 16 + e14 * 1, (e15 & (e14 >= 0) & (e14 < 512)), other=0)).to(tl.int32)
    e17 = (tl.full((32, 16), 7, tl.int32)).to(tl.int32)
    e18 = ((e16 ^ e17)).to(tl.int32)
    e19 = (tl.full((32, 16), 3, tl.int32)).to(tl.int32)
    e20 = ((e18 < e19)).to(tl.int1)
    e21 = ((((511 - (tl.arange(0, 32)[:, None] * 16 + tl.arange(0, 16)[None, :])) + 166) % 512)).to(tl.int32)
    e22 = (limit).to(tl.int32)
    e23 = ((e21 < e22)).to(tl.int1)
    e24 = ((e20 & e23)).to(tl.int1)
    e25 = (tl.full((32, 16), -0.125, tl.float16)).to(tl.float16)
    e26 = (tl.where(e24, e13, e25)).to(tl.float16)
    e27 = ((((tl.arange(0, 32)[:, None] * 16 + tl.arange(0, 16)[None, :]) + 363) % 512)).to(tl.int32)
    e28 = (limit).to(tl.int32)
    e29 = ((e27 < e28)).to(tl.int1)
    tl.store(mem3 + bid * 1060 + 17 + e27 * 2, e26, (e29 & (e27 >= 0) & (e27 < 512)))
    tl.debug_barrier()
    e30 = (tl.load(mem3 + bid * 1060 + 17 + e27 * 2, (e29 & (e27 >= 0) & (e27 < 512)), other=0)).to(tl.float16)
    e31 = (tl.full((32, 16), 0.125, tl.float16)).to(tl.float16)
    e32 = ((e30 + e31)).to(tl.float16)
    tl.debug_barrier()
    tl.store(mem3 + bid * 1060 + 18 + e27 * 1, e32, (e29 & (e27 >= 0) & (e27 < 512)))
    tl.debug_barrier()
    e33 = (tl.full((32, 16), True, tl.int1)).to(tl.int1)
    e34 = (tl.load(mem3 + bid * 1060 + 17 + e27 * 2, (e33 & (e27 >= 0) & (e27 < 512)), other=0)).to(tl.float16)
    e35 = (tl.full((32, 16), 7, tl.int32)).to(tl.int32)
    e36 = ((e27 ^ e35)).to(tl.int32)
    e37 = (tl.load(mem3 + bid * 1060 + 18 + e36 * 1, (e29 & (e36 >= 0) & (e36 < 512)), other=0)).to(tl.float16)
    e38 = ((e34 + e37)).to(tl.float16)
    e39 = (e38.to(tl.float16)).to(tl.float16)
    e40 = (tl.trans(e39)).to(tl.float16)
    e41 = (tl.full((16, 16), 0.0, tl.float32)).to(tl.float32)
    e42 = (tl.dot(e40, e34, e41, input_precision='ieee')).to(tl.float32)
    e43 = (tl.where(tl.sum((e42 != e42).to(tl.int32), 1) > 0, float('nan'), tl.max(e42.to(tl.float32), 1))).to(tl.float32)
    e44 = (tl.reshape(e43, (16, 1))).to(tl.float32)
    e45 = ((e42 - e44)).to(tl.float32)
    e46 = (tl.full((16, 16), 0.125, tl.float32)).to(tl.float32)
    e47 = ((e45 * e46)).to(tl.float32)
    e48 = ((((tl.arange(0, 16)[:, None] * 16 + tl.arange(0, 16)[None, :]) + 87) % 256)).to(tl.int32)
    e49 = (tl.full((16, 16), True, tl.int1)).to(tl.int1)
    e50 = (tl.load(mem6 + bid * 288 + 16 + e48 * 1, (e49 & (e48 >= 0) & (e48 < 256)), other=0)).to(tl.float16)
    e51 = (e47.to(tl.float16)).to(tl.float16)
    e52 = (tl.dot(e51, e50, e41, input_precision='ieee')).to(tl.float32)
    e53 = (tl.reshape(e52, (16, 16))).to(tl.float32)
    e54 = (tl.reshape(e53, (16, 16))).to(tl.float32)
    e55 = (e54.to(tl.float16)).to(tl.float16)
    e64, e65 = mixed_helper_extended_0_1(e55, e43, mem0, mem1, mem2, mem3, mem6, mem7, mem8, steps, limit, bid)
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
    for e73 in tl.range(0, tl.minimum(tl.maximum(e66, 0), 4), num_stages=1):
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
        e84, e85 = mixed_helper_extended_0_1(e83, e43, mem0, mem1, mem2, mem3, mem6, mem7, mem8, steps, limit, bid)
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
        e106, e107 = mixed_helper_extended_0_1(e105, e43, mem0, mem1, mem2, mem3, mem6, mem7, mem8, steps, limit, bid)
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
    e117 = ((e114 + e50)).to(tl.float16)
    e118 = ((((tl.arange(0, 16)[:, None] * 16 + tl.arange(0, 16)[None, :]) + 33) % 256)).to(tl.int32)
    e119 = (tl.full((16, 16), True, tl.int1)).to(tl.int1)
    e120 = (tl.load(mem8 + bid * 288 + 16 + e118 * 1, (e119 & (e118 >= 0) & (e118 < 256)), other=0)).to(tl.float16)
    e121 = ((e117 * e120)).to(tl.float16)
    e122 = ((e121 - e120)).to(tl.float16)
    tl.store(out_e122 + bid * 288 + 16 + (tl.arange(0, 16)[:, None] * 16 + tl.arange(0, 16)[None, :]), e122)
    tl.store(out_e18 + bid * 544 + 16 + (tl.arange(0, 32)[:, None] * 16 + tl.arange(0, 16)[None, :]), e18)
    tl.store(out_e24 + bid * 544 + 16 + (tl.arange(0, 32)[:, None] * 16 + tl.arange(0, 16)[None, :]), e24)
    tl.store(out_e115 + bid * 33 + 16 + tl.full((), 0, tl.int32), e115)
    tl.store(out_e116 + bid * 33 + 16 + tl.full((), 0, tl.int32), e116)
    tl.store(out_e3 + bid * 544 + 16 + (tl.arange(0, 32)[:, None] * 16 + tl.arange(0, 16)[None, :]), e3)
    tl.store(out_e13 + bid * 544 + 16 + (tl.arange(0, 32)[:, None] * 16 + tl.arange(0, 16)[None, :]), e13)
    tl.store(out_e30 + bid * 544 + 16 + (tl.arange(0, 32)[:, None] * 16 + tl.arange(0, 16)[None, :]), e30)
    tl.store(out_e34 + bid * 544 + 16 + (tl.arange(0, 32)[:, None] * 16 + tl.arange(0, 16)[None, :]), e34)
    tl.store(out_e42 + bid * 288 + 16 + (tl.arange(0, 16)[:, None] * 16 + tl.arange(0, 16)[None, :]), e42)
    tl.store(out_e47 + bid * 288 + 16 + (tl.arange(0, 16)[:, None] * 16 + tl.arange(0, 16)[None, :]), e47)
    tl.store(out_e19 + bid * 544 + 16 + (tl.arange(0, 32)[:, None] * 16 + tl.arange(0, 16)[None, :]), e19)
    tl.store(out_e28 + bid * 33 + 16 + tl.full((), 0, tl.int32), e28)

@triton.jit
def mixed_helper_extended_1_0(e56, e57, mem0, mem1, mem2, mem3, mem6, mem7, mem8, steps, limit, bid):
    e58 = (tl.sum(e56.to(tl.float32), 1)).to(tl.float32)
    e59 = ((e58 + e57)).to(tl.float32)
    e60 = (tl.sum(e59.to(tl.float32), 0)).to(tl.float32)
    e61 = (tl.reshape(e59, (16, 1))).to(tl.float32)
    e62 = (e61.to(tl.float16)).to(tl.float16)
    e63 = ((e56 + e62)).to(tl.float16)
    return e63, e60
@triton.jit
def extended_1_0(mem0, mem1, mem2, mem3, mem6, mem7, mem8, out_e122, out_e18, out_e24, out_e115, out_e116, steps, limit):
    bid = tl.program_id(0)
    e1 = ((((511 - (tl.arange(0, 32)[:, None] * 16 + tl.arange(0, 16)[None, :])) + 44) % 512)).to(tl.int32)
    e2 = (tl.full((32, 16), True, tl.int1)).to(tl.int1)
    e3 = (tl.load(mem0 + bid * 544 + 16 + e1 * 1, (e2 & (e1 >= 0) & (e1 < 512)), other=0)).to(tl.float32)
    e4 = ((e3 + e3)).to(tl.float32)
    e5 = ((((tl.arange(0, 32)[:, None] * 16 + tl.arange(0, 16)[None, :]) + 370) % 512)).to(tl.int32)
    e6 = (tl.full((32, 16), True, tl.int1)).to(tl.int1)
    e7 = (tl.load(mem1 + bid * 544 + 16 + e5 * 1, (e6 & (e5 >= 0) & (e5 < 512)), other=0)).to(tl.float32)
    e8 = ((e4 * e7)).to(tl.float32)
    e9 = (e8.to(tl.float16)).to(tl.float16)
    e10 = ((e9 + e9)).to(tl.float16)
    e11 = (e10.to(tl.float16)).to(tl.float16)
    e12 = (tl.full((32, 16), 0.5, tl.float16)).to(tl.float16)
    e13 = ((e11 * e12)).to(tl.float16)
    e14 = ((((511 - (tl.arange(0, 32)[:, None] * 16 + tl.arange(0, 16)[None, :])) + 389) % 512)).to(tl.int32)
    e15 = (tl.full((32, 16), True, tl.int1)).to(tl.int1)
    e16 = (tl.load(mem2 + bid * 544 + 16 + e14 * 1, (e15 & (e14 >= 0) & (e14 < 512)), other=0)).to(tl.int32)
    e17 = (tl.full((32, 16), 7, tl.int32)).to(tl.int32)
    e18 = ((e16 ^ e17)).to(tl.int32)
    e19 = (tl.full((32, 16), 3, tl.int32)).to(tl.int32)
    e20 = ((e18 < e19)).to(tl.int1)
    e21 = ((((511 - (tl.arange(0, 32)[:, None] * 16 + tl.arange(0, 16)[None, :])) + 166) % 512)).to(tl.int32)
    e22 = (limit).to(tl.int32)
    e23 = ((e21 < e22)).to(tl.int1)
    e24 = ((e20 & e23)).to(tl.int1)
    e25 = (tl.full((32, 16), -0.125, tl.float16)).to(tl.float16)
    e26 = (tl.where(e24, e13, e25)).to(tl.float16)
    e27 = ((((tl.arange(0, 32)[:, None] * 16 + tl.arange(0, 16)[None, :]) + 363) % 512)).to(tl.int32)
    e28 = (limit).to(tl.int32)
    e29 = ((e27 < e28)).to(tl.int1)
    tl.store(mem3 + bid * 1060 + 17 + e27 * 2, e26, (e29 & (e27 >= 0) & (e27 < 512)))
    tl.debug_barrier()
    e30 = (tl.load(mem3 + bid * 1060 + 17 + e27 * 2, (e29 & (e27 >= 0) & (e27 < 512)), other=0)).to(tl.float16)
    e31 = (tl.full((32, 16), 0.125, tl.float16)).to(tl.float16)
    e32 = ((e30 + e31)).to(tl.float16)
    tl.debug_barrier()
    tl.store(mem3 + bid * 1060 + 18 + e27 * 1, e32, (e29 & (e27 >= 0) & (e27 < 512)))
    tl.debug_barrier()
    e33 = (tl.full((32, 16), True, tl.int1)).to(tl.int1)
    e34 = (tl.load(mem3 + bid * 1060 + 17 + e27 * 2, (e33 & (e27 >= 0) & (e27 < 512)), other=0)).to(tl.float16)
    e35 = (tl.full((32, 16), 7, tl.int32)).to(tl.int32)
    e36 = ((e27 ^ e35)).to(tl.int32)
    e37 = (tl.load(mem3 + bid * 1060 + 18 + e36 * 1, (e29 & (e36 >= 0) & (e36 < 512)), other=0)).to(tl.float16)
    e38 = ((e34 + e37)).to(tl.float16)
    e39 = (e38.to(tl.float16)).to(tl.float16)
    e40 = (tl.trans(e39)).to(tl.float16)
    e41 = (tl.full((16, 16), 0.0, tl.float32)).to(tl.float32)
    e42 = (tl.dot(e40, e34, e41, input_precision='ieee')).to(tl.float32)
    e43 = (tl.where(tl.sum((e42 != e42).to(tl.int32), 1) > 0, float('nan'), tl.max(e42.to(tl.float32), 1))).to(tl.float32)
    e44 = (tl.reshape(e43, (16, 1))).to(tl.float32)
    e45 = ((e42 - e44)).to(tl.float32)
    e46 = (tl.full((16, 16), 0.125, tl.float32)).to(tl.float32)
    e47 = ((e45 * e46)).to(tl.float32)
    e48 = ((((tl.arange(0, 16)[:, None] * 16 + tl.arange(0, 16)[None, :]) + 87) % 256)).to(tl.int32)
    e49 = (tl.full((16, 16), True, tl.int1)).to(tl.int1)
    e50 = (tl.load(mem6 + bid * 288 + 16 + e48 * 1, (e49 & (e48 >= 0) & (e48 < 256)), other=0)).to(tl.float16)
    e51 = (e47.to(tl.float16)).to(tl.float16)
    e52 = (tl.dot(e51, e50, e41, input_precision='ieee')).to(tl.float32)
    e53 = (tl.reshape(e52, (16, 16))).to(tl.float32)
    e54 = (tl.reshape(e53, (16, 16))).to(tl.float32)
    e55 = (e54.to(tl.float16)).to(tl.float16)
    e64, e65 = mixed_helper_extended_1_0(e55, e43, mem0, mem1, mem2, mem3, mem6, mem7, mem8, steps, limit, bid)
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
    for e73 in tl.range(0, tl.minimum(tl.maximum(e66, 0), 4), num_stages=2):
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
        e84, e85 = mixed_helper_extended_1_0(e83, e43, mem0, mem1, mem2, mem3, mem6, mem7, mem8, steps, limit, bid)
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
        e106, e107 = mixed_helper_extended_1_0(e105, e43, mem0, mem1, mem2, mem3, mem6, mem7, mem8, steps, limit, bid)
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
    e117 = ((e114 + e50)).to(tl.float16)
    e118 = ((((tl.arange(0, 16)[:, None] * 16 + tl.arange(0, 16)[None, :]) + 33) % 256)).to(tl.int32)
    e119 = (tl.full((16, 16), True, tl.int1)).to(tl.int1)
    e120 = (tl.load(mem8 + bid * 288 + 16 + e118 * 1, (e119 & (e118 >= 0) & (e118 < 256)), other=0)).to(tl.float16)
    e121 = ((e117 * e120)).to(tl.float16)
    e122 = ((e121 - e120)).to(tl.float16)
    tl.store(out_e122 + bid * 288 + 16 + (tl.arange(0, 16)[:, None] * 16 + tl.arange(0, 16)[None, :]), e122)
    tl.store(out_e18 + bid * 544 + 16 + (tl.arange(0, 32)[:, None] * 16 + tl.arange(0, 16)[None, :]), e18)
    tl.store(out_e24 + bid * 544 + 16 + (tl.arange(0, 32)[:, None] * 16 + tl.arange(0, 16)[None, :]), e24)
    tl.store(out_e115 + bid * 33 + 16 + tl.full((), 0, tl.int32), e115)
    tl.store(out_e116 + bid * 33 + 16 + tl.full((), 0, tl.int32), e116)

@triton.jit
def mixed_helper_extended_1_1(e56, e57, mem0, mem1, mem2, mem3, mem6, mem7, mem8, steps, limit, bid):
    e58 = (tl.sum(e56.to(tl.float32), 1)).to(tl.float32)
    e59 = ((e58 + e57)).to(tl.float32)
    e60 = (tl.sum(e59.to(tl.float32), 0)).to(tl.float32)
    e61 = (tl.reshape(e59, (16, 1))).to(tl.float32)
    e62 = (e61.to(tl.float16)).to(tl.float16)
    e63 = ((e56 + e62)).to(tl.float16)
    return e63, e60
@triton.jit
def extended_1_1(mem0, mem1, mem2, mem3, mem6, mem7, mem8, out_e122, out_e18, out_e24, out_e115, out_e116, out_e3, out_e13, out_e30, out_e34, out_e42, out_e47, out_e19, out_e28, steps, limit):
    bid = tl.program_id(0)
    e1 = ((((511 - (tl.arange(0, 32)[:, None] * 16 + tl.arange(0, 16)[None, :])) + 44) % 512)).to(tl.int32)
    e2 = (tl.full((32, 16), True, tl.int1)).to(tl.int1)
    e3 = (tl.load(mem0 + bid * 544 + 16 + e1 * 1, (e2 & (e1 >= 0) & (e1 < 512)), other=0)).to(tl.float32)
    e4 = ((e3 + e3)).to(tl.float32)
    e5 = ((((tl.arange(0, 32)[:, None] * 16 + tl.arange(0, 16)[None, :]) + 370) % 512)).to(tl.int32)
    e6 = (tl.full((32, 16), True, tl.int1)).to(tl.int1)
    e7 = (tl.load(mem1 + bid * 544 + 16 + e5 * 1, (e6 & (e5 >= 0) & (e5 < 512)), other=0)).to(tl.float32)
    e8 = ((e4 * e7)).to(tl.float32)
    e9 = (e8.to(tl.float16)).to(tl.float16)
    e10 = ((e9 + e9)).to(tl.float16)
    e11 = (e10.to(tl.float16)).to(tl.float16)
    e12 = (tl.full((32, 16), 0.5, tl.float16)).to(tl.float16)
    e13 = ((e11 * e12)).to(tl.float16)
    e14 = ((((511 - (tl.arange(0, 32)[:, None] * 16 + tl.arange(0, 16)[None, :])) + 389) % 512)).to(tl.int32)
    e15 = (tl.full((32, 16), True, tl.int1)).to(tl.int1)
    e16 = (tl.load(mem2 + bid * 544 + 16 + e14 * 1, (e15 & (e14 >= 0) & (e14 < 512)), other=0)).to(tl.int32)
    e17 = (tl.full((32, 16), 7, tl.int32)).to(tl.int32)
    e18 = ((e16 ^ e17)).to(tl.int32)
    e19 = (tl.full((32, 16), 3, tl.int32)).to(tl.int32)
    e20 = ((e18 < e19)).to(tl.int1)
    e21 = ((((511 - (tl.arange(0, 32)[:, None] * 16 + tl.arange(0, 16)[None, :])) + 166) % 512)).to(tl.int32)
    e22 = (limit).to(tl.int32)
    e23 = ((e21 < e22)).to(tl.int1)
    e24 = ((e20 & e23)).to(tl.int1)
    e25 = (tl.full((32, 16), -0.125, tl.float16)).to(tl.float16)
    e26 = (tl.where(e24, e13, e25)).to(tl.float16)
    e27 = ((((tl.arange(0, 32)[:, None] * 16 + tl.arange(0, 16)[None, :]) + 363) % 512)).to(tl.int32)
    e28 = (limit).to(tl.int32)
    e29 = ((e27 < e28)).to(tl.int1)
    tl.store(mem3 + bid * 1060 + 17 + e27 * 2, e26, (e29 & (e27 >= 0) & (e27 < 512)))
    tl.debug_barrier()
    e30 = (tl.load(mem3 + bid * 1060 + 17 + e27 * 2, (e29 & (e27 >= 0) & (e27 < 512)), other=0)).to(tl.float16)
    e31 = (tl.full((32, 16), 0.125, tl.float16)).to(tl.float16)
    e32 = ((e30 + e31)).to(tl.float16)
    tl.debug_barrier()
    tl.store(mem3 + bid * 1060 + 18 + e27 * 1, e32, (e29 & (e27 >= 0) & (e27 < 512)))
    tl.debug_barrier()
    e33 = (tl.full((32, 16), True, tl.int1)).to(tl.int1)
    e34 = (tl.load(mem3 + bid * 1060 + 17 + e27 * 2, (e33 & (e27 >= 0) & (e27 < 512)), other=0)).to(tl.float16)
    e35 = (tl.full((32, 16), 7, tl.int32)).to(tl.int32)
    e36 = ((e27 ^ e35)).to(tl.int32)
    e37 = (tl.load(mem3 + bid * 1060 + 18 + e36 * 1, (e29 & (e36 >= 0) & (e36 < 512)), other=0)).to(tl.float16)
    e38 = ((e34 + e37)).to(tl.float16)
    e39 = (e38.to(tl.float16)).to(tl.float16)
    e40 = (tl.trans(e39)).to(tl.float16)
    e41 = (tl.full((16, 16), 0.0, tl.float32)).to(tl.float32)
    e42 = (tl.dot(e40, e34, e41, input_precision='ieee')).to(tl.float32)
    e43 = (tl.where(tl.sum((e42 != e42).to(tl.int32), 1) > 0, float('nan'), tl.max(e42.to(tl.float32), 1))).to(tl.float32)
    e44 = (tl.reshape(e43, (16, 1))).to(tl.float32)
    e45 = ((e42 - e44)).to(tl.float32)
    e46 = (tl.full((16, 16), 0.125, tl.float32)).to(tl.float32)
    e47 = ((e45 * e46)).to(tl.float32)
    e48 = ((((tl.arange(0, 16)[:, None] * 16 + tl.arange(0, 16)[None, :]) + 87) % 256)).to(tl.int32)
    e49 = (tl.full((16, 16), True, tl.int1)).to(tl.int1)
    e50 = (tl.load(mem6 + bid * 288 + 16 + e48 * 1, (e49 & (e48 >= 0) & (e48 < 256)), other=0)).to(tl.float16)
    e51 = (e47.to(tl.float16)).to(tl.float16)
    e52 = (tl.dot(e51, e50, e41, input_precision='ieee')).to(tl.float32)
    e53 = (tl.reshape(e52, (16, 16))).to(tl.float32)
    e54 = (tl.reshape(e53, (16, 16))).to(tl.float32)
    e55 = (e54.to(tl.float16)).to(tl.float16)
    e64, e65 = mixed_helper_extended_1_1(e55, e43, mem0, mem1, mem2, mem3, mem6, mem7, mem8, steps, limit, bid)
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
    for e73 in tl.range(0, tl.minimum(tl.maximum(e66, 0), 4), num_stages=2):
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
        e84, e85 = mixed_helper_extended_1_1(e83, e43, mem0, mem1, mem2, mem3, mem6, mem7, mem8, steps, limit, bid)
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
        e106, e107 = mixed_helper_extended_1_1(e105, e43, mem0, mem1, mem2, mem3, mem6, mem7, mem8, steps, limit, bid)
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
    e117 = ((e114 + e50)).to(tl.float16)
    e118 = ((((tl.arange(0, 16)[:, None] * 16 + tl.arange(0, 16)[None, :]) + 33) % 256)).to(tl.int32)
    e119 = (tl.full((16, 16), True, tl.int1)).to(tl.int1)
    e120 = (tl.load(mem8 + bid * 288 + 16 + e118 * 1, (e119 & (e118 >= 0) & (e118 < 256)), other=0)).to(tl.float16)
    e121 = ((e117 * e120)).to(tl.float16)
    e122 = ((e121 - e120)).to(tl.float16)
    tl.store(out_e122 + bid * 288 + 16 + (tl.arange(0, 16)[:, None] * 16 + tl.arange(0, 16)[None, :]), e122)
    tl.store(out_e18 + bid * 544 + 16 + (tl.arange(0, 32)[:, None] * 16 + tl.arange(0, 16)[None, :]), e18)
    tl.store(out_e24 + bid * 544 + 16 + (tl.arange(0, 32)[:, None] * 16 + tl.arange(0, 16)[None, :]), e24)
    tl.store(out_e115 + bid * 33 + 16 + tl.full((), 0, tl.int32), e115)
    tl.store(out_e116 + bid * 33 + 16 + tl.full((), 0, tl.int32), e116)
    tl.store(out_e3 + bid * 544 + 16 + (tl.arange(0, 32)[:, None] * 16 + tl.arange(0, 16)[None, :]), e3)
    tl.store(out_e13 + bid * 544 + 16 + (tl.arange(0, 32)[:, None] * 16 + tl.arange(0, 16)[None, :]), e13)
    tl.store(out_e30 + bid * 544 + 16 + (tl.arange(0, 32)[:, None] * 16 + tl.arange(0, 16)[None, :]), e30)
    tl.store(out_e34 + bid * 544 + 16 + (tl.arange(0, 32)[:, None] * 16 + tl.arange(0, 16)[None, :]), e34)
    tl.store(out_e42 + bid * 288 + 16 + (tl.arange(0, 16)[:, None] * 16 + tl.arange(0, 16)[None, :]), e42)
    tl.store(out_e47 + bid * 288 + 16 + (tl.arange(0, 16)[:, None] * 16 + tl.arange(0, 16)[None, :]), e47)
    tl.store(out_e19 + bid * 544 + 16 + (tl.arange(0, 32)[:, None] * 16 + tl.arange(0, 16)[None, :]), e19)
    tl.store(out_e28 + bid * 33 + 16 + tl.full((), 0, tl.int32), e28)

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
    compiled_0 = compile(ASTSource(extended_0_0, {0: '*fp32', 1: '*fp32', 2: '*i32', 3: '*fp16', 4: '*fp16', 5: '*i32', 6: '*fp16', 7: '*fp16', 8: '*i32', 9: '*i1', 10: '*fp32', 11: '*i32', 12: 'i32', 13: 'i32'}), target=GPUTarget("cuda", arch, 32), options={'num_warps': 4, 'num_stages': 1, 'enable_fp_fusion': False})
    record_extended_compilation('triton_0', compiled_0.asm, {'num_warps': 4, 'num_stages': 1, 'enable_fp_fusion': False})
    def launch_0(memories, outputs, steps, limit):
        arguments = [memories[n] for n in ['mem0', 'mem1', 'mem2', 'mem3', 'mem6', 'mem7', 'mem8']] + [outputs[n] for n in ['e122', 'e18', 'e24', 'e115', 'e116']]
        compiled_0[(2, 1, 1)](*arguments, steps, limit)
    variants.append(('triton_0', ['e122', 'e18', 'e24', 'e115', 'e116'], launch_0))
    extended_stage("compile", 'triton_1')
    compiled_1 = compile(ASTSource(extended_0_1, {0: '*fp32', 1: '*fp32', 2: '*i32', 3: '*fp16', 4: '*fp16', 5: '*i32', 6: '*fp16', 7: '*fp16', 8: '*i32', 9: '*i1', 10: '*fp32', 11: '*i32', 12: '*fp32', 13: '*fp16', 14: '*fp16', 15: '*fp16', 16: '*fp32', 17: '*fp32', 18: '*i32', 19: '*i32', 20: 'i32', 21: 'i32'}), target=GPUTarget("cuda", arch, 32), options={'num_warps': 4, 'num_stages': 1, 'enable_fp_fusion': False})
    record_extended_compilation('triton_1', compiled_1.asm, {'num_warps': 4, 'num_stages': 1, 'enable_fp_fusion': False})
    def launch_1(memories, outputs, steps, limit):
        arguments = [memories[n] for n in ['mem0', 'mem1', 'mem2', 'mem3', 'mem6', 'mem7', 'mem8']] + [outputs[n] for n in ['e122', 'e18', 'e24', 'e115', 'e116', 'e3', 'e13', 'e30', 'e34', 'e42', 'e47', 'e19', 'e28']]
        compiled_1[(2, 1, 1)](*arguments, steps, limit)
    variants.append(('triton_1', ['e122', 'e18', 'e24', 'e115', 'e116', 'e3', 'e13', 'e30', 'e34', 'e42', 'e47', 'e19', 'e28'], launch_1))
    extended_stage("compile", 'triton_2')
    compiled_2 = compile(ASTSource(extended_1_0, {0: '*fp32', 1: '*fp32', 2: '*i32', 3: '*fp16', 4: '*fp16', 5: '*i32', 6: '*fp16', 7: '*fp16', 8: '*i32', 9: '*i1', 10: '*fp32', 11: '*i32', 12: 'i32', 13: 'i32'}), target=GPUTarget("cuda", arch, 32), options={'num_warps': 8, 'num_stages': 2, 'enable_fp_fusion': False})
    record_extended_compilation('triton_2', compiled_2.asm, {'num_warps': 8, 'num_stages': 2, 'enable_fp_fusion': False})
    def launch_2(memories, outputs, steps, limit):
        arguments = [memories[n] for n in ['mem0', 'mem1', 'mem2', 'mem3', 'mem6', 'mem7', 'mem8']] + [outputs[n] for n in ['e122', 'e18', 'e24', 'e115', 'e116']]
        compiled_2[(2, 1, 1)](*arguments, steps, limit)
    variants.append(('triton_2', ['e122', 'e18', 'e24', 'e115', 'e116'], launch_2))
    extended_stage("compile", 'triton_3')
    compiled_3 = compile(ASTSource(extended_1_1, {0: '*fp32', 1: '*fp32', 2: '*i32', 3: '*fp16', 4: '*fp16', 5: '*i32', 6: '*fp16', 7: '*fp16', 8: '*i32', 9: '*i1', 10: '*fp32', 11: '*i32', 12: '*fp32', 13: '*fp16', 14: '*fp16', 15: '*fp16', 16: '*fp32', 17: '*fp32', 18: '*i32', 19: '*i32', 20: 'i32', 21: 'i32'}), target=GPUTarget("cuda", arch, 32), options={'num_warps': 8, 'num_stages': 2, 'enable_fp_fusion': False})
    record_extended_compilation('triton_3', compiled_3.asm, {'num_warps': 8, 'num_stages': 2, 'enable_fp_fusion': False})
    def launch_3(memories, outputs, steps, limit):
        arguments = [memories[n] for n in ['mem0', 'mem1', 'mem2', 'mem3', 'mem6', 'mem7', 'mem8']] + [outputs[n] for n in ['e122', 'e18', 'e24', 'e115', 'e116', 'e3', 'e13', 'e30', 'e34', 'e42', 'e47', 'e19', 'e28']]
        compiled_3[(2, 1, 1)](*arguments, steps, limit)
    variants.append(('triton_3', ['e122', 'e18', 'e24', 'e115', 'e116', 'e3', 'e13', 'e30', 'e34', 'e42', 'e47', 'e19', 'e28'], launch_3))
    return variants

PROGRAM = {'type': 'extended', 'version': 1, 'body': {'arguments': [], 'operations': [{'op': 'index', 'results': [{'name': 'e1', 'type': {'dtype': 'int32', 'shape': (32, 16)}}], 'operands': [], 'attrs': {'shift': 44, 'reverse': True}, 'regions': []}, {'op': 'constant', 'results': [{'name': 'e2', 'type': {'dtype': 'bool', 'shape': (32, 16)}}], 'operands': [], 'attrs': {'value': True}, 'regions': []}, {'op': 'load', 'results': [{'name': 'e3', 'type': {'dtype': 'float32', 'shape': (32, 16)}}], 'operands': ['e1', 'e2'], 'attrs': {'buffer': 'mem0'}, 'regions': []}, {'op': 'add', 'results': [{'name': 'e4', 'type': {'dtype': 'float32', 'shape': (32, 16)}}], 'operands': ['e3', 'e3'], 'attrs': {}, 'regions': []}, {'op': 'index', 'results': [{'name': 'e5', 'type': {'dtype': 'int32', 'shape': (32, 16)}}], 'operands': [], 'attrs': {'shift': 370, 'reverse': False}, 'regions': []}, {'op': 'constant', 'results': [{'name': 'e6', 'type': {'dtype': 'bool', 'shape': (32, 16)}}], 'operands': [], 'attrs': {'value': True}, 'regions': []}, {'op': 'load', 'results': [{'name': 'e7', 'type': {'dtype': 'float32', 'shape': (32, 16)}}], 'operands': ['e5', 'e6'], 'attrs': {'buffer': 'mem1'}, 'regions': []}, {'op': 'mul', 'results': [{'name': 'e8', 'type': {'dtype': 'float32', 'shape': (32, 16)}}], 'operands': ['e4', 'e7'], 'attrs': {}, 'regions': []}, {'op': 'cast', 'results': [{'name': 'e9', 'type': {'dtype': 'float16', 'shape': (32, 16)}}], 'operands': ['e8'], 'attrs': {}, 'regions': []}, {'op': 'add', 'results': [{'name': 'e10', 'type': {'dtype': 'float16', 'shape': (32, 16)}}], 'operands': ['e9', 'e9'], 'attrs': {}, 'regions': []}, {'op': 'cast', 'results': [{'name': 'e11', 'type': {'dtype': 'float16', 'shape': (32, 16)}}], 'operands': ['e10'], 'attrs': {}, 'regions': []}, {'op': 'constant', 'results': [{'name': 'e12', 'type': {'dtype': 'float16', 'shape': (32, 16)}}], 'operands': [], 'attrs': {'value': 0.5}, 'regions': []}, {'op': 'mul', 'results': [{'name': 'e13', 'type': {'dtype': 'float16', 'shape': (32, 16)}}], 'operands': ['e11', 'e12'], 'attrs': {}, 'regions': []}, {'op': 'index', 'results': [{'name': 'e14', 'type': {'dtype': 'int32', 'shape': (32, 16)}}], 'operands': [], 'attrs': {'shift': 389, 'reverse': True}, 'regions': []}, {'op': 'constant', 'results': [{'name': 'e15', 'type': {'dtype': 'bool', 'shape': (32, 16)}}], 'operands': [], 'attrs': {'value': True}, 'regions': []}, {'op': 'load', 'results': [{'name': 'e16', 'type': {'dtype': 'int32', 'shape': (32, 16)}}], 'operands': ['e14', 'e15'], 'attrs': {'buffer': 'mem2'}, 'regions': []}, {'op': 'constant', 'results': [{'name': 'e17', 'type': {'dtype': 'int32', 'shape': (32, 16)}}], 'operands': [], 'attrs': {'value': 7}, 'regions': []}, {'op': 'bitxor', 'results': [{'name': 'e18', 'type': {'dtype': 'int32', 'shape': (32, 16)}}], 'operands': ['e16', 'e17'], 'attrs': {}, 'regions': []}, {'op': 'constant', 'results': [{'name': 'e19', 'type': {'dtype': 'int32', 'shape': (32, 16)}}], 'operands': [], 'attrs': {'value': 3}, 'regions': []}, {'op': 'lt', 'results': [{'name': 'e20', 'type': {'dtype': 'bool', 'shape': (32, 16)}}], 'operands': ['e18', 'e19'], 'attrs': {}, 'regions': []}, {'op': 'index', 'results': [{'name': 'e21', 'type': {'dtype': 'int32', 'shape': (32, 16)}}], 'operands': [], 'attrs': {'shift': 166, 'reverse': True}, 'regions': []}, {'op': 'parameter', 'results': [{'name': 'e22', 'type': {'dtype': 'int32', 'shape': ()}}], 'operands': [], 'attrs': {'name': 'limit'}, 'regions': []}, {'op': 'lt', 'results': [{'name': 'e23', 'type': {'dtype': 'bool', 'shape': (32, 16)}}], 'operands': ['e21', 'e22'], 'attrs': {}, 'regions': []}, {'op': 'and', 'results': [{'name': 'e24', 'type': {'dtype': 'bool', 'shape': (32, 16)}}], 'operands': ['e20', 'e23'], 'attrs': {}, 'regions': []}, {'op': 'constant', 'results': [{'name': 'e25', 'type': {'dtype': 'float16', 'shape': (32, 16)}}], 'operands': [], 'attrs': {'value': -0.125}, 'regions': []}, {'op': 'select', 'results': [{'name': 'e26', 'type': {'dtype': 'float16', 'shape': (32, 16)}}], 'operands': ['e24', 'e13', 'e25'], 'attrs': {}, 'regions': []}, {'op': 'index', 'results': [{'name': 'e27', 'type': {'dtype': 'int32', 'shape': (32, 16)}}], 'operands': [], 'attrs': {'shift': 363, 'reverse': False}, 'regions': []}, {'op': 'parameter', 'results': [{'name': 'e28', 'type': {'dtype': 'int32', 'shape': ()}}], 'operands': [], 'attrs': {'name': 'limit'}, 'regions': []}, {'op': 'lt', 'results': [{'name': 'e29', 'type': {'dtype': 'bool', 'shape': (32, 16)}}], 'operands': ['e27', 'e28'], 'attrs': {}, 'regions': []}, {'op': 'store', 'results': [], 'operands': ['e27', 'e29', 'e26'], 'attrs': {'buffer': 'mem4'}, 'regions': []}, {'op': 'barrier', 'results': [], 'operands': [], 'attrs': {}, 'regions': []}, {'op': 'load', 'results': [{'name': 'e30', 'type': {'dtype': 'float16', 'shape': (32, 16)}}], 'operands': ['e27', 'e29'], 'attrs': {'buffer': 'mem4'}, 'regions': []}, {'op': 'constant', 'results': [{'name': 'e31', 'type': {'dtype': 'float16', 'shape': (32, 16)}}], 'operands': [], 'attrs': {'value': 0.125}, 'regions': []}, {'op': 'add', 'results': [{'name': 'e32', 'type': {'dtype': 'float16', 'shape': (32, 16)}}], 'operands': ['e30', 'e31'], 'attrs': {}, 'regions': []}, {'op': 'barrier', 'results': [], 'operands': [], 'attrs': {}, 'regions': []}, {'op': 'store', 'results': [], 'operands': ['e27', 'e29', 'e32'], 'attrs': {'buffer': 'mem5'}, 'regions': []}, {'op': 'barrier', 'results': [], 'operands': [], 'attrs': {}, 'regions': []}, {'op': 'constant', 'results': [{'name': 'e33', 'type': {'dtype': 'bool', 'shape': (32, 16)}}], 'operands': [], 'attrs': {'value': True}, 'regions': []}, {'op': 'load', 'results': [{'name': 'e34', 'type': {'dtype': 'float16', 'shape': (32, 16)}}], 'operands': ['e27', 'e33'], 'attrs': {'buffer': 'mem4'}, 'regions': []}, {'op': 'constant', 'results': [{'name': 'e35', 'type': {'dtype': 'int32', 'shape': (32, 16)}}], 'operands': [], 'attrs': {'value': 7}, 'regions': []}, {'op': 'bitxor', 'results': [{'name': 'e36', 'type': {'dtype': 'int32', 'shape': (32, 16)}}], 'operands': ['e27', 'e35'], 'attrs': {}, 'regions': []}, {'op': 'load', 'results': [{'name': 'e37', 'type': {'dtype': 'float16', 'shape': (32, 16)}}], 'operands': ['e36', 'e29'], 'attrs': {'buffer': 'mem5'}, 'regions': []}, {'op': 'add', 'results': [{'name': 'e38', 'type': {'dtype': 'float16', 'shape': (32, 16)}}], 'operands': ['e34', 'e37'], 'attrs': {}, 'regions': []}, {'op': 'cast', 'results': [{'name': 'e39', 'type': {'dtype': 'float16', 'shape': (32, 16)}}], 'operands': ['e38'], 'attrs': {}, 'regions': []}, {'op': 'transpose', 'results': [{'name': 'e40', 'type': {'dtype': 'float16', 'shape': (16, 32)}}], 'operands': ['e39'], 'attrs': {}, 'regions': []}, {'op': 'constant', 'results': [{'name': 'e41', 'type': {'dtype': 'float32', 'shape': (16, 16)}}], 'operands': [], 'attrs': {'value': 0.0}, 'regions': []}, {'op': 'matmul', 'results': [{'name': 'e42', 'type': {'dtype': 'float32', 'shape': (16, 16)}}], 'operands': ['e40', 'e34', 'e41'], 'attrs': {}, 'regions': []}, {'op': 'reduce', 'results': [{'name': 'e43', 'type': {'dtype': 'float32', 'shape': (16,)}}], 'operands': ['e42'], 'attrs': {'axis': 1, 'kind': 'max'}, 'regions': []}, {'op': 'reshape', 'results': [{'name': 'e44', 'type': {'dtype': 'float32', 'shape': (16, 1)}}], 'operands': ['e43'], 'attrs': {}, 'regions': []}, {'op': 'sub', 'results': [{'name': 'e45', 'type': {'dtype': 'float32', 'shape': (16, 16)}}], 'operands': ['e42', 'e44'], 'attrs': {}, 'regions': []}, {'op': 'constant', 'results': [{'name': 'e46', 'type': {'dtype': 'float32', 'shape': (16, 16)}}], 'operands': [], 'attrs': {'value': 0.125}, 'regions': []}, {'op': 'mul', 'results': [{'name': 'e47', 'type': {'dtype': 'float32', 'shape': (16, 16)}}], 'operands': ['e45', 'e46'], 'attrs': {}, 'regions': []}, {'op': 'index', 'results': [{'name': 'e48', 'type': {'dtype': 'int32', 'shape': (16, 16)}}], 'operands': [], 'attrs': {'shift': 87, 'reverse': False}, 'regions': []}, {'op': 'constant', 'results': [{'name': 'e49', 'type': {'dtype': 'bool', 'shape': (16, 16)}}], 'operands': [], 'attrs': {'value': True}, 'regions': []}, {'op': 'load', 'results': [{'name': 'e50', 'type': {'dtype': 'float16', 'shape': (16, 16)}}], 'operands': ['e48', 'e49'], 'attrs': {'buffer': 'mem6'}, 'regions': []}, {'op': 'cast', 'results': [{'name': 'e51', 'type': {'dtype': 'float16', 'shape': (16, 16)}}], 'operands': ['e47'], 'attrs': {}, 'regions': []}, {'op': 'matmul', 'results': [{'name': 'e52', 'type': {'dtype': 'float32', 'shape': (16, 16)}}], 'operands': ['e51', 'e50', 'e41'], 'attrs': {}, 'regions': []}, {'op': 'reshape', 'results': [{'name': 'e53', 'type': {'dtype': 'float32', 'shape': (16, 16)}}], 'operands': ['e52'], 'attrs': {}, 'regions': []}, {'op': 'reshape', 'results': [{'name': 'e54', 'type': {'dtype': 'float32', 'shape': (16, 16)}}], 'operands': ['e53'], 'attrs': {}, 'regions': []}, {'op': 'cast', 'results': [{'name': 'e55', 'type': {'dtype': 'float16', 'shape': (16, 16)}}], 'operands': ['e54'], 'attrs': {}, 'regions': []}, {'op': 'call', 'results': [{'name': 'e64', 'type': {'dtype': 'float16', 'shape': (16, 16)}}, {'name': 'e65', 'type': {'dtype': 'float32', 'shape': ()}}], 'operands': ['e55', 'e43'], 'attrs': {'callee': 'mixed_helper'}, 'regions': []}, {'op': 'parameter', 'results': [{'name': 'e66', 'type': {'dtype': 'int32', 'shape': ()}}], 'operands': [], 'attrs': {'name': 'steps'}, 'regions': []}, {'op': 'constant', 'results': [{'name': 'e67', 'type': {'dtype': 'int32', 'shape': ()}}], 'operands': [], 'attrs': {'value': 0}, 'regions': []}, {'op': 'constant', 'results': [{'name': 'e68', 'type': {'dtype': 'int32', 'shape': ()}}], 'operands': [], 'attrs': {'value': 0}, 'regions': []}, {'op': 'constant', 'results': [{'name': 'e69', 'type': {'dtype': 'bool', 'shape': ()}}], 'operands': [], 'attrs': {'value': True}, 'regions': []}, {'op': 'load', 'results': [{'name': 'e70', 'type': {'dtype': 'int32', 'shape': ()}}], 'operands': ['e68', 'e69'], 'attrs': {'buffer': 'mem7'}, 'regions': []}, {'op': 'constant', 'results': [{'name': 'e71', 'type': {'dtype': 'int32', 'shape': ()}}], 'operands': [], 'attrs': {'value': 0}, 'regions': []}, {'op': 'lt', 'results': [{'name': 'e72', 'type': {'dtype': 'bool', 'shape': ()}}], 'operands': ['e70', 'e71'], 'attrs': {}, 'regions': []}, {'op': 'for', 'results': [{'name': 'e92', 'type': {'dtype': 'float16', 'shape': (16, 16)}}, {'name': 'e93', 'type': {'dtype': 'float32', 'shape': ()}}, {'name': 'e94', 'type': {'dtype': 'int32', 'shape': ()}}], 'operands': ['e66', 'e64', 'e65', 'e67'], 'attrs': {'max_steps': 4, 'pipelined': True}, 'regions': [{'arguments': [{'name': 'e73', 'type': {'dtype': 'int32', 'shape': ()}}, {'name': 'e74', 'type': {'dtype': 'float16', 'shape': (16, 16)}}, {'name': 'e75', 'type': {'dtype': 'float32', 'shape': ()}}, {'name': 'e76', 'type': {'dtype': 'int32', 'shape': ()}}], 'operations': [{'op': 'if', 'results': [{'name': 'e83', 'type': {'dtype': 'float16', 'shape': (16, 16)}}], 'operands': ['e72', 'e74'], 'attrs': {}, 'regions': [{'arguments': [{'name': 'e77', 'type': {'dtype': 'float16', 'shape': (16, 16)}}], 'operations': [{'op': 'constant', 'results': [{'name': 'e78', 'type': {'dtype': 'float16', 'shape': (16, 16)}}], 'operands': [], 'attrs': {'value': 0.125}, 'regions': []}, {'op': 'add', 'results': [{'name': 'e79', 'type': {'dtype': 'float16', 'shape': (16, 16)}}], 'operands': ['e77', 'e78'], 'attrs': {}, 'regions': []}], 'returns': ['e79']}, {'arguments': [{'name': 'e80', 'type': {'dtype': 'float16', 'shape': (16, 16)}}], 'operations': [{'op': 'constant', 'results': [{'name': 'e81', 'type': {'dtype': 'float16', 'shape': (16, 16)}}], 'operands': [], 'attrs': {'value': 0.125}, 'regions': []}, {'op': 'sub', 'results': [{'name': 'e82', 'type': {'dtype': 'float16', 'shape': (16, 16)}}], 'operands': ['e80', 'e81'], 'attrs': {}, 'regions': []}], 'returns': ['e82']}]}, {'op': 'call', 'results': [{'name': 'e84', 'type': {'dtype': 'float16', 'shape': (16, 16)}}, {'name': 'e85', 'type': {'dtype': 'float32', 'shape': ()}}], 'operands': ['e83', 'e43'], 'attrs': {'callee': 'mixed_helper'}, 'regions': []}, {'op': 'constant', 'results': [{'name': 'e86', 'type': {'dtype': 'float16', 'shape': (16, 16)}}], 'operands': [], 'attrs': {'value': 0.125}, 'regions': []}, {'op': 'mul', 'results': [{'name': 'e87', 'type': {'dtype': 'float16', 'shape': (16, 16)}}], 'operands': ['e84', 'e86'], 'attrs': {}, 'regions': []}, {'op': 'add', 'results': [{'name': 'e88', 'type': {'dtype': 'float32', 'shape': ()}}], 'operands': ['e75', 'e85'], 'attrs': {}, 'regions': []}, {'op': 'constant', 'results': [{'name': 'e89', 'type': {'dtype': 'int32', 'shape': ()}}], 'operands': [], 'attrs': {'value': 1}, 'regions': []}, {'op': 'add', 'results': [{'name': 'e90', 'type': {'dtype': 'int32', 'shape': ()}}], 'operands': ['e73', 'e89'], 'attrs': {}, 'regions': []}, {'op': 'add', 'results': [{'name': 'e91', 'type': {'dtype': 'int32', 'shape': ()}}], 'operands': ['e76', 'e90'], 'attrs': {}, 'regions': []}], 'returns': ['e87', 'e88', 'e91']}]}, {'op': 'while', 'results': [{'name': 'e114', 'type': {'dtype': 'float16', 'shape': (16, 16)}}, {'name': 'e115', 'type': {'dtype': 'float32', 'shape': ()}}, {'name': 'e116', 'type': {'dtype': 'int32', 'shape': ()}}], 'operands': ['e66', 'e92', 'e93', 'e94'], 'attrs': {'max_steps': 4, 'pipelined': False}, 'regions': [{'arguments': [{'name': 'e95', 'type': {'dtype': 'int32', 'shape': ()}}, {'name': 'e96', 'type': {'dtype': 'float16', 'shape': (16, 16)}}, {'name': 'e97', 'type': {'dtype': 'float32', 'shape': ()}}, {'name': 'e98', 'type': {'dtype': 'int32', 'shape': ()}}], 'operations': [{'op': 'if', 'results': [{'name': 'e105', 'type': {'dtype': 'float16', 'shape': (16, 16)}}], 'operands': ['e72', 'e96'], 'attrs': {}, 'regions': [{'arguments': [{'name': 'e99', 'type': {'dtype': 'float16', 'shape': (16, 16)}}], 'operations': [{'op': 'constant', 'results': [{'name': 'e100', 'type': {'dtype': 'float16', 'shape': (16, 16)}}], 'operands': [], 'attrs': {'value': 0.125}, 'regions': []}, {'op': 'add', 'results': [{'name': 'e101', 'type': {'dtype': 'float16', 'shape': (16, 16)}}], 'operands': ['e99', 'e100'], 'attrs': {}, 'regions': []}], 'returns': ['e101']}, {'arguments': [{'name': 'e102', 'type': {'dtype': 'float16', 'shape': (16, 16)}}], 'operations': [{'op': 'constant', 'results': [{'name': 'e103', 'type': {'dtype': 'float16', 'shape': (16, 16)}}], 'operands': [], 'attrs': {'value': 0.125}, 'regions': []}, {'op': 'sub', 'results': [{'name': 'e104', 'type': {'dtype': 'float16', 'shape': (16, 16)}}], 'operands': ['e102', 'e103'], 'attrs': {}, 'regions': []}], 'returns': ['e104']}]}, {'op': 'call', 'results': [{'name': 'e106', 'type': {'dtype': 'float16', 'shape': (16, 16)}}, {'name': 'e107', 'type': {'dtype': 'float32', 'shape': ()}}], 'operands': ['e105', 'e43'], 'attrs': {'callee': 'mixed_helper'}, 'regions': []}, {'op': 'constant', 'results': [{'name': 'e108', 'type': {'dtype': 'float16', 'shape': (16, 16)}}], 'operands': [], 'attrs': {'value': 0.125}, 'regions': []}, {'op': 'mul', 'results': [{'name': 'e109', 'type': {'dtype': 'float16', 'shape': (16, 16)}}], 'operands': ['e106', 'e108'], 'attrs': {}, 'regions': []}, {'op': 'add', 'results': [{'name': 'e110', 'type': {'dtype': 'float32', 'shape': ()}}], 'operands': ['e97', 'e107'], 'attrs': {}, 'regions': []}, {'op': 'constant', 'results': [{'name': 'e111', 'type': {'dtype': 'int32', 'shape': ()}}], 'operands': [], 'attrs': {'value': 1}, 'regions': []}, {'op': 'add', 'results': [{'name': 'e112', 'type': {'dtype': 'int32', 'shape': ()}}], 'operands': ['e95', 'e111'], 'attrs': {}, 'regions': []}, {'op': 'add', 'results': [{'name': 'e113', 'type': {'dtype': 'int32', 'shape': ()}}], 'operands': ['e98', 'e112'], 'attrs': {}, 'regions': []}], 'returns': ['e109', 'e110', 'e113']}]}, {'op': 'add', 'results': [{'name': 'e117', 'type': {'dtype': 'float16', 'shape': (16, 16)}}], 'operands': ['e114', 'e50'], 'attrs': {}, 'regions': []}, {'op': 'index', 'results': [{'name': 'e118', 'type': {'dtype': 'int32', 'shape': (16, 16)}}], 'operands': [], 'attrs': {'shift': 33, 'reverse': False}, 'regions': []}, {'op': 'constant', 'results': [{'name': 'e119', 'type': {'dtype': 'bool', 'shape': (16, 16)}}], 'operands': [], 'attrs': {'value': True}, 'regions': []}, {'op': 'load', 'results': [{'name': 'e120', 'type': {'dtype': 'float16', 'shape': (16, 16)}}], 'operands': ['e118', 'e119'], 'attrs': {'buffer': 'mem8'}, 'regions': []}, {'op': 'mul', 'results': [{'name': 'e121', 'type': {'dtype': 'float16', 'shape': (16, 16)}}], 'operands': ['e117', 'e120'], 'attrs': {}, 'regions': []}, {'op': 'sub', 'results': [{'name': 'e122', 'type': {'dtype': 'float16', 'shape': (16, 16)}}], 'operands': ['e121', 'e120'], 'attrs': {}, 'regions': []}], 'returns': ['e122', 'e18', 'e24', 'e115', 'e116']}, 'buffers': [{'name': 'mem0', 'dtype': 'float32', 'size': 512, 'role': 'input', 'base': None, 'offset': 0, 'stride': 1}, {'name': 'mem1', 'dtype': 'float32', 'size': 512, 'role': 'input', 'base': None, 'offset': 0, 'stride': 1}, {'name': 'mem2', 'dtype': 'int32', 'size': 512, 'role': 'input', 'base': None, 'offset': 0, 'stride': 1}, {'name': 'mem3', 'dtype': 'float16', 'size': 1028, 'role': 'scratch', 'base': None, 'offset': 0, 'stride': 1}, {'name': 'mem4', 'dtype': 'float16', 'size': 512, 'role': 'scratch', 'base': 'mem3', 'offset': 1, 'stride': 2}, {'name': 'mem5', 'dtype': 'float16', 'size': 512, 'role': 'scratch', 'base': 'mem3', 'offset': 2, 'stride': 1}, {'name': 'mem6', 'dtype': 'float16', 'size': 256, 'role': 'input', 'base': None, 'offset': 0, 'stride': 1}, {'name': 'mem7', 'dtype': 'int32', 'size': 1, 'role': 'input', 'base': None, 'offset': 0, 'stride': 1}, {'name': 'mem8', 'dtype': 'float16', 'size': 256, 'role': 'input', 'base': None, 'offset': 0, 'stride': 1}], 'functions': [{'name': 'mixed_helper', 'body': {'arguments': [{'name': 'e56', 'type': {'dtype': 'float16', 'shape': (16, 16)}}, {'name': 'e57', 'type': {'dtype': 'float32', 'shape': (16,)}}], 'operations': [{'op': 'reduce', 'results': [{'name': 'e58', 'type': {'dtype': 'float32', 'shape': (16,)}}], 'operands': ['e56'], 'attrs': {'axis': 1, 'kind': 'sum'}, 'regions': []}, {'op': 'add', 'results': [{'name': 'e59', 'type': {'dtype': 'float32', 'shape': (16,)}}], 'operands': ['e58', 'e57'], 'attrs': {}, 'regions': []}, {'op': 'reduce', 'results': [{'name': 'e60', 'type': {'dtype': 'float32', 'shape': ()}}], 'operands': ['e59'], 'attrs': {'axis': 0, 'kind': 'sum'}, 'regions': []}, {'op': 'reshape', 'results': [{'name': 'e61', 'type': {'dtype': 'float32', 'shape': (16, 1)}}], 'operands': ['e59'], 'attrs': {}, 'regions': []}, {'op': 'cast', 'results': [{'name': 'e62', 'type': {'dtype': 'float16', 'shape': (16, 1)}}], 'operands': ['e61'], 'attrs': {}, 'regions': []}, {'op': 'add', 'results': [{'name': 'e63', 'type': {'dtype': 'float16', 'shape': (16, 16)}}], 'operands': ['e56', 'e62'], 'attrs': {}, 'regions': []}], 'returns': ['e63', 'e60']}}], 'observations': ['e3', 'e13', 'e30', 'e34', 'e42', 'e47', 'e19', 'e28'], 'blocks': 2, 'input_pattern': 'integer', 'runtime_cases': [(0, 1), (1, 511), (4, 512)], 'configuration_pair': True, 'observation_pair': True, 'family': 'mixed'}

if __name__ == '__main__':
    if os.environ.get('TILESMITH_COMPILE_ONLY', '0') == '1':
        prepare_extended(device_compile=False)
        extended_stage('complete', 'compile_only')
        print('COMPILE PASSED (no GPU execution)')
    else:
        run_extended(PROGRAM, prepare_extended, 0, 3)
        extended_stage('complete', 'execute')
        print('ALL PASSED')
