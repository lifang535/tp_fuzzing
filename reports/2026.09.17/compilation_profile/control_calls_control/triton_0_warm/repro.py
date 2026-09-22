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
def mixed_helper_extended_0_0(e6, e7, mem0, mem1, mem2, steps, limit, bid):
    e8 = (tl.sum(e6.to(tl.float32), 1)).to(tl.float32)
    e9 = ((e8 + e7)).to(tl.float32)
    e10 = (tl.sum(e9.to(tl.float32), 0)).to(tl.float32)
    e11 = (tl.reshape(e9, (32, 1))).to(tl.float32)
    e12 = (e11.to(tl.float16)).to(tl.float16)
    e13 = ((e6 + e12)).to(tl.float16)
    return e13, e10
@triton.jit
def extended_0_0(mem0, mem1, mem2, out_e71, out_e68, out_e69, out_e72, steps, limit):
    bid = tl.program_id(0)
    e1 = ((((511 - (tl.arange(0, 32)[:, None] * 16 + tl.arange(0, 16)[None, :])) + 25) % 512)).to(tl.int32)
    e2 = (tl.full((32, 16), True, tl.int1)).to(tl.int1)
    e3 = (tl.load(mem0 + bid * 544 + 16 + e1 * 1, (e2 & (e1 >= 0) & (e1 < 512)), other=0)).to(tl.float16)
    e4 = ((e3 * e3)).to(tl.float16)
    e5 = (e4.to(tl.float16)).to(tl.float16)
    e14 = (((tl.arange(0, 32) + 27) % 32)).to(tl.int32)
    e15 = (tl.full((32,), True, tl.int1)).to(tl.int1)
    e16 = (tl.load(mem1 + bid * 64 + 16 + e14 * 1, (e15 & (e14 >= 0) & (e14 < 32)), other=0)).to(tl.float32)
    e17, e18 = mixed_helper_extended_0_0(e5, e16, mem0, mem1, mem2, steps, limit, bid)
    e19 = (steps).to(tl.int32)
    e20 = (tl.full((), 0, tl.int32)).to(tl.int32)
    e21 = (tl.full((), 0, tl.int32)).to(tl.int32)
    e22 = (tl.full((), True, tl.int1)).to(tl.int1)
    e23 = (tl.load(mem2 + bid * 33 + 16 + e21 * 1, (e22 & (e21 >= 0) & (e21 < 1)), other=0)).to(tl.int32)
    e24 = (tl.full((), 0, tl.int32)).to(tl.int32)
    e25 = ((e23 < e24)).to(tl.int1)
    e45 = e17
    e46 = e18
    e47 = e20
    for e26 in tl.range(0, tl.minimum(tl.maximum(e19, 0), 4), num_stages=1):
        e27 = e45
        e28 = e46
        e29 = e47
        if e25:
            e30 = e27
            e31 = (tl.full((32, 16), 0.125, tl.float16)).to(tl.float16)
            e32 = ((e30 + e31)).to(tl.float16)
            e36 = e32
        else:
            e33 = e27
            e34 = (tl.full((32, 16), 0.125, tl.float16)).to(tl.float16)
            e35 = ((e33 - e34)).to(tl.float16)
            e36 = e35
        e37, e38 = mixed_helper_extended_0_0(e36, e16, mem0, mem1, mem2, steps, limit, bid)
        e39 = (tl.full((32, 16), 0.125, tl.float16)).to(tl.float16)
        e40 = ((e37 * e39)).to(tl.float16)
        e41 = ((e28 + e38)).to(tl.float32)
        e42 = (tl.full((), 1, tl.int32)).to(tl.int32)
        e43 = ((e26 + e42)).to(tl.int32)
        e44 = ((e29 + e43)).to(tl.int32)
        e45 = e40
        e46 = e41
        e47 = e44
    e67 = e45
    e68 = e46
    e69 = e47
    e48 = tl.full((), 0, tl.int32)
    while e48 < tl.minimum(tl.maximum(e19, 0), 4):
        e49 = e67
        e50 = e68
        e51 = e69
        if e25:
            e52 = e49
            e53 = (tl.full((32, 16), 0.125, tl.float16)).to(tl.float16)
            e54 = ((e52 + e53)).to(tl.float16)
            e58 = e54
        else:
            e55 = e49
            e56 = (tl.full((32, 16), 0.125, tl.float16)).to(tl.float16)
            e57 = ((e55 - e56)).to(tl.float16)
            e58 = e57
        e59, e60 = mixed_helper_extended_0_0(e58, e16, mem0, mem1, mem2, steps, limit, bid)
        e61 = (tl.full((32, 16), 0.125, tl.float16)).to(tl.float16)
        e62 = ((e59 * e61)).to(tl.float16)
        e63 = ((e50 + e60)).to(tl.float32)
        e64 = (tl.full((), 1, tl.int32)).to(tl.int32)
        e65 = ((e48 + e64)).to(tl.int32)
        e66 = ((e51 + e65)).to(tl.int32)
        e67 = e62
        e68 = e63
        e69 = e66
        e48 += 1
    e70 = ((e67 * e45)).to(tl.float16)
    e71 = (e70.to(tl.float32)).to(tl.float32)
    e72 = (tl.sum(e71.to(tl.float32), 1)).to(tl.float32)
    tl.store(out_e71 + bid * 544 + 16 + (tl.arange(0, 32)[:, None] * 16 + tl.arange(0, 16)[None, :]), e71)
    tl.store(out_e68 + bid * 33 + 16 + tl.full((), 0, tl.int32), e68)
    tl.store(out_e69 + bid * 33 + 16 + tl.full((), 0, tl.int32), e69)
    tl.store(out_e72 + bid * 64 + 16 + tl.arange(0, 32), e72)

@triton.jit
def mixed_helper_extended_0_1(e6, e7, mem0, mem1, mem2, steps, limit, bid):
    e8 = (tl.sum(e6.to(tl.float32), 1)).to(tl.float32)
    e9 = ((e8 + e7)).to(tl.float32)
    e10 = (tl.sum(e9.to(tl.float32), 0)).to(tl.float32)
    e11 = (tl.reshape(e9, (32, 1))).to(tl.float32)
    e12 = (e11.to(tl.float16)).to(tl.float16)
    e13 = ((e6 + e12)).to(tl.float16)
    return e13, e10
@triton.jit
def extended_0_1(mem0, mem1, mem2, out_e71, out_e68, out_e69, out_e72, out_e3, out_e22, out_e24, steps, limit):
    bid = tl.program_id(0)
    e1 = ((((511 - (tl.arange(0, 32)[:, None] * 16 + tl.arange(0, 16)[None, :])) + 25) % 512)).to(tl.int32)
    e2 = (tl.full((32, 16), True, tl.int1)).to(tl.int1)
    e3 = (tl.load(mem0 + bid * 544 + 16 + e1 * 1, (e2 & (e1 >= 0) & (e1 < 512)), other=0)).to(tl.float16)
    e4 = ((e3 * e3)).to(tl.float16)
    e5 = (e4.to(tl.float16)).to(tl.float16)
    e14 = (((tl.arange(0, 32) + 27) % 32)).to(tl.int32)
    e15 = (tl.full((32,), True, tl.int1)).to(tl.int1)
    e16 = (tl.load(mem1 + bid * 64 + 16 + e14 * 1, (e15 & (e14 >= 0) & (e14 < 32)), other=0)).to(tl.float32)
    e17, e18 = mixed_helper_extended_0_1(e5, e16, mem0, mem1, mem2, steps, limit, bid)
    e19 = (steps).to(tl.int32)
    e20 = (tl.full((), 0, tl.int32)).to(tl.int32)
    e21 = (tl.full((), 0, tl.int32)).to(tl.int32)
    e22 = (tl.full((), True, tl.int1)).to(tl.int1)
    e23 = (tl.load(mem2 + bid * 33 + 16 + e21 * 1, (e22 & (e21 >= 0) & (e21 < 1)), other=0)).to(tl.int32)
    e24 = (tl.full((), 0, tl.int32)).to(tl.int32)
    e25 = ((e23 < e24)).to(tl.int1)
    e45 = e17
    e46 = e18
    e47 = e20
    for e26 in tl.range(0, tl.minimum(tl.maximum(e19, 0), 4), num_stages=1):
        e27 = e45
        e28 = e46
        e29 = e47
        if e25:
            e30 = e27
            e31 = (tl.full((32, 16), 0.125, tl.float16)).to(tl.float16)
            e32 = ((e30 + e31)).to(tl.float16)
            e36 = e32
        else:
            e33 = e27
            e34 = (tl.full((32, 16), 0.125, tl.float16)).to(tl.float16)
            e35 = ((e33 - e34)).to(tl.float16)
            e36 = e35
        e37, e38 = mixed_helper_extended_0_1(e36, e16, mem0, mem1, mem2, steps, limit, bid)
        e39 = (tl.full((32, 16), 0.125, tl.float16)).to(tl.float16)
        e40 = ((e37 * e39)).to(tl.float16)
        e41 = ((e28 + e38)).to(tl.float32)
        e42 = (tl.full((), 1, tl.int32)).to(tl.int32)
        e43 = ((e26 + e42)).to(tl.int32)
        e44 = ((e29 + e43)).to(tl.int32)
        e45 = e40
        e46 = e41
        e47 = e44
    e67 = e45
    e68 = e46
    e69 = e47
    e48 = tl.full((), 0, tl.int32)
    while e48 < tl.minimum(tl.maximum(e19, 0), 4):
        e49 = e67
        e50 = e68
        e51 = e69
        if e25:
            e52 = e49
            e53 = (tl.full((32, 16), 0.125, tl.float16)).to(tl.float16)
            e54 = ((e52 + e53)).to(tl.float16)
            e58 = e54
        else:
            e55 = e49
            e56 = (tl.full((32, 16), 0.125, tl.float16)).to(tl.float16)
            e57 = ((e55 - e56)).to(tl.float16)
            e58 = e57
        e59, e60 = mixed_helper_extended_0_1(e58, e16, mem0, mem1, mem2, steps, limit, bid)
        e61 = (tl.full((32, 16), 0.125, tl.float16)).to(tl.float16)
        e62 = ((e59 * e61)).to(tl.float16)
        e63 = ((e50 + e60)).to(tl.float32)
        e64 = (tl.full((), 1, tl.int32)).to(tl.int32)
        e65 = ((e48 + e64)).to(tl.int32)
        e66 = ((e51 + e65)).to(tl.int32)
        e67 = e62
        e68 = e63
        e69 = e66
        e48 += 1
    e70 = ((e67 * e45)).to(tl.float16)
    e71 = (e70.to(tl.float32)).to(tl.float32)
    e72 = (tl.sum(e71.to(tl.float32), 1)).to(tl.float32)
    tl.store(out_e71 + bid * 544 + 16 + (tl.arange(0, 32)[:, None] * 16 + tl.arange(0, 16)[None, :]), e71)
    tl.store(out_e68 + bid * 33 + 16 + tl.full((), 0, tl.int32), e68)
    tl.store(out_e69 + bid * 33 + 16 + tl.full((), 0, tl.int32), e69)
    tl.store(out_e72 + bid * 64 + 16 + tl.arange(0, 32), e72)
    tl.store(out_e3 + bid * 544 + 16 + (tl.arange(0, 32)[:, None] * 16 + tl.arange(0, 16)[None, :]), e3)
    tl.store(out_e22 + bid * 33 + 16 + tl.full((), 0, tl.int32), e22)
    tl.store(out_e24 + bid * 33 + 16 + tl.full((), 0, tl.int32), e24)

@triton.jit
def mixed_helper_extended_1_0(e6, e7, mem0, mem1, mem2, steps, limit, bid):
    e8 = (tl.sum(e6.to(tl.float32), 1)).to(tl.float32)
    e9 = ((e8 + e7)).to(tl.float32)
    e10 = (tl.sum(e9.to(tl.float32), 0)).to(tl.float32)
    e11 = (tl.reshape(e9, (32, 1))).to(tl.float32)
    e12 = (e11.to(tl.float16)).to(tl.float16)
    e13 = ((e6 + e12)).to(tl.float16)
    return e13, e10
@triton.jit
def extended_1_0(mem0, mem1, mem2, out_e71, out_e68, out_e69, out_e72, steps, limit):
    bid = tl.program_id(0)
    e1 = ((((511 - (tl.arange(0, 32)[:, None] * 16 + tl.arange(0, 16)[None, :])) + 25) % 512)).to(tl.int32)
    e2 = (tl.full((32, 16), True, tl.int1)).to(tl.int1)
    e3 = (tl.load(mem0 + bid * 544 + 16 + e1 * 1, (e2 & (e1 >= 0) & (e1 < 512)), other=0)).to(tl.float16)
    e4 = ((e3 * e3)).to(tl.float16)
    e5 = (e4.to(tl.float16)).to(tl.float16)
    e14 = (((tl.arange(0, 32) + 27) % 32)).to(tl.int32)
    e15 = (tl.full((32,), True, tl.int1)).to(tl.int1)
    e16 = (tl.load(mem1 + bid * 64 + 16 + e14 * 1, (e15 & (e14 >= 0) & (e14 < 32)), other=0)).to(tl.float32)
    e17, e18 = mixed_helper_extended_1_0(e5, e16, mem0, mem1, mem2, steps, limit, bid)
    e19 = (steps).to(tl.int32)
    e20 = (tl.full((), 0, tl.int32)).to(tl.int32)
    e21 = (tl.full((), 0, tl.int32)).to(tl.int32)
    e22 = (tl.full((), True, tl.int1)).to(tl.int1)
    e23 = (tl.load(mem2 + bid * 33 + 16 + e21 * 1, (e22 & (e21 >= 0) & (e21 < 1)), other=0)).to(tl.int32)
    e24 = (tl.full((), 0, tl.int32)).to(tl.int32)
    e25 = ((e23 < e24)).to(tl.int1)
    e45 = e17
    e46 = e18
    e47 = e20
    for e26 in tl.range(0, tl.minimum(tl.maximum(e19, 0), 4), num_stages=2):
        e27 = e45
        e28 = e46
        e29 = e47
        if e25:
            e30 = e27
            e31 = (tl.full((32, 16), 0.125, tl.float16)).to(tl.float16)
            e32 = ((e30 + e31)).to(tl.float16)
            e36 = e32
        else:
            e33 = e27
            e34 = (tl.full((32, 16), 0.125, tl.float16)).to(tl.float16)
            e35 = ((e33 - e34)).to(tl.float16)
            e36 = e35
        e37, e38 = mixed_helper_extended_1_0(e36, e16, mem0, mem1, mem2, steps, limit, bid)
        e39 = (tl.full((32, 16), 0.125, tl.float16)).to(tl.float16)
        e40 = ((e37 * e39)).to(tl.float16)
        e41 = ((e28 + e38)).to(tl.float32)
        e42 = (tl.full((), 1, tl.int32)).to(tl.int32)
        e43 = ((e26 + e42)).to(tl.int32)
        e44 = ((e29 + e43)).to(tl.int32)
        e45 = e40
        e46 = e41
        e47 = e44
    e67 = e45
    e68 = e46
    e69 = e47
    e48 = tl.full((), 0, tl.int32)
    while e48 < tl.minimum(tl.maximum(e19, 0), 4):
        e49 = e67
        e50 = e68
        e51 = e69
        if e25:
            e52 = e49
            e53 = (tl.full((32, 16), 0.125, tl.float16)).to(tl.float16)
            e54 = ((e52 + e53)).to(tl.float16)
            e58 = e54
        else:
            e55 = e49
            e56 = (tl.full((32, 16), 0.125, tl.float16)).to(tl.float16)
            e57 = ((e55 - e56)).to(tl.float16)
            e58 = e57
        e59, e60 = mixed_helper_extended_1_0(e58, e16, mem0, mem1, mem2, steps, limit, bid)
        e61 = (tl.full((32, 16), 0.125, tl.float16)).to(tl.float16)
        e62 = ((e59 * e61)).to(tl.float16)
        e63 = ((e50 + e60)).to(tl.float32)
        e64 = (tl.full((), 1, tl.int32)).to(tl.int32)
        e65 = ((e48 + e64)).to(tl.int32)
        e66 = ((e51 + e65)).to(tl.int32)
        e67 = e62
        e68 = e63
        e69 = e66
        e48 += 1
    e70 = ((e67 * e45)).to(tl.float16)
    e71 = (e70.to(tl.float32)).to(tl.float32)
    e72 = (tl.sum(e71.to(tl.float32), 1)).to(tl.float32)
    tl.store(out_e71 + bid * 544 + 16 + (tl.arange(0, 32)[:, None] * 16 + tl.arange(0, 16)[None, :]), e71)
    tl.store(out_e68 + bid * 33 + 16 + tl.full((), 0, tl.int32), e68)
    tl.store(out_e69 + bid * 33 + 16 + tl.full((), 0, tl.int32), e69)
    tl.store(out_e72 + bid * 64 + 16 + tl.arange(0, 32), e72)

@triton.jit
def mixed_helper_extended_1_1(e6, e7, mem0, mem1, mem2, steps, limit, bid):
    e8 = (tl.sum(e6.to(tl.float32), 1)).to(tl.float32)
    e9 = ((e8 + e7)).to(tl.float32)
    e10 = (tl.sum(e9.to(tl.float32), 0)).to(tl.float32)
    e11 = (tl.reshape(e9, (32, 1))).to(tl.float32)
    e12 = (e11.to(tl.float16)).to(tl.float16)
    e13 = ((e6 + e12)).to(tl.float16)
    return e13, e10
@triton.jit
def extended_1_1(mem0, mem1, mem2, out_e71, out_e68, out_e69, out_e72, out_e3, out_e22, out_e24, steps, limit):
    bid = tl.program_id(0)
    e1 = ((((511 - (tl.arange(0, 32)[:, None] * 16 + tl.arange(0, 16)[None, :])) + 25) % 512)).to(tl.int32)
    e2 = (tl.full((32, 16), True, tl.int1)).to(tl.int1)
    e3 = (tl.load(mem0 + bid * 544 + 16 + e1 * 1, (e2 & (e1 >= 0) & (e1 < 512)), other=0)).to(tl.float16)
    e4 = ((e3 * e3)).to(tl.float16)
    e5 = (e4.to(tl.float16)).to(tl.float16)
    e14 = (((tl.arange(0, 32) + 27) % 32)).to(tl.int32)
    e15 = (tl.full((32,), True, tl.int1)).to(tl.int1)
    e16 = (tl.load(mem1 + bid * 64 + 16 + e14 * 1, (e15 & (e14 >= 0) & (e14 < 32)), other=0)).to(tl.float32)
    e17, e18 = mixed_helper_extended_1_1(e5, e16, mem0, mem1, mem2, steps, limit, bid)
    e19 = (steps).to(tl.int32)
    e20 = (tl.full((), 0, tl.int32)).to(tl.int32)
    e21 = (tl.full((), 0, tl.int32)).to(tl.int32)
    e22 = (tl.full((), True, tl.int1)).to(tl.int1)
    e23 = (tl.load(mem2 + bid * 33 + 16 + e21 * 1, (e22 & (e21 >= 0) & (e21 < 1)), other=0)).to(tl.int32)
    e24 = (tl.full((), 0, tl.int32)).to(tl.int32)
    e25 = ((e23 < e24)).to(tl.int1)
    e45 = e17
    e46 = e18
    e47 = e20
    for e26 in tl.range(0, tl.minimum(tl.maximum(e19, 0), 4), num_stages=2):
        e27 = e45
        e28 = e46
        e29 = e47
        if e25:
            e30 = e27
            e31 = (tl.full((32, 16), 0.125, tl.float16)).to(tl.float16)
            e32 = ((e30 + e31)).to(tl.float16)
            e36 = e32
        else:
            e33 = e27
            e34 = (tl.full((32, 16), 0.125, tl.float16)).to(tl.float16)
            e35 = ((e33 - e34)).to(tl.float16)
            e36 = e35
        e37, e38 = mixed_helper_extended_1_1(e36, e16, mem0, mem1, mem2, steps, limit, bid)
        e39 = (tl.full((32, 16), 0.125, tl.float16)).to(tl.float16)
        e40 = ((e37 * e39)).to(tl.float16)
        e41 = ((e28 + e38)).to(tl.float32)
        e42 = (tl.full((), 1, tl.int32)).to(tl.int32)
        e43 = ((e26 + e42)).to(tl.int32)
        e44 = ((e29 + e43)).to(tl.int32)
        e45 = e40
        e46 = e41
        e47 = e44
    e67 = e45
    e68 = e46
    e69 = e47
    e48 = tl.full((), 0, tl.int32)
    while e48 < tl.minimum(tl.maximum(e19, 0), 4):
        e49 = e67
        e50 = e68
        e51 = e69
        if e25:
            e52 = e49
            e53 = (tl.full((32, 16), 0.125, tl.float16)).to(tl.float16)
            e54 = ((e52 + e53)).to(tl.float16)
            e58 = e54
        else:
            e55 = e49
            e56 = (tl.full((32, 16), 0.125, tl.float16)).to(tl.float16)
            e57 = ((e55 - e56)).to(tl.float16)
            e58 = e57
        e59, e60 = mixed_helper_extended_1_1(e58, e16, mem0, mem1, mem2, steps, limit, bid)
        e61 = (tl.full((32, 16), 0.125, tl.float16)).to(tl.float16)
        e62 = ((e59 * e61)).to(tl.float16)
        e63 = ((e50 + e60)).to(tl.float32)
        e64 = (tl.full((), 1, tl.int32)).to(tl.int32)
        e65 = ((e48 + e64)).to(tl.int32)
        e66 = ((e51 + e65)).to(tl.int32)
        e67 = e62
        e68 = e63
        e69 = e66
        e48 += 1
    e70 = ((e67 * e45)).to(tl.float16)
    e71 = (e70.to(tl.float32)).to(tl.float32)
    e72 = (tl.sum(e71.to(tl.float32), 1)).to(tl.float32)
    tl.store(out_e71 + bid * 544 + 16 + (tl.arange(0, 32)[:, None] * 16 + tl.arange(0, 16)[None, :]), e71)
    tl.store(out_e68 + bid * 33 + 16 + tl.full((), 0, tl.int32), e68)
    tl.store(out_e69 + bid * 33 + 16 + tl.full((), 0, tl.int32), e69)
    tl.store(out_e72 + bid * 64 + 16 + tl.arange(0, 32), e72)
    tl.store(out_e3 + bid * 544 + 16 + (tl.arange(0, 32)[:, None] * 16 + tl.arange(0, 16)[None, :]), e3)
    tl.store(out_e22 + bid * 33 + 16 + tl.full((), 0, tl.int32), e22)
    tl.store(out_e24 + bid * 33 + 16 + tl.full((), 0, tl.int32), e24)

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
    compiled_0 = compile(ASTSource(extended_0_0, {0: '*fp16', 1: '*fp32', 2: '*i32', 3: '*fp32', 4: '*fp32', 5: '*i32', 6: '*fp32', 7: 'i32', 8: 'i32'}), target=GPUTarget("cuda", arch, 32), options={'num_warps': 4, 'num_stages': 1, 'enable_fp_fusion': False})
    record_extended_compilation('triton_0', compiled_0.asm, {'num_warps': 4, 'num_stages': 1, 'enable_fp_fusion': False})
    def launch_0(memories, outputs, steps, limit):
        arguments = [memories[n] for n in ['mem0', 'mem1', 'mem2']] + [outputs[n] for n in ['e71', 'e68', 'e69', 'e72']]
        compiled_0[(3, 1, 1)](*arguments, steps, limit)
    variants.append(('triton_0', ['e71', 'e68', 'e69', 'e72'], launch_0))
    extended_stage("compile", 'triton_1')
    compiled_1 = compile(ASTSource(extended_0_1, {0: '*fp16', 1: '*fp32', 2: '*i32', 3: '*fp32', 4: '*fp32', 5: '*i32', 6: '*fp32', 7: '*fp16', 8: '*i1', 9: '*i32', 10: 'i32', 11: 'i32'}), target=GPUTarget("cuda", arch, 32), options={'num_warps': 4, 'num_stages': 1, 'enable_fp_fusion': False})
    record_extended_compilation('triton_1', compiled_1.asm, {'num_warps': 4, 'num_stages': 1, 'enable_fp_fusion': False})
    def launch_1(memories, outputs, steps, limit):
        arguments = [memories[n] for n in ['mem0', 'mem1', 'mem2']] + [outputs[n] for n in ['e71', 'e68', 'e69', 'e72', 'e3', 'e22', 'e24']]
        compiled_1[(3, 1, 1)](*arguments, steps, limit)
    variants.append(('triton_1', ['e71', 'e68', 'e69', 'e72', 'e3', 'e22', 'e24'], launch_1))
    extended_stage("compile", 'triton_2')
    compiled_2 = compile(ASTSource(extended_1_0, {0: '*fp16', 1: '*fp32', 2: '*i32', 3: '*fp32', 4: '*fp32', 5: '*i32', 6: '*fp32', 7: 'i32', 8: 'i32'}), target=GPUTarget("cuda", arch, 32), options={'num_warps': 8, 'num_stages': 2, 'enable_fp_fusion': False})
    record_extended_compilation('triton_2', compiled_2.asm, {'num_warps': 8, 'num_stages': 2, 'enable_fp_fusion': False})
    def launch_2(memories, outputs, steps, limit):
        arguments = [memories[n] for n in ['mem0', 'mem1', 'mem2']] + [outputs[n] for n in ['e71', 'e68', 'e69', 'e72']]
        compiled_2[(3, 1, 1)](*arguments, steps, limit)
    variants.append(('triton_2', ['e71', 'e68', 'e69', 'e72'], launch_2))
    extended_stage("compile", 'triton_3')
    compiled_3 = compile(ASTSource(extended_1_1, {0: '*fp16', 1: '*fp32', 2: '*i32', 3: '*fp32', 4: '*fp32', 5: '*i32', 6: '*fp32', 7: '*fp16', 8: '*i1', 9: '*i32', 10: 'i32', 11: 'i32'}), target=GPUTarget("cuda", arch, 32), options={'num_warps': 8, 'num_stages': 2, 'enable_fp_fusion': False})
    record_extended_compilation('triton_3', compiled_3.asm, {'num_warps': 8, 'num_stages': 2, 'enable_fp_fusion': False})
    def launch_3(memories, outputs, steps, limit):
        arguments = [memories[n] for n in ['mem0', 'mem1', 'mem2']] + [outputs[n] for n in ['e71', 'e68', 'e69', 'e72', 'e3', 'e22', 'e24']]
        compiled_3[(3, 1, 1)](*arguments, steps, limit)
    variants.append(('triton_3', ['e71', 'e68', 'e69', 'e72', 'e3', 'e22', 'e24'], launch_3))
    return variants

PROGRAM = {'type': 'extended', 'version': 1, 'body': {'arguments': [], 'operations': [{'op': 'index', 'results': [{'name': 'e1', 'type': {'dtype': 'int32', 'shape': (32, 16)}}], 'operands': [], 'attrs': {'shift': 25, 'reverse': True}, 'regions': []}, {'op': 'constant', 'results': [{'name': 'e2', 'type': {'dtype': 'bool', 'shape': (32, 16)}}], 'operands': [], 'attrs': {'value': True}, 'regions': []}, {'op': 'load', 'results': [{'name': 'e3', 'type': {'dtype': 'float16', 'shape': (32, 16)}}], 'operands': ['e1', 'e2'], 'attrs': {'buffer': 'mem0'}, 'regions': []}, {'op': 'mul', 'results': [{'name': 'e4', 'type': {'dtype': 'float16', 'shape': (32, 16)}}], 'operands': ['e3', 'e3'], 'attrs': {}, 'regions': []}, {'op': 'cast', 'results': [{'name': 'e5', 'type': {'dtype': 'float16', 'shape': (32, 16)}}], 'operands': ['e4'], 'attrs': {}, 'regions': []}, {'op': 'index', 'results': [{'name': 'e14', 'type': {'dtype': 'int32', 'shape': (32,)}}], 'operands': [], 'attrs': {'shift': 27, 'reverse': False}, 'regions': []}, {'op': 'constant', 'results': [{'name': 'e15', 'type': {'dtype': 'bool', 'shape': (32,)}}], 'operands': [], 'attrs': {'value': True}, 'regions': []}, {'op': 'load', 'results': [{'name': 'e16', 'type': {'dtype': 'float32', 'shape': (32,)}}], 'operands': ['e14', 'e15'], 'attrs': {'buffer': 'mem1'}, 'regions': []}, {'op': 'call', 'results': [{'name': 'e17', 'type': {'dtype': 'float16', 'shape': (32, 16)}}, {'name': 'e18', 'type': {'dtype': 'float32', 'shape': ()}}], 'operands': ['e5', 'e16'], 'attrs': {'callee': 'mixed_helper'}, 'regions': []}, {'op': 'parameter', 'results': [{'name': 'e19', 'type': {'dtype': 'int32', 'shape': ()}}], 'operands': [], 'attrs': {'name': 'steps'}, 'regions': []}, {'op': 'constant', 'results': [{'name': 'e20', 'type': {'dtype': 'int32', 'shape': ()}}], 'operands': [], 'attrs': {'value': 0}, 'regions': []}, {'op': 'constant', 'results': [{'name': 'e21', 'type': {'dtype': 'int32', 'shape': ()}}], 'operands': [], 'attrs': {'value': 0}, 'regions': []}, {'op': 'constant', 'results': [{'name': 'e22', 'type': {'dtype': 'bool', 'shape': ()}}], 'operands': [], 'attrs': {'value': True}, 'regions': []}, {'op': 'load', 'results': [{'name': 'e23', 'type': {'dtype': 'int32', 'shape': ()}}], 'operands': ['e21', 'e22'], 'attrs': {'buffer': 'mem2'}, 'regions': []}, {'op': 'constant', 'results': [{'name': 'e24', 'type': {'dtype': 'int32', 'shape': ()}}], 'operands': [], 'attrs': {'value': 0}, 'regions': []}, {'op': 'lt', 'results': [{'name': 'e25', 'type': {'dtype': 'bool', 'shape': ()}}], 'operands': ['e23', 'e24'], 'attrs': {}, 'regions': []}, {'op': 'for', 'results': [{'name': 'e45', 'type': {'dtype': 'float16', 'shape': (32, 16)}}, {'name': 'e46', 'type': {'dtype': 'float32', 'shape': ()}}, {'name': 'e47', 'type': {'dtype': 'int32', 'shape': ()}}], 'operands': ['e19', 'e17', 'e18', 'e20'], 'attrs': {'max_steps': 4, 'pipelined': True}, 'regions': [{'arguments': [{'name': 'e26', 'type': {'dtype': 'int32', 'shape': ()}}, {'name': 'e27', 'type': {'dtype': 'float16', 'shape': (32, 16)}}, {'name': 'e28', 'type': {'dtype': 'float32', 'shape': ()}}, {'name': 'e29', 'type': {'dtype': 'int32', 'shape': ()}}], 'operations': [{'op': 'if', 'results': [{'name': 'e36', 'type': {'dtype': 'float16', 'shape': (32, 16)}}], 'operands': ['e25', 'e27'], 'attrs': {}, 'regions': [{'arguments': [{'name': 'e30', 'type': {'dtype': 'float16', 'shape': (32, 16)}}], 'operations': [{'op': 'constant', 'results': [{'name': 'e31', 'type': {'dtype': 'float16', 'shape': (32, 16)}}], 'operands': [], 'attrs': {'value': 0.125}, 'regions': []}, {'op': 'add', 'results': [{'name': 'e32', 'type': {'dtype': 'float16', 'shape': (32, 16)}}], 'operands': ['e30', 'e31'], 'attrs': {}, 'regions': []}], 'returns': ['e32']}, {'arguments': [{'name': 'e33', 'type': {'dtype': 'float16', 'shape': (32, 16)}}], 'operations': [{'op': 'constant', 'results': [{'name': 'e34', 'type': {'dtype': 'float16', 'shape': (32, 16)}}], 'operands': [], 'attrs': {'value': 0.125}, 'regions': []}, {'op': 'sub', 'results': [{'name': 'e35', 'type': {'dtype': 'float16', 'shape': (32, 16)}}], 'operands': ['e33', 'e34'], 'attrs': {}, 'regions': []}], 'returns': ['e35']}]}, {'op': 'call', 'results': [{'name': 'e37', 'type': {'dtype': 'float16', 'shape': (32, 16)}}, {'name': 'e38', 'type': {'dtype': 'float32', 'shape': ()}}], 'operands': ['e36', 'e16'], 'attrs': {'callee': 'mixed_helper'}, 'regions': []}, {'op': 'constant', 'results': [{'name': 'e39', 'type': {'dtype': 'float16', 'shape': (32, 16)}}], 'operands': [], 'attrs': {'value': 0.125}, 'regions': []}, {'op': 'mul', 'results': [{'name': 'e40', 'type': {'dtype': 'float16', 'shape': (32, 16)}}], 'operands': ['e37', 'e39'], 'attrs': {}, 'regions': []}, {'op': 'add', 'results': [{'name': 'e41', 'type': {'dtype': 'float32', 'shape': ()}}], 'operands': ['e28', 'e38'], 'attrs': {}, 'regions': []}, {'op': 'constant', 'results': [{'name': 'e42', 'type': {'dtype': 'int32', 'shape': ()}}], 'operands': [], 'attrs': {'value': 1}, 'regions': []}, {'op': 'add', 'results': [{'name': 'e43', 'type': {'dtype': 'int32', 'shape': ()}}], 'operands': ['e26', 'e42'], 'attrs': {}, 'regions': []}, {'op': 'add', 'results': [{'name': 'e44', 'type': {'dtype': 'int32', 'shape': ()}}], 'operands': ['e29', 'e43'], 'attrs': {}, 'regions': []}], 'returns': ['e40', 'e41', 'e44']}]}, {'op': 'while', 'results': [{'name': 'e67', 'type': {'dtype': 'float16', 'shape': (32, 16)}}, {'name': 'e68', 'type': {'dtype': 'float32', 'shape': ()}}, {'name': 'e69', 'type': {'dtype': 'int32', 'shape': ()}}], 'operands': ['e19', 'e45', 'e46', 'e47'], 'attrs': {'max_steps': 4, 'pipelined': False}, 'regions': [{'arguments': [{'name': 'e48', 'type': {'dtype': 'int32', 'shape': ()}}, {'name': 'e49', 'type': {'dtype': 'float16', 'shape': (32, 16)}}, {'name': 'e50', 'type': {'dtype': 'float32', 'shape': ()}}, {'name': 'e51', 'type': {'dtype': 'int32', 'shape': ()}}], 'operations': [{'op': 'if', 'results': [{'name': 'e58', 'type': {'dtype': 'float16', 'shape': (32, 16)}}], 'operands': ['e25', 'e49'], 'attrs': {}, 'regions': [{'arguments': [{'name': 'e52', 'type': {'dtype': 'float16', 'shape': (32, 16)}}], 'operations': [{'op': 'constant', 'results': [{'name': 'e53', 'type': {'dtype': 'float16', 'shape': (32, 16)}}], 'operands': [], 'attrs': {'value': 0.125}, 'regions': []}, {'op': 'add', 'results': [{'name': 'e54', 'type': {'dtype': 'float16', 'shape': (32, 16)}}], 'operands': ['e52', 'e53'], 'attrs': {}, 'regions': []}], 'returns': ['e54']}, {'arguments': [{'name': 'e55', 'type': {'dtype': 'float16', 'shape': (32, 16)}}], 'operations': [{'op': 'constant', 'results': [{'name': 'e56', 'type': {'dtype': 'float16', 'shape': (32, 16)}}], 'operands': [], 'attrs': {'value': 0.125}, 'regions': []}, {'op': 'sub', 'results': [{'name': 'e57', 'type': {'dtype': 'float16', 'shape': (32, 16)}}], 'operands': ['e55', 'e56'], 'attrs': {}, 'regions': []}], 'returns': ['e57']}]}, {'op': 'call', 'results': [{'name': 'e59', 'type': {'dtype': 'float16', 'shape': (32, 16)}}, {'name': 'e60', 'type': {'dtype': 'float32', 'shape': ()}}], 'operands': ['e58', 'e16'], 'attrs': {'callee': 'mixed_helper'}, 'regions': []}, {'op': 'constant', 'results': [{'name': 'e61', 'type': {'dtype': 'float16', 'shape': (32, 16)}}], 'operands': [], 'attrs': {'value': 0.125}, 'regions': []}, {'op': 'mul', 'results': [{'name': 'e62', 'type': {'dtype': 'float16', 'shape': (32, 16)}}], 'operands': ['e59', 'e61'], 'attrs': {}, 'regions': []}, {'op': 'add', 'results': [{'name': 'e63', 'type': {'dtype': 'float32', 'shape': ()}}], 'operands': ['e50', 'e60'], 'attrs': {}, 'regions': []}, {'op': 'constant', 'results': [{'name': 'e64', 'type': {'dtype': 'int32', 'shape': ()}}], 'operands': [], 'attrs': {'value': 1}, 'regions': []}, {'op': 'add', 'results': [{'name': 'e65', 'type': {'dtype': 'int32', 'shape': ()}}], 'operands': ['e48', 'e64'], 'attrs': {}, 'regions': []}, {'op': 'add', 'results': [{'name': 'e66', 'type': {'dtype': 'int32', 'shape': ()}}], 'operands': ['e51', 'e65'], 'attrs': {}, 'regions': []}], 'returns': ['e62', 'e63', 'e66']}]}, {'op': 'mul', 'results': [{'name': 'e70', 'type': {'dtype': 'float16', 'shape': (32, 16)}}], 'operands': ['e67', 'e45'], 'attrs': {}, 'regions': []}, {'op': 'cast', 'results': [{'name': 'e71', 'type': {'dtype': 'float32', 'shape': (32, 16)}}], 'operands': ['e70'], 'attrs': {}, 'regions': []}, {'op': 'reduce', 'results': [{'name': 'e72', 'type': {'dtype': 'float32', 'shape': (32,)}}], 'operands': ['e71'], 'attrs': {'axis': 1, 'kind': 'sum'}, 'regions': []}], 'returns': ['e71', 'e68', 'e69', 'e72']}, 'buffers': [{'name': 'mem0', 'dtype': 'float16', 'size': 512, 'role': 'input', 'base': None, 'offset': 0, 'stride': 1}, {'name': 'mem1', 'dtype': 'float32', 'size': 32, 'role': 'input', 'base': None, 'offset': 0, 'stride': 1}, {'name': 'mem2', 'dtype': 'int32', 'size': 1, 'role': 'input', 'base': None, 'offset': 0, 'stride': 1}], 'functions': [{'name': 'mixed_helper', 'body': {'arguments': [{'name': 'e6', 'type': {'dtype': 'float16', 'shape': (32, 16)}}, {'name': 'e7', 'type': {'dtype': 'float32', 'shape': (32,)}}], 'operations': [{'op': 'reduce', 'results': [{'name': 'e8', 'type': {'dtype': 'float32', 'shape': (32,)}}], 'operands': ['e6'], 'attrs': {'axis': 1, 'kind': 'sum'}, 'regions': []}, {'op': 'add', 'results': [{'name': 'e9', 'type': {'dtype': 'float32', 'shape': (32,)}}], 'operands': ['e8', 'e7'], 'attrs': {}, 'regions': []}, {'op': 'reduce', 'results': [{'name': 'e10', 'type': {'dtype': 'float32', 'shape': ()}}], 'operands': ['e9'], 'attrs': {'axis': 0, 'kind': 'sum'}, 'regions': []}, {'op': 'reshape', 'results': [{'name': 'e11', 'type': {'dtype': 'float32', 'shape': (32, 1)}}], 'operands': ['e9'], 'attrs': {}, 'regions': []}, {'op': 'cast', 'results': [{'name': 'e12', 'type': {'dtype': 'float16', 'shape': (32, 1)}}], 'operands': ['e11'], 'attrs': {}, 'regions': []}, {'op': 'add', 'results': [{'name': 'e13', 'type': {'dtype': 'float16', 'shape': (32, 16)}}], 'operands': ['e6', 'e12'], 'attrs': {}, 'regions': []}], 'returns': ['e13', 'e10']}}], 'observations': ['e3', 'e22', 'e24'], 'blocks': 3, 'input_pattern': 'integer', 'runtime_cases': [(0, 1), (1, 511), (2, 512)], 'configuration_pair': True, 'observation_pair': True, 'family': 'control_calls'}

if __name__ == '__main__':
    if os.environ.get('TILESMITH_COMPILE_ONLY', '0') == '1':
        prepare_extended(device_compile=False)
        extended_stage('complete', 'compile_only')
        print('COMPILE PASSED (no GPU execution)')
    else:
        run_extended(PROGRAM, prepare_extended, 0, 3)
        extended_stage('complete', 'execute')
        print('ALL PASSED')
