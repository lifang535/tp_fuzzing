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


@tilelang.jit
def extended_0_0():
    @T.prim_func
    def impl(mem0: T.Buffer((3, 544), "float16"), mem1: T.Buffer((3, 1060), "float16"), out_e17: T.Buffer((3, 544), "float16"), steps: T.int32, limit: T.int32):
        with T.Kernel(3, threads=128) as bid:
            e1 = T.alloc_fragment((32, 16), "int32")
            e2 = T.alloc_fragment((32, 16), "bool")
            e3 = T.alloc_fragment((32, 16), "float16")
            e4 = T.alloc_fragment((32, 16), "float16")
            e5 = T.alloc_fragment((32, 16), "int32")
            e6 = T.alloc_fragment((1,), "int32")
            e7 = T.alloc_fragment((32, 16), "bool")
            e8 = T.alloc_fragment((32, 16), "float16")
            e9 = T.alloc_fragment((32, 16), "float16")
            e10 = T.alloc_fragment((32, 16), "float16")
            e11 = T.alloc_fragment((32, 16), "bool")
            e12 = T.alloc_fragment((32, 16), "float16")
            e13 = T.alloc_fragment((32, 16), "int32")
            e14 = T.alloc_fragment((32, 16), "int32")
            e15 = T.alloc_fragment((32, 16), "float16")
            e16 = T.alloc_fragment((32, 16), "float16")
            e17 = T.alloc_fragment((32, 16), "float16")
            for i, j in T.Parallel(32, 16):
                e1[i, j] = T.cast((((511 - (i * 16 + j)) + 25) % 512), "int32")
            for i, j in T.Parallel(32, 16):
                e2[i, j] = T.cast(True, "bool")
            for i, j in T.Parallel(32, 16):
                e3[i, j] = T.cast(T.if_then_else((e2[i, j] and e1[i, j] >= 0 and e1[i, j] < 512), mem0[bid, 16 + e1[i, j] * 1], T.cast(0, "float16")), "float16")
            for i, j in T.Parallel(32, 16):
                e4[i, j] = T.cast((e3[i, j] * e3[i, j]), "float16")
            for i, j in T.Parallel(32, 16):
                e5[i, j] = T.cast((((511 - (i * 16 + j)) + 89) % 512), "int32")
            e6[0] = T.cast(limit, "int32")
            for i, j in T.Parallel(32, 16):
                e7[i, j] = T.cast((e5[i, j] < e6[0]), "bool")
            for i, j in T.Parallel(32, 16):
                if (e7[i, j] and e5[i, j] >= 0 and e5[i, j] < 512):
                    mem1[bid, 17 + e5[i, j] * 2] = e4[i, j]
            T.sync_threads()
            for i, j in T.Parallel(32, 16):
                e8[i, j] = T.cast(T.if_then_else((e7[i, j] and e5[i, j] >= 0 and e5[i, j] < 512), mem1[bid, 17 + e5[i, j] * 2], T.cast(0, "float16")), "float16")
            for i, j in T.Parallel(32, 16):
                e9[i, j] = T.cast(0.125, "float16")
            for i, j in T.Parallel(32, 16):
                e10[i, j] = T.cast((e8[i, j] + e9[i, j]), "float16")
            T.sync_threads()
            for i, j in T.Parallel(32, 16):
                if (e7[i, j] and e5[i, j] >= 0 and e5[i, j] < 512):
                    mem1[bid, 18 + e5[i, j] * 1] = e10[i, j]
            T.sync_threads()
            for i, j in T.Parallel(32, 16):
                e11[i, j] = T.cast(True, "bool")
            for i, j in T.Parallel(32, 16):
                e12[i, j] = T.cast(T.if_then_else((e11[i, j] and e5[i, j] >= 0 and e5[i, j] < 512), mem1[bid, 17 + e5[i, j] * 2], T.cast(0, "float16")), "float16")
            for i, j in T.Parallel(32, 16):
                e13[i, j] = T.cast(1, "int32")
            for i, j in T.Parallel(32, 16):
                e14[i, j] = T.cast((e5[i, j] ^ e13[i, j]), "int32")
            for i, j in T.Parallel(32, 16):
                e15[i, j] = T.cast(T.if_then_else((e7[i, j] and e14[i, j] >= 0 and e14[i, j] < 512), mem1[bid, 18 + e14[i, j] * 1], T.cast(0, "float16")), "float16")
            for i, j in T.Parallel(32, 16):
                e16[i, j] = T.cast((e12[i, j] + e15[i, j]), "float16")
            for i, j in T.Parallel(32, 16):
                e17[i, j] = T.cast((e16[i, j] * e9[i, j]), "float16")
            for i, j in T.Parallel(32, 16):
                out_e17[bid, 16 + (i * 16 + j)] = e17[i, j]
    return impl

@tilelang.jit
def extended_0_1():
    @T.prim_func
    def impl(mem0: T.Buffer((3, 544), "float16"), mem1: T.Buffer((3, 1060), "float16"), out_e17: T.Buffer((3, 544), "float16"), out_e3: T.Buffer((3, 544), "float16"), out_e8: T.Buffer((3, 544), "float16"), out_e12: T.Buffer((3, 544), "float16"), out_e13: T.Buffer((3, 544), "int32"), out_e2: T.Buffer((3, 544), "bool"), steps: T.int32, limit: T.int32):
        with T.Kernel(3, threads=128) as bid:
            e1 = T.alloc_fragment((32, 16), "int32")
            e2 = T.alloc_fragment((32, 16), "bool")
            e3 = T.alloc_fragment((32, 16), "float16")
            e4 = T.alloc_fragment((32, 16), "float16")
            e5 = T.alloc_fragment((32, 16), "int32")
            e6 = T.alloc_fragment((1,), "int32")
            e7 = T.alloc_fragment((32, 16), "bool")
            e8 = T.alloc_fragment((32, 16), "float16")
            e9 = T.alloc_fragment((32, 16), "float16")
            e10 = T.alloc_fragment((32, 16), "float16")
            e11 = T.alloc_fragment((32, 16), "bool")
            e12 = T.alloc_fragment((32, 16), "float16")
            e13 = T.alloc_fragment((32, 16), "int32")
            e14 = T.alloc_fragment((32, 16), "int32")
            e15 = T.alloc_fragment((32, 16), "float16")
            e16 = T.alloc_fragment((32, 16), "float16")
            e17 = T.alloc_fragment((32, 16), "float16")
            for i, j in T.Parallel(32, 16):
                e1[i, j] = T.cast((((511 - (i * 16 + j)) + 25) % 512), "int32")
            for i, j in T.Parallel(32, 16):
                e2[i, j] = T.cast(True, "bool")
            for i, j in T.Parallel(32, 16):
                e3[i, j] = T.cast(T.if_then_else((e2[i, j] and e1[i, j] >= 0 and e1[i, j] < 512), mem0[bid, 16 + e1[i, j] * 1], T.cast(0, "float16")), "float16")
            for i, j in T.Parallel(32, 16):
                e4[i, j] = T.cast((e3[i, j] * e3[i, j]), "float16")
            for i, j in T.Parallel(32, 16):
                e5[i, j] = T.cast((((511 - (i * 16 + j)) + 89) % 512), "int32")
            e6[0] = T.cast(limit, "int32")
            for i, j in T.Parallel(32, 16):
                e7[i, j] = T.cast((e5[i, j] < e6[0]), "bool")
            for i, j in T.Parallel(32, 16):
                if (e7[i, j] and e5[i, j] >= 0 and e5[i, j] < 512):
                    mem1[bid, 17 + e5[i, j] * 2] = e4[i, j]
            T.sync_threads()
            for i, j in T.Parallel(32, 16):
                e8[i, j] = T.cast(T.if_then_else((e7[i, j] and e5[i, j] >= 0 and e5[i, j] < 512), mem1[bid, 17 + e5[i, j] * 2], T.cast(0, "float16")), "float16")
            for i, j in T.Parallel(32, 16):
                e9[i, j] = T.cast(0.125, "float16")
            for i, j in T.Parallel(32, 16):
                e10[i, j] = T.cast((e8[i, j] + e9[i, j]), "float16")
            T.sync_threads()
            for i, j in T.Parallel(32, 16):
                if (e7[i, j] and e5[i, j] >= 0 and e5[i, j] < 512):
                    mem1[bid, 18 + e5[i, j] * 1] = e10[i, j]
            T.sync_threads()
            for i, j in T.Parallel(32, 16):
                e11[i, j] = T.cast(True, "bool")
            for i, j in T.Parallel(32, 16):
                e12[i, j] = T.cast(T.if_then_else((e11[i, j] and e5[i, j] >= 0 and e5[i, j] < 512), mem1[bid, 17 + e5[i, j] * 2], T.cast(0, "float16")), "float16")
            for i, j in T.Parallel(32, 16):
                e13[i, j] = T.cast(1, "int32")
            for i, j in T.Parallel(32, 16):
                e14[i, j] = T.cast((e5[i, j] ^ e13[i, j]), "int32")
            for i, j in T.Parallel(32, 16):
                e15[i, j] = T.cast(T.if_then_else((e7[i, j] and e14[i, j] >= 0 and e14[i, j] < 512), mem1[bid, 18 + e14[i, j] * 1], T.cast(0, "float16")), "float16")
            for i, j in T.Parallel(32, 16):
                e16[i, j] = T.cast((e12[i, j] + e15[i, j]), "float16")
            for i, j in T.Parallel(32, 16):
                e17[i, j] = T.cast((e16[i, j] * e9[i, j]), "float16")
            for i, j in T.Parallel(32, 16):
                out_e17[bid, 16 + (i * 16 + j)] = e17[i, j]
            for i, j in T.Parallel(32, 16):
                out_e3[bid, 16 + (i * 16 + j)] = e3[i, j]
            for i, j in T.Parallel(32, 16):
                out_e8[bid, 16 + (i * 16 + j)] = e8[i, j]
            for i, j in T.Parallel(32, 16):
                out_e12[bid, 16 + (i * 16 + j)] = e12[i, j]
            for i, j in T.Parallel(32, 16):
                out_e13[bid, 16 + (i * 16 + j)] = e13[i, j]
            for i, j in T.Parallel(32, 16):
                out_e2[bid, 16 + (i * 16 + j)] = e2[i, j]
    return impl

@tilelang.jit
def extended_1_0():
    @T.prim_func
    def impl(mem0: T.Buffer((3, 544), "float16"), mem1: T.Buffer((3, 1060), "float16"), out_e17: T.Buffer((3, 544), "float16"), steps: T.int32, limit: T.int32):
        with T.Kernel(3, threads=256) as bid:
            e1 = T.alloc_fragment((32, 16), "int32")
            e2 = T.alloc_fragment((32, 16), "bool")
            e3 = T.alloc_fragment((32, 16), "float16")
            e4 = T.alloc_fragment((32, 16), "float16")
            e5 = T.alloc_fragment((32, 16), "int32")
            e6 = T.alloc_fragment((1,), "int32")
            e7 = T.alloc_fragment((32, 16), "bool")
            e8 = T.alloc_fragment((32, 16), "float16")
            e9 = T.alloc_fragment((32, 16), "float16")
            e10 = T.alloc_fragment((32, 16), "float16")
            e11 = T.alloc_fragment((32, 16), "bool")
            e12 = T.alloc_fragment((32, 16), "float16")
            e13 = T.alloc_fragment((32, 16), "int32")
            e14 = T.alloc_fragment((32, 16), "int32")
            e15 = T.alloc_fragment((32, 16), "float16")
            e16 = T.alloc_fragment((32, 16), "float16")
            e17 = T.alloc_fragment((32, 16), "float16")
            for i, j in T.Parallel(32, 16):
                e1[i, j] = T.cast((((511 - (i * 16 + j)) + 25) % 512), "int32")
            for i, j in T.Parallel(32, 16):
                e2[i, j] = T.cast(True, "bool")
            for i, j in T.Parallel(32, 16):
                e3[i, j] = T.cast(T.if_then_else((e2[i, j] and e1[i, j] >= 0 and e1[i, j] < 512), mem0[bid, 16 + e1[i, j] * 1], T.cast(0, "float16")), "float16")
            for i, j in T.Parallel(32, 16):
                e4[i, j] = T.cast((e3[i, j] * e3[i, j]), "float16")
            for i, j in T.Parallel(32, 16):
                e5[i, j] = T.cast((((511 - (i * 16 + j)) + 89) % 512), "int32")
            e6[0] = T.cast(limit, "int32")
            for i, j in T.Parallel(32, 16):
                e7[i, j] = T.cast((e5[i, j] < e6[0]), "bool")
            for i, j in T.Parallel(32, 16):
                if (e7[i, j] and e5[i, j] >= 0 and e5[i, j] < 512):
                    mem1[bid, 17 + e5[i, j] * 2] = e4[i, j]
            T.sync_threads()
            for i, j in T.Parallel(32, 16):
                e8[i, j] = T.cast(T.if_then_else((e7[i, j] and e5[i, j] >= 0 and e5[i, j] < 512), mem1[bid, 17 + e5[i, j] * 2], T.cast(0, "float16")), "float16")
            for i, j in T.Parallel(32, 16):
                e9[i, j] = T.cast(0.125, "float16")
            for i, j in T.Parallel(32, 16):
                e10[i, j] = T.cast((e8[i, j] + e9[i, j]), "float16")
            T.sync_threads()
            for i, j in T.Parallel(32, 16):
                if (e7[i, j] and e5[i, j] >= 0 and e5[i, j] < 512):
                    mem1[bid, 18 + e5[i, j] * 1] = e10[i, j]
            T.sync_threads()
            for i, j in T.Parallel(32, 16):
                e11[i, j] = T.cast(True, "bool")
            for i, j in T.Parallel(32, 16):
                e12[i, j] = T.cast(T.if_then_else((e11[i, j] and e5[i, j] >= 0 and e5[i, j] < 512), mem1[bid, 17 + e5[i, j] * 2], T.cast(0, "float16")), "float16")
            for i, j in T.Parallel(32, 16):
                e13[i, j] = T.cast(1, "int32")
            for i, j in T.Parallel(32, 16):
                e14[i, j] = T.cast((e5[i, j] ^ e13[i, j]), "int32")
            for i, j in T.Parallel(32, 16):
                e15[i, j] = T.cast(T.if_then_else((e7[i, j] and e14[i, j] >= 0 and e14[i, j] < 512), mem1[bid, 18 + e14[i, j] * 1], T.cast(0, "float16")), "float16")
            for i, j in T.Parallel(32, 16):
                e16[i, j] = T.cast((e12[i, j] + e15[i, j]), "float16")
            for i, j in T.Parallel(32, 16):
                e17[i, j] = T.cast((e16[i, j] * e9[i, j]), "float16")
            for i, j in T.Parallel(32, 16):
                out_e17[bid, 16 + (i * 16 + j)] = e17[i, j]
    return impl

@tilelang.jit
def extended_1_1():
    @T.prim_func
    def impl(mem0: T.Buffer((3, 544), "float16"), mem1: T.Buffer((3, 1060), "float16"), out_e17: T.Buffer((3, 544), "float16"), out_e3: T.Buffer((3, 544), "float16"), out_e8: T.Buffer((3, 544), "float16"), out_e12: T.Buffer((3, 544), "float16"), out_e13: T.Buffer((3, 544), "int32"), out_e2: T.Buffer((3, 544), "bool"), steps: T.int32, limit: T.int32):
        with T.Kernel(3, threads=256) as bid:
            e1 = T.alloc_fragment((32, 16), "int32")
            e2 = T.alloc_fragment((32, 16), "bool")
            e3 = T.alloc_fragment((32, 16), "float16")
            e4 = T.alloc_fragment((32, 16), "float16")
            e5 = T.alloc_fragment((32, 16), "int32")
            e6 = T.alloc_fragment((1,), "int32")
            e7 = T.alloc_fragment((32, 16), "bool")
            e8 = T.alloc_fragment((32, 16), "float16")
            e9 = T.alloc_fragment((32, 16), "float16")
            e10 = T.alloc_fragment((32, 16), "float16")
            e11 = T.alloc_fragment((32, 16), "bool")
            e12 = T.alloc_fragment((32, 16), "float16")
            e13 = T.alloc_fragment((32, 16), "int32")
            e14 = T.alloc_fragment((32, 16), "int32")
            e15 = T.alloc_fragment((32, 16), "float16")
            e16 = T.alloc_fragment((32, 16), "float16")
            e17 = T.alloc_fragment((32, 16), "float16")
            for i, j in T.Parallel(32, 16):
                e1[i, j] = T.cast((((511 - (i * 16 + j)) + 25) % 512), "int32")
            for i, j in T.Parallel(32, 16):
                e2[i, j] = T.cast(True, "bool")
            for i, j in T.Parallel(32, 16):
                e3[i, j] = T.cast(T.if_then_else((e2[i, j] and e1[i, j] >= 0 and e1[i, j] < 512), mem0[bid, 16 + e1[i, j] * 1], T.cast(0, "float16")), "float16")
            for i, j in T.Parallel(32, 16):
                e4[i, j] = T.cast((e3[i, j] * e3[i, j]), "float16")
            for i, j in T.Parallel(32, 16):
                e5[i, j] = T.cast((((511 - (i * 16 + j)) + 89) % 512), "int32")
            e6[0] = T.cast(limit, "int32")
            for i, j in T.Parallel(32, 16):
                e7[i, j] = T.cast((e5[i, j] < e6[0]), "bool")
            for i, j in T.Parallel(32, 16):
                if (e7[i, j] and e5[i, j] >= 0 and e5[i, j] < 512):
                    mem1[bid, 17 + e5[i, j] * 2] = e4[i, j]
            T.sync_threads()
            for i, j in T.Parallel(32, 16):
                e8[i, j] = T.cast(T.if_then_else((e7[i, j] and e5[i, j] >= 0 and e5[i, j] < 512), mem1[bid, 17 + e5[i, j] * 2], T.cast(0, "float16")), "float16")
            for i, j in T.Parallel(32, 16):
                e9[i, j] = T.cast(0.125, "float16")
            for i, j in T.Parallel(32, 16):
                e10[i, j] = T.cast((e8[i, j] + e9[i, j]), "float16")
            T.sync_threads()
            for i, j in T.Parallel(32, 16):
                if (e7[i, j] and e5[i, j] >= 0 and e5[i, j] < 512):
                    mem1[bid, 18 + e5[i, j] * 1] = e10[i, j]
            T.sync_threads()
            for i, j in T.Parallel(32, 16):
                e11[i, j] = T.cast(True, "bool")
            for i, j in T.Parallel(32, 16):
                e12[i, j] = T.cast(T.if_then_else((e11[i, j] and e5[i, j] >= 0 and e5[i, j] < 512), mem1[bid, 17 + e5[i, j] * 2], T.cast(0, "float16")), "float16")
            for i, j in T.Parallel(32, 16):
                e13[i, j] = T.cast(1, "int32")
            for i, j in T.Parallel(32, 16):
                e14[i, j] = T.cast((e5[i, j] ^ e13[i, j]), "int32")
            for i, j in T.Parallel(32, 16):
                e15[i, j] = T.cast(T.if_then_else((e7[i, j] and e14[i, j] >= 0 and e14[i, j] < 512), mem1[bid, 18 + e14[i, j] * 1], T.cast(0, "float16")), "float16")
            for i, j in T.Parallel(32, 16):
                e16[i, j] = T.cast((e12[i, j] + e15[i, j]), "float16")
            for i, j in T.Parallel(32, 16):
                e17[i, j] = T.cast((e16[i, j] * e9[i, j]), "float16")
            for i, j in T.Parallel(32, 16):
                out_e17[bid, 16 + (i * 16 + j)] = e17[i, j]
            for i, j in T.Parallel(32, 16):
                out_e3[bid, 16 + (i * 16 + j)] = e3[i, j]
            for i, j in T.Parallel(32, 16):
                out_e8[bid, 16 + (i * 16 + j)] = e8[i, j]
            for i, j in T.Parallel(32, 16):
                out_e12[bid, 16 + (i * 16 + j)] = e12[i, j]
            for i, j in T.Parallel(32, 16):
                out_e13[bid, 16 + (i * 16 + j)] = e13[i, j]
            for i, j in T.Parallel(32, 16):
                out_e2[bid, 16 + (i * 16 + j)] = e2[i, j]
    return impl

def prepare_extended(device_compile=True):
    import torch
    from tilelang import tvm
    from tilelang.engine import lower as tilelang_lower
    arch = int(os.environ.get('TILESMITH_CUDA_ARCH', '0'))
    if not arch:
        major, minor = torch.cuda.get_device_capability() if torch.cuda.is_available() else (8, 9)
        arch = major * 10 + minor
    target = tvm.target.Target({"kind": "cuda", "arch": "sm_" + str(arch)})
    variants = []
    extended_stage("lowering", 'tilelang_0')
    with target:
        ir_0 = extended_0_0.get_tir()
    record_extended_compilation('tilelang_0', {"tir": str(ir_0)}, {'threads': 128, 'stages': 1, 'pass_configs': {}}, complete=False)
    extended_stage("device_compile", 'tilelang_0')
    if device_compile:
        compiled_0 = tilelang.compile(ir_0, target=target, pass_configs={})
        artifact_0 = compiled_0.artifact
        artifacts_0 = {"cuda": compiled_0.get_kernel_source()}
        if artifact_0 is not None:
            artifacts_0["lowered_tir"] = str(artifact_0.device_mod)
        record_extended_compilation('tilelang_0', artifacts_0, {'threads': 128, 'stages': 1, 'pass_configs': {}})
        def launch_0(memories, outputs, steps, limit):
            arguments = [memories[n] for n in ['mem0', 'mem1']] + [outputs[n] for n in ['e17']]
            compiled_0(*arguments, steps, limit)
        variants.append(('tilelang_0', ['e17'], launch_0))
    else:
        with target, tvm.transform.PassContext(opt_level=3, config={}):
            lowered_0 = tilelang_lower(ir_0, target=target, enable_device_compile=False)
        record_extended_compilation('tilelang_0', {"lowered_tir": str(lowered_0.device_mod), "cuda": lowered_0.kernel_source}, {'threads': 128, 'stages': 1, 'pass_configs': {}}, complete=False)
        from tilelang.engine.lower import tilelang_callback_cuda_compile
        cubin_0 = tilelang_callback_cuda_compile(lowered_0.kernel_source, target, {})
        record_extended_compilation('tilelang_0', {"cubin": cubin_0}, {'threads': 128, 'stages': 1, 'pass_configs': {}})
    extended_stage("lowering", 'tilelang_1')
    with target:
        ir_1 = extended_0_1.get_tir()
    record_extended_compilation('tilelang_1', {"tir": str(ir_1)}, {'threads': 128, 'stages': 1, 'pass_configs': {}}, complete=False)
    extended_stage("device_compile", 'tilelang_1')
    if device_compile:
        compiled_1 = tilelang.compile(ir_1, target=target, pass_configs={})
        artifact_1 = compiled_1.artifact
        artifacts_1 = {"cuda": compiled_1.get_kernel_source()}
        if artifact_1 is not None:
            artifacts_1["lowered_tir"] = str(artifact_1.device_mod)
        record_extended_compilation('tilelang_1', artifacts_1, {'threads': 128, 'stages': 1, 'pass_configs': {}})
        def launch_1(memories, outputs, steps, limit):
            arguments = [memories[n] for n in ['mem0', 'mem1']] + [outputs[n] for n in ['e17', 'e3', 'e8', 'e12', 'e13', 'e2']]
            compiled_1(*arguments, steps, limit)
        variants.append(('tilelang_1', ['e17', 'e3', 'e8', 'e12', 'e13', 'e2'], launch_1))
    else:
        with target, tvm.transform.PassContext(opt_level=3, config={}):
            lowered_1 = tilelang_lower(ir_1, target=target, enable_device_compile=False)
        record_extended_compilation('tilelang_1', {"lowered_tir": str(lowered_1.device_mod), "cuda": lowered_1.kernel_source}, {'threads': 128, 'stages': 1, 'pass_configs': {}}, complete=False)
        from tilelang.engine.lower import tilelang_callback_cuda_compile
        cubin_1 = tilelang_callback_cuda_compile(lowered_1.kernel_source, target, {})
        record_extended_compilation('tilelang_1', {"cubin": cubin_1}, {'threads': 128, 'stages': 1, 'pass_configs': {}})
    extended_stage("lowering", 'tilelang_2')
    with target:
        ir_2 = extended_1_0.get_tir()
    record_extended_compilation('tilelang_2', {"tir": str(ir_2)}, {'threads': 256, 'stages': 2, 'pass_configs': {'tirx.disable_vectorize': True}}, complete=False)
    extended_stage("device_compile", 'tilelang_2')
    if device_compile:
        compiled_2 = tilelang.compile(ir_2, target=target, pass_configs={'tirx.disable_vectorize': True})
        artifact_2 = compiled_2.artifact
        artifacts_2 = {"cuda": compiled_2.get_kernel_source()}
        if artifact_2 is not None:
            artifacts_2["lowered_tir"] = str(artifact_2.device_mod)
        record_extended_compilation('tilelang_2', artifacts_2, {'threads': 256, 'stages': 2, 'pass_configs': {'tirx.disable_vectorize': True}})
        def launch_2(memories, outputs, steps, limit):
            arguments = [memories[n] for n in ['mem0', 'mem1']] + [outputs[n] for n in ['e17']]
            compiled_2(*arguments, steps, limit)
        variants.append(('tilelang_2', ['e17'], launch_2))
    else:
        with target, tvm.transform.PassContext(opt_level=3, config={'tirx.disable_vectorize': True}):
            lowered_2 = tilelang_lower(ir_2, target=target, enable_device_compile=False)
        record_extended_compilation('tilelang_2', {"lowered_tir": str(lowered_2.device_mod), "cuda": lowered_2.kernel_source}, {'threads': 256, 'stages': 2, 'pass_configs': {'tirx.disable_vectorize': True}}, complete=False)
        from tilelang.engine.lower import tilelang_callback_cuda_compile
        cubin_2 = tilelang_callback_cuda_compile(lowered_2.kernel_source, target, {'tirx.disable_vectorize': True})
        record_extended_compilation('tilelang_2', {"cubin": cubin_2}, {'threads': 256, 'stages': 2, 'pass_configs': {'tirx.disable_vectorize': True}})
    extended_stage("lowering", 'tilelang_3')
    with target:
        ir_3 = extended_1_1.get_tir()
    record_extended_compilation('tilelang_3', {"tir": str(ir_3)}, {'threads': 256, 'stages': 2, 'pass_configs': {'tirx.disable_vectorize': True}}, complete=False)
    extended_stage("device_compile", 'tilelang_3')
    if device_compile:
        compiled_3 = tilelang.compile(ir_3, target=target, pass_configs={'tirx.disable_vectorize': True})
        artifact_3 = compiled_3.artifact
        artifacts_3 = {"cuda": compiled_3.get_kernel_source()}
        if artifact_3 is not None:
            artifacts_3["lowered_tir"] = str(artifact_3.device_mod)
        record_extended_compilation('tilelang_3', artifacts_3, {'threads': 256, 'stages': 2, 'pass_configs': {'tirx.disable_vectorize': True}})
        def launch_3(memories, outputs, steps, limit):
            arguments = [memories[n] for n in ['mem0', 'mem1']] + [outputs[n] for n in ['e17', 'e3', 'e8', 'e12', 'e13', 'e2']]
            compiled_3(*arguments, steps, limit)
        variants.append(('tilelang_3', ['e17', 'e3', 'e8', 'e12', 'e13', 'e2'], launch_3))
    else:
        with target, tvm.transform.PassContext(opt_level=3, config={'tirx.disable_vectorize': True}):
            lowered_3 = tilelang_lower(ir_3, target=target, enable_device_compile=False)
        record_extended_compilation('tilelang_3', {"lowered_tir": str(lowered_3.device_mod), "cuda": lowered_3.kernel_source}, {'threads': 256, 'stages': 2, 'pass_configs': {'tirx.disable_vectorize': True}}, complete=False)
        from tilelang.engine.lower import tilelang_callback_cuda_compile
        cubin_3 = tilelang_callback_cuda_compile(lowered_3.kernel_source, target, {'tirx.disable_vectorize': True})
        record_extended_compilation('tilelang_3', {"cubin": cubin_3}, {'threads': 256, 'stages': 2, 'pass_configs': {'tirx.disable_vectorize': True}})
    return variants

PROGRAM = {'type': 'extended', 'version': 1, 'body': {'arguments': [], 'operations': [{'op': 'index', 'results': [{'name': 'e1', 'type': {'dtype': 'int32', 'shape': (32, 16)}}], 'operands': [], 'attrs': {'shift': 25, 'reverse': True}, 'regions': []}, {'op': 'constant', 'results': [{'name': 'e2', 'type': {'dtype': 'bool', 'shape': (32, 16)}}], 'operands': [], 'attrs': {'value': True}, 'regions': []}, {'op': 'load', 'results': [{'name': 'e3', 'type': {'dtype': 'float16', 'shape': (32, 16)}}], 'operands': ['e1', 'e2'], 'attrs': {'buffer': 'mem0'}, 'regions': []}, {'op': 'mul', 'results': [{'name': 'e4', 'type': {'dtype': 'float16', 'shape': (32, 16)}}], 'operands': ['e3', 'e3'], 'attrs': {}, 'regions': []}, {'op': 'index', 'results': [{'name': 'e5', 'type': {'dtype': 'int32', 'shape': (32, 16)}}], 'operands': [], 'attrs': {'shift': 89, 'reverse': True}, 'regions': []}, {'op': 'parameter', 'results': [{'name': 'e6', 'type': {'dtype': 'int32', 'shape': ()}}], 'operands': [], 'attrs': {'name': 'limit'}, 'regions': []}, {'op': 'lt', 'results': [{'name': 'e7', 'type': {'dtype': 'bool', 'shape': (32, 16)}}], 'operands': ['e5', 'e6'], 'attrs': {}, 'regions': []}, {'op': 'store', 'results': [], 'operands': ['e5', 'e7', 'e4'], 'attrs': {'buffer': 'mem2'}, 'regions': []}, {'op': 'barrier', 'results': [], 'operands': [], 'attrs': {}, 'regions': []}, {'op': 'load', 'results': [{'name': 'e8', 'type': {'dtype': 'float16', 'shape': (32, 16)}}], 'operands': ['e5', 'e7'], 'attrs': {'buffer': 'mem2'}, 'regions': []}, {'op': 'constant', 'results': [{'name': 'e9', 'type': {'dtype': 'float16', 'shape': (32, 16)}}], 'operands': [], 'attrs': {'value': 0.125}, 'regions': []}, {'op': 'add', 'results': [{'name': 'e10', 'type': {'dtype': 'float16', 'shape': (32, 16)}}], 'operands': ['e8', 'e9'], 'attrs': {}, 'regions': []}, {'op': 'barrier', 'results': [], 'operands': [], 'attrs': {}, 'regions': []}, {'op': 'store', 'results': [], 'operands': ['e5', 'e7', 'e10'], 'attrs': {'buffer': 'mem3'}, 'regions': []}, {'op': 'barrier', 'results': [], 'operands': [], 'attrs': {}, 'regions': []}, {'op': 'constant', 'results': [{'name': 'e11', 'type': {'dtype': 'bool', 'shape': (32, 16)}}], 'operands': [], 'attrs': {'value': True}, 'regions': []}, {'op': 'load', 'results': [{'name': 'e12', 'type': {'dtype': 'float16', 'shape': (32, 16)}}], 'operands': ['e5', 'e11'], 'attrs': {'buffer': 'mem2'}, 'regions': []}, {'op': 'constant', 'results': [{'name': 'e13', 'type': {'dtype': 'int32', 'shape': (32, 16)}}], 'operands': [], 'attrs': {'value': 1}, 'regions': []}, {'op': 'bitxor', 'results': [{'name': 'e14', 'type': {'dtype': 'int32', 'shape': (32, 16)}}], 'operands': ['e5', 'e13'], 'attrs': {}, 'regions': []}, {'op': 'load', 'results': [{'name': 'e15', 'type': {'dtype': 'float16', 'shape': (32, 16)}}], 'operands': ['e14', 'e7'], 'attrs': {'buffer': 'mem3'}, 'regions': []}, {'op': 'add', 'results': [{'name': 'e16', 'type': {'dtype': 'float16', 'shape': (32, 16)}}], 'operands': ['e12', 'e15'], 'attrs': {}, 'regions': []}, {'op': 'mul', 'results': [{'name': 'e17', 'type': {'dtype': 'float16', 'shape': (32, 16)}}], 'operands': ['e16', 'e9'], 'attrs': {}, 'regions': []}], 'returns': ['e17']}, 'buffers': [{'name': 'mem0', 'dtype': 'float16', 'size': 512, 'role': 'input', 'base': None, 'offset': 0, 'stride': 1}, {'name': 'mem1', 'dtype': 'float16', 'size': 1028, 'role': 'scratch', 'base': None, 'offset': 0, 'stride': 1}, {'name': 'mem2', 'dtype': 'float16', 'size': 512, 'role': 'scratch', 'base': 'mem1', 'offset': 1, 'stride': 2}, {'name': 'mem3', 'dtype': 'float16', 'size': 512, 'role': 'scratch', 'base': 'mem1', 'offset': 2, 'stride': 1}], 'functions': [], 'observations': ['e3', 'e8', 'e12', 'e13', 'e2'], 'blocks': 3, 'input_pattern': 'boundary', 'runtime_cases': [(0, 1), (1, 511), (3, 512)], 'configuration_pair': True, 'observation_pair': True, 'family': 'indexed_memory'}

if __name__ == '__main__':
    if os.environ.get('TILESMITH_COMPILE_ONLY', '0') == '1':
        prepare_extended(device_compile=False)
        extended_stage('complete', 'compile_only')
        print('COMPILE PASSED (no GPU execution)')
    else:
        run_extended(PROGRAM, prepare_extended, 0, 3)
        extended_stage('complete', 'execute')
        print('ALL PASSED')
