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
    @T.macro
    def mixed_helper_extended_0_0(e56, e57, mem0, mem1, mem2, mem3, mem6, mem7, mem8, steps, limit, bid, result_e63, result_e60):
        e58 = T.alloc_fragment((16,), "float32")
        e59 = T.alloc_fragment((16,), "float32")
        e60 = T.alloc_fragment((1,), "float32")
        e61 = T.alloc_fragment((16, 1), "float32")
        e62 = T.alloc_fragment((16, 1), "float16")
        e63 = T.alloc_fragment((16, 16), "float16")
        e58_wide = T.alloc_fragment((16, 16), "float32")
        e60_wide = T.alloc_fragment((16,), "float32")
        for i, j in T.Parallel(16, 16):
            e58_wide[i, j] = T.cast(e56[i, j], 'float32')
        T.reduce_sum(e58_wide, e58, dim=1, clear=True)
        for i in T.Parallel(16):
            e59[i] = T.cast((e58[i] + e57[i]), "float32")
        for i in T.Parallel(16):
            e60_wide[i] = T.cast(e59[i], 'float32')
        T.reduce_sum(e60_wide, e60, dim=0, clear=True)
        for i, j in T.Parallel(16, 1):
            e61[i, 0] = T.cast(e59[(i * 1 + j)], "float32")
        for i, j in T.Parallel(16, 1):
            e62[i, 0] = T.cast(e61[i, 0], "float16")
        for i, j in T.Parallel(16, 16):
            e63[i, j] = T.cast((e56[i, j] + e62[i, 0]), "float16")
        for i, j in T.Parallel(16, 16):
            result_e63[i, j] = e63[i, j]
        result_e60[0] = e60[0]
    @T.prim_func
    def impl(mem0: T.Buffer((2, 544), "float32"), mem1: T.Buffer((2, 544), "float32"), mem2: T.Buffer((2, 544), "int32"), mem3: T.Buffer((2, 1060), "float16"), mem6: T.Buffer((2, 288), "float16"), mem7: T.Buffer((2, 33), "int32"), mem8: T.Buffer((2, 288), "float16"), out_e122: T.Buffer((2, 288), "float16"), out_e18: T.Buffer((2, 544), "int32"), out_e24: T.Buffer((2, 544), "bool"), out_e115: T.Buffer((2, 33), "float32"), out_e116: T.Buffer((2, 33), "int32"), steps: T.int32, limit: T.int32):
        with T.Kernel(2, threads=32) as bid:
            e1 = T.alloc_fragment((32, 16), "int32")
            e2 = T.alloc_fragment((32, 16), "bool")
            e3 = T.alloc_fragment((32, 16), "float32")
            e4 = T.alloc_fragment((32, 16), "float32")
            e5 = T.alloc_fragment((32, 16), "int32")
            e6 = T.alloc_fragment((32, 16), "bool")
            e7 = T.alloc_fragment((32, 16), "float32")
            e8 = T.alloc_fragment((32, 16), "float32")
            e9 = T.alloc_fragment((32, 16), "float16")
            e10 = T.alloc_fragment((32, 16), "float16")
            e11 = T.alloc_fragment((32, 16), "float16")
            e12 = T.alloc_fragment((32, 16), "float16")
            e13 = T.alloc_fragment((32, 16), "float16")
            e14 = T.alloc_fragment((32, 16), "int32")
            e15 = T.alloc_fragment((32, 16), "bool")
            e16 = T.alloc_fragment((32, 16), "int32")
            e17 = T.alloc_fragment((32, 16), "int32")
            e18 = T.alloc_fragment((32, 16), "int32")
            e19 = T.alloc_fragment((32, 16), "int32")
            e20 = T.alloc_fragment((32, 16), "bool")
            e21 = T.alloc_fragment((32, 16), "int32")
            e22 = T.alloc_fragment((1,), "int32")
            e23 = T.alloc_fragment((32, 16), "bool")
            e24 = T.alloc_fragment((32, 16), "bool")
            e25 = T.alloc_fragment((32, 16), "float16")
            e26 = T.alloc_fragment((32, 16), "float16")
            e27 = T.alloc_fragment((32, 16), "int32")
            e28 = T.alloc_fragment((1,), "int32")
            e29 = T.alloc_fragment((32, 16), "bool")
            e30 = T.alloc_fragment((32, 16), "float16")
            e31 = T.alloc_fragment((32, 16), "float16")
            e32 = T.alloc_fragment((32, 16), "float16")
            e33 = T.alloc_fragment((32, 16), "bool")
            e34 = T.alloc_fragment((32, 16), "float16")
            e35 = T.alloc_fragment((32, 16), "int32")
            e36 = T.alloc_fragment((32, 16), "int32")
            e37 = T.alloc_fragment((32, 16), "float16")
            e38 = T.alloc_fragment((32, 16), "float16")
            e39 = T.alloc_fragment((32, 16), "float16")
            e40 = T.alloc_fragment((16, 32), "float16")
            e41 = T.alloc_fragment((16, 16), "float32")
            e42 = T.alloc_fragment((16, 16), "float32")
            e43 = T.alloc_fragment((16,), "float32")
            e44 = T.alloc_fragment((16, 1), "float32")
            e45 = T.alloc_fragment((16, 16), "float32")
            e46 = T.alloc_fragment((16, 16), "float32")
            e47 = T.alloc_fragment((16, 16), "float32")
            e48 = T.alloc_fragment((16, 16), "int32")
            e49 = T.alloc_fragment((16, 16), "bool")
            e50 = T.alloc_fragment((16, 16), "float16")
            e51 = T.alloc_fragment((16, 16), "float16")
            e52 = T.alloc_fragment((16, 16), "float32")
            e53 = T.alloc_fragment((16, 16), "float32")
            e54 = T.alloc_fragment((16, 16), "float32")
            e55 = T.alloc_fragment((16, 16), "float16")
            e64 = T.alloc_fragment((16, 16), "float16")
            e65 = T.alloc_fragment((1,), "float32")
            e66 = T.alloc_fragment((1,), "int32")
            e67 = T.alloc_fragment((1,), "int32")
            e68 = T.alloc_fragment((1,), "int32")
            e69 = T.alloc_fragment((1,), "bool")
            e70 = T.alloc_fragment((1,), "int32")
            e71 = T.alloc_fragment((1,), "int32")
            e72 = T.alloc_fragment((1,), "bool")
            e92 = T.alloc_fragment((16, 16), "float16")
            e93 = T.alloc_fragment((1,), "float32")
            e94 = T.alloc_fragment((1,), "int32")
            e83 = T.alloc_fragment((16, 16), "float16")
            e78 = T.alloc_fragment((16, 16), "float16")
            e79 = T.alloc_fragment((16, 16), "float16")
            e81 = T.alloc_fragment((16, 16), "float16")
            e82 = T.alloc_fragment((16, 16), "float16")
            e84 = T.alloc_fragment((16, 16), "float16")
            e85 = T.alloc_fragment((1,), "float32")
            e86 = T.alloc_fragment((16, 16), "float16")
            e87 = T.alloc_fragment((16, 16), "float16")
            e88 = T.alloc_fragment((1,), "float32")
            e89 = T.alloc_fragment((1,), "int32")
            e90 = T.alloc_fragment((1,), "int32")
            e91 = T.alloc_fragment((1,), "int32")
            e114 = T.alloc_fragment((16, 16), "float16")
            e115 = T.alloc_fragment((1,), "float32")
            e116 = T.alloc_fragment((1,), "int32")
            e105 = T.alloc_fragment((16, 16), "float16")
            e100 = T.alloc_fragment((16, 16), "float16")
            e101 = T.alloc_fragment((16, 16), "float16")
            e103 = T.alloc_fragment((16, 16), "float16")
            e104 = T.alloc_fragment((16, 16), "float16")
            e106 = T.alloc_fragment((16, 16), "float16")
            e107 = T.alloc_fragment((1,), "float32")
            e108 = T.alloc_fragment((16, 16), "float16")
            e109 = T.alloc_fragment((16, 16), "float16")
            e110 = T.alloc_fragment((1,), "float32")
            e111 = T.alloc_fragment((1,), "int32")
            e112 = T.alloc_fragment((1,), "int32")
            e113 = T.alloc_fragment((1,), "int32")
            e117 = T.alloc_fragment((16, 16), "float16")
            e118 = T.alloc_fragment((16, 16), "int32")
            e119 = T.alloc_fragment((16, 16), "bool")
            e120 = T.alloc_fragment((16, 16), "float16")
            e121 = T.alloc_fragment((16, 16), "float16")
            e122 = T.alloc_fragment((16, 16), "float16")
            e73 = T.alloc_fragment((1,), "int32")
            e74 = T.alloc_fragment((16, 16), "float16")
            e75 = T.alloc_fragment((1,), "float32")
            e76 = T.alloc_fragment((1,), "int32")
            e77 = T.alloc_fragment((16, 16), "float16")
            e80 = T.alloc_fragment((16, 16), "float16")
            e95 = T.alloc_fragment((1,), "int32")
            e96 = T.alloc_fragment((16, 16), "float16")
            e97 = T.alloc_fragment((1,), "float32")
            e98 = T.alloc_fragment((1,), "int32")
            e99 = T.alloc_fragment((16, 16), "float16")
            e102 = T.alloc_fragment((16, 16), "float16")
            e42_a_shared = T.alloc_shared((16, 32), "float16")
            e42_b_shared = T.alloc_shared((32, 16), "float16")
            e43_wide = T.alloc_fragment((16, 16), "float32")
            e43_nan = T.alloc_fragment((16, 16), "int32")
            e43_nan_count = T.alloc_fragment((16,), "int32")
            e52_a_shared = T.alloc_shared((16, 16), "float16")
            e52_b_shared = T.alloc_shared((16, 16), "float16")
            e114_iteration = T.alloc_var("int32")
            for i, j in T.Parallel(32, 16):
                e1[i, j] = T.cast((((511 - (i * 16 + j)) + 44) % 512), "int32")
            for i, j in T.Parallel(32, 16):
                e2[i, j] = T.cast(True, "bool")
            for i, j in T.Parallel(32, 16):
                e3[i, j] = T.cast(T.if_then_else((e2[i, j] and e1[i, j] >= 0 and e1[i, j] < 512), mem0[bid, 16 + e1[i, j] * 1], T.cast(0, "float32")), "float32")
            for i, j in T.Parallel(32, 16):
                e4[i, j] = T.cast((e3[i, j] + e3[i, j]), "float32")
            for i, j in T.Parallel(32, 16):
                e5[i, j] = T.cast((((i * 16 + j) + 370) % 512), "int32")
            for i, j in T.Parallel(32, 16):
                e6[i, j] = T.cast(True, "bool")
            for i, j in T.Parallel(32, 16):
                e7[i, j] = T.cast(T.if_then_else((e6[i, j] and e5[i, j] >= 0 and e5[i, j] < 512), mem1[bid, 16 + e5[i, j] * 1], T.cast(0, "float32")), "float32")
            for i, j in T.Parallel(32, 16):
                e8[i, j] = T.cast((e4[i, j] * e7[i, j]), "float32")
            for i, j in T.Parallel(32, 16):
                e9[i, j] = T.cast(e8[i, j], "float16")
            for i, j in T.Parallel(32, 16):
                e10[i, j] = T.cast((e9[i, j] + e9[i, j]), "float16")
            for i, j in T.Parallel(32, 16):
                e11[i, j] = T.cast(e10[i, j], "float16")
            for i, j in T.Parallel(32, 16):
                e12[i, j] = T.cast(0.5, "float16")
            for i, j in T.Parallel(32, 16):
                e13[i, j] = T.cast((e11[i, j] * e12[i, j]), "float16")
            for i, j in T.Parallel(32, 16):
                e14[i, j] = T.cast((((511 - (i * 16 + j)) + 389) % 512), "int32")
            for i, j in T.Parallel(32, 16):
                e15[i, j] = T.cast(True, "bool")
            for i, j in T.Parallel(32, 16):
                e16[i, j] = T.cast(T.if_then_else((e15[i, j] and e14[i, j] >= 0 and e14[i, j] < 512), mem2[bid, 16 + e14[i, j] * 1], T.cast(0, "int32")), "int32")
            for i, j in T.Parallel(32, 16):
                e17[i, j] = T.cast(7, "int32")
            for i, j in T.Parallel(32, 16):
                e18[i, j] = T.cast((e16[i, j] ^ e17[i, j]), "int32")
            for i, j in T.Parallel(32, 16):
                e19[i, j] = T.cast(3, "int32")
            for i, j in T.Parallel(32, 16):
                e20[i, j] = T.cast((e18[i, j] < e19[i, j]), "bool")
            for i, j in T.Parallel(32, 16):
                e21[i, j] = T.cast((((511 - (i * 16 + j)) + 166) % 512), "int32")
            e22[0] = T.cast(limit, "int32")
            for i, j in T.Parallel(32, 16):
                e23[i, j] = T.cast((e21[i, j] < e22[0]), "bool")
            for i, j in T.Parallel(32, 16):
                e24[i, j] = T.cast((e20[i, j] and e23[i, j]), "bool")
            for i, j in T.Parallel(32, 16):
                e25[i, j] = T.cast(-0.125, "float16")
            for i, j in T.Parallel(32, 16):
                e26[i, j] = T.cast(T.if_then_else(e24[i, j], e13[i, j], e25[i, j]), "float16")
            for i, j in T.Parallel(32, 16):
                e27[i, j] = T.cast((((i * 16 + j) + 363) % 512), "int32")
            e28[0] = T.cast(limit, "int32")
            for i, j in T.Parallel(32, 16):
                e29[i, j] = T.cast((e27[i, j] < e28[0]), "bool")
            for i, j in T.Parallel(32, 16):
                if (e29[i, j] and e27[i, j] >= 0 and e27[i, j] < 512):
                    mem3[bid, 17 + e27[i, j] * 2] = e26[i, j]
            T.sync_threads()
            for i, j in T.Parallel(32, 16):
                e30[i, j] = T.cast(T.if_then_else((e29[i, j] and e27[i, j] >= 0 and e27[i, j] < 512), mem3[bid, 17 + e27[i, j] * 2], T.cast(0, "float16")), "float16")
            for i, j in T.Parallel(32, 16):
                e31[i, j] = T.cast(0.125, "float16")
            for i, j in T.Parallel(32, 16):
                e32[i, j] = T.cast((e30[i, j] + e31[i, j]), "float16")
            T.sync_threads()
            for i, j in T.Parallel(32, 16):
                if (e29[i, j] and e27[i, j] >= 0 and e27[i, j] < 512):
                    mem3[bid, 18 + e27[i, j] * 1] = e32[i, j]
            T.sync_threads()
            for i, j in T.Parallel(32, 16):
                e33[i, j] = T.cast(True, "bool")
            for i, j in T.Parallel(32, 16):
                e34[i, j] = T.cast(T.if_then_else((e33[i, j] and e27[i, j] >= 0 and e27[i, j] < 512), mem3[bid, 17 + e27[i, j] * 2], T.cast(0, "float16")), "float16")
            for i, j in T.Parallel(32, 16):
                e35[i, j] = T.cast(7, "int32")
            for i, j in T.Parallel(32, 16):
                e36[i, j] = T.cast((e27[i, j] ^ e35[i, j]), "int32")
            for i, j in T.Parallel(32, 16):
                e37[i, j] = T.cast(T.if_then_else((e29[i, j] and e36[i, j] >= 0 and e36[i, j] < 512), mem3[bid, 18 + e36[i, j] * 1], T.cast(0, "float16")), "float16")
            for i, j in T.Parallel(32, 16):
                e38[i, j] = T.cast((e34[i, j] + e37[i, j]), "float16")
            for i, j in T.Parallel(32, 16):
                e39[i, j] = T.cast(e38[i, j], "float16")
            for i, j in T.Parallel(16, 32):
                e40[i, j] = T.cast(e39[j, i], "float16")
            for i, j in T.Parallel(16, 16):
                e41[i, j] = T.cast(0.0, "float32")
            T.copy(e40, e42_a_shared)
            T.copy(e34, e42_b_shared)
            T.copy(e41, e42)
            T.gemm(e42_a_shared, e42_b_shared, e42)
            for i, j in T.Parallel(16, 16):
                e43_wide[i, j] = T.cast(e42[i, j], 'float32')
            T.reduce_max(e43_wide, e43, dim=1, clear=True)
            for i, j in T.Parallel(16, 16):
                e43_nan[i, j] = T.if_then_else(T.isnan(e43_wide[i, j]), 1, 0)
            T.reduce_sum(e43_nan, e43_nan_count, dim=1, clear=True)
            for i in T.Parallel(16):
                e43[i] = T.if_then_else(e43_nan_count[i] > 0, T.cast(float('nan'), 'float32'), e43[i])
            for i, j in T.Parallel(16, 1):
                e44[i, 0] = T.cast(e43[(i * 1 + j)], "float32")
            for i, j in T.Parallel(16, 16):
                e45[i, j] = T.cast((e42[i, j] - e44[i, 0]), "float32")
            for i, j in T.Parallel(16, 16):
                e46[i, j] = T.cast(0.125, "float32")
            for i, j in T.Parallel(16, 16):
                e47[i, j] = T.cast((e45[i, j] * e46[i, j]), "float32")
            for i, j in T.Parallel(16, 16):
                e48[i, j] = T.cast((((i * 16 + j) + 87) % 256), "int32")
            for i, j in T.Parallel(16, 16):
                e49[i, j] = T.cast(True, "bool")
            for i, j in T.Parallel(16, 16):
                e50[i, j] = T.cast(T.if_then_else((e49[i, j] and e48[i, j] >= 0 and e48[i, j] < 256), mem6[bid, 16 + e48[i, j] * 1], T.cast(0, "float16")), "float16")
            for i, j in T.Parallel(16, 16):
                e51[i, j] = T.cast(e47[i, j], "float16")
            T.copy(e51, e52_a_shared)
            T.copy(e50, e52_b_shared)
            T.copy(e41, e52)
            T.gemm(e52_a_shared, e52_b_shared, e52)
            for i, j in T.Parallel(16, 16):
                e53[i, j] = T.cast(e52[((i * 16 + j)) // 16, ((i * 16 + j)) % 16], "float32")
            for i, j in T.Parallel(16, 16):
                e54[i, j] = T.cast(e53[((i * 16 + j)) // 16, ((i * 16 + j)) % 16], "float32")
            for i, j in T.Parallel(16, 16):
                e55[i, j] = T.cast(e54[i, j], "float16")
            mixed_helper_extended_0_0(e55, e43, mem0, mem1, mem2, mem3, mem6, mem7, mem8, steps, limit, bid, e64, e65)
            e66[0] = T.cast(steps, "int32")
            e67[0] = T.cast(0, "int32")
            e68[0] = T.cast(0, "int32")
            e69[0] = T.cast(True, "bool")
            e70[0] = T.cast(T.if_then_else((e69[0] and e68[0] >= 0 and e68[0] < 1), mem7[bid, 16 + e68[0] * 1], T.cast(0, "int32")), "int32")
            e71[0] = T.cast(0, "int32")
            e72[0] = T.cast((e70[0] < e71[0]), "bool")
            for i, j in T.Parallel(16, 16):
                e92[i, j] = e64[i, j]
            e93[0] = e65[0]
            e94[0] = e67[0]
            for e92_iteration in T.Pipelined(T.min(T.max(e66[0], 0), 4), num_stages=1):
                e73[0] = e92_iteration
                for i, j in T.Parallel(16, 16):
                    e74[i, j] = e92[i, j]
                e75[0] = e93[0]
                e76[0] = e94[0]
                if e72[0]:
                    for i, j in T.Parallel(16, 16):
                        e77[i, j] = e74[i, j]
                    for i, j in T.Parallel(16, 16):
                        e78[i, j] = T.cast(0.125, "float16")
                    for i, j in T.Parallel(16, 16):
                        e79[i, j] = T.cast((e77[i, j] + e78[i, j]), "float16")
                    for i, j in T.Parallel(16, 16):
                        e83[i, j] = e79[i, j]
                else:
                    for i, j in T.Parallel(16, 16):
                        e80[i, j] = e74[i, j]
                    for i, j in T.Parallel(16, 16):
                        e81[i, j] = T.cast(0.125, "float16")
                    for i, j in T.Parallel(16, 16):
                        e82[i, j] = T.cast((e80[i, j] - e81[i, j]), "float16")
                    for i, j in T.Parallel(16, 16):
                        e83[i, j] = e82[i, j]
                mixed_helper_extended_0_0(e83, e43, mem0, mem1, mem2, mem3, mem6, mem7, mem8, steps, limit, bid, e84, e85)
                for i, j in T.Parallel(16, 16):
                    e86[i, j] = T.cast(0.125, "float16")
                for i, j in T.Parallel(16, 16):
                    e87[i, j] = T.cast((e84[i, j] * e86[i, j]), "float16")
                e88[0] = T.cast((e75[0] + e85[0]), "float32")
                e89[0] = T.cast(1, "int32")
                e90[0] = T.cast((e73[0] + e89[0]), "int32")
                e91[0] = T.cast((e76[0] + e90[0]), "int32")
                for i, j in T.Parallel(16, 16):
                    e92[i, j] = e87[i, j]
                e93[0] = e88[0]
                e94[0] = e91[0]
            for i, j in T.Parallel(16, 16):
                e114[i, j] = e92[i, j]
            e115[0] = e93[0]
            e116[0] = e94[0]
            e114_iteration = 0
            while e114_iteration < T.min(T.max(e66[0], 0), 4):
                e95[0] = e114_iteration
                for i, j in T.Parallel(16, 16):
                    e96[i, j] = e114[i, j]
                e97[0] = e115[0]
                e98[0] = e116[0]
                if e72[0]:
                    for i, j in T.Parallel(16, 16):
                        e99[i, j] = e96[i, j]
                    for i, j in T.Parallel(16, 16):
                        e100[i, j] = T.cast(0.125, "float16")
                    for i, j in T.Parallel(16, 16):
                        e101[i, j] = T.cast((e99[i, j] + e100[i, j]), "float16")
                    for i, j in T.Parallel(16, 16):
                        e105[i, j] = e101[i, j]
                else:
                    for i, j in T.Parallel(16, 16):
                        e102[i, j] = e96[i, j]
                    for i, j in T.Parallel(16, 16):
                        e103[i, j] = T.cast(0.125, "float16")
                    for i, j in T.Parallel(16, 16):
                        e104[i, j] = T.cast((e102[i, j] - e103[i, j]), "float16")
                    for i, j in T.Parallel(16, 16):
                        e105[i, j] = e104[i, j]
                mixed_helper_extended_0_0(e105, e43, mem0, mem1, mem2, mem3, mem6, mem7, mem8, steps, limit, bid, e106, e107)
                for i, j in T.Parallel(16, 16):
                    e108[i, j] = T.cast(0.125, "float16")
                for i, j in T.Parallel(16, 16):
                    e109[i, j] = T.cast((e106[i, j] * e108[i, j]), "float16")
                e110[0] = T.cast((e97[0] + e107[0]), "float32")
                e111[0] = T.cast(1, "int32")
                e112[0] = T.cast((e95[0] + e111[0]), "int32")
                e113[0] = T.cast((e98[0] + e112[0]), "int32")
                for i, j in T.Parallel(16, 16):
                    e114[i, j] = e109[i, j]
                e115[0] = e110[0]
                e116[0] = e113[0]
                e114_iteration += 1
            for i, j in T.Parallel(16, 16):
                e117[i, j] = T.cast((e114[i, j] + e50[i, j]), "float16")
            for i, j in T.Parallel(16, 16):
                e118[i, j] = T.cast((((i * 16 + j) + 33) % 256), "int32")
            for i, j in T.Parallel(16, 16):
                e119[i, j] = T.cast(True, "bool")
            for i, j in T.Parallel(16, 16):
                e120[i, j] = T.cast(T.if_then_else((e119[i, j] and e118[i, j] >= 0 and e118[i, j] < 256), mem8[bid, 16 + e118[i, j] * 1], T.cast(0, "float16")), "float16")
            for i, j in T.Parallel(16, 16):
                e121[i, j] = T.cast((e117[i, j] * e120[i, j]), "float16")
            for i, j in T.Parallel(16, 16):
                e122[i, j] = T.cast((e121[i, j] - e120[i, j]), "float16")
            for i, j in T.Parallel(16, 16):
                out_e122[bid, 16 + (i * 16 + j)] = e122[i, j]
            for i, j in T.Parallel(32, 16):
                out_e18[bid, 16 + (i * 16 + j)] = e18[i, j]
            for i, j in T.Parallel(32, 16):
                out_e24[bid, 16 + (i * 16 + j)] = e24[i, j]
            out_e115[bid, 16 + 0] = e115[0]
            out_e116[bid, 16 + 0] = e116[0]
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
    record_extended_compilation('tilelang_0', {"tir": str(ir_0)}, {'threads': 32, 'stages': 1, 'pass_configs': {}}, complete=False)
    extended_stage("device_compile", 'tilelang_0')
    if device_compile:
        compiled_0 = tilelang.compile(ir_0, target=target, pass_configs={})
        artifact_0 = compiled_0.artifact
        artifacts_0 = {"cuda": compiled_0.get_kernel_source()}
        if artifact_0 is not None:
            artifacts_0["lowered_tir"] = str(artifact_0.device_mod)
        record_extended_compilation('tilelang_0', artifacts_0, {'threads': 32, 'stages': 1, 'pass_configs': {}})
        def launch_0(memories, outputs, steps, limit):
            arguments = [memories[n] for n in ['mem0', 'mem1', 'mem2', 'mem3', 'mem6', 'mem7', 'mem8']] + [outputs[n] for n in ['e122', 'e18', 'e24', 'e115', 'e116']]
            compiled_0(*arguments, steps, limit)
        variants.append(('tilelang_0', ['e122', 'e18', 'e24', 'e115', 'e116'], launch_0))
    else:
        with target, tvm.transform.PassContext(opt_level=3, config={}):
            lowered_0 = tilelang_lower(ir_0, target=target, enable_device_compile=False)
        record_extended_compilation('tilelang_0', {"lowered_tir": str(lowered_0.device_mod), "cuda": lowered_0.kernel_source}, {'threads': 32, 'stages': 1, 'pass_configs': {}}, complete=False)
        from tilelang.engine.lower import tilelang_callback_cuda_compile
        cubin_0 = tilelang_callback_cuda_compile(lowered_0.kernel_source, target, {})
        record_extended_compilation('tilelang_0', {"cubin": cubin_0}, {'threads': 32, 'stages': 1, 'pass_configs': {}})
    return variants

PROGRAM = {'type': 'extended', 'version': 1, 'body': {'arguments': [], 'operations': [{'op': 'index', 'results': [{'name': 'e1', 'type': {'dtype': 'int32', 'shape': (32, 16)}}], 'operands': [], 'attrs': {'reverse': True, 'shift': 44}, 'regions': []}, {'op': 'constant', 'results': [{'name': 'e2', 'type': {'dtype': 'bool', 'shape': (32, 16)}}], 'operands': [], 'attrs': {'value': True}, 'regions': []}, {'op': 'load', 'results': [{'name': 'e3', 'type': {'dtype': 'float32', 'shape': (32, 16)}}], 'operands': ['e1', 'e2'], 'attrs': {'buffer': 'mem0'}, 'regions': []}, {'op': 'add', 'results': [{'name': 'e4', 'type': {'dtype': 'float32', 'shape': (32, 16)}}], 'operands': ['e3', 'e3'], 'attrs': {}, 'regions': []}, {'op': 'index', 'results': [{'name': 'e5', 'type': {'dtype': 'int32', 'shape': (32, 16)}}], 'operands': [], 'attrs': {'reverse': False, 'shift': 370}, 'regions': []}, {'op': 'constant', 'results': [{'name': 'e6', 'type': {'dtype': 'bool', 'shape': (32, 16)}}], 'operands': [], 'attrs': {'value': True}, 'regions': []}, {'op': 'load', 'results': [{'name': 'e7', 'type': {'dtype': 'float32', 'shape': (32, 16)}}], 'operands': ['e5', 'e6'], 'attrs': {'buffer': 'mem1'}, 'regions': []}, {'op': 'mul', 'results': [{'name': 'e8', 'type': {'dtype': 'float32', 'shape': (32, 16)}}], 'operands': ['e4', 'e7'], 'attrs': {}, 'regions': []}, {'op': 'cast', 'results': [{'name': 'e9', 'type': {'dtype': 'float16', 'shape': (32, 16)}}], 'operands': ['e8'], 'attrs': {}, 'regions': []}, {'op': 'add', 'results': [{'name': 'e10', 'type': {'dtype': 'float16', 'shape': (32, 16)}}], 'operands': ['e9', 'e9'], 'attrs': {}, 'regions': []}, {'op': 'cast', 'results': [{'name': 'e11', 'type': {'dtype': 'float16', 'shape': (32, 16)}}], 'operands': ['e10'], 'attrs': {}, 'regions': []}, {'op': 'constant', 'results': [{'name': 'e12', 'type': {'dtype': 'float16', 'shape': (32, 16)}}], 'operands': [], 'attrs': {'value': 0.5}, 'regions': []}, {'op': 'mul', 'results': [{'name': 'e13', 'type': {'dtype': 'float16', 'shape': (32, 16)}}], 'operands': ['e11', 'e12'], 'attrs': {}, 'regions': []}, {'op': 'index', 'results': [{'name': 'e14', 'type': {'dtype': 'int32', 'shape': (32, 16)}}], 'operands': [], 'attrs': {'reverse': True, 'shift': 389}, 'regions': []}, {'op': 'constant', 'results': [{'name': 'e15', 'type': {'dtype': 'bool', 'shape': (32, 16)}}], 'operands': [], 'attrs': {'value': True}, 'regions': []}, {'op': 'load', 'results': [{'name': 'e16', 'type': {'dtype': 'int32', 'shape': (32, 16)}}], 'operands': ['e14', 'e15'], 'attrs': {'buffer': 'mem2'}, 'regions': []}, {'op': 'constant', 'results': [{'name': 'e17', 'type': {'dtype': 'int32', 'shape': (32, 16)}}], 'operands': [], 'attrs': {'value': 7}, 'regions': []}, {'op': 'bitxor', 'results': [{'name': 'e18', 'type': {'dtype': 'int32', 'shape': (32, 16)}}], 'operands': ['e16', 'e17'], 'attrs': {}, 'regions': []}, {'op': 'constant', 'results': [{'name': 'e19', 'type': {'dtype': 'int32', 'shape': (32, 16)}}], 'operands': [], 'attrs': {'value': 3}, 'regions': []}, {'op': 'lt', 'results': [{'name': 'e20', 'type': {'dtype': 'bool', 'shape': (32, 16)}}], 'operands': ['e18', 'e19'], 'attrs': {}, 'regions': []}, {'op': 'index', 'results': [{'name': 'e21', 'type': {'dtype': 'int32', 'shape': (32, 16)}}], 'operands': [], 'attrs': {'reverse': True, 'shift': 166}, 'regions': []}, {'op': 'parameter', 'results': [{'name': 'e22', 'type': {'dtype': 'int32', 'shape': ()}}], 'operands': [], 'attrs': {'name': 'limit'}, 'regions': []}, {'op': 'lt', 'results': [{'name': 'e23', 'type': {'dtype': 'bool', 'shape': (32, 16)}}], 'operands': ['e21', 'e22'], 'attrs': {}, 'regions': []}, {'op': 'and', 'results': [{'name': 'e24', 'type': {'dtype': 'bool', 'shape': (32, 16)}}], 'operands': ['e20', 'e23'], 'attrs': {}, 'regions': []}, {'op': 'constant', 'results': [{'name': 'e25', 'type': {'dtype': 'float16', 'shape': (32, 16)}}], 'operands': [], 'attrs': {'value': -0.125}, 'regions': []}, {'op': 'select', 'results': [{'name': 'e26', 'type': {'dtype': 'float16', 'shape': (32, 16)}}], 'operands': ['e24', 'e13', 'e25'], 'attrs': {}, 'regions': []}, {'op': 'index', 'results': [{'name': 'e27', 'type': {'dtype': 'int32', 'shape': (32, 16)}}], 'operands': [], 'attrs': {'reverse': False, 'shift': 363}, 'regions': []}, {'op': 'parameter', 'results': [{'name': 'e28', 'type': {'dtype': 'int32', 'shape': ()}}], 'operands': [], 'attrs': {'name': 'limit'}, 'regions': []}, {'op': 'lt', 'results': [{'name': 'e29', 'type': {'dtype': 'bool', 'shape': (32, 16)}}], 'operands': ['e27', 'e28'], 'attrs': {}, 'regions': []}, {'op': 'store', 'results': [], 'operands': ['e27', 'e29', 'e26'], 'attrs': {'buffer': 'mem4'}, 'regions': []}, {'op': 'barrier', 'results': [], 'operands': [], 'attrs': {}, 'regions': []}, {'op': 'load', 'results': [{'name': 'e30', 'type': {'dtype': 'float16', 'shape': (32, 16)}}], 'operands': ['e27', 'e29'], 'attrs': {'buffer': 'mem4'}, 'regions': []}, {'op': 'constant', 'results': [{'name': 'e31', 'type': {'dtype': 'float16', 'shape': (32, 16)}}], 'operands': [], 'attrs': {'value': 0.125}, 'regions': []}, {'op': 'add', 'results': [{'name': 'e32', 'type': {'dtype': 'float16', 'shape': (32, 16)}}], 'operands': ['e30', 'e31'], 'attrs': {}, 'regions': []}, {'op': 'barrier', 'results': [], 'operands': [], 'attrs': {}, 'regions': []}, {'op': 'store', 'results': [], 'operands': ['e27', 'e29', 'e32'], 'attrs': {'buffer': 'mem5'}, 'regions': []}, {'op': 'barrier', 'results': [], 'operands': [], 'attrs': {}, 'regions': []}, {'op': 'constant', 'results': [{'name': 'e33', 'type': {'dtype': 'bool', 'shape': (32, 16)}}], 'operands': [], 'attrs': {'value': True}, 'regions': []}, {'op': 'load', 'results': [{'name': 'e34', 'type': {'dtype': 'float16', 'shape': (32, 16)}}], 'operands': ['e27', 'e33'], 'attrs': {'buffer': 'mem4'}, 'regions': []}, {'op': 'constant', 'results': [{'name': 'e35', 'type': {'dtype': 'int32', 'shape': (32, 16)}}], 'operands': [], 'attrs': {'value': 7}, 'regions': []}, {'op': 'bitxor', 'results': [{'name': 'e36', 'type': {'dtype': 'int32', 'shape': (32, 16)}}], 'operands': ['e27', 'e35'], 'attrs': {}, 'regions': []}, {'op': 'load', 'results': [{'name': 'e37', 'type': {'dtype': 'float16', 'shape': (32, 16)}}], 'operands': ['e36', 'e29'], 'attrs': {'buffer': 'mem5'}, 'regions': []}, {'op': 'add', 'results': [{'name': 'e38', 'type': {'dtype': 'float16', 'shape': (32, 16)}}], 'operands': ['e34', 'e37'], 'attrs': {}, 'regions': []}, {'op': 'cast', 'results': [{'name': 'e39', 'type': {'dtype': 'float16', 'shape': (32, 16)}}], 'operands': ['e38'], 'attrs': {}, 'regions': []}, {'op': 'transpose', 'results': [{'name': 'e40', 'type': {'dtype': 'float16', 'shape': (16, 32)}}], 'operands': ['e39'], 'attrs': {}, 'regions': []}, {'op': 'constant', 'results': [{'name': 'e41', 'type': {'dtype': 'float32', 'shape': (16, 16)}}], 'operands': [], 'attrs': {'value': 0.0}, 'regions': []}, {'op': 'matmul', 'results': [{'name': 'e42', 'type': {'dtype': 'float32', 'shape': (16, 16)}}], 'operands': ['e40', 'e34', 'e41'], 'attrs': {}, 'regions': []}, {'op': 'reduce', 'results': [{'name': 'e43', 'type': {'dtype': 'float32', 'shape': (16,)}}], 'operands': ['e42'], 'attrs': {'axis': 1, 'kind': 'max'}, 'regions': []}, {'op': 'reshape', 'results': [{'name': 'e44', 'type': {'dtype': 'float32', 'shape': (16, 1)}}], 'operands': ['e43'], 'attrs': {}, 'regions': []}, {'op': 'sub', 'results': [{'name': 'e45', 'type': {'dtype': 'float32', 'shape': (16, 16)}}], 'operands': ['e42', 'e44'], 'attrs': {}, 'regions': []}, {'op': 'constant', 'results': [{'name': 'e46', 'type': {'dtype': 'float32', 'shape': (16, 16)}}], 'operands': [], 'attrs': {'value': 0.125}, 'regions': []}, {'op': 'mul', 'results': [{'name': 'e47', 'type': {'dtype': 'float32', 'shape': (16, 16)}}], 'operands': ['e45', 'e46'], 'attrs': {}, 'regions': []}, {'op': 'index', 'results': [{'name': 'e48', 'type': {'dtype': 'int32', 'shape': (16, 16)}}], 'operands': [], 'attrs': {'reverse': False, 'shift': 87}, 'regions': []}, {'op': 'constant', 'results': [{'name': 'e49', 'type': {'dtype': 'bool', 'shape': (16, 16)}}], 'operands': [], 'attrs': {'value': True}, 'regions': []}, {'op': 'load', 'results': [{'name': 'e50', 'type': {'dtype': 'float16', 'shape': (16, 16)}}], 'operands': ['e48', 'e49'], 'attrs': {'buffer': 'mem6'}, 'regions': []}, {'op': 'cast', 'results': [{'name': 'e51', 'type': {'dtype': 'float16', 'shape': (16, 16)}}], 'operands': ['e47'], 'attrs': {}, 'regions': []}, {'op': 'matmul', 'results': [{'name': 'e52', 'type': {'dtype': 'float32', 'shape': (16, 16)}}], 'operands': ['e51', 'e50', 'e41'], 'attrs': {}, 'regions': []}, {'op': 'reshape', 'results': [{'name': 'e53', 'type': {'dtype': 'float32', 'shape': (16, 16)}}], 'operands': ['e52'], 'attrs': {}, 'regions': []}, {'op': 'reshape', 'results': [{'name': 'e54', 'type': {'dtype': 'float32', 'shape': (16, 16)}}], 'operands': ['e53'], 'attrs': {}, 'regions': []}, {'op': 'cast', 'results': [{'name': 'e55', 'type': {'dtype': 'float16', 'shape': (16, 16)}}], 'operands': ['e54'], 'attrs': {}, 'regions': []}, {'op': 'call', 'results': [{'name': 'e64', 'type': {'dtype': 'float16', 'shape': (16, 16)}}, {'name': 'e65', 'type': {'dtype': 'float32', 'shape': ()}}], 'operands': ['e55', 'e43'], 'attrs': {'callee': 'mixed_helper'}, 'regions': []}, {'op': 'parameter', 'results': [{'name': 'e66', 'type': {'dtype': 'int32', 'shape': ()}}], 'operands': [], 'attrs': {'name': 'steps'}, 'regions': []}, {'op': 'constant', 'results': [{'name': 'e67', 'type': {'dtype': 'int32', 'shape': ()}}], 'operands': [], 'attrs': {'value': 0}, 'regions': []}, {'op': 'constant', 'results': [{'name': 'e68', 'type': {'dtype': 'int32', 'shape': ()}}], 'operands': [], 'attrs': {'value': 0}, 'regions': []}, {'op': 'constant', 'results': [{'name': 'e69', 'type': {'dtype': 'bool', 'shape': ()}}], 'operands': [], 'attrs': {'value': True}, 'regions': []}, {'op': 'load', 'results': [{'name': 'e70', 'type': {'dtype': 'int32', 'shape': ()}}], 'operands': ['e68', 'e69'], 'attrs': {'buffer': 'mem7'}, 'regions': []}, {'op': 'constant', 'results': [{'name': 'e71', 'type': {'dtype': 'int32', 'shape': ()}}], 'operands': [], 'attrs': {'value': 0}, 'regions': []}, {'op': 'lt', 'results': [{'name': 'e72', 'type': {'dtype': 'bool', 'shape': ()}}], 'operands': ['e70', 'e71'], 'attrs': {}, 'regions': []}, {'op': 'for', 'results': [{'name': 'e92', 'type': {'dtype': 'float16', 'shape': (16, 16)}}, {'name': 'e93', 'type': {'dtype': 'float32', 'shape': ()}}, {'name': 'e94', 'type': {'dtype': 'int32', 'shape': ()}}], 'operands': ['e66', 'e64', 'e65', 'e67'], 'attrs': {'max_steps': 4, 'pipelined': True}, 'regions': [{'arguments': [{'name': 'e73', 'type': {'dtype': 'int32', 'shape': ()}}, {'name': 'e74', 'type': {'dtype': 'float16', 'shape': (16, 16)}}, {'name': 'e75', 'type': {'dtype': 'float32', 'shape': ()}}, {'name': 'e76', 'type': {'dtype': 'int32', 'shape': ()}}], 'operations': [{'op': 'if', 'results': [{'name': 'e83', 'type': {'dtype': 'float16', 'shape': (16, 16)}}], 'operands': ['e72', 'e74'], 'attrs': {}, 'regions': [{'arguments': [{'name': 'e77', 'type': {'dtype': 'float16', 'shape': (16, 16)}}], 'operations': [{'op': 'constant', 'results': [{'name': 'e78', 'type': {'dtype': 'float16', 'shape': (16, 16)}}], 'operands': [], 'attrs': {'value': 0.125}, 'regions': []}, {'op': 'add', 'results': [{'name': 'e79', 'type': {'dtype': 'float16', 'shape': (16, 16)}}], 'operands': ['e77', 'e78'], 'attrs': {}, 'regions': []}], 'returns': ['e79']}, {'arguments': [{'name': 'e80', 'type': {'dtype': 'float16', 'shape': (16, 16)}}], 'operations': [{'op': 'constant', 'results': [{'name': 'e81', 'type': {'dtype': 'float16', 'shape': (16, 16)}}], 'operands': [], 'attrs': {'value': 0.125}, 'regions': []}, {'op': 'sub', 'results': [{'name': 'e82', 'type': {'dtype': 'float16', 'shape': (16, 16)}}], 'operands': ['e80', 'e81'], 'attrs': {}, 'regions': []}], 'returns': ['e82']}]}, {'op': 'call', 'results': [{'name': 'e84', 'type': {'dtype': 'float16', 'shape': (16, 16)}}, {'name': 'e85', 'type': {'dtype': 'float32', 'shape': ()}}], 'operands': ['e83', 'e43'], 'attrs': {'callee': 'mixed_helper'}, 'regions': []}, {'op': 'constant', 'results': [{'name': 'e86', 'type': {'dtype': 'float16', 'shape': (16, 16)}}], 'operands': [], 'attrs': {'value': 0.125}, 'regions': []}, {'op': 'mul', 'results': [{'name': 'e87', 'type': {'dtype': 'float16', 'shape': (16, 16)}}], 'operands': ['e84', 'e86'], 'attrs': {}, 'regions': []}, {'op': 'add', 'results': [{'name': 'e88', 'type': {'dtype': 'float32', 'shape': ()}}], 'operands': ['e75', 'e85'], 'attrs': {}, 'regions': []}, {'op': 'constant', 'results': [{'name': 'e89', 'type': {'dtype': 'int32', 'shape': ()}}], 'operands': [], 'attrs': {'value': 1}, 'regions': []}, {'op': 'add', 'results': [{'name': 'e90', 'type': {'dtype': 'int32', 'shape': ()}}], 'operands': ['e73', 'e89'], 'attrs': {}, 'regions': []}, {'op': 'add', 'results': [{'name': 'e91', 'type': {'dtype': 'int32', 'shape': ()}}], 'operands': ['e76', 'e90'], 'attrs': {}, 'regions': []}], 'returns': ['e87', 'e88', 'e91']}]}, {'op': 'while', 'results': [{'name': 'e114', 'type': {'dtype': 'float16', 'shape': (16, 16)}}, {'name': 'e115', 'type': {'dtype': 'float32', 'shape': ()}}, {'name': 'e116', 'type': {'dtype': 'int32', 'shape': ()}}], 'operands': ['e66', 'e92', 'e93', 'e94'], 'attrs': {'max_steps': 4, 'pipelined': False}, 'regions': [{'arguments': [{'name': 'e95', 'type': {'dtype': 'int32', 'shape': ()}}, {'name': 'e96', 'type': {'dtype': 'float16', 'shape': (16, 16)}}, {'name': 'e97', 'type': {'dtype': 'float32', 'shape': ()}}, {'name': 'e98', 'type': {'dtype': 'int32', 'shape': ()}}], 'operations': [{'op': 'if', 'results': [{'name': 'e105', 'type': {'dtype': 'float16', 'shape': (16, 16)}}], 'operands': ['e72', 'e96'], 'attrs': {}, 'regions': [{'arguments': [{'name': 'e99', 'type': {'dtype': 'float16', 'shape': (16, 16)}}], 'operations': [{'op': 'constant', 'results': [{'name': 'e100', 'type': {'dtype': 'float16', 'shape': (16, 16)}}], 'operands': [], 'attrs': {'value': 0.125}, 'regions': []}, {'op': 'add', 'results': [{'name': 'e101', 'type': {'dtype': 'float16', 'shape': (16, 16)}}], 'operands': ['e99', 'e100'], 'attrs': {}, 'regions': []}], 'returns': ['e101']}, {'arguments': [{'name': 'e102', 'type': {'dtype': 'float16', 'shape': (16, 16)}}], 'operations': [{'op': 'constant', 'results': [{'name': 'e103', 'type': {'dtype': 'float16', 'shape': (16, 16)}}], 'operands': [], 'attrs': {'value': 0.125}, 'regions': []}, {'op': 'sub', 'results': [{'name': 'e104', 'type': {'dtype': 'float16', 'shape': (16, 16)}}], 'operands': ['e102', 'e103'], 'attrs': {}, 'regions': []}], 'returns': ['e104']}]}, {'op': 'call', 'results': [{'name': 'e106', 'type': {'dtype': 'float16', 'shape': (16, 16)}}, {'name': 'e107', 'type': {'dtype': 'float32', 'shape': ()}}], 'operands': ['e105', 'e43'], 'attrs': {'callee': 'mixed_helper'}, 'regions': []}, {'op': 'constant', 'results': [{'name': 'e108', 'type': {'dtype': 'float16', 'shape': (16, 16)}}], 'operands': [], 'attrs': {'value': 0.125}, 'regions': []}, {'op': 'mul', 'results': [{'name': 'e109', 'type': {'dtype': 'float16', 'shape': (16, 16)}}], 'operands': ['e106', 'e108'], 'attrs': {}, 'regions': []}, {'op': 'add', 'results': [{'name': 'e110', 'type': {'dtype': 'float32', 'shape': ()}}], 'operands': ['e97', 'e107'], 'attrs': {}, 'regions': []}, {'op': 'constant', 'results': [{'name': 'e111', 'type': {'dtype': 'int32', 'shape': ()}}], 'operands': [], 'attrs': {'value': 1}, 'regions': []}, {'op': 'add', 'results': [{'name': 'e112', 'type': {'dtype': 'int32', 'shape': ()}}], 'operands': ['e95', 'e111'], 'attrs': {}, 'regions': []}, {'op': 'add', 'results': [{'name': 'e113', 'type': {'dtype': 'int32', 'shape': ()}}], 'operands': ['e98', 'e112'], 'attrs': {}, 'regions': []}], 'returns': ['e109', 'e110', 'e113']}]}, {'op': 'add', 'results': [{'name': 'e117', 'type': {'dtype': 'float16', 'shape': (16, 16)}}], 'operands': ['e114', 'e50'], 'attrs': {}, 'regions': []}, {'op': 'index', 'results': [{'name': 'e118', 'type': {'dtype': 'int32', 'shape': (16, 16)}}], 'operands': [], 'attrs': {'reverse': False, 'shift': 33}, 'regions': []}, {'op': 'constant', 'results': [{'name': 'e119', 'type': {'dtype': 'bool', 'shape': (16, 16)}}], 'operands': [], 'attrs': {'value': True}, 'regions': []}, {'op': 'load', 'results': [{'name': 'e120', 'type': {'dtype': 'float16', 'shape': (16, 16)}}], 'operands': ['e118', 'e119'], 'attrs': {'buffer': 'mem8'}, 'regions': []}, {'op': 'mul', 'results': [{'name': 'e121', 'type': {'dtype': 'float16', 'shape': (16, 16)}}], 'operands': ['e117', 'e120'], 'attrs': {}, 'regions': []}, {'op': 'sub', 'results': [{'name': 'e122', 'type': {'dtype': 'float16', 'shape': (16, 16)}}], 'operands': ['e121', 'e120'], 'attrs': {}, 'regions': []}], 'returns': ['e122', 'e18', 'e24', 'e115', 'e116']}, 'buffers': [{'name': 'mem0', 'dtype': 'float32', 'size': 512, 'role': 'input', 'base': None, 'offset': 0, 'stride': 1}, {'name': 'mem1', 'dtype': 'float32', 'size': 512, 'role': 'input', 'base': None, 'offset': 0, 'stride': 1}, {'name': 'mem2', 'dtype': 'int32', 'size': 512, 'role': 'input', 'base': None, 'offset': 0, 'stride': 1}, {'name': 'mem3', 'dtype': 'float16', 'size': 1028, 'role': 'scratch', 'base': None, 'offset': 0, 'stride': 1}, {'name': 'mem4', 'dtype': 'float16', 'size': 512, 'role': 'scratch', 'base': 'mem3', 'offset': 1, 'stride': 2}, {'name': 'mem5', 'dtype': 'float16', 'size': 512, 'role': 'scratch', 'base': 'mem3', 'offset': 2, 'stride': 1}, {'name': 'mem6', 'dtype': 'float16', 'size': 256, 'role': 'input', 'base': None, 'offset': 0, 'stride': 1}, {'name': 'mem7', 'dtype': 'int32', 'size': 1, 'role': 'input', 'base': None, 'offset': 0, 'stride': 1}, {'name': 'mem8', 'dtype': 'float16', 'size': 256, 'role': 'input', 'base': None, 'offset': 0, 'stride': 1}], 'functions': [{'name': 'mixed_helper', 'body': {'arguments': [{'name': 'e56', 'type': {'dtype': 'float16', 'shape': (16, 16)}}, {'name': 'e57', 'type': {'dtype': 'float32', 'shape': (16,)}}], 'operations': [{'op': 'reduce', 'results': [{'name': 'e58', 'type': {'dtype': 'float32', 'shape': (16,)}}], 'operands': ['e56'], 'attrs': {'axis': 1, 'kind': 'sum'}, 'regions': []}, {'op': 'add', 'results': [{'name': 'e59', 'type': {'dtype': 'float32', 'shape': (16,)}}], 'operands': ['e58', 'e57'], 'attrs': {}, 'regions': []}, {'op': 'reduce', 'results': [{'name': 'e60', 'type': {'dtype': 'float32', 'shape': ()}}], 'operands': ['e59'], 'attrs': {'axis': 0, 'kind': 'sum'}, 'regions': []}, {'op': 'reshape', 'results': [{'name': 'e61', 'type': {'dtype': 'float32', 'shape': (16, 1)}}], 'operands': ['e59'], 'attrs': {}, 'regions': []}, {'op': 'cast', 'results': [{'name': 'e62', 'type': {'dtype': 'float16', 'shape': (16, 1)}}], 'operands': ['e61'], 'attrs': {}, 'regions': []}, {'op': 'add', 'results': [{'name': 'e63', 'type': {'dtype': 'float16', 'shape': (16, 16)}}], 'operands': ['e56', 'e62'], 'attrs': {}, 'regions': []}], 'returns': ['e63', 'e60']}}], 'observations': ['e3', 'e13', 'e30', 'e34', 'e42', 'e47', 'e19', 'e28'], 'blocks': 2, 'input_pattern': 'integer', 'runtime_cases': [(0, 1), (1, 511), (4, 512)], 'configuration_pair': False, 'observation_pair': False, 'family': 'mixed'}

if __name__ == '__main__':
    if os.environ.get('TILESMITH_COMPILE_ONLY', '0') == '1':
        prepare_extended(device_compile=False)
        extended_stage('complete', 'compile_only')
        print('COMPILE PASSED (no GPU execution)')
    else:
        run_extended(PROGRAM, prepare_extended, 0, 3)
        extended_stage('complete', 'execute')
        print('ALL PASSED')
