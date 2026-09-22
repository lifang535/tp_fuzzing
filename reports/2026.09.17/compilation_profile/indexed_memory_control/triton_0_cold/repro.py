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
def extended_0_0(mem0, mem1, out_e17, steps, limit):
    bid = tl.program_id(0)
    e1 = ((((511 - (tl.arange(0, 32)[:, None] * 16 + tl.arange(0, 16)[None, :])) + 25) % 512)).to(tl.int32)
    e2 = (tl.full((32, 16), True, tl.int1)).to(tl.int1)
    e3 = (tl.load(mem0 + bid * 544 + 16 + e1 * 1, (e2 & (e1 >= 0) & (e1 < 512)), other=0)).to(tl.float16)
    e4 = ((e3 * e3)).to(tl.float16)
    e5 = ((((511 - (tl.arange(0, 32)[:, None] * 16 + tl.arange(0, 16)[None, :])) + 89) % 512)).to(tl.int32)
    e6 = (limit).to(tl.int32)
    e7 = ((e5 < e6)).to(tl.int1)
    tl.store(mem1 + bid * 1060 + 17 + e5 * 2, e4, (e7 & (e5 >= 0) & (e5 < 512)))
    tl.debug_barrier()
    e8 = (tl.load(mem1 + bid * 1060 + 17 + e5 * 2, (e7 & (e5 >= 0) & (e5 < 512)), other=0)).to(tl.float16)
    e9 = (tl.full((32, 16), 0.125, tl.float16)).to(tl.float16)
    e10 = ((e8 + e9)).to(tl.float16)
    tl.debug_barrier()
    tl.store(mem1 + bid * 1060 + 18 + e5 * 1, e10, (e7 & (e5 >= 0) & (e5 < 512)))
    tl.debug_barrier()
    e11 = (tl.full((32, 16), True, tl.int1)).to(tl.int1)
    e12 = (tl.load(mem1 + bid * 1060 + 17 + e5 * 2, (e11 & (e5 >= 0) & (e5 < 512)), other=0)).to(tl.float16)
    e13 = (tl.full((32, 16), 1, tl.int32)).to(tl.int32)
    e14 = ((e5 ^ e13)).to(tl.int32)
    e15 = (tl.load(mem1 + bid * 1060 + 18 + e14 * 1, (e7 & (e14 >= 0) & (e14 < 512)), other=0)).to(tl.float16)
    e16 = ((e12 + e15)).to(tl.float16)
    e17 = ((e16 * e9)).to(tl.float16)
    tl.store(out_e17 + bid * 544 + 16 + (tl.arange(0, 32)[:, None] * 16 + tl.arange(0, 16)[None, :]), e17)

@triton.jit
def extended_0_1(mem0, mem1, out_e17, out_e3, out_e8, out_e12, out_e13, out_e2, steps, limit):
    bid = tl.program_id(0)
    e1 = ((((511 - (tl.arange(0, 32)[:, None] * 16 + tl.arange(0, 16)[None, :])) + 25) % 512)).to(tl.int32)
    e2 = (tl.full((32, 16), True, tl.int1)).to(tl.int1)
    e3 = (tl.load(mem0 + bid * 544 + 16 + e1 * 1, (e2 & (e1 >= 0) & (e1 < 512)), other=0)).to(tl.float16)
    e4 = ((e3 * e3)).to(tl.float16)
    e5 = ((((511 - (tl.arange(0, 32)[:, None] * 16 + tl.arange(0, 16)[None, :])) + 89) % 512)).to(tl.int32)
    e6 = (limit).to(tl.int32)
    e7 = ((e5 < e6)).to(tl.int1)
    tl.store(mem1 + bid * 1060 + 17 + e5 * 2, e4, (e7 & (e5 >= 0) & (e5 < 512)))
    tl.debug_barrier()
    e8 = (tl.load(mem1 + bid * 1060 + 17 + e5 * 2, (e7 & (e5 >= 0) & (e5 < 512)), other=0)).to(tl.float16)
    e9 = (tl.full((32, 16), 0.125, tl.float16)).to(tl.float16)
    e10 = ((e8 + e9)).to(tl.float16)
    tl.debug_barrier()
    tl.store(mem1 + bid * 1060 + 18 + e5 * 1, e10, (e7 & (e5 >= 0) & (e5 < 512)))
    tl.debug_barrier()
    e11 = (tl.full((32, 16), True, tl.int1)).to(tl.int1)
    e12 = (tl.load(mem1 + bid * 1060 + 17 + e5 * 2, (e11 & (e5 >= 0) & (e5 < 512)), other=0)).to(tl.float16)
    e13 = (tl.full((32, 16), 1, tl.int32)).to(tl.int32)
    e14 = ((e5 ^ e13)).to(tl.int32)
    e15 = (tl.load(mem1 + bid * 1060 + 18 + e14 * 1, (e7 & (e14 >= 0) & (e14 < 512)), other=0)).to(tl.float16)
    e16 = ((e12 + e15)).to(tl.float16)
    e17 = ((e16 * e9)).to(tl.float16)
    tl.store(out_e17 + bid * 544 + 16 + (tl.arange(0, 32)[:, None] * 16 + tl.arange(0, 16)[None, :]), e17)
    tl.store(out_e3 + bid * 544 + 16 + (tl.arange(0, 32)[:, None] * 16 + tl.arange(0, 16)[None, :]), e3)
    tl.store(out_e8 + bid * 544 + 16 + (tl.arange(0, 32)[:, None] * 16 + tl.arange(0, 16)[None, :]), e8)
    tl.store(out_e12 + bid * 544 + 16 + (tl.arange(0, 32)[:, None] * 16 + tl.arange(0, 16)[None, :]), e12)
    tl.store(out_e13 + bid * 544 + 16 + (tl.arange(0, 32)[:, None] * 16 + tl.arange(0, 16)[None, :]), e13)
    tl.store(out_e2 + bid * 544 + 16 + (tl.arange(0, 32)[:, None] * 16 + tl.arange(0, 16)[None, :]), e2)

@triton.jit
def extended_1_0(mem0, mem1, out_e17, steps, limit):
    bid = tl.program_id(0)
    e1 = ((((511 - (tl.arange(0, 32)[:, None] * 16 + tl.arange(0, 16)[None, :])) + 25) % 512)).to(tl.int32)
    e2 = (tl.full((32, 16), True, tl.int1)).to(tl.int1)
    e3 = (tl.load(mem0 + bid * 544 + 16 + e1 * 1, (e2 & (e1 >= 0) & (e1 < 512)), other=0)).to(tl.float16)
    e4 = ((e3 * e3)).to(tl.float16)
    e5 = ((((511 - (tl.arange(0, 32)[:, None] * 16 + tl.arange(0, 16)[None, :])) + 89) % 512)).to(tl.int32)
    e6 = (limit).to(tl.int32)
    e7 = ((e5 < e6)).to(tl.int1)
    tl.store(mem1 + bid * 1060 + 17 + e5 * 2, e4, (e7 & (e5 >= 0) & (e5 < 512)))
    tl.debug_barrier()
    e8 = (tl.load(mem1 + bid * 1060 + 17 + e5 * 2, (e7 & (e5 >= 0) & (e5 < 512)), other=0)).to(tl.float16)
    e9 = (tl.full((32, 16), 0.125, tl.float16)).to(tl.float16)
    e10 = ((e8 + e9)).to(tl.float16)
    tl.debug_barrier()
    tl.store(mem1 + bid * 1060 + 18 + e5 * 1, e10, (e7 & (e5 >= 0) & (e5 < 512)))
    tl.debug_barrier()
    e11 = (tl.full((32, 16), True, tl.int1)).to(tl.int1)
    e12 = (tl.load(mem1 + bid * 1060 + 17 + e5 * 2, (e11 & (e5 >= 0) & (e5 < 512)), other=0)).to(tl.float16)
    e13 = (tl.full((32, 16), 1, tl.int32)).to(tl.int32)
    e14 = ((e5 ^ e13)).to(tl.int32)
    e15 = (tl.load(mem1 + bid * 1060 + 18 + e14 * 1, (e7 & (e14 >= 0) & (e14 < 512)), other=0)).to(tl.float16)
    e16 = ((e12 + e15)).to(tl.float16)
    e17 = ((e16 * e9)).to(tl.float16)
    tl.store(out_e17 + bid * 544 + 16 + (tl.arange(0, 32)[:, None] * 16 + tl.arange(0, 16)[None, :]), e17)

@triton.jit
def extended_1_1(mem0, mem1, out_e17, out_e3, out_e8, out_e12, out_e13, out_e2, steps, limit):
    bid = tl.program_id(0)
    e1 = ((((511 - (tl.arange(0, 32)[:, None] * 16 + tl.arange(0, 16)[None, :])) + 25) % 512)).to(tl.int32)
    e2 = (tl.full((32, 16), True, tl.int1)).to(tl.int1)
    e3 = (tl.load(mem0 + bid * 544 + 16 + e1 * 1, (e2 & (e1 >= 0) & (e1 < 512)), other=0)).to(tl.float16)
    e4 = ((e3 * e3)).to(tl.float16)
    e5 = ((((511 - (tl.arange(0, 32)[:, None] * 16 + tl.arange(0, 16)[None, :])) + 89) % 512)).to(tl.int32)
    e6 = (limit).to(tl.int32)
    e7 = ((e5 < e6)).to(tl.int1)
    tl.store(mem1 + bid * 1060 + 17 + e5 * 2, e4, (e7 & (e5 >= 0) & (e5 < 512)))
    tl.debug_barrier()
    e8 = (tl.load(mem1 + bid * 1060 + 17 + e5 * 2, (e7 & (e5 >= 0) & (e5 < 512)), other=0)).to(tl.float16)
    e9 = (tl.full((32, 16), 0.125, tl.float16)).to(tl.float16)
    e10 = ((e8 + e9)).to(tl.float16)
    tl.debug_barrier()
    tl.store(mem1 + bid * 1060 + 18 + e5 * 1, e10, (e7 & (e5 >= 0) & (e5 < 512)))
    tl.debug_barrier()
    e11 = (tl.full((32, 16), True, tl.int1)).to(tl.int1)
    e12 = (tl.load(mem1 + bid * 1060 + 17 + e5 * 2, (e11 & (e5 >= 0) & (e5 < 512)), other=0)).to(tl.float16)
    e13 = (tl.full((32, 16), 1, tl.int32)).to(tl.int32)
    e14 = ((e5 ^ e13)).to(tl.int32)
    e15 = (tl.load(mem1 + bid * 1060 + 18 + e14 * 1, (e7 & (e14 >= 0) & (e14 < 512)), other=0)).to(tl.float16)
    e16 = ((e12 + e15)).to(tl.float16)
    e17 = ((e16 * e9)).to(tl.float16)
    tl.store(out_e17 + bid * 544 + 16 + (tl.arange(0, 32)[:, None] * 16 + tl.arange(0, 16)[None, :]), e17)
    tl.store(out_e3 + bid * 544 + 16 + (tl.arange(0, 32)[:, None] * 16 + tl.arange(0, 16)[None, :]), e3)
    tl.store(out_e8 + bid * 544 + 16 + (tl.arange(0, 32)[:, None] * 16 + tl.arange(0, 16)[None, :]), e8)
    tl.store(out_e12 + bid * 544 + 16 + (tl.arange(0, 32)[:, None] * 16 + tl.arange(0, 16)[None, :]), e12)
    tl.store(out_e13 + bid * 544 + 16 + (tl.arange(0, 32)[:, None] * 16 + tl.arange(0, 16)[None, :]), e13)
    tl.store(out_e2 + bid * 544 + 16 + (tl.arange(0, 32)[:, None] * 16 + tl.arange(0, 16)[None, :]), e2)

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
    compiled_0 = compile(ASTSource(extended_0_0, {0: '*fp16', 1: '*fp16', 2: '*fp16', 3: 'i32', 4: 'i32'}), target=GPUTarget("cuda", arch, 32), options={'num_warps': 4, 'num_stages': 1, 'enable_fp_fusion': False})
    record_extended_compilation('triton_0', compiled_0.asm, {'num_warps': 4, 'num_stages': 1, 'enable_fp_fusion': False})
    def launch_0(memories, outputs, steps, limit):
        arguments = [memories[n] for n in ['mem0', 'mem1']] + [outputs[n] for n in ['e17']]
        compiled_0[(3, 1, 1)](*arguments, steps, limit)
    variants.append(('triton_0', ['e17'], launch_0))
    extended_stage("compile", 'triton_1')
    compiled_1 = compile(ASTSource(extended_0_1, {0: '*fp16', 1: '*fp16', 2: '*fp16', 3: '*fp16', 4: '*fp16', 5: '*fp16', 6: '*i32', 7: '*i1', 8: 'i32', 9: 'i32'}), target=GPUTarget("cuda", arch, 32), options={'num_warps': 4, 'num_stages': 1, 'enable_fp_fusion': False})
    record_extended_compilation('triton_1', compiled_1.asm, {'num_warps': 4, 'num_stages': 1, 'enable_fp_fusion': False})
    def launch_1(memories, outputs, steps, limit):
        arguments = [memories[n] for n in ['mem0', 'mem1']] + [outputs[n] for n in ['e17', 'e3', 'e8', 'e12', 'e13', 'e2']]
        compiled_1[(3, 1, 1)](*arguments, steps, limit)
    variants.append(('triton_1', ['e17', 'e3', 'e8', 'e12', 'e13', 'e2'], launch_1))
    extended_stage("compile", 'triton_2')
    compiled_2 = compile(ASTSource(extended_1_0, {0: '*fp16', 1: '*fp16', 2: '*fp16', 3: 'i32', 4: 'i32'}), target=GPUTarget("cuda", arch, 32), options={'num_warps': 8, 'num_stages': 2, 'enable_fp_fusion': False})
    record_extended_compilation('triton_2', compiled_2.asm, {'num_warps': 8, 'num_stages': 2, 'enable_fp_fusion': False})
    def launch_2(memories, outputs, steps, limit):
        arguments = [memories[n] for n in ['mem0', 'mem1']] + [outputs[n] for n in ['e17']]
        compiled_2[(3, 1, 1)](*arguments, steps, limit)
    variants.append(('triton_2', ['e17'], launch_2))
    extended_stage("compile", 'triton_3')
    compiled_3 = compile(ASTSource(extended_1_1, {0: '*fp16', 1: '*fp16', 2: '*fp16', 3: '*fp16', 4: '*fp16', 5: '*fp16', 6: '*i32', 7: '*i1', 8: 'i32', 9: 'i32'}), target=GPUTarget("cuda", arch, 32), options={'num_warps': 8, 'num_stages': 2, 'enable_fp_fusion': False})
    record_extended_compilation('triton_3', compiled_3.asm, {'num_warps': 8, 'num_stages': 2, 'enable_fp_fusion': False})
    def launch_3(memories, outputs, steps, limit):
        arguments = [memories[n] for n in ['mem0', 'mem1']] + [outputs[n] for n in ['e17', 'e3', 'e8', 'e12', 'e13', 'e2']]
        compiled_3[(3, 1, 1)](*arguments, steps, limit)
    variants.append(('triton_3', ['e17', 'e3', 'e8', 'e12', 'e13', 'e2'], launch_3))
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
