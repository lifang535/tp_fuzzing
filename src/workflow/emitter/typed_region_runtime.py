"""Independent batched-tile interpreter with masked memory side effects."""


def _typed_region_reference(A, B, body, block_m, block_n, output_dtype, functions=()):
    import torch
    m, n = A.shape[0], B.shape[1] if body['operations'][0]['kind'] == 'gemm' else A.shape[1]
    tm, tn = (m + block_m - 1) // block_m, (n + block_n - 1) // block_n
    by = torch.arange(tm, device=A.device)[:, None, None, None]
    bx = torch.arange(tn, device=A.device)[None, :, None, None]
    pool = {fn['name']: fn['body'] for fn in functions}
    memory = {}

    def tiles(x):
        x = torch.nn.functional.pad(x, (0, tn * block_n - n, 0, tm * block_m - m))
        return x.reshape(tm, block_m, tn, block_n).permute(0, 2, 1, 3)

    def index(axis, iteration):
        return {'row': by, 'column': bx, 'checkerboard': by + bx, 'iteration': iteration}[axis]

    def read_input(attrs):
        source = A if attrs['source'] == 'A' else B
        rows = by * block_m + torch.arange(block_m, device=A.device)[None, None, :, None] + attrs.get('row_offset', 0)
        cols = bx * block_n + torch.arange(block_n, device=A.device)[None, None, None, :] + attrs.get('col_offset', 0)
        valid = (rows < source.shape[0]) & (cols < source.shape[1])
        data = source[rows.clamp_max(source.shape[0] - 1), cols.clamp_max(source.shape[1] - 1)]
        return torch.where(valid, data, 0).to(getattr(torch, attrs.get('dtype', 'float32')))

    def execute(region, inherited, scope, argument=None, iteration=0, active=True):
        values = dict(inherited)
        if region['arguments']:
            values[region['arguments'][0]] = argument
        for op in region['operations']:
            kind, attrs = op['kind'], op['attrs']
            args = [values[v] for v in op['operands']]
            if kind == 'load':
                result = tiles(A.float())
            elif kind == 'gemm':
                result = tiles(A.float() @ B.float())
            elif kind == 'load_input':
                result = read_input(attrs)
            elif kind == 'store_tile':
                slot = (scope, op['result'])
                old = memory.get(slot, torch.zeros_like(args[0]))
                memory[slot] = torch.where(torch.as_tensor(active, device=A.device), args[0], old)
                result = slot
            elif kind == 'write_tile':
                memory[args[0]] = torch.where(torch.as_tensor(active, device=A.device), args[1], memory[args[0]])
                result = args[0]
            elif kind == 'load_tile':
                result = memory[args[0]].clone()
            elif kind == 'call':
                callee = pool[attrs['callee']]
                bindings = dict(zip(callee['arguments'], args))
                result = execute({**callee, 'arguments': []}, bindings, attrs['callee'], iteration=iteration, active=active)
            elif kind == 'if':
                condition = index(attrs.get('predicate', 'row'), iteration) % attrs.get('modulus', 2) == attrs['parity']
                condition = torch.as_tensor(condition, device=A.device)
                left = execute(op['regions'][0], values, scope, args[0], iteration, active & condition)
                right = execute(op['regions'][1], values, scope, args[0], iteration, active & ~condition)
                result = torch.where(condition, left, right)
            elif kind == 'for':
                result = args[0]
                for i in range(attrs['trip_count']):
                    result = execute(op['regions'][0], values, scope, result,
                                     attrs.get('start', 0) + i * attrs.get('step', 1), active)
            elif kind == 'cast':
                result = args[0].to(getattr(torch, attrs['dtype']))
            elif kind in ('broadcast_tile', 'to_tile'):
                result = args[0].expand(tm, tn, block_m, block_n)
                if kind == 'to_tile':
                    result = result.float()
            elif kind == 'reduce_tile':
                fn = {'sum': 'sum', 'max': 'amax', 'min': 'amin'}[attrs['reduction']]
                result = getattr(args[0].float(), fn)(attrs['axis'] + 2, keepdim=True)
            elif kind == 'tile_transpose':
                result = args[0].transpose(-1, -2)
            elif kind.startswith('row_'):
                x = args[0].float()
                if kind == 'row_softmax':
                    result = x.softmax(-1)
                else:
                    fn = {'row_sum': 'sum', 'row_max': 'amax', 'row_min': 'amin'}[kind]
                    result = getattr(x, fn)(-1, keepdim=True).expand_as(x)
            else:
                # Half arithmetic has an explicit rounding point at every SSA
                # definition; this is independent of the backend expression text.
                x = args[0].float()
                y = args[1].float() if len(args) > 1 else None
                if kind == 'scale': result = x * attrs['alpha']
                elif kind == 'neg': result = -x
                elif kind == 'abs': result = x.abs()
                elif kind == 'sqrt': result = x.abs().sqrt()
                elif kind == 'round': result = x.to(getattr(torch, attrs['dtype'])).float()
                elif kind == 'copy': result = x.clone()
                elif kind == 'exp': result = x.clamp(-10, 10).exp()
                elif kind == 'tanh': result = x.tanh()
                elif kind == 'erf': result = torch.erf(x)
                elif kind == 'log': result = x.abs().clamp_min(1e-3).log()
                elif kind == 'log2': result = x.abs().clamp_min(1e-3).log2()
                elif kind == 'exp2': result = x.clamp(-10, 10).exp2()
                elif kind == 'rsqrt': result = x.abs().clamp_min(1e-6).rsqrt()
                elif kind == 'sin': result = x.sin()
                elif kind == 'cos': result = x.cos()
                elif kind == 'floor': result = x.floor()
                elif kind == 'ceil': result = x.ceil()
                elif kind == 'add': result = x + y
                elif kind == 'mul': result = x * y
                elif kind == 'sub': result = x - y
                elif kind == 'maximum': result = torch.maximum(x, y)
                elif kind == 'minimum': result = torch.minimum(x, y)
                elif kind == 'div': result = x / y.abs().clamp_min(0.001)
                elif kind == 'where': result = torch.where(x > 0, y, args[2].float())
                elif kind == 'index_add': result = x + index(attrs['axis'], iteration) * attrs['scale']
                else: raise ValueError('Unknown typed operation: ' + kind)
                result = result.to(args[0].dtype)
            values[op['result']] = result
        return values[region['yield_value']]

    output = execute(body, {}, 'main')
    return output.permute(0, 2, 1, 3).reshape(tm * block_m, tn * block_n)[:m, :n].to(getattr(torch, output_dtype))
