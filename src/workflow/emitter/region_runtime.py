"""Independent interpreter for the executable structured tile subset."""


def _region_reference(A, B, body, block_m, block_n, output_dtype, functions=()):
    import torch
    m = A.shape[0]
    n = B.shape[1] if body['operations'][0]['kind'] == 'gemm' else A.shape[1]
    pm, pn = (m + block_m - 1) // block_m * block_m, (n + block_n - 1) // block_n * block_n
    by = torch.arange(pm, device=A.device) // block_m
    bx = torch.arange(pn, device=A.device) // block_n
    function_pool = {fn['name']: fn['body'] for fn in functions}
    def padded(x):
        return torch.nn.functional.pad(x, (0, pn - n, 0, pm - m))
    def tile_index(axis, iteration):
        if axis == 'row':
            return by[:, None]
        if axis == 'column':
            return bx[None, :]
        if axis == 'checkerboard':
            return by[:, None] + bx[None, :]
        return torch.tensor(iteration, device=A.device)
    def execute(region, parent, argument=None, iteration=0):
        values = dict(parent)
        if region['arguments']:
            values[region['arguments'][0]] = argument
        for op in region['operations']:
            kind, attrs = op['kind'], op['attrs']
            args = [values[v] for v in op['operands']]
            if kind == 'load':
                result = padded(A.float())
            elif kind == 'gemm':
                # int8 x int8 accumulates in int32. torch.matmul does not
                # implement integer dtypes, so promote to fp64: 8-bit
                # products and their sums stay far below 2^53, making the
                # fp64 matmul an exact int32 reference on device.
                if A.dtype == torch.int8:
                    result = padded((A.to(torch.float64) @ B.to(torch.float64)).to(torch.int32))
                else:
                    result = padded(A.float() @ B.float())
            elif kind == 'call':
                callee = function_pool[attrs['callee']]
                # A callee sees only its own formal arguments, never caller locals.
                bindings = dict(zip(callee['arguments'], args))
                result = execute({**callee, 'arguments': []}, bindings, iteration=iteration)
            elif kind == 'index_add':
                index = tile_index(attrs['axis'], iteration)
                result = args[0] + index * attrs['scale']
            elif kind == 'scale':
                result = args[0] * attrs['alpha']
            elif kind == 'neg':
                result = -args[0]
            elif kind == 'abs':
                result = args[0].abs()
            elif kind == 'sqrt':
                result = args[0].abs().sqrt()
            elif kind == 'round':
                result = args[0].to(getattr(torch, attrs['dtype'])).float()
            elif kind == 'add':
                result = args[0] + args[1]
            elif kind == 'mul':
                result = args[0] * args[1]
            elif kind == 'copy':
                result = args[0].clone()
            elif kind == 'exp':
                result = args[0].clamp(-10, 10).exp()
            elif kind == 'tanh':
                result = args[0].tanh()
            elif kind == 'erf':
                result = torch.erf(args[0])
            elif kind == 'log':
                result = args[0].abs().clamp_min(1e-3).log()
            elif kind == 'log2':
                result = args[0].abs().clamp_min(1e-3).log2()
            elif kind == 'exp2':
                result = args[0].clamp(-10, 10).exp2()
            elif kind == 'rsqrt':
                result = args[0].abs().clamp_min(1e-6).rsqrt()
            elif kind == 'sin':
                result = args[0].sin()
            elif kind == 'cos':
                result = args[0].cos()
            elif kind == 'floor':
                result = args[0].floor()
            elif kind == 'ceil':
                result = args[0].ceil()
            elif kind == 'sub':
                result = args[0] - args[1]
            elif kind == 'maximum':
                result = torch.maximum(args[0], args[1])
            elif kind == 'minimum':
                result = torch.minimum(args[0], args[1])
            elif kind == 'div':
                # Defined division avoids trivial zero-denominator failures.
                result = args[0] / args[1].abs().clamp_min(1e-3)
            elif kind == 'where':
                result = torch.where(args[0] > 0, args[1], args[2])
            elif kind == 'tile_transpose':
                result = args[0].reshape(pm//block_m, block_m, pn//block_n, block_n).permute(0, 3, 2, 1).reshape(pm, pn)
            elif kind.startswith('row_'):
                tiles = args[0].reshape(pm, pn//block_n, block_n)
                if kind == 'row_softmax':
                    result = tiles.softmax(-1).reshape(pm, pn)
                else:
                    fn = {'row_sum': 'sum', 'row_max': 'amax', 'row_min': 'amin'}[kind]
                    result = getattr(tiles, fn)(-1, keepdim=True).expand_as(tiles).reshape(pm, pn)
            elif kind == 'for':
                result = args[0]
                for i in range(attrs['trip_count']):
                    index = attrs.get('start', 0) + i * attrs.get('step', 1)
                    result = execute(op['regions'][0], values, result, iteration=index)
            elif kind == 'if':
                left = execute(op['regions'][0], values, args[0], iteration)
                right = execute(op['regions'][1], values, args[0], iteration)
                index = tile_index(attrs.get('predicate', 'row'), iteration)
                result = torch.where(index % attrs.get('modulus', 2) == attrs['parity'], left, right)
            else:
                raise ValueError('Unsupported region reference operation: ' + kind)
            values[op['result']] = result
        return values[region['yield_value']]
    return execute(body, {})[:m, :n].to(getattr(torch, output_dtype))
