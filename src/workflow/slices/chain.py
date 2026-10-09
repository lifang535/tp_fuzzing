"""Elementwise steps shared by every slice: legalization, lowering, reference.

A step applies one operation in the current dtype and then converts to the
step's dtype. Legalization keeps each knob when the transferred domain stays
exactly representable and falls back to 'none' / no conversion otherwise, so
the stored parameters always describe the executed program.
"""
import math

from .dtypes import DTYPES, Domain, bit_bound

UNARY = ('abs', 'neg', 'half', 'floor', 'ceil')
BINARY = ('add', 'sub', 'mul', 'max', 'min', 'select', 'and', 'or', 'xor', 'shl', 'shr', 'div', 'mod')
OPS = ('none',) + UNARY + BINARY
FLOAT_ONLY = ('half', 'floor', 'ceil')
INT_ONLY = ('and', 'or', 'xor', 'shl', 'shr', 'div', 'mod')
# Per-step constants; the step index picks one so chains vary their operands.
CONSTANTS = {'add': (3, -1, 5), 'sub': (2, -3, 1), 'mul': (2, -1, 3), 'max': (1, -2, 0),
             'min': (-1, 2, 0), 'select': (0, 1, -1), 'and': (7, 6, 12), 'or': (5, 2, 9),
             'xor': (6, 3, 10), 'shl': (1, 2, 1), 'shr': (1, 2, 1), 'div': (3, -2, 4), 'mod': (3, 5, -3)}


def constant(op, index, dtype):
    c = CONSTANTS[op][index % 3]
    if DTYPES[dtype].kind == 'uint' or op in ('and', 'or', 'xor'):
        c = abs(c)
    if op in ('div', 'mod') and c == 0:
        c = 3
    return c


def op_allowed(op, dtype, backend=None):
    d = DTYPES[dtype]
    if op == 'none':
        return True
    if d.is_fp8:
        return False
    # Triton's tl.floor/tl.ceil are libdevice calls typed for fp32/fp64 only.
    if backend == 'triton' and op in ('floor', 'ceil') and dtype not in ('f32', 'f64'):
        return False
    if op in FLOAT_ONLY and not d.is_float:
        return False
    if op in INT_ONLY and d.is_float:
        return False
    if op in ('abs', 'neg') and d.kind == 'uint':
        return False
    return True


def transfer(op, domain, operand, semantics):
    """Domain of op(x, operand) for x in domain; semantics names the integer
    division rounding of the backend ('trunc' or 'floor')."""
    lo, hi, f = domain.lo, domain.hi, domain.frac
    if op == 'none':
        return domain
    if op == 'abs':
        low = 0 if lo <= 0 <= hi else min(abs(lo), abs(hi))
        return Domain(low, domain.magnitude, f)
    if op == 'neg':
        return Domain(-hi, -lo, f)
    if op == 'half':
        return Domain(lo / 2, hi / 2, f + 1)
    if op == 'floor':
        return Domain(math.floor(lo), math.floor(hi), 0)
    if op == 'ceil':
        return Domain(math.ceil(lo), math.ceil(hi), 0)
    o = operand
    if op == 'add':
        return Domain(lo + o.lo, hi + o.hi, max(f, o.frac))
    if op == 'sub':
        return Domain(lo - o.hi, hi - o.lo, max(f, o.frac))
    if op == 'mul':
        corners = [a * b for a in (lo, hi) for b in (o.lo, o.hi)]
        return Domain(min(corners), max(corners), f + o.frac)
    if op == 'max':
        return Domain(max(lo, o.lo), max(hi, o.hi), max(f, o.frac))
    if op == 'min':
        return Domain(min(lo, o.lo), min(hi, o.hi), max(f, o.frac))
    if op == 'select':
        return domain.join(o)
    if op == 'and':
        if o.lo >= 0 and o.lo == o.hi:
            return Domain(0, o.hi)
        return bit_bound(domain.join(o))
    if op in ('or', 'xor'):
        return bit_bound(domain.join(o))
    if op == 'shl':
        return Domain(lo * 2 ** o.hi, hi * 2 ** o.hi)
    if op == 'shr':
        return Domain(math.floor(lo / 2 ** o.lo), math.floor(hi / 2 ** o.lo))
    if op == 'div':
        bound = math.ceil(domain.magnitude / min(abs(o.lo), abs(o.hi)))
        return Domain(-bound if lo < 0 or o.lo < 0 else 0, bound)
    if op == 'mod':
        m = max(abs(o.lo), abs(o.hi)) - 1
        if semantics == 'trunc':
            return Domain(-m if lo < 0 else 0, m if hi > 0 else 0)
        return Domain(-m if o.lo < 0 else 0, m if o.hi > 0 else 0)
    raise ValueError('Unknown slice step: ' + op)


def cast_domain(domain, dtype):
    """Domain after converting to dtype (float to integer truncates)."""
    if not DTYPES[dtype].is_float and domain.frac:
        return Domain(math.trunc(domain.lo), math.trunc(domain.hi), 0)
    return domain


def cast_allowed(src, dst, backend):
    s, d = DTYPES[src], DTYPES[dst]
    if src == dst:
        return True
    # Both front ends reject integer <-> fp8 conversions outright.
    if (s.is_fp8 and not d.is_float) or (d.is_fp8 and not s.is_float):
        return False
    return True


def legalize_steps(params, prefix, count, dtype, domain, backend, operand=None, semantics='trunc'):
    """Legalize steps prefix1..prefix{count}; returns (dtype, domain, steps).

    operand is the domain of the second input buffer (stored in the input
    dtype) or None when binary steps use constants only."""
    steps = []
    for i in range(1, count + 1):
        op_key, dt_key = f'{prefix}{i}_op', f'{prefix}{i}_dt'
        op = params.get(op_key, 'none')
        source = 'const'
        if op != 'none' and not op_allowed(op, dtype, backend):
            op = 'none'
        if op in BINARY:
            if operand is not None and op not in ('shl', 'shr', 'div', 'mod') and operand.fits(dtype) \
                    and (DTYPES[dtype].kind != 'uint' or operand.lo >= 0):
                other, source = operand, 'input'
            else:
                c = constant(op, i - 1, dtype)
                other = Domain(c, c)
        else:
            other = None
        result = transfer(op, domain, other, semantics) if op != 'none' else domain
        if op != 'none' and (not result.fits(dtype) or (DTYPES[dtype].kind == 'uint' and result.lo < 0)):
            op, result, source = 'none', domain, 'const'
        target = params.get(dt_key, dtype)
        converted = cast_domain(result, target)
        if (target != dtype and (not cast_allowed(dtype, target, backend) or not converted.fits(target)
                                 or (DTYPES[target].kind == 'uint' and result.lo < 0))):
            target, converted = dtype, result
        params[op_key], params[dt_key] = op, target
        steps.append((op, dtype, target, source, constant(op, i - 1, dtype) if op in BINARY else None))
        dtype, domain = target, converted
    return dtype, domain, steps


# ---- Triton lowering --------------------------------------------------------

def triton_step(value, step, shape, operand_expr):
    """Triton statements applying one legalized step to `value`."""
    op, dtype, target, source, c = step
    t = DTYPES[dtype].triton
    lines = []
    if op != 'none':
        if op in BINARY:
            other = f'{operand_expr}.to({t})' if source == 'input' else f'tl.full({shape}, {c}, {t})'
        if op in ('add', 'sub', 'mul', 'and', 'or', 'xor', 'shl', 'shr', 'div', 'mod'):
            symbol = {'add': '+', 'sub': '-', 'mul': '*', 'and': '&', 'or': '|', 'xor': '^',
                      'shl': '<<', 'shr': '>>', 'div': '//', 'mod': '%'}[op]
            expr = f'{value} {symbol} {other}'
        elif op == 'max':
            expr = f'tl.maximum({value}, {other})'
        elif op == 'min':
            expr = f'tl.minimum({value}, {other})'
        elif op == 'select':
            expr = f'tl.where({value} > {other}, {value}, {other})'
        elif op == 'abs':
            expr = f'tl.abs({value})'
        elif op == 'neg':
            expr = f'-{value}'
        elif op == 'half':
            expr = f'{value} * tl.full({shape}, 0.5, {t})'
        else:
            expr = f'tl.{op}({value})'
        lines.append(f'{value} = ({expr}).to({t})')
    if target != dtype:
        lines.append(f'{value} = {value}.to({DTYPES[target].triton})')
    return lines


# ---- TileLang lowering ------------------------------------------------------

def tilelang_step(expr, step, operand_expr):
    """A TIR expression applying one legalized step to expression `expr`."""
    op, dtype, target, source, c = step
    t = DTYPES[dtype].tilelang
    if op != 'none':
        if op in BINARY:
            other = f'T.cast({operand_expr}, "{t}")' if source == 'input' else f'T.cast({c}, "{t}")'
        if op in ('add', 'sub', 'mul', 'and', 'or', 'xor', 'shl', 'shr'):
            symbol = {'add': '+', 'sub': '-', 'mul': '*', 'and': '&', 'or': '|', 'xor': '^',
                      'shl': '<<', 'shr': '>>'}[op]
            expr = f'({expr} {symbol} {other})'
        elif op == 'div':
            expr = f'T.floordiv({expr}, {other})'
        elif op == 'mod':
            expr = f'T.floormod({expr}, {other})'
        elif op == 'max':
            expr = f'T.max({expr}, {other})'
        elif op == 'min':
            expr = f'T.min({expr}, {other})'
        elif op == 'select':
            expr = f'T.if_then_else({expr} > {other}, {expr}, {other})'
        elif op == 'abs':
            expr = f'T.abs({expr})'
        elif op == 'neg':
            expr = f'(-{expr})'
        elif op == 'half':
            expr = f'({expr} * T.cast(0.5, "{t}"))'
        else:
            expr = f'T.{op}({expr})'
    if target != dtype:
        expr = f'T.cast({expr}, "{DTYPES[target].tilelang}")'
    return expr


# ---- exact reference (embedded in the harness as data) -----------------------

def reference_steps(steps):
    """JSON-serializable step list interpreted by the harness reference."""
    return [list(step) for step in steps]
