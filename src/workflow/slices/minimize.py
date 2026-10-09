"""One-knob-at-a-time reduction of a failing slice program.

A failing knob assignment is reduced by resetting one knob at a time to its
simplest value and keeping the reset when the failure persists (ddmin at
knob granularity). The knobs left away from their simplest values form the
failure's core: two reproducers of one mechanism through different dtype
paths reduce to the same core, so the core, not the full assignment, names a
wrong-result mechanism (cf. test-case reduction for deduplication, Donaldson
et al., PLDI'21), and the scheduler can steer away from it.
"""
from . import SLICES, make_program

# Simplest value per knob; dtype-step knobs default to "no conversion".
SIMPLEST = {
    'pair': 0, 'tail': 'none', 'dynamic': 0, 'operand': 'const', 'values': 'small', 'warps': 4,
    'threads': 128, 'stage': 'fragment', 'shape': None, 'kind': None, 'axis': 0, 'keep': 0,
    'reverse': 0, 'clear': 1, 'scope': 'fragment', 'batch': 1, 'nanprop': 0, 'dst': 'inplace',
    'mma_dt': 'f16', 'trans_a': 0, 'trans_b': 0, 'acc': 'f32', 'm': 32, 'n': 32, 'k': 32,
    'kloop': 1, 'stages': 1, 'acc_mode': 'ret', 'prec': 'ieee', 'bm': 64, 'bn': 64, 'bk': 32,
    'gm': 1, 'gn': 1, 'gk': 1, 'areg': 0, 'kpack': 1, 'policy': 'Square', 'loop': 'pipelined',
    'clear_accum': 0, 'op': 'add', 'blocks': 1, 'slots': 16, 'contention': 'mod', 'mask': 0,
    'sem': 'relaxed', 'in_dt': 'f32',
}
# Cheap, rarely essential knobs first; the operation-defining ones last.
# Conversions are reset front to back: a reset step's conversion moves to
# the next step, then to the output store, where one conversion remains.
ORDER = ('pair', 'warps2', 'threads2', 'tail', 'dynamic', 'operand', 'values', 'warps', 'threads',
         'post2_op', 'post1_op', 'pre3_op', 'pre2_op', 'pre1_op', 'pre1_dt', 'pre2_dt', 'pre3_dt',
         'post1_dt', 'post2_dt', 'out_dt', 'stage', 'scope', 'batch', 'nanprop', 'dst', 'keep', 'clear',
         'shape', 'axis', 'reverse', 'sem', 'mask', 'contention', 'slots', 'blocks', 'n',
         'acc_mode', 'kloop', 'stages', 'loop', 'kpack', 'areg', 'policy', 'clear_accum', 'gm', 'gn',
         'gk', 'm', 'k', 'bm', 'bn', 'bk', 'trans_a', 'trans_b', 'prec', 'acc', 'kind', 'core_dt',
         'op', 'in_dt', 'mma_dt')


def simplest(slice_, knob, params, plan):
    """The simplest value of knob in the current assignment, or None."""
    space = slice_.space(plan['backend'])
    if knob not in space:
        return None
    if knob.endswith('_op'):
        return 'none'
    if knob in ('warps2', 'threads2'):
        return params[knob[:-1]]
    if knob[:3] in ('pre', 'pos') and knob.endswith('_dt'):
        steps = plan['pre'] if knob.startswith('pre') else plan['post']
        index = int(knob[-4]) - 1
        return steps[index][1] if index < len(steps) else None
    if knob == 'out_dt':
        return plan.get('final_dt', plan.get('out_dt'))
    if knob == 'core_dt':
        return plan['pre'][-1][2] if plan['pre'] else plan['in_dt']
    if knob == 'scope' and 'sem' in space:
        return 'gpu'
    value = SIMPLEST.get(knob)
    return space[knob][0] if value is None else value


def minimize(program, still_fails, budget=24):
    """(reduced program, core, tests used); still_fails runs one test."""
    slice_ = SLICES[program.slice]
    current = program
    used = 0
    tried = set()
    for _ in range(3):  # until a pass keeps nothing, budget permitting
        before = current
        for knob in ORDER:
            if used >= budget:
                break
            params = dict(current.params)
            if knob not in params:
                continue
            plan = slice_.legalize(dict(params), current.backend)
            value = simplest(slice_, knob, params, plan)
            if value is None or value == params[knob]:
                continue
            params[knob] = value
            candidate = make_program(current.slice, params, current.backend)
            key = tuple(sorted(candidate.params.items()))
            if candidate.params == current.params or key in tried:
                continue
            tried.add(key)
            used += 1
            if still_fails(candidate):
                current = candidate
        if current is before or used >= budget:
            break
    return current, core(current), used


def core(program):
    """Knobs of a (reduced) program away from their simplest values."""
    slice_ = SLICES[program.slice]
    params = dict(program.params)
    plan = slice_.legalize(dict(params), program.backend)
    kept = {}
    for knob, value in sorted(params.items()):
        simple = simplest(slice_, knob, params, plan)
        if knob in ('warps2', 'threads2') and not params.get('pair'):
            continue
        if simple is None or simple != value:
            kept[knob] = value
    return kept


def signature(program):
    """Canonical name of a reduced program: its executed dtype path (input
    dtype, operations, conversions, output conversion) and its other core
    knobs. A conversion spelled at the first chain step or at the output
    store is one conversion here, so both reductions get one name."""
    slice_ = SLICES[program.slice]
    params = dict(program.params)
    plan = slice_.legalize(dict(params), program.backend)
    path = [plan['in_dt']]

    def steps(chain):
        for op, dtype, target, source, _ in chain:
            if op != 'none':
                path.append(op + ('(y)' if source == 'input' else ''))
            if target != dtype:
                path.append('>' + target)
    steps(plan['pre'])
    path.append('|')
    steps(plan['post'])
    if plan['out_dt'] != plan.get('final_dt', plan['out_dt']):
        path.append('>' + plan['out_dt'])
    chain = {'in_dt', 'out_dt'} | {k for k in params if k[:3] in ('pre', 'pos') and k[-3:] in ('_op', '_dt')}
    knobs = ' '.join(f'{k}={v}' for k, v in sorted(core(program).items()) if k not in chain)
    return f"{program.slice} {' '.join(path)} {knobs}".strip()


def matches(params, kept):
    return all(params.get(k) == v for k, v in kept.items())
