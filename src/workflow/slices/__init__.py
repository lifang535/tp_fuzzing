"""Feature-slice generation (FFTG-style focused generators per DSL feature).

Each slice owns a discrete knob space. A program is a legalized assignment
of its knobs; src/workflow/slices/scheduler.py chooses slices and knob
assignments by interaction coverage and estimated bug-species discovery.
"""
from .cast import CastSlice
from .reduce import ReduceSlice
from .scan import ScanSlice
from .gemm import GemmSlice
from .atomic import AtomicSlice
from .round import RoundSlice
from .mathfn import MathSlice
from .layout import LayoutSlice
from .loop import LoopSlice
from .memory import MemorySlice

SLICES = {s.name: s for s in (CastSlice(), ReduceSlice(), ScanSlice(), GemmSlice(), AtomicSlice(), RoundSlice(),
                              MathSlice(), LayoutSlice(), LoopSlice(), MemorySlice())}


def available(backend, names=None):
    chosen = [n for n in (names or SLICES) if n in SLICES and backend in SLICES[n].backends]
    if names and len(chosen) != len(names):
        unknown = sorted(set(names) - set(chosen))
        raise ValueError(f'Unknown or unsupported slices for {backend}: {unknown}')
    return chosen


def validate_program(program):
    if program.slice not in SLICES:
        raise ValueError('Unknown slice: ' + str(program.slice))
    slice_ = SLICES[program.slice]
    if program.backend not in slice_.backends:
        raise ValueError(f'Slice {program.slice} does not support backend {program.backend!r}')
    params = dict(program.params)
    slice_.legalize(params, program.backend)
    if params != program.params:
        raise ValueError('Slice parameters are not a legalized assignment')


def make_program(name, params, backend):
    """Legalize params (in place on a copy) and wrap them as a program."""
    from src.ir.slice import SliceProgram
    params = dict(params)
    SLICES[name].legalize(params, backend)
    return SliceProgram(name, params, backend)


def emit_slice(program, config):
    return SLICES[program.slice].emit(program, config)


def slice_origin(program):
    """The focus of a slice program, separating wrong-result mechanisms that
    share the checker's message: its core knobs and dtype path."""
    p = program.params
    if program.slice == 'gemm':
        return f"gemm {p['mma_dt']}>{p['acc']} tA{p['trans_a']} tB{p['trans_b']}"
    if program.slice == 'atomic':
        return f"atomic {p['op']} {p['in_dt']} {p['contention']} slots{p['slots']}"
    if program.slice == 'round':
        return f"round {p['src']}>{p['via']}>{p['dst']} {p.get('mode', 'rtne')}"
    if program.slice == 'math':
        return f"math {p['fn']} {p['in_dt']} {p['values']}" + (' widen' if p['widen'] else '')
    if program.slice == 'layout':
        ops = ','.join(p[f'op{i}'] for i in (1, 2, 3) if p[f'op{i}'] != 'none')
        return f"layout {p['source']} {p['dt']} {ops}".strip()
    if program.slice == 'loop':
        return f"loop {p['loop']} {p['body']} {p['in_dt']}>{p['acc_dt']}"
    if program.slice == 'memory':
        return f"memory {p['access']} {p['dt']} {p['view']} mask={p['mask']}"
    path = '>'.join([p['in_dt']] + [p[f'pre{i}_dt'] for i in (1, 2, 3) if f'pre{i}_dt' in p])
    ops = ','.join(p[f'pre{i}_op'] for i in (1, 2, 3) if p.get(f'pre{i}_op', 'none') != 'none')
    if program.slice in ('reduce', 'scan'):
        return f"{program.slice} {p['kind']} {p['core_dt']} ax{p['axis']} from {path} {ops}".strip()
    return f"cast {path}>{p['out_dt']} {ops}".strip()


def path_features(program):
    """Position-free features of the executed program: every operation with
    the dtype it computes in, every conversion as a source>target pair and
    the core operation, wherever in the chain they occur. A defect in one
    conversion is reached from any chain position; knob features name the
    position and cannot generalize across them."""
    slice_ = SLICES[program.slice]
    params = dict(program.params)
    plan = slice_.legalize(params, program.backend)
    features = set()

    def conversion(source, target):
        if source != target:
            features.add(f'path:conv {source}>{target}')

    def chain(steps):
        for op, dtype, target, _, _ in steps:
            if op != 'none':
                features.add(f'path:op {op}@{dtype}')
            conversion(dtype, target)
    if program.slice == 'math':
        fn = plan['fn']
        features.add(f"path:math {params['fn']}@{params['in_dt']}")
        features.add(f'path:sem {fn.sem}:{fn.mode}@{params["in_dt"]}>{plan["store"]}')
        return features
    if program.slice == 'layout':
        source = f"{plan['source']}@{plan['in_dt']}"
        features.add(f'path:layout-source {source}')
        previous = plan['source']
        for step in plan['steps']:
            features.add(f"path:layout {step['op']}@{step['dt']}")
            features.add(f"path:layout {previous}>{step['op']}")
            if 'scope' in step:
                features.add(f"path:layout {step['op']}>{step['scope']}")
            previous = step['op']
        conversion(plan['final_dt'], plan['out_dt'])
        return features
    if program.slice == 'loop':
        options = [k for k in ('stages', 'unroll', 'flatten', 'disallow', 'ws', 'sched', 'inner') if params.get(k)
                   not in (None, 0, 'none')]
        features.add(f"path:loop {params['loop']}>{plan['body']}@{plan['acc_dt']}")
        for option in options:
            features.add(f"path:loop {params['loop']}+{option}={params[option]}")
        if 'addr' in params:
            features.add(f"path:loop addr={params['addr']}>{plan['body']}")
        conversion(plan['in_dt'], plan['acc_dt'])
        return features
    if program.slice == 'memory':
        features.add(f"path:memory {params['access']}@{params['dt']}")
        features.add(f"path:memory {params['access']}>{params['view']}>{params['mask']}")
        for knob in ('hint', 'cache', 'evict', 'scache', 'sevict', 'coalesced', 'padding', 'transform'):
            if params.get(knob) not in (None, 'none', 0, 'zero'):
                features.add(f"path:memory {params['access']}+{knob}={params[knob]}")
        return features
    if program.slice == 'round':
        via = params['via']
        if via != 'none':
            conversion(params['src'], via)
        conversion(via if via != 'none' else params['src'], params['dst'])
        features.add(f"path:round {params.get('mode', 'rtne')}>{params['dst']}")
        return features
    chain(plan.get('pre', []))
    current = plan['pre'][-1][2] if plan.get('pre') else plan.get('in_dt')
    if program.slice in ('reduce', 'scan'):
        conversion(current, plan['core_dt'])
        features.add(f"path:{program.slice} {plan['kind']}@{plan['core_dt']}")
    elif program.slice == 'gemm':
        features.add(f"path:dot {params['mma_dt']}>{params['acc']}")
    elif program.slice == 'atomic':
        features.add(f"path:atomic {params['op']}@{params['in_dt']}")
    chain(plan.get('post', []))
    final = plan.get('final_dt', plan['post'][-1][2] if plan.get('post') else current)
    conversion(final, plan['out_dt'])
    return features
