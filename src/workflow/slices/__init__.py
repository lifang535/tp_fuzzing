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

SLICES = {s.name: s for s in (CastSlice(), ReduceSlice(), ScanSlice(), GemmSlice(), AtomicSlice(), RoundSlice())}


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
