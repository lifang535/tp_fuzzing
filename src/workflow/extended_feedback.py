"""Bounded structural and potentially observable dependency features.

Liveness is a static approximation; retained compiler IR has separate counters.
No source or IR feature is described as compiler edge coverage.
"""
from src.ir.extended import analyze, walk
from src.workflow.feedback import key


def extended_features(program):
    types, live = analyze(program)
    features = {key('extended_family', program.family),
                key('extended_oracle', program.configuration_pair, program.observation_pair)}
    buffers = {b.name: b for b in program.buffers}

    def visit(block, scope, inherited=None, parent='function'):
        sources = dict(inherited or {})
        sources.update((v.name, ('argument', ())) for v in block.arguments)
        memory = {}
        for n in block.operations:
            sig = tuple((v.type.dtype, v.type.shape) for v in n.results)
            local = {key('op', n.op), key('extended_signature', n.op, sig),
                     key('extended_context', parent, n.op, sig)}
            for name in n.operands:
                producer, ancestors = sources[name]
                ty = types[scope, name]
                local.add(key('extended_edge', producer, n.op, ty.dtype, ty.shape))
                for ancestor in ancestors:
                    local.add(key('dependency_chain', ancestor, producer, n.op, parent, ty.dtype))
            if n.op in ('load', 'store'):
                buf = buffers[n.attrs['buffer']]
                root = buf.base or buf.name
                local.add(key('memory_view', n.op, buf.role, bool(buf.base), buf.stride, buf.dtype))
                local.add(key('memory_dependency', memory.get(root, 'initial'), n.op, buf.dtype, parent))
                memory[root] = n.op
            if n.op in ('for', 'while', 'if', 'call'):
                local.add(key('region_signature', n.op, sig))
            features.update(local)
            if (scope, id(n)) in live:
                features.update(key('observable', f) for f in local)
            for child in n.regions:
                visit(child, scope, sources, n.op)
            ancestry = tuple(sorted({sources[a][0] for a in n.operands}))
            sources.update((v.name, (n.op, ancestry)) for v in n.results)

    for fn in program.functions:
        visit(fn.body, fn.name)
    visit(program.body, 'main')
    return features
