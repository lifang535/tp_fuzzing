"""Campaign-level structural feedback, inspired by MLIRSmith's DiversityCriteria.

These are source/IR features, not measured compiler branch coverage. Successful
executions drive selection; attempted features are retained separately so invalid
programs cannot masquerade as successfully exercised combinations.
"""
from collections import Counter
import hashlib
import json
import re


def key(*parts):
    return json.dumps(parts, separators=(',', ':'))


def program_digest(program):
    from src.ir.serialization import program_to_dict
    encoded = json.dumps(program_to_dict(program), sort_keys=True).encode()
    return hashlib.sha256(encoded).hexdigest()


def confirmed_failure(bug_or_message, backend):
    """Conservative signatures of manually audited compiler mechanisms.

    A broad root_cause label must never be used to suppress an unreviewed
    failure. New signatures remain unpenalized and every failure is saved.
    """
    message = (bug_or_message if isinstance(bug_or_message, str)
               else bug_or_message.error_message)
    if backend == 'tilelang':
        if 'Cannot convert type bool' in message and 'to CUDA type' in message:
            return 'tilelang_bool_cuda_type'
        if 'ReduceOp cannot lower' in message or 'CanProveEqual' in message:
            return 'tilelang_reduce_layout'
        if 'IsValidCPAsyncTransferBytes' in message:
            return 'tilelang_two_byte_cp_async'
    elif backend == 'triton':
        if ('triton.compiler.errors.CompilationError' in message
                and re.search(r'tl\.flip\([^\n]*\)[^\n]*\n\s*\^', message)):
            return 'triton_flip_default_axis'
        if ('TritonGPURemoveLayoutConversions' in message
                and 'operand #0 does not dominate this use' in message):
            return 'triton_layout_conversion_dominance'
    return None


def program_features(program):
    from src.ir.extended import ExtendedProgram
    if isinstance(program, ExtendedProgram):
        from .extended_feedback import extended_features
        return extended_features(program)
    from src.ir.region import RegionProgram, walk
    if not isinstance(program, RegionProgram):
        raise TypeError(f'Unsupported program: {type(program).__name__}')
    p, dtype = program.spec, program.spec.dtype.value
    features = set()
    if program.body.operations[0].kind == 'probe':
        op = p.compute_kind.value
        features.add(key('op', op))
        features.add(key('probe', op, p.input_layout, p.input_pattern, dtype))
        if p.cache_cycle:
            features.add(key('cache_cycle', op, p.threads, p.schedule_pair, dtype))
    else:
        if program.typed:
            from src.ir.region_types import infer_program
            types, _, slots = infer_program(program)
            for scope, body in [('main', program.body)] + [(fn.name, fn.body) for fn in program.functions]:
                for op in walk(body):
                    t = types[scope, op.result]
                    features.add(key('value_type', op.kind, t.kind, t.dtype, t.shape))
                    for operand in op.operands:
                        source = types[scope, operand]
                        features.add(key('typed_edge', op.kind, source.kind, source.dtype, source.shape, t.dtype, t.shape))
                    if op.kind == 'load_input':
                        features.add(key('input_access', op.attrs['source'], op.attrs.get('row_offset', 0), op.attrs.get('col_offset', 0)))
                    elif op.kind == 'reduce_tile':
                        features.add(key('compact_reduce', op.attrs['axis'], op.attrs['reduction']))
            features.add(key('scratch_slots', len(slots)))
        if program.execution is not None:
            execution = program.execution
            features.add(key('region_input', execution.input_pattern))
            features.add(key('region_layout', program.body.operations[0].kind,
                             execution.input_layout_a, execution.input_layout_b, program.spec.dtype.value))
            features.add(key('region_oracle', execution.input_seed_count, execution.repeat_count,
                             execution.schedule_pair))
        def visit(region, inherited, parent="function", sources=()):
            producers = dict(inherited)
            producers.update(zip(region.arguments, sources))
            for op in region.operations:
                features.add(key("op", op.kind))
                features.add(key("nest", parent, op.kind))
                if op.kind == 'round':
                    features.add(key('attribute', 'round', op.attrs['dtype']))
                elif op.kind == 'for':
                    features.add(key('attribute', 'for', op.attrs['trip_count']))
                    features.add(key('loop_index', op.attrs.get('start', 0), op.attrs.get('step', 1)))
                    child = op.regions[0]
                    uses_carry = child.yield_value == child.arguments[0] or any(
                        child.arguments[0] in nested.operands for nested in walk(child))
                    features.add(key('loop_carry', uses_carry))
                elif op.kind == 'if':
                    features.add(key('attribute', 'if', op.attrs['parity']))
                    features.add(key('predicate', op.attrs.get('predicate', 'row'),
                                     op.attrs.get('modulus', 2), op.attrs['parity']))
                elif op.kind == 'index_add':
                    features.add(key('index_add', op.attrs['axis'], op.attrs['scale']))
                elif op.kind == 'scale':
                    alpha = op.attrs['alpha']
                    features.add(key('attribute', 'scale', 'zero' if alpha == 0 else 'negative' if alpha < 0 else 'positive'))
                inputs = [producers[v] for v in op.operands]
                for source in inputs:
                    features.add(key("data", source, op.kind))
                for child in op.regions:
                    yielded = visit(child, producers, op.kind, inputs)
                    features.add(key("yield", yielded, op.kind))
                producers[op.result] = op.kind
            return producers[region.yield_value]
        visit(program.body, {})
        depths = {}
        for fn in program.functions:
            visit(fn.body, {}, sources=['argument'] * len(fn.body.arguments))
            calls = [op for op in walk(fn.body) if op.kind == 'call']
            depths[fn.name] = 1 + max((depths[op.attrs['callee']] for op in calls), default=0)
            features.add(key('function_arity', len(fn.body.arguments)))
        if program.functions:
            features.add(key('function_count', len(program.functions)))
            for op in program.all_operations():
                if op.kind == 'call':
                    features.add(key('call', len(op.operands), depths[op.attrs['callee']]))
    loop = p.loop_kind.value if hasattr(p.loop_kind, 'value') else p.loop_kind
    # Bound feature cardinality: dimensions represented as tail/full/singleton.
    shape = tuple('singleton' if getattr(p, d) == 1 else
                  'tail' if getattr(p, d) % getattr(p, 'block_' + d) else 'full'
                  for d in ('M', 'N', 'K'))
    ops = [f for f in features if json.loads(f)[0] == 'op']
    for op in ops:
        features.add(key('schedule', json.loads(op)[1], dtype, loop, p.num_stages, p.threads, shape))
    return features


class StructuralFeedback:
    def __init__(self):
        self.attempted = Counter()
        self.passed = Counter()
        self.compiled = Counter()
        self.compiler = Counter()
        self.seed_compiler = {}
        self.known_seed_failures = Counter()
        self.known_signatures = Counter()

    def observe(self, program, passed):
        features = program_features(program)
        novel = features - self.passed.keys() if passed else set()
        self.attempted.update(features)
        if passed:
            self.passed.update(features)
        return len(novel)


    def observe_compilation(self, program, records, complete=False):
        features = {f for r in records for f in r.get('features', [])}
        novel = features - self.compiler.keys()
        self.compiler.update(features)
        if complete:
            self.compiled.update(program_features(program))
        return len(novel)

    def register_seed(self, program, records):
        """Associate observed post-lowering features with a passing seed."""
        features = {f for record in records for f in record.get('features', [])}
        if features:
            self.seed_compiler[program_digest(program)] = sorted(features)

    def observe_known_failure(self, source_digest, signature):
        if signature is None:
            return
        self.known_signatures[signature] += 1
        if source_digest:
            self.known_seed_failures[source_digest] += 1

    def retain_seeds(self, digests):
        self.seed_compiler = {digest: features for digest, features in self.seed_compiler.items()
                              if digest in digests}
        self.known_seed_failures = Counter({digest: count for digest, count in self.known_seed_failures.items()
                                            if digest in digests})

    def seed_weight(self, program):
        features = program_features(program)
        structural = 1.0 + 4.0 * sum(1 / (1 + self.passed[f]) for f in sorted(features)) / max(1, len(features))
        digest = program_digest(program)
        compiler = self.seed_compiler.get(digest, ())
        # Only actually retained compiler features guide this component.
        rare = sum(1 / (1 + self.compiler[f]) for f in compiler) / max(1, len(compiler))
        repeats = max(0, self.known_seed_failures[digest] - 1)
        # Keep a nonzero exploration floor so one early failure cannot ban a seed.
        return (structural + 4.0 * rare) / (1.0 + min(repeats, 12) * 0.5)

    def weight(self, feature, base=1.0, passed_decay=2.0, uncovered_boost=0.0):
        """Selection weight for one structural feature.

        MLIRSmith's DiversityCriteria boost: a feature that has never been
        attempted gets a large fixed additive weight (one-shot — a tried
        feature loses the boost even if it failed), while the passed-count
        decay remains the secondary pressure towards combinations that work.

        A feature that has been attempted but never passed is demoted below
        `base`: MLIRSmith counts programs rejected before real compilation as
        wasted effort, so a combination that only produced front-end
        rejections must not out-compete untried or proven combinations. The
        0.6 factor keeps the composed op+nest weight in the region template
        positive (0.6 + 0.6 - 1.0), so no choice list collapses to all-zero
        weights.
        """
        if not self.attempted[feature]:
            return base + passed_decay + uncovered_boost
        if not self.passed[feature]:
            return base * 0.6
        return base + passed_decay / (1 + self.passed[feature])

    def save(self, path):
        data = {'version': 2, 'attempted': dict(self.attempted), 'passed': dict(self.passed),
                'compiled': dict(self.compiled), 'compiler': dict(self.compiler),
                'seed_compiler': self.seed_compiler,
                'known_seed_failures': dict(self.known_seed_failures),
                'known_signatures': dict(self.known_signatures)}
        temporary = path.with_suffix('.tmp')
        temporary.write_text(json.dumps(data, sort_keys=True))
        temporary.replace(path)

    def restore(self, path):
        if not path.exists():
            return  # Old campaigns start with empty structural feedback.
        data = json.loads(path.read_text())
        if data.get('version') not in (1, 2):
            raise ValueError('Unsupported structural feedback version')
        counters = []
        for name in ('attempted', 'passed', 'compiled', 'compiler'):
            values = data.get(name, {})
            if not isinstance(values, dict) or any(type(v) is not int or v < 0 for v in values.values()):
                raise ValueError('Invalid structural feedback counts')
            counters.append(Counter(values))
        self.attempted, self.passed, self.compiled, self.compiler = counters
        if data.get('version') == 2:
            features = data.get('seed_compiler', {})
            if not isinstance(features, dict) or any(
                    not isinstance(digest, str) or not isinstance(values, list)
                    or any(not isinstance(value, str) for value in values)
                    for digest, values in features.items()):
                raise ValueError('Invalid seed compiler features')
            self.seed_compiler = features
            for name in ('known_seed_failures', 'known_signatures'):
                values = data.get(name, {})
                if not isinstance(values, dict) or any(type(v) is not int or v < 0 for v in values.values()):
                    raise ValueError('Invalid known failure counts')
                setattr(self, name, Counter(values))
