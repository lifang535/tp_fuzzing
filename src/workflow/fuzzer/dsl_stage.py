"""Coverage-guided DSL derivatives of passing common Extended programs."""

from collections import Counter
import hashlib
import json
import random

from src.ir.serialization import program_from_dict
from src.workflow.generator.dsl_extend import DSL_OPS, eligible_ops, extend_passed


class DSLStage:
    def __init__(self, config, backend, grids=None, feedback=None):
        self.config, self.backend, self.grids = config, backend, grids
        self.feedback = feedback
        self.sources = []  # (program, saved source path, verified, source digest)
        self.tried = set()  # source digest and op pairs already derived
        self.counts = Counter()
        self.compiler = Counter()  # Target-only features; never update common feedback.
        self.source_compiler = {}
        self.baseline_rejected = 0
        self.invalid_extension = 0
        self._seen = 0

    def add(self, program, source_file, verified=True):
        if not eligible_ops(program, self.backend):
            return False
        self._seen += 1
        digest = hashlib.sha256(json.dumps(program.to_dict(), sort_keys=True).encode()).hexdigest()
        entry = (program, str(source_file), verified, digest)
        if len(self.sources) < self.config.dsl_seed_pool_max:
            self.sources.append(entry)
        else:
            # Rotate a bounded pool so newly passing structures keep entering.
            slot = random.randrange(len(self.sources))
            old_digest = self.sources[slot][3]
            self.tried = {pair for pair in self.tried if pair[0] != old_digest}
            self.source_compiler.pop(old_digest, None)
            self.sources[slot] = entry
        return True

    def generate(self, oracle):
        """Return a derivative and lineage; a restored baseline is rechecked."""
        for _ in range(max(1, len(self.sources) * 2)):
            candidates = {op: [] for op in DSL_OPS[self.backend]}
            for index, (parent, _, _, digest) in enumerate(self.sources):
                for op in eligible_ops(parent, self.backend):
                    if (digest, op) not in self.tried:
                        candidates[op].append(index)
            available = [op for op, indices in candidates.items() if indices]
            if not available:
                return None
            least = min(self.counts[op + ':attempted'] for op in available)
            op = random.choice([op for op in available if self.counts[op + ':attempted'] == least])
            indices = candidates[op]
            if self.feedback is not None and self.config.structural_feedback:
                index = random.choices(indices, weights=[self.source_weight(self.sources[i][0],
                                                                            self.sources[i][3])
                                                         for i in indices], k=1)[0]
            else:
                index = random.choice(indices)
            parent, source_file, verified, digest = self.sources[index]
            if not verified:
                if oracle.test(parent) is not None:
                    self.baseline_rejected += 1
                    self.sources.pop(index)
                    self.tried = {pair for pair in self.tried if pair[0] != digest}
                    continue
                self.sources[index] = (parent, source_file, True, digest)
            self.counts[op + ':attempted'] += 1
            self.tried.add((digest, op))
            try:
                child = extend_passed(parent, self.backend, op, self.config, self.grids)
            except ValueError:
                self.invalid_extension += 1
                continue
            return child, {'extension_op': op, 'source_file': source_file,
                           'source_sha256': digest, 'baseline_revalidated': True}
        return None

    def record(self, op, passed):
        self.counts[op + (':passed' if passed else ':failed')] += 1

    def observe_compilation(self, digest, records, passed):
        if not passed:
            return
        features = {feature for record in records for feature in record.get('features', [])}
        self.compiler.update(features)
        if features:
            self.source_compiler[digest] = sorted(set(self.source_compiler.get(digest, ())) | features)

    def source_weight(self, program, digest):
        base = self.feedback.seed_weight(program)
        features = self.source_compiler.get(digest, ())
        if not features:
            return base
        rarity = sum(1 / (1 + self.compiler[feature]) for feature in features) / len(features)
        return base * (1.0 + rarity)

    def snapshot(self):
        return {'backend': self.backend, 'counts': dict(self.counts),
                'compiler': dict(self.compiler), 'source_compiler': self.source_compiler,
                'baseline_rejected': self.baseline_rejected,
                'invalid_extension': self.invalid_extension, 'seen': self._seen,
                'tried': [list(pair) for pair in sorted(self.tried)],
                'sources': [{'program': program.to_dict(), 'source_file': path}
                            for program, path, _, _ in self.sources]}

    def restore(self, state):
        if state.get('backend') != self.backend:
            raise ValueError('DSL stage backend mismatch')
        self.counts = Counter(state.get('counts', {}))
        compiler = state.get('compiler', {})
        if not isinstance(compiler, dict) or any(type(value) is not int or value < 0
                                                  for value in compiler.values()):
            raise ValueError('Invalid DSL compiler feedback')
        self.compiler = Counter(compiler)
        self.baseline_rejected = state.get('baseline_rejected', 0)
        self.invalid_extension = state.get('invalid_extension', 0)
        for entry in state.get('sources', [])[:self.config.dsl_seed_pool_max]:
            program = program_from_dict(entry['program'])
            if eligible_ops(program, self.backend):
                digest = hashlib.sha256(json.dumps(program.to_dict(), sort_keys=True).encode()).hexdigest()
                self.sources.append((program, entry['source_file'], False, digest))
        digests = {entry[3] for entry in self.sources}
        source_compiler = state.get('source_compiler', {})
        if not isinstance(source_compiler, dict) or any(not isinstance(values, list)
                or any(not isinstance(feature, str) for feature in values)
                for values in source_compiler.values()):
            raise ValueError('Invalid DSL source compiler feedback')
        self.source_compiler = {digest: features for digest, features in source_compiler.items()
                                if digest in digests}
        self.tried = {(digest, op) for digest, op in state.get('tried', []) if digest in digests}
        self._seen = max(state.get('seen', 0), len(self.sources))
