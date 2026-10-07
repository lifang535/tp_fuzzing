"""Coverage-guided DSL derivatives of passing common Extended programs."""

from collections import Counter
import hashlib
import json
import random

from src.ir.serialization import program_from_dict
from src.workflow.generator.dsl_extend import (DSL_OPS, MATRIX_OPS, eligible_ops, extend_passed, loop_target,
                                               respell_target)
from src.workflow.feedback import corpus_eviction, program_features
from .dsl_schedule import DSLSchedule


class DSLStage:
    def __init__(self, config, backend, grids=None, feedback=None):
        self.config, self.backend, self.grids = config, backend, grids
        self.feedback = feedback
        self.sources = []  # (program, saved source path, verified, source digest)
        self.tried = set()  # source digest and op pairs already derived
        self.source_attempts = Counter()
        self.duplicate_derivatives = 0
        self.counts = Counter()
        self.compiler = Counter()  # Target-only features; never update common feedback.
        self.compiler_passed = Counter()
        self.schedule = DSLSchedule()
        self.source_compiler = {}
        self.baseline_rejected = 0
        self.invalid_extension = 0
        self._seen = 0
        # Passing target programs never enter the common mutation pool.
        self.targets = []
        self.target_meta = {}
        self.target_structural = Counter()
        self.structures = {}
        self.evolution_counts = Counter()
        if (not 0 <= config.dsl_evolve_prob <= 1 or not 1 <= config.dsl_max_depth <= 8
                or config.dsl_max_ops < 1):
            raise ValueError('Invalid DSL evolution probability, depth or operation limit')
        if type(config.dsl_source_variants) is not int or not 1 <= config.dsl_source_variants <= 32:
            raise ValueError('DSL source variants must be an integer in [1, 32]')

    @staticmethod
    def digest(program):
        return hashlib.sha256(json.dumps(program.to_dict(), sort_keys=True).encode()).hexdigest()

    @staticmethod
    def checked_output(parent, child, previous=None):
        added = [name for name in child.body.returns if name not in parent.body.returns]
        return added[-1] if added else previous or child.body.returns[0]

    def _prune(self):
        digests = {entry[3] for entry in self.sources + self.targets}
        self.structures = {d: f for d, f in self.structures.items() if d in digests}
        self.source_compiler = {d: f for d, f in self.source_compiler.items() if d in digests}
        self.target_meta = {d: m for d, m in self.target_meta.items() if d in digests}
        self.tried = {(d, op) for d, op in self.tried if d in digests}
        self.source_attempts = Counter({pair: count for pair, count in self.source_attempts.items()
                                        if pair[0] in digests})
        self.schedule.prune(digests)

    @property
    def adaptive(self):
        return self.config.dsl_adaptive_schedule and self.config.structural_feedback

    def _choose(self, values, weights):
        if self.adaptive:
            return self.schedule.choose(values, weights)
        return random.choices(values, weights=weights, k=1)[0]

    def _bound(self, pool):
        if len(pool) <= self.config.dsl_seed_pool_max:
            return
        if self.config.corpus_feedback:
            features = [self.structures[digest] | set(self.source_compiler.get(digest, ()))
                        for _, _, _, digest in pool]
            index = corpus_eviction(features)
        else:
            index = random.randrange(len(pool) - 1)
        pool.pop(index)
        self._prune()

    def add(self, program, source_file, verified=True):
        if not eligible_ops(program, self.backend):
            return False
        self._seen += 1
        digest = self.digest(program)
        if any(entry[3] == digest for entry in self.sources):
            return False
        entry = (program, str(source_file), verified, digest)
        self.structures[digest] = program_features(program)
        self.sources.append(entry)
        self._bound(self.sources)
        return any(entry[3] == digest for entry in self.sources)

    def retain_target(self, program, source_file, lineage, records, compiler_novelty=0):
        features = program_features(program)
        novel = features - self.target_structural.keys()
        self.target_structural.update(features)
        if not self.config.dsl_evolve_prob:
            return False
        # Terminal descendants remain saved reproducers, but must not crowd
        # mutable ancestors out of the bounded exploration corpus.
        if lineage.get('extension_depth', 1) >= self.config.dsl_max_depth:
            return False
        digest = self.digest(program)
        if any(entry[3] == digest for entry in self.targets):
            return False
        if not (novel or compiler_novelty or random.random() < self.config.seed_add_prob):
            return False
        self.structures[digest] = features
        self.source_compiler[digest] = sorted({f for r in records for f in r.get('features', [])})
        self.target_meta[digest] = dict(lineage)
        self.targets.append((program, str(source_file), True, digest))
        self._bound(self.targets)
        return any(entry[3] == digest for entry in self.targets)

    def _source_limit(self, op):
        variable = (op in ('join', 'split', 'interleave')
                    or op in MATRIX_OPS and (self.config.dsl_attributes or self.config.dsl_matrix_prob > 0))
        return self.config.dsl_source_variants if variable else 1

    def generate(self, oracle, accept=None):
        """Return a derivative and lineage; a restored baseline is rechecked."""
        if self.targets and self.config.dsl_evolve_prob and random.random() < self.config.dsl_evolve_prob:
            derivative = self._evolve(oracle, accept)
            if derivative is not None:
                return derivative
        for _ in range(max(1, len(self.sources) * 2)):
            candidates = {op: [] for op in DSL_OPS[self.backend]}
            for index, (parent, _, _, digest) in enumerate(self.sources):
                for op in eligible_ops(parent, self.backend):
                    if self.source_attempts[digest, op] < self._source_limit(op):
                        candidates[op].append(index)
            available = [op for op, indices in candidates.items() if indices]
            if not available:
                return None
            least = min(self.counts[op + ':attempted'] for op in available)
            if not self.adaptive or least == 0:
                # Cover eligible operations once before exploiting past outcomes.
                op = random.choice([op for op in available if self.counts[op + ':attempted'] == least])
            else:
                op = self._choose(available, [self.schedule.weight('extend', op) for op in available])
            indices = candidates[op]
            if self.feedback is not None and self.config.structural_feedback:
                weights = [self.source_weight(self.sources[i][0], self.sources[i][3]) for i in indices]
                if self.adaptive:
                    weights = [weight * self.schedule.weight('extend', op, self.sources[i][3])
                               for i, weight in zip(indices, weights)]
                index = self._choose(indices, weights)
            else:
                index = random.choice(indices)
            parent, source_file, verified, digest = self.sources[index]
            if not verified:
                if oracle.test(parent) is not None:
                    self.baseline_rejected += 1
                    self.sources.pop(index)
                    self._prune()
                    continue
                self.sources[index] = (parent, source_file, True, digest)
            self.counts[op + ':attempted'] += 1
            self.tried.add((digest, op))
            self.source_attempts[digest, op] += 1
            try:
                child = extend_passed(parent, self.backend, op, self.config, self.grids)
            except ValueError:
                self.invalid_extension += 1
                continue
            if accept is not None and not accept(child):
                self.duplicate_derivatives += 1
                continue
            return child, {'extension_op': op, 'source_file': source_file,
                           'source_sha256': digest, 'baseline_revalidated': True,
                           'extension_depth': 1, 'extension_action': 'extend',
                           'extension_variant': self.source_attempts[digest, op],
                           'extension_output': self.checked_output(parent, child),
                           'root_source_sha256': digest}
        return None

    def _evolve(self, oracle, accept=None):
        from src.workflow.generator.extended import mutate_extended
        for _ in range(16):
            entries = [entry for entry in self.targets
                       if self.target_meta[entry[3]].get('extension_depth', 1) < self.config.dsl_max_depth]
            if not entries:
                return None
            weights = ([self.source_weight(entry[0], entry[3]) for entry in entries]
                       if self.config.structural_feedback else None)
            parent, source_file, verified, digest = self._choose(entries, weights)
            if not verified:
                index = next(i for i, entry in enumerate(self.targets) if entry[3] == digest)
                if oracle.test(parent) is not None:
                    self.baseline_rejected += 1
                    self.targets.pop(index)
                    self._prune()
                    continue
                self.targets[index] = (parent, source_file, True, digest)
            meta = self.target_meta[digest]
            # Respelling target attributes takes half of the mutation share.
            respell = .125 if self.config.dsl_attributes else 0
            actions = ('compose', 'mutate', 'loop', 'respell')
            weights = (0.55, 0.25 - respell, 0.20, respell)
            if self.adaptive:
                weights = [weight * self.schedule.weight(action, digest=digest)
                           for action, weight in zip(actions, weights)]
            action = self._choose(actions, weights)
            op = meta['extension_op']
            try:
                if action == 'compose':
                    ops = eligible_ops(parent, self.backend, allow_target=True)
                    if not ops:
                        continue
                    weights = ([self.schedule.weight('compose', name, digest) for name in ops]
                               if self.adaptive else
                               [1 / (1 + self.evolution_counts['compose:' + name]) for name in ops])
                    op = self._choose(ops, weights)
                    # Follow the target result, not an unrelated auxiliary
                    # output that happened to be last in the common program.
                    child = extend_passed(parent, self.backend, op, self.config, self.grids,
                                          allow_target=True,
                                          input_name=meta.get('extension_output', parent.body.returns[0]))
                elif action == 'loop':
                    child = loop_target(parent, self.backend)
                elif action == 'respell':
                    child = respell_target(parent, self.backend, self.grids)
                else:
                    child = mutate_extended(parent, self.config, self.backend, regenerate=False)
                if sum(1 for _ in child.all_operations()) > self.config.dsl_max_ops:
                    raise ValueError('DSL operation budget exceeded')
                if self.digest(child) == digest:
                    continue
            except ValueError:
                self.invalid_extension += 1
                continue
            if accept is not None and not accept(child):
                self.duplicate_derivatives += 1
                continue
            self.evolution_counts[action + ':' + op] += 1
            self.counts[op + ':attempted'] += 1
            return child, {'extension_op': op, 'extension_action': action,
                           'extension_depth': meta.get('extension_depth', 1) + 1,
                           'extension_output': self.checked_output(parent, child, meta.get('extension_output')),
                           'source_file': source_file, 'source_sha256': digest,
                           'root_source_sha256': meta.get('root_source_sha256', meta['source_sha256']),
                           'baseline_revalidated': True}
        return None

    def record(self, op, passed):
        self.counts[op + (':passed' if passed else ':failed')] += 1

    def observe_compilation(self, digest, records, passed):
        features = {feature for record in records for feature in record.get('features', [])}
        novel = features - self.compiler.keys()
        self.compiler.update(features)
        if passed:
            # A feature first observed before a failure still deserves a passing
            # representative when it later becomes executable and checkable.
            novel |= features - self.compiler_passed.keys()
            self.compiler_passed.update(features)
            if features:
                self.source_compiler[digest] = sorted(set(self.source_compiler.get(digest, ())) | features)
        return len(novel)

    def observe_outcome(self, program, lineage, records, *, passed, seconds, known_repeat=False):
        self.record(lineage['extension_op'], passed)
        compiler_novelty = self.observe_compilation(lineage['source_sha256'], records, passed)
        structural_novelty = bool(program_features(program) - self.target_structural.keys()) if passed else False
        self.schedule.observe(lineage['source_sha256'], lineage.get('extension_action', 'extend'),
                              lineage['extension_op'], structural_novelty=structural_novelty,
                              compiler_novelty=compiler_novelty, seconds=seconds,
                              known_repeat=known_repeat)
        return compiler_novelty

    def source_weight(self, program, digest):
        if digest in self.target_meta:
            features = self.structures[digest]
            base = 1 + 4 * sum(1 / (1 + self.target_structural[f]) for f in sorted(features)) / max(1, len(features))
            if self.feedback is not None:
                base *= self.feedback.known_failure_weight(digest)
        else:
            base = self.feedback.seed_weight(program) if self.feedback is not None else 1.0
        if self.adaptive:
            base *= self.schedule.parent_weight(digest)
        features = self.source_compiler.get(digest, ())
        if not features:
            return base
        rarity = sum(1 / (1 + self.compiler[feature]) for feature in features) / len(features)
        return base * (1.0 + rarity)

    def snapshot(self):
        return {'version': 4, 'backend': self.backend, 'counts': dict(self.counts),
                'compiler': dict(self.compiler), 'source_compiler': self.source_compiler,
                'compiler_passed': dict(self.compiler_passed), 'schedule': self.schedule.snapshot(),
                'baseline_rejected': self.baseline_rejected,
                'invalid_extension': self.invalid_extension, 'seen': self._seen,
                'tried': [list(pair) for pair in sorted(self.tried)],
                'source_attempts': [[digest, op, count] for (digest, op), count
                                    in sorted(self.source_attempts.items())],
                'duplicate_derivatives': self.duplicate_derivatives,
                'target_structural': dict(self.target_structural),
                'evolution_counts': dict(self.evolution_counts),
                'targets': [{'program': program.to_dict(), 'source_file': path,
                             'lineage': self.target_meta[digest]}
                            for program, path, _, digest in self.targets],
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
        # Before v3 only passing DSL compilation features were recorded.
        compiler_passed = state.get('compiler_passed', compiler)
        if (not isinstance(compiler_passed, dict)
                or any(type(value) is not int or not 0 <= value <= self.compiler[feature]
                       for feature, value in compiler_passed.items())):
            raise ValueError('Invalid passing DSL compiler feedback')
        self.compiler_passed = Counter(compiler_passed)
        self.baseline_rejected = state.get('baseline_rejected', 0)
        self.invalid_extension = state.get('invalid_extension', 0)
        for entry in state.get('sources', [])[:self.config.dsl_seed_pool_max]:
            program = program_from_dict(entry['program'])
            if eligible_ops(program, self.backend):
                digest = hashlib.sha256(json.dumps(program.to_dict(), sort_keys=True).encode()).hexdigest()
                self.sources.append((program, entry['source_file'], False, digest))
                self.structures[digest] = program_features(program)
        if self.config.dsl_evolve_prob:
            from src.backends import get_backend
            for entry in state.get('targets', [])[:self.config.dsl_seed_pool_max]:
                program = program_from_dict(entry['program'])
                get_backend(self.backend).validate_program(program)
                lineage = entry['lineage']
                if (type(lineage.get('extension_depth')) is not int
                        or not 1 <= lineage['extension_depth'] <= 8
                        or lineage.get('extension_op') not in DSL_OPS[self.backend]
                        or not isinstance(lineage.get('source_sha256'), str)):
                    raise ValueError('Invalid DSL target lineage')
                if lineage['extension_depth'] >= self.config.dsl_max_depth:
                    continue
                digest = self.digest(program)
                self.targets.append((program, entry['source_file'], False, digest))
                self.target_meta[digest] = lineage
                self.structures[digest] = program_features(program)
        for name in ('target_structural', 'evolution_counts'):
            values = state.get(name, {})
            if not isinstance(values, dict) or any(type(v) is not int or v < 0 for v in values.values()):
                raise ValueError('Invalid DSL evolution feedback')
            setattr(self, name, Counter(values))
        digests = {entry[3] for entry in self.sources + self.targets}
        source_compiler = state.get('source_compiler', {})
        if not isinstance(source_compiler, dict) or any(not isinstance(values, list)
                or any(not isinstance(feature, str) for feature in values)
                for values in source_compiler.values()):
            raise ValueError('Invalid DSL source compiler feedback')
        self.source_compiler = {digest: features for digest, features in source_compiler.items()
                                if digest in digests}
        self.tried = {(digest, op) for digest, op in state.get('tried', []) if digest in digests}
        attempts = state.get('source_attempts', [[digest, op, 1] for digest, op in self.tried])
        self.source_attempts = Counter()
        for digest, op, count in attempts:
            if not isinstance(digest, str) or op not in DSL_OPS[self.backend] or type(count) is not int or count < 0:
                raise ValueError('Invalid DSL source attempt feedback')
            if digest in digests:
                self.source_attempts[digest, op] = count
        duplicates = state.get('duplicate_derivatives', 0)
        if type(duplicates) is not int or duplicates < 0:
            raise ValueError('Invalid DSL duplicate count')
        self.duplicate_derivatives = duplicates
        self._seen = max(state.get('seen', 0), len(self.sources))
        if 'schedule' in state:
            self.schedule.restore(state['schedule'])
        self.schedule.prune(digests)
