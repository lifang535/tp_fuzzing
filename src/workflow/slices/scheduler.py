"""Choose the next slice test by expected discovery per second.

Across slices (Böhme, STADS, TOSEM'18): a slice's tests sample two kinds of
species, its non-noise failure buckets and its knob-pair cells. The
Good-Turing estimate of the chance that the next test shows an unseen bucket
is (f1 + 1) / (n + 2) with Laplace smoothing, f1 the buckets seen once; for
cells it is the singleton share of a test's cells. Their weighted sum per
second of test time sets each slice's draw weight, after a short round-robin
warm-up and with an epsilon-uniform floor so no slice starves.

Within a slice (AETG greedy, Cohen et al. TSE'97): the candidate covering
the most uncovered knob-pair cells among `candidates` random legalized
samples, or a mutation of 1-2 knobs of an interesting earlier program (one
that showed a new bucket or many new cells). A candidate containing the
reduced core of a bucket already hit `k` times is skipped with probability
1 - max(min_explore, 3 / k), as in the known-bug quarantine.

Lowering feedback (Zest, Padhye et al. ISSTA'19; FairFuzz, Lemieux and Sen
ASE'18): every harness reports features of the code its kernels lowered to
(PTX opcodes, TTGIR operations and layout conversions, CUDA intrinsics). A
passing program that reaches a lowering feature first becomes a seed; half
of the mutations start from a seed drawn with weight 1 / (tests showing its
rarest such feature), steering mutation toward rarely reached lowering. The
Good-Turing estimate of unseen lowering features (features seen in one test
per test) joins the slice's discovery rate.
"""
from collections import Counter
import itertools
import random

from . import SLICES, available

NOISE_ROOTS = ('oracle_unstable', 'unsupported_feature', 'shared_memory_overflow', 'warp_partition',
               'gpu_oom', 'timeout')


def pairs(params):
    items = sorted((k, str(v)) for k, v in params.items())
    return {f'{a}={x}&{b}={y}' for (a, x), (b, y) in itertools.combinations(items, 2)}


class SliceState:
    def __init__(self):
        self.tests = 0
        self.passed = 0
        self.rejected = 0
        self.covered = set()      # pair cells seen at least once
        self.single = set()       # pair cells seen in exactly one test
        self.buckets = Counter()  # non-noise failure buckets of this slice
        self.noise = Counter()    # noise roots (unsupported feature, resources, ...)
        self.interesting = []     # params dicts worth mutating, newest last
        self.seconds = 0.0        # moving average test time
        self.cells_per_test = 0   # knob-pair cells of one program
        self.cores = []           # [core dict, bucket, root cause] from reduced failures
        self.lowering = Counter()  # lowering feature -> tests that showed it
        self.lowered = 0          # tests that reported lowering features
        self.seeds = []           # [params, lowering features it reached first] of passing programs

    def discovery(self, cell_weight, lowering_weight=0.0):
        bugs = sum(1 for c in self.buckets.values() if c == 1)
        cells = len(self.single) / max(1, self.tests * self.cells_per_test)
        lowering = self.lowering_single() / max(1, self.lowered)
        return (bugs + 1) / (self.tests + 2) + cell_weight * cells + lowering_weight * lowering

    def lowering_single(self):
        return sum(1 for c in self.lowering.values() if c == 1)

    def rarity(self, seed):
        return 1.0 / min(self.lowering[f] or 1 for f in seed[1])

    def snapshot(self):
        return {'tests': self.tests, 'passed': self.passed, 'rejected': self.rejected,
                'covered': sorted(self.covered), 'single': sorted(self.single),
                'buckets': dict(self.buckets), 'noise': dict(self.noise),
                'interesting': self.interesting, 'seconds': self.seconds,
                'cells_per_test': self.cells_per_test, 'cores': self.cores,
                'lowering': dict(self.lowering), 'lowered': self.lowered, 'seeds': self.seeds}

    @classmethod
    def restore(cls, data):
        state = cls()
        state.tests, state.passed, state.rejected = data['tests'], data['passed'], data['rejected']
        state.covered, state.single = set(data['covered']), set(data['single'])
        state.buckets, state.noise = Counter(data['buckets']), Counter(data['noise'])
        state.interesting, state.seconds = list(data['interesting']), float(data['seconds'])
        state.cells_per_test = int(data.get('cells_per_test', 0))
        state.cores = [[dict(kept), bucket, cause] for kept, bucket, cause in data.get('cores', [])]
        state.lowering = Counter(data.get('lowering', {}))
        state.lowered = int(data.get('lowered', 0))
        state.seeds = [[dict(params), list(novel)] for params, novel in data.get('seeds', [])]
        return state


class SliceScheduler:
    def __init__(self, backend, names=None, candidates=16, mutate_prob=0.3, epsilon=0.15,
                 cell_weight=0.25, interesting_max=64, adaptive=True, warmup=3, min_explore=0.02,
                 lowering_weight=0.25, seeds_max=256):
        self.backend = backend
        self.names = available(backend, names)
        if not self.names:
            raise ValueError(f'No slices available for {backend}')
        self.candidates, self.mutate_prob, self.epsilon = candidates, mutate_prob, epsilon
        self.cell_weight, self.interesting_max, self.adaptive = cell_weight, interesting_max, adaptive
        self.lowering_weight, self.seeds_max = lowering_weight, seeds_max
        self.warmup, self.min_explore = warmup, min_explore
        self.avoided = Counter()
        self.states = {name: SliceState() for name in self.names}
        self.choices = Counter()

    # ---- selection ------------------------------------------------------------
    def estimate(self, name):
        state = self.states[name]
        return state.discovery(self.cell_weight, self.lowering_weight) / max(state.seconds, 0.5)

    def choose_slice(self, rng=random):
        if not self.adaptive or rng.random() < self.epsilon:
            return rng.choice(self.names)
        cold = [name for name in self.names if self.states[name].tests < self.warmup]
        if cold:
            return min(cold, key=lambda name: (self.states[name].tests, self.names.index(name)))
        weights = [self.estimate(name) for name in self.names]
        return rng.choices(self.names, weights=weights, k=1)[0]

    def next(self, rng=random, accept=None):
        """A legalized (slice name, params) pair; accept filters candidates."""
        from . import make_program
        name = self.choose_slice(rng)
        slice_ = SLICES[name]
        state = self.states[name]
        best, best_score = None, -1
        if (state.interesting or state.seeds) and rng.random() < self.mutate_prob:
            if state.seeds and (not state.interesting or rng.random() < 0.5):
                seed = rng.choices(state.seeds, weights=[state.rarity(s) for s in state.seeds], k=1)[0]
                base = dict(seed[0])
            else:
                base = dict(rng.choice(state.interesting[-16:]))
            space = slice_.space(self.backend)
            for knob in rng.sample(sorted(space), k=min(len(space), rng.choice((1, 1, 2)))):
                base[knob] = rng.choice(space[knob])
            program = make_program(name, base, self.backend)
            if accept is None or accept(program):
                return program
        for _ in range(self.candidates):
            program = make_program(name, slice_.sample(rng, self.backend), self.backend)
            if accept is not None and not accept(program):
                continue
            if self.adaptive and self.known(program, rng):
                continue
            score = len(pairs(program.params) - state.covered) if self.adaptive else 0
            if score > best_score:
                best, best_score = program, score
            if not self.adaptive:
                break
        if best is not None:
            self.choices[name] += 1
        return best

    def known(self, program, rng=random):
        """True to skip a candidate containing a well-sampled bug's core."""
        from .minimize import matches
        state = self.states[program.slice]
        for kept, bucket, _ in state.cores:
            hits = state.buckets[bucket]
            if hits >= 3 and matches(program.params, kept) and rng.random() >= max(self.min_explore, 3 / hits):
                self.avoided[program.slice, bucket] += 1
                return True
        return False

    def add_core(self, program, kept, bucket, root_cause):
        state = self.states.get(program.slice)
        entry = [dict(kept), bucket, root_cause]
        if state is not None and entry not in state.cores:
            state.cores.append(entry)

    def find_core(self, program, root_cause, bucket=None):
        """(core, bucket) of a recorded core of this root cause (and bucket,
        if given) that program contains, or None."""
        from .minimize import matches
        state = self.states.get(program.slice)
        for kept, known, cause in (state.cores if state else []):
            if cause == root_cause and bucket in (None, known) and matches(program.params, kept):
                return kept, known
        return None

    def cores_of(self, program, bucket):
        state = self.states.get(program.slice)
        return sum(1 for _, known, _ in (state.cores if state else []) if known == bucket)

    # ---- feedback -------------------------------------------------------------
    def observe(self, program, root_cause=None, bucket=None, seconds=0.0, lowering=None):
        state = self.states.get(program.slice)
        if state is None:
            return False
        fresh_lowering = []
        if lowering:
            state.lowered += 1
            fresh_lowering = sorted(f for f in lowering if f not in state.lowering)
            state.lowering.update(lowering)
            if fresh_lowering and bucket is None:
                # a valid program reaching new lowering is a seed (Zest)
                state.seeds.append([dict(program.params), fresh_lowering])
                if len(state.seeds) > self.seeds_max:
                    state.seeds.remove(min(state.seeds, key=state.rarity))
        state.tests += 1
        state.seconds = seconds if state.tests == 1 else 0.9 * state.seconds + 0.1 * seconds
        cells = pairs(program.params)
        state.cells_per_test = max(state.cells_per_test, len(cells))
        fresh = cells - state.covered
        state.single -= cells - fresh
        state.single |= fresh
        state.covered |= fresh
        novel_bucket = False
        if bucket is None:
            state.passed += 1
        elif root_cause in NOISE_ROOTS:
            state.noise[root_cause] += 1
            state.rejected += root_cause == 'unsupported_feature'
        else:
            novel_bucket = bucket not in state.buckets
            state.buckets[bucket] += 1
        if novel_bucket or len(fresh) >= 8:
            state.interesting.append(dict(program.params))
            del state.interesting[:-self.interesting_max]
        return novel_bucket or bool(fresh) or bool(fresh_lowering)

    # ---- reporting and persistence --------------------------------------------
    def stats(self):
        from src.workflow.triage import species
        return {name: {'tests': s.tests, 'passed': s.passed, 'rejected': s.rejected,
                       'noise': dict(s.noise), 'buckets': dict(s.buckets.most_common()),
                       'bug_species': species(s.buckets, samples=s.tests),
                       'pair_cells_covered': len(s.covered), 'pair_cells_single': len(s.single),
                       'lowering_features': len(s.lowering), 'lowering_single': s.lowering_single(),
                       'lowering_tests': s.lowered, 'lowering_seeds': len(s.seeds),
                       'mean_seconds': round(s.seconds, 2), 'estimate': round(self.estimate(name), 4),
                       'chosen': self.choices[name],
                       'cores': [{'bucket': bucket, 'root_cause': cause, 'core': kept}
                                 for kept, bucket, cause in s.cores],
                       'avoided': {b: n for (slice_name, b), n in self.avoided.items() if slice_name == name}}
                for name, s in self.states.items()}

    def lowering_stats(self):
        """Lowering features over all slices, in total and per kind."""
        union = Counter()
        for state in self.states.values():
            union.update(state.lowering)
        kinds = Counter(feature.split(':', 1)[0] for feature in union)
        return {'features': len(union), 'single': sum(1 for c in union.values() if c == 1),
                'tests': sum(s.lowered for s in self.states.values()), 'by_kind': dict(kinds.most_common()),
                'seeds': sum(len(s.seeds) for s in self.states.values())}

    def snapshot(self):
        return {'version': 1, 'backend': self.backend, 'names': self.names,
                'states': {name: s.snapshot() for name, s in self.states.items()},
                'choices': dict(self.choices),
                'avoided': [[slice_name, bucket, n] for (slice_name, bucket), n in self.avoided.items()]}

    def restore(self, data):
        if data.get('version') != 1 or data.get('backend') != self.backend:
            raise ValueError('Incompatible slice scheduler state')
        for name, state in data['states'].items():
            if name in self.states:
                self.states[name] = SliceState.restore(state)
        self.choices = Counter(data.get('choices', {}))
        avoided = data.get('avoided', [])
        # Earlier snapshots kept one global count per bucket.
        self.avoided = Counter({(entry[0], entry[1]): entry[2] for entry in avoided} if isinstance(avoided, list) else {})
