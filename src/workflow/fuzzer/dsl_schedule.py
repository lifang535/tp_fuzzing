"""Cost-aware DSL action scheduling with bounded, recent outcome feedback.

Rewards measure observed compiler artifacts and passing source features, not
independent bugs or compiler edge coverage. Only audited repeats incur a penalty.
"""
from dataclasses import asdict, dataclass
import math
import random


@dataclass
class Outcome:
    tested: int = 0
    novel: int = 0
    known_repeats: int = 0
    reward: float = 0.0
    seconds: float = 0.0
    repeat_rate: float = 0.0

    def observe(self, reward, seconds, known_repeat):
        alpha = 0.2 if self.tested else 1.0
        self.tested += 1
        self.novel += int(reward > 0)
        self.known_repeats += int(known_repeat)
        self.reward += alpha * (reward - self.reward)
        self.seconds += alpha * (seconds - self.seconds)
        self.repeat_rate += alpha * (float(known_repeat) - self.repeat_rate)

    def weight(self, mean_seconds):
        # Bound the cost correction: cheap front-end rejections must not crowd
        # out programs that actually reach expensive compilation/execution.
        cost = math.sqrt(max(mean_seconds, 0.001) / max(self.seconds, 0.001))
        cost = max(0.5, min(2.0, cost))
        uncertainty = 0.5 / math.sqrt(1 + self.tested)
        return 0.05 + cost * (self.reward + uncertainty) / (1 + 4 * self.repeat_rate)


class DSLSchedule:
    EXPLORATION = 0.10
    PLATEAU_WINDOW = 256

    def __init__(self):
        self.actions = {}
        self.parents = {}
        self.total = Outcome()
        self.stagnant = 0

    @staticmethod
    def keys(action, op):
        return (action, action + ':' + op)

    def observe(self, digest, action, op, *, structural_novelty, compiler_novelty,
                seconds, known_repeat=False):
        if not math.isfinite(seconds) or seconds < 0:
            raise ValueError('Invalid DSL outcome duration')
        seconds = max(0.001, seconds)
        reward = 0.5 * bool(structural_novelty) + 0.5 * bool(compiler_novelty)
        # New lexical artifacts inside an audited repeat do not constitute
        # progress toward a new mechanism. Keep accounting, but give no reward.
        if known_repeat:
            reward = 0.0
        self.stagnant = 0 if reward else self.stagnant + 1
        self.total.observe(reward, seconds, known_repeat)
        local = self.parents.setdefault(digest, {})
        local.setdefault('*', Outcome()).observe(reward, seconds, known_repeat)
        for key in self.keys(action, op):
            self.actions.setdefault(key, Outcome()).observe(reward, seconds, known_repeat)
            local.setdefault(key, Outcome()).observe(reward, seconds, known_repeat)

    def weight(self, action, op=None, digest=None):
        key = action if op is None else action + ':' + op
        aggregate = self.actions.get(key, Outcome())
        weight = aggregate.weight(self.total.seconds)
        local = self.parents.get(digest, {}).get(key)
        if local is not None:
            # Gradually trust a parent's own outcomes over the cross-parent prior.
            trust = local.tested / (local.tested + 4)
            weight = (1 - trust) * weight + trust * local.weight(self.total.seconds)
        return weight

    def parent_weight(self, digest):
        return self.parents.get(digest, {}).get('*', Outcome()).weight(self.total.seconds)

    def choose(self, values, weights):
        available = [(value, weight) for value, weight in zip(values, weights) if weight > 0]
        if not available:
            raise ValueError('No eligible DSL scheduling choices')
        values, weights = zip(*available)
        # An explicit mixture guarantees every eligible action remains reachable,
        # including after an unlucky run of failures or an old campaign restart.
        if random.random() < self.exploration:
            return random.choice(values)
        return random.choices(values, weights=weights, k=1)[0]

    @property
    def exploration(self):
        # Ramp up exploration after a plateau; reset on useful novelty.
        return min(0.5, self.EXPLORATION + 0.4 * self.stagnant / self.PLATEAU_WINDOW)

    def prune(self, digests):
        self.parents = {d: outcomes for d, outcomes in self.parents.items() if d in digests}

    def snapshot(self, include_parents=True):
        state = {'version': 1, 'total': asdict(self.total),
                 'stagnant': self.stagnant, 'exploration': self.exploration,
                 'actions': {k: asdict(v) for k, v in self.actions.items()}}
        if include_parents:
            state['parents'] = {d: {k: asdict(v) for k, v in outcomes.items()}
                                for d, outcomes in self.parents.items()}
        return state

    def restore(self, state):
        def outcome(data):
            if not isinstance(data, dict):
                raise ValueError('Invalid DSL schedule outcome')
            try:
                result = Outcome(**data)
            except TypeError as error:
                raise ValueError('Invalid DSL schedule outcome') from error
            for name in ('tested', 'novel', 'known_repeats'):
                value = getattr(result, name)
                if type(value) is not int or not 0 <= value <= result.tested:
                    raise ValueError('Invalid DSL schedule count')
            for name in ('reward', 'seconds', 'repeat_rate'):
                value = getattr(result, name)
                if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                    raise ValueError('Invalid DSL schedule measurement')
            if result.reward > 1 or result.repeat_rate > 1:
                raise ValueError('Invalid DSL schedule rate')
            return result

        def mapping(data):
            if not isinstance(data, dict) or any(not isinstance(k, str) for k in data):
                raise ValueError('Invalid DSL schedule keys')
            return {k: outcome(v) for k, v in data.items()}

        if not isinstance(state, dict) or state.get('version') != 1:
            raise ValueError('Invalid DSL schedule version')
        total = outcome(state.get('total', {}))
        actions = mapping(state.get('actions', {}))
        parents = state.get('parents', {})
        if not isinstance(parents, dict) or any(not isinstance(d, str) for d in parents):
            raise ValueError('Invalid DSL schedule parents')
        parents = {d: mapping(outcomes) for d, outcomes in parents.items()}
        stagnant = state.get('stagnant', 0)
        if type(stagnant) is not int or not 0 <= stagnant <= total.tested:
            raise ValueError('Invalid DSL plateau count')
        self.total, self.actions, self.parents = total, actions, parents
        self.stagnant = stagnant
