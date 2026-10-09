"""Failure triage: diagnostic buckets, known-bug quarantine and species counts.

Every failure is still executed and saved. Buckets group failures whose
diagnostic normalizes to the same text; the quarantine only lowers how often
fresh and mutated generation re-enters a program region that, on recent
evidence, reliably reproduces an already well-sampled bucket.
"""
from collections import Counter
import hashlib
import math
import random
import re

_EXCEPTION = re.compile(r'^(?:[A-Za-z_]\w*\.)*[A-Za-z_]\w*(?:Error|Exception|OutOfResources)(?::|$)')
_CHAINED = ('The above exception was the direct cause of the following exception:',
            'During handling of the above exception, another exception occurred:')
_FRAME = re.compile(r'^\s*File "[^"]*", line \d+, in (.+)$')
# Logging and re-raise helpers say nothing about where a failure happened.
_PLUMBING = {'LogFatal', '~LogFatal', 'raise_', '__call__', '<module>'}
_DTYPE = re.compile(r'(?:u?int|b?float|fp)\d+\w*')
_PATH = re.compile(r'(?<![\w.])(?:/[\w.+-]+)+')
_SIGNAL = re.compile(r'Process terminated by signal \d+ \((\w+)\)')


def _normalize(text, limit=200):
    # Temporary files, addresses, sizes, SSA names (e12), variant indices
    # (triton_8_prec) and vector widths (boolx16) differ between reproducers
    # of one mechanism; dtype names do not.
    text = _PATH.sub('<path>', text)
    text = re.sub(r'0x[0-9a-fA-F]+', '0x#', text)
    text = re.sub(r'(?<![\w.#])-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?', '#', text)
    text = re.sub(r'(?<![\w.])[A-Za-z_]\w*',
                  lambda m: m.group() if _DTYPE.fullmatch(m.group()) else re.sub(r'\d+', '#', m.group()),
                  text)
    return ' '.join(text.split())[:limit]


def _diagnostic(line):
    # A failed TVM check is identified by its condition, not by the values.
    return _normalize(re.sub(r'(Check failed: \(.*?\) is false):.*', r'\1', line))


def _frame(lines):
    """Innermost function named by the traceback above the final exception."""
    for line in reversed(lines):
        match = _FRAME.match(line)
        if match:
            name = match.group(1).split('(')[0].split()[-1].split('::')[-1]
            if name not in _PLUMBING:
                return name
    return ''


# MLIR element types keep their digits: f64 -> f8E5M2 and f32 -> f8E4M3FN
# conversions are different mechanisms.
_MLIR_TYPE = re.compile(r'\b(?:bf16|[fi]\d+(?:E\d+M\d+(?:FNUZ|FN|B\d+)?)?|ui\d+)\b')
_PASS = re.compile(r'Pipeline failed while executing \[`(\w+)`')
_MLIR_DETAIL = re.compile(r'^(?:LLVM ERROR: .*|Unsupported .*|.*Assertion `.*|.*\berror: (?!Failures have been detected).*)$',
                          re.MULTILINE)
# The first nvcc error and the header it is reported in (atomic.h, common.h
# or the generated tvm_kernels.cu name different template mechanisms).
_NVCC = re.compile(r'([\w.]+)\.(cu|h|hpp|cuh)\(\d+\): error: (.*)')


def _protected(text):
    """Normalize a backend diagnostic but keep MLIR element types."""
    kept = []
    text = _MLIR_TYPE.sub(lambda m: kept.append(m.group()) or f'TYPE{len(kept) - 1}T', text)
    text = _normalize(text)
    return re.sub(r'TYPE#T', lambda m: kept.pop(0) if kept else 'TYPE', text)


def _backend_detail(message):
    """The compiler's own diagnosis behind a generic final exception: the
    failing MLIR pass and its first error, or the first nvcc error of a
    TileLang CUDA compilation; both final exceptions read the same for every
    mechanism behind them."""
    detail = []
    if 'PassManager::run failed' in message:
        failed = _PASS.search(message)
        if failed:
            detail.append('pass ' + failed.group(1))
        first = _MLIR_DETAIL.search(message)
        if first:
            line = first.group()
            if 'Assertion `' in line:
                line = 'Assertion ' + line.split('Assertion `', 1)[1]
            detail.append(_protected(line.split(': error: ')[-1]))
    nvcc = _NVCC.search(message)
    if nvcc:
        where = '' if nvcc.group(1) == 'tvm_kernels' else f'{nvcc.group(1)}.{nvcc.group(2)}: '
        detail.append('nvcc ' + where + _normalize(nvcc.group(3)))
    return detail


def failure_key(message, location=''):
    """Normalized diagnostic: the final exception (a TVM check reduced to its
    condition) and its innermost frame, the root of a chained traceback, the
    call and message under a Triton compile-error caret, and a fatal signal.
    A message without any exception line (a timeout) keeps its location."""
    lines = message.rstrip().splitlines()
    exceptions = [i for i, line in enumerate(lines) if _EXCEPTION.match(line)]
    last = exceptions[-1] if exceptions else len(lines)
    text = lines[last] if exceptions else next((line for line in reversed(lines) if line.strip()), '')
    parts = [_diagnostic(text)]
    frame = _frame(lines[:last])
    if frame:
        parts.append('in ' + _normalize(frame))
    marker = min((message.find(m) for m in _CHAINED if m in message), default=-1)
    if marker >= 0:
        roots = [line for line in message[:marker].splitlines() if _EXCEPTION.match(line)]
        if roots and _diagnostic(roots[-1]) != parts[0]:
            parts.append('from ' + _diagnostic(roots[-1]))
    carets = [i for i in range(last, len(lines)) if lines[i].strip() == '^']
    if exceptions and 'CompilationError' in text and carets and carets[-1] > 0:
        caret = carets[-1]
        call = re.match(r'\(*([\w.]+)', lines[caret - 1][lines[caret].index('^'):])
        if call:
            parts.append('at ' + _normalize(call.group(1)))
        detail = next((line for line in lines[caret + 1:] if line.strip()), '')
        if detail:
            parts.append(_normalize(detail))
    parts += _backend_detail(message)
    signal = _SIGNAL.search(message)
    if signal and signal.group(1) not in text:
        parts.append('signal ' + signal.group(1))
    if not exceptions and location:
        parts.append('at ' + _normalize(location))
    return ' | '.join(parts)


def failure_bucket(message, root_cause, confirmed=None, location='', origin=''):
    """(bucket, key). An audited signature is its own bucket; any other failure
    is grouped as root_cause:digest of its normalized diagnostic and, for a
    wrong result, the origin of its earliest wrong value."""
    key = failure_key(message, location)
    if origin:
        key += ' | origin ' + origin
    if confirmed:
        return confirmed, key
    return f'{root_cause or "other"}:{hashlib.sha1(key.encode()).hexdigest()[:10]}', key


# Attributes that change what an operation computes or how a loop is
# compiled. Constants, index shifts, buffer names and loop bounds vary between
# reproducers of one mechanism; a flag that is off is spelled as if absent.
_ORIGIN_ATTRIBUTES = ('kind', 'axis', 'k', 'dtype', 'reverse', 'descending', 'keep_dims', 'pipelined')
_STORES = ('store', 'atomic_add', 'atomic_max', 'atomic_min', 'atomic_and', 'atomic_or', 'atomic_xor')


def wrong_result_origin(program, message, limit=160):
    """Where the earliest wrong value of an extended program (its dict form)
    comes from, or '' when the message names none of its values.

    The wrong values are the WRONG VALUES line of the message (otherwise the
    value names in its WRONG RESULT line); the earliest is the one the first
    operation of the entry block produces. Every wrong value of one launch
    shares the diagnostic text, while the operation producing the earliest
    tells mechanisms apart: its name, result dtype and rank, operand dtypes
    that differ from the result, and semantic attributes, followed through
    loops, conditionals and calls to the operation yielding the value. A
    wrong buffer is attributed to the first store or atomic writing it.
    Attribution never stops a campaign: an unreadable program has no origin."""
    try:
        return _origin(program, message)[:limit]
    except (AttributeError, IndexError, KeyError, StopIteration, TypeError):
        return ''


def _origin(program, message):
    functions = {f['name']: f['body'] for f in program.get('functions', [])}
    producers, types, arguments = {}, {}, set()

    def index(block):
        for value in block['arguments']:
            arguments.add(value['name'])
            types[value['name']] = value['type']
        for node in block['operations']:
            for position, value in enumerate(node['results']):
                producers[value['name']] = node, position
                types[value['name']] = value['type']
            for region in node['regions']:
                index(region)
    index(program['body'])
    for body in functions.values():
        index(body)
    roots = {b['name']: b['base'] or b['name'] for b in program.get('buffers', [])}

    def flags(node):
        settings = ((attribute, node['attrs'].get(attribute, False)) for attribute in _ORIGIN_ATTRIBUTES)
        return ''.join(f' {attribute}' if setting is True else f' {attribute}={setting}'
                       for attribute, setting in settings if setting is not False)

    def describe(name, depth=0):
        if name in arguments:
            return 'argument'  # a parameter, or a value carried through unchanged
        if name not in producers or depth > 8:
            return '?'
        node, position = producers[name]
        op = node['op']
        if op in ('for', 'while', 'if', 'call'):
            bodies = [functions[node['attrs']['callee']]] if op == 'call' else node['regions']
            inner = dict.fromkeys(describe(body['returns'][position], depth + 1) for body in bodies)
            return f"{op}{flags(node)}{{{' | '.join(inner)}}}"
        dtype, shape = node['results'][position]['type']['dtype'], node['results'][position]['type']['shape']
        # A load's operands are always its int32 indices and bool mask.
        sources = sorted({types[v]['dtype'] for v in node['operands'] if v in types} - {dtype}) if op != 'load' else []
        return f"{op} {dtype} r{len(shape)}{' <- ' + '+'.join(sources) if sources else ''}{flags(node)}"

    def writer(block, root):
        """The first store or atomic into `root`, in execution order."""
        for node in block['operations']:
            if node['op'] in _STORES and roots.get(node['attrs'].get('buffer')) == root:
                return node
            bodies = ([functions[node['attrs']['callee']]] if node['op'] == 'call' and node['attrs'].get('callee')
                      in functions else node['regions'])
            for body in bodies:
                found = writer(body, root)
                if found is not None:
                    return found
        return None

    # The wrong values in the order of the entry-block operation they come from.
    top = program['body']['operations']
    position = {value['name']: i for i, node in enumerate(top) for value in node['results']}
    match = re.findall(r'^WRONG VALUES: (.+)$', message, flags=re.MULTILINE)
    if match:
        names = [name.strip() for name in match[-1].split(',')]
    else:
        line = next((line for line in reversed(message.splitlines()) if 'WRONG RESULT' in line), '')
        tokens = re.findall(r'(?<![\w.])[A-Za-z_]\w*', line)
        names = [t for t in tokens if t in position] + ['memory:' + t for t in tokens if roots.get(t) == t]
    candidates = []
    for name in names:
        if name in position:
            candidates.append((position[name], describe(name)))
        elif name.startswith('memory:'):
            for i, node in enumerate(top):
                store = writer({'operations': [node]}, name[len('memory:'):])
                if store is not None:
                    dtype = next(b['dtype'] for b in program['buffers'] if b['name'] == store['attrs']['buffer'])
                    candidates.append((i, f"{store['op']} {dtype} of {describe(store['operands'][-1])}"))
                    break
    # Ties (results of one operation) keep the check order of the message.
    return min(candidates, key=lambda candidate: candidate[0])[1] if candidates else ''


def species(counts, samples=None):
    """Abundance statistics (Chao 1984; Good 1953): the Chao1 lower bound on
    the number of species and, given the number of sampling units, the
    Good-Turing probability that the next one shows an unseen species."""
    f1 = sum(1 for count in counts.values() if count == 1)
    f2 = sum(1 for count in counts.values() if count == 2)
    chao1 = len(counts) + (f1 * f1 / (2 * f2) if f2 else f1 * (f1 - 1) / 2)
    result = {'observed': len(counts), 'singletons': f1, 'doubletons': f2, 'chao1': round(chao1, 2)}
    if samples is not None:
        result.update(samples=samples, unseen_probability=round(f1 / samples, 6) if samples else 1.0)
    return result


class Quarantine:
    """Steer fresh and mutated generation away from known-bug regions.

    A sliding window keeps the structural features and outcomes of recent
    tests. When a failure is not explained by an existing rule of its bucket,
    a greedy conjunction of features is learned from the bucket's unexplained
    failures in the window: it must cover at least `min_coverage` of them, and
    at least `precision` of all window samples matching it must be that
    bucket. Rules accumulate per bucket (sequential covering), so suppressing
    one region does not make the next relearning forget it. A candidate
    matching a rule is tested only with probability max(min_explore, K/hits),
    K = 3 for an audited signature and 8 for an automatic bucket, which may
    still merge mechanisms: a bucket is revisited in inverse proportion to how
    often it was already seen (cf. AFLFast's power schedules). The explored
    samples keep the window evidence current, and a rule whose recent matches
    fall below `precision - 0.2` is retired.
    """

    def __init__(self, window=1024, precision=0.9, min_explore=0.01, min_samples=5,
                 min_coverage=0.5, max_terms=6, max_rules=4):
        if window < 64 or not 0.5 < precision <= 1 or not 0 < min_explore <= 1:
            raise ValueError('Invalid quarantine window, precision or exploration floor')
        self.window, self.precision, self.min_explore = window, precision, min_explore
        self.min_samples, self.min_coverage = min_samples, min_coverage
        self.max_terms, self.max_rules = max_terms, max_rules
        self.vocabulary = {}
        self.samples = []  # (frozenset of feature ids, bucket or None for a pass), oldest first
        self.hits = Counter()
        self.rules = {}  # bucket -> [frozenset of feature ids]
        self.rejected, self.explored = Counter(), Counter()
        self.forced = self.learned = self.retired = 0

    def ids(self, features):
        # Sorted: string hashing is salted per process, and ids break ties.
        return frozenset(self.vocabulary.setdefault(f, len(self.vocabulary)) for f in sorted(features))

    def epsilon(self, bucket):
        k = 8.0 if ':' in bucket else 3.0  # automatic buckets are root_cause:digest
        return max(self.min_explore, min(1.0, k / max(1, self.hits[bucket])))

    def matches(self, features):
        known = {i for i in map(self.vocabulary.get, features) if i is not None}
        return sorted(bucket for bucket, rules in self.rules.items() if any(rule <= known for rule in rules))

    def admit(self, features, force=False):
        """True to test a candidate; False to draw another one instead."""
        matched = self.matches(features)
        if not matched:
            return True
        if random.random() < min(map(self.epsilon, matched)):
            self.explored.update(matched)
            return True
        if force:
            self.forced += 1
            return True
        self.rejected.update(matched)
        return False

    def observe(self, features, bucket=None, learn=True):
        """Record a test outcome. learn=False still counts a failure, but its
        bucket gets no rule: an oracle mismatch message does not identify the
        miscompilation, so one bucket may hold several."""
        ids = self.ids(features)
        self.samples.append((ids, bucket))
        del self.samples[:-self.window]
        if bucket is not None:
            self.hits[bucket] += 1
            rules = self.rules.get(bucket, [])
            if learn and len(rules) < self.max_rules and not any(rule <= ids for rule in rules):
                rule = self.learn(bucket)
                if rule is not None:
                    self.rules[bucket] = rules + [rule]
                    self.learned += 1
        for name, rules in list(self.rules.items()):
            kept = [rule for rule in rules if not (rule <= ids and self._stale(name, rule))]
            self.retired += len(rules) - len(kept)
            if kept:
                self.rules[name] = kept
            else:
                del self.rules[name]

    def _stale(self, bucket, rule):
        labels = [label for ids, label in self.samples if rule <= ids]
        return len(labels) >= self.min_samples and labels.count(bucket) < (self.precision - 0.2) * len(labels)

    def learn(self, bucket):
        rules = self.rules.get(bucket, [])
        positives = [ids for ids, label in self.samples
                     if label == bucket and not any(rule <= ids for rule in rules)]
        negatives = [ids for ids, label in self.samples if label != bucket]
        # Few negatives make any conjunction look precise.
        if len(positives) < self.min_samples or len(negatives) < 4 * self.min_samples:
            return None
        floor = max(self.min_samples, math.ceil(self.min_coverage * len(positives)))
        candidates = {f for f, count in Counter(f for ids in positives for f in ids).items() if count >= floor}
        chosen = []
        while len(chosen) < self.max_terms:
            covered = Counter(f for ids in positives for f in ids & candidates)
            wrong = Counter(f for ids in negatives for f in ids & candidates)
            # Highest precision, then coverage; the feature id breaks ties.
            best = max(((covered[f] / (covered[f] + wrong[f]), covered[f], -f)
                        for f in candidates if covered[f] >= floor), default=None)
            current = len(positives) / (len(positives) + len(negatives))
            if best is None or (chosen and best[0] <= current):
                return None
            chosen.append(-best[2])
            candidates.discard(-best[2])
            positives = [ids for ids in positives if -best[2] in ids]
            negatives = [ids for ids in negatives if -best[2] in ids]
            if len(positives) >= self.precision * (len(positives) + len(negatives)):
                return frozenset(chosen)
        return None

    def stats(self):
        reverse = {i: f for f, i in self.vocabulary.items()}
        return {'rules': {bucket: [sorted(reverse[f] for f in rule) for rule in rules]
                          for bucket, rules in sorted(self.rules.items())},
                'rejected': dict(self.rejected), 'explored': dict(self.explored),
                'forced': self.forced, 'learned': self.learned, 'retired': self.retired,
                'window': len(self.samples)}

    def snapshot(self):
        used = sorted({f for ids, _ in self.samples for f in ids}
                      | {f for rules in self.rules.values() for rule in rules for f in rule})
        remap = {f: i for i, f in enumerate(used)}
        reverse = {i: f for f, i in self.vocabulary.items()}
        return {'version': 1, 'vocabulary': [reverse[f] for f in used],
                'samples': [[sorted(remap[f] for f in ids), label] for ids, label in self.samples],
                'rules': {bucket: [sorted(remap[f] for f in rule) for rule in rules]
                          for bucket, rules in self.rules.items()},
                'hits': dict(self.hits), 'rejected': dict(self.rejected), 'explored': dict(self.explored),
                'forced': self.forced, 'learned': self.learned, 'retired': self.retired}

    def restore(self, state):
        if state.get('version') != 1:
            raise ValueError('Unsupported quarantine state version')
        vocabulary = state.get('vocabulary')
        if not isinstance(vocabulary, list) or any(not isinstance(f, str) for f in vocabulary):
            raise ValueError('Invalid quarantine vocabulary')
        size = len(vocabulary)

        def valid(ids):
            return isinstance(ids, list) and all(type(i) is int and 0 <= i < size for i in ids)
        samples, rules = state.get('samples', []), state.get('rules', {})
        if (not isinstance(samples, list) or not isinstance(rules, dict)
                or any(not isinstance(s, list) or len(s) != 2 or not valid(s[0])
                       or not (s[1] is None or isinstance(s[1], str)) for s in samples)
                or any(not isinstance(r, list) or not all(map(valid, r)) for r in rules.values())):
            raise ValueError('Invalid quarantine samples or rules')
        counts = []
        for name in ('hits', 'rejected', 'explored'):
            values = state.get(name, {})
            if not isinstance(values, dict) or any(type(v) is not int or v < 0 for v in values.values()):
                raise ValueError('Invalid quarantine counts')
            counts.append(Counter(values))
        totals = [state.get(name, 0) for name in ('forced', 'learned', 'retired')]
        if any(type(v) is not int or v < 0 for v in totals):
            raise ValueError('Invalid quarantine counts')
        # Validated in full first: a rejected state leaves this one unchanged.
        self.vocabulary = {f: i for i, f in enumerate(vocabulary)}
        self.samples = [(frozenset(ids), label) for ids, label in samples][-self.window:]
        self.rules = {bucket: [frozenset(rule) for rule in values] for bucket, values in rules.items() if values}
        self.hits, self.rejected, self.explored = counts
        self.forced, self.learned, self.retired = totals
