"""Audit the installed DSL facade against executed extend campaigns.

This is an inventory, not a claim that one successful sample covers every
argument, dtype, architecture, or compiler path of an API.
"""
import argparse
import ast
from collections import Counter
from importlib.metadata import version
import inspect
import json
from pathlib import Path

from src.workflow.generator.dsl_extend import DSL_OPS


# An extension maps to the public facade entry it actually invokes. Entries
# without execution evidence stay "implemented", never "verified".
EXTENSION_API = {
    'triton': {
        'join': 'join', 'split': 'split', 'interleave': 'interleave',
        'scan_sum': 'cumsum', 'scan_product': 'cumprod', 'sort': 'sort',
        'histogram': 'histogram', 'argmax': 'argmax', 'argmin': 'argmin',
        'xor_sum': 'xor_sum', 'dsl_sigmoid': 'sigmoid',
        'dsl_clamp': 'clamp', 'softmax': 'softmax',
        'topk': 'topk', 'gather': 'gather',
        'atomic_and': 'atomic_and', 'atomic_or': 'atomic_or', 'atomic_xor': 'atomic_xor',
    },
    'tilelang': {
        'pipelined_for': 'Pipelined', 'scan_sum': 'cumsum',
        'scan_max': 'cummax', 'reduce_abssum': 'reduce_abssum',
        'reduce_absmax': 'reduce_absmax', 'reduce_bitand': 'reduce_bitand',
        'reduce_bitor': 'reduce_bitor', 'reduce_bitxor': 'reduce_bitxor',
        'dsl_sigmoid': 'sigmoid', 'dsl_clamp': 'clamp',
    },
}


def public_callables(backend):
    if backend == 'triton':
        import triton.language as facade
        package = 'triton'
    elif backend == 'tilelang':
        import tilelang.language as facade
        package = 'tilelang'
    else:
        raise ValueError(backend)
    prefix = 'triton.language' if backend == 'triton' else ('tilelang.language', 'tilelang.cuda.language')
    entries = {}
    for name in dir(facade):
        if name.startswith('_'):
            continue
        value = getattr(facade, name)
        module = getattr(value, '__module__', '')
        if callable(value) and module.startswith(prefix):
            entries[name] = {'module': module, 'kind': ('type' if inspect.isclass(value)
                                                        else 'function' if inspect.isfunction(value)
                                                        else 'callable')}
    return version(package), entries


def campaign_passes(paths, backend):
    passed = Counter()
    for path in paths:
        data = json.loads(Path(path).read_text())
        by_op = data.get('by_op')
        if not isinstance(by_op, dict):
            by_op = data.get('dsl_extension', {}).get('by_op')
        if data.get('backend') != backend or not isinstance(by_op, dict):
            raise ValueError(f'{path} is not a {backend} extension summary')
        for op in DSL_OPS[backend]:
            passed[op] += by_op.get(op + ':passed', 0)
    return passed


def passing_code_calls(paths, backend):
    """Record direct facade call sites in saved, passing Python reproducers."""
    alias = 'tl' if backend == 'triton' else 'T'
    files = Counter()
    for directory in paths:
        if not directory.is_dir() or directory.name != 'passed':
            raise ValueError(f'{directory} must be a passing-reproducer directory named passed')
        for path in directory.glob('*.py'):
            tree = ast.parse(path.read_text(), filename=str(path))
            names = {node.func.attr for node in ast.walk(tree)
                     if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                     and isinstance(node.func.value, ast.Name) and node.func.value.id == alias}
            files.update(names)
    return files


def audit(backend, names, passed, passing_code=None):
    mapping = EXTENSION_API[backend]
    if set(mapping) != set(DSL_OPS[backend]):
        raise ValueError('Extension API map differs from the generator registry')
    reverse = {api: op for op, api in mapping.items()}
    passing_code = passing_code or Counter()
    rows = []
    for name, meta in sorted(names.items()):
        op = reverse.get(name)
        status = ('executed_pass' if op and passed.get(op, 0) > 0 else
                  'seen_in_passing_code' if passing_code.get(name, 0) > 0 else
                  'implemented_no_run' if op else 'unmapped')
        rows.append({'api': name, **meta, 'extension_op': op,
                     'status': status, 'passing_cases': passed.get(op, 0) if op else 0,
                     'passing_code_files': passing_code.get(name, 0)})
    missing = {api: op for op, api in mapping.items() if api not in names}
    return {'api_count': len(rows), 'kind_counts': dict(Counter(row['kind'] for row in rows)),
            'status_counts': dict(Counter(row['status'] for row in rows)),
            'missing_in_installed_version': missing, 'entries': rows}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend', required=True, choices=tuple(DSL_OPS))
    parser.add_argument('--summary', type=Path, action='append', default=[],
                        help='summary.json from a standalone or integrated extend campaign; repeatable')
    parser.add_argument('--passed-code-dir', type=Path, action='append', default=[],
                        help='passed/ directory with successful .py reproducers; repeatable')
    parser.add_argument('--output', type=Path, help='Write JSON inventory here')
    args = parser.parse_args(argv)
    installed_version, names = public_callables(args.backend)
    result = {'backend': args.backend, 'installed_version': installed_version,
              'scope': 'public callable names owned by the installed language facade namespaces',
              **audit(args.backend, names, campaign_passes(args.summary, args.backend),
                      passing_code_calls(args.passed_code_dir, args.backend))}
    rendered = json.dumps(result, indent=2, ensure_ascii=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + '\n')
    print(rendered)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
