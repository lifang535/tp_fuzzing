"""Add observation-only hooks to copies of selected saved reproducers."""
import json
from pathlib import Path
import sys
from replay import ROOT, OUT, run

HOOK = r'''
import json as _json
_original_finite = _finite_compare
def _observed_finite(actual, expected, *args, **kwargs):
    a, b = actual.detach().float(), expected.detach().float()
    delta = (a-b).abs()
    finite = torch.isfinite(delta)
    coords = torch.nonzero(finite & (delta > 0))[:8]
    record = {'shape': list(a.shape), 'a_max': a.abs().max().item(), 'b_max': b.abs().max().item(),
              'a_mean': a.abs().mean().item(), 'b_mean': b.abs().mean().item(),
              'maxdiff': delta[finite].max().item() if finite.any() else None,
              'differences': int((a != b).sum()),
              'samples': [(c.tolist(), a[tuple(c)].item(), b[tuple(c)].item()) for c in coords]}
    print('NUMERIC', _json.dumps(record), flush=True)
    return _original_finite(actual, expected, *args, **kwargs)
_finite_compare = _observed_finite
if '_region_check' in globals():
    _original_check = _region_check
    def _region_check(actual, expected, relative, tolerance, label):
        print('CHECK', label, flush=True)
        try: return _original_check(actual, expected, relative, tolerance, label)
        except RuntimeError as e: print('OBSERVED_FAILURE', str(e), flush=True)
if '_region_equal' in globals():
    _original_equal = _region_equal
    def _region_equal(actual, expected, label):
        try: return _original_equal(actual, expected, label)
        except RuntimeError as e:
            print('EXACT_FAILURE', label, flush=True)
            _observed_finite(actual, expected)
'''
records = [json.loads(line) for line in (OUT/'replays.jsonl').read_text().splitlines()]
for ident in sys.argv[1:]:
    source = next(ROOT.glob('results/*/failed/*/*'+ident+'.py'))
    target = OUT/(ident+'_observed.py')
    code = source.read_text().replace("if __name__ == '__main__':", HOOK+"\nif __name__ == '__main__':")
    target.write_text(code)
    previous = next((r for r in reversed(records) if r['label'] == ident), {})
    run(ident+'_observed', target, 90, cache=previous.get('cache'))
