"""DSL versions this harness revision targets, and what is installed here.

The harness is written against one pair of upstream releases and records that
pair in every campaign's summary.json. Upstream moves internal layout between
releases (TileLang 0.1.14 moved the device-compile callback out of
engine/lower.py; Triton 3.8 requires string signature keys), so two kinds of
coupling exist:

* adaptive shims, resolved at run time in the generated harnesses
  (`backends/tilelang/extended.py`, `backends/triton/extended.py`), which work
  on both the target and the legacy pair;
* wording-coupled diagnostics (`backends/*/diagnostics.py`) and pass-config
  pools (`backends/common/knobs.py`), which cannot adapt on their own -- a
  version check is the only thing that turns their silent rot into a visible
  message.

A version mismatch is therefore reported, never raised: the harness still runs
(that is how a newer DSL gets fuzzed before the shims catch up), but the
campaign records what it actually ran against.
"""
import importlib
import importlib.metadata
import sys

# `tp_fuzzing_latest` on both campaign servers: the pair this revision is
# validated against.
TARGET_TILELANG = '0.1.14'
TARGET_TRITON = '3.8.0'
TARGET_TORCH = '2.4.0'
# `tp_fuzzing`: the pair the previous revision was written against. The
# adaptive shims keep both working; diagnostics/pool parity is only claimed
# for the target pair.
LEGACY_TILELANG = '0.1.11'
LEGACY_TRITON = '3.0.0'

DISTRIBUTIONS = {'tilelang': TARGET_TILELANG, 'triton': TARGET_TRITON, 'torch': TARGET_TORCH}


def installed(name: str):
    """Installed version of a distribution, or None when it is not present.

    Distribution metadata first (no import cost, works for torch), module
    ``__version__`` as the fallback for locally built packages.
    """
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        pass
    except Exception:
        pass
    try:
        return getattr(importlib.import_module(name), '__version__', None)
    except ImportError:
        return None
    except Exception:
        return None


def environment() -> dict:
    """Versions recorded in summary.json for every campaign."""
    record = {name: installed(name) for name in DISTRIBUTIONS}
    record['python'] = '%d.%d.%d' % sys.version_info[:3]
    return record


def mismatches() -> list:
    """Human-readable notes for every installed version != target.

    Empty when this machine runs the target pair; missing packages (the
    harness machine need not import the DSLs at all) are not reported.
    """
    notes = []
    for name, target in DISTRIBUTIONS.items():
        found = installed(name)
        if found is not None and found != target:
            notes.append(f'{name} {found} (target {target})')
    return notes


def describe() -> str:
    """One-line version banner for campaign logs."""
    record = environment()
    text = ', '.join(f'{name}={record[name] or "absent"}' for name in DISTRIBUTIONS)
    notes = mismatches()
    if notes:
        text += '  [not the target pair: ' + '; '.join(notes) + ']'
    return text
