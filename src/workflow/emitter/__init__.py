"""Standalone numeric helpers and backend emitter entry points."""
from importlib import import_module

_EXPORTS = {name: (f'src.backends.{backend}.emitter', name)
            for backend, name in [('tilelang', 'TileLangEmitter'), ('triton', 'TritonEmitter')]}
__all__ = list(_EXPORTS) + ['get_emitter']


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(name)
    module, attribute = _EXPORTS[name]
    value = getattr(import_module(module), attribute)
    globals()[name] = value
    return value


def get_emitter(backend: str, config=None):
    """Return the appropriate emitter for a backend, optionally with config."""
    from src.backends import get_backend
    return get_backend(backend).make_emitter(config)


def _load_runtime_sources():
    # Snapshot at import time so editing source files during a running campaign
    # cannot mix code objects with inspect's newer source line offsets.
    import inspect
    from .runtime import _finite_compare
    return {fn.__name__: inspect.getsource(fn) for fn in
            (_finite_compare,)}


_RUNTIME_SOURCES = _load_runtime_sources()


def _threshold_header(config) -> str:
    """Embed standalone numeric checks and a reproducible input seed."""
    if config is None:
        from src.config import DEFAULT_CONFIG
        config = DEFAULT_CONFIG
    return f'torch.manual_seed({config.input_seed!r})\n\n' + _RUNTIME_SOURCES['_finite_compare']
