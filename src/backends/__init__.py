"""Backend registry. Importing the registry does not import a DSL runtime.

External modules may register a Backend instance, then select it by name through
the same generator, mutator, oracle and CLI used by the built-in backends.
"""
from importlib import import_module
import re

from .base import Backend

_backends = {}
_builtin_modules = {'tilelang': 'src.backends.tilelang.backend',
                    'triton': 'src.backends.triton.backend'}


def backend_names():
    return tuple(sorted(_builtin_modules.keys() | _backends.keys()))


def register_backend(backend):
    if not isinstance(backend, Backend):
        raise TypeError('A backend must implement Backend')
    if not re.fullmatch(r'[a-z][a-z0-9-]*', backend.name):
        raise ValueError('Backend names must contain lowercase letters, digits or hyphens')
    if backend.name in _backends or backend.name in _builtin_modules:
        raise ValueError('Backend already registered: ' + backend.name)
    _backends[backend.name] = backend


def get_backend(name):
    if name not in _backends:
        module = _builtin_modules.get(name)
        if module is None:
            raise ValueError(f'Unknown backend: {name}; available: {", ".join(backend_names())}')
        _backends[name] = import_module(module).BACKEND
    return _backends[name]


def load_backend_plugins(modules):
    """Load explicitly requested Python modules whose import registers backends."""
    for module in modules:
        import_module(module)
