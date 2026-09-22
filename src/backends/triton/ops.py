"""Shared Triton expression builders for native and typed region lowering.

One table per backend keeps the backend-specific DSL surface in the backend
module instead of duplicated lambda dicts across the region/typed emitters.
Entries operate on pre-cast operand strings: native call sites pass bare value
names, typed call sites pass their .to(tl.float32) strings, and `index_term`
carries each caller's iteration-index expression. Emitted source stays
byte-identical to the pre-refactor dicts.
"""

import triton.language as tl

# tanh has no tl.tanh in any supported Triton; only the libdevice bindings
# provide it. Triton 2.x exposes tl.extra.libdevice, Triton 3.x moved it to
# tl.extra.cuda.libdevice. Emitting a path this Triton does not provide fails
# in the JIT front-end (AttributeError while visiting the AST) before any real
# compilation, turning every tanh program into wasted fuzzing. Probe once at
# import time; when neither binding exists, fall back to an exp-based identity
# that matches torch.tanh within fp32 tolerance.
_TANH_BINDING = None
_extra = getattr(tl, 'extra', None)
for _mod in (_extra, getattr(_extra, 'cuda', None)):
    if _mod is not None and hasattr(getattr(_mod, 'libdevice', None), 'tanh'):
        _TANH_BINDING = 'tl.extra.cuda.libdevice.tanh' if _mod is not _extra else 'tl.extra.libdevice.tanh'
        break


def elementwise_expr(kind, x, y, z, attrs, index_term):
    """Expression string for one elementwise region op.

    x/y/z are operand strings in the caller's access convention; index_term is
    the index string for `index_add` (built by the caller's axis machinery,
    empty for every other op).
    """
    return {
        'scale': lambda: f'{x} * {attrs["alpha"]}',
        'neg': lambda: f'-{x}',
        'index_add': lambda: f'{x} + {index_term} * {attrs["scale"]}',
        'abs': lambda: f'tl.abs({x})',
        'sqrt': lambda: f'tl.sqrt(tl.abs({x}))',
        'round': lambda: f'{x}.to(tl.{attrs["dtype"]}).to(tl.float32)',
        'copy': lambda: x,
        'exp': lambda: f'tl.exp(tl.minimum(tl.maximum({x}, -10.), 10.))',
        'sub': lambda: f'{x} - {y}',
        'maximum': lambda: f'tl.maximum({x}, {y})',
        'minimum': lambda: f'tl.minimum({x}, {y})',
        'div': lambda: f'{x} / tl.maximum(tl.abs({y}), 0.001)',
        'where': lambda: f'tl.where({x} > 0, {y}, {z})',
        'tile_transpose': lambda: f'tl.trans({x})',
        'add': lambda: f'{x} + {y}',
        'mul': lambda: f'{x} * {y}',
        # Transcendentals: sanitization must stay byte-identical to the
        # reference interpreters (region_runtime / typed_region_runtime).
        # tanh uses the probed libdevice binding (see module header).
        'tanh': (lambda: f'{_TANH_BINDING}({x})' if _TANH_BINDING
                 else f'(1.0 - 2.0 / (1.0 + tl.exp(2.0 * ({x}))))'),
        'erf': lambda: f'tl.erf({x})',
        'log': lambda: f'tl.log(tl.maximum(tl.abs({x}), 0.001))',
        'log2': lambda: f'tl.log2(tl.maximum(tl.abs({x}), 0.001))',
        'exp2': lambda: f'tl.exp2(tl.minimum(tl.maximum({x}, -10.), 10.))',
        'rsqrt': lambda: f'tl.rsqrt(tl.maximum(tl.abs({x}), 1e-6))',
        'sin': lambda: f'tl.sin({x})',
        'cos': lambda: f'tl.cos({x})',
        'floor': lambda: f'tl.floor({x})',
        'ceil': lambda: f'tl.ceil({x})',
    }[kind]()
