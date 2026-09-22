"""Shared TileLang expression builders for native and typed region lowering.

One table per backend keeps the backend-specific DSL surface in the backend
module instead of duplicated lambda dicts across the region/typed emitters.
Entries operate on pre-cast operand strings: native call sites pass raw
fragment accesses, typed call sites pass their T.cast(...) strings, and
`index_term` carries each caller's iteration-index expression. Emitted source
stays byte-identical to the pre-refactor dicts.
"""


def elementwise_expr(kind, x, y, z, attrs, index_term):
    """Expression string for one elementwise region op.

    x/y/z are operand strings in the caller's access convention; index_term is
    the index string for `index_add` (built by the caller's axis machinery,
    empty for every other op).
    """
    return {
        'scale': lambda: f'{x} * {attrs["alpha"]}',
        'neg': lambda: f'-{x}',
        'index_add': lambda: f'{x} + T.cast({index_term}, "float32") * {attrs["scale"]}',
        'abs': lambda: f'T.abs({x})',
        'sqrt': lambda: f'T.sqrt(T.abs({x}))',
        'round': lambda: f'T.cast(T.cast({x}, "{attrs["dtype"]}"), "float32")',
        'copy': lambda: x,
        'exp': lambda: f'T.exp(T.min(T.max({x}, T.float32(-10)), T.float32(10)))',
        'sub': lambda: f'{x} - {y}',
        'maximum': lambda: f'T.max({x}, {y})',
        'minimum': lambda: f'T.min({x}, {y})',
        'div': lambda: f'{x} / T.max(T.abs({y}), T.float32(0.001))',
        'where': lambda: f'T.if_then_else({x} > 0, {y}, {z})',
        'add': lambda: f'{x} + {y}',
        'mul': lambda: f'{x} * {y}',
        # Transcendentals: sanitization must stay byte-identical to the
        # reference interpreters (region_runtime / typed_region_runtime).
        'tanh': lambda: f'T.tanh({x})',
        'erf': lambda: f'T.erf({x})',
        'log': lambda: f'T.log(T.max(T.abs({x}), T.float32(0.001)))',
        'log2': lambda: f'T.log2(T.max(T.abs({x}), T.float32(0.001)))',
        'exp2': lambda: f'T.exp2(T.min(T.max({x}, T.float32(-10)), T.float32(10)))',
        'rsqrt': lambda: f'T.rsqrt(T.max(T.abs({x}), T.float32(1e-6)))',
        'sin': lambda: f'T.sin({x})',
        'cos': lambda: f'T.cos({x})',
        'floor': lambda: f'T.floor({x})',
        'ceil': lambda: f'T.ceil({x})',
    }[kind]()
