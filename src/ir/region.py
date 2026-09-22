"""Structured tile IR: lexical regions, explicit values, and loop-carried results.

Versions 1–3 use full float32 tiles. Version 4 infers a dtype, shape and
tensor/buffer kind for each value; storage dtype belongs to the input/output
specification.
"""
from dataclasses import dataclass, field, asdict
import math
from src.ir.region_ops import OPS
from src.ir.ir import TileKernel, ComputeKind, DataType, LoopKind


@dataclass
class Operation:
    kind: str
    result: str
    operands: list[str] = field(default_factory=list)
    attrs: dict = field(default_factory=dict)
    regions: list = field(default_factory=list)


@dataclass
class Region:
    arguments: list[str] = field(default_factory=list)
    operations: list[Operation] = field(default_factory=list)
    yield_value: str = ''

    @classmethod
    def from_dict(cls, data):
        return cls(data['arguments'], [Operation(o['kind'], o['result'], o['operands'],
                   o['attrs'], [cls.from_dict(r) for r in o['regions']])
                   for o in data['operations']], data['yield_value'])


def walk(region):
    for op in region.operations:
        yield op
        for child in op.regions:
            yield from walk(child)


@dataclass
class Function:
    """A tensor function with explicit arguments and a single tensor result.

    Typed functions may read inputs and use private scratch internally.
    """
    name: str
    body: Region
    argument_types: list[dict] | None = None

    def to_dict(self):
        data = {'name': self.name, 'body': asdict(self.body)}
        if self.argument_types is not None:
            data['argument_types'] = self.argument_types
        return data


@dataclass
class RegionExecution:
    """Reproducible input and oracle choices, included in the program identity."""
    input_pattern: str = 'normal'
    input_seed_count: int = 2
    repeat_count: int = 3
    schedule_pair: bool = True
    input_layout_a: str = 'contiguous'
    input_layout_b: str = 'contiguous'
    # MLIRSmith-style schedule sweep: one program is executed across several
    # num_stages / loop_kind configurations sharing the same reference.
    stage_sweep: bool = False
    loop_sweep: bool = False
    # Layout sweep: physical programs also run each input case against an
    # alternate matrix layout pair (the reference recomputes from the actual
    # strided views, so no per-layout reference bookkeeping is needed).
    layout_sweep: bool = False
    # Compilation-knob sweeps: a pass-config pair (tilelang pass_configs via
    # the jit decorator; triton enable_fp_fusion), a tilelang threadblock
    # rasterization swizzle pair and a tilelang GemmWarpPolicy pair (FullRow /
    # FullCol variants of the base gemm). All keep the kernel source's math
    # intact, so every variant still shares the one reference.
    pass_sweep: bool = False
    swizzle_sweep: bool = False
    warp_policy_sweep: bool = False

    def to_dict(self):
        data = asdict(self)
        # Preserve identities of v3 programs saved before layout support.
        for field_name in ('input_layout_a', 'input_layout_b'):
            if data[field_name] == 'contiguous':
                del data[field_name]
        for field_name in ('stage_sweep', 'loop_sweep', 'layout_sweep',
                           'pass_sweep', 'swizzle_sweep', 'warp_policy_sweep'):
            if not data[field_name]:
                del data[field_name]
        return data

    def validate(self):
        from src.ir.layout import MATRIX_LAYOUTS
        if self.input_layout_a not in MATRIX_LAYOUTS or self.input_layout_b not in MATRIX_LAYOUTS:
            raise ValueError('Unsupported region input layout')
        if self.input_pattern not in ('normal', 'integer', 'alternating'):
            raise ValueError('Unsupported region input pattern')
        if type(self.input_seed_count) is not int or not 1 <= self.input_seed_count <= 8:
            raise ValueError('Region input_seed_count must be in 1..8')
        if type(self.repeat_count) is not int or not 1 <= self.repeat_count <= 8:
            raise ValueError('Region repeat_count must be in 1..8')
        if type(self.schedule_pair) is not bool:
            raise ValueError('Region schedule_pair must be boolean')
        if type(self.stage_sweep) is not bool or type(self.loop_sweep) is not bool \
                or type(self.layout_sweep) is not bool or type(self.pass_sweep) is not bool \
                or type(self.swizzle_sweep) is not bool or type(self.warp_policy_sweep) is not bool:
            raise ValueError('Region stage_sweep/loop_sweep/layout_sweep/pass_sweep/swizzle_sweep/warp_policy_sweep must be boolean')


@dataclass
class RegionProgram:
    spec: TileKernel
    body: Region
    value_type: str = 'tile<float32>'  # Entry result contract; v4 infers internal types.
    input_scale: float = 0.1  # v1 programs used this scale; new generation sets it explicitly.
    functions: list[Function] = field(default_factory=list)
    execution: RegionExecution | None = None  # None uses the single-execution harness.
    typed: bool = False

    def all_operations(self):
        for function in self.functions:
            yield from walk(function.body)
        yield from walk(self.body)

    def call_label(self):
        """Static call sites in lexical order, not a runtime execution trace."""
        names = {fn.name: f'f{i}' for i, fn in enumerate(self.functions)}
        def label(name, body):
            calls = [names[op.attrs['callee']] for op in walk(body) if op.kind == 'call']
            return name + '(' + '+'.join(calls) + ')'
        return '__'.join([label('main', self.body)] +
                         [label(names[fn.name], fn.body) for fn in self.functions])

    def to_dict(self):
        data = {'type': 'region', 'version': 1, 'value_type': self.value_type,
                'spec': {**self.spec.params_dict, 'dtype': self.spec.dtype.value,
                         'name': self.spec.name}, 'body': asdict(self.body), 'input_scale': self.input_scale}
        if self.functions:
            data.update(version=2, functions=[fn.to_dict() for fn in self.functions])
        if self.execution is not None:
            data.update(version=3, execution=self.execution.to_dict())
        if any(o.kind == 'index_add' or set(o.attrs) & {'predicate', 'modulus', 'start', 'step'}
               for o in self.all_operations()):
            data['version'] = 3
        if self.typed:
            data.update(version=4, typed=True)
        return data

    @classmethod
    def from_dict(cls, data):
        if data.get('version') not in (1, 2, 3, 4):
            raise ValueError('Unsupported region IR version')
        if data.get('version') not in (3, 4) and data.get('execution') is not None:
            raise ValueError('Execution settings require region IR version 3 or 4')
        if data.get('version') == 1 and data.get('functions'):
            raise ValueError('Functions require region IR version 2')
        if data.get('legacy') is not None:
            raise ValueError('Legacy region wrappers are no longer supported')
        p = dict(data['spec'])
        p.pop('alpha', None)  # Ignored field in previously saved native region specs.
        p['dtype'] = DataType(p['dtype'])
        p['compute_kind'] = ComputeKind(p['compute_kind'])
        p['loop_kind'] = LoopKind(p['loop_kind'])
        obj = cls(TileKernel(**p), Region.from_dict(data['body']), data['value_type'], data.get('input_scale', 0.1))
        obj.functions = [Function(fn['name'], Region.from_dict(fn['body']), fn.get('argument_types')) for fn in data.get('functions', [])]
        obj.typed = data.get('typed', False)
        if obj.typed != (data['version'] == 4):
            raise ValueError('Typed regions require IR version 4')
        if data.get('execution') is not None:
            obj.execution = RegionExecution(**data['execution'])
        obj.validate()
        return obj

    @property
    def params_dict(self):
        return {'region_program': self.to_dict()}

    def validate(self):
        from src.ir.region_ops import TYPED_OPS
        contracts = {**OPS, **TYPED_OPS} if self.typed else OPS
        if type(self.typed) is not bool:
            raise ValueError('Invalid typed flag')
        if self.typed and self.execution is None:
            raise ValueError('Typed regions require execution settings and native IR')
        if self.execution is not None:
            self.execution.validate()
        if self.value_type != 'tile<float32>':
            raise ValueError('Unsupported region value type')
        p = self.spec
        if self.body.operations and self.body.operations[0].kind == 'probe':
            op = self.body.operations[0]
            if (self.typed or self.execution is not None or self.functions or not p.coverage_probe or self.body.arguments or len(self.body.operations) != 1
                    or op.operands or op.regions or self.body.yield_value != op.result):
                raise ValueError('Probe must be a complete function with a probe specification')
            return
        from src.backends.common.region_profile import validate_dimensions, validate_target
        validate_dimensions(p)
        if not math.isfinite(self.input_scale) or self.input_scale <= 0:
            raise ValueError('Invalid input scale')
        validate_target(self)
        if self.body.arguments or not self.body.operations or self.body.operations[0].kind not in ('load', 'gemm'):
            raise ValueError('Function must start with a load or GEMM')
        if (self.execution is not None and self.body.operations[0].kind == 'load'
                and self.execution.input_layout_b != 'contiguous'):
            raise ValueError('Load regions do not use a second input layout')
        definitions = set()
        available = {}
        if len(self.functions) > 8:
            raise ValueError('Function budget exceeded')
        count = 0
        def define(name):
            if not name.isidentifier() or name in definitions:
                raise ValueError('Invalid or duplicate SSA value: ' + name)
            definitions.add(name)
        def check(region, parent, depth, helper=False):
            nonlocal count
            if depth > 4:
                raise ValueError('Region nesting limit exceeded')
            visible = set(parent)
            for arg in region.arguments:
                define(arg)
                visible.add(arg)
            for i, op in enumerate(region.operations):
                count += 1
                if count > 128:
                    raise ValueError('Region operation budget exceeded')
                if any(v not in visible for v in op.operands):
                    raise ValueError('Operand is not visible: ' + op.kind)
                contract = contracts.get(op.kind)
                if op.kind == 'call':
                    callee = available.get(op.attrs.get('callee'))
                    if callee is None or len(op.operands) != len(callee.body.arguments) or op.regions:
                        raise ValueError('Invalid call target/arity (forward or recursive calls are forbidden)')
                elif contract is None or len(op.operands) != contract.arity:
                    raise ValueError('Invalid operation/arity: ' + op.kind)
                if op.kind in ('load', 'gemm') and (helper or depth or i != 0):
                    raise ValueError('Input initialization is only legal at function entry')
                if op.kind == 'scale' and (not isinstance(op.attrs.get('alpha'), (float, int))
                                          or not math.isfinite(op.attrs['alpha'])):
                    raise ValueError('Invalid scale')
                if op.kind == 'round' and op.attrs.get('dtype') not in ('float16', 'float32'):
                    raise ValueError('Invalid rounding dtype')
                if op.kind == 'index_add':
                    if (op.attrs.get('axis') not in ('row', 'column', 'iteration')
                            or op.attrs.get('scale') not in (0.25, 0.5, 1.0, 2.0)):
                        raise ValueError('Invalid index_add attributes')
                if op.kind == 'for':
                    if type(op.attrs.get('trip_count')) is not int or not 0 <= op.attrs['trip_count'] <= 4:
                        raise ValueError('Loop must have a bounded nonnegative trip count')
                    for attr, lo, hi in (('start', 0, 3), ('step', 1, 3)):
                        value = op.attrs.get(attr, 0 if attr == 'start' else 1)
                        if type(value) is not int or not lo <= value <= hi:
                            raise ValueError('Invalid loop ' + attr)
                if op.kind == 'if':
                    modulus = op.attrs.get('modulus', 2)
                    parity = op.attrs.get('parity')
                    if (op.attrs.get('predicate', 'row') not in ('row', 'column', 'checkerboard', 'iteration')
                            or type(modulus) is not int or modulus not in (2, 3, 4)
                            or type(parity) is not int or not 0 <= parity < modulus):
                        raise ValueError('Unsupported branch predicate')
                expected = 0 if op.kind == 'call' else contract.regions
                if len(op.regions) != expected:
                    raise ValueError('Wrong region count')
                for child in op.regions:
                    if len(child.arguments) != 1:
                        raise ValueError('Region requires one carried/input tile argument')
                    check(child, visible, depth + 1, helper)
                define(op.result)
                visible.add(op.result)
            if region.yield_value not in visible:
                raise ValueError('Yield value is not visible')
        import keyword
        for fn in self.functions:
            if fn.argument_types is not None and not self.typed:
                raise ValueError('Typed function signatures require region v4')
            if (not fn.name.isidentifier() or keyword.iskeyword(fn.name)
                    or not fn.name.startswith('fn_') or fn.name in available):
                raise ValueError('Invalid or duplicate function name')
            if not 1 <= len(fn.body.arguments) <= 3:
                raise ValueError('Function requires one to three tile arguments')
            definitions.clear()
            check(fn.body, set(), 0, helper=True)
            available[fn.name] = fn
        definitions.clear()
        check(self.body, set(), 0)
        if self.typed:
            from src.ir.region_types import infer_program
            infer_program(self)
