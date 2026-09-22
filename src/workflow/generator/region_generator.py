"""Two-phase generation: recursive structure first, lexical SSA binding second."""
import random
from dataclasses import dataclass, field
from src.ir import DataType
from src.ir.region import Operation, Region, RegionProgram, Function, RegionExecution, walk
from src.ir.region_ops import OPS, TYPED_OPS
from src.ir.region_types import FULL, ValueType, result_type, scratch_bytes


@dataclass
class TemplateOp:
    kind: str
    regions: list = field(default_factory=list)
    callee: str | None = None
    carry_merge: bool = False


@dataclass
class FunctionTemplate:
    name: str
    arguments: list[str]
    operations: list[TemplateOp]


@dataclass
class ProgramTemplate:
    entry: list[TemplateOp]
    functions: list[FunctionTemplate] = field(default_factory=list)


# Boundary step ops: ceil/floor/round (and the typed cast) quantize their
# input. A fp32 GEMM computes TF32 tensor-core math in the kernel (tilelang
# T.gemm and triton tl.dot) while the torch reference is exact fp32, so
# ~0.5-1% of elements land on opposite sides of a rounding boundary and flip
# by one ulp. The reference check then reports a large "relative" error for
# pure numeric noise — no compiler bug can be distinguished inside it, and
# ceil/floor of near-integers is degenerate anyway. Other dtypes are exact
# (fp16 products accumulate in fp32; int8 in int32), so the trap is
# fp32-GEMM-specific.
_STEP_OPS = ('ceil', 'floor', 'round', 'cast')


def program_has_gemm(program) -> bool:
    """Any GEMM anywhere in the op tree (body or function bodies)."""
    from src.ir.region import walk
    regions = [program.body] + [fn.body for fn in program.functions]
    return any(op.kind == 'gemm' for region in regions for op in walk(region))


def gemm_step_op_program(program) -> bool:
    """A GEMM and a boundary step op coexist anywhere in the op tree."""
    from src.ir.region import walk
    regions = [program.body] + [fn.body for fn in program.functions]
    ops = [op for region in regions for op in walk(region)]
    return any(op.kind == 'gemm' for op in ops) and any(op.kind in _STEP_OPS for op in ops)


def step_op_noise_trap(program) -> bool:
    """fp32 GEMM + step op: kernel/reference TF32-vs-fp32 rounding flips step
    boundaries and drowns the oracle in wrong_result noise."""
    return program.spec.dtype == DataType.FLOAT32 and gemm_step_op_program(program)


class RegionGenerator:
    LEAVES = tuple(k for k, spec in OPS.items() if not spec.entry and not spec.regions)

    def __init__(self, config, backend="tilelang", type_gen=None, grids=None):
        if not 0 <= config.region_max_depth <= 4 or not 1 <= config.region_max_ops <= 127 or config.region_max_length < 1:
            raise ValueError("Invalid region depth/operation/length budget")
        if not 0 <= config.coverage_probe_prob <= 1:
            raise ValueError("coverage_probe_prob must be between 0 and 1")
        if not 0 <= config.region_gemm_prob <= 1 or not 0 <= config.latest_value_prob <= 1:
            raise ValueError('Invalid entry/operand selection probability')
        if not 0 <= config.function_min_count <= config.function_max_count <= 8 or not 0 <= config.function_call_prob <= 1:
            raise ValueError('Invalid function count/call probability')
        if config.region_max_ops < 2 * config.function_min_count:
            raise ValueError('Region operation budget is too small for minimum function count')
        if config.coverage_probe_prob and config.probe_repeat_count < 2:
            raise ValueError("probe_repeat_count must be at least 2")
        self.config = config
        if not 0 <= config.region_layout_prob <= 1:
            raise ValueError('region_layout_prob must be between 0 and 1')
        RegionExecution(input_seed_count=config.region_input_seed_count,
                        repeat_count=config.region_repeat_count,
                        schedule_pair=config.region_schedule_pair).validate()
        self.backend = backend
        from src.backends import get_backend
        self.backend_impl = get_backend(backend)
        if type_gen is None:
            from .generator import TypeGenerator
            type_gen = TypeGenerator(config)
        self.type_gen = type_gen
        self.feedback = None
        self.grids = grids
        self._int8 = False
        if not 0 <= config.region_int8_prob <= 1:
            raise ValueError('region_int8_prob must be between 0 and 1')
        if not 0 <= config.region_typed_prob <= 1 or config.region_scratch_max_bytes < 4096:
            raise ValueError('Invalid typed-region probability or scratch budget')

    def template(self, depth=0, budget=None, parent="function", functions=(), has_buffer=False):
        if budget is None:
            budget = [self.config.region_max_ops]
        nodes = []
        # One explicit adapter closes each generated region. This keeps its
        # interface stable while its interior can hold mixed types and shapes.
        normalize = self.config.region_typed_prob > 0 and budget[0] > 0
        if normalize:
            budget[0] -= 1
        for _ in range(random.randint(self.config.region_min_length if depth == 0 else 1, self.config.region_max_length)):
            if budget[0] <= 0:
                break
            choices = list(self.LEAVES)
            if depth < self.config.region_max_depth and budget[0] >= 3:
                choices += ['for', 'if']
            controls = [k for k in choices if k in ("if", "for")]
            if controls and random.random() < self.config.region_control_prob:
                choices = controls
            else:
                choices = [k for k in choices if k not in ("if", "for")]
                if random.random() < self.config.region_typed_prob:
                    choices = ['cast', 'reduce_tile', 'broadcast_tile', 'load_input', 'store_tile']
                    if has_buffer:
                        choices += ['load_tile', 'write_tile']
            if self.feedback is None:
                kind = random.choice(choices)
            else:
                from src.workflow.feedback import key
                fb, boost = self.feedback, self.config.uncovered_boost
                weights = [fb.weight(key('op', k), passed_decay=2.0, uncovered_boost=boost)
                           + fb.weight(key('nest', parent, k), passed_decay=2.0, uncovered_boost=boost) - 1.0
                           for k in choices]
                kind = random.choices(choices, weights=weights, k=1)[0]
            budget[0] -= 1
            if kind not in ('if', 'for') and functions and random.random() < self.config.function_call_prob:
                nodes.append(TemplateOp('call', callee=random.choice(functions).name))
                continue
            # Reserve a final merge with the carried tile. This keeps a direct
            # output dependence on the previous iteration within the op budget.
            if kind == 'for':
                budget[0] -= 1
            children = [self.template(depth + 1, budget, kind, functions, has_buffer) for _ in range(2 if kind == 'if' else 1 if kind == 'for' else 0)]
            if kind == 'for':
                children[0].append(TemplateOp('add', carry_merge=True))
            nodes.append(TemplateOp(kind, children))
            has_buffer |= kind == 'store_tile'
        if normalize:
            nodes.append(TemplateOp('to_tile'))
        return nodes

    def instantiate(self, template, initial='load', arguments=None, functions=()):
        function_pool = {fn.name: fn for fn in functions}
        producers = {arg: 'argument' for arg in (arguments or [])}
        value_types = {arg: FULL for arg in (arguments or [])}
        counter = 0
        def fresh():
            nonlocal counter
            counter += 1
            return 'v' + str(counter)
        def fill(nodes, inherited, current, argument=None, in_loop=False, predicates=()):
            values = list(inherited)
            if argument:
                producers[argument] = producers[current]
                value_types[argument] = value_types[current]
                values.append(argument)
                current = argument
            region = Region([argument] if argument else [])
            for node in nodes:
                def choose(expected=None, buffer=False):
                    compatible = [v for v in values if (value_types[v] == expected if expected is not None
                                  else value_types[v].kind == ('buffer' if buffer else 'tensor'))]
                    if not compatible:
                        raise ValueError('Template has no compatible operand for ' + node.kind)
                    if self.feedback is None:
                        return random.choice(compatible)
                    from src.workflow.feedback import key
                    weights = [self.feedback.weight(key('data', producers[v], node.kind), passed_decay=4.0,
                                                    uncovered_boost=self.config.uncovered_boost)
                               for v in compatible]
                    return random.choices(compatible, weights=weights, k=1)[0]
                # Every operand may reuse an earlier dominating tile. Successful
                # data-edge counts guide reuse, not just operation selection.
                expected = FULL if node.kind in ('if', 'for', 'call') else value_types[current]
                operands = [current if value_types[current] == expected and random.random() < self.config.latest_value_prob else choose(expected)]
                attrs = {}
                arity = len(function_pool[node.callee].body.arguments) if node.kind == 'call' else {**OPS, **TYPED_OPS}[node.kind].arity
                if node.kind == 'call':
                    attrs['callee'] = node.callee
                for _ in range(arity - 1):
                    operands.append(choose(expected))
                if not arity:
                    operands = []
                if node.kind in ('load_tile', 'write_tile'):
                    buffer = choose(buffer=True)
                    operands = [buffer]
                    if node.kind == 'write_tile':
                        bt = value_types[buffer]
                        operands.append(choose(ValueType(bt.dtype, bt.shape)))
                if node.kind == 'cast':
                    attrs['dtype'] = random.choice(('float16', 'float32'))
                if node.kind == 'reduce_tile':
                    attrs.update(axis=random.choice((0, 1)), reduction=random.choice(('sum', 'max', 'min')))
                if node.kind == 'load_input':
                    attrs.update(source=random.choice(('A', 'B')) if getattr(self, '_initial', initial) == 'gemm' else 'A',
                                 dtype=random.choice(('float16', 'float32')),
                                 row_offset=random.randrange(4), col_offset=random.randrange(4))
                if node.carry_merge:
                    if argument is None or node.kind != 'add':
                        raise ValueError('Carry merge requires a loop region argument')
                    operands = [current, argument]
                if node.kind == 'index_add':
                    attrs['axis'] = random.choice(('row', 'column', 'iteration') if in_loop else ('row', 'column'))
                    attrs['scale'] = random.choice((0.25, 0.5, 1.0, 2.0))
                if node.kind == 'scale':
                    attrs['alpha'] = random.uniform(self.config.scale_alpha_min, self.config.scale_alpha_max)
                if node.kind == 'round':
                    attrs['dtype'] = random.choice(('float16', 'float32'))
                if node.kind == 'for':
                    attrs.update(trip_count=random.choice((0, 1, 2, 3, 4)),
                                 start=random.randrange(4), step=random.randint(1, 3))
                child_predicates = predicates
                if node.kind == 'if':
                    axes = ('row', 'column', 'checkerboard') + (('iteration',) if in_loop else ())
                    choices = [(axis, mod) for axis in axes for mod in (2, 3, 4)
                               if (axis, mod) not in predicates]
                    axis, modulus = random.choice(choices)
                    attrs.update(predicate=axis, modulus=modulus, parity=random.randrange(modulus))
                    child_predicates = predicates + ((axis, modulus),)
                # An inner loop rebinds the induction value, so old iteration
                # predicates need not constrain the new induction variable.
                if node.kind == 'for':
                    child_predicates = tuple(p for p in predicates if p[0] != 'iteration')
                children = [fill(child, values, operands[0], fresh(), in_loop or node.kind == 'for',
                                 child_predicates) for child in node.regions]
                result = fresh()
                region.operations.append(Operation(node.kind, result, operands, attrs, children))
                values.append(result)
                producers[result] = node.kind
                out_type = (FULL if node.kind == 'call' else value_types[operands[0]] if node.kind in ('for', 'if')
                            else result_type(node.kind, [value_types[v] for v in operands], attrs))
                value_types[result] = out_type
                if out_type.kind == 'tensor':
                    current = result
            region.yield_value = current
            return region
        if arguments is not None:
            body = fill(template, arguments, arguments[0])
            body.arguments = list(arguments)
            return body
        first = fresh()
        producers[first] = initial
        value_types[first] = FULL
        body = fill(template, [first], first)
        body.operations.insert(0, Operation(initial, first))
        return body

    def function_template(self):
        """Select the complete function skeleton before semantic instantiation."""
        if random.random() < self.config.coverage_probe_prob:
            return [TemplateOp('probe')]
        return [TemplateOp('gemm' if random.random() < self.config.region_gemm_prob else 'load')] + self.template()

    def program_template(self, initial=None):
        """Build all function skeletons before binding any SSA operands."""
        if initial is None and random.random() < self.config.coverage_probe_prob:
            return ProgramTemplate([TemplateOp('probe')])
        if initial is None:
            # int8 x int8 GEMM: gemm-only entry, no functions, no typed tail
            # ops. An int32 accumulator cannot feed the float32 fragment
            # contract of the elementwise/typed operations, so the whole body
            # is the gemm and sample_spec switches to the pre-validated
            # INT8_SPEC_GRID. Only the random-entry path rolls it (an explicit
            # initial= keeps deterministic entry control), and only when the
            # config does not require functions (int8 programs have none).
            if (self.config.function_min_count == 0
                    and random.random() < self.config.region_int8_prob):
                self._int8 = True
                return ProgramTemplate([TemplateOp('gemm')])
            initial = 'gemm' if random.random() < self.config.region_gemm_prob else 'load'
        # Templates expose names/signatures, not instantiated SSA values.
        # Reserve one mandatory call per helper/entry connection; all helpers
        # are reachable, while random call sites also occur in nested regions.
        count = min(random.randint(self.config.function_min_count, self.config.function_max_count),
                    self.config.region_max_ops // 2)
        per_body = (self.config.region_max_ops - count) // (count + 1)
        functions = []
        for i in range(count):
            nodes = self.template(budget=[per_body], functions=functions)
            if functions:
                nodes.append(TemplateOp('call', callee=functions[-1].name))
            arguments = [f'arg{j}' for j in range(random.randint(1, 3))]
            functions.append(FunctionTemplate(f'fn_{i}', arguments, nodes))
        nodes = self.template(budget=[per_body], functions=functions)
        if functions:
            nodes.append(TemplateOp('call', callee=functions[-1].name))
        return ProgramTemplate([TemplateOp(initial)] + nodes, functions)

    def instantiate_program(self, template):
        self._initial = template.entry[0].kind
        functions = []
        for fn in template.functions:
            body = self.instantiate(fn.operations, arguments=fn.arguments, functions=functions)
            functions.append(Function(fn.name, body))
        return self.instantiate_function(template.entry, functions)

    def generate(self, initial=None):
        # fp32 GEMM + step-op programs are oracle noise (see step_op_noise_trap);
        # re-roll instead of emitting them. The trap draws are independent and
        # rare enough that the retry bound never binds in practice.
        for _ in range(10):
            program = self.instantiate_program(self.program_template(initial))
            if not step_op_noise_trap(program):
                return program
        raise ValueError('Exhausted generation retries avoiding fp32 GEMM step-op noise')

    def instantiate_function(self, template, functions=()):
        if not template or template[0].kind not in ('load', 'gemm', 'probe') or template[0].regions:
            raise ValueError('Invalid function entry template')
        initial = template[0].kind
        if initial == 'probe':
            if len(template) != 1:
                raise ValueError('Probe requires a whole-function template')
            return self.instantiate_probe()
        dtype = None
        if self._int8:
            # Consume the flag set by program_template for this entry.
            self._int8 = False
            dtype = DataType.INT8
        body = self.instantiate(template[1:], initial, functions=functions)
        spec = self.sample_spec(body, initial, functions, dtype)
        execution = RegionExecution(input_pattern=random.choice(('normal', 'normal', 'integer', 'alternating')),
                                    input_seed_count=self.config.region_input_seed_count,
                                    repeat_count=self.config.region_repeat_count,
                                    schedule_pair=self.config.region_schedule_pair,
                                    stage_sweep=self.config.region_stage_sweep,
                                    loop_sweep=self.config.region_loop_sweep,
                                    layout_sweep=self.config.region_layout_sweep,
                                    pass_sweep=self.config.region_pass_config,
                                    swizzle_sweep=self.config.region_swizzle_pair,
                                    warp_policy_sweep=self.config.region_warp_policy_pair)
        from src.ir.layout import MATRIX_LAYOUTS
        layout_fields = ('input_layout_a', 'input_layout_b') if initial == 'gemm' else ('input_layout_a',)
        for field_name in layout_fields:
            if random.random() < self.config.region_layout_prob:
                setattr(execution, field_name, random.choice(MATRIX_LAYOUTS[1:]))
        input_scale = self.config.region_input_scale
        if spec.dtype == DataType.INT8:
            # Sub-unit scales truncate every int8 input to zero (float->int
            # conversion truncates toward zero), degenerating the int8 GEMM
            # surface into an all-zero matmul. Integer scales >= 1 keep inputs
            # exact and products deep inside the int32 accumulator range.
            input_scale = max(input_scale, 1.0)
        program = RegionProgram(spec, body, input_scale=input_scale,
                                functions=list(functions), execution=execution,
                                typed=any(o.kind in TYPED_OPS for region in [body] + [fn.body for fn in functions]
                                          for o in walk(region)))
        self.backend_impl.validate_program(program)
        self.bound_scratch(program)
        return program

    def bound_scratch(self, program):
        """Bound physical scratch independently of the requested dimension pool."""
        if not program.typed:
            return
        while scratch_bytes(program) > self.config.region_scratch_max_bytes:
            p = program.spec
            if p.M <= p.block_M and p.N <= p.block_N:
                raise ValueError('A single block exceeds the configured scratch budget')
            if p.M >= p.N and p.M > p.block_M:
                p.M = max(1, p.M // 2)
            else:
                p.N = max(1, p.N // 2)

    def instantiate_probe(self):
        program = self.backend_impl.generate_probe(self.config)
        self.backend_impl.validate_program(program)
        return program

    def sample_spec(self, body, initial, functions=(), dtype=None, unchecked_ok=True):
        return self.backend_impl.sample_region_spec(self, body, initial, functions, dtype, unchecked_ok)
