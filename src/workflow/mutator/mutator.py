"""Mutation through the same recursive generator used for fresh programs."""
import copy
import random
from src.config import DEFAULT_CONFIG
from src.ir import DataType
from src.ir.region import RegionProgram
from src.workflow.generator.region_generator import RegionGenerator, gemm_step_op_program, program_has_gemm

class Mutator:
    def __init__(self, config=DEFAULT_CONFIG, backend='tilelang'):
        if not 0 <= config.dtype_mutate_prob <= 1:
            raise ValueError('dtype_mutate_prob must be between 0 and 1')
        if not 0 <= config.local_mutate_prob <= 1:
            raise ValueError('local_mutate_prob must be between 0 and 1')
        self.config, self.backend = config, backend
        from src.backends import get_backend
        self.backend_impl = get_backend(backend)
        self.feedback = None

    def mutate(self, program):
        from src.ir.extended import ExtendedProgram
        if isinstance(program, ExtendedProgram):
            from src.workflow.generator.extended import mutate_extended
            return mutate_extended(program, self.config, self.backend)
        gen = RegionGenerator(self.config, self.backend, getattr(self, "type_gen", None))
        gen.feedback = self.feedback
        if not isinstance(program, RegionProgram):
            raise TypeError(f'Unsupported program: {type(program).__name__}')
        result = copy.deepcopy(program)
        if result.body.operations[0].kind == 'probe':
            result = self.backend_impl.mutate_probe(result, self.config)
        elif (alternatives := [d for d in dict.fromkeys(map(DataType, self.config.supported_dtypes))
                               if d != result.spec.dtype
                               and (d != DataType.FLOAT32 or not gemm_step_op_program(result))]) and random.random() < self.config.dtype_mutate_prob:
            result.spec.dtype = random.choice(alternatives)
            p = result.spec
            if not self.backend_impl.dtype_parameters_valid(p):
                # A wider dtype may no longer fit. Repair scheduling, retaining
                # the problem dimensions, functions, and selected target dtype.
                # The repair must stay within the validated schedule domain so
                # the mutation reaches the frontend cache instead of dying on a
                # resource error before it.
                result.spec = gen.sample_spec(result.body, result.body.operations[0].kind,
                                              result.functions, dtype=p.dtype, unchecked_ok=False)
                result.spec.M, result.spec.N, result.spec.K = p.M, p.N, p.K
        elif random.random() < self.config.local_mutate_prob and self.mutate_local(result):
            pass
        elif random.random() < .5:
            result = gen.generate(initial=result.body.operations[0].kind)
        elif gemm_step_op_program(result):
            # Never roll a fresh random spec onto a gemm+step body: an fp32
            # draw would recreate the TF32-noise trap. fp16 GEMMs accumulate
            # exactly in fp32, so they stay trap-free.
            result.spec = gen.sample_spec(result.body, result.body.operations[0].kind,
                                          result.functions, dtype=DataType.FLOAT16)
        else:
            result.spec = gen.sample_spec(result.body, result.body.operations[0].kind, result.functions)
        self.backend_impl.validate_program(result)
        gen.bound_scratch(result)
        return result

    def mutate_local(self, program):
        if not program.typed:
            return self._mutate_local_once(program)
        # Type-changing edits must remain valid for every downstream use and
        # both branch yields. Failed proposals never modify the original seed.
        for _ in range(32):
            candidate = copy.deepcopy(program)
            if not self._mutate_local_once(candidate):
                return False
            try:
                self.backend_impl.validate_program(candidate)
            except (ValueError, KeyError):
                continue
            program.body, program.functions, program.execution = candidate.body, candidate.functions, candidate.execution
            return True
        return False

    def _mutate_local_once(self, program):
        """Change one attribute, leaf op, or dominating operand in place.

        No definitions or call sites are removed, so nested scopes, function
        reachability, and the operation budget remain intact.
        """
        from src.ir.region_ops import OPS
        from src.ir.region_types import infer_program
        value_types = infer_program(program)[0] if program.typed else None
        edits = {'operand': [], 'operation': [], 'attribute': [], 'execution': []}
        if program.execution is not None:
            edits['execution'].append((program.execution, 'input_pattern',
                [p for p in ('normal', 'integer', 'alternating') if p != program.execution.input_pattern]))
            from src.ir.layout import MATRIX_LAYOUTS
            fields = ('input_layout_a', 'input_layout_b') if program.body.operations[0].kind == 'gemm' else ('input_layout_a',)
            for field_name in fields:
                current = getattr(program.execution, field_name)
                if self.config.region_layout_prob > 0 or current != 'contiguous':
                    edits['execution'].append((program.execution, field_name,
                        [layout for layout in MATRIX_LAYOUTS if layout != current]))

        def visit(region, inherited, scope):
            visible = list(inherited) + list(region.arguments)
            for op in region.operations:
                for index, operand in enumerate(op.operands):
                    choices = [v for v in visible if v != operand and
                               (value_types is None or value_types[scope, v] == value_types[scope, operand])]
                    if choices:
                        edits['operand'].append((op, index, choices))
                contract = OPS.get(op.kind)
                if contract and not contract.entry and not contract.regions:
                    choices = [kind for kind, other in OPS.items()
                               if not other.entry and not other.regions and other.arity == contract.arity
                               and kind not in (op.kind, 'scale', 'round', 'index_add')
                               and (kind != 'tile_transpose' or program.spec.block_M == program.spec.block_N)]
                    if program.spec.dtype == DataType.FLOAT32 and program_has_gemm(program):
                        # fp32 GEMM + boundary step ops are oracle noise; the
                        # kind swap must not manufacture the trap either.
                        choices = [kind for kind in choices if kind not in ('ceil', 'floor', 'cast')]
                    if choices:
                        edits['operation'].append((op, choices))
                fields = {
                    'round': ('dtype', ['float16', 'float32']),
                    'for': ('trip_count', [0, 1, 2, 3, 4]),
                    'if': ('parity', list(range(op.attrs.get('modulus', 2)))),
                    'scale': ('alpha', [-1.0, 0.0, 0.5, 1.0, 2.0]),
                    'index_add': ('axis', ['row', 'column', 'iteration']),
                }
                if op.kind in fields:
                    attr, values = fields[op.kind]
                    edits['attribute'].append((op, attr, [v for v in values if v != op.attrs[attr]]))
                extra = []
                if op.kind == 'cast':
                    cast_dtypes = ['float16', 'float32']
                    if program.spec.dtype == DataType.FLOAT32 and program_has_gemm(program):
                        # Casting an fp32 GEMM output to fp16 straddles fp16
                        # rounding boundaries under TF32 noise; keep fp32.
                        cast_dtypes = ['float32']
                    extra = [('dtype', cast_dtypes, 'float32')]
                elif op.kind == 'reduce_tile':
                    extra = [('axis', [0, 1], 1), ('reduction', ['sum', 'max', 'min'], 'sum')]
                elif op.kind == 'load_input':
                    extra = [('row_offset', list(range(4)), 0), ('col_offset', list(range(4)), 0)]
                if op.kind == 'for':
                    extra = [('start', [0, 1, 2, 3], 0), ('step', [1, 2, 3], 1)]
                elif op.kind == 'if':
                    extra = [('predicate', ['row', 'column', 'checkerboard', 'iteration'], 'row')]
                elif op.kind == 'index_add':
                    extra = [('scale', [0.25, 0.5, 1.0, 2.0], 1.0)]
                for attr, values, default in extra:
                    filtered = [v for v in values if v != op.attrs.get(attr, default)]
                    if filtered:
                        edits['attribute'].append((op, attr, filtered))
                for child in op.regions:
                    visit(child, visible, scope)
                visible.append(op.result)

        for fn in program.functions:
            visit(fn.body, [], fn.name)
        visit(program.body, [], 'main')
        available = [kind for kind, candidates in edits.items() if candidates]
        if not available:
            return False
        kind = random.choice(available)
        edit = random.choice(edits[kind])
        op = edit[0]
        if kind == 'operand':
            op.operands[edit[1]] = random.choice(edit[2])
        elif kind == 'operation':
            op.kind = random.choice(edit[1])
            op.attrs = {}
        elif kind == 'execution':
            setattr(op, edit[1], random.choice(edit[2]))
        else:
            op.attrs[edit[1]] = random.choice(edit[2])
        return True
