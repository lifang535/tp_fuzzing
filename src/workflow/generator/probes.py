"""Bounded, directed cases for layouts, tail masks, and fused index reductions."""
import copy
import random

from src.ir import TileKernel, TileProgram, ComputeKind, DataType, LoopKind

KINDS = (ComputeKind.COPY, ComputeKind.REDUCE_SUM, ComputeKind.REDUCE_MAX,
         ComputeKind.REDUCE_MIN, ComputeKind.SOFTMAX, ComputeKind.ARGMAX,
         ComputeKind.GEMM_ARGMAX)
LAYOUTS = ('contiguous', 'transposed', 'strided', 'offset')
BOUNDARIES = (1, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129)


def patterns(kind):
    # NaN max/argmax policies differ between DSLs. Test exceptional bit patterns
    # with exact copies; keep index reductions mathematically unambiguous.
    if kind == ComputeKind.COPY:
        return ('normal', 'special', 'subnormal', 'indexed')
    if kind == ComputeKind.GEMM_ARGMAX:
        return ('integer', 'ties')  # Exact dot products avoid unstable near ties.
    if kind == ComputeKind.ARGMAX:
        return ('integer', 'ties', 'negative')
    return ('normal', 'integer', 'negative', 'zeros')


def repair_probe(k):
    k.block_M = 32
    k.block_N = max(32, 1 << (k.N - 1).bit_length())
    k.block_K = 32
    if k.loop_kind == LoopKind.SERIAL:
        k.num_stages = 1
    if k.compute_kind == ComputeKind.GEMM_ARGMAX:
        from src.constraints import tilelang_check_shared_memory, triton_check_shared_memory
        # The paired schedules must both stay within the shared-memory budget.
        candidates = ((bm, bk, stages) for bm in (32, 16) for bk in (32, 16)
                      for stages in range(k.num_stages, 0, -1))
        for bm, bk, stages in candidates:
            if all(check(bm, k.block_N, bk, k.dtype, stages)
                   for check in (tilelang_check_shared_memory, triton_check_shared_memory)):
                k.block_M, k.block_K, k.num_stages = bm, bk, stages
                break
        else:
            raise ValueError('No GEMM probe tile fits the shared-memory budget')
    if k.input_pattern not in patterns(k.compute_kind):
        k.input_pattern = random.choice(patterns(k.compute_kind))
    return k


def generate_probe(config):
    dims = (16, 32, 64, 128) if config.easy_shape else BOUNDARIES
    kind = random.choice(KINDS)
    k = TileKernel('kernel_0', compute_kind=kind,
        M=random.choice(dims), N=random.choice(dims), K=random.choice(dims),
        dtype=DataType(random.choice(config.supported_dtypes)),
        threads=random.choice((128, 256)), num_stages=random.choice((1, 2, 3)),
        loop_kind=random.choice(list(LoopKind)), coverage_probe=True,
        input_layout=random.choice(LAYOUTS), input_pattern=random.choice(patterns(kind)),
        repeat_count=config.probe_repeat_count, schedule_pair=config.probe_schedule_pair)
    return TileProgram([repair_probe(k)])


def mutate_probe(program):
    result = copy.deepcopy(program)
    k = result.kernels[0]
    field = random.choice(('M', 'N', 'K', 'input_layout', 'input_pattern',
                           'compute_kind', 'threads', 'num_stages', 'loop_kind'))
    choices = {'M': BOUNDARIES, 'N': BOUNDARIES, 'K': BOUNDARIES,
               'input_layout': LAYOUTS, 'input_pattern': patterns(k.compute_kind),
               'compute_kind': KINDS, 'threads': (128, 256), 'num_stages': (1, 2, 3),
               'loop_kind': tuple(LoopKind)}
    setattr(k, field, random.choice(choices[field]))
    repair_probe(k)
    return result
