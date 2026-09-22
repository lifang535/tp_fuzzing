"""Typed dataflow and mutable-memory semantics, including emitted instructions."""
import ast
import copy
import json
import random
import types
import unittest
from dataclasses import asdict

import torch
from src.config import Config
from src.ir.region import Region, RegionProgram, RegionExecution, Function, Operation as Op, walk
from src.ir.region_types import ValueType, infer_program, scratch_bytes
from src.workflow.emitter.typed_region_runtime import _typed_region_reference
from src.backends.common.typed_emitter import TypedLowering, _prepare_typed_launch
from src.workflow.emitter.region_checks import _region_input_storage
from src.workflow.feedback import program_features, key
from src.workflow.fuzzer.fuzzer import TileSmith
from src.workflow.generator.region_generator import RegionGenerator
from src.workflow.mutator import Mutator
from src.workflow.oracle import Oracle, BugReport, BugType
from test_regions import nested_program


def typed_program(dtype='float32', initial='load'):
    p = nested_program(dtype, initial)
    p.typed = True
    p.execution = RegionExecution(input_pattern='integer', input_seed_count=1, repeat_count=2,
                                  input_layout_a='offset', input_layout_b='transposed' if initial == 'gemm' else 'contiguous')
    p.input_scale = 0.125
    loop = Region(['carry'], [
        Op('load_tile', 'before', ['buffer']),
        Op('scale', 'half', ['before'], {'alpha': 0.5}),
        Op('write_tile', 'alias', ['buffer', 'half']),
        Op('load_input', 'reloaded', attrs={'source':'A', 'dtype':'float32', 'row_offset':1, 'col_offset':1}),
        Op('add', 'next', ['carry', 'reloaded'])], 'next')
    fn = Function('fn_0', Region(['small', 'full'], [
        Op('store_tile', 'buffer', ['small']),
        Op('for', 'looped', ['full'], {'trip_count':2, 'start':1, 'step':2}, [loop]),
        Op('load_tile', 'read', ['buffer']),
        Op('to_tile', 'wide', ['read']),
        Op('add', 'result', ['wide', 'looped'])], 'result'),
        [ValueType('float16', 'row').to_dict(), ValueType().to_dict()])
    yes = Region(['yes_arg'], [
        Op('call', 'first_call', ['row_half', 'yes_arg'], {'callee':'fn_0'}),
        Op('write_tile', 'written', ['outer_buffer', 'first_call'])], 'first_call')
    no = Region(['no_arg'], [
        Op('neg', 'negative', ['no_arg']),
        Op('write_tile', 'other_write', ['outer_buffer', 'negative'])], 'negative')
    p.functions = [fn]
    p.body = Region([], [Op(initial, 'entry'),
        Op('reduce_tile', 'row', ['entry'], {'axis':1, 'reduction':'sum'}),
        Op('cast', 'row_half', ['row'], {'dtype':'float16'}),
        Op('store_tile', 'outer_buffer', ['entry']),
        Op('if', 'branch', ['entry'], {'predicate':'checkerboard', 'modulus':2, 'parity':0}, [yes,no]),
        Op('load_tile', 'after_branch', ['outer_buffer']),
        Op('call', 'answer', ['row_half', 'after_branch'], {'callee':'fn_0'})], 'answer')
    p.validate()
    return p


def reference(p, a, b):
    return _typed_region_reference(a, b, asdict(p.body), p.spec.block_M, p.spec.block_N,
                                  p.spec.dtype.value, [fn.to_dict() for fn in p.functions])


def run_emitted_triton(p, a_storage, b_storage):
    """Run emitted Python statements with a CPU implementation of tl primitives."""
    class Pointer:
        def __init__(self, value, offset=0): self.value, self.offset = value.reshape(-1), offset
        def __add__(self, offset): return Pointer(self.value, self.offset + offset)
    def load(ptr, mask=None, other=0, **kwargs):
        if mask is None: return ptr.value[ptr.offset].clone()
        return torch.where(mask, ptr.value[torch.where(mask, ptr.offset, 0)], other)
    def store(ptr, value, mask=None):
        offsets, values = torch.broadcast_tensors(ptr.offset, value)
        if mask is None: ptr.value[offsets] = values.to(ptr.value.dtype)
        else: ptr.value[offsets[mask]] = values[mask].to(ptr.value.dtype)
    tile = [0, 0]
    tl = types.SimpleNamespace(program_id=lambda axis: tile[axis], arange=torch.arange,
        load=load, store=store, float32=torch.float32, float16=torch.float16,
        full=lambda shape, value, dtype: torch.full(shape, value, dtype=dtype),
        range=lambda start, stop, **kw: range(start, stop), debug_barrier=lambda: None,
        dot=lambda a,b,c: a.float() @ b.float() + c, broadcast_to=torch.broadcast_to,
        sum=lambda x,axis,keep_dims=False: x.sum(axis,keepdim=keep_dims),
        max=lambda x,axis,keep_dims=False: x.amax(axis,keepdim=keep_dims),
        min=lambda x,axis,keep_dims=False: x.amin(axis,keepdim=keep_dims),
        trans=lambda x: x.T, abs=torch.abs, sqrt=torch.sqrt, exp=torch.exp, where=torch.where,
        maximum=lambda x,y: torch.maximum(x,torch.as_tensor(y)),
        minimum=lambda x,y: torch.minimum(x,torch.as_tensor(y)),
        tanh=torch.tanh, erf=torch.erf, log=torch.log, log2=torch.log2, exp2=torch.exp2,
        rsqrt=torch.rsqrt, sin=torch.sin, cos=torch.cos, floor=torch.floor, ceil=torch.ceil,
        extra=types.SimpleNamespace(
            libdevice=types.SimpleNamespace(tanh=torch.tanh),
            cuda=types.SimpleNamespace(libdevice=types.SimpleNamespace(tanh=torch.tanh))))
    lower = TypedLowering(p, 'triton', 'kernel')
    namespace = {'triton': types.SimpleNamespace(jit=lambda fn: fn), 'tl':tl}
    exec(lower.emit(), namespace)
    tm, tn = (p.spec.M+p.spec.block_M-1)//p.spec.block_M, (p.spec.N+p.spec.block_N-1)//p.spec.block_N
    buffers = [torch.full((tm*tn, t.dimensions(p.spec)[0]*t.dimensions(p.spec)[1]+32), 23,
                          dtype=getattr(torch,t.dtype)) for t in lower.slots.values()]
    output = torch.full((p.spec.M,p.spec.N), float('nan'), dtype=getattr(torch,p.spec.dtype.value))
    for by in range(tm):
        for bx in range(tn):
            tile[:] = [by,bx]
            namespace['kernel'](*(Pointer(x) for x in [a_storage,b_storage]+buffers+[output]))
    for buffer in buffers:
        assert torch.all(buffer[:,:16] == 23) and torch.all(buffer[:,-16:] == 23)
    return output


class TypedRegionTests(unittest.TestCase):
    def setUp(self):
        state = random.getstate()
        self.addCleanup(random.setstate, state)
        random.seed(203)

    def test_branch_side_effects_and_zero_loop(self):
        p = nested_program()
        p.typed, p.execution = True, RegionExecution()
        yes = Region(['y'], [Op('scale','twice',['y'],{'alpha':2}),
                            Op('write_tile','a',['buf','twice'])], 'twice')
        no = Region(['n'], [Op('scale','triple',['n'],{'alpha':3}),
                           Op('write_tile','b',['buf','triple'])], 'triple')
        skipped = Region(['s'], [Op('write_tile','never',['buf','s'])], 's')
        p.body = Region([], [Op('load','x'), Op('store_tile','buf',['x']),
            Op('if','chosen',['x'],{'parity':0},[yes,no]),
            Op('for','zero',['x'],{'trip_count':0},[skipped]),
            Op('load_tile','out',['buf'])], 'out')
        p.validate()
        a = torch.arange(33*35).reshape(33,35).float()/32
        expected = a * torch.where(torch.arange(33)//32%2 == 0,2,3)[:,None]
        torch.testing.assert_close(reference(p,a,torch.empty(33,35)),expected,rtol=0,atol=0)

    def test_compact_types_rounding_and_buffer_aliases(self):
        p = nested_program(); p.typed, p.execution = True, RegionExecution()
        p.body = Region([], [Op('load','x'), Op('cast','h',['x'],{'dtype':'float16'}),
            Op('reduce_tile','col',['h'],{'axis':0,'reduction':'sum'}),
            Op('reduce_tile','scalar',['col'],{'axis':1,'reduction':'sum'}),
            Op('store_tile','buf',['scalar']), Op('neg','negative',['scalar']),
            Op('write_tile','alias',['buf','negative']), Op('load_tile','read',['buf']),
            Op('to_tile','out',['read'])], 'out')
        p.spec.M,p.spec.N=2,3
        p.validate()
        a=torch.tensor([[0.0003,1.0003,3.0003],[2.0003,-4.0003,0.0007]])
        expected=(-a.half().float().sum()).expand_as(a)
        torch.testing.assert_close(reference(p,a,torch.empty(33,3)),expected,rtol=0,atol=0)
        types_,aliases,_=infer_program(p)
        self.assertEqual(types_['main','col'],ValueType('float32','column'))
        self.assertEqual(types_['main','scalar'],ValueType('float32','scalar'))
        self.assertEqual(aliases['main','buf'],aliases['main','alias'])

    def test_rejects_bad_types_scopes_and_buffer_escape(self):
        changes = [
            lambda p: p.body.operations[2].attrs.update(dtype='int32'),
            lambda p: p.functions[0].body.operations[1].regions[0].operations[2].operands.__setitem__(1,'full'),
            lambda p: p.body.operations[-1].operands.__setitem__(0,'entry'),
            lambda p: setattr(p.body,'yield_value','outer_buffer'),
            lambda p: p.body.operations[4].regions[0].operations[1].operands.__setitem__(0,'row_half'),
            lambda p: p.body.operations[-2].operands.__setitem__(0,'written'),
        ]
        for edit in changes:
            p=typed_program(); edit(p)
            with self.assertRaises(ValueError): p.validate()

    def test_emitted_memory_calls_and_typed_arithmetic_match_reference(self):
        for dtype in ('float16','float32'):
            for initial in ('load','gemm'):
                p=typed_program(dtype,initial)
                for layout in ('contiguous','transposed','strided','offset','broadcast_rows','broadcast_cols'):
                    p.execution.input_layout_a=layout
                    a_storage,a=_region_input_storage((p.spec.M,p.spec.K if initial=='gemm' else p.spec.N),
                        getattr(torch,dtype),'integer',0.125,layout,'cpu')
                    b_storage,b=_region_input_storage((p.spec.K,p.spec.N),getattr(torch,dtype),'integer',0.125,p.execution.input_layout_b,'cpu')
                    actual=run_emitted_triton(p,a_storage,b_storage)
                    torch.testing.assert_close(actual,reference(p,a,b),rtol=0,atol=0)
                for backend in ('triton','tilelang'):
                    code=Oracle(Config(),backend)._emit_code(p)
                    ast.parse(code)
                    self.assertNotIn('from src.',code)
                    self.assertIn('scratch_',code)

    def test_random_generation_mutation_roundtrip_and_execution(self):
        config=Config(coverage_probe_prob=0,region_typed_prob=0.7,dim_range=(1,40),
                      tile_size_choices=[32],block_k_choices=[32],region_gemm_prob=0,
                      region_int8_prob=0)
        kinds=set(); shapes=set(); dtypes=set(); nested_memory=False
        gen=RegionGenerator(config)
        for i in range(70):
            p=gen.generate()
            if i%2: p=Mutator(config).mutate(p)
            p.validate()
            raw=json.loads(json.dumps(p.to_dict()))
            self.assertEqual(RegionProgram.from_dict(raw).to_dict(),raw)
            self.assertEqual(TileSmith._make_sig(p),TileSmith._make_sig_from_dict(raw))
            for op in p.all_operations():
                kinds.add(op.kind)
                nested_memory |= any(x.kind in ('store_tile','load_tile','write_tile','load_input') for child in op.regions for x in walk(child))
            for t in infer_program(p)[0].values(): shapes.add(t.shape); dtypes.add(t.dtype)
            a_storage,a=_region_input_storage((p.spec.M,p.spec.N),getattr(torch,p.spec.dtype.value),'integer',0.125,p.execution.input_layout_a,'cpu')
            b_storage,b=_region_input_storage((p.spec.K,p.spec.N),getattr(torch,p.spec.dtype.value),'integer',0.125,'contiguous','cpu')
            torch.testing.assert_close(run_emitted_triton(p,a_storage,b_storage),reference(p,a,b),rtol=1e-5,atol=1e-5,equal_nan=True)
            ast.parse(Oracle(config,'tilelang')._emit_code(p))
        self.assertTrue({'cast','reduce_tile','store_tile','load_tile','write_tile','load_input'} <= kinds)
        self.assertTrue({'tile','row','column','scalar'} <= shapes)
        self.assertEqual(dtypes,{'float16','float32'})
        self.assertTrue(nested_memory)

    def test_scratch_guards_and_initialization(self):
        a=torch.zeros(1)
        def bad(A,B,S,C): S[0]=0
        launch=_prepare_typed_launch(bad,(a,a),[(2,8,'float16')])
        with self.assertRaisesRegex(RuntimeError,'scratch canary'): launch(a)
        def good(A,B,S,C):
            self.assertTrue(torch.isnan(S.reshape(2,40)[:,16:-16]).all())
            S.reshape(2,40)[:,16:-16]=1
        launch=_prepare_typed_launch(good,(a,a),[(2,8,'float16')])
        launch(a); launch(a)

    def test_feedback_and_scratch_budget(self):
        p=typed_program()
        self.assertIn(key('value_type','cast','tensor','float16','row'),program_features(p))
        p.spec.M=p.spec.N=4096
        gen=RegionGenerator(Config(region_scratch_max_bytes=1<<20))
        gen.bound_scratch(p)
        self.assertLessEqual(scratch_bytes(p),1<<20)
        p.validate()

    def test_memory_failures_keep_distinct_classifications(self):
        for message, expected in [('scratch canary modified', 'scratch_out_of_bounds'),
                                  ('output canary modified', 'output_out_of_bounds'),
                                  ('input storage modified', 'input_corruption')]:
            report = BugReport(BugType.WRONG_RESULT, 'WRONG RESULT: ' + message)
            report.classify_root_cause()
            self.assertEqual(report.root_cause, expected)

    def test_old_generation_and_signatures_remain_available(self):
        p=RegionGenerator(Config(coverage_probe_prob=0,region_typed_prob=0)).generate()
        self.assertFalse(p.typed)
        self.assertEqual(p.to_dict()['version'],3)
        self.assertNotIn('argument_types',p.to_dict()['functions'][0])


if __name__=='__main__': unittest.main()
