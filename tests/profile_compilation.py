"""Profile the same saved IR on both backends, without changing campaign code.

Each cold trial has private DSL compiler caches and a fresh subprocess. An
optional warm trial reuses that trial's caches in another fresh subprocess.
Compiler instrumentation is installed only in the worker, never in site-packages.
Inclusive spans are nested; sum exclusive_seconds to avoid double counting.
"""
import argparse
from contextlib import contextmanager
import functools
import hashlib
import importlib
import inspect
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
import tempfile
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class Recorder:
    def __init__(self, path):
        self.started = time.perf_counter()
        self.stream = path.open('w', buffering=1)
        self.stack = []
        self.serial = 0
        self.events = []

    def begin(self, name):
        self.serial += 1
        self.stack.append(dict(id=self.serial, name=name,
                               parent=self.stack[-1]['id'] if self.stack else None,
                               start=time.perf_counter() - self.started, children=0.0))

    def end(self):
        end = time.perf_counter() - self.started
        event = self.stack.pop()
        event['seconds'] = end - event['start']
        event['exclusive_seconds'] = event['seconds'] - event.pop('children')
        if self.stack:
            self.stack[-1]['children'] += event['seconds']
        self.events.append(event)
        self.stream.write(json.dumps(event) + '\n')

    @contextmanager
    def span(self, name):
        self.begin(name)
        try:
            yield
        finally:
            self.end()

    def wrap(self, function, name, synchronize=False):
        @functools.wraps(function)
        def measured(*args, **kwargs):
            with self.span(name):
                result = function(*args, **kwargs)
                if synchronize:
                    import torch
                    torch.cuda.synchronize()
                return result
        return measured

    def patch(self, owner, attribute, name):
        descriptor = inspect.getattr_static(owner, attribute)
        wrapped = self.wrap(getattr(owner, attribute), name)
        setattr(owner, attribute, staticmethod(wrapped) if isinstance(descriptor, staticmethod) else wrapped)


def instrument_tilelang(recorder):
    import tilelang
    from tilelang import tvm
    from tilelang.jit import JITImpl
    recorder.patch(JITImpl, 'get_tir', 'tilelang.frontend')
    lower = importlib.import_module('tilelang.engine.lower')
    for name in ('lower_to_host_device_ir', 'device_codegen', 'host_codegen'):
        recorder.patch(lower, name, 'tilelang.' + name)
    from tilelang.contrib import nvcc
    recorder.patch(nvcc, 'compile_cuda', 'tilelang.nvcc')
    from tilelang.jit.kernel import JITKernel
    recorder.patch(JITKernel, '_compile_and_create_adapter', 'tilelang.compile_total')
    from tilelang.jit.adapter.tvm_ffi import TVMFFIKernelAdapter
    recorder.patch(TVMFFIKernelAdapter, '_convert_torch_func', 'tilelang.adapter')

    @tvm.ir.instrument.pass_instrument
    class PassTimer:
        def run_before_pass(self, mod, info):
            recorder.begin('tilelang.pass.' + info.name)

        def run_after_pass(self, mod, info):
            recorder.end()

    original = tvm.transform.PassContext.__init__

    def context_init(self, opt_level=2, required_pass=None, disabled_pass=None,
                     instruments=None, config=None):
        return original(self, opt_level, required_pass, disabled_pass,
                        list(instruments or []) + [PassTimer()], config)

    tvm.transform.PassContext.__init__ = context_init


def instrument_triton(recorder):
    from triton.compiler import ASTSource
    from triton.backends import backends
    # Triton 3.0 loads its backend under an alias (nvi); importing the file by
    # its package name creates a different class that the compiler never uses.
    CUDABackend = backends['nvidia'].compiler
    recorder.patch(ASTSource, 'make_ir', 'triton.frontend')
    for stage in ('ttir', 'ttgir', 'llir', 'ptx', 'cubin'):
        recorder.patch(CUDABackend, 'make_' + stage, 'triton.' + stage)
    import triton.compiler
    import triton.compiler.compiler
    # Extended uses triton.compiler.compile; native JIT uses the package export.
    original = triton.compiler.compile
    wrapped = recorder.wrap(original, 'triton.compile_total')
    triton.compiler.compile = wrapped
    triton.compiler.compiler.compile = wrapped
    triton.compile = wrapped


def instrument_harness(namespace, recorder):
    # runpy's returned mapping can differ from a function's globals mapping.
    scopes = [namespace]
    scopes.extend(fn.__globals__ for fn in namespace.values()
                  if inspect.isfunction(fn) and fn.__code__.co_filename == str(Path(namespace['__file__'])))
    scopes = list({id(scope): scope for scope in scopes}.values())
    labels = {
        'extended_reference': 'runtime.reference', '_region_reference': 'runtime.reference',
        '_typed_region_reference': 'runtime.reference', 'extended_inputs': 'runtime.inputs',
        '_region_input_storage': 'runtime.inputs', '_probe_input': 'runtime.inputs',
        'extended_check': 'runtime.compare', '_finite_compare': 'runtime.compare',
        '_region_equal': 'runtime.compare', '_probe_exact': 'runtime.compare',
        'record_extended_compilation': 'artifacts.write', 'extended_stage': 'artifacts.progress',
    }
    for name, label in labels.items():
        if name in namespace:
            wrapped = recorder.wrap(namespace[name], label)
            for scope in scopes:
                scope[name] = wrapped
    if 'prepare_extended' in namespace:
        original = namespace['prepare_extended']

        def prepare(*args, **kwargs):
            with recorder.span('prepare_all'):
                variants = original(*args, **kwargs)
                return [(label, watched, recorder.wrap(launch, 'runtime.kernel', synchronize=True))
                        for label, watched, launch in variants]

        for scope in scopes:
            scope['prepare_extended'] = prepare
    else:
        for name in list(namespace):
            if name.startswith('prepare_') and inspect.isfunction(namespace[name]):
                def prepare(*args, _original=namespace[name], **kwargs):
                    with recorder.span('prepare_variant'):
                        launch = _original(*args, **kwargs)
                    return recorder.wrap(launch, 'runtime.kernel', synchronize=True)
                for scope in scopes:
                    scope[name] = prepare


def worker(args):
    directory = args.output.resolve()
    recorder = Recorder(directory / 'events.jsonl')
    info = {'backend': args.backend, 'success': False}
    try:
        with recorder.span('worker'):
            with recorder.span('import.torch'):
                import torch
            with recorder.span('import.backend'):
                backend = importlib.import_module(args.backend)
            info.update(torch=torch.__version__, backend_version=backend.__version__,
                        python=sys.version, cuda_available=torch.cuda.is_available())
            if not info['cuda_available']:
                raise RuntimeError('GPU execution is required; run outside a device-isolated sandbox')
            info.update(gpu=torch.cuda.get_device_name(0), capability=torch.cuda.get_device_capability(0))
            # Context initialization belongs to each real fuzzer subprocess too.
            with recorder.span('runtime.cuda_context'):
                torch.cuda.init()
                torch.cuda.synchronize()
            with recorder.span('instrumentation.setup'):
                (instrument_tilelang if args.backend == 'tilelang' else instrument_triton)(recorder)
            with recorder.span('module.load'):
                namespace = runpy.run_path(str(directory / 'repro.py'), run_name='profiled_case')
            instrument_harness(namespace, recorder)
            with recorder.span('harness.total'):
                if 'run_extended' in namespace:
                    namespace['run_extended'](namespace['PROGRAM'], namespace['prepare_extended'], 0, 3)
                    namespace['extended_stage']('complete', 'execute')
                else:
                    tests = [v for k, v in namespace.items() if k.startswith('test_') and callable(v)]
                    if len(tests) != 1:
                        raise ValueError('Expected exactly one standalone test entry')
                    tests[0]()
            info['success'] = True
    except Exception:
        info['error'] = traceback.format_exc()
        traceback.print_exc()
    finally:
        # Failed TVM passes may omit their after callback. Retain those spans.
        while recorder.stack:
            recorder.end()
        totals = {}
        for event in recorder.events:
            bucket = totals.setdefault(event['name'], {'calls': 0, 'seconds': 0., 'exclusive_seconds': 0.})
            bucket['calls'] += 1
            for key in ('seconds', 'exclusive_seconds'):
                bucket[key] += event[key]
        info['stages'] = totals
        info['span_accounting_valid'] = (
            not recorder.stack
            and all(e['exclusive_seconds'] >= -1e-9 for e in recorder.events)
            and abs(sum(e['exclusive_seconds'] for e in recorder.events)
                    - sum(e['seconds'] for e in recorder.events if e['parent'] is None)) < 1e-6
        )
        (directory / 'timings.json').write_text(json.dumps(info, indent=2))
        recorder.stream.close()
    return 0 if info['success'] else 1


def campaign(args):
    from src.backends import get_backend
    from src.config import Config
    from src.ir.serialization import program_from_dict, program_to_dict
    from src.workflow.oracle.process import run_isolated
    directory = args.output.resolve()
    directory.mkdir(parents=True, exist_ok=False)
    data = json.loads(args.program.read_text())
    program = program_from_dict(data)
    if args.single_variant:
        if hasattr(program, 'configuration_pair'):
            program.configuration_pair = program.observation_pair = False
        elif program.execution is not None:
            program.execution.schedule_pair = False
    canonical = json.dumps(program_to_dict(program), sort_keys=True)
    (directory / 'program.json').write_text(canonical)
    summary = dict(program_source=str(args.program.resolve()), program_sha256=hashlib.sha256(canonical.encode()).hexdigest(),
                   complete=False, trials=[], instrumented=True,
                   cache_policy='Private DSL caches per backend/trial; warm reuses them; CUDA driver cache is inherited',
                   timing_note='Wall time; stages nest. Use exclusive_seconds for additive totals. Kernel wrappers synchronize.')
    config = Config(region_repeat_count=3)
    backends = ('triton', 'tilelang') if args.backend == 'both' else (args.backend,)
    with tempfile.TemporaryDirectory(prefix='tilesmith_profile_cache_') as caches:
        for repeat in range(args.repeats):
            # Alternate backend order to reduce systematic order bias.
            for backend in backends if repeat % 2 == 0 else tuple(reversed(backends)):
                cache = Path(caches) / f'{backend}_{repeat}'
                for mode in ('cold', 'warm') if args.warm else ('cold',):
                    label = f'{backend}_{repeat}_{mode}'
                    case = directory / label
                    case.mkdir()
                    started = time.perf_counter()
                    source = get_backend(backend).make_emitter(config).emit(program)
                    emit_seconds = time.perf_counter() - started
                    (case / 'repro.py').write_text(source)
                    env = dict(os.environ, TILESMITH_ARTIFACT_DIR=str(case),
                               TILELANG_CACHE_DIR=str(cache / 'tilelang'),
                               TRITON_CACHE_DIR=str(cache / 'triton'))
                    started = time.perf_counter()
                    command = [sys.executable, '-B', __file__, '--worker', '--backend', backend, '--output', str(case)]
                    try:
                        result = run_isolated(command, timeout=args.timeout, env=env)
                        output, returncode = result.stdout + result.stderr, result.returncode
                    except subprocess.TimeoutExpired as error:
                        def decode(value):
                            return value.decode(errors='replace') if isinstance(value, bytes) else value or ''
                        output = decode(error.stdout) + decode(error.stderr) + '\nTIMEOUT\n'
                        returncode = None
                    elapsed = time.perf_counter() - started
                    (case / 'run.log').write_text(output)
                    record = dict(label=label, backend=backend, repeat=repeat, cache=mode,
                                  subprocess_seconds=elapsed, emit_seconds=emit_seconds, returncode=returncode,
                                  source_sha256=hashlib.sha256(source.encode()).hexdigest())
                    if (case / 'timings.json').exists():
                        record.update(json.loads((case / 'timings.json').read_text()))
                    summary['trials'].append(record)
                    (directory / 'summary.json').write_text(json.dumps(summary, indent=2))
                    print(f'{label}: {elapsed:.3f}s, success={record.get("success", False)}', flush=True)
                    if returncode:
                        print(output[-2000:], flush=True)
    summary['complete'] = True
    (directory / 'summary.json').write_text(json.dumps(summary, indent=2))
    return int(any(not trial.get('success') for trial in summary['trials']))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--program', type=Path, help='Saved Region/Extended IR or bug report')
    parser.add_argument('--output', type=Path, required=True, help='New experiment directory')
    parser.add_argument('--backend', choices=('both', 'tilelang', 'triton'), default='both')
    parser.add_argument('--repeats', type=int, default=1)
    parser.add_argument('--warm', action='store_true')
    parser.add_argument('--single-variant', action='store_true')
    parser.add_argument('--timeout', type=int, default=360)
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.repeats < 1 or args.timeout < 1 or (not args.worker and args.program is None):
        parser.error('A program, positive repeats and a positive timeout are required')
    return worker(args) if args.worker else campaign(args)


if __name__ == '__main__':
    raise SystemExit(main())
