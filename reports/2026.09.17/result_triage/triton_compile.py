"""Compile the saved Triton kernels for sm_89 without allocating their inputs."""
import ast
import json
from pathlib import Path
import sys
from replay import ROOT, OUT, run

snapshot = json.loads((OUT/'snapshot.json').read_text())
done = {json.loads(line)['label'] for line in (OUT/'replays.jsonl').read_text().splitlines()}
for item in snapshot['files']:
    path = (ROOT/item['path']).with_suffix('.py')
    if 'triton' not in str(path) or path.parent.name != 'segfault':
        continue
    ident = path.stem[-16:]
    if ident+'_compile' in done:
        continue
    meta = json.loads(path.with_suffix('.json').read_text())
    dtype = {'float16': '*fp16', 'float32': '*fp32'}[meta['dtype']]
    tree = ast.parse(path.read_text())
    specs = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name.startswith('prepare_'):
            kernel_call = next(x for x in ast.walk(node) if isinstance(x, ast.Call) and isinstance(x.func, ast.Subscript))
            layout_call = next(x for x in ast.walk(node) if isinstance(x, ast.Call) and isinstance(x.func, ast.Name) and x.func.id == '_prepare_typed_launch')
            layouts = ast.literal_eval(layout_call.args[2])
            options = {kw.arg: ast.literal_eval(kw.value) for kw in kernel_call.keywords}
            specs.append((kernel_call.func.value.id, layouts, options))
    code = f'''import runpy
import triton
from triton.compiler import ASTSource
from triton.backends.compiler import GPUTarget
ns = runpy.run_path({str(path)!r})
for name, layouts, options in {specs!r}:
    fn = ns[name]
    signature = {{arg: {dtype!r} for arg in fn.arg_names}}
    for arg, layout in zip([x for x in fn.arg_names if x not in ('A', 'B', 'C')], layouts):
        signature[arg] = {{'float16': '*fp16', 'float32': '*fp32'}}[layout[2]]
    print('COMPILING', name, signature, options, flush=True)
    result = triton.compile(ASTSource(fn, signature, {{}}), target=GPUTarget('cuda', 89, 32), options=options)
    print('COMPILED', name, flush=True)
print('ALL COMPILED')
'''
    driver = OUT/(ident+'_compile.py')
    driver.write_text(code)
    run(ident+'_compile', driver, timeout=60)
