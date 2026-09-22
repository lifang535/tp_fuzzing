"""Queue sequential controls after the repeated mixed experiment finishes."""
import json
from pathlib import Path
import subprocess
import sys
import time

root=Path(__file__).resolve().parents[3]
base=root/'reports/2026.09.17/compilation_profile'
while True:
    state=base/'mixed/summary.json'
    try:
        if json.loads(state.read_text()).get('complete'): break
    except (OSError,ValueError):
        pass
    time.sleep(5)
commands=[]
for family in ('arithmetic','indexed_memory','shape_matmul','control_calls','native'):
    commands.append((family,[sys.executable,'-B',str(root/'tests/profile_compilation.py'),
       '--program',str(base/'corpus'/f'{family}.json'),
       '--output',str(base/(family+'_control')),'--repeats','1','--warm']))
commands.append(('mixed_single',[sys.executable,'-B',str(root/'tests/profile_compilation.py'),
       '--program',str(base/'corpus/mixed.json'),
       '--output',str(base/'mixed_single'),'--single-variant','--repeats','1']))
status=[]
for label,command in commands:
    print('START',label,flush=True)
    result=subprocess.run(command,cwd=root)
    status.append({'case':label,'returncode':result.returncode})
    (base/'controls_status.json').write_text(json.dumps(status,indent=2))
    print('DONE',label,result.returncode,flush=True)
