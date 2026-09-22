from replay import run, ROOT, OUT
import json
for label,path in json.loads((OUT/"selected.json").read_text()).items():
    run(label,ROOT/path.replace(".json",".py"),timeout=70)
