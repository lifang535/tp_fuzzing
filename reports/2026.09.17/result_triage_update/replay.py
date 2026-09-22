"""Bounded replay of saved reproducers; never modifies campaign files."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent

def run(label, source, timeout=90, cache=None):
    env = dict(os.environ, PYTHONFAULTHANDLER='1', OMP_NUM_THREADS='2')
    cache = Path(cache or tempfile.mkdtemp(prefix='triage_cache_'))
    env.update(TILELANG_CACHE_DIR=str(cache/'tilelang'), TRITON_CACHE_DIR=str(cache/'triton'))
    log = OUT / (label + '.log')
    start = time.monotonic()
    with log.open('w') as f:
        p = subprocess.Popen([sys.executable, '-u', str(source)], cwd=ROOT, env=env,
                             stdout=f, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            rc = p.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(p.pid, signal.SIGKILL)
            p.wait()
            rc = 'timeout'
    record = dict(label=label, source=str(source), returncode=rc,
                  seconds=round(time.monotonic()-start, 3), log=str(log), cache=str(cache))
    with (OUT/'replays.jsonl').open('a') as f:
        f.write(json.dumps(record)+'\n')
    print(json.dumps(record), flush=True)
    print(log.read_text()[-1100:], flush=True)
    return record

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('ids', nargs='+')
    ap.add_argument('--timeout', type=int, default=90)
    args = ap.parse_args()
    for ident in args.ids:
        source = next(ROOT.glob('results/*/failed/*/*'+ident+'.py'))
        run(ident, source, args.timeout)
