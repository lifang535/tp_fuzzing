"""Rebuild additive phase summaries from retained profiling results."""
from pathlib import Path
from collections import defaultdict
import csv
import json
import statistics

BASE=Path(__file__).resolve().parent
EXCLUDED={'mixed_initial','arithmetic'}  # Method-validation pilots.

def phases(trial):
    groups=defaultdict(float)
    for name, item in trial['stages'].items():
        value=item['exclusive_seconds']
        if name=='tilelang.pass.tl.LayoutInference': group='layout_inference'
        elif name=='tilelang.nvcc': group='nvcc'
        elif name.startswith(('tilelang.', 'triton.')) or name in ('prepare_all','prepare_variant'): group='other_compilation'
        elif name=='runtime.reference': group='reference'
        elif name.startswith('runtime.') and name!='runtime.cuda_context' or name=='harness.total': group='execution_checks'
        elif name.startswith('artifacts.'): group='artifact_io'
        else: group='startup_imports'
        groups[group]+=value
    groups['startup_imports']+=trial['subprocess_seconds']-trial['stages']['worker']['seconds']
    assert abs(sum(groups.values())-trial['subprocess_seconds'])<1e-5
    return dict(groups)

rows=[];experiments={}
for path in sorted(BASE.glob('*/summary.json')):
    case=path.parent.name
    if case in EXCLUDED:continue
    result=json.loads(path.read_text());experiments[case]=result
    # This source passed Triton but its warp schedule was rejected by TileLang.
    # Retain its status and raw failure, but do not compare failed compilation
    # against successful compilation or publish a one-sided timing comparison.
    if case=='native_control':continue
    for trial in result['trials']:
        if not trial.get('success') or not trial.get('span_accounting_valid'):
            continue
        values=phases(trial)
        rows.append(dict(case=case,backend=trial['backend'],cache=trial['cache'],repeat=trial['repeat'],
                         total=trial['subprocess_seconds'],emit=trial['emit_seconds'],**values))
columns=['case','backend','cache','repeat','total','emit','startup_imports','layout_inference','nvcc','other_compilation','reference','execution_checks','artifact_io']
with (BASE/'phases.csv').open('w') as stream:
    writer=csv.DictWriter(stream,fieldnames=columns);writer.writeheader();writer.writerows(rows)

stats=[]
for key in sorted({(r['case'],r['backend'],r['cache']) for r in rows}):
    cases=[r for r in rows if (r['case'],r['backend'],r['cache'])==key]
    stats.append(dict(case=key[0],backend=key[1],cache=key[2],n=len(cases),
                      mean={k:statistics.mean(r.get(k,0) for r in cases) for k in columns[4:]},
                      median_seconds=statistics.median(r['total'] for r in cases),
                      min_seconds=min(r['total'] for r in cases),max_seconds=max(r['total'] for r in cases)))
(BASE/'aggregates.json').write_text(json.dumps(stats,indent=2))
lines=['# 编译耗时实验结果','','同一 IR、独立 DSL 编译缓存、新进程；GPU 驱动缓存保留。详细方法见 [EXPERIMENT.md](EXPERIMENT.md)。',
       '','以下均为墙钟秒数。阶段使用 exclusive 时间汇总，未重复叠加嵌套 pass。用户两个 fuzzer 同时运行，因此绝对时间和 GPU 检查耗时有并发噪声。',
       '','## 主试验：实际慢 mixed 用例','',
       '| 后端 / 缓存 | 次数 | 总时间均值 | 中位数 | 范围 | 布局推导 | NVCC | 其它编译 | 启动/导入 | 参考计算 | 执行和检查 | 产物 I/O |',
       '|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|']
for row in stats:
    if row['case']!='mixed':continue
    m=row['mean'];vals=[m.get(k,0) for k in ('layout_inference','nvcc','other_compilation','startup_imports','reference','execution_checks','artifact_io')]
    lines.append(f"| {row['backend']} / {row['cache']} | {row['n']} | {m['total']:.3f} | {row['median_seconds']:.3f} | {row['min_seconds']:.3f}–{row['max_seconds']:.3f} | "+' | '.join(f'{v:.3f}' for v in vals)+' |')
lines+=['','## 路线对照（每格一次，不能替代大样本吞吐基准）','',
        '| 用例 | 后端 | 缓存 | 总时间 | 布局推导 | NVCC | 其它编译 | 执行和检查 |',
        '|---|---|---|---:|---:|---:|---:|---:|']
for row in stats:
    if row['case']=='mixed':continue
    m=row['mean'];lines.append(f"| {row['case']} | {row['backend']} | {row['cache']} | "+' | '.join(f'{m.get(k,0):.3f}' for k in ('total','layout_inference','nvcc','other_compilation','execution_checks'))+' |')
lines+=['','## TileLang 最慢 pass（mixed 冷编译，按 exclusive 时间汇总）','', '| Pass | 每次完整程序平均耗时 |','|---|---:|']
passes=defaultdict(float);n=0
for trial in experiments.get('mixed',{}).get('trials',[]):
    if trial['backend']=='tilelang' and trial['cache']=='cold' and trial.get('success'):
        n+=1
        for name,item in trial['stages'].items():
            if name.startswith('tilelang.pass.'):passes[name]+=item['exclusive_seconds']
for name,value in sorted(passes.items(),key=lambda x:x[1],reverse=True)[:12]:lines.append(f'| `{name}` | {value/max(1,n):.3f} |')
lines+=['','## Triton 编译阶段（mixed 冷编译）','', '| 阶段 | 每次完整程序平均耗时 |','|---|---:|']
parts=defaultdict(float);n=0
for trial in experiments.get('mixed',{}).get('trials',[]):
    if trial['backend']=='triton' and trial['cache']=='cold' and trial.get('success'):
        n+=1
        for name,item in trial['stages'].items():
            if name.startswith('triton.'):parts[name]+=item['exclusive_seconds']
for name,value in parts.items():lines.append(f'| `{name}`（exclusive） | {value/max(1,n):.3f} |')
lines+=['','## 状态与原始记录','','| 实验 | 完成 | 成功 / 已尝试 |','|---|---|---:|']
for name,result in experiments.items():
    lines.append(f"| [{name}]({name}/summary.json) | {result['complete']} | {sum(bool(r.get('success')) for r in result['trials'])} / {len(result['trials'])} |")
lines+=['','[逐次阶段数据 CSV](phases.csv) · [聚合 JSON](aggregates.json) · [实际 campaign 快照](campaign_snapshot.json) · [生成开销](generation.json)','']
(BASE/'RESULTS.md').write_text('\n'.join(lines))
print('Wrote',len(rows),'successful trial rows')
