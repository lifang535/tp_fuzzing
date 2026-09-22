"""Emit an auditable per-record triage, with uncertainty kept explicit."""
import csv
import json
from collections import Counter
from pathlib import Path

OUT = Path(__file__).resolve().parent
snapshot = json.loads((OUT/'snapshot.json').read_text())
rows=[]
for item in snapshot['files']:
    d=item['report']
    p=Path(item['path'])
    ident=p.stem[-16:]
    rp=d['params'].get('region_program',{})
    spec=rp.get('spec',{})
    cat=d['root_cause']
    fp16_k8=spec.get('compute_kind')=='gemm' and spec.get('dtype')=='float16' and spec.get('block_K')==8
    status='待确认'
    reason='需要逐条缩减及独立语义验证'
    if cat=='dtype_mismatch':
        status='已确认缺陷机制，同类记录'
        reason='源码 dtype 与输入匹配；frontend cache 未纳入全局 dtype；三步缓存对照已确认机制，未逐条重放489条'
    elif cat=='ptx_async_boundary':
        status='已确认缺陷机制，同类记录'
        reason='49条均为fp16、N=1或K=1，普通T.copy自动生成非法2字节cp.async；代表6ffb52d51060569a已复现'
    elif cat in ('shared_memory_overflow','gpu_oom','assertion_failure','other'):
        status='非有效目标编译器bug'
        reason={
            'shared_memory_overflow':'请求动态shared memory超过sm_89每block 99328字节限制；生成/资源预算不足',
            'gpu_oom':'56条全部在PyTorch参考解释器中OOM；属于oracle内存预算',
            'assertion_failure':'日志实际为CUDA out of memory，DSA提示被关键词分类器误认成assertion',
            'other':'10条均为probe检查器对末维stride=0的单列张量view(uint8)异常；CPU可复现'
        }[cat]
    elif ident in ('2432022d37c01733','c6f65c66acaafb6f','8407027ddaa21060'):
        status='非有效目标编译器bug'
        reason={
            '2432022d37c01733':'约3893128的输出差0.5（约2 ULP），绝对容差0.001误报',
            'c6f65c66acaafb6f':'实际0、参考约4.56e-5，纯归一化误差放大为0.9785，缺少绝对容差',
            '8407027ddaa21060':'约1.18e6的输出差0.5，固定绝对容差误报；已观测第一调度，第二调度编译超时'
        }[ident]
    elif ident=='119930b0d5c760cc':
        status='疑似数值不稳定，暂不报目标bug'
        reason='原误差3.3；改输入scale=0.1为可精确表示的0.125后两调度、两组输入均通过；仍需逐操作误差轨迹才能最终定性'
    elif cat=='segfault' and 'triton' in str(p):
        status='已复现目标编译器bug'
        reason='原始kernel在无GPU分配/执行的sm_89离线编译中SIGSEGV，make_ttgir的pass manager内崩溃'
    elif cat=='segfault':
        status='历史NVCC崩溃，待缩减'
        reason='原日志明确为nvcc编译子进程segfault；非GPU kernel越界证据，非直接TileLang崩溃；本轮未重放'
    elif ident=='28f7a4c7a8eda0b4':
        status='已复现目标编译器bug'
        reason='重复执行从有限正确值变成Inf；block_K=32；shared转置前后加显式同步后2输入x2调度x20重复全部通过'
    elif ident=='b5c2b8bb73c6097d':
        status='已复现目标编译器bug'
        reason='schedule0精确匹配参考，schedule1第8行起错误；独立K=8整数GEMM复现列重复，源码明确支持m16n8k8'
    elif ident=='25a869b78ca36590':
        status='已复现目标编译器bug'
        reason='第二次tl.dot三个输入均有限且匹配参考，输出NaN；不含循环/别名的16x16 fp16常量矩阵tl.dot最小核同样全NaN'
    elif cat in ('wrong_result','schedule_mismatch') and fp16_k8:
        status='K=8已知错误机制候选'
        reason='与已确认K=8布局/列重复机制同配置；不能因block_K=8判非法，也不能未经缩减把全部219条算独立bug'
    elif cat=='layout_inference':
        status='布局编译失败候选，待缩减'
        reason='no available layout；代表2ffa308929b8d63e冷缓存复现且无transpose；其他记录仍需排查布局约束/支持范围'
    elif cat=='timeout':
        status='超时证据不足'
        reason='历史Region日志只有Execution timed out，不能区分编译慢、参考慢、资源压力和kernel hang'
    elif cat=='schedule_mismatch':
        status='数值容差候选，待确认'
        reason='非GEMM的两个copy调度差异，绝对误差0.0068359375或0.25；需相对尺度/ULP验证'
    row=dict(path=str(p),case_id=ident,backend='triton' if 'triton' in str(p) else 'tilelang',
             original_root_cause=cat,triage=status,reason=reason,version=rp.get('version','extended'),
             dtype=d['dtype'],shape=str(tuple(spec.get(k) for k in ('M','N','K'))),
             tile=str(tuple(spec.get(k) for k in ('block_M','block_N','block_K'))),
             sha256=item['sha256'])
    rows.append(row)
with (OUT/'triage.csv').open('w',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=list(rows[0]))
    writer.writeheader();writer.writerows(rows)
summary={'captured_at':snapshot['captured_at'],'total':len(rows),
         'original_categories':dict(Counter(r['original_root_cause'] for r in rows)),
         'triage':dict(Counter(r['triage'] for r in rows))}
(OUT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
print(json.dumps(summary,ensure_ascii=False,indent=2))
