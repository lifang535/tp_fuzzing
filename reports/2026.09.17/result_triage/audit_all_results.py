"""Read-only audit of saved results; historical replay evidence is labelled explicitly."""
import csv
import hashlib
import json
import re
from collections import Counter
from datetime import datetime
from pathlib import Path

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
previous = {r['path']: r for r in csv.DictReader((OUT / 'triage.csv').open())}
rows = []
for path in sorted((ROOT / 'results').glob('*/failed/*/*.json')):
    raw = path.read_bytes()
    d = json.loads(raw)
    rel = str(path.relative_to(ROOT))
    source = path.with_suffix('.py').read_text()
    category = d['root_cause']
    params = d['params']
    digest = hashlib.sha256(raw).hexdigest()
    flags = []
    status, reason = '待确认', '没有充分证据判定；不能按错误类别直接计算真实bug数'
    provenance = '本轮静态审查，未GPU重放'
    prior = previous.get(rel)
    if prior and prior['sha256'] == digest:
        status, reason = prior['triage'], prior['reason']
        provenance = '复核既有审查证据；JSON SHA256一致；复现是既有记录，非本轮执行'
    elif category == 'dtype_mismatch':
        declared = re.findall(r'^    dtype = "(float\d+)"', source, re.M)
        host = set(re.findall(r'torch.randn\([^\n]*dtype=torch\.(float\d+)', source))
        if declared == [d['dtype']] and host == {d['dtype']}:
            status = '已确认缓存机制的同类候选，未逐条复现'
            reason = '声明dtype与host输入一致；闭包dtype未作为JIT实参；已有独立缓存碰撞对照。不是混合精度本身非法'
    elif category == 'ptx_async_boundary':
        status = '已确认cp.async机制的同类候选，未逐条复现'
        reason = 'fp16普通T.copy被自动降为非法2字节cp.async；已有最小复现，同症状尚未逐条缩减'
    elif category == 'shared_memory_overflow':
        status = '非有效目标编译器bug'
        reason = '保存日志明确Required超过Hardware limit；调度资源预算问题'
    elif category == 'gpu_oom':
        status = '非有效目标编译器bug'
        reason = 'PyTorch参考计算cublasCreate分配失败；不能据此报告目标编译器错误'
    elif category == 'assertion_failure':
        status = '非有效目标编译器bug'
        reason = '实际为CUDA设备busy/unavailable；分类器把TORCH_USE_CUDA_DSA提示误作assertion'
    elif category == 'warp_partition':
        status = '不支持的调度组合，暂不计真实bug'
        reason = '16x16 tile配128/256线程；ComputeDefaultWarpPartition没有满足约束的warp划分；至多另议错误诊断质量'
    elif category == 'timeout':
        status = '超时证据不足'
        reason = '不能从超时区分编译慢、参考慢、环境竞争与kernel hang'
    if not prior and category == 'wrong_result':
        if ('accumulate_reduce' in source and params.get('N', 0) > params.get('block_N', 0)
                and ('sum(dim=-1, keepdim=True)' in source or 'max(dim=-1, keepdim=True)' in source)):
            flags.append('tile/global归约语义不一致')
        if ('row_stat_' in source and '.clamp(min=1e-6)' in source
                and re.search(r'row_stat_\d+\[i\] \+ 1e-6', source)):
            flags.append('分母sum+eps与clamp(sum)语义不一致')
        if flags:
            status = 'oracle语义不一致，当前记录不能作目标bug证据'
            reason = '；'.join(flags) + '；可能同时存在编译器问题，需修正oracle后重测'
        if d['dtype'] == 'float16' and params.get('block_K') == 8:
            flags.append('fp16_K8：已知真实缺陷配置候选，非逐条确认')
    rows.append(dict(path=rel, code_name=path.with_suffix('.py').name,
                     root_cause=category, triage=status, reason=reason,
                     flags='；'.join(flags), evidence=provenance, sha256=digest))
with (OUT / 'all_results_triage.csv').open('w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
summary = dict(captured_at=datetime.now().astimezone().isoformat(), total=len(rows),
               categories=dict(Counter(r['root_cause'] for r in rows)),
               decisions=dict(Counter(r['triage'] for r in rows)),
               unchanged_prior=sum(r['evidence'].startswith('复核') for r in rows),
               flags=dict(Counter(flag for r in rows for flag in r['flags'].split('；') if flag)))
(OUT / 'all_results_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2))
print(json.dumps(summary, ensure_ascii=False, indent=2))
