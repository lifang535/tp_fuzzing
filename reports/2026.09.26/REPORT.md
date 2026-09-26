# 2026-09-26：暂停实验后的 DSL bug 审计

结论：以第四轮（针对 TileLang 0.1.14 / Triton 3.8.0 完成适配后的运行）为口径，保守确认 **4 类 DSL 编译缺陷**：TileLang 3 类、Triton 1 类。对应 **534 条已保存的同签名失败记录**，跨服务器按程序参数与输入种子去重后为 **520 个样本**。这不是 534 个独立 bug，也不是“4 个首次发现的上游问题”。其余结果尚不足以全部定性，以下明确区分。

两台服务器的 TileLang、Triton 实验均已收到 SIGINT 并退出；第四轮四份 summary.json 已落盘。审计结束后再次检查实验进程，状态保存在 各服务器子目录的 stop_status.txt。本次未修改运行中的项目源代码、原始失败样本或 conda 环境；用于对照的修改仅存在于 reports/2026.09.26-audit/ 内的独立审计脚本。

## 按服务器阅读

- [32906 / NVIDIA vGPU-32GB](server_32906_vGPU32GB/REPORT.md)：307 条已确认签名记录；本机磁盘异常与诊断不足记录单列。
- [41790 / NVIDIA GeForce RTX 4090](server_41790_RTX4090/REPORT.md)：227 条已确认签名记录；包含已验证的 fuzzer atomic 检查误报。
- [复现证据及执行服务器](evidence/README.md)：明确区分样本来源机器和复现实验执行机器。

## 口径与数据范围

| 服务器 | 后端 | 第四轮目录时间 | 实际测试数 total_tested | 通过 | 失败 bugs_total | oracle_unstable |
|---|---|---|---:|---:|---:|---:|
| 32906 / vGPU | TileLang | 2026.09.25-23.14 | 21,110 | 2,644 | 18,402 | 64 |
| 32906 / vGPU | Triton | 2026.09.25-23.14 | 23,537 | 8,830 | 14,478 | 229 |
| 41790 / RTX 4090 | TileLang | 2026.09.25-23.10 | 2,301 | 2,009 | 246 | 46 |
| 41790 / RTX 4090 | Triton | 2026.09.25-23.10 | 7,291 | 6,908 | 222 | 161 |
| 合计 | | | **54,239** | **20,391** | **33,348** | **500** |

- 数据来自停止后 summary.json 与 failed/**/*.json；日志中的迭代序号包含重复生成，不能与 total_tested 混用。
- 环境均为 tp_fuzzing_latest：TileLang 0.1.14、Triton 3.8.0、PyTorch 2.4.0+cu124。
- 第四轮保存策略为不限制每类样本数量。前三轮仅作历史对照，不混入当前版本有效性结论；早期每类保存上限为 10，历史保存数不是实际触发总次数。
- 534 条为相同错误签名的记录数；只对代表样本进行独立复现及对照，没有逐一重跑全部 534 条。4 类是经审计归并的缺陷机制/错误签名数，不替代上游开发者最终根因归并。
- 去重键：按缺陷类别分组，对 JSON 中 params（包含程序 IR 和 input_seed）排序序列化后计算 SHA-256。同参数跨服务器重复只计一次。

## 可计入当前 DSL 缺陷统计的四类

| 编号 | DSL | 缺陷 | 32906 记录 | 41790 记录 | 合计 | 去重样本 |
|---|---|---|---:|---:|---:|---:|
| TL-01 | TileLang | bool 向量 CUDA 代码生成失败 | 95 | 61 | **156** | 153 |
| TL-02 | TileLang | 自动选择的归约布局无法 lowering | 80 | 74 | **154** | 146 |
| TL-03 | TileLang | 流水线拷贝生成非法 cp.async 字节宽度 | 30 | 21 | **51** | 49 |
| TR-01 | Triton | tl.flip 默认 dim=None 导致编译异常 | 102 | 71 | **173** | 172 |
| 合计 | | **4 类** | **307** | **227** | **534** | **520** |

### TL-01：bool 向量代码生成

签名：`Cannot convert type boolx8/boolx16 to CUDA type`。不同向量宽度归并为一类。

从失败样本中提取独立 kernel 定义，去掉 fuzzer 的参考计算、数值判定与调度框架，仅执行 `tilelang.compile`。默认配置失败；同一 kernel 使用 `tirx.disable_vectorize=True` 后编译通过。两个输出观测变体均呈现该对照结果。因此不是 fuzzer 数值判定错误或 GPU 运行资源限制，而是向量化与 CUDA 类型生成之间的问题。

证据：isolated_bool_vector.py / isolated_bool_vector.log。最初尝试的简单 bool 输出小程序可以通过，已如实保留在 minimal_checks.log；它没有复现复杂表达式的触发条件，不作为缺陷证据。

上游已有相同 boolx8 代码生成症状报告，不能宣称本轮首次发现：[TileLang #2206](https://github.com/tile-ai/tilelang/issues/2206)。

### TL-02：归约布局 lowering

签名：`ReduceOp cannot lower a layout where a source index depends on a thread-owned reduce segment`，栈位于 `src/backend/common/op/reduce.h:534`。

提取的 kernel 使用正常的 fragment、copy、gemm、reduce API，布局由编译器自动推断；不包含 fuzzer oracle。单独编译仍失败，关闭向量化或关闭 enable_async_copy 后同一断言仍出现。源码检查及此独立编译复现支持将其归为 DSL 自动布局/归约 lowering 缺陷，而不是参考结果误差。不同索引表达式归并为同一个签名；尚未获得上游修复或维护者确认，不声称已经完成补丁级根因定位。

证据：isolated_reduce_layout.py / isolated_reduce_layout.log；原始样本完整重放亦复现相同断言。

### TL-03：流水线拷贝生成非法 PTX 字节宽度

签名：`IsValidCPAsyncTransferBytes(total_bytes)` 失败，生成的传输宽度为 2 字节，而 cp.async 要求 4、8 或 16 字节。

已进一步去掉生成器的函数调用、复杂运算及参考实现，化简为独立矩阵乘法：A=(1,9157)、B=(9157,3709)、块大小 64×64×128、256 线程。源码仅使用 T.copy、T.gemm 和 T.Pipelined，没有手写非法 PTX。`num_stages=4` 复现同一错误，`num_stages=0` 编译通过。这说明非法字节宽度由流水线编译路径引入，不是 fuzzer 主动生成了非法 cp.async 调用。

证据：minimal_cp_async.py / minimal_cp_async.log；isolated_cp_async.py 保留原始 kernel 的独立编译对照。

上游已有同一断言的相关报告：[TileLang #2172](https://github.com/tile-ai/tilelang/issues/2172)。相同诊断不自动证明是完全相同的上游根因，也不能据此宣称新问题。

### TR-01：flip 的默认参数错误

签名：`tl.flip(x)` 编译失败，内层错误是整数与 None 比较的 TypeError。

独立 Triton kernel（读取 128 个元素、flip、写回）通过 ASTSource 直接编译，无需 fuzzer，也不执行 GPU 数值比较。省略 dim 时失败；改为 `dim=0` 后通过。安装版本中的 flip 在处理默认 None 之前执行了维度范围比较。

证据：minimal_checks.py / minimal_checks.log。原始失败样本重放亦失败在 tl.flip。

官方 API 暴露 dim=None 默认值：[Triton flip 文档](https://triton-lang.org/main/python-api/generated/triton.language.flip.html)。上游已有完全匹配的问题及修复关联：[Triton #10790](https://github.com/triton-lang/triton/issues/10790)。本次确认的是安装的 3.8.0 仍可触发，不能计为首次发现。

## 排除及待定结果

以下数量均不计入“4 类已确认 DSL 编译缺陷”。

| 处理 | 记录数 | 依据 |
|---|---:|---|
| 明确磁盘错误 | 19 | 错误信息明确包含 Disk quota exceeded / No space left |
| 诊断不足的 other / segfault | 32,283 | 多数发生于第一台异常时段，缺少足够栈或仅有退出码；不能全部证明来自磁盘，也不能当真实 DSL bug |
| 资源限制或非法生成配置 | 230 | shared_memory_overflow 185、warp_partition 29、timeout 12、gpu_oom 4 |
| 已确认 fuzzer 判定缺陷 | 1 | atomic_min 严格检查错误地作用到普通 store 的 mem2 |
| 待定候选 | 281 | 数值语义、参考实现与布局合法性尚未全部排除 |
| 已确认 DSL 同签名记录 | 534 | 上表四类 |
| 失败总计 | **33,348** | 与四份 summary.json 的 bugs_total 总和一致 |

超时记录尚未区分机器负载、编译器耗时或潜在死循环；将它们排除确认统计不等于证明 DSL 没有问题。

此外，500 条 oracle_unstable 是主动跳过的参考不稳定情况，不在 33,348 条失败中。

待定候选细分：wrong_result 190、layout_mismatch 15、schedule_mismatch 9、pass_config_mismatch 10、precision_mismatch 27、algebraic_identity 3、layout_inference 26、stride_alignment 1，共 281 条。其中 stride_alignment 代表记录实际也包含 no available layout found，说明标签还需要二次归并。

对代表数值样本的重放能够重复报警，但“可重复报警”不等于“真实 DSL 数值 bug”：

- precision 路径的 fp16 累加参考模型按每 16 个乘积进行一次舍入；需核对硬件允许的累加精度/舍入顺序，不能仅凭与这个模型不同即定性。
- pass_config 包括 FP fusion 等选项；浮点变换不普遍保证逐位相同。需要针对样本做数值语义与误差分析。
- algebraic_identity 代表样本差值约 0.000255，靠近当前检查阈值；不能仅凭恒等式名称视为编译错误。
- layout_inference 的 has_best 断言在独立 kernel 中也能复现，但尚未排除不支持的布局组合，保守留在候选。

### 确认的一处 fuzzer 误报

样本：`failed_extended_indexed_memory_702e6b89a19c62c0`，保存于第二台第四轮 TileLang 的 atomic_mismatch。

该程序仅对 mem5 执行 atomic_min，而 mem2 是普通 store（通过 mem3、mem4 别名）写入的缓冲区。检查代码却使用整个程序级别的 atomic_kind，将所有 memory 的比较均派发给 atomic_min 的严格相等判定。mem2 的普通浮点写入因此因约 7.45e-9（约 1 ULP）的差异被误标为 atomic_mismatch。

在独立审计副本中，仅把 atomic 检查的作用范围限定为真正的 mem5；kernel 计算没有改动。原始样本失败，修改检查范围后的副本输出 ALL PASSED、退出码 0。因此这 1 条明确排除。证据：atomic_checker_control.py / atomic_checker_control.log；代码位置见 仓库 src/workflow/emitter/extended_runtime.py 中 _atomic_kind 与 memory 比较分支。

## 历史对照与论文表述

前三轮的保存类别计数见 historical_saved_counts.json，完整抽取数据见 raw_32906.json、raw_41790.json。早期 callback 导入失败、ASTSource signature 的整数键等 fuzzer 兼容性问题，不应混入第四轮 DSL 缺陷数。第一轮旧 TileLang 的 dtype 缓存问题也不能当作当前 0.1.14 仍存在的 bug；本轮未重审其旧版全部样本。

建议论文采用以下保守表述：

> 在适配 TileLang 0.1.14 和 Triton 3.8.0 的实验中，经代表样本重放、独立 kernel 编译和对照检查，确认了四类 DSL 编译缺陷，包含三类 TileLang 缺陷和一类 Triton 缺陷。四类缺陷对应 534 条保存记录，按程序参数跨设备去重后为 520 个样本。环境失败、已确认的测试框架误报及尚未完成语义验证的候选未计入确认缺陷数。

不要将“4 类”写成“4 个新发现的 bug”，不要将“520 个样本”写成“520 个独立 bug”。本次没有将 281 条未定性候选判为无 bug；确认数是现有证据支持的保守下限，后续最小化及参考实现核验可能增加或合并类别。

## 仓库内交付文件

- audit_summary.json：总统计与停止后的四份 campaign 摘要。
- server_32906_vGPU32GB/、server_41790_RTX4090/：每台报告、运行摘要、逐条分类压缩 CSV、已确认签名 CSV、停止状态。
- evidence/：独立编译脚本、原始代表程序、完整重放日志与检查器对照；执行位置见该目录 README.md。
- historical_saved_counts.json：四轮历史保存类别计数。
- SHA256SUMS.json：本报告目录的文件摘要，便于服务器拉取后验证。

完整原始抽取快照 raw_32906.json、raw_41790.json 与源代码快照保存在本地 `/home/lifang535/tp_fuzzing_audit_20260926/`，两台服务器也各自保留 `reports/2026.09.26-audit/` 审计归档。仓库报告使用分服务器 CSV 和代表证据，避免重复提交整份原始快照。
