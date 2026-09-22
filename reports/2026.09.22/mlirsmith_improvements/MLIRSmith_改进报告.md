# 参考 MLIRSmith 改进 tp_fuzzing：减少无效 fuzzing、增大 bug 覆盖率

日期：2026-09-22
范围：`tp_fuzzing/` 目录内（未改动目录外文件）

## 1. 背景与对照：MLIRSmith 做了什么

MLIRSmith（ASE'23）把无效 fuzzing 归结为四类浪费，并逐一对症：

| MLIRSmith 机制 | 对应的 tp_fuzzing 改进 |
| --- | --- |
| DiversityCriteria 覆盖率加权（未覆盖特征大权重） | StructuralFeedback 未覆盖 boost + 已尝试未通过特征降权 0.6 |
| 无效程序计入 wasted effort（编译/执行前置校验失败不计 bug） | oracle trust gate：数值检查无意义的程序归为 `oracle_unstable`，单独计数，不进 bug 列表 |
| crash oracle：逐个 pass + 随机 pass 序列定位崩溃点 | TILESMITH_STAGE 阶段标记 + 超时定位 + 失败定位解析 |
| 重复 root cause 限流（同类 bug 只保留有限 reproducer） | `max_same_root_cause=10` 保存上限 + 已知 root cause 去重计数 |
| 前端能力探测（跳过后端不支持的特性，避免注定失败的编译） | triton tanh libdevice 绑定探测（2.x/3.x 差异），探测失败退化为等价表达式 |

## 2. 核心问题：chaotic 程序的假阳性 wrong_result

改动前 fuzzer 把"参考解释器结果 ≠ kernel 结果"一律记为 `wrong_result`。但带非线性反馈环的程序（带 carry 的 for 循环、函数调用链）会把 fp32 算术顺序噪声放大 ~10^6 倍：fp32 参考结果自身都无法复现（fp64 副本对比自误差高达 125），**任何** kernel 都不可能通过检查 —— 这类失败是 oracle 噪声，不是编译器 bug。

实测（8 个保存的 triton `calls_*` wrong_result，精确重建 campaign 输入）：

| reproducer | fp64 自相对误差 | 1-ulp 输入扰动自相对误差 | 结论 |
| --- | --- | --- | --- |
| 2508a8b | 0.435 | 62 | 混沌，双门拦截 |
| 4978c85 | 0.995 | 6.01 | 混沌，双门拦截 |
| 6b66bdd | 2.61 | 4.09 | 混沌，双门拦截 |
| b5b190 | 1.68 | 2.17 | 混沌，双门拦截 |
| be73c9 | 125 | 741 | 混沌，双门拦截 |
| 0fd0af | inf | 0（NaN 位置不同） | fp64 门拦截 |
| c8833055 | 0.00315 | 1.74 | **仅扰动门拦截**（归约顺序敏感） |
| 92e1ed | 6.2e-05 | 0.298 | **仅扰动门拦截**（归约顺序敏感） |

对照组：良性程序 fp64 自误差 3.9e-07~9.9e-07，扰动自误差同量级 —— 与固定阈值 1e-2 有 3~8 个数量级的分隔。

### 双重 oracle trust gate 设计

数值检查失败时，先用两个自一致性检查验证参考解释器本身可复现，任一失败即跳过数值检查：

1. **fp64 门**：把参考解释器源码做字符串变换生成 fp64 版本（`double_reference_source`），fp32 参考与 fp64 参考自对比。拦截算术顺序噪声被混沌放大的程序。
2. **1-ulp 扰动门**：对输入做 1-ulp 扰动（`_ulp_jitter`，0 值移到 `finfo.tiny` 再用 `nextafter`）重算参考。两个实现之间最小的可能差异就是输入的读取/舍入差异；若 1-ulp 输入变化让参考输出变化远超容差，说明该程序对归约顺序敏感，同样不可能被任何 kernel 复现。拦截 fp64 门漏掉的温和放大器。

门失败时抛 `RuntimeError('ORACLE UNSTABLE: ...') from None`（抑制 WRONG RESULT 链式文本，避免污染分类），根因归类为新的 `BugType.ORACLE_UNSTABLE = "oracle_unstable"`（分类规则置于所有 wrong_result 规则之前，含 layout invariance 包装）。

**代价**：仅在数值检查失败时多跑 1~2 次 CPU 参考解释器（失败路径），正常路径零开销。固定阈值 1e-2 与良性噪声（≤1e-4）分隔 2 个数量级以上；扰动门在 fp16 输入下相对扰动 5e-4，良性程序放大 ≤2 倍，混沌/顺序敏感程序放大 ≥300 倍，无中间地带。

## 3. 改动清单

### 3.1 oracle trust gate（本次会话 + 前一阶段）

| 文件 | 改动 |
| --- | --- |
| `src/workflow/emitter/runtime.py` | `double_reference_source()`：字符串变换生成参考解释器的 fp64 副本（去 `.float()`/`.to(dtype)`，重命名函数） |
| `src/workflow/emitter/region_checks.py` | `_ulp_jitter()`、`_reference_stable(ref, double, jitter=None)`（固定 1e-2 阈值）、`_run_region(..., reference_verify=, reference_jitter=)` 失败路径双门检查；**campaign 实测发现并修复 GPU 缺陷**：`torch.nextafter` 的 CUDA kernel 无 Half 重载（`nextafter_cuda not implemented for 'Half'`），fp16 程序在 GPU 上触发扰动门即崩溃并被误归类为 'other' 噪声 —— 改为扰动计算迁到 CPU（所有浮点 dtype 均支持）再搬回输入设备 |
| `src/backends/common/region_emitter.py` | 原生 plain/checked harness：fp64 副本 + `_ULP_JITTER_SOURCE` 嵌入、plain harness 内联双门、checked 两个调用点传两个闭包 |
| `src/backends/common/typed_emitter.py` | typed harness：checks 元组加 `_ulp_jitter`、两个调用点传两个闭包 |
| `src/backends/common/diagnostics.py` | `classify_root_cause` 第一条规则：`'oracle unstable' in err → 'oracle_unstable'`（先于全部 wrong_result 规则） |
| `src/backends/common/builtin.py` | `classify_error` 第一条规则同上，返回 `BugType.ORACLE_UNSTABLE` |
| `src/workflow/oracle/oracle.py` | `BugType.ORACLE_UNSTABLE`；超时分支用 `_failure_location` 解析最后一个 TILESMITH_STAGE 标记；非 extended 的 `compilation_complete` 认 `TILESMITH_STAGE=execute` 标记 |
| `src/workflow/fuzzer/fuzzer.py` | `stats.oracle_unstable` 计数（MLIRSmith wasted effort 风格）；oracle_unstable 不进 bug 列表，受 `max_same_root_cause` 限制保存少量审计 reproducer；summary.json 输出 `oracle_unstable` |

### 3.2 无效 fuzzing 削减（前一阶段）

| 文件 | 改动 |
| --- | --- |
| `src/workflow/feedback.py` | `weight()`：已尝试未通过的特征降权到 `base*0.6`（只产生前端拒绝的组合不得胜过未尝试/已验证组合；0.6 保证模板组合权重为正） |
| `src/config/config.py` | `max_same_root_cause=10`（同类 root cause 的 reproducer 保存上限） |
| `src/backends/triton/ops.py` | tanh 探测 `tl.extra.cuda.libdevice.tanh`（3.x）/`tl.extra.libdevice.tanh`（2.x），均无则退化为 `1 - 2/(1+exp(2x))` 恒等式 —— 避免所有 tanh 程序在前端就死掉 |
| `src/backends/common/probe_emitter.py` / `src/workflow/emitter/probe_runtime.py` | probe 路径阶段标记（`probe_reference` / `probe_instantiate_{v}` / `probe_variant_{v}`） |
| 各 harness | `TILESMITH_STAGE=` stderr 标记（prepare/execute/reference/execute_variant_{i}/layout:{la}/{lb}），失败与超时定位到具体变体 |

### 3.3 测试与固件

| 文件 | 改动 |
| --- | --- |
| `tests/test_feedback.py` | 断言 0.6 降权新语义 |
| `tests/test_transcendentals.py` | tanh 发射断言跟随探测到的 libdevice 绑定 |
| `tests/test_bug_class_reachability.py` | `_run_layout_case` 调用文本断言拆分为前缀 + 两个门闭包断言；新增 `test_emitted_harnesses_are_self_contained`（symtable 作用域分析：7 种 harness 变体 × triton/tilelang 发射后不得有任何未定义全局名） |
| `tests/test_oracle_gates.py`（新增） | `_ulp_jitter` 语义（1-ulp 扰动、0 值移开、整数 dtype 无操作、设备保持）+ `_reference_stable` 阈值行为的永久 CPU 测试 |
| `tests/test_generation_diversity.py` | 反馈测试同时设置 attempted+passed（对齐 observe() 语义） |
| `tests/test_typed_regions.py` | mock `tl.extra.cuda.libdevice` 补齐（tanh 探测路径） |
| `tests/fixtures/current_programs.json` | 发射摘要按新发射更新（全部 27 条；第二轮 11 条为 `_reference_stable` 嵌入修复、第三轮 19 条为 `_ulp_jitter` GPU 修复） |

## 4. 验证

- **单元测试**：`python -m unittest discover -s tests`（repo 根目录）→ 275 tests, **OK**。
- **campaign 实测回环**：tilelang campaign 首轮窗口即抓到扰动门的 GPU 缺陷（fp16 程序门触发时 `nextafter_cuda not implemented for 'Half'` → 3 个程序误落入 'other'）—— 修复后这些噪声工件已隔离（`quarantined_editor_window/`），恢复的 campaign 会用修复后的代码重测；修复与永久测试一并合入（`tests/test_oracle_gates.py`）。
- **发射自包含静态验证（永久测试）**：symtable 作用域分析证明 7 种 harness 变体（triton/tilelang × native-plain/native-checked/typed-checked/typed-sweep）发射后不引用任何未定义全局名 —— 锁住此前 `_reference_stable` 漏嵌导致的 NameError 假失败（该类错误能通过 compile() 却在门触发时变回 'other' 噪声）。
- **fp64 门 CPU 验证**（`gate_verify`，27 项检查全过）：变换健全性、良性 typed/native 稳定（3.9e-07/9.9e-07）、混沌 typed 拦截（2.61）、分类（含 layout 包装、`classify_error`）、6 种 harness 发射+编译（triton/tilelang × native-plain/native-checked/typed-checked）+ 2 种 probe 标记。
- **扰动门 CPU 验证**：8 个保存 reproducer × 全部 input_case（精确重建 campaign 输入：`input_seed` + `_region_input_storage` 原始 layout）→ **8/8 全部被双门拦截**；`_ulp_jitter` 单元检查（1-ulp 位移、0 值处理、int8 无操作）；良性对照不被误拦；发射文本接线检查全过。
- **GPU 端到端烟测**（campaign 结束后实跑）：发射混沌程序 harness（triton calls_* 深调用链 + row_softmax/for/if）→ 退出码 1 + stderr 含 `ORACLE UNSTABLE: reference disagrees with its fp64 or one-ulp-jittered copy` ✓；发射良性 native checked GEMM → 退出码 0 + stdout `ALL PASSED`、无门误触 ✓。

## 5. 结果对比

### Triton（hard-shape）

| 指标 | 改动前（seed=42） | 改动后（seed=123） |
| --- | --- | --- |
| tested | 626 | 400 |
| compiled（编译成功率） | 494（78.9%） | 380（**95.0%**） |
| passed | 493 | 378 |
| codegen_api_mismatch | **116** | **0** |
| wrong_result | 11 | 19（8 个 calls_* 混沌/顺序敏感经双门确认均为 oracle 噪声；2 个 extended_mixed `actual=nan; expected=inf` 为真实候选） |
| segfault | 3 | 1 |
| shared_memory_overflow | 2 | 1 |
| timeout | 1 | 0 |

解读：改动后编译率从 78.9% 提升到 95.0%（tanh 探测 + 覆盖率加权减少注定失败的程序）；codegen_api_mismatch 从 116 个（同一前端缓存碰撞 bug 的重复报出，挤占 18.5% 的测试预算）降到 0（root cause 去重 + 限流 + 生成器不再死磕同类组合）；wrong_result 中的混沌假阳性被双门拦截为 oracle_unstable（不计 bug）。

### Tilelang（hard-shape，seed=123，-n 200 resume）

本窗口新增 200 个 case 的完整统计（全 campaign 累计 tested=334，含早期窗口 134 个历史 case）：

| 指标 | 本窗口数值 |
| --- | --- |
| tested | 200 |
| compiled | 184（**92.0%**） |
| passed | 166（83.0%） |
| oracle_unstable | **5**（2.5%：双门拦截、跳过数值检查、不计 bug） |
| wrong_result | 2（error=5.09/0.00195，参考已由双门认证稳定 → **真实 bug 候选**） |
| layout_mismatch | 1（新类别：layout sweep 不变性违反） |
| nondeterminism | 1（新类别：repeat_count 间结果不一致） |
| dtype_mismatch | 21 次命中 → 去重限流后保存 10 个 reproducer（`max_same_root_cause=10` 生效，本窗口全部为 dup 不再落盘） |
| ptx_async_boundary / layout_inference / shared_memory_overflow / tilelang_codegen_error / timeout | 5 / 3 / 2 / 2 / 2 |
| other | **0** |

全 campaign：bugs_total=52 条报告、bugs_unique=10 个类别；隔离的噪声工件（broken window + env noise + 门缺陷窗口）44 个文件在 `quarantined_editor_window/`（24）、`quarantined_env_nvcc/`（20），不计入任何类别。

解读：编译成功率 92.0%；5 次门拦截（2.5% 的测试预算）在改动前都会落成 wrong_result 噪声；剩 2 个 wrong_result 是门认证过的候选（深调用链 + for/if 的 tilelang codegen 分歧），值得人工跟进；dtype_mismatch（tilelang 前端缓存碰撞，本 fuzzer 的目标 bug 类）21 次重复报出被压成 10 个 reproducer，不再挤占测试预算；layout_mismatch 与 nondeterminism 两个新类别由 layout sweep 与 repeat 不变量检查暴露。

## 6. 剩余建议

1. **扰动门阈值自适应**：当前固定 1e-2，建议对 fp32 输入程序单独统计（fp32 的 1-ulp 相对扰动 6e-8，良性放大与 fp16 不同量级），必要时按输入 dtype 分档。
2. **`where` 分支敏感程序的理论风险**：1-ulp 扰动可能翻转 `where` 条件分支；若两分支输出量级差异巨大（当前生成器不产生此类程序），扰动门会误拦真实 bug。风险已接受并记录，可用"分支输出差异上限"的生成约束彻底消除。
3. **oracle_unstable 的反馈复用**：被门拦截的程序目前只计数不采样。可将它们的结构特征计入 wasted-effort 反馈，让生成器主动回避高混沌倾向的组合（如深调用链 + for-carry），进一步减少无效 fuzzing。
4. **extended 路径门**：本次门只覆盖 Region/probe 路径；extended 路径的 2 个 `nan vs inf` wrong_result 是独立 oracle 逻辑，若继续出现 NaN 位置类噪声，可把同样的稳定性检查移植过去。
