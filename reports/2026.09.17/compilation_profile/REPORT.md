# tp_fuzzing 编译耗时对比

2026-09-17，RTX 4060 Laptop GPU / sm_89，TileLang 0.1.11、Triton 3.0.0、PyTorch 2.4.0+cu124。

**实测主要瓶颈是 TileLang 编译器的 `tl.LayoutInference`，尤其是带控制流和函数调用的 Extended 用例。默认四个变体会重复支付这部分成本。** 较简单的 Extended 用例中，NVCC 的占比更高；生成和变异本身只有毫秒级。

## 实验如何比较

- 主试验取实际 campaign 中已通过但运行很慢的 `mixed` 程序，同一份保存的 IR 分别送给两种后端。每个后端做三次冷编译及对应的新进程缓存重放，共 12 次，全部通过。
- 每次冷编译使用独立 DSL 缓存，不删除或清空用户缓存；CUDA 驱动缓存继承当前环境。每轮交替后端顺序，所有性能试验串行执行。
- 另测 `arithmetic`、`indexed_memory`、`shape_matmul`、`control_calls`，普通 Region，以及同一 mixed 程序的单变体版本。路线对照每个条件一次，用于定位而非估计整个程序空间的平均性能。
- 在子进程中包装编译阶段；TileLang 通过 TVM pass instrument 直接记录每个 pass，Triton 记录 frontend、TTIR、TTGIR、LLVM IR、PTX、cubin。不是根据日志时间戳猜测阶段成本。
- 汇总使用 exclusive 时间，消除嵌套阶段重复计数；总时间还包括新进程、导入、CUDA 初始化、参考计算、执行/检查和保存编译产物。
- 用户的两个 fuzzer 保持运行。共享 CPU/GPU 的负载会影响绝对时间，因此保留每次测量及范围。本文的倍率仅针对被测程序，不能直接推广到整个 campaign。

正式可比结果共 34 次运行，均成功且通过嵌套计时加和检查。另外保留了 4 次不匹配 schedule 的普通 Region 尝试，其中 TileLang 两次提前失败；它们不纳入可比耗时统计。前期 6 次方法验证运行也单独排除。

表内总时间为完整子进程耗时，不含父进程生成 IR 和源码。源码生成时间单列在 `phases.csv` 的 `emit` 列；每个实验父进程首次加载 emitter 依赖约一秒，其后通常为毫秒级，不会在同一个正常 fuzzer 进程中每轮重复导入。

详细方法和失败样本处理见 [EXPERIMENT.md](EXPERIMENT.md)。

## 主试验：慢 mixed 程序

冷编译结果，单位秒，三轮均值；每轮均包括四个变体：

| 阶段 | TileLang | Triton |
|---|---:|---:|
| `tl.LayoutInference` | 108.970 | 不适用 |
| NVCC | 14.887 | 不适用 |
| 其它编译/准备；Triton 为全部编译/准备 | 14.116 | 1.452 |
| 启动、导入和 CUDA 初始化等 | 4.422 | 1.799 |
| 参考计算 | 0.056 | 0.034 |
| GPU 启动/同步、运行与结果检查 | 9.512 | 2.704 |
| 编译产物 I/O | 0.045 | 0.058 |
| **完整子进程总耗时** | **152.007** | **6.046** |

TileLang 冷编译总耗时范围 140.931–164.966 秒，中位数 150.123 秒；Triton 为 5.941–6.235 秒，中位数 5.963 秒。这个程序的完整流程约慢 25.1 倍。

TileLang 的编译/准备合计约 137.97 秒，占完整流程 **90.8%**。其中 `LayoutInference` 单项占完整流程 **71.7%**，占编译/准备时间约 **79.0%**。因此，减少 Python 生成时间或少执行几次 kernel，无法消除这里的主要差距。

TileLang 次慢的具体 pass 是 `tl.LowerTileOp`，平均 5.03 秒；其余单项 pass 明显小于布局推导。Triton 的所有编译/准备约 1.45 秒，其中 LLVM IR 阶段约 0.42 秒、PTX 约 0.09 秒、cubin/ptxas 约 0.20 秒。Triton 也有布局相关转换，只是没有 TileLang 这个同名 pass，不能把表中的“不适用”理解为完全不做布局工作。

## 为什么布局推导会放大耗时

`LayoutInference` 是编译器把逻辑 tile/fragment 映射到线程、寄存器布局并协调操作约束的阶段，不是 fuzzer 随机选择输入 contiguous/strided 布局的那一步。

当前安装版 `tilelang/src/transform/layout_inference.cc` 的 `Run()` 从第 329 行开始，包含严格约束推导、工作队列迭代、放宽约束后的推导，以及别名布局传播。当前测量已经定位到整个 pass，但没有进一步区分其内部哪一个 C++ 子函数最贵。

被测 mixed 程序含 106 个操作、1 个 helper、3 处调用，以及 if/for/while、归约、变形、访存和两个内部 matmul；单个结果最多 512 个元素。它的慢编译不能简单解释成输入矩阵很大。各对照 IR 的结构记录在 [corpus_summary.json](corpus_summary.json)。

两种后端的 Extended 默认都编译 `2 个配置 × 2 种观测模式 = 4 个变体`，对应：

- [TileLang 变体构造](../../../src/backends/tilelang/backend.py)，`extended_variants()`。
- [Triton 变体构造](../../../src/backends/triton/backend.py)，`extended_variants()`。
- [TileLang 编译调用](../../../src/backends/tilelang/extended.py)，`compile_source()` 为每个变体生成一次 `tilelang.compile()`。

这是编译数量相同、每个变体编译成本不同。mixed 的每个 TileLang 变体都花约 27 秒做布局推导。

## 单变体和缓存对照

| 条件 | TileLang 总时间 | TileLang 布局推导 | Triton 总时间 |
|---|---:|---:|---:|
| mixed 默认四变体，冷编译，三轮均值 | 152.007 | 108.970 | 6.046 |
| mixed 单变体，冷编译，一轮 | 37.830 | 26.995 | 3.535 |
| mixed 默认四变体，新进程复用 DSL 缓存，三轮中位数 | 7.814 | 0 | 4.016 |

单变体使 TileLang 完整流程约快 4.0 倍，布局推导时间也约降为四分之一。此实验同时关闭配置对和额外中间值观测，kernel launch 数从 72 降至 18，因此不是严格的“只改变编译次数、所有检查都保持不变”。它会减少差分和观测覆盖，当前正常 fuzzer 的策略没有改动。

缓存命中后的 `LayoutInference` 和 NVCC 均未执行。TileLang 热缓存总耗时三次为 29.826、7.010、7.814 秒，其中第一次执行和检查受共享负载影响明显，所以这里报告中位数。热缓存结果证明重复编译成本大，但 fuzzer 持续生成/变异新源码，不能把反复执行同一程序的缓存吞吐当作真实 fuzz 吞吐。

## 不同路线与整个 campaign

各路线完整阶段表见 [RESULTS.md](RESULTS.md)。简单的算术、索引访存、shape/matmul Extended 对照中，TileLang 总耗时分别为 36.28、32.03、27.51 秒，NVCC 分别约 13.75、13.41、12.90 秒；这些用例的布局推导只有 5–12 秒。

`control_calls` 对照没有 matmul，TileLang 总耗时 169.97 秒，其中布局推导 125.16 秒、NVCC 25.22 秒；Triton 总耗时 4.36 秒。这说明控制流/调用组合也能触发慢布局推导。不同 family 的 IR 还有其它差别，这个对照没有单独证明某一个操作是唯一原因；要继续定位，应对保存的慢程序做最小化，并在布局推导内部细分计时。

普通 Region 的成功匹配对照 `native_matched` 使用已保存的 typed GEMM 程序，M=7165、N=977、K=977，两个 helper，默认两种调度：

| 后端 | 冷编译总时间 | 布局推导 | NVCC | 其它编译/准备 | 参考计算 | 热缓存总时间 |
|---|---:|---:|---:|---:|---:|---:|
| TileLang | 41.460 | 21.751 | 7.723 | 3.580 | 2.316 | 4.741 |
| Triton | 4.315 | 不适用 | 不适用 | 0.848 | 0.361 | 3.315 |

这个普通 Region 样本也以编译为主，TileLang 编译/准备约 33.05 秒，占总时间约 79.7%。不过普通 Region 的矩阵规模和 oracle 成本变化很大，单一样本不能代表所有普通 Region。

生成逻辑微测试，每个后端 40 次，均值如下，单位毫秒：

| 操作 | TileLang | Triton |
|---|---:|---:|
| 生成 IR | 2.746 | 2.270 |
| 变异 IR | 1.718 | 1.651 |
| 程序签名 | 0.667 | 0.763 |
| 反馈处理 | 1.432 | 1.510 |

这些 CPU 操作的量级远小于被测编译时间；详见 [generation.json](generation.json)。

另取正在运行的两个 campaign 在 20:27 的结果快照，按连续完成记录的时间间隔分组。TileLang 的 32 个有效间隔中，Extended 是 11 个，占约 **66.4%** 的墙钟时间；普通 Region 约 30.6%，probe 约 3.0%。Triton 的 Extended 约占 20.0%，普通 Region 约占 69.8%。所以就这个窗口而言，TileLang 的 Extended 长尾确实明显拖慢了整体推进速度。

快照不是同 IR 公平对比，也不是完整 campaign 的最终分布；它排除了首个记录和未完成尾部，同时包含生成、运行、保存等时间，并受到当时并行任务影响。原始统计见 [campaign_snapshot.json](campaign_snapshot.json)。

## 优化优先级

1. 对 TileLang 优先处理布局推导长尾：用 `mixed` 和 `control_calls` 固定复现，缩减 IR，继续测 `LayoutInference` 内部操作/约束传播。先确认昂贵结构，再决定调整 emitter 或上游编译器。
2. 评估默认四变体的预算。单变体实测可显著加快此类慢例，但要明确它减少覆盖。可以另行设计额外变体的抽样或分阶段验证，不能把减少检查伪装成无损优化。
3. 对较简单 TileLang 用例，进一步关注 NVCC 和重复进程/编译初始化。跨配置复用编译结果必须保证配置与布局兼容，不能直接共享不匹配的布局结果。
4. Triton 在被测 mixed 程序中，编译只占约 24%；新进程导入、启动和运行检查合计更大。若优化 Triton 吞吐，应分别评估进程复用及检查成本，而不是照搬 TileLang 的优先级。
5. 当前没有证据支持优先优化 Python 生成/变异，或者仅靠减少执行 repeat 解决 TileLang 的百秒级慢例。

## 文件与复现

实验脚本：[tests/profile_compilation.py](../../../tests/profile_compilation.py)。所有 instrumentation 只作用于实验 worker，不修改第三方库文件或正常 fuzzer 流程。

在项目根目录、CUDA 可见的环境中，输出目录须尚不存在：

```bash
python -B tests/profile_compilation.py \
  --program reports/2026.09.17/compilation_profile/corpus/mixed.json \
  --output reports/recheck_mixed --repeats 3 --warm

python -B tests/profile_compilation.py \
  --program reports/2026.09.17/compilation_profile/corpus/mixed.json \
  --output reports/recheck_mixed_single --single-variant
```

每个试验保留 `program.json`、`summary.json`；每个子进程保留 `repro.py`、`events.jsonl`、`timings.json`、`run.log` 和适用的编译产物。事件记录父子关系、inclusive 和 exclusive 耗时；结果保存 IR/源码哈希和成功状态。

[RESULTS.md](RESULTS.md) 为各阶段表，[phases.csv](phases.csv) 为逐次测量，[aggregates.json](aggregates.json) 为均值、中位数和范围。可运行 `python -B reports/2026.09.17/compilation_profile/analyze.py` 从原始结果重新汇总。

计时器的嵌套加和、异常清理、static method 包装，以及 runpy 函数作用域包装，共 4 个针对性测试已通过。
