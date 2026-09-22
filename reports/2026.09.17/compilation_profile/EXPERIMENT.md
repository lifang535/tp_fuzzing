# 编译阶段对比实验

实验脚本：`tests/profile_compilation.py`。同一份已保存 IR 分别交给 TileLang 和 Triton，均保留默认编译配置和观测变体。每个试验使用新 Python 子进程；冷编译只隔离 DSL 缓存，热缓存试验在新进程复用该次 DSL 缓存。CUDA 驱动缓存继承现有环境，不清空用户缓存。

计时使用 `perf_counter`。TileLang 使用 TVM pass instrument 和局部 Python 包装，Triton 使用实际注册后端的阶段包装。不会修改第三方库文件或 campaign 源码。事件同时记录 inclusive 和 exclusive 时间；累计分类时使用 exclusive，避免重复计数。GPU launch 包装后同步，耗时包含启动与同步，不能当作纯 device kernel latency。

环境：RTX 4060 Laptop GPU，sm_89，PyTorch 2.4.0+cu124，TileLang 0.1.11，Triton 3.0.0。用户的两个 fuzzer 同时继续运行；不停止它们，因而绝对耗时有并发负载噪声。

正式 mixed 用例：用户 20:03 运行中耗时约 115 秒的已通过程序，保存在 `corpus/mixed.json`。四个编译变体、2 个输入 seed、3 个运行边界组合、每组合重复 3 次，总共 72 次 kernel launch。正式试验重复 3 轮，两种后端交替顺序。

`mixed_initial/` 是探路试验，清空了独立 CUDA 驱动缓存且最初 Triton 阶段 hook 没有绑定到动态加载的实际后端类，因此不纳入正式统计。`arithmetic/` 是修正后 Triton hook 的验证试验。后续跨路线对照另存。

`native_control/` 最初选用 Triton 已通过的普通 Region，但 TileLang 拒绝其 warp 划分（32×16 tile、256 threads）。两次编译失败完整保留，不将提前失败时间当作成功编译时间比较。替换对照 `native_matched/` 使用 TileLang 已通过且 block_K >= 16 的原始程序，两边读取同一个文件，来源见 `native_matched_source.json`。

原始输出：每个 trial 的 `events.jsonl`、`timings.json`、`run.log`、独立 `repro.py` 和编译产物；上层 `summary.json` 保存源码哈希、同一 IR 哈希、成功状态和总时间。独立 DSL 缓存在试验结束时自动清理。

生成开销微测试 `generation.json`：每个后端 40 次生成、变异、签名和失败反馈观察，仅用于分离控制逻辑开销，不将它当作实际 campaign 的操作分布。

## 解释边界

- 缓存命中试验是同 IR 的重复重放，不代表持续生成新 IR 的 fuzzer 吞吐量。
- 当前 backend 运行保持原 emitter 的调度参数：TileLang 与 Triton 的线程/warp 默认值不同；比较的是相同语义程序在各自正常编译路线上的成本，不声称指令或 schedule 完全相同。
- 阶段耗时是仪器包装的墙钟时间。用户仍在进行两个 fuzz campaign，GPU 队列竞争会增加 launch+sync、torch 比较和总体耗时；以三轮中位数及范围报告。
- `tl.LayoutInference` 用 TVM pass 的 before/after hook 直接计时，不由文件 mtime 推测；它会为 fragment、并行操作等推导线程/寄存器布局并传播约束。名称类似的“输入 layout 选择”只是生成参数，不能混为一个阶段。
- `runtime.kernel` 包含 host launch、同步与可能的等待。普通 Region 的 Triton 首次 launch 内部还会触发编译；应查看 exclusive 值，不能将 inclusive launch 时间全部归为 GPU 计算。
- `generation.json` 的控制逻辑耗时是 CPU 微测试，不计 DSL import 或 GPU 执行。
