# Triton / TileLang：论文 bug 覆盖核查

核查对象：本目录当前源码，包含本轮新增定向 probes。论文为 `../paper/Characterizing Real-World Bugs in Tile Programs for Automated Bug Detection.pdf`，重点对照第 4 节、图 2/3 及自动检测建议。

**结论：目前不能生成并检测论文中所有 Triton / TileLang bug。** 已覆盖若干触发结构；尚无按历史 issue、受影响编译器版本和 GPU 架构逐项执行的复现矩阵。论文的 301 个 bug 是八个框架的合计，不能作为这两个后端的覆盖率分母。

“覆盖”须区分三件事：生成器能否表达触发程序；当前编译器和硬件是否进入有问题的编译路径；oracle 能否观测该问题。下面的“部分覆盖”仅指第一项及部分第三项，不代表已复现历史 issue。

| 论文类别 / 具体问题 | 当前能力 | 主要缺口 |
|---|---|---|
| 分支谓词、尾块掩码；Triton #5265 的 4/8 warp 差异 | 原有动态分支，新增部分行/列、128/256 threads 配对 | 分支嵌套、复杂 loop guard、布局组合不完整；尚未逐项复现 #5265 |
| Warp control；Triton #2658 | 可改变普通 launch 的 num_warps | 没有显式 warp specialization / producer-consumer warpgroup 构造；普通 num_warps 配对不能替代 |
| 指令调度；Triton AMD #6750 | 串行和流水 K 循环可触发部分调度变换 | 没有 AMD local-prefetch 指令序列、ROCm 测试矩阵 |
| IR 构造；TileLang #313 | 能检查同一已编译内核重复执行是否一致 | 没有重复编译后 IR/layout_map 对比、缓存命中统计；IR 变化但结果正确时不会报告 |
| IR transformation | 算子链、分支、归约、融合会经过优化 pass | 不变异 pass 开关/顺序，不比较不同 pass 设置的结果或中间 IR |
| Tile mapping / launch | 现有二维 grid，加上定向一维 row grid、边界形状 | 没有 batch/head 多维映射、grouped launch、多轴 flatten、大 grid 和索引宽度系统测试 |
| 索引与 stride；Triton #443 | 连续、转置、双步长和 offset 输入，带尾块 | 不等于 dds_matmul 的稀疏间接索引；没有通用 gather/scatter、广播 stride=0 或高维布局 |
| 资源生命周期；TileLang #359 | 新增 T.Parallel global→shared、流水 GEMM，以及后续索引归约 | 没有论文中完整 scale 计算/producer warpgroup 寄存器回收路径；测试 GPU 非 Hopper |
| Ordering/cache；TileLang #1604 | 输入完整性、重复执行、两个线程配置配对 | 没有专门的 A→B→A 编译/调用序列、缓存键字段变异、缓存开关对照和显式同步变异 |
| 特殊浮点值 | copy 的 NaN/Inf/±0/次正规数逐位检查 | 未将这些输入扩展至所有算术、归约和融合算子；FTZ、NaN max/argmax 策略仍未系统建模 |
| 类型语义 | fp16/fp32 输入与 fp32 累加；新增 int32 argmax 输出 | 小整数数值输入仍是浮点 tensor；未覆盖 int8/int32 算术、bf16/fp8 混合运算和广泛 cast 链 |
| 算子实现；Triton #1846 | 显式 dot(A, trans(B))→argmax；并列取首个索引、尾块、int32 输出 | 能生成相关结构，但未在受影响旧版本上确认复现；其他复杂算子组合仍有限 |
| 架构及性能 | 当前 CUDA GPU 上检查结果、崩溃和超时 | 无 H100/AMD 等跨设备对照、MMA/WGMMA/TMA 选择矩阵；无可靠性能回归 oracle |

## 新增实现及边界

- [生成与变异](src/workflow/generator/probes.py)：7 种定向计算，尺寸包含 1、15/16/17、31/32/33、63/64/65、127/128/129；GEMM 参数按共享内存预算修复。默认全新生成有 20% 概率选择定向用例，原有动态序列和模板仍保留。
- [两个后端的代码生成](src/workflow/emitter/probes.py)：Triton 显式 transpose/dot/argmax；TileLang 通过 max 归约和最小匹配索引实现 argmax。TileLang 用一维物理 buffer 和显式步长表达式，并未测试前端任意 stride 描述符。
- [输入和执行 oracle](src/workflow/emitter/probe_runtime.py)：同一输入上比较 128/256 threads 两个配置，各默认执行 3 次；检查参考结果、重复结果、前后各 16 元素输出保护区和输入存储完整性。复制用例逐位比较；算术线程配置比较允许正常浮点误差。
- [IR](src/ir/ir.py) 和 [持久化](src/workflow/fuzzer/fuzzer.py)：保存布局、输入模式、重复次数、配对开关；纳入去重、文件名和恢复。新生成器不保证沿用旧版本完全相同的后续随机程序序列。

这些 oracle 只对定向 probes 默认启用，没有自动扩展到每个原有模板/动态程序。输出保护区只能检测写入保护区且可观测的破坏，无法保证发现越界读取、越过保护区的写入或所有数据竞争。重复执行相同已编译内核也不能证明重复编译 IR 稳定。

## 下一步优先级

1. 为论文的 Triton/TileLang issue 建独立回归清单：保存最小触发结构、受影响/修复版本、硬件前提、预期症状及执行结果，先建立可计量的覆盖率分母。
2. 给普通模板和动态链引入相同的输入布局/特殊值模型及配对 oracle，补足类型转换、算术特殊值和复杂操作组合。
3. 独立实现编译过程测试：重复编译 IR 对比、A→B→A 缓存调用序列、pass 配置配对。这些问题不能单靠多跑数值测试解决。
4. 实现 warp specialization、producer global→register→shared 计算、间接索引和多维 launch；在具备相应硬件时增加 Hopper/ROCm 用例。
5. 接入内存检查器与性能测量；区分编译错误、运行错误、数值差异、缓存回归和性能异常，避免把所有失败分类都当成已确认编译器 bug。

## 本轮验证

- CPU 回归：`python -B -m unittest discover -s tests -v`，29/29 通过；包括对未写输出、越界保护区、输入破坏、重复不一致及 schedule 不一致的故障注入测试。
- GPU：Triton/TileLang × fp16/fp32，首批 44/44 通过；补充宽尾块并重测复制和融合路径，24/24 通过（含重叠用例，共 48 个不同新增用例）。GPU 为 RTX 4060 Laptop；每个用例独立 TileLang 缓存。日志/复现脚本保留于 `/tmp/tilesmith_gpu_smoke__q8gr2wd` 和 `/tmp/tilesmith_gpu_smoke_uluwrw2c`。
- 两个后端的 `--probe-prob 1 --seed 42 --dump` 输出均通过 Python AST 解析；`git diff --check` 通过。
- 以上验证确认新增生成器/参考实现能在当前环境工作，不能用来推算论文历史 bug 覆盖率。
