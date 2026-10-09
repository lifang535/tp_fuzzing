# TileSmith

TileSmith 为 TileLang 和 Triton 生成 GPU tile 程序，结合结构化生成、变异、PyTorch 参考计算和重复执行检查，发现编译崩溃、运行异常和结果不一致。

当前执行链只接受两种程序表示：`RegionProgram` 和 `ExtendedProgram`。定向 probe 是 Region 中的一种整函数形式。旧 `TileProgram`、`TilePipeline`、`DynamicSequence` 及其算子注册表、emitter 和兼容转发模块已移除。

[English](README.en.md) · [工作流说明](src/workflow/README.cn.md)

## 代码布局

```text
main.py                           命令行参数、后端加载、dump / campaign 入口
src/
  config/config.py                生成、变异、数值检查和资源限制
  ir/
    ir.py                         TileKernel：形状、dtype、调度和 probe 参数
    region.py                     Operation / Region / Function / RegionProgram
    region_ops.py                 普通及 typed 操作契约
    region_types.py               类型推导、作用域和值池
    extended.py                   Extended 的类型、节点、多值区域和验证
    slice.py                      特征切片程序（参数在 src/workflow/slices 中展开）
    layout.py                     输入布局及物理地址描述
    serialization.py              Region / Extended / 切片程序的保存和恢复
  backends/
    base.py                       后端接口
    common/                       共用调度策略、probe、程序分发和测试脚本组装
    tilelang/                     TileLang 参数约束及各类 IR 的 lowering
    triton/                       Triton 参数约束及各类 IR 的 lowering
  workflow/
    generator/                    原生模板实例化、Extended 生成
    mutator/                      变异入口及原生局部变异
    emitter/                      嵌入独立脚本的参考解释器和运行检查
    oracle/                       子进程执行、超时、诊断和编译证据
    fuzzer/                       主循环、种子池、去重、持久化和恢复
    slices/                       特征切片生成器、精确参考、失败约简与调度
    feedback.py                   Region/probe 的结构反馈
    extended_feedback.py          Extended 结构及编译特征
    coverage_audit.py              覆盖证据审计
tests/                            单元测试、离线编译及 GPU smoke
```

`TileKernel` 是 Region 附带的参数对象，不是旧的单算子程序容器。普通 Region 的计算逻辑由 `body` 和 `functions` 中的操作定义。

## 运行方式

在项目目录运行。本分支针对 **TileLang 0.1.14**、**Triton 3.8.0** 和 **PyTorch 2.4.0+cu124**；两台已审计服务器的 `tp_fuzzing_latest` 环境使用 Python 3.11。仓库中的 [requirements.txt](requirements.txt) 仍固定*旧版* TileLang 0.1.11 / Triton 3.0.0，不能原样用于复现本分支实验。安装带 `+cu124` 的 PyTorch wheel 还需使用对应的 PyTorch CUDA wheel 索引，不能只依赖普通 PyPI 镜像。运行前检查当前环境：

```bash
python -c "import sys; from importlib.metadata import version; print(sys.version.split()[0], {name: version(name) for name in ('torch', 'tilelang', 'triton')})"
```

Fuzzer 会把实际安装版本记录在 campaign 的 `summary.json`，恢复时若环境发生变化也会提示。生成源码和 CPU 单元测试不要求可用的 GPU；执行 kernel 需要对应 DSL 和 CUDA 环境。

```bash
# 查看参数 / 原生操作契约
python main.py --help
python main.py --list-kernels

# 仅输出一个程序的独立 Python 脚本
python main.py --backend triton --seed 42 --dump

# 两种后端分别启动 campaign
python main.py --backend tilelang --seed 42 -n 100 -o results
python main.py --backend triton --seed 42 -n 100 -o results

# 仅普通 Region / 仅 probe / 仅 Extended
python main.py --extended-prob 0 --probe-prob 0 -n 100
python main.py --extended-prob 0 --probe-prob 1 -n 100
python main.py --extended-prob 1 -n 100

# 仅编译 Extended，不执行 GPU kernel
python main.py --backend triton --compile-only -n 10

# 恢复当前格式的 campaign；沿用原 backend、seed、形状模式及生成参数
python main.py --backend triton --seed 42 --resume results/<campaign目录> -n 100

# 不长期保存 Extended 编译证据；结果记录仍会保存
python main.py --backend triton --seed 42 -n 100 --no-save-artifacts
```

`-n` 是本次新增执行数量，去重跳过的候选不计数。`--input-seed` 控制测试输入；`--seed` 控制生成与变异。`--easy-shape` 使用 2 的幂尺寸，小于 tile 的尺寸仍然需要边界掩码。

内置两种后端共用 IR 和 campaign 主循环，TileLang 与 Triton 的可用配置、lowering、启动方式和诊断规则由 `src/backends/` 中各自的实现决定。扩展其他 DSL 需实现并注册 `src/backends/base.py` 中的接口，再以 `--backend-plugin MODULE` 加载注册模块；参见[工作流说明](src/workflow/README.cn.md)。如果还要加入新的 IR 格式，则需同时扩展生成、序列化和反馈。

## 路线与概率

种子池为空时只能全新生成；非空时默认 **50% 变异、50% 全新生成**，由 `Config.mutate_prob` 控制。

全新生成先按 `extended_prob` 选择 Extended；剩余候选进入 Region 路线，再按 `coverage_probe_prob` 选择 probe。新 CLI campaign 的默认值分别为 0.25 和 0.20，因此全新生成候选的期望比例为：25% Extended、15% probe、60% 普通 Region。去重和失败会改变最终执行、通过样本的比例。

库 `Config()` 的 `extended_prob` 默认为 0。CLI 恢复时读取 summary 中保存的 Extended 概率，缺失时使用 0。

| 参数 | 默认值 | 作用 |
|---|---:|---|
| `--backend` | `tilelang` | `tilelang` / `triton` / 已注册后端 |
| `--extended-prob` | 新 CLI：0.25 | 全新生成时选择 Extended 的概率 |
| `--probe-prob` | 0.20 | Region 路线内选择 probe 的条件概率 |
| `--gemm-prob` | 0.50 | 普通 Region 从 GEMM 开始，否则从 load 开始 |
| `--typed-op-prob` | 0.35 | 原生类型、形状和内存操作的生成概率；0 使用 v3 |
| `--function-min-count` / `--function-max-count` | 1 / 3 | 原生辅助函数数量 |
| `--function-call-prob` | 0.30 | 可用 callee 存在时选择调用的概率 |
| `--dtype-mutate-prob` | 0.25 | 变异时显式切换存储 dtype |
| `--local-mutate-prob` | 0.35 | 未选择 dtype 变异后的局部变异概率 |
| `--region-input-seeds` | 2 | 普通 Region 的输入组数 |
| `--region-repeat-count` | 3 | 每组输入、每种调度的重复执行次数 |
| `--region-layout-prob` | 0.35 | 每个使用中的输入采用非连续布局的概率 |
| `--no-region-schedule-pair` | 关闭 | 指定后禁用普通 Region 的调度配对 |
| `--no-region-stage-sweep` | 关闭 | 指定后禁用普通 Region 的 num_stages 扫掠 |
| `--no-region-loop-sweep` | 关闭 | 指定后禁用普通 Region 的 loop_kind 扫掠 |
| `--no-region-layout-sweep` | 关闭 | 指定后禁用普通 Region 的备用布局对扫掠 |
| `--extended-config-depth` | 1 | Extended 编译配置扫掠深度：0 = 单配置；1 = 线程/stages 配对；2 = 追加第二 pass 配置 |
| `--no-extended-configurations` | 关闭 | `--extended-config-depth 0` 的别名 |
| `--extended-fast-math` | 关闭 | 追加 `tl.enable_fast_math` 编译配置对（改变数值） |
| `--no-extended-precision` | 关闭 | 指定后禁用累加器宽度扫掠（fp16 累加副本 + triton ieee→tf32） |
| `--no-extended-identities` | 关闭 | 指定后禁用代数恒等式扫掠（分配律副本） |
| `--random-config-count` | 2 | 每个 Extended 程序额外采样的随机 pass 管线配置数（按程序+seed 确定性采样；0 关闭） |
| `--extended-atomic-prob` | 0.25 | Extended 程序中全局内存原子 add/max/min 的生成概率 |
| `--extended-elementwise-prob` | 0.50 | 在 Extended 程序中组合一个通用 float32 数学操作；`elementwise` 骨架固定选择三个 |
| `--extended-fma-prob` | 0.30 | Extended 程序中标量 FMA 链的生成概率 |
| `--extended-shape-op-prob` | 0.30 | 共用 Extended 种子中 `flip` 的生成概率；Triton 专属形状操作在单独的 extend 阶段生成 |
| `--extended-int8-prob` | 0.30 | Extended matmul 使用 int8 × int8（int32 累加）的概率 |
| `--region-int8-prob` | 0.15 | 普通 Region 生成 int8 GEMM-only 程序的概率（规格取自预校验网格） |
| `--no-region-pass-config` | 关闭 | 指定后禁用 Region 的 pass-config 不变性配对 |
| `--no-region-swizzle` | 关闭 | 指定后禁用 tilelang `T.use_swizzle` Region 变体配对 |
| `--no-region-warp-policy` | 关闭 | 指定后禁用 tilelang `GemmWarpPolicy`（FullRow/FullCol）Region 变体配对 |
| `--no-instance-grids` | 关闭 | 指定后禁用新 op 面的 per-(op, backend) 轮转实例网格 |
| `--uncovered-boost` | 50.0 | 从未尝试的结构特征的权重加成（MLIRSmith 式多样性优先；0 恢复旧权重） |
| `--no-structural-feedback` | 关闭 | 指定后禁用结构反馈引导 |
| `--compile-only` | 关闭 | 指定后仅编译，并强制 Extended 概率为 1 |
| `--slice-prob` | 新 CLI：0.4 | 每个测试为特征切片程序的概率（恢复无切片的旧 campaign 时为 0） |
| `--slices` | 全部 | 逗号分隔的切片子集：`cast,reduce,scan,gemm,atomic` |
| `--no-slice-adaptive` | 关闭 | 切片与参数均匀采样，不按发现估计和未覆盖参数对选择 |

更多参数通过 `--help` 查看；未开放 CLI 的配置在 `src/config/config.py`，包括尺寸池、模板深度、操作数量、scratch 预算及数值阈值。

## 模板如何生成

普通 Region 先调用 `program_template()` 生成辅助函数和入口的操作树。模板保存操作种类、if/for 子区域、调用目标和接口信息，此时尚未为每个操作绑定具体 SSA 值。`instantiate_program()` 随后按函数顺序实例化：为操作选择当前作用域中类型合适的值，填写属性，分配结果名，再采样形状和合法调度。后面的函数只调用前面定义的函数，从而避免递归环。

v3 使用完整 fp32 tile；v4 在此基础上加入 fp16/fp32、tile/row/column/scalar 形状、tensor/buffer 区分和 scratch 读写。验证器检查作用域、调用签名、区域返回值、类型和形状。

probe 使用整函数 `probe` 节点，其参数选择 copy、reduce_sum/max/min、softmax、argmax 或 GEMM+argmax。它专门测试物理步长、偏移、广播、尾部掩码、特殊数值、重复执行和缓存复用，使用专用参考检查。生成、变异、保存和恢复全部采用 `RegionProgram`。

Extended 使用单独的 IR 和生成器，提供 `arithmetic`、`indexed_memory`、`shape_matmul`、`control_calls`、`elementwise`、`mixed` 六类程序骨架；在各骨架中实例化带类型的操作、操作数和属性。`elementwise` 骨架按轮次覆盖两种后端共用的 18 种 float32 数学操作。它支持显式内部 matmul、索引访存、多值控制流/函数接口，以及中间结果观测和编译配置配对。这里覆盖的是有界的操作族，不代表 TileLang/Triton 的全部 API、dtype、形状及硬件特性。它不是旧 `DynamicSequence` 的改名。

### 共用种子与 DSL 专属 extend 阶段

新 campaign 的 Extended 生成默认只产生两种 DSL 共用的 IR 结构：Triton 专属的 `join/split/interleave` 和流水化 `for` 不再混入共用种子；两后端的 `mixed` 骨架也采用相同的结构。`--legacy-extended-mix` 可恢复旧生成行为，以便继续原先的实验设置。这里的“共用”指同一 IR 的语义和结构，不表示两种编译器生成相同代码。

现在 `main.py` 可在同一实验中交替执行两条路线。新实验的 `--dsl-extend-prob` 默认为 0.35：存在执行通过的共用 Extended 样例后，每次生成有 35% 的概率从有界种子池派生目标 DSL 操作；其余仍走原本的 Region/Extended 生成路线（该路线的 `--extended-prob` 默认 0.25）。扩展池为空或“样例×操作”组合耗尽时继续生成共用程序。首次探索优先选择尚未尝试的目标操作，同一输入样例与操作的派生次数由 `--dsl-source-variants` 限制；开启自适应调度后，已尝试操作根据近期收益分配预算，否则继续优先选择尝试次数最少的操作；在可选父样例间优先考虑实际编译产物中的稀有阶段、相邻 IR 操作与编译配置／操作组合。DSL 专属编译特征保存在独立的 `dsl_stage.json` 账本，不影响共用 IR 生成反馈。某个父样例反复触发人工确认过的具体失败签名时，其选择权重逐步降低但不会归零；未知签名和同目录的其它错误不受该惩罚，失败样例仍照常保存。`--no-structural-feedback` 可关闭这种种子引导。这些编译产物特征只是代理指标，并非实测编译 pass 或分支覆盖；Region 程序目前仅有结构反馈。`summary.json` 记录 `dsl_extension.by_op` 和 `confirmed_failure_signatures`；`dsl_stage.json` 支持恢复，恢复的共用样例复用前会重新验证。指定 `--dsl-extend-prob 0` 可关闭一体化阶段。

通过测试的 DSL 扩展程序现在会进入独立的目标种子池。在 DSL 路线内，`--dsl-evolve-prob`（默认 0.5）选择继续组合目标操作、保留目标操作的局部变异，或将保持形状的目标操作放入有界循环。组合会使用前一步被检查的目标输出。`--dsl-max-depth` 默认 3；到达上限的程序仍保存，但不会挤占可继续变异的祖先种子。恢复的目标种子复用前重新验证。结果记录直接父程序与初始父程序的哈希、派生深度、动作和目标输出。`--dsl-evolve-prob 0` 恢复单步扩展。

`--corpus-feedback` 默认开启，在共用和目标种子池满时优先淘汰覆盖冗余的种子，保留稀有特征代表；`--no-corpus-feedback` 恢复随机淘汰。旧 CLI campaign 恢复时默认不启用这两项新策略，除非已保存设置或显式参数启用。原生和 DSL 扩展程序共用隔离准入与学习策略，保留抽样探索和最后一次强制重试。数值错误的诊断桶不会用于隔离或结构特征降权，oracle 不稳定结果也不会建立隔离规则。`coverage_progress.json` 和 `coverage_progress.jsonl` 每 100 个测试及退出时记录进度；这些是结构／编译产物特征计数，不是编译器分支覆盖率或独立 bug 数。

`--dsl-adaptive-schedule` 在新 CLI 实验中默认开启，旧实验恢复时保留原设置；要启用它，在原恢复命令中追加 `--dsl-adaptive-schedule`。`--no-dsl-adaptive-schedule` 关闭动作自适应，`--no-structural-feedback` 同时关闭种子与动作引导。调度按动作、操作和直接父种子记录近期新增的通过程序结构特征、实际编译产物特征、测试耗时及人工确认签名的重复命中。奖励和重复率使用移动平均，耗时修正有上下界，均匀探索概率从 10% 随无收益测试数线性增加，连续 256 次无收益时达到 50%，出现有效新颖性后恢复 10%。已确认签名的重复失败即使带来新的编译产物词法特征也不获得奖励；未知诊断和数值错误不按重复已知缺陷惩罚。共用种子和目标种子均应用已有的已确认失败降权，失败保存策略保持不变。

`--dsl-source-variants` 在新 CLI 实验中默认为 4，旧实验恢复时默认为 1；它为同一通过的共用父程序保留有限的重试机会，即使第一个派生程序失败，仍可探索不同轴、方向或形状。只对可能变化的操作重复派生，确定性的操作仍只尝试一次。开启 `--dsl-attributes` 和 instance grids 时，属性组合按操作、输入类型与合法属性域轮换；`--no-instance-grids` 保留随机属性采样。目标属性变异会排除原属性，并保持结果类型与 top-k 大小，避免返回未改变的程序。已测试的完整程序在 DSL 生成阶段即被过滤，重试次数有界，耗尽后回到共用生成；`duplicate_derivatives` 单独计数，不计作新测试或新缺陷。变体尝试次数与属性游标均支持断点恢复。

DSL 编译账本现在记录失败前实际获得的中间产物；`compiler_passed` 单独记录最终执行通过的证据，失败程序不会进入通过种子池。首次通过的编译特征仍可保留种子，即使此前曾在失败程序中出现。`dsl_stage.json` 保存父种子与动作反馈并兼容旧状态；`summary.json` 的 `dsl_extension.schedule` 和进度文件的 `dsl_schedule` 记录 `tested`、`novel`、`known_repeats` 及近期收益/耗时；调度快照同时记录 `stagnant` 和当前 `exploration`，支持平台期断点恢复。`novel` 表示获得上述代理特征的测试数，不能作为独立 bug 数；按动作及“动作:操作”同时聚合的计数不能相加，使用 `total` 查看总数。`seconds` 是 oracle 测试耗时的移动平均，不包含生成和父种子重新验证耗时。

短时 GPU 对照可运行 `python tests/dsl_evolution_comparison.py --backend triton --passed-dir results/<campaign>/passed --output /tmp/dsl-comparison`。两组使用相同的已重新验证父程序和编译／oracle 设置，并在不同 seed 间交替执行顺序。默认短测不包含 precision、identity 和随机编译配置扫描，指定 `--full-oracle` 可启用；输出保留特征集合、失败与派生关系。增加 `--comparison schedule` 可在相同演化概率、种子保留规则和 oracle 设置下，仅比较固定与自适应调度；默认 `evolution` 对照的两组均关闭动作自适应。短测按测试数运行，判断 bug 收益仍需多 seed、固定硬件与时间预算的实验。

```bash
# 第一阶段：生成、实例化并执行共用 Extended IR；仅通过的样例进入 passed/
python main.py --backend triton --extended-prob 1 -n 10000
# 一体化运行：持续生成共用 IR，并按概率从通过样例派生目标 DSL 操作
python main.py --backend triton --extended-prob 0.5 --dsl-extend-prob 0.4 -n 10000
# 也可以单独对既有 passed/ 结果运行 extend
# 第二阶段：读取第一阶段的 passed/*.json，定向测试目标 DSL 的操作
python extend.py --backend triton --passed-dir results/<campaign>/passed -n 10000
python extend.py --backend tilelang --passed-dir results/<campaign>/passed -n 10000
# 可用 --op 指定下表中的一项操作
```

extend 的输入是**已实例化且执行通过的 ExtendedProgram JSON**，不是 IR 模板，也不读取仅编译通过的 `compiled/` 或普通 Region 样例。程序在目标环境、当前输入 seed 下复跑成功后，才注入目标操作并执行派生程序。输出是独立目录中的 `passed/`、`failed/`、`summary.json`；每个派生 JSON 记录原样例路径、IR 哈希、操作和输入 seed，供复现与归因。脚本每轮更换输入 seed，可用 `-n` 长时间运行；默认从通过样例中等概率保留最多 5000 个“样例×操作”组合，避免读取大型语料时占满内存（`--max-sources 0` 不限制）。失败实例全部保存，通过实例默认每种操作仅保存 20 个（`--max-passed-saved 0` 全部保存），完整执行计数写入 `summary.json`。第一阶段和第二阶段的失败数应分开统计；派生失败仍需人工核验，不能直接等同于 DSL bug。

| 后端 | extend 阶段的 `--op` | 检查内容 |
| --- | --- | --- |
| Triton | `join`、`split`、`interleave`、`scan_sum`、`scan_product`、`sort`、`histogram`、`argmax`、`argmin`、`xor_sum`、`dsl_sigmoid`、`dsl_clamp`、`softmax`、`topk`、`gather`、`atomic_and`、`atomic_or`、`atomic_xor` | 形状变换、扫描/排序、整数归约、逐元素数学运算、按位原子更新；`topk`/`gather` 仅在安装版本导出对应 API 时启用 |
| TileLang | `pipelined_for`、`scan_sum`、`scan_max`、`reduce_abssum`、`reduce_absmax`、`reduce_bitand`、`reduce_bitor`、`reduce_bitxor`、`dsl_sigmoid`、`dsl_clamp` | 流水化循环、扫描、归约、逐元素数学运算 |

数值扩展保留父样例的原有检查输出，额外返回 DSL 专属操作结果。浮点输入由父结果导出并限幅；直方图和按位归约优先使用父程序的精确整数值，否则使用依赖运行时参数的整数索引，初始直方图输入限制在 0–15；变异后允许越界，参考解释器按 Triton 语义忽略负数及大于等于 16 的值。二维 Triton softmax 沿行归约时强制保留归约维度，以正确广播；旧保存程序在重新发射时也会修正。这样避免把父程序容差内的浮点误差放大成离散结果误报。表中仍是有界子集，不能声称覆盖全部 API；异步/TMA 等硬件专属指令、自定义扫描组合函数及任意 dtype/形状组合尚未纳入。

用 `python api_coverage.py --backend triton --summary results/<扩展实验>/summary.json --passed-code-dir results/<共用实验>/passed --output triton_api.json`（TileLang 将后端改为 `tilelang`）审计当前 Conda 环境导出的 CUDA 语言 API。清单逐项记录安装版本与状态：`executed_pass` 表示扩展实验至少有一个通过样例；`seen_in_passing_code` 表示保存的通过样例代码中有该 DSL 的直接调用；`implemented_no_run` 表示有生成代码但缺少实验通过证据；`unmapped` 表示清单尚无证据，不代表 fuzzer 其它阶段从未调用。单一样例也不代表覆盖该 API 的全部参数、数据类型与硬件路径。应在目标版本 Triton 3.8、TileLang 0.1.14 的环境分别生成清单。

MLIRSmith 式 op 面扩张在两层 IR 之上新增编译器代码路径。Extended 程序加入全局内存原子操作（在刻意竞态的地址上做可交换 add/max/min）、带数据依赖操作数的标量 FMA 链、triton 形状原语（flip/interleave/join/split）以及 int32 累加的 int8 × int8 matmul。普通 Region 加入超越函数元素操作（tanh/erf/log/log2/exp2/rsqrt/sin/cos/floor/ceil）和 int8 GEMM-only 程序（规格来自预校验网格：block_K ∈ {32, 64}、int32 累加、精确整数参考）。fp32 GEMM 永不与边界台阶 op（ceil/floor/round/cast）组合：TF32 张量核计算相对精确 fp32 参考会翻动取整边界，使 oracle 淹没在无法与 bug 区分的噪声里。

新 op 的属性取自**有界实例网格**（`src/workflow/generator/grids.py`）：per-(op, backend) 轮转游标保证每轮扫掠每个角落实例恰好出现一次（MLIRSmith 穷举实例思想在新 op 面上的针对性版本；老 op 保持随机采样）。网格游标随 campaign 的 rng 状态持久化；`--no-instance-grids` 恢复纯随机采样。

## 特征切片

运行数周后，Region/Extended 路线反复命中少数几种机制（Chao1 ≈ 已观测桶数）：调度只能重排生成器能表达的程序，而两种 IR 只覆盖 fp16/fp32/int8/int32/bool、秩不超过 2 和内置归约组合函数。参照特征聚焦测试生成（FFTG，Zamudio Amaya 等，ASE'26）与 tile 程序 bug 实证研究（Rathnasuriya 等，ISSTA'26：301 个代码生成 bug 中类型与算子处理占 49%），`src/workflow/slices/` 为每个 bug 密集特征提供聚焦生成器；每个切片有离散参数空间，其他维度保持简单：

| 切片 | 聚焦 | 参数（节选） |
|---|---|---|
| `cast` | 转换/逐元素链 | 三步（操作，dtype），覆盖 bf16、f16、f32、f64、fp8 e4m3/e5m2、i8–i64、u8–u32；秩 1–3；不整齐尾部；动态尺寸 |
| `reduce` | 归约 | Triton：sum/max/min/argmax/argmin/xor_sum 及自定义 `tl.reduce` 组合（含 (值, 下标)、(min, max) 元组）；TileLang：`T.reduce_*` 的 clear/batch/nan_propagate/共享内存源 |
| `scan` | 扫描 | cumsum/cumprod、自定义 `associative_scan`（含不可交换的线性递推）、reverse；`T.cumsum/T.cummax` 原地或写入另一缓冲 |
| `gemm` | 矩阵乘 | MMA dtype（f16、bf16、fp8、i8，f32 的 ieee/tf32/tf32x3）、累加器、转置、批量（三维）`tl.dot`、K 循环与 stages；TileLang 寄存器操作数、k_pack、warp policy、clear_accum、串行/流水循环 |
| `atomic` | 全局原子 | dtype × add/max/min/and/or/xor/xchg（Triton 的 sem/scope）或 addx2/addx4（TileLang）× 槽位竞争 × 掩码 |

输入是小整数分子的二进分数 k·2^-f，每一步传递静态取值域；结果不能精确表示的参数值在合法化阶段被替换（YARPGen 式范围追踪）。因此 float64/int64 参考是精确的，输出逐位比较，不一致不可能来自舍入噪声。每个测试还检查带保护区的输出、重复执行确定性和第二组启动配置。安装版本不支持的特征在前端被拒绝时会打标记并归为 `unsupported_feature`，不计为失败。

调度器（`slices/scheduler.py`）在轮转预热后，按每秒新失败桶与新参数对的 incidence Good-Turing 估计选择切片（STADS，Böhme TOSEM'18）；切片内从若干合法候选中选覆盖最多未覆盖参数对的一个（AETG 式两两覆盖），或对先前有意义的程序变异 1–2 个参数。失败会逐个参数约简到核心（`slices/minimize.py`，每次最多 24 个额外测试，总量不超过切片测试的三分之一）：错误结果按约简后的 dtype 路径分桶（错误值共用检查器消息），之后包含高频桶核心的候选以 1 − max(0.02, 3/命中数) 的概率跳过。约简后的程序与失败一起保存为 `*.min.py`，JSON 记录核心。`summary.json` 的 `slices` 项按切片记录测试数、桶、核心、Good-Turing/Chao1 与跳过数；`slice_state.json` 用于恢复。失败键现在保留失败的 MLIR pass 及其首个诊断、首个 nvcc 错误，`PassManager::run failed` 与 TileLang CUDA 编译失败按机制分开。

## 多样性机制与 oracle 维度

参考 MLIRSmith 的"一程序 × 多配置复用、未覆盖特征优先、细粒度定位"，每个程序除基础管线外还执行若干配对检查。同一参考解释器与调度无关（`_region_reference` 是纯 IR 解释器），因此调度类扫掠共享一个参考；改变数值语义的扫掠（累加器宽度、代数恒等式）改为对变换后的程序副本分别计算期望值。

| 机制 | 默认 | 检查内容 | 失败标签（root_cause） |
|---|---|---|---|
| 未覆盖特征优先 | 开（`--uncovered-boost 50`） | 从未尝试的结构特征获得一次性权重加成 | — |
| Region 调度扫掠 | 开（`--no-region-stage-sweep` / `--no-region-loop-sweep` 关闭） | 线程配对之外的合法 num_stages、loop_kind 变体共享同一参考 | `stage_mismatch` / `loop_kind_mismatch` |
| Region 布局扫掠 | 开（`--no-region-layout-sweep` 关闭） | 非连续布局程序另跑一组备用布局对；每个布局对编译自己的 kernel 集（布局常量烘焙在 kernel 源码里） | `layout_mismatch` |
| Region pass-config 配对 | 开（`--no-region-pass-config` 关闭） | Region kernel 另经 `@tilelang.jit(pass_configs=...)` 以数值中性的 region 池（knobs.py）确定性采样子集编译；triton 以 `enable_fp_fusion=True` 启动。int8 region 池排除 `tirx.disable_vectorize`（去向量化的 int8 cp_async 拷贝会被 tilelang codegen 直接拒绝） | `pass_config_mismatch` |
| Region swizzle 配对 | 开（`--no-region-swizzle` 关闭） | tilelang gemm 变体另用 `T.use_swizzle(panel_size=10)` 标注 kernel；triton 无此旋钮，不生成该配对 | `swizzle_mismatch` |
| Region warp-policy 配对 | 开（`--no-region-warp-policy` 关闭） | tilelang gemm 另以 `GemmWarpPolicy.FullRow`/`FullCol` 变体编译（warp 沿 M/N 全分配，逐 tile 数学不变，共享同一参考）；按 warp 划分可行性逐 policy 过滤，仅 tilelang 生成 | `warp_policy_mismatch` |
| Extended pass 配置扫掠 | 深度 1（`--extended-config-depth`） | 线程/stages 配对（深度 1）+ 第二 pass 配置 `tl.disable_loop_unswitching` / `enable_fp_fusion`（深度 2） | `configuration_mismatch`（跨变体一致性） |
| 随机 pass 管线采样 | 2 配置/程序（`--random-config-count`，0 关闭） | 每个 Extended 程序额外编译确定性随机配置：tilelang 从 14 个已核实的语义保持 `pass_configs` 开关随机取子集（可另加数值旋钮、随机线程/stages），triton 随机取 `num_warps`/`num_stages`/`maxnreg`；plain 变体自动与基线对比 | `configuration_mismatch`（跨变体一致性） |
| 累加器宽度扫掠 | 开（`--no-extended-precision` 关闭） | 每个基础配置的 fp16 累加副本（tilelang `T.gemm` fp16 fragment、triton `tl.dot` fp16 累加），解释器按每 k=16 的 MMA 舍入建模；triton 另加 ieee→tf32 输入精度变体 | `precision_mismatch` |
| 代数恒等式扫掠 | 开（`--no-extended-identities` 关闭） | 无 matmul 程序中 `mul(x, add/sub(y, z))` 改写为分配律形式，按变换后程序单独计算期望 | `algebraic_identity` |

tilelang 的 `opt_level` 无法穿透 `tilelang.compile`（所有 s_tir pass 声明 `opt_level=0`），因此 RC2 pass 管线差异用已核实的 `pass_configs` 键实现；`tl.enable_fast_math` 会改变数值，默认关闭。随机采样池（`src/backends/common/knobs.py`）只含针对目标 TileLang 0.1.14 重新核实过消费者的键（排除竞态、去掉安全合法化、Hopper-only 和 debug 键），且采样是（程序签名, seed）的纯函数——证据读取和超时缩放会重新推导同一变体列表。

每个失败报告的定位信息保存在 summary.json 的新键 `root_cause_locations`（`root_cause → 位置 → 次数`）中：位置依次取不变性标签本身、崩溃前最后一个 `TILESMITH_STAGE` 标记、TVM pass 名或报错源文件。`root_causes` 键保持 `{str: int}` 形状不变，`failed/` 目录命名不变。

详细调用关系见 [工作流说明](src/workflow/README.cn.md)。

## 检查、结果与恢复

生成的 `.py` 包含 kernel、输入初始化、参考计算和检查，可独立运行。普通 Region 检查数值、重复执行、调度配对（线程、num_stages、loop_kind）、布局配对、输入完整性和输出保护区；typed Region 还检查 scratch。Extended 额外保存编译阶段证据，并检查中间结果观测、pass 配置变化、随机采样管线、累加器宽度、代数恒等式及 scratch 内容。

结构反馈记录操作、数据依赖、嵌套、类型、布局和调度等特征。尝试、成功执行和编译特征分别统计；它不是编译器分支覆盖率。错误分类用于分组，`wrong_result` 仍需排查数值容差和参考语义，不能直接当作已确认的编译器 bug。

**如何判读失败：**`failed/<root_cause>/` 是自动症状标签，不是已确认的 DSL 缺陷。统计独立 bug 前，应在记录的环境中重跑代表性的 `.py`，检查相邻 `.json`，并把生成的 kernel 与参考实现、检查器分离验证。即使可重复出现 `WRONG RESULT`，也可能是 oracle 或容差错误：已有一例确认的 `atomic_mismatch` 误报，是把原子操作专用的严格比较错误地应用到普通 store 写入的缓冲区。磁盘耗尽、资源限制及不支持的配置也需单列。[9 月 26 日审计](reports/2026.09.26/REPORT.md)仅针对当时的第四轮实验，保守确认 **4 类缺陷机制**（TileLang 3 类、Triton 1 类）；534 条同签名保存记录不是 534 个独立 bug。这个日期明确的审计结论不能自动用于后续 campaign。

```text
results/<时间>_<backend>_<形状模式>_seed=<seed>/
  passed/                         成功执行的 .json IR 和 .py（--no-save-passed-code 时只保存 .json）
  compiled/                       compile-only 结果，不能等同成功执行
  failed/<root_cause>/             失败报告和独立复现脚本
  artifacts/                      Extended 编译证据（--no-save-artifacts 时不保留）
  summary.json                    计数与生成配置（含 root_cause_locations 定位统计）
  structural_feedback.json        反馈计数
  seed_pool.json                  可变异的程序种子
  dim_pool.json                   尺寸池
  rng_state.json                  随机状态
  pending_program.pkl             中断时尚未完成的程序（若有）
```

默认保存 `artifacts/`。添加 `--no-save-artifacts` 可关闭证据目录的持久保存，例如：
`python main.py --backend triton -n 100 --no-save-artifacts`。
测试仍使用临时文件完成编译证据检查和特征提取，每个测试结束后清理（包括失败和超时）；
`passed/`、`failed/`、`compiled/` 等结果照常保存。该参数也可用于 `--resume`，不会删除已有的 `artifacts/`。

通过样例的 `.py` 复现脚本约占 `passed/` 磁盘用量的 80%，且 fuzzer 从不读回（恢复和 DSL 演化只读 `.json`）。长时间运行时可添加 `--no-save-passed-code`，只保存通过样例的 `.json` IR；失败样例始终同时保存 `.json` 和 `.py`。该参数同样可用于 `--resume`，不会删除已有的 `.py`。

原生文件名使用调用关系和完整 IR 哈希，Extended 文件名使用 family 和哈希。当前 Region v1–v4 与 Extended 的 JSON 可以恢复；此前原生记录中无效的 `legacy: null` 和 spec `alpha` 字段会被忽略。非空 legacy 包装及旧 single_op/pipeline/dynamic 格式不再支持恢复，应新建 campaign。已有 `results/`、`reports/` 和独立复现脚本不受此次源码清理影响。

本次清理还删除了会被全部覆盖的旧参数预采样，因此同一 seed 在清理前后的后续随机程序序列不保证一致；保存的原生 IR 仍可重放。旧 `pipeline_rtol_fp16/fp32` 配置名改为 `region_rtol_fp16/fp32`，阈值不变。GEMM 的 `LoopKind.PIPELINED`、`num_stages` 和 `pipeline_stages_choices` 是仍在使用的硬件调度参数，保留。

## 验证

```bash
python -B -m unittest discover -s tests -v
python -B tests/offline_typed_smoke.py
python -B tests/gpu_smoke.py
python -B tests/extended_smoke.py --seeds 1
python -B tests/region_int8_smoke.py           # 双后端 int8 GEMM oracle 实跑
python -B tests/region_int8_smoke.py --compile-only
```

单元测试覆盖参考解释器、类型/作用域、布局、变异、结构反馈、持久化、后端派发和故障注入。`tests/fixtures/current_programs.json` 保存清理前的 24 个原生/probe/Extended 程序及生成函数 AST 摘要，用于防止清理改变已有程序的执行代码。

离线 typed smoke 编译 Triton PTX/cubin 并进行 TileLang lowering/CUDA 源码生成，不等同于 GPU 执行；后三个 smoke 需要可用 CUDA。

同一 IR 的编译阶段对比（需要 CUDA）：

```bash
python -B tests/profile_compilation.py --program <程序.json> \
  --output reports/<新的实验目录> --repeats 3 --warm
```

两种后端串行运行相同程序；冷编译使用独立 DSL 缓存，`--warm` 在新进程复用缓存。脚本记录 TileLang 各个 TIR pass/NVCC、Triton 各编译阶段、导入、参考计算和运行检查。`--single-variant` 可测单个编译版本。计时只在实验子进程中插入，不修改正常 fuzzer。阶段存在嵌套，汇总时使用 `exclusive_seconds`，避免重复计数。

本机实测结果与瓶颈分析见 [编译耗时实验报告](reports/2026.09.17/compilation_profile/REPORT.md)。
