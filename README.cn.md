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
    layout.py                     输入布局及物理地址描述
    serialization.py              Region / Extended 的保存和恢复
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
    feedback.py                   Region/probe 的结构反馈
    extended_feedback.py          Extended 结构及编译特征
    coverage_audit.py              覆盖证据审计
tests/                            单元测试、离线编译及 GPU smoke
```

`TileKernel` 是 Region 附带的参数对象，不是旧的单算子程序容器。普通 Region 的计算逻辑由 `body` 和 `functions` 中的操作定义。

## 运行方式

在项目目录运行；依赖版本见 [requirements.txt](requirements.txt)。生成源码和 CPU 单元测试不要求可用的 GPU，执行生成的 kernel 需要对应 DSL 和 CUDA 环境。

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
```

`-n` 是本次新增执行数量，去重跳过的候选不计数。`--input-seed` 控制测试输入；`--seed` 控制生成与变异。`--easy-shape` 使用 2 的幂尺寸，小于 tile 的尺寸仍然需要边界掩码。

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
| `--extended-fma-prob` | 0.30 | Extended 程序中标量 FMA 链的生成概率 |
| `--extended-shape-op-prob` | 0.30 | Extended 程序中形状 op 的生成概率（flip 双 DSL 支持；interleave/join/split 仅 triton） |
| `--extended-int8-prob` | 0.30 | Extended matmul 使用 int8 × int8（int32 累加）的概率 |
| `--region-int8-prob` | 0.15 | 普通 Region 生成 int8 GEMM-only 程序的概率（规格取自预校验网格） |
| `--no-region-pass-config` | 关闭 | 指定后禁用 Region 的 pass-config 不变性配对 |
| `--no-region-swizzle` | 关闭 | 指定后禁用 tilelang `T.use_swizzle` Region 变体配对 |
| `--no-region-warp-policy` | 关闭 | 指定后禁用 tilelang `GemmWarpPolicy`（FullRow/FullCol）Region 变体配对 |
| `--no-instance-grids` | 关闭 | 指定后禁用新 op 面的 per-(op, backend) 轮转实例网格 |
| `--uncovered-boost` | 50.0 | 从未尝试的结构特征的权重加成（MLIRSmith 式多样性优先；0 恢复旧权重） |
| `--no-structural-feedback` | 关闭 | 指定后禁用结构反馈引导 |
| `--compile-only` | 关闭 | 指定后仅编译，并强制 Extended 概率为 1 |

更多参数通过 `--help` 查看；未开放 CLI 的配置在 `src/config/config.py`，包括尺寸池、模板深度、操作数量、scratch 预算及数值阈值。

## 模板如何生成

普通 Region 先调用 `program_template()` 生成辅助函数和入口的操作树。模板保存操作种类、if/for 子区域、调用目标和接口信息，此时尚未为每个操作绑定具体 SSA 值。`instantiate_program()` 随后按函数顺序实例化：为操作选择当前作用域中类型合适的值，填写属性，分配结果名，再采样形状和合法调度。后面的函数只调用前面定义的函数，从而避免递归环。

v3 使用完整 fp32 tile；v4 在此基础上加入 fp16/fp32、tile/row/column/scalar 形状、tensor/buffer 区分和 scratch 读写。验证器检查作用域、调用签名、区域返回值、类型和形状。

probe 使用整函数 `probe` 节点，其参数选择 copy、reduce_sum/max/min、softmax、argmax 或 GEMM+argmax。它专门测试物理步长、偏移、广播、尾部掩码、特殊数值、重复执行和缓存复用，使用专用参考检查。生成、变异、保存和恢复全部采用 `RegionProgram`。

Extended 使用单独的 IR 和生成器，提供 `arithmetic`、`indexed_memory`、`shape_matmul`、`control_calls`、`mixed` 五类程序骨架；在各骨架中实例化带类型的操作、操作数和属性。它支持显式内部 matmul、索引访存、多值控制流/函数接口，以及中间结果观测和编译配置配对。它不是旧 `DynamicSequence` 的改名。

MLIRSmith 式 op 面扩张在两层 IR 之上新增编译器代码路径。Extended 程序加入全局内存原子操作（在刻意竞态的地址上做可交换 add/max/min）、带数据依赖操作数的标量 FMA 链、triton 形状原语（flip/interleave/join/split）以及 int32 累加的 int8 × int8 matmul。普通 Region 加入超越函数元素操作（tanh/erf/log/log2/exp2/rsqrt/sin/cos/floor/ceil）和 int8 GEMM-only 程序（规格来自预校验网格：block_K ∈ {32, 64}、int32 累加、精确整数参考）。fp32 GEMM 永不与边界台阶 op（ceil/floor/round/cast）组合：TF32 张量核计算相对精确 fp32 参考会翻动取整边界，使 oracle 淹没在无法与 bug 区分的噪声里。

新 op 的属性取自**有界实例网格**（`src/workflow/generator/grids.py`）：per-(op, backend) 轮转游标保证每轮扫掠每个角落实例恰好出现一次（MLIRSmith 穷举实例思想在新 op 面上的针对性版本；老 op 保持随机采样）。网格游标随 campaign 的 rng 状态持久化；`--no-instance-grids` 恢复纯随机采样。

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

tilelang 的 `opt_level` 无法穿透 `tilelang.compile`（所有 s_tir pass 声明 `opt_level=0`），因此 RC2 pass 管线差异用已核实的 `pass_configs` 键实现；`tl.enable_fast_math` 会改变数值，默认关闭。随机采样池（`src/backends/common/knobs.py`）只含在本机 tilelang 0.1.11 中核实过消费者的键（排除竞态、去掉安全合法化、Hopper-only 和 debug 键），且采样是（程序签名, seed）的纯函数——证据读取和超时缩放会重新推导同一变体列表。

每个失败报告的定位信息保存在 summary.json 的新键 `root_cause_locations`（`root_cause → 位置 → 次数`）中：位置依次取不变性标签本身、崩溃前最后一个 `TILESMITH_STAGE` 标记、TVM pass 名或报错源文件。`root_causes` 键保持 `{str: int}` 形状不变，`failed/` 目录命名不变。

详细调用关系见 [工作流说明](src/workflow/README.cn.md)。

## 检查、结果与恢复

生成的 `.py` 包含 kernel、输入初始化、参考计算和检查，可独立运行。普通 Region 检查数值、重复执行、调度配对（线程、num_stages、loop_kind）、布局配对、输入完整性和输出保护区；typed Region 还检查 scratch。Extended 额外保存编译阶段证据，并检查中间结果观测、pass 配置变化、随机采样管线、累加器宽度、代数恒等式及 scratch 内容。

结构反馈记录操作、数据依赖、嵌套、类型、布局和调度等特征。尝试、成功执行和编译特征分别统计；它不是编译器分支覆盖率。错误分类用于分组，`wrong_result` 仍需排查数值容差和参考语义，不能直接当作已确认的编译器 bug。

```text
results/<时间>_<backend>_<形状模式>_seed=<seed>/
  passed/                         成功执行的 .json IR 和 .py
  compiled/                       compile-only 结果，不能等同成功执行
  failed/<root_cause>/             失败报告和独立复现脚本
  artifacts/                      Extended 编译证据
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
