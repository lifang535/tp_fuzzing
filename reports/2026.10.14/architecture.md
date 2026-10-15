# TileSmith 代码框架与运行逻辑（ea9ecc41，2026-10-14）

本文说明当前代码的组成、一次测试从生成到判定的完整流程、各条生成路线、特征切片子系统、失败分类与反馈，以及服务器上的实验管理和本地监控。行号、参数默认值以 `claude/cross-target-20261011` 分支的 ea9ecc41 为准；命令行参数与非切片路线的更多细节见仓库根目录 `README.cn.md`。

## 1. 整体组成

TileSmith 是针对 tile 级 GPU DSL（**TileLang 0.1.14**、**Triton 3.8.0**）的编译器 fuzzer。它生成合法的 tile 程序，在子进程中编译、运行，并与独立的参考计算比对，发现编译器崩溃、生成代码被拒绝、运行时错误和静默算错。

整个系统由三部分组成：

| 部分 | 位置 | 作用 |
|---|---|---|
| fuzzer 本体 | `tp_fuzzing/`（git 仓库，约 1.74 万行 Python） | 生成、执行、判定、分桶、约简、反馈、持久化 |
| 服务器实验管理 | 服务器 `/root/autodl-tmp/tilesmith_campaigns/<管理目录>/`（源文件在本地 `claude_work/campaign_20261009/`） | 按计划启动两个 backend 的长时间实验，监督进程、清理缓存、断点停止与恢复 |
| 本地监控与分诊 | 本地 `claude_work/campaign_20261009/` | 通过 SSH 定期拉取状态和失败桶，按已知机制规则自动标注，人工调查新桶 |

## 2. 代码布局

```text
main.py                         命令行解析、后端加载、--dump / campaign 入口（402 行）
src/
  config/config.py              全部可调参数（Config dataclass）
  ir/                           程序表示
    region.py, region_ops.py, region_types.py   Region IR：带词法区域、SSA 值和循环携带值的结构化 tile 程序
    extended.py                 Extended IR：带具体标量/向量/矩阵形状、多结果、块内别名缓冲的探索 IR
    slice.py                    SliceProgram：(切片名, 旋钮赋值, backend)，不存代码
    ir.py, layout.py            TileKernel 参数（形状、dtype、调度）、输入物理布局
    serialization.py            三种程序的保存与恢复
  backends/
    base.py                     后端接口：参数约束、lowering、执行命令、诊断
    common/                     共用调度、probe、程序分发（emitter.py）、harness 组装
    tilelang/, triton/          各 DSL 的 lowering、参数、诊断分类
  workflow/
    fuzzer/fuzzer.py            主循环（1151 行）：路线选择、去重、执行、记录、反馈、保存、恢复
    fuzzer/dsl_stage.py, dsl_schedule.py   DSL 派生阶段及其代价感知调度
    generator/                  Region 生成（region_generator.py）、Extended 生成（extended.py）、
                                DSL 派生（dsl_extend.py）、精度/恒等变换（identities.py）
    mutator/mutator.py          用同一个递归生成器做变异
    emitter/                    嵌入独立脚本的参考解释器和检查（Region、Extended、probe）
    oracle/oracle.py            子进程执行、超时、错误分类、生成 BugReport
    slices/                     特征切片子系统（约 4900 行，见第 5 节）
    triage.py                   失败分桶、wrong result 来源定位、quarantine
    feedback.py, extended_feedback.py   结构特征反馈（类 MLIRSmith DiversityCriteria）
    coverage_audit.py           覆盖证据审计
tests/                          514 个 CPU 单元测试 + 离线编译 / GPU smoke 脚本
reports/                        各阶段报告与 bug 复现脚本
```

## 3. 一次测试的生命周期

主循环在 `TileSmith.run()`（`fuzzer.py:525`）中，每次迭代处理一个程序：

```text
            ┌──────────────────────── 生成 ─────────────────────────┐
            │ _generate_test_case()                                  │
            │   ① 切片路线  (概率 slice_prob)  → SliceScheduler.next │
            │   ② DSL 派生  (概率 dsl_extend_prob)→ DslStage.generate│
            │   ③ 原生路线  fresh / mutate → Region 或 Extended      │
            │   quarantine.admit()：命中已知失败规则则重抽（≤8 次）  │
            └───────────────┬────────────────────────────────────────┘
                            ▼
                去重：_make_sig(program) 已测过 → 跳过，不计数
                            ▼
            ┌──────────────────────── 执行 ─────────────────────────┐
            │ Oracle.test(program)                                   │
            │   emit：程序 → 自包含 Python 脚本（kernel+参考+检查）  │
            │   写入临时文件，独立子进程运行（私有 TMPDIR、超时）    │
            │   退出码 0 且打印 ALL PASSED → 通过                    │
            │   否则：错误分类 → BugType / root_cause → failure_bucket│
            └───────────────┬────────────────────────────────────────┘
                            ▼
      失败且为切片程序 → _minimize_slice()：逐旋钮约简，得到核心与 .min.py
                            ▼
      _record_bucket()：统计桶；切片调度器 observe()；结构反馈 / quarantine 更新
                            ▼
      保存：失败 → failed/<root_cause>/*.json + .py (+ .min.py)
            通过 → passed/*.json（只存 IR；可作为变异种子或 DSL 派生源）
                            ▼
      每 100 个测试写一次进度；中断时写 pending_program.pkl 和全部状态
```

要点：

- **每个测试都在新的子进程里运行**，生成的脚本是自包含的（kernel、输入构造、参考实现、检查函数都在一个文件里），所以失败文件本身就是复现脚本，脱离 fuzzer 也能直接 `python X.py` 运行。
- **去重**：同一程序签名只测一次；`-n` 计的是新测的程序数。
- **不丢失败**：`max_same_root_cause=0`，每个失败都保存复现文件；桶计数只用于统计和调度。

## 4. 生成路线

当前实验的参数（`make_plan.py`）：`--slice-prob 0.85 --extended-prob 0.5 --dsl-extend-prob 0.4 --dsl-evolve-prob 0.5 --quarantine --corpus-feedback --swarm-prob 0.5 ...`。每次生成先以 85% 概率走切片路线；没选中时再以 40% 概率做 DSL 派生（需已有派生源）；其余进入原生路线。

| 路线 | 程序表示 | 做法 | 参考 |
|---|---|---|---|
| **slice（切片）** | SliceProgram | 10 个聚焦某一 DSL 特性的生成器，在离散旋钮空间上采样并合法化（第 5 节） | float64/int64 精确参考，逐位比较 |
| fresh（全新生成） | Region / Extended | Region：先递归生成结构，再做词法 SSA 绑定；有 50% 概率只用一半操作（swarm testing）；部分为定向 probe（布局、尾部掩码、融合下标归约）。Extended：类型导向生成，带覆盖锚点 | 嵌入脚本的独立解释器（PyTorch），多输入种子、多次重复、128/256 线程等调度对 |
| mutate（变异） | Region | 从种子池按稀有度加权选种子，用同一个递归生成器重写一部分 | 同上 |
| dsl_extend（DSL 派生） | Extended | 从通过的公共 Extended 程序派生出使用目标 DSL 特有操作的程序，最多 3 步，按新颖度与代价调度 | 同上 |

非切片路线还有若干差分检查：调度不变性（不同线程数/流水线级数结果应相同）、pass 配置不变性、精度对（fp16 累加、tf32 变体）、代数恒等式（分配律变体）。它们产生的 `schedule_mismatch`、`pass_config_mismatch`、`precision_mismatch` 等桶，因为参考是浮点近似，误报率较高，监控协议里只列出、不调查。

## 5. 特征切片子系统（`src/workflow/slices/`）

切片是当前实验的主力（约 85% 的测试，也是 #1–#29 中绝大多数机制的来源）。

### 5.1 十个切片

| 切片 | 聚焦的特性 |
|---|---|
| `cast` | dtype 转换链（含 bf16/fp8/fp64/int16/int64/无符号） |
| `round` | 舍入模式、整数↔浮点、fp8 转换 |
| `math` | Triton `tl.*`、libdevice 约 215 个函数条目；TileLang `T.ieee_*`、取整、位运算、x2 打包运算等 99 个 |
| `reduce` / `scan` | 归约、扫描、自定义 combine |
| `gemm` | `tl.dot` / `T.gemm`：dtype、块形状、累加类型、后处理 |
| `atomic` | 原子操作：类型、语义（acquire/release…）、作用域、竞争方式 |
| `layout` | reshape/permute/trans/join/split/gather/sort 等至多三步形状变换；TileLang 共享内存与寄存器片段间的拷贝、重排、swizzle |
| `loop` | `range`、`tl.range`（num_stages、warp_specialize…）、while、`T.Pipelined` 等循环形式 |
| `memory` | 窗口、填充、掩码、cache 提示、块指针、tensor descriptor、`T.copy` 选项 |

### 5.2 程序 = 旋钮赋值

每个切片有一个离散旋钮空间（`Slice.space()`），由三类旋钮组成：

- **公共旋钮**：输入 dtype、数值范围（`values`）、形状、尾部掩码、输出 dtype、前后各 1–2 步逐元素运算（`pre*_op/post*_op`）、warps / threads、是否成对比较两种启动配置（`pair`）。
- **核心旋钮**：切片特有，如 gemm 的 `bm/bn/bk`、atomic 的 `sem/scope`、layout 的 `op1..op3/arg1..arg3`。
- **配置旋钮**（`config_space`）：不应改变语义的编译选项。
  - Triton：`cfg_maxnreg`（0/32/64/128，寄存器上限）、`cfg_fusion`（FMA 合并）、`cfg_stages`、`cfg_ftz`。
  - TileLang：`cfg_pass`，17 种 pass 配置（index64、关向量化、关 CSE、ldgstg 降级、unswitch 等）。
  - `cfg_target`：编译目标（第 5.6 节）；`cfg_amd`：Triton 的 AMD 降级选项。
  - 采样时 50% 的程序用默认配置，35% 的程序编到非本机目标。

### 5.3 合法化与精确参考

`Slice.legalize(params, backend)` 把任意赋值修成可执行程序，并返回 plan（实际执行的步骤序列）：

- 输入是 `k·2^-frac` 形式的小二进有理数；每一步运算都传递一个静态值域 (lo, hi, frac)（YARPGen 式范围跟踪），只有当结果在所用 dtype 中**精确可表示**时才保留这一步，否则退回 `none` 或不转换。
- 因此 float64/int64 的参考计算是精确的，输出可以**逐位比较**：任何不一致都不是舍入噪声。math 切片的近似函数是例外，按规定的 ulp 容差检查。
- 合法化必须**幂等**（再合法化一次不变）：`emit` 时会重新合法化并检查参数没变；`make_program` 现在反复合法化到不动点（ea9ecc41 修复的崩溃即源于此）。

### 5.4 harness 的检查

`Slice.emit()`（`base.py:258`）生成的脚本结构：

```text
helpers（runtime.py 中的函数源码原样嵌入）
PLAN = {...}                        # 合法化后的执行计划
kernel(...)                         # Triton 或 TileLang kernel
reference(x, y)                     # float64/int64 精确参考
main():
  _slice_target(cfg_target, ...)    # 非本机目标：改写编译目标；返回是否只编译
  x, y = _slice_values(...)         # 确定性输入
  expected = reference(x, y)
  for 每种启动配置 (label, options):    # pair=1 时两种 warps/threads
    for run in (0, 1):                     # 每种配置跑两次
      out = 带 64 元素前后保护区的输出缓冲，填 7
      launch(...)；异常 → _slice_reject 判断是否为“前端明确不支持”
      只编译目标 → 跳过检查
      检查 1：保护区未被改写（越界写）
      检查 2：两次运行逐位相同（确定性/数据竞争）
      检查 3：与 expected 逐位比较（或 ulp 容差）
  print('ALL PASSED')
finally: _slice_lowering() 打印降级特征（PTX 操作码、TTGIR 操作、CUDA 内建）
```

- 输出的阶段标记 `TILESMITH_STAGE=reference / execute:<label>:<run>` 用来定位失败发生的阶段。
- `_slice_reject`：前端给出清晰的“不支持”报错（如 `not supported in this architecture`）时打印 `TILESMITH_REJECTED=` 标记，oracle 据此归为 `unsupported_feature` 噪声，而不是 bug；内部断言、`PassManager` 失败、`LLVM ERROR` 等不算拒绝。

### 5.5 调度（`scheduler.py`）

- **选哪个切片**：先每个切片各跑 3 次（warm-up），之后按“每秒期望发现数”加权抽取。期望发现数用 Good-Turing 估计（Böhme STADS）：新失败桶概率 (f1+1)/(n+2)、新旋钮对覆盖、新降级特征三者加权；另有 15% 均匀抽取，保证没有切片被饿死。
- **切片内选哪个程序**：随机采样 16 个合法候选，选覆盖最多未见旋钮对（pairwise）的那个（AETG 贪心）；30% 概率改为对“有趣”的旧程序或种子改 1–2 个旋钮做变异。
- **降级反馈**（Zest / FairFuzz 式）：第一次到达某个降级特征的通过程序成为种子；变异时按其最稀有特征的稀有度加权选种子。
- **避开已知机制**：候选含有某个已命中 k 次的失败核心时，（k ≥ 3）以 1 − max(0.02, 3/k) 的概率跳过。

### 5.6 编译目标（跨架构）

`_slice_target()`（`runtime.py:813`）在第一次编译前改写 Triton 的 driver 目标和 JIT、TileLang 的 `tilelang.jit`：

| 目标 | 处理 |
|---|---|
| native（sm_89） | 编译并运行，完整检查 |
| sm80 | 编译为 sm_80 后在 sm_89 上运行，完整检查 |
| sm75（仅 TileLang）、sm90、sm100、sm120、gfx90a、gfx942、gfx950、gfx1100（仅 Triton） | 只编译（Triton 到 cubin/hsaco；TileLang 到 CUDA/HIP 源码，CUDA 目标再交给 nvcc），崩溃或构建失败才算失败 |

目标本身缺少的特性（Triton 架构断言、AMD wave64 下不足一个 wavefront、AMD 缺少的 fp8 转换、Blackwell 上寄存器上限过低）记为噪声。

### 5.7 失败约简（`minimize.py`）与核心

切片失败后，`_minimize_slice()` 在预算内（每次最多 28 个额外测试，总量不超过切片测试数的约 30%）逐个旋钮把值重置为最简值（首先把 `cfg_target` 恢复为 native），失败仍在就保留重置。剩下不能重置的旋钮构成**核心**（core）：

- 错误结果的桶按核心命名：同一机制经不同 dtype 路径触发会约简到同一核心，因此落入同一个桶。
- 核心交给调度器，用于避开已知机制。
- 约简后的程序保存为 `.min.py`。

## 6. 判定、分类与分桶

1. **BugType**（`oracle.py`）：`compile_crash`、`runtime_crash`、`wrong_result`、`timeout`、`oracle_unstable`（参考自身不稳定，不是 bug）。
2. **root_cause**：后端的诊断规则（`backends/*/diagnostics.py`）把错误信息归类，如 `triton_pass_failure`、`tilelang_codegen_error`、`tilelang_cuda_compile_error`、`assertion_failure`、`segfault`、`other`。噪声类：`oracle_unstable`、`unsupported_feature`、`shared_memory_overflow`、`warp_partition`、`gpu_oom`、`timeout`、`register_limit`。
3. **failure_bucket**（`triage.py`）：`root_cause:` + 规范化诊断（去掉数字、路径、变量名）的 SHA1 前 10 位。错误结果的诊断都是检查器自己的消息，所以再拼上**来源**：切片程序用核心旋钮，Extended 程序用最早出错的操作。经审计的已知签名直接作为桶名。
4. **截断**：报告中只保留错误信息的有界尾部（2000 字符），但分类在完整信息上进行（d6046da9 起；之前截断会丢失 `TILESMITH_TARGET/REJECTED` 标记，造成一类 fuzzer 缺陷）。

## 7. 反馈与 quarantine

- **结构反馈**（`feedback.py`，非切片路线）：记录程序的结构特征（操作、类型、组合）是否被尝试、是否通过；从未通过的特征获得额外权重（`--uncovered-boost 50`），通过且带来新特征的程序进入种子池（上限 200，按稀有特征保留代表）。这是源码/IR 特征，不是编译器分支覆盖。
- **quarantine**（`triage.py: Quarantine`）：在最近 1024 个样本上学习“特征组合 → 反复出现的已知失败桶”的规则（精度 ≥ 0.9），之后命中规则的 fresh/mutate 候选会被重抽（最多 8 次，最后一次强制放行），减少重复命中同一 bug 的浪费。错误结果和 oracle_unstable 不参与学习。
- **explained feedback**：重复失败所用的特征视为“已探索”，降低其稀有度权重。

## 8. 持久化与断点恢复

结果目录 `results/<时间>_<backend>_hard-shape_seed=<seed>/`：

| 文件 | 内容 |
|---|---|
| `failed/<root_cause>/failed_*.{json,py,min.py}` | 失败记录、原始复现脚本、约简后复现脚本 |
| `passed/passed_*.json` | 通过程序的 IR（`--no-save-passed-code` 时不存 .py） |
| `summary.json` | 统计、生成配置、环境版本、路线耗时 |
| `rng_state.json`、`pending_program.pkl` | 随机状态；中断时正在测的程序 |
| `seed_pool.json`、`dim_pool.json`、`structural_feedback.json`、`quarantine.json`、`slice_state.json`、`dsl_stage.json` | 各反馈与调度状态 |
| `coverage_progress.json(l)` | 覆盖/发现进度时间线 |

`--resume <目录>` 时：读取全部历史结果用于去重，恢复随机状态和上述所有状态，先重跑 pending 程序，测试编号接着往下排。程序通过 `SliceProgram.from_dict` 载入时会按当前代码重新合法化，所以代码升级后旧程序仍能恢复。

## 9. 服务器实验管理

每台服务器一个管理目录（当前为 `/root/autodl-tmp/tilesmith_campaigns/20261012_xtarget_b1eded9f/`）：

| 文件 | 作用 |
|---|---|
| `make_plan.py` | 生成 `plan.json`：两个 backend 的完整命令行（seed、各路线概率、`-n 500000`、结果目录、可选 `--resume`）和环境变量（各自的编译缓存目录） |
| `supervise.py` | 校验 HEAD 等于 `commit.txt`，启动两个 fuzzer 进程；每 10 分钟写 `status.json`、清理 30 分钟前的编译缓存；剩余磁盘 < 8 GiB 时请求停止；收到 SIGINT 时让两个进程正常保存检查点后退出 |
| `pause.py` | 向 supervisor 发 SIGINT，正常暂停 |
| `resume_campaign.sh` | 检查 HEAD 与 `commit.txt` 一致、磁盘 ≥ 12 GiB，暂停仍在运行的 backend，归档旧日志为 `segmentN/`，重新生成计划并以 `--resume` 启动 |
| `commit.txt`、`README.txt`、`tests.log` | 运行的 commit、变更记录、部署时的单测结果 |

部署流程：本地修改 → push 到 GitHub → 服务器 `git fetch`（需 `source /etc/network_turbo`）→ `pause.py` → checkout → 服务器单测 → 更新 `commit.txt` → `resume_campaign.sh`。

## 10. 本地监控与分诊

监控只在本地通过 SSH 进行，不在服务器上部署任何东西（`claude_work/campaign_20261009/`）：

| 文件 | 作用 |
|---|---|
| `rs.sh`、`../ssh/remote.py` | 远程执行命令、上传下载文件（密码从 `collect.sh` 读取，不入仓库） |
| `watch.sh` + `watch_check.py` | 后台 watchdog：每 10 分钟检查两台的 `status.json`（进程退出、停止请求、supervisor 超过 15 分钟未更新、磁盘 < 15 GiB）和各 backend 45 分钟内是否写过结果；连续 4 次连不上也算异常。异常时打印 `ANOMALY` 并退出以唤醒会话 |
| `check.sh` → `collect.sh` / `snapshot.py` / `check.py` | 每 3 小时一次：拉取两台的状态和失败桶快照（`snapshots/<时间>/`），列出新桶并按 `known_mechanisms.json` 自动标注，与基线实验按同时长对比 |
| `known_mechanisms.json` | 已确认机制及 harness 已修问题的匹配规则（按诊断子串、切片、参数或约简核心匹配）；`_FIXED` 结尾的规则若重现会报警 |
| `compare_mechanisms.py`、`baseline_*/` | 与前几轮实验在同一时长的机制数、桶数对比 |
| `CHECK_PROTOCOL.md`、`CHECKS.md` | 检查流程，以及每次检查的记录 |
| `cases/<桶>/` | 下载的待调查失败样例与本地二分脚本 |

新桶的处理：已标注的只记数；切片路线的未标注桶下载 `.json/.py/.min.py`，先去掉配置旋钮或目标判断其是否必要，再排除 harness/类型模型问题，能复现的在本地 GPU（RTX 4060，sm_89）复现并写独立最小脚本，确认后加入 `known_mechanisms.json` 和报告；fuzzer 自身问题只记录，不在实验中途改代码（除非导致进程崩溃）。

## 11. 已知局限

- **只编译的目标**查不出静默算错；这些架构上的运行结果需要真实硬件。
- **合法化缺口**：仍有少数参数组合会为目标生成它不支持的程序（TileLang HIP 的 nan_propagate、MFMA k_dim、Triton gfx90a 上强开 async copy），产生 fuzzer 缺陷桶。
- **非切片路线**的差分检查基于浮点近似参考，误报多，目前只列出不调查。
- **反馈是源码/IR/降级特征**，不是编译器的真实代码覆盖；在覆盖平台期后发现新机制的速度明显下降。
- **环境版本**：`requirements.txt` 仍固定旧版 TileLang 0.1.11 / Triton 3.0.0，实验环境以服务器 `tp_fuzzing_latest`（TileLang 0.1.14、Triton 3.8.0、PyTorch 2.4.0+cu124、Python 3.11）为准。

## 12. 参考的相关工作

下表只列在代码注释或设计文档中明确引用、并实际影响了实现的工作；“对应实现”给出代码位置。FFTG 和 tile bug 实证研究两项最初按项目已有文档引用，10-14 已核对出处（见对应行）；实证研究中的百分比仍是旧文档的说法，未对照原文。

### 12.1 程序生成

| 工作 | 思想 | 在 TileSmith 中的对应 |
|---|---|---|
| **MLIRSmith**（MLIR 随机程序生成，ASE'23） | 面向多层 IR 的随机生成；DiversityCriteria：优先选择从未覆盖的结构；对同一程序扫描多种 pass 流水线、调度和精度配置 | Region 生成器与结构反馈（`feedback.py`，`--uncovered-boost`）；调度扫描、pass 配置扫描、累加精度扫描、代数恒等式扫描、随机 pass 流水线采样（`config.py`、`backends/common/knobs.py`）；小实例域枚举（`generator/grids.py`） |
| **FFTG**（“Feature-Focused Test Generation”，ASE'26，doi 10.1145/3832783.3837479） | 基于规约的测试生成：从规约提取特性模型，由测试者选定要聚焦的特性，生成只聚焦这些特性的测试切片（原型 Compás）；本身不是编译器 fuzzing，TileSmith 借用“按特性切片、其余维度保持简单”的思路 | 特征切片（`slices/`，10 个切片） |
| **tile 程序 bug 实证研究**（“Characterizing Real-World Bugs in Tile Programs for Automated Bug Detection”，ISSTA'26，arXiv 2605.19652） | 301 个真实 tile 程序代码生成 bug（8 个框架）的根因、症状与 oracle 分类；旧文档称数据类型语义约占 19%、类型/算子合计约 49%（未对照原文） | 切片优先覆盖 dtype 转换链、fp8/bf16/fp64/窄整数（`slices/cast.py`、`round.py`、`mathfn.py`） |
| **YARPGen**（Livinskii, Babokin, Regehr, OOPSLA'20） | 生成时跟踪取值范围，保证程序无未定义行为 | 切片的静态值域传递：只保留结果精确可表示的步骤，使参考精确、可逐位比较（`slices/dtypes.py`、`chain.py`） |
| **Swarm testing**（Groce 等，ISSTA'12） | 每个测试随机关掉一部分特性，反而能更深地测剩下的特性 | `--swarm-prob 0.5`：原生 Region 只从随机一半的操作中抽取（`region_generator.py`） |

### 12.2 反馈与调度

| 工作 | 思想 | 在 TileSmith 中的对应 |
|---|---|---|
| **STADS**（Böhme，TOSEM'18） | 把 fuzzing 看作物种发现，用 Good-Turing 估计下一个测试发现新物种的概率 | 切片调度按“每秒期望发现”加权（`slices/scheduler.py`）；发现率统计（`fuzzer._discovery`、`triage.py`） |
| **Entropic**（Böhme, Manès, Cha，FSE'20） | 按信息熵（物种发现率）分配种子能量 | 同上，切片间预算分配的思路来源 |
| **AFLFast**（Böhme, Pham, Roychoudhury，CCS'16） | 功率调度：少给已被频繁命中的路径分配能量 | quarantine 中按已见次数降低重复区域的采样（`triage.py`） |
| **Zest**（Padhye 等，ISSTA'19） | 在结构化生成器的参数空间上做覆盖引导的变异，产生语义合法的输入 | 切片程序就是生成器参数（旋钮赋值）；以降级特征（PTX 操作码、TTGIR 操作、CUDA 内建）为覆盖反馈选种子变异 |
| **FairFuzz**（Lemieux, Sen，ASE'18） | 把变异集中到触及稀有分支的输入上 | 种子按其最稀有降级特征的稀有度加权（`scheduler.py`） |
| **AETG**（Cohen 等，TSE'97） | 组合测试：贪心选择覆盖最多未覆盖参数对的测试 | 切片内 16 个候选中选覆盖最多新旋钮对的那个 |

### 12.3 判定、约简与去重

| 工作 | 思想 | 在 TileSmith 中的对应 |
|---|---|---|
| **ddmin / Delta Debugging**（Zeller, Hildebrandt，TSE'02） | 逐步删减失败输入直到最小 | 切片失败按旋钮逐个重置为最简值（`slices/minimize.py`） |
| **spirv-fuzz 的约简去重**（Donaldson 等，PLDI'21） | 约简到同一结果的失败视为同一 bug，用约简结果去重 | 错误结果按约简后的核心分桶；核心交给调度器避开已知机制 |
| **差分测试 / 等价变体**（MLIRSmith 的配置扫描及 EMI 一类思想） | 同一程序在不改变语义的变换或配置下结果应相同 | 切片的配置旋钮（`cfg_maxnreg`、`cfg_pass` 等）、成对启动配置、两次运行确定性检查；非切片路线的调度/pass 配置/精度/恒等式不变性检查 |

### 12.4 支撑实验设计判断的工作

| 工作 | 用途 |
|---|---|
| **Böhme & Falk，“Fuzzing: On the Exponential Cost of Vulnerability Discovery”**（FSE'20） | 在同一输入空间里，发现新 bug 的代价指数增长；据此在每轮实验出现平台期后扩大输入空间（加切片、加编译配置、加跨架构目标），而不是延长运行时间 |

### 12.5 考虑过但未采用

| 工作 | 原因 |
|---|---|
| **CodaMOSA**（Lemieux 等，ICSE'23）、Gentoo、ChatAFL 等 LLM 辅助生成 | 需要接入模型 API；计划在覆盖平台期用模型针对未覆盖的 API / 降级特征补充切片模板 |
| **AFLGo**（Böhme 等，CCS'17）等定向灰盒 fuzzing | 需要对 TVM/TileLang C++ 与 Triton MLIR pass 做代码覆盖插桩并重新编译编译器；目前反馈仍是源码/IR/降级特征 |
