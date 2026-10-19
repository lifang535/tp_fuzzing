# TileSmith 最新实验检查报告

日期：2026-10-19。所有时间均为北京时间（UTC+8）。[English report](REPORT_EN.md)

**本轮从 10 月 16 日中午开始；截至 10 月 19 日 18:25，已运行约 78 小时 16 分钟，四路实验仍在运行，最新检查点累计测试 164,000 个程序。** 真实问题统计固定使用当天 17:35 采集的 6,176 条失败记录：其中 4,669 条匹配 **43 类真实编译器或下游工具链问题**，87 条对应 **4 类 fuzzer 缺陷**；另有 19 条数值差异和 8 条 nvcc 崩溃尚未完成归因。

本报告独立编号为 **TileLang#1–TileLang#26、Triton#1–Triton#17**，不沿用历史总表的编号；中英文报告、数据表与复现说明使用同一套编号。编号仅在本报告中有效。按同一口径回查，上轮单独长跑命中 46 类，本轮目前命中 43 类，少 3 类。

## 1. 运行时间与实验范围

实验源码提交为 `5a790b0235ec59e927583f7a02ac64763f17ebef`，分支为 `tilelang-0.1.15-triton-3.8_2026.10.15`。两台服务器分别运行 TileLang 和 Triton，每路目标为 500,000 个程序。

| 服务器 | GPU / seed | 本轮开始时间 | 状态读取时间 | 墙钟时长 |
| --- | --- | --- | --- | --- |
| 41790 | RTX 4090 / 43 | 2026-10-16 12:09:22 | 2026-10-19 18:25:32 | 78 小时 16 分 10 秒 |
| 32906 | 32 GB vGPU / 42 | 2026-10-16 12:10:02 | 2026-10-19 18:25:32 | 78 小时 15 分 31 秒 |

### 1.1 最新进度与固定审计样本

最新进度仅说明实验仍在推进，不把后续新增失败混入已经完成分类的样本。

| 服务器 / 后端 | 18:25 读取的累计测试数 | 该检查点时间 | 17:35 审计对应测试数 | 固定失败记录数 |
| --- | ---: | --- | ---: | ---: |
| 41790 / TileLang | 19,000 | 18:08:53 | 18,800 | 1,225 |
| 41790 / Triton | 42,600 | 18:19:25 | 42,100 | 1,312 |
| 32906 / TileLang | 32,300 | 18:18:12 | 31,900 | 2,136 |
| 32906 / Triton | 70,100 | 18:23:15 | 69,400 | 1,503 |
| 合计 | **164,000** | | **162,200** | **6,176** |

失败文件采集窗口为 **2026-10-19 17:35:22–17:35:41**；相应进度检查点在 17:26–17:33，已运行约 77.3 小时。固定检查点的 passed 合计 156,027，失败计数为 6,173；另外 3 份失败文件在读取间隙产生。因此分类总数取实际保存的 6,176 份 JSON，全部可解析。

统计使用 `coverage_progress.json.total_tested`，不是日志中的生成尝试流水号。`passed` 包括跨架构编译成功，不能解释成所有程序都已在相应目标 GPU 上完成数值验证。

### 1.2 环境与运行状态

- 两台服务器：TileLang **0.1.15**、Triton **3.8.0**、PyTorch **2.4.0+cu124**、apache-tvm-ffi **0.1.11**、Python **3.11**，系统 nvcc **12.4.131**。
- 结果目录分别为 `2026.10.16-12.09_{tilelang,triton}_hard-shape_seed=43`、`2026.10.16-12.10_{tilelang,triton}_hard-shape_seed=42`。凌晨两轮短试跑未计入。
- 四个主进程在 18:25 读取时均存在。17:51 健康检查中退出码和停止请求均为空，剩余磁盘约 44.20 / 49.92 GiB。
- 检查时本地与远端共同采集的 152 个源码、文档、测试文件哈希一致；远端检查前后源码哈希未变化。SSH 子进程直连，显式关闭代理。

可核对的时间、计数、版本与采样统计见 [audit_summary.json](data/audit_summary.json)。

## 2. 失败记录如何分类

| 分类 | 记录数 | 去重或归因结果 |
| --- | ---: | --- |
| 匹配真实编译器 / 工具链机制 | 4,669 | TileLang 路径 26 类，Triton 路径 17 类 |
| 确认的生成器 / 检查器缺陷 | 87 | 4 类，见第 5 节 |
| 资源限制、明确不支持、参考不稳、超时 | 1,393 | 不计为真实编译器 bug，也不统一归罪于 fuzzer |
| nvcc 崩溃，归因待定 | 8 | 单列工具链候选 |
| 数值差异，归因待定 | 19 | 尚未排除所有参考模型或浮点语义因素 |
| 合计 | **6,176** | |

“记录数”是重复触发次数，“机制数”才是去重后的问题种类。跨机器合并有 267 个不同的 `(backend, failure_bucket)` 诊断桶，但同一根因可以产生多个桶，同一桶也可能混合不同根因。

43 类的证据分为两层：保存的诊断、参数及约简核心与已独立验证机制匹配；本次又对其中 **5 类**运行了独立小样例和对照。并未声称重新独立验证了全部 4,669 条记录。跨架构问题中许多仅验证目标编译路径。逐记录数据见 [classification.csv](data/classification.csv)，完整机制表见 [bug_catalog.json](data/bug_catalog.json)。

## 3. 本轮真实问题目录

### 3.1 TileLang 路径：26 类、3,018 条记录

| 编号 | 实际问题 | 41790 记录 | 32906 记录 | 合计 |
| --- | --- | ---: | ---: | ---: |
| TileLang#1 | bool 向量 CUDA codegen 失败 | 43 | 68 | 111 |
| TileLang#2 | 自动选择的归约布局无法降级 | 38 | 52 | 90 |
| TileLang#3 | 流水线生成非法 2 字节 cp.async | 17 | 40 | 57 |
| TileLang#4 | 嵌套整数 floordiv codegen 崩溃 | 2 | 6 | 8 |
| TileLang#5 | int64→f16/bf16 `T.cast` 被 nvcc 拒绝 | 352 | 687 | 1039 |
| TileLang#6 | int64 `T.atomic_max/min/or` 被 nvcc 拒绝 | 45 | 57 | 102 |
| TileLang#7 | int→bf16 截断而非就近舍入（259→258） | 9 | 35 | 44 |
| TileLang#8 | while 循环拷贝到共享内存再 gemm 缺同步 | 38 | 112 | 150 |
| TileLang#9 | 窄整数运算后加宽不按窄类型回绕 | 73 | 152 | 225 |
| TileLang#10 | 半精度数学内建生成错误；本轮为 hpow/htanh 重载失败 | 6 | 33 | 39 |
| TileLang#11 | 缺降级规则的内建到达 codegen | 40 | 51 | 91 |
| TileLang#12 | 共享内存取模下标重排丢掉列偏移 | 29 | 92 | 121 |
| TileLang#13 | 区域拷贝降级为 ldmatrix 时丢列偏移 | 9 | 5 | 14 |
| TileLang#14 | 开 `tl.enable_lower_ldgstg` 后向量元素访问失败 | 3 | 5 | 8 |
| TileLang#15 | index64 改写值或向量类型，引发错误结果或 codegen 失败 | 40 | 74 | 114 |
| TileLang#16 | sm100/sm120 的 256 位 int16/uint16 向量类型无法打印 | 35 | 68 | 103 |
| TileLang#17 | IEEE 舍入内建在 AMD 上无降级规则 | 121 | 149 | 270 |
| TileLang#18 | 编译到 sm_80 的 fp8 `T.gemm` 用了 sm_89 才有的 MMA，运行时 device assert | 12 | 17 | 29 |
| TileLang#19 | 窄 N 的 f16 gemm 触发 `Not a canonical GMMA_MN layout` 断言 | 0 | 3 | 3 |
| TileLang#20 | HIP 后端 fp8 处理缺口：e5m2 常量无法打印；`float8_e4m3` gemm 报 KeyError | 47 | 68 | 115 |
| TileLang#21 | 8 位整数 xor 在 HIP codegen 中 `Check failed: (p)` | 42 | 66 | 108 |
| TileLang#22 | bf16 `T.pow` 报 `power only applies to float` | 5 | 11 | 16 |
| TileLang#23 | 显式 order/stage 的 `T.Pipelined` 把 fragment 多缓冲，随后报“只允许 fragment[0]”或 `Layout::Forward` 维数不符 | 58 | 71 | 129 |
| TileLang#24 | 浮点 `x - x`、`x * 0` 被化简成 0，inf/NaN 输入得 0 而不是 NaN | 1 | 1 | 2 |
| TileLang#25 | gemm 累加器加上另一个 fragment 的行和时布局推断失败（`has_best`） | 5 | 4 | 9 |
| TileLang#26 | sm75 bf16 atomic_min/max 错选 CUDA 12.4 atomic_ref 分支 | 8 | 13 | 21 |

TileLang#15 中的样例 `194926aa6a83ea49` 同时保留 index64 与 while/GEMM 条件，可能与 TileLang#8 的同步问题重叠；它暂作为记录索引保留，不因该条样例增加机制数。

### 3.2 Triton 路径：17 类、1,651 条记录

| 编号 | 实际问题 | 41790 记录 | 32906 记录 | 合计 |
| --- | --- | ---: | ---: | ---: |
| Triton#1 | f16/bf16 `tl.abs` 有一半元素仍为负 | 24 | 49 | 73 |
| Triton#2 | f32→fp8 默认舍入错（0.53125006→0.5） | 0 | 17 | 17 |
| Triton#3 | f64↔fp8 转换 LLVM abort | 319 | 317 | 636 |
| Triton#4 | fp8→fp8 转换断言 | 93 | 104 | 197 |
| Triton#5 | `fp_downcast_rounding='rtz'` pass 崩溃 | 65 | 72 | 137 |
| Triton#6 | bf16 自定义 combine 的 reduce/scan 编译失败 | 52 | 74 | 126 |
| Triton#7 | 常量折叠后转 fp8，部分 lane 是垃圾值 | 0 | 6 | 6 |
| Triton#8 | 向量化窄整数掩码加载，负填充值读错 | 0 | 12 | 12 |
| Triton#9 | `tl.fdiv` 用于 fp16/bf16 编译失败 | 18 | 19 | 37 |
| Triton#10 | `tl.gather` 沿 axis 0 断言 | 26 | 37 | 63 |
| Triton#11 | 有符号 `libdevice.mul24` 返回完整 48 位乘积 | 7 | 4 | 11 |
| Triton#12 | warp specialize 循环里有扫描时 unreachable | 8 | 0 | 8 |
| Triton#13 | gfx942/gfx950 的 atomic_xchg 生成不存在的 exch 内建 | 15 | 16 | 31 |
| Triton#14 | bf16 原子加在 `maxnreg=32` 下丢失更新 | 0 | 5 | 5 |
| Triton#15 | sm<89 上 f16→fp8e5 转换：平局一律进位，NaN 变成 inf 或 −0.0 | 14 | 19 | 33 |
| Triton#16 | gfx1100 的 fp16 反向 cumsum 被 LLVM 汇编器拒绝，外层报 LLD 失败 | 95 | 120 | 215 |
| Triton#17 | sm100/sm120 的 warp specialization + unroll 触发 dyn_cast 断言 | 19 | 25 | 44 |

“路径”包含 DSL 编译器及其下游：例如 Triton#1、#7、#14 涉及 ptxas，Triton#11 涉及 libdevice，Triton#16 涉及 LLVM AMDGPU 后端。它们不是全部都由 Triton 前端本身造成。

## 4. 具体例子、复现方法与本次实测

### 4.1 运行条件

GPU 小样例在本地 **RTX 4060 Laptop、sm89、Python 3.10.20** 上运行，TileLang / Triton / PyTorch 版本与服务器相同，系统 nvcc 和 Triton 自带 ptxas 均为 **12.9.86**。TileLang#26 单独在服务器原有 **nvcc 12.4.131** 下做 CPU 编译检查，不在服务器执行 GPU 测试。完整环境见 [environment.json](validation/environment.json)。

使用满足项目 [依赖版本](../../../requirements.txt) 的独立 Python 环境。从仓库根目录进入本报告目录，后续命令都在此运行：

```bash
cd reports/2026.10.19/latest_experiment_audit
python -c "import importlib.metadata as m; print({p: m.version(p) for p in ('tilelang', 'triton', 'torch', 'apache-tvm-ffi')})"
```

前五个脚本不导入 fuzzer 的生成器或 oracle。它们输出观察结果；脚本退出 0 表示检查执行完毕，不表示编译器行为正确。**以下结果均在本次实际测试中观察到。**

### 4.2 Triton#1：绝对值仍保留负号

256 个负数先转成 fp16，执行 `tl.abs`，再转成整数后存储。例如 `[-1, -2, -3, -4]` 应得到 `[1, 2, 3, 4]`，实测为 `[-1, 2, -3, 4]`。

```bash
python -B repro/triton_abs.py
```

本次结果：**128/256 个输出仍为负数**；同结构的 float32 对照 **0 个错误**。该机制已有相同 PTX 分别交给不同 ptxas 的因果定位；本次小样例验证错误现象和类型对照，没有重做全部汇编器版本比较。

源码：[triton_abs.py](repro/triton_abs.py)。输出：[triton_abs.py.txt](validation/triton_abs.py.txt)。

### 4.3 Triton#8：掩码加载的负填充值被错误打包

对 int8 数组执行 `tl.load(..., mask=r < 48, other=-2)`。被屏蔽行的 8 个元素应该全为 `-2`，实测一行为 `[-2, -1, -1, -1, -2, -1, -1, -1]`。

```bash
python -B repro/triton_masked_load.py
```

本次结果：**128 个被屏蔽元素中 96 个错误**。改为 int32、仍填 `-2` 时 0 错误；保持 int8、改填 `+2` 时也是 0 错误。这对应窄整数向量打包时没有正确重复填充位模式的机制。

源码：[triton_masked_load.py](repro/triton_masked_load.py)。输出：[triton_masked_load.py.txt](validation/triton_masked_load.py.txt)。

### 4.4 TileLang#22：bf16 的 pow 被当作非浮点类型拒绝

小核对 bfloat16 输入计算 `T.pow(X[i], Y[i])`。构建时类型检查报：

```text
Check failed: (x.dtype().is_float()) is false: power only applies to float
```

```bash
python -B repro/tilelang_pow_bf16.py
```

本次结果：**bf16 pow 被拒绝**；float32 pow 与 bf16 exp 两组对照均编译运行成功。问题位于类型检查没有把 bfloat16 纳入允许类型。

源码：[tilelang_pow_bf16.py](repro/tilelang_pow_bf16.py)。输出：[tilelang_pow_bf16.py.txt](validation/tilelang_pow_bf16.py.txt)。

### 4.5 TileLang#24：把浮点 x−x 恒定折叠为 0

输入 `[+inf, -inf, NaN, 1.5]`，计算 `X[i] - X[i]`。没有启用 fast math，预期 `[NaN, NaN, NaN, 0]`，本次输出却为 **`[0, 0, 0, 0]`**。

```bash
python -B repro/tilelang_self_subtract.py
```

3 个特殊值均错误，有限值对照正确。打印出的 CUDA 源码直接写入常量 0，输入参数已经消失，证明错误发生在表达式化简 / 降级阶段，而不是 GPU 对 NaN 的计算波动。

源码及生成 CUDA：[tilelang_self_subtract.py](repro/tilelang_self_subtract.py)、[输出](validation/tilelang_self_subtract.py.txt)。

### 4.6 TileLang#26：sm75 的 bf16 原子 min/max 错选 CUDA 模板

TileLang 0.1.15 的 `atomic.h` 在 sm75 上保留 `cutlass::bfloat16_t`，未进入现有的 16 位 CAS 分支，而是实例化通用 `cuda::atomic_ref`。CUDA 12.4 对此报 `atomic_ref does not support 1 or 2 byte types`。

在有 **CUDA 12.4** 的环境执行；将 `NVCC` 指向实际的 12.4 安装：

```bash
NVCC=/usr/local/cuda/bin/nvcc python -B repro/tilelang_atomic_compile.py
```

脚本只经 stdin 编译很小的 CUDA 内核，输出到 `/dev/null`，不启动 GPU 核函数。修正对照只替换内存中的头文件文本，不修改安装文件。

| 条件 | 本次结果 |
| --- | --- |
| 原始头文件，bf16 Min / sm75 | 编译失败，atomic_ref 静态断言 |
| 原始头文件，bf16 Max / sm75 | 编译失败，同一静态断言 |
| 原始头文件，fp16 Min / sm75 | 编译成功 |
| 原始头文件，bf16 Min / sm80 | 编译成功 |
| 让 bf16 进入现有 CAS 分支，Min / sm75 | 编译成功 |
| 同上，Max / sm75 | 编译成功 |

该问题依赖工具链组合；先前本地 CUDA 12.9 对照可以编译，不能期望任意 CUDA 版本都失败。本轮有 21 条，上轮 TileLang 0.1.14 日志也有 14 条同条件记录，因此是**本次补充定位的问题，并非本轮首次出现或已经证明的 0.1.15 回归**。

源码：[tilelang_atomic_compile.py](repro/tilelang_atomic_compile.py)。六组实测：[tl26_cuda124.json](validation/tl26_cuda124.json)。

## 5. Fuzzer 缺陷与误报对照

| 编号 | 缺陷 | 记录数 |
| --- | --- | ---: |
| Fuzzer#1 | 生成 gfx942/gfx950 的 f32 MFMA + kpack=2 组合，布局 k_dim 超出实现支持范围 | 21 |
| Fuzzer#2 | 在 HIP half/bf16 reduce 上开启仅支持 CUDA 的 nan_propagate=True | 30 |
| Fuzzer#3 | 在不支持该 async copy 路径的 gfx90a/gfx1100 上强开 cfg_amd=async | 29 |
| Fuzzer#4 | 浮点 oracle 未充分建模允许的乘加融合，正常结果差异被判错 | 7 |
| 合计 | | **87** |

前三类来自生成合法化缺口，对应 [gemm.py](../../../src/workflow/slices/gemm.py)、[reduce.py](../../../src/workflow/slices/reduce.py)、[runtime.py](../../../src/workflow/slices/runtime.py) 的目标 / 类型组合处理。它们不是新发现的编译器根因。

### 5.1 Fuzzer#4 的实际配对复现

样例 `7c40f5b1a54c5f71` 含 fp16 乘积相减和后续 scan。保留原输入、失败的 identity 变体、全部 seed/runtime case 和检查器，只改变 nvcc 的融合选项：

```bash
python -B repro/fz04_default.py
python -B repro/fz04_no_fma.py
```

本次第一条命令按预期 **退出 1**，报 `actual=0.00103759765625, expected=0.0`；第二条命令启用 `--fmad=false`，**退出 0，ALL PASSED**。脚本是保存程序的对照副本，体积较大，不是独立最小样例。

这里不能仅凭“改一个选项就通过”认定误报。已有六组 identity 控制的生成 CUDA 逐字节相同，半精度乘 / 减使用允许收缩的 `__hmul2` / `__hsub2`，差异发生在下游融合及结果；CUDA 提供禁止相应融合的 `_rn` 变体。允许的浮点融合与 oracle 的固定计算模型不一致，才是归类依据。[CUDA half2 算术说明](https://docs.nvidia.com/cuda/archive/12.3.2/cuda-math-api/group__CUDA__MATH____HALF2__ARITHMETIC.html)、[nvcc fmad 说明](https://docs.nvidia.com/cuda/archive/12.9.1/cuda-compiler-driver-nvcc/index.html#fmad-true-false-fmad)。

本次输出：[默认融合](validation/fz04_default.py.txt)、[禁用融合](validation/fz04_no_fma.py.txt)；原审计全部融合对照摘要：[fusion_controls.json](data/fusion_controls.json)。只将满足完整配对证据的 7 条归入 Fuzzer#4，不把其余数值报错批量当作误报。

### 5.2 自动标签的纠正

- `other:3bb67ac4a2` 同时包含 AMD atomic_xchg 与 IEEE fdiv；已拆分为 Triton#13、Triton#9。
- 半精度 atomic 的旧宽泛规则把 bf16 min/max 编译错误混入 atomic_add，现单列 TileLang#26。
- f32→fp8 的旧规则误包含 AMD rtz 断言和 sm80 fp8e5 软件转换，现分别归入 Triton#5、Triton#15。
- bf16 CAS 的 5 条同机制记录统一归入 Triton#14。
- 普通内建缺降级与 HIP IEEE 缺降级分为 TileLang#11、TileLang#17；流水线 fragment 的两种断言合并为 TileLang#23。

这些是报告中的逐记录校正，原监控规则和项目源码未修改。

## 6. 未确认与资源类记录

| 类型 | 记录数 | 处理 |
| --- | ---: | --- |
| oracle_unstable | 247 | 参考模型自己不稳定，跳过数值定罪 |
| unsupported_feature | 981 | 明确不支持的目标 / 类型 / 配置 |
| shared_memory_overflow | 116 | 共享内存超过硬件限制 |
| register_limit | 17 | 寄存器预算不足 |
| warp_partition | 11 | 线程或 tile 分块非法 |
| timeout | 20 | 未独立区分慢编译、慢执行或真正挂起 |
| gpu_oom | 1 | 显存不足 |
| 合计 | **1,393** | |

其中 101 条 unsupported_feature 是 HIP 线程块小于完整 wavefront，另有 11 条非法 warp_partition；它们说明合法化还可改善，但已在此表计数，不能再加入 87 条确认 fuzzer 缺陷。

另外 **8 条 sm90 + nvcc 12.4 段错误**没有独立最小根因，单列工具链候选。**19 条数值差异**保留待定，见 [numeric_pending.csv](data/numeric_pending.csv)。原审计重放中，一个样例转成 ORACLE UNSTABLE，另一个超过 120 秒；其 no-FMA 对照因本地 8 GiB 显存不足而未完成，不能据此判远端 bug 已修复或归罪于 fuzzer。

不开 fast math 不等于禁止 FMA；variant invariance 检查的直接比较也不能自动获得 structured-reference 稳定性保护。不同累加顺序、浮点融合及误差放大仍需逐条排除。

## 7. 与上一轮长跑的比较

上一轮从 10 月 12 日 00:45 / 00:47 开始，于 10 月 16 日 02:02 停止，墙钟约 97 小时，共 204,285 个测试。其 vGPU / Triton 的 `campaign_seconds` 为 94.86 小时，其余三路约 97.2 小时，不能混称四路有效运行时间完全一致。

按本次校正后的规则回查上轮 8,225 份失败 JSON，得到以下机制匹配统计。此次没有重新独立重放上轮每条记录。

| 范围 | TileLang 路径 | Triton 路径 | 合计 |
| --- | ---: | ---: | ---: |
| 上轮最终长跑 | 28 | 18 | **46** |
| 上轮截取至本轮对应的约 77.3 小时 | 28 | 18 | **46** |
| 本轮固定审计快照 | 26 | 17 | **43** |

本轮少 **3 类，约 6.5%**。两轮共有 41 类；本轮多触发了 TileLang#13（ldmatrix 列偏移）和 Triton#7（常量 fp8 的 ptxas 错误），上轮则多触发了以下 5 类：fp16 atomic_add 下标向量化、向量 bitwise_not、loop unswitching 破坏 SSA、maxnreg 下 dot 非法访存、AMD fp32 dot + kpack=2 断言。这两类本轮独有命中也曾在更早实验中确认，不能称为首次发现。

TileLang#26 在两轮中都有记录，计入双方，不增加“相对上轮首次出现”数量。历史累计总表的 51 类来自多轮实验和本地探测，不能当作上轮单独数量。两轮之间 TileLang 从 0.1.14 升到 0.1.15，生成器也有变化，因此这个计数差不能单独证明 fuzzer 效果退步或未触发的问题已修复。

可复核统计：[comparison.json](data/comparison.json)。

## 8. 设计检查与证据范围

四路均覆盖全部 10 个 slice；slice 路线占测试量约 85%，fresh、mutate、dsl_extend 也在推进。非 native 目标比例约 43.8%–54.3%，native + sm80 比例约 50.5%–62.7%。41790 / TileLang 全默认配置为 29.63%，略低于原 30% 观察下限，尚无某个切片停止采样的证据。

旧的非 HIP 目标携带 cfg_amd 问题、旧 TileLang 关键字 `k_pack/sync/group` 报错，在本次快照中均为 0；oracle_unstable 比例为 0.141%–0.191%。各路详细统计保存在 [audit_summary.json](data/audit_summary.json)。

本次工作只改写本地报告并增加复现材料，没有修改原项目源码、远端源码或实验结果，没有停止、重启或部署实验。远端额外执行仅为 TileLang#26 的少量 CPU 编译对照。

随报告提供的数据与测试证据：

- [43 类机制目录](data/bug_catalog.json)、[6,176 条分类](data/classification.csv)、[19 条待定数值记录](data/numeric_pending.csv)。
- [运行时间、检查点及版本](data/audit_summary.json)、[两轮比较](data/comparison.json)。
- [复现说明](repro/README.md)、[本次本地运行退出码](validation/local_runs.json)、[本次融合对照退出码](validation/fusion_runs.json)、[CUDA 12.4 六组编译结果](validation/tl26_cuda124.json)。

完整原始远端快照、批量重放产物和编译缓存保留在本地 `evidence/`；GitHub 提供本报告引用的精简数据、复现脚本与测试输出。测试输出的编号和文件名已统一成最终报告编号，数值、诊断、退出码及时间未改变，见 [validation/README.md](validation/README.md)。
