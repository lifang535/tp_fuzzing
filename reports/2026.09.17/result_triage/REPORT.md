# results 真实性审查

审查结论：这里确实有有效的编译器 bug，但 `failed/` 不能整体作为 bug 集合。已经确认 **6 类缺陷机制**：TileLang frontend 缓存忽略 dtype、fp16 尾部拷贝生成非法 cp.async、K=8 GEMM 列重复、shared-memory 同步缺失、Triton 编译器崩溃，以及 Triton fp16 负常量矩阵乘法产生 NaN。另有大量资源超限、oracle 异常和数值容差误报。

本报告基于 **2026-09-17 20:38:22 +08:00** 的快照，共 **2,295 条失败记录**。目录中的 campaign 仍在运行，之后新增的结果不在本次统计内。记录数不是独立 bug 数；同一机制可能产生数百份程序。

完整逐条结论在 [triage.csv](triage.csv)，包含路径、case ID、原始分类、审查分类、理由和原文件 SHA256。原始元数据快照为 [snapshot.json](snapshot.json)，统计为 [summary.json](summary.json)，本轮所有复现进程的退出码和耗时为 [replays.jsonl](replays.jsonl)。[代表样本索引](CASES.md) 提供到原始程序的直接链接。

## 统计与判定

| 原始分类 | 条数 | 本次判断 |
|---|---:|---|
| `dtype_mismatch` | 489 | **真实缓存缺陷机制**，不是生成了类型不匹配的调用；已做独立小程序对照 |
| `ptx_async_boundary` | 49 | **真实 lowering/codegen 缺陷机制**；代表原始脚本冷缓存复现 |
| `shared_memory_overflow` | 559 | **排除**：生成程序的资源预算不足，全部超过本机每 block shared-memory 上限 |
| `gpu_oom` | 56 | **排除**：全部发生在 PyTorch 参考解释器中 |
| `other` | 10 | **排除**：probe 自己的 `.view(torch.uint8)` 检查器异常 |
| `assertion_failure` | 1 | **排除**：实际是 CUDA OOM，自动分类错误 |
| `segfault` | 15 | **13 条 Triton 全部离线编译复现**；另外 2 条原日志是 NVCC 子进程崩溃，待缩减 |
| `nondeterminism` | 1 | **真实错误**；重复执行产生 Inf，显式同步后 80 次执行通过 |
| `schedule_mismatch` | 17 | 15 条是 fp16 GEMM K=8，同类错误机制已确认；其余 2 条需要排查数值容差 |
| `wrong_result` | 797 | 混合类别：204 条 fp16 GEMM K=8；3 条已核实是容差误报；1 条强数值不稳定嫌疑；1 条 Extended NaN 已缩减确认；其余 588 条未逐条确认 |
| `layout_inference` | 163 | **保留候选**；小尺寸合法操作组合能复现，但需缩减区分推导缺陷和不支持的布局组合 |
| `timeout` | 138 | **不能凭现有日志定性**，未证实 kernel hang |

保守地说：**至少 629 条可以排除为有效目标编译器 bug**，其中 626 条是资源/oracle 问题，3 条是具体核实的数值容差误报。另有 538 条属于已经验证机制的缓存/cp.async 家族，但并没有逐条重新执行这 538 个程序。直接重放确认的 13 条 Triton 崩溃、1 条 nondeterminism、1 条 schedule mismatch 和 1 条 Extended 错误共 16 条。其余记录保留明确的不确定性，不能用“总失败数减去误报数”当作真 bug 数。

## 可以保留的真实 bug

### 1. TileLang frontend cache 没有把全局 dtype 纳入缓存键

489 条记录的源程序中，module-level `dtype`、保存的 spec dtype 和可直接提取的 host 输入 dtype 没有发现冲突。错误消息却说缓存中的 kernel 期待另一种类型。

本轮用独立缓存和 32×32 copy kernel 做了三步对照：

1. fp16 首次编译、执行：通过。
2. 同一 JIT 函数源码，换成 fp32，复用缓存：报 `input A dtype mismatch, expected float16`。
3. 相同 fp32 程序换独立缓存：通过。

见 [fp16 日志](dtype16_first.log)、[同缓存 fp32 日志](dtype32_same_cache.log)、[独立缓存 fp32 日志](dtype32_fresh_cache.log)。最小程序为 [dtype32_same_cache.py](dtype32_same_cache.py)，运行顺序由 [controls.py](controls.py) 的 `dtype` 分支定义。

本机 `tilelang/jit/__init__.py` 的 `_frontend_cache_key_data`（约第 430 行）使用函数源码、签名、实参等信息，未包含这里引用的全局 dtype 的值。两个独立进程各自使用稳定、匹配的 dtype，仍发生碰撞。因此这不是“运行中随意修改全局变量”的无效测试，也不是 fuzzer 把 fp16 张量传给显式 fp32 签名。

应当按**一个缓存机制**归并；单独运行某份 `.py` 在空缓存下通过，并不能否定历史记录。

### 2. fp16 窄维度的普通拷贝被降成非法的 2 字节 cp.async

49 条记录全部满足 `dtype=float16` 且 `N=1` 或 `K=1`，诊断完全一致：`tl::ptx_cp_async requires a final PTX byte width in {4, 8, 16}, but got 2`。

代表 `6ffb52d51060569a`：M=N=K=1，tile=64×64×16，原始脚本在空缓存下稳定复现，见 [日志](6ffb52d51060569a.log)。程序使用高层 `T.copy`，并没有手工发出 2 字节 PTX 指令。小于 tile 的逻辑维度本身不是非法输入。

这属于自动异步拷贝选择/边界 lowering 的缺陷，应回退到合法拷贝或正确处理边界。[TileLang 语言文档](https://www.tilelang.com/programming_guides/language_basics.html) 描述了高层拷贝和安全访存处理；本结论还以实际保存的脚本及本机复现为依据。

### 3. fp16 GEMM 的 K=8 路径发生列重复，不能误判成“不合法的 block_K”

有 **204 条 wrong_result + 15 条 schedule_mismatch** 使用该配置。它们是优先归并/缩减的候选集合，并不代表 219 个独立 bug，也不保证每条都由相同机制导致。

本轮另行构造了不含参考解释器、控制流、转置或函数调用的简单 GEMM：M=N=K=32，tile=32×32×8。

| 对照 | 最大绝对误差 |
|---|---:|
| 随机输入，K tile=8，128 threads | 29.81209 |
| 同一输入，K tile=8，256 threads | 5.72e-6 |
| 同一输入，K tile=16，128/256 threads | 5.72e-6 |
| 全 1 的 A、每行是 0…31 的 B，K tile=8，128 threads | **256，精确整数计算仍错** |
| 同一整数输入，K tile=8，256 threads | **0** |

整数对照的正确首行为 `0,32,...,992`，错误输出把第 0–7 列复制到第 8–15 列，也把第 16–23 列复制到第 24–31 列。见 [最小整数复现](block_k_8_exact.py)、[日志](block_k_8_exact.log)、[随机对照](block_k_8.log)、[K=16 对照](block_k_16.log)。

本机 `cuda/intrinsics/macro/mma_macro_generator.py` 第 121 行明确使用 `k_dim=min(256 // bits, chunk)`，K=8 会选择 `m16n8k8`；生成的 [128-thread CUDA](block_k_8.py128.cuda) 也确实调用了 `mma_sync<...,16,8,8,...>`。这是可用的 fp16 指令形态，见 [NVIDIA PTX ISA](https://docs.nvidia.com/cuda/archive/11.1.0/pdf/ptx_isa_7.1.pdf)。不能套用其他 lowering 路径要求 K≥16 的约束，直接过滤这些结果。

原始 `b5c2b8bb73c6097d` 的复现也吻合：第一个调度逐元素等于参考，第二个调度在第 8 行起发生明显错误，最大差 1.314453125。见 [原始复现日志](b5c2b8bb73c6097d.log) 和 [逐调度观测](b5c2b8bb73c6097d_observed.log)。

### 4. shared-memory 转置附近的同步缺失

`28f7a4c7a8eda0b4` 使用 K tile=32，与上面的 K=8 问题不同。相同输入第一次可以完全正确，后续重复执行出现明显错误甚至 Inf；本次观测到 199 个元素变化。见 [观测日志](28f7a4c7a8eda0b4_observed.log)。这不是 signed zero、ULP 或普通浮点容差问题。

只在 `T.copy(arg1, v1_shared)` 前后各加一次 `T.sync_threads()`，其余算法和输入保持原样，**2 组输入 × 2 种调度 × 20 次重复，共 80 次执行全部通过**。见 [同步对照程序](28f7a4c7a8eda0b4_explicit_barrier.py) 和 [日志](28f7a4c7a8eda0b4_explicit_barrier.log)。

证据指向 shared-memory 同步插入/复用问题；具体是哪一个 pass 漏掉屏障还没有定位。普通 `T.copy` 的共享内存依赖应由编译器处理，这里也没有用户显式使用“不等待”的 `T.async_copy`。

### 5. Triton 在 TTGIR 优化阶段崩溃

快照中的 **13 条 Triton segfault 已全部复现**。重放使用原始 kernel、原始指针 dtype 和编译选项，调用 `triton.compile(..., target=GPUTarget('cuda',89,32))`，**没有分配原始大型输入，也没有执行 GPU kernel**。

13 条都以 SIGSEGV 退出，Python faulthandler 指向 `triton/backends/nvidia/compiler.py:189` 的 `make_ttgir` 中 `pm.run(mod)`。因此不是 GPU OOM、kernel 越界或 oracle 运算导致的进程崩溃。尚未定位到该 pass manager 内的具体 pass，也不能把 13 条称为 13 个独立根因。

批处理脚本为 [triton_compile.py](triton_compile.py)，代表日志为 [4abdeda508af655d_compile.log](4abdeda508af655d_compile.log)。完整样本 ID 可从 CSV 筛选 `backend=triton`、`original_root_cause=segfault` 获得。

另外两条 `12125edd1cbbc6f2`、`d55f8abb84d0887d` 的旧日志明确显示 **NVCC 编译子进程 segfault**，并含巨大的 `float[32768]` 局部数组。它们值得作为 NVCC/后端资源压力下的崩溃问题缩减，但本次没有重放，不能混算为上述 Triton 或直接的 TileLang 进程崩溃。

### 6. Triton 的 fp16 负常量矩阵作为 tl.dot 操作数时产生 NaN

原始 Extended case `25a869b78ca36590` 的四种编译/观测变体均失败，首元素实际为 NaN、参考为 0.544921875。进一步观测定位到第二次 matmul `e41=tl.dot(e40,e17,e33)`：e40 在 [-0.01220703125,0] 内，e17 全为 -0.125，e33 全为 0.125，三个操作数均与参考一致，输出却变成 NaN。见 [原始重放](25a869b78ca36590.log)、[中间值观测](extended_more_observations.log)。

已缩减到只有一个 16×16 GEMM 的 [最小程序](triton_constant_dot.py)，没有循环、别名访存或复杂 oracle：A 是幅度约 0.01 的有限 fp16 输入，B 使用 `tl.full((16,16),-0.125,tl.float16)`，fp32 accumulator 为 0.125。

| B 的构造方式 | 本轮结果 |
|---|---|
| kernel 内 fp16 常量 -0.125 | **256/256 个元素均 NaN** |
| kernel 内 fp16 常量 +0.125 | 通过，最大差约1.49e-8 |
| 从内存加载 fp16 的 -0.125 矩阵 | 通过，最大差约7.45e-9 |

见 [失败日志](triton_constant_dot.log)、[正常量对照](triton_positive_constant_dot.log)、[内存操作数对照](triton_memory_operand_dot.log)。这排除了输入不合理、fp16 溢出、运行时零次循环和参考解释器错误；证据指向 Triton 对 fp16 负常量 dot 操作数的代码生成。尚未定位具体 C++ pass 或指令打包代码。

## 应当排除或暂缓报告的结果

### 资源预算与检查器错误：626 条

- **559 条 shared_memory_overflow**：记录中请求的动态共享内存最小 103,424 字节，最大 1,082,880 字节，全部超过本机 sm_89 的每 block 99,328 字节上限。最常见请求是 262,144 字节。当前 `src/backends/tilelang/params.py::check_shared_memory` 只估算 GEMM A/B staging，没有覆盖转置 staging、其他临时区等全部开销。它们不能作为错误计算/非法编译的有效证据。
- **56 条 gpu_oom**：全部堆栈在 `_region_reference` / `_typed_region_reference` 的 PyTorch 运算中。参考实现为全部 tile 保留 SSA 张量，并计算分支，容易在 8GB GPU 上耗尽内存。应给 oracle 单独做峰值预算或分块参考计算。
- **1 条 assertion_failure**：`46168fe75af99982` 实际报 `CUDA error: out of memory`。后面的 `Compile with TORCH_USE_CUDA_DSA to enable device-side assertions` 是提示语，却被分类器的 assertion 规则先匹配到了。
- **10 条 other**：单列 broadcast view 的末维 stride=0。PyTorch 会将它视为 contiguous，所以 `.contiguous()` 没有改掉 stride；随后 `.view(torch.uint8)` 报错。这在 CPU 上即可复现，和目标 kernel 是否正确无关。检查时可先 flatten/materialize 再按字节比较。

### 数值 oracle 的误报与不稳定输入

| case ID | 本轮实际数值 | 判定 |
|---|---|---|
| `2432022d37c01733` | actual=3,893,128.25，ref=3,893,128.75，差0.5，约2个fp32 ULP | **误报**：用绝对容差0.001比较此量级不合理 |
| `8407027ddaa21060` | 输出约1.05–1.18百万，最大差0.5 | **误报**：同类绝对容差问题；第一调度已完成数值观测，第二调度编译超出本轮90秒限额 |
| `c6f65c66acaafb6f` | actual=0，ref≈±4.561e-5，被归一化成0.9785 | **误报**：接近零的结果仅使用相对尺度，缺少绝对容差 |
| `119930b0d5c760cc` | 原输入最大差3.3；仅把scale从0.1改为0.125后全部通过 | **强数值不稳定嫌疑**，暂不提交；这项对照本身还不足以严格证明最终根因 |

这些程序在语法和类型上可以合法，问题在于 oracle 把普通舍入差异或病态计算放大成 bug 信号。建议使用逐元素 `atol + rtol*abs(reference)` / ULP，并对接近零的分支、反复乘法/归约、溢出与非有限值传播单独分析。不能因为出现 Inf/NaN 差异就自动确认为编译器 bug，也不能统一放宽容差掩盖真正的列重复或未初始化读取。

特别注意：当前原生 Triton GEMM 的 `tl.dot` 没有指定 `input_precision='ieee'`，而参考是 `A.float() @ B.float()`；TileLang 本机 fp32 MMA 路径也会转换 TF32 操作数。深层非线性组合必须先核对这种计算语义和误差放大。

数值诊断副本会打印 `OBSERVED_FAILURE` 后继续收集两种调度，因此其末尾 `ALL PASSED` **不表示检查通过**。原始脚本的重放退出码和真正通过的控制实验分别保存在 `replays.jsonl`；不要把诊断脚本当作回归测试。

## 保留但尚未确认的候选

- **163 条 layout_inference**：代表 `2ffa308929b8d63e` 在空缓存下重现 `no available layout`，M=1、N=16、K=32、tile=32×32×32，操作序列没有 transpose，值得优先缩减。其余程序中可能存在直接 fragment 转置或其他互相约束的布局，不能据同一错误字符串整体定性。见 [代表日志](2ffa308929b8d63e.log)。
- **138 条 timeout**：Region 报告只有 `Execution timed out`，没有可靠的编译/执行阶段标记。已有仓库编译耗时报告也说明布局推导能很慢。此类证据不足以证明 deadlock；需要分阶段超时、保留最后执行阶段和编译日志。
- **其余 588 条 wrong_result**：没有逐条重放和缩减，可能同时包含真实错误、TF32/归约误差、病态数值和参考语义问题。CSV 明确标为待确认，不强行二分。

## 下一步优先级与方法边界

先保留 K=8 GEMM、shared-memory 同步、缓存键、cp.async、Triton SIGSEGV 和 fp16 负常量 dot 这六类，使用本报告的对照和最小程序进一步归并。给资源错误和 oracle 异常单独计数，不进入 compiler bug 指标；对 K=8 不应简单加过滤规则，因为本机实现确实支持该指令路径。修正数值 oracle 后再重放剩余 wrong_result，比继续累计失败数量更有价值。

本轮使用本机 TileLang **0.1.11**、Triton **3.0.0**、PyTorch **2.4.0+cu124**，GPU 为 RTX 4060 Laptop、sm_89、8GB。结论针对实际安装源码和保存的 repro，未验证其他版本是否已修复。并行运行的 campaign 可能影响耗时，不能把本轮耗时当作性能基准。

已读取全部失败 JSON，静态核对关键类别，并对快照中所有 2,294 个 Region 程序执行当前 IR validator，均通过；这只证明 fuzzer 自己的类型/作用域规则接受它们，**不能代替 DSL 合法性和数值语义审查**。单独的 Extended 程序另行分析。所有控制程序、快照及报告均写在此目录，原始 results 和生成器源码没有修改。
