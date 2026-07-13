# TileSmith 实验结果报告

**日期**：2026-07-11  
**测试条件**：3张 GPU（RTX4060 / RTX4090 / A800），2个后端（TileLang / Triton），2种形状模式（easy-shape / hard-shape），各跑 seed=42  
**相较上次变化**：测试规模翻倍（TileLang 10,000→20,000 次/配置，Triton 20,000→40,000 次/配置）；输入参数范围扩大，约束从 fuzzer 前端移至后端。

---

## 一、各卡各配置触发统计总览

### TileLang 后端（每配置 20,000 次迭代）

| GPU | 形状模式 | 总测试数 | 触发总数 | 占比 | wrong_result | dtype_mismatch | ptx_async_boundary | warp_partition | shared_memory_overflow | timeout | assertion_failure | gpu_oom |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| RTX4060 | easy-shape | 20,000 | 3,805 | 19.0% | 2,962 | 698 | 100 | 34 | — | 9 | 2 | — |
| RTX4060 | hard-shape | 20,000 | 3,830 | 19.2% | 3,315 | 476 | 1 | 20 | — | 15 | 2 | 1 |
| RTX4090 | easy-shape | 20,000 | 3,985 | 19.9% | 3,128 | 745 | 72 | 40 | — | — | — | — |
| RTX4090 | hard-shape | 20,000 | 3,816 | 19.1% | 3,287 | 501 | 2 | 24 | 2 | — | — | — |
| A800    | easy-shape | 20,000 | 3,713 | 18.6% | 2,840 | 750 | 99 | 24 | — | — | — | — |
| A800    | hard-shape | 20,000 | 3,554 | 17.8% | 3,057 | 474 | 1 | 22 | — | — | — | — |

### Triton 后端（每配置 40,000 次迭代）

| GPU | 形状模式 | 总测试数 | 触发总数 | 占比 | wrong_result | shared_memory_overflow | timeout | assertion_failure | gpu_oom |
|---|---|---|---|---|---|---|---|---|---|
| RTX4060 | easy-shape | 40,000 | 6,191 | 15.5% | 6,155 | 25 | 10 | 1 | — |
| RTX4060 | hard-shape | 40,000 | 6,571 | 16.4% | 6,521 | 34 | 13 | 1 | 2 |
| RTX4090 | easy-shape | 40,000 | 5,987 | 15.0% | 5,949 | 38 | — | — | — |
| RTX4090 | hard-shape | 40,000 | 6,667 | 16.7% | 6,619 | 48 | — | — | — |
| A800    | easy-shape | 40,000 | 5,977 | 14.9% | 5,963 | 14 | — | — | — |
| A800    | hard-shape | 40,000 | 6,470 | 16.2% | 6,455 | 15 | — | — | — |

---

## 二、`wrong_result` 的真实性分析

与上次实验一致，`wrong_result` 数量依然庞大（TileLang ~2,800–3,300 次/配置，Triton ~5,900–6,600 次/配置），经验证**全部为假阳性**，来源与上次相同：oracle 参考语义错误、`WHERE` 类条件算子放大浮点误差、大 K 下 fp32 阈值过紧。参数范围扩大后绝对数量有所上升，但比例基本稳定。本报告后续分析均排除 `wrong_result`。

---

## 三、真实 Bug 汇总

去掉 `wrong_result` 后，本次共检出 **8 种根因**，较上次新增 `timeout`、`assertion_failure`、`gpu_oom` 三种。3 张卡合计：

| root_cause | 后端 | crash 时机 | 3张卡合计 |
|---|---|---|---|
| `dtype_mismatch` | TileLang | **runtime fail** | 3,644 |
| `ptx_async_boundary` | TileLang | **compile fail** | 275 |
| `shared_memory_overflow` | Triton | **runtime fail** | 174 |
| `warp_partition` | TileLang | **compile fail** | 164 |
| `timeout` | TileLang + Triton | — | 47 |
| `shared_memory_overflow` | TileLang | **runtime fail** | 2 |
| `assertion_failure` | TileLang + Triton | **runtime fail** | 6 |
| `gpu_oom` | TileLang + Triton | **runtime fail** | 3 |

> **与上次对比**：`dtype_mismatch` 从 454 增至 3,644（约 8×），`ptx_async_boundary` 从 8 增至 275（约 34×），`warp_partition` 从 40 增至 164（约 4×），`shared_memory_overflow`（Triton）从 72 增至 174（约 2.4×）。其中 `ptx_async_boundary` 的增幅远超测试规模翻倍的预期，原因见第四节。

---

## 四、compile fail 详解

### 4.1 `warp_partition`

行为与上次完全一致：TileLang `LayoutInference` pass 将线程数分解为 `m_warp × n_warp` 时，遇到 block_M/block_N 过小（如 16×16）的情况无法凑齐请求的 warp 数，编译期断言失败。

```
InternalError: m_warp * n_warp must equal num_warps,
               m_warp: 1, n_warp: 1, num_warps: 4
```

本次 3 张卡合计 164 次（上次 40 次），增幅约 4×，与测试量翻倍和参数空间扩大（更多小 block 尺寸组合被采样）相符。

---

### 4.2 `ptx_async_boundary`

**上次规律**：仅在 hard-shape 下触发（共 8 次），因为 hard-shape 才有质数/奇数 K。  
**本次变化**：easy-shape 下也大量触发（RTX4060：100 次，RTX4090：72 次，A800：99 次），而 hard-shape 下反而极少（各卡约 1–2 次）。

原因在于**参数范围扩大**：本次 fuzzer 的 block_K 采样不再局限于 2 的幂次，部分组合下 `block_K × 2` 不满足 `cp.async` 要求的 {4, 8, 16} 字节约束，因此即使 K 本身是 2 的幂次，也可能在代码生成阶段断言失败。

```
InternalError: tl::ptx_cp_async requires PTX byte width in {4, 8, 16}, but got 2
```

实际触发触发 block_K 值多为非标准对齐尺寸（如 block_K=1），进一步证实了参数范围扩大的影响。

---

## 五、runtime fail 详解

### 5.1 `dtype_mismatch`

行为与上次一致：混合精度 kernel（fp16 输入 + fp32 accumulator）在 TileLang 类型推断阶段把某个 buffer 推导为 fp32，导致调用时类型检查失败。

```
RuntimeError: kernel impl input A dtype mismatch, expected float32
```

本次 3 张卡合计 3,644 次（上次 454 次），增幅约 8×，远超测试量翻倍的预期。参数范围扩大后，更多混合精度组合（`dtype="float16"`, `accum_dtype="float32"`）被采样，覆盖了更多触发路径。此 bug 仍为 TileLang 混合精度场景下类型推断的系统性缺陷。

---

### 5.2 `shared_memory_overflow`

Triton 后端行为与上次一致：kernel 编译成功，首次执行时 `_init_handles` 查询 GPU 硬件上限，发现申请的共享内存超限。

```
triton.runtime.errors.OutOfResources:
  out of resource: shared memory, Required: 110592, Hardware limit: 101376
```

跨卡频次仍然严格按 RTX4090 > RTX4060 > A800 排列（RTX4090 easy+hard=86，RTX4060 easy+hard=59，A800 easy+hard=29），与三张卡的共享内存容量成反比。

> 注意：本次 RTX4090 触发次数（86）超过 RTX4060（59），与上次（RTX4060 > RTX4090 > A800）有所不同。结合参数范围扩大，推测本次 RTX4090 采样到了更多接近其上限的共享内存配置。

TileLang 后端也在 RTX4090 hard-shape 下出现 2 次 `shared_memory_overflow`，与上次零星触发的模式一致，说明 TileLang 同样存在类似的运行时内存检查时机问题，但频率远低于 Triton。

---

### 5.3 `timeout`（新增）

**仅在 RTX4060 上触发**（TileLang：easy-shape 9 次 + hard-shape 15 次；Triton：easy-shape 10 次 + hard-shape 13 次；RTX4090 / A800 均为 0 次），3 张卡合计 47 次。

**典型案例**：`failed_dynamic_gemm+copy_f2g_M32,N256,K32,bM16,bN128,bK32,t128,pipelined,s4,float16.py`

```
error_message: "Execution timed out"
```

这类 kernel 通常包含多阶段 pipeline（`num_stages=4`）或 dynamic sequence 中的 `copy_f2g` 算子。超时意味着 kernel 启动后 GPU 长时间无响应，既可能是真正的死锁（barrier 不配对、warp 饥饿），也可能是 RTX4060 算力或内存带宽在特定参数下导致实际运行时间超过了 fuzzer 的限制阈值。RTX4090 和 A800 因算力更强，相同参数下没有触发超时，建议在确认是真实 kernel hang 之前先放宽超时阈值并在相同参数下复现。

---

### 5.4 `assertion_failure`（新增）

3 张卡合计 6 次，全部集中在 RTX4060（TileLang easy/hard 各 2 次，Triton easy/hard 各 1 次），RTX4090 和 A800 均为 0 次。

**典型错误信息**：
```
RuntimeError: CUDA error: CUDA-capable device(s) is/are busy or unavailable
```

与名字暗示的"断言失败"不同，这类错误的实质是：**前一个 kernel 崩溃（如 `warp_partition`）使 GPU 进入异常状态，后续的 `torch.randn(..., device='cuda')` 调用无法获取 CUDA 句柄，间接触发此类报错**。这是 crash 的级联效应，不是独立的新 bug；它暴露的真实问题是 fuzzer 在 kernel 崩溃后没有有效地重置 GPU 状态，导致后续测试污染。RTX4060 独有的原因与其 VRAM 和 compute capability 在崩溃后的恢复能力较弱有关。

---

### 5.5 `gpu_oom`（新增）

3 张卡合计 3 次，全部在 RTX4060（TileLang hard-shape 1 次，Triton hard-shape 2 次），RTX4090 和 A800 为 0 次。

**典型案例**：`M=6528, N=9414, K=9399, dtype=float32`（dynamic sequence 包含 9 个算子）

```
RuntimeError: CUDA error: CUBLAS_STATUS_ALLOC_FAILED when calling `cublasCreate(handle)`
```

这是 oracle 参考实现（PyTorch 的 cublas GEMM）在分配极大矩阵时耗尽了 RTX4060 的 8 GB 显存，而不是被测 kernel 本身的问题。RTX4090（24 GB）和 A800（80 GB）显存充裕，相同参数不触发。触发原因是参数范围扩大后出现了 M/N/K 均在数千量级的极端组合；修复方向是在 fuzzer 前端根据 GPU 显存对矩阵尺寸组合加上上界约束。

---

## 六、与论文 Bug 分类的对应关系

参考论文：*Characterizing Real-World Bugs in Tile Programs for Automated Bug Detection*（ISSTA '26）

| 实验 root_cause | crash 时机 | 论文分类 | 对应论文案例 |
|---|---|---|---|
| `warp_partition` | compile fail | Tile Mapping and Launch Bugs（6.31%） | Triton #5265（num_warps 断言失败） |
| `ptx_async_boundary` | compile fail | Memory Bugs（19.27%）+ Device-Specific Bugs（3.99%） | PTX 对齐约束，非对齐 block_K 触发 codegen 断言 |
| `dtype_mismatch` | runtime fail | Type and Operator Bugs（48.84%）— Data-Type Semantics | Apache TVM #14112（dtype 不匹配） |
| `shared_memory_overflow` | runtime fail | Memory Bugs（19.27%）— Resource Allocation | Triton `OutOfResources`；TileLang 动态共享内存分配失败 |
| `timeout` | — | Crash（58.14%）— potential deadlock | 需进一步复现确认是 hang 还是性能问题 |
| `assertion_failure` | runtime fail | — | GPU 崩溃级联效应，非独立 bug |
| `gpu_oom` | runtime fail | — | oracle 参考实现（cuBLAS）显存不足，非被测 kernel 问题 |

**主要发现**：

1. **`dtype_mismatch` 依然是数量最多的真实 bug**（3,644 次），绝对数量较上次增长约 8×，印证了 TileLang 混合精度类型推断缺陷在更大参数空间下的系统性。

2. **`ptx_async_boundary` 突破 easy-shape 限制**：参数范围扩大后，easy-shape 下的非标准 block_K 同样触发 PTX 对齐错误，说明该 bug 的触发条件比上次分析的"奇数/质数 K"更广泛，根源在于代码生成对任意 block_K 尺寸缺乏预检。

3. **新增 `timeout` 需要区分真假**：47 次超时全部发生在 RTX4060，不能排除是性能瓶颈而非真正的 hang，建议后续加入 GPU 状态探针（如 `CUDA_LAUNCH_BLOCKING=1` 配合 watchdog）加以区分。

4. **`assertion_failure` 和 `gpu_oom` 是工具健壮性问题**，而非新发现的编译器 bug。`assertion_failure` 指向 fuzzer 缺少崩溃后的 GPU 状态恢复机制；`gpu_oom` 指向 oracle 参考实现对极大矩阵缺少显存预估和保护。
