# TileSmith 实验结果报告

**日期**：2026-07-11  
**测试条件**：3张 GPU（RTX4060 / RTX4090 / A800），2个后端（TileLang / Triton），2种形状模式（easy-shape / hard-shape），各跑 seed=42

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

## 二、假阳性分析

以下两类触发均为假阳性，不反映真实的编译器 bug，排除在后续分析之外。

### 2.1 `wrong_result`

`wrong_result` 数量庞大（TileLang ~2,800–3,300 次/配置，Triton ~5,900–6,600 次/配置），经验证**全部为假阳性**，来源分三类：

**1. Oracle 参考实现语义错误（主因）**  
Dynamic sequence 中的复合算子，oracle 参考对整个矩阵做全局统计，但 kernel 实际在每个 tile 内做局部统计。tile 尺寸远小于矩阵尺寸时，局部与全局结果相差数倍乃至数十倍，相对误差远超阈值。这是 fuzzer 自身的 oracle 缺陷，不是编译器 bug。

**2. 浮点误差被条件算子放大**  
`WHERE` 算子对 GEMM 输出做正负判断，当结果在零附近时，Triton 与 PyTorch 对 K 维的不同累加顺序导致符号误差，WHERE 把该误差放大成 `max_diff ≈ |D2|`（可达 1.0 以上），远超阈值。

**3. 阈值对大 K 的 fp32 GEMM 过紧**  
fp32 GEMM 在 K 较大时，Triton 与 PyTorch 的累加误差稳定在 ~3%–5%，刚好超过 fp32 阈值 5%，属于纯浮点精度问题。

### 2.2 `timeout`

**仅在 RTX4060 上触发**（TileLang：easy-shape 9 次 + hard-shape 15 次；Triton：easy-shape 10 次 + hard-shape 13 次；RTX4090 / A800 均为 0 次），3 张卡合计 47 次。

经排查，这些超时均由**本地测试机多线程卡死**引起，并非 kernel 在 GPU 上发生死锁或真实挂起。RTX4060 所在的测试机在并发跑多个配置时，Python 多线程调度出现卡顿，导致超时计时器误触发。RTX4090 和 A800 运行在算力更强、调度更稳定的机器上，相同参数下不触发，进一步印证了这一判断。此类超时属于**测试环境噪声**，不是编译器 bug。

---

## 三、真实 Bug 汇总

去掉 `wrong_result` 和 `timeout` 后，共检出 **6 种根因**，分为 compile fail、runtime fail 以及工具健壮性问题三类：

| root_cause | 后端 | crash 时机 | 3张卡合计 |
|---|---|---|---|
| `dtype_mismatch` | TileLang | **runtime fail** | 3,644 |
| `ptx_async_boundary` | TileLang | **compile fail** | 275 |
| `shared_memory_overflow` | Triton | **runtime fail** | 174 |
| `warp_partition` | TileLang | **compile fail** | 164 |
| `shared_memory_overflow` | TileLang | **runtime fail** | 2 |
| `assertion_failure` | TileLang + Triton | **runtime fail** | 6 |
| `gpu_oom` | TileLang + Triton | **runtime fail** | 3 |

---

## 四、compile fail 详解

### 4.1 `warp_partition`

**crash 时机**：`tilelang.lower()` 内部的 Layout 推断阶段

**典型案例**：`results_RTX4060/.../failed/warp_partition/failed_pipeline_gemm+sqrt.py`

```python
# kernel 声明：block_M=16, block_N=16, threads=128（= 4 个 warp）
with T.Kernel(..., threads=128) as (bx, by):
    ...
    T.gemm(A_shared, B_shared, C_local)   # GEMM 算子触发 warp partition 推断
```

执行输出：
```
TileLang begins to compile kernel `impl`
  tilelang/jit/__init__.py compile
    JITKernel.__init__
      _compile_and_create_adapter
        tilelang.lower()
          LayoutInference pass
            GemmNode.InferLayout
              compute_warp_partition
InternalError: m_warp * n_warp must equal num_warps,
               m_warp: 1, n_warp: 1, num_warps: 4
```

TileLang 在 `LayoutInference` pass 中将线程数分解为 `m_warp × n_warp` 的 warp 网格。block_M=16、block_N=16 的组合下只能得到 `m_warp=1, n_warp=1`（共 1 个 warp），无法凑齐请求的 4 个 warp，编译期断言失败。

**本质**：TileLang 的 `ComputeDefaultWarpPartition` 对小 tile（如 16×16）没有合法的 fallback，tile 尺寸无法支持所请求的 warp 数量时，编译器直接断言失败。3 张卡合计 164 次，RTX4060 / RTX4090 / A800 均有触发，与形状模式无明显关联。

---

### 4.2 `ptx_async_boundary`

**crash 时机**：`tilelang.lower()` 内部的 PTX 代码生成阶段

**典型报错**：
```
_compile_and_create_adapter
  tilelang.lower()
    device_codegen → BuildTileLangCUDA
      CodeGenTileLangCUDA → GetTileLangCPAsyncTransferBytes
InternalError: tl::ptx_cp_async requires PTX byte width in {4, 8, 16}, but got 2
```

`cp.async` 是 PTX 异步内存拷贝指令，要求拷贝字节数必须是 {4, 8, 16} 之一。per-row 拷贝量 = `block_K × sizeof(dtype)`，当 `block_K` 取非标准对齐值（如 block_K=1，float16 时每行 2 字节）时，不满足约束，代码生成阶段断言失败。

**与形状模式的关系**：本次 easy-shape 下触发频繁（RTX4060：100 次，RTX4090：72 次，A800：99 次），hard-shape 下反而极少（各卡约 1–2 次）。原因在于 `block_K` 采样范围扩大后，easy-shape 下也会出现非标准对齐尺寸（如 block_K=1），即使 K 本身是 2 的幂次也可能触发。该 bug 的根源是编译器代码生成对任意 `block_K` 尺寸缺乏预检。3 张卡合计 275 次。

---

## 五、runtime fail 详解

### 5.1 `dtype_mismatch`

**crash 时机**：kernel 编译成功后，第一次调用 `kernel(A, B)` 时

**典型案例**：`results_RTX4060/.../failed/dtype_mismatch/failed_dynamic_gemm+copy_f2g.py`

```python
# kernel 声明：dtype="float16"，accum_dtype="float32"
dtype = "float16"
accum_dtype = "float32"
...
C_local_1 = T.alloc_fragment((block_M, block_N), accum_dtype)  # fp32 accumulator
T.gemm(A_shared_1, B_shared_1, C_local_1)
T.copy(C_local_1, C[...])   # 将 fp32 accumulator 拷贝回 fp16 输出
```

执行输出：
```
C = kernel(A, B)             ← kernel 已编译好，这里是第一次调用
  tilelang/jit/kernel.py __call__
    tilelang/jit/adapter/tvm_ffi.py func
      executable(*tensor_list)
RuntimeError: kernel impl input A dtype mismatch, expected float32
```

**根本原因**：kernel 声明 `dtype="float16"`，但 TileLang 在处理 `accum_dtype="float32"` 的 fragment buffer 时，类型推断将某个 buffer 的期望类型推导成了 float32，与外部传入的 float16 tensor 不一致。编译阶段未检测到这个矛盾，直到 JIT adapter 在执行前做参数类型检查时才发现。

**触发规律**：凡是同时使用 `dtype` 和 `accum_dtype` 的 kernel（GEMM 累加器为 fp32，输入输出为 fp16）几乎全部触发，说明是 TileLang 类型推断在混合精度 kernel 中的系统性缺陷。3 张卡合计 3,644 次，是本次触发最多的真实 bug。

---

### 5.2 `shared_memory_overflow`

**crash 时机**：kernel 编译成功后，第一次调用执行时 Triton 初始化 GPU handle

**典型报错**：
```
kernel_0_kernel[grid](...)
  triton/runtime/jit.py run
    triton/compiler/compiler.py _init_handles
      raise OutOfResources(...)
triton.runtime.errors.OutOfResources:
  out of resource: shared memory, Required: 110592, Hardware limit: 101376
```

Triton kernel 编译本身成功，`_init_handles` 在首次实际执行时向 GPU 查询硬件限制，此时才发现申请的共享内存（108 KB）超过了 GPU 上限（99 KB）。

**跨卡频次规律**：本次排列为 RTX4090 > RTX4060 > A800（RTX4090 easy+hard=86，RTX4060 easy+hard=59，A800 easy+hard=29）。A800 单 SM 共享内存最大，相同参数在 A800 上不越界；RTX4090 和 RTX4060 上限相近，因此两者频次接近。

TileLang 后端也在 RTX4090 hard-shape 下出现 2 次 `shared_memory_overflow`，说明 TileLang 同样存在类似的运行时内存检查时机问题，但频率远低于 Triton。

---

## 六、工具健壮性问题

以下两类触发不是编译器 bug，而是暴露了 fuzzer 自身的工程缺陷。

### 6.1 `assertion_failure`

3 张卡合计 6 次，全部集中在 RTX4060（TileLang easy/hard 各 2 次，Triton easy/hard 各 1 次），RTX4090 和 A800 均为 0 次。

**典型错误信息**：
```
RuntimeError: CUDA error: CUDA-capable device(s) is/are busy or unavailable
```

实质是：**前一个 kernel 崩溃（如 `warp_partition`）使 GPU 进入异常状态，后续的 `torch.randn(..., device='cuda')` 调用无法获取 CUDA 句柄，间接触发此类报错**。这是 crash 的级联效应，不是独立的新 bug。它暴露的问题是 fuzzer 在 kernel 崩溃后缺少有效的 GPU 状态恢复机制，导致后续测试受到污染。

### 6.2 `gpu_oom`

3 张卡合计 3 次，全部在 RTX4060（TileLang hard-shape 1 次，Triton hard-shape 2 次），RTX4090 和 A800 为 0 次。

**典型案例**：`M=6528, N=9414, K=9399, dtype=float32`（dynamic sequence 包含 9 个算子）

```
RuntimeError: CUDA error: CUBLAS_STATUS_ALLOC_FAILED when calling `cublasCreate(handle)`
```

这是 **oracle 参考实现**（PyTorch 的 cuBLAS GEMM）在分配极大矩阵时耗尽了 RTX4060 的 8 GB 显存，而不是被测 kernel 本身的问题。RTX4090（24 GB）和 A800（80 GB）显存充裕，相同参数不触发。修复方向是在 fuzzer 前端根据 GPU 显存对矩阵尺寸组合加上上界约束。

---

## 七、与论文 Bug 分类的对应关系

参考论文：*Characterizing Real-World Bugs in Tile Programs for Automated Bug Detection*（ISSTA '26）

论文按**症状**将 301 个 bug 分为三类：Crash（58.14%）、Correctness issues（36.21%）、Performance Bottleneck（5.65%）。

| 实验 root_cause | crash 时机 | 论文分类 | 对应论文案例 |
|---|---|---|---|
| `warp_partition` | compile fail | Tile Mapping and Launch Bugs（6.31%） | Triton #5265（num_warps 断言失败） |
| `ptx_async_boundary` | compile fail | Memory Bugs（19.27%）+ Device-Specific Bugs（3.99%） | PTX 对齐约束，非对齐 block_K 触发 codegen 断言 |
| `dtype_mismatch` | runtime fail | Type and Operator Bugs（48.84%）— Data-Type Semantics | Apache TVM #14112（dtype 不匹配） |
| `shared_memory_overflow` | runtime fail | Memory Bugs（19.27%）— Resource Allocation | Triton `OutOfResources`；TileLang 动态共享内存分配失败 |

**主要发现**：

1. **`dtype_mismatch` 是数量最多的真实 bug**（3,644 次），对应论文中最大类 Type and Operator Bugs（48.84%）。混合精度 kernel（fp16 输入 + fp32 accumulator）几乎全部触发，说明 TileLang 在混合精度场景下的类型推断存在系统性缺陷。

2. **`warp_partition` 精确复现了论文引用的 Triton #5265**（`num_warps` 无法分解为合法 warp grid 导致断言失败），说明该类问题跨框架存在，TileLang 同样受影响。

3. **`ptx_async_boundary` 的触发条件比"奇数/质数 K"更广泛**：`block_K` 取任意非标准对齐值即可触发，根源在于编译器代码生成对 `block_K` 缺乏预检，easy-shape 下同样会出现。

4. **`shared_memory_overflow` 的跨卡频次差异**（RTX4090 > RTX4060 > A800）与三张卡的共享内存容量成反比，直接印证了论文中 Resource Allocation Bugs 与硬件参数强相关的论断。
