# 工作流模块说明

本目录包含 TileSmith 的核心工作流组件。TileSmith 是一个**通用的 tile 程序模糊测试框架**——它在后端无关的抽象 IR 层生成和变异程序，然后通过不同的 emitter 翻译到具体后端（TileLang、Triton 等）。

**核心设计原则**：
- **IR 层**（`src/ir/`）定义计算语义，与任何具体后端无关
- **Emitter 层**（`workflow/emitter/`）将 IR 翻译为后端特定代码，是唯一接触后端 API 的地方
- **约束层**（`src/constraints/`）检查硬件限制，按后端分开实现
- 新增后端只需加一个 emitter + 约束文件，生成器/变异器/oracle 无需改动

支持的后端：
- **TileLang**：tile-level DSL，基于 TVM，使用 `T.gemm`/`T.copy`/`T.Parallel` 等原语
- **Triton**：OpenAI 的 tile-level DSL，使用 `tl.dot`/`tl.load`/`tl.store` 等原语
- **（可扩展）**：CUTLASS、Hidet 等其他 tile 编译器

按功能划分为 5 个子模块。

---

## 模块结构

```
workflow/
├── generator/     程序生成器
├── mutator/       变异引擎
├── emitter/       代码发射器
│   ├── tilelang/  TileLang 后端
│   └── triton/    Triton 后端
├── oracle/        测试预言机
└── fuzzer/        主循环调度
```

---

## 完整工作流

### 第一步：程序生成（generator/）

`ProgramGenerator.generate()` 按概率选择三种生成策略：

**策略 A — 单 op（30%）**
从 15 种 `ComputeKind` 中随机选一个（GEMM / COPY / SOFTMAX 等），用硬件约束过滤后生成 `TileKernel`。

**策略 B — 模板 pipeline（40%）**
从预定义的结构模板生成 `TilePipeline`，有两种子策略：
- GEMM epilogue：`GEMM → [0~2个 epilogue op] → [可选 terminal]`
- Elementwise chain：`COPY → [1~2个 elementwise op]`

**策略 C — 动态序列（30%，类 MLIRSmith）**
`DynamicSequenceGenerator` 参考 MLIRSmith 的 `TypedValuePool` 机制，不预定义结构，而是逐步决策：

```
初始化 TileValuePool（含输入 buffer A/B）
循环 3~8 步:
    1. 扫描所有 OpGen，找到当前 pool 状态下可用的 op
    2. 按权重随机选一个（未覆盖的 op 权重 +50，多样性引导）
    3. 生成 KernelStep，更新 pool 和 torch_ref
固定以 GEMM 开始
```

每个 buffer 携带 `torch_ref`（如 `"A.float() @ B.float()"`），随每步 op 动态更新，最终用于正确性验证。

**支持的 Op（共 16 种）：**

| 类型 | Op | 描述 |
|---|---|---|
| **基础内存** | `gemm`, `copy_g2s`, `copy_s2f`, `copy_f2g` | GEMM 累加、global↔shared↔fragment 搬运 |
| **逐元素** | `scale`, `exp`, `sqrt`, `elemwise_add`, `elemwise_mul`, `elemwise_max` | 标量乘、exp/sqrt、二元 add/mul/max |
| **归约** | `reduce_sum`, `reduce_max`, `softmax` | 按行归约（terminal op） |
| **嵌套结构** | `if_epilogue`, `double_pipeline`, `accumulate_reduce` | 类 MLIRSmith 的控制流和数据流嵌套 |

**三种嵌套结构（类比 MLIRSmith 的 scf.if / scf.for）：**

这些嵌套结构是**后端无关**的——由 IR 层定义语义，由各后端的 emitter 分别翻译为 TileLang / Triton 语法：

- **`if_epilogue`**（类比 `scf.if`）：对 tile 内每个元素做条件分支（`x > threshold ? path_A : path_B`）。测试编译器对**条件计算**的代码生成，如分支预测和掩码处理。
  - TileLang 实现：`T.Parallel` 中的 if/else
  - Triton 实现：`tl.where(condition, a, b)`

- **`double_pipeline`**（类比嵌套 `affine.for`）：两个**独立的 K 维度 GEMM 循环**，各自累加到不同 fragment，最后相加。测试编译器对**多个并发 pipeline 写同一输出 tile** 的正确性。
  - TileLang 实现：两套独立的 alloc_shared + Pipelined 循环 + gemm
  - Triton 实现：两段独立的 K-loop + tl.dot，结果 acc1 + acc2

- **`accumulate_reduce`**（类比 `scf.for` + reduce）：先对 tile 做行级 reduce（max 或 sum），再将结果**广播回 2D fragment**（如 `x[i,j] -= row_max[i]`）。这是 online softmax 和 layer normalization 的核心模式，测试 **reduce → broadcast 数据流**是否正确。
  - TileLang 实现：`T.reduce_max/sum` + `T.Parallel` 逐元素操作
  - Triton 实现：`tl.max/sum(axis=1)` + 广播减/除

---

### 第二步：变异（mutator/）

主循环 60% 概率从 `seed_pool` 取一个已通过的程序进行变异：

- **参数变异**：shape 换为 2^n / 2^n±1 / 质数 / 极端值，tile size 换合法值，dtype 切换，threads 切换
- **结构变异**：loop_kind 切换（pipelined ↔ serial），num_stages 调整，compute_kind 替换
- **边界变异**：`M = block_M * n + r`（r≠0），制造不整除边界，专门触发边界处理 bug
- **pipeline 专用**：增/删/替换 epilogue 步骤

变异后调用 `_enforce_constraints()` 确保参数满足硬件约束。

---

### 第三步：代码发射（emitter/）

将抽象 IR 翻译为可执行的 Python 代码字符串。同一个 IR 可以发射到不同后端：

| IR 类型 | TileLang | Triton |
|---|---|---|
| `TileKernel`（单 op） | `tilelang/emitter.py` | `triton/emitter.py` |
| `TilePipeline`（模板） | `tilelang/pipeline_emitter.py` | `triton/pipeline_emitter.py` |
| `DynamicSequence`（动态） | `tilelang/dynamic_emitter.py` | `triton/dynamic_emitter.py` |

每种发射器生成完整的 Python 文件，包含：
1. kernel 函数定义（调用编译器 API）
2. test 函数（创建 GPU tensor，执行 kernel，与参考实现对比）

**参考实现的一致性保证**：参考计算 `ref` 从 IR 语义自动推导，不是手写的：
- 单 op：`ops.py` 里每个 op 类写死（GEMM: `ref = A @ B`）
- pipeline：链式追踪（`ref = ref * alpha` → `ref = torch.exp(ref)` → ...）
- 动态序列：`TileBuffer.torch_ref` 随每步 op 动态更新

---

### 第四步：测试执行（oracle/）

在独立子进程中执行生成的代码，隔离崩溃：

```python
subprocess.run([python3, tmp_file], timeout=compile_timeout + execute_timeout)
```

错误分类（按 root_cause，后端通用 + 后端特定）：

**通用分类（所有 tile 编译器均可能触发）：**

| 分类 | 含义 | 真实 bug? |
|---|---|---|
| `wrong_result` | kernel 计算结果与参考实现不一致 | ✅ |
| `dtype_mismatch` | 编译器内部类型推断与用户声明不一致 | ✅ |
| `shared_memory_overflow` | tile 参数组合超出 GPU shared memory 限制 | ❌ 硬件限制 |
| `gpu_oom` | GPU 显存不足 (transient) | ❌ 环境问题 |
| `segfault` | 编译器 segfault | ✅ |
| `ptx_async_boundary` | 异步拷贝指令在边界 tile 产生非法字节宽度 | ✅ |
| `tilelang_codegen_error` | 编译器 codegen 内部 assertion 失败 | ✅ |

**TileLang 特定：**

| 分类 | 含义 |
|---|---|
| `warp_partition` | MMA warp 分区无法匹配 block 大小 |
| `layout_inference` | layout inference 找不到合法的内存布局 |
| `alignment` | block 大小不满足 MMA 对齐要求 |

**Triton 特定：**

| 分类 | 含义 |
|---|---|
| `dtype_unsupported_op` | 算子不支持指定类型（如 tl.sqrt 不支持 fp16） |
| `triton_compile_error` | Triton compiler 报错 |

---

### 第五步：结果保存与 Resume（fuzzer/）

```
results/{日期-时间}_{backend}_{easy/hard-shape}_seed={seed}/
├── passed/
│   ├── passed_{op_label}.py       可复现的通过代码
│   └── passed_{op_label}.json     元信息
├── failed/
│   └── {root_cause}/
│       ├── failed_{op_label}.py   可复现的失败代码
│       └── failed_{op_label}.json 元信息（含完整错误信息）
└── summary.json                    累计统计汇总（跨所有 session）
```

**op_label 命名规则**：单 op 用类型名（`gemm`），多步序列用 `+` 连接（`gemm+exp+softmax`）。  
文件名相同时覆盖写，因此磁盘文件数 ≤ 触发次数。

**Resume 机制**：`--resume <dir>` 恢复已有实验，核心步骤：

1. **配置校验**：解析目录名，检查 backend / easy-shape / seed 与当前命令行参数一致，不一致立即报错
2. **重建 `tested_configs`**：遍历 `passed/` 和 `failed/` 下所有 `.json`，用 `_make_sig_from_dict` 还原 sig（与运行时 `_make_sig` 格式完全一致），加入去重集合
3. **恢复 `known_root_causes`**：先从 `failed/` 目录文件数统计，再用 `summary.json` 的触发次数覆盖（`max(文件数, summary值)`），确保 dup bug 的计数不丢失
4. **恢复 `total_tested`**：取 `max(文件总数, summary["total_tested"])`，因为去重跳过的 case 不写文件但计入测试数
5. **继续运行**：新测试写入同一目录，session 结束后将累计统计写回 `summary.json`

**`summary.json` 字段语义**：
- `bugs_total` = `sum(root_causes.values())`，触发次数之和
- `bugs_unique` = `len(root_causes)`，不同 root_cause 类型数
- `root_causes` 存触发次数（不是文件数），是 resume 时恢复 `known_root_causes` 的权威来源

---

## 去重与 pool 轮换

**去重**：相同 `(compute_kind, M, N, K, block, dtype, loop, stages)` 的程序只测一次，避免重复编译相同 kernel。

**dim_pool 轮换**：每 `pool_rotation_interval`（默认 100）次迭代，重新随机化 dim_pool。避免长时间运行后 M/N/K 的取值空间被耗尽，显著降低重复率（实测从 63% 降到 4%）。

---

## 与 MLIRSmith 的对应关系

| MLIRSmith 组件 | TileSmith 对应 |
|---|---|
| `TypedValuePool` | `TileValuePool`（`ir/dynamic_seq.py`） |
| `RegionGen.apply()` | `DynamicSequenceGenerator.generate()` |
| `OpGenerator` × 200+ | `OpGenBase` 子类 × 13（`ir/dynamic_seq.py`） |
| `DiversityCriteria` | diversity boost（未覆盖 op 权重 +50） |
| `config.h` / `OpConf` | `config/config.py` |
| 模板 JSON + 实例化 | `TilePipeline` + `PipelineGenerator` |
| 只检测 crash | crash + 正确性验证（differential testing） |
