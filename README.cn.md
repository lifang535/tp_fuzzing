# TileSmith — 面向 Tile 程序的结构感知模糊测试工具

TileSmith 是一个专为 Tile 程序编译器（TileLang、Triton）设计的模糊测试工具，
灵感来源于 MLIRSmith 的两阶段生成方法（结构模板 + 参数实例化）。

---

## 目录结构

```
tp_fuzzing/
├── main.py                    # 入口程序
├── src/
│   ├── config/                # 超参数集中配置
│   │   └── config.py
│   ├── ir/                    # 抽象表示层（IR）
│   │   ├── ir.py              # 基础数据结构（TileKernel, ComputeKind 等）
│   │   ├── pipeline.py        # 多步骤流水线 IR
│   │   └── dynamic_seq.py     # 动态序列 IR（仿 MLIRSmith TypedValuePool）
│   ├── constraints/           # 硬件约束检查
│   │   └── constraints.py
│   ├── ops/                   # 算子注册表（每个 ComputeKind 对应一个类）
│   │   └── ops.py
│   └── workflow/              # 模糊测试工作流
│       ├── generator/         # 程序生成器
│       ├── mutator/           # 变异引擎
│       ├── emitter/           # 代码生成器（TileLang / Triton）
│       │   ├── tilelang/
│       │   └── triton/
│       ├── oracle/            # 测试预言机（执行 + 检测 bug）
│       └── fuzzer/            # 主模糊测试循环
```

---

## 快速开始

```bash
# 使用默认参数运行（100 次迭代，TileLang 后端）
python main.py

# 指定迭代次数和随机种子（可复现）
python main.py -n 500 --seed 42

# 打印生成的代码（不执行）
python main.py --dump --seed 42

# 列出所有支持的算子类型
python main.py --list-kernels

# 使用 Triton 后端
python main.py --backend triton -n 200

# 指定输出目录
python main.py -o /tmp/fuzz_results -n 1000

# 使用 easy-shape 模式（只生成 2 的幂次方大小的 shape）
# 效果：pass 率约提升 14%，适合验证 fuzzer 本身或构建干净 seed 语料库
python main.py --easy-shape -n 200

# 对比两种模式的 pass 率
python main.py --seed 42 -n 100 -o results/normal
python main.py --seed 42 -n 100 --easy-shape -o results/easy

# 恢复上一次未完成的实验（继续写入同一个结果目录）
python main.py --resume 2026.06.29-16.41_tilelang_easy-shape_seed=42 -n 200 --seed 42 --easy-shape
```

---

## 核心设计

### 三种程序类型

| 类型 | 比例 | 描述 |
|------|------|------|
| `TilePipeline` | 40% | 基于模板的多步骤流水线（GEMM + epilogue） |
| `DynamicSequence` | 30% | 基于 TypedValuePool 的动态序列（仿 MLIRSmith） |
| `TileProgram` | 30% | 单算子程序 |

### 支持的算子（共 15 种）

- 矩阵乘法：`gemm`
- 内存操作：`copy`
- 逐元素：`add`, `mul`, `max`, `sub`, `scale`, `exp`, `sqrt`, `where`
- 转置：`transpose`
- 规约：`reduce_sum`, `reduce_max`, `reduce_min`
- 复合：`softmax`

### Bug 分类

工具自动将发现的 bug 分类为 10 种：

| 分类 | 含义 |
|------|------|
| `wrong_result` | 计算结果与参考不一致 |
| `dtype_mismatch` | 编译器内部类型推断与声明不一致 |
| `warp_partition` | warp 分区无法满足 block 大小 |
| `shared_memory_overflow` | shared memory 超出硬件限制 |
| `layout_inference` | TileLang layout inference 找不到可用布局 |
| `dtype_unsupported_op` | 算子不支持指定类型（如 tl.sqrt 不支持 fp16） |
| `codegen_duplicate_arg` | 代码生成的 kernel 参数重复 |
| `triton_compile_error` | Triton 编译阶段报错 |
| `segfault` | 编译器 segfault |
| `other` | 其他未分类错误 |

> **说明：**
> - `wrong_result` 不一定意味着编译器存在真实 bug。对于链式计算（如多步 pipeline 或动态序列），浮点运算的累积误差可能导致结果与参考实现存在细微差异，从而被误判为错误。
> - `shared_memory_overflow` 等硬件约束类错误，部分情况下是由于运行环境无法准确获取 GPU 硬件信息（如 shared memory 容量），导致约束检查阶段使用了不准确的上限，生成了实际上超出当前硬件限制的 kernel。

---

## 输出结构

```
results/
└── 2026.06.26-10.30_tilelang_hard-shape_seed=42/
    ├── summary.json                              # 统计摘要（累计跨所有 session）
    ├── passed/
    │   ├── passed_single_gemm_M128,N256,K64,bM64,bN128,bK32,t128,pipelined,s2,float16.py
    │   ├── passed_pipeline_gemm+scale+add_M512,N512,K128,bM64,bN64,bK32,t128,serial,s1,float16.py
    │   └── passed_dynamic_gemm+exp+copy_f2g_M256,N128,K64,bM32,bN64,bK16,t128,pipelined,s2,float16.py
    └── failed/
        └── {root_cause}/
            ├── failed_single_gemm_M128,N256,K64,bM64,bN128,bK32,t128,pipelined,s2,float16.py
            ├── failed_pipeline_gemm+where_M512,N512,K128,bM64,bN64,bK32,t128,serial,s1,float16.py
            └── failed_dynamic_gemm+sqrt+mul_M256,N128,K64,bM32,bN64,bK16,t128,pipelined,s2,float16.py
```

文件名命名规则：`{passed/failed}_{type}_{ops}_{params}`

- 参数格式：`M{m},N{n},K{k},bM{block_M},bN{block_N},bK{block_K},t{threads},{loop_kind},s{num_stages},{dtype}`
- 单算子：`{passed/failed}_single_{op}_{params}`
- 模板 pipeline：`{passed/failed}_pipeline_{op1}+{op2}+..._{params}`
- 动态序列：`{passed/failed}_dynamic_{op1}+{op2}+..._{params}`

只有程序结构和所有输入参数完全一致时才视为同一测试用例，不同参数的测试不会互相覆盖。

`summary.json` 格式：

```json
{
  "backend": "tilelang",
  "total_tested": 2011,
  "bugs_total": 940,
  "bugs_unique": 4,
  "root_causes": {
    "wrong_result": 859,
    "shared_memory_overflow": 2,
    "warp_partition": 5,
    "dtype_mismatch": 74
  }
}
```

- `bugs_total`：所有 root_cause 触发次数之和（`sum(root_causes.values())`）
- `bugs_unique`：不同 root_cause 类型数（`len(root_causes)`）
- `root_causes`：每种 root_cause 的触发次数（包含未保存到文件的 dup）

## Resume 机制

`--resume` 将新测试写入已有结果目录，接续上一次中断的实验：

```bash
python main.py --resume 2026.06.29-16.41_tilelang_easy-shape_seed=42 \
               -n 1000 --seed 42 --easy-shape
```

- `--backend`、`--easy-shape`、`--seed` 必须与目录名一致，否则报错
- 从 `passed/` 和 `failed/` 目录重建已测试集合，避免重复测试
- 从 `summary.json` 恢复每个 root_cause 的精确触发次数（包含未写文件的 dup）
- 若 `summary.json` 不存在（上次中断过早），则从文件数推断，不影响继续运行
- 所有统计数字（`total_tested`、`bugs_total` 等）在 session 结束后累计写回 `summary.json`

---

## 配置

所有超参数集中在 `src/config/config.py` 的 `Config` 数据类中，包含详细注释。
常用配置项：

```python
Config(
    seed=42,              # 程序生成随机种子（None 表示不固定）
    input_seed=0,         # 张量输入随机种子，写入每个生成的复现文件
    backends=["tilelang"],# 目标后端
    output_dir="results", # 输出目录
    compile_timeout=60,   # 编译超时（秒）
    execute_timeout=60,   # 执行超时（秒）
)
```


## 正确性与回归验证

- `--input-seed` 控制 PyTorch 张量输入，独立于程序生成用的 `--seed`。新结果会保存该设置；恢复新格式结果时必须匹配。旧结果目录仍可恢复，历史文件保持原样。
- 数值检查先验证 NaN/Inf 的位置与 Inf 符号，再比较有限值。参考值会按输出存储类型舍入；copy/transpose 还检查零的符号。
- 动态序列参考实现解释完整 buffer 数据流，保留填充 lane、tile 内归约和中间类型转换。动态去重包含操作属性和 buffer 身份；旧格式缺少这些信息的签名不会抑制新的精确签名。
- 恢复运行优先保留 `summary.json` 的累计计数。`bugs_unique` 是失败分类数，不是经过人工确认的独立编译器 bug 数。

```bash
# CPU 回归，包括参考语义、代码生成、去重和恢复运行
python -B -m unittest discover -s tests -v

# GPU 冒烟测试：两个后端、两种 dtype，每个用例使用独立 TileLang 缓存
python -B tests/gpu_smoke.py

# 可选：使用共享缓存，检查跨用例缓存行为
python -B tests/gpu_smoke.py --shared-cache
```

GPU 冒烟测试将复现文件及失败日志保存到打印出的 `/tmp/tilesmith_gpu_smoke_*` 目录，不写入 fuzzing 结果目录。共享缓存下观察到的 TileLang dtype mismatch 需要单独排查；独立缓存测试用于验证生成代码及参考语义。

## 论文导向的定向用例

默认每次全新生成有 20% 概率选择定向用例（其余按原 pipeline/dynamic/single 比例生成）；通过用例也进入种子池变异。`--probe-prob 1` 只生成定向用例，`--probe-prob 0` 关闭全新定向生成。已有定向种子仍可变异。

```bash
python main.py --backend triton --probe-prob 1 --seed 42 -n 100
python main.py --backend tilelang --probe-prob 1 --seed 42 -n 100
python -B tests/gpu_smoke.py --filter probe
```

- 新增 `argmax` 与融合 `gemm_argmax`，int32 输出，并列最大值取第一个索引。Triton 转置布局路径显式生成 `tl.dot(x, tl.trans(y))` 后接 `tl.argmax`。使用小整数输入避免浮点近似导致 argmax 的参考答案不稳定。
- copy、sum/max/min、softmax、argmax 定向测试 singleton、31/32/33、63/64/65、127/128/129 等边界。每行由一个完整 tile 处理，归约尾部按运算使用 0、负无穷或正无穷填充。
- 输入包含连续、转置、双步长和偏移布局。TileLang 使用一维物理 buffer 加显式地址表达式，覆盖地址计算，但不等同于测试任意 stride 的前端 buffer 描述符。
- 特殊值（NaN、Inf、正负零、次正规数）目前只进入 copy 的逐位检查；尚未覆盖这些值参与所有算术操作的语义。
- 每个定向用例默认比较 128/256 threads 两个配置，在同一组输入上各运行 3 次；检查参考值、重复执行逐位一致性、线程配置之间的一致性、输出前后各 16 元素保护区以及输入存储未修改。保护区不替代内存检查器，不能保证发现所有越界读写。
- 布局、输入模式、重复次数与配对开关写入 IR、去重键和保存文件，支持恢复。老格式结果仍可恢复；生成器扩展后不保证与旧版本产生相同的后续随机序列。

这些扩展增加了论文所述 bug 的触发空间，**不代表已复现全部 Triton/TileLang 历史 bug**。warp specialization、Hopper producer warpgroup 寄存器回收、AMD 指令调度、编译 pass 配对、重复编译 IR/缓存稳定性、性能回归、多维 launch 以及更多 dtype 仍未系统覆盖。
