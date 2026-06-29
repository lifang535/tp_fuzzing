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
    ├── summary.json                              # 统计摘要
    ├── passed/
    │   ├── passed_single_gemm.py                 # 单算子通过
    │   ├── passed_pipeline_gemm+scale+add.py     # 模板 pipeline 通过
    │   └── passed_dynamic_gemm+exp+copy_f2g.py   # 动态序列通过
    └── failed/
        └── {root_cause}/
            ├── failed_single_gemm.py             # 单算子失败
            ├── failed_pipeline_gemm+where.py     # pipeline 失败
            └── failed_dynamic_gemm+sqrt+mul.py   # 动态序列失败
```

文件名命名规则：
- 单算子：`{passed/failed}_single_{op}`，如 `passed_single_gemm`
- 模板 pipeline：`{passed/failed}_pipeline_{op1}+{op2}+...`，如 `failed_pipeline_gemm+softmax`
- 动态序列：`{passed/failed}_dynamic_{op1}+{op2}+...`，如 `passed_dynamic_gemm+exp+copy_f2g`

---

## 配置

所有超参数集中在 `src/config/config.py` 的 `Config` 数据类中，包含详细注释。
常用配置项：

```python
Config(
    seed=42,              # 随机种子（None 表示不固定）
    backends=["tilelang"],# 目标后端
    output_dir="results", # 输出目录
    compile_timeout=60,   # 编译超时（秒）
    execute_timeout=60,   # 执行超时（秒）
)
```
