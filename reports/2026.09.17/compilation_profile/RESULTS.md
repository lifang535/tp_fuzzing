# 编译耗时实验结果

同一 IR、独立 DSL 编译缓存、新进程；GPU 驱动缓存保留。详细方法见 [EXPERIMENT.md](EXPERIMENT.md)。

以下均为墙钟秒数。阶段使用 exclusive 时间汇总，未重复叠加嵌套 pass。用户两个 fuzzer 同时运行，因此绝对时间和 GPU 检查耗时有并发噪声。

## 主试验：实际慢 mixed 用例

| 后端 / 缓存 | 次数 | 总时间均值 | 中位数 | 范围 | 布局推导 | NVCC | 其它编译 | 启动/导入 | 参考计算 | 执行和检查 | 产物 I/O |
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| tilelang / cold | 3 | 152.007 | 150.123 | 140.931–164.966 | 108.970 | 14.887 | 14.116 | 4.422 | 0.056 | 9.512 | 0.045 |
| tilelang / warm | 3 | 14.883 | 7.814 | 7.010–29.826 | 0.000 | 0.000 | 1.725 | 4.703 | 0.042 | 8.388 | 0.026 |
| triton / cold | 3 | 6.046 | 5.963 | 5.941–6.235 | 0.000 | 0.000 | 1.452 | 1.799 | 0.034 | 2.704 | 0.058 |
| triton / warm | 3 | 4.162 | 4.016 | 3.764–4.707 | 0.000 | 0.000 | 0.300 | 1.740 | 0.037 | 2.030 | 0.056 |

## 路线对照（每格一次，不能替代大样本吞吐基准）

| 用例 | 后端 | 缓存 | 总时间 | 布局推导 | NVCC | 其它编译 | 执行和检查 |
|---|---|---|---:|---:|---:|---:|---:|
| arithmetic_control | tilelang | cold | 36.285 | 11.805 | 13.745 | 3.540 | 4.583 |
| arithmetic_control | tilelang | warm | 4.194 | 0.000 | 0.000 | 0.353 | 0.997 |
| arithmetic_control | triton | cold | 4.107 | 0.000 | 0.000 | 0.716 | 1.713 |
| arithmetic_control | triton | warm | 2.737 | 0.000 | 0.000 | 0.274 | 0.966 |
| control_calls_control | tilelang | cold | 169.971 | 125.159 | 25.224 | 7.845 | 3.927 |
| control_calls_control | tilelang | warm | 6.706 | 0.000 | 0.000 | 1.190 | 1.354 |
| control_calls_control | triton | cold | 4.362 | 0.000 | 0.000 | 0.816 | 1.785 |
| control_calls_control | triton | warm | 4.239 | 0.000 | 0.000 | 0.390 | 2.110 |
| indexed_memory_control | tilelang | cold | 32.028 | 6.866 | 13.411 | 5.815 | 2.208 |
| indexed_memory_control | tilelang | warm | 3.940 | 0.000 | 0.000 | 0.300 | 0.910 |
| indexed_memory_control | triton | cold | 4.349 | 0.000 | 0.000 | 0.683 | 1.794 |
| indexed_memory_control | triton | warm | 3.971 | 0.000 | 0.000 | 0.431 | 1.154 |
| mixed_single | tilelang | cold | 37.830 | 26.995 | 3.591 | 3.418 | 0.777 |
| mixed_single | triton | cold | 3.535 | 0.000 | 0.000 | 0.576 | 1.037 |
| native_matched | tilelang | cold | 41.460 | 21.751 | 7.723 | 3.580 | 1.700 |
| native_matched | tilelang | warm | 4.741 | 0.000 | 0.000 | 0.163 | 1.192 |
| native_matched | triton | cold | 4.315 | 0.000 | 0.000 | 0.848 | 1.109 |
| native_matched | triton | warm | 3.315 | 0.000 | 0.000 | 0.424 | 0.611 |
| shape_matmul_control | tilelang | cold | 27.509 | 5.442 | 12.895 | 4.237 | 2.039 |
| shape_matmul_control | tilelang | warm | 3.883 | 0.000 | 0.000 | 0.386 | 0.709 |
| shape_matmul_control | triton | cold | 3.872 | 0.000 | 0.000 | 0.731 | 1.424 |
| shape_matmul_control | triton | warm | 2.728 | 0.000 | 0.000 | 0.280 | 0.710 |

## TileLang 最慢 pass（mixed 冷编译，按 exclusive 时间汇总）

| Pass | 每次完整程序平均耗时 |
|---|---:|
| `tilelang.pass.tl.LayoutInference` | 108.970 |
| `tilelang.pass.tl.LowerTileOp` | 5.034 |
| `tilelang.pass.tl.Simplify` | 1.208 |
| `tilelang.pass.tl.VerifyParallelLoop` | 0.924 |
| `tilelang.pass.tirx.Simplify` | 0.718 |
| `tilelang.pass.tl.LegalizeVectorizedLoop` | 0.385 |
| `tilelang.pass.tirx.RemoveNoOp` | 0.280 |
| `tilelang.pass.tl.ThreadSync` | 0.220 |
| `tilelang.pass.tl.FlattenBuffer` | 0.166 |
| `tilelang.pass.tl.LowerIntrin` | 0.151 |
| `tilelang.pass.tl.LegalizeNegativeIndex` | 0.148 |
| `tilelang.pass.s_tir.RenormalizeSplitPattern` | 0.116 |

## Triton 编译阶段（mixed 冷编译）

| 阶段 | 每次完整程序平均耗时 |
|---|---:|
| `triton.frontend`（exclusive） | 0.140 |
| `triton.ttir`（exclusive） | 0.037 |
| `triton.ttgir`（exclusive） | 0.173 |
| `triton.llir`（exclusive） | 0.417 |
| `triton.ptx`（exclusive） | 0.090 |
| `triton.cubin`（exclusive） | 0.195 |
| `triton.compile_total`（exclusive） | 0.400 |

## 状态与原始记录

| 实验 | 完成 | 成功 / 已尝试 |
|---|---|---:|
| [arithmetic_control](arithmetic_control/summary.json) | True | 4 / 4 |
| [control_calls_control](control_calls_control/summary.json) | True | 4 / 4 |
| [indexed_memory_control](indexed_memory_control/summary.json) | True | 4 / 4 |
| [mixed](mixed/summary.json) | True | 12 / 12 |
| [mixed_single](mixed_single/summary.json) | True | 2 / 2 |
| [native_control](native_control/summary.json) | True | 2 / 4 |
| [native_matched](native_matched/summary.json) | True | 4 / 4 |
| [shape_matmul_control](shape_matmul_control/summary.json) | True | 4 / 4 |

[逐次阶段数据 CSV](phases.csv) · [聚合 JSON](aggregates.json) · [实际 campaign 快照](campaign_snapshot.json) · [生成开销](generation.json)
