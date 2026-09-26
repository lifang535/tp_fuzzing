# 服务器 32906：NVIDIA vGPU-32GB 审计结果

服务器：`connect.westc.seetacloud.com:32906`。项目目录：`/root/autodl-tmp/project/tile_program_fuzzing/tp_fuzzing`。

口径：2026-09-26 SIGINT 停止后的第四轮结果。两条实验均已停止；审计时 Git 提交见 stop_status.txt。

| 后端 | 结果目录 | 实际测试 | 通过 | 失败 | 参考不稳定 |
|---|---|---:|---:|---:|---:|
| tilelang | `2026.09.25-23.14_tilelang_hard-shape_seed=42` | 21110 | 2644 | 18402 | 64 |
| triton | `2026.09.25-23.14_triton_hard-shape_seed=42` | 23537 | 8830 | 14478 | 229 |

## 已确认签名记录

| 缺陷类别 | 本机保存记录数 |
|---|---:|
| TileLang bool 向量 CUDA 代码生成失败 | 95 |
| TileLang 自动归约布局 lowering 断言 | 80 |
| TileLang 异步拷贝生成非法 PTX 字节宽度 | 30 |
| Triton flip 默认 dim=None 编译失败 | 102 |

## 全量分类

| 分类 | 记录数 |
|---|---:|
| 已验证 DSL 错误签名 | 307 |
| 明确磁盘错误 | 19 |
| 诊断不足 | 32283 |
| 资源/配置相关，未计确认 | 124 |
| 已确认 fuzzer 判定缺陷 | 0 |
| 待定候选 | 147 |
| 参考不稳定（不含于 bugs_total） | 293 |

记录不等同于独立 bug；本机四类签名与另一台归并后仍是四类。具体代表复现的执行位置见 [证据说明](../evidence/README.md)。

本机完整分类见 `classification.csv.gz`，已确认签名的原始结果路径见 `confirmed_cases.csv`，停止时的 campaign 摘要见 `summary.json`。

明确磁盘错误集中在本机；其余缺少诊断的退出不能自动归因于磁盘。

[返回总报告](../REPORT.md)
