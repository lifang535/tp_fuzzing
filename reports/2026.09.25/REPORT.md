# 2026-09-25 审计：最新版 tilelang / triton 下的真实 bug

审计对象：四次 campaign，`2026.09.24-00.45` 与 `2026.09.24-00.46` 各 tilelang / triton。

机器（两台同 seed 同参数，生成流相同）：

| 机器 | GPU | 架构（sm，CUDA compute capability） |
|---|---|---|
| A | NVIDIA GeForce RTX 4090，24564 MiB | sm_89（Ada，cc 8.9） |
| B | NVIDIA vGPU-32GB，32760 MiB | sm_89（Ada，cc 8.9） |

环境：tilelang **0.1.14**、triton **3.8.0**、torch 2.4.0+cu124。
命令行四次相同：`main.py --backend <b> --seed 42 --function-min-count 1 --function-max-count 2 --function-call-prob 0.05 -n 100000 --no-save-artifacts`。

## tilelang

| 标签 | 现象（触发条件 → 判定依据） | 真实 bug | 次数 |
|---|---|---:|--:|
| `tilelang_codegen_error` | 编译期失败，不产出可运行 kernel：CUDA codegen / lowering 内部报错，消息含 `Cannot convert type boolxN to CUDA type`（向量化 bool 无法打印成 CUDA 类型）或 `ReduceOp cannot lower a layout where a source index depends on a thread-owned reduce segment`（reduce layout lowering 断言）。region 与 extended 程序都命中 | ✅ 是 | 415 |
| `ptx_async_boundary` | 编译期失败：`T.ptx_cp_async` 降到最终 PTX 时单次传输字节数不落在 {4, 8, 16}，断言 `IsValidCPAsyncTransferBytes(total_bytes)` 失败。由非整除的向量化 copy（int8 / 小尾块）触发 | ✅ 是 | 84 |
| `wrong_result` | 编译运行都成功，但输出与 fp64 参考不符，误差远超容差（不是 ULP 级噪声）→ 结果算错 | ✅ 是 | 75 |
| `layout_inference` | 编译期失败：layout inference pass 报 `no available layout found`，给不出某个 fragment 的 layout（`has_best` 断言） | ✅ 是（候选） | 32 |
| `schedule_mismatch` | 数值不变量违背：同一 kernel 只换线程划分（threads sweep，不改变数学）两次执行，结果应逐位相同却不同 | ✅ 是 | 19 |
| `layout_mismatch` | 数值不变量违背：同一 kernel 换输入 layout（contiguous / offset 等，源码里 stride 变了）后结果应一致却不同 → 索引计算 bug | ✅ 是 | 12 |
| `pass_config_mismatch` | 数值不变量违背：同一份 kernel 源码，只换 `@tilelang.jit(pass_configs=...)`（如 `tl.enable_async_copy`、`tl.disable_shared_memory_reuse` 等数值中性 pass）重新编译，结果应一致却不同 → 某个 pass 改了语义 | ✅ 是 | 1 |
| import `tilelang_callback_cuda_compile` 失败 | harness 缺陷：生成的 reproducer 脚本自己 import 一个 0.1.14 已删除的内部符号，脚本直接起不来 | ❌ | 878 |
| `shared_memory_overflow` | 生成程序要的 dynamic shared memory 超过设备上限（如 262144 B，硬件 101376 B/SM），生成侧没约束 | ❌ | 154 |
| `No valid warp partition for T.gemm ... M=16, N=32 ... 8 warps` | 生成侧漏洞：`block_M=16` 时 warp 划分本来就不可行（生成器自己的 `check_warp_partition` 能算出来，hard-shape 路径没调它） | ❌ | 2 |
| `timeout` / `gpu_oom` | 资源限制：编译或执行超时；显存被同机其他进程占用 | ❌ | 6 / 8 |
| `oracle_unstable` | 信任门：fp64 参考与自身/抖动副本相对误差 >1e-2（混沌 recurrence），主动跳过数值判定 | ❌ | 175 |
| `other`（region 侧，未定性） | 未查到具体报错归属 | ❓ | 54 |

## triton

| 标签 | 现象（触发条件 → 判定依据） | 真实 bug | 次数 |
|---|---|---:|--:|
| `wrong_result` | 编译运行都成功，但输出与 fp64 参考不符，误差远超容差 | ✅ 是 | 234 |
| `pass_config_mismatch` | 数值不变量违背：同一 kernel 换编译配置（`enable_fp_fusion` 开关、随机采样的 pass 组合）重编，结果应一致却不同。本类误差最夸张，`67108864.0` 对 `0.001` 容差 | ✅ 是 | 20 |
| `layout_mismatch` | 数值不变量违背：同一 kernel 换输入 layout（contiguous / offset 等）后结果应一致却不同 → 索引计算 bug | ✅ 是 | 20 |
| `schedule_mismatch` | 数值不变量违背：同一 kernel 只换 `num_warps` / `num_stages`，结果应逐位相同却不同 | ✅ 是 | 7 |
| `ASTSource` signature 用 int 键 | harness 缺陷：构造 `ASTSource` 时 signature 用 `0:'*fp16'` 这类整数键，triton 3.8 要求参数名字符串键，直接抛 `Signature keys must be string`，没有一个 extended 程序能跑到编译 | ❌ | 3591 |
| `shared_memory_overflow` | 生成程序要的 shared memory 超过设备上限（`OutOfResources: Required: 131072–262144, Hardware limit: 101376`），生成侧没约束 | ❌ | 172 |
| `timeout` / `gpu_oom` | 资源限制 | ❌ | 6 / 7 |
| `oracle_unstable` | 信任门主动跳过检查 | ❌ | 648 |
| `other`（region 侧，未定性） | 未查到具体报错归属 | ❓ | 13 |

## 真实 bug 举例

**tilelang `tilelang_codegen_error`** — 两个签名，region 与 extended 程序都有：

```
Cannot convert type boolx16 to CUDA type
Cannot convert type boolx8 to CUDA type
Check failed: (analyzer->CanProveEqual(projected_index, simplified_index)) is false:
  ReduceOp cannot lower a layout where a source index depends on a thread-owned reduce segment
```

前两个是 CUDA 打印器不认向量化 bool，第三个是 reduce 的 layout lowering 内部断言。代表程序：
`M=163 N=4673 K=1 block 256x128x128 threads=128 stages=4 fp16`、`M=4968 N=122 K=1240 block 16x256x32 threads=256 stages=1 fp16`。

**tilelang `ptx_async_boundary`** — cp.async 最终 PTX 字节宽度落在 {4,8,16} 之外：

```
Check failed: (IsValidCPAsyncTransferBytes(total_bytes)) is false:
T.ptx_cp_async(T.address_of(As[T.shift_right(thread_binding, 3) * 64 + ...
```

代表程序：`M=1 N=9282 K=10937 block 16x64x64 threads=128 stages=4 fp16`。

**两边的数值不变量违背** — 同一程序换 schedule / pass config / layout 后结果与参考或变体不一致，oracle gate（fp64 + 1-ulp jitter 自比对）已排除参考自身不稳定：

| 类别 | 最大 error / tolerance |
|---|---|
| triton `pass_config_mismatch` | `67108864.0 / 0.001`（2^26）、`841.6 / 0.1`、`6.89 / 0.001` |
| tilelang `schedule_mismatch` | `16.98 / 0.1`、`5.17 / 0.1`、`1.64 / 0.1` |
| tilelang `layout_mismatch` | `4.0 / 0.001`、`1.0 / 0.001`、`0.5 / 0.001` |
| triton `layout_mismatch` | `8.0 / 0.001`、`1.0 / 0.001` |
| triton `wrong_result` | `2.79 / 0.05`、`1.80 / 0.05` |
| tilelang `wrong_result` | `5.12 / 0.1`、`1.32 / 0.05` |

容差尺度：`elemwise_atol=1e-3` ≈ fp16 在 [1,2) 的 1 ULP，`region_rtol_fp16=0.10` / `fp32=0.05`。样本里 `error=0.001953125`(2 ULP)、`0.00390625`(4 ULP) 这类属容差边缘噪声，未计入上表。

**harness 缺陷造成 extended 轨道全部失败** — 本轮没有一个 extended 程序通过。triton 侧 100% 死在 `src/backends/triton/extended.py:194` 的 int 键 signature；tilelang 侧死在 `src/backends/tilelang/extended.py:328` 函数体顶层 import 已被 0.1.14 删除的 `tilelang_callback_cuda_compile`。tilelang 另有 146/67 条 extended 程序在 lowering 阶段就因**真实编译错误**退出（早于该 import 行），已计入上面的 tilelang 计数。
