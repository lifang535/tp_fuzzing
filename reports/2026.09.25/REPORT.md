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

## 修复：最新版适配（分支 `tilelang-0.1.14-triton-3.8`，提交 `2fe7c95b`）

上面两张表里两个 harness 缺陷行（tilelang 的 callback import、triton 的 int 键 signature）已经修掉，并做了实测。

**根因（比审计时更精确）**

- **tilelang**：0.1.14 把 `tilelang_callback_cuda_compile` 从 `tilelang.engine.lower` 移到 `tilelang.cuda.backend`（同名同签名）。生成程序里这句 import 位于**各 variant 的编译循环之后、`make_launch` 之前**，所以它丢掉的是已经编译成功、本来能跑的程序；真正在 `tilelang.compile` 里就抛错的程序会被记成 `device_compile:<variant>`，走不到那行。修复方式是在生成程序里 `try: from tilelang.cuda.backend ... except ImportError: from tilelang.engine.lower ...`，旧版本继续走 fallback。
- **triton**：3.8 要求 `ASTSource` 的 signature 用**参数名字符串键**（`triton/compiler/compiler.py:69`，`Signature keys must be string`），整数键直接 TypeError。修复方式是用生成程序真实的参数名做键，3.0 同样接受。

**修复前的实测**（两台服务器，`tp_fuzzing_latest`，2026-09-24 23:39 启动、2026-09-25 23:00/23:14 停止；口径为磁盘上保存的 `.json` 个案数，与上表的"次数"口径不同）

| 机器 | 后端 | 通过 | extended 通过 | extended 失败 |
|---|---|---:|---:|---|
| A | tilelang | 2441 | **0** | 458 = 370 `other` + 88 codegen |
| A | triton | 8537 | **0** | 1499（全部 TypeError） |
| B | tilelang | 3850 | **0** | 745 = 611 `other` + 134 codegen |
| B | triton | 13684 | **0** | 2324（全部 TypeError） |

`other` 里 369/370（A）、610/611（B）就是那句 ImportError。codegen 那部分（88 / 134，例如 `boolx16` 的 PrintType 崩溃）是**真 bug**，在 import 之前就挂了——修完 import 它们仍会失败，不计入修复收益。

停止后落盘的 `summary.json`（运行自身的计数）与之一致，location 分布把两类失败分得更开：

| 机器 | 后端 | tested | passed | `other` 的 location |
|---|---|---:|---:|---|
| A | tilelang | 3266 | 2461 | 398 例中 375 例 `lowering:tilelang_7/8_ident/9_prec`，21 例 `tvm.error` |
| A | triton | 10612 | 8632 | 1517 例中 1513 例 `compile:triton_0` |
| B | tilelang | 5157 | 3910 | 640 例中 616 例同上 |
| B | triton | 16989 | 13912 | 2359 例中 2348 例同上 |

tilelang 的 `tilelang_codegen_error` 里落在 `device_compile:*` 的（A 88、B 137）就是上面那批真 bug。

**修复后的实测**（服务器 A，target pair，30 次迭代，与线上同参数）

- 8 个 extended fixture 变体在两台机器上 8/8 通过；
- 30 次迭代：tilelang 4 个 extended 通过、extended 失败 0；triton 10 个 extended 通过、extended 失败 0（另有 1 例 `oracle_unstable`，非 extended）。对照线上两条轨道都是 0；
- 同一份 harness 在旧组合（0.1.11 / 3.0.0）上 6 个 fixture 全 PASS，适配没有弄坏旧版本。

**顺带修掉的 harness 缺陷**

- **probe 按位比较**：退化 stride 的参考张量无法比较——torch 的 contiguity 检查跳过 size-1 维，`.contiguous()` 之后 `stride(-1)` 仍为 0，`view(uint8)` 抛 `stride(-1) must be 1 to view Float/Half as Byte`。线上被误判的个案：A 机 5 例（tilelang 1、triton 4）、B 机 12 例（tilelang 1、triton 11），也就是两台机器 triton 侧非 extended 的 `other` 全部。改为只在字节视图非法时才物化张量。
- **diagnostics 措辞**：tilelang 补 0.1.14 的 layout inference / warp partition 新措辞，triton 补 `PassManager::run failed` 一类；`'triton.compiler'` 保持点号形式不变，否则会把 harness 侧错误（如上面那个 signature TypeError）从 `other` 里挪走。
- **版本标记**：`src/backends/common/versions.py` + 启动横幅 + `summary.json` 的 `environment` / `target_versions`；resume 时环境漂移会告警。
- **两个候选旋钮实测后继续排除**：`tl.config_index_bitwidth` 在 0.1.14 上仍是"一击必杀"（MakePackedAPI：`impl variables (limit,) are used, but are not passed in as API arguments`，`make_packed_api.cc:1060`；0.1.11 为 :577），bfloat16 则根本不是改池子的事（`ir/ir.py` 的 DataType、`ir/extended.py` 的 DTYPES、triton 的 signature 表都不认），需要先扩 IR 白名单、emitter 与容差。

测试：分支 303 OK / main 282 OK，新增 21 个测试；4 个 fixture 的 emission digest 属故意更新（逐条 diff 过，其余 23 条字节一致）。

## 部署与验收（2026-09-25 23:00–23:15）

- 分支 `tilelang-0.1.14-triton-3.8`：本地 `2fe7c95b` → `17984092` → `e23f6810`，已 push 到 origin。两台服务器都连不上 github.com（`git ls-remote` 与 `curl` 均超时），所以按 patch 应用：16 个文件的 sha256 清单在本地与两台机器上逐条一致（清单 md5 `413e56fc3ffafd3631c08f737b77e76d`）；服务器侧提交是 patch 复刻，哈希不同（A：`ad597aab` + `beaa67b6`，B：`4ee2ba63` + `85d6d796`）。服务器要归位到远端分支：`git fetch origin && git reset --hard origin/tilelang-0.1.14-triton-3.8`。
- 旧 campaign 用 `run_fuzzers.sh stop` 正常停止：SIGINT 送达 worker，`finally` 落盘，`summary.json` 完整（上面第二张表就来自它）。
- 新 campaign：A 于 23:10、B 于 23:14 启动，参数不变，新结果目录 `results/2026.09.25-23.10_*`、`results/2026.09.25-23.14_*`。
- 验收：重启后 **20 秒内**（A）、**60 秒内**（B）出现第一批 extended 通过。A：tilelang 12 通过 / 6 extended / 1 例真 codegen 失败，triton 40 通过 / 13 extended / 0 失败；B：tilelang 4 通过 / 1 extended / 1 例真 codegen 失败。对照修复前 23 小时 0 例。
- 部署树上两台机器各 303 tests OK，6 个 fixture（含 2 个 probe）全 PASS。

## dtype_mismatch 类在 0.1.14 上已不可达

`tests/test_dtype_mismatch.py` 钉住的是前端缓存碰撞：tilelang 0.1.11 的 `jit/__init__.py:_frontend_cache_key_data` 用 `inspect.getsource(impl)` 做键，而 region emitter 故意把 dtype 绑在 module 作用域（源码里看不到），于是两个 dtype 共用一个缓存条目、第二个程序拿到第一个的 kernel。0.1.14 删掉了这个方法，把 kernel cache 的键换成 `func.script(show_meta=True)` 的哈希（`cache/kernel_cache.py:_generate_key`）——解析后的 TIR 里 dtype 是具体的 buffer 类型，两个 dtype 不再共用条目。

实测：0.1.14 上第二个程序 returncode 0、且没有任何 dtype 报错（旧断言失败）；在未打补丁的 `ed7fb818` 上同样失败，所以这是既有差异、不是本轮改动引入的。测试改为按安装版本选择断言：0.1.11 及更旧继续断言碰撞，0.1.14 及更新改为断言修复（不跳过），上游若重新引入就会在这里报错。前提条件仍然成立——`test_impl_source_is_dtype_insensitive` 在 0.1.14 上照样通过，dtype 依旧不在 jit 源码里，变的只是缓存键。
