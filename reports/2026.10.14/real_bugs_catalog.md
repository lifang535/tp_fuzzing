# TileSmith 发现的真实 bug 总表（截至 2026-10-15）

本表汇总 TileSmith 从 2026-09 起所有实验（早期 hard-shape 实验、特征切片三轮 82c18be2 / b9e08b4c / 637deff6、跨架构 b1eded9f→ea9ecc41）以及本地探测中确认的真实 bug，按 bug 本身组织，不按发现轮次。

“确认”的标准：有不依赖 fuzzer oracle 的独立复现（最小脚本或原样重放），并排除了 harness / 参考实现 / 类型模型的问题；同一根因在不同程序、不同服务器上多次出现只计一种。复现脚本除特别注明外都在 `reports/2026.10.09/feature_slices/`。

## 总览

| 分组 | 数量 |
|---|---|
| Triton | **21**（Triton#1–#21） |
| TileLang | **30**（TileLang#1–#30） |
| 合计确认 | **51**（其中 13 种只在非本机目标出现：Triton#15、Triton#18–#21、TileLang#19–#26） |
| 候选（证据不足，未计入） | 0（老候选和跨架构候选已于 10-15 全部清理，见第三节） |

编号规则：每个 DSL 单独编号，按确认时间排序：9 月 hard-shape 实验的早期发现 → 特征切片第一至三轮（82c18be2 / b9e08b4c / 637deff6）→ 跨架构上线前的本地冒烟 → 第四轮跨架构实验。总表最后一列是旧编号（`#N` 为 `reports/2026.10.09/feature_slices/REPORT.md` 中跨 DSL 的确认序号；`E1`–`E5` 为早期发现，即 `reports/2026.09.26/REPORT.md` 的 TL-01/02/03、TR-01 和 09-30 的 RemoveLayoutConversions；`X1`–`X4` 为本地冒烟；Triton#18–#21、TileLang#22–#30 是 10-15 清理积压时确认的，没有旧编号）。`CHECKS.md`、`known_mechanisms.json` 和其他报告仍用旧编号。

按后果分：

| 后果 | 数量 | 编号 |
|---|---|---|
| **静默错误结果**（编译运行都“成功”，数值错） | 16 | Triton：Triton#3、Triton#4、Triton#9、Triton#10、Triton#13、Triton#17、Triton#18；TileLang：TileLang#4、TileLang#8、TileLang#9、TileLang#10、TileLang#14、TileLang#15、TileLang#16、TileLang#18、TileLang#29 |
| 运行时错误（越界写、非法访存、对齐错误、device assert；运行时） | 3 | Triton#16、TileLang#4、TileLang#22 |
| 编译器崩溃 / 断言 / LLVM abort（编译时） | 14 | Triton：Triton#2、Triton#5、Triton#6、Triton#7、Triton#12、Triton#14、Triton#20、Triton#21；TileLang：TileLang#2、TileLang#17、TileLang#23、TileLang#24、TileLang#28、TileLang#30 |
| 生成的代码被下游拒绝（nvcc/LLVM/codegen 报错；编译时） | 19 | Triton：Triton#1、Triton#8、Triton#11、Triton#15、Triton#19；TileLang：TileLang#1、TileLang#3、TileLang#5、TileLang#6、TileLang#7、TileLang#11、TileLang#12、TileLang#13、TileLang#19、TileLang#20、TileLang#21、TileLang#25、TileLang#26、TileLang#27 |

TileLang#4 同时有错误结果和越界写两种表现，在两行都列出；TileLang#18 以错误结果为主，也有编译失败形式。

按责任方分：Triton 编译器 15 种；NVIDIA 工具链（ptxas / libdevice）经 Triton 触发 5 种（Triton#3、Triton#9、Triton#13、Triton#16、Triton#17）；LLVM AMDGPU 后端经 Triton 触发 1 种（Triton#19）；TileLang/TVM 30 种。

上游状态：TileLang#1、TileLang#3、Triton#1 上游已有相同报告（TileLang #2206、#2172，Triton #10790），不算首次发现；其余尚未系统查重或提交。

## 总表

后果：**错误结果** = 编译运行都成功但数值错；**运行时错误** = 越界、非法访存等；**崩溃** = 编译器断言 / abort / unreachable；**代码被拒** = 生成的代码被下游（nvcc、LLVM、codegen）拒绝。目标“sm_89”表示在本机（4090 / vGPU / 4060）上触发。详细说明见第一、二节。

### Triton（21 种）

| 编号 | 现象 | 后果 | 责任方 / 位置 | 触发条件 | 目标 | 复现 | 旧编号 |
|---|---|---|---|---|---|---|---|
| Triton#1 | `tl.flip(x)` 省略 dim 编译失败 | 代码被拒 | Triton 前端（None 与整数比较） | `tl.flip` 不传 dim | 任意 | 上游 Triton #10790 已有 | E4 |
| Triton#2 | RemoveLayoutConversions 生成违反支配关系的 IR | 崩溃 | Triton TritonGPURemoveLayoutConversions | 一致分支 + scratch store + debug_barrier | sm_89 | 原样重放，未化简 | E5 |
| Triton#3 | f16/bf16 `tl.abs` 有一半元素仍为负 | 错误结果 | ptxas 12.9（只掩码打包寄存器的高半部分） | f16/bf16 abs，如 `abs(dot)` 存 fp16 | sm_89 | `abs_dot.py`、`abs_f16_elementwise.py` | #1 |
| Triton#4 | f32→fp8 默认舍入错（0.53125006→0.5） | 错误结果 | Triton f32→fp8 降级（经 fp16 用 RZ 双重舍入；sm_90+ 用直接转换正确） | `x.to(fp8)`，输入略大于两个 fp8 值的正中间 | sm_89 | `f32_to_fp8.py` | #2 |
| Triton#5 | f64↔fp8 转换 LLVM abort | 崩溃 | Triton ConvertTritonGPUToLLVM | f64 与 fp8 间 `.to()` | sm_89；AMD 上为 Unsupported conversion | campaign `.min.py` | #3 |
| Triton#6 | fp8→fp8 转换断言 | 崩溃 | Triton（应为前端报错） | fp8 格式之间 `.to()` | sm_89 | — | #4 |
| Triton#7 | `fp_downcast_rounding='rtz'` pass 崩溃 | 崩溃 | Triton 转换降级 | rtz + f64 源或 fp8 目标 | sm_89；AMD 上为 `maybeOutVals` 断言 | `probe_round_tr.py` | #5 |
| Triton#8 | bf16 自定义 combine 的 reduce/scan 编译失败 | 代码被拒 | Triton 前端类型提升（combine 返回 f32） | bf16 + `tl.maximum` 等 combine | sm_89 | — | #6 |
| Triton#9 | 常量折叠后转 fp8，部分 lane 是垃圾值 | 错误结果 | ptxas 12.9（`F2FP…MERGE_C` 合并无关寄存器） | 编译期常量转 fp8 后存储 | sm_89 | `ptxas_const_fp8.py` | #12 |
| Triton#10 | 向量化窄整数掩码加载，负填充值读错 | 错误结果 | Triton PTX（填充值未按字节重复） | int8/int16/uint8 `tl.load(mask, other<0)` | sm_89 | `masked_load_negative_other.py` | #13 |
| Triton#11 | `tl.fdiv` 用于 fp16/bf16 编译失败 | 代码被拒 | Triton 降级（位宽 / intrinsic 返回类型不符） | fdiv 半精度；`ieee_rounding=True` 用于 fp16/bf16/fp64 | sm_89 | `tr_fdiv_half.py` | #16 |
| Triton#12 | `tl.gather` 沿 axis 0 断言 | 崩溃 | Triton OptimizeThreadLocality（`isWarpLocal`） | 秩 3、下标 tile 比源维度长、num_warps=8 | sm_89 | `tr_gather_warp_local_assert.py` | #21 |
| Triton#13 | 有符号 `libdevice.mul24` 返回完整 48 位乘积 | 错误结果 | NVIDIA libdevice（`mul nsw` 声明不成立） | `mul24` 结果被加宽使用 | sm_89 | `tr_libdevice_mul24_nsw.py` | #24 |
| Triton#14 | warp specialize 循环里有扫描时 unreachable | 崩溃 | Triton TaskIdPropagation（不处理 `tt.scan`） | `warp_specialize=True` 循环内 cumsum/associative_scan | sm_80/89/90 | `tr_warp_specialize_scan_unreachable.py` | #25 |
| Triton#15 | `tl.atomic_xchg` 全部无法编译 | 代码被拒 | Triton AMD buffer atomic 降级（`exch` 内建不存在） | 任意类型 `atomic_xchg` | gfx942 / gfx950 | `tr_amd_atomic_xchg.py` | X1 |
| Triton#16 | f16 dot 在 `maxnreg=128` 下非法访存 | 运行时错误 | ptxas 12.4 / 12.9（-O2 起提前改写地址寄存器） | 32×64@64×128、num_warps=1、maxnreg=128 | sm_89 | `tr_maxnreg_spill_illegal_access.py` | #28 |
| Triton#17 | bf16 原子加在 `maxnreg=32` 下丢失更新 | 错误结果 | ptxas 12.9（CAS 循环寄存器分配；12.4 未测） | sem=acquire/acq_rel、num_warps=1、maxnreg=32 | sm_89 | `tr_maxnreg_atomic_cas_lost_update.py` | #29 |
| Triton#18 | sm<89 上 f16→fp8e5 转换：平局一律进位，NaN 变成 inf 或 −0.0 | 错误结果 | Triton sm<89 软件转换（`and 0xfffefffe; add 0x00800080` 后取高字节，没有 RNE 和 NaN 处理） | `x.to(tl.float8e5)`（默认 rtne）；`padding_option='nan'` 的 fp8e5 加载 | sm80（在 sm_89 上真实执行） | `tr_sm80_f16_to_e5m2_ties.py`、`tr_sm80_nan_to_e5m2.py`、`tr_sm80_fp8_padding_nan.py` | — |
| Triton#19 | fp16 反向 cumsum 无法生成 hsaco（报 LLD 错） | 代码被拒 | Triton 自带 LLVM 的 AMDGPU 后端（DPP 合并生成 GFX11 不允许的 `v_bfe_i32_e64_dpp …, 0, 16`，被自己的汇编器拒绝）；Triton 把汇编错误吞掉，只报 LLD 错 | `tl.cumsum(x_f16, reverse=True)` | gfx1100 | `tr_gfx1100_scan_lld.py` | — |
| Triton#20 | fp32 `tl.dot` 加 `kpack=2` 断言 `idx < size()` | 崩溃 | Triton AMD 降级（ConvertTritonAMDGPUToLLVM） | fp32 dot、K=16、`kpack=2`（K=32 或 kpack=1 正常） | gfx942 | `tr_gfx942_kpack2_dot.py` | — |
| Triton#21 | warp specialize 循环加展开时断言 `dyn_cast on a non-existent value` | 崩溃 | Triton TritonGPUAutomaticWarpSpecialization | `tl.range(..., loop_unroll_factor≥2, warp_specialize=True)` + block pointer 加载 | sm100 / sm120（sm90 正常） | `tr_blackwell_ws_unroll.py` | — |

小计（按主要表现）：错误结果 7（Triton#3 Triton#4 Triton#9 Triton#10 Triton#13 Triton#17 Triton#18）、运行时错误 1（Triton#16）、崩溃 8（Triton#2 Triton#5 Triton#6 Triton#7 Triton#12 Triton#14 Triton#20 Triton#21）、代码被拒 5（Triton#1 Triton#8 Triton#11 Triton#15 Triton#19）。责任方：Triton 15 种，NVIDIA 工具链 5 种（Triton#3 Triton#9 Triton#13 Triton#16 Triton#17），LLVM AMDGPU 1 种（Triton#19）。

### TileLang（30 种）

| 编号 | 现象 | 后果 | 责任方 / 位置 | 触发条件 | 目标 | 复现 | 旧编号 |
|---|---|---|---|---|---|---|---|
| TileLang#1 | bool 向量 CUDA codegen 失败 | 代码被拒 | CUDA 类型打印（boolx8/x16） | bool 运算被向量化 | sm_89 | 上游 TileLang #2206 已有 | E1 |
| TileLang#2 | 自动选择的归约布局无法降级 | 崩溃 | ReduceOp 布局（`reduce.h:534`） | fragment + reduce 的推断布局 | sm_89 | 原样重放 | E2 |
| TileLang#3 | 流水线生成非法 2 字节 cp.async | 代码被拒 | 流水线 / cp.async 降级 | GEMM、`num_stages=4`、奇数 K | sm_89 | 上游 TileLang #2172 已有 | E3 |
| TileLang#4 | fp16 `T.atomic_add` 向量化后忽略下标映射 | 错误结果 + 运行时错误 | 原子加向量化（`AtomicAddx2`） | fp16 原子加被向量化 | sm_89 | `atomic_f16.py` | #7 |
| TileLang#5 | 嵌套整数 floordiv codegen 崩溃 | 代码被拒 | 向量化后 Let 绑定冲突 | 嵌套 floordiv、threads=32 | sm_89 | `floordiv_i16.py` | #8 |
| TileLang#6 | int64→f16/bf16 `T.cast` 被 nvcc 拒绝 | 代码被拒 | CUDA codegen（`half_t` 构造歧义） | int64 转半精度 | sm_89 | — | #9 |
| TileLang#7 | int64 `T.atomic_max/min/or` 被 nvcc 拒绝 | 代码被拒 | CUDA 模板（`atomic_ref<unsigned long long>`） | int64 原子 max/min/or | sm_89 | — | #10 |
| TileLang#8 | int→bf16 截断而非就近舍入（259→258） | 错误结果 | cutlass `bfloat16_t(int)` 直接取高 16 位 | int16/int32 → bf16 | sm_89 | `int_to_bf16.py` | #11 |
| TileLang#9 | while 循环拷贝到共享内存再 gemm 缺同步 | 错误结果（数据竞争） | 同步插入（循环体末尾缺 `__syncthreads`） | `T.alloc_var` 计数的 while + copy + gemm | sm_89 | `tl_while_gemm_race.py` | #14 |
| TileLang#10 | 窄整数运算后加宽不按窄类型回绕 | 错误结果 | CUDA codegen（C 整数提升） | int8/16、uint8/16 运算后 cast 到宽类型 | sm_89 | `tl_narrow_int_widen.py` | #15 |
| TileLang#11 | 向量化 `T.bitwise_not` 生成 `~` 作用于向量类型 | 代码被拒 | CUDA codegen | 向量化按位取反 | sm_89 | `tl_intrinsic_lowering.py` | #17 |
| TileLang#12 | 半精度 `nearbyint/fmod/tanh` 生成未定义函数 | 代码被拒 | CUDA 内建降级（`hnearbyint` 等不存在） | f16/bf16 上这些内建 | sm_89 | `tl_intrinsic_lowering.py` | #18 |
| TileLang#13 | 缺降级规则的内建到达 codegen | 代码被拒 | CUDA 内建降级规则缺失 | 有符号 popcount；半精度 copysign/nextafter/ldexp/hypot | sm_89 | `tl_intrinsic_lowering.py` | #19 |
| TileLang#14 | 共享内存取模下标重排丢掉列偏移 | 错误结果 | 下标化简 | 行列偏移同时存在，与形状、线程数有关 | sm_89 | `tl_remap_column_shift_dropped.py` | #20 |
| TileLang#15 | 嵌套 floordiv 不可靠化简（取负移进内层） | 错误结果 | 算术化简器 | `floordiv(floordiv(x,3)*-1, 4)` | sm_89 | `tl_nested_floordiv_wrong_value.py` | #22 |
| TileLang#16 | 区域拷贝降级为 ldmatrix 时丢列偏移 | 错误结果 | copy → ldmatrix 降级 | 2 字节类型、列偏移非 0、threads=256 | sm_89 | `tl_ldmatrix_region_offset_dropped.py` | #23 |
| TileLang#17 | 开 `tl.enable_lower_ldgstg` 后向量元素访问失败 | 崩溃 | `lower_ldg_stg.cc`（忽略向量 lanes） | float32x2 / int32x2 / float32x4 元素 | sm_89 | `tl_ldgstg_vector_lanes.py` | #26 |
| TileLang#18 | `tl.config_index_bitwidth=64` 改写存储的值 | 错误结果 + 代码被拒 | `ConfigIndexBitwidthRewriter` | 该 pass 配置 + 值中有窄有符号 cast/常量 | sm_89 | `tl_index64_rewrites_stored_values.py` | #27 |
| TileLang#19 | int16 逐元素运算无法打印 int16x16 | 代码被拒 | CUDA 类型打印（Blackwell 256 位向量化） | 最简单的 int16 逐元素运算 | sm100 / sm120 | `tl_blackwell_256bit_vector_type.py` | X2 |
| TileLang#20 | 标准流水线 GEMM 用了 sm_120 没有的 setmaxnreg | 代码被拒 | warp specialization（未按目标关闭） | `T.Pipelined` + `T.gemm` | sm120 | `tl_sm120_setmaxnreg.py` | X3 |
| TileLang#21 | IEEE 舍入内建在 AMD 上无降级规则 | 代码被拒 | HIP 内建降级规则缺失 | `T.ieee_add/mul/fdiv/...` | gfx942 / gfx950 | `tl_hip_ieee_unresolved.py` | X4 |
| TileLang#22 | 编译到 sm_80 的 fp8 `T.gemm` 用了 sm_89 才有的 MMA，运行时 device assert | 运行时错误 | gemm 指令选择（不按目标架构检查 fp8 MMA，应在编译时拒绝） | float8_e4m3 `T.gemm`、target sm_80 | sm80 | `tl_sm80_fp8_gemm.py` | — |
| TileLang#23 | `clear_accum=(k==0)` 加 loop unswitching 后 IR 不再是 SSA | 崩溃 | `tl.loop_unswitching_allow_non_trivial_else` 复制循环体，`accum` 被定义两次 | int8 或 fp8 gemm、`clear_accum=(k==0)`、该 pass 配置（f16 正常） | sm75 | `tl_unswitch_else_ssa.py` | — |
| TileLang#24 | 窄 N 的 f16 gemm 触发 `Not a canonical GMMA_MN layout` 断言 | 崩溃 | WGMMA 描述符计算（`compute_gmma_descriptor`） | 64×32@32×16、threads=256（N=8、threads=128 时是另一条 WGMMA 断言；N≥32 正常） | sm90 | `tl_sm90_gmma_narrow_n.py` | — |
| TileLang#25 | HIP 后端 fp8 处理缺口：e5m2 常量无法打印；`float8_e4m3` gemm 报 KeyError | 代码被拒 | HIP codegen `PrintConst`（不支持 e5m2/e5m2fnuz 常量）；MFMA 类型表缺 `float8_e4m3`（CUDA 后端两者都支持） | 含 e5m2 常量的表达式；float8_e4m3 `T.gemm` | gfx942 / gfx950 | `tl_amd_probe.py fp8_const …`、`tl_amd_probe.py fp8_gemm …` | — |
| TileLang#26 | 8 位整数 xor 在 HIP codegen 中 `Check failed: (p)` | 代码被拒 | HIP codegen（`VisitExpr_`） | int8/uint8 `^`（int16/int32、CUDA 正常） | gfx942 / gfx950 / gfx1100 | `tl_amd_narrow_select.py <target> uint8 xor` | — |
| TileLang#27 | bf16 `T.pow` 报 `power only applies to float` | 代码被拒 | TVM `pow` 的类型检查（`is_float()` 不含 bfloat16；bf16 的 exp/sqrt/log 正常） | bf16 `T.pow(x, y)`、浮点指数 | sm_89 | `tl_pow_bf16.py bfloat16 pow` | 老候选 |
| TileLang#28 | 显式 order/stage 的 `T.Pipelined` 把 fragment 多缓冲，随后报“只允许 fragment[0]”或 `Layout::Forward` 维数不符 | 崩溃 | 软件流水线（显式 stage 时对 fragment 做多缓冲；同一程序只给 num_stages 时正确） | 循环体把全局数据拷到 fragment 再计算，`order`/`stage` 显式给出 | sm_89 | `tl_pipelined_stage_fragment.py fragment explicit 2`、`tl_pipelined_stage_reduce.py fragment explicit` | 老候选 ×2 |
| TileLang#29 | 浮点 `x - x`、`x * 0` 被化简成 0，inf/NaN 输入得 0 而不是 NaN | 错误结果 | TVM 算术化简器（未开 fast math 也按实数化简） | 任何浮点 `x - x` / `x * 0`，x 为 inf 或 NaN | sm_89 | `tl_float_x_minus_x.py float32` | — |
| TileLang#30 | gemm 累加器加上另一个 fragment 的行和时布局推断失败（`has_best`） | 崩溃 | 布局推断（未把 gemm 决定的布局传播到可自由选择布局的 fragment） | `stat = reduce_sum(wide)`，`wide` 来自全局内存而非 acc，再算 `stat[i] + acc[i, j]` | sm_89 | `tl_layout_has_best.py 256 other` | 老候选（09-27） |

小计（按主要表现）：错误结果 9（TileLang#4 TileLang#8 TileLang#9 TileLang#10 TileLang#14 TileLang#15 TileLang#16 TileLang#18 TileLang#29；TileLang#4 也会越界写，TileLang#18 也有编译失败形式）、运行时错误 1（TileLang#22）、崩溃 6（TileLang#2 TileLang#17 TileLang#23 TileLang#24 TileLang#28 TileLang#30）、代码被拒 14（TileLang#1 TileLang#3 TileLang#5 TileLang#6 TileLang#7 TileLang#11 TileLang#12 TileLang#13 TileLang#19 TileLang#20 TileLang#21 TileLang#25 TileLang#26 TileLang#27）。TileLang#22–#30 在 0.1.15 上复验仍然存在。

---

## 一、Triton

### 数值错误

**Triton#3 f16/bf16 上的 `tl.abs` 有一半元素仍为负**（责任方：ptxas 12.9）
- 现象：`tl.abs` 作用于 f16/bf16（最初是 `tl.abs(tl.dot(a, b))` 存成 fp16），输出交替为 [-1, 2, -3, 4, …]。不需要 dot：256 个元素、`num_warps=2`、f16→abs→int32 即可触发；abs 在 f32 中算完再收窄到 f16 同样触发。
- 根因：Triton 生成的 PTX 对每个 half 都做 `and.b16 0x7fff`，是正确的；Triton 3.8 自带的 ptxas 12.9 用 `F2FP.PACK_AB` 把两个 half 打包后，只对高半部分（`PRMT 0x7632`）做掩码，低半部分保留符号位。同一份 PTX 用 ptxas 12.4 编译结果正确。
- 复现：`abs_dot.py`、`abs_f16_elementwise.py`。
- 10-14 在 4090 上排除 fuzzer 因素（脚本 `claude_work/verify_abs/`，不含 TileSmith 代码）：把同一份 PTX（逐字节相同，每个 half 都有 `and.b16 ..., 32767`）分别用 ptxas 12.9 和 12.4 汇编，再用 CUDA 驱动直接加载运行，绕过 Triton。12.9 的 cubin 有 266/1024 个负数输出，12.4 全部正确。SASS 中 12.9 每个线程 8 个 half 只生成 4 条 `LOP3.LUT ... 0x7fff`（只给 `PRMT 0x7632` 取出的高半部分），`PRMT ..., 0x5410, RZ` 取出的低半部分没有掩码；12.4 生成 8 条。

**Triton#4 f32→fp8 默认舍入错误**
- 现象：f32 转 e4m3/e5m2：0.53125006→0.5（应 0.5625），36.000004→32（应 40）。
- 根因（PTX 已确认）：先 `cvt.rz.f16.f32` 向零舍入到 fp16，再 `cvt.rn.satfinite.e5m2x2.f16x2`；中间的 RZ 丢掉了粘滞位，就近舍入的平局判断出错（双重舍入）。
- 复现：`f32_to_fp8.py`。

**Triton#9 常量折叠后转 fp8，部分 lane 是垃圾值**（责任方：ptxas 12.9）
- 现象：编译期被折叠成常量的值（如非负输入的 `min(x, 0)`）转 fp8 后存储，每 4 个元素错 1–2 个，出现 -0.0、-inf、-384 等。
- 根因：Triton 的 PTX 正确；ptxas 12.9 生成 `F2FP.SATFINITE.E5M2.F16.UNPACK_B_MERGE_C` 或 `...E4M3.F32.PACK_AB_MERGE_C ... R1`，未用到的字节从无关寄存器（甚至栈指针 R1）合并进来。ptxas 12.4 正确。
- 复现：`ptxas_const_fp8.py`。

**Triton#10 带掩码的向量化窄整数加载，负填充值读错**
- 现象：int8/int16/uint8 的 `tl.load(..., mask, other=负数)`（或 uint8 ≥128）被向量化时，被掩码的元素除每个 32 位字的第一个外都读出 -1/255。
- 根因：PTX 中填充值是 `mov.u32 %r, 0xfffffffffffffffe`（符号扩展后的标量），而不是按字节重复的 0xfefefefe。ptxas 12.4 结果相同，是 Triton 的 PTX 错误。int32、浮点和正的填充值正确。
- 复现：`masked_load_negative_other.py`。

**Triton#13 有符号 `libdevice.mul24` 返回完整 48 位乘积**（责任方：NVIDIA libdevice，经 Triton 触发）
- 现象：`libdevice.mul24(x, y)` 的结果只被加宽使用（`.to(tl.int64)`）时，返回完整乘积而不是文档规定的“低 32 位”，128/128 错。
- 根因：Triton 附带的 `libdevice.10.bc` 中 `__nv_mul24` 是 `shl/ashr 8` 后 `mul nsw i32`；两个 24 位数的乘积可达 2⁴⁶，`nsw`（不会有符号溢出）的声明不成立，NVPTX 据此把 `sext(mul nsw)` 合并成 `mul.wide.s32`。无符号 `umul24` 和普通 `x * y` 正确。
- 复现：`tr_libdevice_mul24_nsw.py`。

**Triton#17 bf16 原子加在 `maxnreg=32` 下丢失更新**（责任方：ptxas 12.9）
- 现象：bf16 `tl.atomic_add`（`sem='acquire'` 或 `'acq_rel'`），`num_warps=1, maxnreg=32` 时一个槽得 10，应为 12，确定性复现。
- 根因：bf16 原子加在 sm_89 上降级为 32 位 CAS 循环加 fence。加与不加 maxnreg 的 PTX 只差 `.maxnreg 32` 一行，两者都没有寄存器溢出，所以错误出在 ptxas 对 CAS 循环的寄存器分配。maxnreg≥40、num_warps≥2、`sem='relaxed'` 都正确。
- 复现：`tr_maxnreg_atomic_cas_lost_update.py`。

### 运行时错误

**Triton#16 f16 dot 在 `maxnreg=128` 下非法访存**（责任方：ptxas，12.4 和 12.9 都有）
- 现象：f16 `tl.dot`（32×64 @ 64×128，f32 累加），`num_warps=1` 加文档化的启动选项 `maxnreg=128`，运行报 `illegal memory access`。
- 分析：maxnreg 只应限制寄存器、强制溢出（138 次溢出，552 字节栈帧），不应改变语义。不加 maxnreg（214 寄存器）、maxnreg=255、num_warps=2 都正确；把存储改成 `acc * 1.0 + 0.0` 也能通过，说明依赖具体的溢出代码。
- 复现：`tr_maxnreg_spill_illegal_access.py`。
- 10-14 在 4090 上定位（脚本 `/root/autodl-tmp/abs_repro/maxnreg_ptxas.py`）：加与不加 maxnreg 的 PTX 只差 `.maxnreg 128` 一行。同一份 PTX 用 ptxas 12.9 和 12.4 汇编后经 CUDA 驱动运行都报非法访存（CUresult 700）；`-O0`/`-O1` 正确、`-O2`/`-O3` 出错；`.maxnreg` 改成 96、160、200 都正确；不加 maxnreg 的 cubin 在 compute-sanitizer 下 0 错误。compute-sanitizer 报 `STS.128 [R114]`（把 B 块写入共享内存，共享内存共 20480 字节）越界写，地址 0x5b00–0x7f00。PTX 中该地址是 `((tid & 15) << 4) ^ (bfe(tid, 4, 1) & 272)`，不超过 0x1f0；SASS 中 `R115 = tid & 15` 之后，`/*0cb0*/ IMAD R115, R115, 0x80, R114` 先把 R115 改写成另一个地址（供后面 `LOP3 R40, R115, R40` 用），`/*0cc0*/ SHF.L.U32 R114, R115, 0x4` 才读 R115，读到的是改写后的值，算出 `(tid & 15) * 0x800 + (tid & 7) * 0x100`，与实测越界地址逐一吻合。是 ptxas 在寄存器紧张时的调度/寄存器分配错误，PTX 正确。

### 编译器崩溃

**Triton#5 f64↔fp8 转换 LLVM abort**：`.to()` 在 f64 与 fp8 之间转换时，ConvertTritonGPUToLLVM 中 LLVM abort，进程被杀；前端不报错。AMD 目标上表现为 `Unsupported conversion`。复现：campaign `.min.py`。

**Triton#6 fp8→fp8 转换断言**：报 `Rounding mode must be specified for convertsions to fp8`，断言杀掉进程，而不是前端报错。

**Triton#7 `fp_downcast_rounding='rtz'` 用于 f64 源或 fp8 目标时 pass 崩溃**；AMD 上为 `maybeOutVals` 断言。复现：`probe_round_tr.py`、`tr_round.jsonl`。

**Triton#12 `tl.gather` 沿 axis 0 断言**
- 现象：秩 3 的 tile 沿 axis 0 做 `tl.gather`，下标 tile 比被 gather 的那一维更长时（源 (2,16,4)、下标 (4,16,4)、num_warps=8），`OptimizeThreadLocality.cpp:220` 断言 `GatherLoweringHelper(op).isWarpLocal()` 失败，进程被杀。
- 两台服务器各自独立发现，本地复现。复现：`tr_gather_warp_local_assert.py`。

**Triton#14 warp specialization 循环里有扫描时 `llvm_unreachable`**
- 现象：`tl.range(..., warp_specialize=True)` 的循环体里有 `tl.cumsum` 或 `tl.associative_scan`，报 `UNREACHABLE executed at .../WarpSpecialization/TaskIdPropagation.cpp:107`（`Other parent ops are not supported`）。
- 根因：`TaskIdBackwardPropagation::propagateToParent` 只接受 scf.for、scf.if、tt.func 和 tt.reduce 作为父操作，没有处理 `tt.scan` 的 combine 区域（上游 main 至今如此）。同一循环里的 `tl.sum`、自定义 `tl.reduce` 能编译；sm_80/89/90 失败，sm_100 能编译。
- 复现：`tr_warp_specialize_scan_unreachable.py`（只编译，不需要 GPU）。

**Triton#2 `TritonGPURemoveLayoutConversions` 生成违反 SSA 支配关系的 MLIR**
- 现象：含一致分支、全局 scratch `tl.store` 和 `tl.debug_barrier` 的程序，编译器报 `operand #0 does not dominate this use`，失败 pass 为 TritonGPURemoveLayoutConversions。
- 证据：09-30 实验中原样复跑复现；尚未化简为最小样例。

### 编译失败（生成的 IR 不合法）

**Triton#8 bf16 上自定义 combine 的 `tl.reduce` / `tl.associative_scan` 编译失败**：combine 用 `tl.maximum` 等时，combine 区域返回 f32 而不是 bf16（前端类型提升问题）。

**Triton#11 `tl.fdiv` 用于 fp16/bf16**：ConvertTritonGPUToLLVM 报 `LLVM ERROR: Unsupported bitwidth`；`ieee_rounding=True` 用于 fp16/bf16/fp64 时 LLVM 翻译失败（intrinsic 返回 half/double，而 `llvm.nvvm.div.rn.f` 返回 float）。复现：`tr_fdiv_half.py`。

**Triton#1 `tl.flip(x)` 省略 dim 时编译失败**：内层是整数与 None 比较的 TypeError；文档规定默认 `dim=None`。`dim=0` 通过。上游已有完全匹配的问题（Triton #10790），Triton 3.8.0 仍可触发。

### 只在非本机目标出现（已有独立复现）

**Triton#15 gfx942/gfx950 上任何 `tl.atomic_xchg` 都无法编译**
- 现象：i32/i64/f32/f64 的 `tl.atomic_xchg` 报 `LLVM Translation failed`。
- 根因：buffer atomic 降级生成 LLVM 内建 `llvm.amdgcn.raw.ptr.buffer.atomic.exch`，Triton 自带的 LLVM 里只有 `...buffer.atomic.swap`。gfx90a/gfx1100 不走 buffer atomics，能编译；`AMDGCN_USE_BUFFER_ATOMICS=0` 也能编译。
- 复现：`tr_amd_atomic_xchg.py`。

**Triton#18 sm<89 上 f16→fp8e5 的软件转换：平局舍入错，NaN 变成 inf 或 −0.0**
- 现象：目标为 sm_80（在 sm_89 上真实执行），`x.to(tl.float8e5)`（默认 rtne）对全部 248 个“正好在中间”的有限 f16 中的 124 个舍入错误，也就是所有应舍到偶数的那一半都进位了，如 3.8147e-05→4.5776e-05（应 3.0518e-05）。1024 个 f16 NaN 中 256 个没变成 NaN：尾数小的变成 inf（0x7c01→inf），尾数大的进位溢出到符号位变成 −0.0（0x7fff→−0.0）。fp8e5 块指针加载用 `padding_option='nan'` 时填充值是 −0.0，也是这个原因。本机目标（sm_89）用 `cvt.rn.satfinite.e5m2x2.f16x2`，结果正确。
- 根因：sm<89 没有 fp8 转换指令，Triton 用内联 PTX `and.b32 a, x, 0xfffefffe; add.u32 a, a, 0x00800080; prmt …0x7531` 取高字节，等于“加 0x80 再截断”，平局一律远离零，也没有 NaN 判断。
- 复现：`tr_sm80_f16_to_e5m2_ties.py 80`（对照 `native`）、`tr_sm80_nan_to_e5m2.py 80`、`tr_sm80_fp8_padding_nan.py 80 float8_e5m2`。合并了两个跨架构候选（f16→f8e5 平局、fp8e5 `padding=nan`）。

**Triton#19 gfx1100 上 fp16 反向 cumsum 无法生成代码**（责任方：Triton 自带 LLVM 的 AMDGPU 后端）
- 现象：`tl.cumsum(x_f16, axis=0, reverse=True)`（32 或 128 个元素，num_warps 1 或 4）编译到 gfx1100，报 `LLD failed to link hsaco … target emulation unknown`。正向 cumsum、fp32、bf16、gfx942 都正常。
- 根因：Triton 生成的 LLVM IR 合法（`llvm.amdgcn.update.dpp.i32` 后做 16 位符号扩展）；LLVM 的 DPP 合并把它们合成 `v_bfe_i32_e64_dpp v2, v2, 0, 16 row_xmask:8 …`，GFX11 的 VOP3-DPP 不允许 src1 是立即数，LLVM 自己的汇编器报 `src1 immediate operand invalid for instruction`。Triton 没有把汇编错误报出来，只留下随后的 LLD 错误。
- 复现：`tr_gfx1100_scan_lld.py gfx1100 fp16 1 32 4`（只编译）。加 `TRITON_KERNEL_DUMP=1` 可以看到 `.amdgcn` 中的指令。

**Triton#20 gfx942 上 fp32 `tl.dot` 加 `kpack=2` 断言**
- 现象：16×16 @ 16×64 的 fp32 `tl.dot`，启动选项 `kpack=2`（gfx942 上文档化的选项，只在 gfx950 上被废弃并改回 1），ConvertTritonAMDGPUToLLVM 中 `SmallVector::operator[]` 断言 `idx < size()`，进程被杀。K=32、kpack=1、fp16/bf16 都正常。
- 复现：`tr_gfx942_kpack2_dot.py fp32 2 1 16`（只编译）。

**Triton#21 Blackwell 上 warp specialization 循环加展开时断言**
- 现象：`tl.range(0, n, loop_unroll_factor=2, warp_specialize=True)`，循环体通过 block pointer 加载并 `tl.advance`，编译到 sm100 或 sm120 时 TritonGPUAutomaticWarpSpecialization 断言 `detail::isPresent(Val) && "dyn_cast on a non-existent value"`。不展开、改用普通指针、sm90 都正常。
- 复现：`tr_blackwell_ws_unroll.py 100 2 1`（只编译）。

---

## 二、TileLang

### 数值错误

**TileLang#4 fp16 `T.atomic_add` 被向量化后忽略下标映射**
- 现象：fp16 原子加被向量化为 `AtomicAddx2`，忽略目标下标映射：写出界（输出两侧的 canary 被改）、结果错、或 misaligned address。64 线程写单个 slot 时 O[1] 被加 64。
- 复现：`atomic_f16.py`。

**TileLang#8 int16/int32→bfloat16 的 `T.cast` 截断而不是就近舍入**
- 现象：259→258（应 260）。int→float16 正确。
- 根因：生成 `(bfloat16_t)x`；cutlass 的 `bfloat16_t(int)` 先转 float，再直接取高 16 位（`bits >> 16`），没有舍入。
- 复现：`int_to_bf16.py`。

**TileLang#9 while 循环里拷贝到共享内存再 gemm，缺少同步**
- 现象：用 `T.alloc_var` 计数的 while 循环中先拷贝到共享内存再 `T.gemm`，20 次运行 15 次错、10 种不同输出；`T.serial` 正确。
- 根因：生成的 CUDA 只在拷贝与 gemm 之间有 `__syncthreads()`，循环体末尾没有，下一轮拷贝与上一轮 gemm 的 ldmatrix 读冲突（数据竞争）。
- 复现：`tl_while_gemm_race.py`。实验中出现次数最多的机制之一。

**TileLang#10 窄整数运算后加宽，不按窄类型回绕**
- 现象：int8/uint8/int16/uint16 的加、减、乘、移位后 `T.cast` 到更宽类型，结果是未回绕的值；直接存成窄类型时正确。
- 根因：生成 `((int64_t)((int)(a + b)))`，C 的整数提升保留了未回绕的值，而 TIR 中该运算的类型是窄类型。int32 也有类似路径（乘法先加宽到 int64 再乘），但属于有符号溢出，上游可能视为未定义行为；结论主要由回绕语义明确的无符号情形支撑。
- 复现：`tl_narrow_int_widen.py`、`tl_int32_widen_in_wide_type.py`。

**TileLang#14 共享内存取模下标重排丢掉列偏移**
- 现象：`b1[i,j] = b0[(i+3)%16, (j+2)%16]` 的列下标被生成为 `(threadIdx.x & 7) * 2`，`+2` 和 `%16` 都消失，结果静默等于“只换行不换列”；32×32 时生成正确。确定性，不是竞争。
- 触发：行、列偏移同时存在；与向量化无关（标量读也出现）；是否触发取决于形状和线程数怎样拆分下标。
- 复现：`tl_remap_column_shift_dropped.py`、`tl_remap_column_shift_dropped_scalar.py`。

**TileLang#15 嵌套整数 floordiv 的不可靠化简**
- 现象：`T.floordiv(T.floordiv(x, 3) * -1, 4)` 被编译成 `floordiv(-x, 12)`，x = 1、2 时得 −1，应为 0；`* -2` 被合并成 `floordiv(-x, 6)`，−8..8 中 6/17 个输入错。正乘数正确。
- 根因：把取负移进内层 floordiv 不成立：−⌊x/3⌋ 等于 ⌈−x/3⌉ 而不是 ⌊−x/3⌋。数值很小，与溢出无关。
- 复现：`tl_nested_floordiv_wrong_value.py`。

**TileLang#16 区域拷贝降级为 ldmatrix 时丢掉列偏移**
- 现象：`T.copy(b1[0:64, 8:16], b2)`（2 字节类型、threads=256）生成 `ptx_ldmatrix_x1(&b1[(tid >> 5) * 128 + (tid & 7) * 16], ...)`，没有 `+ 8`，片段静默拿到第 0–7 列。
- threads=128（普通 uint2 加载，带偏移）、4 字节类型、偏移为 0 都正确。效果与 TileLang#14 相同，但降级路径不同。
- 复现：`tl_ldmatrix_region_offset_dropped.py`。

**TileLang#18 `tl.config_index_bitwidth=64` 改写了存储的值**
- 现象：这个 pass 配置（文档：“配置索引的位宽”）不只改索引：(1) 错误结果：`O[i] = T.cast(T.cast(X[i], "int8"), "int32")` 原样返回 X[i]（−640 得 −640，应 −128）；(2) 编译失败：int8/int16 元素运算变成 int64x16 向量运算，报 `Cannot convert type int64x16 to CUDA type`。
- 根因：`ConfigIndexBitwidthRewriter` 为改索引，在访问整个 BufferStore/BufferLoad 时打开改写开关，被存储的值里所有窄于 64 位的有符号整数常量和 Cast 也被改成 int64。默认配置正确。
- 复现：`tl_index64_rewrites_stored_values.py`。

### 编译器崩溃 / 断言

**TileLang#17 打开 `tl.enable_lower_ldgstg` 后，向量元素类型的全局访问失败**
- 现象：`Check failed: tgt_dtype.lanes() * tgt_dtype.bits() == src_dtype.lanes() * src_dtype.bits() (32 vs. 64)`。
- 根因：`lower_ldg_stg.cc` 用 `buffer->dtype.bits()`（单个 lane 的位宽）决定访问宽度，忽略向量元素的 lanes：float32x2 元素被生成 `ldg32`，再把 32 位结果 reinterpret 成 64 位。float32x4、int32x2 同样失败。
- 复现：`tl_ldgstg_vector_lanes.py`。

**TileLang#2 自动选择的归约布局无法降级**
- 现象：`ReduceOp cannot lower a layout where a source index depends on a thread-owned reduce segment`（`reduce.h:534`）。
- 证据：只用 fragment、copy、gemm、reduce 等正常 API，布局由编译器推断；单独编译仍失败，关闭向量化或 async copy 仍失败。

### 生成的代码被拒绝 / codegen 失败

**TileLang#5 嵌套整数 floordiv codegen 崩溃**：threads=32 时向量化，报 `Let cannot bind the same var to two different values`；int8/16/32/64 都复现。复现：`floordiv_i16.py`。

**TileLang#6 int64→float16/bfloat16 的 `T.cast` 被 nvcc 拒绝**：`half_t` 构造函数歧义。

**TileLang#7 int64 的 `T.atomic_max/min/or` 被 nvcc 拒绝**：`atomic_ref<unsigned long long>` 不匹配。

**TileLang#11 向量化的 `T.bitwise_not` 对 CUDA 向量类型生成 `~`**：nvcc 报 `no operator "~" matches these operands`（int4、ulonglong2 等）。复现：`tl_intrinsic_lowering.py`。

**TileLang#12 float16/bfloat16 上的 `T.nearbyint`/`T.fmod`（及 `T.tanh`）生成未定义的 `hnearbyint`/`hfmod`/`htanh`**。复现：`tl_intrinsic_lowering.py`。

**TileLang#13 缺少 CUDA 降级规则的内建函数到达代码生成**：有符号 int32/int64 的 `T.popcount`，float16/bfloat16 的 `T.copysign/nextafter/ldexp/hypot`，报 `Unresolved call ir.Op(tirx.*)`。复现：`tl_intrinsic_lowering.py`。

**TileLang#1 bool 向量 CUDA 代码生成失败**：`Cannot convert type boolx8/boolx16 to CUDA type`；关闭向量化后通过。上游已有相同症状报告（TileLang #2206）。

**TileLang#3 流水线拷贝生成非法的 2 字节 cp.async**：`IsValidCPAsyncTransferBytes(total_bytes)` 失败（cp.async 要求 4/8/16 字节）。最小化为 A=(1,9157)、B=(9157,3709)、64×64×128 块、256 线程的 GEMM，`num_stages=4` 失败、`num_stages=0` 通过。上游已有相关报告（TileLang #2172）。

### 只在非本机目标出现（已有独立复现）

**TileLang#19 sm100/sm120 上 int16 逐元素运算无法打印向量类型**：最简单的 `O[i] = X[i] + X[i]`（512 元素、threads=32）报 `Cannot convert type int16x16 to CUDA type`。Blackwell 上向量化宽度变为 256 位，CUDA 打印器没有 int16x16；sm_89/sm_90a 用 128 位 int16x8 正常。实验中也被命中。复现：`tl_blackwell_256bit_vector_type.py`。

**TileLang#20 sm120 上标准流水线 GEMM 构建失败**：`T.Pipelined` + `T.gemm` 被 warp specialization 加上 `tl::warpgroup_reg_alloc/dealloc`（PTX `setmaxnreg`），sm_120（RTX 50 系列）没有这条指令，nvcc 停在 `static_assert`；sm_90a、sm_100a 正常。复现：`tl_sm120_setmaxnreg.py`（需 nvcc ≥ 12.8）。

**TileLang#21 AMD 上 IEEE 舍入内建没有降级规则**：gfx942/gfx950 上 `T.ieee_add/sub/mul/fmaf/fsqrt/frcp/fdiv` 报 `Unresolved call tl.ieee_*`（CUDA 正常）。与 TileLang#13 同类（缺少降级规则），但在 HIP 后端。复现：`tl_hip_ieee_unresolved.py`。


**TileLang#22 编译到 sm_80 的 fp8 gemm 使用 sm_89 的 MMA**
- 现象：target `sm_80` 的 float8_e4m3 `T.gemm` 编译、nvcc 构建都成功，但代码用的是 `cute::SM89_16x8x32_F32E4M3E4M3F32_TN`，运行时触发 `Attempting to use SM89_16x8x32_F32E4M3E4M3F32_TN without CUTE_ARCH_MMA_F32_SM89_ENABLED` 的 device assert。sm_80 没有 fp8 MMA，应在编译时拒绝或换实现；target sm_89 正常。
- 复现：`tl_sm80_fp8_gemm.py sm_80 float8_e4m3`（在 sm_89 GPU 上运行）。

**TileLang#23 `clear_accum=(k==0)` 加 loop unswitching 后 IR 不再是 SSA**
- 现象：sm75 上 int8 或 fp8 的 `T.gemm(..., clear_accum=(k == 0))` 位于 `T.Pipelined` 中，开文档化的 pass 配置 `tl.loop_unswitching_allow_non_trivial_else` 后报 `buffer accum has already been defined, the Stmt is not SSA`。不开配置、f16、sm80/sm89 都正常。unswitching 把循环复制成 if/else 两份，sm75 上 int8/fp8 gemm 走的实现在循环体内定义了 `accum`，于是被定义两次。
- 复现：`tl_unswitch_else_ssa.py sm_75 int8 1`（只降级）。

**TileLang#24 sm90 上窄 N 的 gemm 触发 WGMMA 断言**
- 现象：f16 `T.gemm`，64×32 @ 32×16、threads=256，sm_90a 上 `compute_gmma_descriptor` 报 `AssertionError: Not a canonical GMMA_MN layout: stride<1,0>=2 != W=1`；N=8、threads=128 时报另一条 WGMMA 布局断言。threads=128 且 N=16、N≥32、sm_89 都正常。N=16 是 WGMMA 合法的 N，应能选择合适的划分或退回 MMA。
- 复现：`tl_sm90_gmma_narrow_n.py 16 256`（只降级）。

**TileLang#25 HIP 后端的 fp8 处理缺口**
- 现象 1：表达式中出现 e5m2 常量（如 `T.if_then_else(c, x, T.cast(0, "float8_e5m2"))`）时，HIP codegen 报 `Bad bit-width for float: float8_e5m2`（`PrintConst`）；gfx942 和 gfx950、e5m2 和 e5m2fnuz 都一样，e4m3 常量正常，CUDA 正常。
- 现象 2：gfx950 上 `float8_e4m3` 的 `T.gemm` 报 `KeyError: "dtype('float8_e4m3')"`（`_initialize_mfma_prefix`），MFMA 类型表只有 `float8_e4m3fn`；CUDA 后端接受 `float8_e4m3`。这一条也可能被上游视为“类型不支持但报错不友好”。
- 复现：`tl_amd_probe.py fp8_const gfx950 float8_e5m2`、`tl_amd_probe.py fp8_gemm gfx950 float8_e4m3`（只降级）。

**TileLang#26 8 位整数 xor 在 HIP codegen 中失败**
- 现象：int8/uint8 的 `a ^ b`（如 `T.cast(X[i, j], "uint8") ^ T.cast(3, "uint8")`）在 gfx942、gfx950、gfx1100 上报 `Check failed: (p) is false:`（空消息，`VisitExpr_`）。int16/int32 和 CUDA 正常。
- 复现：`tl_amd_narrow_select.py gfx942 uint8 xor`（只降级）。

以上 TileLang#22–#26 在 TileLang 0.1.15 上复验仍然存在。

### 10-15 清理老候选和非切片待定桶时确认（sm_89）

**TileLang#27 bf16 `T.pow` 被拒绝**（原老候选）
- 现象：`T.pow(x, y)`（x 为 bfloat16、y 为浮点）报 `Check failed: (x.dtype().is_float()) is false: power only applies to float`。同样的 bf16 输入用 `T.exp`、`T.sqrt`、`T.log` 都正常，`T.pow` 的文档也没有类型限制。
- 根因：TVM 的 `pow` 用 `is_float()` 检查类型，而 TVM 中 bfloat16 不算 `is_float()`。整数指数走 `pow_of_int`，不受影响。0.1.14、0.1.15 都存在。另外，0.1.14 上 f16 `T.pow` 生成未定义的 `hpow`（属 TileLang#12 一族），0.1.15 已修。
- 复现：`tl_pow_bf16.py bfloat16 pow`。

**TileLang#28 显式 order/stage 的流水线把 fragment 多缓冲**（原老候选 ×2 合并）
- 现象：`T.Pipelined(n, num_stages=2, order=[0, 1], stage=[0, 1])` 的循环体先 `T.copy` 到 fragment，再在 `T.Parallel` 中使用。编译报 `Fragment buffer access with non-zero index [1] is not supported. Only fragment[0] access is allowed within T.Parallel loop.`；循环体是“拷到 fragment → `T.reduce_sum` → 累加”时报 `Check failed: vars.size() == InputDim() (1 vs. 2)`（`Layout::Forward`）。同一程序去掉 order/stage、只给 `num_stages=2` 时编译运行都正确；暂存在共享内存时也正确。
- 根因：显式 stage 时，流水线把在 stage 0 写、stage 1 读的 fragment 也做成双缓冲，多出来的一维违反了 fragment 只能用下标 0 访问的约束，也让布局维数对不上。用户代码中并没有这样的访问。0.1.14、0.1.15 都存在。
- 复现：`tl_pipelined_stage_fragment.py fragment explicit 2`（对照 `fragment auto 2`）、`tl_pipelined_stage_reduce.py fragment explicit`。

**TileLang#29 浮点 `x - x`、`x * 0` 被化简成 0**（来自非切片路线的 2 个待定桶）
- 现象：float16/float32 的 `X[i] - X[i]`、`X[i] * 0`、`X[i] + 0 - X[i]` 被直接编译成常量 0（生成的 CUDA 为 `O[...] = 0x0p+0f`），x 为 inf、-inf、NaN 时得 0，按 IEEE 应为 NaN。`x / x` 没有被化简，结果正确。TileLang 默认不开 fast math（`tl.enable_fast_math` 默认 False）。
- 实验中：TileLang 的 elementwise/control 程序里 `e24 = e23 - e23`（e23 为 NaN）得 0，`e75 = e74 - e74` 使后续结果为 -inf 而不是 NaN（32906 wrong_result:f62d92c68b、41790 wrong_result:d9dc98c83a）。
- 根因：TVM 算术化简器按实数规则化简浮点表达式。上游可能认为这是 TVM 的既有行为，但与 TileLang 默认的 IEEE 语义不一致。0.1.14、0.1.15 都存在。
- 复现：`tl_float_x_minus_x.py float32`。

**TileLang#30 gemm 累加器与另一个 fragment 的行统计量组合时布局推断失败**（原老候选，09-27 起反复出现）
- 现象：`T.gemm(As, Bs, acc)` 之后，把另一个从全局内存读入的 fragment `wide` 做 `T.reduce_sum(wide, stat, dim=1)`，再算 `out[i, j] = stat[i] + acc[i, j]`，报 `Check failed: (has_best) is false: no available layout found`。threads 为 128 或 256 都失败；`wide` 改为来自 `acc` 时正常。
- 分析：`wide` 的布局不受约束，推断器本可以按 `acc`（gemm 决定）的布局为 `wide`、`stat` 选择兼容的布局，却在候选中找不到可行解。这是“行统计量 + 累加器”的常见写法。原始用例（fresh 路线 layout_inference:065f998191，3000 多行）经自动块删除约简到 51 行，再手工删到上面的形式。0.1.14、0.1.15 都存在。
- 复现：`tl_layout_has_best.py 256 other`（对照 `256 acc`）。
---

## 三、候选（未计入确认数）

### 老候选（10-15 清理）

| DSL | 现象 | 结论 |
|---|---|---|
| TileLang | `T.pow` 用于 bfloat16 报 `Check failed: x.dtype().is_float()` | 确认，TileLang#27 |
| TileLang | `T.Pipelined` + `num_stages=2` + order/stage 把 fragment 暂存的拷贝多缓冲，报 `Fragment buffer access with non-zero index is not supported` | 确认，TileLang#28 |
| TileLang | `T.Pipelined` 带 order/stage、循环体是 fragment 暂存的归约时 `Layout::Forward` 失败 | 与上一条同根因，并入 TileLang#28 |
| TileLang | 布局推断 `Check failed: (has_best) is false` | 确认，TileLang#30（约简到 gemm 累加器 + 另一 fragment 的行和） |

### 跨架构候选（b1eded9f 实验中出现；10-15 已全部清理）

12 个跨架构候选都已在本地（RTX 4060，Triton 3.8.0 / TileLang 0.1.14 与 0.1.15）用最小脚本复现并归类：

| DSL | 目标 | 原现象 | 结论 |
|---|---|---|---|
| Triton | sm80 | f16→f8e5 rtne 平局舍入错 | 确认，Triton#18 |
| Triton | sm80 | fp8e5 `padding=nan` 得 −0.0 | 与上一条同根因，并入 Triton#18 |
| Triton | gfx1100 | AMDGCN 被汇编器拒绝，随后 LLD 报错 | 确认，Triton#19（LLVM AMDGPU） |
| Triton | sm100/sm120 | AutomaticWarpSpecialization 断言 | 确认，Triton#21 |
| Triton | gfx942 | `idx < size()` 断言（`kpack=2`） | 确认，Triton#20 |
| TileLang | sm80 | fp8 gemm 运行时 device assert | 确认，TileLang#22 |
| TileLang | sm75 | `the Stmt is not SSA` | 确认，TileLang#23 |
| TileLang | sm90 | `Not a canonical GMMA_MN layout` | 确认，TileLang#24 |
| TileLang | gfx950 | OCP fp8 `PrintConst` / `_initialize_mfma_prefix` KeyError | 确认，TileLang#25 |
| TileLang | gfx942/950 | `Check failed: (p)`（空消息） | 确认，TileLang#26（8 位整数 xor） |
| TileLang | gfx942/950 | `PrintVecElemStore` 的 type_name 为空 | TileLang#18 的变体：只在 `tl.config_index_bitwidth=64` 时出现，同一配置在 CUDA 上报 `Cannot convert type int64x8` |
| TileLang | gfx942 | gemm 化简器除零 | 不算 DSL bug：K=32 时 `k_pack=2` 本身非法（A 在共享内存时 TileLang 明确报 `block_K must be >= micro_size_k * k_pack`，A 在 fragment 时漏了这个检查，化简时除零）；属于 fuzzer 的合法化缺口，TileLang 只是缺少检查 |

另有 nvcc 12.4 编译 sm_90a 时段错误（CUDA 工具链问题，不计入 DSL）。

---

## 四、附：不算真实 bug 的情况

以下在实验中出现过，但已查明是 fuzzer 自身的问题，不计入上表：

- **harness / 参考模型错误（已修 11 类）**：如 Triton 窄类型 `tl.max/min` 的提升规则、libdevice 返回类型、cdiv 公式、flush-to-zero、近似函数参考溢出等，在 637deff6 之前修复，`known_mechanisms.json` 中以 `_FIXED` 结尾，若重现会报警。
- **错误信息截断**：目标明确不支持的程序（如 sm80 不支持 fp8e4nv）因拒绝标记被截掉而被记为失败；d6046da9 已修。
- **合法化缺口**：fuzzer 为目标生成了它不支持的参数（AMD 上 tf32 gemm、TileLang AMD 的 block_K 已修；TileLang HIP 的 nan_propagate、MFMA k_dim、Triton gfx90a 上强开 async copy、TileLang AMD 上 A 在 fragment 时 K < micro_size_k × k_pack 未修）。
- **进程崩溃**：layout 切片合法化不幂等导致 vGPU Triton 进程退出，ea9ecc41 已修。
- **非切片路线的数值判定不稳定（10-15 清理，11 个待定桶）**：b1eded9f 中 fresh/mutate 路线的 precision、identity、schedule/pass-config 不变性、structured reference 不一致，除 TileLang#29 的两个外都属于这一类。本地全部稳定复现，但误差来自判定本身：
  - pass-config 不变性（Triton，af0ff50a66）：变体只差 `enable_fp_fusion=True`；程序中 `x*x - x*x` 被合并成 FMA 后得到 x² 的舍入误差（不再是 0），再被循环放大到 0.25。FMA 合并是该选项的本意（也是 Triton 的默认值），检查应允许它。
  - structured reference（Triton，20d6491ff5）：`sqrt(|行和|)` 在行和接近 0 时条件数极差，求和顺序不同就造成较大相对误差；独立脚本 `tr_structured_605af3d.py` 中去掉 scratch 写入或 barrier 结果一样。
  - schedule 不变性等（764d8d706c 等）：float16 累加之后接 `round`/`floor`/`max` 等不连续运算，不同调度的累加顺序使值跨过取整边界。
  - precision（`_prec` 变体，fp16 累加）和 identity：float16 累加顺序不同带来的差异超过 2% 容差。
  - 这些属于 fuzzer 缺陷（判定没有覆盖 FMA 合并、病态运算和不连续运算），按约定只记录，不在实验中修改。
- **噪声**：共享内存超限、warp 划分不合法、超时、oracle 不稳定、Blackwell 上寄存器上限过低等，按资源限制处理。
