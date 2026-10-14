# TileSmith 发现的真实 bug 总表（截至 2026-10-14）

本表汇总 TileSmith 从 2026-09 起所有实验（早期 hard-shape 实验、特征切片三轮 82c18be2 / b9e08b4c / 637deff6、跨架构 b1eded9f→ea9ecc41）以及本地探测中确认的真实 bug，按 bug 本身组织，不按发现轮次。

“确认”的标准：有不依赖 fuzzer oracle 的独立复现（最小脚本或原样重放），并排除了 harness / 参考实现 / 类型模型的问题；同一根因在不同程序、不同服务器上多次出现只计一种。复现脚本除特别注明外都在 `reports/2026.10.09/feature_slices/`。

## 总览

| 分组 | 数量 |
|---|---|
| 已确认、已编号（#1–#29 + 早期 E1–E5） | **34** |
| 只在非本机目标出现、已有独立复现（X1–X4） | **4** |
| 合计确认 | **38**（TileLang 21、Triton 17） |
| 候选（证据不足，未计入） | 16（老候选 4 + 跨架构候选 12） |

按后果分：

| 后果 | 数量 | 编号 |
|---|---|---|
| **静默错误结果**（编译运行都“成功”，数值错） | 14 | #1 #2 #11 #12 #13 #14 #15 #20 #22 #23 #24 #27 #29，以及 #7 的错误结果形式 |
| 运行时错误（越界写、非法访存、对齐错误） | 2 | #7、#28 |
| 编译器崩溃 / 断言 / LLVM abort（无清晰报错） | 9 | #3 #4 #5 #21 #25 #26 E2 E5，X1 |
| 生成的代码被下游拒绝（nvcc/LLVM/codegen 报错） | 14 | #6 #8 #9 #10 #16 #17 #18 #19 E1 E3 E4，X2 X3 X4 |

#7 同时有错误结果和越界写两种表现，在两行都列出；#27 以错误结果为主，也有编译失败形式。

按责任方分：Triton 编译器 12 种；NVIDIA 工具链（ptxas / libdevice）经 Triton 触发 5 种（#1、#12、#24、#28、#29）；TileLang/TVM 21 种。

上游状态：E1、E3、E4 上游已有相同报告（TileLang #2206、#2172，Triton #10790），不算首次发现；其余尚未系统查重或提交。

---

## 一、Triton

### 数值错误

**#1 f16/bf16 上的 `tl.abs` 有一半元素仍为负**（责任方：ptxas 12.9）
- 现象：`tl.abs` 作用于 f16/bf16（最初是 `tl.abs(tl.dot(a, b))` 存成 fp16），输出交替为 [-1, 2, -3, 4, …]。不需要 dot：256 个元素、`num_warps=2`、f16→abs→int32 即可触发；abs 在 f32 中算完再收窄到 f16 同样触发。
- 根因：Triton 生成的 PTX 对每个 half 都做 `and.b16 0x7fff`，是正确的；Triton 3.8 自带的 ptxas 12.9 用 `F2FP.PACK_AB` 把两个 half 打包后，只对高半部分（`PRMT 0x7632`）做掩码，低半部分保留符号位。同一份 PTX 用 ptxas 12.4 编译结果正确。
- 复现：`abs_dot.py`、`abs_f16_elementwise.py`。
- 10-14 在 4090 上排除 fuzzer 因素（脚本 `claude_work/verify_abs/`，不含 TileSmith 代码）：把同一份 PTX（逐字节相同，每个 half 都有 `and.b16 ..., 32767`）分别用 ptxas 12.9 和 12.4 汇编，再用 CUDA 驱动直接加载运行，绕过 Triton。12.9 的 cubin 有 266/1024 个负数输出，12.4 全部正确。SASS 中 12.9 每个线程 8 个 half 只生成 4 条 `LOP3.LUT ... 0x7fff`（只给 `PRMT 0x7632` 取出的高半部分），`PRMT ..., 0x5410, RZ` 取出的低半部分没有掩码；12.4 生成 8 条。

**#2 f32→fp8 默认舍入错误**
- 现象：f32 转 e4m3/e5m2：0.53125006→0.5（应 0.5625），36.000004→32（应 40）。
- 根因（PTX 已确认）：先 `cvt.rz.f16.f32` 向零舍入到 fp16，再 `cvt.rn.satfinite.e5m2x2.f16x2`；中间的 RZ 丢掉了粘滞位，就近舍入的平局判断出错（双重舍入）。
- 复现：`f32_to_fp8.py`。

**#12 常量折叠后转 fp8，部分 lane 是垃圾值**（责任方：ptxas 12.9）
- 现象：编译期被折叠成常量的值（如非负输入的 `min(x, 0)`）转 fp8 后存储，每 4 个元素错 1–2 个，出现 -0.0、-inf、-384 等。
- 根因：Triton 的 PTX 正确；ptxas 12.9 生成 `F2FP.SATFINITE.E5M2.F16.UNPACK_B_MERGE_C` 或 `...E4M3.F32.PACK_AB_MERGE_C ... R1`，未用到的字节从无关寄存器（甚至栈指针 R1）合并进来。ptxas 12.4 正确。
- 复现：`ptxas_const_fp8.py`。

**#13 带掩码的向量化窄整数加载，负填充值读错**
- 现象：int8/int16/uint8 的 `tl.load(..., mask, other=负数)`（或 uint8 ≥128）被向量化时，被掩码的元素除每个 32 位字的第一个外都读出 -1/255。
- 根因：PTX 中填充值是 `mov.u32 %r, 0xfffffffffffffffe`（符号扩展后的标量），而不是按字节重复的 0xfefefefe。ptxas 12.4 结果相同，是 Triton 的 PTX 错误。int32、浮点和正的填充值正确。
- 复现：`masked_load_negative_other.py`。

**#24 有符号 `libdevice.mul24` 返回完整 48 位乘积**（责任方：NVIDIA libdevice，经 Triton 触发）
- 现象：`libdevice.mul24(x, y)` 的结果只被加宽使用（`.to(tl.int64)`）时，返回完整乘积而不是文档规定的“低 32 位”，128/128 错。
- 根因：Triton 附带的 `libdevice.10.bc` 中 `__nv_mul24` 是 `shl/ashr 8` 后 `mul nsw i32`；两个 24 位数的乘积可达 2⁴⁶，`nsw`（不会有符号溢出）的声明不成立，NVPTX 据此把 `sext(mul nsw)` 合并成 `mul.wide.s32`。无符号 `umul24` 和普通 `x * y` 正确。
- 复现：`tr_libdevice_mul24_nsw.py`。

**#29 bf16 原子加在 `maxnreg=32` 下丢失更新**（责任方：ptxas 12.9）
- 现象：bf16 `tl.atomic_add`（`sem='acquire'` 或 `'acq_rel'`），`num_warps=1, maxnreg=32` 时一个槽得 10，应为 12，确定性复现。
- 根因：bf16 原子加在 sm_89 上降级为 32 位 CAS 循环加 fence。加与不加 maxnreg 的 PTX 只差 `.maxnreg 32` 一行，两者都没有寄存器溢出，所以错误出在 ptxas 对 CAS 循环的寄存器分配。maxnreg≥40、num_warps≥2、`sem='relaxed'` 都正确。
- 复现：`tr_maxnreg_atomic_cas_lost_update.py`。

### 运行时错误

**#28 f16 dot 在 `maxnreg=128` 下非法访存**（责任方：ptxas，12.4 和 12.9 都有）
- 现象：f16 `tl.dot`（32×64 @ 64×128，f32 累加），`num_warps=1` 加文档化的启动选项 `maxnreg=128`，运行报 `illegal memory access`。
- 分析：maxnreg 只应限制寄存器、强制溢出（138 次溢出，552 字节栈帧），不应改变语义。不加 maxnreg（214 寄存器）、maxnreg=255、num_warps=2 都正确；把存储改成 `acc * 1.0 + 0.0` 也能通过，说明依赖具体的溢出代码。
- 复现：`tr_maxnreg_spill_illegal_access.py`。
- 10-14 在 4090 上定位（脚本 `/root/autodl-tmp/abs_repro/maxnreg_ptxas.py`）：加与不加 maxnreg 的 PTX 只差 `.maxnreg 128` 一行。同一份 PTX 用 ptxas 12.9 和 12.4 汇编后经 CUDA 驱动运行都报非法访存（CUresult 700）；`-O0`/`-O1` 正确、`-O2`/`-O3` 出错；`.maxnreg` 改成 96、160、200 都正确；不加 maxnreg 的 cubin 在 compute-sanitizer 下 0 错误。compute-sanitizer 报 `STS.128 [R114]`（把 B 块写入共享内存，共享内存共 20480 字节）越界写，地址 0x5b00–0x7f00。PTX 中该地址是 `((tid & 15) << 4) ^ (bfe(tid, 4, 1) & 272)`，不超过 0x1f0；SASS 中 `R115 = tid & 15` 之后，`/*0cb0*/ IMAD R115, R115, 0x80, R114` 先把 R115 改写成另一个地址（供后面 `LOP3 R40, R115, R40` 用），`/*0cc0*/ SHF.L.U32 R114, R115, 0x4` 才读 R115，读到的是改写后的值，算出 `(tid & 15) * 0x800 + (tid & 7) * 0x100`，与实测越界地址逐一吻合。是 ptxas 在寄存器紧张时的调度/寄存器分配错误，PTX 正确。

### 编译器崩溃

**#3 f64↔fp8 转换 LLVM abort**：`.to()` 在 f64 与 fp8 之间转换时，ConvertTritonGPUToLLVM 中 LLVM abort，进程被杀；前端不报错。AMD 目标上表现为 `Unsupported conversion`。复现：campaign `.min.py`。

**#4 fp8→fp8 转换断言**：报 `Rounding mode must be specified for convertsions to fp8`，断言杀掉进程，而不是前端报错。

**#5 `fp_downcast_rounding='rtz'` 用于 f64 源或 fp8 目标时 pass 崩溃**；AMD 上为 `maybeOutVals` 断言。复现：`probe_round_tr.py`、`tr_round.jsonl`。

**#21 `tl.gather` 沿 axis 0 断言**
- 现象：秩 3 的 tile 沿 axis 0 做 `tl.gather`，下标 tile 比被 gather 的那一维更长时（源 (2,16,4)、下标 (4,16,4)、num_warps=8），`OptimizeThreadLocality.cpp:220` 断言 `GatherLoweringHelper(op).isWarpLocal()` 失败，进程被杀。
- 两台服务器各自独立发现，本地复现。复现：`tr_gather_warp_local_assert.py`。

**#25 warp specialization 循环里有扫描时 `llvm_unreachable`**
- 现象：`tl.range(..., warp_specialize=True)` 的循环体里有 `tl.cumsum` 或 `tl.associative_scan`，报 `UNREACHABLE executed at .../WarpSpecialization/TaskIdPropagation.cpp:107`（`Other parent ops are not supported`）。
- 根因：`TaskIdBackwardPropagation::propagateToParent` 只接受 scf.for、scf.if、tt.func 和 tt.reduce 作为父操作，没有处理 `tt.scan` 的 combine 区域（上游 main 至今如此）。同一循环里的 `tl.sum`、自定义 `tl.reduce` 能编译；sm_80/89/90 失败，sm_100 能编译。
- 复现：`tr_warp_specialize_scan_unreachable.py`（只编译，不需要 GPU）。

**E5 `TritonGPURemoveLayoutConversions` 生成违反 SSA 支配关系的 MLIR**
- 现象：含一致分支、全局 scratch `tl.store` 和 `tl.debug_barrier` 的程序，编译器报 `operand #0 does not dominate this use`，失败 pass 为 TritonGPURemoveLayoutConversions。
- 证据：09-30 实验中原样复跑复现；尚未化简为最小样例。

### 编译失败（生成的 IR 不合法）

**#6 bf16 上自定义 combine 的 `tl.reduce` / `tl.associative_scan` 编译失败**：combine 用 `tl.maximum` 等时，combine 区域返回 f32 而不是 bf16（前端类型提升问题）。

**#16 `tl.fdiv` 用于 fp16/bf16**：ConvertTritonGPUToLLVM 报 `LLVM ERROR: Unsupported bitwidth`；`ieee_rounding=True` 用于 fp16/bf16/fp64 时 LLVM 翻译失败（intrinsic 返回 half/double，而 `llvm.nvvm.div.rn.f` 返回 float）。复现：`tr_fdiv_half.py`。

**E4 `tl.flip(x)` 省略 dim 时编译失败**：内层是整数与 None 比较的 TypeError；文档规定默认 `dim=None`。`dim=0` 通过。上游已有完全匹配的问题（Triton #10790），Triton 3.8.0 仍可触发。

### 只在非本机目标出现（已有独立复现）

**X1 gfx942/gfx950 上任何 `tl.atomic_xchg` 都无法编译**
- 现象：i32/i64/f32/f64 的 `tl.atomic_xchg` 报 `LLVM Translation failed`。
- 根因：buffer atomic 降级生成 LLVM 内建 `llvm.amdgcn.raw.ptr.buffer.atomic.exch`，Triton 自带的 LLVM 里只有 `...buffer.atomic.swap`。gfx90a/gfx1100 不走 buffer atomics，能编译；`AMDGCN_USE_BUFFER_ATOMICS=0` 也能编译。
- 复现：`tr_amd_atomic_xchg.py`。

---

## 二、TileLang

### 数值错误

**#7 fp16 `T.atomic_add` 被向量化后忽略下标映射**
- 现象：fp16 原子加被向量化为 `AtomicAddx2`，忽略目标下标映射：写出界（输出两侧的 canary 被改）、结果错、或 misaligned address。64 线程写单个 slot 时 O[1] 被加 64。
- 复现：`atomic_f16.py`。

**#11 int16/int32→bfloat16 的 `T.cast` 截断而不是就近舍入**
- 现象：259→258（应 260）。int→float16 正确。
- 根因：生成 `(bfloat16_t)x`；cutlass 的 `bfloat16_t(int)` 先转 float，再直接取高 16 位（`bits >> 16`），没有舍入。
- 复现：`int_to_bf16.py`。

**#14 while 循环里拷贝到共享内存再 gemm，缺少同步**
- 现象：用 `T.alloc_var` 计数的 while 循环中先拷贝到共享内存再 `T.gemm`，20 次运行 15 次错、10 种不同输出；`T.serial` 正确。
- 根因：生成的 CUDA 只在拷贝与 gemm 之间有 `__syncthreads()`，循环体末尾没有，下一轮拷贝与上一轮 gemm 的 ldmatrix 读冲突（数据竞争）。
- 复现：`tl_while_gemm_race.py`。实验中出现次数最多的机制之一。

**#15 窄整数运算后加宽，不按窄类型回绕**
- 现象：int8/uint8/int16/uint16 的加、减、乘、移位后 `T.cast` 到更宽类型，结果是未回绕的值；直接存成窄类型时正确。
- 根因：生成 `((int64_t)((int)(a + b)))`，C 的整数提升保留了未回绕的值，而 TIR 中该运算的类型是窄类型。int32 也有类似路径（乘法先加宽到 int64 再乘），但属于有符号溢出，上游可能视为未定义行为；结论主要由回绕语义明确的无符号情形支撑。
- 复现：`tl_narrow_int_widen.py`、`tl_int32_widen_in_wide_type.py`。

**#20 共享内存取模下标重排丢掉列偏移**
- 现象：`b1[i,j] = b0[(i+3)%16, (j+2)%16]` 的列下标被生成为 `(threadIdx.x & 7) * 2`，`+2` 和 `%16` 都消失，结果静默等于“只换行不换列”；32×32 时生成正确。确定性，不是竞争。
- 触发：行、列偏移同时存在；与向量化无关（标量读也出现）；是否触发取决于形状和线程数怎样拆分下标。
- 复现：`tl_remap_column_shift_dropped.py`、`tl_remap_column_shift_dropped_scalar.py`。

**#22 嵌套整数 floordiv 的不可靠化简**
- 现象：`T.floordiv(T.floordiv(x, 3) * -1, 4)` 被编译成 `floordiv(-x, 12)`，x = 1、2 时得 −1，应为 0；`* -2` 被合并成 `floordiv(-x, 6)`，−8..8 中 6/17 个输入错。正乘数正确。
- 根因：把取负移进内层 floordiv 不成立：−⌊x/3⌋ 等于 ⌈−x/3⌉ 而不是 ⌊−x/3⌋。数值很小，与溢出无关。
- 复现：`tl_nested_floordiv_wrong_value.py`。

**#23 区域拷贝降级为 ldmatrix 时丢掉列偏移**
- 现象：`T.copy(b1[0:64, 8:16], b2)`（2 字节类型、threads=256）生成 `ptx_ldmatrix_x1(&b1[(tid >> 5) * 128 + (tid & 7) * 16], ...)`，没有 `+ 8`，片段静默拿到第 0–7 列。
- threads=128（普通 uint2 加载，带偏移）、4 字节类型、偏移为 0 都正确。效果与 #20 相同，但降级路径不同。
- 复现：`tl_ldmatrix_region_offset_dropped.py`。

**#27 `tl.config_index_bitwidth=64` 改写了存储的值**
- 现象：这个 pass 配置（文档：“配置索引的位宽”）不只改索引：(1) 错误结果：`O[i] = T.cast(T.cast(X[i], "int8"), "int32")` 原样返回 X[i]（−640 得 −640，应 −128）；(2) 编译失败：int8/int16 元素运算变成 int64x16 向量运算，报 `Cannot convert type int64x16 to CUDA type`。
- 根因：`ConfigIndexBitwidthRewriter` 为改索引，在访问整个 BufferStore/BufferLoad 时打开改写开关，被存储的值里所有窄于 64 位的有符号整数常量和 Cast 也被改成 int64。默认配置正确。
- 复现：`tl_index64_rewrites_stored_values.py`。

### 编译器崩溃 / 断言

**#26 打开 `tl.enable_lower_ldgstg` 后，向量元素类型的全局访问失败**
- 现象：`Check failed: tgt_dtype.lanes() * tgt_dtype.bits() == src_dtype.lanes() * src_dtype.bits() (32 vs. 64)`。
- 根因：`lower_ldg_stg.cc` 用 `buffer->dtype.bits()`（单个 lane 的位宽）决定访问宽度，忽略向量元素的 lanes：float32x2 元素被生成 `ldg32`，再把 32 位结果 reinterpret 成 64 位。float32x4、int32x2 同样失败。
- 复现：`tl_ldgstg_vector_lanes.py`。

**E2 自动选择的归约布局无法降级**
- 现象：`ReduceOp cannot lower a layout where a source index depends on a thread-owned reduce segment`（`reduce.h:534`）。
- 证据：只用 fragment、copy、gemm、reduce 等正常 API，布局由编译器推断；单独编译仍失败，关闭向量化或 async copy 仍失败。

### 生成的代码被拒绝 / codegen 失败

**#8 嵌套整数 floordiv codegen 崩溃**：threads=32 时向量化，报 `Let cannot bind the same var to two different values`；int8/16/32/64 都复现。复现：`floordiv_i16.py`。

**#9 int64→float16/bfloat16 的 `T.cast` 被 nvcc 拒绝**：`half_t` 构造函数歧义。

**#10 int64 的 `T.atomic_max/min/or` 被 nvcc 拒绝**：`atomic_ref<unsigned long long>` 不匹配。

**#17 向量化的 `T.bitwise_not` 对 CUDA 向量类型生成 `~`**：nvcc 报 `no operator "~" matches these operands`（int4、ulonglong2 等）。复现：`tl_intrinsic_lowering.py`。

**#18 float16/bfloat16 上的 `T.nearbyint`/`T.fmod`（及 `T.tanh`）生成未定义的 `hnearbyint`/`hfmod`/`htanh`**。复现：`tl_intrinsic_lowering.py`。

**#19 缺少 CUDA 降级规则的内建函数到达代码生成**：有符号 int32/int64 的 `T.popcount`，float16/bfloat16 的 `T.copysign/nextafter/ldexp/hypot`，报 `Unresolved call ir.Op(tirx.*)`。复现：`tl_intrinsic_lowering.py`。

**E1 bool 向量 CUDA 代码生成失败**：`Cannot convert type boolx8/boolx16 to CUDA type`；关闭向量化后通过。上游已有相同症状报告（TileLang #2206）。

**E3 流水线拷贝生成非法的 2 字节 cp.async**：`IsValidCPAsyncTransferBytes(total_bytes)` 失败（cp.async 要求 4/8/16 字节）。最小化为 A=(1,9157)、B=(9157,3709)、64×64×128 块、256 线程的 GEMM，`num_stages=4` 失败、`num_stages=0` 通过。上游已有相关报告（TileLang #2172）。

### 只在非本机目标出现（已有独立复现）

**X2 sm100/sm120 上 int16 逐元素运算无法打印向量类型**：最简单的 `O[i] = X[i] + X[i]`（512 元素、threads=32）报 `Cannot convert type int16x16 to CUDA type`。Blackwell 上向量化宽度变为 256 位，CUDA 打印器没有 int16x16；sm_89/sm_90a 用 128 位 int16x8 正常。实验中也被命中。复现：`tl_blackwell_256bit_vector_type.py`。

**X3 sm120 上标准流水线 GEMM 构建失败**：`T.Pipelined` + `T.gemm` 被 warp specialization 加上 `tl::warpgroup_reg_alloc/dealloc`（PTX `setmaxnreg`），sm_120（RTX 50 系列）没有这条指令，nvcc 停在 `static_assert`；sm_90a、sm_100a 正常。复现：`tl_sm120_setmaxnreg.py`（需 nvcc ≥ 12.8）。

**X4 AMD 上 IEEE 舍入内建没有降级规则**：gfx942/gfx950 上 `T.ieee_add/sub/mul/fmaf/fsqrt/frcp/fdiv` 报 `Unresolved call tl.ieee_*`（CUDA 正常）。与 #19 同类（缺少降级规则），但在 HIP 后端。复现：`tl_hip_ieee_unresolved.py`。

---

## 三、候选（未计入确认数）

### 老候选（实验中反复出现，但未证明是合法程序）

| DSL | 现象 | 未确认的原因 |
|---|---|---|
| TileLang | `T.pow` 用于 bfloat16 报 `Check failed: x.dtype().is_float()`（TVM 中 bf16 不算 is_float） | 可能被上游视为不支持 |
| TileLang | `T.Pipelined` + `num_stages=2` + sync 注解把片段暂存的拷贝多缓冲，报 `Fragment buffer access with non-zero index is not supported` | 未确认该组合是否合法 |
| TileLang | `T.Pipelined` 带 order/stage、循环体是片段暂存的归约时，`Layout::Forward` 推断失败 | 同上 |
| TileLang | 布局推断 `Check failed: (has_best) is false` | 09-27 已独立复现，但未证明是受支持的布局组合 |

### 跨架构候选（b1eded9f 实验中出现，未约简）

| DSL | 目标 | 现象 |
|---|---|---|
| Triton | sm80（真实执行） | f16→f8e5 rtne 平局舍入错（28672 vs 24576） |
| Triton | sm80（真实执行） | fp8e5 `padding=nan` 结果不是 nan（得 −0.0），两台服务器各自触发 |
| Triton | gfx1100 | 生成的 AMDGCN 被汇编器拒绝（`src1 immediate operand invalid`），随后 LLD 报错（i16 `cmax` 扫描、i8 `flip`） |
| Triton | sm120 | AutomaticWarpSpecialization 断言 |
| Triton | gfx942 | `idx < size()` 断言（`kpack=2`） |
| TileLang | sm80（真实执行） | fp8 `T.gemm` 选用 sm_89 才有的 `SM89_16x8x32` MMA，运行时 device assert |
| TileLang | sm75 | 只在 sm75 + `tl.loop_unswitching_allow_non_trivial_else` 时报 `buffer accum has already been defined, the Stmt is not SSA` |
| TileLang | sm90 | `Not a canonical GMMA_MN layout` 断言 |
| TileLang | gfx950 | OCP fp8：`PrintConst` 失败 / `Bad bit-width for float: float8_e5m2` / `_initialize_mfma_prefix` KeyError |
| TileLang | gfx942/950 | i16/u8 unroll 循环 `Check failed: (p) is false:`（空消息） |
| TileLang | gfx942/950 | `PrintVecElemStore` 的 type_name 为空（可能属 #27 族） |
| TileLang | gfx942 | gemm 化简器除零 |

另有 nvcc 12.4 编译 sm_90a 时段错误（CUDA 工具链问题，不计入 DSL）。

---

## 四、附：不算真实 bug 的情况

以下在实验中出现过，但已查明是 fuzzer 自身的问题，不计入上表：

- **harness / 参考模型错误（已修 11 类）**：如 Triton 窄类型 `tl.max/min` 的提升规则、libdevice 返回类型、cdiv 公式、flush-to-zero、近似函数参考溢出等，在 637deff6 之前修复，`known_mechanisms.json` 中以 `_FIXED` 结尾，若重现会报警。
- **错误信息截断**：目标明确不支持的程序（如 sm80 不支持 fp8e4nv）因拒绝标记被截掉而被记为失败；d6046da9 已修。
- **合法化缺口**：fuzzer 为目标生成了它不支持的参数（AMD 上 tf32 gemm、TileLang AMD 的 block_K 已修；TileLang HIP 的 nan_propagate、MFMA k_dim、Triton gfx90a 上强开 async copy 未修）。
- **进程崩溃**：layout 切片合法化不幂等导致 vGPU Triton 进程退出，ea9ecc41 已修。
- **噪声**：共享内存超限、warp 划分不合法、超时、oracle 不稳定、Blackwell 上寄存器上限过低等，按资源限制处理。
