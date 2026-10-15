# 特征切片（feature slices）：新增 bug 类型的方法与首批结果

2026-10-09，分支 `claude/feature-slices-20261009`（fbd08fb4 → 82c18be2），TileLang 0.1.14 / Triton 3.8.0 / torch 2.4.0+cu124。

## 为什么旧方法只出重复 bug

8ddfbb9d 两台各跑 9.6 h：TileLang 只有 3 种已知机制（bool 向量类型、ReduceOp 布局、2 字节 cp.async），Triton 0 种。调度、隔离、swarm 只能重排生成器**能表达**的程序；Region/Extended IR 只覆盖 fp16/fp32/int8/int32/bool、秩 ≤ 2、内置归约组合，已经采样饱和。本地探针（`claude_work/explore/`）表明，大量崩溃位于这个空间之外：bf16/fp8/fp64/16-64 位整数、无符号数、秩 3、自定义 combine、MMA 操作数布局、原子操作的 dtype 与竞争模式。

## 方法与文献对应

| 机制 | 文献 | 实现 |
|---|---|---|
| 每个 bug 密集特征一个聚焦生成器，其余维度保持简单 | FFTG（Zamudio Amaya, Böhme, Zeller 等，ASE'26）；tile bug 实证（ISSTA'26：类型/算子占 49%） | `src/workflow/slices/` 六个切片：cast、reduce、scan、gemm、atomic、round |
| 精确参考 + 逐位比较，避免数值噪声 | YARPGen 取值域追踪 | 二进分数输入，逐步传递取值域，不可精确表示的参数在合法化阶段替换 |
| 按 Good-Turing 估计“每秒发现新桶/新参数对”的概率分配预算 | STADS（Böhme, TOSEM'18）；Entropic | `slices/scheduler.py` |
| 切片内选覆盖最多未覆盖参数对的候选 | 组合测试（AETG） | 同上 |
| 失败逐参数约简为核心，错误结果按核心分桶，命中多次的核心降采样 | 测试约简去重（spirv-fuzz, PLDI'21）；已有 quarantine | `slices/minimize.py`、`fuzzer._minimize_slice` |
| 与位置无关的路径特征，供 quarantine 学规则 | — | `slices.path_features`（如 `path:conv f64>f8e4`） |

未采用：LLM/agent 生成模板（Gentoo、CodaMOSA、ChatAFL）需要接入模型 API，留作后续；编译器分支覆盖插桩需要重编 Triton/TVM。

## 部署

两台服务器 20:58 冷启动，seed 4090=43、vGPU=42（不再同 seed），参数同上一轮 + `--slice-prob 0.85`；22:33 与 23:00 两次带断点更新代码后 resume（修复见提交记录）。管理目录 `/root/autodl-tmp/tilesmith_campaigns/20261009_feature_slices/`，结果 `results/2026.10.09-20.58_*`。

## 1.8 h 时的对比（非噪声桶，同一时长）

| 运行 | 新 | 旧 8ddfbb9d 同时长 | 旧 9.6 h 全程 |
|---|---:|---:|---:|
| vGPU TileLang | 8 | 3 | 8 |
| vGPU Triton | 13 | 0 | 2 |
| 4090 TileLang | 7 | 4 | 9 |
| 4090 Triton | 8 | 0 | 1 |

桶不等于机制：TileLang 的 int64→half 有 6 个桶（类型名不同）。按机制计，下表是独立复现过的新机制。前后对比夹杂 seed 与版本差异，不是严格 A/B。

## 已用独立最小脚本复现的新机制

| # | DSL | 现象 | 最小复现 | 备注 |
|---|---|---|---|---|
| 1 | Triton | f16/bf16 上的 `tl.abs` 有一半元素仍为负；最初由 `tl.abs(tl.dot(a, b))` 存成 fp16/bf16 发现，10-10 确认不需要 dot：256 个元素、`num_warps=2`、f16→abs→int32 即可触发 | `abs_dot.py`、`abs_f16_elementwise.py` | Triton 生成的 PTX 对每个 half 都做了 `and.b16 0x7fff`；Triton 3.8 自带的 ptxas 12.9 用 `F2FP.PACK_AB` 把两个 half 打包，只对高半部分（`PRMT 0x7632`）做掩码，低半部分（`PRMT 0x5410`）保留符号，输出交替为 [-1, 2, -3, 4, …]；换 ptxas 12.4 结果正确 |
| 2 | Triton | f32→fp8（e4m3/e5m2）默认舍入错：0.53125006→0.5（应 0.5625），36.000004→32（应 40） | `f32_to_fp8.py` | 根因（PTX 已确认）：先 `cvt.rz.f16.f32`（向零舍入到 fp16），再 `cvt.rn.satfinite.e5m2x2.f16x2`；中间的 RZ 丢掉了粘滞位，平局判断出错。服务器 campaign 也独立发现，约简为 `src=f32, dst=f8e5` |
| 3 | Triton | f64↔fp8 的 `.to()` 在 ConvertTritonGPUToLLVM 里 LLVM abort，进程被杀 | campaign `*.min.py` | 前端不报错 |
| 4 | Triton | fp8→fp8 转换触发断言 `Rounding mode must be specified for convertsions to fp8` | 同上 | |
| 5 | Triton | `fp_downcast_rounding='rtz'` 用于 f64 源或 fp8 目标时 pass 崩溃 | `probe_round_tr.py` / `tr_round.jsonl` | |
| 6 | Triton | bf16 上自定义 combine（`tl.maximum`）的 `tl.reduce` / `tl.associative_scan` 编译失败：combine 区域返回 f32 | campaign | 前端类型提升问题 |
| 7 | TileLang | fp16 `T.atomic_add` 被向量化为 `AtomicAddx2`，忽略目标下标映射：写出界（canary 被改）、结果错、或 misaligned address | `atomic_f16.py` | 64 线程写单个 slot 时 O[1] 被加 64 |
| 8 | TileLang | 嵌套整数 `T.floordiv`（threads=32 时向量化）codegen 崩溃 `Let cannot bind the same var to two different values` | `floordiv_i16.py` | int8/16/32/64 都复现 |
| 9 | TileLang | int64→float16/bfloat16 的 `T.cast` 生成的 CUDA 被 nvcc 拒绝（`half_t` 构造函数歧义） | campaign | |
| 10 | TileLang | int64 `T.atomic_max/min/or` 生成的 CUDA 被 nvcc 拒绝（`atomic_ref<unsigned long long>` 不匹配） | campaign | |
| 11 | TileLang | int16/int32→bfloat16 的 `T.cast` 不按就近舍入，而是截断：259→258（应 260） | `int_to_bf16.py` | 10-10 定期检查发现。生成 `(bfloat16_t)x`，cutlass 的 `bfloat16_t(int)` 先转 float，再直接取高 16 位（`bits >> 16`）；int→float16 正确 |
| 12 | Triton | 编译期被折叠成常量的值转成 fp8 后存储，部分 lane 是垃圾值（-0.0、-inf、-384 等） | `ptxas_const_fp8.py` | 10-10 定期检查发现，最初触发方式是 i8 gemm 后接 min/select；06:47 又在 cast 切片中出现：非负输入的 `min(x, 0)` 被折叠成 0，再经 bf16→e4m3，每 4 个元素错 1 个。Triton 的 PTX 正确；ptxas 12.9 生成 `F2FP.SATFINITE.E5M2.F16.UNPACK_B_MERGE_C` 或 `...E4M3.F32.PACK_AB_MERGE_C ... R1`，未用到的字节从无关寄存器（栈指针 R1）合并；改用 ptxas 12.4 后结果正确 |

尚未查上游 issue；3、5 可能被上游视为“不支持”，但都以 LLVM abort/断言崩溃而非前端报错出现。

1 和 12 都是 Triton 3.8 自带的 ptxas 12.9 的问题：PTX 正确，同一份 PTX 用 ptxas 12.4 编译结果正确。10-10 的定期检查中，又有 6 个错误结果在换用 ptxas 12.4 后全部通过，已归入这两类：3 个 gemm（abs 后存成窄类型、i8 gemm 后接 min/select 存成 fp8），3 个 cast 切片（f16→abs→int 两个，常量→fp8 一个）。

## 已排除的误报

- Triton 对窄类型 `tl.max/tl.min` 会先扩宽到 int32/float32，旧类型模型只考虑了 bf16，导致 uint16 除法被拒绝（739fbf58 修复）。
- torch 的 float64→fp16/bf16/fp8 转换会经 float32 两次舍入，不能当参考；round 切片改用 harness 内的正确舍入实现（与 numpy 一致）。
- 82c18be2 的已知 harness 问题（实验结束后再修）：cast 切片在 `operand=input` 时把 fp8 输入操作数 `w` 直接 `.to(int)`，Triton 前端拒绝 fp8↔int 转换（`cannot cast fp8e5 to int8 at w.to`）。合法化只检查了取值域，没检查操作数的转换是否合法。由于报错是 AssertionError，`_slice_reject` 没把它标成 unsupported_feature。

## 第二轮：扩大切片的程序范围（b9e08b4c，2026-10-10）

82c18be2 两台各跑约 13.2 h（vGPU：TileLang 4842 / Triton 10104 个测试；4090：3263 / 6951）后停止。最后 10.5 h 没有出现新机制，最后 2.7 h 的新桶全部对应已知机制，说明六个切片的参数空间已采样饱和。b9e08b4c 新增四个切片，每个都在本地 sm_89 上先探测 API 的实际行为（`claude_work/explore/expand/`），再按探测结果确定参数空间：

| 切片 | 覆盖的程序 | 参考 |
|---|---|---|
| `math` | Triton `tl.*` 数学、libdevice 约 120 个函数（四种舍入模式的 add/sub/mul/div/fma/rcp/sqrt、float↔int 转换、位运算、fmod/remainder/nextafter/ldexp 等精确函数、近似函数）、Philox 随机数、`inline_asm_elementwise`（舍入模式 PTX、x2/x4 打包、双输出）、`map_elementwise`；TileLang `T.ieee_*`（四种舍入）、取整、位运算、`T.reinterpret`、x2 打包运算。共 Triton 215 / TileLang 99 个函数条目，只列前端接受的 dtype | 每个元素用有理数精确求值，再按所请求的模式舍入一次；与 numpy 逐位一致，Philox 与 Triton 输出逐位一致；近似函数允许若干 ulp |
| `layout` | 从 load、arange、`tl.dot` 累加器、cumsum、广播归约产生 tile，再接至多三步 reshape/permute/trans/expand/join/split/interleave/cat/flip/sort/gather/keep-dims 归约/ravel（秩 1–4）；TileLang 在共享内存与寄存器片段之间做 reshape、view、transpose、区域拷贝、下标重排、swizzle、归约回广播 | torch 中执行同样的形状操作 |
| `loop` | Triton `range`、`tl.range`（num_stages、unroll、flatten、disallow_acc_multi_buffer、warp_specialize）、`static_range`、while；正向/反向/步长 2；下标、指针递增、块指针推进寻址；内层循环、条件、扫描、dot 累加。TileLang `T.Pipelined`（order/stage/sync/group）、serial、unroll、while、`T.loop_break` | torch 中逐次迭代 |
| `memory` | 窗口起点、行填充、转置视图、越界/不规则掩码与填充值；Triton 指针（真实的 `multiple_of`/`max_contiguous`/`assume` 提示、cache modifier、eviction、volatile）、块指针（边界检查、zero/nan 填充）、tensor descriptor；TileLang `T.copy`（coalesced_width、disable_tma、eviction）、区域拷贝、带掩码的循环加载 | torch 中同样的下标与掩码 |

同时修复了 82c18be2 的已知 harness 问题：cast 切片在 `operand=input` 时把 fp8 操作数直接转成整数（`legalize_steps` 现在检查操作数转换是否合法）。

本地冒烟测试（新切片各约 100–150 个程序）已经发现以下新机制，均已写成独立最小脚本：

| # | DSL | 现象 | 复现 |
|---|---|---|---|
| 13 | Triton | 被向量化的 int8/int16/uint8 带掩码 `tl.load`，若 `other` 为负（或 uint8 ≥ 128），被掩码的元素除每个 32 位字的第一个外都读出 -1/255。PTX 中是 `mov.u32 %r, 0xfffffffffffffffe`（符号扩展后的标量）而不是按字节重复的 0xfefefefe；ptxas 12.4 结果相同，所以是 Triton 的 PTX 错误。int32、浮点和正的填充值正确 | `masked_load_negative_other.py` |
| 14 | TileLang | 用 `T.alloc_var` 计数的 while 循环里先拷贝到共享内存再 `T.gemm`：生成的 CUDA 只在拷贝与 gemm 之间有 `__syncthreads()`，循环体末尾没有，下一轮的拷贝与上一轮 gemm 的 ldmatrix 读冲突。20 次运行 15 次错误、10 种不同输出；`T.serial` 正确 | `tl_while_gemm_race.py` |
| 15 | TileLang | int8/uint8/int16/uint16 的加、减、乘、移位后再 `T.cast` 到更宽类型，生成 `((int64_t)((int)(a + b)))`：C 整数提升保留了未回绕的值，而 TIR 中该运算的类型是窄类型（直接存成窄类型时正确） | `tl_narrow_int_widen.py` |
| 16 | Triton | `tl.fdiv` 用于 fp16/bf16 时在 ConvertTritonGPUToLLVM 中 `LLVM ERROR: Unsupported bitwidth`；`ieee_rounding=True` 用于 fp16/bf16/fp64 时 LLVM 翻译失败（intrinsic 返回 half/double，而 `llvm.nvvm.div.rn.f` 返回 float） | `tr_fdiv_half.py` |
| 17 | TileLang | 被向量化的 `T.bitwise_not` 对 CUDA 向量类型（int4、ulonglong2 等）生成 `~`，nvcc 报 `no operator "~" matches these operands` | `tl_intrinsic_lowering.py` |
| 18 | TileLang | float16/bfloat16 上的 `T.nearbyint`/`T.fmod` 生成未定义的 `hnearbyint`/`hfmod` | 同上 |
| 19 | TileLang | 缺少 CUDA 降级规则的内建函数到达代码生成：有符号 int32/int64 的 `T.popcount`，float16/bfloat16 的 `T.copysign/nextafter/ldexp/hypot`，报 `Unresolved call ir.Op(tirx.*)` | `tl_intrinsic_lowering.py`（popcount）；half 类型的函数见 `claude_work/explore/expand/probe_tl_features.py` |
| 20 | TileLang | 共享内存取模下标重排丢掉列偏移：`b1[i,j] = b0[(i+3)%16,(j+2)%16]` 的列下标被生成为 `(threadIdx.x & 7) * 2`（即原样的 j），`+2` 和 `%16` 都消失，结果静默等于“只换行不换列”的重排；32×32 生成 `((threadIdx.x & 15) * 2 + 2) & 31`，正确。确定性复现，加 `T.sync_threads()` 无影响，不是竞争。10-11 06:20 补充：与向量化无关，(32,64) f16 共享内存→寄存器片段、threads=32 时以标量读出现同样的丢失，(i+2, j+1) 错而 (i+0, j+1) 对，threads=64/128 也对，所以触发条件是行、列偏移同时存在；10-11 09:19 又见 f32 reshape 成 (4,256) 后的共享内存→共享内存重排，在 threads=32/64/128 下都丢列偏移，触发与否取决于形状和线程数怎样拆分下标 | `tl_remap_column_shift_dropped.py`、`tl_remap_column_shift_dropped_scalar.py` |
| 21 | Triton | 对秩 3 的 tile 沿 axis 0 做 `tl.gather`，当下标 tile 比被 gather 的那一维更长时，编译器在 TritonGPUOptimizeThreadLocality 中断言失败（`OptimizeThreadLocality.cpp:220`，`GatherLoweringHelper(op).isWarpLocal()`）。断言直接杀掉进程，前端不报错。两台服务器各自独立发现（源 (2,16,4)、下标 (4,16,4)、num_warps=8），本地也复现 | `tr_gather_warp_local_assert.py` |

| 22 | TileLang | 嵌套整数 floordiv 的不可靠化简：`T.floordiv(T.floordiv(x, 3) * -1, 4)` 被编译成 `floordiv(-x, 12)`（`rmod = x % 12; rdiv = x / -12; rmod <= 0 ? rdiv : rdiv - 1`），x = 1、2 时得 −1 而应为 0；`* -2` 被合并成 `floordiv(-x, 6)`，−8..8 中 6/17 个输入错；正乘数（对照）正确。`floordiv(floordiv(y,3),4) == floordiv(y,12)` 本身成立，所以错的一步是把取负移进内层 floordiv（−⌊x/3⌋ 是 ⌈−x/3⌉ 而不是 ⌊−x/3⌋）。数值很小，与溢出无关 | `tl_nested_floordiv_wrong_value.py` |
| 23 | TileLang | 从共享内存带列偏移的区域 `T.copy` 到寄存器片段，在被降级为 ldmatrix 时丢掉偏移：`T.copy(b1[0:64, 8:16], b2)`（2 字节类型、threads=256）生成 `ptx_ldmatrix_x1(&b1[(tid >> 5) * 128 + (tid & 7) * 16], ...)`，没有 `+ 8`，片段静默拿到第 0–7 列。threads=128（普通 uint2 加载，带偏移）、4 字节类型、偏移为 0 时都正确；不需要 `T.reshape`。效果上与 #20 同为“丢列偏移”，但降级路径不同（这里是 copy 选用 ldmatrix，#20 是 T.Parallel 的向量化下标化简） | `tl_ldmatrix_region_offset_dropped.py` |
| 24 | Triton（根因在 libdevice） | 有符号 `libdevice.mul24` 的结果只被加宽使用时（`r.to(tl.int64)`），返回完整的 48 位乘积，而不是文档规定的“乘积的低 32 位”。Triton 附带的 `libdevice.10.bc` 中 `__nv_mul24` 是 `shl/ashr 8` 取低 24 位后 `mul nsw i32`；两个 24 位数的乘积可达 2⁴⁶，`nsw`（不会有符号溢出）的声明不成立，NVPTX 据此把 `sext(mul nsw)` 合并成 `mul.wide.s32`。128/128 错；无符号 `umul24` 和 Triton 自己的 `x * y`（普通 `mul`）都正确 | `tr_libdevice_mul24_nsw.py` |

#20、#21 是 2026-10-10 15:22 定期检查在实验中发现的（前者由 layout 切片、后者由 layout 切片的 gather 步骤触发），#22 是 18:19 检查由 cast 切片发现的，#23 是 10-11 00:19 检查由 layout 切片（reshape 后区域拷贝）发现的，#24 是 03:19 检查由 math 切片发现的，都不在本地冒烟测试之列。#24 与 #1、#12 一样，根因在 NVIDIA 的工具（这里是 libdevice 的 bitcode），经由 Triton 触发。

#15 的补充（18:19 检查）：int32 也有同类问题但路径不同——`T.cast(x * 3, "int64")` 先把两个操作数加宽到 int64（`make_longlong2`）再相乘，乘积不按 32 位回绕（`tl_int32_widen_in_wide_type.py`）。int32 情形属于有符号溢出，TVM 可能视其为未定义行为，上游可能据此认为程序非法；#15 的结论主要由回绕语义明确的 uint8/uint16 情形支撑。

另有待确认的候选：bfloat16 上的 `T.pow` 触发 `power only applies to float` 检查失败；`T.Pipelined` 加 sync 注解时把从全局拷贝到寄存器片段的缓冲多缓冲化后报 `Fragment buffer access with non-zero index`；流水线中寄存器片段归约触发 `vars.size() >= InputDim()`；`T.Pipelined` 的 order/stage 配合寄存器片段归约体在 `Layout::Forward` / `ForwardThread` 失败（`vars.size() >= InputDim()`，10-10 两台实验中各出现数次）；寄存器片段的子区域拷贝出现多种布局推断崩溃（`Could not normalize iterators` 等，可能被上游视为不支持，切片已改为只从共享内存做子区域拷贝）。

新发现的 harness 问题（实验运行期间只记录，不改 src；全部已在 d31c08cb 修复，见第三轮）：
- math 切片的 `double2uint_*`/`float2uint_*`：Triton 的 libdevice 表把它们的返回类型声明为 int32，`widen=1` 时按有符号扩宽，于是高位为 1 的 lane 变成负数（3464260925 与 −830706371 位相同）。应把这些条目的 out 声明为 i32，或先按位转换再扩宽。
- memory 切片仍会选出 `T.copy` 无法证明的 `coalesced_width`（b9e08b4c 的 origin/row_pad/access 条件不够）。
- reduce 切片的 64 位中性值在传给 kernel 时溢出 TVM FFI 的 C long（旧问题，这次才暴露）。
- 18:19 检查又确认 5 个，都只记录、实验结束后再修（合计约占已标注失败的 3.7%；21:19 升至 7.6%，10-11 00:19 升至 12.5%，主要来自 Triton math 切片的三项：uint 返回 int32、fma 特殊值、近似函数参考不饱和）：
  - math 切片近似函数的参考是未经输出格式饱和的 float64 值：bf16 最大值 / 0.75 = 4.52e38 超出 bf16 范围，设备正确地给出 inf，却被判错。
  - fma 遇到非有限操作数时，参考退回未融合的 `x*y+z`，乘积先溢出：x = f64 最大值、y = −1478974、z = +∞ 时参考给 NaN，而精确的 fma（和设备）是 +∞。
  - Triton 默认以 flush-to-zero 链接 libdevice（`enable_reflect_ftz=True`），f32 的 `nextafter(0, +)` 返回最小正规数 2⁻¹²⁶ 而不是 2⁻¹⁴⁹；这是 FTZ 下的预期行为，参考没有建模。
  - `tl.maximum(u16, Python 整数)` 被提升到 32 位，math 切片按 16 位按位转换存储时失败（reduce 切片在 739fbf58 已建模这种提升，math 切片没有）。
  - TileLang 把 f16/bf16 与 Python 浮点字面量的运算提升到 f32（`X / 0.75` 生成 `((float)X) / 0.75f`），显式写 `T.cast(0.75, "float16")` 时才是 f16 运算。math 切片把常量写成裸 Python 浮点数却按 f16 建模，widen=1 时暴露出 f32 的商（21:19 检查）。这是前端的类型提升规则，不是错误编译。
  - Triton 的 `tl.cdiv` 定义为 `(x + (div - 1)) // div`，在操作数类型里计算：int8 下 x = 93、div = 43 时 135 回绕成 −121，结果 −2；设备结果与该公式完全一致。参考按精确的向上取整计算，而目录允许 x + div − 1 超出 int8（10-11 00:19 检查）。
  - loop 切片的合法化按输入取值 [−2, 2] 估计累加范围，但 TileLang `inputs=2` 时每步加 x + y ∈ [−4, 4]：coef 2、6 轮时范围是 ±252，i8 累加器回绕而 float64 参考不回绕。

已排除的探测结果：`inline_asm_elementwise` 在 pack>1 时跨元素的汇编（如 prmt 字节反转）结果随布局变化，文档写明“哪些输入组成一组未指定”，不是 bug，目录里只保留逐元素的打包汇编；`tl.sort` 只支持最后一维（前端 static_assert）。

部署：两台服务器 12:31 停止 82c18be2，在 `20261010_slices_b9e08b4c` 管理目录下冷启动，seed 与参数同上一轮（4090=43、vGPU=42，`--slice-prob 0.85`）。上一轮的最终 summary 在 `claude_work/campaign_20261009/baseline_82c18be2/`，最终快照为 `snapshots/1010-1232`。

## 第三轮：修复 harness 问题，加入编译配置与降级覆盖反馈（d31c08cb + 637deff6，2026-10-11）

b9e08b4c 跑了约 22 h：最后一个新机制（#24）出现在 13.7 h，之后约 8 h 没有新机制，harness 误报在已标注失败中的占比从 3.7% 升到 20.9%。按 Böhme 等人（FSE'20）的“发现的指数代价”，继续跑同一个输入空间收益很低，所以这一轮先修掉测试代码的问题，再扩大输入空间。

### 修复的 11 个 harness 问题

| 问题 | 修复 |
|---|---|
| `double2uint_*`/`float2uint_*` 在 libdevice 表里返回 int32 | 目录把结果声明为 i32，参考先按 u32 计算再按位回绕到 i32（`Fn.wraps`） |
| 近似函数参考未饱和到输出格式 | `_slice_mcheck`：参考值在容差内超出格式最大值时，接受同号的 ±inf |
| fma 非有限操作数时参考用未融合的 `x*y+z` | 按 IEEE 精确 fma 处理特殊值：有限乘积是精确的，不会先溢出 |
| Triton 以 FTZ 链接 libdevice | f32 的 libdevice 条目在操作数或结果落在次正规范围时不检查（`_slice_ftz`）；`cfg_ftz=0` 时关闭 FTZ 并精确检查 |
| `tl.maximum(u16, 3)` 被提升到 32 位 | Triton math kernel 先 `r.to(<目录声明的类型>)` 再存储 |
| loop 切片 `inputs=2` 的累加范围 | 合法化按每步 (1 + inner) × inputs 个输入估计范围 |
| memory 切片的 `coalesced_width` | 先合法化线程数；宽度须整除 tile/threads（`ComputePlanCandidate` 会把向量减半到每线程元素数），不满足时减半 |
| reduce 切片 u64 中性值溢出 C long | TileLang 用 `T.max_value("uint64")`（`T.cast(-1, "uint64")` 会被拒绝） |
| `tl.cdiv` 的定义 | 参考按 `(x + (y - 1)) // y` 在操作数类型里回绕、截断除法计算 |
| TileLang f16/bf16 与 Python 浮点常量 | `yform=const` 的浮点常量写成 `T.cast(c, "<类型>")` |
| libdevice 整数 `abs` 在 INT_MIN 处（637deff6） | `__nv_abs`/`__nv_llabs` 调用 `llvm.nvvm.abs.i/ll`（llvm-bcanalyzer 可见声明），Triton 3.8 内置的新版 LLVM 把它自动升级为 `llvm.abs(x, true)`，INT_MIN 为 poison（上游 llvm-project #183851/#198646 有意如此：PTX `abs.s` 对 INT_MIN 未定义；C 的 `abs(INT_MIN)` 也未定义），加宽后得到 +2³¹；LLVM 18 仍升级为回绕的 select。参考改为 C 语义 `cabs`，INT_MIN 不检查；`tl.abs` 仍按回绕检查。b9e08b4c 最后 2 h 在 vGPU 上出现（21.2 h），停机后的最终检查发现 |

验证：用新代码在本地 RTX 4060（Triton 3.8 / TileLang 0.1.14）重放 1011-0919 快照里每个带标签切片桶的首个程序。harness 桶 39/39 通过（Triton 30、TileLang 9）；真实机制 Triton 10 种 20/20、TileLang 15 种 27/28 仍失败，没失败的那个是 #14 `tl_while_loop_missing_barrier`（数据竞争，本来就时有时无）。单测 505 个通过（637deff6 加一个，506 个通过；abs 用例重放后通过）。冒烟：随机配置 Triton 80 个、TileLang 60 个，强制非默认配置 Triton 42 个、TileLang 每种 pass 配置 3 个（共 48 个）；全部失败（12 个）都是已知机制，且去掉配置后以同样方式失败，没有配置引起的误报。本地端到端 25 个测试得到 212 个降级特征。

顺带核对了 #24：libdevice 位码里没有 `llvm.nvvm.mul24` 内建函数，`__nv_mul24` 的 `mul nsw` 确实在位码中，#24 的结论不变。

部署：11:15 停止 b9e08b4c（22.7 h），11:18/11:19 两台以 d31c08cb 冷启动；11:20 对旧实验最终快照的检查发现上面的 abs 问题，修复为 637deff6 后约 12 分钟内停掉 d31c08cb 重新冷启动（管理目录 `20261011_slices_637deff6`），本轮从这里计时。b9e08b4c 的最终 summary 与快照在 `claude_work/campaign_20261009/baseline_b9e08b4c/`。

### 新方法 B：编译配置作为切片旋钮

同一程序在不同编译配置下应得到相同结果（配置只改变降级决策，不改变语义），所以每个配置都是一个差分 oracle，能到达默认配置永远走不到的编译路径（cf. 编译器选项空间测试）。

- Triton：`maxnreg`（0/32/64/128，寄存器上限触发 ptxas 溢出）、`enable_fp_fusion`（FMA 收缩）；loop 切片加 `num_stages`（默认软件流水深度 1/2/4）；math 切片加 `enable_reflect_ftz`（libdevice FTZ）。
- TileLang：`pass_configs` 一个类别旋钮，16 种非默认值：64 位索引、关闭向量化、关闭 CSE、关闭 cp.async、LDG/STG 降级（含谓词形式）、激进共享内存合并、关闭共享内存复用、强制 let 内联、StorageRewrite 原地复用、关闭/放宽 loop unswitching、reducer 强制基线方案（文档明确说用于差分测试）、`io-aware` 布局代价模型、关闭 noalias、ptxas 寄存器使用等级 0/10。排除了会改变语义或关闭安全检查的配置（fast math、关闭安全访存、关闭线程同步、关闭各类检查）。
- 采样时一半程序保持默认配置；配置不进入失败标签（同一机制在不同配置下仍是一个桶）；约简最先尝试把配置恢复默认，只有必要的配置才留在核心里。

### 新方法 A：降级特征覆盖反馈（Zest / FairFuzz 式）

每个 harness 结束时打印其已编译 kernel 的降级特征：Triton 为 PTX 操作码（含类型和修饰符）、TTGIR 操作、布局编码种类和 `convert_layout` 的源/目标布局对；TileLang 为生成 CUDA 中调用的内建函数（模板参数里的线程数归一化）、向量指针转换、内联 PTX 助记符和 pragma。oracle 读取并从输出中去掉这一行。

- 通过的程序首次到达某个降级特征时成为种子（Zest 的有效性反馈：失败程序不做种子）。
- 一半的变异从种子出发，按 1 / (显示其最稀有新特征的测试数) 加权抽取（FairFuzz 的稀有目标）。
- 每个切片的发现率加入降级特征的 Good-Turing 估计（只出现在一次测试中的特征数 / 测试数）。
- `summary.json` 的 `slice_lowering` 和 `coverage_progress.json` 的 `slice_lowering_features` 记录覆盖的特征数；检查脚本报告其增长。

### 第三轮实验中的新机制

| # | DSL | 现象 | 复现 |
|---|---|---|---|
| 25 | Triton | `tl.range(..., warp_specialize=True)` 的循环体里有扫描（`tl.cumsum`、`tl.associative_scan`）时，NVGPUWarpSpecialization 触发 `llvm_unreachable`：`Other parent ops are not supported.` / `UNREACHABLE executed at .../WarpSpecialization/TaskIdPropagation.cpp:107!`。`TaskIdBackwardPropagation::propagateToParent` 从每个操作向外找所在循环，只接受 scf.for、scf.if、tt.func 和 tt.reduce 作为父操作，`tt.scan` 的 combine 区域没有处理（上游 main 至今如此）。同一循环里的 `tl.sum`、自定义 `tl.reduce` 能编译；为 sm_80/89/90 编译失败，sm_100（Blackwell 流水线）能编译 | `tr_warp_specialize_scan_unreachable.py`（只编译，不需要 GPU） |
| 26 | TileLang | 打开 pass 配置 `tl.enable_lower_ldgstg`（文档：把 Ramp 形式的全局 load/store 降级为 ldg/stg 内建函数）后，元素类型是向量的全局访问编译失败：`Check failed: tgt_dtype.lanes() * tgt_dtype.bits() == src_dtype.lanes() * src_dtype.bits() (32 vs. 64) : reinterpret expects source and target to have the same number of bits`。`src/cuda/transform/lower_ldg_stg.cc` 用 `buffer->dtype.bits()`（单个 lane 的位宽）决定访问宽度，忽略了向量元素类型的 lanes：float32x2 元素（64 位）被生成 `ldg32`/`stg32`，再把 32 位结果 reinterpret 成 float32x2。int32x2 同样失败，float32x4 报 32 vs. 128；float16x2 因为 16 位 lane 根本不被降级而正常。默认配置和只开谓词版本（`tl.enable_lower_ldgstg_predicated`）下都能编译且结果正确；上游 main 的宽度计算相同。触发途径：`T.view(A, ..., "float32x2")`，math 切片的 `T.add2/sub2/mul2/max2/fma2/abs2`（f32）都会这样访问 | `tl_ldgstg_vector_lanes.py` |
| 27 | TileLang | pass 配置 `tl.config_index_bitwidth=64`（文档：“配置索引的位宽”）不只改索引，也改存储的值：`src/transform/config_index_bitwidth.cc` 的 `ConfigIndexBitwidthRewriter` 为了把索引改成 int64，在访问整个 BufferStore/BufferLoad 时打开改写开关，被存储的值也在其中，于是值里所有窄于 64 位的有符号整数常量和 Cast 都被改成 int64。后果有两种：(1) **错误结果**：值里显式的收窄转换被丢掉，`O[i] = T.cast(T.cast(X[i], "int8"), "int32")` 原样返回 X[i]（x = −640 得 −640，应为 −128），int16 同样；uint8 不受影响（改写只匹配有符号类型）。(2) **编译失败**：int8/int16 元素运算（`-b`、`b * T.cast(3, "int8")`）在向量化循环里变成 int64x16 / int64x8 向量运算，`CodeGenTileLangCUDA::PrintType` 无法打印（`Cannot convert type int64x16 to CUDA type`）。默认配置下全部正确；上游 main 的改写器相同 | `tl_index64_rewrites_stored_values.py` |

#25 是 10-11 14:09 检查发现的（4090，0.43 h，loop 切片 `body=cumsum, loop=tl_range, ws=1`）。`ws` 旋钮在 b9e08b4c 里就有，但 22.7 h 两台都没碰到这个组合。

14:09 检查的另一个候选当时判为不是新机制：vGPU 上 cast 切片在 `cfg_pass=index64` 下报 `Cannot convert type int64x8 to CUDA type`。去掉配置后同一程序报 `boolx8`（即 TL-01），两者都经过 BufferStore→Let→Mod→PrintVecBinaryOp→PrintType 这条路径：被向量化的 `floormod(floordiv(...))` 产生了 CUDA 打印器没有的向量类型，64 位索引只改变了它的标量类型。归入 TL-01。

20:24 更正：int64x8 这个类型本身来自 #27（index64 把存储值里的窄整数运算改成了 int64）。该 cast 切片程序去掉配置后仍因 TL-01 失败，两种机制同时存在；同一个桶在 4090 的 memory 切片上再次出现时，去掉配置就能通过，是单纯的 #27。规则已改为把 index64 下的 `int64x` 打印失败归入 #27。

#26 是 10-11 17:47 检查发现的（vGPU，4.76 h，math 切片 `fn=sub2, in_dt=f32, cfg_pass=ldgstg`），是第一个依赖新方法 B（编译配置旋钮）才能到达的机制：去掉配置就不触发。opt-in 的配置默认关闭，上游可能把优先级排得较低，但这是 pass 自身的宽度计算错误，被它自己生成的 reinterpret 检查拦下，不是清晰的“不支持”报错。

17:47 检查的另一个候选归入 #1：scan 切片 `f32 → abs → f16 → u32`（abs 在 f32 中计算再收窄）。LLVM 把 fabs 移到 fptrunc 之后，生成 `cvt.rn.f16.f32` 加 `and.b16 …, 32767`，正是 #1 的模式；num_warps=4 的变体走这条路径而出错，num_warps=1 的变体用 `abs.f16x2` 而正确，换 ptxas 12.4 后全部正确。

#27 是 10-11 20:24 检查发现的：4090 上 memory 切片（int8、`access=region`、`transform=neg`）在 `cfg_pass=index64` 下报 `int64x16` 打印失败，去掉配置能通过；顺着改写器的源码推断并验证了更严重的错误结果形式。和 #26 一样只能通过编译配置旋钮到达。

## 后续

- 继续运行，关注 `summary.json` 的 `slices`、`route_stats` 和桶的 `route`。
- 对 1–10 查上游 issue、写最小报告。
- 下一步候选：LLM 补充切片（Gentoo/CodaMOSA 式，需模型 API）；严格 A/B（`--slice-prob 0` 对照）。
- 637deff6 实验：按机制（不按桶）与 b9e08b4c 在相同时长对比（`compare_mechanisms.py`）；配置相关的新桶要用去掉配置的 `.min.py` 对照，确认配置是否必要。

## 第四轮：跨架构编译（b1eded9f，2026-10-12）

### 为什么要换方向

637deff6 两台各跑约 13.1 h：最后一个新机制 #27 出现在约 7 h，此后约 6 h 的新桶全部是已知机制；Triton 的降级特征稳定在约 1000 个（每 3 h 只增 3–4%）。前三轮的所有手段（切片、编译配置、降级覆盖反馈）都只在一个目标上编译：sm_89。编译器里只有其他架构才走的代码——Hopper 的 wgmma/TMA/warp specialization、Blackwell、Turing、两个 DSL 的整个 AMD 后端——从没被测过。按 Böhme 等人（FSE'20）的“发现的指数代价”，在同一输入空间里继续跑收益很低；要扩大输入空间，而这个方向不需要新硬件：两个编译器都能在没有对应 GPU 的情况下为任意目标编译。

### 方法

- `cfg_target` 旋钮（35% 的切片程序）：Triton 为 native、sm80、sm90、sm100、sm120、gfx90a、gfx942、gfx950、gfx1100（Triton 3.x 只支持计算能力 8.0 以上，不含 sm75）；TileLang 为 native、sm80、sm75、sm90、sm100、sm120、gfx942、gfx950。
- `_slice_target`（runtime.py）在第一次编译前改写 Triton 的 driver 目标与 JIT、TileLang 的 `tilelang.jit`。sm80 的二进制能在 sm_89 上运行，按完整 oracle 检查结果；其余目标只编译，走完所有不需要设备的阶段（Triton 到 cubin/hsaco；TileLang 到 CUDA 源码再由 nvcc 生成 cubin，或到 HIP 源码），只有崩溃/构建失败算失败。服务器的 nvcc 是 12.4，TileLang 的 sm100/sm120 只做降级、不跑 nvcc（本地 12.9 两步都做）。
- `cfg_amd` 旋钮（只在 AMD 目标下生效）：buffer ops/atomics 及其范围分析、block ping-pong、in-thread transpose、async copy、两种备选调度器、packed 浮点标量化、`waves_per_eu`、`matrix_instr_nonkdim`、`kpack`。
- math 切片：没有 AMD 降级的条目（PTX 内联汇编、只有 CUDA libdevice 的函数）改到 sm90；AMD 目标导入 HIP libdevice。
- AMDGCN 操作码进入降级特征；非原生目标进入路径特征（quarantine 可学目标相关规则）；约简最先把目标恢复为 native（预算 24 → 28），所以核心里带 `cfg_target` 的桶才是目标相关的，去掉目标仍失败的归入已有机制。
- 目标缺少的特性记为噪声：Triton 的架构断言、AMD wave64 下不足一个 wavefront 的 block、AMD 缺少的 fp8 转换、Blackwell 上低于指令需求的寄存器上限（新噪声类 `register_limit`）。

### 本地冒烟（RTX 4060，约 1000 个程序）发现的目标相关候选

均已确认原生目标下通过；带脚本的已写成独立最小复现。

| # | DSL | 目标 | 现象 | 复现 |
|---|---|---|---|---|
| X1 | Triton | gfx942、gfx950 | **任何** `tl.atomic_xchg`（i32/i64/f32/f64）都无法编译：buffer atomic 降级生成 LLVM 内建函数 `llvm.amdgcn.raw.ptr.buffer.atomic.exch`，而 Triton 自带的 LLVM 里只有 `...buffer.atomic.swap`，于是 `LLVM Translation failed`。gfx90a/gfx1100 不走 buffer atomics，能编译；`AMDGCN_USE_BUFFER_ATOMICS=0` 也能编译 | `tr_amd_atomic_xchg.py` |
| X2 | TileLang | sm100、sm120 | 最简单的 int16 逐元素运算（`O[i] = X[i] + X[i]`，512 元素、threads=32）报 `Cannot convert type int16x16 to CUDA type`：Blackwell 上向量化宽度变为 256 位，CUDA 打印器没有 int16x16；int8/f16/i32/f32 正常，sm_89/sm_90a 用 128 位 int16x8 正常 | `tl_blackwell_256bit_vector_type.py` |
| X3 | TileLang | sm120 | 标准的流水线 GEMM（`T.Pipelined` + `T.gemm`）被 warp specialization 加上 `tl::warpgroup_reg_alloc/dealloc`（PTX setmaxnreg），sm_120（RTX 50 系列）没有这条指令，nvcc 停在模板的 `static_assert`；sm_90a、sm_100a 能构建 | `tl_sm120_setmaxnreg.py`（需 nvcc ≥ 12.8） |
| X4 | TileLang | gfx942、gfx950 | IEEE 舍入内建函数（`T.ieee_add/sub/mul/fmaf/fsqrt/frcp/fdiv`）在 HIP 代码生成中没有降级规则：`Unresolved call tl.ieee_*`（CUDA 正常） | `tl_hip_ieee_unresolved.py` |
| X5 | TileLang | sm80（真实执行） | fp8 `T.gemm` 以 sm_80 为目标编译成功，却选用 sm_89 才有的 `cute::SM89_16x8x32` MMA，运行时触发 device-side assert | campaign 用例待约简 |
| X6 | TileLang | gfx942、gfx950 | OCP fp8 在 HIP 代码生成中 `Bad bit-width for float: float8_e5m2`，或 Python 层 `KeyError: dtype('float8_e4m3')`；gfx950 硬件支持 OCP fp8 | 待约简 |
| X7 | Triton | gfx1100 | i16 `cmax` 扫描、i8 `flip`：生成的 AMDGCN 被汇编器拒绝（`src1 immediate operand invalid for instruction`），随后 LLD 报错 | 待约简 |
| X8 | TileLang | gfx950 | i16/u8 的 unroll 循环：`Check failed: (p) is false:`（空消息） | 待约简 |

另：Triton 为 sm_75 编译 int8 `tl.dot` 时 TritonGPUAccelerateMatmul 生成非法的 `arith.extf`（`tr_sm75_int8_dot_unsupported_arch.py`），但 Triton 3.x 官方只支持 8.0 以上，上游可能不受理，因此没有把 sm75 列为 Triton 目标。

本地冒烟的非原生程序约 8%（Triton）/ 10%（TileLang）失败，其余失败都是已知机制（#3/#4/#5 fp8/f64/rtz、#6、#9、#10、#17、TL 候选 pow/布局）在其他架构上重现，约简恢复 native 后会归入原机制。

### 部署

两台服务器 10-12 00:44 停止 637deff6（约 13.1 h），00:45/00:47 以 b1eded9f 冷启动（服务器单测 512 个通过），seed 与参数同上一轮（4090=43、vGPU=42，`--slice-prob 0.85`）。管理目录 `20261012_xtarget_b1eded9f`，结果 `results/2026.10.12-00.45_*`（4090）、`results/2026.10.12-00.47_*`（vGPU）。637deff6 的最终 summary 与快照在 `claude_work/campaign_20261009/baseline_637deff6/`，check.sh 现在先与它按机制同时长对比。

### 第四轮实验中的新机制

| # | DSL | 现象 | 复现 |
|---|---|---|---|
| 28 | Triton 经 ptxas（12.4、12.9 都有） | f16 `tl.dot`（32×64 @ 64×128，f32 累加）用 `num_warps=1` 加文档化的启动选项 `maxnreg=128` 时，运行报 `illegal memory access`；不加 maxnreg（214 寄存器）、maxnreg=255 或 num_warps=2 都正确。maxnreg 只限制寄存器、强制溢出（138 次溢出，552 字节栈帧），不应改变语义；把存储改成 `acc * 1.0 + 0.0` 又能通过，说明依赖具体的溢出代码。原生目标，确定性复现。10-14 定位为 ptxas 错误：PTX 只差 `.maxnreg 128` 一行，同一份 PTX 用 ptxas 12.4、12.9 汇编都越界，`-O0`/`-O1` 正确、`-O2`/`-O3` 出错；compute-sanitizer 报把 B 块写入共享内存的 `STS.128` 越界（地址 0x5b00–0x7f00，共享内存 0x5000），SASS 中保存 `tid & 15` 的 R115 在被 `SHF.L.U32 R114, R115, 0x4` 读取前被另一条 `IMAD` 改写，而 PTX 中该地址不超过 0x1f0 | `tr_maxnreg_spill_illegal_access.py` |
| 29 | Triton 经 ptxas 12.9 | bf16 `tl.atomic_add`（`sem='acquire'`/`'acq_rel'`，sm_89 上降级为 32 位 CAS 循环加 fence）用 `num_warps=1, maxnreg=32` 时丢失更新：一个槽得到 10，应为 12，确定性复现。加与不加 maxnreg 的 PTX 只差 `.maxnreg 32` 一行，两者都没有溢出，所以错误出在 ptxas。不加 maxnreg、maxnreg≥40、num_warps≥2 或 `sem='relaxed'` 都正确 | `tr_maxnreg_atomic_cas_lost_update.py` |

#28 是 10-12 17:28 检查发现的（4090，约 16.3 h，gemm 切片 `cfg_maxnreg=128, warps=1`），在本地 RTX 4060 上复现。和 #26/#27 一样，是编译配置旋钮（方法 B）到达的机制。#29 是 10-14 02:33 检查发现的（4090，约 48.7 h，atomic 切片 `cfg_maxnreg=32, warps=1`），同样在本地 RTX 4060 上复现，也是 maxnreg 旋钮到达的。

### 下一步

- 关注核心里带 `cfg_target` 的新桶；X5–X8 从 campaign 的 `.min.py` 约简。
- 若跨架构也饱和：CodaMOSA（Lemieux 等，ICSE'23）式的 LLM 补充——在覆盖平台期由模型针对未覆盖的 API/降级特征写新的切片模板（需模型 API）；或对 TVM/TileLang C++ 与 Triton MLIR pass 做真正的代码覆盖插桩，再用 AFLGo（Böhme 等，CCS'17）式的定向调度把预算导向未覆盖的 pass。
