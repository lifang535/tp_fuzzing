# 第四轮实验报告：跨架构编译（b1eded9f → ea9ecc41，2026-10-12 至 10-14）

截至 2026-10-14 16:08（约 63 h）。监控记录见 `claude_work/campaign_20261009/CHECKS.md`，机制规则见 `known_mechanisms.json`，复现脚本在 `reports/2026.10.09/feature_slices/`（方法细节见该目录 `REPORT.md` 的“第四轮”一节）。

## 结论

- **真实 bug**：实验命中已知真实机制 **32 种**（28 种已确认 + 4 个老候选），与上一轮 637deff6 同时长持平；本轮新确认 **2 种**（#28、#29），并第一次在实验中命中 #8。累计已确认 **34 种**（#1–#29 + 5 个早期发现），都有独立复现脚本。
- **新真实候选**：约 **13 种**，几乎全部只在非本机目标（sm80/sm90/sm100/sm120/sm75/AMD）上触发，尚未约简确认。
- **fuzzer 缺陷**：失败桶中 **19 个**是 fuzzer 自身造成的（3 类），另有 **1 个导致进程崩溃的代码 bug**（10-14 13:18 vGPU Triton 退出），已在 ea9ecc41 修复并部署到两台服务器，实验从断点恢复。

## 实验设置

| 项 | 内容 |
|---|---|
| 代码 | b1eded9f（10-12 00:45 冷启动）；10-14 15:4x 起两台都换成 ea9ecc41 = b1eded9f + d6046da9 + ea9ecc41，从断点恢复 |
| 服务器 | 4090（端口 41790，seed 43，结果 `results/2026.10.12-00.45_*`）；vGPU-32GB（端口 32906，seed 42，结果 `results/2026.10.12-00.47_*`） |
| 方法 | 10 个特征切片 + 编译配置旋钮 + 降级覆盖反馈（前三轮），本轮新增 `cfg_target`：35% 的切片程序编译到非本机目标（Triton：sm80/90/100/120、gfx90a/942/950/1100；TileLang：sm75/80/90/100/120、gfx942/950）。sm80 在 sm_89 上真实执行并做完整检查，其余目标只编译 |
| 监控 | 本地 SSH watchdog（每 10 分钟）+ 每 3 小时定期检查，共约 25 次检查 |

### 已测数（10-14 16:07）

| 服务器 | TileLang | Triton |
|---|---|---|
| 4090 | 15983（通过 14773） | 35220（通过 33907） |
| vGPU | 25123（通过 23235） | 51974（通过 50273） |
| 合计 | 41106 | 87194 |

吞吐：4090 约 TileLang 240/h、Triton 550/h；vGPU 约 TileLang 420/h、Triton 900/h。磁盘每台每天增长约 0.2 GB（通过的程序只存 IR json，切片程序平均 0.6 KB），剩余 46–53 GiB。

## 一、真实 bug

### 本轮新确认的机制

| # | DSL | 现象 | 发现 | 复现 |
|---|---|---|---|---|
| 28 | Triton（或 ptxas 12.9） | f16 `tl.dot`（32×64 @ 64×128）用 `num_warps=1, maxnreg=128` 运行报 `illegal memory access`；不加 maxnreg、maxnreg=255 或 num_warps=2 都正确。maxnreg 只应强制溢出（138 次溢出），不应改变语义 | 10-12 17:28，4090，16.3 h，gemm 切片 | `tr_maxnreg_spill_illegal_access.py` |
| 29 | Triton 经 ptxas 12.9 | bf16 `tl.atomic_add`（`sem=acquire/acq_rel`，sm_89 上降级为 32 位 CAS 循环）用 `num_warps=1, maxnreg=32` 时丢失更新（得 10，应 12），确定性复现。加与不加 maxnreg 的 PTX 只差 `.maxnreg 32` 一行，都没有溢出，错误在 ptxas。maxnreg≥40、num_warps≥2、`sem=relaxed` 都正确 | 10-14 02:33，4090，48.7 h，atomic 切片 | `tr_maxnreg_atomic_cas_lost_update.py` |

两者都由编译配置旋钮 `cfg_maxnreg`（方法 B）到达，在原生目标上触发，本地 RTX 4060 确定性复现。另外，#8（TileLang 嵌套 floordiv 的 Let 冲突）以前只在本地探测中出现，本轮第一次在实验中被命中（vGPU，约 25.7 h）。

### 本轮命中的已知机制（32 种）

| 类别 | 机制 |
|---|---|
| Triton 已确认（12） | #1 ptxas f16 abs、#2 f32→fp8 舍入、#3 f64↔fp8 abort、#4 fp8 舍入断言、#5 rtz 降级、#6 bf16 自定义 combine、#13 负填充值掩码加载、#16 fdiv half、#21 gather 非 warp-local、#25 warp specialize + scan、#28、#29 |
| TileLang 已确认（16） | #7 f16 原子向量化、#8 嵌套 floordiv、#9 int64→half、#10 int64 原子、#11 int→bf16 截断、#14 while 循环缺同步、#15 窄整数加宽不回绕、#17 向量按位取反、#18 half 内建未定义、#19 未解析内建、#20 下标重排丢列偏移、#26 ldgstg 向量宽度、#27 index64 改写存储值、早期 TL-01 bool 类型、ReduceOp 布局、2 字节 cp.async |
| 老候选（4） | TileLang `T.pow` bf16 检查失败、Pipelined 片段多缓冲、Pipelined 片段布局前向、布局推断 `has_best` |

按服务器：4090 TileLang 18、Triton 12；vGPU TileLang 20、Triton 8。与 637deff6 同时长相比：本轮独有 #8、#28、#29；本轮未命中 6 种（#12 ptxas 常量 fp8、#22 floordiv 取负合并、#23 ldmatrix 丢偏移、#24 libdevice mul24、TR flip 默认轴、TR 布局转换支配）。对这 6 种做过可达性抽样：在当前代码下都能生成，只是 35% 预算分给非本机目标后频率下降，且它们本来就是单次命中的稀有桶。

### 新真实候选（约 13 种，未约简）

| DSL | 目标 | 现象 |
|---|---|---|
| Triton | sm80（真实执行） | f16→f8e5 rtne 平局舍入错（28672 vs 24576） |
| Triton | sm80（真实执行） | fp8e5 `padding=nan` 结果不是 nan（得 −0.0），两台服务器各自触发 |
| Triton | gfx1100 | X7：生成的 AMDGCN 被汇编器拒绝，随后 LLD 报错 |
| Triton | sm120 | AutomaticWarpSpecialization 断言 |
| Triton | gfx942 | `idx < size()` 断言（`kpack=2`） |
| TileLang | gfx950 | X6：OCP fp8 `PrintConst` 失败 / `_initialize_mfma_prefix` KeyError |
| TileLang | sm100 | X2：`Cannot convert type int16x16 to CUDA type`（256 位向量化） |
| TileLang | gfx942/950 | X8：`Check failed: (p)`（空消息） |
| TileLang | gfx942/950 | `PrintVecElemStore` 的 type_name 为空（核心约简为 memory i16 + `cfg_pass=index64`，可能属 #27 族） |
| TileLang | gfx942 | gemm 化简器除零 |
| TileLang | sm90 | `Not a canonical GMMA_MN layout` 断言 |
| TileLang | sm75 | 只在 sm75 + `tl.loop_unswitching_allow_non_trivial_else` 时报 `buffer accum has already been defined, the Stmt is not SSA`；native + 该配置、sm75 不加配置都通过 |
| TileLang | sm80（真实执行） | X5：fp8 gemm 选用 sm_89 才有的 MMA，运行时 device assert |

另有 nvcc 12.4 编译 sm_90a 时段错误（工具链问题，不计入）。已知族在 AMD 上的变体（f64↔fp8 Unsupported conversion 属 #3 族、rtz→fp8 `maybeOutVals` 断言属 #5 族）归入原机制。非切片路线（fresh/mutate/extended）的少量 wrong_result、precision、schedule/pass-config 不一致桶按协议只列出，没有调查。

## 二、fuzzer 缺陷

### 失败桶中的 fuzzer 缺陷（19 个桶，3 类）

| 类别 | 例子 | 状态 |
|---|---|---|
| 错误信息截断：error_message 截到 2000 字符，`TILESMITH_TARGET`/拒绝标记丢失，目标明确不支持的程序被当成失败 | Triton sm80 `fp8e4nv not supported in this architecture`（27ec397856、8d8362782c 等）；TileLang AMD wavefront 不足；Triton sm90/sm100 `maxnreg=32` 寄存器不足（ff1a781e90、39a1a80878） | d6046da9 已修，10-14 15:4x 部署；部署后 vGPU 第一次把这类程序正确归为 `unsupported_feature` 噪声 |
| 合法化缺口（已修）：生成了目标不支持的参数 | AMD 上 tf32/tf32x3 gemm、TileLang AMD gemm 的 block_K | d6046da9 已修并部署 |
| 合法化缺口（未修） | TileLang HIP 的 `nan_propagate`；TileLang MFMA 的 k_dim（9054268b38、959d3506d3）；Triton gfx90a 上 `cfg_amd=async` 强开 async copy（gemm 977ca8af90、loop 9c65d2c443） | 待修；部署后新增的 fuzzer 缺陷应只剩这一类 |

### 导致进程退出的代码 bug（已修）

- **现象**：10-14 13:18 vGPU Triton 进程退出（约 60.5 h，已测 51866），`emit → ValueError: slice program parameters are not legalized`，主循环未捕获。
- **根因**：layout 切片 `legalize_triton` 不是幂等的。源为 cumsum/bcast 时 `arg1` 决定源的轴，但 `op1=none` 时步骤循环把 `arg1` 置 0。第一次合法化按原轴算出的数值范围允许 `out_dt=i8`；emit 时再合法化，轴变了、范围变大，`out_dt` 被改成 same，emit 判定参数被改动而抛异常。第一次 emit 时 plan 里的轴与范围也不一致。
- **修复（ea9ecc41）**：cumsum/bcast 源保留 `arg1`；`make_program` 反复合法化直到参数不再变化；新测试 `test_legalization_is_idempotent`（含崩溃程序 + 每个切片和 backend 各 3000 个样本）。修复前该测试报出与服务器相同的错误；本地与两台服务器 514 个单测全部通过；5 万样本扫描所有切片没有其他不幂等的情况。
- **恢复**：两台都 pause → checkout ea9ecc41 → 服务器单测 → 更新 `commit.txt`（旧值存为 `commit.txt.b1eded9f`）→ `resume_campaign.sh` 从断点恢复（历史结果、随机状态、反馈、pending 程序都恢复，测试编号连续）。崩溃的程序 #51867 在新代码下通过。中断时间：vGPU Triton 约 2.5 h，4090 约 15 分钟。

## 三、与前几轮的比较与解读

- 按机制同时长对比，本轮与 637deff6 持平（32 vs 32），按桶数本轮明显更多（约 47 h 时：4090 TileLang 88 vs 48、Triton 50 vs 21；vGPU TileLang 98 vs 57、Triton 49 vs 22），多出的桶主要是非本机目标的候选和 fuzzer 缺陷。
- 本轮新确认的 #28、#29 都来自 maxnreg 旋钮，在原生目标上触发；跨架构本身带来的是约 13 个目标相关候选，需要约简后才能计入确认数。
- 第 3 天的新桶几乎都是已知机制（#14、#15、#20 等反复出现），出现饱和迹象。

## 四、下一步建议

1. 约简并确认候选，优先 sm80 上真实执行出错的三个（f16→f8e5 平局舍入、fp8e5 padding=nan、X5），以及 sm75 unswitch 的 not-SSA。
2. 修复未修的合法化缺口（HIP nan_propagate、MFMA k_dim、gfx90a async），减少 fuzzer 缺陷桶。
3. 若继续饱和：按 `REPORT.md` “下一步”，考虑 LLM 补充切片模板（CodaMOSA 式），或对编译器 pass 做代码覆盖插桩后定向调度（AFLGo 式）。
