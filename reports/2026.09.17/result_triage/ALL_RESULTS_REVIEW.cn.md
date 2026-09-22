# results 全目录复核

合并总表： [Markdown](ALL_RESULTS_TABLE.cn.md) · [蓝色表格网页版](ALL_RESULTS_TABLE.cn.html) · [CSV](ALL_RESULTS_TABLE.cn.csv)

快照：2026-09-17 21:16 +08:00。读取全部12个campaign下14,425份失败JSON及对应Python程序。不是14,425个独立bug。

本轮做了全目录静态审查，并复核仓库中已有的最小复现源码、实验日志和本机安装源码。既有审查覆盖的2,295份JSON全部SHA256一致。本轮GPU访问被操作系统阻止，因此以下“确认”引用既有复现证据，不声称本轮重新执行了GPU实验，也不声称最新上游版本仍然存在这些问题。既有实验环境为TileLang 0.1.11、Triton 3.0.0、sm_89。

**结论：有证据支持6类真实缺陷；大量其他记录来自资源配置、oracle语义和数值判断问题；剩余样本必须保留待确认状态。**

## 真实缺陷及代码中的名称

这里同时列出results中的分类名称、代表复现文件名、相关代码符号。分类名不是已定位的根因函数名；未定位具体pass的情况明确标注。

| 真实缺陷 | results分类名称 | 代表代码名称 / case ID | 相关代码符号及依据 |
|---|---|---|---|
| TileLang JIT缓存没有区分捕获的dtype | `dtype_mismatch` | [dtype32_same_cache.py](dtype32_same_cache.py)，函数`triage_cache_probe` | `tilelang/jit/__init__.py::JITImpl._frontend_cache_key_data`：缓存键含源码、签名、实参，但没有这里引用的外部dtype值。同缓存fp16→fp32失败，独立fp32缓存通过。当前2,442条同类记录；旧版新增1,953条均核对声明和host输入类型一致，不代表全部已逐条重放。 |
| TileLang普通fp16拷贝生成非法2字节cp.async | `ptx_async_boundary` | `failed_calls_main(f0+f0)__f0()_6ffb52d51060569a.py` | 高层`T.copy`；诊断点`GetTileLangCPAsyncTransferBytes`。既有冷缓存复现报传输宽度2，不属于合法PTX宽度集合。当前209条同症状；旧版160条全部fp16，154条N/K=1，另6条N=191、K为奇数。不是“所有block_K=1”，也不是用户主动发出非法PTX。 |
| TileLang fp16 K-tile=8 GEMM列重复 | `wrong_result` / `schedule_mismatch` | [block_k_8_exact.py](block_k_8_exact.py)，函数`control`；原始`failed_calls_main(f1+f1)__f0()__f1(f0)_b5c2b8bb73c6097d.py` | `T.gemm`、`mma_macro_generator.py`的K=8 MMA路径；具体错误映射尚未定位。整数输入32×32 GEMM在128线程下误差256且列重复，256线程同输入误差0。不能把K=8统一过滤为非法程序。旧版4,086条fp16/K8 wrong_result仅是配置候选，部分同时含oracle缺陷。 |
| TileLang共享内存依赖缺少同步 | `nondeterminism` | `failed_calls_main(f1)__f0()__f1(f0)_28f7a4c7a8eda0b4.py`；[28f7a4c7a8eda0b4_explicit_barrier.py](28f7a4c7a8eda0b4_explicit_barrier.py) | `T.copy(arg1, v1_shared)`附近共享内存转置。原核相同输入重复执行出现Inf；加`T.sync_threads()`后80次运行通过。尚不能命名具体漏插屏障的pass。 |
| Triton编译阶段SIGSEGV | `segfault` | `failed_calls_main(f0)__f0()_4abdeda508af655d.py`；[triton_compile.py](triton_compile.py) | `triton/backends/nvidia/compiler.py::CUDABackend.make_ttgir`中的`pm.run(mod)`。既有13条离线编译均SIGSEGV，不分配大型输入、不执行GPU kernel；具体C++ pass尚未定位，13条不能算13种根因。 |
| Triton fp16负常量dot操作数生成NaN | `wrong_result` | `failed_extended_mixed_25a869b78ca36590.py`；[triton_constant_dot.py](triton_constant_dot.py)，函数`dot_control` | `tl.full((16,16), -0.125, tl.float16)`与`tl.dot`组合。256个输出全NaN；正常量、从内存加载相同负数矩阵均通过。最小程序无复杂控制流和oracle；具体codegen pass尚未定位。 |

原始文件直接链接在[代表样本索引](CASES.md)，完整实验解释及日志链接见[已有复现审查](REPORT.md)。

## 应排除的结果和fuzzer问题

| 问题 | 数量 | 判断与代码证据 |
|---|---:|---|
| `shared_memory_overflow` | 562 | 保存日志的申请量超过设备上限；属于生成的调度资源不合理，不是正确性bug。检查`src/backends/tilelang/params.py::check_shared_memory`及Triton对应函数；不能只预算A/B staging而忽略临时共享区。 |
| `gpu_oom` | 57 | 56条既有Region记录在参考解释器中OOM；另1条旧版在PyTorch参考GEMM中`cublasCreate`分配失败。 |
| `assertion_failure` | 5 | 4条实际为设备busy/unavailable，1条为OOM。`TORCH_USE_CUDA_DSA`提示被误分类；不是5个device assertion bug，也没有证据把设备不可用归因于某个编译失败。 |
| `other` | 10 | probe的`.view(torch.uint8)`在末维stride=0的单列张量上报错；是检查器问题。 |
| 已有具体数值容差误报 | 3 | `2432022d37c01733`、`8407027ddaa21060`为百万级结果差0.5；`c6f65c66acaafb6f`为近零差约4.56e-5被相对尺度放大。 |
| 旧版oracle语义不一致 | **3,255（去重）** | 静态规则标出2,947条tile/global归约不一致，以及1,689条分母`sum+eps` / `clamp(sum)`不一致，两者有重叠。当前记录不能作为目标bug证据；不代表这些程序排除了同时存在其他编译器缺陷。 |
| `warp_partition` | 88 | 全部tile=16×16，其中62条128线程、26条256线程。当前`ComputeDefaultWarpPartition`要求warp分解能覆盖tile且每warp具有最小M/N；这些配置没有满足约束的划分。暂按不支持的调度组合排除，不计入6类确认bug；如讨论诊断健壮性，应另立议题。 |

前五项共637条，连同3,255条oracle语义不一致，**3,892条当前不能用作有效目标bug证据**；另外88条不支持的warp配置暂不计入真实bug。不要从总数中减去这些数字，把余数全算作真实bug。

静态确认的oracle不一致例子：kernel在`block_N`个元素上执行`T.reduce_max`，oracle却对长度N的整行执行`.max(dim=-1)`；N>block_N时不是同一数学程序。另一个例子是kernel除以`row_stat[i] + 1e-6`，oracle除以`sum.clamp(min=1e-6)`：当sum=-1时两个分母分别接近-1和正1e-6。这不需要GPU即可证明公式不一致。

此外，抽查发现部分旧程序的oracle在exp前加了`clamp(-80,80)`而kernel没有；该问题没有额外批量计数。旧版`_finite_compare`只比较两边都finite的位置、全部非有限时直接通过，也会漏报。程序本身合法，不等于参考计算与被测kernel等价。

## 不足以确认的结果

- `layout_inference`：163条，不能仅凭`no available layout`断定是真bug或非法程序，需逐例核对布局支持范围。
- `timeout`：169条，缺少足够阶段证据；不能统一称为死锁，也不能照搬旧报告全部归因为多线程卡死。
- 另2条TileLang目录内`segfault`是NVCC子进程崩溃，需独立缩减，不混进13条Triton编译崩溃。
- 大量`wrong_result`仍待验证，包括数值病态、精度模式、边界归约及真实codegen错误。Triton的float32 `tl.dot`默认精度不能当作IEEE fp32精度；参见[官方dot说明](https://triton-lang.org/main/python-api/generated/triton.language.dot.html)。应核对保存代码及安装版本，不能以放宽一个全局阈值代替分析。

## 对旧报告的修正

`reports/2026.07.01/report.cn.md`和`reports/2026.07.11/report.cn.md`中的“wrong_result全部是假阳性”“shared_memory_overflow是真实编译器bug”“dtype_mismatch由fp32累加器污染输入类型导致”等判断不能继续直接采用。当前目录的K=8及负常量dot最小实验支持真实wrong-result缺陷；资源超限本身没有证明编译器缺陷；dtype问题有更直接的缓存对照解释。

本次未修改results和生成器。逐条分类、完整Python文件名、证据来源和SHA256见[all_results_triage.csv](all_results_triage.csv)；计数见[all_results_summary.json](all_results_summary.json)。可使用[audit_all_results.py](audit_all_results.py)重新生成静态清单。该脚本是证据分流工具，不是自动证明程序合法性或bug真实性的判定器。
