# results 统一汇总表

依据 [ALL_RESULTS_REVIEW.cn.md](ALL_RESULTS_REVIEW.cn.md) 的三部分内容合并，后端与数量按 `all_results_triage.csv` 核对。快照为2026-09-17 21:16，共14,425条失败记录。

“真实缺陷机制”表示该机制有复现证据，不表示该行全部样本已逐条复现；数量不是独立bug数。正确性问题没有crash，因此标明检测阶段。前19行互斥覆盖全部14,425条记录，末2行是未单独计数的补充风险。K=8最小控制程序不额外计入results记录；同配置未确认样本仍列入待确认行。

| root_cause | 后端 | crash 时机 / 检测阶段 | 判定 | 记录数 | 说明 | 对应代码名称 / 样本 |
|---|---|---|---|---|---|---|
| dtype_mismatch | TileLang | runtime fail（参数检查） | 真实缺陷机制 | 2,442 | JIT缓存未区分捕获的dtype；同类记录未全部重放 | dtype32_same_cache.py；JITImpl._frontend_cache_key_data |
| ptx_async_boundary | TileLang | compile fail | 真实缺陷机制 | 209 | 普通fp16拷贝生成非法2字节cp.async；同类记录未全部重放 | 6ffb52d51060569a；GetTileLangCPAsyncTransferBytes |
| wrong_result / schedule_mismatch | TileLang | 执行后结果校验（非crash） | 真实缺陷 | 1（原始样本已复现） | fp16、block_K=8 GEMM列重复；其余同配置候选计入待确认行 | block_k_8_exact.py::control；b5c2b8bb73c6097d；T.gemm |
| nondeterminism | TileLang | 重复执行结果校验（非crash） | 真实缺陷 | 1 | 共享内存依赖缺少同步；补屏障后80次运行通过 | 28f7a4c7a8eda0b4；T.copy / T.sync_threads |
| segfault | Triton | compile crash（SIGSEGV） | 真实缺陷 | 13 | 离线编译崩溃；具体C++ pass未定位，不代表13种根因 | 4abdeda508af655d；CUDABackend.make_ttgir / pm.run(mod) |
| wrong_result | Triton | 执行后结果校验（非crash） | 真实缺陷 | 1 | fp16负常量dot操作数产生NaN，正常量和内存操作数对照通过 | triton_constant_dot.py::dot_control；25a869b78ca36590 |
| shared_memory_overflow | Triton | runtime fail（加载/启动） | 排除 | 3 | 共享内存申请量超过硬件上限 | check_shared_memory；CompiledKernel._init_handles |
| shared_memory_overflow | TileLang | runtime fail（加载/启动） | 排除 | 559 | 生成调度的共享内存预算不合理 | src/backends/tilelang/params.py::check_shared_memory |
| warp_partition | TileLang | compile fail | 暂按不支持的配置排除 | 88 | 16×16 tile配128/256线程，无满足当前实现约束的warp划分 | ComputeDefaultWarpPartition |
| gpu_oom | TileLang | 参考计算阶段 runtime fail | 排除 | 57 | PyTorch参考计算显存不足，不是目标kernel正确性证据 | _region_reference / _typed_region_reference；cublasCreate |
| assertion_failure | TileLang | 运行阶段（输入分配/资源操作） | 排除：分类错误 | 5 | 4条设备busy/unavailable、1条OOM；DSA提示被误识别 | TORCH_USE_CUDA_DSA（日志提示，非实际断言） |
| other | TileLang | probe检查阶段 runtime fail | 排除：检查器问题 | 10 | 末维stride=0的单列张量无法按当前方式转换字节视图 | .view(torch.uint8) |
| wrong_result | TileLang | 结果校验（非crash） | 排除：容差误报 | 3 | 百万级结果差0.5，或近零误差被相对尺度放大 | 2432022d37c01733；8407027ddaa21060；c6f65c66acaafb6f |
| wrong_result | TileLang + Triton | 参考计算/结果比较（非crash） | 排除：oracle语义不一致 | 3,255（TileLang 3,190；Triton 65） | tile与整行归约范围不同，或sum+eps与clamp(sum)分母不同；两类去重计数 | accumulate_reduce；T.reduce_max / T.reduce_sum；sum.clamp(min=1e-6) |
| layout_inference | TileLang | compile fail | 待确认 | 163 | no available layout；需核对布局约束与支持范围 | 2ffa308929b8d63e；LayoutInference |
| timeout | TileLang + Triton | 阶段未知（超时终止） | 待确认 | 169（TileLang 164；Triton 5） | 无法区分编译慢、参考计算慢、资源竞争及kernel hang | Execution timed out |
| segfault | TileLang（NVCC子进程） | compile crash | 待确认 | 2 | NVCC编译子进程崩溃，需缩减；不混入Triton崩溃 | 12125edd1cbbc6f2；d55f8abb84d0887d；nvcc |
| wrong_result | TileLang + Triton | 执行后结果校验（非crash） | 待确认 | 7,428（TileLang 7,254；Triton 174） | 含204条Region K=8候选、1条疑似数值不稳定及其他未确认结果；可能涉及精度模式、边界归约或codegen | 119930b0d5c760cc；_finite_compare；tl.dot；其余见逐条CSV |
| schedule_mismatch | TileLang | 跨调度结果比较（非crash） | 待确认 | 16 | 14条K=8候选及2条数值容差候选，不能全部计为已确认bug | schedule_mismatch；具体文件名见逐条CSV |
| wrong_result相关风险（未单独归类） | 旧版程序（已抽查TileLang） | 参考计算（非crash） | fuzzer问题：待进一步统计 | 未单独计数 | 部分oracle在exp前clamp(-80,80)，kernel没有；公式不同 | torch.exp(...clamp(-80,80)) / T.exp |
| wrong_result相关风险（可能漏报） | 旧版TileLang + Triton检查代码 | 结果校验（非crash） | fuzzer问题：检查器漏报风险 | 未单独计数 | 只比较双方finite位置，全部非有限时直接通过 | _finite_compare |

既有复现证据来自此前实验；本轮仅整理表格。原始结果及逐条判定未修改。
