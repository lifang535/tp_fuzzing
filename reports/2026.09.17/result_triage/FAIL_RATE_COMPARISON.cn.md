# 22.43两轮实验失败数量对比

本报告只读检查当前源码、各轮summary.json和保存的程序；未重新运行GPU实验。数字是本次读取时的快照，不表示各campaign已经完成。fail表示fuzzer发现的异常记录，不等于已确认编译器bug。

## 实际数量

| 后端 | campaign | 已测试 | 失败记录 | 失败率 |
|---|---|---:|---:|---:|
| TileLang | 07.08-15.44 hard | 30,058 | 5,781 | 19.23% |
| TileLang | 09.15-23.03 hard | 3,923 | 755 | 19.25% |
| TileLang | 09.17-00.09 hard | 2,878 | 265 | 9.21% |
| TileLang | **09.17-22.43 hard** | **966** | **91** | **9.42%** |
| Triton | 07.08-15.02 hard | 703 | 132 | 18.78% |
| Triton | 09.17-20.03 hard | 586 | 32 | 5.46% |
| Triton | **09.17-22.43 hard** | **7,821** | **439** | **5.61%** |

相对于最近版本，两个后端的失败率都没有明显下降。新TileLang失败数少主要是已测试数量少；新Triton失败数实际比上述旧轮次更多。相对于7月版本，失败率确实下降，但测试对象和oracle已经变化，不能直接解释为发现bug能力下降或编译器修复。

## 当前测试构成

从每份成功/失败JSON提取IR类型；Region v1为probe，v4为typed普通Region。

| 后端 / 路线 | 测试数 | 失败数 | 失败率 |
|---|---:|---:|---:|
| TileLang普通Region | 517 | 87 | 16.83% |
| TileLang probe | 197 | 0 | 0% |
| TileLang Extended | 252 | 4 | 1.59% |
| Triton普通Region | 4,176 | 422 | 10.11% |
| Triton probe | 1,967 | 5 | 0.25% |
| Triton Extended | 1,678 | 12 | 0.72% |

当前近半数样本来自本轮失败率很低的probe/Extended路线，拉低了总失败率。对照09.17-00.09 TileLang：普通Region 1,940个、262失败（13.51%），probe 938个、3失败；当时没有Extended。因此新轮普通Region的观察失败率反而更高。这是样本统计，不是受控因果实验。

CLI新实验默认extended_prob=0.25，Region内部probe概率0.20；这些概率只控制fresh生成，另有50% mutation和种子反馈，所以实际比例不严格等于25%与15%。源码：main.py、src/workflow/generator/generator.py::ProgramGenerator.generate、src/workflow/fuzzer/fuzzer.py::_generate_test_case。

## 源码解释

1. **旧版高频误报来源减少。** 旧single/pipeline/dynamic已退出当前生成路线。旧版存在tile归约与全矩阵参考不一致、分母公式不一致；当前Region有结构化参考解释器。此前全目录审查已静态标出3,255条旧版oracle不一致记录。这能解释部分长期差异，但没有同程序A/B重放，不能声称本次下降全部来自这项。
2. **dtype缓存缺陷的触发路径改变。** 新TileLang的517个普通Region全部typed=true；09.15旧轮2,312个普通Region全部未typed。common/region_emitter.py按typed分流；tilelang/typed.py将dtype字符串直接写入T.Buffer和分配代码，因此换dtype也会改变JIT源码。原有非typed路径tilelang/region.py仍保留外部dtype绑定及对应回归测试，但本轮普通Region没有走它。09.17-00.09的1,940个普通Region也已全部typed，因此这是较早版本到typed版本的变化，不是22.43才发生的回归。不能把没有dtype_mismatch误解为目标缓存bug已经修复。
3. **拷贝与边界路径改变。** typed GEMM入口使用T.Parallel + T.if_then_else的显式边界读写，区别于非typed连续输入下的普通T.copy路径。新TileLang普通Region里N/K=1的样本为0，窄fp16拷贝触发条件较少；hard-shape与旧easy-shape也不应直接混比。
4. **资源/调度约束前置。** common/builtin.py::sample_region_spec过滤warp划分与共享内存预算；region_generator.py::bound_scratch将typed scratch控制在配置上限内，必要时缩小M/N。这减少无效或过重程序，但共享内存预算并不完整，本轮仍有21条shared_memory_overflow。
5. **没有一刀切过滤K=8真实bug。** TileLangBackend.min_block_k=8仍有效，新轮普通Region中有21条fp16/K8/GEMM配置。它们只是配置样本数，并非21条确认bug。Triton有独立的最小K约束，不应混为TileLang过滤。
6. **检查并非统一放宽。** 新旧近期配置仍使用fp16相对阈值0.1、fp32为0.05；当前检查含NaN/Inf一致性、输出保护区、输入完整性、重复执行及调度配对。7月旧检查会忽略部分非有限值差异，当前这方面更严格；当前容差仍可能误报，不能称已彻底解决。

## 没有发现失败保存丢失

新TileLang：91份failed JSON = summary.bugs_total=91；875份passed + 91份failed = 966。
新Triton：439份failed JSON = summary.bugs_total=439；7,382份passed + 439份failed = 7,821。

max_same_root_cause默认1,000,000，本轮未触及。文件名包含完整IR签名的哈希，而非只用错误类别，因此不同程序不会仅因同一root_cause互相覆盖。此处以实际计数吻合作为没有丢失迹象的主要依据，不声称哈希永不碰撞。

--no-save-artifacts仅将Extended证据放入临时目录后清理，仍执行证据校验；不控制_save_bug或_save_passed。两份新summary未包含后加的save_artifacts字段，不能从当前源码反推当时使用过这个参数。

## 为什么TileLang测试量小

同22.43命名目录中TileLang完成966个、Triton完成7,821个，相差约8.1倍。目录时间和计数不足以证明两进程持续运行了相同时间，因此不能把它直接当吞吐基准。

已有compilation_profile/RESULTS.md的控制实验支持TileLang编译更慢：一个mixed样本冷启动TileLang平均152秒、Triton约6秒，TileLang布局推导约109秒；这是特定样本而非整轮平均。Extended默认配对配置与中间观测可产生4个编译变体，普通Region还有多输入、重复及调度对比，单条程序代价比旧版单核单次检查高。新TileLang的27条timeout也提示耗时值得单独分析，但超时不能直接判定为编译卡死。

## 结论与比较口径

优先同时看测试总数、每路线失败率、确认的独立缺陷机制、单位时间覆盖与发现数。不要单比较failed文件夹大小。若要比较生成器检错能力，应固定测试数/预算、按Region/probe/Extended分层，或回放同一份IR语料；并为dtype外部绑定、普通边界T.copy及K=8等已知机制保留专门回归路线。当前证据没有证明22.43版本漏存失败或突然失去检错能力；确实存在默认生成分布改变导致部分旧bug路径触达减少的情况。
