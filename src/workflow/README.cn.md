# 工作流模块

当前 campaign 混合原生 Region/probe 与 ExtendedProgram 两种生成域。使用说明与参数表见[项目 README](../../README.cn.md)。下面的编号步骤侧重原生路线。

生成器、变异器和 Oracle 通过 `src.backends.get_backend()` 调用所选 DSL 的参数策略、emitter、启动配置和诊断规则。实现位于 `src/backends/tilelang/`、`triton/`；公用测试脚本组装位于 `src/backends/common/`，参考解释器和运行检查位于 `src/workflow/emitter/`。添加后端需实现 `src/backends/base.py` 的接口，并通过 `register_backend()` 注册；`--backend-plugin MODULE` 可加载注册模块。

1. `generator/generator.py` 按 `extended_prob` 选择扩展域，未选中则调用 `RegionGenerator.generate()`。CLI 新 campaign 默认概率 0.25；库默认 0。
2. `program_template()` 先生成全部辅助函数和入口的操作树与签名，之后 `instantiate_program()` 按顺序实例化。默认有 1–3 个辅助函数，后续函数和入口只调用先前定义的函数，调用也可以位于 if/for 内。普通入口默认各以 50% 概率选择 load/GEMM。
3. 最后通过 `instantiate_function()` 构建入口并选择整个程序的形状、类型和调度参数。v4 值池区分 fp16/fp32、tile/row/column/scalar 和 tensor/buffer；生成的函数接口仍采用完整 fp32 tile，IR 也支持显式紧凑类型签名。
4. `RegionProgram.validate()` 检查作用域、操作契约、循环边界与类型，以及调用签名和无递归环。Triton 输出 JIT 函数，TileLang 输出宏；参考解释器按独立函数作用域执行调用。
5. `mutator/` 支持显式 dtype 切换、局部操作数/操作/属性变异，以及结构重生成和参数重采样。局部变异保留调用图与作用域。只接受 Region 和 Extended 程序；probe 的生成与变异也返回 RegionProgram。
6. `src/backends/common/region_emitter.py` 分发到相应后端；v4 通过同目录的 `typed_emitter.py` 递归翻译混合形状与内存操作，生成含独立参考解释器与输入 seed 的可执行文件。
7. `oracle/` 在独立子进程编译执行，检查数值、崩溃和超时。新普通 region 的独立测试默认执行 2 组输入、每配置重复 3 次和合法的 128/256 threads 配对，检查输出保护区与输入完整性；v4 还检查每个 block 的 scratch 保护区。线程配对之外还有调度扫掠（num_stages、loop_kind 变体共享同一参考）、备用布局对、tilelang pass-config 不变性配对（按 (程序, seed) 从数值中性池确定性采样子集 `pass_configs`，int8 池排除 `tirx.disable_vectorize`，池含 `tl.disable_warp_specialized` —— sm_89 上无操作、TMA 机器上为真实 warp specialization 差分；triton 以 `enable_fp_fusion=True` 对应）、`T.use_swizzle` 变体和 `GemmWarpPolicy` FullRow/FullCol 变体（warp 沿 M/N 全分配的 gemm，逐 tile 数学不变，共享同一参考；按 warp 划分可行性逐 policy 过滤）。配置随 region v3/v4 IR 保存。
8. `fuzzer/` 在种子池非空时以 50% / 50% 选择变异或全新生成，负责去重、结果保存、恢复和种子调度；文件名仅保留调用关系及完整 IR 哈希。summary.json 的 `root_cause_locations` 键记录 `root_cause → 位置 → 次数` 的细粒度定位（不变性标签 / TILESMITH_STAGE 标记 / TVM pass 名 / 报错源文件）。`feedback.py` 累计操作、嵌套、数据依赖、类型/形状、内存访问、函数数量和调用链等结构特征。

定向 probe 是受限的整函数模板，使用专用布局与数值检查；`--probe-prob 0` 可以关闭。它从生成到执行一直采用 RegionProgram，不经过额外的单算子容器。

普通 region 的分支可以依赖 tile 行/列号、行列号之和或循环索引；`index_add` 将这些索引加入 tile 数据流。循环支持 0–4 次、索引起点 0–3 和步长 1–3。新生成循环预留末尾 carry 合并节点；局部变异仍可改变依赖。辅助函数接收行号、列号和当前循环索引上下文，嵌套循环退出后恢复外层索引。

普通输入布局由 `ir/layout.py` 统一描述，输入初始化和两个 emitter 共享步长、偏移和物理大小计算。GEMM 的 A/B 可以独立选择六种布局；load 仅变异 A 的布局。非连续输入用一维物理 buffer 加显式地址读取，参考端使用逻辑 tensor view；保护检查保存并比较整个物理输入存储，包含填充区。零步长广播只初始化唯一地址。

MLIRSmith 式 op 面扩张在两层 IR 上新增编译器代码路径：普通 Region 加入超越函数元素操作（tanh/erf/log/log2/exp2/rsqrt/sin/cos/floor/ceil）和 int8 GEMM-only 程序（规格取自预校验网格 `INT8_SPEC_GRID`：block_K ∈ {32, 64}、int32 累加、精确整数参考）。fp32 GEMM 不与边界台阶 op（ceil/floor/round/cast）组合，避免 TF32 张量核取整边界相对精确参考的噪声淹没 oracle。新 op 的属性取自 `generator/grids.py` 的 per-(op, backend) 有界实例网格，轮转游标保证每轮扫掠每个角落实例恰好一次（游标随 campaign rng 状态持久化；`--no-instance-grids` 恢复纯随机采样）。

`--typed-op-prob` 默认 0.35；设为 0 生成 v3 对照。新内存操作可嵌套于 if/for 和辅助函数：`load_input` 重读输入，`store_tile` 初始化块私有 global scratch，`write_tile` 更新同一 buffer，`load_tile` 获取快照。buffer 可以由内层区域捕获，但不能通过 yield 或函数调用逃逸。参考端掩码控制分支副作用，零次循环不执行写入。每次 launch 前毒化 scratch 并检查其保护区；默认 scratch 预算为 64 MiB，超限时生成器缩小 M/N。

旧 single_op/pipeline/dynamic 模型、emitter 和兼容包装已删除，旧格式恢复会报错；原生 Region v1–v4 与 Extended 仍可读取。原生路线的 GEMM 仍限入口，归约与转置采用含填充 lane 的 tile 内语义。扩展路线将 matmul 变成显式 SSA 操作，并加入异构多值接口、全 scratch 内容检查和编译阶段反馈，以及全局内存原子操作（可交换 add/max/min）、标量 FMA 链、triton 形状原语（flip/interleave/join/split）和 int8 × int8 matmul（int32 累加）。每个 Extended 程序另按 (程序, seed) 确定性采样随机 pass 管线配置（`backends/common/knobs.py`；池中键均在本机 tilelang 0.1.11 核实过消费者）。显式共享内存分配与异步拷贝仍不作为 IR 构造存在，只经 pass 旋钮间接触及；warp specialization 同样受 TMA（sm_90+）门控、经 `tl.disable_warp_specialized` 旋钮触及，本机 sm_89 上由 `GemmWarpPolicy` FullRow/FullCol 变体配对覆盖 warp 划分层面的构造差异。
