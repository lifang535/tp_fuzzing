# 复现证据的来源与执行服务器

- original_representatives/ 的 12 个原始代表程序全部取自 **41790 / RTX 4090** 第四轮结果；完整来源路径见 selected.json。
- replay_41790/ 为这 12 个代表程序在 **41790** 停止实验后的重放日志，原始报警均可复现；数值报警可复现不等于已证明 DSL 数值 bug。
- isolated_*.py / .log 在 **32906 / vGPU-32GB** 执行，从上述代表程序中提取 kernel，去掉 fuzzer oracle 后独立编译。
- minimal_checks.py / .log 在 **32906** 执行：Triton flip 默认参数失败、显式 dim=0 通过；其中简单 bool 小程序未复现，不能作为 bool 缺陷证据。
- minimal_cp_async.py / .log 在 **32906** 执行：同一简单矩阵乘法 stages=4 失败、stages=0 编译通过。
- atomic_checker_control.py / .log 在 **41790** 执行：只修改审计副本中 atomic 检查范围，kernel 不改，输出 ALL PASSED。

两台机器上同签名的原始个案分别列在对应服务器目录的 confirmed_cases.csv。代表样本验证不代表全部个案均逐一重跑。所有审计脚本使用 tp_fuzzing_latest；实验任务停止后没有自动恢复。
