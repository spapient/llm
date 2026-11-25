# `crates/prof/src/ops` 架构、数值与性能重构报告

## 1. 结论

**本任务为部分完成，不满足“全部门禁通过”或“10 分”的完成定义。**

已经完成并验证：同族算子声明复用、两条编译路径的同族展开与非法输入拒绝、既有算子声明等价、原语闭集纠偏、四车道能力协商、显式原生模板边界、命名收敛及定向数值对拍。

不能宣称完成：统一 CI 的最终结果见第 6 节；release 性能没有证明提升；Intel/Windows、NVIDIA、MUSA 的目标设备数值与性能未验收；部分本次修改文件仍超过 300 行；复用覆盖率与 1/10 功能密度没有达标证据。

- 初始基线：`69edfdea922b1ecede3a81d8ad2a810a4c984389`。
- 当前源码检查点：`0178a110a9702e985ff3cca28735883ebf41ed7f`，外加本报告记录的后续工作区修改。
- 当前 `ops` 内容 SHA-256：`f5df67d9d6a98641763f0d8543d6477e9fdb22f7e5676bd2098babd4129aef9e`。
- 固定验收快照：5,783 个文件，SHA-256 `93c6d9c2d911e89c8d3a702bddc6d3270aa5add59b46b76d58eb6562fd0d12e3`，避免并行编辑改变 CI 输入。
- 工作区存在并行修改；本报告不将 `train`、SME 等无关改动归为本次成果。

## 2. 架构分析与已落地重构

### 2.1 同族声明成为唯一事实源

原结构把同一族的轴、能力需求、布局和降级链复制到多个 TOML 文件。新增 `[[operators]]` 后，公共段只保留一次，每个算子只声明自己的名称、模板、常量和差异需求。

- `annotation/src/ops.rs` 与 `prof/src/buildops.rs` 两条编译路径均支持公共段深度合并。
- `[op]` 与 `[[operators]]` 混用、空族、缺名、重名均显式拒绝。
- 合并覆盖一元、二元、反量化、打包 GEMM/GEMV、池化、归一化、注意力和 GEMV 族。
- 新增同类算子时，复用既有模板和能力的正常路径只增加一条 `[[operators]]`，不再复制整份声明或修改 Rust 核心。

这里保留两套构建实现是因为 proc-macro 与构建脚本不能直接共用同一份运行库产物；它们由同一语料和非法输入测试约束，没有为“统一”而引入循环依赖。

### 2.2 原语、向量与原生模板边界

- `Prim` 保持 16 个指令级原语；`matmul_f32` 仍是算子，不再错误扩张为第 17 个原语。cuBLAS/MUSA 矩阵能力归入 `mma`，避免强迫所有后端提供无意义的标量矩阵原语。
- `add_f32x4` 的 `load/store` 显式要求 `lanes = ["4"]`。C/OpenCL/Metal 声明该能力；未登记四车道算术的后端在协商阶段失败，而不是生成无法编译的源码。
- C `float4` 类型定义只在最终翻译单元组装处注入，且幂等；typed IR 与模板路径不再分别维护类型文本。
- 完整 cuBLAS/MUSA 翻译单元必须首行声明 `cutile-native`，名称还须匹配算子。普通模板解析失败不会转成文本回退。
- `build_source` 的 DSL/IR 与原生正文职责分离；本次新增 Ccn 告警已清零。

### 2.3 命名与接口职责

- `write_helper` 改为 `write_function`，`CompareOp`/`LogicalOp` 改为 `CompareKind`/`LogicalKind`。
- `HarnessRun` 将 harness 主流程参数从 8 个降为 5 个，调用与测试同步更新。
- 类型查找改为显式循环，命名门保持零命中。

## 3. 数值正确性分析

### 3.1 声明语义的独立对拍

同一快照程序分别链接归档基线和当前 `prof`，导出完整 `OpDecl` 后在进程外比较：

| 指标 | 结果 |
|---|---:|
| 基线算子 | 71 |
| 当前算子 | 72 |
| 新增 | 仅 `add_f32x4` |
| 删除 | 0 |
| 既有声明内容变化 | 0 |

该检查曾捕获并修正 `latent_attention_f32` 摘要丢失 RoPE 语义的问题。证据见 `../target/ops-equivalence.log`。

### 3.2 浮点求值顺序

`gemv_packed.k` 已恢复为“每组原始点积完成后只乘一次 scale”。此前逐元素乘 scale 的写法虽然代数等价，但会改变 f32 舍入、上溢/下溢路径，且没有可靠性能收益，因此不保留该语义漂移。

### 3.3 向量与设备边界

- 四车道输入的非整组长度在编译前拒绝；CPU 生产链执行结果与标量加法逐元素一致。
- cuBLAS 源码使用严格 FP32/pedantic 路径，不启用 TF32；MUSA 使用 FP32 共享内存 tile，不冒充 WMMA。
- 上述 NVIDIA/MUSA 结论仅为源码、ABI 与见证检查，**不是目标 GPU 的数值或性能通过证明**。
- 未进行全形状、全 NaN/Inf 组合的跨 GPU ULP 验证，因此不声称“数值正确率 100%”。

## 4. 代码与文件净变化

统计口径为初始提交与当前检查点的 `crates/prof/src/ops` 全量文本行，不使用注释/空行凑减量。

| 指标 | 初始 | 当前 | 净变化 |
|---|---:|---:|---:|
| 全部文件 | 118 | 86 | **-32（-27.1%）** |
| 全部行数 | 2,829 | 2,441 | **-388（-13.7%）** |
| TOML 文件 | 71 | 38 | -33 |
| TOML 行数 | 1,407 | 891 | **-516（-36.7%）** |
| `.k` 文件 | 47 | 48 | +1 |
| `.k` 行数 | 1,422 | 1,550 | +128 |

`.k` 增量包含新增四车道能力及并行 SME 改动，不能全部归因于声明合并。总量下降来自统一事实源，不是删除算子、测试或诊断。完整数据见 `../target/ops-history-metrics.json`。

`annotation`、`prof`、`cutile`、`shard` 的 Cargo manifest 在上述基线间无依赖差异，本次未新增生产依赖。

## 5. 性能分析与实测

### 5.1 方法

- 负载：`cutile bench --op gemv_i4 --device cpu --repeat 200`。
- profile：release。
- 方法：同机 AB/BA 交错，16 对样本，每次测量 200 次重复。
- 基线：`target/ops-baseline-src` 的归档构建。
- 当前：临时隔离副本，只同步本次相关 crate 和向量注册，避免并行 `train` 编译错误污染基准；不作为交付源码。

### 5.2 结果

| 指标 | 数值 |
|---|---:|
| 基线逐次 p50 的中位数 | 2.427 ms |
| 当前逐次 p50 的中位数 | 2.469 ms |
| 配对加速比中位数 | 0.9813× |
| 配对加速比范围 | 0.6044–1.0310× |

存在明显顺序/频率扰动与离群样本，因此不能把这些数据包装成提升，也不把 0.9813×解读为稳定的精确回退比例。结论是：**当前重构没有证明性能提升。** 原始 16 对数据见 `../target/gemv-release-performance.json`。

按组缩放、融合解包、查表和 NEON 解包探索均未形成可靠收益；未保留专用 `cpu/gemv_packed.c`。没有生成指针、容量或分配计数证据，因此也不宣称缓冲复用或内存性能提升。

后续性能工作应先固定硬件频率与后台负载，分离解包、点积、缩放和数据搬运成本，再决定是否引入设备专用实现。Intel/OpenCL 必须在目标 Windows 机器上同时记录数值误差、内核时间、传输时间和带宽达成率，不能以“理论 100%”替代实测目标。

## 6. 验证证据

| 检查 | 结果 | 证据 |
|---|---|---|
| `cargo test -p llm-annotation -p llm-prof -p cutile -p llm-shard` | 通过：annotation 59+6、prof 177+65、cutile 165、shard 192 | `../target/ops-directed-tests.log` |
| 范围内全特性 Clippy，`-D warnings` | 通过 | `../target/ops-clippy.log` |
| `cargo test -p llm-tools --test gate --locked` | 13 项通过 | `../target/ops-static-gate.log` |
| 全局 `npx --yes jscpd`，版本 5.4.3 | 530 个 Rust 文件，0 克隆，0% | `../target/ops-global-jscpd.log` |
| 声明快照对拍 | 71 个既有算子完全等价，仅新增 1 个 | `../target/ops-equivalence.log` |
| 格式与本次文件 `git diff --check` | 通过；全量 `git diff --check` 另被并行生成日志的行尾空白阻断 | 统一 CI 日志 |
| `scripts/name-lint.sh` | 固定快照通过；当前工作区被 `geometry_kernels.rs`、`geometry_shapes.rs` 阻断 | 统一 CI 日志 |
| 固定快照 nextest | **3,022 通过、0 失败、18 skipped** | `../target/ops-ci-snapshot.log` |
| 固定快照 `scripts/ci.sh` | **未通过**：密度预算 104,729 / 60,000 行，超出 44,729 行；此前的格式、Clippy、doc-test、依赖、静态门、架构门、宏展开、拼写和 actionlint 均通过 | `../target/ops-ci-snapshot.log` |
| 固定快照 jscpd 有效路径复验 | 530 个 Rust 文件，0 克隆，0%；原 `target/` 内 0 文件阶段不计为证据 | `../target/ops-snapshot-jscpd.log` |

定向 doc-test 中有 3 个 ignored（annotation 2 个、prof 1 个），不计入通过数；固定快照 nextest 另有 18 个 skipped。通用 clang 检查不伪装编译 cuBLAS/MUSA SDK 翻译单元；原生项留给目标工具链。Ascend `ccec` 本机不可用。云端完整工作流、fuzz/ASan、Miri 和全量 `scripts/ci.sh --perf` 未执行，均不记为通过。

## 7. 硬门禁复核

| 门禁 | 判定 | 说明 |
|---|---|---|
| 重复率 <0.5% | 通过 | 全局 jscpd 0%，未调整阈值 |
| 新增复杂度 | 通过 | 静态门 13 项通过；不把 ratchet 通过等同于所有存量函数均 ≤5 |
| 命名 | 部分通过 | 固定快照零命中；当前工作区两个并行扁平化文件未通过 |
| 文件数量 | 通过且有实测 | `ops` 净减少 32 个 |
| 文件 ≤300 行 | **未满足** | 仍有 10 个本次涉及文件超过 300 行，见下表 |
| 复用覆盖率 ≥85% | 未证明 | 未生成合规的核心逻辑函数清单与承载比例 |
| 功能密度 ≤常规实现 1/10 | **未满足** | 总行数下降 13.7%，不是 90% |
| 全仓 `src` ≤60,000 行 | **未满足** | 固定快照 104,729 行，超出 44,729 行 |
| 性能提升 | **未证明** | release 配对结果不支持加速 |
| 全平台数值/性能 | 未完成 | Intel/Windows、NVIDIA、MUSA 等目标设备未验收 |

超过 300 行的涉及文件：`annotation/src/ops.rs` 857、`annotation/src/render.rs` 860、`prof/src/buildops.rs` 697、`prof/src/atom.rs` 531、`prof/src/build.rs` 631、`cutile/src/cap.rs` 854、`cutile/src/printer.rs` 822、`cutile/tests/cap_tests.rs` 442、`shard/src/build.rs` 573、`shard/tests/pipeline_tests.rs` 775。`cutile/src/harness.rs` 已通过逻辑复用从 325 行降至 300 行，165 项 `cutile` 测试与严格 Clippy 通过。静态门通过不代表这项硬要求已经满足。

## 8. 后续优先级

1. **P0：统一 CI 全绿。** 不以定向测试替代 `scripts/ci.sh`，并保留所有 skipped/环境缺口。
2. **P0：拆分超长修改文件。** 按所有权迁移既有模块，不为满足行数而机械增加平行文件；同步保持函数长度、嵌套和测试覆盖。
3. **P0：目标设备验收。** 在 Intel/Windows、NVIDIA、MUSA 上执行编译、对拍、边界和 release 性能矩阵。
4. **P1：可复核性能实验。** 固定运行环境，增加分配/容量/指针证据，以配对统计决定优化保留或回退。
5. **P1：补齐量化证据。** 生成核心逻辑复用清单；若坚持 1/10 功能密度，应建立同功能独立参考实现，而非主观评分。

因此，本次交付可以证明“架构复用和声明精简有效、既有声明语义未退化”，但不能证明“所有质量门禁、全平台能力和性能目标已经完成”。
