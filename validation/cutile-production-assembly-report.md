# cutile 生产装配路径调研与四层纵向优化报告

- 日期：2026-11-25
- 基线：`235dc945`
- 范围：`cutile` DSL/IR、`engine::Program`、`nn` 算子语义、`serve` 图装配契约
- 结论级别：纵向切片已落地；完整目标与“业界领先 50 年”均不作无证据声明

## 执行摘要

**cutile 已经进入生产计算路径，但只是原生微内核通道，不是 typed DSL/IR 通道。** 当前调用关系为 `serve → nn → cutile::native → prof::native`，其中 `cutile::native` 是既有兼容重导出；`serve::assembly` 仍把 `nn` 层栈装配成 `engine::Graph`，没有直接消费 `cutile::Catalogue`、能力协商、模板拼接或 typed IR。换言之，“cutile 是否进入生产装配路径”不能回答成简单的全有或全无：原生算法体已经在生产，通用 DSL/IR 尚未进入。

**本轮不将 typed DSL/IR 提前接入生产。** NVIDIA 官方在 2026 年 9 月仍明确称 `cutile-rs` 尚非 production-ready，且限定 Linux、CUDA 13.3 与计算能力 8.0+；本仓 ADR 0056 也记录 typed IR 的设备闭环、全元素档与 L3 证据未闭合。将这样的实现替换已经工作的跨设备生产路径，会同时降低可移植性、可恢复性和证据强度。[NVIDIA](kimi-cite://web?url=https%3A%2F%2Fdeveloper.nvidia.com%2Fblog%2Fintroducing-cuda-rust-two-tracks-for-writing-gpu-kernels%2F&text=Both%20projects%20are%20early-stage%20and%20neither%20is%20production-ready "citation")

落地的纵向切片包括：`cutile` 融合参数合并预分配、`engine` 与 `prof` 访问器宏单一事实源、`nn` RoPE 位置表零转换缓冲读取、`serve` 稠密/量化与注意力适配职责分离。生产代码相对基线净减 **30 行**，其中 `serve` 净减 **50 行**；新增一个独立生产文件，未新增第三方依赖。这个幅度是有效但有限的减法，未达到“1/10 功能密度”或“大量减少代码”。

RoPE 在 release、相同 64 元素输入和 32 个批位置、每路径 220,000 次迭代的条件下，从 **154.406 ns/次**降至 **146.919 ns/次**，同机中位数改善约 **4.85%**；每次调用的分配从 **3 次降为 2 次**。剩余两次来自 `tensor::View::one` 的维度和步长 `Vec`，因此只宣称删除位置转换分配，不宣称整条路径零分配。

## 1. 调研裁决

### 1.1 上游方案比较

| 方案 | 可借鉴机制 | 不应照搬的部分 | 对本仓裁决 |
|---|---|---|---|
| NVIDIA cuTile / Tile IR | tile 一等值、所有权分区、按目标后端复用 ABI 与加载 | NVIDIA 专用、官方仍称非生产就绪、硬件与工具链约束强 | 未来可作可选 NVIDIA 后端，不作唯一生产 IR |
| CubeCL | Rust 前端生成 IR、硬件参数编译期特化、缓存 autotune | 通用 IR 与多 runtime 的供应链和维护成本 | 保留现有窄 IR，暂不引入外部编译栈 |
| TileLang | 源码定位、pass 追踪、按能力回退、统一基准协议 | Python/TVM 依赖、自动布局推导的复杂度 | 采用其证据与诊断方法，不引入实现 |
| CuTe / CUTLASS | 少量布局和原子组合出大量内核、静态整除检查 | 完整布局代数与 NVIDIA 专用模板元编程 | 继续描述符表和闭集校验 |
| Burn / CubeK | 共享算子与设备 runtime 选择分离；轻量 CPU 与 JIT 双轨 | 大型后端 trait 面和额外运行时装配 | 保持 `nn` 语义、`engine` 提交、设备后端分层 |
| MLIR | 分层 dialect、可复用 pass、渐进 lowering | MLIR 本身不是现成编译器，前端和转换仍需自建 | 不以新增 MLIR 依赖冒充减码 |

TileLang 的 CUDA Tile IR RFC 明确建议复用既有 pass pipeline、host/device split、packed API 与模块加载，仅在有新产物或运行约定时增加后端。这支持“目标后端变体”而不是“重写生产编译栈”的路线。[TileLang RFC](kimi-cite://web?url=https%3A%2F%2Fgithub.com%2Ftile-ai%2Ftilelang%2Fissues%2F4403&text=This%20backend%20reuses%20PassPipeline%2C%20the%20host%2Fdevice%20split%20and%20the%20packed%20API "citation") CubeCL 同样把 Rust 前端、IR、硬件特化和 autotune 缓存分开，但其通用性并不自动等于本仓净收益。[CubeCL](kimi-cite://web?url=https%3A%2F%2Fgithub.com%2Ftracel-ai%2Fcubecl&text=Comptime%20specializes%20kernels%20at%20compile%20time "citation")

TileBench 要求统一参考实现、容差、数据类型、输入形状、预热和计时协议，并分别报告默认与 autotune 路径。Burn 的仓库边界则把共享算子放在 `burn-cubecl`，设备 crate 只选择 runtime，同时保留轻量 CPU 路径。这两个实践分别约束了本轮的性能测量和分层方式。[TileBench](kimi-cite://web?url=https%3A%2F%2Farxiv.org%2Fhtml%2F2609.29067v1&text=20%20warmup%20runs%2C%20and%20100%20timed%20runs "citation"), [Burn](kimi-cite://web?url=https%3A%2F%2Fgithub.com%2Ftracel-ai%2Fburn%2Fblob%2Fmain%2FAGENTS.md "citation")

### 1.2 生产准入矩阵

| 通道 | 当前生产状态 | 本轮动作 | 晋升到通用生产前必须补齐 |
|---|---|---|---|
| `prof::native` 经 `cutile::native` 重导出 | 已在 `nn` 高性能路径使用 | 保持单一事实源 | 持续真机对拍和 release 基准 |
| `cutile` 声明、协商、模板拼接 | 服务 `doctor/verify/bench`，未由 `serve::assembly` 直接调用 | 不改变边界 | 明确生产调用者和错误恢复契约 |
| `cutile` typed DSL/IR | 有部分 C/设备 lowering 与融合验证，但覆盖不完整 | 仅做融合分配减法 | 全元素档、设备 L2、匹配 roofline L3、真实消费者 |
| NVIDIA `cutile-rs` | 未进入本仓依赖图 | 不引入 | 官方成熟度、跨设备替代路径、供应链评估 |

## 2. 架构与实现

### 2.1 不变量与分层

数据定义仍由 `prof::Operator`、`nn::registry` 和算子声明持有；`cutile` 负责 DSL/IR 与生成机制；`engine::Program` 只负责已登记内核在绑定缓冲上的执行；`serve` 只负责层栈到图节点的装配和视图适配。没有新增 `serve → cutile` 依赖，也没有让 `engine` 反向依赖 `nn` 或 `cutile`。

可变差异继续由表项、函数指针和声明式宏承载。`serve` 的算术没有复制到适配层，`engine` 没有新增第二份算子目录，`prof` 的访问器宏成为唯一实现，`nn` 的两种批位置输入共用同一个几何校验、频率表和旋转写回核心。

### 2.2 实际变更

| 层 | 变更 | 复用/质量收益 |
|---|---|---|
| `cutile::fuse` | 合并生产者与消费者参数时按两侧容量之和预分配 | 避免参数 Vec 增长重分配；不改变融合语义 |
| `prof::exec` / `engine::cpu` | `value_methods!` 导出并复用，删除 `engine` 的 7 行副本 | jscpd 从 1 个克隆降为 0，行为只有一份 |
| `nn::rope` | 新增 `forward_batch_from_f32`，与 `forward_batch` 共用 `forward_batch_with` | 删除 `Vec<usize>` 转换，保留先全段校验再写回 |
| `serve::kernels` | 公共注册、视图检查与注意力/RoPE 同文件；稠密/量化独立成 `dense.rs` | 294 + 233 行，满足 230–899 的正式文件粒度门；宏生成同构适配 |

`serve` 从 577 行单文件变成合计 527 行的两个职责文件，净减 50 行（约 8.67%）。全部相关生产文件相对基线净减 30 行；测试新增 18 行用于两种位置档逐元素等价对拍。新增生产文件一个，新增第三方依赖零个。

### 2.3 被否决的实验

有序 `Vec` 二分查找曾把执行器查询改为 O(log n)，但 release 实测反而从约 32.7 ns/节点变为约 37.3 ns/节点，原因是比较过程中反复经 `dyn Program::name()` 间接调用。随后实现 `HashMap<String, usize>` 索引，并在五轮、每轮 11 组采样、每组 2,700,000 次节点执行的交替对照中，得到约 1.0007 的同轮比值中位数，未形成稳定收益。

两个方案均未保留。最终 `engine::Program` 继续使用线性函数表，避免为理论复杂度牺牲实测性能与内存。实验原始输出保留在 `validation/engine-program-experiment.log`，该文件只用于说明否决过程，不作为最终性能提升证据。

## 3. 正确性与性能证据

### 3.1 RoPE 对照

探针位于 `validation/rope-position-bench`，使用 release、`lto = "thin"`、`codegen-units = 1`。输入为 64 个 `f32` 元素、32 个位置、头宽 2；每种路径先预热，再执行 11 轮，每轮 20,000 次，合计 220,000 次。分配计数覆盖整个计时区，不只覆盖位置转换语句。

| 指标 | `Vec<usize>` 转换路径 | `f32` 直接读取路径 | 变化 |
|---|---:|---:|---:|
| 中位耗时 | 154.406 ns | 146.919 ns | **−4.85%** |
| 计时区分配次数 | 660,000 | 440,000 | **−33.33%** |
| 每次调用分配 | 3 | 2 | 删除 1 次位置转换分配 |

两种公开入口共用同一份旋转实现，新增测试对 `[3, 9]` 两种位置档的输出逐元素相等进行验证。既有空位置、零头宽、非整数倍、输入输出不等长和单段等价测试保持原样并继续通过。

### 3.2 回归与质量

定向回归最终保留结果：`cutile` 166、`engine` 101（2 个既有忽略）、`nn` 562（4 个既有忽略）、`serve` 198 全部通过；`prof` 177 全部通过。严格 Clippy 通过，`npx --yes jscpd` 报告 535 个 Rust 文件、163,030 行中 **0 克隆、0.00% 重复**。

统一入口使用 `PATH="$PWD/.tools:$PATH" scripts/ci.sh --perf`，日志写入 `validation/cutile-production-assembly-ci.log`。最终 nextest 为 **3031 passed / 0 failed / 18 skipped**；doc-test、cargo-deny、cargo-machete、jscpd、静态门、复杂度、架构门、命名门、宏展开、拼写和 actionlint 均通过。随后密度预算失败：全仓 `src` 为 **104,130 / 60,000 行**，超出 **44,130 行**，因此脚本在基准编译与 criterion 实测之前停止，完整 CI 与完整 `--perf` 均标记为未完成。

## 4. 交付前五项自检

### 4.1 重复性

潜在重复点包括访问器宏、逐元素适配、量化 GEMV/GEMM 适配、注意力 prefill/decode 路由和两种 RoPE 位置档。公共承载分别位于 `prof::value_methods!`、`serve` 的声明式宏与共享 qkv/几何函数、`gemv_quant`/`gemm_quant`、`attention_paths`/`decode_paths` 和 `nn::rope::forward_batch_with`。jscpd 的零克隆只是文本证据，上述单一调用链是结构证据。

### 4.2 精简度

精简来自删除宏副本、合并同构适配和共享旋转核心，不来自删除测试、诊断、注释或边界检查。生产净减 30 行是可复核的小幅架构减法；功能密度 1/10 的目标没有达成，文件数也未下降，不能包装成已完成的大规模压缩。

### 4.3 正确性

正常路径由既有 1,000+ 相关单测覆盖；空位置、零头宽、错误 lane、非法几何、未注册/重复注册、内核失败后的缓冲归还继续由原测试覆盖。RoPE 新入口共享“先校验全部段再写回”的路径，因此部分失败不会留下半写输出。新增位置档对拍验证正常行为，未改变原有错误分类。

### 4.4 扩展性

新增一元/二元算子只需在对应宏列表与 CPU 注册表加入口；新增量化 GEMV/GEMM 只需登记函数与标记类型；新增另一种位置表示只需投影到 `forward_batch_with`，不必复制几何校验和旋转写回。新增生产 typed IR 后端则不属于普通表项扩展，仍需先通过 ADR 0056 的证据门。

### 4.5 可维护性

诊断仍使用 `diagnostic::Error` 和原有 site；`engine` 重名与未注册错误不变，`serve` 的输入输出 lane 错误不变。模块拆分按算子族定位代码，宏只隐藏同构拆包，不隐藏算法、副作用恢复或错误路径。被否决的查找实验有原始数据，后续维护者可以复核为何没有采用理论更优的索引。

## 5. 未完成项与后续门槛

1. **“超越业界顶尖 50 年”不可验证且未达成。** 没有该指标的定义、对照集和测量协议，不能签发完成结论。
2. **typed DSL/IR 尚未进入 `serve` 生产装配。** 需要全元素档、设备 L2、匹配硬件 roofline L3、真实消费者与失败恢复证据后另案晋升。
3. **功能密度与大规模减码未达成。** 本轮生产净减 30 行、增加一个生产文件，距 1/10 目标很远；下一轮应优先消除 `train` 与 `serve` 的适配协议重复，而不是继续拆小文件。
4. **视图构造仍分配。** `tensor::View::one` 每次创建维度和步长 `Vec`，RoPE 计时区仍有 2 次分配；需用基准证明内联小秩形状或借用形状的收益后另行优化。
5. **完整 CI 与固定硬件性能门未完成。** 统一 `--perf` 在密度预算处以 104,130 / 60,000 行失败，后续基准编译和 criterion 未运行；nextest 另有 18 项 skipped，平台专项未在本轮执行，不能用本地部分通过替代完整工作流。
