# S++ 第一阶段：现状审计报告（只读诊断，未改业务代码）

- 基线：工作树 HEAD 之上的在途 WIP；审计方法为只读扫描 `Cargo.toml` 依赖、`src` 公开面、门禁与既有测试/基准。
- 范围：25 个 workspace 成员，`src` 约 51 161 行；1292 个 `#[test]`；18 条 criterion 基准；39 份 ADR。
- 口径：本报告只给诊断、分层判断与分级改动方案；**改动经对齐后再进入第二阶段**，不留下半成品。
- 既有回链：[`ARCHITECTURE.md`](ARCHITECTURE.md)、[`audit.md`](audit.md)、[`code-audit-2026-11-17.md`](code-audit-2026-11-17.md)、[`REPO-review-2026-11.md`](REPO-review-2026-11.md)。

## 1. 模块全景

### 1.1 分层与成员（权威定义见 `ARCHITECTURE.md` 一、二节）

| 层 | 成员 | 职责 |
|---|---|---|
| L1 基础设施 | `diagnostic`、`dtype`、`wire`、`tool`、`testkit` | 错误类别/声明数据树、精度闭集、字节码闭集、静态质量门与人可读输出、测试夹具 |
| L2 数据层 | `prof`、`op`、`prim` | 设备档案闭集、算子声明与词汇闭集、原语片段 |
| L3 内核 | `cutile` | 协商 → 拼接 → emit → 编译执行流水线与见证命令 |
| L4 运行时 | `tensor`、`storage`、`tokenizer` | 张量元数据/布局、字节缓冲与唯一 mmap 安全岛、文本边界 |
| L5 模型与训练 | `weights`、`autograd`、`data`、`nn`、`shard` | 权重容器与格式适配、反向图、样本、层定义、切分拓扑与集合通信 |
| L6 调度与服务 | `train`、`engine`、`serve`、`backend`、`cli` | 训练会话/缓冲、连续批与 KV 页、提交执行、端点契约与模型内核桥、设备后端边界、命令行文法 |
| L7 入口 | `llm`（bin：`llm`、`cutile`） | 组装，不定义新类型 |
| 工具成员 | `annotation`、`benches`、`fuzz` | 编译期注解、集中测量容器、模糊测试契约；不参与运行时分层与方向判定 |

### 1.2 运行时路径依赖矩阵（`[dependencies]` 内 path 依赖；外部依赖单列）

| crate | 路径依赖（向下/向左） | 外部依赖 |
|---|---|---|
| diagnostic | — | `toml`（buildgen） |
| dtype | — | `serde`(opt)、`bytemuck`(opt)、`serde_json` |
| wire | diagnostic | — |
| tool | diagnostic | — |
| testkit | — | — |
| prof | diagnostic, dtype | — |
| op | diagnostic, annotation, prof | — |
| prim | diagnostic, prof | — |
| cutile | diagnostic, prim, op, prof | — |
| tensor | diagnostic, dtype | — |
| storage | diagnostic, tensor | — |
| tokenizer | diagnostic | — |
| weights | diagnostic, dtype, tensor, wire, storage(opt) | — |
| autograd | diagnostic, tensor | — |
| data | diagnostic, tensor | — |
| nn | diagnostic, annotation, op, tensor, weights | — |
| shard | diagnostic | — |
| train | autograd, data, diagnostic, nn, op, shard, tensor, weights, wire | — |
| engine | diagnostic, tensor, storage | — |
| serve | diagnostic, engine, nn, op, tokenizer, weights | — |
| backend | diagnostic, engine, storage | — |
| cli | diagnostic | — |
| annotation | dtype | — |
| llm(bin) | cli, cutile, diagnostic, engine, nn, op, prim, prof, serve, tensor, tokenizer, tool, train, weights | — |

外部运行时依赖极薄：`serde`/`bytemuck`/`toml` 均 feature-gated 可选；`serde_json` 仅 `dtype`；`criterion`、`assert_cmd` 为 dev 依赖。**无** clap、regex、rayon、thiserror、anyhow、tracing、log。

### 1.3 依赖方向结论

- 架构守护测试 `tests/architecture.rs` 已把成员集合、边集合、方向（向下/同层向左）、类型归属、L7 不定义类型、错误边界不回退字符串固化为可执行断言；**当前无循环依赖**，依赖单向成立。
- 边界观察：`annotation` 分层表列为工具成员，但被 `nn`/`op` 在运行时路径依赖（`crates/nn/Cargo.toml`、`crates/op/Cargo.toml`）。它实际承担编译期注解设施角色，建议在分层表里明确"工具成员但被 L2/L5 以 buildgen 方式消费"，避免方向判定歧义。

### 1.4 端口（trait）清单——六边形雏形

`exec::Executor/Program/Allocator`、`serve::{Predict,Sampling,Endpoint}`、`nn::Contract`、`op::{Unary,Binary,Gemm,Gemm,Reduce,Element,Format}`、`train::{Kernel,Optimizer,Lane,Inputs}`、`weights::{Reader,Loader}`、`storage::Allocator`、`shard::Collective`、`tokenizer::Tokenizer`、`data::Dataset`、`wire::{Sink,Uint}`、`tensor::Element`、`fuzz::Target` 等 35 个 `pub trait`。
核心领域经端口对接外部设施的骨架已具备；`Box<dyn>` 仅出现在真正的多实现边界（`exec` 程序注册、`nn` 层栈、`data` 数据集、`fuzz` 目标），未见类型擦除滥用。

## 2. 问题清单（按域，定位到文件）

严重度：**高** = 影响正确性/安全/契约；**中** = 影响可维护性/可观测/覆盖；**低** = 风格/收口/重复。

### A. 架构与边界

- **A1（中）根入口命令编排超重**：`src/main.rs` 835 行、22 个自由函数，承载 status/run/train 三条命令的装载、解析、排版编排。L7 规则是"只组装、不写业务"；命令实现应按文法/业务/排版分别下沉 `cli`/业务层/`tool`。现状已在瘦身（见 #62–#64），剩余解析与排版仍卡在既有裁决（见 §6 待决策 D1/D2）。
- **A2（中）单文件接近上限**：`crates/serve/src/assembly.rs` 855 行（门限 899），计划（`*_plan`）、校验（`check_*`）、接线（`wire_*`/`push_block`/`chain`）三段共处一文件，可按 plan/check/wire 拆子模块。其余大文件：`op/decl.rs` 758、`annotation/syntax.rs` 755、`prof/atom.rs` 726、`engine/exec.rs` 717、`cutile/device.rs` 711、`engine/lib.rs` 698。
- **A3（低）工具成员运行时依赖未注释清**：见 §1.3，`annotation` 被 `nn`/`op` 路径依赖，分层表措辞需澄清。

### B. 类型系统 / 泛型 / 生命周期

- **B1（中）端口泛型程度参差**：算子族已有泛型骨架（`op::gemv::Element`、`train::Lane` 密封 trait 双 lane），但 `exec::Program`、`serve::Predict` 仍以 `&[f32]` 扁平切片握手，未把形状/ lane 收进类型；形状合法性靠运行时 `check_*` 校验。可在不破坏现有契约的前提下，逐步把关键端口的操作数按形状/lane 泛型化。
- **B2（低）newtype 覆盖不足**：宽度、头数、步数等原始 `usize` 直接跨端口传递，存在参数错位风险（例如 `q_rows` 与 `hist` 传反）。关键边界可引入 newtype（`HeadDim`、`SeqLen`、`KvHist`）编译期消歧。
- **B3（低）生命周期显式度**：现有借用视图（`tensor::View`、`wire` 游标）语义清晰；`exec` 绑定以 `Vec<f32>` 拥有缓冲为主，未见悬垂或不必要生命周期标注，暂无需改动。

### C. 错误处理 / 安全

- **C1（中）错误无底层 source 链**：`crates/diagnostic/src/error.rs` 的 `Error{code,message,context,notes}` 把 `io::Error` 经 `from_io` 格式化进 message（:80-87），未保留底层错误类型与 `source()` 链；`std::error::Error` 为空 impl（:107）。可回溯性弱于 thiserror 风格的枚举包裹。
- **C2（中，口径已修正）**：下表"全仓 `src` 命中 379 处"是**未剥测试的表面计数**（含 `#[cfg(test)]` 模块与集成测试，测试代码惯例可用 `expect`/`unwrap`）。实施核对：静态门扫描生产代码前先 `strip_tests` 剔除测试条目，生产侧裸 `unwrap/expect/panic/unreachable` 经门禁 **Call 限值 0 且 debt 表无条目**，即生产侧裸 panic 为零；样本核实 `weights/archive.rs` 的 33 个 `expect` 全在 503/555/580/589 行之后的 `cfg(test)` 模块内。剩余裸调用面仅在测试夹具，不属生产错误处理缺口。
- **C3（低）unsafe 收敛良好**：唯一生产 unsafe 岛在 `storage/src/mmap.rs`（FFI `extern "C"`、`Send/Sync` impl、`from_raw_parts`、`munmap`），其余 crate 文档明确不装 unsafe；`unsafe_code = "deny"` 工作空间级生效。后续 Metal/CUDA 边界需按 ADR 白名单逐块写安全不变式。
- **C4（低）无错误枚举派生宏**：`Code` 为手写枚举 + `ALL`/`as_str` 对照；新增类别需多处同步。可考虑 derive/代码生成收口（对照 `annotation::buildgen` 既有模式）。

### D. 并发 / 所有权

- **D1（低）共享面很小**：`Arc/Rc/Mutex/RwLock/RefCell` 全仓仅 10 处，分布 `autograd/graph.rs`、`serve/completion.rs`（端点 `Mutex<State>` 串行化）、`testkit`。无锁竞争热点证据；`Completion` 单序列锁内循环是刻意的串行边界（连续批在 #51），非隐患。
- **D2（中）无异步/背压/取消安全设计**：当前为同步阻塞模型，无 async 链路、无线程池隔离、无背压控制。若未来接入在线服务并发，需显式标注 `Send/Sync` 边界并把阻塞 IO 隔离到专门线程池；本轮不引入 async，避免过度设计。

### E. 性能 / 分配

- **E1（中）前向热路径分配**：`exec` 绑定以 `Bindings: u32 → Vec<f32>` 拥有缓冲，每步前向存在向量分配；`serve/assembly.rs` 装配期 `entry.clone()`/`shape.clone()`（:450-454）属装配期一次性成本，热路径克隆仅 13 处且多为解码器 poison 快照与路由表快照，暂非热点。
- **E2（中）基准覆盖**：18 条 criterion 基准以算子级为主，已含 `model_decode.rs`、`decode_attention_f32.rs`、`train.rs`、`mem_stream.rs`；缺整模型多序列步、TPOT/吞吐/内存对照的固化口径（#53 部分在途）。无 iai 缓存敏感型测量，无 proptest 属性测试。
- **E3（低）profiling 闭环未建**：无 pprof/flamegraph/perf 接入与"先度量再优化"的门禁流程；当前优化依据以算子语义流量下界（`nn::Bounds`）对账，而非真机采样。
- **E4（低）批量处理口径**：既有偏好为整块 slice/单次 put/向量化查表；算子内核多为逐行循环（`op::attention` 行循环），属串行依赖可接受；SIMD/向量化留待 profiling 确认热点后再做。

### F. 配置化

- **F1（中）阈值/常数散落**：`serve/assembly.rs` 的 `INPUT_SLOT/ACTIVATION_BASE/WEIGHT_BASE`（:25-31）、`main.rs::status_pager` 页池尺寸、各算子头宽/ tile 常数。目录类配置已收口 `cli::settings`（三键闭集 + 来源追踪），但算子/页池/采样阈值仍硬编码在源码。
- **F2（低）配置反序列化与启动校验**：`diagnostic::buildgen` 在构建期解析 `prof`/`op` TOML 闭集；运行期配置类型反序列化、默认值、按环境覆盖的统一机制尚未形成。

### G. 测试 / 文档 / 供应链

- **G1（中）测试覆盖不均**：`prof`、`prim`、`tokenizer`、`tensor` 各仅 1 个测试文件，`testkit` 为夹具库无自身测试；`op` 31、`cutile` 22、`fuzz` 14 覆盖较厚。缺属性测试（proptest）与模糊测试在核心算子的系统性接入（`fuzz` 契约框架已存在）。
- **G2（低）文档**：`missing_docs = "warn"`（非 deny）；公共 API rustdoc 头在多数 crate 已具备，部分访问器/内部类型注释不足。
- **G3（低）供应链已有基础**：`deny.toml` 配齐 advisories/licenses/bans/sources；`CHANGELOG.md` 存在；edition 2024、MSRV 1.100 已锁。缺覆盖率门槛、变异测试与 CI 中 `cargo deny`/基准回归的固化记录。

### H. 宏 / 编译期能力

- **H1（低）宏复用面**：`annotation` 提供声明式注解与 buildgen；`serve::model` 有 `kernel!` 宏收敛适配。横切样板（断言、序列化、操作数绑定）可进一步识别重复后抽宏，当前未见明显宏卫生问题。
- **H2（低）编译期求值**：闭集检索（`REGISTRY`、设备/算子词汇表）已在编译期锁定；运行期字符串分派集中在命令文法层，符合"能编译期完成不留运行时"的方向。

## 3. 架构合理性评估与调整建议

| # | 现状判断 | 建议 | 理由 | 风险 |
|---|---|---|---|---|
| R1 | L7 `main.rs` 命令编排超重（A1） | 按第 1 轮已验证的顺序下沉：纯解析 → cli；装载解包 → weights/nn/train；行排版 → tool 或业务层；L7 只留解析→分派→退出码 | 依赖方向合法（L7 向下）；每刀门禁兜底 | 低 |
| R2 | `assembly.rs` 三段共处 855 行（A2） | 拆 `plan.rs`/`check.rs`/`wire.rs` 子模块，私有函数随段迁移，公开面不变 | 降单文件复杂度、内聚清晰；不改图模型 | 低 |
| R3 | `annotation` 工具成员被运行时依赖（A3） | 在 `ARCHITECTURE.md` 分层表注明其 buildgen/注解设施角色与消费边，不改依赖 | 消除方向判定歧义 | 低 |
| R4 | 端口以 `&[f32]` 扁平握手（B1） | 新端口按形状/lane 泛型化；既有 `&[f32]` 握手保留兼容层，逐算子迁移 | 类型状态消除非法形状；避免一次性破坏契约 | 中 |
| R5 | 原始宽度/步数跨端口传递（B2） | 关键边界引入 newtype 消歧 | 编译期防参数错位 | 低 |
| R6 | 错误无 source 链（C1） | 评估为 `Error` 增加可选 `source: Option<Box<dyn Error>>` 或保留现有聚合模型并补 `source()`；不强行换 thiserror | 仓内错误模型统一且绑定退出码映射，替换面大 | 中 |
| R7 | 生产裸 unwrap/expect 偏高（C2） | 逐 crate 审计分类：不可达断言保留注释，可恢复路径改 `Result` | 降低 panic 面 | 中 |
| R8 | 阈值常数散落（F1） | 算子/页池/采样阈值外置为类型化配置，带默认值与启动校验 | 配置与业务分离 | 中 |
| R9 | 测试覆盖不均（G1） | 补 prof/prim/tokenizer/tensor 单测；核心算子接 proptest/fuzz 契约 | 不变式钉住 | 低 |
| R10 | 无 profiling 闭环（E3） | 建 criterion/iai 基准 + 回归门禁；热点先采样后优化 | 度量驱动，禁止直觉微优化 | 低 |

**不建议本轮做的事**：引入 async/运行时线程池（D2，无并发需求证据）；上 rayon/SIMD（E4，未采样）；把统一 `diagnostic::Error` 整体换成 thiserror/anyhow（C6，破坏退出码契约）；为单实现端口强行抽 trait（杜绝过度设计）。

## 4. 改动方案、风险等级与回滚方式

按依赖先后排序；每刀独立可回滚，门禁全绿再进下一刀。

| 刀次 | 内容 | 依据 | 风险 | 回滚方式 | 验收 |
|---|---|---|---|---|---|
| P1 | L7 命令编排下沉续刀（R1）：解析/装载/排版按 §3 归位；不放宽 cli 既有裁决前只沉零依赖项 | A1、#62–#64 已验证路径 | 低 | 单刀 git 回退；纯函数迁移，无接口变更 | workspace 测试/clippy/fmt/arch/contracts/gate 全绿；main.rs 行数下降 |
| P2 | `assembly.rs` 拆子模块（R2） | A2 | 低 | 模块拆分回退到单文件；私有项迁移不影响公开 API | 编译与 serve 测试不变；文件行数回落限内 |
| P3 | 端口类型化与 newtype（R4/R5） | B1/B2 | 中 | 保留 `&[f32]` 兼容握手，新泛型端口可_feature-gated 回退 | 契约对拍测试通过；无非法形状进入热路径 |
| P4 | 错误 source 链与裸 panic 收口（R6/R7） | C1/C2 | 中 | source 字段可选新增；panic 改 Result 的逐处回退 | 错误边界测试通过；生产 src 裸 panic 计数下降 |
| P5 | 配置外置（R8） | F1/F2 | 中 | 配置键表回退到源码常数；默认值兜底 | 启动校验测试；既有行为不变 |
| P6 | 测试与基准补强（R9/R10） | G1/E2/E3 | 低 | 测试/基准为增量，不影响生产 | 覆盖率/基准回归门禁纳入 CI |
| P7 | 文档/供应链收口（G2/G3） | G2/G3 | 低 | 文档与 deny 配置增量 | clippy missing-docs 收敛；cargo deny check 干净 |

风险等级定义：**低** = 局部重构/增量测试，门禁可兜底，无公开接口变更；**中** = 触碰 trait 签名、依赖边或错误模型，需兼容层与对拍；**高** = 不兼容变更/unsafe/跨设备 FFI，必须 ADR + 白名单 + 对拍。当前清单无高风险项；Metal/CUDA 落地时另立高风险刀次。

## 5. 九阶段就绪度矩阵

| 阶段 | 现状 | 就绪度 | 主要缺口 |
|---|---|---|---|
| 二、架构/设计模式 | 35 端口 trait、单向依赖、架构守护测试、39 ADR | 较好 | L7 编排超重；装配文件待拆；端口抽象度参差 |
| 三、类型系统 | 算子泛型骨架、密封 trait、newtype 局部 | 中 | 端口操作数未类型化；宽度/步数缺 newtype |
| 四、宏/编译期 | annotation buildgen、kernel! 宏、闭集编译期锁定 | 中 | 横切样板重复识别不足 |
| 五、配置化 | cli::settings 目录键表、buildgen TOML 闭集 | 中 | 阈值/页池/采样常数仍硬编码；缺统一运行期配置机制 |
| 六、性能 | Bounds 语义下界对账、18 criterion 基准 | 中 | 缺 profiling 闭环、整模型基准、缓存敏感测量；热路径分配待查 |
| 七、并发/异步 | 同步模型、Mutex 串行边界、共享面极小 | 初段 | 无 async/背压/线程池隔离设计（按需再引入） |
| 八、错误/安全/可观测 | 统一 Code+Error 聚合模型、unsafe 唯一岛、deny unsafe | 中 | 无 source 链；生产裸 panic 待分类；无 tracing/metrics |
| 九、质量/规范/供应链 | workspace lint、criterion、deny.toml、CHANGELOG、edition/MSRV 锁定 | 较好 | 覆盖率门槛、proptest/fuzz 系统接入、missing-docs 升 deny、CI 基准回归 |

## 6. 待决策项（需对齐后执行）

- **D1（cli 裁决）**：是否放宽"cli 只搬字符串、不解析数值"的既有注释/ADR，把 `parse_optional_*`/`parse_input` 下沉 cli；维持现状则这些函数留根入口 debt 条目。
- **D2（排版归属）**：`floor_row`/`device_images`/`report_*` 行格式化，tool(L1) 不能向上依赖 nn/prim；需决定沉对应业务层还是留命令层。
- **D3（错误模型）**：是否为 `diagnostic::Error` 补可选 source 链（中风险），还是维持现有聚合模型、仅补 `source()` 空实现文档。
- **D4（基准深度）**：本轮是否引入 iai 缓存敏感测量与 proptest，还是先扩充 criterion 整模型基准。
- **D5（提交粒度）**：当前在途 WIP 未提交；后续每刀是否单独成提交，还是按阶段汇总提交。

> 下一步：确认 §6 决策与 §4 刀次顺序后，从 P1（L7 续刀）进入第二阶段实施；每刀先跑基线门禁，改动后复跑全部门禁，不留编译失败或半成品。

## 7. 实施回写：P1–P7 核对结论（2026-11-17）

> 本节记录第一阶段落地后的核对结论。决策口径按保守默认推进（用户未逐条回答 D1–D5）：D1 cli 维持"只搬字符串、不解析数值"既有裁决；D4 基准先扩 criterion，本轮不引 iai/proptest；D5 改动留工作树，不擅自 commit/stage。业务行为与对外接口契约不变。

### 7.1 刀次结论

| 刀次 | 结论 | 落地改动 | 公共契约 |
|---|---|---|---|
| P1 L7 拆分 | **已完成，实质改动** | `src/main.rs` 837→78 行；拆 `status.rs`（命令装配/汇报 + 4 测试）、`run.rs`（前向命令 + 输入解析 + 2 测试）、`train.rs`（配置解析与汇报）；`main.rs` 只留 main/分派/退出码。守护测试 `the_entry_point_defines_no_types` 禁止 L7 定义新类型，拆分后仍通过 | 不变 |
| P2 serve 拆分 | **已核对，无新代码** | 在途 `plan.rs`（335 行，计划/族别分流/校验）、`block.rs`（236 行，注意力块 q/k/v/o）已被 `lib.rs` 声明且编译通过；`assembly.rs` 855→312 行，仅剩编号分区常数与装配/接线，内聚清晰；plan/block 内类型均 `pub(crate)`，不进公共所有权表 | 不变 |
| P3 类型系统 | **评审结论：现状到位，不改代码** | 无 downcast/类型擦除滥用——`dyn Any` downcast 仅在 `fuzz/contract.rs` 提取 panic 消息的固有场景；泛型 trait 仅 `data::Dataset`，其余用关联类型或 f32 握手。`Program`/`Predict`/`Sampling` 的扁平 f32 握手保留；形状/lane 收进类型属中风险专项（需全 Program 实现迁移 + 契约对拍），本轮不动。`nn::Linear::new(in, out)` 两相邻 usize 存在传反可能，但改 newtype 到 pub 签名属不兼容变更，保留 | 不变 |
| P4 错误/安全 | **已收口一处文档化，其余维持** | 保留自研 `diagnostic::Error` 值类型（derive Debug/Clone/PartialEq/Eq），不换 thiserror/anyhow，不引入 `Box<dyn Error>` source 链（会破 Eq/Clone）；`error.rs` 补显式 `source() -> None` + 取舍注释；`from_io` 已按 ErrorKind 归类（NotFound→Missing，否则 Io）。生产裸 panic 为零（见 C2 修正）。唯一 unsafe 岛 `storage/mmap.rs` 本轮不动。tracing/metrics 按 ADR 0039 依赖策略与"外部依赖极薄"刻意决策，本轮不引入 | 不变 |
| P5 配置收口 | **评审结论 + 一处单点收口** | 常数归类：编号分区基址、算子粒度、缓存 TOUCH/LOCK_WINDOW 属算法与编号不变式及 cutile 内部策略（ADR 0023 cutile 不读 env），留源码；采样温度/top-k/top-p、train 超参已是类型化档案 + `with_overrides` + `Default`；无跨文件散落重复魔法数（4096 仅出现在文档示例）。落地：`serve/generate.rs` 温度默认 `unwrap_or(1.0)` 提为具名常量 `DEFAULT_TEMPERATURE` | 不变 |
| P6 测试/基准 | **评审结论：覆盖充足，不堆测试** | 按测试数纠正 G1 口径：tensor 26、tokenizer 36、prof 32、prim 12 + device_tests；覆盖标量/行列主序/步幅/视图越界/溢出饱和/owned 分配/借读/签名/峰值公式/协商拒绝。基准 18 条 criterion：`model_decode` single/batch、`train` schedule/snapshot/buffers、`decode_attention` hist 128/512；`benches/tests/regress.rs` 基线按 os-arch/口径/版本登记，含 roofline read/write 双向对账。proptest/iai 本轮不引（新依赖需走 cargo-deny 白名单与 ADR）；变异测试列入后续专项 | 不变 |
| P7 文档/供应链 | **已完成核对与修复** | `cargo deny check`：advisories/bans/licenses/sources 全 ok；编译期 missing-docs 告警为零；`RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps` 全绿。本轮修复一批 rustdoc 内链：engine `Pager::share` 失效路径、tool/op/serve 冗余显式目标、backend/train 指向私有模块/条目的链接、tool `bench` 与 op `model` 同名歧义。CHANGELOG Unreleased 已补记本轮内部整理；edition 2024、MSRV 1.100 锁定 | 不变 |

### 7.2 门禁终验（P7 收尾复跑）

- `cargo test --workspace`：**1391 passed / 0 failed / 1 ignored**。
- `cargo clippy --workspace --all-targets --all-features -- -D warnings`：零告警。
- `cargo fmt --all --check`：0 diff。
- `cargo test -p llm --test architecture --test contracts`：arch 8、contracts 6 全绿；`cargo test -p llm-tool --test gate`：1 绿。
- rustdoc `-D warnings`：全绿；`cargo deny check`：四项全 ok。
- 已知现象：gate 在 workspace 并行跑时曾误报 FAILED；单跑稳定通过，判定为并行中间态/竞争，非真实违例。

### 7.3 未动工与后续专项（不在本轮干净增量内）

- **#56 Metal（本机 M5 Pro / Metal 4）/ CUDA（win 远端 T600 sm_75 / CUDA 13.3）挂载与跨后端契约对拍**：边界 `llm-backend` 占位骨架已从工作树移除，树内只保留运行时一侧 `Executor`/`Allocator`/`Collective`/`Program` trait 与 CPU/进程内实现；真实命令队列/设备缓冲映射/kernel launch/事件同步未接。口径：设备内核管线未建前只做编译 + 契约/形状对拍，不跑真内核、不出设备数字、不引 FFI crate、不写 unsafe；硬件两侧均具备，剩余前置是 ADR-0035 评审、FFI 策略与跨机构建打通。
- **#57 量化 lane 适配与通信原型**：u8/i4/i8 lane 的 Program 适配与反量化内核挂载（参照 `train/src/kernel.rs` u8 View 先例）；`shard::Collective` 通信后端原型；拓扑簿记不改。
- **后续专项**：Program 形状/lane 泛型化（P3，中风险）；profiling 闭环（pprof/flamegraph）、proptest 属性测试、变异测试（P6）；tracing/metrics 可观测接入（P4，需先定依赖策略 ADR）。
- **待用户拍板**：D1 cli 数值解析是否放宽下沉；D2 排版函数归属；D3 错误是否补 source 链（本轮选保留值类型模型）；D4 是否引 iai/proptest；D5 提交粒度与未跟踪分析文档、`report/` 目录归属。
