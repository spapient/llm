# 架构契约

这份文档回答三个问题：**谁属于哪一层、允许依赖谁、每个类型归谁**。
它是评审用的检查表：改动依赖方向或新增成员时，逐条核对下面所有约定。

`crates/cutile/DESIGN.md` 描述的是 kernel 层（`cutile` + `op` / `prof` / `prim`）的内部机制；
本文描述的是**整仓的模块划分与依赖方向**。两者不重叠：改 DSL 机制看 DESIGN，加模块看本文。

## 一、分层

| 层 | 成员 | 职责 | 不允许出现 |
|---|---|---|---|
| L1 基础设施 | `diagnostic`、`testkit`、`llm-config`、`tool`、`benches` | 错误类型、测试夹具、配置分层与路径、人可读输出、基准判定 | 任何领域语义 |
| L2 数据层 | `einsum`、`prof`、`op`、`prim` | 求和表达式的语义代价（flops/流量）、设备档案与能力闭集、算子声明与词汇表闭集（布局/调度/模板词法）、原语片段 | 任何机制代码（拼接、协商、emit）、任何设备知识 |
| L3 内核 | `cutile`、`audit` | `cutile` 做机制（协商 → 拼接 → emit → 编译执行），`audit` 在其上做命令策略（doctor/verify/bench 的结构化报告与见证执行）；两者只消费 L2 的数据 | 设备厂商特判、运行时依赖 |
| L4 运行时 | `mem`、`tensor`、`storage`、`exec`、`tokenizer` | 分配、张量元数据、布局搬运、提交执行、文本边界 | 模型语义、调度策略 |
| L5 模型与训练 | `weights`、`kv`、`autograd`、`comm`、`proto`、`data`、`nn`、`shard` | 权重结构、KV 页、反向图、集合通信、线协议、样本、层定义、切分拓扑 | 进程/服务语义 |
| L6 调度与服务 | `checkpoint`、`train`、`engine`、`serve`、`cli` | 检查点格式、训练循环状态、连续批与序列、端点契约、命令行解析 | 直接调用 `cutile` 生成 kernel；执行任何业务 |
| L7 入口 | `llm`（bin：`llm` 与 `cutile`） | 组装：把上面各层连起来，**不定义新类型**；`cutile` 二进制只做解析 → `audit` → 排版 → 退出码（`Code::exit_code`） | 任何新类型、任何业务逻辑 |

`annotation`（编译期注解）与 `fuzz`（模糊测试契约）是**工具成员**：它们不属于运行时分层，
也不参与“至少一条依赖边”的统计。它们的**依赖边不参与方向判定**——工具没有运行时层级。

## 二、依赖规则（唯一一条，但严格执行）

> **依赖只能指向更低的层；同层内只能依赖表中更靠左的成员，不允许反向。**

于是：

- 向上依赖不可能通过：`op` 无法依赖 `cutile`，`tensor` 无法依赖 `engine`。
- 同层内也可以有先后：`prim` → `prof`，因为片段文件名必须落在原语闭集内。
- 环一定被拒绝：层与表内顺序构成偏序，严格递减无法成环，不需要额外的环检测。
- 数据层永远不知道机制：`einsum` / `op` / `prof` / `prim` 的依赖只有彼此与 L1。
- 语义先于设备：`op` 只写"一条求和表达式 + 操作数位宽"，`flops` / `bytes` 由 `einsum` 推出；
  `cutile` 的 roofline 只消费这个数对，不认识任何算子名。

允许的边（当前全量，共 83 条，含 4 条 `+dev` 边；工具成员 `annotation` / `fuzz` 的边不参与运行时方向判定）：

```
cutile   → diagnostic, llm-config, op, prim, prof    +dev: testkit
audit    → cutile, diagnostic, llm-config, op, prim, prof
einsum   → diagnostic
op       → einsum, diagnostic, llm-config, prof
prim     → prof
prof     → llm-config
mem      → diagnostic
tensor   → mem, diagnostic
storage  → tensor, diagnostic
exec     → diagnostic
tokenizer→ diagnostic
weights  → tensor, diagnostic
kv       → tensor, diagnostic
autograd → tensor, diagnostic
comm     → diagnostic
proto    → diagnostic
data     → tensor, diagnostic
nn       → tensor, op, diagnostic, annotation
shard    → comm, diagnostic
benches  → diagnostic
checkpoint → diagnostic, tensor, weights
train    → checkpoint, nn, autograd, data, weights, shard, diagnostic   +dev: tensor
engine   → kv, exec, tensor, diagnostic
serve    → engine, proto, tokenizer, diagnostic            +dev: exec, kv
cli      → diagnostic
fuzz     → diagnostic（工具成员，不参与方向判定）
llm      → audit, cli, cutile, diagnostic, engine, exec, kv, llm-config, nn, op,
           prof, serve, tensor, tokenizer, tool, train, weights
tool     → （叶：无出边，被 llm 消费）
testkit  → （叶：无出边，被 cutile 的 dev 依赖消费）
diagnostic / llm-config / annotation → （叶：被大量成员消费）
```

## 三、所有权表（一个类型只能有一个主人）

| 概念 | 唯一主人 | 说明 |
|---|---|---|
| `Error` / `Code` / `Result` | `diagnostic` | 全仓库唯一的错误类型；`op` / `cutile` / `audit` 三层边界都用它，`prof` / `prim` 仍是存量（在边界处用 `Error::new` 显式给类别）。`Code::exit_code` 是类别到 shell 退出码的唯一映射（2/3/4/5/70） |
| `Settings` / `Key` / `Roots` / `App` | `llm-config` | 配置分层、路径解析与**应用身份**；其它 crate 用 `Settings::<自己的 App>::load()` 而不是复制分层逻辑 |
| `Expr` / `Extents` / `Operand` / `Axes` / `Bits` / `Cost` | `einsum` | 求和表达式的语义：一条式子推出 `flops` 与最少流量；不认识设备、tile 或片段 |
| `Op` / 模板文本 / `Signature` / `Sample` / `Model` | `op` | 算子声明、模板与算子语义（签名、样本、标量参考）；签名是 ABI 的唯一事实源，实参顺序与输出名由它派生；代价由 `einsum` 推出，`Cost` 只是再导出 |
| `OpDecl` / `Segment` / `LayoutSpec` / `Schedule` | `op` | 声明解析的结果与三张闭集（模板词法、布局构造子、调度模板）；`cutile` 只消费，不再持有第二份 |
| `Profile` / 能力词汇表 | `prof` | 设备档案与闭集；机制代码不得出现 `vendor == "..."` |
| `FragmentSet` | `prim` | 原语片段与见证源码 |
| `Catalogue` | `cutile` | 已装载的三层数据（声明 + 档案 + 片段）与它们的交叉校验；装载要随数据格式演进，所以不属于任何纯数据成员 |
| `Plan` / `Path` / `NoPath` / `VariantCheck` / `Source` / `Usage` | `cutile` | kernel 层的中间产物，全部可见、无 IR；只有 `NoPath` 自带文本（`Display`，错误契约）。`Usage` 是构建期存储契约比较过的一条声明用量 |
| `CacheKey` / `Cache` / `Entry` / `State` / `Changed` / `Residue` / `Gc` / `ArtifactKey` / `Artifact` | `cutile` | 缓存指纹、编译产物与存量：`CacheKey::fields` 是哈希、`meta.txt` 与差异比较的唯一来源；`ArtifactKey` 是产物身份（源文本 + 真正调用的编译器 + 主函数），`Artifact` 是命中结果（路径、字节、是否复用）；`Entry`/`State` 让"有 `source`"与"只写了一半"都成为可扫描的记录，`Residue`/`Gc` 是回收的结构化结果 |
| `Doctor` / `OpInfo` / `DeviceInfo` / `LayoutRow` / `Storage` / `Declared` / `Built` / `CacheWhy` / `CacheState` | `audit` | `doctor` 的结构化结论：每步产物都是数据，排版由前端拼；`Storage` 逐层给出档案容量与 `Declared` 用量，`CacheWhy` 答“这个键与本次构建差在哪”，`CacheState` 把"从未落盘 / 半截落盘 / 一致 / 不同"分成四类（文本在前端） |
| `VerifyReport` / `VerifyEntry` / `Verified` / `WitnessRun` | `audit` | `verify` 的逐设备结果与见证执行；失败如实记录，不写成“跳过” |
| `Bench` / `Throughput` / `Peak` / `Verdict` | `audit` | `bench` 的实测口径与达成率判定；门槛常数（`L3_THRESHOLD`）也在这里 |
| `Arena` / `Region` | `mem` | 分配；不定义布局 |
| `Tensor` / `View` / `Element` / `DType` / `Shape` / `Strides` | `tensor` | `Tensor` 是**字节**视图，`View<E>` 是**元素**零拷贝视图（不分配、不装 `unsafe`） |
| `Layout` / `Order` / `remap` | `storage` | 物理布局与搬运；不分配 |
| `Stream` / `Graph` / `Launch` | `exec` | 提交与执行；不编译 kernel |
| `Tokenizer` / `Token` | `tokenizer` | 文本 ↔ token 的唯一边界 |
| `Header` / `Entry` / `Reader` | `weights` | 权重文件结构；不持有张量 |
| `Format` / `Snapshot` | `checkpoint` | 检查点的文件头与恢复事实（步数 / 并行度 / 层数 / 批行数 + 权重条目表）；条目表本身归 `weights`，不复制 |
| `Page` / `Table` / `Cache` | `kv` | 分页 KV 的页管理 |
| `Tape` / `Var` / `Node` / `Requirement` | `autograd` | 反向图的记录：变量的需求（不进图 / 中间 / 终点）、边、深度 |
| `Schedule` / `Account` | `autograd` | 反向调度（可达剪枝 + 深度降序 + 同深度分层）与扇出账本（谁的梯度已齐全）；两者都在原对象上重填，训练循环不必每步新建 |
| `Plan` / `Budget` / `Choice` / `Footprint` / `Sizes` | `autograd` | 显存计划：预算内的保留集、释放点与重放步数。`Plan::of` 只读 `Tape`、`Schedule` 与一张形状表，不分配、不执行；与 `cutile::Plan`（单条 kernel 选哪个 tile）不是同一件事 |
| `Replay` / `Store` | `autograd` | 重放执行：把计划编成"第 `k` 步之前重算哪几步、之后交还哪些激活"的扁平序列表，并驱动整次反向。`Store` 是唯一的口子（`compute` / `free`），真实缓冲归 `train`，这一层不碰数值 |
| `Group` / `Reduce` / `Collective` | `comm` | 集合通信契约 |
| `Envelope` / `Kind` / `Version` | `proto` | 线协议帧格式 |
| `Batch` / `Split` / `Dataset` | `data` | 样本与切分 |
| `Layer` / `LayerSpec` / `Linear` | `nn` | 层定义；产出的算子名必须存在于 `op` 且有语义表 |
| `Bounds` | `nn` | 层的性能下界（流量 + 运算次数）；由算子语义推出，不认识设备峰值 |
| `Topology` / `Partition` | `shard` | TP/PP/DP 拓扑 |
| `Schedule` / `Step` / `Checkpoint` | `train` | 训练循环状态 |
| `Sequence` / `Scheduler` | `engine` | 连续批与序列状态 |
| `Route` / `Endpoint` / `Response` | `serve` | 端点契约 |
| `Command` / `Invocation` / `USAGE` | `cli` | 命令行解析；只分派，不执行任何业务 |
| `Report` / `join_names` | `tool` | 人可读输出；只接字符串，不接领域类型 |
| `layer` 属性宏 | `annotation` | 把注解展开成 `nn::Layer` 的实现 + 一条校验测试；**不持有运行期注册表** |
| `Case` / `Baseline` / `Judgement` | `benches` | 基准场景与回退判定；不做测量。名字刻意避开 `Verdict`，见下节第三条 |

三个容易混淆、但**不是重复**的东西：

- **kernel 侧操作数视图**（DSL 的宿主 API：`&mut` 分片即 `grid`，见 DESIGN §4.4）与
  `tensor::Tensor` 是两回事：前者不得依赖 `tensor`（内核层不能向上依赖运行时），后者也不得
  假装能做 kernel 分区。这个视图在 P0 **尚未实现**——`cutile` 只接收裸指针与标量，分片由
  调用方给；等 `llm` 真要直接调用 kernel 时再按 DESIGN §4.4 重建，见
  [`adr/0003-kernel-and-runtime-types.md`](adr/0003-kernel-and-runtime-types.md) 的修订段。
- `cutile::space::Space`（GM/FAST/REG）是 **语言概念**，`prof::atom::Space` 是它的档案表示；
  分配器（`mem`）只认字节，不认地址空间——地址空间由档案声明。
- `nn::LmHead` 这类**注解声明**与 `Linear::new` 这类**运行期构造**不是两套机制：前者把形状与
  算子写成属性、由 `#[layer]` 生成实现与校验测试，后者按配置计算宽度；两者最终都走
  `LayerSpec::new` 的同一条校验路径。注解不建注册表（注册表会成为第二个事实源），
  见 [`adr/0005-annotations-as-contracts.md`](adr/0005-annotations-as-contracts.md)。
- `audit::Verdict`（达成率口径：`reached` / `below` / `over-peak` / `cache-resident` / `unknown`）
  与 `benches::Judgement`（基准回退判定：`pass` / `regressed` / `improved`）**不是同一件事**：
  前者判定“这次测量能不能与档案峰值比较”，后者判定“与基线相比是否回退”。两者会同时出现在
  一份 CI 报告里，所以名字必须不同——`benches` 里的类型因此叫 `Judgement`。

## 四、占位策略

一个 crate 还没实现时，它占的是**接口形状**，不是 `add(2, 2)`：
类型、trait、字段名与错误语义先定，函数体后填。判断标准是"删掉它会谁编译失败"——
如果谁都不失败，它就该从 `members` 里移出去，而不是留在 workspace 里复制同一个占位函数。

规则见 [`adr/0004-placeholder-policy.md`](adr/0004-placeholder-policy.md)。

## 五、新增一个成员的三步

1. `crates/<name>/Cargo.toml`：`version/edition/rust-version/license/repository/readme` 全部
   `.workspace = true`，写 `description`、`publish = false`、`[lints] workspace = true`。
2. `src/lib.rs` 第一行起写 `//!` crate 文档，说明**拥有什么、不拥有什么、依赖谁、被谁依赖**。
3. 在第一节的层表里加上它，并让它至少有一条依赖边（被消费或消费别人），否则它只是一个目录，
   不是模块。

做完这三步，按同一口径自查：

- `Cargo.toml` 的成员列表与本文件第一节的层表一一对应；
- 每条依赖边都指向更低的层（同层只能依赖表中更靠左的成员，不允许反向）；
- 每个成员至少一条依赖边（工具成员豁免）；
- 全部成员 `publish = false`，`description` 与 `readme` 齐备；
- 仓库内只有一个 `target/`（workspace 根）。

## 六、这份契约怎么被守住

| 手段 | 位置 | 挡住什么 |
|---|---|---|
| `clippy -D warnings` | CI `clippy` job | 缺元数据（`cargo::cargo_common_metadata`）、缺文档、`unsafe` 风格 |
| `RUSTDOCFLAGS=-D warnings cargo doc` | CI `doc` job | 文档里的坏链接、未文档化的公开项 |
| `cargo test --workspace` | CI `test` job | 行为回归 |
| `tests/architecture.rs` | CI `test` job | 成员集合、边集合与方向、所有权表的 `pub` 定义与 `cargo metadata` / 源码不一致；以及 `op` / `cutile` / `audit` 是否又退回 `Result<_, String>` |

`cargo metadata` 只能告诉你"谁依赖谁"，不能告诉你"这个依赖方向对不对"——后者由本文的层表定义，
再由 `tests/architecture.rs` 执行：它读 `cargo metadata` 与源码，把第一节的成员表、第二节的边集
（含总数与 `+dev` 数）、第三节的所有权表逐条比对。新增成员或调整依赖方向时，改完文档再跑
`cargo test`——不一致会直接失败，不再依赖评审清单。
