# 架构契约

这份文档回答三个问题：**谁属于哪一层、允许依赖谁、每个类型归谁**。
它是评审用的检查表：改动依赖方向或新增成员时，逐条核对下面所有约定。

`../crates/cutile/README.md` 描述的是 kernel 层（`cutile` + `op` / `prof` / `prim`）的内部机制；
本文描述的是**整仓的模块划分与依赖方向**。两者不重叠：改 DSL 机制看该 README，加模块看本文。

## 一、分层

| 层 | 成员 | 职责 | 不允许出现 |
|---|---|---|---|
| L1 基础设施 | `diagnostic`、`testkit`、`dtype`、`wire`、`tool` | 错误类型与声明数据树（`diagnostic::data`：构建期语料在运行期的形状，**没有解析器**）、测试夹具、精度闭集、与领域无关的字节码闭集（`wire`：CRC-32、LEB128 变长整数、计数/切片/文件 Sink 与 CRC 密封）、静态质量门与人可读输出、回退判定（`tool::regress`） | 任何运行时领域语义、任何 TOML 解析 |
| L2 数据层 | `prof`、`op`、`prim` | 设备档案与能力闭集（`devices/*.toml` 在构建期解析成表）、算子声明与词汇表闭集（`ops/*.toml` 同上）（布局/调度/模板词法）与求和表达式的语义代价（flops/流量，`op::einsum`）、原语片段 | 任何机制代码（拼接、协商、emit）、任何设备知识 |
| L3 内核 | `cutile` | 机制（协商 → 拼接 → emit → 编译执行）与其上的命令策略（doctor/verify/bench 的结构化报告与见证执行，按命令分成 `cutile::{bench, doctor, verify, execution}`，共用一条 `pipeline` 与 `witness`）；只消费 L2 的数据 | 设备厂商特判、运行时依赖 |
| L4 运行时 | `tensor`、`storage`、`tokenizer` | 张量元数据、布局搬运（含 feature-gated 只读内存映射，全仓唯一 unsafe 岛）、文本边界 | 模型语义、调度策略 |
| L5 模型与训练 | `weights`、`autograd`、`data`、`nn`、`shard` | 权重结构（含自有 LLPK 容器读写与外部格式只读适配）、反向图、样本、层定义、切分拓扑与集合通信 | 进程/服务语义 |
| L6 调度与服务 | `train`、`engine`、`serve`、`cli` | 训练循环状态与激活缓冲、检查点格式（`train::snapshot`）、连续批与序列、KV 页（`engine::kv`）、提交（`engine::exec`）、端点契约与其线格式（`serve::proto`）、命令行解析 | 直接调用 `cutile` 生成 kernel；执行任何业务 |
| L7 入口 | `llm`（bin：`llm` 与 `cutile`） | 组装：把上面各层连起来，**不定义新类型**；`cutile` 二进制只做解析 → `cutile::{doctor, verify, bench, execution}` → 排版 → 退出码（`Code::exit_code`） | 任何新类型、任何业务逻辑 |

`annotation`（编译期注解）、`fuzz`（模糊测试契约）与 `benches`（集中测量容器）是**工具成员**：
它们不属于运行时分层，也不参与“至少一条依赖边”的统计。它们的**依赖边不参与方向判定**——
工具没有运行时层级。

## 二、依赖规则（唯一一条，但严格执行）

> **依赖只能指向更低的层；同层内只能依赖表中更靠左的成员，不允许反向。**

于是：

- 向上依赖不可能通过：`op` 无法依赖 `cutile`，`tensor` 无法依赖 `engine`。
- 同层内也可以有先后：`prim` → `prof`，因为片段文件名必须落在原语闭集内。
- 环一定被拒绝：层与表内顺序构成偏序，严格递减无法成环，不需要额外的环检测。
- 数据层永远不知道机制：`op` / `prof` / `prim` 的依赖只有彼此与 L1。
- 语义先于设备：`op` 只写"一条求和表达式 + 操作数位宽"，`flops` / `bytes` 由 `op::einsum` 推出；
  `cutile` 的 roofline 只消费这个数对，不认识任何算子名。

允许的边（当前全量，共 90 条，含 27 条 `+dev` 边；工具成员 `annotation` / `fuzz` / `benches` 的边不参与运行时方向判定）：

```
cutile   → diagnostic, op, prim, prof    +dev: testkit
op       → annotation, diagnostic, prof    +dev: testkit（工具边：`#[model]` 注解，不参与方向判定）
annotation → dtype（工具边：变体名唯一事实源 `DType::variant`，不参与方向判定）
prim     → prof, diagnostic
prof     → diagnostic, dtype
dtype    → （叶：精度闭集，无运行期工作区依赖）    +dev: testkit
tensor   → diagnostic, dtype    +dev: testkit
storage  → tensor, diagnostic
tokenizer→ diagnostic
weights  → tensor, diagnostic, dtype, wire    +dev: storage, testkit
autograd → tensor, diagnostic    +dev: testkit
data     → tensor, diagnostic
nn       → tensor, op, weights, diagnostic, annotation
shard    → diagnostic
train    → nn, autograd, data, weights, shard, op, tensor, diagnostic, wire
engine   → tensor, diagnostic    +dev: testkit
serve    → engine, tokenizer, diagnostic    +dev: testkit
cli      → diagnostic
fuzz     → diagnostic    +dev: cli, weights, train, serve, wire, tensor（工具成员，不参与方向判定；`+dev` 的是 `drivers.rs` 里的六个驱动对象：`cli` 参数文法、`weights` JSON 与 safetensors、`train` 检查点、`serve` 帧解码、`wire` 接收器不变量，`tensor` 只为给检查点夹具写形状；`prof` 的档案入口随 ADR 0022 离开运行期，这条边随之删除）
benches  → +dev: prof, op, cutile, wire, weights, tokenizer, tool, testkit（工具成员：集中测量容器，不参与方向判定；预热、采样与测量时长由 criterion 配置，本成员只声明被测对象；`tool::regress` 判回退、`testkit` 登记"本机无基线"的跳过）
llm      → cli, cutile, diagnostic, engine, nn, op,
           prim, prof, serve, tensor, tokenizer, tool, train, weights
           +dev: autograd, testkit（`tests/cross_layer.rs` 打通权重、层声明与训练缓冲三段对拍：`autograd` 出 `Tape`/`Sizes`，`testkit` 出三档浮点口径）
tool     → diagnostic（配置驱动静态质量门 + 人可读排版 + 回退判定；阈值文件在 `tool` 的 `build.rs` 里解析成 `diagnostic::data` 的表）    +dev: testkit
testkit  → （叶：无出边，被 cutile / autograd / serve / engine / wire / op / dtype / tensor / tool / weights 的 dev 依赖消费）
wire     → diagnostic（纯字节码闭集：CRC-32、LEB128、Sink；不认识张量或容器布局）    +dev: testkit
diagnostic → （叶：错误类型 + `toml`，被大量成员消费；它自己的出边为零）
```

七条吸收链（`checkpoint→train`、`audit→cutile`、`regress→tool`、`exec→engine`、`kv→engine`、`einsum→op`、`proto→serve`）
**已落地**（ADR 0019）：成员从 33（含根包）降到 26，边从 106 条降到 82 条：消失的边是被吸收方的全部出边与全部入边，
新增的边只有 `cutile → annotation`（工具边）与 `serve → +dev: testkit`（`proto` 的 CRC 对拍随迁）。
方向合法性没有变化：七条链都是"同层向左"或"跨层向下"，因此 `tests/architecture.rs` 的方向断言一条都不需要放宽。

第八次是**拆分后删除**（ADR 0020 → 0021）：`llm-config` 没有单一消费者，不能像上面七条那样被某个 crate 整体吸收，
所以按"谁真的用得上"分成两半——设置层（分层、来源、键闭集、平台目录）当时只有一个生产消费者，先进了
`crates/cutile/src/settings/`（第十次又被上移到 `cli::settings`，见下）；TOML 语法层有六个跨层消费者，进 `diagnostic::toml`（该模块在第九次 ADR 0022 即被删除，现不存在，构建期共享内核现为 `diagnostic::buildgen`；每个读 TOML 的成员本来就已依赖
`diagnostic`，因此这条边不新增、只消失）。成员 26 → **25**，边 82 → **76**，`+dev` 13 → **12**；消失的六条边是
`op` / `prof` / `tool` / `cutile` / `llm` / `fuzz` 各自指向语法层的那一条。

第九次是**解析器退场**（ADR 0022）：那六个消费者真正需要的不是"能解析 TOML"，而是"已经解析好的那棵树"。
于是 `ops/`、`devices/`、`lints/vocabulary.toml` 改由 `op` / `prof` / `tool` 各自的 `build.rs` 在构建期解析
（`toml` 只作 `[build-dependencies]`），生成 `diagnostic::data::Table` 字面量；`cutile` 的运行期配置文件层
（内置默认 < 环境变量 < `--set` 之外的第四层）连同它的 `schema` 校验、文件相对路径基准、未知键校验一起删除。
585 行解析器与 8 条复杂度基线消失，成员数与边数不变（解析器原本就在 `diagnostic` 里）；唯一减少的边是
`fuzz → +dev: prof`——它的 TOML 目标没有入口可灌了。

第十次是**策略上移**（ADR 0023）：删掉第四层之后，`cutile/src/settings/` 只剩"默认值 + `CUTILE_*` +
`--set`"，而这三件事没有一件是内核机制该决定的——默认目录是**部署策略**，`--set` 是**命令行文法**。
于是 924 行的分层机制缩成 `cutile::dirs::Dirs`（三个路径的纯数据，64 行），解析、来源追踪与
`--set` 的 argv 扫描整体搬到 `cli::settings`（`cli` 本来就是"argv + 环境 → 强类型调用参数"的那一层）。
成员数与边数**一条未增**：`cli` 仍只依赖 `diagnostic`（键表不再借 `prof::atom::Closed`，三个键不值得
拉一套闭集机制），`cutile` 的五个出边原样；两个二进制各写一个 `kernel_dirs` 投影函数把 `Settings`
变成 `Dirs`，因此"配置策略"与"内核输入"之间只剩一次显式转换。

## 三、所有权表（一个类型只能有一个主人）

这张表由 `tests/architecture.rs` **双向**校验：表里写的必须在 `crates/<主人>/src` 里有 `pub` 定义；
反方向也成立——每个成员源码里的 `pub struct|enum|trait` 必须在表里，或登记在
`lints/vocabulary.toml` 的 `[[ownership.unlisted]]` 存量基线，且基线不允许过期（某个名字进了表
或不再是 `pub`，对应条目必须删除）。L7 组装点另有单独一条：`src/**/*.rs` 不得定义任何类型。

| 概念 | 唯一主人 | 说明 |
|---|---|---|
| `Error` / `Code` / `Result` | `diagnostic` | 全仓库唯一的错误类型；同一 crate 的 `data` 子模块（ADR 0022）还拥有 `Table` / `Value`——它们是**声明数据树**，由构建期生成代码构造，运行期没有解析器，因此也不存在"语法错误"这一类；`op` / `prof` / `prim` / `cutile`（含 `report`）用同一份它，类别由**产生错误的那一层**给（`ProfileError::code`、`prim` 的装载诊断），上层只补上下文。`Error::also` 把同一次校验里的第二条失败追加为同级诊断（各自带类别，不再被 join 成一行 `invalid`）；`Error::from_io` 是"路径不存在 `missing` / 存在但读不出 `io`"的唯一判定；`Code::exit_code` 是类别到 shell 退出码的唯一映射（2/3/4/5/70） |
| `DType` / `QType` / `MxType` / `Element` | `dtype` | 全仓唯一的精度事实源，三个格式轴分文件（`Element` 是 MX 的子闭集）：`dtype.rs` 的 `DType` 是标量闭集（24 个变体，Bool、U1–U64、I1–I64、F8 两种、F4、F16、BF16、F32、F64、`F8E4M3FNUZ`/`F8E5M2FNUZ`），稳定数字 code 与 LLPK 容器逐位兼容（前 21 个与旧 Wire 同码，F4 追加 22、FNUZ 追加 23/24）；`qtype.rs` 的 `QType` 是 GGML 块量化（25 个格式，传统 Q、K-quant、IQ 查表、TQ 三元四族；`Scale` 记块内 scale 前缀形状 f16/双 f16/f32/sub-scale，`Family` 记编码族，二者正交，`q8_K` 的 f32 d 与 `iq4_nl` 的单 d 各按源码如实分类）；`mxtype.rs` 的 `MxType` 是微缩放（OCP MX 六个标准格式 + NVFP4 厂商扩展，块共享纯指数 scale），同文件的 `Element` 是 MX 块内元素格式（E2M1/E2M3/E3M2/E4M3/E5M2/INT8，6 个，只回答位宽与标量对应）。这些格式只放身份与布局元数据，反量化算法留在 `weights`；除 `parse_name`（转调 trait 默认实现）外 inherent 访问器全部 `const fn`，零运行时开销，同一批读法再由 `Format` / `BlockFormat` 提供一次泛型视角（stable 无 const trait：几何的唯一实现是 inherent `const fn`，trait 只转发；`parse_name` 反过来由 trait 默认实现唯一提供），`Layout<B>` 是「标量元素或块格式」的唯一几何（`weights` 的 GGUF 支持表用 `Layout<QType>`，接 MXFP4 时换 `Layout<MxType>`）。各层以再导出（`tensor`、`weights` 的 `Wire`）、trait 或子集表表达局部能力，不再手写跨层映射 |
| `Dirs` | `cutile` | 内核要的三个目录（缓存根、工作目录、可选数据根）——**纯数据**：不带默认值、不读环境变量、不认识命令行。地址是调用方的策略，因此 `cutile` 只依赖 `std::path`，换一个嵌入方（测试给临时目录、服务给容器卷）不用改内核一行（ADR 0023） |
| `Settings` / `Roots` | `cli` | 三个目录开关的**解析策略**（`crates/cli/src/settings.rs`，ADR 0023）：内置默认 < `CUTILE_*` < `--set`、来源追踪（`值 ← 来源`）、平台默认目录与相对路径基准。键表是一行一个的声明宏（`cache.dir` / `work.dir` / `data.root`，含 `CUTILE_CACHE` 兼容名），`describe` 与"键未登记"的清单都从同一张表派生；**没有配置文件**（ADR 0022）。它曾经是独立成员 `llm-config` 的设置层，并入 `cutile` 后又上移到本层：默认目录是部署策略、`--set` 是命令行文法，都不是内核的职责，上移后成员数与依赖边一条未增——`cli` 仍只依赖 `diagnostic`，`cutile` 仍只收纯数据 |
| `Expr` / `Extents` / `Operand` / `Axes` / `Bits` / `Cost` / `Buffer` | `op` | 求和表达式的语义：一条式子推出 `flops` 与最少流量；`Buffer` 是内核样本的具名缓冲（形参名 + 取值）；不认识设备、tile 或片段 |
| `Op` / 模板文本 / `Signature` / `Sample` / `Model` | `op` | 算子声明、模板与算子语义（签名、样本、标量参考），`REGISTRY` 现有 22 条注册（GEMV 与 GEMM 各三个格式、逐元素与归约、形态算子、注意力两条、解量化两条），其中 GEMV 三条共用一份泛型骨架、只差格式契约；签名是 ABI 的唯一事实源，实参顺序、输出名与输出类型都由它派生（`output_type` 与 `Model::dtype` 交叉核对，回读只认 `float`）；代价由 `op::einsum` 推出，`Cost` 只是再导出；`bind` 是真实数据进入样本的唯一入口（只换 `x` / `w`，量化缓冲明确拒绝） |
| `Format` / `OperandSpec` / `Element` / `FormulaCache` | `op` | GEMV 骨架的格式契约：`Format` 把"随精度变化"的部分（权重元素、位宽、粒度、派生轴、标度、解包循环）收成关联常量与方法，`OperandSpec` 是表达式的操作数表（`const` 数组，装载时再转 `Operand`），`Element` 是样本缓冲的元素闭集（`f32` / `u8`，让 `reference` 不必按变体分发），`FormulaCache` 是四条骨架（GEMV / 逐元素 / 一元 / 归约）共用的唯一一份表达式缓存实现，单元格仍由每个具体格式用 `static` 持有（泛型作用域里的 `static` 只有一份，共用会让两种精度拿到错误的轴表），`Format::CHECK` 是格式对自身声明的自证（粒度 > 0、派生轴除数就是粒度——两处都拿同一个常数算分组，否则代价与样本是两种语义），由 `#[model]` 生成物在别名上求值；`Quantized<G>` / `Dense<B>` 的 const 参数分别承载声明粒度与累加块宽，`cost` / `sample` / `reference` 按格式单态化，`Model` 里仍是三个 `fn` 指针 |
| `OpDecl` / `Template` / `Segment` / `Registration` / `LayoutSpec` / `Schedule` / `Masking` | `op` | 声明解析的结果与三张闭集（模板词法、布局构造子、调度模板），以及唯一的算子注册表 `REGISTRY`（名字 / 声明 / 模板 / 语义表同属一条注册，`BUILTIN` 与 `model()` 都是它的投影）；`cutile` 只消费，不再持有第二份 |
| `Profile` / 能力词汇表 | `prof` | 设备档案与闭集；机制代码不得出现 `vendor == "..."` |
| `Closed` / `Set` | `prof` | 词汇表契约：`Closed` 要求每个闭集只声明 `ALL`（顺序即 `index`）、`as_str` 与 `index`，`parse` 由契约派生；`Set<E, WORDS>` 是全仓唯一的位集（`WORDS` 是 const 参数，容量在单态化时由 `const FITS` 断言，写宽或写窄是编译错误），`DTypeSet` / `SpaceSet` 只是它的别名；`inherent_names!` 把 `ALL` / `as_str` / `parse` 以**同名固有项**转发，于是调用方不必 `use` 一个 trait，而名字表仍然只有一份 |
| `FragmentSet` / `Device` / `Image` / `Status` | `prim` | 原语片段与见证源码；`Device` 是每台设备这次构建带上的设备侧代码（原语枚举 + 入口映射表 + 可选镜像），`Image` 是编译产物、`Status` 是它的结局。三张内嵌表（`BUILTIN` / `WITNESS` / `DEVICES`）由 `build.rs` 扫描 `<vendor>/` 生成，`src/` 里只有类型 |
| `Catalogue` | `cutile` | 已装载的三层数据（声明 + 档案 + 片段）与它们的交叉校验；装载要随数据格式演进，所以不属于任何纯数据成员 |
| `Plan` / `Path` / `NoPath` / `VariantCheck` / `Proof` / `Source` / `Usage` | `cutile` | kernel 层的中间产物，全部可见、无 IR；只有 `NoPath` 自带文本（`Display`，错误契约）。`Usage` 是构建期存储契约比较过的一条声明用量。`Proof` 是这次真正执行过的供给（`(原语, 实现名)` 集合）：协商只有在"档案写了 `L2`"与"这次跑通了见证"同时成立时才认一条能力，`Proof::declared` 只给解释性报告用 |
| `Toolchain` / `CacheKey` / `Cache` / `Entry` / `State` / `Changed` / `Residue` / `Gc` / `ArtifactKey` / `Artifact` | `cutile` | 缓存指纹、编译产物与存量：`Toolchain` 是实测工具链（`命令@版本`，来自 `--version`；档案里的 `[meta].tool_ver` 只是登记值），`CacheKey::fields` 是哈希、`meta.txt` 与差异比较的唯一来源且**不含工具链**（源文本与编译器无关）；`ArtifactKey` 是产物身份（源文本 + 实测工具链身份 + 编译参数指纹 + 主函数；参数一变就换一份产物，见证因此不会拿旧参数的二进制替新参数作证），`Artifact` 是命中结果（路径、字节、是否复用）；`Entry`/`State` 让"有 `source`"与"只写了一半"都成为可扫描的记录，`Residue`/`Gc` 是回收的结构化结果；每份产物旁的 `bin/<hex>.key` 记着它的身份（谁编的、拼了哪个主函数），`Cache::artifacts_in` 靠它在没有编译器时复用产物 |
| `Doctor` / `OpInfo` / `DeviceInfo` / `LayoutRow` / `Storage` / `Declared` / `Built` / `CacheWhy` / `CacheState` | `cutile` | `doctor` 的结构化结论：每步产物都是数据，排版由前端拼；`Storage` 逐层给出档案容量与 `Declared` 用量，`CacheWhy` 答“这个键与本次构建差在哪”，`CacheState` 把"从未落盘 / 半截落盘 / 一致 / 不同"分成四类（文本在前端） |
| `VerifyReport` / `VerifyEntry` / `Verified` / `Execution` / `WitnessRun` / `Evidence` / `Basis` | `cutile` | `verify` 的逐设备结果与见证执行；失败如实记录，不写成“跳过”。`Evidence` 是三条命令（`doctor` / `verify` / `bench`）共用的那一份证据：设备执行与自检各几条、依据是 `Executed` 还是 `Declared`、哪一步不成立；`Basis` 说清"这份结论来自执行还是档案自述"；`Execution` 是 `execute` 的结论：用调用方给的样本（真实权重）走同一条流水线，带回内核输出与标量参考 |
| `Negotiation` | `cutile` | 一台设备的协商结论（首档 / 降级 / 无路径 / 无档案）：只读声明与档案，与执行成败分开记；文本在前端 |
| `Fragment` | `cutile` | 一份生成片段的行数与 §16 口径（`FRAGMENT_LIMIT` / `FRAGMENT_GOAL`）：上限管成败、目标只是余量，`within_limit` 就是判据本身 |
| `Bench` / `Throughput` / `Peak` / `Verdict` / `Performance` | `cutile` | `bench` 的实测口径与达成率判定；门槛常数（`L3_THRESHOLD`）也在这里。`Performance` 把达成率与它的尺子绑在一起：口径无效（超过 100%、工作集装得进 `FAST`）或没有峰值时 `is_measured` 为假，`needs_specialisation` 就是 §16 的“某设备 < 70%” |
| `Tensor` / `View` / `Element` / `Shape` / `Strides` / `OwnedTensor` | `tensor` | `Tensor` 是**字节**视图，`View<E>` 是**元素**零拷贝视图（不分配、不装 `unsafe`）；`SUPPORTED` 子集表声明本层可构造的五种精度；`OwnedTensor` 是分配出来、可写入的拥有形态（元数据 + 持有字节向量），与只读 `Tensor` 视图互补，设备地址不泄漏到本层 |
| `Layout` / `Order` / `remap` / `Buffer` / `Allocator` / `CpuAllocator` | `storage` | 物理布局与搬运；`Buffer` 是拥有的扁平字节缓冲（设备句柄不泄漏到本层），`Allocator` 是分配/释放/显式拷贝的边界 trait，`CpuAllocator` 是安全 `Vec<u8>` 实现，设备后端在边界 crate 实现同一 trait；本层不做隐式跨设备拷贝 |
| `Stream` / `Graph` / `Launch` / `Device` | `engine` | 提交与执行；不编译 kernel。`Device` 是运行时设备标识标签（厂商 / 架构字符串），不持私有句柄；与 `prim::Device`（编译期设备侧代码镜像）是两个类型，路径隔离 |
| `Tokenizer` / `Token` / `Bytes` / `Bpe` / `SpecialTokens` / `Encoding` / `Strategy` / `Truncation` / `Padding` / `PadDirection` / `Fingerprint` / `DecoderStream` / `Vocab` / `VocabSource` / `VocabFamily` / `Merge` | `tokenizer` | 文本 ↔ token 的唯一边界：编号、编码对齐记录（编号/字节区间/特殊标记等长）、序列策略配置、词表身份指纹与 UTF-8 边界安全的流式解码；`Vocab` 是格式无关的中性词表值对象（双向查表、特殊编号范围与族附属表在构造期闭合），`VocabSource` 是装配点传入的纯数据，`Merge` 为 BPE 合并对；`Bytes` 是字节级基线实现，`Bpe` 是在 `Vocab` 上按合并秩迭代归并的 BPE 具体实现（预切分层待 D2） |
| `Header` / `Entry` / `Reader` / `Loader` / `Archive` / `File` / `Quantized`；自有容器 `Writer` / `Stored` / `Segment` / `Quant` / `Codec` / `Proof`（llpk 模块） | `weights` | 权重文件，分两半：外部格式（safetensors）只读适配 + 自有 LLPK 容器读写。外部两条读取契约分工在**所有权**：`Reader` 把字节借出来（内存/映射），`Loader` 把字节拷进调用方缓冲区（文件）。LLPK 侧 `Writer`/`Archive` 读写对称，精度 code 由 `dtype` 单一事实源提供（`Wire` 是它的再导出别名，前 22 个 code 与旧线格式逐位兼容），另有显式量化、rANS、Merkle 证明、对齐与两级 CRC、原子写。`Quantized` 是外部量化条目与其标度伙伴的推断结果：分组、逻辑输入宽度、输出宽度出自**同一次**推断 |
| `Format` / `Facts` / `Snapshot` | `train` | 检查点的字节格式与恢复事实：`Format` 是魔数与版本（当前第 3 版），`Facts` 是恢复后必须逐项相同的四个数（步号 / 并行度 / 层数 / 批行数）——放在一个类型里才能整体比较，`Snapshot` 是"事实 + 权重条目表"（条目表本身归 `weights`，不复制）。字节只写**相对上一条**的差：名字共享前后缀、布局同型同形时只写一个标签、偏移存增量，全部 LEB128；尾部 CRC-32（八路查表）让截断与改动在**解码期**失败，每种"还能省却没省"的写法也被解码器拒掉，于是同一份快照只有一种字节形态。`encoded_len` 与 `encode` 共用一个编码器，`write` 把同一个编码器直接接到同目录临时文件上（整份字节不进内存，`rename` 不留半截）。恢复前的两道校验分开：`verify` 只验字节不建表，`check_weights` 要求条目表与手头的权重文件逐条相同——形状对、内容错位不会在读取时炸，只会在训练若干步后表现成 loss 不降 |
| `Page` / `Table` / `Cache` | `engine` | 分页 KV 的页管理：`Table::grow` 只抬容量（页仍由 `Cache` 发），`Cache` 用每页一字节的存活位图判重复归还（O(1)），不持有页里的数据 |
| `Tape` / `Var` / `Node` / `Requirement` / `Role` | `autograd` | 反向图的记录：变量的需求（不进图 / 中间 / 终点）、出身（参数 / 常量 / 激活，由需求与是否有生产者派生）、边、深度 |
| `Schedule` / `Account` / `Ordering` / `Config` | `autograd` | 反向调度（可达剪枝 + 可配置排序 + 同深度分层）与扇出账本（谁的梯度已齐全）；`Ordering` 是排序策略闭集，`Config` 把排序与显存策略收成一份纯数据约定；两者都在原对象上重填，训练循环不必每步新建 |
| `Plan` / `Budget` / `Choice` / `Footprint` / `Sizes` / `MemoryPolicy` | `autograd` | 显存计划：预算内的保留集、释放点与重放步数；`MemoryPolicy` 是自动 / 强制全留 / 强制分段的策略闭集。`Plan::of` 只读 `Tape`、`Schedule` 与一张形状表，不分配、不执行；与 `cutile::Plan`（单条 kernel 选哪个 tile）不是同一件事 |
| `Replay` / `Store` | `autograd` | 重放执行：把计划编成"第 `k` 步之前重算哪几步、之后交还哪些激活"的扁平序列表，并驱动整次反向。`Store` 是唯一的口子（`compute` / `free`），真实缓冲归 `train`，这一层不碰数值 |
| `Summary` | `autograd` | 图规模与可达性的只读计数快照：角色分布、边数、最大深度、可达变量与死分支；不存变量列表，供诊断、缓存键与容量预估 |
| `Envelope` / `Frame` / `Decoder` / `Kind` / `Version` | `serve`（`proto` 子模块） | 端点契约的线格式（ADR 0019 第七次吸收）：版本化信封、零拷贝帧视图与增量解码；帧布局、版本规则、载荷上限与校验尾见 ADR 0016。它不再独立成成员——唯一消费者就是同一个 crate 的 [`Response::frame`] 与 `Endpoint`，独立成员只会让"改协议"与"改端点"跨两次提交 |
| `Batch` / `Split` / `Dataset` / `Chunks` | `data` | 样本与切分。`Batch` 与 `Chunks` 都借用调用方的 token 缓冲（取批零分配、零拷贝），两个构造函数都按 `(rows, cols)` 收几何、与 `Shape::new([rows, cols])` 同序——此前 `Chunks::new` 收 `(cols, rows)`，同一对数字换个构造函数就是一次静默转置。批几何与批总数都在 `Chunks::new` 构造期算好，所以 `Dataset::next_batch` 的 `None` 只有一个含义：剩余不足一整批。三个计数各说一件事：`cols` 是列宽、`batches` 是完整批总数、`remaining_batches` 是还没取走的批数，`Chunks::rows` 只表示每批行数；`Chunks` 还是 `ExactSizeIterator`，`len()` 与 `size_hint` 都取自 `remaining_batches`，「还剩几条」因此只有一个来源。`Dataset<'a>` 是对象安全的（`&mut dyn Dataset<'a>` 可用），而 `Split` 的单位是样本行、`Chunks` 的单位是批，两者不互相换算 |
| `Contract` / `ContractSpec` / `Linear` / `LmHead` / `WeightFormat` | `nn` | 层定义；产出的算子名必须存在于 `op` 且有语义表。`LmHead` 是注解声明的固定形状层（形状、输入宽度与算子写在 `#[spec]` 上，`impl Contract` 与校验测试由宏生成）——它曾经住在 L7 组装点，而组装点不定义类型。`WeightFormat` 是权重编码的唯一事实源：格式 → 算子名（int4 → `gemv_i4`、int8 → `gemv_i8`、f32 → `gemv_f32`），没有算子的格式（f16 / bf16 / 裸 i8 / 未声明分组的 4 位打包）仍是合法声明，只是推不出下界，并在理由里点名缺什么；`Contract::linear` 决定"这一层能不能被真实数据执行"——注解层没有参数表，返回 `None` |
| `RmsNorm` / `Softmax` / `Gelu` / `Silu` / `Rope` / `Scale` / `Add` / `Mul` / `ResidualAdd` / `ReduceSum` / `ReduceMax` / `Embedding` / `Attention` / `AttentionAlgo` / `MatMul` | `nn` | 单算子层：归一化、激活、位置编码、逐元素二元、标量缩放、归约、查表、注意力与矩阵乘。各层只固定自己绑定的已注册算子名与形状口径，规格校验、访问器委托与下界路径共享内核；下界仍由算子语义表推出，层里不写第二份流量常数。`AttentionAlgo` 在朴素与 Flash 两条已注册注意力算子之间选型；`MatMul` 是矩阵 × 矩阵的三轴骨架，收缩轴由 `gemm` 算子固定，层只声明输出行/列宽 |
| `Bounds` | `nn` | 层的性能下界：`read` / `write` / `flops` 三笔，按方向分账（`bytes()` 是派生合计）；由算子语义推出，不认识设备峰值、也不做 roofline 编排 |
| `Stack` / `Floor` / `Unpriced` | `nn` | 层栈：一串层 + 它们的总下界（`Bounds::plus`）。层的清单可来自声明或权重文件的条目；不是矩阵的条目列进未覆盖清单，是矩阵但推不出下界的列进无法计价清单——两者都不允许静默丢弃，也都不让整张模型算不出下界 |
| `Group` / `Reduce` / `Collective` / `Topology` / `Parallel` / `Partition` | `shard` | 集合通信契约（进程组、归约、后端 trait）与 TP/PP/DP 拓扑；`world` 是构造期算好的精确值（`tp · pp · dp` 必须能用 `u32` 表示，因为 rank 就是 `u32`），因此 `check` 不会接受一个不可能的进程数、`group` 也不会用回绕的乘积发出"看起来合法"的 rank |
| `Schedule` / `Anneal` / `Step` / `Checkpoint` | `train` | 训练循环状态：`Schedule` 是纯函数学习率调度，`Anneal` 是 warmup 之后的衰减曲线闭集（线性 / 余弦 / 恒定）与最小学习率下限 |
| `Buffers` / `Kernel` / `Reference` / `Data` / `View` / `ViewMut` / `Kind` / `Inputs` | `train` | 激活缓冲：`autograd::Store` 在训练侧的落地——按 `Sizes` 分配、按计划的释放点当场交还、按等长空闲表复用。`load` 要求变量已登记形状与元素类型（缺登记是 `internal`，不是"跳过校验"），分配走 `try_reserve_exact`（形状可能来自权重文件头，分配失败是一条错误而不是 `abort`）。算术交给 `Kernel`（默认 `Reference` 调 `op` 的标量语义），本层不定义算子、不决定显存计划、不碰 `unsafe`。两条存储 lane（`f32` 与按位打包 `u8`）由密封 trait `Lane` 承载，形态标签 `Kind` 只在"元素类型 → 形态"处出现一次；`Reference` 的算子表长度与顺序同 `op::REGISTRY` 在编译期锁死，`Inputs` 把每组输入的 lane 写成类型 |
| `Sequence` / `Scheduler` / `Progress` | `engine` | 连续批与序列状态；准入只预留提示的页，之后每生成一个 token 由 `Scheduler::advance` 预留它要落的那页（池不够返回 `Blocked` 且不改任何状态），`Sequence::step` 因此不是公开 API；`blocked()` 给出批级阻塞集 |
| `Route` / `Endpoint` / `Response` | `serve` | 端点契约。`Endpoint::call` 的 `Err` 只留给"服务端自己坏了"，请求侧的不合法必须落成 `Response::error(BadRequest, …)`——准入就是个例子：id 重复、序列已结束、页宽与调度器不一致都是客户端能改的错，`503` 只用于"现在满、可重试"，`400` 只用于"永远装不下或请求自相矛盾" |
| `Command` / `Invocation` / `USAGE` | `cli` | 命令行解析；只分派，不执行任何业务 |
| `Report` / `join_names` | `tool` | 人可读输出；只接字符串，不接领域类型 |
| `Token` / `Pos` / `Delim` / `Finding` / `Gate` | `tool` | 零依赖词法 token 树、单函数复杂度 finding 与配置化阈值；分析器只产出数据，不打印不退出 |
| `spec` / `model` 属性宏 | `annotation` | 把注解展开成契约实现 + 编译期断言 + 一条校验测试；**不持有运行期注册表**。`model` 标的是 `type 名字 = <具体格式>`，生成 `MODEL`（`cost` / `sample` / `reference` 都是 `<名字 as Format>` 的单态化实例、`group` 取 `Format::GRAIN`）与"生产形状是粒度整数倍"的 `const` 断言，于是注解与泛型 bound 强绑定：别名不实现 `Format` 就编译不过。语言身份不走注解：设备执行是 `cutile` 的 `const SPECS` 描述符表按 `Emit` 登记探测、程序生成、执行回读三个函数指针，新增设备 = 表里加一行；语法门 `const GATES` 表行只带工具名与语法检查参数，"见证必须在它自己声称的语言里编得过"做成第三种结论——它不执行任何一行、不改等级，但被拒绝就是证据缺陷 |
| `Case` / `Baseline` / `Judgement` | `tool` | 回退判定（`tool::regress`）；不做测量（测量集中在工具成员 `benches`）。名字刻意避开 `Verdict`，见下节第三条 |

三个容易混淆、但**不是重复**的东西：

- **kernel 侧操作数视图**（DSL 的宿主 API：`&mut` 分片即 `grid`，见 cutile README §4.4）与
  `tensor::Tensor` 是两回事：前者不得依赖 `tensor`（内核层不能向上依赖运行时），后者也不得
  假装能做 kernel 分区。这个视图在 P0 **尚未实现**——`cutile` 只接收裸指针与标量，分片由
  调用方给；等 `llm` 真要直接调用 kernel 时再按 cutile README §4.4 重建，见
  [`adr/0003-kernel-and-runtime-types.md`](adr/0003-kernel-and-runtime-types.md) 的修订段。
- `cutile::space::Space`（GM/FAST/REG）是 **语言概念**，`prof::atom::Space` 是它的档案表示；
  档案侧只认字节，不认地址空间——地址空间由档案声明。
- `nn::LmHead` 这类**注解声明**与 `Linear::new` 这类**运行期构造**不是两套机制：前者把形状与
  算子写成属性、由 `#[spec]` 生成实现与校验测试，后者按配置计算宽度；两者最终都走
  `ContractSpec::new` 的同一条校验路径。注解不建注册表（注册表会成为第二个事实源），
  见 [`adr/0005-annotations-as-contracts.md`](adr/0005-annotations-as-contracts.md)。
- `cutile::bench::Verdict`（达成率口径：`reached` / `below` / `over-peak` / `cache-resident` / `unknown`）
  与 `tool::regress::Judgement`（基准回退判定：`pass` / `regressed` / `improved`）**不是同一件事**：
  前者判定“这次测量能不能与档案峰值比较”，后者判定“与基线相比是否回退”。两者会同时出现在
  一份 CI 报告里，所以名字必须不同——`tool::regress` 里的类型因此叫 `Judgement`。

### 闭集与位集：一份声明，编译期核对

「限定了取值范围的词汇表」（原语、数据类型、存储层、属性键、证据等级、输出语言、骨架）都必须
回答同样三个问题：**全量是什么、跨层名字是什么、位集下标是什么**。写三遍就是三份会漂移的事实源，
所以它们由一个契约收口：

- `impl Closed for X { const ALL: &'static [Self]; fn as_str(..); fn index(..); }` 是**唯一**声明处，
  `parse` 由 `ALL` + `as_str` 派生，不另写；
- L1 已经拥有闭集契约的成员（`DType`/`QType`/`MxType`/`mxtype::Element` 都实现了 `dtype::Format`）
  由一条 `impl<T: dtype::Format> Closed for T` blanket 接上，不再逐类型手写：`ALL` / `as_str` / `index`
  仍然只有一份来源（精度表 + 判别值）。四个闭集的 `Display` / `FromStr` / 可选 `serde` 名字也全部
  取自同一张行表（`Format::name` / `Format::parse_name`），因此不存在「序列化一套、显示一套」的漂移；
- `prof::inherent_names!(X)` 生成同名的固有 `ALL` / `as_str` / `parse` 转发，调用点写法不变，
  逻辑仍只有一份；
- `Set<E, WORDS>` 是唯一的位集：`WORDS` 是 const 参数，`Set<E, 1>` 装不下 `E::ALL.len() > 64`
  是**编译错误**（`const FITS: () = assert!(..)` 在单态化时求值），因此"加第 65 个原语"会先在
  编译期被拦住，而不是等到越界；
- 不在契约可达范围内的成员（`diagnostic` / `tensor` / `shard` / `serve`：都在
  `prof` 之下，反向依赖会成环）允许把词汇表留在本地，但必须登记在
  `lints/vocabulary.toml` 并写明理由；`serve` 的豁免条目指的是它的 `proto` 子模块（帧类型 `Kind`）。

这份契约由 `tests/contracts.rs` 执行：它扫描 `crates/*/src/**/*.rs`，禁止新增手写 `const ALL: [..]`
与手写位集（白名单在 `lints/vocabulary.toml`：dylint 规则已按 `ui-toml` 的格式设计、计划读同一份，
但**那条规则目前还没有落地**，今天只有测试在消费它），并禁止白名单过期（某成员已经改用 `Closed`
却还挂在白名单上，测试会要求删掉那条豁免）。

#### 注解里的名字也是词汇表的名字

属性宏收的是**字符串**（`dtype = "f32"`），词汇表里是**变体**，`codegen` 按
`DType::variant` 把前者翻成后者。三处必须同时成立，否则改名会变成一处静默漂移：

- `crates/annotation/src/syntax.rs` 的 `DTYPES` 与「`cutile::reference::tolerance` 给出非零阈值」的 `DType` 集合逐项相等
  （`tests/contracts.rs` 的 `annotation_names_come_from_the_vocabulary`）。这是**等式**：
  少一个名字就有一种输出永远无法被声明，多一个就拼出匹配不上的代码；
- 每个注解调用点生成的 `#[cfg(test)]` 模块把字面量与变体对齐
  （`DType::parse_name(dtype) == Some(MODEL.dtype)`），
  并把 `Model::dtype` 与 `.k` 的可写输出类型对齐（`signature.output_type()` 的 `CType::dtype()`）——
  注解说"按什么解释"、模板说"写的什么"，两者分家就对拍不上；失败指向被标注的那一行，而不是另一张表；
- 变体名取 `DType::variant`：`dtype` 名 → 变体名不再在注解里手写一份映射（缩写词全大写
  如 `bf16 → BF16` 的规则也住在 dtype 里），`annotation` 依赖 `dtype` 这个 L1 零依赖基座直接取
  唯一事实源，`tests/contracts.rs` 再把 `Debug` 名与 `variant` 对齐。

#### 语义表的唯一生产者

`Model` 不能有第二份来源。`#[annotation::model(name = .., shape = [..], dtype = ..)]` 标在
`type 名字 = <具体格式>;` 上，生成 `MODEL` 常量、`const _: () = <名字 as Format>::CHECK;`
与一条校验测试；注册表里的 `model:` 只能写 `..::MODEL`。两侧互为镜像，由
`tests/contracts.rs::the_registry_and_the_model_annotations_are_mirrors` 判：

- 标了 `#[model]` 却没在 `REGISTRY` 里加一行——生成物还在，发布出去的二进制里没有这条算子，
  `verify --op <名字>` 报"未知算子"，而作者以为已经接上；
- 注册表里手写 `Model { .. }` 字面量——语义表就有第二份来源，生成的那一份与它漂移时没有判据
  （P0-4 之前 `call_args` / `output` 的老毛病正是这个形状）。

规则按文本判（注解的参数顺序由 `syntax` 定死，注册表一条一行），且**没有豁免节**：标注了却不注册
永远错，注册的语义表不是注解生成的也永远错——出现例外说明契约本身该改，而不是某个 crate 该豁免。
`Format::CHECK` 同理：规则属于格式契约（`gemv::derived_matches_grain`），宏只负责让它在生产别名上
被求值，不另抄一遍。

`annotation` 仍然零外部依赖，只依赖 `dtype` 这个 L1 零依赖基座：`syn` / `quote` / `proc-macro2`
仍然不进入供应链（见 ADR 0005），而 `dtype` 名 → 变体名的映射取唯一事实源 `DType::variant`，
不再靠手写映射 + 门禁核对两份规则；给过程宏拉一条到 `prof` 的依赖边仍然被拒绝（`prof` 会带入
`diagnostic` 的传递依赖，且 `dtype` 已经提供了注解所需的全部类型身份）。

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
| `tests/architecture.rs` | CI `test` job | 成员集合、边集合与方向、所有权表的 `pub` 定义与 `cargo metadata` / 源码不一致；以及 `op` / `prof` / `prim` / `cutile`（含 `report`）是否又退回 `Result<_, String>` |
| `tests/contracts.rs` + `lints/vocabulary.toml` | CI `test` job | 闭集表或位集被手写第二份；白名单过期；扫描器什么都没扫到（有源文件数下限，避免“空门禁假绿”）；注解与注册表不互为镜像（标了 `#[model]` 没注册、注册的语义表不是注解生成的常量） |
| `#[model]` 生成的两个 `const` + 一条测试 | 被标注 crate 的 `cargo test` | 格式自证不成立（粒度 ≤ 0、派生轴除数 ≠ 粒度，编译期即失败）；注解的 `dtype` 与 `.k` 的可写输出不是同一个类型 |

`cargo metadata` 只能告诉你"谁依赖谁"，不能告诉你"这个依赖方向对不对"——后者由本文的层表定义，
再由 `tests/architecture.rs` 执行：它读 `cargo metadata` 与源码，把第一节的成员表、第二节的边集
（含总数与 `+dev` 数）、第三节的所有权表逐条比对。新增成员或调整依赖方向时，改完文档再跑
`cargo test`——不一致会直接失败，不再依赖评审清单。
