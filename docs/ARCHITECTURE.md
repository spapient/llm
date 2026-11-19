# 架构契约

这份文档回答三个问题：**谁属于哪一层、允许依赖谁、每个类型归谁**。
它是评审用的检查表：改动依赖方向或新增成员时，逐条核对下面所有约定。

`../crates/cutile/README.md` 描述的是 kernel 层（`cutile` + `op` / `prof` / `prim`）的内部机制；
本文描述的是**整仓的模块划分与依赖方向**。两者不重叠：改 DSL 机制看该 README，加模块看本文。
`cutile` 25 个模块的清单、职责判决与五设备取证见 [`cutile-analysis.md`](cutile-analysis.md)；设备数据与机制的边界裁决见 [ADR 0028](adr/0028-device-data-vs-mechanism-boundary.md)。

## 一、分层

| 层 | 成员 | 职责 | 不允许出现 |
|---|---|---|---|
| L1 基础设施 | `diagnostic`、`rng`、`dtype`、`wire`、`tool` | 错误类型与声明数据树（`diagnostic::data`：构建期语料在运行期的形状，**没有解析器**）、确定性计数器随机基础件（`rng`：Philox4x32-10 显式种子/流号/下标派生，均匀/正态/伯努利分布）、精度闭集、与领域无关的字节码闭集（`wire`：CRC-32、LEB128 变长整数、计数/切片/文件 Sink 与 CRC 密封）、静态质量门与人可读输出、回退判定（`tool::regress`） | 任何运行时领域语义、任何 TOML 解析 |
| L2 数据层 | `prof`、`op`、`prim` | 设备档案与能力闭集（`devices/*.toml` 在构建期解析成表）、算子声明与词汇表闭集（`ops/*.toml` 同上）（布局/调度/模板词法）与求和表达式的语义代价（flops/流量，`op::einsum`）、原语片段 | 任何机制代码（拼接、协商、emit）、任何设备知识 |
| L3 内核 | `cutile` | 机制（协商 → 拼接 → emit → 编译执行）与其上的命令策略：执行类 doctor/verify/bench 与 `execution` 共用一条 `pipeline` 与 `witness`；`ledger` 跨档案聚合证据行、`cache` 按 LRU 回收，二者不走流水线；只消费 L2 的数据 | 设备厂商特判、运行时依赖 |
| L4 运行时 | `tensor`、`storage`、`tokenizer` | 张量元数据、布局搬运（含 feature-gated 只读内存映射，全仓唯一 unsafe 岛）、文本边界 | 模型语义、调度策略 |
| L5 模型与训练 | `weights`、`autograd`、`data`、`nn`、`shard` | 权重结构（含自有 `.u` 容器 v2 读写与外部格式只读适配）、反向图、样本、层定义、切分拓扑与集合通信 | 进程/服务语义 |
| L6 调度与服务 | `train`、`engine`、`serve` | 训练循环状态与激活缓冲、检查点格式（`train::snapshot`）、连续批与序列、KV 页（`engine::kv`）、提交（`engine::exec`）、设备后端 FFI（`engine::metal`，Metal 越界岛，ADR 0035）、端点契约与其线格式（`serve::proto`）、CPU 模型内核桥（`serve::model`） | 直接调用 `cutile` 生成 kernel；执行任何业务 |
| L7 入口 | `cli`（bin：`llm` 与 `cutile`） | 组合根：命令行文法与配置解析归本层，把上面各层连起来装配、取值解析与人可读排版；编排入口（`cmd/` 与两个 bin）**不定义新类型**；`cutile` 二进制只做解析 → `cutile::{doctor, verify, bench, ledger, cache}` → 排版 → 退出码（`Code::exit_code`） | 编排入口新增领域类型；在本层复制内核机制 |

`annotation`（编译期注解）、`fuzz`（模糊测试契约）、`benches`（集中测量容器）与 `testkit`（测试夹具与工作区契约门禁）是**工具成员**：
它们不属于运行时分层，也不参与“至少一条依赖边”的统计。它们的**依赖边不参与方向判定**——
工具没有运行时层级。`testkit` 的库体只出浮点口径、随机数与暂存目录等夹具；跨层对拍与架构/契约守卫在其 `tests/` 下以集成测试运行，被测内核只出现在它的 dev-dependencies 里，因此不会被夹具库反向依赖进任何产品路径。

## 二、依赖规则（唯一一条，但严格执行）

> **依赖只能指向更低的层；同层内只能依赖表中更靠左的成员，不允许反向。**

于是：

- 向上依赖不可能通过：`op` 无法依赖 `cutile`，`tensor` 无法依赖 `engine`。
- 同层内也可以有先后：`prim` → `prof`，因为片段文件名必须落在原语闭集内。
- 环一定被拒绝：层与表内顺序构成偏序，严格递减无法成环，不需要额外的环检测。
- 数据层永远不知道机制：`op` / `prof` / `prim` 的依赖只有彼此与 L1。
- 语义先于设备：`op` 只写"一条求和表达式 + 操作数位宽"，`flops` / `bytes` 由 `op::einsum` 推出；
  `cutile` 的 roofline 只消费这个数对，不认识任何算子名。

允许的边（当前全量，共 110 条，含 42 条 `+dev` 边；工具成员 `annotation` / `fuzz` / `benches` / `testkit` 的边不参与运行时方向判定）：

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
weights  → tensor, diagnostic, dtype, wire, storage（`storage` 仅由 `mmap` feature 可选启用，复用其唯一 unsafe 岛）    +dev: storage, testkit
autograd → tensor, diagnostic    +dev: testkit
data     → tensor, diagnostic
nn       → tensor, op, weights, diagnostic, annotation
shard    → diagnostic
train    → nn, autograd, data, weights, shard, op, tensor, diagnostic, wire    +dev: testkit
engine   → tensor, diagnostic, storage    +dev: testkit
serve    → engine, nn, op, rng, tokenizer, diagnostic, weights    +dev: testkit
cli      → cutile, diagnostic, engine, nn, op, prim, prof, serve, tensor, tokenizer, tool, train, weights（L7 组合根：只做命令分派与各层装配，不定义新类型；运行时依赖覆盖被装配的各层）
fuzz     → diagnostic    +dev: cli, weights, train, serve, wire, tensor, tool（工具成员，不参与方向判定；`+dev` 的是 `tests/support/drivers.rs` 里的驱动对象：`cli` 参数文法、`weights` JSON / safetensors / `.u` 容器、`train` 检查点、`serve` 帧解码、`wire` LEB128 游标与接收器不变量、`tensor` 为检查点与 `.u` 夹具写形状、`tool` 词汇片段解析）
benches  → +dev: prof, op, cutile, wire, weights, tokenizer, tool, testkit, train, autograd, tensor, nn, engine, serve（工具成员：集中测量容器，不参与方向判定；预热、采样与测量时长由 criterion 配置，本成员只声明被测对象；`tool::regress` 判回退、`testkit` 登记"本机无基线"的跳过；train 基准覆盖调度求值、快照编解码与缓冲池复用，`autograd` 出 `Tape`/`Sizes`，`tensor` 出形状与元素类型，`nn` 出层下界与栈合计；`engine` 出页账与序列几何、`serve` 出宿主与 Predict 路线 B，供整模型一步夹具装配）
tool     → diagnostic（配置驱动静态质量门 + 人可读排版 + 回退判定；阈值文件在 `tool` 的 `build.rs` 里解析成 `diagnostic::data` 的表）    +dev: testkit
testkit  → +dev: autograd, cutile, nn, op, tensor, tool, train, weights（工具成员：夹具库 + 工作区契约门禁；跨层对拍 `crates/testkit/tests/cross_layer.rs` 打通权重、层声明与训练缓冲三段，`autograd` 出 `Tape`/`Sizes`，`tool` 出词汇片段解析；被测层只出现在 dev-dependencies，不进产品路径）
wire     → diagnostic（纯字节码闭集：CRC-32、LEB128、Sink；不认识张量或容器布局）    +dev: testkit
rng      → diagnostic（计数器型 Philox4x32-10 引擎与均匀/正态/伯努利分布；显式种子/流号/下标派生，不持全局状态、不引 `rand`）    +dev: testkit
diagnostic → （叶：错误类型 + `toml`，被大量成员消费；它自己的出边为零）
```

七条吸收链（`checkpoint→train`、`audit→cutile`、`regress→tool`、`exec→engine`、`kv→engine`、`einsum→op`、`proto→serve`）
**已落地**（ADR 0019）：成员从 33（含根包）降到 26，边从 106 条降到 82 条：消失的边是被吸收方的全部出边与全部入边，
新增的边只有 `cutile → annotation`（工具边）与 `serve → +dev: testkit`（`proto` 的 CRC 对拍随迁）。
方向合法性没有变化：七条链都是"同层向左"或"跨层向下"，因此 `crates/testkit/tests/architecture.rs` 的方向断言一条都不需要放宽。

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
成员数与边数**一条未增**（ADR 0023 时点）：`cli` 当时只依赖 `diagnostic`（键表不再借 `prof::atom::Closed`，三个键不值得
拉一套闭集机制），`cutile` 的五个出边原样；两个二进制各写一个 `kernel_dirs` 投影函数把 `Settings`
变成 `Dirs`，因此"配置策略"与"内核输入"之间只剩一次显式转换。

## 三、所有权表（一个类型只能有一个主人）

这张表由 `crates/testkit/tests/architecture.rs` **双向**校验：表里写的必须在 `crates/<主人>/src` 里有 `pub` 定义；
反方向也成立——每个成员源码里的 `pub struct|enum|trait` 必须在表里，或登记在
`lints/vocabulary.toml` 的 `[[ownership.unlisted]]` 存量基线，且基线不允许过期（某个名字进了表
或不再是 `pub`，对应条目必须删除）。L7 组装点另有单独一条：`src/**/*.rs` 不得定义任何类型。

| 概念 | 唯一主人 | 说明 |
|---|---|---|
| `Error` / `Code` / `Result` | `diagnostic` | 全仓库唯一的错误类型；同一 crate 的 `data` 子模块（ADR 0022）还拥有 `Table` / `Value`——它们是**声明数据树**，由构建期生成代码构造，运行期没有解析器，因此也不存在"语法错误"这一类；`op` / `prof` / `prim` / `cutile`（含 `report`）用同一份它，类别由**产生错误的那一层**给（`ProfileError::code`、`prim` 的装载诊断），上层只补上下文。`Error::also` 把同一次校验里的第二条失败追加为同级诊断（各自带类别，不再被 join 成一行 `invalid`）；`Error::from_io` 是"路径不存在 `missing` / 存在但读不出 `io`"的唯一判定；`Code::exit_code` 是类别到 shell 退出码的唯一映射（2/3/4/5/70） |
| `Rng` / `Uniform` / `Normal` / `Bernoulli` | `rng` | 计数器型确定性随机：`Rng` 持有种子 / 流号 / 下标与四词缓冲，`block` 是不推进状态的纯计数器派生；三个分布在构造期校验参数，采样是 `u ∈ [0,1) → 元素` 的纯映射，中间计算在 f32，半精度输出舍入归 `dtype` 转换，本层不持有全局状态也不引 `rand` |
| `DType` / `QType` / `MxType` / `Element` | `dtype` | 全仓唯一的精度事实源，三个格式轴分文件（`Element` 是 MX 的子闭集）：`dtype.rs` 的 `DType` 是标量闭集（24 个变体，Bool、U1–U64、I1–I64、F8 两种、F4、F16、BF16、F32、F64、`F8E4M3FNUZ`/`F8E5M2FNUZ`），稳定数字 code 与 `.u` 容器逐位兼容（前 21 个与旧 Wire 同码，F4 追加 22、FNUZ 追加 23/24）；`qtype.rs` 的 `QType` 是 GGML 块量化（25 个格式，传统 Q、K-quant、IQ 查表、TQ 三元四族；`Scale` 记块内 scale 前缀形状 f16/双 f16/f32/sub-scale，`Family` 记编码族，二者正交，`q8_K` 的 f32 d 与 `iq4_nl` 的单 d 各按源码如实分类）；`mxtype.rs` 的 `MxType` 是微缩放（OCP MX 六个标准格式 + NVFP4 厂商扩展，块共享纯指数 scale），同文件的 `Element` 是 MX 块内元素格式（E2M1/E2M3/E3M2/E4M3/E5M2/INT8，6 个，只回答位宽与标量对应）。这些格式只放身份与布局元数据，反量化算法留在 `weights`；除 `parse_name`（转调 trait 默认实现）外 inherent 访问器全部 `const fn`，零运行时开销，同一批读法再由 `Format` / `BlockFormat` 提供一次泛型视角（stable 无 const trait：几何的唯一实现是 inherent `const fn`，trait 只转发；`parse_name` 反过来由 trait 默认实现唯一提供），`Layout<B>` 是「标量元素或块格式」的唯一几何（`weights` 的 GGUF 支持表用 `Layout<QType>`，接 MXFP4 时换 `Layout<MxType>`）。各层以再导出（`tensor`、`weights` 的 `Wire`）、trait 或子集表表达局部能力，不再手写跨层映射 |
| `Dirs` | `cutile` | 内核要的三个目录（缓存根、工作目录、可选数据根）——**纯数据**：不带默认值、不读环境变量、不认识命令行。地址是调用方的策略，因此 `cutile` 只依赖 `std::path`，换一个嵌入方（测试给临时目录、服务给容器卷）不用改内核一行（ADR 0023） |
| `Settings` / `Roots` | `cli` | 三个目录开关的**解析策略**（`crates/cli/src/settings.rs`，ADR 0023）：内置默认 < `CUTILE_*` < `--set`、来源追踪（`值 ← 来源`）、平台默认目录与相对路径基准。键表是一行一个的声明宏（`cache.dir` / `work.dir` / `data.root`，含 `CUTILE_CACHE` 兼容名），`describe` 与"键未登记"的清单都从同一张表派生；**没有配置文件**（ADR 0022）。它曾经是独立成员 `llm-config` 的设置层，并入 `cutile` 后又上移到本层：默认目录是部署策略、`--set` 是命令行文法，都不是内核的职责；`cutile` 仍只收纯数据 |
| `Expr` / `Extents` / `Operand` / `Axes` / `Bits` / `Cost` / `Buffer` | `op` | 求和表达式的语义：一条式子推出 `flops` 与最少流量；`Buffer` 是内核样本的具名缓冲（形参名 + 取值）；不认识设备、tile 或片段 |
| `Op` / 模板文本 / `Signature` / `Sample` / `Model` | `op` | 算子声明、模板与算子语义（签名、样本、标量参考），`REGISTRY` 现有 24 条注册（GEMV 与 GEMM 各三个格式、逐元素与归约、形态算子、注意力两条及其 decode 变体两条、解量化两条），其中 GEMV 三条共用一份泛型骨架、只差格式契约；签名是 ABI 的唯一事实源，实参顺序、输出名与输出类型都由它派生（`output_type` 与 `Model::dtype` 交叉核对，回读只认 `float`）；代价由 `op::einsum` 推出，`Cost` 只是再导出；`bind` 是真实数据进入样本的唯一入口（只换 `x` / `w`，量化缓冲明确拒绝） |
| `Format` / `OperandSpec` / `Element` / `FormulaCache` / `Row` / `Conf` | `op` | `Row` 是一档对称量化格式的静态配置表（位宽 / 每字节码数 / 对称零点 / 解包表 / 样本偏移，全部 `const` 可编译期求值；GEMV / GEMM / 反量化三族打包格式按这张表读常量，不再各抄一份除法与码表）；`Conf` 是 `Row` 喂给打包骨架（取样 / 长度 / 解包 / 点积）的派生配置，只收骨架运行期需要的 `cpb` / `zero` / `offsets` 三项，位宽事实仍留在 `Row`。GEMV 骨架的格式契约：`Format` 把"随精度变化"的部分（权重元素、位宽、粒度、派生轴、标度、解包循环）收成关联常量与方法，`OperandSpec` 是表达式的操作数表（`const` 数组，装载时再转 `Operand`），`Element` 是样本缓冲的元素闭集（`f32` / `u8`，让 `reference` 不必按变体分发），`FormulaCache` 是四条骨架（GEMV / 逐元素 / 一元 / 归约）共用的唯一一份表达式缓存实现，单元格仍由每个具体格式用 `static` 持有（泛型作用域里的 `static` 只有一份，共用会让两种精度拿到错误的轴表），`Format::CHECK` 是格式对自身声明的自证（粒度 > 0、派生轴除数就是粒度——两处都拿同一个常数算分组，否则代价与样本是两种语义），由 `#[model]` 生成物在别名上求值；`Quantized<G>` / `Dense<B>` 的 const 参数分别承载声明粒度与累加块宽，`cost` / `sample` / `reference` 按格式单态化，`Model` 里仍是三个 `fn` 指针 |
| `OpDecl` / `Template` / `Segment` / `Registration` / `LayoutSpec` / `Schedule` / `Masking` | `op` | 声明解析的结果与三张闭集（模板词法、布局构造子、调度模板），以及唯一的算子注册表 `REGISTRY`（名字 / 声明 / 模板 / 语义表同属一条注册，`BUILTIN` 与 `model()` 都是它的投影）；`cutile` 只消费，不再持有第二份 |
| `Scalar` / `Monoid` / `SsmStep` / `Scan` | `op` | β 指数梯形扫描的数学底座与注册算子：`Scalar` 是扫描算术的精度约束（f32/f64），`Monoid` 给恒等元与结合 `combine`，是串行与 Blelloch 两种扫描同一个代数底；`SsmStep` 是 SSM 半群 `(a,b)`，`combine` 实现 `(a₁,b₁)⊕(a₂,b₂)=(a₂a₁, a₂b₁+b₂)`，`exp_trapezoid` 用 `expm1` 消 λ≈0 的灾难性消去；`Scan` 是 `scan_f32` 算子的模型标记（`#[annotation::model]` 生成 `MODEL`，`DECAY`/`INPUT` 是模板常量而非内核标量） |
| `Profile` / 能力词汇表 | `prof` | 设备档案与闭集；机制代码不得出现 `vendor == "..."` |
| `Closed` / `Set` / `KernelKey` | `prof` | 词汇表契约：`Closed` 要求每个闭集只声明 `ALL`（顺序即 `index`）、`as_str` 与 `index`，`parse` 由契约派生；`Set<E, WORDS>` 是全仓唯一的位集（`WORDS` 是 const 参数，容量在单态化时由 `const FITS` 断言，写宽或写窄是编译错误），`DTypeSet` / `SpaceSet` 只是它的别名；`inherent_names!` 把 `ALL` / `as_str` / `parse` 以**同名固有项**转发，于是调用方不必 `use` 一个 trait，而名字表仍然只有一份；`KernelKey` 是一条内核实现的登记键（ADR 0027 七元组：原语 / 实现名 / 精度 / 层级 / 宽度 / 架构标签 / 块宽），只承载身份，不持有可执行句柄 |
| `FragmentSet` / `Device` / `Image` / `Status` | `prim` | 原语片段与见证源码；`Device` 是每台设备这次构建带上的设备侧代码（原语枚举 + 入口映射表 + 可选镜像），`Image` 是编译产物、`Status` 是它的结局。三张内嵌表（`BUILTIN` / `WITNESS` / `DEVICES`）由 `build.rs` 扫描 `<vendor>/` 生成，`src/` 里只有类型 |
| `Catalogue` | `cutile` | 已装载的三层数据（声明 + 档案 + 片段）与它们的交叉校验；装载要随数据格式演进，所以不属于任何纯数据成员 |
| `Backend` / `Readback` | `cutile` | 设备后端契约（trait）：cutile 只依赖这条缝，不认识 Metal/CUDA/C 的具体实现；`emit` / `available` / `program` / `run` 四个方法把「生成程序 → 编译 → 运行 → 回读」抽象成一条 `dyn` 分派，具体后端由装配点（`cli`）注入，本层不硬编码后端清单。`Readback` 是回读结果的元组别名（每条写入个数 + f32 值） |
| `Plan` / `Path` / `NoPath` / `VariantCheck` / `Proof` / `Source` / `Usage` | `cutile` | kernel 层的中间产物，全部可见、无 IR；只有 `NoPath` 自带文本（`Display`，错误契约）。`Usage` 是构建期存储契约比较过的一条声明用量。`Proof` 是这次真正执行过的供给（`(原语, 实现名)` 集合）：协商只有在"档案写了 `L2`"与"这次跑通了见证"同时成立时才认一条能力，`Proof::declared` 只给解释性报告用 |
| `Toolchain` / `CacheKey` / `Cache` / `Entry` / `State` / `Changed` / `Residue` / `Gc` / `ArtifactKey` / `Artifact` | `cutile` | 缓存指纹、编译产物与存量：`Toolchain` 是实测工具链（`命令@版本`，来自 `--version`；档案里的 `[meta].tool_ver` 只是登记值），`CacheKey::fields` 是哈希、`meta.txt` 与差异比较的唯一来源且**不含工具链**（源文本与编译器无关）；`ArtifactKey` 是产物身份（源文本 + 实测工具链身份 + 编译参数指纹 + 主函数；参数一变就换一份产物，见证因此不会拿旧参数的二进制替新参数作证），`Artifact` 是命中结果（路径、字节、是否复用）；`Entry`/`State` 让"有 `source`"与"只写了一半"都成为可扫描的记录，`Residue`/`Gc` 是回收的结构化结果；每份产物旁的 `bin/<hex>.key` 记着它的身份（谁编的、拼了哪个主函数），`Cache::artifacts_in` 靠它在没有编译器时复用产物 |
| `Doctor` / `OpInfo` / `DeviceInfo` / `LayoutRow` / `Storage` / `Declared` / `Built` / `CacheWhy` / `CacheState` | `cutile` | `doctor` 的结构化结论：每步产物都是数据，排版由前端拼；`Storage` 逐层给出档案容量与 `Declared` 用量，`CacheWhy` 答“这个键与本次构建差在哪”，`CacheState` 把"从未落盘 / 半截落盘 / 一致 / 不同"分成四类（文本在前端） |
| `VerifyReport` / `VerifyEntry` / `Verified` / `Execution` / `WitnessRun` / `Evidence` / `Basis` | `cutile` | `verify` 的逐设备结果与见证执行；失败如实记录，不写成“跳过”。`Evidence` 是三条命令（`doctor` / `verify` / `bench`）共用的那一份证据：设备执行与自检各几条、依据是 `Executed` 还是 `Declared`、哪一步不成立；`Basis` 说清"这份结论来自执行还是档案自述"；`Execution` 是 `execute` 的结论：用调用方给的样本（真实权重）走同一条流水线，带回内核输出与标量参考 |
| `LedgerCell` / `BackendRow` | `cutile` | `ledger` 跨档案后端能力台账：`LedgerCell` 是一台设备对一条原语一条供给的结论（原语 / 实现名 / 自述等级 / 本次状态 / 说明），`BackendRow` 是一台设备的行（档案元信息、依据、设备与自检计数、语言门结论、可用性、逐原语格子）；台账不产生新证据，只聚合 `witness` 已跑出的 `Evidence` / `Proof`。台账本地的 `Status` 闭集把档案自述与本次执行结论分开，与 `prim::Status`（编译产物结局）同名不同义，路径隔离 |
| `Negotiation` | `cutile` | 一台设备的协商结论（首档 / 降级 / 无路径 / 无档案）：只读声明与档案，与执行成败分开记；文本在前端 |
| `Fragment` | `cutile` | 一份生成片段的行数与 §16 口径（`FRAGMENT_LIMIT` / `FRAGMENT_GOAL`）：上限管成败、目标只是余量，`within_limit` 就是判据本身 |
| `Bench` / `Throughput` / `Peak` / `Verdict` / `Performance` | `cutile` | `bench` 的实测口径与达成率判定；门槛常数（`L3_THRESHOLD`）也在这里。`Performance` 把达成率与它的尺子绑在一起：口径无效（超过 100%、工作集装得进 `FAST`）或没有峰值时 `is_measured` 为假，`needs_specialisation` 就是 §16 的“某设备 < 70%” |
| `Tensor` / `View` / `Element` / `Shape` / `Strides` / `Order` / `OwnedTensor` | `tensor` | `Tensor` 是**字节**视图，`View<E>` 是**元素**零拷贝视图（不分配、不装 `unsafe`）；`SUPPORTED` 子集表声明本层可构造的五种精度；`Order` 是排列顺序元数据，两种顺序的步幅累积是 `Strides` 的唯一实现；`OwnedTensor` 是分配出来、可写入的拥有形态（元数据 + 持有字节向量），与只读 `Tensor` 视图互补，设备地址不泄漏到本层 |
| `Layout` / `remap` / `Buffer` / `Allocator` / `CpuAllocator` | `storage` | 物理布局与搬运；`Order` 由 `tensor` 再导出、本层不另持枚举；`Buffer` 是拥有的扁平字节缓冲（设备句柄不泄漏到本层），`Allocator` 是分配/释放/显式拷贝的边界 trait，`CpuAllocator` 是安全 `Vec<u8>` 实现，设备后端在边界 crate 实现同一 trait；本层不做隐式跨设备拷贝 |
| `Stream` / `Graph` / `Launch` / `Device` / `Operand` / `OperandRole` / `ScalarSlots` | `engine` | 提交与执行；不编译 kernel。`Device` 是运行时设备标识标签（厂商 / 架构字符串），不持私有句柄；与 `prim::Device`（编译期设备侧代码镜像）是两个类型，路径隔离。`Operand`/`OperandRole` 是节点操作数的不透明引用（编号 + 只读输入/可写输出角色）；编号到实际缓冲的解析、生命周期与设备句柄都在后端边界侧，本层不持缓冲字节、不引用算子闭集。`ScalarSlots` 是定长类型化闭集标量槽（首批仅 rope 解码位置）：槽位集合冻结在 ADR 0043，不用字符串键名表或 `HashMap`，位置这类语义参数不编进 1 元素 f32 缓冲；变长批的逐序列位置表不进槽，走独立输入缓冲切片通道（ADR 0043 §5，整数按既有先例折 f32 承载） |
| `Tokenizer` / `Token` / `Bytes` / `Bpe` / `Unigram` / `SpecialTokens` / `Encoding` / `Strategy` / `Truncation` / `Padding` / `PadDirection` / `Fingerprint` / `DecoderStream` / `Vocab` / `VocabSource` / `VocabFamily` / `Merge` | `tokenizer` | 文本 ↔ token 的唯一边界：编号、编码对齐记录（编号/字节区间/特殊标记等长）、序列策略配置、词表身份指纹与 UTF-8 边界安全的流式解码；`Vocab` 是格式无关的中性词表值对象（双向查表、特殊编号范围与族附属表在构造期闭合），`VocabSource` 是装配点传入的纯数据，`Merge` 为 BPE 合并对；`Bytes` 是字节级基线实现，`Bpe` 是在 `Vocab` 上按合并秩迭代归并的 BPE 具体实现，`Unigram` 是按逐片分数做前向 Viterbi 最优切分的 Unigram 具体实现（预切分层待 D2） |
| `Header` / `Entry` / `Reader` / `Loader` / `Archive` / `File` / `MmapSource` / `Quantized` / `Scheme`；自有容器 `Writer` / `Stored` / `Segment` / `Quant` / `Codec` / `Proof` / `Metrics`（自有 .u 容器 v2） | `weights` | 权重文件，分两半：外部格式（safetensors）只读适配 + 自有 `.u` 容器 v2 读写。外部两条读取契约分工在**所有权**：`Reader` 把字节借出来（内存/映射），`Loader` 把字节拷进调用方缓冲区（文件）；`MmapSource` 是 `Reader` 的只读内存映射实现（`mmap` feature，复用 `storage` 全仓唯一 unsafe 岛，非 Unix 目标不编译）。`.u` 侧 `Writer`/`Archive` 读写对称，精度 code 由 `dtype` 单一事实源提供（`Wire` 是它的再导出别名，前 22 个 code 与旧线格式逐位兼容），另有显式量化、rANS、Merkle 证明、对齐与两级 CRC、原子写、REF 去重段与打开后现场现算的 `Metrics`。`Quantized` 是外部量化条目与其标度伙伴的推断结果：分组、逻辑输入宽度、输出宽度出自**同一次**推断；`Scheme` 是 2/4/8-bit 对称 PTQ 的运行时格式事实，`ptq` 的 `quantize`/`dequantize` 是 f32↔打包码的离线换算（ADR 0018，不进设备内核注册表） |
| `Format` / `Facts` / `Snapshot` | `train` | 检查点的字节格式与恢复事实：`Format` 是魔数与版本（当前第 4 版），`Facts` 是恢复后必须逐项相同的五个数（步号 / 并行度 / 层数 / 批行数 / 批列宽）——放在一个类型里才能整体比较，`Snapshot` 是"事实 + 权重条目表 + 优化器矩状态 + 累积窗口梯度"（条目表本身归 `weights`，不复制；矩/累积段类型见优化器行）。字节只写**相对上一条**的差：名字共享前后缀、布局同型同形时只写一个标签、偏移存增量，全部 LEB128；尾部 CRC-32（八路查表）让截断与改动在**解码期**失败，每种"还能省却没省"的写法也被解码器拒掉，于是同一份快照只有一种字节形态。`encoded_len` 与 `encode` 共用一个编码器，`write` 把同一个编码器直接接到同目录临时文件上（整份字节不进内存，`rename` 不留半截）。恢复前的两道校验分开：`verify` 只验字节不建表，`check_weights` 要求条目表与手头的权重文件逐条相同——形状对、内容错位不会在读取时炸，只会在训练若干步后表现成 loss 不降 |
| `Page` / `Table` / `Cache` / `Pager` | `engine` | 分页 KV 的页管理与页里的数据：`Table::grow` 只抬容量（页仍由 `Cache` 发），`Cache` 给每页记引用计数（`alloc` 置 1、`share` +1、`release` −1，O(1) 判越界/重复归还/下溢），计数归 0 才回空闲池且不持有数据；`Pager` 是页池加上它背后的 `f32` 缓冲——借出一页就有可写的行，只有最后一个持有者归还时才清零，因此"复用的页带出上一条序列的字节"没有发生的路径；行宽（一个 token 位置的 KV 宽度）由装载模型的一侧给，页池不认识层与头 |
| `Tape` / `Var` / `Node` / `Requirement` / `Role` | `autograd` | 反向图的记录：变量的需求（不进图 / 中间 / 终点）、出身（参数 / 常量 / 激活，由需求与是否有生产者派生）、边、深度 |
| `Schedule` / `Account` / `Ordering` / `Config` | `autograd` | 反向调度（可达剪枝 + 可配置排序 + 同深度分层）与扇出账本（谁的梯度已齐全）；`Ordering` 是排序策略闭集，`Config` 把排序与显存策略收成一份纯数据约定；两者都在原对象上重填，训练循环不必每步新建 |
| `Plan` / `Budget` / `Choice` / `Footprint` / `Sizes` / `MemoryPolicy` | `autograd` | 显存计划：预算内的保留集、释放点与重放步数；`MemoryPolicy` 是自动 / 强制全留 / 强制分段的策略闭集。`Plan::of` 只读 `Tape`、`Schedule` 与一张形状表，不分配、不执行；与 `cutile::Plan`（单条 kernel 选哪个 tile）不是同一件事 |
| `Replay` / `Store` | `autograd` | 重放执行：把计划编成"第 `k` 步之前重算哪几步、之后交还哪些激活"的扁平序列表，并驱动整次反向。`Store` 是唯一的口子（`compute` / `free`），真实缓冲归 `train`，这一层不碰数值 |
| `Summary` | `autograd` | 图规模与可达性的只读计数快照：角色分布、边数、最大深度、可达变量与死分支；不存变量列表，供诊断、缓存键与容量预估 |
| `Envelope` / `Frame` / `Decoder` / `Kind` / `Version` | `serve`（`protocol` 模块；`proto` 为兼容重导出路径） | 端点契约的线格式（ADR 0019 第七次吸收）：版本化信封、零拷贝帧视图与增量解码；帧布局、版本规则、载荷上限与校验尾见 ADR 0016。它不再独立成成员——唯一消费者就是同一个 crate 的 [`Response::frame`] 与 `Endpoint`，独立成员只会让"改协议"与"改端点"跨两次提交 |
| `Batch` / `Split` / `Dataset` / `Chunks` | `data` | 样本与切分。`Batch` 与 `Chunks` 都借用调用方的 token 缓冲（取批零分配、零拷贝），两个构造函数都按 `(rows, cols)` 收几何、与 `Shape::new([rows, cols])` 同序——此前 `Chunks::new` 收 `(cols, rows)`，同一对数字换个构造函数就是一次静默转置。批几何与批总数都在 `Chunks::new` 构造期算好，所以 `Dataset::next_batch` 的 `None` 只有一个含义：剩余不足一整批。三个计数各说一件事：`cols` 是列宽、`batches` 是完整批总数、`remaining_batches` 是还没取走的批数，`Chunks::rows` 只表示每批行数；`Chunks` 还是 `ExactSizeIterator`，`len()` 与 `size_hint` 都取自 `remaining_batches`，「还剩几条」因此只有一个来源。`Dataset<'a>` 是对象安全的（`&mut dyn Dataset<'a>` 可用），而 `Split` 的单位是样本行、`Chunks` 的单位是批，两者不互相换算 |
| `Contract` / `ContractSpec` / `Linear` / `LmHead` / `WeightFormat` / `ParamRole` | `nn` | 层定义；产出的算子名必须存在于 `op` 且有语义表。`ParamRole` 是参数角色闭集（Trainable/Bias/Norm/Embedding，ADR 0037）：权重衰减豁免与可训练判定从角色派生，优化器按角色取组，不按字段名猜 |`LmHead` 是注解声明的固定形状层（形状、输入宽度与算子写在 `#[spec]` 上，`impl Contract` 与校验测试由宏生成）——它曾经住在 L7 组装点，而组装点不定义类型。`WeightFormat` 是权重编码的唯一事实源：格式 → 算子名（int4 → `gemv_i4`、int8 → `gemv_i8`、f32 → `gemv_f32`），没有算子的格式（f16 / bf16 / 裸 i8 / 未声明分组的 4 位打包）仍是合法声明，只是推不出下界，并在理由里点名缺什么；`Contract::linear` 决定"这一层能不能被真实数据执行"——注解层没有参数表，返回 `None` |
| `RmsNorm` / `Softmax` / `Gelu` / `Silu` / `Rope` / `Scale` / `Add` / `Mul` / `ResidualAdd` / `ReduceSum` / `ReduceMax` / `Embedding` / `Attention` / `AttentionAlgo` / `MatMul` | `nn` | 单算子层：归一化、激活、位置编码、逐元素二元、标量缩放、归约、查表、注意力与矩阵乘。各层只固定自己绑定的已注册算子名与形状口径，规格校验、访问器委托与下界路径共享内核；下界仍由算子语义表推出，层里不写第二份流量常数。`AttentionAlgo` 在朴素与 Flash 两条已注册注意力算子之间选型；`MatMul` 是矩阵 × 矩阵的三轴骨架，收缩轴由 `gemm` 算子固定，层只声明输出行/列宽 |
| `AttentionBlock` | `nn` | 注意力组合块：Q/K/V 三个投影 + 注意力 + 输出投影内聚成一个 `Contract`，子层按 `{块名}.q` / `.k` / `.v` / `.attn` / `.o` 前缀对齐权重条目；下界逐子层求和。残差旁路与层归一化引用块外主干隐藏向量，归装配侧接线，不内聚；子层经 `Contract::attention_block` 类型化视图取回，外层不按层名前缀分组扇出 |
| `Bounds` | `nn` | 层的性能下界：`read` / `write` / `flops` 三笔，按方向分账（`bytes()` 是派生合计）；由算子语义推出，不认识设备峰值、也不做 roofline 编排 |
| `Stack` / `Floor` / `Unpriced` | `nn` | 层栈：一串层 + 它们的总下界（`Bounds::plus`）。层的清单可来自声明或权重文件的条目；不是矩阵的条目列进未覆盖清单，是矩阵但推不出下界的列进无法计价清单——两者都不允许静默丢弃，也都不让整张模型算不出下界 |
| `Calib` / `Lif` | `nn` | 决策与脉冲的参考语义：`Calib` 是 DecisionHead 的校准闭集（spherical/softmax/temperature/brier，`brier` 是严格适当评分，`HEAD_LAYERS_TOKEN=0`）；`Lif` 是 γ 脉冲的漏积分-发放神经元（积分 + 泄漏 + 阈值 + 重置），`encode_id` 给 IEEE-754 恒等编码 |
| `Group` / `Reduce` / `Collective` / `Topology` / `Parallel` / `Partition` | `shard` | 集合通信契约（进程组、归约、后端 trait）与 TP/PP/DP 拓扑；`world` 是构造期算好的精确值（`tp · pp · dp` 必须能用 `u32` 表示，因为 rank 就是 `u32`），因此 `check` 不会接受一个不可能的进程数、`group` 也不会用回绕的乘积发出"看起来合法"的 rank |
| `Schedule` / `Anneal` / `Step` / `Session` / `TrainConfig` / `ParamSlot` / `Checkpoint` / `Distributed` | `train` | `Distributed` 是分布式编排装配：拓扑、本 rank 与通信后端（泛型 `E: Collective`，单进程恒等回路不持通信句柄），梯度按数据并行组归约、是否取平均与归约种类随配置走；后端契约在 `shard`，训练侧只挂装配与同步口径。训练循环状态：`Schedule` 是纯函数学习率调度，`Anneal` 是 warmup 之后的衰减曲线闭集（线性 / 余弦 / 恒定）与最小学习率下限；`Session` 是一次训练会话（复用同一张图、同一批缓冲与同一个优化器，逐 micro-batch 推进），`TrainConfig` 收学习率峰值 / warmup / 总步数 / 裁剪上限 / 梯度累积窗口，`ParamSlot` 是已登记的参数槽（变量编号、角色与常驻参数值，梯度不在槽内、由缓冲自持） |
| `Optimizer` / `Sgd` / `Adam` / `Group` / `Slot` / `OwnedSlot` / `Moments` / `SlotMoments` / `GradAccum` | `train` | 优化器：对象安全 trait，矩状态由实现按槽位编号自持；`Group` 是角色派生的参数组闭集（`Group::from_role` 唯一映射，不按字段名猜组）；`OwnedSlot` 是一次步进取走的参数/梯度向量配对（owned 副本），`view()` 借出可变视图；`Slot` 是 `OwnedSlot` 内部向量借出的参数/梯度/角色视图；`Moments` 是矩状态表（按槽位编号排列 + 优化器全局步数，快照只搬运这一份表，无矩优化器写 `empty`），`SlotMoments` 是单槽的一阶矩 `m` 与二阶矩 `v`（两张表长度一致）；`GradAccum` 是累积窗口内尚未步进的梯度汇总（各槽 f32 表 + 已汇总 micro-batch 数），与矩状态平行入快照。autograd 只出 tape / plan / Replay，不持梯度存储 |
| `Buffers` / `Kernel` / `Reference` / `Data` / `View` / `ViewMut` / `Kind` / `Lane` / `Sealed` / `Inputs` | `train` | 激活缓冲：`autograd::Store` 在训练侧的落地——按 `Sizes` 分配、按计划的释放点当场交还、按等长空闲表复用。`load` 要求变量已登记形状与元素类型（缺登记是 `internal`，不是"跳过校验"），分配走 `try_reserve_exact`（形状可能来自权重文件头，分配失败是一条错误而不是 `abort`）。算术交给 `Kernel`（默认 `Reference` 调 `op` 的标量语义），本层不定义算子、不决定显存计划、不碰 `unsafe`。两条存储 lane（`f32` 与按位打包 `u8`）由密封 trait `Lane` 承载，形态标签 `Kind` 只在"元素类型 → 形态"处出现一次；`Reference` 的算子表长度与顺序同 `op::REGISTRY` 在编译期锁死，`Inputs` 把每组输入的 lane 写成类型 |
| `MetricPoint` / `MetricsBoard` / `Learner` / `ForwardOut` | `train` | 评估与编排：`MetricPoint` 是一条步指标（步号、样本数、平均 loss、命中率），`MetricsBoard` 在内存按样本数加权聚合 loss / accuracy、记最优 loss 并渲染可读行，不做持久化；`Learner` 是独立训练编排器，遍历数据集驱动「前向 → 评估入账 → 装梯度 → 优化步」闭环，模型拓扑与梯度来源由前向 / 梯度闭包注入；`ForwardOut` 是一次前向产出的 logits 与 loss 两个终点变量。交叉熵与命中统计是 train 侧纯函数，不进 op 设备核注册表 |
| `Distill` / `Grpo` / `Loss` / `Metric` / `Outcome` | `train` | 损失与度量：`Loss` 是损失闭集（分类交叉熵 / Hinton 蒸馏 / GRPO），`Metric` 是前向产出携带的度量闭集（蒸馏/GRPO 的中间量与配置，配置随度量走、默认 `None` 即分类），两者都是纯数据标记 + 分发入口，实际数值由 `objective`/`learner` 的纯函数按三臂（classify/distill/grpo）拆开算；`Distill` 收温度 `T` 与软损失权重 `α`，`Grpo` 收裁剪 `ε` 与参考 KL 惩罚 `β`，两者取极值退化为纯 CE/软蒸馏/REINFORCE；`Outcome` 是损失与命中数的成对结果（分类/蒸馏为 argmax 命中、RL 为 0）。损失族是 train 侧纯函数，不进 op 设备核注册表 |
| `Sequence` / `Scheduler` / `Progress` | `engine` | 连续批与序列状态；准入只预留提示的页，之后每生成一个 token 由 `Scheduler::advance` 预留它要落的那页（池不够返回 `Blocked` 且不改任何状态），`Sequence::step` 因此不是公开 API；`blocked()` 给出批级阻塞集。`Sequence` 还是已生成 token 的唯一存放处（`tokens` 只读、`push_token` 仅同 crate 可见）：`Predict` 只看得到序列，历史存到别处就等于把"模型能读到自己的输出"降级成调用方约定。记账与推进成对走 `Batch::emit`——`Blocked` 时不多记一个 token，否则重试会看见一个并不存在的输出 |
| `Route` / `Endpoint` / `Response` | `serve` | 端点契约。`Endpoint::call` 的 `Err` 只留给"服务端自己坏了"，请求侧的不合法必须落成 `Response::error(BadRequest, …)`——准入就是个例子：id 重复、序列已结束、页宽与调度器不一致都是客户端能改的错，`503` 只用于"现在满、可重试"，`400` 只用于"永远装不下或请求自相矛盾" |
| `Command` / `Invocation` / `USAGE` | `cli` | 命令行文法（argv/env → 强类型调用参数）与编排分派；编排入口只装配各层、取值解析与人可读排版，不定义新类型 |
| `Report` / `join_names` | `tool` | 人可读输出；只接字符串，不接领域类型 |
| `Token` / `Pos` / `Delim` / `Finding` / `Gate` | `tool` | 零依赖词法 token 树、单函数复杂度 finding 与配置化阈值；分析器只产出数据，不打印不退出 |
| `spec` / `model` 属性宏 | `annotation` | 把注解展开成契约实现 + 编译期断言 + 一条校验测试；**不持有运行期注册表**。`model` 标的是 `type 名字 = <具体格式>`，生成 `MODEL`（`cost` / `sample` / `reference` 都是 `<名字 as Format>` 的单态化实例、`group` 取 `Format::GRAIN`）与"生产形状是粒度整数倍"的 `const` 断言，于是注解与泛型 bound 强绑定：别名不实现 `Format` 就编译不过。语言身份不走注解：设备执行是 `cutile` 的 `const SPECS` 描述符表按 `Emit` 登记探测、程序生成、执行回读三个函数指针，新增设备 = 表里加一行；语法门 `const GATES` 表行只带工具名与语法检查参数，"见证必须在它自己声称的语言里编得过"做成第三种结论——它不执行任何一行、不改等级，但被拒绝就是证据缺陷 |
| `Case` / `Baseline` / `Judgement` | `tool` | 回退判定（`tool::regress`）；不做测量（测量集中在工具成员 `benches`）。名字刻意避开 `Verdict`，见下节第三条 |
| `EntryKind` / `Harness` / `Roofline` / `Geometry` / `BenchRecord` | `tool` | 统一基准记录结构（`tool::bench`）：`EntryKind` 是条目层级闭集（算子 / 层 / 整模型），`Harness` 是硬件与工具链身份，`Roofline` 是峰值带宽 / 算力上界，`Geometry` 是形状 / 批次 / 精度三元几何，`BenchRecord` 是一条基准记录（身份 + 几何 + 后端 + roofline + 可选实测）。只持有记录、不测量不写性能结论；实测值由调用方跑完后填入，基线是否登记由 `regress` 判定，未测得的峰值留 `None` 不预填 |

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

### 运行时实现新增的类型

下面这张表与上面那张同属所有权表（判据完全相同：表里写的必须在主人里有 `pub` 定义，反向亦然），
分开列是因为它们全部来自"把引擎、分片与服务从契约推进到可执行"这一轮改动，评审时应当能一眼看到
新增了什么、每个类型归谁。

| 概念 | 唯一主人 | 说明 |
|---|---|---|
| `Program` / `Bindings` / `Buffer` / `View` / `ViewMut` / `CpuExecutor` | `engine` | CPU 执行体：`Program` 是一段可执行内核体（名字 + 在只读输入与可写输出上就地计算），按名字登记、重名拒绝；`Buffer` 是已绑定缓冲的存储形态，`F32(Vec<f32>)` 或按位打包的 `U8(Vec<u8>)`（i4/i8 lane），只承载元素宽度，形状与条目名留装配侧；`View`/`ViewMut` 是一次 `Program::run` 的只读/可写窄视图，分别给 f32 切片或打包字节切片，形态不符取槽给 `Invalid`；`Bindings` 是编号 ↔ `Buffer` 的操作数表（f32 与打包字节两路 lane），按编号排序，因此诊断输出与插入顺序无关；`CpuExecutor` 把图**入队**、在 `sync` 时按图序真的执行——查内核、借出操作数、调用、放回，失败路径也放回，而没跑完的条目留在队列里（`pending` 查得到）。`sync` 只排指定流：跨流先后不属于本层，`submit` 返回只表示"已入队"，这与 `Executor` 契约里"提交不等于完成"一致。它与只记账的 `LocalExecutor` 共用 `Executor` trait，"换后端不改上层"不因多了实现而改变。借出/放回是安全的：编号到缓冲的映射是运行期值，借用检查器无法证明两组编号不重叠，所以执行器把缓冲从表里取出来、用完放回，不装 `unsafe` |
| `Slot` | `engine` | 一个 token 在 KV 缓存里的位置（页号 + 页内偏移）：`Table::slot` 用页宽除出偏移，`Scheduler::slot` 连"先准入"这一前提一起校验。写 kernel 的一侧因此不必自己拿页表再算一遍偏移 |
| `Pager` | `engine` | 页池加上页里的数据：一页 = `page_size` 行、每行 `width` 个 `f32`，整池构造时一次分配，借出不再碰分配器。`row` / `row_mut` / `write` 只经 `Slot` 寻址，并且先验"页还活着"与"偏移在页内"——失效的 `Slot` 不是合法地址。`share` 是另一张表采纳同一前缀页的入口（只 +1 计数）；`release` 是唯一清零点：只有最后一个持有者归还时才清零，所以复用页读到的永远是零。行宽由装载模型的一侧给，页池不认识层与头；多层模型拿整行再按自己的层偏移切 |
| `Batch` / `Admission` | `engine` | 连续批：`Batch` 把页账（`Scheduler`）与它正在跑的 `Sequence` 收在一处，准入、推进、结束与 KV 读写都只有这一个来源，`sequences` 与 `running` 顺序相同；`emit(id, token)` 是"记账 + 推进"的唯一入口，`Blocked` 时不多记一个 token；`Admission` 是准入结局——已进批，或页不够时把序列原样退回（服务层据此报可重试的 503）。与 `data::Batch`（一批 token 行）同名不同物：前者是调度单位，后者是数据单位 |
| `Sharding` / `Tiling` | `shard` | 张量并行的两种线性层切法（`Column` 切输出维、分片直接进下一层；`Row` 切输入维、每 rank 一份部分和）与一层权重的切分计划：每个 rank 的权重形状、被切维区间、以及"合并要不要 `all_reduce(Sum)`"。切分必须整除（各 rank 形状一致），不整除在构造期失败，而不是摊余数 |
| `Local` | `shard` | 进程内的集合通信实现：每个 rank 一份镜像，`load` 写入输入、`all_reduce` 按 `Group` 的 rank 升序归约并写回每个成员。它让 TP/DP 的数值语义在没有 NCCL 的机器上也能被 `cargo test` 验证 |
| `Remote` | `shard` | 远端集合通信占位后端：与 `Local` 同一 `Collective` 端口，不持网络/通信句柄、不接 FFI、不写 `unsafe`；`all_reduce` 统一返回 `Unsupported`，拒绝时不改动调用方缓冲。真实跨进程实现（NCCL/MPI/Gloo，组内握手与句柄池）随后续后端刀填入，上层调用点不变 |
| `Status` / `Request` | `serve` | 响应状态闭集（200 / 400 / 404 / 503，三个失败态各对应一种客户端动作）与请求文法：`id` / `prompt` / `max_new` 三行的行文法，未知键、重复键、缺字段与坏数字都是 `invalid`（服务层据此报 400）。停止符不在请求里——"采到哪个 token 算停"是采样策略的事 |
| `Predict` / `Sampling` / `Greedy` / `Classical` / `Multinomial` / `SamplingParams` / `Generation` / `Ending` | `serve` | 生成的协作者与结局：`Predict` 给一步 logits、`Sampling` 选一个 token（`None` 即停止）、`Greedy` 是确定性贪心（等值取最小下标、非有限值报错）；`Classical` 是经典采样（温度缩放 → top-k → top-p 核采样 → 保留集 argmax，截断后仍是确定性 argmax，不接 RNG）；`Multinomial` 是多项式采样（与经典采样共用同一套截断，保留集上按 softmax 概率逆变换抽样，随机流由构造种子经显式 `rng::Rng` 派生、不读全局熵源，同种子同 logits 逐位可复现；Gumbel-max 为并行后端等价口径预留），`SamplingParams` 是它的温度 / top-k / top-p 闭集参数档案（`None` 表示该档不启用，越界值采样时对照候选条数校验）；`Generation` 是"采到的 token + 为何停下"，`Ending` 是闭集（`Limit` / `Stop`）。端点不认识模型，模型也不认识页账 |
| `Completion` / `Router` | `serve` | `Completion` 是第一条真实端点：解析请求 → 编词 → 准入 → 逐步前向 + 采样 → 归还页 → 包成帧；客户端错误落 400、页不够落 503、只有模型或采样失败才 `Err`，失败路径同样归还页。`Router` 按 `方法 + 路径` 分派（方法大小写不敏感、重复路由拒绝、未命中 404）。二者合起来让"一条请求能不能跑完"在进程内可验证，传输层只需调用 `Router::call` |
| `Httpd` | `serve` | std-only HTTP/1.1 薄适配器：单线程 `accept → 解析请求行 + Content-Length + 主体 → Router::call → Response::frame → 回写 → 关闭`；连接级错误回写 500 后返回、不向上抛。首刀只认请求行 + `Content-Length` + 主体，响应固定 `Connection: close`，排除 TLS / keep-alive / chunked / 多线程；gRPC 未来在同一 `Endpoint` / `Router` 上再挂一个适配器即可，不动端点模型 |
| `CpuModel` / `Params` / `Block` | `serve` | CPU 模型宿主与驻留参数表：`Params` 是条目名 → 权重块的内存表，装载期一次性解码进来，生成循环内不再读文件；`Block` 是驻留块的两种形态——稠密 `f32` 或量化打包字节（i4 每字节两码、i8 每字节一码），量化标度作为独立 f32 条目登记、`Params` 记录权重条目到标度条目的对应。`CpuModel` 把 `nn::Stack` 的投影 / 一元 / 归一化 / 查表层与注意力块装配成 `engine::Graph`，权重整块在构造期驻留一次，`forward` 每步只覆写输入与逐层激活缓冲。形状、宽度链与权重缺失都在装配期报出，int4/int8 量化投影（含注意力块内 q/k/v/o 四路）走 `gemv_i4`/`gemv_i8`（打包字节 + 标度两条输入，量化输入宽度须为分组 32 的整数倍，否则装配期 `Invalid`）；`from_path` 是文件装配根：打开头部 → 层栈 → 驻留参数 → 图装配，字节来源按 `Reader`（Unix + `mmap` feature 的 `MmapSource` 映射借用）/ `Loader`（回退 `File` seek+read 拷贝）在装配点类型可见分流，不立平行 `Source` 父 trait；`Predict` 路线 B 经 `&Sequence` + `&mut Batch` 读历史 K/V、写回当前步 `[k|v]` 页行，页账归调度器，宿主不另存 KV 账 |
| `MetalDevice` / `MetalCommandQueue` / `MetalBuffer` / `MetalPipeline` / `MetalExecutor` | `engine` | 设备后端边界（M3，ADR 0035 白名单第五类「进程级库句柄」，`engine::metal` 越界岛，不单独成 crate）：`MetalDevice` 是本机 Metal 默认设备的进程内句柄（`MTLCreateSystemDefaultDevice` 按 Create 规则 +1 保留，`Drop` 时 `-release` 恰好一次）；`MetalCommandQueue` 是它经 `newCommandQueue` 派生的命令队列句柄（new 规则 +1 保留、`Drop` 释放）；`MetalBuffer` 是经 `newBufferWithLength:options:`（`MTLResourceStorageModeShared`）分配的共享缓冲句柄，`contents` 指针缓存在内、`as_mut_bytes` 暴露 CPU 可读写字节视图（读写同一块内存），同样 +1 保留、`Drop` 释放；`MetalPipeline` 是经 `newLibraryWithSource:options:error:` → `newFunctionWithName:` → `newComputePipelineStateWithFunction:error:` 在进程内编译 MSL 得到的计算管线句柄（编译路径用 `objc_autoreleasePoolPush/Pop` 包住自动释放的 `NSString`，`library`/`function` 建管线后立即释放）。五者指针私有、不暴露、不跨线程，本刀打通「设备 → 命令队列 → 共享缓冲 → 计算管线 → 派发回读」完整闭环（`MetalCommandQueue::execute` 走 `commandBuffer → computeCommandEncoder → setComputePipelineState → setBuffer → dispatchThreads → endEncoding → commit → waitUntilCompleted`，`contents` 回读）与 `objc_msgSend`（`(id, SEL) -> id` / `-> void` / `usize` / 指针 / 结构体参数多种形状，`dispatchThreads:threadsPerThreadgroup:` 的 `MTLSize` 按间接指针 ABI 登记）；`MetalExecutor` 把这条闭环接进 `engine::Executor` 契约：持设备/队列句柄 + 内核表（`register` 按算子名进程内编译 MSL，入口 `kernel_main`）+ 操作数表（`bind`/`bind_bytes` 按编号绑共享缓冲），`submit` 只入队、`sync` 按图序逐节点查内核→按「输入后输出」绑缓冲→`execute` 派发，`waitUntilCompleted` 后输出即可回读；MSL 源码由装配层注入，本层不调用 `cutile` 生成 kernel（L6 禁止列） |

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
  `lints/vocabulary.toml` 并写明理由；`serve` 的豁免条目指的是它的 `protocol` 模块（帧类型 `Kind`）。

这份契约由 `crates/testkit/tests/contracts.rs` 执行：它扫描 `crates/*/src/**/*.rs`，禁止新增手写 `const ALL: [..]`
与手写位集（白名单在 `lints/vocabulary.toml`：dylint 规则已按 `ui-toml` 的格式设计、计划读同一份，
但**那条规则目前还没有落地**，今天只有测试在消费它），并禁止白名单过期（某成员已经改用 `Closed`
却还挂在白名单上，测试会要求删掉那条豁免）。

#### 注解里的名字也是词汇表的名字

属性宏收的是**字符串**（`dtype = "f32"`），词汇表里是**变体**，`codegen` 按
`DType::variant` 把前者翻成后者。三处必须同时成立，否则改名会变成一处静默漂移：

- `crates/annotation/src/syntax.rs` 的 `DTYPES` 与「`cutile::reference::tolerance` 给出非零阈值」的 `DType` 集合逐项相等
  （`crates/testkit/tests/contracts.rs` 的 `annotation_names_come_from_the_vocabulary`）。这是**等式**：
  少一个名字就有一种输出永远无法被声明，多一个就拼出匹配不上的代码；
- 每个注解调用点生成的 `#[cfg(test)]` 模块把字面量与变体对齐
  （`DType::parse_name(dtype) == Some(MODEL.dtype)`），
  并把 `Model::dtype` 与 `.k` 的可写输出类型对齐（`signature.output_type()` 的 `CType::dtype()`）——
  注解说"按什么解释"、模板说"写的什么"，两者分家就对拍不上；失败指向被标注的那一行，而不是另一张表；
- 变体名取 `DType::variant`：`dtype` 名 → 变体名不再在注解里手写一份映射（缩写词全大写
  如 `bf16 → BF16` 的规则也住在 dtype 里），`annotation` 依赖 `dtype` 这个 L1 零依赖基座直接取
  唯一事实源，`crates/testkit/tests/contracts.rs` 再把 `Debug` 名与 `variant` 对齐。

#### 语义表的唯一生产者

`Model` 不能有第二份来源。`#[annotation::model(name = .., shape = [..], dtype = ..)]` 标在
`type 名字 = <具体格式>;` 上，生成 `MODEL` 常量、`const _: () = <名字 as Format>::CHECK;`
与一条校验测试；注册表里的 `model:` 只能写 `..::MODEL`。两侧互为镜像，由
`crates/testkit/tests/contracts.rs::the_registry_and_the_model_annotations_are_mirrors` 判：

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
| `crates/testkit/tests/architecture.rs` | CI `test` job | 成员集合、边集合与方向、所有权表的 `pub` 定义与 `cargo metadata` / 源码不一致；以及 `op` / `prof` / `prim` / `cutile`（含 `report`）是否又退回 `Result<_, String>` |
| `crates/testkit/tests/contracts.rs` + `lints/vocabulary.toml` | CI `test` job | 闭集表或位集被手写第二份；白名单过期；扫描器什么都没扫到（有源文件数下限，避免“空门禁假绿”）；注解与注册表不互为镜像（标了 `#[model]` 没注册、注册的语义表不是注解生成的常量） |
| `#[model]` 生成的两个 `const` + 一条测试 | 被标注 crate 的 `cargo test` | 格式自证不成立（粒度 ≤ 0、派生轴除数 ≠ 粒度，编译期即失败）；注解的 `dtype` 与 `.k` 的可写输出不是同一个类型 |

`cargo metadata` 只能告诉你"谁依赖谁"，不能告诉你"这个依赖方向对不对"——后者由本文的层表定义，
再由 `crates/testkit/tests/architecture.rs` 执行：它读 `cargo metadata` 与源码，把第一节的成员表、第二节的边集
（含总数与 `+dev` 数）、第三节的所有权表逐条比对。新增成员或调整依赖方向时，改完文档再跑
`cargo test`——不一致会直接失败，不再依赖评审清单。
