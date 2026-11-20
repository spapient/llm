# 架构契约

这份文档回答三个问题：**谁属于哪一层、允许依赖谁、每个类型归谁**。它不是评审清单，而是
**可执行的契约**——[`crates/testkit/tests/architecture.rs`](../crates/testkit/tests/architecture.rs)
读 `cargo metadata` 与源码，把下面第一节的成员表、第二节的边集（含总数与 `+dev` 数）、
第三节的所有权表逐条比对；不一致时 `cargo test` 直接失败。

机制层的内部设计看 [`crates/cutile/README.md`](../crates/cutile/README.md)，规格与判据看
[`lld.md`](lld.md)，决策理由看 [`adr/`](adr/)。本文只写**结构**。

## 一、分层

| 层 | 成员 | 职责 | 不允许出现 |
|---|---|---|---|
| L1 基础设施 | `diagnostic`、`rng`、`dtype`、`wire`、`tool` | 错误类型与声明数据树（`data`：构建期语料在运行期的形状，**没有解析器**）、计数器型确定性随机、精度闭集、与领域无关的字节码闭集（CRC-32 / LEB128 / Sink）、静态质量门与回退判定 | 任何运行时领域语义、任何 TOML 解析 |
| L2 数据层 | `prof`、`op`、`prim` | 设备档案与能力闭集、算子声明与模板（布局 / 调度 / 模板词法）及其求和语义代价（`op::einsum`）、原语片段与见证；语料在各自 `build.rs` 里解析成表 | 任何机制代码（拼接、协商、emit）、任何设备知识 |
| L3 内核 | `cutile` | 协商 → 拼接 → emit → 编译执行；`doctor` / `verify` / `bench` / `execution` 共用一条 `pipeline`，报告只是数据、排版在前端 | 设备厂商特判、运行时依赖 |
| L4 运行时 | `tensor`、`storage`、`tokenizer` | 张量元数据、布局搬运、文本边界 | 模型语义、调度策略 |
| L5 模型与训练 | `weights`、`autograd`、`nn`、`shard` | 权重结构与自有 `.u` 容器、反向图与显存计划、层定义与执行侧机制、切分拓扑与集合通信 | 进程 / 服务语义 |
| L6 调度与服务 | `train`、`engine`、`serve` | 训练样本与循环、检查点、连续批与 KV 页账、提交与执行体、设备后端实现（`engine::metal` / `backends`，实现 `cutile` 的 `Backend` 缝）、端点契约与线格式、CPU 模型宿主 | 直接调用 `cutile` 生成 kernel；执行任何业务 |
| L7 入口 | `cli`（bin：`llm` 与 `cutile`） | 组合根：命令行文法、配置解析、各层装配、取值解析与人可读排版；编排入口不定义新类型 | 编排入口新增领域类型；在本层复制内核机制 |

`annotation`（编译期注解）、`fuzz`（模糊测试契约）、`benches`（集中测量容器）与 `testkit`（测试夹具与门禁）是**工具成员**：没有运行时层级，依赖边不参与方向判定。

`testkit` 的库体只出浮点口径、随机数与暂存目录等夹具，跨层对拍与门禁在它的 `tests/` 下运行，
被测内核只出现在 dev-dependencies 里，不进任何产品路径。

## 二、依赖规则

> **依赖只能指向更低的层；同层内只能依赖表中更靠左的成员，不允许反向。**

于是：向上依赖不可能通过（`op` 无法依赖 `cutile`，`tensor` 无法依赖 `engine`）；同层内仍有先后
（`prim → prof`，因为片段文件名必须落在原语闭集内）；环一定被拒绝（层号与表内顺序构成偏序，
严格递减无法成环）；数据层永远不知道机制（`op` / `prof` / `prim` 的依赖只有彼此与 L1）；
语义先于设备（`op` 只写"一条求和表达式 + 操作数位宽"，`flops` / `bytes` 由 `op::einsum` 推出，
`cutile` 的 roofline 只消费这个数对，不认识任何算子名）。

历史的成员合并（七条吸收链、配置层拆分、样本并入 `train`）都已落地，**当前成员数与边数以本节为准**，
不再复述每一次的数字；配置层的现状见 [ADR 0022](adr/0022-build-time-corpora.md) 与
[ADR 0023](adr/0023-settings-to-cli.md)。

允许的边（当前全量，共 110 条，含 42 条 `+dev` 边；工具成员 `annotation` / `fuzz` / `benches` / `testkit` 的边不参与运行时方向判定）：

```
cutile   → diagnostic, op, prim, prof    +dev: testkit
op       → annotation, diagnostic, prof    +dev: testkit（工具边：`#[model]` 注解，不参与方向判定）
prim     → prof, diagnostic
prof     → diagnostic, dtype
dtype    → （叶：精度闭集，无运行期工作区依赖）    +dev: testkit
tensor   → diagnostic, dtype    +dev: testkit
storage  → tensor, diagnostic
tokenizer→ diagnostic
weights  → tensor, diagnostic, dtype, wire, storage（`storage` 仅由 `mmap` feature 可选启用，复用其唯一 unsafe 岛）    +dev: storage, testkit
autograd → tensor, diagnostic    +dev: testkit
nn       → tensor, op, weights, diagnostic, annotation
shard    → diagnostic
train    → nn, autograd, weights, shard, op, tensor, diagnostic, wire    +dev: testkit
engine   → tensor, diagnostic, storage, cutile, op, prof    +dev: testkit
serve    → engine, nn, op, rng, tokenizer, diagnostic, weights, wire    +dev: testkit
cli      → cutile, diagnostic, engine, nn, op, prim, prof, serve, tensor, tokenizer, tool, train, weights（L7 组合根：只做命令分派与各层装配，不定义新类型）
fuzz     → diagnostic    +dev: cli, weights, train, serve, wire, tensor, tool（工具成员；`+dev` 是 `tests/support.rs` 里的驱动对象）
benches  → +dev: prof, op, cutile, wire, weights, tokenizer, tool, testkit, train, autograd, tensor, nn, engine, serve（工具成员：只声明被测对象，测量配置归 criterion）
tool     → diagnostic（配置驱动静态质量门 + 人可读排版 + 回退判定）    +dev: testkit
testkit  → +dev: autograd, cutile, nn, op, tensor, tool, train, weights（工具成员：夹具库 + 门禁；被测层只出现在 dev-dependencies）
wire     → diagnostic（纯字节码闭集，不认识张量或容器布局）    +dev: testkit
rng      → diagnostic（Philox4x32-10，显式种子 / 流号 / 下标派生，不持全局状态、不引 `rand`）    +dev: testkit
diagnostic → （叶：错误类型 + `toml`，被大量成员消费；它自己的出边为零）
```

## 三、所有权表

一个概念只能有一个主人；这张表**双向**校验：表里写的必须在 `crates/<主人>/src` 里有 `pub` 定义，
源码里每个 `pub struct|enum|trait` 也必须在表里（或登记在 `lints/vocabulary.toml` 的
`[[ownership.unlisted]]` 存量基线，且基线不允许过期）。L7 组装点另有单独一条：`src/**/*.rs`
不得定义任何类型。

| 概念 | 唯一主人 | 负责什么 |
|---|---|---|
| `Error` / `Code` / `Result` / `Cursor` | `diagnostic` | 全仓唯一错误类型：类别由产生错误的那一层给，上层只补上下文；`also` 追加同级诊断，`from_io` 区分"缺失"与"读不出"，`exit_code` 是类别到退出码的唯一映射。`Cursor`（`scan` 模块）是带字节位置的逐字节游标：只做读字节 / 跳空白 / 吃必需字节三件事，不认语法、不建树——否则它就成了第二份解析器 。同名局部读取器在 `wire`（LEB）/ `train`（快照）/ `tokenizer` / `cutile` / `annotation` 各有一份，按各自的格式边界读字节，与本行不是同一个类型——合并它们要按格式逐个收，不在这里主张 |
| `Table` / `Value` | `diagnostic` | 声明数据树：构建期生成的代码构造、运行期只读，因此不存在"语法错误"这一类 |
| `Rng` / `Uniform` / `Normal` / `Bernoulli` | `rng` | 计数器型确定性随机与三个构造期校验参数的分布；采样是 `u ∈ [0,1) → 元素` 的纯映射 |
| `DType` / `QType` / `MxType` / `Element` | `dtype` | 全仓唯一精度事实源，三轴分文件（标量 / GGML 块量化 / OCP MX）；只放身份与布局元数据，反量化算法留在 `weights` |
| `Dirs` | `cutile` | 内核要的三个目录（缓存根 / 工作目录 / 可选数据根）——纯数据，不带默认值、不读环境变量、不认识命令行 |
| `Settings` / `Roots` / `SettingValue` | `cli` | 配置键的解析策略：来源分层与追踪、平台默认目录、相对路径基准；键表一行一个，`describe` 与"未登记键"都从同一张表派生 |
| `Expr` / `Extents` / `Operand` / `Axes` / `Bits` / `Cost` / `Buffer` | `op` | 求和表达式的语义：一条式子推出 `flops` 与最少流量；`Buffer` 是内核样本的具名缓冲。不认识设备、tile 或片段 |
| `Op` / `Signature` / `Sample` / `Model` / `Precision` | `op` | 算子声明与语义（签名 / 样本 / 标量参考）；签名是 ABI 的唯一事实源，实参顺序、输出名与输出类型都由它派生；`bind` 是真实数据进入样本的唯一入口；`Precision` 是收缩轴可变 GEMM 的精度契约（元素类型 + 累加器 + 一次乘加），遍历与尺寸校验只写一份 |
| `Format` / `OperandSpec` / `Element` / `FormulaCache` / `Row` / `Conf` / `Operator` | `op` | 打包格式的静态表与派生配置（`Row` 全 `const`，`Conf` 只收骨架运行期要的 `cpb` / `zero` / `offsets`）、表达式缓存、样本缓冲元素闭集，以及六族骨架之上的统一语义契约（[ADR 0050](adr/0050-operator-base.md)） |
| `OpDecl` / `Template` / `Segment` / `Registration` / `LayoutSpec` / `Schedule` / `Masking` | `op` | 声明解析结果与三张闭集（模板词法 / 布局构造子 / 调度模板），以及唯一注册表 `REGISTRY`；`cutile` 只消费，不持第二份 |
| `Scalar` / `Monoid` / `SsmStep` / `PlaneStep` / `Scan` | `op` | β 指数梯形扫描的数学底座与 `scan_f32` 算子：半群、两种离散化（`exp_zero_order` / `exp_trapezoid`）、复值状态的 2×2 实值块 |
| `Profile` | `prof` | 设备档案；机制代码不得出现 `vendor == "..."` |
| `Closed` / `Set` / `KernelKey` | `prof` | 词汇表契约、全仓唯一位集（容量在单态化时由 `const FITS` 断言）、以及一条内核实现的登记键（[ADR 0027](adr/0027-kernel-registry-and-fragment-contract.md) 七元组，只承载身份） |
| `FragmentSet` / `Device` / `Image` / `Status` | `prim` | 原语片段与见证源码、每台设备这次构建带上的设备侧代码、编译产物与其结局；三张内嵌表由 `build.rs` 扫描生成，`src/` 里只有类型 |
| `Catalogue` | `cutile` | 已装载的三层数据（声明 + 档案 + 片段）与它们的交叉校验 |
| `Backend` / `Codegen` / `Driver` / `Readback` / `Source` / `Lower` | `cutile` | 设备后端契约（`Backend = Codegen + Driver`）：`Codegen` 是纯函数，`Driver` 是唯一碰工具链 / FFI 的一半，运行期只经 `dyn` 分派；Metal / CUDA 的 `Driver` 由 `engine` 实现、`cli` 注入；`Lower` 是生产态 lowering 的中性请求 |
| `Plan` / `Path` / `NoPath` / `VariantCheck` / `Proof` / `Source` / `Usage` | `cutile` | kernel 层中间产物，全部可见、无 IR；只有 `NoPath` 自带文本（错误契约）；`Proof` 记这次真正执行过的供给，`Proof::declared` 只给解释性报告 |
| `Toolchain` / `CacheKey` / `Cache` / `Entry` / `State` / `Changed` / `Residue` / `Gc` / `ArtifactKey` / `Artifact` | `cutile` | 缓存指纹、编译产物与存量：`CacheKey::fields` 是哈希与差异比较的唯一来源且**不含工具链**，`ArtifactKey` 含实测工具链与编译参数，参数一变就换一份产物 |
| `Doctor` / `OpInfo` / `DeviceInfo` / `LayoutRow` / `Storage` / `Declared` / `Built` / `CacheWhy` / `CacheState` | `cutile` | `doctor` 的结构化结论：每步产物都是数据，排版在前端 |
| `VerifyReport` / `VerifyEntry` / `Verified` / `Execution` / `WitnessRun` / `Evidence` / `Basis` | `cutile` | `verify` 的逐设备结果与见证执行；失败如实记录，不写成"跳过"；`Evidence` 是三条命令共用的那一份证据 |
| `LedgerCell` / `BackendRow` | `cutile` | `ledger` 跨档案能力台账：只聚合 `witness` 已跑出的证据，不产生新证据 |
| `Negotiation` | `cutile` | 一台设备的协商结论（首档 / 降级 / 无路径 / 无档案）：只读声明与档案，与执行成败分开记 |
| `Fragment` | `cutile` | 一份生成片段的行数与 §16 口径（`FRAGMENT_LIMIT` 上限管成败，`FRAGMENT_GOAL` 只是余量） |
| `Bench` / `Throughput` / `Peak` / `Verdict` / `Performance` | `cutile` | `bench` 的实测口径与达成率判定；口径无效或没有峰值时 `is_measured` 为假 |
| `Tensor` / `View` / `Element` / `Shape` / `Strides` / `Order` / `AxisLen` / `OwnedTensor` | `tensor` | `Tensor` 是**字节**视图，`View<E>` 是**元素**零拷贝视图；`SUPPORTED` 子集表声明本层可构造的精度；设备地址不泄漏到本层 |
| `Layout` / `remap` / `Buffer` / `Allocator` / `CpuAllocator` | `storage` | 物理布局与搬运；`Allocator` 是分配 / 释放 / 显式拷贝的边界 trait，设备后端在边界 crate 实现同一 trait；本层不做隐式跨设备拷贝 |
| `Stream` / `Graph` / `Launch` / `Device` / `Operand` / `OperandRole` / `ScalarSlots` | `engine` | 提交与执行，不编译 kernel；`Device` 只是运行时标识标签，不持私有句柄；`ScalarSlots` 是定长类型化闭集标量槽（[ADR 0043](adr/0043-program-scalar-slots.md)），位置参数不折进 1 元素 f32 缓冲 |
| `Sequence` / `Scheduler` / `Progress` | `engine` | 连续批与序列状态；准入后每生成一个 token 才预留它要落的那页，`Blocked` 时不改任何状态；`Sequence` 是已生成 token 的唯一存放处 |
| `Page` / `Table` / `Cache` / `Pager` | `engine` | 分页 KV 的页管理与页内数据：引用计数 O(1) 判定越界 / 重复归还 / 下溢，归零才回池且只有最后一个持有者归还时清零；行宽由装载模型的一侧给 |
| `Slot` | `engine` | 一个 token 在 KV 缓存里的位置（页号 + 页内偏移），写 kernel 的一侧不必自己再算一遍 |
| `Radix` | `engine` | 满块前缀树：自己就是页的持有者，`evict` 只摘 LRU 叶；页复用与位置账（`Sequence::prefilled`）是同一份 |
| `Batch` / `Admission` | `engine` | 连续批：`emit` 是"记账 + 推进"的唯一入口；`Admission` 让"页不够"成为可重试的 503 而不是请求非法。与 `train::Batch`（数据单位）同名不同物 |
| `Budget` / `Candidate` / `Phase` / `Verdict` | `engine`（`ifb` 模块） | 一次迭代的可行域与相位：`pack` 判"这一轮塞得下几行"，与 `admit` 判"这辈子跑不跑得完"分开；相位是序列自己的事实，不是调用方贴的标签 |
| `Executor` / `Trace` / `Program` / `Bindings` / `Buffer` / `View` / `ViewMut` / `CpuExecutor` | `engine` | CPU 执行体（[ADR 0051](adr/0051-execution-backend-base.md)）：`Program` 按名字登记、重名拒绝，`Bindings` 是编号 ↔ 缓冲的操作数表，`CpuExecutor` 入队并在 `sync` 时就地执行；`Trace` 只记录不改语义 |
| `MetalDevice` / `MetalCommandQueue` / `MetalBuffer` / `MetalPipeline` / `MetalExecutor` | `engine` | 设备后端边界（[ADR 0035](adr/0035-unsafe-ffi-boundary.md) 白名单第五类「进程级库句柄」，越界岛，不单独成 crate）：句柄私有、不跨线程，MSL 由装配层注入，本层不调用 `cutile` |
| `Abi` / `Arch` / `Host` / `Machine` / `Row` / `Standing` / `Why` | `engine`（`platform` 模块） | 六平台执行合同（[ADR 0052](adr/0052-execution-platform-contract.md)）：`Row` 是执行族闭集（CPU / Metal / CUDA / ROCm-HIP / Vulkan-SYCL 合一的 GPU 行，顺序即合同登记顺序），`Standing` 是三态裁决（覆盖 + 证据 或 未覆盖 + `Why`），`Why` 只分 `NoDevice` / `NoToolchain`——两者处置不同（换机器 vs 装工具链），混成一个"未覆盖"就丢了唯一的行动信息。`audit` 每次都出六行（少一行会让"未覆盖"与"根本没问"变成同一输出），`require` 才报 `E-PLAT`；它只判**平台有没有资格跑**，不判内核在某台设备上是否正确（那是 `cutile::verify` 与 `tool::card` 的事）。`Host` 是注入缝（架构 + 该行工具链），`Machine` 是这台机器的实现：架构取 `std::env::consts`、工具链复用 `backends` 那份进程级快照探测（不新开第二次 `--version`），于是测试能用一台假机器摆出九种组合。**不拥有**设备探测实现、设备后端代码（Metal / CUDA 的越界边界见 [ADR 0035](adr/0035-unsafe-ffi-boundary.md)，本层只裁决"能不能跑"）、性能与正确性结论（E1–E15 归协议卡）、以及设备档案本身（那是 `cutile::profile` 的声明层） |
| `Tokenizer` / `Token` / `Bytes` / `Bpe` / `Unigram` / `SpecialTokens` / `Encoding` / `Strategy` / `Truncation` / `Padding` / `PadDirection` / `Fingerprint` / `DecoderStream` / `Vocab` / `VocabSource` / `VocabFamily` / `Merge` | `tokenizer` | 文本 ↔ token 的唯一边界：编号、编码对齐记录、序列策略、词表身份指纹与 UTF-8 边界安全的流式解码 |
| `Adaptive` / `AdaptiveCompensated` / `Header` / `Entry` / `Reader` / `Loader` / `Archive` / `File` / `MmapSource` / `Quantized` / `Scheme` / `Compensated` / `Writer` / `Stored` / `Segment` / `Quant` / `Codec` / `Proof` / `Metrics` | `weights` | 权重文件：外部 safetensors 只读适配（`Reader` 借出 / `Loader` 拷贝两条契约）+ 自有 `.u` 容器 v2 读写（rANS、Merkle 证明、两级 CRC、原子写、现场度量）；`Quantized` 的分组、输入宽度与输出宽度出自同一次推断 |
| `Certified` / `Note` / `Stable` / `Mode` | `weights` | `ledger` 的谱隙证书与它门控的秩选择：判定与产出出自同一次分解，证书写在 `AX02` 的 `rank` 段并由 `audit` 重算，对不上就拒读 |
| `Tape` / `Var` / `Node` / `Requirement` / `Role` | `autograd` | 反向图的记录：变量需求、出身、边与深度 |
| `Schedule` / `Account` / `Ordering` / `Config` | `autograd` | 反向调度（可达剪枝 + 可配置排序 + 同深度分层）与扇出账本；都在原对象上重填，训练循环不必每步新建 |
| `Plan` / `Budget` / `Choice` / `Footprint` / `Sizes` / `MemoryPolicy` | `autograd` | 显存计划：只读 `Tape`、`Schedule` 与形状表，不分配、不执行。与 `cutile::Plan`（单条 kernel 选哪个 tile）不是同一件事 |
| `Replay` / `Store` | `autograd` | 重放执行：把计划编成扁平序列表；`Store` 是唯一口子，真实缓冲归 `train` |
| `Summary` | `autograd` | 图规模与可达性的只读计数快照，供诊断、缓存键与容量预估 |
| `Contract` / `ContractSpec` / `Linear` / `LmHead` / `WeightFormat` / `ParamRole` | `nn` | 层定义与契约校验：产出的算子名必须存在于 `op`；`WeightFormat` 是格式 → 算子名的唯一事实源，`ParamRole` 让权重衰减豁免与可训练判定从角色派生 |
| `RmsNorm` / `Softmax` / `Gelu` / `Silu` / `Rope` / `Scale` / `Add` / `Mul` / `ResidualAdd` / `ReduceSum` / `ReduceMax` / `Embedding` / `Attention` / `AttentionAlgo` / `MatMul` | `nn` | 单算子层：只固定自己绑定的算子名与形状口径；下界仍由算子语义表推出，层里不写第二份流量常数 |
| `AttentionBlock` | `nn` | 注意力组合块：Q/K/V 投影 + 注意力 + 输出投影内聚成一个 `Contract`，子层按 `{块名}.q` / `.k` / `.v` / `.attn` / `.o` 对齐权重条目 |
| `Bounds` | `nn` | 层的性能下界：`read` / `write` / `flops` 三笔按方向分账，不认识设备峰值、不做 roofline 编排 |
| `Stack` / `Floor` / `Unpriced` | `nn` | 层栈与它的总下界；未覆盖与无法计价两类清单都不允许静默丢弃，也不让整张模型算不出下界 |
| `Calib` / `Lif` / `Encoder` / `Events` / `Audit` | `nn` | 决策与脉冲的参考语义：`Calib` 是唯一决策端口（DecisionHead / MoE / 层间路由共用），`Lif` 是漏积分-发放神经元，事件臂只管码位与稀疏度。不拥有判定语汇：PASS/FAIL 仍只由 `tool::card` 出 |
| `Spiking` / `Membrane` | `nn` | 脉冲前向的执行侧（`src/spike.rs`）：`Spiking` 持膜时间常数与阈值，`Membrane` 是每个神经元的状态（电位 + 不应期）；`step` 先校验形状再推进，`drive` 走一次多步发放 |
| `BlockTable` / `Bypass` / `MemoryBlock` / `Channel` / `Gating` / `MissRate` | `nn` | §5.8.2 / §5.8.3 / §1.5–1.6 的执行侧：块表因果寻址、条件记忆旁路与可装配的 `MemoryBlock`、α/β/γ 通道与一步门控、`ρ_t` 的因果窗口；三者不持参数、不注册算子 |
| `Mla` / `Latent` / `Path` | `nn` | MLA 的 KV 记账从公式变成结构：`Latent` 只有 `c^{KV}` 与解耦 RoPE 键，两条路径按 §5.8.1 逐元素对拍；不拥有分页与批调度 |
| `Moe` / `K3_EXPERTS` / `K3_ACTIVE` | `nn` | 专家侧：路由走唯一端口 `policy::Calib`，选中集重新归一，共享专家恒参与且不占 top-k 名额；不拥有门控标量、专家参数与训练状态 |
| `Mhc` / `MHC_STREAMS` | `nn` | 残差侧：双随机矩阵重组残差流，非负性是入口硬条件（否则"实测 ≤ 1"会假通过）；不拥有门控标量、层参数与训练状态 |
| `Hybrid` / `Query` | `nn` | 装配：一层 = 层首混流 → MLA 注意力残差 → MoE 前馈残差，三件都调各自模块的唯一入口；增量只加回主路（同一份增量加给每条流会让多流退化成单流），缓存只读、失败不回写；不拥有投影矩阵、归一化算术、KV 页账与训练状态 |
| `Tower` / `State` / `Kind` / `MemoryStep` / `MEMORY_PER_ATTENTION` | `nn` | 层栈装配：每三个通道记忆层配一个混合层（配比是构造条件，不是注释），层序由 `Kind` 逐格读出；记忆状态与 KV 缓存分离（`State` 随序列长累积、`Latent` 按页账走），一次推进要么整栈走一步要么一格不动；不拥有投影、页账与训练状态 |
| `Kda` / `Memory` | `nn` | 混合 delta 轴：纠错式写入（先减去记忆已预测的部分），逐通道门控；不拥有投影、层配方与训练状态 |
| `Mamba` / `Carry` | `nn` | 指数梯形状态空间：状态存成 `2N` 个实值分量、低秩 MIMO、读数取本 token 更新之后的状态；与 delta 规则判据正交，两份实现互不借用语义 |
| `AttnRes` / `Sources` / `splice` | `nn` | 深度轴：按注意力取深层残差（凸组合），`Sources` 按层号记来源，`splice` 保证并回是凸组合 |
| `Boundary` / `Patches` | `nn` | byte-latent 编码侧：边界规则闭集（参数随变体走），`Patches` 只存起点与总字节数，因此"区间首尾相接"由构造保证 |
| `ByteEncoder` / `ByteDecoder` | `nn` | byte-latent 的表示侧与解码侧：哈希 N-gram 桶向量 + patch 内注意力池化，一次投影给出 256-way 分布与贪心字节 |
| `Emission` / `Emitter` / `Halt` | `nn` | byte-latent 生成侧：把边界 / 编码 / 解码串成一次生成，条件只取**已收口** patch；`Halt` 是闭集，`None` 说不清"截断"与"空结果" |
| `Global` / `Backread` / `Layer` | `nn` | byte-latent 的潜在栈：`Global` 的遍数就是 patch 数，`Backread` 的可见集由 `Patches` 给出，`Layer` 的宽度由栈校验 |
| `Group` / `Reduce` / `Collective` / `Topology` / `Parallel` / `Partition` | `shard` | 集合通信契约与 TP/PP/DP 拓扑；`world = tp · pp · dp` 在构造期算成精确值，因此不存在回绕出来的"合法" rank |
| `Sharding` / `Tiling` | `shard` | 张量并行的两种线性层切法与一层权重的切分计划；切分必须整除，在构造期失败而不是摊余数 |
| `Local` | `shard` | 进程内集合通信：每个 rank 一份镜像，让 TP/DP 的数值语义在没有 NCCL 的机器上也能被 `cargo test` 验证 |
| `Remote` | `shard` | 远端集合通信占位：同一 `Collective` 端口，`all_reduce` 返回 `Unsupported` 且不改动调用方缓冲 |
| `Format` / `Facts` / `Snapshot` | `train` | 检查点字节格式与恢复事实：字节只写相对上一条的差、尾部 CRC-32；`verify` 只验字节，`check_weights` 要求条目表与权重文件逐条相同 |
| `Schedule` / `Anneal` / `Step` / `Session` / `TrainConfig` / `ParamSlot` / `Checkpoint` / `Distributed` | `train` | 训练循环状态：纯函数学习率调度与衰减曲线、一次会话复用同一张图 / 缓冲 / 优化器、参数槽与分布式装配（后端契约在 `shard`） |
| `Batch` / `Dataset` / `Chunks` | `train`（`samples` 模块） | 训练样本：借用调用方 token 缓冲（零分配 / 零拷贝），两个构造函数共用同一份几何校验，批总数在构造期算好，`Dataset` 对象安全 |
| `Optimizer` / `Sgd` / `Adam` / `Group` / `Slot` / `OwnedSlot` / `Moments` / `SlotMoments` / `GradAccum` | `train` | 优化器：对象安全 trait，矩状态按槽位编号自持；参数组由 `ParamRole` 派生，不按字段名猜 |
| `Muon` | `train` | 矩阵参数的近似正交化更新：更新量只由方向决定，非矩阵槽**拒绝**而不是悄悄跳过 |
| `Buffers` / `Kernel` / `Reference` / `Data` / `View` / `ViewMut` / `Kind` / `Lane` / `Sealed` / `Inputs` / `OperatorSetOps` | `train` | 激活缓冲：`autograd::Store` 在训练侧的落地（按计划分配与交还、按等长空闲表复用），算术交给 `Kernel`；本层不定义算子、不碰 `unsafe` |
| `MetricPoint` / `MetricsBoard` / `Learner` / `ForwardOut` | `train` | 评估与编排：`Learner` 驱动"前向 → 评估入账 → 装梯度 → 优化步"闭环，模型拓扑与梯度来源由闭包注入 |
| `Distill` / `Grpo` / `Loss` / `Metric` / `Outcome` | `train` | 损失与度量闭集：分类 CE / Hinton 蒸馏 / GRPO 三臂，配置随度量走；都是 train 侧纯函数，不进 `op` 的设备核注册表 |
| `Envelope` / `Frame` / `Decoder` / `Kind` / `Version` | `serve`（`protocol` 模块；`proto` 为兼容重导出路径） | 端点契约的线格式：版本化信封、零拷贝帧视图与增量解码（帧布局 = 10 字节头 + 载荷 + v2 起的 CRC-32 尾，见 `serve::protocol` 的 `//!`） |
| `Route` / `Endpoint` / `Response` | `serve` | 端点契约：`Endpoint::call` 的 `Err` 只留给"服务端自己坏了"，请求侧不合法必须落成 `Response::error(BadRequest, …)` |
| `Status` / `Request` | `serve` | 响应状态闭集与请求文法：未知键、重复键、缺字段与坏数字都是 `invalid`；停止符不在请求里——那是采样策略的事 |
| `Predict` / `Sampling` / `Greedy` / `Classical` / `Multinomial` / `SamplingParams` / `Generation` / `Ending` | `serve` | 生成的协作者与结局：给出一步 logits、选一个 token、说明为何停下；端点不认识模型，模型也不认识页账 |
| `Completion` / `Router` | `serve` | 第一条真实端点：解析 → 编词 → 准入 → 逐步前向 + 采样 → 归还页 → 包帧；客户端错误落 400、页不够落 503，失败路径同样归还页 |
| `Verdict` / `accept` / `Drafter` | `serve`（`speculate` 模块） | 投机解码的接受判定：取最长一致前缀，第一个不一致的位置由目标模型补一个；只判"接受几个"、不写 KV |
| `Httpd` | `serve` | std-only HTTP/1.1 薄适配器：单线程一问一答，固定 `Connection: close`；换传输层只需在同一个 `Router` 上再挂一个适配器 |
| `CpuModel` / `Params` / `Block` | `serve` | CPU 模型宿主与驻留参数表：装载期一次解码、生成循环不再读文件；七个层族装配成 `engine::Graph`，执行器泛化为 `E: Executor`；`from_path` 按 `Reader` / `Loader` 在装配点分流 |
| `Command` / `Invocation` / `USAGE` | `cli` | 命令行文法（argv / env → 强类型调用参数）与编排分派 |
| `Report` / `join_names` | `tool` | 人可读输出；只接字符串，不接领域类型 |
| `Token` / `Pos` / `Delim` / `Finding` / `Gate` | `tool` | 零依赖词法 token 树、单函数复杂度 finding 与配置化阈值；分析器只产出数据，不打印不退出 |
| `Case` / `Baseline` / `Judgement` | `tool` | 回退判定（与基线相比有没有退）；不做测量 |
| `EntryKind` / `Harness` / `Roofline` / `Geometry` / `BenchRecord` | `tool` | 统一基准记录结构：只持有记录，不测量、不下性能结论，未测得的峰值留 `None` |
| `Protocol` / `Statistic` / `Tally` / `Standing` / `Shortfall` / `Card` | `tool` | 实测卡三态判定：闭集 → 样本量 → 统计量定义 → 区间与门槛，次序有意；样本与时钟来源归 `benches` |
| `spec` / `model` 属性宏 | `annotation` | 把注解展开成契约实现 + 编译期断言 + 一条校验测试；**不持有运行期注册表**。`model` 生成 `MODEL` 并断言注解与 `Operator` 对齐；设备执行由 `cutile` 的 `const SPECS` / `const GATES` 描述符表按函数指针登记 |

几个**同名不同物**的邻居，评审时容易混淆：

- `cutile::space::Space`（GM / FAST / REG）是**语言概念**，`prof::atom::Space` 是它的档案表示；档案侧只认字节；
- `cutile::bench::Verdict`（达成率：能否与档案峰值比较）、`tool::regress::Judgement`（相对基线有无回退）、
  `tool::card::Standing`（这一格成立了没有）三态名不同是刻意的，它们会同时出现在一份报告里；
- kernel 侧操作数视图与 `tensor::Tensor` 是两回事：前者不得依赖 `tensor`，后者不得假装能做 kernel 分区。
  该视图在 P0 **尚未实现**——`cutile` 只接收裸指针与标量，分片由调用方给；
- `nn::LmHead` 这类注解声明与 `Linear::new` 这类运行期构造走同一条 `ContractSpec::new` 校验路径，
  注解不建注册表（否则就是第二份事实源，见 [ADR 0005](adr/0005-annotations-as-contracts.md)）。

## 四、闭集与位集

「限定取值范围的词汇表」（原语、精度、存储层、属性键、证据等级、输出语言、骨架）必须回答同一组问题：
**全量是什么、跨层名字是什么、位集下标是什么**。写三遍就是三份会漂移的事实源，因此由一个契约收口：

- `impl Closed for X { const ALL; fn as_str; fn index; }` 是**唯一**声明处，`parse` 由 `ALL` + `as_str` 派生；
- 已实现 `dtype::Format` 的精度类型由一条 blanket 接上，`Display` / `FromStr` / 可选 `serde` 名字
  都取自同一张行表，不存在"序列化一套、显示一套"；
- `prof::inherent_names!(X)` 生成同名固有转发，调用点写法不变、逻辑仍只有一份；
- `Set<E, WORDS>` 是唯一位集：容量在单态化时由 `const FITS` 断言，"加第 65 个原语"是编译错误；
- 契约不可达的成员（`diagnostic` / `tensor` / `shard` / `serve` 在 `prof` 之下，反向依赖会成环）
  可以把词汇表留在本地，但必须登记在 `lints/vocabulary.toml` 并写明理由。

由 `crates/testkit/tests/contracts.rs` 执行：禁止新增手写 `const ALL` 与手写位集，禁止白名单过期，
并要求扫描到的源文件数不为零（避免"空门禁假绿"）。

#### 注解里的名字也是词汇表的名字

属性宏收字符串（`dtype = "f32"`），词汇表里是变体，`codegen` 按 `DType::variant` 转换。三处必须同时
成立：`annotation` 的 `DTYPES` 与 `cutile::reference` 给出非零阈值的 `DType` 集合逐项**相等**；
注解调用点生成的 `#[cfg(test)]` 把字面量与变体、`Model::dtype` 与 `.k` 的可写输出类型对齐；变体名
取 `DType::variant` 而不是在注解里手写映射。`annotation` 只依赖 `dtype` 这个 L1 零依赖基座，
`syn` / `quote` / `proc-macro2` 不进入供应链。

#### 语义表的唯一生产者

`#[annotation::model(name = .., shape = [..], dtype = ..)]` 标在 `type 名字 = <具体格式>;` 上，生成
`MODEL`、`const _: () = <名字 as Format>::CHECK;` 与一条校验测试；注册表里的 `model:` 只能写
`..::MODEL`。两侧互为镜像，由 `crates/testkit/tests/contracts.rs` 判，且**没有豁免节**：标了却不注册
永远错（发布出去的二进制里没有这条算子），注册的语义表不是注解生成的也永远错——出现例外说明契约
本身该改，而不是某个 crate 该豁免。

## 五、占位策略

一个 crate 还没实现时，它占的是**接口形状**，不是 `add(2, 2)`：类型、trait、字段名与错误语义先定，
函数体后填。判断标准是"删掉它会谁编译失败"——如果谁都不失败，它就该从 `members` 里移出去，而不是
留在 workspace 里复制同一个占位函数。

## 六、新增一个成员的三步

1. `crates/<name>/Cargo.toml`：`version/edition/rust-version/license/repository/readme` 全部
   `.workspace = true`，写 `description`、`publish = false`、`[lints] workspace = true`；
2. `src/lib.rs` 第一行起写 `//!` 文档：**拥有什么、不拥有什么、依赖谁、被谁依赖**；
3. 在第一节层表登记、在第二节边清单加边、在第三节所有权表登记新类型——删掉它会有人编译失败，
   才叫模块。

自查口径：`Cargo.toml` 成员与层表一一对应；每条边指向更低的层（同层更靠左）；每个成员至少一条
依赖边（工具成员豁免）；全部成员 `publish = false` 且 `description` / `readme` 齐备；仓库只有一个
`target/`。

## 七、这份契约怎么被守住

| 手段 | 位置 | 挡住什么 |
|---|---|---|
| `clippy -D warnings` | CI `clippy` job | 缺元数据、缺文档、`unsafe` 风格 |
| `RUSTDOCFLAGS=-D warnings cargo doc` | CI `doc` job | 文档坏链接、未文档化的公开项 |
| `cargo test --workspace` | CI `test` job | 行为回归 |
| `crates/testkit/tests/architecture.rs` | CI `test` job | 成员集合、边集合与方向、所有权表与 `cargo metadata` / 源码不一致；`op` / `prof` / `prim` / `cutile` 退回 `Result<_, String>` |
| `crates/testkit/tests/contracts.rs` + `lints/vocabulary.toml` | CI `test` job | 闭集 / 位集被手写第二份、白名单过期、注解与注册表不互为镜像 |
| `#[model]` 生成的两个 `const` + 一条测试 | 被标注 crate 的 `cargo test` | 格式自证不成立（粒度 ≤ 0、派生轴除数 ≠ 粒度）；注解的 `dtype` 与 `.k` 可写输出不是同一个类型 |

`cargo metadata` 只能告诉你"谁依赖谁"，不能告诉你"方向对不对"——后者由本文的层表定义，再由
`architecture.rs` 执行。新增成员或调整依赖方向时，**先改文档再跑 `cargo test`**：不一致会直接失败，
不再依赖评审清单。
