# 架构契约

这份文档回答三个问题：**谁属于哪一层、允许依赖谁、每个类型归谁**。它不是评审清单，而是
**可执行的契约**——[`crates/testkit/src/architecture.rs`](../crates/testkit/src/architecture.rs)（事实面与断言按主题分在 `src/` 下的 `layer.rs` / `ownership.rs` / `boundary.rs`，随本 crate 的单测一起跑）
读 `cargo metadata` 与源码，把下面第一节的成员表、第二节的边集（含总数与 `+dev` 数）、
第三节的所有权表逐条比对；不一致时 `cargo test` 直接失败。

机制层的内部设计看 [`crates/cutile/README.md`](../crates/cutile/README.md)，规格与判据看
[`lld.md`](lld.md)，决策理由看 [`adr/`](adr/)。本文只写**结构**。

## 一、分层

| 层 | 成员 | 职责 | 不允许出现 |
|---|---|---|---|
| L1 基础设施 | `diagnostic`、`dtype`、`wire`、`tools` | 错误类型与声明数据树（`data`：构建期语料在运行期的形状，**没有解析器**）、精度闭集、与领域无关的字节码闭集（CRC-32 / LEB128 / Sink / 对齐落位 / `Invalid` 构造点）、静态质量门与回退判定 | 任何运行时领域语义、任何 TOML 解析 |
| L2 数据层 | `prof` | 算子契约与设备档案同住一个 crate。契约侧出签名 / 样本 / 语义表 / 求和表达式 / 模板词法，以及 `atom` 这一层词汇（原语、存储位、数据类型、属性键、证据等级、需求与供给的代数 `Need` / `Provide` / `Span` / `KernelKey`）；档案侧出声明层与模板正文：`OpDecl` 解析、`catalogue` 的 57 条算子模板、六台设备的原语片段与见证（一台设备一个模块，`src/<设备>/` 放语料，`build.rs` 一次把它解析成每台设备的 `FRAGMENTS` / `PRIMS` / `EPILOGUE` / `DEVICE` 表，并在编译期拼成一张聚合表，按 `Layout` 决定读内嵌表还是读盘）。两者同住是因为它们是**同一条文法**：档案声明"这台设备能拼出 `dot`"，片段声明"`dot` 这么写"，模板与签名就是两边的共同语言——拆成两个 crate 只会多一层 `pub use` 转发，一行代码也不会少 | 契约/档案子域不得出现机制代码（拼接、协商、emit）、设备知识与厂商内核正文；主机 CPU 微内核本体只允许收在 `native` 受控岛（arch 门 / 特性探测 / `unsafe` 由 unsafe 岛白名单钉死），其余子域不持内核正文 |
| L3 内核 | `cutile` | 纯机制：协商 → 拼接 → emit → 缓存指纹，一行也不执行。只产设备源与指纹，换掉工具链不影响它；执行侧（工具链探测、编译、验见证、测速）随原 `optim` 并入 `shard`，因为那部分要碰编译器与设备，换掉执行方式不该动框架 | 设备厂商特判、运行时依赖 |
| L4 运行时 | `tensor`、`storage`、`tokenizer` | 张量元数据、布局搬运、文本边界、检查点字节格式（`snapshot` 折进 `storage`：`Snapshot<Manifest>` 泛型，字节只写相对上一条的差 + CRC-32 尾部，不认识训练循环） | 模型语义、调度策略 |
| L5 模型与训练 | `weights`、`autograd`、`nn`、`algorithm`、`shard` | 权重结构与自有 `.u` 容器、外部权重格式只读适配（torch `.bin` 的 ZIP+pickle 栈机、GGUF、JSON，只吐四要素不判语义）、反向图与显存计划、层定义与内核族（`nn` 同时拥有内核装配、神经算子与唯一注册表 `REGISTRY`——注册表必须持有内核类型才能装进语义表，故它与内核同住）、架构变体的参考语义（不产出 `ContractSpec`）、切分拓扑与集合通信、以及原 `optim` 的编译执行面 | 进程 / 服务语义 |
| L6 调度与服务 | `train`、`engine`、`device`、`serve` | 训练样本与循环、检查点事实集、连续批与 KV 页账、提交与执行体、设备源 lowering 与厂商后端（`device::backends` / `device::metal::host` / `device::metal::fold` / `device::cuda` / `device::dialect` / `device::platform`，实现 `cutile` 的 `Backend` 缝与 `engine` 的 `Executor` 缝；`device::metal::executor` 是唯一 `unsafe` 岛（[ADR 0035](adr/0035-unsafe-ffi-boundary.md)））、端点契约与线格式、CPU 模型宿主 | 直接调用 `cutile` 生成 kernel；执行任何业务 |
| L7 入口 | `cli`（bin：`llm` 与 `cutile`） | 组合根：命令行文法、配置解析、各层装配、取值解析与人可读排版；编排入口不定义新类型 | 编排入口新增领域类型；在本层复制内核机制 |

`annotation`（编译期注解）、`fuzz`（模糊测试契约）、`benchmarks`（集中测量容器）与 `testkit`（测试夹具与门禁）是**工具成员**：没有运行时层级，依赖边不参与方向判定。
机制层与模型层之间另有一条 dev 环：模型层的写回尾块测试要真的装配并执行，装配器在机制层；机制层的计划测试要拿真的语义表，语义表在模型层。它只存在于测试图里——产品路径上机制层不认识模型层。

`testkit` 的库体只出浮点口径、随机数与暂存目录等夹具，跨层对拍与门禁在它的 `tests/` 下运行，
被测内核只出现在 dev-dependencies 里，不进任何产品路径。

## 二、依赖规则

> **依赖只能指向更低的层；同层内只能依赖表中更靠左的成员，不允许反向。**

于是：向上依赖不可能通过（`prof` 无法依赖 `cutile`，`tensor` 无法依赖 `engine`）；同层内仍有先后
（片段文件名必须落在 `prof::atom` 的原语闭集内）；环一定被拒绝（层号与表内顺序构成偏序，
严格递减无法成环）；数据层永远不知道机制（`prof` 的依赖只有彼此与 L1）；
语义先于设备（`prof` 只写"一条求和表达式 + 操作数位宽"，`flops` / `bytes` 由 `prof::einsum` 推出，
`cutile` 的 roofline 只消费这个数对，不认识任何算子名）。

历史的成员合并（七条吸收链、配置层拆分、样本并入 `train`）都已落地，**当前成员数与边数以本节为准**，
不再复述每一次的数字；配置层的现状见 [ADR 0022](adr/0022-build-time-corpora.md) 与
[ADR 0023](adr/0023-settings-to-cli.md)。

**合并判据是「同构」，不是「消费者数量」。** 一个成员只有一条入边时，它通常是边界而不是肥肉：`storage` 独自持有全仓唯一的 `unsafe` 岛（`tensor` 是 `deny(unsafe_code)` 且有 11 个消费者），`algorithm` 刻意不产 `ContractSpec`（`nn` 专产），`shard` 坐 L5 而 `train` 坐 L6。可合并的硬信号只有一个——两个成员的 `lib.rs` / `build.rs` 除专名外**逐字相同**：六台设备 crate 并入 `prof`（原 `frag`）时按这条判据删掉 512 行、补回 55 行，净省 457 行。反面的账同样要算：一次合并能省的至多是该 crate 的 `Cargo.toml`（全仓 1060 行 / 33 个，约 32 行）加一段头注释，指望靠合并兑现「代码大量减少」是算错账。裁决过程与测量数字见 [lld.md](lld.md) v3.0.90。

允许的边（当前全量，共 124 条，含 55 条 `+dev` 边；工具成员 `annotation` / `fuzz` / `benchmarks` / `testkit` 的边不参与运行时方向判定）：

```
diagnostic → （叶：错误类型 + `toml`，被大量成员消费；运行时出边为零；`+dev` 只登记夹具）
dtype      → （叶    +dev: testkit（精度闭集）
wire       → diagnostic    +dev: testkit（叶：纯字节码闭集，不认识张量或容器布局）
tools      → diagnostic    +dev: testkit（配置驱动静态质量门 + 回退判定）
prof       → diagnostic, dtype    +dev: testkit（契约词汇 + 档案 + 声明层 + 模板正文；片段与见证的语料在这里，机制也只有它）
cutile     → annotation, diagnostic, prof    +dev: nn, testkit（机制：协商 → 拼接 → emit → 指纹，一行也不执行）
tensor     → diagnostic, dtype    +dev: testkit（`dtype` 之上的形状与索引代数；`diagnostic` 只用于构造期 `Invalid`）
storage    → diagnostic, tensor, wire    +dev: testkit（扁平缓冲与分配，以及折进来的检查点字节格式）
tokenizer  → diagnostic（词表与分词）
weights    → diagnostic, dtype, storage, tensor, wire    +dev: storage, testkit（`storage` 由 `mmap` feature 可选启用，复用其唯一 unsafe 岛）
autograd   → diagnostic, tensor    +dev: testkit（反向图与显存计划）
nn         → annotation, diagnostic, cutile, prof, tensor, weights    +dev: testkit（内核装配 + 神经算子 + 唯一注册表；`cutile::native` 是到 `prof::native` 的零成本转发面，nn 经它上提 i8/sme 探针与瓦片几何，不再直引 prof 内核层）
algorithm  → diagnostic, nn    +dev: prof, testkit（架构变体的参考语义：产出张量流而非 `ContractSpec`，故不认识 `tensor` / `weights`）
shard      → cutile, diagnostic, nn, prof    +dev: testkit（切分拓扑、集合通信与原 `optim` 的编译执行面）
train      → autograd, diagnostic, nn, prof, shard, storage, tensor, weights    +dev: annotation, testkit
engine     → diagnostic, storage, tensor    +dev: testkit
device     → cutile, diagnostic, engine, prof, shard    +dev: nn, tensor, testkit（设备源 lowering 与厂商后端（`metal::executor` 是唯一 unsafe 岛）；`nn` 只在 dev 图里供对拍参考核与注册表，生产面不认识算子层）
serve      → diagnostic, engine, nn, train, tensor, tokenizer, weights, wire    +dev: testkit（形状积只经 `tensor::product` 的饱和乘法：回绕出的偏小数会骗过权重长度核对）
cli        → cutile, device, diagnostic, engine, nn, prof, serve, shard, tensor, tokenizer, train, weights    +dev: storage（L7 组合根：只做命令分派与各层装配，不定义新类型）
annotation → （叶：工具边，`#[model]` 注解，不参与方向判定）
fuzz       → diagnostic    +dev: cli, cutile, serve, storage, tensor, testkit, tools, train, weights, wire（工具成员；`+dev` 是 `src/drivers.rs` 里的驱动对象与 `registry()` 登记表，合法种子经 `#[path]` 编入 `src/seeds.rs`，`src/fuzz_*.rs` 只剩一行入口）
benchmarks → （叶    +dev: autograd, cutile, engine, nn, prof, serve, storage, tensor, testkit, tokenizer, tools, train, weights, wire（工具成员：只声明被测对象，测量配置归 criterion）
testkit    → （叶    +dev: autograd, cutile, nn, prof, tensor, tools, train, weights（工具成员：夹具库 + 门禁；被测层只出现在 dev-dependencies）
```

## 三、所有权表

一个概念只能有一个主人；这张表**双向**校验：表里写的必须在 `crates/<主人>/src` 里有 `pub` 定义，
源码里每个 `pub struct|enum|trait` 也必须在表里（或登记在 `.config/lints/vocabulary.toml` 的
`[[ownership.unlisted]]` 存量基线，且基线不允许过期）。L7 组装点另有单独一条：`src/**/*.rs`
不得定义任何类型。

| 概念 | 唯一主人 | 负责什么 |
|---|---|---|
| `Error` / `Code` / `Result` / `Cursor` | `diagnostic` | 全仓唯一错误类型：类别由产生错误的那一层给，上层只补上下文；`also` 追加同级诊断，`from_io` 区分"缺失"与"读不出"，`exit_code` 是类别到退出码的唯一映射。`Cursor`（`scan` 模块）是带字节位置的逐字节游标：只做读字节 / 跳空白 / 吃必需字节三件事，不认语法、不建树——否则它就成了第二份解析器 。同名局部读取器在 `wire`（LEB）/ `train`（快照）/ `tokenizer` / `cutile` / `annotation` 各有一份，按各自的格式边界读字节，与本行不是同一个类型——合并它们要按格式逐个收，不在这里主张 |
| `Table` / `Value` | `diagnostic` | 声明数据树：构建期生成的代码构造、运行期只读，因此不存在"语法错误"这一类 |
| `Panel` / `Sheet` / `Align` / `join_names` / `human_bytes` / `per_run` | `diagnostic`（`render` 模块） | 人读渲染：带标题的两列面板、Markdown 表（表头 / 分隔行 / 逐列对齐只在这里拼）、顿号列表、1024 进制字节与单次耗时；只接字符串与数，不做判定 |
| `Rng` / `Uniform` / `Normal` / `Bernoulli` | `train` | 计数器型确定性随机与三个构造期校验参数的分布；采样是 `u ∈ [0,1) → 元素` 的纯映射 |
| `LoaderConfig` | `train` | 批装配配置：行列几何与采样三态（顺序 / 种子洗牌 / 权重有放回）一次性收口到构造期校验，热路径不再做合法性分支；洗牌与加权随机流只由种子派生 |
| `DType` / `QType` / `MxType` / `Element` / `FloatLayout` | `dtype` | 全仓唯一精度事实源，三轴分文件（标量 / GGML 块量化 / OCP MX）；只放身份与布局元数据，反量化算法留在 `weights` |
| `Dirs` | `cutile` | 内核要的三个目录（缓存根 / 工作目录 / 可选数据根）——纯数据，不带默认值、不读环境变量、不认识命令行 |
| `Settings` / `Roots` / `SettingValue` | `cli` | 配置键的解析策略：来源分层与追踪、平台默认目录、相对路径基准；键表一行一个，`describe` 与"未登记键"都从同一张表派生 |
| `Expr` / `Extents` / `Operand` / `Axes` / `Bits` / `Cost` / `Buf` | `prof` | 求和表达式的语义：一条式子推出 `flops` 与最少流量；`Buf` 是内核样本的具名缓冲。不认识设备、tile 或片段 |
| `Signature` / `Sample` / `Model` | `prof` | 算子声明与语义（签名 / 样本 / 标量参考）；签名是 ABI 的唯一事实源，实参顺序、输出名与输出类型都由它派生；`bind` 是真实数据进入样本的唯一入口 |
| `Op` | `prof` | 算子的注册身份：注册名与可微性由 `Operator` 派生，模板正文归目录，注册表是它唯一的投影面 |
| `Capability` | `prof` | 后端能力表：一条登记「一副骨架族支持哪个元素档」，`supports` 按骨架族 × 元素档两维查表，缺项返回 `false` 由上层按缺的是哪一族、哪一档定位，不回泛化 `Unsupported`；第三维 `backend` 留待接 `Executor`/后端真实实现逐条对账后再补，本层不认识设备字面量 |
| `Precision` | `nn` | 收缩轴可变 GEMM 的精度契约（元素类型 + 累加器 + 一次乘加），遍历与尺寸校验只写一份 |
| `OperandSpec` / `FormulaCache` / `FormulaOwner` / `Row` / `Operator` | `prof` | 打包格式的静态表（`Row` 全 `const`）、表达式缓存，以及六族骨架之上的统一语义契约（[ADR 0050](adr/0050-operator-base.md)）。`FormulaOwner` 是一条求和表达式的唯一载体：`EXPR` / `OPERANDS` / `FRAME` 加**每个实现一份**的缓存单元格——写在泛型骨架里的 `static` 只有一份，先解析的式子会覆盖后面的，故缓存按实现分而不是按骨架分 |
| `Format` / `Element` / `Conf` | `nn` | 打包格式的派生配置（`Conf` 只收骨架运行期要的 `cpb` / `zero` / `offsets`）与样本缓冲元素闭集 |
| `Template` / `Segment` / `LayoutSpec` / `Schedule` / `Masking` | `prof` | 声明侧的三张闭集：模板词法、布局构造子与调度模板；`cutile` 只消费，不持第二份 |
| `OpDecl` | `prof` | 声明解析结果（名字 + 槽位 + 档位 + 供给）：只由 `src/ops/` 语料解析出来，不认识内核类型 |
| `Registration` | `prof` | 注册表 `REGISTRY` 的一行：名字 + 内核类型 + 语义表构造器。行形状与 `Op`/`TEMPLATES` 同一张表，故住声明层；`nn` 只按模板名填表，`cutile` 只消费 |
| `Phase` / `Seam` / `Fusion` | `prof` | 槽位代数：`compose` 判定生产者能否拼到消费者前面——`Phase` 是一段「读哪些槽位 → 写哪些槽位」，`Seam` 是两侧同类型的一处缝，`Fusion` 是拼接后的整体；只做判定，不碰设备源 |
| `Scalar` | `nn` | 元素运算超 trait（`Add` / `Sub` / `Mul` / `Div`）：让泛型内核写 `a + b` 而不是逐方法调用 |
| `Probe` | `nn` | 端口峰值探针——**一条指令路径**的分母：组名前缀 / 通道后缀 / 一轮 MAC 数 / 本机跑一轮的 `run` 闭包四字段是它与测量容器之间的全部协议 |
| `Monoid` / `SsmStep` / `PlaneStep` / `Scan` | `nn` | β 指数梯形扫描的数学底座与 `scan_f32` 算子：半群、两种离散化、复值状态的步进；语义表注解是唯一对拍口径 |
| `Profile` | `prof` | 设备档案；机制代码不得出现 `vendor == "..."` |
| `Device` / `Image` / `Status` | `prof` | 一台设备这次构建带上的设备侧代码、编译产物与其结局（`Source` / `Compiled` / `NoRecipe` / `Skipped`）；放在最底层是为了让六台设备的片段与档案都能引用同一份类型而不成环 |
| `WitnessLookup` | `prof` | 「见证文件名 → 磁盘路径」的唯一缝：档案校验只问这一个方法，因此内嵌根（多 crate）与数据根（单目录）两种布局共用同一条校验路径 |
| `Closed` / `Set` / `KernelKey` / `Epilogue` / `ActKind` | `prof` | 词汇表契约、全仓唯一位集（容量在单态化时由 `const FITS` 断言）、以及一条内核实现的登记键（[ADR 0027](adr/0027-kernel-registry-and-fragment-contract.md) 七元组，只承载身份）。`Epilogue` 是写回尾块闭集（顺序即施加顺序，只有 `store` 有副作用），`ActKind` 是 `act:<kind>` 的参数闭集且与 `nn::kernel::unary` 的执行核一一对应：两者只描述「做了什么」，算术仍归 `nn`。五个名字都在 `crate::atom` 里由 `closed!` 宏展开成 `pub enum`，源码没有那行 `pub`——所有权门禁按宏块参数行的行首标识符认名，两向扫描共用 `public_type_names` 这一份口径 |
| `FragmentSet` / `Fragment` / `Layout` | `prof` | 原语片段与见证源码的读入形状（文件名落在原语闭集内）、以及「内嵌根 / 数据根」二选一的布局；六张内嵌表由同一个 `build.rs` 生成 |
| 片段文本 + 见证 | `prof` | 一台设备一个模块（`src/<设备>/`）：片段文本、见证与生成器给出的 `FRAGMENTS` / `PRIMS` / `EPILOGUE` / `DEVICE` 表都由该成员拥有，不认识机制侧的厂商特判。六台设备共用一份 `Cargo.toml` / `build.rs` / 测试目标，是因为它们的机制逐字相同——差别只有 vendor 字符串与片段正文 |
| `Native` / `Tile` / `Gram` / `Gemm` / `GEMM_KERNELS` / `gemm_negotiate` / `with_panels` / `threads_override` / `align_up` | `prof` | `native` 主机原生微内核岛：`prof` 内唯一允许 `cfg(target_arch)` / 特性探测 / `unsafe` 的子域，边界由 unsafe 岛白名单钉死。`Native` 登记一条编译期内核（名字 / 瓦片形状 / 入口），`Tile` 是 f32 8×8 微内核的坐标描述，`Gram` 是 i8 4×4 列块的坐标描述，`Gemm` 是内核入口签名，`GEMM_KERNELS` 是编译期通道表，`gemm_negotiate` 是唯一协商入口；`with_panels` / `threads_override` / `align_up` 是设备无关公共件。`nn` / `cutile` 只经协商面问「这块谁能算」，不写 arch/feature 门与汇编片段 |
| `Catalogue` | `cutile` | 已装载的三层数据（声明 + 档案 + 片段）与它们的交叉校验 |
| `Kernel` / `Write` / `Parameter` / `ValueType` / `ScalarType` / `Access` / `Expr` / `Arith` / `VerificationError` / `Pass` / `Pipeline` | `cutile` | `#[dsl]` 的 typed IR：一份目标无关的内核**签名 + 循环 + 一次写回**；写回是数据而非类型——逐点 `Write::Map`（`output[idx] = value`）与归约 `Write::Reduce { at }`（`output[at] += value`，如点积）落在同一个 `Kernel` 上，因此名字/形参/表达式校验只有一套（`ScalarType` 是标量闭集，`Access` 只区分只读与可读写，`ValueType` 把「标量」与「缓冲区 + 元素类型 + 访问」合成签名里的一种值，`Arith` 是算术闭集），`Kernel::fuse` 再把两个逐点核拼成一个。它只描述「算什么」，不含任何设备、线程或内存层级——`verify` 在构造时挡住越界与类型不符，`Pass` / `Pipeline` 把化简与死形参消除扫到**不动点**。化简不是一堆专用函数，而是一张**定律表**——`-(-x)→x`、`x*1.0→x`、`x/1.0→x`、`x-0.0→x`、常量折叠、常量取负，一条定律一行，签名统一为 `fn(&Expr) -> Option<Expr>`，加一条律只加一行；`normalize` 先规约子节点再在本节点上把定律跑到不动点，因此一趟就出范式（`x*(0.5+0.5)` 折成 `x*1.0` 后当趟即消成 `x`，不需要第二轮），每条律是否严格收缩由 `debug_assert` 机器检查，内层循环另以「起始节点数」为上界，所以定律即便写错，任何构建下都只会少化简、不会打转。表里只收**逐位成立**的恒等：`x+0.0→x` 被排除是因为 `-0.0+0.0=+0.0`，非有限结果（`0.0/0.0`）也不折叠，留给目标端自己的 IEEE 语义。驱动按**结构是否变化**判停，而不是拿代价当「变了没变」的代理，所以「没变小但有变化」的中间态不会被误当成终点；轮次上限取初始 `cost`（形参数 + 节点数）——严格收缩的 pass 必然在此之前收敛，上限耗尽说明流水线在震荡，此时 `run` **报错**（MLIR 的 did not converge）而不是悄悄交出半成品。注册顺序只在两个 pass 真正冲突时才决定结果：互相交换的 pass 无论如何注册都会收敛到同一范式，所以「调用方排好依赖」不是正确性前提。`run` 在**每个 pass 之后**都 `verify`（MLIR 的 verifyEach），坏内核在发布构建里也到不了发射器，报错还直接点名出问题的 pass——「管理器交出的内核一定合法」因此由构造成立，而不是只在 debug 下成立。落地有两条：`emit_c`（portable C）与 `emit_k`（`.k` 正文，经 `shard::ir` 接 `Lower` / 设备 dispatch）；`Image` 仍未接上，因此设备路径之外它**不产生任何性能结论**，只是把「宏写出来的表达式」变成可校验的数据 |
| `Backend` / `Codegen` / `Driver` / `Readback` / `Source` / `Lower` / `Launch` / `Stages` | `shard` | 设备后端契约与其实现面：缝在机制层，实现在这里——换后端不动框架 |
| `Plan` / `Path` / `NoPath` / `VariantCheck` / `Proof` | `cutile` | kernel 层中间产物，全部可见、无 IR；只有 `NoPath` 自带文本（错误契约）；`Proof` 记这次真正执行过的供给，`Proof::declared` 只给解释性报告 |
| `Payoff` / `Verdict` | `cutile` | 融合打分的结果：`Payoff` 是结构性 roofline 增益（不融合两次定界之和 vs 复合代价一次定界，`saved_bytes` 是中间量的往返），`speedup ≥ 1` 由 `max` 次可加保证而非拟合；`Verdict` 是三态判决——打得出分、拼不出来（带几何原因）、无从打分（缺语义投影 / 接缝跨多条输出 / 档案缺峰值，都不猜一个数顶上）。打分只读代价表、不跑标量参考，所以不需要样本也不分配内存；`bound()` 报融合后哪一侧定界 |
| `Usage` / `Toolchain` | `shard` | kernel 产物的**运行期**两份：`Usage` 记这次编译真正占了什么，`Toolchain` 是实测工具链身份（`ArtifactKey` 才含，`CacheKey` 不含） |
| `CacheKey` / `Cache` / `Entry` / `State` / `Changed` / `Residue` / `Gc` / `ArtifactKey` / `Artifact` | `cutile` | 缓存指纹、编译产物与存量：`CacheKey::fields` 是哈希与差异比较的唯一来源且**不含工具链**，`ArtifactKey` 含实测工具链与编译参数，参数一变就换一份产物 |
| `Doctor` / `OpInfo` / `DeviceInfo` / `LayoutRow` / `Storage` / `Declared` / `Built` / `CacheWhy` / `CacheState` | `shard` | `doctor` 的结论面：一台设备为什么能 / 不能跑某个算子，逐行可追到档案与缓存状态 |
| `VerifyReport` / `VerifyEntry` / `Verified` / `Execution` / `WitnessRun` / `Evidence` / `Basis` | `shard` | `verify` 的逐设备结果与见证执行；失败如实记录，不降级成警告 |
| `LedgerCell` / `BackendRow` | `shard` | `ledger` 跨档案能力台账：只聚合 `witness` 已跑出的证据，不产生新证据 |
| `Negotiation` | `shard` | 一台设备的协商结论（首档 / 降级 / 无路径 / 无档案）：只读声明与档案，与执行成败分开记 |
| `Fragment` / `EpiArgs` | `cutile` | 一份生成片段的行数与 §16 口径（`FRAGMENT_LIMIT` 上限管成败，`FRAGMENT_GOAL` 只是余量）；`EpiArgs` 是尾块片段写回要的四个实参 `$0`..`$3`——做成结构体而不是一袋字符串，「装配不猜名字、缺参即错误」才判得出来 |
| `Payoff` | `cutile` | 一条融合候选的 roofline 增益：`unfused_s` 是两侧各自 `max(算力, 带宽)` 之和，`fused_s` 是复合代价的定界。`max` 次可加保证 `fused ≤ unfused`，`speedup` 因此恒 ≥ 1——代数结论，不是拟合 |
| `Oracle` | `cutile` | 融合对拍的读数：`got` 是消费者在**全部缝**都桥接之后的产出，`bridged` 是实际桥接的 `(生产者输出, 消费者槽位)` 对。`reference::verify_fusion` 是 `Fusion` 的唯一实例化入口——缝从代数的证据里来，不从调用点的字符串来；自融合、多输出生产者、非 `F32` 缝、缝上形状不符一律显式拒绝（`Invalid` / `Unsupported`），而不是算出一个「差不多」的值。README §9 的契约「每个被接受的融合都必须能生成一条对拍」由 `crates/cutile/tests/reference_tests.rs` 在整张内置注册表上执行 |
| `Bench` / `Throughput` / `Peak` / `Verdict` / `Performance` / `CostCoefficients` | `shard` | `bench` 的实测口径与达成率判定；口径无效或没有峰值时 `is_measure` 直接给否，不许补一个乐观值 |
| `Tensor` / `View` / `Owned` / `Store` / `StoreMut` / `Rank` / `Element` / `NumericElement` / `OrderedElement` / `FloatElement` / `IntElement` / `BoolElement` / `Abs` / `Shape` / `Slice` / `Order` / `AxisLen` / `product` | `tensor` | 只有一个 `Tensor<B, R, K>`，三个泛型位各管一件事：`B: Store<K>` 是元素存在哪（借来的 `&[K]` / 持有的 `Owned<K>`，`StoreMut` 追加就地写自己的缓冲）、`R: Rank` 是秩从哪来（`[usize; D]` 编译期定长 / `Vec<usize>` 运行期长度）、`K: Element` 是元素是什么（`bool` / `u8`–`u64` / `i8`–`i64` / `f32` / `f64` / `f16` / `bf16`，精度是类型常量而不是字段）。元素按能力分封闭标记：`NumericElement` 收可参与乘加的数值，下分 `FloatElement` / `IntElement`；`OrderedElement` 在数值之上收可比较排序的标量，供极值归约取序；`Abs` 是绝对值能力标记；`BoolElement` 直挂 `Element`，不入数值列。与 Burn `Tensor<B, const D, K>` 同形，两处差异刻意的：`B` 是**存储**不是后端算子集（算子由 `nn` 的注册表给出，`B: Backend` 会带来百方法接口，见 ADR 0053）、`R` 是 `const D` 的**超集**（运行期 batch / segment 几何进同一个类型，ADR 0046）。`View<'a, K>` 就是 `Tensor<&'a [K], Vec<usize>, K>` 的别名。形状算子面与 Burn 齐平并多出零拷贝分段：`broadcast` 把被扩展轴的步长写成 0，`narrow` / `slice` 只前移起点，`chunks` / `split` / `unbind` 借同一存储分段，`select` / `squeeze` / `unsqueeze` / `flatten` / `diagonal` 改秩，`transpose` / `permute` / `movedim` 换轴序，`slice_fill` / `slice_assign` 就地写自己的缓冲；带步长切片（`Slice::stepped`，对应 Burn `s![..;2]`）同样零拷贝。只有 `flip` 按镜像下标重算并返回自持张量，因为无符号步长表达不出倒走。`Shape` 的 `numel` 是饱和乘积（溢出到 `usize::MAX` 而不是回绕——长度可能来自不受信任的权重文件头），`element_bytes(dtype)` 折字节数，亚字节按位向上取整的唯一实现是 `DType::bytes_for`；偏移、可达性、连续性与取行只有 `layout` 一份实现，连续步长只有 `shape::accumulate` 一份实现，于是两种秩、两种存储不可能漂移。元素类型面与 Burn 同宽（含 `f16`/`bf16`），且亚字节精度（`U1`/`U2`/`U4`/`I4`/`F4`）是 `DType` 位宽表的一等公民、`bytes_for` 位级进位，比 Burn 的 `size_of` 整除约束更宽；`DType` **只转发不拥有**，闭集与 `Class` 数值类别的唯一源在 L1 `dtype`。设备地址不泄漏到本层 |
| `Buf` / `Allocator` / `CpuAllocator` / `LaneBuf` / `LaneView` / `LaneViewMut` | `storage` | 扁平缓冲与分配（排布归算子声明，本层不持有布局）；`Allocator` 是分配 / 释放 / 显式拷贝的边界 trait，设备后端在边界 crate 实现同一 trait；本层不做隐式跨设备拷贝。`LaneBuf` / `LaneView` / `LaneViewMut` 是 `f32` 与打包 `u8` 两条 lane 的公共缓冲与视图枚举，`engine` 的 `Buf`/`View`/`ViewMut` 与 `train` 的 `Data`/`View`/`ViewMut` 只做类型别名引用，不再各抄一份 `F32/U8` 副本；形状不在这里解释 |
| `Manifest` / `ManifestEntry` | `storage` | 检查点字节格式的两条契约：`Manifest` 是"有序条目 + 数据段起点"，`ManifestEntry` 是一条条目的名字 / 布局 / 字节区间。字节格式因此不必反向依赖 `weights`——`weights::Header` 实现它，环不存在 |
| `Stream` / `Graph` / `Launch` / `Device` / `Operand` / `OperandRole` / `ScalarSlots` | `engine` | 提交与执行，不编译 kernel；`Device` 只是运行时标识标签，不持私有句柄；`ScalarSlots` 是定长类型化闭集标量槽（[ADR 0043](adr/0043-program-scalar-slots.md)），位置参数不折进 1 元素 f32 缓冲 |
| `VacantExecutor` | `engine` | 尚未接入运行时的设备后端位：只登记厂商标签，不持设备句柄 / 内核表 / 宿主缓冲；提交、登记内核、绑定与取回操作数一律回 `Code::Unsupported` 并带站点；真机管线落地后在同一 `Executor` 缝上升格，缝的其余消费方不改 |
| `Sequence` / `Scheduler` / `Progress` | `engine` | 连续批与序列状态；准入后每生成一个 token 才预留它要落的那页，`Blocked` 时不改任何状态；`Sequence` 是已生成 token 的唯一存放处 |
| `Page` / `Table` / `Cache` / `Pager` | `engine` | 分页 KV 的页管理与页内数据：引用计数 O(1) 判定越界 / 重复归还 / 下溢，归零才回池且只有最后一个持有者归还时清零；行宽由装载模型的一侧给 |
| `Slot` | `engine` | 一个 token 在 KV 缓存里的位置（页号 + 页内偏移），写 kernel 的一侧不必自己再算一遍 |
| `Radix` | `engine` | 满块前缀树：自己就是页的持有者，`evict` 只摘 LRU 叶；页复用与位置账（`Sequence::prefilled`）是同一份 |
| `Batch` / `Admission` | `engine` | 连续批：`emit` 是"记账 + 推进"的唯一入口；`Admission` 让"页不够"成为可重试的 503 而不是请求非法。与 `train::Batch`（数据单位）同名不同物 |
| `Budget` / `Candidate` / `Phase` / `Verdict` | `engine`（`ifb` 模块） | 一次迭代的可行域与相位：`pack` 判"这一轮塞得下几行"，与 `admit` 判"这辈子跑不跑得完"分开；相位是序列自己的事实，不是调用方贴的标签。唯一消费者是 `serve::Completion` 的批调度：`round` 先把本层一次提交的粒度收进预算、再把 `Packed` 的行原样下发，于是"这一轮算几行"只有一个来源（打包给配额、`chunk` 给块尾，取小的那个） |
| `Executor` / `Trace` / `Program` / `Bindings` / `Buf` / `View` / `ViewMut` / `CpuExecutor` | `engine` | CPU 执行体（[ADR 0051](adr/0051-execution-backend-base.md)）：`Program` 按名字登记、重名拒绝，`Bindings` 是编号 ↔ 缓冲的操作数表，`CpuExecutor` 入队并在 `sync` 时就地执行；`Trace` 只记录不改语义。`Buf`/`View`/`ViewMut` 是 `storage::LaneBuf`/`LaneView`/`LaneViewMut` 的别名，执行缝只按编号绑定缓冲、不解释形状（ADR 0044） |
| `MetalDevice` / `MetalCommandQueue` / `MetalBuf` / `MetalPipeline` / `MetalExecutor` | `device` | 设备后端边界（[ADR 0035](adr/0035-unsafe-ffi-boundary.md) 白名单第五类「进程级库句柄」，越界岛，不单独成 crate）：句柄私有、不跨线程，MSL 由装配层注入，本层不调用 `cutile` |
| `Abi` / `Arch` / `Host` / `Machine` / `Row` / `Standing` / `Why` | `device`（`platform` 模块） | 六平台执行合同（[ADR 0052](adr/0052-execution-platform-contract.md)）：`Row` 是执行族闭集（CPU / Metal / CUDA / ROCm-HIP / Vulkan-SYCL 合一的 GPU 行，顺序即合同登记顺序），`Standing` 是三态裁决（覆盖 + 证据 或 未覆盖 + `Why`），`Why` 只分 `NoDevice` / `NoToolchain`——两者处置不同（换机器 vs 装工具链），混成一个"未覆盖"就丢了唯一的行动信息。`audit` 每次都出六行（少一行会让"未覆盖"与"根本没问"变成同一输出），`require` 才报 `E-PLAT`；它只判**平台有没有资格跑**，不判内核在某台设备上是否正确（那是 `shard::verify` 与 `tools::card` 的事）。`Host` 是注入缝（架构 + 该行工具链），`Machine` 是这台机器的实现：架构取 `std::env::consts`、工具链复用 `backends` 那份进程级快照探测（不新开第二次 `--version`），于是测试能用一台假机器摆出九种组合。**不拥有**设备探测实现、设备后端代码（Metal / CUDA 的越界边界见 [ADR 0035](adr/0035-unsafe-ffi-boundary.md)，本层只裁决"能不能跑"）、性能与正确性结论（E1–E15 归协议卡）、以及设备档案本身（那是 `prof` 的声明层） |
| `Tokenizer` / `Token` / `Bytes` / `Bpe` / `Unigram` / `SpecialTokens` / `Encoding` / `Strategy` / `Truncation` / `Padding` / `PadDirection` / `Fingerprint` / `DecoderStream` / `Vocab` / `VocabSource` / `VocabFamily` / `Merge` | `tokenizer` | 文本 ↔ token 的唯一边界：编号、编码对齐记录、序列策略、词表身份指纹与 UTF-8 边界安全的流式解码 |
| `Adaptive` / `AdaptiveCompensated` / `Header` / `Entry` / `Reader` / `Loader` / `Archive` / `File` / `MmapSource` / `Quantized` / `Scheme` / `Compensated` / `Writer` / `Stored` / `Segment` / `Quant` / `Codec` / `Proof` / `Metrics` | `weights` | 权重文件：外部 safetensors 只读适配（`Reader` 借出 / `Loader` 拷贝两条契约）+ 自有 `.u` 容器 v2 读写（rANS、Merkle 证明、两级 CRC、原子写、现场度量）；`Quantized` 的分组、输入宽度与输出宽度出自同一次推断 |
| `Certified` / `Note` / `Stable` / `Mode` / `EpsMethod` / `Bound` | `weights` | `ledger` 的谱隙证书与它门控的秩选择：判定与产出出自同一次分解，证书写在 `AX03` 的 `rank` 段并由 `audit` 重算，对不上就拒读；证书自报 `ε` 的**方法**（`EpsMethod`）与这张界的**类型**（`Bound`，L0 诊断 vs L0' 候选），于是「幂迭代只够排序」与「候选界不签无损」是可执行约束而不是注释；`E_decomp` 与 `E_comp` 分开记，合成误差只有 `Note::synth` 一处算式 |
| `Rung` / `Chain` / `Bracket` / `Ruling` | `weights`（`l3` 模块） | 网络级误差合成（LLD §4.5）：逐段 `(ρ, ρ̂, ε, L, b)` 收成 `e_L` 上界与**适用域**，`Ruling` 按「有非零 `ε` 就拒 → 缺产物逐位证据就存疑 → 有段在定理外就只到部分 → 三条都过才无损」判定；`ε` 可吃谱隙证书的 `Note::eps`（同一次分解的后验残差），`L` 无默认值，`ρ̂ ≠ ρ` 分开记。只做算术与判决：不持权重、不解码产物，产物是否逐位还原由调用方实测后传入 |
| `Cert` / `Settled` / `Settlement` / `Reason` | `weights`（`settle` 模块） | 逐张量沉降账本（LLD §4.7）：体积、熵、实际位宽、无损与原因全部由**落盘字节**现算，不读文件里存的度量字段；`lossless` 与 `Metrics` 共用同一份 `lossless_entry`，于是容器的两个视图不会对同一条给出两种“无损”。“原因”只答“为什么是这个体积”（引用去重 / 小段 / 轴字典 / 不可压），不答“够不够 Must”——后者归协议卡；本层只读产物，不重写产物。账本的读回也归本层：`Settlement::verify` 用同一份字节现算的账逐格核对（按规范写法比、不能少也不能多），因此“记账”与“核账”不可能各自算一遍；CLI 的 `audit` 只是把这条读回出口接到命令行 |
| `Tape` / `Var` / `Node` / `Requirement` / `Role` | `autograd` | 反向图的记录：变量需求、出身、边与深度 |
| `Schedule` / `Account` / `Ordering` / `Config` | `autograd` | 反向调度（可达剪枝 + 可配置排序 + 同深度分层）与扇出账本；都在原对象上重填，训练循环不必每步新建 |
| `Plan` / `Budget` / `Choice` / `Footprint` / `Sizes` / `MemoryPolicy` | `autograd` | 显存计划：只读 `Tape`、`Schedule` 与形状表，不分配、不执行。与 `cutile::plan::Plan`（单条 kernel 选哪个 tile）不是同一件事 |
| `Replay` / `Store` | `autograd` | 重放执行：把计划编成扁平序列表；`Store` 是唯一口子，真实缓冲归 `train` |
| `Summary` | `autograd` | 图规模与可达性的只读计数快照，供诊断、缓存键与容量预估 |
| `Contract` / `ContractSpec` / `Linear` / `LmHead` / `WeightFormat` / `ParamRole` / `OpLayer` | `nn` | 层定义与契约校验：产出的算子名必须存在于 `nn` 注册表；`WeightFormat` 是格式 → 算子名的唯一事实源（精度档是一条线：2 / 4 / 8 位量化与 2 / 4 位打包各占一档，量化档的分组从对应算子的语义表读出，层里不写第二份常数），`ParamRole` 让权重衰减豁免与可训练判定从角色派生；`OpLayer` 是运行期配置化的单轴层（算子名 / 长度 / 角色都是构造参数，契约与编译期层同源同一个 `AxisCore`） |
| `RmsNorm` / `Softmax` / `Sigmoid` / `Tanh` / `Relu` / `Gelu` / `Silu` / `LeakyRelu` / `HardSigmoid` / `Rope` / `Scale` / `Add` / `Sub` / `Mul` / `Div` / `Max` / `Min` / `Exp` / `Log` / `Embedding` / `Scan` / `MatMul` / `Gauge` / `Precision` / `Attention` / `AttentionAlgo` / `LatentAttention` / `DecodeAttention` | `nn` | 单算子层（`src/single.rs` / `src/attention.rs`）：形状完全由构造参数给出、没有窗口/分组几何，两族只差权重口径；解码注意力按算子表声明的历史档给**松下界**，不假装知道真实 KV 长度：只固定自己绑定的算子名与形状口径；下界仍由算子语义表推出，层里不写第二份流量常数。`MatMul` 另带两条轴——`Precision`（元素精度）与 `Gauge`（收缩轴档位）选算子（`Gauge::Short` 是 `gemm_*`、`K = 8`，`Gauge::Long` 是 `matmul_*`、`K = 4096`，同一族在算子表里是两个档，差 512 倍），「精度 × 档位」写成一张表，表里没有的组合报 `Unsupported` 而不是借兄弟算子凑一个数 |
| `Conv1d` / `Conv2d` / `ConvTranspose1d` / `ConvTranspose2d` / `MaxPool1d` / `MaxPool2d` / `AvgPool2d` / `AdaptiveAvgPool2d` / `Interpolate` / `BatchNorm2d` / `GroupNorm` / `InstanceNorm` / `LayerNorm` | `nn` | 经典几何层（`src/spatial.rs`）：每个层只固定自己绑定的 `nn` 算子名与形状口径，窗口 / 步长 / 分组等几何与数值算术都在 `nn` 侧（`conv` / `kernel`），层里不写第二份流量常数。归一化族与卷积 / 池化 / 插值共享同一条形状约定（末轴是通道）与同一档下界口径（问算子语义表），只是不碰窗口——判据是**声明口径相同**而不是行数，故同住 `spatial`；两族都只用到「逐字段必须非零」这一条校验，`nonzero` 因此上提到 `lib` 与 `nonzero_all` 并列 |
| `Cell` / `Recurrent` / `RecurrentTensor` | `nn` | 循环层（`src/recurrent.rs`）：`Cell` 是门结构（RNN 单门 / GRU 三门 / LSTM 四门，门数与激活次数都从它推、下界按真实次数收），`RecurrentTensor` 是方向内的一件权重（输入侧 / 隐侧 × 矩阵 / 偏置四件，名字与形状对齐真实 checkpoint），`Recurrent` 是 `layers` 层 × `dirs` 方向的循环层：绑定 `gemv_f32` 门投影 + 单元的激活算子，不拥有门控数值本身 |
| `AttentionBlock` | `nn` | 注意力组合块：Q/K/V 投影 + 注意力 + 输出投影内聚成一个 `Contract`，子层按 `{块名}.q` / `.k` / `.v` / `.attn` / `.o` 对齐权重条目 |
| `Bounds` | `nn` | 层的性能下界：`read` / `write` / `flops` 三笔按方向分账，不认识设备峰值、不做 roofline 编排 |
| `Stack` / `Floor` / `Unpriced` | `nn` | 层栈与它的总下界；未覆盖与无法计价两类清单都不允许静默丢弃，也不让整张模型算不出下界 |
| `Calib` / `Lif` / `Encoder` / `Events` / `Audit` | `algorithm` | 决策与脉冲的参考语义：`Calib` 是唯一决策端口（DecisionHead / MoE / 层间路由共用），`Lif` 是漏积分-发放神经元，事件臂只管码位与稀疏度。不拥有判定语汇：PASS/FAIL 仍只由 `tools::card` 出 |
| `Spiking` / `Membrane` / `Gradients` / `Surrogate` / `Master` | `algorithm` | 脉冲三段（`src/spike.rs`）：`Spiking` 持 i8 权重与动力学，`Membrane` 是状态（电位 + 发放前电位 `pre`，反传只认这个值，发放一提交它就被写成 `reset`），`Gradients` 是权重/尺度/电流三条梯度，`Surrogate` 是代理导数（矩形/三角窗，面积 1、窗外 0），`Master` 是 QAT 的浮点主权重（`view` 量化、`descend` 一步 SGD）。形状、码位与梯度布局在同一组判据上拒绝；不拥有 STE 之外的整条反传链、批调度与设备后端 |
| `BlockTable` / `Channel` / `Gating` / `MissRate` | `algorithm` | §5.8.2 / §5.8.3 的执行侧：块表因果寻址、α/β/γ 通道与一步门控、`ρ_t` 的因果窗口；三者不持参数、不注册算子 |
| `Bypass` / `MemoryBlock` | `nn` | §1.5–1.6 的条件记忆旁路：可装配的 `MemoryBlock` 只读、失败不回写，形状与槽位由 `engram` 唯一给出；旁路的装配骨架留在 `nn`（它产出 `ContractSpec`），块表寻址与门控语义在 `algorithm`；不持参数、不注册算子 |
| `Mla` / `Latent` / `Path` | `algorithm` | MLA 的 KV 记账从公式变成结构：`Latent` 只有 `c^{KV}` 与解耦 RoPE 键，两条路径按 §5.8.1 逐元素对拍；不拥有分页与批调度 |
| `Moe` / `K3_EXPERTS` / `K3_ACTIVE` | `algorithm` | 专家侧：路由走唯一端口 `policy::Calib`，选中集重新归一，共享专家恒参与且不占 top-k 名额；不拥有门控标量、专家参数与训练状态 |
| `Mhc` / `MHC_STREAMS` | `algorithm` | 残差侧：双随机矩阵重组残差流，非负性是入口硬条件（否则"实测 ≤ 1"会假通过）；不拥有门控标量、层参数与训练状态 |
| `Family` / `Wiring` / `Blocked` | `algorithm` | 接线台账：17 族每族一行，缺算子还是缺层都具名，`mhc` 的流重组落成 `nn::MatMul(streams, streams)`；不拥有层声明、算子实现与执行路径 |
| `Hybrid` / `Query` | `algorithm` | 装配：一层 = 层首混流 → MLA 注意力残差 → MoE 前馈残差，三件都调各自模块的唯一入口；增量只加回主路（同一份增量加给每条流会让多流退化成单流），缓存只读、失败不回写；不拥有投影矩阵、归一化算术、KV 页账与训练状态 |
| `Tower` / `State` / `Kind` / `MemoryStep` / `MEMORY_PER_ATTENTION` | `algorithm` | 层栈装配：每三个通道记忆层配一个混合层（配比是构造条件，不是注释），层序由 `Kind` 逐格读出；记忆状态与 KV 缓存分离（`State` 随序列长累积、`Latent` 按页账走），一次推进要么整栈走一步要么一格不动；不拥有投影、页账与训练状态 |
| `Kda` / `Memory` | `algorithm` | 混合 delta 轴：纠错式写入（先减去记忆已预测的部分），逐通道门控；不拥有投影、层配方与训练状态 |
| `Mamba` / `Carry` | `algorithm` | 指数梯形状态空间：状态存成 `2N` 个实值分量、低秩 MIMO、读数取本 token 更新之后的状态；与 delta 规则判据正交，两份实现互不借用语义 |
| `AttnRes` / `Sources` / `splice` | `algorithm` | 深度轴：按注意力取深层残差（凸组合），`Sources` 按层号记来源，`splice` 保证并回是凸组合 |
| `Boundary` / `Patches` | `algorithm` | byte-latent 编码侧：边界规则闭集（参数随变体走），`Patches` 只存起点与总字节数，因此"区间首尾相接"由构造保证 |
| `ByteEncoder` / `ByteDecoder` | `algorithm` | byte-latent 的表示侧与解码侧：哈希 N-gram 桶向量 + patch 内注意力池化，一次投影给出 256-way 分布与贪心字节 |
| `Emission` / `Emitter` / `Halt` | `algorithm` | byte-latent 生成侧：把边界 / 编码 / 解码串成一次生成，条件只取**已收口** patch；`Halt` 是闭集，`None` 说不清"截断"与"空结果" |
| `Global` / `Backread` / `Layer` | `algorithm` | byte-latent 的潜在栈：`Global` 的遍数就是 patch 数，`Backread` 的可见集由 `Patches` 给出，`Layer` 的宽度由栈校验 |
| `Group` / `Reduce` / `Collective` / `Topology` / `Parallel` | `shard` | 集合通信契约与 TP/PP/DP 拓扑；`world = tp · pp · dp` 在构造期算成精确值，因此不存在回绕出来的"合法" rank |
| `Sharding` / `Tiling` | `shard` | 张量并行的两种线性层切法与一层权重的切分计划；切分必须整除，在构造期失败而不是摊余数（不保留"前几份多拿一个"的均匀切分：TP 要求各 rank 的权重形状一致） |
| `Local` | `shard` | 进程内集合通信：每个 rank 一份镜像，让 TP/DP 的数值语义在没有 NCCL 的机器上也能被 `cargo test` 验证 |
| `Remote` / `FileRemote` | `shard` | 远端集合通信：`Remote` 是未挂载占位（同一 `Collective` 端口，`all_reduce` 返回 `Unsupported` 且不改动调用方缓冲），`FileRemote` 是文件传输的真实实现（`world × width` 槽位、小端字节序、跨进程、无 `unsafe`） |
| `RuntimeConfig` / `LanguageFront` | `shard` | 运行时可配置旋钮（纯数据，由 `cli` 注入）：各语言语法门、host 自检参数、L3 达成率门槛、缓存锁等待；codegen 常量（`POINT_THREADS` 等）不进这里 |
| `ReduceScalar` / `ParallelConfig` / `Priority` | `shard` | 集合通信元素档约束（加法/乘法/偏序/中立元，浮点极值传播 NaN）；并行策略纯数据（TP/PP/DP 度数、归约种类、平均开关、调度优先级），由 `Distributed` 装配；只持部署形态，不读 toml、不发起通信。约束自包含在 shard，不反向依赖 `tensor` 元素层级 |
| `Plan` | `shard` | 并行计划视图：由拓扑与本 rank 派生 TP/PP/DP 三轴下标、各维通信组与 PP 流水层段；只读数据，不持缓冲、不发起通信，层段整除判据与 `Tiling` 同口径（摊余数构造期拒绝） |
| `Job` / `Queue` | `shard` | 集合操作优先级排队：作业绑定调度优先级与泛型载荷，排空时按 `Priority` 序数稳定排序（同档保持入队先后）；只定批次内提交顺序，不引线程池、不改变任何归约数值 |
| `Assembler` / `Placement` / `LinearGeo` | `shard` | 模型层并行归属：把 `nn` 层清单映射到本 rank 的 PP 段与 TP 分片几何；`Placement` 带泛型载荷（装配侧句柄原样带回），段外层不建分片计划，非分层参数各副本持有；可按几何取本 rank 本地子块（`Copy` 元素档泛型切片），但只产归属几何与本地子块，不装载进图、不发通信 |
| `Snapshot` / `Format` / `Facts` | `storage`（`snapshot` 模块） | 检查点字节格式与恢复事实：字节只写相对上一条的差、尾部 CRC-32；`verify` 只验字节，`check_weights` 要求条目表与权重文件逐条相同。不依赖任何训练侧类型，写出同样的字节即可换掉实现 |
| `Moments` / `SlotMoments` / `GradAccum` / `Ties` | `storage`（`snapshot` 模块） | 快照携带的优化器状态段：矩状态按槽位自持、累积梯度带计数、权重绑定表是 `(源槽, 回填槽)` 下标对；编解码与字节格式同源，训练侧只读写不重新定义 |
| `Schedule` / `Anneal` / `Step` / `Session` / `TrainConfig` / `ParamSlot` / `Checkpoint` / `Distributed` | `train` | 训练循环状态：纯函数学习率调度与衰减曲线、一次会话复用同一张图 / 缓冲 / 优化器、参数槽与分布式装配（后端契约在 `shard`） |
| `Batch` / `Dataset` / `Chunks` | `train`（`samples` 模块） | 训练样本：借用调用方 token 缓冲（零分配 / 零拷贝），两个构造函数共用同一份几何校验，批总数在构造期算好，`Dataset` 对象安全 |
| `Optimizer` / `Sgd` / `Adam` / `Group` / `Slot` / `OwnedSlot` | `train` | 优化器：对象安全 trait，矩状态按槽位编号自持；参数组由 `ParamRole` 派生，不按字段名猜（矩状态与累积梯度的**类型**归 `storage`，本层只持有并步进） |
| `Muon` | `train` | 矩阵参数的近似正交化更新：更新量只由方向决定，非矩阵槽**拒绝**而不是悄悄跳过 |
| `Bufs` / `Kernel` / `Reference` / `Data` / `View` / `ViewMut` / `Kind` / `Lane` / `Sealed` / `Inputs` / `OperatorSetOps` | `train` | 激活缓冲：`autograd::Store` 在训练侧的落地（按计划分配与交还、按等长空闲表复用），算术交给 `Kernel`；本层不定义算子、不碰 `unsafe`。`Data`/`View`/`ViewMut`/`Kind` 是 `storage::LaneBuf`/`LaneView`/`LaneViewMut`/`Lane` 的别名，本层只保留 lane 静态契约（`Lane`/`Inputs`/`Sealed`）与 `DType`→lane 映射 |
| `MetricPoint` / `MetricsBoard` / `Learner` / `ForwardOut` | `train` | 评估与编排：`Learner` 驱动"前向 → 评估入账 → 装梯度 → 优化步"闭环，模型拓扑与梯度来源由闭包注入 |
| `Distill` / `Grpo` / `Loss` / `Metric` / `Outcome` | `train` | 损失与度量闭集：分类 CE / Hinton 蒸馏 / GRPO 三臂，配置随度量走；都是 train 侧纯函数，不进 `nn` 的设备核注册表 |
| `Envelope` / `Frame` / `Decoder` / `Kind` / `Version` | `serve`（`protocol` 模块；`proto` 为兼容重导出路径） | 端点契约的线格式：版本化信封、零拷贝帧视图与增量解码（帧布局 = 10 字节头 + 载荷 + v2 起的 CRC-32 尾，见 `serve::protocol` 的 `//!`） |
| `Route` / `Endpoint` / `Response` | `serve` | 端点契约：`Endpoint::call` 的 `Err` 只留给"服务端自己坏了"，请求侧不合法必须落成 `Response::error(BadRequest, …)` |
| `Status` / `Request` | `serve` | 响应状态闭集与请求文法：未知键、重复键、缺字段与坏数字都是 `invalid`；停止符不在请求里——那是采样策略的事 |
| `Predict` / `Sampling` / `Greedy` / `Classical` / `Multinomial` / `SamplingParams` / `Generation` / `Ending` | `serve` | 生成的协作者与结局：给出一步 logits、选一个 token、说明为何停下；端点不认识模型，模型也不认识页账 |
| `Completion` / `Router` | `serve` | 第一条真实端点：解析 → 编词 → 准入 → 逐步前向 + 采样 → 归还页 → 包帧；客户端错误落 400、页不够落 503，失败路径同样归还页 |
| `Verdict` / `accept` / `Drafter` | `serve`（`speculate` 模块） | 投机解码的接受判定：取最长一致前缀，第一个不一致的位置由目标模型补一个；只判"接受几个"、不写 KV |
| `Httpd` | `serve` | std-only HTTP/1.1 薄适配器：单线程一问一答，固定 `Connection: close`；换传输层只需在同一个 `Router` 上再挂一个适配器 |
| `CpuModel` / `Params` / `Block` | `serve` | CPU 模型宿主与驻留参数表：装载期一次解码、生成循环不再读文件；七个层族装配成 `engine::exec::Graph`，执行器泛化为 `E: Executor`；`from_path` 按 `Reader` / `Loader` 在装配点分流 |
| `Command` / `Invocation` / `USAGE` | `cli` | 命令行文法（argv / env → 强类型调用参数）与编排分派 |
| `Token` / `Pos` / `Delim` / `Finding` / `Gate` / `Hit` | `tools` | 零依赖词法 token 树、单函数复杂度 finding 与配置化阈值、标识符层词汇命中（全拼名 + 应改用的缩写）；分析器只产出数据，不打印不退出。词汇判定在**标识符**层而不是文本层，字符串字面量与注释不参与——设备源码里 `[[buffer(0)]]` 这类选择器因此不会被误报 |
| `Case` / `Baseline` / `Judgement` | `tools` | 回退判定（与基线相比有没有退）；不做测量 |
| `EntryKind` / `Harness` / `Roofline` / `Bottleneck` / `Score` / `Geometry` / `BenchRecord` | `tools` | 统一基准记录结构：记录只持有身份、几何与峰值，未测得的峰值留 `None`；`Roofline::score` 把「峰值 + 算术强度 + 实测吞吐」折成占上界百分比与瓶颈侧，测量与上界登记仍归 `shard` / `benchmarks` |
| `Protocol` / `Statistic` / `Tally` / `Standing` / `Shortfall` / `Card` | `tools` | 实测卡三态判定：闭集 → 样本量 → 统计量定义 → 区间与门槛，次序有意；样本与时钟来源归 `benchmarks` |
| `spec` / `model` 属性宏 | `annotation` | 把注解展开成契约实现 + 编译期断言 + 一条校验测试；**不持有运行期注册表**。`model` 生成 `MODEL` 并断言注解与 `Operator` 对齐；设备执行由 `cutile` 的 `const SPECS` / `const GATES` 描述符表按函数指针登记 |
| `Contract` / `Provision` / `Witness` | `testkit` | 设备片段层自述：`Contract` 只答「这台设备声明了哪些 P0 片段、拿哪些见证入口验」，`Provision` 只答「一条原语在这台设备上由哪个实现供给」，`Witness` 只答「用哪个 `witness_*` 入口、期望什么输出」。三者都只持 `&str` / `&[f32]` 借用、不持档案类型，于是夹具库不把 `prof` 拖进依赖，设备档案在测试侧摊平即用 |
| `Call` / `Reading` | `testkit` | 见证对拍的两端：`Call` 是「要跑哪个入口、期望什么」，`Reading` 是「跑完读回了几个、值是什么、最大相对误差多少」。个数与值分路回读，`max_rel_err` 在个数不符时取 [`f64::INFINITY`]，使"少写一个元素"与"值错了"在读数层就可区分 |
| `TOLERANCE` / `Contract` / `Provision` / `Witness` / `Call` / `Reading` | `testkit` | 片段层的"算得对不对"与"写得对不对"分两处：`testkit::device::Contract` 只做结构闭合（P0 集合、签名首行、见证入口逐条对应、档案供给必须有证据、目录是闭集），`evidence` 只做数值闭合（见证必须自足、期望值与档案写的一致、见证执行结果与期望对拍）。`Contract` 上的 `check_p0_set` / `check_signatures` / `check_witness_entries` / `check_evidence` / `check_closed_directory` / `check_self_contained` / `check_expectations` / `check_witnesses` 是两组判据的入口，两边都只**产出缺陷**（`Result<(), String>`），由调用方的测试断言——夹具自己不 panic，于是 `src/` 里没有一个 `unwrap`/`expect`。`TOLERANCE` 住 `evidence`——它是"算出来对不对"的阈值，与结构判据无关 |
| `Scale` | `testkit` | 夹具侧的浮点对拍分母策略：`Scale` 只答「分母按整批最大幅度还是每元素留 1 的地板取」；失败读数与最差元素（`RelErr` / `Worst`）连同服务内部的判定 `rel_err` / `within_ratio` / `within_rel` 一律是 crate 内部载体，公开面只留 `max_rel_err` / `assert_within_rel` / `assert_rel_with`。阈值仍只由 `cutile::reference::tolerance` 拥有——本行不设第二张分档表，`Scale::Batch` 与它由 `crates/testkit/tests/rel_tests.rs` 逐位钉住 |
| `Absence` | `testkit` | 性质测试台账的一行：`members` 只持成员名、`reason` 只持一句**可核查**的替代覆盖说明，台账因此是纯数据——判据全在 `prop_discipline` 那道门里（与源码扫描双向一致、理由必须具体、已登记缺口有上限）。台账自己不做判断，也就不会和扫描各有一套说法 |
| `Gemm` / `Matmul` / `MatmulI8Op` / `Packed` / `PackedRow` | `nn` | 内核族的算子实体：打包格式（`Packed` 行描述子 + `PackedRow` 行视图）与三条收缩装配；`#[model]` 的 `reference` 是语义表唯一来源 |
| `Plain` / `PlainFmt` / `Dense` / `DequantI8` / `QuantI4` / `SymI8` / `Clamp` | `nn` | 未打包与量化格式的算子实体：回写尾块、反量化、4 位分组量化、对称 i8、逐元素裁剪，各带自己的语义表注解 |
| `Binary` / `Unary` | `nn` | 逐元素两族：一张宏表同时生成声明与内核绑定，新算子只加一行 |
| `Embed` / `Sampler` / `FlashAttention` / `DecodeFlashAttention` | `nn` | 词表与注意力侧算子实体：预填充 / 解码两条几何、掩码与归一化、采样；`reference` 是唯一对拍口径 |
| `ConvGeom` | `nn` | 卷积几何：把填充 / 步长 / 膨胀折成输出尺寸与索引映射，卷积与池化共用一份 |
| `Axis` / `Builtin` / `Skeleton` / `Variant` / `TileCombo` / `Chunk` / `Grid` | `prof` | 声明层与模板词法：`Builtin` 是一次声明解析的结果（名字 + 模板正文 + 槽位），`Skeleton` / `Axis` / `Variant` / `TileCombo` 是模板槽位与 tile 组合，`Chunk` / `Grid` 是片段正文的分块与网格形状 |
| `LanguageGate` | `shard` | 每门语言一个语法前端（命令 + 版本 + 诊断解析）：片段正文必须被它接受，门不许自己写第二份语言知识 |
| `Scope` / `TensorGeometry` / `Timing` | `shard` | 测速范围（单点 / 多轮 / 预热）、张量几何的归一化描述、一次测量的耗时口径 |
| `Attr` / `AttrKey` / `Grade` / `Mismatch` / `Need` / `Prim` / `Provide` / `Space` / `Span` | `prof` | 原语闭集（`Prim`）与它的属性键 / 档位（`AttrKey` / `Attr` / `Grade`）、档位空间（`Space`）、供给与需求（`Provide` / `Need`）及失配（`Mismatch`）、源码区间（`Span`）：全是闭集生成的，声明顺序就是 `ALL` 的下标 |

几个**同名不同物**的邻居，评审时容易混淆：

- `cutile::space::Space`（GM / FAST / REG）是**语言概念**，`prof::atom::Space` 是它的档案表示；档案侧只认字节；
- `shard::bench::Verdict`（达成率：能否与档案峰值比较）、`tools::regress::Judgement`（相对基线有无回退）、
  `tools::card::Standing`（这一格成立了没有）三态名不同是刻意的，它们会同时出现在一份报告里；
- kernel 侧操作数视图与 `tensor::Tensor` 是两回事：前者不得依赖 `tensor`，后者不得假装能做 kernel 分区（`slice_fill` / `slice_assign` 只写自己的缓冲，不分区 kernel 操作数）；
  该视图在 P0 **尚未实现**——`cutile` 只接收裸指针与标量，分片由调用方给；
- `nn::LmHead` 这类注解声明与 `Linear::new` 这类运行期构造走同一条 `ContractSpec::new` 校验路径，
  注解不建注册表（否则就是第二份事实源，见 [ADR 0005](adr/0005-annotations-as-contracts.md)）。
## 四、闭集与位集

「限定取值范围的词汇表」（原语、精度、存储层、属性键、证据等级、输出语言、骨架）必须回答同一组问题：
**全量是什么、跨层名字是什么、位集下标是什么**。写三遍就是三份会漂移的事实源，因此由一个契约收口：

- `impl Closed for X { const ALL; fn as_str; fn idx; }` 是**唯一**声明处，`parse` 由 `ALL` + `as_str` 派生；
- 已实现 `dtype::Format` 的精度类型由一条 blanket 接上，`Display` / `FromStr` / 可选 `serde` 名字
  都取自同一张行表，不存在"序列化一套、显示一套"；
- `inherent_names!(X)` 生成同名固有转发，调用点写法不变、逻辑仍只有一份；
- `Set<E, WORDS>` 是唯一位集：容量在单态化时由 `const FITS` 断言，"加第 65 个原语"是编译错误；
- 契约不可达的成员（`diagnostic` / `tensor` / `shard` / `serve` 在 `prof` 之下，反向依赖会成环）
  可以把词汇表留在本地，但必须登记在 `.config/lints/vocabulary.toml` 并写明理由。

由 `crates/testkit/src/contracts.rs` 执行：禁止新增手写 `const ALL` 与手写位集，禁止白名单过期，
并要求扫描到的源文件数不为零（避免"空门禁假绿"）。

#### 注解里的名字也是词汇表的名字

属性宏收字符串（`dtype = "f32"`），词汇表里是变体，注解宏按 `DType::variant` 转换。三处必须同时
成立：`annotation` 的 `DTYPES` 与 `cutile::reference` 给出非零阈值的 `DType` 集合逐项**相等**；
注解调用点生成的 `#[cfg(test)]` 把字面量与变体、`Model::dtype` 与 `.k` 的可写输出类型对齐；变体名
取 `DType::variant` 而不是在注解里手写映射。`annotation` 只依赖 `dtype` 这个 L1 零依赖基座，
`syn` / `quote` / `proc-macro2` 不进入供应链。

#### 语义表的唯一生产者

`#[annotation::model(name = .., shape = [..], dtype = ..)]` 标在 `type 名字 = <具体格式>;` 上，生成
`MODEL`、`const _: () = <名字 as Format>::CHECK;` 与一条校验测试；注册表里的 `model:` 只能写
`..::MODEL`。两侧互为镜像，由 `crates/testkit/src/contracts.rs` 与 `src/mirror.rs` 判，且**没有豁免节**：标了却不注册
永远错（发布出去的二进制里没有这条算子），注册的语义表不是注解生成的也永远错——出现例外说明契约
本身该改，而不是某个 crate 该豁免。

## 五、占位策略

一个 crate 还没实现时，它占的是**接口形状**，不是 `add(2, 2)`：类型、trait、字段名与错误语义先定，
函数体后填。判断标准是"删掉它会谁编译失败"——如果谁都不失败，它就该从 `members` 里移出去，而不是
留在 workspace 里复制同一个占位函数。

片段层没有这条余地：片段正文是**要被设备编译器读的代码**，不是形状。判据见第七节的语法门——
每台设备的每个片段都必须在该设备档案声明的语言与版本下解析通过，正文写坏时门是红的，
注释里写"留待以后"不算实现。

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
| `crates/shard/src/gate.rs` + `tests/fragments_tests.rs` | CI `test` job | 片段正文没被设备自己的语言接受：`GATES` 给每门语言登记一个语法前端，用例把六台设备的每个算子 × 每个 tile 组合真正拼成翻译单元（`substitute::render` → `emit::assemble`，与出厂文本逐字节同源）再交给前端解析；语言版本取自档案 `compile.flags`（`GateSpec::check`），门不许自己再写一份。片段正文写坏时反向用例必红——"占位不报错"在这里不成立 |
| `crates/testkit/src/{layer,ownership,boundary}.rs` | `cargo test -p llm-testkit --lib` | 成员集合、边集合与方向、所有权表与 `cargo metadata` / 源码不一致；`prof` / `cutile` / `shard` 退回 `Result<_, String>`；本文正文点名的 `crate::名字` 不存在；`allow(unsafe_code)` 不在 ADR 0035 §6 的白名单里；机制层（`cutile`）的公开项以厂商名起头；设备后端没在 `cli` 的产品路径上装配 |
| `crates/testkit/src/{contracts,mirror}.rs` + `.config/lints/vocabulary.toml` | `cargo test -p llm-testkit --lib` | 闭集 / 位集被手写第二份、白名单过期、注解与注册表不互为镜像 |
| `#[model]` 生成的两个 `const` + 一条测试 | 被标注 crate 的 `cargo test` | 格式自证不成立（粒度 ≤ 0、派生轴除数 ≠ 粒度）；注解的 `dtype` 与 `.k` 可写输出不是同一个类型 |

`cargo metadata` 只能告诉你"谁依赖谁"，不能告诉你"方向对不对"——后者由本文的层表定义，再由
`architecture.rs` 执行。新增成员或调整依赖方向时，**先改文档再跑 `cargo test`**：不一致会直接失败，
不再依赖评审清单。
