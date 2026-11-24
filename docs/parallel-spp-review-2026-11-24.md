# 分布式 / 并行 S++ 收口分账（2026-11-24）

> 范围：`crates/shard` 集合通信与三维并行本体 + `train` 分布式装配接缝。
> 评分口径：本仓五维自评（能力 / 性能 / 质量 / 代码量 / 门禁可执行性），0–10；不沿用外部框架打分。
> 证据分级：**confirmed** = 源码/metadata/构造期判据回链命中；**inferred** = 由相邻代码推断；**unknown** = 本轮未取得运行/编译观测。

## 1. 已拍板约束

- 范围含跨进程原型后端，但只允许 std 安全通道；NCCL / ROCm / 网络 / GPU 集合通信不在本轮闭合（ADR-0035 仍 Proposed，workspace `deny(unsafe_code)`，仓内无网络/FFI 依赖）。
- 减法仅限 shard 本体内部；16 个并入的 `optim` 模块及其测试保持原位、不拆移、不动架构边表。
- 评分按本仓五维自评，不与 Burn 的行数/方法数做机械倍数对比。

## 2. 本轮落地（confirmed，回链源码）

### 2.1 元素档泛型化（`collective.rs`）

- 新增自包含标量约束 [`ReduceScalar`]：`Copy + Default + 'static + Send + Sync + Add + Mul + PartialOrd`，附带 `ZERO / ONE / MIN_LIMIT / MAX_LIMIT` 四个中立元；浮点档覆写 `max` / `min` 保留 NaN 传播，整数档走偏序默认实现。
- 已实例化：`f32` / `f64`，整数 `u8/u16/u32/u64/i8/i16/i32/i64/usize/isize`（宏批量实现）。
- [`Collective`] 端口参数化为 `Collective<T: ReduceScalar = f32>`；`all_reduce / all_gather / broadcast / reduce_scatter` 缓冲由 `&mut [f32]` 改为 `&mut [T]`，缺省档仍为 `f32`，既有 `C: Collective` 不显式写参数仍指 f32。
- [`Reduce`] 的 `identity / apply / fold` 泛型到 `<T: ReduceScalar>`；旧的 f32 闭包行表（单位元 + `fn(f32,f32)->f32`）移除，单位元按类型常量派生，极值走 `ReduceScalar::max/min`。
- [`Local`] 泛型为 `Local<T: ReduceScalar = f32>`，镜像表 `Vec<Vec<T>>`，零值填 `T::ZERO`。
- [`Remote`] 占位与 [`FileRemote`] 字节协议固定 `Collective<f32>`：文件槽位是小端 4 字节布局，跨进程序列化档不参与泛型化。
- 约束自包含在 shard 内，**未新增 `shard → tensor` 边**（不反向依赖 tensor 的元素标记层级）。
- 浮点 `f32/f64` 两份逐字平行的 `max/min` NaN 覆写收编进 `impl_float_scalar!` 宏批量实例化（与整数族宏同构），消除两份 32 行重复样板。

### 2.2 并行策略配置层（`policy.rs`，新文件）

- [`ParallelConfig`] 纯数据：`tp / pp / dp` 三维度数 + `reduce` 归约种类 + `average` 平均开关 + `priority` 通信优先级；不读 toml、不发起通信，只持部署形态。
- 构造：`single()`（1×1×1 / Sum / 平均 / Normal，恒合法）、`new(tp,pp,dp)`（零维与 `u32` 上界交由 `Topology::new` 同一判据拒绝）、`topology()` 派生拓扑。
- 覆盖链：`with_reduce / with_average / with_priority`；访问器 `tp/pp/dp/reduce/average/priority`。
- 与内核工具链的 [`RuntimeConfig`] 分属两条旋钮，互不混入。

### 2.3 优先级（`policy.rs`）

- [`Priority`] 闭集 `High / Normal / Low`，`#[repr(usize)]`；`ALL` 表、`as_str`、`ordinal`（越小越先调度）、`from_name`（反解短名，`None` 表未知名）；实现 `Ord` 可排序。
- 优先级作为策略元数据挂进 `Distributed`；**批量多组并发时的排队排序执行点尚未接线**（当前训练循环一次只同步一个 DP 组，见 §5.2）。

### 2.4 训练装配接缝（`train/src/session.rs`）

- `Distributed<E: Collective>` 把原 `average: bool` / `reduce: Reduce` 两个字段收编进 `policy: ParallelConfig`。
- 新增 `Distributed::from_config(rank, collective, policy)`；保留既有 `new(topology, rank, collective)` 与 `with_average`，新增 `with_reduce / with_priority / policy()`。
- `sync_grads` 经 `policy.reduce()` / `policy.average()` 取口径；`none()` 仍为恒等回路。
- `scale` 补依赖注释：各槽元素互相独立、无串行依赖；安全 Rust 边界内无可调用的整块浮点向量化原语，故保留逐元素除法，不引入 rayon / unsafe SIMD。
- `new` / `from_config` 逐字重复的 rank 越界校验抽进私有 `ensure_rank` helper，两处共用同一判据与站点标签。

### 2.5 测试与登记

- `tests/policy_tests.rs`：优先级短名/序数/反解一致、单进程恒等策略、三维派生拓扑、零维拒绝、旋钮独立覆盖、非法度数带 `Code::Invalid`、策略拓扑与直接构造一致（7 例）。
- `docs/ARCHITECTURE.md` 第三节类型表登记 `ReduceScalar / ParallelConfig / Priority`（shard 属）；依赖边未变。

### 2.6 并行策略注解入口（`annotation` crate，本轮新增）

- 第三入口 `#[parallel(..)]`：键与运行期 `shard::ParallelConfig` 严格一一对应，单一事实源——`tp / pp / dp`（正整数，缺省 1）、`reduce`（闭集 `sum/max/min/prod`，缺省 `sum`）、`average`（`true/false`，缺省 `true`）、`priority`（闭集 `high/normal/low`，缺省 `normal`）。
- 三层复用既有架构：`syntax.rs` 提供游标/键表工具与被标注项判据；`decl.rs` 解析出 `pub(crate) struct ParallelDecl`（六件事实）；`render.rs` 把声明渲染成 `#[doc(hidden)] #[allow(dead_code)] fn __parallel_config() -> ::shard::ParallelConfig`，走 `ParallelConfig::new(..).expect(..).with_reduce(::shard::Reduce::{..}).with_average(..).with_priority(::shard::Priority::{..})`。
- 注解层不反向依赖 shard/nn：生产依赖仅 `proc-macro2/quote/syn`，生成代码以字符串路径引用 `::shard::`，类型解析推迟到消费方 crate 编译期（与既有 `#[spec]/#[model]/#[dsl]` 同构）。
- 编译期拒绝：零维度数、闭集越界、布尔非法、未知键、重复键；渲染产物经括号配平断言。`decl_tests` 两例（缺省策略 / 9 条非法用例）、`render_tests` 两例（全量覆盖 / 缺省求和-平均-普通）。
- 实测：`cargo test -p llm-annotation --lib` **48 passed / 0 failed**；`clippy -p llm-annotation --lib` 0 警告；`fmt -p llm-annotation --check` 0 diff。
- 现状缺口（本轮已闭合，见 §2.9）：注解产物原无消费方，`__parallel_config()` 访问器为 `allow(dead_code)`；`::shard::` 路径推迟到消费方编译期解析。

### 2.7 并行计划层（`plan.rs`，新文件）

- [`Plan`] 由拓扑与本 rank 派生三维归属：`tp_idx / pp_idx / dp_idx` 三轴下标 + 三条轴通信组（`tp_group / pp_group / dp_group`）+ PP 流水层段 `layer_range(layers)`。
- 组号压缩公式在计划层内固定：TP 组号 `dp·pp + pp_idx`、PP 组号 `dp·tp + tp_idx`、DP 组号 `pp·tp + tp_idx`，与 [`Topology::group`] 同一套 rank 布局；`Plan::new` 委托 `topology.idxs` 做 rank 越界判据，不另造校验。
- `layer_range` 与 [`Tiling`] 同口径要求整除：`layers == 0` 或不能被 `pp` 整除即在构造期拒绝，不保留"前几段多一层"的不均匀切分。
- 纯只读数据：不持缓冲、不发起通信；`single()` 为 1×1×1 全 0 下标恒合法视图。
- 对拍（`tests/plan_tests.rs`，6 例）：单进程全下标 0 与整段层范围；2·3·2 下 rank 2 的 TP/PP/DP 组成员对照拓扑表；非整除与零层数拒绝；3·2·4 全 rank 必属自己三轴组；12 层 4 段层段恰好划开一遍无重叠无遗漏。

### 2.8 优先级有序提交队列（`schedule.rs`，新文件）

- [`Job<T>`] 绑定调度优先级与泛型载荷；[`Queue<T>`] 入队不排序，`drain()` 时按 `Priority` 序数稳定排序——同优先级保持入队先后，不打乱同档提交次序。
- 只决定同一批次内的提交顺序，不改变任何归约数值；std `Vec` 实现，不引线程池、不 `unsafe`。
- 对拍（`tests/schedule_tests.rs`，3 例）：空队列排空为空；乱序入队的 Low/High/Normal/High 排空为 High·High·Normal·Low 且同档保序；载荷经队列原样解出。

### 2.9 注解端到端消费方（`train` dev 集成测试）

- `train` dev-dependency 引入 `annotation`（工具成员边，不进生产分层）；`tests/parallel_config_tests.rs` 在测试描述类型上挂 `#[parallel(tp=2, pp=1, dp=2, reduce="max", average=false, priority="high")]`，调用生成的 `__parallel_config()` 取出 [`ParallelConfig`]，再经 `Distributed::from_config(0, Remote, policy)` 装配。
- 至此打通 `#[parallel]` → `__parallel_config()` → `Distributed::from_config` 全链：`::shard::` 字符串路径在消费方编译期解析为真实类型，注解键与运行期配置仍是单一事实源，未造第二份键表。
- 裸 `#[parallel]` 用例回落单进程默认口径（1×1×1 / Sum / 平均 / Normal）。实测该文件 **2 passed / 0 failed**。
- 边界：消费方落在 dev 测试侧，生产 `Session` 装配仍由既有 `from_config / new` 驱动；模型层把注解访问器接入真实 `nn` 装配（TP 权重切分、PP 层段）属 §5.1 未闭合项。

### 2.10 模型层归属计划（`placement.rs`，新文件）

- [`Assembler`] 消费 `nn::Stack` 层清单（`shard → nn` 既有边，不反向让 nn 认识 shard），由 [`Plan`] 或 [`ParallelConfig`] 驱动，逐条产出 [`Placement<T>`]：
  - **PP**：层名首个数字段解析为块号，落在 `Plan::layer_range` 段内才 `held`；无块号的非分层参数（嵌入、归一化、块外项）所有 PP 副本持有；
  - **TP**：段内投影按后缀角色定切法（`.q/.k/.v` 列切、`.o` 行切，未登记角色回退列切），用 [`Tiling`] 算本 rank 分片形状与权重行/列范围；段外层不建分片计划；
  - **DP**：副本不切层不切权重，归属视图不做 DP 裁剪（DP 只驱动梯度归约）。
- [`Placement<T>`] 载荷泛型：装配侧自带的层句柄/键原样带回，计划层不解引用；[`LinearGeo`] 是脱离 `nn::Linear` 引用的投影几何只读副本。`Assembler::from_config(rank, &policy)` 与 `Distributed::from_config` 走同一条策略→拓扑判据。
- 只产归属几何与权重范围计划：不装载真实权重、不发起通信。
- 对拍（`tests/placement_tests.rs`，7 例）：单进程全持完整形状；PP=2 两块各归一个 rank、非分层双副本持有；TP=2 Q 列切输出维、O 行切输入维；段外层无分片计划；每个块恰好被一个 PP rank 持有；奇数输出宽不可整除拒绝；后缀切法表逐项核对。
- 端到端 dev 链（`train/tests/parallel_config_tests.rs` 第 3 例）：注解策略 → `Assembler::from_config` → `place_stack` 归属一份 nn 栈，分片形状断言通过。

### 2.11 归属几何驱动的本地权重子块切片（`tiling.rs` / `placement.rs`）

- [`Tiling::split`] 泛型到 `<E: Copy>`：给定行主序完整权重 `&[E]` 与份号，产出本 rank 本地子块 `Vec<E>`。列切（输出维）行段在缓冲里连续，一次切出整段，不逐元素拷；行切（输入维）列段跨行分散，外层按行取列段（数据布局决定的分散拷贝，保留外层循环并注释理由），每行内部整块 `extend_from_slice`。
- 构造期校验：`full.len()` 必须等于完整形状积（`checked_mul` 防溢出），不符报 [`Code::Invalid`]；份下标越界复用 [`Tiling::range`] 同一条判据。元素档约束只要 `Copy`，独立于归约标量 [`ReduceScalar`]——切片不是归约，不挂加法/乘法/偏序。
- [`Placement::split`] 委托入口：段外层或非投影层返回 `None`；段内投影按归属时记下的本 rank TP 份号调用 [`Tiling::split`]。`Placement` 新增私有 `part` 字段记份号，不改变既有访问器。
- 只产本地子块：不装载进 serve `Graph`/`Params`、不发起通信；行切子块合并仍停在 [`Sharding::merge`] 的求和归约标注，运行时接线未闭合。
- 对拍（`tests/tiling_tests.rs` 4 例 + `tests/placement_tests.rs` 2 例）：列切两份按行序拼回原样；行切每行左右两段接回原行；u32 档验证泛型元素独立；长度不符与份号越界拒绝；归属侧段内投影子块长度等于分片形状积，嵌入层与段外层返回 `None`。train dev 第 4 例：注解策略 → 归属 → 完整权重取本 rank 子块，首行元素与完整权重首行一致。

## 3. 五维自评

| 维度 | 分 | 依据与边界 |
| --- | --- | --- |
| 能力 | **8 / 10** | 集合端口元素档泛型、策略三旋钮、`#[parallel(..)]` 入口、`Plan` 三轴视图、`Job/Queue` 排队原语就位；归属计划接到 `nn` 层清单（PP 段过滤、TP 列/行几何）；本轮再把几何推进到**本地权重子块切片**（泛型 `Copy` 档，列切整块、行切按行取列段，拼回原样对拍），注解策略→归属→本地子块 dev 链打通（confirmed）。缺口仍在：子块尚未装载进 serve `Graph`/`Params`；PP 跨段流水调度、TP 行切求和归约未接运行图；接通的运行通路只有 DP 梯度同步；真实跨进程/设备通信缺（§5.1/5.3）。既有对比基线分布式维 llm=3。 |
| 性能 | **6 / 10** | `reduce_into` 直写调用方缓冲、一次归约不分配堆；组内归约内层预取镜像切片；标量安全后端无 SIMD，平均除法保留循环（已注释保留理由）。切片侧列切行段连续、一次整块 `to_vec`；行切列段不连续，外层按行拷（布局决定的分散拷贝），行内整块复制；归属与切片都是装配期动作，不进前向/反向热路径；队列排序是批次提交点稳定排序。未新增热路径通路，无基准测量观测（unknown），不夸大量化收益。 |
| 质量 | **9 / 10** | 构造期校验（零维 / 上界 / rank 越界 / 宽度不符 / 组非空升序 / 层段整除 / TP 切维整除 / 切片形状积与溢出 / 份号越界）、浮点极值 NaN 传播、归约闭集与行序断言；计划层全 rank 归属与层段无重叠无遗漏对拍；归属侧每块恰归一个 PP rank、非分层双副本、段外无计划对拍；切片拼回原轴/原行、泛型档、长度不符拒绝对拍；队列同档保序；注解编译期闭集/度数/布尔/重复键拒绝 + 端到端类型解析；新顶层类型回登所有权表，dev 工具边回登边表。 |
| 代码量 | **6 / 10** | 上一轮本体减法：移除 f32 闭包行表、浮点实例化收编进宏、rank 越界校验抽 helper。本轮切片能力加在既有 `tiling.rs`/`placement.rs` 方法上，未新增源文件；测试追加到既有测试文件；16 个并入 optim 模块保持原位。能力件仍以新增为主，未再拆本体。 |
| 门禁可执行性 | **7 / 10** | 本批 crate 绿窗口实测：`llm-shard` lib **162**、`llm-train` lib **235** + doctest **5**、`llm-nn` **528**（4 ignored）、`llm-annotation` **48** 全 0 failed；`cargo clippy` 本批 lib 0 警告；本批文件 `fmt` 0 diff。`llm-testkit` lib **99 passed / 1 failed**：唯一失败是上游 cutile 在途一元算子族的未登记类型 `cutile:UnaryOp`（源码 `UnaryFn`/`printer.rs`/`unary_tests.rs` 一条在途线，非本批并行文件），按约定未介入。其余残留：上游 device Metal pointwise 1 条、fuzz 两条 broken link、`cargo deny` 受网络限制；全仓 doc / `--all-targets` clippy 未在本轮跑完。 |

> 结论：距 "S++ / 超越 Burn 10 分" 仍未达成。本轮闭合了**本地权重子块切片算法**；子块装载进图、PP 跨段流水、TP 行切运行归约、生产装配持有归属结果、跨进程后端仍是缺口。分数为绿窗口实测口径，被上游半成品阻塞的 crate 不重复计绿。

## 4. 与 Burn 对照（沿用 §10 已核事实，不重复取证）

- Burn 0.21 以可微集合通信为底座重做分布式；0.22 内置 `tensor::distributed`、`DistributedContext`、`ExecutionStrategy::ddp(devices, DistributedConfig)`；Remote 装饰器走 Iroh 加密跨网；TP/PP 模型切分主要靠用户侧组织。
- 本仓：拓扑/切分/Reduce/集合端口结构在位，DP 梯度同步接通；无内置分布式执行策略对象、无可微集合通信、无加密 remote 网络后端。

## 5. 未闭合项与阻塞

### 5.1 TP / PP 未接运行图

归属计划已接 `nn` 层清单：`Assembler` 按块号做 PP 段过滤、按后缀角色出 TP 列/行分片几何（§2.10）；本轮再闭合**本地子块切片算法**：`Tiling::split`/`Placement::split` 按几何把完整权重取成本 rank 本地子块，拼回原样对拍通过（§2.11）。仍未闭合：

- 本地子块尚未驱动**真实装载**：`Params`/装配管线仍按整块权重绑定，没有把 `Placement` 子块接进图权重槽；生产装配侧（`serve` Graph / `train` tape）尚未持有 `Assembler` 归属结果——当前消费方在 dev 测试；
- PP 只做了段内层过滤，跨段的流水调度（段间激活传递、micro-batch 排队）未接；
- TP 行切后的求和归约还停在 `Sharding::merge` 标注，未接 `Collective` 运行调用；列切子块直接进下一层也未在图里接线；
- 嵌入/归一化等非投影参数的归属策略未定义（当前不切片，各副本持有）。

### 5.2 优先级排序执行点

`Priority` 序数/比较与本轮 `Queue<T>` 排队原语就位（按序数稳定排序、同档保序）；但一次训练步仍只同步一个 DP 组，不存在多组并发提交场景，队列尚未被 `Distributed` 持有/驱动。批量多组并发接线待 TP/DP 多组装配落地后闭合。

### 5.3 真实跨进程 / 设备通信

`FileRemote` 是文件槽位跨进程原型（小端、无 unsafe）；`Remote` 为未挂载占位。loopback socket / 共享内存 / NCCL / ROCm / GPU 集合通信受 ADR-0035、`deny(unsafe_code)` 与无网络/FFI 依赖阻塞，本轮不闭合。

### 5.4 上游并行流半成品与残留告警

取证期间工作树存在外部并行工作流连续提交与未提交文件周期收编。本轮收尾绿窗口实测：`llm-annotation` lib **48 passed / 0 failed**；`llm-testkit` lib **100 passed / 0 failed**；`llm-shard` lib **156 passed / 0 failed**；`llm-train` lib **234 passed / 0 failed** + doctest **5 passed / 0 failed**；`llm-nn` **528 passed**（4 ignored）。

过程中 cutile 多次进入合并中间态：`E0515`（子节点返回 `&[&Expr]` 临时切片悬垂）、`E0282`（类型标注缺失）、`E0592`（`children / map_children / rebuild_unary / rebuild_binary / rebuild_ternary` 在 `expr.rs` 与 `kernel.rs` 跨文件各定义一份，同 impl `kernel::Expr`）、`E0034`（多个同签名项在作用域内）。这些方法本应只住 `expr.rs`（子节点遍历与按元数重建的分派单一家），`kernel.rs` 内同名副本是迁移残留；该中间态在上一轮收尾前被上游收编——`kernel.rs` 副本已移除。

当前残留均属上游线或环境、非本批并行文件——

- `llm-device` Metal pointwise 对拍 1 条失败：`remainder_f32 Metal 输出 对拍失败：max_rel_err=1.000e0 > 1.000e-5（第 3 个元素 got=-1.125 want=-0.125）`；改动源文件 `crates/device/tests/metal/correctness/pointwise.rs`（263 行在途改动）与 `crates/prof/src/ops/{max,min}_f32.toml` 均非本批并行文件，本轮未介入，待上游收编后复跑；
- `llm-fuzz` `lib.rs:3/22` 两条 broken intra-doc link（`[`check`]` 指向作用域内不存在的 `check` 项），`RUSTDOCFLAGS=-D warnings` 下阻断全仓 doc；上游既有缺陷，本轮未改 fuzz；
- `cargo deny check` 因当前环境无法访问 github.com rustsec advisory-db（`SSL_ERROR_SYSCALL`）未执行，属网络限制而非代码供应链告警；
- 边表经本轮复核：`train → annotation` 为 dev 工具边，已补登 `docs/ARCHITECTURE.md`；实测 `cargo metadata` 运行时边 69、`+dev` 边 55、总计 **124** 条，文档计数与门禁 `testkit::layer` 断言一致；
- 干净窗口 `clippy --workspace --lib` 0、`fmt --all --check` 0 diff；`--all-targets / --all-features` clippy 与全仓 doc 未在本轮跑完。

这些不属于 shard 并行主线，未在本轮擅改其设计。全量门禁待上游 fuzz doc 链接与网络条件具备后按 §6 一次性复跑。

### 5.5 注解端到端消费方（dev 链已通，生产装配未接）

`#[parallel(..)]` → `__parallel_config()` → `Distributed::from_config` 与 `Assembler::from_config` → `place_stack` 两条 dev 链均在 `train` 集成测试打通（§2.9/§2.10，3 passed / 0 failed），`::shard::` 路径在消费方编译期解析为真实类型。剩余缺口：消费方目前只在 dev 测试侧，生产 `Session`/`serve` 装配尚未持有归属结果；那一步与 §5.1 的真实权重切片装载同批闭合，在此之前不造第二事实源。

## 6. 收尾门禁命令（待上游闭合后一次跑）

```
cargo test --workspace
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo fmt --all --check
RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps
cargo deny check
cargo test -p llm-testkit --lib
```

成员名 / 测试目标名以 `cargo metadata --no-deps` 当前结果为准；旧 `llm-tool` / `tool gate` 命名可能已随 `tools` 改名变化，testkit 架构/所有权/层边门经 `--lib` 跑（旧 `--test architecture/contracts` 目标名已失效）。
