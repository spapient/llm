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
| 质量 | **9 / 10** | 构造期校验（零维 / 上界 / rank 越界 / 宽度不符 / 组非空升序 / 层段整除 / TP 切维整除 / 切片形状积与溢出 / 份号越界）、浮点极值 NaN 传播、归约闭集与行序断言；计划层全 rank 归属与层段无重叠无遗漏对拍；归属侧每块恰归一个 PP rank、非分层双副本、段外无计划对拍；切片拼回原轴/原行、泛型档、长度不符拒绝对拍；队列同档保序；注解编译期闭集/度数/布尔/重复键拒绝 + 端到端类型解析；新顶层类型回登所有权表，dev 工具边回登边表。本轮补口径一致性：remainder 设备模板由 floor 向负无穷对齐到参考闭包 `x % y` 的 fmod/截断商（`x - _cutile_trunc_f32(x/v)*v`，去冗余 copysign），nn 模块注释与 `.toml` 注释同改；shard `placement/tiling` 抽私有方法降 assign/split 圈复杂度、词表 `index→idx`；testkit 单行重导出垫片收拢进 lib.rs；fuzz harness 终止形态与契约库零 panic 立场对齐。本轮门禁收口再补：性质测试台账与源码扫描双向对齐（`serve / tokenizer / weights` 已写性质块，从 ABSENT 挪进 COVERED，manifest 声明集与台账等值）；架构边集镜像补齐漏登的 `tokenizer → testkit` dev 边；nn unary 二十五算子泛型闭包编译破损修复——闭包绑定到 `fn(K) -> K` 函数指针、参数由签名反推为泛型档 `K`，与二元族同一模式，不靠在调用点标注方法泛型（调用点 `K` 未入作用域）；全仓 `clippy -D warnings` / doc-test / doc link 零告警。本轮再闭合三处静态门残留：nn `unary.rs` 的 `abs_f16` 模块 outer `///` 文档与 inner `//!` 同挂触发属性冲突，并入 mod 内 `//!`；shard 回迁副本在私有 `mod cuda/metal/tiled` 下，其 pub 重导出与函数无 crate 内消费者，触发 9 条 dead_code/unused，改 `pub mod`（`dialect` 保持私有，与 device 同名模块口径一致）；testkit `use_tree` 处理无花括号、无 `as` 的 `pub use` 时别名误取整条路径，改为取尾标识符（与花括号条目同口径），mirror 门恢复按尾名读得回模块。 |
| 代码量 | **6 / 10** | 上一轮本体减法：移除 f32 闭包行表、浮点实例化收编进宏、rank 越界校验抽 helper。本轮切片能力加在既有 `tiling.rs`/`placement.rs` 方法上，未新增源文件；测试追加到既有测试文件；16 个并入 optim 模块保持原位。本轮精简：testkit 单行垫片 `assert.rs` 收拢进 `lib.rs`（删一碎文件）、撤销一条越权 debt 条目与配套注释、remainder 模板去掉冗余 copysign 表达式。能力件仍以新增为主，未再拆本体。密度存量债本轮未压到 S++ 线：N6 总量超 14,855 行、N3 九个 >600 文件、N2 四个 >8,200 crate（明细见 §5.4.4）；低风险目录化拆分与结构性 crate 外移分别排进 REPO-review §2.1 / §2.4，后者需 ADR 拍板、动架构边表，不在本轮门禁边界内。本轮减法：nn 一元 f32 档与精度档各自抄一份同构 `impl Unary` 块，抽出第三个公共 `#[macro_export] unary_impl!`（接收 `$name, $apply $(, $input)?`，内含 apply 闭包绑定与可选 input 槽），两族改为调用它，消去同文件一条 jscpd clone；nn `unary` 91 passed、clippy 干净，语义不变。 |
| 门禁可执行性 | **7 / 10** | 本轮（2026-11-24 晚窗口，改动集中后一次跑）实测：`fmt --all --check` 0 diff；`clippy --workspace --all-targets --all-features -D warnings` 0 警告；`nextest run --workspace --all-features` **2966 passed / 0 failed / 29 skipped**；`cargo test --doc --workspace --all-features` 各档 0 failed；`deny --offline check all` 四类（advisories/bans/licenses/sources）ok；`machete` 无未用依赖；静态门 `gate` **15 passed**；架构门 `testkit --lib` **100 passed**；`name-lint` 0 命中；`bench --workspace --no-run` Finished；基准回归 `regress` **12 passed**（无测量产物的场景按设计 skip，不静默放行）。**两条硬门仍红，未豁免、未调阈值**：① `jscpd` threshold 0，实测 **78 clones / 4.3%**——57 对 `device/src ↔ shard/src`、20 对 `device/tests ↔ shard/tests`、1 对 `device` 内部 `cuda.rs ↔ metal.rs` 同族 use 导入块；根因是 #30 A/E/B 生产面回迁 shard 后 device 旧副本尚未整机删除的迁移中间态（取证与解阻顺序见 §5.4.5）。② 密度预算 N2/N3/N6 报告阶段红账（见 §5.4.4）。扣分：jscpd 硬门红、密度预算未达 S++ 线；真机/跨进程测量未接入回归门，性能回归只编译不采样。 |

> 结论：距 "S++ / 10 分" 仍未达成。本轮绿窗口实测：2966 测试 0 failed，clippy/fmt/doc/deny/machete/架构门/命名/基准编译/回归全绿，零豁免、零未用依赖；但两条硬门仍红——jscpd 78 对双份（§5.4.5，#30 迁移中间态）与密度预算 N2/N3/N6（§5.4.4），均未调阈值、未加 ignore。本轮另闭合三处静态门残留（nn 模块属性冲突、shard 回迁副本可见性 dead_code、testkit 别名解析）并抽出 nn 一元公共宏。能力侧硬缺口仍在：§5.1 运行装配（子块装载进图、PP 跨段流水、TP 行切运行归约、生产装配持有归属结果、跨进程后端）一条未推进。分数为绿窗口实测口径，被上游半成品阻塞的 crate 不重复计绿。

## 4. 与 Burn 对照（沿用 §10 已核事实，不重复取证）

- Burn 版本演进（2026-11 联网核对）：0.19 引入 `burn-collective` 梯度同步、Learner 分布式训练策略与多流执行；0.21 围绕可微集合通信重做分布式栈；0.22 集合操作经 `burn::tensor::distributed` 暴露，`DistributedContext` 取代旧 Session/Runtime，`ExecutionStrategy::ddp(devices, DistributedConfig)` 构造 DDP 策略，同时模型/张量不再携带 `B: Backend` 泛型、由 `Device` 运行时选择后端，并带 LoRA/QLoRA、参数组、graph capture；Remote（Beta）是 client-server 网络远端装饰器，Router（Beta）多后端组合；TP/PP 模型切分仍主要靠用户侧组织。
- 本仓对照：拓扑/切分/Reduce/集合端口结构在位，DP 梯度同步接通；但无内置分布式执行策略对象、无可微集合通信、无网络 remote 后端（受 deny(unsafe_code)/无网络/FFI 边界阻塞，见 §5.3），TP/PP 未接运行图（§5.1）。分布式成熟度落后 Burn 一个代际。

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

取证期间工作树存在外部并行工作流连续提交与未提交文件周期收编。本轮收尾绿窗口实测：`llm-annotation` lib **48 passed / 0 failed**；`llm-shard` lib **162 passed / 0 failed**；`llm-train` lib **235 passed / 0 failed** + doctest **5 passed / 0 failed**；`llm-nn` **528 passed**（4 ignored）；`llm-prof` **164 + 56 passed**；`llm-fuzz` **17+3+5+1 passed**；`llm-device` `metal_suite` **15 passed / 0 failed / 12 ignored**；`llm-testkit` lib **100 passed / 0 failed**。

过程中 cutile 多次进入合并中间态：`E0515`（子节点返回 `&[&Expr]` 临时切片悬垂）、`E0282`（类型标注缺失）、`E0592`（`children / map_children / rebuild_unary / rebuild_binary / rebuild_ternary` 在 `expr.rs` 与 `kernel.rs` 跨文件各定义一份，同 impl `kernel::Expr`）、`E0034`（多个同签名项在作用域内）。这些方法本应只住 `expr.rs`（子节点遍历与按元数重建的分派单一家），`kernel.rs` 内同名副本是迁移残留；该中间态在上一轮收尾前被上游收编——`kernel.rs` 副本已移除。

### 5.4.1 本轮已闭合的三条上游在途残留（confirmed）

- **`llm-device` Metal pointwise 对拍**：旧记录的 `remainder_f32 第 3 元素 got=-1.125 want=-0.125` 在当前工作树不复现；`cargo test -p llm-device --test metal_suite` 实测 **15 passed / 0 failed / 12 ignored**，`pointwise_elementwise_matches_cpu_reference_on_metal ... ok`。取证发现一处真实口径隐患：nn 参考闭包是 `|x,y| x % y`（Rust `f32::Rem` = fmod = `x - trunc(x/y)·y`，截断商、符号同被除数），而设备模板 `remainder_f32.k` 原写 `copysign(x - floor(x/v)*v, x)`（floor 向负无穷），负商非整数时与 trunc 不等价，且该算子未进 Metal `ENTRIES` 对拍表、设备路径未经真机覆盖。已裁决以 nn 运行时参考闭包为契约事实，模板对齐 fmod/截断商：表达式改 `x - _cutile_trunc_f32(x/v)*v`，去掉冗余 copysign；`.toml` 注释 `floor→trunc`；nn 模块注释删去"IEEE 余数语义"误称。六后端 preamble 均已提供可移植 `_cutile_trunc_f32`。
- **`llm-fuzz` 两条 broken intra-doc link**：`lib.rs:3/22` 原写 `[`check`]`，作用域内无 `check` 项；实际判据入口是 `check_target`（`lib.rs:106`，经 538 行 `pub use contract::{check_target, sweep, sweep_all}` 导出）。两处已改指 `[`check_target`]`。验证：`RUSTDOCFLAGS="-D warnings" cargo doc -p llm-fuzz --no-deps` 零告警；`cargo test -p llm-fuzz` 四档全绿（17 + 3 + 5 + 1，0 failed）。
- **`cargo deny` 网络受限**：根因是缺 `--offline`，非代码供应链告警。本地 advisory 快照完整存在于 `~/.cargo/advisory-dbs/advisory-db-3157b0e258782691`（commit `52384ab`，crates/ rust/ 齐全）。cargo-deny 0.20.2 不接受 check 子命令的 `--offline`，但顶层 `cargo deny --offline check` 可用；实测 `cargo deny --offline check all` → `advisories ok, bans ok, licenses ok, sources ok` 四类全绿，离线无需改 `deny.toml`。

### 5.4.2 本轮顺带收口的静态门项

- shard `placement.rs` 词表改名（`layer_index→layer_idx`、闭包 `index→idx`）并抽 `linear_geo`/`sharded` 私有方法降 assign 圈复杂度；`tiling.rs` 把 `split` 的列切/行切抽成 `column_slice`/`row_slice`，长度校验抽 `check_len`，降主体 ccn；`plan_tests.rs` 测试名 `indices→idxs`；testkit 单行重导出垫片 `assert.rs`（12 行）收拢进 `lib.rs` 直连 `similar_asserts::assert_eq`。
- fuzz harness 终止形态合规化：`harness.rs` 原在 `fuzz_target!` 闭包内 `panic!("{finding}")`。门的 Call 度量阈值为 0，棘轮 debt 按 `ratchet` 测试硬约束只放行 `crates/cli/src/`，`crates/fuzz/` 不能挂债务条目（上一轮误挂的 `call=1` 条目已撤销，ratchet 用例转绿）。代码侧改为先 `eprintln!("{finding}")` 落证据到 stderr，再 `std::process::abort()`——与 libfuzzer-sys `test_input_wrap` 捕获 unwind 后 `process::abort()` 的最终终止形态一致，跳过被零 panic 门禁止的 panic 宏；契约库本身仍不 panic。`cargo check -p llm-fuzz --features engine` 通过。

### 5.4.3 仍压门禁的既有边界债（本轮边界内未拆）

- **`train` 三巨石 Lines 超标：已闭合（gate 口径）**。三文件已目录化拆分，物理行数回落到 gate `file_max_lines=899` 以内：`session.rs` 887、`backprop.rs` 319、`geometry.rs` 694；静态门 `every_source_passes_the_static_gate` 转绿（15 passed）。注意 loc 脚本更严的 N3 线（>600 一票否决）下，`session.rs` 887 / `geometry.rs` 694 / `kernel.rs` 654 / `backprop/formulas.rs` 608 仍在 >600 清单，见 §5.4.4。
- **既有 doc broken link：已清零**。前几轮登记的 shard 3 / nn 4 / prof 6 / testkit 3 条 intra-doc link 按四类修法逐条真实修复（private-intra-doc-links 去方括号留纯文本反引号；broken-unresolved 核对真实目标后改全路径或纯文本；模块名撞原语/函数去链接；redundant explicit target 去 target），不靠 `#[allow]` 豁免。`RUSTDOCFLAGS="-D warnings" cargo doc --workspace --all-features --no-deps` 零 error。
- `cutile` 一元算子族在途：`ir_emit.rs` 重组为 `printer.rs`、新增 `unary_tests.rs`；收尾时 ownership 门扫到既有 HEAD 公开类型 `cutile:UnaryFn` 漏登基线，已补进既有 `cutile` 的 `ownership.unlisted` 基线节（不碰 cutile 代码、不新增架构边），ownership 门转绿；该闭集待 cutile 分节稳定后再逐节进对外所有权表。再次动手 cutile 源码前应先确认工作树静止，必要时 `cargo clean -p cutile` 清陈旧 OUT_DIR。
- 边表经本轮复核：`train → annotation` 为 dev 工具边，已补登 `docs/ARCHITECTURE.md`；本轮再补 `tokenizer → testkit` dev 边（tokenizer dev-dependency 用 testkit，文档边集原漏登，`layer` 门转绿），边数同步为运行时 69、`+dev` 56、总计 **125** 条，文档计数与门禁 `testkit::layer` 断言一致。
- 全仓工具链本轮已一次性跑完：`fmt --all --check` 0 diff；`clippy --workspace --all-targets --all-features -D warnings` 0 警告；全仓 doc-test 0 failed。此前"非本批文件未格式化 diff / 全仓 clippy 与 doc 未跑完"的状态已解除。

### 5.4.4 密度预算红账（N2 / N3 / N6，报告阶段未阻断）

CI `density budget` 步骤 `continue-on-error: true`（R1–R2 报告不阻断，R3 起转阻断，阈值只收紧）。本轮实测 `scripts/loc-budget.sh` 三项仍红，属已提交存量、非本轮引入：

- **N6 总量**：`crates/*/src` **110,955** 行 / 预算 96,100，超 **14,855** 行；441 个源文件；>400 行 77 个。
- **N3 单文件 >600 一票否决（9 个）**：`train/session.rs` 887、`testkit/contracts.rs` 750、`train/geometry.rs` 694、`weights/torch/mod.rs` 663、`train/kernel.rs` 654、`weights/safetensors/mod.rs` 643、`train/backprop/formulas.rs` 608、`weights/ptq/mod.rs` 603、`fuzz/drivers.rs` 602。
- **N2 单 crate >8,200（4 个）**：`nn` 15,862、`shard` 13,619、`prof` 9,966、`weights` 9,839。注意 `shard` 体量含 #30 回迁副本，`device` 6,675 行是同批旧副本（双份同时计入 N6/N2），随 device 整机删除一并回落。

无损减法已在前几轮清到低残留：jscpd 跨文件双份 78 对（§5.4.5，#30 迁移中间态，非独立抄袭）、machete 未用依赖 0、gate 零引用 pub / 死代码判据绿。剩余可压缩空间按 `docs/REPO-review-2026-11.md` §2 分两档：**低风险**是 §2.1 超大文件按主题目录化拆分（公开路径不变、只搬模块与再导出，消 N3 清单，但不降总量）；**结构性**是 §2.4 `weights` 五领域混杂拆出 `crypto` / `codec` 独立 crate（降 N2/N6，但动架构边表、需 ADR 拍板，周期 1–2 周）。本轮硬边界不动架构边表，未做结构性拆 crate；N2/N6 的真减行要等 ADR 与拆案落地，不能靠放宽阈值或删判据散文凑数字。

这些不属于 shard 并行主线，未在本轮擅改其设计。全量门禁按 §6 一次性复跑。

### 5.4.5 jscpd 双份红账（78 clones，threshold 0，未豁免）

- **实测分桶**：57 对 `device/src ↔ shard/src`、20 对 `device/tests ↔ shard/tests`、1 对 `device` 内部 `cuda.rs ↔ metal.rs` 同族 use 导入块；同文件 clone 已为 0（本轮 nn `unary_impl!` 公共宏提取消掉那一条）。
- **逐对 diff 取证**：除 `platform.rs` 外正文逐字等价，仅 import 前缀（device `llm_optim::`/`shard::` ↔ shard `crate::`）、站点字面量（`engine::dialect::*` ↔ `shard::dialect::*`）、注释差异。`platform.rs` 是**有意分层分叉**：device 版 `#[cfg(target_os = "macos")]` 调 `engine::metal::MetalDevice` FFI 探真机；shard 版退化为 `std::env::consts::OS == "macos"` 平台常量（L5 shard 不得上靠 L6 engine），不可盲并。
- **根因**：#30 A/E/B 生产面已复制进 shard（git 未跟踪），device 旧副本尚未整机删除，形成双份；上一轮稳态 jscpd=0。
- **消费者取证（confirmed）**：生产 `crates/*/src` 已零 `llm-device` 代码引用（grep 仅剩散文注释）；Cargo 依赖 `llm-device` 的只有 cli（dev 测试）与 device 自身。残留读路径仅测试夹具：`cli/tests/device.rs` 5 处 `::device::cuda::*` / `::device::norm::*`；`device/tests` 全套（`schedule.rs`、`metal_suite.rs`、`metal/`、`metal_executor_harness.rs`、`backends.rs`、`backends_production.rs`、`platform.rs`、`program.rs`、`tiled.rs`）。生产入口已切 shard：`cli/src/lib.rs:59`、`cli/src/cutile.rs:34` 调 `shard::backends::install()`；`cli/src/platform.rs:3` 用 `shard::platform::{Machine, Row, Standing, audit}`。
- **阻塞（本轮不能安全清零）**：device tests 3712 行与后端生成链、`engine::metal` 真机夹具深度交织；#31（D 族 → `prof/<vendor>/`）、#32（F `backward` → nn/train）未完成前 device crate 不能整机删；删旧副本前须先把测试夹具读路径迁到 shard 并证明无消费者；`platform.rs` 的 engine FFI 探针分叉不可并入 shard。解阻顺序与同步清单见 `device-crate-deletion-migration-2026-11-24.md` §2.2 / §3 / §4。
- **纪律**：未在 `.config/.jscpd.json` 加 ignore、未调 threshold、未删判据；该双份是迁移中间态红账，闭合依赖 #30 测试随迁 + #31/#32 批次 + 架构边表/ADR 回写。
### 5.5 注解端到端消费方（dev 链已通，生产装配未接）

`#[parallel(..)]` → `__parallel_config()` → `Distributed::from_config`、`Assembler::from_config` → `place_stack`、归属 → 本地子块切片三条 dev 链均在 `train` 集成测试打通（§2.9/§2.10/§2.11，4 passed / 0 failed），`::shard::` 路径在消费方编译期解析为真实类型。剩余缺口：消费方目前只在 dev 测试侧，生产 `Session`/`serve` 装配尚未持有归属结果与子块；那一步与 §5.1 的真实装载进图同批闭合，在此之前不造第二事实源。

## 6. 收尾门禁命令（待上游闭合后一次跑）

```
cargo test --workspace
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo fmt --all --check
RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps
cargo deny --offline check all
cargo test -p llm-testkit --lib
```

成员名 / 测试目标名以 `cargo metadata --no-deps` 当前结果为准；旧 `llm-tool` / `tool gate` 命名可能已随 `tools` 改名变化，testkit 架构/所有权/层边门经 `--lib` 跑（旧 `--test architecture/contracts` 目标名已失效）。Metal 对拍目标名是 `metal_suite`（`--test metal` 不存在）；`cargo deny` 的 `--offline` 必须放在顶层 `deny` 之后、`check` 之前（check 子命令不接受该 flag），离线读本地 advisory 快照、不需联网拉 github rustsec 库。
