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
- 现状缺口：尚无模型类型挂载该注解，`__parallel_config()` 访问器当前无消费方（故 `allow(dead_code)`）；`::shard::` 路径要到消费方编译时才解析，端到端展开未取得观测（§5.5）。

## 3. 五维自评

| 维度 | 分 | 依据与边界 |
| --- | --- | --- |
| 能力 | **6 / 10** | 集合端口元素档泛型、策略三旋钮、优先级键位、`#[parallel(..)]` 声明入口就位（confirmed）。但 TP 权重切分与 PP 流水调度尚未接模型层；接通的只有 DP 梯度同步接缝；注解产物暂无消费方；真实跨进程/设备通信缺（§5）。既有对比基线分布式维 llm=3。 |
| 性能 | **6 / 10** | `reduce_into` 直写调用方缓冲、一次归约不分配堆；组内归约内层预取镜像切片；标量安全后端无 SIMD，平均除法保留循环（已注释保留理由）。本轮未新增性能通路；无基准测量观测（unknown），不夸大量化收益。 |
| 质量 | **8 / 10** | 构造期校验（零维 / 上界 / rank 越界 / 宽度不符 / 组非空升序）、浮点极值 NaN 传播、归约闭集与行序断言、策略对拍测试；注解编译期闭集/度数/布尔/重复键拒绝 + 括号配平对拍；新顶层类型回登所有权表；注解产物 `pub(crate)` 不引入新运行时依赖边。 |
| 代码量 | **6 / 10** | 本体减法：移除 f32 闭包行表、浮点 `f32/f64` 实例化收编进宏、rank 越界校验抽 helper；`collective.rs` 433→419 行。生产 diff 实测：`shard+train` **+42 / -56**（本体净减 14 行）；`annotation` **+263 / -5**（注解入口 + 两份对拍测试，要素新增）。生产合计 **+305 / -61**；泛型约束样板使集合端口净增。本体未拆 16 个 optim 模块（裁决保留）。与历史基线的全仓机制行数对照待整链门禁后复核（unknown）。 |
| 门禁可执行性 | **7 / 10** | 干净窗口整链实测：`cargo test --workspace --lib` **2565 passed / 0 failed**；`llm-annotation` **48**、`llm-shard` lib+集成各 **141**、`llm-train` lib **231** + doctest **5**、`llm-testkit` **100** 全绿。`cargo clippy --workspace --lib` 0 警告/错误；`cargo fmt --all --check` 0 diff；`RUSTDOCFLAGS=-D warnings cargo doc -p llm-annotation --no-deps` 干净。本批 `shard/annotation/train` 自身 clippy/doc/fmt 全 0。残留均属上游或环境：`llm-fuzz` 两条 broken intra-doc link（`lib.rs:3/22` 指向不存在的 `check` 项）；`cargo deny check` 因环境无法访问 github advisory-db（SSL_ERROR_SYSCALL）未执行；`--all-targets / --all-features` clippy 与全仓 doc 未在本轮跑完。 |

> 结论：距 "S++ / 超越 Burn 10 分" 仍未达成。本轮闭合了**注解入口**这一接口/配置面；运行能力（TP/PP 模型装配）、真实通信、注解端到端消费方与干净整链观测仍是缺口。分数为绿窗口实测口径，被上游半成品阻塞的 crate 不重复计绿。

## 4. 与 Burn 对照（沿用 §10 已核事实，不重复取证）

- Burn 0.21 以可微集合通信为底座重做分布式；0.22 内置 `tensor::distributed`、`DistributedContext`、`ExecutionStrategy::ddp(devices, DistributedConfig)`；Remote 装饰器走 Iroh 加密跨网；TP/PP 模型切分主要靠用户侧组织。
- 本仓：拓扑/切分/Reduce/集合端口结构在位，DP 梯度同步接通；无内置分布式执行策略对象、无可微集合通信、无加密 remote 网络后端。

## 5. 未闭合项与阻塞

### 5.1 TP / PP 未接模型层

拓扑与 `Tiling` 切分结构在位，但权重按 TP 列/行切分、PP 层分段与流水调度尚未接到 `nn` 装配与 `Session`。当前 `Distributed` 只驱动 DP 组内梯度同步。

### 5.2 优先级排序执行点

`Priority` 序数与比较就位，但一次训练步只同步一个 DP 组，不存在多组排队场景；按优先级批量下发集合操作的调度器待多组并发接线后落地。

### 5.3 真实跨进程 / 设备通信

`FileRemote` 是文件槽位跨进程原型（小端、无 unsafe）；`Remote` 为未挂载占位。loopback socket / 共享内存 / NCCL / ROCm / GPU 集合通信受 ADR-0035、`deny(unsafe_code)` 与无网络/FFI 依赖阻塞，本轮不闭合。

### 5.4 上游并行流半成品与残留告警

取证期间工作树存在外部并行工作流连续提交与未提交文件周期收编。本轮收尾绿窗口实测：`llm-annotation` lib **48 passed / 0 failed**；`llm-testkit` lib **100 passed / 0 failed**；`llm-shard` lib + 集成测试各 **141 passed / 0 failed**；`llm-train` lib **231 passed / 0 failed** + doctest **5 passed / 0 failed**。

过程中 cutile 多次进入合并中间态：`E0515`（子节点返回 `&[&Expr]` 临时切片悬垂）、`E0282`（类型标注缺失）、`E0592`（`children / map_children / rebuild_unary / rebuild_binary / rebuild_ternary` 在 `expr.rs` 与 `kernel.rs` 跨文件各定义一份，同 impl `kernel::Expr`）、`E0034`（多个同签名项在作用域内）。这些方法本应只住 `expr.rs`（子节点遍历与按元数重建的分派单一家），`kernel.rs` 内同名副本是迁移残留；该中间态在本轮收尾前被上游收编——`kernel.rs` 副本已移除，整链 `cargo test --workspace --lib` 复跑 **2565 passed / 0 failed**。

当前残留均属上游线或环境、非本批并行文件——

- `llm-fuzz` `lib.rs:3/22` 两条 broken intra-doc link（`[`check`]` 指向作用域内不存在的 `check` 项），`RUSTDOCFLAGS=-D warnings` 下阻断全仓 doc；上游既有缺陷，本轮未改 fuzz；
- `cargo deny check` 因当前环境无法访问 github.com rustsec advisory-db（`SSL_ERROR_SYSCALL`）未执行，属网络限制而非代码供应链告警；
- 上游 `device` 新增对 `storage` 的生产依赖，本轮已补登 `docs/ARCHITECTURE.md` 边表（`device → storage`），总数 123→124（`+dev` 边 54 不变）；`prof` 算子模板表 52→57 条同步订正；
- 干净窗口 `clippy --workspace --lib` 0、`fmt --all --check` 0 diff；`--all-targets / --all-features` clippy 与全仓 doc 未在本轮跑完。

这些不属于 shard 并行主线，未在本轮擅改其设计。全量门禁待上游 fuzz doc 链接与网络条件具备后按 §6 一次性复跑。

### 5.5 注解端到端消费方

`#[parallel(..)]` 入口本体已落地并在 annotation crate 内编译/测试通过（§2.6），但尚无模型类型挂载：生成的 `__parallel_config()` 访问器当前无人调用（`allow(dead_code)`），`::shard::` 路径要到消费方 crate 编译时才解析。模型层 TP/PP 装配落地后，由具体类型实现并行 trait 并经该访问器取 `ParallelConfig`，才构成端到端展开验证；在此之前不造第二事实源。

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
