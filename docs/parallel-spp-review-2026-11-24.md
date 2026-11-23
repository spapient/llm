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

### 2.2 并行策略配置层（`policy.rs`，新文件）

- [`ParallelConfig`] 纯数据：`tp / pp / dp` 三维度数 + `reduce` 归约种类 + `average` 平均开关 + `priority` 通信优先级；不读 toml、不发起通信，只持部署形态。
- 构造：`single()`（1×1×1 / Sum / 平均 / Normal，恒合法）、`new(tp,pp,dp)`（零维与 `u32` 上界交由 `Topology::new` 同一判据拒绝）、`topology()` 派生拓扑。
- 覆盖链：`with_reduce / with_average / with_priority`；访问器 `tp/pp/dp/reduce/average/priority`。
- 与内核工具链的 [`RuntimeConfig`] 分属两条旋钮，互不混入。

### 2.3 优先级（`policy.rs`）

- [`Priority`] 闭集 `High / Normal / Low`，`#[repr(usize)]`；`ALL` 表、`as_str`、`ordinal`（越小越先调度）、`from_str`；实现 `Ord` 可排序。
- 优先级作为策略元数据挂进 `Distributed`；**批量多组并发时的排队排序执行点尚未接线**（当前训练循环一次只同步一个 DP 组，见 §5.2）。

### 2.4 训练装配接缝（`train/src/session.rs`）

- `Distributed<E: Collective>` 把原 `average: bool` / `reduce: Reduce` 两个字段收编进 `policy: ParallelConfig`。
- 新增 `Distributed::from_config(rank, collective, policy)`；保留既有 `new(topology, rank, collective)` 与 `with_average`，新增 `with_reduce / with_priority / policy()`。
- `sync_grads` 经 `policy.reduce()` / `policy.average()` 取口径；`none()` 仍为恒等回路。
- `scale` 补依赖注释：各槽元素互相独立、无串行依赖；安全 Rust 边界内无可调用的整块浮点向量化原语，故保留逐元素除法，不引入 rayon / unsafe SIMD。

### 2.5 测试与登记

- `tests/policy_tests.rs`：优先级短名/序数/反解一致、单进程恒等策略、三维派生拓扑、零维拒绝、旋钮独立覆盖、非法度数带 `Code::Invalid`、策略拓扑与直接构造一致（7 例）。
- `docs/ARCHITECTURE.md` 第三节类型表登记 `ReduceScalar / ParallelConfig / Priority`（shard 属）；依赖边未变。

## 3. 五维自评

| 维度 | 分 | 依据与边界 |
| --- | --- | --- |
| 能力 | **5 / 10** | 集合端口元素档泛型、策略三旋钮、优先级键位就位（confirmed）。但 TP 权重切分与 PP 流水调度尚未接模型层；接通的只有 DP 梯度同步接缝；真实跨进程/设备通信缺（§5）。既有对比基线分布式维 llm=3。 |
| 性能 | **6 / 10** | `reduce_into` 直写调用方缓冲、一次归约不分配堆；组内归约内层预取镜像切片；标量安全后端无 SIMD，平均除法保留循环（已注释保留理由）。无基准测量观测（unknown），不夸大量化收益。 |
| 质量 | **7 / 10** | 构造期校验（零维 / 上界 / rank 越界 / 宽度不符 / 组非空升序）、浮点极值 NaN 传播、归约闭集与行序断言、策略对拍测试、新顶层类型回登所有权表。 |
| 代码量 | **6 / 10** | 消除 f32 闭包行表、收编 `Distributed` 两字段进策略对象；泛型约束样板使 `collective.rs` 净增（trait + 浮点/整数实例化）。本体未拆 16 个 optim 模块（裁决保留）。净增减待门禁行数对照复核（unknown）。 |
| 门禁可执行性 | **4 / 10** | 静态自查、类型闭合与登记到位。本轮取得的观测：`llm-shard` lib 测试 140 通过 / 1 失败（唯一失败 `gate::fragments_tests` 的 `remainder_f32_0` C 片段被 clang 拒绝，属 optim 并入模块的上游算子线）；`collective / policy / topology / tiling` 本体对拍全绿；`llm-train` lib `check` 通过。`llm-train` test 目标被 `samples_tests.rs:84` dataset 类型注解缺失阻塞（上游并行流改动，与本接缝无关）；全量 `clippy / fmt / doc / deny / 架构契约门` 未跑。 |

> 结论：距 "S++ / 超越 Burn 10 分" 未达成。接口面与配置面补齐是本轮可闭合的部分；运行能力、真实通信、注解入口与干净门禁观测仍是缺口。

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

### 5.4 上游并行流半成品阻塞门禁

取证期间工作树存在外部并行工作流连续提交与未提交文件周期收编。本轮观测：`shard` 与 `train` 的 **lib** 已能 `check` 通过，shard 本体测试全绿；残留阻塞集中在——

- `shard` optim 并入模块的内核片段测试：`remainder_f32_0` portable C 片段被 clang 拒绝（`gate::fragments_tests`）；
- `train` test 目标：`samples_tests.rs:84` `Chunks::new` 缺数据集类型注解（dataset 并行流改动）；
- 全量 `clippy / fmt / doc / deny / 架构契约门` 尚未在本轮跑。

这些不属于 shard 并行主线，未在本轮擅改其设计。全量门禁须待上游中间态闭合后按 §6 一次性复跑。

### 5.5 并行策略注解入口

`#[parallel(..)]` proc-macro 第三入口未落地：当前模型层 TP/PP 装配尚无消费方，先造注解产物会形成第二个事实源（违反仓内"注解即契约、行为在 trait 上"原则）。`ParallelConfig` 已作为注解键与运行期配置对齐的单一事实源；注解入口待模型层并行装配落地后补，复用 `annotation` 既有的 `Streams` 依赖倒置路径。

## 6. 收尾门禁命令（待上游闭合后一次跑）

```
cargo test --workspace
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo fmt --all --check
RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps
cargo deny check
cargo test -p llm-testkit --test architecture --test contracts
```

成员名 / 测试目标名以 `cargo metadata --no-deps` 当前结果为准；旧 `llm-tool` / `tool gate` 命名可能已随 `tools` 改名变化。
