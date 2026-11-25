# `crates/train` 训练架构与改造基线

日期：2026-11-25　范围：`crates/train`　性质：现状说明与后续改造约束，不把规划写成既有能力。

## 1. 实测基线

| 项目 | 当前值 |
|---|---|
| 生产文件 / 行数 | 13 / 7,971 |
| 测试文件 / 行数 | 16 / 5,728 |
| 最近完整训练测试 | 246 passed / 0 failed / 1 ignored |
| 静态质量门 | `crates/train` 零命中 |
| 生产文件门 | `file_min_lines = 300`（薄入口除外）、`file_max_lines = 899` |
| 其它硬门 | `ccn_max = 5`、`nloc_max = 60`、测试文件不超过 800 行 |

本文的模块名、配置字段、注解产物和恢复语义均对照当前源码。后续重构不得通过删测试、
降低门禁、引入 `unsafe` 或把职责转嫁给其它 crate 来改善表面数字。

## 2. crate 的职责与数据流

`train` 把样本、前向评估、反向梯度、参数更新和恢复状态串成一次训练闭环：

```text
token 缓冲
  → Loader / Dataset（借用批次）
  → Learner（前向、指标、优化编排）
  → Session（图、调度、参数槽、累积窗口）
  → autograd（反向计划与重放）
  → Bufs / Kernel（激活存储与参考计算）
  → Optimizer（裁剪、同步、参数更新）
  → Checkpoint / Snapshot（恢复事实与字节编码）
```

职责边界：

| 所属 crate | 负责 | train 不复制的内容 |
|---|---|---|
| `autograd` | tape、反向调度、显存计划、重放序列 | 不保存参数梯度、不执行优化器更新 |
| `nn` | 层定义、算子注册、参数角色、算子语义 | 不另建层注册表或参数角色表 |
| `shard` | TP/PP/DP 拓扑、分片归属、集合通信 | 不实现第二套 rank 布局或归约协议 |
| `storage` | 快照字节格式、矩状态与累积段的编码 | 不在训练循环里重新定义二进制布局 |
| `weights` | 权重存储与容器 | 不在会话里解释权重文件格式 |
| `tokenizer` | 文本与 token 的边界 | 不在数据加载器里实现分词 |

`train` 不拥有设备内核注册表，也不以“参考实现”为名复制 `nn` 的声明配置。

## 3. 模块职责图

crate 首页列出 9 个顶层模块；嵌套文件只是这些模块的实现延续，不是新的语义层。

| 模块 | 单一职责 |
|---|---|
| `session`（含 `checkpoint`） | 会话状态、静态配置、学习率、分布式编排、恢复与快照装配 |
| `learner`（含 `dataset`） | 借用批次、数据遍历、前向评估、指标看板与训练编排 |
| `loss` | 分类/回归、蒸馏、GRPO 目标及其共享几何与数值判据 |
| `optimizer` | 对象安全更新接口、SGD/Adam/Muon/Hybrid、矩状态与全局裁剪 |
| `backprop` | 按逆拓扑执行参考反向公式，把梯度送到参数端点 |
| `geometry`（含 `kernels` / `shapes`） | 运行时实例几何、滚动统计与前向/反向参考数值 |
| `kernel` | 存储 lane、内核适配与标量参考接口 |
| `pool` | `autograd::Store` 的真实缓冲实现与等长缓冲复用 |
| `rng` | 计数器型确定性随机及 Uniform/Normal/Bernoulli 分布 |

`lib.rs` 只做模块声明、公共导出和声明式宏，不新增另一套训练流程。

## 4. 配置与注解

### 4.1 配置对象

- `TrainConfig`：学习率峰值、warmup、总步数、裁剪范数、梯度累积、衰减曲线和谷底比例。
- `TrainOverrides`：可选覆盖；缺省字段保留原值，`clip_norm` 不能借覆盖静默清除已有阈值。
- `LoaderConfig`：行列几何与 `Sequential` / `Shuffle` / `Weighted` 采样，构造期完成校验。
- `ObjectiveConfig`：`Classify`、`Distill`、`Grpo` 三态，超参与运行期损失共用判据。
- `TrainSpec`：`session + loader + objective` 的完整静态组合，`validate` 统一重验，
  `config_fingerprint` 同时覆盖优化器类型与配置、批几何、采样和目标。

`train` 不读取 TOML，也不决定配置来源。命令行或文件解析属于 `cli`；公开字段被修改后，
仍须在装配边界重新调用 `validate`，不能把“构造时曾合法”当作永久保证。

### 4.2 并行注解

`annotation::parallel` 只生成 `ParallelConfig` 访问器，不读取环境、不发起通信：

```rust
use annotation::parallel;

#[parallel(tp = 2, pp = 1, dp = 2, reduce = "max", average = false, priority = "high")]
struct Sharded;

let policy = Sharded::__parallel_config();
assert_eq!(policy.topology().expect("合法拓扑").world(), 4);
```

完整策略可继续传给 `Distributed::from_config` 或 `shard::Assembler::from_config`，
同一份配置驱动拓扑、层归属和梯度同步，不复制键表。

## 5. 状态与恢复

`Checkpoint` 记录恢复所需的事实：步数、图规模、批形状、层数、权重条目数、世界大小和配置指纹。
`Session` 另持参数槽、累积窗口、优化器矩状态、梯度累积与权重绑定关系。

恢复必须遵守：

1. 世界大小、层结构和批几何先匹配；
2. 非零配置指纹不匹配时，在写入任何状态前拒绝；
3. 矩状态、累积梯度与 ties 的槽位数和槽内长度必须一致；
4. `storage::Snapshot` 负责字节格式与编码，train 只负责装配和校验；
5. `GradientSnapshot` 按 `Var` 自持，脱离累积窗口与缓冲生命周期。

这套状态不允许“恢复一半后继续训练”：失败必须发生在提交新状态之前。

## 6. 分布式语义

rank 的唯一布局由 `shard` 定义：`rank = dp_idx * (tp * pp) + pp_idx * tp + tp_idx`。
`Distributed` 接收本 rank、`Collective` 后端和 `ParallelConfig`；默认 `Remote` 只是未挂载占位，
不得被描述为生产级 NCCL/RCCL 实现。

- TP/PP：由 `shard::Assembler` / `Placement` 计算层段、投影切法和本 rank 权重几何。
- DP：由 `Distributed` 选择数据并行组并同步梯度，按配置决定归约种类、是否平均和调度优先级。
- `Local` 用于进程内数值验证；`FileRemote` 用于文件槽位协议验证，二者不冒充真实设备通信栈。

## 7. 性能与资源约束

- `Batch` / `Loader` 借用调用方条目缓冲，取批本身不分配、不复制。
- `Bufs` 只有 `f32` 与打包 `u8` 两条存储 lane；`F16` / `BF16` 在编译期拒绝，不做无安全保证的位重解释。
- 等长空闲表只复用容量匹配的块；稳定几何下避免重复分配，`live` / `peak` 与 autograd 计划对账。
- `Session` 复用图、缓冲、参数槽和优化器；指标评估与损失扫描尽量单趟完成。
- 随机流由计数器与种子确定；采样策略改变游标顺序，不改变数据所有权。
- crate 维持 `unsafe_code = deny`，不以性能为由新增第二座 unsafe 岛。

## 8. 验证层次

- 单元测试：配置、损失、优化器、调度、快照和边界拒绝。
- 性质测试：随机参数、随机图和优化器状态转换。
- 数值证据：前向/反向参考与中心差分对拍，区分共享定义与独立实现。
- 注解测试：策略从 `#[parallel]` 流向拓扑、层归属和本地权重切片。
- 分布式测试：组归属、归约/平均、非法 rank 和单进程恒等路径。
- 恢复测试：矩状态、累积梯度、ties、配置漂移与原子提交。

验收命令：

```bash
cargo test -p llm-train
cargo clippy -p llm-train --all-targets --no-deps -- -D warnings
RUSTDOCFLAGS="-D warnings" cargo doc -p llm-train --no-deps
cargo test -p llm-tools --test gate every_source_passes_the_static_gate -- --exact
```

## 9. 后续改造约束

1. 先消除同义配置、第二份公式表和重复校验，再合并文件；不以文件数代替内聚性。
2. 配置只有一份事实源，注解只生成既有契约；不新增平行的配置镜像或第二套类型体系。
3. 只合并共享不变量的实现，保持公开路径或提供明确的兼容导出。
4. 缓冲复用、零拷贝批次、确定性随机和失败原子性是行为合同，不能在重构中削弱。
5. 不降低测试数量与证据强度，不把自比对拍当独立证据，不把占位后端写成真实分布式交付。
6. 每轮同时报告能力、性能、文件与行数变化；净增能力行不得冒充去重减量。
