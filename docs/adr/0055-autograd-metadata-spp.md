# 0055：autograd 反向图层 S++ 收口——概念归并、闭包足迹泛型、契约注解与证据对齐

- 状态：已采纳（2026-11-24 落地；验收以代码、测试与九项门禁为准）
- 日期：2026-11-24
- 相关：[`docs/ARCHITECTURE.md`](../ARCHITECTURE.md) L5 训练侧与所有权表、[0005](0005-annotations-as-contracts.md)（注解即契约）、[`docs/PROMPT-S++-AUDIT.md`](../PROMPT-S++-AUDIT.md) N3 文件内聚 / N7 配置泛型注解优先 / N13 测试分层

## 背景

对比基线（`docs/COMPARISON-2026-11-24-reeval.md` §17.2 第 6 行）：自动微分维度 llm 5 / llm.u 2 / Burn 10。
取证回链当前源码：

- **边界是架构定死的**：`autograd` 只产出反向图的**元数据**——谁在图上、按什么顺序执行、梯度何时齐、
  激活留哪些何时还。梯度缓冲、前向激活值、算子反向公式归 `train`；前向计算归 `nn` / `cutile`。
  本层不认识张量数值，也不注册算子（`crates/autograd/src/lib.rs` 头部四问）。
- **现状基线（2026-11-24 实测）**：src 8 文件 2,045 行（最大 `schedule.rs` 341 行），tests 8 文件 1,909 行、
  68 个 `#[test]`；jscpd 0 克隆；lib 65 例 + `gradcheck` 3 例全绿；`llm-train` / `testkit` / `benchmarks`
  对本层的公开面编译通过。
- **可枚举缺口（在本层边界内）**：
  1. 文件按实现切面拆，而非按概念归并——`account`（就绪账本）与 `schedule`（执行序）同属"反向执行状态"，
     `segment`（分段求解）与 `facts`（计划事实）同属"显存求解"，却各占一文件。
  2. `Footprint` 只收切片 / `Vec` / 引用 / `Sizes` 四型，调用方给不出闭包足迹，只能先分配一张 `Vec<u64>`。
  3. 契约注解不齐：部分只读访问器缺 `#[must_use]`，少数派生 trait 不完整；错误站点与不变量散落在散文里。
  4. 证据错位：`tests/gradcheck.rs` 已对 200 张随机图跑中心差分对拍（含菱形扇出、重复输入、detach 分支、
     图外参数梯度恰为 0、分层与顺序逐位一致、重复反向逐位一致），对比文档却记"梯度数值未对拍"。

Burn 的"后端装饰器任意可微 + 梯度快照 + 分布式梯度"落在后端装饰器与训练执行面，是 `train` / `device`
的职责；本层不复制那条架构税。对齐口径：**在元数据层内把图记录、调度、账本、显存计划、重放序列五件事
做到配置化、泛型化、可审计、可复用**，数值正确性由 `gradcheck` 对拍作证。

## 决策

### 1. 边界与依赖不动

依赖边保持 `autograd → diagnostic, tensor`（dev `testkit`）；不引入张量 IR、梯度缓冲、算子公式表、
proc-macro 或分布式梯度。公开类型名保持不变（所有权表 18 个 `pub` 类型逐条保留），`train` / `testkit` /
`benchmarks` 的调用点不改。

### 2. 模块按概念归并：src 8 → 7，测试 8 → 7

| 目标文件 | 概念 | 吸收自 |
|---|---|---|
| `graph.rs` | 图的静态值：`Role` / `Requirement` / `Var` / `Node` | `graph.rs` |
| `tape.rs` | 前向记录：`Tape` | `tape.rs` |
| `schedule.rs` | 反向执行序与计数快照：`Ordering` / `Schedule` / `Summary` | `schedule.rs` |
| `account.rs` | 就绪账本：`Account`（独立生命周期，按调度重建；不并入调度文件） | `account.rs` |
| `facts.rs` | 显存事实与分段求解：`Facts` / `Ledger` / `Scratch` 与段数试算、段界划分 | `facts.rs` + `segment.rs` |
| `plan.rs` | 显存决策与配置：`Plan` / `Budget` / `Choice` / `MemoryPolicy` / `Footprint` / `Sizes` / `Config` | `plan.rs` |
| `replay.rs` | 重放指令序列：`Store` / `Replay` | `replay.rs` |

归并只收"同一职责、同一不变量"的兄弟文件（N3 / 反模式 #2：不为减文件数把独立语义塞进同一文件）。
唯一吸收的是 `segment.rs`——它整段就是 `impl Facts` 的求解续体，与 `facts.rs` 同一个类型、同一份
字节/候选表；`Account` 虽由 `Schedule::account()` 构造，但账本有独立的扇出计数与结算进度，生命周期
按一次调度重建，留在独立文件。生产文件 ≤400 行优先、>600 一票否决；测试文件 ≤800 行。

测试归并：`account_tests.rs`（7 例）并入 `schedule_tests.rs`；其余按源文件一一对应；`gradcheck.rs`
保留为独立 `[[test]]` target，`support.rs` 夹具不动。

### 3. 泛型：守住已有的泛型边界，不做推测性抽象

- `Footprint` 闭集保留 `[u64]` / `Vec<u64>` / `&F` / `Sizes` 四型。曾尝试补 blanket
  `impl<F: Fn(Var) -> u64> Footprint for F`，与既有 `impl<F: Footprint> Footprint for &F`
  在 `&G: Fn`（std 对引用到闭包的 `Fn` 实现）上触发 E0119 重叠；调用方当前没有闭包足迹需求，
  按一致性回退，不为假想消费者引新表面。
- `Var` 编号保持 `u32`（编号即稠密下标，泛型到宽度是推测性抽象，按 N7 不做）。
- 已有泛型面保持：`Facts::new` 吃 `&impl Footprint`；`Replay::run` 泛型 `S: Store` + 回调
  `F: FnMut(&mut S, &Node) -> Result<()>`；`Tape::record` 吃 `impl Into<String>`。

### 4. 配置化：策略轴收成一份 `Config`，求解启发式留在内部

`Config { ordering, policy }` 是唯一配置入口：排序（深度 / 插入序）× 显存策略（自动 / 全留 / 分段）。
段数中心（√n 与预算一半配额 ±2 档）是求解器内部启发式，不上升为公开旋钮——当前没有第二个消费者，
按"删无消费者公开面"原则不提前暴露。`Config` 补构造期 fluent setter，保持 `new` / `default` 与既有
测试用例不变。

### 5. 注解：契约落到访问器与错误站点

- 只读查询访问器补 `#[must_use]`；值类型补齐 `Debug / Clone / Copy / PartialEq / Eq / Hash` 派生。
- 每个 `pub` 项的文档写清不变量、失败类别（`# Errors`）与前置条件；内部错误统一走 `invalid` /
  `internal` 两个诊断工厂，不在生产面写字符串兜底或 `unwrap`。
- 闭集枚举（`Role` / `Requirement` / `Ordering` / `MemoryPolicy`）的短名方法保持唯一分派点。

### 6. 性能：SoA 与缓冲复用守住，保留的循环写明串行理由

- `Tape` 五平行数组、`Schedule` / `Account` / `Scratch` 的缓冲复用（`clear` / `rebuild` / `plan_into`
  稳态零分配）保持不变。
- 无串行依赖的批量统计走整段 slice / 迭代器收集；保留的逐元素循环只出现在有串行依赖处（按编号递推的
  宽度 / 重算步数、按编号降序传播的释放点），并在注释写明保留理由。
- `Replay` 执行期两张扁平区间表，零分配形态不变。

### 7. 证据：冻结数值与对拍口径保持，对比文档按实测回填

- `plan_tests.rs` 的 `FROZEN` 摘要（chain16 三档预算的 saved / release / replayed / peak）逐字不变。
- `gradcheck.rs` 的容差（绝对 `2e-3` + 相对 `0.05`，步长 `1e-6`）、200 轮随机图与三条固定用例不变。
- 门禁全绿后回填 `COMPARISON` 第 6 行的取证依据：元数据层能力项逐条回链源码，数值正确性以
  `gradcheck` 中心差分对拍为证；任意可微 / 分布式梯度仍记在 `train` / `device` 侧，不在本层计分。

## 不做的事

- 不在 `autograd` 内建梯度张量、前向激活缓冲或算子反向公式（归 `train`）。
- 不引入后端装饰器、分布式梯度聚合、多 loss 根集、检查点段数公开旋钮（无消费者，属推测性抽象）。
- 不把 `annotation` 的 proc-macro 引入本层（注解层服务 `nn` / `cutile` 契约 DSL）。
- 不改 `train` / `testkit` / `benchmarks` 对本层的调用代码；若归并导致可见性降级，逐条恢复到原公开面。

## 验收标尺

- `cargo test -p llm-autograd`：用例数不低于基线 68，`gradcheck` 全绿；冻结摘要逐字一致。
- `cargo check -p llm-train -p llm-testkit -p llm-benchmarks` 编译通过，公开调用点零改动。
- 静态门：ccn ≤5、nloc ≤60、生产文件 ≤600（优先 ≤400）、测试 ≤800、jscpd 0 克隆、命名零命中。
- `cargo fmt --check`、`clippy -D warnings`、doc-test、`cargo deny`、`machete` 全绿。
- 回滚方案：归并为纯文件内联，公开类型名与路径不变；回滚即恢复 8 个源文件与 8 个测试文件，无跨层改动。

## 减码账本（2026-11-24 实测）

| 轮次 | src 行数 | Δ | src 文件 | Δ | tests 行数 | Δ | tests 文件 | Δ | jscpd | 主要动作 | 门禁 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| 基线 | 2,045 | — | 9 | — | 1,909 | — | 8 | — | 0 | 取证：65 lib + 3 gradcheck + 2 doc 全绿 | 待收尾 |
| 本轮 | 2,030 | −15 | 8 | −1 | 1,900 | −9 | 7 | −1 | 0 | `segment` 续体并入 `facts`；`account` 用例并入 `schedule`；剪除 `Schedule` 只写不读的 `vars` 字段；归并后 rustfmt/clippy 0 告警 | 收尾一次跑 |

> 本 crate 归并前已 jscpd 0 克隆、静态门全绿，压缩空间在文件归并与死字段，不在逐行删码；强制砍行
> 会触碰 N3 内聚与反模式 #2，故本轮以"同一概念一处存放"为准，不做行数换结构。

## 风险

- 并发工作树存在 `device` 金属后端、`shard` 测试与对比文档的未提交改动（N20）：本轮只认领
  `crates/autograd/**` 与本 ADR、对比文档第 6 行证据段，不回滚、不并入外部产物。
- 归并后单文件可能逼近 400 行线：超 400 但 ≤600 须在文件头给出内聚证明；无解则拆回独立文件，
  不以放宽门禁容纳胖文件（反模式 #1）。
