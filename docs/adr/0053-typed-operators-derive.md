# 0053：类型化算子入口——一个 derive、一个原语、一张装配表

- 状态：已采纳（机制已落地：`annotation::Operators` + `train::OperatorSetOps` + 36 个在册算子）
- 日期：2026-11-20
- 相关：ADR 0002（单一主人）、ADR 0005（注解即契约）、ADR 0013（加算子只加数据）、ADR 0050（算子基座）

## 背景

Burn 的 Tensor API 是"每个算子手写一条方法"：`crates/burn-tensor/src/tensor/api/*.rs` 共 296 个
`pub fn`，后端侧 `FloatTensorOps` 一个 trait 就有 130 条方法（`burn-backend/src/backend/ops/tensor.rs`，
2000 行；全部 ops trait 合计 424 条）。新增一个算子要同时改：后端 trait、每个后端实现、Tensor 方法、
文档与测试。这是它的架构税，不是它的优势。

本仓的差异点在于：算子语义已经收口在 `op::operator::Operator` 与 `REGISTRY`（ADR 0050），执行入口
已经收口在 `autograd::Store::compute` 一处。缺的只是"类型化 API 面"——一个能从同一张表长出来、
且不会与注册表漂移的门面。

## 决策

### 1. 一个 derive 生成整个 API 面

`#[derive(annotation::Operators)]` 标在一个只承载装配表的结构上：

```rust
#[derive(annotation::Operators)]
#[operators(
    var = "::autograd::Var",
    result = "::diagnostic::Result<()>",
    registry = "::op::REGISTRY",
    add_f32 = 2,
    matmul_f32 = 2,
    // …一行一个算子
)]
pub struct OperatorSet;
```

生成两件东西：

1. `trait OperatorSetOps`：唯一原语
   `apply(name: &'static str, inputs: &[Var], output: Var) -> Result<()>`，加上每个算子一条**默认方法**
   `fn add_f32(&mut self, inputs: &[Var; 2], output: Var)`，方法体只有一句转发。
2. 一条校验测试：装配表与 `REGISTRY` 的**名字、顺序、输入个数**逐条对拍（输入个数取自算子样本里
   非 `y` 的缓冲数）。注册表加算子而装配表没加、或个数写错，测试当场失败。

后端（本仓是 `Buffers<K: Kernel>`）只实现一次 `apply`，就拿到全部 36 条类型化方法；未来 `serve`
或设备执行器实现同一个原语即可复用整张 API 面。方法调用是单态化的默认方法，无虚表、无运行期注册。

### 2. 不做的事

- **不手写每个算子的方法**：方法体对每个算子都同构，手写只会制造 36 份可漂移的重复。
- **不建运行期注册表**：装配表在编译期展开；校验测试对拍的是编译期常量 `REGISTRY`。
- **不把语义塞进 derive**：derive 只认"名字 + 输入个数"，不复制形状/代价/样本；语义仍在 `Operator`。
- **不新增调度层**：`apply` 直接调 `autograd::Store::compute`，执行路径与手写调用完全一致。

## 与 Burn 的对比口径

只比较**同一件事的装配面**，不比较功能广度（Burn 覆盖 f16/bf16/int/bool/量化/autodiff 与多维
张量；本表只统计同一条 36 算子的前向入口）。Burn 数据取 `tracel-ai/burn` 提交
`bb4d9ddd`（2026-11-19）的源码行数统计：

| 指标 | Burn | 本仓 | 说明 |
|---|---|---|---|
| 后端必须实现的原语方法 | 130（`FloatTensorOps`）；全部 ops trait 424 | **1**（`apply`） | 每后端一次 |
| 手写 Tensor 方法 | 296 个 `pub fn` | **0**（36 条由装配表生成） | 生成方法约 6 行/条 |
| 新增一个算子的 **API 面接线** | 后端 trait + 各后端实现 + Tensor 方法 + 测试 | **1 行**（装配表一行；`REGISTRY` 与 `Operator` 是语义层既有改动） | 生成测试兜底 |
| 漂移防线 | 后端 trait 缺方法=编译错误；Tensor 面靠人写 | 生成测试逐条对拍名字/顺序/输入个数 | 已实测：删一行、改错个数都会失败 |

按 36 算子切片折算：后端原语面 **130 → 1**（约 130×），手写 API 面 **296 → 0**，单算子装配
**3.6 条 trait 方法 + 8.2 条 Tensor 方法 → 1 条装配行 + 1 条生成方法**。

**不宣称的部分**：没有做 Burn 与本仓的端到端运行时对比，因此不写"速度快几倍"。两者都是单态化静态
派发；本仓的 `apply` 在 `Kernel::compute` 里按名字做一次线性查找（36 条），Burn 是直接 trait 调用，
这点差异在真实算子粒度上未测，不能当结论。

## 验证

- `cargo test -p llm-train --lib operators`：类型化调用与字符串派发逐元素一致；生成的
  `typed_surface_matches_the_registry` 通过。
- 漂移演练（E1）：删掉 `log_f32 = 1` 一行 → `类型化入口与注册表算子数不同`；把它写成 `= 2` →
  `算子 log_f32 的输入个数与注册表样本不同`。两次都失败在 `cargo test`，不是运行期。
- `cargo test -p llm-annotation --lib`：文法错误（缺表、重复、零输入）与渲染文本的单元测试通过。
- `cargo test -p llm-tool --test gate`：静态门禁通过。

## 后果

- 加算子的流程固定为：`REGISTRY` 一行（含 `Operator` 实现）→ 装配表一行 → 生成测试自动对拍。
- API 面与注册表同源；不存在"注册表有、API 没有"的静默缺口。
- 若将来接入设备执行器，只需为它实现 `apply`；不需要再写 36 条转发方法。
