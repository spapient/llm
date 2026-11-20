# 0053：类型化算子入口——一张清单、一个原语、零手写方法

- 状态：已采纳（机制已落地：`op::for_each_operator!` + `train::OperatorSetOps` + 36 个在册算子）
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

### 1. 一张清单同时驱动注册表与 API 面

算子清单只声明一次（`op::for_each_operator!`），一行一个算子：

```rust
add_f32: $crate::add_f32::AddF32 => "binary",
gemm_i4: $crate::gemm_i4::GemmI4Model => "gemm_packed",
// …一行一个算子
```

两个消费者都从这张表展开：

1. `REGISTRY`：名字、模板、语义表、可微性（`op` 内展开）；清单的方法名与
   `Operator::NAME` 在编译期逐字节对拍（`same_str` 常量断言）。
2. `train::OperatorSetOps`：唯一原语
   `apply(name: &'static str, inputs: &[Var], output: Var) -> Result<()>`，加上每个算子一条**默认方法**
   `fn add_f32(&mut self, inputs: &[Var; <AddF32 as Operator>::ARITY], output: Var)`；方法名与算子名
   来自同一行，输入个数来自 `Operator::ARITY`。`ARITY` 由 `arity_matches_the_samples` 与算子样本
   （非 `y` 缓冲数）对拍。

后端（本仓是 `Buffers<K: Kernel>`）只实现一次 `apply`，就拿到全部 36 条类型化方法；未来 `serve`
或设备执行器实现同一个原语即可复用整张 API 面。方法调用是单态化的默认方法，无虚表、无运行期注册。
`train` 里没有第二份 36 行装配表——`cargo tree -p llm-train | grep annotation` 为空。

### 2. 不做的事

- **不手写每个算子的方法**：方法体对每个算子都同构，手写只会制造 36 份可漂移的重复。
- **不建运行期注册表**：清单在编译期展开；名字与输入个数都在编译期/单测里对拍。
- **不做第二个列表**：注册表与 API 面共享 `for_each_operator!`；`ARITY` 是 `Operator` 的常量，
  不复制形状/代价/样本。
- **不新增调度层**：`apply` 直接调 `autograd::Store::compute`，执行路径与手写调用完全一致。

## 与 Burn 的对比口径

只比较**同一件事的装配面**，不比较功能广度（Burn 覆盖 f16/bf16/int/bool/量化/autodiff 与多维
张量；本表只统计同一条 36 算子的前向入口）。Burn 数据取 `tracel-ai/burn` 提交
`bb4d9ddd`（2026-11-19）的源码行数统计：

| 指标 | Burn | 本仓 | 说明 |
|---|---|---|---|
| 后端必须实现的原语方法 | 130（`FloatTensorOps`）；全部 ops trait 424 | **1**（`apply`） | 每后端一次 |
| 手写 Tensor 方法 | 296 个 `pub fn` | **0**（36 条由装配表生成） | 生成方法约 6 行/条 |
| 新增一个算子的 **API 面接线** | 后端 trait + 各后端实现 + Tensor 方法 + 测试 | **0 行**（清单一行同时驱动注册表与 API；`Operator` 实现仍在语义层） | 编译期名字断言 + ARITY 测试 |
| 漂移防线 | 后端 trait 缺方法=编译错误；Tensor 面靠人写 | 生成测试逐条对拍名字/顺序/输入个数 | 已实测：删一行、改错个数都会失败 |

按 36 算子切片折算：后端原语面 **130 → 1**（约 130×），手写 API 面 **296 → 0**，单算子装配
**3.6 条 trait 方法 + 8.2 条 Tensor 方法 → 1 条装配行 + 1 条生成方法**。

**不宣称的部分**：没有做 Burn 与本仓的端到端运行时对比，因此不写"速度快几倍"。两者都是单态化静态
派发；本仓的 `apply` 在 `Kernel::compute` 里按名字做一次线性查找（36 条），Burn 是直接 trait 调用，
这点差异在真实算子粒度上未测，不能当结论。

## 验证

- `cargo test -p llm-op --lib arity_matches`：36 个算子的 `ARITY` 与样本输入个数逐条相等。
- `cargo test -p llm-train --lib operators`：类型化调用与字符串派发逐元素一致。
- 漂移演练（E1）：把清单里的方法名改成与 `Operator::NAME` 不同的拼写 → 编译期 `const` 断言失败；
  把某个算子的 `ARITY` 改错 → `arity_matches_the_samples` 失败。
- `cargo test -p llm-tool --test gate`：静态门禁通过。

## 后果

- 加算子的流程固定为：`Operator` 实现 → 清单一行。注册表、API 面与 ARITY 检查同时到位。
- API 面与注册表同源；不存在"注册表有、API 没有"的静默缺口。
- 若将来接入设备执行器，只需为它实现 `apply`；不需要再写 36 条转发方法。
