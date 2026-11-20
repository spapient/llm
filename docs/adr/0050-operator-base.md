# 0050：算子基座——`Operator` 契约收口六族语义,装配宏退化为投影

- 状态：已采纳（本轮落地 `op::operator` 契约与 reduce 族示范；其余族按 §5 顺序迁移）
- 日期：2026-11-20
- 相关：[0002](0002-single-owner-per-type.md)（单一主人）、[0005](0005-annotations-as-contracts.md)（注解即契约）、[0013](0013-dense-projection-operator.md)（加算子只加数据）、[0026](0026-gemm-generic-skeleton.md)（GEMM 泛型骨架）、[0031](0031-kernel-macro-derived-table.md)（宏派生表）、[0047](0047-2bit-quantized-lane.md)（量化 lane 的配置表）、[0049](0049-refactor-summary-2026-11.md)（上一轮重构结论）

## 背景

全仓的**变化点**只有三类:算子语义、精度格式、执行后端。但收口它们的抽象只覆盖了一半:

| 变化点 | 已有收口 | 缺口 |
|---|---|---|
| 精度格式 | `dtype::Format`（L1，单一闭集） | — |
| 执行后端 | `cutile::backend::Backend = Codegen + Driver`（L3，`engine` 实现、`cli` 注入） | 上层装配不使用它 |
| **算子语义** | 六族 trait（`Unary` / `Binary` / `Reduce` / `Format` / `Gemm` / `Plain`）**形状同构却各自为政**；`Model`（`op::model`）是统一的数据投影，但没有统一的行为契约 | **没有 `Operator`**：同一个概念（一个算子的名字/形状/代价/样本/参考/可微）在不同族用不同范式表达 |

后果与用户诊断一致:装配宏 53 处散在 26 个文件（每族自造一套）、`model/decl/sig/tpl` 手写、`train/kernel` 与 `serve/kernels` 各写适配表；文件只能按族堆大。**这不是缺设计模式，是缺"一个"模式。**

## 决策

### 1. 基座不是单个 crate，而是一条被同一契约贯穿的链

Burn 的单 crate 泛型塔与本仓的分层契约（依赖只能向下，ADR 0001）不可兼得。本仓的等价形态是:

```text
L1 dtype::Format        精度变化点（闭集 + 布局几何）
L2 op::Operator         语义变化点（名字 / 形状 / 代价 / 样本 / 参考 / 可微）
L3 cutile::backend      执行变化点（Codegen + Driver，engine 实现）
L5/L6 train/serve/cli   装配层：同一张 operator 表投影成各自的内核表
```

`Operator` 住在 `op`（L2）：`cutile`（L3）、`nn`/`train`（L5）、`serve`（L6）都已依赖 `op`，**不新增任何依赖边**，也不出现环。后端不进入 `Operator`——那是 L3 的契约，跨越即破坏分层。

### 2. `Operator` 契约（唯一语义收口）

```rust
pub trait Operator {
    const NAME: &'static str;
    const SHAPE: (usize, usize);
    const DTYPE: DType;
    const AXES: &'static [AxisSpec];
    const DIFFERENTIABLE: bool;
    const GROUP: u32 = 1;
    fn shape_ctx(m: usize, n: usize) -> Result<ShapeContext, Error>;
    fn cost(shape: &ShapeContext) -> Result<Cost, Error>;
    fn sample(m: usize, n: usize) -> Sample;
    fn reference(sample: &Sample) -> Result<Vec<f32>>;
}
```

- 六族 trait **保留**为族内骨架配置（`EXPR` / `OPERANDS` / `cache()` / `forward` 等），每个具体算子同时 `impl` 族 trait 与 `Operator`；`Operator` 的方法体**转发**到族 trait，不写第二份算术。
- `Model` 由 `Operator` 投影（`operator::model_of::<T>()`）；`#[model]` 注解仍是语法糖，生成物改为对 `model_of` 的调用（下一轮，见 §5）。
- **单一主人不变**：`Model` 的字段仍只有一个主人（`op::model`），`Operator` 是它的行为侧投影；族 trait 的算术不变、`reference` 仍是唯一标量参考。
- 由 `testkit` 的一条契约测试锁住:每个在册算子的 `Operator` 投影与 `#[model]` 生成常量在 `cost/sample/reference` 上逐位一致（过度期双写不许漂移）。

### 3. 装配宏退化为投影

`train::kernel_table!` / `serve::kernels` 的适配行宏只保留"函数签名适配"（`&[View] → ViewMut`），
**名字与顺序不再手写**：由 `Operator::NAME` 的表驱动生成（`operator_table!`），每一行只剩适配函数名。
`op::kernel_table!`（本轮已收口）继续作为行展开。

### 4. 不做的事

- 不引入运行期注册表（ADR 0005）：`Operator` 是 trait，表仍是编译期常量。
- 不把后端塞进 `op`：`Operator` 不认识 `Emit` / `Backend` / 设备。
- 不删族 trait：它们是骨架配置的唯一声明处；`Operator` 是它们之上的统一契约，不是替代。

## 迁移顺序（每族独立可验证，测试数不跌）

1. **reduce 族（本轮示范）**：`Operator` 契约 + `reduce_sum/max` 两个 impl + 投影一致性测试。
2. unary / binary / dequant：三族形状同构，照 reduce 范式。
3. gemv / gemm：族 trait 已最完整（`Format` / `Gemm`），迁移中把 `PackedRow` / `config()` 收进 `Operator` 的精度关联项。
4. attention / softmax / rmsnorm / rope / embed / scan：standalone 家族，迁移时顺带评估骨架化（此前研究过，需实证收益）。
5. `annotation` 的 `#[model]` 改生成 `model_of::<T>()`；镜像门禁改为"注解 ↔ Operator 表"双向。

## 验收门（每步）

- `cargo nextest run --workspace` 测试数不跌；`clippy -D warnings`、`fmt`、静态门 2/2、架构门（新增 `Operator` 相关 pub 类型进所有权表）全绿。
- 投影一致性测试对在册全部已迁移算子逐位通过。
- 不新增依赖边、不新增 unsafe、不在 `op` 出现后端字面量。

## 影响与回滚

- 回滚单位是族：一族迁移是一个 commit，`git revert` 即回到族 trait 直连的状态。
- 过渡期允许"族 trait + Operator 双实现"，由投影一致性测试守住漂移；每族迁移完成后不留双写。
