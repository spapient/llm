# einsum —— 语义层

一条求和表达式 + 轴长度 + 每个操作数的位宽，推出这件计算**至少**要算多少次浮点、**至少**要搬多少字节。
`cutile` 的 roofline 只消费这两个数，所以本 crate 不认识设备、tile、片段，也不认识任何一个算子名。

## 目录

```
crates/einsum/
├── Cargo.toml
└── src/
    ├── lib.rs       # crate 文档、位宽闭集 Bits、代价 Cost
    ├── expr.rs      # 表达式文法与解析（Expr），只做结构校验
    ├── extents.rs   # 轴长度表（Extents），含派生轴（Extents::group）
    └── formula.rs   # 操作数（Operand / Access）→ 轴分解（Axes）→ 代价（Cost）
```

## 文法

```text
expr   = term { "," term } "->" output
term   = { label }                # 空项表示标量缓冲
output = { label }
label  = "a".."z" | "A".."Z"
```

`ij,jk,ik->ik` 就是 `C = A·B`：**输出缓冲也是左侧的一项**，这样每个操作数都有自己的位宽与方向，
输出不必在表达式之外再声明一次形状。

三种写法是有意不支持的：省略号（广播，形状会开始靠猜）、同一项里的重复字母（对角提取，属于索引
变换而不是算子语义）、省略 `->` 的隐式模式（输出顺序变成一条只有作者记得的规则）。

## 两个容易写错的地方

**分组轴不是求和维。** 量化 GEMV 的 `scales[n, g]` 里 `g = m / 32`：

```rust
let extents = Extents::new([('n', 4096), ('m', 4096)])?
    .group('g', 'm', 32)?;        // 长度由 m 推出，m 不整除即报错
```

`g` 只影响流量（这块张量确实只有 `n · m/32` 个元素），不能参与运算次数——遍历 `g` 与遍历 `m` 是同一批
元素的两层看法，乘起来就是重复计数。`Axes` 因此把每条轴标成批 / 自由 / 规约 / 分组四类角色。

**位宽是闭集。** `Bits::{Four, Eight, Sixteen, ThirtyTwo, SixtyFour}` 决定流量怎么算：`i4` 元素按 nibble
计费并向上取整（3 个 int4 元素要 2 字节），不会被悄悄当成 `f32` 或 `u8`。

## 新增一个算子要写什么

在算子 crate 里写一条表达式与操作数表即可，不必再手写任何流量常数：

```rust
let formula = Formula::new(
    Expr::parse("nm,m,ng,n->n")?,
    vec![
        Operand::read("w", Bits::Four),
        Operand::read("x", Bits::ThirtyTwo),
        Operand::read("scales", Bits::ThirtyTwo),
        Operand::write("y", Bits::ThirtyTwo),
    ],
)?;
let cost = formula.cost(&extents)?;   // flops = 2·n·m，bytes = m·n/2 + 4m + 4ng + 4n
```

## 什么不放进来

- **设备峰值与 roofline 编排**：峰值在 `prof`，`max(flops/peak, bytes/bw) + 未掩盖项` 在 `cutile::plan`。
  本 crate 只给分子，不给时间。
- **算子声明、样本与标量参考**：在 `crates/op`。本 crate 不认识任何算子名字。
- **布局与片段**：在 `storage` / `cutile::layout` / `prim`。本 crate 只知道"每个操作数有几条轴"。

决策背景与未决项见 [`docs/adr/0007`](../../docs/adr/0007-semantic-cost-layer.md)。
