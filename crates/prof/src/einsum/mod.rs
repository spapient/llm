//! 语义层：一条爱因斯坦求和表达式 + 轴长度 + 操作数位宽 → [`Cost`]（flops 与最少字节）。
//!
//! 这一层回答的是**与设备、与 tile、与片段都无关**的问题：这个算子至少要算多少次浮点、至少
//! 要搬多少字节。它不认识 `cutile`、设备档案与片段，因此可以在任何机器上离线复现——roofline
//! 的分子从此只有一个来源，而不是每个算子各写一遍公式。
//!
//! **拥有**：[`Expr`]、[`Extents`]、[`Axis`] / [`Axes`]、[`Operand`] / [`OperandSpec`]、
//! [`Bits`]、[`Cost`]，以及四条骨架共用的表达式缓存 [`FormulaCache`]。
//! **不拥有**：设备峰值（[`crate::prof`]）、roofline 的编排（`cutile::plan`）、算子声明与样本（[`crate`]）。
//! **依赖**：`diagnostic`（全仓唯一的错误类型）。**被依赖**：本 crate 的算子语义模型（用本模块推导代价），并经 `nn` 注册表对齐。
//!
//! 职责收口在四个文件：[`axis`] 轴分解与轴长度表、[`expr`] 表达式文法、[`formula`] 配对/流量/缓存、
//! [`operand`] 操作数、位宽与代价分量。对外路径仍为 `prof::einsum::*`。
//!
//! # 表达式
//!
//! 文法与 `numpy.einsum` 的**显式**模式一致，只保留能穷举的那部分：
//!
//! ```text
//! expr   = term { "," term } "->" output
//! term   = { label }                # 空项表示标量缓冲
//! output = { label }
//! label  = "a".."z" | "A".."Z"
//! ```
//!
//! 不支持的三种写法都是刻意的：
//!
//! - **省略号（广播）**：广播让同一个字母在不同操作数里有不同长度，代价模型就必须开始猜形状；
//! - **同一项里重复字母（对角提取）**：那是索引变换而不是算子语义，应当由调用方先 reshape；
//! - **省略 `->`（隐式模式）**：输出轴顺序变成"输入出现顺序的排序规则"，读者与机器都会各猜一套。
//!
//! 三件都能用两条显式式子改写出来，而支持它们会让这份文档变成唯一事实源之外的第二个解释器。
//!
//! **输出缓冲同样是左侧的一项**：`ij,jk,ik->ik` 表示 `C = A·B`，第三项就是 `C` 的轴。这样每个
//! 操作数的位宽与读写方向都有出处，输出不必在表达式之外再声明一次形状。
//!
//! # 代价口径
//!
//! | 量 | 口径 |
//! |---|---|
//! | `flops` | 有规约轴时 `2 × Π(参与运算的轴)`（每个规约项一次乘加）；没有规约轴时 `Π(输出轴)`（每个输出元素一次运算） |
//! | `read` | 只读操作数各自的 `ceil(Π(自己的轴) × 位宽 / 8)` 之和 |
//! | `write` | 只写操作数各自的 `ceil(Π(自己的轴) × 位宽 / 8)` 之和（至少写一遍输出） |
//!
//! 读与写**分开给**而不是合成一个 `bytes`：它们的上限不是同一个（只流式读权重的 GEMV 可以
//! 跑过设备的拷贝上限），合成之后"这层是读受限还是写受限"就再也判断不了。`Cost::bytes()` 仍然
//! 提供合计值，但它是**派生**量，不是口径本身。
//!
//! 位宽用 [`Bits`] 闭集表达，因此 `i4` 权重按 nibble 计费，而不是被当成一个 `f32` 或一个 `u8`。
//!
//! 有一类轴必须被排除在运算次数之外：**分组轴**。量化 GEMV 的 `scales[n, g]` 里，`g` 是 `m`
//! 每 `GROUP` 个元素切出来的一面，遍历 `g` 与遍历 `m` 是同一批元素的两层看法；把它乘进
//! `Π(轴)` 会让 `flops` 多出一个因子。这类轴用 [`Extents::group`] 声明为派生轴：长度由基准轴
//! 推出，只计流量、不计运算次数。
//!
//! # 例子
//!
//! ```
//! use llm_prof::einsum::{Bits, Expr, Extents, Formula, Operand};
//!
//! // int4 权重、按 32 个元素一组缩放的 GEMV：
//! let expr = Expr::parse("nm,m,ng,n->n").expect("表达式合法");
//! let formula = Formula::new(
//!     expr,
//!     vec![
//!         Operand::read("w", Bits::Four),
//!         Operand::read("x", Bits::ThirtyTwo),
//!         Operand::read("scales", Bits::ThirtyTwo),
//!         Operand::write("y", Bits::ThirtyTwo),
//!     ],
//! )
//! .expect("操作数与表达式一一对应");
//!
//! // `g` 是 `m` 每 32 个元素一组切出来的派生轴：长度由 `m` 推出，不参与运算次数。
//! let extents = Extents::new([('n', 8), ('m', 64)])
//!     .expect("轴长度")
//!     .group('g', 'm', 32)
//!     .expect("64 能被 32 整除");
//! let cost = formula.cost(&extents).expect("可估算");
//!
//! assert_eq!(cost.flops, 2.0 * 8.0 * 64.0);
//! // w: 8·64 元素 × 4 bit/8；x: 64 × 4；scales: 8·2 × 4。
//! assert_eq!(cost.read, 8 * 64 / 2 + 64 * 4 + 8 * 2 * 4);
//! // 只有一个输出缓冲，所以写方向只有它。
//! assert_eq!(cost.write, 8 * 4);
//! ```

pub mod axis;
pub mod expr;
pub mod formula;
pub mod operand;

pub use axis::{Axes, Axis, AxisRole};
pub use operand::Bits;
pub use formula::FormulaCache;
pub use operand::Cost;
pub use expr::Expr;
pub use axis::Extents;
pub use formula::Formula;
pub use operand::{
    Access, Operand, OperandSpec, bf16_io_operands, f16_io_operands, f32_binary_operands,
    f32_io_operands,
};

#[cfg(test)]
mod tests {
    include!("../../tests/spec/einsum_tests.rs");
    include!("../../tests/spec/cost_tests.rs");
}
