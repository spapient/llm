//! 骨架的表达式协议：一条 einsum 表达式、一张操作数表、一个按实现分开的缓存单元格。
//!
//! 六族骨架（[`crate::unary::Unary`] / [`crate::elementwise::Binary`] / [`crate::reduce::Reduce`] /
//! [`crate::gemv::Format`] / [`crate::gemv::Plain`]）在「表达式从哪来」上是同一件事：族给出表达式
//! 与操作数表，实现只决定算术。协议收在这里，族 trait 只声明随语义变化的那部分（ADR 0050 §2）。
//!
//! [`crate::gemm`] 不在其列：它的代价由分块几何现算，没有表达式可缓存。

use crate::einsum::{Formula, FormulaCache, OperandSpec};
use diagnostic::Error;

/// 表达式的主人：一副骨架怎么声明表达式、缓存在哪、解析失败报什么帧。
///
/// 实现者是族 trait 的实现（标记类型，以及 [`crate::gemv::Packed`] 这类配置驱动类型）：它们只给
/// 三个常量与一个单元格，[`FormulaOwner::formula`] 是全仓唯一一份解析与缓存逻辑。
pub trait FormulaOwner {
    /// 一条 einsum 表达式。
    const EXPR: &'static str;

    /// 表达式各项的位宽与方向，顺序与 [`FormulaOwner::EXPR`] 的项一一对应。
    const OPERANDS: &'static [OperandSpec];

    /// 解析失败时的诊断帧：点名是哪一族在读表达式（如 `op::unary::formula`）。
    const FRAME: &'static str;

    /// 本实现的缓存单元格。
    ///
    /// 必须**每个实现一份**：写在泛型骨架里的 `static` 在 Rust 里只有一份，先解析的表达式与操作数
    /// 表会被后跑的格式复用（位宽串档）。函数体写成 [`formula_cell!`](crate::formula_cell)，全仓一个写法。
    fn cache() -> &'static FormulaCache;

    /// 表达式：第一次调用时解析并校验，之后复用。
    ///
    /// # Errors
    ///
    /// 表达式与操作数表不自洽时返回说明（帧即 [`FormulaOwner::FRAME`]）。
    fn formula() -> Result<&'static Formula, Error> {
        Self::cache().get(Self::EXPR, Self::OPERANDS, Self::FRAME)
    }
}

/// 一个实现自己的表达式缓存单元格。
///
/// 展开成一段块表达式：`static` 落在展开处，于是每个 `impl` 拿到自己的单元格，而不是全仓共用
/// 一个——见 [`FormulaOwner::cache`] 的说明。
#[macro_export]
macro_rules! formula_cell {
    () => {{
        static CACHE: $crate::einsum::FormulaCache = $crate::einsum::FormulaCache::new();
        &CACHE
    }};
}
