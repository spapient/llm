//! 表达式缓存：一条表达式只解析一次，四条骨架共用同一份缓存逻辑。
//!
//! GEMV / 逐元素 / 一元 / 归约原先各写一份 `OnceLock<Result<Formula, Error>>`，代码逐字相同，
//! 只有表达式文本、操作数表与错误帧不同。这里把**逻辑**收成一处：`get` 收表达式与操作数表，
//! 单元格仍由调用方给出。
//!
//! 缓存**必须由每个具体类型自己持有**：写在泛型函数里的 `static` 在 Rust 里只有一份，两种精度
//! 会共用同一个单元格，先跑的那一份会让另一份拿到错误的轴表——这是一次实测里"`g` 不在
//! `nm,m,n->n` 里"的来源。所以本模块提供的是类型，`static` 留在各骨架 trait 的 `cache()` 里。

use crate::einsum::{Expr, Formula, Operand, OperandSpec};
use diagnostic::Error;
use std::sync::OnceLock;

/// 一份表达式缓存：构造一次，之后复用到任意形状。
///
/// 初始化失败也一起缓存——同一条常量表达式不会第二次成功。
pub struct FormulaCache(OnceLock<Result<Formula, Error>>);

impl FormulaCache {
    /// 空缓存（`static` 初始化用，必须是 `const fn`）。
    #[must_use]
    pub const fn new() -> Self {
        Self(OnceLock::new())
    }

    /// 取这条表达式：第一次调用时解析并校验，之后直接复用。
    ///
    /// `site` 是错误帧（如 `op::gemv::formula`），由**骨架**给出而不是由格式给出：同一条骨架的
    /// 所有精度共享同一个位置，重复写在每个实现里只会多出几份可以写歪的事实。
    ///
    /// # Errors
    ///
    /// 表达式或操作数表不自洽时返回第一次得到的说明（带 `site` 帧）。
    pub fn get(
        &'static self,
        expr: &'static str,
        operands: &'static [OperandSpec],
        site: &'static str,
    ) -> Result<&'static Formula, Error> {
        self.0
            .get_or_init(|| {
                let operands = operands
                    .iter()
                    .map(|spec| Operand::new(spec.name, spec.bits, spec.access))
                    .collect();
                Expr::parse(expr)
                    .and_then(|expr| Formula::new(expr, operands))
                    .map_err(|error| error.with(site))
            })
            .as_ref()
            .map_err(Clone::clone)
    }
}

impl Default for FormulaCache {
    fn default() -> Self {
        Self::new()
    }
}
