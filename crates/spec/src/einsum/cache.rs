//! 表达式缓存：一条表达式只解析一次，四条骨架共用同一份缓存逻辑。
use crate::einsum::{Expr, Formula, Operand, OperandSpec};
use diagnostic::Error;
use std::sync::OnceLock;

/// 一份表达式缓存：构造一次，之后复用到任意形状。
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
