//! 矩阵乘层：`Y[m,n] = Σ_k A[m,k] · B[k,n]`。

use diagnostic::{Code, Error, Result};

use crate::{Bounds, BoundsResult, Contract, spec_accessors};

/// 矩阵乘层：结果形状 `[rows, cols]`，收缩轴由算子固定。
#[crate::spec(name = $name, out = [$rows, $cols], ops = [$op])]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MatMul {
    rows: usize,
    cols: usize,
}

impl MatMul {
    /// 绑定的稠密矩阵乘算子名。
    pub const OP: &str = "gemm_f32";

    /// 由输出行数与列数构造；两者都必须非零。
    ///
    /// # Errors
    ///
    /// `rows` / `cols` 任一为零（[`Code::Invalid`]）；算子未在 `op` 注册或没有语义表
    /// （[`crate::ContractSpec::new`] 上报）。
    pub fn new(name: impl Into<String>, rows: usize, cols: usize) -> Result<Self> {
        if rows == 0 || cols == 0 {
            return Err(Error::new(
                Code::Invalid,
                format!("矩阵乘维度必须非零（rows = {rows}，cols = {cols}）"),
            )
            .with("nn::MatMul::new"));
        }
        let spec = __layer_spec_MatMul!(name, rows, cols, Self::OP)?;
        Ok(Self { spec, rows, cols })
    }

    crate::accessors! {
        /// 输出行数（轴 `m`）。
        rows -> usize = rows,
        /// 输出列数（轴 `n`）。
        cols -> usize = cols,
    }
}

impl Contract for MatMul {
    spec_accessors!();

    fn bounds(&self) -> BoundsResult {
        // 三轴骨架：收缩轴 `k` 由算子的 `project_shape` 固定，层只给 `(m, n)`，走通用形状入口，
        // 而不是投影入口——投影入口只填输入/输出两条角色轴，填不出独立的收缩轴。
        Bounds::of_ctx(Self::OP, self.rows, self.cols).map(Some)
    }
}

#[cfg(test)]
#[path = "../tests/matmul_tests.rs"]
mod matmul_tests;
