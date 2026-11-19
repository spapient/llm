//! 矩阵乘层：`Y[m,n] = Σ_k A[m,k] · B[k,n]`。
//!
//! 与 [`crate::Linear`]（向量 × 矩阵的 GEMV）的差别在骨架：本层是矩阵 × 矩阵的三轴骨架，
//! 收缩轴 `k` 由算子语义表固定（`gemm` 家族的固定档），层只声明输出
//! 行数 `rows`（轴 `m`）与列数 `cols`（轴 `n`）。下界仍由 `gemm_f32` 的语义表推出，层里不写
//! 第二份流量常数；收缩轴宽度属于算子，不由层猜。

use diagnostic::{Code, Error, Result};
use tensor::Shape;

use crate::{Bounds, BoundsResult, Contract, ContractSpec, ParamRole, spec_accessors};

/// 矩阵乘层：结果形状 `[rows, cols]`，收缩轴由算子固定。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MatMul {
    spec: ContractSpec,
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
    /// （[`ContractSpec::new`] 上报）。
    pub fn new(name: impl Into<String>, rows: usize, cols: usize) -> Result<Self> {
        if rows == 0 || cols == 0 {
            return Err(Error::new(
                Code::Invalid,
                format!("矩阵乘维度必须非零（rows = {rows}，cols = {cols}）"),
            )
            .with("nn::MatMul::new"));
        }
        let spec = ContractSpec::new_with_role(
            name,
            Shape::new([rows, cols]),
            vec![Self::OP.to_string()],
            ParamRole::Trainable,
        )?;
        Ok(Self { spec, rows, cols })
    }

    /// 输出行数（轴 `m`）。
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// 输出列数（轴 `n`）。
    pub fn cols(&self) -> usize {
        self.cols
    }
}

impl Contract for MatMul {
    spec_accessors!();

    fn bounds(&self) -> BoundsResult {
        // 三轴骨架：收缩轴 `k` 由算子的 `project_shape` 固定，层只给 `(m, n)`，走通用形状入口，
        // 而不是投影入口——投影入口只填输入/输出两条角色轴，填不出独立的收缩轴。
        Bounds::of_context(Self::OP, self.rows, self.cols).map(Some)
    }
}

#[cfg(test)]
mod tests {
use super::*;

#[test]
fn a_matmul_layer_binds_its_operator_and_shape() {
    let layer = MatMul::new("blk.0.matmul", 16, 12).expect("合法");
    assert_eq!(layer.name(), "blk.0.matmul");
    assert_eq!(layer.output().dims(), [16, 12]);
    assert_eq!(layer.spca().ops(), [MatMul::OP]);
    assert_eq!(layer.rows(), 16);
    assert_eq!(layer.cols(), 12);
}

#[test]
fn the_matmul_floor_follows_the_gemm_semantics() {
    let layer = MatMul::new("blk.0.matmul", 16, 12).expect("合法");
    let floor = layer.bounds().expect("合法").expect("有算子");
    // 层不写第二份公式：下界必须与直接问语义表逐项相等。
    let same = Bounds::of_context(MatMul::OP, 16, 12).expect("同一份语义");
    assert_eq!(floor, same, "矩阵乘下界来自算子语义表");
    assert!(floor.read() > 0 && floor.write() > 0 && floor.flops() > 0.0);
}

#[test]
fn a_zero_dimension_matmul_is_rejected() {
    assert_eq!(MatMul::new("x", 0, 12).unwrap_err().code(), Code::Invalid);
    assert_eq!(MatMul::new("x", 16, 0).unwrap_err().code(), Code::Invalid);
}

#[test]
fn a_matmul_works_as_a_trait_object() {
    let layer: &dyn Contract = &MatMul::new("proj", 16, 12).expect("合法");
    assert_eq!(layer.output().dims(), [16, 12]);
    assert!(layer.bounds().expect("合法").is_some());
}
}

