//! 一个批：`rows x cols` 个 token，行主序；借用调用方的缓冲，不拥有。

use diagnostic::{Code, Error, Result};
use tensor::Shape;

/// 一个批：`rows x cols` 个 token，行主序；借用调用方的缓冲，不拥有。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Batch<'a> {
    tokens: &'a [u32],
    rows: usize,
    cols: usize,
}

impl<'a> Batch<'a> {
    /// 构造：`rows` 与 `cols` 都非零，且 `tokens.len() == rows * cols`。
    pub fn new(tokens: &'a [u32], rows: usize, cols: usize) -> Result<Self> {
        // 行列非零与乘积不溢出是同一件事：只要乘积是 `None` 或 `0`，批的形状就不成立。
        let expected = rows
            .checked_mul(cols)
            .filter(|expected| *expected != 0)
            .ok_or_else(|| {
                Error::new(
                    Code::Invalid,
                    format!("批的行列必须非零且乘积不溢出（rows = {rows}，cols = {cols}）"),
                )
                .with("data::Batch::new")
            })?;
        if tokens.len() != expected {
            return Err(Error::new(
                Code::Invalid,
                format!("token 数 {} 与 {rows} x {cols} 不符", tokens.len()),
            )
            .with("data::Batch::new"));
        }
        Ok(Self { tokens, rows, cols })
    }

    /// 不变量已由调用方在构造期保证（`tokens.len() == rows * cols` 且两者非零）时使用。
    ///
    /// 形状只在 debug 构建里断言；release 下不会越界，因为切片长度由调用方给的跨度推导。
    pub(crate) fn exact(tokens: &'a [u32], rows: usize, cols: usize) -> Self {
        debug_assert_eq!(
            tokens.len(),
            rows * cols,
            "内部批次的长度必须等于 rows * cols"
        );
        Self { tokens, rows, cols }
    }

    /// token 序列（行主序）。
    pub fn tokens(&self) -> &'a [u32] {
        self.tokens
    }

    /// 行数。
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// 列数。
    pub fn cols(&self) -> usize {
        self.cols
    }

    /// 逻辑形状：`[rows, cols]`。
    pub fn shape(&self) -> Shape {
        Shape::new([self.rows, self.cols])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_batch_must_match_its_shape() {
        let tokens = [1, 2, 3, 4];
        let batch = Batch::new(&tokens, 2, 2).expect("2 x 2");
        assert_eq!(batch.tokens(), tokens);
        assert_eq!(batch.shape().dims(), [2, 2]);
        assert_eq!(
            Batch::new(&[1, 2, 3], 2, 2).unwrap_err().code(),
            Code::Invalid
        );
        assert_eq!(Batch::new(&[], 0, 2).unwrap_err().code(), Code::Invalid);
    }

    #[test]
    fn a_batch_borrows_the_callers_buffer() {
        let tokens = [7, 8, 9, 10, 11, 12];
        for (rows, cols) in [(1, 6), (2, 3), (3, 2), (6, 1)] {
            let batch = Batch::new(&tokens, rows, cols).expect("形状合法");
            assert_eq!(
                batch.tokens().as_ptr(),
                tokens.as_ptr(),
                "批必须借用调用方的缓冲而不是拷贝"
            );
            assert_eq!(rows * cols, batch.tokens().len());
        }
    }
}
