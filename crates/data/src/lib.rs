//! 训练样本：[`Batch`]、[`Split`]、[`Dataset`] 与最简实现 [`Chunks`]。
//!
//! **拥有**：`Batch`、`Split`、`Dataset`、`Chunks`。**不拥有**：分词（`tokenizer`）、打乱与打包策略
//! （P2：它们只改变 `Chunks` 里游标的推进方式，不改变这些类型）。
//! **依赖**：`tensor`、`diagnostic`。**被依赖**：`train`。
//!
//! 契约的重点是"形状即校验"：`rows * cols` 必须等于 token 数，否则训练循环会在第 N 步才发现
//! 偏移错位——那时已经烧掉了几个小时。
//!
//! # 例子
//!
//! ```
//! use llm_data::{Batch, Split};
//!
//! let batch = Batch::new(vec![1, 2, 3, 4, 5, 6], 2, 3).expect("2 x 3");
//! assert_eq!(batch.shape().dims(), [2, 3]);
//! let split = Split::new(100, 10).expect("10% 验证集");
//! assert_eq!((split.train(), split.valid()), (90, 10));
//! ```

#![deny(unsafe_code)]

use diagnostic::{Code, Error, Result};
use tensor::Shape;

/// 一个批：`rows x cols` 个 token，行主序。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Batch {
    tokens: Vec<u32>,
    rows: usize,
    cols: usize,
}

impl Batch {
    /// 构造：`rows` 与 `cols` 都非零，且 `tokens.len() == rows * cols`。
    pub fn new(tokens: Vec<u32>, rows: usize, cols: usize) -> Result<Self> {
        if rows == 0 || cols == 0 {
            return Err(Error::new(
                Code::Invalid,
                format!("批的行列必须非零（rows = {rows}，cols = {cols}）"),
            )
            .with("data::Batch::new"));
        }
        let expected = rows.checked_mul(cols).ok_or_else(|| {
            Error::new(Code::Invalid, "rows * cols 溢出").with("data::Batch::new")
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

    /// token 序列（行主序）。
    pub fn tokens(&self) -> &[u32] {
        &self.tokens
    }

    /// token 数。
    pub fn len(&self) -> usize {
        self.tokens.len()
    }

    /// 是否为空批（构造期已禁止，恒为 `false`；保留以对称）。
    pub fn is_empty(&self) -> bool {
        self.tokens.is_empty()
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

/// 训练集 / 验证集的行数划分。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Split {
    train: usize,
    valid: usize,
}

impl Split {
    /// 按验证集百分比切分：`0..=100`，且总行数非零。
    pub fn new(total: usize, valid_percent: u8) -> Result<Self> {
        if total == 0 {
            return Err(Error::new(Code::Invalid, "总行数不能为 0").with("data::Split::new"));
        }
        if valid_percent > 100 {
            return Err(Error::new(
                Code::Invalid,
                format!("验证集百分比 {valid_percent} 超出 0..=100"),
            )
            .with("data::Split::new"));
        }
        let valid = total.saturating_mul(usize::from(valid_percent)) / 100;
        // 训练集至少要留一行：否则"训练"会在第一次取批时就失败。
        let valid = valid.min(total - 1);
        Ok(Self {
            train: total - valid,
            valid,
        })
    }

    /// 训练集行数。
    pub fn train(&self) -> usize {
        self.train
    }

    /// 验证集行数。
    pub fn valid(&self) -> usize {
        self.valid
    }

    /// 行区间：`(0..train, train..train+valid)`。
    pub fn ranges(&self) -> (core::ops::Range<usize>, core::ops::Range<usize>) {
        (0..self.train, self.train..self.train + self.valid)
    }
}

/// 数据集：按行取批。
pub trait Dataset {
    /// 总行数。
    fn rows(&self) -> usize;

    /// 取下 `rows` 行；不足一整个批时返回 `None`（尾部残批直接丢弃，不做补齐）。
    fn next_batch(&mut self, rows: usize) -> Option<Batch>;
}

/// 最简实现：一段连续的 token 流按 `cols` 宽度切批。
///
/// 它不做打乱、不做跨样本拼接；那两件事只改变游标推进方式，不改变 [`Batch`] 的形状契约。
#[derive(Debug, Clone)]
pub struct Chunks {
    tokens: Vec<u32>,
    cols: usize,
    cursor: usize,
}

impl Chunks {
    /// 由 token 流与列宽构造。
    pub fn new(tokens: Vec<u32>, cols: usize) -> Result<Self> {
        if cols == 0 {
            return Err(Error::new(Code::Invalid, "列宽不能为 0").with("data::Chunks::new"));
        }
        Ok(Self {
            tokens,
            cols,
            cursor: 0,
        })
    }

    /// 还没被取走的 token 数。
    pub fn remaining(&self) -> usize {
        self.tokens.len().saturating_sub(self.cursor)
    }
}

impl Dataset for Chunks {
    fn rows(&self) -> usize {
        self.tokens.len() / self.cols
    }

    fn next_batch(&mut self, rows: usize) -> Option<Batch> {
        if rows == 0 {
            return None;
        }
        let take = rows.checked_mul(self.cols)?;
        let slice = self
            .tokens
            .get(self.cursor..self.cursor.checked_add(take)?)?;
        let batch = Batch::new(slice.to_vec(), rows, self.cols).ok()?;
        self.cursor += take;
        Some(batch)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_batch_must_match_its_shape() {
        let batch = Batch::new(vec![1, 2, 3, 4], 2, 2).expect("2 x 2");
        assert_eq!(batch.len(), 4);
        assert!(!batch.is_empty());
        assert_eq!(batch.shape().dims(), [2, 2]);
        assert_eq!(
            Batch::new(vec![1, 2, 3], 2, 2).unwrap_err().code(),
            Code::Invalid
        );
        assert_eq!(Batch::new(vec![], 0, 2).unwrap_err().code(), Code::Invalid);
    }

    #[test]
    fn a_split_always_leaves_a_training_row() {
        let split = Split::new(10, 30).expect("切分");
        assert_eq!((split.train(), split.valid()), (7, 3));
        assert_eq!(split.ranges(), (0..7, 7..10));
        let all_valid = Split::new(1, 100).expect("切分");
        assert_eq!((all_valid.train(), all_valid.valid()), (1, 0));
        assert_eq!(Split::new(0, 10).unwrap_err().code(), Code::Invalid);
        assert_eq!(Split::new(10, 101).unwrap_err().code(), Code::Invalid);
    }

    #[test]
    fn chunks_drop_the_partial_tail() {
        let mut dataset = Chunks::new((0..7).collect(), 2).expect("数据");
        assert_eq!(dataset.rows(), 3);
        let first = dataset.next_batch(2).expect("第一批");
        assert_eq!(first.tokens(), [0, 1, 2, 3]);
        let second = dataset.next_batch(1).expect("第二批");
        assert_eq!(second.tokens(), [4, 5]);
        assert!(
            dataset.next_batch(1).is_none(),
            "只剩 1 个 token，凑不满一行"
        );
        assert_eq!(dataset.remaining(), 1);
        assert!(dataset.next_batch(0).is_none(), "0 行不是有效批");
    }
}
