//! 训练样本：`Batch` 批几何、`Dataset` 取批契约与最简连续流实现 `Chunks`。

use diagnostic::{Code, Error, Result};

/// 批几何：`rows`、`cols` 非零且乘积不溢出，返回每批 token 数（`rows * cols`）。
fn stride(rows: usize, cols: usize, site: &str) -> Result<usize> {
    if rows == 0 {
        return Err(Error::new(Code::Invalid, "批行数不能为 0").with(site));
    }
    if cols == 0 {
        return Err(Error::new(Code::Invalid, "列宽不能为 0").with(site));
    }
    rows.checked_mul(cols).ok_or_else(|| {
        Error::new(
            Code::Invalid,
            format!("rows * cols 溢出（rows = {rows}，cols = {cols}）"),
        )
        .with(site)
    })
}

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
        // 行列非零与乘积不溢出都由同一份几何校验给出，`Chunks::new` 走的是同一条。
        let expected = stride(rows, cols, "train::Batch::new")?;
        if tokens.len() != expected {
            return Err(Error::new(
                Code::Invalid,
                format!("token 数 {} 与 {rows} x {cols} 不符", tokens.len()),
            )
            .with("train::Batch::new"));
        }
        Ok(Self { tokens, rows, cols })
    }

    /// 不变量已由调用方在构造期保证（`tokens.len() == rows * cols` 且两者非零）时使用。
    fn exact(tokens: &'a [u32], rows: usize, cols: usize) -> Self {
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
}

/// 数据集：按批取样本。三个计数语义互不重叠：列宽、完整批总数、还没取走的完整批数。
pub trait Dataset<'a> {
    /// 列宽（每行多少 token）。
    fn cols(&self) -> usize;

    /// 完整批的**总数**——不随消费推进。
    fn batches(&self) -> usize;

    /// **还没取走**的完整批数。
    fn remaining_batches(&self) -> usize;

    /// 取下一批；`None` 当且仅当剩余 token 不足一整批（尾部残批丢弃，不补齐）。
    fn next_batch(&mut self) -> Option<Batch<'a>>;
}

/// 最简实现：一段连续 token 流按固定几何切批。
#[derive(Debug, Clone)]
pub struct Chunks<'a> {
    tokens: &'a [u32],
    rows: usize,
    cols: usize,
    stride: usize,
    /// 完整批总数：`tokens.len() / stride`，构造期算好，不随消费推进。
    batches: usize,
    /// 已取走的批数（不是 token 数：游标永远是 `stride` 的整数倍，那是不变量的另一半）。
    taken: usize,
}

impl<'a> Chunks<'a> {
    /// 由 token 流、每批行数与列宽构造：三者任一不合法都报 `Code::Invalid`。
    pub fn new(tokens: &'a [u32], rows: usize, cols: usize) -> Result<Self> {
        let stride = stride(rows, cols, "train::Chunks::new")?;
        Ok(Self {
            tokens,
            rows,
            cols,
            stride,
            batches: tokens.len() / stride,
            taken: 0,
        })
    }
}

/// 迭代器：`next` 就是 [`Dataset::next_batch`]，所以「还剩几条」只有一个来源。
impl<'a> Iterator for Chunks<'a> {
    type Item = Batch<'a>;

    fn next(&mut self) -> Option<Batch<'a>> {
        self.next_batch()
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.remaining_batches();
        (remaining, Some(remaining))
    }
}

impl ExactSizeIterator for Chunks<'_> {}

impl core::iter::FusedIterator for Chunks<'_> {}

impl<'a> Dataset<'a> for Chunks<'a> {
    fn cols(&self) -> usize {
        self.cols
    }

    fn batches(&self) -> usize {
        self.batches
    }

    fn remaining_batches(&self) -> usize {
        self.batches - self.taken
    }

    fn next_batch(&mut self) -> Option<Batch<'a>> {
        if self.taken == self.batches {
            return None;
        }
        // `taken < batches == len / stride` 蕴含 `end <= len`：越界在结构上不可能，因此这里不需要
        // 一条只能表示「取完了」的 `?`（那会把「内部不变量破了」重新折成 `None`）。
        let start = self.taken * self.stride;
        let end = start + self.stride;
        self.taken += 1;
        Some(Batch::exact(&self.tokens[start..end], self.rows, self.cols))
    }
}

#[cfg(test)]
#[path = "../tests/samples_tests.rs"]
mod samples_tests;
