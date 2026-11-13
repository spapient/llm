//! 数据集契约与最简实现：[`Dataset`] 按批取样本，[`Chunks`] 把一段连续 token 流切成固定几何的批。

use diagnostic::{Code, Error, Result};

use crate::Batch;

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
///
/// 它不做打乱、不做跨样本拼接；那两件事只改变游标推进方式，不改变 [`Batch`] 的形状契约。
/// 批几何在构造期校验，所以取批只剩「取完了」一种失败。
#[derive(Debug, Clone)]
pub struct Chunks<'a> {
    tokens: &'a [u32],
    cols: usize,
    rows: usize,
    stride: usize,
    cursor: usize,
}

impl<'a> Chunks<'a> {
    /// 由 token 流、列宽与每批行数构造：三者任一不合法都报 `Code::Invalid`。
    pub fn new(tokens: &'a [u32], cols: usize, rows: usize) -> Result<Self> {
        if cols == 0 {
            return Err(Error::new(Code::Invalid, "列宽不能为 0").with("data::Chunks::new"));
        }
        if rows == 0 {
            return Err(Error::new(Code::Invalid, "批行数不能为 0").with("data::Chunks::new"));
        }
        let stride = rows.checked_mul(cols).ok_or_else(|| {
            Error::new(
                Code::Invalid,
                format!("rows * cols 溢出（rows = {rows}，cols = {cols}）"),
            )
            .with("data::Chunks::new")
        })?;
        Ok(Self {
            tokens,
            cols,
            rows,
            stride,
            cursor: 0,
        })
    }

    /// 每批的行数（构造时给的批大小）。
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// 每批的 token 数（`rows * cols`，构造期算好）。
    pub fn stride(&self) -> usize {
        self.stride
    }

    /// 还没被取走的 token 数。
    pub fn remaining(&self) -> usize {
        self.tokens.len().saturating_sub(self.cursor)
    }
}

impl<'a> Dataset<'a> for Chunks<'a> {
    fn cols(&self) -> usize {
        self.cols
    }

    fn batches(&self) -> usize {
        self.tokens.len() / self.stride
    }

    fn remaining_batches(&self) -> usize {
        self.remaining() / self.stride
    }

    fn next_batch(&mut self) -> Option<Batch<'a>> {
        let end = self.cursor.checked_add(self.stride)?;
        let slice = self.tokens.get(self.cursor..end)?;
        self.cursor = end;
        Some(Batch::exact(slice, self.rows, self.cols))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_chunks_stream_rejects_bad_geometry() {
        let tokens = [1, 2, 3, 4];
        for (cols, rows) in [(0, 1), (1, 0), (2, usize::MAX)] {
            assert_eq!(
                Chunks::new(&tokens, cols, rows).unwrap_err().code(),
                Code::Invalid,
                "cols = {cols}、rows = {rows} 必须在构造期被拒"
            );
        }
    }

    #[test]
    fn an_empty_stream_yields_no_batch() {
        let mut dataset = Chunks::new(&[], 2, 1).expect("空流也是合法输入");
        assert_eq!(dataset.cols(), 2);
        assert_eq!(dataset.batches(), 0);
        assert_eq!(dataset.remaining_batches(), 0);
        assert_eq!(dataset.remaining(), 0);
        assert!(dataset.next_batch().is_none());
    }

    #[test]
    fn chunks_drop_the_partial_tail() {
        let tokens = [0, 1, 2, 3, 4, 5, 6];
        let mut dataset = Chunks::new(&tokens, 2, 1).expect("数据");
        assert_eq!(dataset.batches(), 3, "7 个 token 按 2 个一行只能凑 3 批");
        assert_eq!(dataset.remaining_batches(), 3);
        for expected in [&tokens[0..2], &tokens[2..4], &tokens[4..6]] {
            assert_eq!(dataset.next_batch().expect("整批").tokens(), expected);
        }
        assert!(
            dataset.next_batch().is_none(),
            "只剩 1 个 token，凑不满一行"
        );
        assert_eq!(dataset.remaining(), 1);
        assert_eq!(dataset.remaining_batches(), 0);
    }

    #[test]
    fn consuming_advances_both_counters() {
        let tokens: [u32; 6] = [0, 1, 2, 3, 4, 5];
        let mut dataset = Chunks::new(&tokens, 2, 1).expect("数据");
        assert_eq!(
            (
                dataset.cols(),
                dataset.batches(),
                dataset.remaining_batches()
            ),
            (2, 3, 3)
        );
        let batch = dataset.next_batch().expect("第一批");
        assert_eq!(batch.tokens(), [0, 1]);
        assert_eq!(dataset.remaining_batches(), 2);
        assert_eq!(dataset.remaining(), 4);
        assert_eq!(
            (dataset.rows(), dataset.batches(), dataset.cols()),
            (1, 3, 2),
            "每批行数、批总数、列宽都不随消费推进"
        );
        assert_eq!(dataset.stride(), 2, "每批 token 数在构造期算好");
    }

    #[test]
    fn two_batches_can_be_held_at_once() {
        let tokens: [u32; 6] = [0, 1, 2, 3, 4, 5];
        let mut dataset = Chunks::new(&tokens, 2, 1).expect("数据");
        let first = dataset.next_batch().expect("第一批");
        let second = dataset.next_batch().expect("第二批");
        // 两批的寿命来自 token 缓冲而不是数据集，所以可以同时存活且不重叠。
        assert_eq!(first.tokens(), [0, 1]);
        assert_eq!(second.tokens(), [2, 3]);
        assert_eq!(
            first.tokens().as_ptr().wrapping_add(first.tokens().len()),
            second.tokens().as_ptr(),
            "两批必须落在缓冲里相邻但不重叠的位置"
        );
    }
}
