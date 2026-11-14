//! 数据集契约与最简实现：[`Dataset`] 按批取样本，[`Chunks`] 把一段连续 token 流切成固定几何的批。

use diagnostic::{Code, Error, Result};

use crate::Batch;

/// 数据集：按批取样本。三个计数语义互不重叠：列宽、完整批总数、还没取走的完整批数。
///
/// 它是**对象安全**的（四个方法都只取 `&self` / `&mut self`，没有泛型参数），所以按批消费的代码
/// 可以只依赖这个 trait，用 `&mut dyn Dataset<'a>` 接收任意实现。
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
/// 批几何与批总数都在构造期算好，所以取批只剩「取完了」一种失败。
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
    ///
    /// 参数顺序是 `(rows, cols)`，与 [`Batch::new`] 和 `Shape::new([rows, cols])` 一致。两个构造函数
    /// 曾经一个收 `(rows, cols)`、另一个收 `(cols, rows)`，而两种顺序都通得过校验——同一对数字换个
    /// 构造函数就是一次静默转置，正是这个 crate 要挡住的那类错值。
    pub fn new(tokens: &'a [u32], rows: usize, cols: usize) -> Result<Self> {
        if rows == 0 {
            return Err(Error::new(Code::Invalid, "批行数不能为 0").with("data::Chunks::new"));
        }
        if cols == 0 {
            return Err(Error::new(Code::Invalid, "列宽不能为 0").with("data::Chunks::new"));
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
            rows,
            cols,
            stride,
            batches: tokens.len() / stride,
            taken: 0,
        })
    }

    /// 每批的行数（构造时给的批大小）。
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// 列宽（每行多少 token）。
    pub fn cols(&self) -> usize {
        self.cols
    }

    /// 每批的 token 数（`rows * cols`，构造期算好）。
    pub fn stride(&self) -> usize {
        self.stride
    }

    /// 完整批的**总数**——构造期算好，不随消费推进。
    pub fn batches(&self) -> usize {
        self.batches
    }

    /// **还没取走**的完整批数。
    pub fn remaining_batches(&self) -> usize {
        self.batches - self.taken
    }

    /// 还没被取走的 token 数（含凑不满一批的尾巴）。
    pub fn remaining(&self) -> usize {
        // `taken <= batches = len / stride`，所以 `taken * stride <= len`：既不下溢也不上溢。
        self.tokens.len() - self.taken * self.stride
    }
}

/// 迭代器：`next` 就是 [`Dataset::next_batch`]，所以「还剩几条」只有一个来源。
///
/// `ExactSizeIterator` 不是装饰：`len()` 与 `size_hint()` 都取自 [`Dataset::remaining_batches`]，
/// 于是 `collect()` 能一次分配到位，「还能产出几条」也不需要调用方自己再数一遍。
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
        Chunks::cols(self)
    }

    fn batches(&self) -> usize {
        Chunks::batches(self)
    }

    fn remaining_batches(&self) -> usize {
        Chunks::remaining_batches(self)
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
mod tests {
    use super::*;

    #[test]
    fn a_chunks_stream_rejects_bad_geometry() {
        let tokens = [1, 2, 3, 4];
        for (rows, cols) in [(1, 0), (0, 1), (usize::MAX, 2)] {
            assert_eq!(
                Chunks::new(&tokens, rows, cols).unwrap_err().code(),
                Code::Invalid,
                "rows = {rows}、cols = {cols} 必须在构造期被拒"
            );
        }
    }

    #[test]
    fn both_constructors_read_geometry_in_the_same_order() {
        let tokens: [u32; 6] = [0, 1, 2, 3, 4, 5];
        // `(2, 3)` 在两个构造函数里都必须表示「2 行 3 列」。此前 `Chunks::new` 收的是 `(cols, rows)`，
        // 于是同一对数字换个构造函数就是一次静默转置——而两种顺序都能通过校验，没有一条用例会红。
        let batch = Batch::new(&tokens, 2, 3).expect("2 x 3");
        let mut dataset = Chunks::new(&tokens, 2, 3).expect("2 x 3");
        let first = dataset.next_batch().expect("6 个 token 恰好是一批");
        assert_eq!(first.shape(), batch.shape(), "两个构造函数必须读出同一形状");
        assert_eq!(first.tokens(), batch.tokens());
        assert_eq!(
            (first.rows(), first.cols()),
            (2, 3),
            "顺序反了会得到 (3, 2)"
        );
        assert_eq!(dataset.remaining_batches(), 0);
    }

    #[test]
    fn an_empty_stream_yields_no_batch() {
        let mut dataset = Chunks::new(&[], 1, 2).expect("空流也是合法输入");
        assert_eq!(dataset.cols(), 2);
        assert_eq!(dataset.batches(), 0);
        assert_eq!(dataset.remaining_batches(), 0);
        assert_eq!(dataset.remaining(), 0);
        assert!(dataset.next_batch().is_none());
    }

    #[test]
    fn chunks_drop_the_partial_tail() {
        let tokens = [0, 1, 2, 3, 4, 5, 6];
        let mut dataset = Chunks::new(&tokens, 1, 2).expect("数据");
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
        let mut dataset = Chunks::new(&tokens, 1, 2).expect("数据");
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
        let mut dataset = Chunks::new(&tokens, 1, 2).expect("数据");
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

    #[test]
    fn iterating_yields_exactly_the_declared_batches() {
        let tokens: [u32; 9] = [0, 1, 2, 3, 4, 5, 6, 7, 8];
        let mut dataset = Chunks::new(&tokens, 1, 3).expect("数据");
        assert_eq!(dataset.len(), 3, "`len()` 就是还没取走的批数");
        assert_eq!(dataset.size_hint(), (3, Some(3)));
        let collected: Vec<_> = dataset.by_ref().collect();
        assert_eq!(collected.len(), 3, "迭代条数必须等于 `batches()`");
        assert_eq!(collected[2].tokens(), [6, 7, 8]);
        assert_eq!(dataset.len(), 0);
        // 取空之后 `len()` 与两种取法都不再变化（`FusedIterator`）。
        assert!(dataset.next().is_none());
        assert!(dataset.next_batch().is_none());
        assert_eq!(dataset.size_hint(), (0, Some(0)));
    }

    #[test]
    fn next_and_next_batch_advance_the_same_cursor() {
        let tokens: [u32; 9] = [0, 1, 2, 3, 4, 5, 6, 7, 8];
        let mut dataset = Chunks::new(&tokens, 1, 3).expect("数据");
        assert_eq!(dataset.next().expect("第一条").tokens(), [0, 1, 2]);
        assert_eq!(dataset.next_batch().expect("第二条").tokens(), [3, 4, 5]);
        assert_eq!(
            dataset.by_ref().count(),
            1,
            "两种取法共用同一个游标，不会跳批"
        );
        assert_eq!(dataset.remaining_batches(), 0);
    }

    #[test]
    fn the_trait_object_reports_and_yields_the_same_stream() {
        let tokens: [u32; 9] = [0, 1, 2, 3, 4, 5, 6, 7, 8];
        let mut chunks = Chunks::new(&tokens, 1, 3).expect("数据");
        // 走 trait 对象路径：具体类型上的同名调用会解析到 inherent 方法，只有 `dyn Dataset`
        // 才真正执行 trait 实现里的三个转发，从而验证对象安全契约与转发正确性。
        let ds: &mut dyn Dataset<'_> = &mut chunks;
        assert_eq!(ds.cols(), 3);
        assert_eq!(ds.batches(), 3);
        assert_eq!(ds.remaining_batches(), 3);
        for expected in [&tokens[0..3], &tokens[3..6], &tokens[6..9]] {
            assert_eq!(ds.next_batch().expect("整批").tokens(), expected);
        }
        assert_eq!(ds.remaining_batches(), 0);
        assert!(ds.next_batch().is_none(), "取空后 trait 路径同样终止");
    }
}
