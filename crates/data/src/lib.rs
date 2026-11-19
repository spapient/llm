//! 训练样本：[`Batch`]、[`Split`]、[`Dataset`] 与最简实现 [`Chunks`]。
//!
//! **拥有**：`Batch`、`Split`、`Dataset`、`Chunks`。**不拥有**：分词（`tokenizer`）、打乱与打包策略
//! （P2：它们只改变 `Chunks` 里游标的推进方式，不改变这些类型）。
//! **依赖**：`tensor`、`diagnostic`。**被依赖**：`train`。
//!
//! 契约的重点是"形状即校验"：`rows * cols` 必须等于 token 数，否则训练循环会在第 N 步才发现
//! 偏移错位——那时已经烧掉了几个小时。
//!
//! # 拥有与借用
//!
//! 这些类型**不拥有** token 缓冲：调用方持有它，本 crate 只给出一段视图。取一批因此没有分配、
//! 也没有拷贝，批次的生命周期停在调用方那份缓冲上，而不停在数据集对象上。
//!
//! # 按批取样本
//!
//! [`Chunks`] 是 `Iterator`，并且它知道自己还剩几批——`ExactSizeIterator::len` 与 `size_hint`
//! 都取自 [`Dataset::remaining_batches`]，「还剩几条」只有一个来源：
//!
//! ```
//! use llm_data::{Chunks, Dataset};
//!
//! let tokens = [0, 1, 2, 3, 4, 5];
//! let mut chunks = Chunks::new(&tokens, 1, 2).expect("每批 1 行、每行 2 个 token");
//! assert_eq!(chunks.len(), 3);
//! let rows: usize = chunks.by_ref().map(|batch| batch.rows()).sum();
//! assert_eq!((rows, chunks.len()), (3, 0));
//! // `next()` 与 `Dataset::next_batch` 是同一个游标，混用不会跳批。
//! let mut again = Chunks::new(&tokens, 1, 2).expect("同上");
//! assert_eq!(again.next_batch().expect("第一批").tokens(), [0, 1]);
//! assert_eq!(again.next().expect("第二批").tokens(), [2, 3]);
//! ```
//!
//! 它也是**对象安全**的：`&mut dyn Dataset<'a>` 与具体类型走同一条取批路径，所以按批消费的
//! 代码可以只依赖 trait。
//!
//! ```
//! use llm_data::{Chunks, Dataset};
//!
//! fn drain<'a>(dataset: &mut dyn Dataset<'a>) -> usize {
//!     core::iter::from_fn(|| dataset.next_batch()).count()
//! }
//!
//! let tokens = [0, 1, 2, 3];
//! let mut chunks = Chunks::new(&tokens, 1, 2).expect("每批 1 行、每行 2 个 token");
//! assert_eq!(drain(&mut chunks), 2);
//! ```
//!
//! # 两个单位
//!
//! [`Split`] 的单位是**样本行**，[`Chunks`] 的单位是**批**（一批 `rows` 行、每行 `cols` 个 token）。
//! 两者刻意不互相换算：切分不知道列宽，取批也不需要知道切分。要从切分走到「验证集的 token 段」，
//! 由调用方把行区间乘上列宽——`Split::ranges` 给的就是行区间。
//!
//! ```
//! use llm_data::{Chunks, Split};
//!
//! let tokens: Vec<u32> = (0..12).collect(); // 6 行，每行 2 个 token
//! let cols = 2;
//! let (train, valid) = Split::new(tokens.len() / cols, 25).expect("25% 验证集").ranges();
//! assert_eq!((train.len(), valid.len()), (5, 1));
//! let mut chunks = Chunks::new(&tokens[valid.start * cols..valid.end * cols], 1, cols)
//!     .expect("每批 1 行、每行 2 个 token");
//! assert_eq!(chunks.remaining_batches(), 1);
//! ```
//!
//! # 模块划分
//!
//! `batch` 管形状即校验，`split` 管行数划分，`dataset` 管按批取样本的游标。
//! P2 起三者内联在本文件里（不再一个类型一个文件），模块名与再导出路径不变。
//!
//! # 例子
//!
//! ```
//! use llm_data::{Batch, Split};
//!
//! let tokens = [1, 2, 3, 4, 5, 6];
//! let batch = Batch::new(&tokens, 2, 3).expect("2 x 3");
//! assert_eq!(batch.shape().dims(), [2, 3]);
//! let split = Split::new(100, 10).expect("10% 验证集");
//! assert_eq!((split.train(), split.valid()), (90, 10));
//! ```

#![deny(unsafe_code)]

mod batch {
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
        fn a_batch_must_reject_an_overflowing_shape() {
            let tokens = [1, 2];
            // 溢出必须报错，而不是回绕成一个恰好等于 `tokens.len()` 的形状：后者会让一个非法的
            // 2 x usize::MAX 被当成合法批，形状从此不再可信。
            assert_eq!(
                Batch::new(&tokens, usize::MAX, 2).unwrap_err().code(),
                Code::Invalid
            );
            assert_eq!(
                Batch::new(&tokens, 2, usize::MAX).unwrap_err().code(),
                Code::Invalid
            );
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
}

mod dataset {
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
}

mod split {
    //! 训练集 / 验证集的行数划分。

    use diagnostic::{Code, Error, Result};

    /// 训练集 / 验证集的行数划分。
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct Split {
        train: usize,
        valid: usize,
    }

    impl Split {
        /// 按验证集百分比切分：`0..=99`，且总行数非零。
        ///
        /// 100% 验证集直接报错而不是夹成 0%：训练集至少要留一行，否则"训练"会在第一次取批时
        /// 就失败，而调用方看到的却是一个合法的 `(total, 0)`。
        pub fn new(total: usize, valid_percent: u8) -> Result<Self> {
            if total == 0 {
                return Err(Error::new(Code::Invalid, "总行数不能为 0").with("data::Split::new"));
            }
            if valid_percent >= 100 {
                return Err(Error::new(
                    Code::Invalid,
                    format!("验证集百分比 {valid_percent} 必须在 0..=99：训练集至少要留一行"),
                )
                .with("data::Split::new"));
            }
            let valid = valid_rows(total, valid_percent);
            // 兜底：`valid_percent <= 99` 时它恒等；留着是为了将来放宽百分比时不再产生 0 训练行。
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

    /// `total` 行的 `percent`%：拆成「整百部分」与「余数部分」两次乘法，两项都装得进 `usize`。
    ///
    /// 记 `total = 100q + r`（`r < 100`），则 `total * p / 100 == q * p + r * p / 100`：前者因
    /// `p <= 99` 而 `q * p <= total * p / 100 <= total`，后者 `r * p / 100 < 100`，所以既不会溢出，
    /// 也不需要「中间宽度」或饱和。饱和正是上一轮要修掉的那件事——`saturating_mul` 在 `total` 接近
    /// `usize::MAX` 时先把乘积钉在 `usize::MAX` 再除以 100，会把比例放大近一倍。
    fn valid_rows(total: usize, percent: u8) -> usize {
        let percent = usize::from(percent);
        let (hundreds, rest) = (total / 100, total % 100);
        hundreds * percent + rest * percent / 100
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn a_split_always_leaves_a_training_row() {
            let split = Split::new(10, 30).expect("切分");
            assert_eq!((split.train(), split.valid()), (7, 3));
            assert_eq!(split.ranges(), (0..7, 7..10));
            let almost_all_valid = Split::new(100, 99).expect("切分");
            assert_eq!(
                (almost_all_valid.train(), almost_all_valid.valid()),
                (1, 99)
            );
            let half = Split::new(2, 50).expect("切分");
            assert_eq!((half.train(), half.valid()), (1, 1));
            // 2 行的 99% 是 1.98 行：向下取整得 1，于是训练集恰好留下 1 行（夹值在这里恒等）。
            let almost_half = Split::new(2, 99).expect("切分");
            assert_eq!((almost_half.train(), almost_half.valid()), (1, 1));
            let all_training = Split::new(100, 0).expect("切分");
            assert_eq!((all_training.train(), all_training.valid()), (100, 0));
            // 总行数只有一行时，任何非零比例都只能得到 0 验证行。
            for percent in [0, 1, 99] {
                let split = Split::new(1, percent).expect("切分");
                assert_eq!((split.train(), split.valid()), (1, 0), "百分比 {percent}");
            }
            assert_eq!(Split::new(0, 10).unwrap_err().code(), Code::Invalid);
            assert_eq!(Split::new(10, 101).unwrap_err().code(), Code::Invalid);
        }

        #[test]
        fn a_split_may_not_hand_over_every_row() {
            for total in [1, 100] {
                assert_eq!(
                    Split::new(total, 100).unwrap_err().code(),
                    Code::Invalid,
                    "total = {total} 时不该接受 100% 验证集"
                );
            }
        }

        #[test]
        fn a_split_rounds_the_same_way_as_an_exact_proportion() {
            // 字面表，重点是余数：`100q + r` 的拆分只要漏掉 `r * p / 100` 这一项，
            // 下面的 `(99, 99)`、`(199, 50)`、`(12345, 37)` 就会各差一行或更多。
            for (total, percent, valid) in [
                (99, 99, 98),
                (101, 99, 99),
                (199, 50, 99),
                (999, 99, 989),
                (1000, 1, 10),
                (12345, 37, 4567),
            ] {
                assert_eq!(
                    Split::new(total, percent).expect("切分").valid(),
                    valid,
                    "{total} 行的 {percent}% 向下取整应为 {valid}"
                );
            }
        }

        #[test]
        fn a_split_at_usize_max_keeps_the_proportion() {
            let total = usize::MAX;
            let split = Split::new(total, 50).expect("一半");
            // 只断言性质，不重算实现里的同一公式。
            assert_eq!(
                split.train() + split.valid(),
                total,
                "两半必须恰好占满全部行"
            );
            assert!(split.train() >= 1, "训练集至少要留一行");
            assert!(
                split.valid() * 2 <= total,
                "验证集不能超过一半，饱和乘法会让它接近两倍"
            );
        }
    }
}

pub use batch::Batch;
pub use dataset::{Chunks, Dataset};
pub use split::Split;
