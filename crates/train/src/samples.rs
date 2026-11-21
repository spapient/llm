//! 训练样本：[`Batch`]、[`Dataset`] 与最简实现 [`Chunks`]。
//!
//! 这三件套原来是独立成员 `llm-data`：唯一消费者就是本 crate（[`crate::Learner`] 经
//! `&mut dyn Dataset` 取批、[`crate::Checkpoint`] 记录批形状），因此按"单一消费者"
//! 判据并入本 crate——样本与训练循环状态同居一个成员，`train → data` 与 `data → diagnostic`
//! 两条边一起消失，也不再有一个只为 `train` 存在的成员。
//!
//! **拥有**：`Batch`、`Dataset`、`Chunks`。**不拥有**：分词（`tokenizer`）、打乱、打包与训练/验证集
//! 切分（打乱与打包只改变 [`Chunks`] 里游标的推进方式，切分是调用方按行区间的选择）。
//!
//! 契约的重点是"形状即校验"：`rows * cols` 必须等于 token 数，否则训练循环会在第 N 步才发现
//! 偏移错位——那时已经烧掉了几个小时。两个构造函数共用同一份几何校验（行列非零、乘积不溢出），
//! 所以同一对非法行列不管走哪条路径，拿到的是同一个 `Code::Invalid`。
//!
//! # 拥有与借用
//!
//! 这些类型**不拥有** token 缓冲：调用方持有它，本模块只给出一段视图。取一批因此没有分配、
//! 也没有拷贝，批次的生命周期停在调用方那份缓冲上，而不停在数据集对象上。
//!
//! # 按批取样本
//!
//! [`Chunks`] 是 `Iterator`，并且它知道自己还剩几批——`ExactSizeIterator::len` 与 `size_hint`
//! 都取自 [`Dataset::remaining_batches`]，「还剩几条」只有一个来源：
//!
//! ```
//! use llm_train::{Chunks, Dataset};
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
//! use llm_train::{Chunks, Dataset};
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
//! # 例子
//!
//! ```
//! use llm_train::Batch;
//!
//! let tokens = [1, 2, 3, 4, 5, 6];
//! let batch = Batch::new(&tokens, 2, 3).expect("2 x 3");
//! assert_eq!((batch.rows(), batch.cols()), (2, 3));
//! ```

use diagnostic::{Code, Error, Result};

/// 批几何：`rows`、`cols` 非零且乘积不溢出，返回每批 token 数（`rows * cols`）。
///
/// 两个构造函数共用这一份校验——此前 `Batch::new` 与 `Chunks::new` 各写一套，同一对非法行列在
/// 两条路径上给出不同措辞。`site` 只用于错误出处，不改判定。
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
    ///
    /// 形状只在 debug 构建里断言；release 下不会越界，因为切片长度由调用方给的跨度推导。
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
    /// 参数顺序是 `(rows, cols)`，与 [`Batch::new`] 一致。两个构造函数
    /// 曾经一个收 `(rows, cols)`、另一个收 `(cols, rows)`，而两种顺序都通得过校验——同一对数字换个
    /// 构造函数就是一次静默转置，正是这套类型要挡住的那类错值。
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
mod tests;
