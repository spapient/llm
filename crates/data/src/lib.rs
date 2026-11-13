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
//! 一个类型一个模块，测试跟着它自己的契约走：`batch` 管形状即校验，`split` 管行数划分，
//! `dataset` 管按批取样本的游标。这里只做声明与再导出。
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

mod batch;
mod dataset;
mod split;

pub use batch::Batch;
pub use dataset::{Chunks, Dataset};
pub use split::Split;
