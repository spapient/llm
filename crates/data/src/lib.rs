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
