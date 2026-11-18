//! L1 确定性随机基础件：计数器型 Philox4x32-10 引擎与自包含的分布闭集。
//!
//! 随机数由显式种子、流号与元素下标派生，不持有全局可变状态、不读线程熵源，也不引入
//! `rand` 生态依赖。调用方持有 [`Rng`] 实例；同种子、同流号、同下标序列逐位可复现，
//! CPU 顺序填充与 GPU 按索引寻址走同一计数器函数，两条后端可对拍。
//!
//! **拥有**：[`Rng`]、[`block`] 计数器派生，以及 [`Uniform`] / [`Normal`] /
//! [`Bernoulli`] 分布。**不拥有**：张量元数据（`tensor`）、缓冲分配（`storage`）、
//! 采样策略截断（`serve`）；也不做半精度输出舍入，那属于 `dtype` 转换职责。
//!
//! **依赖**：`diagnostic`（错误类型）。**不依赖**任何运行时工作区成员。

#![deny(unsafe_code)]

pub mod dist;
pub mod philox;

pub use dist::{Bernoulli, Normal, Uniform};
pub use philox::{Rng, block};
