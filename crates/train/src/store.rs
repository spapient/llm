//! 训练侧的激活缓冲：[`Buffers`] 把 [`autograd::Store`] 落到真实内存，算的部分交给 [`Kernel`]。
//!
//! **拥有**：`Buffers`、`Kernel`、`Reference`、`Data`、`View`、`ViewMut`，以及承载静态契约的
//! `Kind`、`Lane`、`Inputs`。**不拥有**：显存计划（`autograd::Plan`）、反向调度、形状表、算子语义
//! （`op`）。**依赖**：`autograd`、`tensor`、`op`、`diagnostic`。
//!
//! 三条硬约束：缓冲只有 `f32` 与按位打包的 `u8` 两种形态（`unsafe_code = deny`，"把字节重解释成
//! 浮点"这条路关着，`F16` / `BF16` 于是在编译期就被拒绝）；内核表与 [`op::REGISTRY`] 同名同序，
//! "注册了算子却没有参考实现"是编译错误；`live` / `peak` 都是实测值，与 [`autograd::Plan`] 的承诺
//! 对账，而计划保留的检查点必须在重放开始前就在场。
//!
//! 三条语义轴平铺在 crate 根：[`shape`] 是形态/视图/拆包契约，[`kernel`] 是标量参考内核适配，
//! [`pool`] 是等长空闲表与 [`Buffers`] 落地；本模块只做类型再导出与诊断站点。
//!
//! # 例子
//!
//! ```
//! use autograd::{Sizes, Store, Tape};
//! use llm_train::{Buffers, Data, Reference};
//! use tensor::{DType, Shape};
//!
//! let mut tape = Tape::new();
//! let (x, w) = (tape.leaf(), tape.leaf());
//! let y = tape.record("gemv_f32", &[w, x]).expect("记录");
//! let mut sizes = Sizes::new(tape.vars());
//! sizes.set(x, Shape::new([2]), DType::F32);
//! sizes.set(w, Shape::new([2, 2]), DType::F32);
//! sizes.set(y, Shape::new([2]), DType::F32);
//!
//! let mut store = Buffers::new(sizes, Reference);
//! store.load(x, Data::F32(vec![1.0, 2.0])).expect("输入");
//! store.load(w, Data::F32(vec![1.0, 0.0, 0.0, 1.0])).expect("权重");
//! store.compute("gemv_f32", &[w, x], y).expect("可算");
//! assert!(store.value(y).and_then(|view| view.f32()).is_some());
//! ```

/// 本模块的诊断站点：出错时用它标明"是这一层拒绝了这次请求"。
pub(crate) const SITE: &str = "train::Buffers";

// 测试白盒可见：形状表下标与登记元数据经入口再导出，`super::*` 仍能解析。
#[cfg(test)]
pub(crate) use autograd::{Sizes, Var};
#[cfg(test)]
pub(crate) use tensor::{DType, Shape};

pub use crate::kernel::{Kernel, Reference};
pub use crate::pool::Buffers;
#[cfg(test)]
pub(crate) use crate::pool::Pool;
pub use crate::shape::{Data, Inputs, Kind, Lane, View, ViewMut};

#[cfg(test)]
#[path = "../tests/store_tests.rs"]
mod tests;
