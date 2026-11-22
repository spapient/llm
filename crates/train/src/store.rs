//! 训练侧的激活缓冲：[`Bufs`] 把 [`autograd::Store`] 落到真实内存，算的部分交给 [`Kernel`]。
//!
//! **拥有**：`Bufs`、`Kernel`、`Reference`、`Data`、`View`、`ViewMut`，以及承载静态契约的
//! `Kind`、`Lane`、`Inputs`。**不拥有**：显存计划（`autograd::Plan`）、反向调度、形状表、算子语义
//! （`op`）。**依赖**：`autograd`、`tensor`、`op`、`diagnostic`。
//!
//! 三条硬约束：缓冲只有 `f32` 与按位打包的 `u8` 两种形态（`unsafe_code = deny`，"把字节重解释成
//! 浮点"这条路关着，`F16` / `BF16` 于是在编译期就被拒绝）；内核表与 [`op::REGISTRY`] 同名同序，
//! "注册了算子却没有参考实现"是编译错误；`live` / `peak` 都是实测值，与 [`autograd::Plan`] 的承诺
//! 对账，而计划保留的检查点必须在重放开始前就在场。
//!
//! 三条语义轴平铺在 crate 根：[`shape`] 是形态/视图/拆包契约，[`kernel`] 是标量参考内核适配，
//! [`pool`] 是等长空闲表与 [`Bufs`] 落地；本模块是这一层的**门面**——把三者的类型收成一个名字，
//! 持有该层统一的诊断站点，并把 `op` 的算子清单展开成 [`OperatorSetOps`] 的默认方法。
//!
//! # 例子
//!
//! ```
//! use autograd::{Sizes, Store, Tape};
//! use llm_train::{Bufs, Data, Reference};
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
//! let mut store = Bufs::new(sizes, Reference);
//! store.load(x, Data::F32(vec![1.0, 2.0])).expect("输入");
//! store.load(w, Data::F32(vec![1.0, 0.0, 0.0, 1.0])).expect("权重");
//! store.compute("gemv_f32", &[w, x], y).expect("可算");
//! assert!(store.value(y).and_then(|view| view.f32()).is_some());
//! ```

/// 本模块的诊断站点：出错时用它标明"是这一层拒绝了这次请求"。
pub(crate) const SITE: &str = "train::Buffers";

pub use crate::kernel::{Kernel, Reference};

pub use crate::pool::Bufs;

pub use crate::shape::{Data, Inputs, Kind, Lane, View, ViewMut};

#[cfg(test)]
#[path = "../tests/store_tests.rs"]
mod tests;

// ---- 类型化算子入口 --------------------------------------------------------
//
// 从 `op::for_each_operator!` 的同一张清单展开，没有第二份装配表：后端只需实现
// `OperatorSetOps::apply` 一个原语，其余方法都是默认方法，名字取 `Operator::NAME`、
// 输入个数取 `Operator::ARITY`。方法名与 `Operator::NAME` 在 `op` 里编译期逐字节对拍，
// 因此注册表、API 面与样本不会各自漂移。

use autograd::{Store, Var};
use diagnostic::Result;

/// 把算子清单展开成 trait：一个原语 + 每个算子一条默认方法。
macro_rules! typed_ops {
    ($($name:ident: $ty:ty => $template:literal),* $(,)?) => {
        /// 由 `op::for_each_operator!` 展开的类型化算子入口。
        pub trait OperatorSetOps {
            /// 唯一原语：按注册名派发一次。
            ///
            /// # Errors
            ///
            /// 算子未注册、输入个数或形状不符时返回说明。
            fn apply(&mut self, name: &'static str, inputs: &[Var], output: Var) -> Result<()>;

            $(
                #[doc = concat!(
                    "`", stringify!($name),
                    "`：名字取 `Operator::NAME`，输入个数取 `Operator::ARITY`。"
                )]
                #[inline]
                fn $name(
                    &mut self,
                    inputs: &[Var; <$ty as ::op::operator::Operator>::ARITY],
                    output: Var,
                ) -> Result<()> {
                    <Self as OperatorSetOps>::apply(
                        self,
                        <$ty as ::op::operator::Operator>::NAME,
                        inputs.as_slice(),
                        output,
                    )
                }
            )*
        }
    };
}

::op::for_each_operator!(typed_ops);

/// 运行时缓冲实现唯一原语：按注册名派发到既有内核表。
impl<K: crate::Kernel> OperatorSetOps for crate::Bufs<K> {
    fn apply(&mut self, name: &'static str, inputs: &[Var], output: Var) -> Result<()> {
        Store::compute_step(self, name, inputs, output)
    }
}
