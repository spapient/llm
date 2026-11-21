//! 类型化算子入口：从 `op::for_each_operator!` 的同一张清单展开，没有第二份装配表。
//!
//! 后端只需实现 [`OperatorSetOps::apply`] 一个原语；其余方法都是默认方法，名字取
//! `Operator::NAME`，输入个数取 `Operator::ARITY`。清单的方法名与 `Operator::NAME` 在
//! `op` 里编译期逐字节对拍，`ARITY` 与算子样本在 `op` 的 `arity_matches_the_samples` 对拍，
//! 因此注册表、API 面与样本不会各自漂移。
//!
//! 例子（`use llm_train::OperatorSetOps;` 之后）：
//!
//! ```ignore
//! store.add_f32(&[a, b], sum)?;          // 定长输入；名字在编译期确定
//! store.matmul_f32(&[lhs, rhs], out)?;   // 与 store.compute("matmul_f32", ..) 同一条路径
//! ```

use autograd::{Store, Var};
use diagnostic::Result;

/// 把算子清单展开成 trait：一个原语 + 每个算子一条默认方法。
macro_rules! typed_ops {
    ($($name:ident: $ty:ty => $template:literal),* $(,)?) => {
        /// 由 `op::for_each_operator!` 展开的类型化算子入口。
        ///
        /// 后端实现一次 [`apply`](Self::apply)，就拿到全部算子的定长输入方法。
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
impl<K: crate::Kernel> OperatorSetOps for crate::Buffers<K> {
    fn apply(&mut self, name: &'static str, inputs: &[Var], output: Var) -> Result<()> {
        Store::compute(self, name, inputs, output)
    }
}

#[cfg(test)]
#[path = "../tests/operators_tests.rs"]
mod tests;
