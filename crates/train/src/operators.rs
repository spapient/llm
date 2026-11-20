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
mod tests {
    use super::*;

    use crate::{Buffers, Data, Reference};
    use autograd::{Sizes, Tape};
    use tensor::{DType, Shape};

    /// 一条 `add_f32`：类型化调用与字符串派发逐元素一致。
    #[test]
    fn a_typed_call_matches_the_string_path() {
        let mut tape = Tape::new();
        let a = tape.parameter();
        let b = tape.leaf();
        let typed = tape.record("add_f32", &[a, b]).expect("记录");
        let raw = tape.record("add_f32", &[a, b]).expect("记录");
        let mut sizes = Sizes::new(tape.vars());
        for var in [a, b, typed, raw] {
            sizes.set(var, Shape::new([4]), DType::F32);
        }
        let mut store = Buffers::new(sizes, Reference);
        store
            .load(a, Data::F32(vec![1.0, 2.0, 3.0, 4.0]))
            .expect("a");
        store
            .load(b, Data::F32(vec![0.5, 0.5, 0.5, 0.5]))
            .expect("b");
        store.add_f32(&[a, b], typed).expect("类型化调用");
        Store::compute(&mut store, "add_f32", &[a, b], raw).expect("字符串调用");
        let typed = store.value(typed).expect("有值").f32().expect("f32").to_vec();
        let raw = store.value(raw).expect("有值").f32().expect("f32").to_vec();
        assert_eq!(typed, raw);
        assert_eq!(typed, vec![1.5, 2.5, 3.5, 4.5]);
    }
}
