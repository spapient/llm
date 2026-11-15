//! 归约层：把一维输入折叠成单个标量输出。
//!
//! 与单轴族（[`crate::pointwise`]）的差别只在输出形状：归约消掉输入轴 `n`，输出轴 `o` 恒为 1，
//! 因此层的声明形状是标量 [`Shape`]（空轴）。下界仍由归约算子的语义表推出，层不手写公式。

use diagnostic::{Code, Error, Result};
use tensor::Shape;

use crate::{Bounds, BoundsResult, Contract, ContractSpec};

/// 归约层共享内核：一份已校验规格 + 输入元素数。
#[derive(Debug, Clone, PartialEq, Eq)]
struct ReduceCore {
    spec: ContractSpec,
    len: usize,
}

impl ReduceCore {
    /// 构造：输入元素数必须非零；算子名由具体层给出，构造期经 [`ContractSpec::new`] 校验。
    fn new(name: impl Into<String>, op: &'static str, len: usize) -> Result<Self> {
        if len == 0 {
            return Err(Error::new(
                Code::Invalid,
                format!("归约层输入元素数必须非零（len = {len}）"),
            )
            .with("nn::ReduceCore::new"));
        }
        // 输出轴恒为 1：声明形状是标量，与归约骨架 `n,o->o` 的输出一致。
        let spec = ContractSpec::new(name, Shape::new([]), vec![op.to_string()])?;
        Ok(Self { spec, len })
    }

    fn name(&self) -> &str {
        self.spec.name()
    }

    fn output(&self) -> &Shape {
        self.spec.output()
    }

    fn spca(&self) -> &ContractSpec {
        &self.spec
    }

    /// 输入元素数（被规约的轴 `n` 长度）。
    fn len(&self) -> usize {
        self.len
    }

    fn bounds(&self) -> BoundsResult {
        let op = self
            .spec
            .ops()
            .first()
            .map(String::as_str)
            .unwrap_or_default();
        Bounds::of_context(op, self.len, 1).map(Some)
    }
}

/// 为一个已显式声明的归约层绑定算子名、构造器、访问器与 [`Contract`] 委托。
///
/// 类型声明留在源码里（架构门禁按 `pub struct` 行核对）；本宏只展开构造与委托。
macro_rules! reduce_delegate {
    ($ty:ident, $op:literal) => {
        impl $ty {
            /// 本层绑定的算子名。
            pub const OP: &str = $op;

            /// 由层名与输入元素数构造；元素数为零、或算子未在 `op` 注册时返回说明。
            pub fn new(name: impl Into<String>, len: usize) -> Result<Self> {
                Ok(Self {
                    inner: ReduceCore::new(name, Self::OP, len)?,
                })
            }

            /// 输入元素数（被规约的轴 `n` 长度）。
            pub fn len(&self) -> usize {
                self.inner.len()
            }

            /// 是否为空：构造期已拒绝零长度，因此恒为 `false`。
            pub fn is_empty(&self) -> bool {
                self.inner.len() == 0
            }
        }

        impl Contract for $ty {
            fn name(&self) -> &str {
                self.inner.name()
            }

            fn output(&self) -> &Shape {
                self.inner.output()
            }

            fn spca(&self) -> &ContractSpec {
                self.inner.spca()
            }

            fn bounds(&self) -> BoundsResult {
                self.inner.bounds()
            }
        }
    };
}

/// 求和归约：`y = Σ a[0..n]`，输出单个标量。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReduceSum {
    inner: ReduceCore,
}
reduce_delegate!(ReduceSum, "reduce_sum_f32");

/// 最大值归约：`y = max(a[0..n])`，输出单个标量。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReduceMax {
    inner: ReduceCore,
}
reduce_delegate!(ReduceMax, "reduce_max_f32");

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_reduction_emits_a_scalar_shape() {
        let sum = ReduceSum::new("loss.sum", 128).expect("合法");
        assert_eq!(sum.output().dims(), [], "归约输出是标量");
        assert_eq!(sum.spca().ops(), [ReduceSum::OP]);
        let floor = sum.bounds().expect("合法").expect("有算子");
        assert_eq!(
            floor,
            Bounds::of_context(ReduceSum::OP, 128, 1).expect("同一份语义")
        );
        // 读侧覆盖整个输入，写侧只有一个标量。
        assert!(floor.read() > floor.write());
    }

    #[test]
    fn a_max_reduction_binds_its_operator() {
        let max = ReduceMax::new("pool.max", 64).expect("合法");
        assert_eq!(max.output().dims(), []);
        assert_eq!(max.spca().ops(), [ReduceMax::OP]);
        assert!(max.bounds().expect("合法").is_some());
    }

    #[test]
    fn a_zero_length_reduction_is_rejected() {
        assert_eq!(ReduceSum::new("x", 0).unwrap_err().code(), Code::Invalid);
        assert_eq!(ReduceMax::new("x", 0).unwrap_err().code(), Code::Invalid);
    }
}
