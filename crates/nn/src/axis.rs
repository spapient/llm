//! 单轴族与归约族层共享的规格内核与委托宏。

use diagnostic::{Code, Error, Result};
use tensor::Shape;

use crate::{Bounds, BoundsResult, ContractSpec, ParamRole};

/// 层声明的输出形状口径：单轴族透传一条轴，归约族折叠成标量。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum OutputShape {
    /// 输出与输入等长的一条轴 `[len]`。
    Single,
    /// 输出标量（空轴 `[]`）。
    Scalar,
}

/// 一份已校验规格 + 元素数，被单轴族与归约族的具体层包裹。
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct AxisCore {
    spec: ContractSpec,
    len: usize,
}

impl AxisCore {
    /// 构造：元素数必须非零；输出形状口径由具体层给出，算子名由层绑定，构造期经 [`ContractSpec::new`]
    /// 校验。
    ///
    /// # Errors
    ///
    /// 元素数为零（[`Code::Invalid`]）。`kind` 只用于诊断文案点名族别。
    pub(crate) fn new(
        name: impl Into<String>,
        op: &'static str,
        len: usize,
        output: OutputShape,
        kind: &str,
        role: ParamRole,
    ) -> Result<Self> {
        if len == 0 {
            return Err(Error::new(
                Code::Invalid,
                format!("{kind}元素数必须非零（len = {len}）"),
            )
            .with("nn::AxisCore::new"));
        }
        let shape = match output {
            OutputShape::Single => Shape::new([len]),
            OutputShape::Scalar => Shape::new([]),
        };
        let spec = ContractSpec::new_with_role(name, shape, vec![op.to_string()], role)?;
        Ok(Self { spec, len })
    }

    pub(crate) fn name(&self) -> &str {
        self.spec.name()
    }

    pub(crate) fn output(&self) -> &Shape {
        self.spec.output()
    }

    pub(crate) fn spca(&self) -> &ContractSpec {
        &self.spec
    }

    /// 元素数（被处理 / 被归约的轴长度）。
    pub(crate) fn len(&self) -> usize {
        self.len
    }

    /// 本层下界：单轴/归约算子的第二形参被骨架忽略，补 `1` 只满足 `shape_ctx` 签名，不参与代价。
    pub(crate) fn bounds(&self) -> BoundsResult {
        let op = self
            .spec
            .ops()
            .first()
            .map(String::as_str)
            .unwrap_or_default();
        Bounds::of_context(op, self.len, 1).map(Some)
    }
}

/// 为一个已显式声明的层绑定算子名、构造器、访问器与 [`Contract`] 委托。
#[doc(hidden)]
#[macro_export]
macro_rules! layer_delegate {
    ($ty:ident, $op:literal, $kind:literal, $shape:expr, $role:expr) => {
        impl $ty {
            /// 本层绑定的算子名。
            pub const OP: &str = $op;

            /// 由层名与元素数构造；元素数为零、或算子未在 `op` 注册时返回说明。
            pub fn new(name: impl Into<String>, len: usize) -> diagnostic::Result<Self> {
                Ok(Self {
                    inner: $crate::axis::AxisCore::new(name, Self::OP, len, $shape, $kind, $role)?,
                })
            }

            /// 元素数（被处理 / 被归约的轴长度）。
            pub fn len(&self) -> usize {
                self.inner.len()
            }

            /// 是否为空：构造期已拒绝零长度，因此恒为 `false`。
            pub fn is_empty(&self) -> bool {
                self.inner.len() == 0
            }
        }

        impl $crate::Contract for $ty {
            fn name(&self) -> &str {
                self.inner.name()
            }

            fn output(&self) -> &$crate::Shape {
                self.inner.output()
            }

            fn spca(&self) -> &$crate::ContractSpec {
                self.inner.spca()
            }

            fn bounds(&self) -> $crate::BoundsResult {
                self.inner.bounds()
            }
        }
    };
}

/// 对"持有 `spec: ContractSpec` 字段"的层，把层名/输出形状/规格三件委托收口一份；
/// 各层仍在自己的 `Contract` impl 里补类型化访问器与下界。
#[doc(hidden)]
#[macro_export]
macro_rules! spec_accessors {
    () => {
        fn name(&self) -> &str {
            self.spec.name()
        }

        fn output(&self) -> &$crate::Shape {
            self.spec.output()
        }

        fn spca(&self) -> &$crate::ContractSpec {
            &self.spec
        }
    };
}
