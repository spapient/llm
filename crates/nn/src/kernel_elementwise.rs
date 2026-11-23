//! 逐元素二元骨架：`y[n] = op(a[n], b[n])`。

use crate::formula::FormulaOwner;
use crate::model::{Buf, Cost, Sample, ShapeCtx, Values};
use diagnostic::{Code, Error};
use tensor::{View, ViewMut};

// 单轴定义与形状投影复用一元骨架（与 dequant / softmax / rmsnorm / rope / scan 同一模式）。
pub use crate::unary::{AXES, project_shape};

/// 逐元素二元算子随语义变化的那部分。
pub trait Binary: FormulaOwner {
    /// 逐元素运算：两个输入值，一个输出值。
    fn apply(x: f32, y: f32) -> f32;

    /// 第一路输入的确定性取值：默认线性夹具 `−1.0 + 0.5·i`；需要不同取值时覆盖。
    fn input_a(len: usize) -> Vec<f32> {
        crate::model::linear(len, 0.5, -1.0)
    }

    /// 第二路输入的确定性取值：默认线性夹具 `0.5 + 0.25·i`；需要不同取值时覆盖。
    fn input_b(len: usize) -> Vec<f32> {
        crate::model::linear(len, 0.25, 0.5)
    }
}

/// 标量代价：完全由表达式与操作数位宽推出，本函数不含流量常数。
pub fn cost<B: Binary>(shape: &ShapeCtx) -> Result<Cost, Error> {
    B::formula()?
        .cost(shape.extents())
        .map_err(|error| error.with("nn::elementwise::cost"))
}

/// 确定性样本：第一个参数是元素数（轴 `n`），第二个参数逐元素算子不用。
pub fn sample<B: Binary>(len: usize, _: usize) -> Sample {
    Sample {
        m: len,
        n: len,
        scalars: vec![("N", len as i64)],
        bufs: vec![
            Buf {
                name: "a",
                values: Values::F32(B::input_a(len)),
            },
            Buf {
                name: "b",
                values: Values::F32(B::input_b(len)),
            },
            Buf {
                name: "y",
                values: Values::F32(vec![0.0; len]),
            },
        ],
    }
}

/// 标量参考：逐元素应用 [`Binary::apply`]，不依赖任何设备。
pub fn reference<B: Binary>(sample: &Sample) -> diagnostic::Result<Vec<f32>> {
    let a = sample.f32("a")?;
    let b = sample.f32("b")?;
    if a.len() != b.len() {
        return Err(Error::new(
            Code::Invalid,
            format!("两个输入长度不一致：{} 与 {}", a.len(), b.len()),
        ));
    }
    Ok(a.iter()
        .zip(b.iter())
        .map(|(x, y)| B::apply(*x, *y))
        .collect())
}

/// 内核本体：结果写进调用方给的缓冲，不自带分配。
///
/// # Errors
///
/// 两个输入等长、且与输出等长，否则返回说明。
pub fn forward<B: Binary>(
    a: View<'_, f32>,
    b: View<'_, f32>,
    out: &mut [f32],
) -> diagnostic::Result<()> {
    let a = a.as_slice();
    let b = b.as_slice();
    if a.len() != b.len() || a.len() != out.len() {
        return Err(Error::new(
            Code::Invalid,
            format!(
                "缓冲长度不一致：a {}、b {}、out {}",
                a.len(),
                b.len(),
                out.len()
            ),
        ));
    }
    // 三个切片 zip：连续内存、无下标边界检查，编译器最容易自动向量化。
    for ((out, x), y) in out.iter_mut().zip(a).zip(b) {
        *out = B::apply(*x, *y);
    }
    Ok(())
}

/// 生成一份 f32 二元逐元素算子的公开面：标记类型、[`Binary`] 实现、[`annotation::model`] 别名、
/// `forward` 入口与测试挂载。`add` / `mul` / `residual_add` 只差一行 `apply`（可选样本夹具）；
/// 新增一种二元运算加一次调用即可。
#[macro_export]
macro_rules! binary_f32 {
    (
        $name:ident, $alias:ident, $op:tt, $kind:ident,
        $apply:expr
        $(, $input_a:expr, $input_b:expr)?
    ) => {
        #[doc = concat!("`", $op, "` 的标记类型：逐元素二元运算见 `apply`，骨架见 `crate::elementwise`。")]
        pub struct $name;

        $crate::formula_owner!(
            $name,
            "n,n,n->n",
            $crate::einsum::f32_binary_operands(),
            "nn::elementwise::formula"
        );

        impl $crate::elementwise::Binary for $name {
            fn apply(x: f32, y: f32) -> f32 {
                const APPLY: fn(f32, f32) -> f32 = $apply;
                APPLY(x, y)
            }

            $( fn input_a(len: usize) -> Vec<f32> {
                const INPUT: fn(usize) -> Vec<f32> = $input_a;
                INPUT(len)
            }

            fn input_b(len: usize) -> Vec<f32> {
                const INPUT: fn(usize) -> Vec<f32> = $input_b;
                INPUT(len)
            } )?
        }

        // 基座契约（ADR 0050）：只转发族骨架，不写第二份算术。
        $crate::operator_via!(
            $name,
            $op,
            2,
            (4096, 4096),
            true,
            super,
            $crate::elementwise::AXES
        );

        #[annotation::model(name = $op, shape = [4096, 4096], dtype = "f32", skeleton = "elementwise")]
        #[doc = concat!("`", $op, "` 的语义别名。")]
        pub type $alias = $name;

        /// 内核入口：结果写进调用方给的缓冲。
        ///
        /// # Errors
        ///
        /// 三个缓冲等长，否则返回说明。
        pub fn forward(a: tensor::View<'_, f32>, b: tensor::View<'_, f32>, out: &mut [f32]) -> diagnostic::Result<()> {
            $crate::elementwise::forward::<$name>(a, b, out)
        }

        #[cfg(test)]
        mod tests {
            use super::*;
            $crate::elementwise::test_support::$kind!($name, forward);
        }
    };
}

#[cfg(test)]
mod test_support {
    include!("../tests/kernel_elementwise_test_support.rs");
}

// 声明式算子表：同一副二元骨架的七个算子，每个算子一个 `pub mod` 保持 `nn::<name>::forward`
// 路径不变（`forward` 同名，必须隔在各自的模块里）。新增二元运算在此补一段。

pub mod add_f32 {
    //! 逐元素加法：`y[n] = a[n] + b[n]`。

    crate::binary_f32!(Add, AddF32, "add_f32", elementwise_binary_tests, |x, y| x
        + y);
}

pub mod sub_f32 {
    //! 逐元素减法：`y[n] = a[n] - b[n]`。

    crate::binary_f32!(Sub, SubF32, "sub_f32", elementwise_binary_tests, |x, y| x
        - y);
}

pub mod mul_f32 {
    //! 逐元素乘法：`y[n] = a[n] * b[n]`。

    crate::binary_f32!(Mul, MulF32, "mul_f32", elementwise_binary_tests, |x, y| x
        * y);
}

pub mod div_f32 {
    //! 逐元素除法：`y[n] = a[n] / b[n]`。

    crate::binary_f32!(Div, DivF32, "div_f32", elementwise_div_tests, |x, y| x / y);
}

pub mod max_f32 {
    //! 逐元素取大：`y[n] = max(a[n], b[n])`。

    crate::binary_f32!(Max, MaxF32, "max_f32", elementwise_binary_tests, |x, y| x
        .max(y));
}

pub mod min_f32 {
    //! 逐元素取小：`y[n] = min(a[n], b[n])`。

    crate::binary_f32!(Min, MinF32, "min_f32", elementwise_binary_tests, |x, y| x
        .min(y));
}

pub mod residual_add_f32 {
    //! 残差相加：`y[n] = a[n] + b[n]`。

    crate::binary_f32!(
        Residual,
        ResidualAddF32,
        "residual_add_f32",
        elementwise_binary_tests,
        |x, y| x + y,
        |len| (0..len).map(|i| (i as f32) * 0.5).collect(),
        |len| (0..len).map(|i| (i as f32) * 0.25).collect()
    );
}
