//! 逐元素二元骨架：`y[n] = op(a[n], b[n])`。

use crate::formula::FormulaOwner;
use crate::kernel::scalar::{FloatSlice, FloatSliceMut, Scalar};
use crate::model::{Buf, Cost, Sample, ShapeCtx, Values};
use diagnostic::{Code, Error};
use tensor::View;

// 单轴定义与形状投影复用一元骨架（与 dequant / softmax / rmsnorm / rope / scan 同一模式）。
pub use crate::unary::{AXES, project_shape};

/// 逐元素二元算子随语义变化的那部分。
///
/// 逐元素运算泛型到浮点标量档 `K: Scalar`（ADR 0054 §1 的「元素档 K」）：同一份闭包
/// 服务四档浮点。对拍通道归一 f32，[`apply_f32`](Self::apply_f32) 即 `apply::<f32>` 的默认别名。
pub trait Binary: FormulaOwner {
    /// 逐元素运算：两个输入值，一个输出值，泛型到浮点标量档 `K`。
    fn apply<K: Scalar>(x: K, y: K) -> K;

    /// f32 标量核：对拍通道归一 f32 的入口，默认即 [`apply`](Self::apply) 的 f32 单态化。
    fn apply_f32(x: f32, y: f32) -> f32 {
        Self::apply(x, y)
    }

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

/// 标量参考：逐元素应用 f32 标量核 [`Binary::apply_f32`]，不依赖任何设备。
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
        .map(|(x, y)| B::apply_f32(*x, *y))
        .collect())
}

/// 内核本体：结果写进调用方给的缓冲，泛型到浮点标量档 `K`，不自带分配。
///
/// # Errors
///
/// 两个输入等长、且与输出等长，否则返回说明。
pub fn forward<K: Scalar, B: Binary>(
    a: View<'_, K>,
    b: View<'_, K>,
    out: &mut [K],
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

/// 运行时精度入口：一次 match 把两份 tagged 输入与输出分派到对应档的单态化 [`forward`]。
///
/// # Errors
///
/// 三者档不一致，或缓冲长度不一致时返回说明。
pub fn dispatch<B: Binary>(
    a: FloatSlice<'_>,
    b: FloatSlice<'_>,
    out: FloatSliceMut<'_>,
) -> diagnostic::Result<()> {
    match (a, b, out) {
        (FloatSlice::F32(a), FloatSlice::F32(b), FloatSliceMut::F32(o)) => {
            forward::<f32, B>(View::one(a), View::one(b), o)
        }
        (FloatSlice::F64(a), FloatSlice::F64(b), FloatSliceMut::F64(o)) => {
            forward::<f64, B>(View::one(a), View::one(b), o)
        }
        (FloatSlice::F16(a), FloatSlice::F16(b), FloatSliceMut::F16(o)) => {
            forward::<half::f16, B>(View::one(a), View::one(b), o)
        }
        (FloatSlice::Bf16(a), FloatSlice::Bf16(b), FloatSliceMut::Bf16(o)) => {
            forward::<half::bf16, B>(View::one(a), View::one(b), o)
        }
        _ => Err(Error::new(Code::Invalid, "输入与输出精度档不一致")),
    }
}

/// 生成一份 f32 二元逐元素算子的公开面：标记类型、[`Binary`] 实现、[`annotation::model`] 别名、
/// `forward` 入口与测试挂载。`add` / `mul` / `residual_add` 只差一行 `apply`（可选样本夹具）；
/// 新增一种二元运算加一次调用即可。
#[macro_export]
macro_rules! binary_f32 {
    (
        $name:ident, $alias:ident, $op:tt, $kind:ident,
        $apply:expr, $diff:literal
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
            fn apply<K: $crate::kernel::scalar::Scalar>(x: K, y: K) -> K {
                let apply: fn(K, K) -> K = $apply;
                apply(x, y)
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
            $diff,
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
            $crate::elementwise::forward::<f32, $name>(a, b, out)
        }

        #[cfg(test)]
        mod tests {
            use super::*;
            $crate::elementwise::test_support::$kind!($name, forward);
        }
    };
    // 缺省可微：既有算子沿用 `true`；不可微调子在显式位传 `false`。
    (
        $name:ident, $alias:ident, $op:tt, $kind:ident,
        $apply:expr
        $(, $input_a:expr, $input_b:expr)?
    ) => {
        $crate::binary_f32! {
            $name, $alias, $op, $kind, $apply, true $(, $input_a, $input_b)?
        }
    };
}

#[cfg(test)]
mod test_support {
    include!("../../tests/kernel_elementwise_test_support.rs");
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
        |len| crate::model::linear(len, 0.5, 0.0),
        |len| crate::model::linear(len, 0.25, 0.0)
    );
}

pub mod remainder_f32 {
    //! 逐元素余数：`y[n] = a[n] % b[n]`（fmod：商向零截断，结果符号同被除数）。

    crate::binary_f32!(
        Remainder,
        RemainderF32,
        "remainder_f32",
        elementwise_pair_tests,
        |x, y| x % y,
        false
    );
}
