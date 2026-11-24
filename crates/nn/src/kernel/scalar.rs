//! 元素精度根与浮点标量闭集：`Precision` 是全算子族共享的乘累加契约，`Scalar` 是其浮点子集。

use core::fmt::Debug;
use tensor::{FloatElement, NumericElement};

/// 全算子族共享的元素精度档：累加器类型 + 乘累加原语（ADR 0054 §1 的「元素档 K」根）。
///
/// 稠密档（`f32` / `f64` / `half::f16` / `half::bf16`）以自身为累加器；`i8` 以 `i32` 防溢出。
/// 骨架族（unary / binary / reduce / matmul）泛型到本 trait 即获多精度执行能力，
/// 无需为每档另写一份算子（不开平行的 `*_f16` 注册族）。打包量化权重走 `gemv::Element`，不在此列。
pub trait Precision: Copy + NumericElement {
    /// 累加器类型（稠密档 = 自身，`i8` = `i32`）。
    type Acc: Copy + Default;
    /// 乘累加：`acc += a * b`。
    fn mac(acc: &mut Self::Acc, a: Self, b: Self);
}

/// 浮点标量闭集：`f32` / `f64` / `half::f16` / `half::bf16` 满足；`nn` 的脉冲 / 合成 / 决策头复用本 trait。
///
/// 元素域收紧到 `FloatElement`：只有封闭浮点标量可进超越函数 / 浮点算术族，
/// 整数与布尔不在其列。作为 [`Precision`] 的浮点子集，一元 / 二元 / 归约骨架泛型到本 trait
/// 即在同一份闭包下服务四个浮点档。
pub trait Scalar:
    Copy
    + PartialEq
    + PartialOrd
    + Debug
    + FloatElement
    + Precision
    + core::ops::Add<Output = Self>
    + core::ops::Sub<Output = Self>
    + core::ops::Mul<Output = Self>
    + core::ops::Div<Output = Self>
    + core::ops::Neg<Output = Self>
    + core::ops::Rem<Output = Self>
{
    /// 加法幺元。
    const ZERO: Self;
    /// 乘法幺元。
    const ONE: Self;
    /// 负无穷（DSA top-k 掩码：`softmax(−∞)=0`，实现真稀疏注意力）。
    const NEG_INFINITY: Self;

    /// 自然指数 `exp(x)`。
    fn exp(self) -> Self;
    /// `exp(x) - 1`（无 `x≈0` 时的灾难性消去）。
    fn exp_m1(self) -> Self;
    /// 平方根。
    fn sqrt(self) -> Self;
    /// 绝对值。
    fn abs(self) -> Self;
    /// 取两数较大者。
    fn max(self, rhs: Self) -> Self;
    /// 取两数较小者。
    fn min(self, rhs: Self) -> Self;
    /// 向下取整（池化/插值的坐标下取整）。
    fn floor(self) -> Self;
    /// 自然对数（自信息 `−ln p` 与交叉熵的唯一来源，不另写近似）。
    fn ln(self) -> Self;
    /// 正弦（复值状态的相位，单位弧度）。
    fn sin(self) -> Self;
    /// 余弦（复值状态的相位，单位弧度）。
    fn cos(self) -> Self;
    /// 由 `f64` 字面量构造（仅用于常量，`f32` 下可能有损）。
    fn from_f64(v: f64) -> Self;
    /// 由 `usize` 构造：形状/索引数学（步长、核尺寸、面积）的通用入口，不绕道 `f64`。
    fn from_usize(v: usize) -> Self;
    /// 取回 `usize`（截断向零）：只用于形状/索引数学，调用方保证值非负且落在 `usize` 范围内。
    fn to_usize(self) -> usize;
    /// 就近取整，半分远离零（`f64::round` 的口径）。
    fn round(self) -> Self;
    /// 向上取整。
    fn ceil(self) -> Self;
    /// 向零截断。
    fn trunc(self) -> Self;
    /// `ln(1 + x)`（无 `x≈0` 时的灾难性消去）。
    fn ln_1p(self) -> Self;
    /// 以 2 为底的指数 `2^x`。
    fn exp2(self) -> Self;
    /// 取回 `f64`。
    fn to_f64(self) -> f64;
}

/// 一份算术骨架为 `f32` / `f64` / `half::f16` / `half::bf16` 四个标量实现，能变的部分只有类型、
/// 零/一/负无穷常量与 `f64`/`usize` 的进出转换式；逐字写四份会把 14 个方法的加/减/乘/除与
/// 超越函数复制成四胞胎（jscpd 的克隆源）。数学函数统一走 [`num_traits::Float`]：`f32`/`f64`
/// 由该 trait 直接委托给原生实现，`half` 类型由 half 自带实现经 `f32` 往返得到，语义与精度由
/// half 侧保证，骨架侧不复制任何一份算术。
macro_rules! impl_scalar {
    ($ty:ty, $zero:expr, $one:expr, $neg_inf:expr, $from_f64:expr, $from_usize:expr, $to_f64:expr, $to_usize:expr) => {
        impl Precision for $ty {
            type Acc = $ty;
            fn mac(acc: &mut Self::Acc, a: Self, b: Self) {
                *acc += a * b;
            }
        }
        impl Scalar for $ty {
            const ZERO: Self = $zero;
            const ONE: Self = $one;
            const NEG_INFINITY: Self = $neg_inf;

            fn exp(self) -> Self {
                num_traits::Float::exp(self)
            }
            fn exp_m1(self) -> Self {
                num_traits::Float::exp_m1(self)
            }
            fn sqrt(self) -> Self {
                num_traits::Float::sqrt(self)
            }
            fn abs(self) -> Self {
                num_traits::Float::abs(self)
            }
            fn max(self, rhs: Self) -> Self {
                num_traits::Float::max(self, rhs)
            }
            fn min(self, rhs: Self) -> Self {
                num_traits::Float::min(self, rhs)
            }
            fn floor(self) -> Self {
                num_traits::Float::floor(self)
            }
            fn ln(self) -> Self {
                num_traits::Float::ln(self)
            }
            fn sin(self) -> Self {
                num_traits::Float::sin(self)
            }
            fn cos(self) -> Self {
                num_traits::Float::cos(self)
            }
            fn from_f64(v: f64) -> Self {
                ($from_f64)(v)
            }
            fn from_usize(v: usize) -> Self {
                ($from_usize)(v)
            }
            fn round(self) -> Self {
                num_traits::Float::round(self)
            }
            fn ceil(self) -> Self {
                num_traits::Float::ceil(self)
            }
            fn trunc(self) -> Self {
                num_traits::Float::trunc(self)
            }
            fn ln_1p(self) -> Self {
                num_traits::Float::ln_1p(self)
            }
            fn exp2(self) -> Self {
                num_traits::Float::exp2(self)
            }
            fn to_f64(self) -> f64 {
                ($to_f64)(self)
            }
            fn to_usize(self) -> usize {
                ($to_usize)(self)
            }
        }
    };
}

impl_scalar!(
    f32,
    0.0_f32,
    1.0_f32,
    f32::NEG_INFINITY,
    |v| v as f32,
    |v| v as f32,
    f64::from,
    |v| v as usize
);
impl_scalar!(
    f64,
    0.0_f64,
    1.0_f64,
    f64::NEG_INFINITY,
    |v| v,
    |v| v as f64,
    |v| v,
    |v| v as usize
);
impl_scalar!(
    half::f16,
    half::f16::ZERO,
    half::f16::ONE,
    half::f16::NEG_INFINITY,
    half::f16::from_f64,
    |v| half::f16::from_f64(v as f64),
    half::f16::to_f64,
    |v| half::f16::to_f64(v) as usize
);
impl_scalar!(
    half::bf16,
    half::bf16::ZERO,
    half::bf16::ONE,
    half::bf16::NEG_INFINITY,
    half::bf16::from_f64,
    |v| half::bf16::from_f64(v as f64),
    half::bf16::to_f64,
    |v| half::bf16::to_f64(v) as usize
);

#[cfg(test)]
mod tests {
    //! 泛型化判据：同一副骨架在 f64 / f16 / bf16 上执行，结果对 f32 参考落在档内精度容差内。
    //! `forward` 仅绑 f32 时这些用例无法编译——它们锁住「Precision 推广到全算子族」的能力增益。
    use super::Scalar;
    use crate::kernel::{elementwise, reduce, unary};
    use tensor::View;

    fn assert_close<K: Scalar>(got: &[K], want: &[f32], rel: f64) {
        for (g, w) in got.iter().zip(want) {
            let (g, w) = (g.to_f64(), f64::from(*w));
            assert!((g - w).abs() <= rel * w.abs().max(1.0), "{g} vs {w} (rel {rel})");
        }
    }

    fn unary_gelu<K: Scalar>(rel: f64) {
        use unary::ops::gelu_f32::Gelu;
        let xs = [-2.0_f64, -0.5, 0.0, 0.7, 1.5, 3.0];
        let a32: Vec<f32> = xs.iter().map(|&v| v as f32).collect();
        let mut want = vec![0.0_f32; xs.len()];
        unary::forward::<f32, Gelu>(View::one(&a32), &mut want).unwrap();
        let ak: Vec<K> = xs.iter().map(|&v| K::from_f64(v)).collect();
        let mut got = vec![K::ZERO; xs.len()];
        unary::forward::<K, Gelu>(View::one(&ak), &mut got).unwrap();
        assert_close(&got, &want, rel);
    }

    fn binary_add<K: Scalar>(rel: f64) {
        use elementwise::add_f32::Add;
        let pairs = [(-1.5_f64, 0.25), (0.5, 2.0), (3.0, -4.0), (0.0, 0.0)];
        let a32: Vec<f32> = pairs.iter().map(|p| p.0 as f32).collect();
        let b32: Vec<f32> = pairs.iter().map(|p| p.1 as f32).collect();
        let mut want = vec![0.0_f32; pairs.len()];
        elementwise::forward::<f32, Add>(View::one(&a32), View::one(&b32), &mut want).unwrap();
        let ak: Vec<K> = pairs.iter().map(|p| K::from_f64(p.0)).collect();
        let bk: Vec<K> = pairs.iter().map(|p| K::from_f64(p.1)).collect();
        let mut got = vec![K::ZERO; pairs.len()];
        elementwise::forward::<K, Add>(View::one(&ak), View::one(&bk), &mut got).unwrap();
        assert_close(&got, &want, rel);
    }

    fn reduce_sum<K: Scalar>(rel: f64) {
        use reduce::reduce_sum_f32::Sum;
        let xs = [-3.0_f64, -1.5, 0.0, 2.5, 4.0];
        let a32: Vec<f32> = xs.iter().map(|&v| v as f32).collect();
        let mut want = vec![0.0_f32; 1];
        reduce::forward::<f32, Sum>(View::one(&a32), &mut want).unwrap();
        let ak: Vec<K> = xs.iter().map(|&v| K::from_f64(v)).collect();
        let mut got = vec![K::ZERO; 1];
        reduce::forward::<K, Sum>(View::one(&ak), &mut got).unwrap();
        assert_close(&got, &want, rel);
    }

    #[test]
    fn unary_runs_in_every_float_tier() {
        unary_gelu::<f64>(1e-6);
        unary_gelu::<half::f16>(2e-2);
        unary_gelu::<half::bf16>(5e-2);
    }

    #[test]
    fn binary_runs_in_every_float_tier() {
        binary_add::<f64>(1e-6);
        binary_add::<half::f16>(2e-2);
        binary_add::<half::bf16>(5e-2);
    }

    #[test]
    fn reduce_runs_in_every_float_tier() {
        reduce_sum::<f64>(1e-6);
        reduce_sum::<half::f16>(2e-2);
        reduce_sum::<half::bf16>(5e-2);
    }
}
