//! 浮点标量闭集：一份算术骨架为 `f32` / `f64` / `half::f16` / `half::bf16` 实现。

use core::fmt::Debug;

/// 浮点标量闭集：`f32` / `f64` / `half::f16` / `half::bf16` 满足；`nn` 的脉冲 / 合成 / 决策头复用本 trait。
pub trait Scalar:
    Copy
    + PartialEq
    + PartialOrd
    + Debug
    + core::ops::Add<Output = Self>
    + core::ops::Sub<Output = Self>
    + core::ops::Mul<Output = Self>
    + core::ops::Div<Output = Self>
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
    |v| v as f64,
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
