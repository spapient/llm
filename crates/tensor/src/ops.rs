//! 张量方法面：逐元素算术、归约与统计，按元素能力分组收紧 bound。
//!
//! 这里只放"纯元素、不依赖设备调度"的参考语义：形状代数与布局在 [`crate::layout`]，
//! 矩阵乘 / 激活 / 归一这类设备内核仍走 `nn` 数据驱动注册表（ADR 0053），本层不重复第二份算术。
//!
//! 能力分组即 trait bound：加减乘除要 [`NumericElement`]，取极值要 [`OrderedElement`]；
//! 分组外的元素类型（如 `bool`）在编译期就调不到对应方法，不需要运行期 `supports(dtype)` 开关。

use core::ops::{Add, Div, Mul, Sub};

use diagnostic::Result;

use crate::Tensor;
use crate::element::{Element, NumericElement, OrderedElement};
use crate::invalid;
use crate::rank::Rank;
use crate::store::{Owned, Store};

/// 元素绝对值：标准库数值与 half 标量各有取负，闭一层避免方法 bound 上挂第二份标记。
///
/// 只对 [`Element`](crate::Element) 闭集内的标量实现；外部类型无法接入本层元素集合。
pub trait Abs: Default {
    /// 自身的非负值。
    fn abs(self) -> Self;
}

macro_rules! abs_impl {
    ($($t:ty),+ $(,)?) => {
        $(
            impl Abs for $t {
                fn abs(self) -> Self {
                    if self < Self::default() { -self } else { self }
                }
            }
        )+
    };
}

abs_impl!(f32, f64, half::f16, half::bf16, i8, i16, i32, i64);

// 无符号标量不存在小于 0 的分支：绝对值即自身。
impl Abs for u8 {
    fn abs(self) -> Self {
        self
    }
}
impl Abs for u16 {
    fn abs(self) -> Self {
        self
    }
}
impl Abs for u32 {
    fn abs(self) -> Self {
        self
    }
}
impl Abs for u64 {
    fn abs(self) -> Self {
        self
    }
}

impl<B, R, K> Tensor<B, R, K>
where
    R: Rank,
    K: Element,
    B: Store<K>,
{
    /// 逐元素映射：按逻辑序对每个元素应用 `f`，产出自持连续张量。
    pub fn map<F>(&self, f: F) -> Tensor<Owned<K>, R, K>
    where
        F: Fn(K) -> K,
    {
        let values = self.to_vec().into_iter().map(f).collect();
        self.respell(values)
    }

    /// 逐元素二元：两张量形状逐轴一致，`op(a[i], b[i])` 写入新自持张量。
    ///
    /// 形状不符返回 `Invalid`——不做静默广播（广播是布局动作，走 [`Tensor::broadcast`]）。
    fn binary<O>(&self, rhs: &Self, op: O, marker: &str) -> Result<Tensor<Owned<K>, R, K>>
    where
        O: Fn(K, K) -> K,
    {
        if self.dims() != rhs.dims() {
            return Err(invalid(
                format!("形状 {:?} 与 {:?} 不一致", self.dims(), rhs.dims()),
                marker,
            ));
        }
        let a = self.to_vec();
        let b = rhs.to_vec();
        let values = a.into_iter().zip(b).map(|(x, y)| op(x, y)).collect();
        Ok(self.respell(values))
    }

    /// 逐元素加：`out[i] = a[i] + b[i]`。
    pub fn add(&self, rhs: &Self) -> Result<Tensor<Owned<K>, R, K>>
    where
        K: NumericElement + Add<Output = K>,
    {
        self.binary(rhs, Add::add, "tensor::Tensor::add")
    }

    /// 逐元素减：`out[i] = a[i] - b[i]`。
    pub fn sub(&self, rhs: &Self) -> Result<Tensor<Owned<K>, R, K>>
    where
        K: NumericElement + Sub<Output = K>,
    {
        self.binary(rhs, Sub::sub, "tensor::Tensor::sub")
    }

    /// 逐元素乘：`out[i] = a[i] * b[i]`。
    pub fn mul(&self, rhs: &Self) -> Result<Tensor<Owned<K>, R, K>>
    where
        K: NumericElement + Mul<Output = K>,
    {
        self.binary(rhs, Mul::mul, "tensor::Tensor::mul")
    }

    /// 逐元素除：`out[i] = a[i] / b[i]`；除零口径沿用标量语义。
    pub fn div(&self, rhs: &Self) -> Result<Tensor<Owned<K>, R, K>>
    where
        K: NumericElement + Div<Output = K>,
    {
        self.binary(rhs, Div::div, "tensor::Tensor::div")
    }

    /// 加标量：`out[i] = a[i] + scalar`。
    pub fn add_scalar(&self, scalar: K) -> Tensor<Owned<K>, R, K>
    where
        K: NumericElement + Add<Output = K>,
    {
        self.map(|x| x + scalar)
    }

    /// 乘标量：`out[i] = a[i] * scalar`。
    pub fn mul_scalar(&self, scalar: K) -> Tensor<Owned<K>, R, K>
    where
        K: NumericElement + Mul<Output = K>,
    {
        self.map(|x| x * scalar)
    }

    /// 全元素求和；空张量返回 `None`。
    ///
    /// 累加是串行依赖：下一项依赖前一项的和，无法整块并行归约，这里保留 fold（见仓内循环纪律）。
    pub fn sum(&self) -> Option<K>
    where
        K: NumericElement + Add<Output = K> + Default,
    {
        let values = self.to_vec();
        if values.is_empty() {
            return None;
        }
        Some(values.into_iter().fold(K::default(), Add::add))
    }

    /// 全局最大元素；空张量返回 `None`。
    pub fn max(&self) -> Option<K>
    where
        K: OrderedElement,
    {
        self.to_vec()
            .into_iter()
            .reduce(|a, b| if b >= a { b } else { a })
    }

    /// 全局最小元素；空张量返回 `None`。
    pub fn min(&self) -> Option<K>
    where
        K: OrderedElement,
    {
        self.to_vec()
            .into_iter()
            .reduce(|a, b| if b <= a { b } else { a })
    }

    /// 全局最大元素的逻辑序下标；空张量返回 `None`。并列时取最先出现的位置。
    pub fn argmax(&self) -> Option<usize>
    where
        K: OrderedElement,
    {
        self.to_vec()
            .into_iter()
            .enumerate()
            .reduce(|(i, a), (j, b)| if b >= a { (j, b) } else { (i, a) })
            .map(|(index, _)| index)
    }

    /// 逐元素绝对差不超过 `eps`；形状不一致或元素数不符直接 `false`。
    pub fn all_close(&self, rhs: &Self, eps: K) -> bool
    where
        K: OrderedElement + Sub<Output = K> + Abs,
    {
        if self.dims() != rhs.dims() {
            return false;
        }
        let a = self.to_vec();
        let b = rhs.to_vec();
        a.len() == b.len() && a.into_iter().zip(b).all(|(x, y)| (x - y).abs() <= eps)
    }
}
