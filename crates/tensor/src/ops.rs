//! 逐元素映射、二元运算与归约；全部直接消费逻辑序迭代器。

use core::ops::{Add, Div, Mul, Sub};

use diagnostic::Result;

use crate::element::{Element, NumericElement, OrderedElement};
use crate::{Owned, Rank, Store, Tensor, invalid};

/// 绝对值与受溢出保护的绝对差比较。
pub trait Abs: Default {
    /// 返回非负绝对值；有符号整数最小值饱和到最大值。
    fn abs(self) -> Self;

    /// 判断 `|self - rhs| <= eps`，整数减法不得溢出。
    fn abs_diff_le(self, rhs: Self, eps: Self) -> bool;
}

macro_rules! float_abs {
    ($($scalar:ty),+ $(,)?) => {
        $(
            impl Abs for $scalar {
                fn abs(self) -> Self {
                    if self < Self::default() { -self } else { self }
                }

                fn abs_diff_le(self, rhs: Self, eps: Self) -> bool {
                    (self - rhs).abs() <= eps
                }
            }
        )+
    };
}

macro_rules! signed_abs {
    ($($scalar:ty),+ $(,)?) => {
        $(
            impl Abs for $scalar {
                fn abs(self) -> Self {
                    self.saturating_abs()
                }

                fn abs_diff_le(self, rhs: Self, eps: Self) -> bool {
                    eps >= 0
                        && (self as i128 - rhs as i128).unsigned_abs() <= eps as u128
                }
            }
        )+
    };
}

macro_rules! unsigned_abs {
    ($($scalar:ty),+ $(,)?) => {
        $(
            impl Abs for $scalar {
                fn abs(self) -> Self {
                    self
                }

                fn abs_diff_le(self, rhs: Self, eps: Self) -> bool {
                    self.abs_diff(rhs) <= eps
                }
            }
        )+
    };
}

float_abs!(f32, f64, half::f16, half::bf16);
signed_abs!(i8, i16, i32, i64);
unsigned_abs!(u8, u16, u32, u64);

impl<B, R, K> Tensor<B, R, K>
where
    R: Rank,
    K: Element,
    B: Store<K>,
{
    /// 逐元素映射，可改变元素类型并保留原秩见证。
    pub fn map<O, F>(&self, f: F) -> Tensor<Owned<O>, R, O>
    where
        O: Element,
        F: FnMut(K) -> O,
    {
        self.respell(self.iter().map(f).collect())
    }

    /// 对两个同形张量按逻辑序执行二元映射。
    ///
    /// 两侧存储和秩见证可以不同，输出元素类型也可以不同；只有形状必须一致。
    pub fn zip_with<B2, R2, O, F>(
        &self,
        rhs: &Tensor<B2, R2, K>,
        mut op: F,
    ) -> Result<Tensor<Owned<O>, R, O>>
    where
        B2: Store<K>,
        R2: Rank,
        O: Element,
        F: FnMut(K, K) -> O,
    {
        if self.dims() != rhs.dims() {
            return Err(invalid(
                format!("形状 {:?} 与 {:?} 不一致", self.dims(), rhs.dims()),
                "tensor::Tensor::zip_with",
            ));
        }
        let values = self
            .iter()
            .zip(rhs.iter())
            .map(|(left, right)| op(left, right))
            .collect();
        Ok(self.respell(values))
    }

    /// 逐元素加法。
    pub fn add<B2: Store<K>, R2: Rank>(
        &self,
        rhs: &Tensor<B2, R2, K>,
    ) -> Result<Tensor<Owned<K>, R, K>>
    where
        K: NumericElement + Add<Output = K>,
    {
        self.zip_with(rhs, Add::add)
    }

    /// 逐元素减法。
    pub fn sub<B2: Store<K>, R2: Rank>(
        &self,
        rhs: &Tensor<B2, R2, K>,
    ) -> Result<Tensor<Owned<K>, R, K>>
    where
        K: NumericElement + Sub<Output = K>,
    {
        self.zip_with(rhs, Sub::sub)
    }

    /// 逐元素乘法。
    pub fn mul<B2: Store<K>, R2: Rank>(
        &self,
        rhs: &Tensor<B2, R2, K>,
    ) -> Result<Tensor<Owned<K>, R, K>>
    where
        K: NumericElement + Mul<Output = K>,
    {
        self.zip_with(rhs, Mul::mul)
    }

    /// 逐元素除法，除零沿用标量语义。
    pub fn div<B2: Store<K>, R2: Rank>(
        &self,
        rhs: &Tensor<B2, R2, K>,
    ) -> Result<Tensor<Owned<K>, R, K>>
    where
        K: NumericElement + Div<Output = K>,
    {
        self.zip_with(rhs, Div::div)
    }

    /// 加标量。
    pub fn add_scalar(&self, scalar: K) -> Tensor<Owned<K>, R, K>
    where
        K: NumericElement + Add<Output = K>,
    {
        self.map(|value| value + scalar)
    }

    /// 乘标量。
    pub fn mul_scalar(&self, scalar: K) -> Tensor<Owned<K>, R, K>
    where
        K: NumericElement + Mul<Output = K>,
    {
        self.map(|value| value * scalar)
    }

    /// 全元素求和；空张量返回 `None`。
    pub fn sum(&self) -> Option<K>
    where
        K: NumericElement + Add<Output = K>,
    {
        if self.numel() == 0 {
            return None;
        }
        Some(self.iter().fold(K::default(), Add::add))
    }

    /// 全局最大元素；空张量返回 `None`。
    pub fn max(&self) -> Option<K>
    where
        K: OrderedElement,
    {
        self.iter().reduce(|a, b| if b > a { b } else { a })
    }

    /// 全局最小元素；空张量返回 `None`。
    pub fn min(&self) -> Option<K>
    where
        K: OrderedElement,
    {
        self.iter().reduce(|a, b| if b < a { b } else { a })
    }

    /// 全局最大元素的逻辑序位置，并列时取最先出现者。
    pub fn argmax(&self) -> Option<usize>
    where
        K: OrderedElement,
    {
        self.extreme_position(|a, b| b > a)
    }

    /// 全局最小元素的逻辑序位置，并列时取最先出现者。
    pub fn argmin(&self) -> Option<usize>
    where
        K: OrderedElement,
    {
        self.extreme_position(|a, b| b < a)
    }

    /// 共享极值位置实现，严格比较确保并列保留首个位置。
    fn extreme_position(&self, replace: impl Fn(K, K) -> bool) -> Option<usize>
    where
        K: OrderedElement,
    {
        self.iter()
            .enumerate()
            .reduce(|(best_index, best), (index, value)| {
                if replace(best, value) {
                    (index, value)
                } else {
                    (best_index, best)
                }
            })
            .map(|(index, _)| index)
    }

    /// 判断同形张量逐元素绝对差是否不超过 `eps`，不分配临时缓冲。
    pub fn all_close<B2: Store<K>, R2: Rank>(&self, rhs: &Tensor<B2, R2, K>, eps: K) -> bool
    where
        K: OrderedElement + Abs,
    {
        self.dims() == rhs.dims()
            && self
                .iter()
                .zip(rhs.iter())
                .all(|(left, right)| left.abs_diff_le(right, eps))
    }
}
