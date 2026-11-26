//! 逐元素映射、二元运算与归约；全部直接消费逻辑序迭代器。

use core::ops::{Add, Div, Mul, Sub};

use diagnostic::Result;

use crate::{Element, NumericElement, OrderedElement};
use crate::{Owned, Rank, Store, Tensor, invalid};

/// 绝对值与受溢出保护的绝对差比较。
pub trait Abs: Default {
    /// 返回非负绝对值；有符号整数最小值饱和到最大值。
    #[must_use]
    fn abs(self) -> Self;

    /// 判断 `|self - rhs| <= tolerance`，整数减法不得溢出。
    fn abs_diff_le(self, rhs: Self, tolerance: Self) -> bool;
}

macro_rules! float_abs {
    ($($scalar:ty),+ $(,)?) => {
        $(
            impl Abs for $scalar {
                fn abs(self) -> Self {
                    if self < Self::default() { -self } else { self }
                }

                fn abs_diff_le(self, rhs: Self, tolerance: Self) -> bool {
                    (self - rhs).abs() <= tolerance
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

                // 守卫 `tolerance >= 0` 排除负值；去符号到 u128 受此不变量约束，clippy 不可见该守卫。
                #[allow(clippy::cast_sign_loss)]
                fn abs_diff_le(self, rhs: Self, tolerance: Self) -> bool {
                    tolerance >= 0
                        && (i128::from(self) - i128::from(rhs)).unsigned_abs() <= tolerance as u128
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

                fn abs_diff_le(self, rhs: Self, tolerance: Self) -> bool {
                    self.abs_diff(rhs) <= tolerance
                }
            }
        )+
    };
}

float_abs!(f32, f64, half::f16, half::bf16);
signed_abs!(i8, i16, i32, i64);
unsigned_abs!(u8, u16, u32, u64);

macro_rules! elementwise_binary {
    ($name:ident, $trait:ident, $operation:ident, $doc:literal) => {
        #[doc = $doc]
        #[doc = "# Errors\n\n两侧形状不一致时返回形状校验错误。"]
        pub fn $name<B2: Store<K>, R2: Rank>(
            &self,
            rhs: &Tensor<B2, R2, K>,
        ) -> Result<Tensor<Owned<K>, R, K>>
        where
            K: NumericElement + $trait<Output = K>,
        {
            self.zip_with(rhs, $trait::$operation)
        }
    };
}

impl<B, R, K> Tensor<B, R, K>
where
    R: Rank,
    K: Element,
    B: Store<K>,
{
    /// 逐元素映射，可改变元素类型并保留原秩见证。
    pub fn map<O, F>(&self, operation: F) -> Tensor<Owned<O>, R, O>
    where
        O: Element,
        F: FnMut(K) -> O,
    {
        self.owned_with_values(self.iter().map(operation).collect())
    }

    /// 对两个同形张量按逻辑序执行二元映射。
    ///
    /// 两侧的存储、秩见证和元素类型均可不同，输出元素类型也可不同；只有形状必须一致。
    ///
    /// # Errors
    ///
    /// 两侧形状不一致时返回形状校验错误。
    pub fn zip_with<B2, R2, K2, O, F>(
        &self,
        rhs: &Tensor<B2, R2, K2>,
        mut operation: F,
    ) -> Result<Tensor<Owned<O>, R, O>>
    where
        B2: Store<K2>,
        R2: Rank,
        K2: Element,
        O: Element,
        F: FnMut(K, K2) -> O,
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
            .map(|(left, right)| operation(left, right))
            .collect();
        Ok(self.owned_with_values(values))
    }

    elementwise_binary!(add, Add, add, "逐元素加法。");
    elementwise_binary!(sub, Sub, sub, "逐元素减法。");
    elementwise_binary!(mul, Mul, mul, "逐元素乘法。");
    elementwise_binary!(div, Div, div, "逐元素除法，除零沿用标量语义。");

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
    fn extreme_position(&self, should_replace: impl Fn(K, K) -> bool) -> Option<usize>
    where
        K: OrderedElement,
    {
        self.iter()
            .enumerate()
            .reduce(|(best_idx, best), (idx, value)| {
                if should_replace(best, value) {
                    (idx, value)
                } else {
                    (best_idx, best)
                }
            })
            .map(|(idx, _)| idx)
    }

    /// 判断同形张量逐元素绝对差是否不超过 `tolerance`，不分配临时缓冲。
    pub fn all_close<B2: Store<K>, R2: Rank>(&self, rhs: &Tensor<B2, R2, K>, tolerance: K) -> bool
    where
        K: OrderedElement + Abs,
    {
        self.dims() == rhs.dims()
            && self
                .iter()
                .zip(rhs.iter())
                .all(|(left, right)| left.abs_diff_le(right, tolerance))
    }
}
