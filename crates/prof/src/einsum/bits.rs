//! 元素位宽闭集（bit）：代价估算按位宽计费，`i4` 权重按 nibble 计，不被当成一个 `f32` 或 `u8`。

use crate::atom::Closed;

/// 元素位宽闭集（bit）。
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Bits {
    /// 4 位（打包的 int4 权重）。
    Four,
    /// 8 位。
    Eight,
    /// 16 位（f16 / bf16 / i16）。
    Sixteen,
    /// 32 位（f32 / i32）。
    ThirtyTwo,
    /// 64 位（f64 / i64）。
    SixtyFour,
    /// 2 位（打包的 int2 权重，每字节四个码）。
    Two,
}

/// 与 [`Bits::ALL`] 同序的名字表；下标即判别值，加变体时与 `ALL`、[`WIDTHS`] 三处对齐。
const NAMES: [&str; Bits::ALL.len()] = ["4", "8", "16", "32", "64", "2"];

/// 与 [`Bits::ALL`] 同序的位宽表；下标即判别值。
const WIDTHS: [u32; Bits::ALL.len()] = [4, 8, 16, 32, 64, 2];

/// `Bits` 的词汇表由 [`crate::atom::Closed`] 单点声明，`ALL` / `as_str` / `parse` 由
/// [`crate::inherent_names!`] 以同名固有项转发。
impl Closed for Bits {
    const ALL: &'static [Self] = &[
        Self::Four,
        Self::Eight,
        Self::Sixteen,
        Self::ThirtyTwo,
        Self::SixtyFour,
        Self::Two,
    ];

    fn as_str(self) -> &'static str {
        NAMES[self as usize]
    }

    fn idx(self) -> usize {
        self as usize
    }
}

crate::inherent_names!(Bits);

impl Bits {
    /// 每元素占多少位。
    pub const fn bits(self) -> u32 {
        WIDTHS[self as usize]
    }

    /// 给定元素个数需要多少字节，**向上取整**。
    pub const fn bytes_for(self, elements: u128) -> u128 {
        (elements * self.bits() as u128).div_ceil(8)
    }
}
