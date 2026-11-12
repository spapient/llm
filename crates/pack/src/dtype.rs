//! 容器线上的元素类型闭集：覆盖布尔、INT1–INT64、F8–F64 的全精度，独立于 `tensor::DType`。
//!
//! 容器能**存**闭集内的任何类型，即使当前 `tensor` 还没有对应元素；能不能被某个内核消费由
//! [`Wire::to_tensor`] 与上层决定。每个类型有稳定的线上 code（`0` 保留，不分配）。

use tensor::DType;

/// 线上元素类型。
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub enum Wire {
    /// 布尔（1 位）。
    Bool,
    /// 无符号整数。
    U1,
    /// 无符号整数。
    U2,
    /// 无符号整数（按位打包）。
    U4,
    /// 无符号整数。
    U8,
    /// 无符号整数。
    U16,
    /// 无符号整数。
    U32,
    /// 无符号整数。
    U64,
    /// 有符号整数。
    I1,
    /// 有符号整数。
    I2,
    /// 有符号整数（按位打包）。
    I4,
    /// 有符号整数。
    I8,
    /// 有符号整数。
    I16,
    /// 有符号整数。
    I32,
    /// 有符号整数。
    I64,
    /// 8 位浮点（1 符号 + 4 指数 + 3 尾数）。
    F8E4M3,
    /// 8 位浮点（1 符号 + 5 指数 + 2 尾数）。
    F8E5M2,
    /// 16 位浮点。
    F16,
    /// bfloat16。
    BF16,
    /// 32 位浮点。
    F32,
    /// 64 位浮点。
    F64,
}

impl Wire {
    /// 全部线上类型，按 code 顺序，供遍历与测试。
    pub const ALL: [Self; 21] = [
        Self::Bool,
        Self::U1,
        Self::U2,
        Self::U4,
        Self::U8,
        Self::U16,
        Self::U32,
        Self::U64,
        Self::I1,
        Self::I2,
        Self::I4,
        Self::I8,
        Self::I16,
        Self::I32,
        Self::I64,
        Self::F8E4M3,
        Self::F8E5M2,
        Self::F16,
        Self::BF16,
        Self::F32,
        Self::F64,
    ];

    /// 稳定线上 code（1..=21）。
    pub fn code(self) -> u16 {
        Self::ALL
            .iter()
            .position(|known| *known == self)
            .expect("Wire 必在 ALL 中") as u16
            + 1
    }

    /// 由 code 还原；`0` 与未知 code 返回 `None`。
    pub fn from_code(code: u16) -> Option<Self> {
        usize::from(code)
            .checked_sub(1)
            .and_then(|index| Self::ALL.get(index).copied())
    }

    /// 单个元素的位宽。
    pub fn bits(self) -> u32 {
        match self {
            Self::Bool | Self::U1 | Self::I1 => 1,
            Self::U2 | Self::I2 => 2,
            Self::U4 | Self::I4 => 4,
            Self::U8 | Self::I8 | Self::F8E4M3 | Self::F8E5M2 => 8,
            Self::U16 | Self::I16 | Self::F16 | Self::BF16 => 16,
            Self::U32 | Self::I32 | Self::F32 => 32,
            Self::U64 | Self::I64 | Self::F64 => 64,
        }
    }

    /// 是否为浮点类型。
    pub fn is_float(self) -> bool {
        matches!(
            self,
            Self::F8E4M3 | Self::F8E5M2 | Self::F16 | Self::BF16 | Self::F32 | Self::F64
        )
    }

    /// 是否为布尔。
    pub fn is_bool(self) -> bool {
        self == Self::Bool
    }

    /// `numel` 个元素需要的字节数（亚字节类型按位向上取整）。
    pub fn element_bytes(self, numel: u64) -> u64 {
        let total = numel.saturating_mul(u64::from(self.bits()));
        total.div_ceil(8)
    }

    /// 映射到当前 `tensor` 的元素类型；没有对应元素时返回 `None`（上层按 Unsupported 处理）。
    ///
    /// `I4` 刻意不映射到 `DType::U4`：有符号与无符号的 4 位语义不同，隐式转换会改语义。
    pub fn to_tensor(self) -> Option<DType> {
        match self {
            Self::F32 => Some(DType::F32),
            Self::F16 => Some(DType::F16),
            Self::BF16 => Some(DType::BF16),
            Self::I8 | Self::U8 => Some(DType::I8),
            Self::U4 => Some(DType::U4),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn codes_are_a_stable_bijection() {
        for wire in Wire::ALL {
            assert_eq!(Wire::from_code(wire.code()), Some(wire));
        }
        assert_eq!(Wire::from_code(0), None);
        assert_eq!(Wire::from_code(99), None);
    }

    #[test]
    fn sub_byte_types_pack_by_bits() {
        assert_eq!(Wire::U4.element_bytes(6), 3);
        assert_eq!(Wire::U1.element_bytes(8), 1);
        assert_eq!(Wire::I2.element_bytes(5), 2);
        assert_eq!(Wire::F32.element_bytes(2), 8);
    }

    #[test]
    fn only_matching_widths_map_to_tensor() {
        assert_eq!(Wire::U8.to_tensor(), Some(DType::I8));
        assert_eq!(Wire::I4.to_tensor(), None, "有符号 4 位不隐式映射");
        assert_eq!(Wire::F64.to_tensor(), None);
        assert_eq!(Wire::F8E4M3.to_tensor(), None);
    }
}
