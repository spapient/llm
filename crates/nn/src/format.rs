//! 权重编码：层的"存储事实"，以及它决定了哪个算子。
//!
//! 在此之前，量化格式被钉死在 [`Linear::OP`] 一个常量上：任何秩 2 的权重条目都按 int4 计价，
//! 哪怕文件里写的是 `F32`。那不是一个"更松的下界"，而是**声明与文件不符**——一份 4096 方阵的
//! f32 权重有 64 MiB，按 int4 报出来只有 10 MiB，差 6.4 倍，而数字看起来仍然漂亮。
//!
//! 所以格式必须是层的事实，而不是全局默认值：
//!
//! - 格式来自权重条目自己（`weights::Entry::dtype`），由 [`WeightFormat::of_dtype`] 翻译；
//! - 格式 → 算子名由 [`WeightFormat::operator`] 给出；**没有**对应算子的格式返回 `None`，
//!   层于是"已声明、无法计价"，而不是拿另一个格式的算子凑一个数；
//! - `U4` 的**分组**不在元素类型里，因此 [`WeightFormat::of_dtype`] 只给出 [`WeightFormat::Packed4`]
//!   （"4 位打包，分组未声明"）；要计价必须显式构造 [`WeightFormat::int4`]，不猜。
//!
//! 目前有算子（也就是有语义表）的格式是 int4（[`WeightFormat::INT4`]）与 f32
//! （[`WeightFormat::F32`]）。f16 / bf16 / i8 的投影算子还不存在，那些层仍然算不出下界——这是
//! 如实报告，不是回归：它们的元素宽度与 f32 不同（16 位 vs 32 位），套上去会把读下界报成两倍。

use std::fmt;

use tensor::DType;

/// 权重编码。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WeightFormat {
    /// 32 位浮点。
    F32,
    /// 16 位浮点。
    F16,
    /// bfloat16。
    Bf16,
    /// 8 位整数（int8 / uint8 的语义由算子决定，本层不猜）。
    I8,
    /// 4 位打包整数，但**没有声明分组**：元素类型本身不携带"多少个元素共享一个缩放因子"。
    Packed4,
    /// 4 位打包整数 + 每 `group` 个元素一个缩放因子。
    Int4 {
        /// 分组宽度（`m` 必须是它的整数倍）。
        group: usize,
    },
}

impl WeightFormat {
    /// 量化侧的默认格式：int4 + 32 元素分组，即 [`crate::Linear::OP`] 的输入格式。
    ///
    /// 这只是"声明层默认用什么"的便利常量；它是否与算子语义表里的分组一致，由
    /// [`WeightFormat::operator`] 与测试共同保证。
    pub const INT4: Self = Self::Int4 { group: 32 };

    /// 由权重的元素类型推出格式。
    ///
    /// `U4` 只给出 [`WeightFormat::Packed4`]：分组没有出现在元素类型里，任何具体数字都只能是
    /// 猜的。需要计价时用 [`WeightFormat::int4`] 明确声明分组。
    #[must_use]
    pub fn of_dtype(dtype: DType) -> Option<Self> {
        match dtype {
            DType::F32 => Some(Self::F32),
            DType::F16 => Some(Self::F16),
            DType::BF16 => Some(Self::Bf16),
            DType::I8 => Some(Self::I8),
            DType::U4 => Some(Self::Packed4),
            // 闭集里的其他类型不是权重存储格式：调用方按未覆盖处理，不强行归类。
            _ => None,
        }
    }

    /// 显式声明 4 位量化的分组。
    pub const fn int4(group: usize) -> Self {
        Self::Int4 { group }
    }

    /// 这种编码在内存里的元素类型（[`WeightFormat::Packed4`] 与 [`WeightFormat::Int4`] 都是 `U4`）。
    pub fn dtype(self) -> DType {
        match self {
            Self::F32 => DType::F32,
            Self::F16 => DType::F16,
            Self::Bf16 => DType::BF16,
            Self::I8 => DType::I8,
            Self::Packed4 | Self::Int4 { .. } => DType::U4,
        }
    }

    /// 本格式对应的已声明算子；`None` = 还没有这个格式的算子，因此这一层无法计价。
    ///
    /// int4 的分组从算子语义表里读（`op::model(Linear::OP).group`），而不是在这里再写一个
    /// 32：那会变成两条可能互相矛盾的"事实"。格式的分组与算子声明的分组不一致时返回 `None`，
    /// 而不是把一个不同分组的算子套上去。
    pub fn operator(self) -> Option<&'static str> {
        match self {
            Self::Int4 { group } if Some(group) == int4_group() => Some(crate::Linear::OP),
            // 稠密权重走另一条算子：它的求和表达式没有标度轴，代价按 32 位元素算。
            // 两者只是同一个骨架的两种输入格式，不能互相顶替。
            Self::F32 => Some(crate::Linear::DENSE),
            _ => None,
        }
    }

    /// 为什么这种编码取不到算子：一句话说清"缺什么"，而不是只说"没有"。
    ///
    /// 与 [`WeightFormat::operator`] 严格互补：取得到算子时它的理由不会被调用，取不到时才拼出
    /// 诊断。三类理由分别对应三种真实文件：
    ///
    /// - 元素类型根本没有算子（f16 / bf16 / i8）——16 位的元素宽度与 f32 不同，套用稠密算子
    ///   会把读下界正好报成两倍；
    /// - 4 位打包但没有标度伙伴，分组不明（[`WeightFormat::Packed4`]）；
    /// - 有分组，但与内核声明的分组不同（[`WeightFormat::Int4`]）——这是最容易被误判成"能算"的一类，
    ///   套上去只会得到一个用错分组的流量。
    pub fn unpriced_reason(self) -> String {
        match self {
            Self::Int4 { group } => match int4_group() {
                Some(declared) => format!(
                    "按 {group} 个元素一组量化，而内核声明的是 {declared} 个元素一组：分组不同不能共用同一条内核"
                ),
                None => format!("按 {group} 个元素一组量化，但没有任何算子声明过 int4 的分组"),
            },
            Self::Packed4 => "是 4 位打包，但找不到它的标度条目，分组不明".to_string(),
            // 稠密算子目前只有 32 位那一条：16 位的元素宽度不同，套上去读下界会正好差两倍。
            Self::F16 | Self::Bf16 => format!("编码是 {self}，还没有 16 位元素的投影算子"),
            other => format!("编码是 {other}，还没有这个格式的投影算子"),
        }
    }
}

impl fmt::Display for WeightFormat {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::F32 => formatter.write_str("f32"),
            Self::F16 => formatter.write_str("f16"),
            Self::Bf16 => formatter.write_str("bf16"),
            Self::I8 => formatter.write_str("i8"),
            Self::Packed4 => formatter.write_str("4 位打包（未声明分组）"),
            Self::Int4 { group } => write!(formatter, "int4（分组 {group}）"),
        }
    }
}

/// [`Linear::OP`] 在语义表里声明的分组。
fn int4_group() -> Option<usize> {
    op::model(crate::Linear::OP).map(|model| model.group as usize)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_element_type_maps_to_a_format_and_back() {
        for dtype in DType::ALL.iter().copied() {
            if let Some(format) = WeightFormat::of_dtype(dtype) {
                assert_eq!(format.dtype(), dtype, "{dtype:?} 往返");
            }
        }
        assert_eq!(
            WeightFormat::of_dtype(DType::U4),
            Some(WeightFormat::Packed4)
        );
    }

    #[test]
    fn only_the_declared_int4_group_reaches_an_operator() {
        // 分组来自算子语义表：这里把它读出来而不是再断言一个 32。
        let group = op::model(crate::Linear::OP).expect("算子有语义表").group as usize;
        assert_eq!(
            WeightFormat::int4(group).operator(),
            Some(crate::Linear::OP),
            "与算子声明的分组一致时才选得到算子"
        );
        assert_eq!(WeightFormat::INT4.operator(), Some(crate::Linear::OP));
        assert_eq!(
            WeightFormat::int4(group * 2).operator(),
            None,
            "分组不同就是另一个算子，不能套用"
        );
    }

    #[test]
    fn a_format_without_an_operator_says_so_instead_of_guessing() {
        for format in [
            WeightFormat::F16,
            WeightFormat::Bf16,
            WeightFormat::I8,
            WeightFormat::Packed4,
        ] {
            assert_eq!(format.operator(), None, "{format} 还没有已声明的算子");
            assert!(
                !format.unpriced_reason().is_empty(),
                "取不到算子时必须说清缺什么"
            );
        }
    }

    /// f32 现在有算子，而且就是稠密那一条：格式与算子名的对应不是注释，是可执行的断言。
    #[test]
    fn a_dense_format_reaches_the_dense_operator() {
        assert_eq!(WeightFormat::F32.operator(), Some(crate::Linear::DENSE));
        assert_eq!(WeightFormat::F32.dtype(), DType::F32);
        assert_ne!(
            WeightFormat::F32.operator(),
            Some(crate::Linear::OP),
            "稠密与量化是两条算子，不能共用"
        );
    }

    /// 每个映射到的算子名都必须在 `op` 注册表里真的存在：否则 `LayerSpec::new` 会在构造时
    /// 以 `missing` 拒绝，这里提前把漂移钉在测试里。
    #[test]
    fn every_mapped_operator_name_is_declared() {
        let formats = [WeightFormat::F32, WeightFormat::INT4];
        for format in formats {
            let operator = format.operator().expect("这两种格式都有算子");
            assert!(
                op::builtin_named(operator).is_some(),
                "{format} 映射到未注册的算子 `{operator}`"
            );
        }
    }

    #[test]
    fn the_display_form_is_stable() {
        assert_eq!(WeightFormat::F32.to_string(), "f32");
        assert_eq!(WeightFormat::INT4.to_string(), "int4（分组 32）");
    }
}
