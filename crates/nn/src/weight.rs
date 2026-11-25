//! 权重编码：层的「存储事实」，以及它决定了哪个投影算子。
//!
//! 编码只声明位宽与分组，算子由 [`crate::Linear`] 的常量给出；没有对应算子的格式（f16/bf16/
//! 未声明分组的打包格式）仍然是合法声明，只是推不出下界。

use std::fmt;

use tensor::DType;

/// 权重编码。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum WeightFormat {
    /// 32 位浮点。
    F32,
    /// 16 位浮点。
    F16,
    /// bfloat16。
    Bf16,
    /// 8 位整数（int8 / uint8 的语义由算子决定，本层不猜）。
    I8,
    /// 2 位打包整数，但**没有声明分组**：与 [`WeightFormat::Packed4`] 同一类"缺标度条目"的编码。
    Packed2,
    /// 4 位打包整数，但**没有声明分组**：元素类型本身不携带"多少个元素共享一个缩放因子"。
    Packed4,
    /// 2 位打包整数 + 每 `group` 个元素一个缩放因子。
    Int2 {
        /// 分组宽度（`m` 必须是它的整数倍）。
        group: usize,
    },
    /// 4 位打包整数 + 每 `group` 个元素一个缩放因子。
    Int4 {
        /// 分组宽度（`m` 必须是它的整数倍）。
        group: usize,
    },
    /// 8 位逐元素整数 + 每 `group` 个元素一个缩放因子（对称 int8，不打包）。
    Int8 {
        /// 分组宽度（`m` 必须是它的整数倍）。
        group: usize,
    },
}

/// 元素类型 ↔ 权重编码的唯一事实源：`of_dtype` 与 `dtype()` 是它的两个方向。
const FORMATS: &[(DType, WeightFormat)] = &[
    (DType::F32, WeightFormat::F32),
    (DType::F16, WeightFormat::F16),
    (DType::BF16, WeightFormat::Bf16),
    (DType::I8, WeightFormat::I8),
    (DType::U4, WeightFormat::Packed4),
    (DType::U2, WeightFormat::Packed2),
];

impl WeightFormat {
    /// 对称 int2 的默认格式：32 元素分组（与 int4 共用内核的 group 声明档），即 `Linear::I2` 的输入格式。
    pub const INT2: Self = Self::Int2 { group: 32 };

    /// 量化侧的默认格式：int4 + 32 元素分组，即 `Linear::OP` 的输入格式。
    pub const INT4: Self = Self::Int4 { group: 32 };

    /// 对称 int8 的默认格式：32 元素分组，即 `Linear::I8` 的输入格式。
    pub const INT8: Self = Self::Int8 { group: 32 };

    /// 由权重的元素类型推出格式：查 `FORMATS` 的那一个方向。
    #[must_use]
    pub fn of_dtype(dtype: DType) -> Option<Self> {
        FORMATS
            .iter()
            .find(|(element, _)| *element == dtype)
            .map(|(_, format)| *format)
    }

    /// 显式声明 4 位量化的分组。
    pub const fn int4(group: usize) -> Self {
        Self::Int4 { group }
    }

    /// 显式声明 2 位量化的分组。
    pub const fn int2(group: usize) -> Self {
        Self::Int2 { group }
    }

    /// 显式声明 8 位逐元素量化的分组。
    pub const fn int8(group: usize) -> Self {
        Self::Int8 { group }
    }

    /// 这种编码在内存里的元素类型：主匹配穷尽所有格式，分组助手把单函数复杂度留在门内。
    pub const fn dtype(self) -> DType {
        match self {
            Self::Int2 { .. } | Self::Int4 { .. } | Self::Int8 { .. } => self.grouped_dtype(),
            Self::F32 => DType::F32,
            Self::F16 => DType::F16,
            Self::Bf16 => DType::BF16,
            Self::I8 | Self::Packed2 | Self::Packed4 => self.plain_integer_dtype(),
        }
    }

    /// 带分组格式的映射；其它格式回派到总入口，因此没有不可达的默认类型。
    const fn grouped_dtype(self) -> DType {
        match self {
            Self::Int2 { .. } => DType::U2,
            Self::Int4 { .. } => DType::U4,
            Self::Int8 { .. } => DType::I8,
            other => other.dtype(),
        }
    }

    /// 无分组整数格式的映射；其它格式同样回派到总入口。
    const fn plain_integer_dtype(self) -> DType {
        match self {
            Self::I8 => DType::I8,
            Self::Packed2 => DType::U2,
            Self::Packed4 => DType::U4,
            other => other.dtype(),
        }
    }

    /// 本格式对应的已声明算子；`None` = 还没有这个格式的算子，因此这一层无法计价。
    pub fn operator(self) -> Option<&'static str> {
        match self {
            Self::F32 => Some(crate::Linear::DENSE),
            other => other
                .quantized()
                .filter(|quant| Some(quant.group) == quant.declared)
                .map(|quant| quant.op),
        }
    }

    /// 为什么这种编码取不到算子：一句话说清"缺什么"，而不是只说"没有"。
    pub fn unpriced_reason(self) -> String {
        // 量化格式的说法只有一种：缺的是「分组对得上」而不是算子本身。
        if let Some(quant) = self.quantized() {
            return quantized_reason(quant);
        }
        match self {
            Self::Packed2 | Self::Packed4 => format!(
                "是 {} 位打包，但找不到它的标度条目，分组不明",
                self.dtype().bits()
            ),
            // 稠密算子目前只有 32 位那一条：16 位的元素宽度不同，套上去读下界会正好差两倍。
            Self::F16 | Self::Bf16 => format!("编码是 {self}，还没有 16 位元素的投影算子"),
            other => format!("编码是 {other}，还没有这个格式的投影算子"),
        }
    }

    /// 带分组的量化格式描述；非量化返回 `None`。
    fn quantized(self) -> Option<Quant> {
        match self {
            Self::Int2 { group } => Some(Quant {
                group,
                op: crate::Linear::I2,
                declared: kernel_group(crate::Linear::I2),
                label: "int2",
            }),
            Self::Int4 { group } => Some(Quant {
                group,
                op: crate::Linear::OP,
                declared: kernel_group(crate::Linear::OP),
                label: "int4",
            }),
            Self::Int8 { group } => Some(Quant {
                group,
                op: crate::Linear::I8,
                declared: kernel_group(crate::Linear::I8),
                label: "int8",
            }),
            _ => None,
        }
    }
}

impl fmt::Display for WeightFormat {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Packed2 | Self::Packed4 => {
                write!(formatter, "{} 位打包（未声明分组）", self.dtype().bits())
            }
            Self::Int2 { group } => write!(formatter, "int2（分组 {group}）"),
            Self::Int4 { group } => write!(formatter, "int4（分组 {group}）"),
            Self::Int8 { group } => write!(formatter, "int8（分组 {group}）"),
            // 其余四种编码的名字就是元素类型的规范名，不在这里再写一遍字面量。
            element => formatter.write_str(element.dtype().name()),
        }
    }
}

/// 带分组的量化格式：文件声明的分组、对应算子、内核声明分组与诊断文案标签。
struct Quant {
    group: usize,
    op: &'static str,
    declared: Option<usize>,
    label: &'static str,
}

/// 量化分组对不上内核声明档时的诊断文案。
fn quantized_reason(quant: Quant) -> String {
    match quant.declared {
        Some(declared) => format!(
            "按 {} 个元素一组量化（{}），而内核声明的是 {declared} 个元素一组：分组不同不能共用同一条内核",
            quant.group, quant.label
        ),
        None => format!(
            "按 {} 个元素一组量化，但没有任何算子声明过 {} 的分组",
            quant.group, quant.label
        ),
    }
}

/// 算子在语义表里声明的分组；`None` = 算子未注册或未声明分组。
fn kernel_group(op: &str) -> Option<usize> {
    crate::registry::model(op).map(|model| model.group as usize)
}

#[cfg(test)]
#[path = "../tests/format_tests.rs"]
mod format_tests;
