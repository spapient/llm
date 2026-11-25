//! 权重编码：复用 `DType` 的存储事实，只补充量化分组与投影算子选择。
//!
//! 没有对应算子的格式（f16/bf16/未声明分组的打包格式）仍然是合法声明，只是推不出下界。

use std::fmt;

use serde::ser::{Error as _, SerializeStruct};
use serde::{Deserialize, Serialize};
use tensor::DType;

/// 权重编码：字段私有，合法组合只能经关联常量、构造器或 `of_dtype` 建立。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct WeightFormat {
    dtype: DType,
    group: usize,
    quantized: bool,
}

/// 格式身份与稳定配置名；`DType` 仍是唯一的存储、位宽和规范名事实源。
struct FormatDescriptor {
    name: &'static str,
    dtype: DType,
    quantized: bool,
}

impl FormatDescriptor {
    const fn new(name: &'static str, dtype: DType, quantized: bool) -> Self {
        Self {
            name,
            dtype,
            quantized,
        }
    }
}

const FORMAT_DESCRIPTORS: &[FormatDescriptor] = &[
    FormatDescriptor::new("f32", DType::F32, false),
    FormatDescriptor::new("f16", DType::F16, false),
    FormatDescriptor::new("bf16", DType::BF16, false),
    FormatDescriptor::new("i8", DType::I8, false),
    FormatDescriptor::new("packed2", DType::U2, false),
    FormatDescriptor::new("packed4", DType::U4, false),
    FormatDescriptor::new("int2", DType::U2, true),
    FormatDescriptor::new("int4", DType::U4, true),
    FormatDescriptor::new("int8", DType::I8, true),
];

impl WeightFormat {
    /// 32 位浮点。
    pub const F32: Self = Self::plain(DType::F32);
    /// 16 位浮点。
    pub const F16: Self = Self::plain(DType::F16);
    /// bfloat16。
    pub const BF16: Self = Self::plain(DType::BF16);
    /// 8 位整数，没有分组声明。
    pub const I8: Self = Self::plain(DType::I8);
    /// 2 位打包整数，没有分组声明。
    pub const PACKED2: Self = Self::plain(DType::U2);
    /// 4 位打包整数，没有分组声明。
    pub const PACKED4: Self = Self::plain(DType::U4);

    /// 对称 int2 的默认格式：32 元素分组，即 `Linear::I2` 的输入格式。
    pub const INT2: Self = Self::int2(32);
    /// 量化侧的默认格式：int4 + 32 元素分组，即 `Linear::OP` 的输入格式。
    pub const INT4: Self = Self::int4(32);
    /// 对称 int8 的默认格式：32 元素分组，即 `Linear::I8` 的输入格式。
    pub const INT8: Self = Self::int8(32);

    /// 无分组编码；非量化时 `group` 固定为零，避免 `Option<usize>` 把类型撑大到 24 字节。
    const fn plain(dtype: DType) -> Self {
        Self {
            dtype,
            group: 0,
            quantized: false,
        }
    }

    /// 有分组量化；布尔标记与 `group` 分离，分组为零仍保持原有可声明语义。
    const fn scaled(dtype: DType, group: usize) -> Self {
        Self {
            dtype,
            group,
            quantized: true,
        }
    }

    /// 由权重的元素类型查出规范的无分组格式。
    #[must_use]
    pub fn of_dtype(dtype: DType) -> Option<Self> {
        FORMAT_DESCRIPTORS
            .iter()
            .find(|descriptor| !descriptor.quantized && descriptor.dtype == dtype)
            .map(|_| Self::plain(dtype))
    }

    /// 显式声明 2 位量化的分组。
    pub const fn int2(group: usize) -> Self {
        Self::scaled(DType::U2, group)
    }

    /// 显式声明 4 位量化的分组。
    pub const fn int4(group: usize) -> Self {
        Self::scaled(DType::U4, group)
    }

    /// 显式声明 8 位逐元素量化的分组。
    pub const fn int8(group: usize) -> Self {
        Self::scaled(DType::I8, group)
    }

    /// 这种编码在内存里的元素类型，直接复用 `DType`，无匹配或重复映射。
    pub const fn dtype(self) -> DType {
        self.dtype
    }

    /// 量化分组；非量化格式返回 `None`，调用方不再匹配各个量化变体。
    pub const fn group(self) -> Option<usize> {
        if self.quantized {
            Some(self.group)
        } else {
            None
        }
    }

    /// 本格式对应的已声明算子；`None` = 还没有这个格式的算子，因此这一层无法计价。
    pub fn operator(self) -> Option<&'static str> {
        match self.dtype {
            DType::F32 => Some(crate::Linear::DENSE),
            _ => self
                .quantized()
                .filter(|quant| Some(quant.group) == quant.declared)
                .map(|quant| quant.op),
        }
    }

    /// 为什么这种编码取不到算子：一句话说清"缺什么"，而不是只说"没有"。
    pub fn unpriced_reason(self) -> String {
        if let Some(quant) = self.quantized() {
            return quantized_reason(quant);
        }
        match self.dtype {
            DType::U2 | DType::U4 => format!(
                "是 {} 位打包，但找不到它的标度条目，分组不明",
                self.dtype.bits()
            ),
            DType::F16 | DType::BF16 => {
                format!("编码是 {self}，还没有 16 位元素的投影算子")
            }
            _ => format!("编码是 {self}，还没有这个格式的投影算子"),
        }
    }

    /// 带分组的量化格式描述；非量化返回 `None`。
    fn quantized(self) -> Option<Quant> {
        if !self.quantized {
            return None;
        }
        let op = match self.dtype {
            DType::U2 => crate::Linear::I2,
            DType::U4 => crate::Linear::OP,
            DType::I8 => crate::Linear::I8,
            _ => return None,
        };
        Some(Quant {
            dtype: self.dtype,
            group: self.group,
            op,
            declared: kernel_group(op),
        })
    }

    /// 稳定配置名，由同一张描述表正向查出。
    fn wire_name(self) -> Option<&'static str> {
        FORMAT_DESCRIPTORS
            .iter()
            .find(|descriptor| {
                descriptor.dtype == self.dtype && descriptor.quantized == self.quantized
            })
            .map(|descriptor| descriptor.name)
    }

    /// 从稳定配置名和分组还原；无分组格式拒绝多余的 `group`，量化格式强制要求它。
    fn from_wire(name: &str, group: Option<usize>) -> Option<Self> {
        let descriptor = FORMAT_DESCRIPTORS
            .iter()
            .find(|descriptor| descriptor.name == name)?;
        match (descriptor.quantized, group) {
            (true, Some(group)) => Some(Self::scaled(descriptor.dtype, group)),
            (false, None) => Some(Self::plain(descriptor.dtype)),
            _ => None,
        }
    }
}

impl fmt::Display for WeightFormat {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.quantized {
            return write!(formatter, "int{}（分组 {}）", self.dtype.bits(), self.group);
        }
        match self.dtype {
            DType::U2 | DType::U4 => {
                write!(formatter, "{} 位打包（未声明分组）", self.dtype.bits())
            }
            _ => formatter.write_str(self.dtype.name()),
        }
    }
}

/// 带分组的量化格式：存储类型、文件分组、对应算子与内核声明分组。
struct Quant {
    dtype: DType,
    group: usize,
    op: &'static str,
    declared: Option<usize>,
}

/// 量化分组对不上内核声明档时的诊断文案。
fn quantized_reason(quant: Quant) -> String {
    match quant.declared {
        Some(declared) => format!(
            "按 {} 个元素一组量化（int{}），而内核声明的是 {declared} 个元素一组：分组不同不能共用同一条内核",
            quant.group,
            quant.dtype.bits()
        ),
        None => format!(
            "按 {} 个元素一组量化，但没有任何算子声明过 int{} 的分组",
            quant.group,
            quant.dtype.bits()
        ),
    }
}

/// 算子在语义表里声明的分组；`None` = 算子未注册或未声明分组。
fn kernel_group(op: &str) -> Option<usize> {
    crate::registry::model(op).map(|model| model.group as usize)
}

impl Serialize for WeightFormat {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let name = self
            .wire_name()
            .ok_or_else(|| S::Error::custom("非法权重编码"))?;
        serialize_fields(self, serializer, name)
    }
}

/// 序列化字段装配：格式名必有，量化分组交给独立可选分支。
fn serialize_fields<S: serde::Serializer>(
    format: &WeightFormat,
    serializer: S,
    name: &str,
) -> Result<S::Ok, S::Error> {
    let mut wire =
        serializer.serialize_struct("WeightFormat", 1 + usize::from(format.quantized))?;
    wire.serialize_field("type", name)?;
    serialize_group(format, &mut wire)?;
    wire.end()
}

/// 只有量化格式写 `group`，无分组编码不产生多余字段。
fn serialize_group<S: SerializeStruct>(
    format: &WeightFormat,
    wire: &mut S,
) -> Result<(), S::Error> {
    if format.quantized {
        wire.serialize_field("group", &format.group)?;
    }
    Ok(())
}

/// 配置反序列化的临时结构；严格拒绝未知字段，正式状态仍由描述表构造。
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct WeightFormatWire {
    #[serde(rename = "type")]
    name: String,
    #[serde(default)]
    group: Option<usize>,
}

impl<'de> Deserialize<'de> for WeightFormat {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let wire = WeightFormatWire::deserialize(deserializer)?;
        Self::from_wire(&wire.name, wire.group)
            .ok_or_else(|| <D::Error as serde::de::Error>::custom("未知权重编码或分组字段不匹配"))
    }
}

#[cfg(test)]
#[path = "../tests/format_tests.rs"]
mod format_tests;
