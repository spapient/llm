//! 四个格式闭集共用的泛型视角：[`Format`]（身份）、[`BlockFormat`]（块布局）与 [`Layout`]。
//!
//! `DType`/`QType`/`MxType`/`Element` 各持自己的行表（`SPECS`/`ELEMENT_SPECS`），但「怎么读
//! 一行」完全一样：闭集全量、判别值即下标、规范名、位宽、按名解析、按元素数算字节。实现只有
//! 一份，trait 只负责把它交给泛型代码：几何走各枚举的 inherent `const fn`（常量求值需要），
//! trait 方法逐个转发；`parse_name` 反过来由本文件的 trait 默认实现唯一提供（stable 无 const
//! trait），inherent 方法转发。两条路径读同一张行表，由 `tests/dtype_contracts.rs` 一处对四个闭集
//! 逐项对拍（身份与块几何都在那里）。
//!
//! 这份抽象不是为「统一」而统一：`prof` 的位集（`Closed` + `Set<E, WORDS>`）需要一个「闭集」
//! 契约（`ALL` + 判别值 + 名字），过去只能在 L2 手写 `impl Closed for DType`；`weights` 的
//! GGUF 支持表也需要「标量元素或块格式」的布局来源（每块元素数/字节数）。两者都是本 crate
//! 已经掌握的事实，因此契约住在这里，上层只消费。
//!
//! # 为什么 trait 方法不是 `const fn`
//!
//! stable Rust 还没有稳定的 const trait，trait 方法只能是普通 `fn`。所以每个枚举同时保留
//! 一组 inherent `const fn`（形状与判别值决定了它们本就该是常量）：**常量求值走那一组，
//! 泛型代码走 trait**。转发方向按是否需要常量求值决定：
//!
//! - 几何（`block_len`/`block_bytes`/`block_count`/`bytes_for`）与身份（`name`/`bits`）的
//!   **唯一实现**是 inherent `const fn`，trait 方法只做转调——公式不会在 trait 默认实现里再抄一遍；
//! - `parse_name` 的唯一实现是 [`Format::parse_name`] 的默认方法，各枚举的 inherent
//!   `parse_name` 只做转调——stable 没有 const trait，解析本来就不可能在常量上下文里调用。
//!
//! 这个分工由 `tests/dtype_contracts.rs` 对四个闭集逐项对拍（trait 视角 == `const fn` 视角），不是靠
//! 约定；同一文件的「块几何」段只管性质（整块取整、单调、饱和与 `Layout` 两条手臂）。
//!
//! # 标准 trait：互操作只有一层转调
//!
//! 每个闭集都实现了 [`core::str::FromStr`]（转调 [`Format::parse_name`]），`DType` 另有
//! `From<DType> for u16` 与 `TryFrom<u16>`（稳定线上 code）。
//! **刻意不提供 `Default`**：一个默认精度会静默宣布一个未经声明的类型，而本 crate 的取舍是
//! 读不到就失败（[`UnknownName`]/[`UnknownCode`]），不拿一个“差不多”的值顶上。
//!
//! # 哪些枚举有规范名
//!
//! 只有**格式**有规范名、[`core::fmt::Display`] 与 [`core::str::FromStr`]：名字是它与档案、
//! 注解、配置之间的对外契约。分类枚举（`dtype::Class`，`qtype` 与 `mxtype` 各自的 `Scale` 与
//! `Family`）描述的是格式的属性，没有对外拼写，因此刻意不给 `Display`/`FromStr`/serde——
//! 给内部建模细节编一套名字，等于把建模方式固化成一份对外承诺，而它随时可能为了更准确地
//! 描述新格式而调整。
//!
//! # 泛型代码的用法
//!
//! 对布局的算法只需要 `B: BlockFormat`，与具体是 GGML 还是 MX 无关：
//!
//! ```
//! use llm_dtype::{BlockFormat, DType, Layout, MxType, QType};
//!
//! fn span<B: BlockFormat>(layout: Layout<B>) -> u64 {
//!     layout.span()
//! }
//!
//! assert_eq!(span(Layout::Block(QType::Q4_K)), 144);
//! assert_eq!(span(Layout::Block(MxType::NvFp4)), 9);
//! assert_eq!(span::<QType>(Layout::Scalar(DType::F16)), 2);
//! ```

use core::fmt;

use crate::dtype::DType;

/// 按规范名解析失败：零分配、不持有输入（调用方手里仍有原始字符串）。
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub struct UnknownName;

impl fmt::Display for UnknownName {
    /// 输出固定描述：错误本身不带名字，避免为了报错而分配。
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("unknown format name")
    }
}

impl core::error::Error for UnknownName {}

/// 按线上 code 还原失败：回显无法识别的 16 位码（`u16` 是 `Copy`，仍然零分配）。
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub struct UnknownCode {
    /// 无法识别的线上 code。
    pub code: u16,
}

impl fmt::Display for UnknownCode {
    /// 输出无法识别的 code，非法字节不需要再回头查上下文。
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        core::write!(formatter, "unknown dtype code {}", self.code)
    }
}

impl core::error::Error for UnknownCode {}

/// 给一个闭集实现 [`core::str::FromStr`]：解析的写法只有 [`Format::parse_name`] 一份。
macro_rules! impl_from_str {
    ($format:ty) => {
        impl core::str::FromStr for $format {
            type Err = $crate::format::UnknownName;

            /// 按规范名解析；未知名字报 [`UnknownName`](crate::format::UnknownName)。
            fn from_str(text: &str) -> Result<Self, Self::Err> {
                <Self as $crate::format::Format>::parse_name(text)
                    .ok_or($crate::format::UnknownName)
            }
        }
    };
}

pub(crate) use impl_from_str;

/// 给一个闭集实现 `serde` 的字符串（反）序列化：名字只取自 [`Format::name`]。
#[cfg(feature = "serde")]
macro_rules! impl_serde {
    ($format:ty) => {
        impl serde::Serialize for $format {
            /// 输出规范名（与 [`core::fmt::Display`] 逐字一致）。
            fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
            where
                S: serde::Serializer,
            {
                serializer.serialize_str(self.name())
            }
        }

        impl<'de> serde::Deserialize<'de> for $format {
            /// 按规范名还原；未知名字是 serde 的自定义错误（不分配表）。
            fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
            where
                D: serde::Deserializer<'de>,
            {
                struct Name;

                impl<'de> serde::de::Visitor<'de> for Name {
                    type Value = $format;

                    fn expecting(
                        &self,
                        formatter: &mut core::fmt::Formatter<'_>,
                    ) -> core::fmt::Result {
                        formatter.write_str(concat!("a canonical ", stringify!($format), " name"))
                    }

                    fn visit_str<E>(self, value: &str) -> Result<Self::Value, E>
                    where
                        E: serde::de::Error,
                    {
                        <$format as $crate::format::Format>::parse_name(value)
                            .ok_or_else(|| E::custom("unknown format name"))
                    }
                }

                deserializer.deserialize_str(Name)
            }
        }
    };
}

#[cfg(feature = "serde")]
pub(crate) use impl_serde;

/// 一个闭集格式的身份读法：全量表、判别值、规范名、位宽与名字解析。
pub trait Format:
    Copy + Eq + Ord + core::hash::Hash + core::fmt::Debug + core::fmt::Display + 'static
{
    /// 闭集全量，顺序即 [`Format::idx`]，也是内部行表的下标。
    const ALL: &'static [Self];

    /// 判别值（`repr(u8)` 的自然数），必须等于本类型在 [`Format::ALL`] 里的位置。
    #[must_use]
    fn idx(self) -> usize;

    /// 规范小写名，用于档案、注解与诊断。
    #[must_use]
    fn name(self) -> &'static str;

    /// 单个元素（块格式为标称元素）的位宽。
    #[must_use]
    fn bits(self) -> u32;

    /// 由规范名还原；未知名字返回 `None`。
    #[must_use]
    fn parse_name(text: &str) -> Option<Self> {
        Self::ALL
            .iter()
            .copied()
            .find(|format| format.name() == text)
    }
}

/// 块格式在 [`Format`] 之上多出来的布局：每块元素数、每块字节数与字节数算法。
pub trait BlockFormat: Format {
    /// 每块元素数（GGML 的 `blck_size`、MX 的 32/16）。
    #[must_use]
    fn block_len(self) -> u32;

    /// 每块字节数（含 scale 与 min/sum）。
    #[must_use]
    fn block_bytes(self) -> u32;

    /// `numel` 个元素按本格式需要的块数（不足一块按整块计）。
    #[must_use]
    fn block_count(self, numel: u64) -> u64;

    /// `numel` 个元素按本格式打包需要的字节数（含末块补齐与 scale）。
    #[must_use]
    fn bytes_for(self, numel: u64) -> u64;
}

/// 一个张量的布局来源：标量元素（[`DType`]）或块格式（`B`）。
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub enum Layout<B> {
    /// 标量元素（无 scale）。
    Scalar(DType),
    /// 块量化格式。
    Block(B),
}

impl<B: BlockFormat> Layout<B> {
    /// 每块元素数：标量元素按 1，块格式取 [`BlockFormat::block_len`]。
    #[must_use]
    pub fn elems(self) -> u64 {
        match self {
            Self::Scalar(_) => 1,
            Self::Block(format) => u64::from(format.block_len()),
        }
    }

    /// 每块字节数：标量元素取单元素宽度，块格式取 [`BlockFormat::block_bytes`]。
    #[must_use]
    pub fn span(self) -> u64 {
        match self {
            Self::Scalar(dtype) => dtype.bytes_for(1),
            Self::Block(format) => u64::from(format.block_bytes()),
        }
    }

    /// `numel` 个元素需要的字节数：标量按位打包，块格式按整块补齐。
    #[must_use]
    pub fn bytes_for(self, numel: u64) -> u64 {
        match self {
            Self::Scalar(dtype) => dtype.bytes_for(numel),
            Self::Block(format) => format.bytes_for(numel),
        }
    }
}

pub mod weight {
    //! 权重编码：复用 [`DType`] 的存储事实，只补充量化分组，不认识任何算子。

    use core::fmt;

    use crate::DType;

    #[cfg(feature = "serde")]
    use serde::ser::{Error as _, SerializeStruct};
    #[cfg(feature = "serde")]
    use serde::{Deserialize, Serialize};

    /// 权重编码：字段私有，合法组合只能经关联常量、构造器或转换入口建立。
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub struct WeightFormat {
        dtype: DType,
        group: usize,
        quantized: bool,
    }

    /// 格式身份与稳定配置名；`DType` 仍是唯一的存储、位宽和规范名事实源。
    #[cfg_attr(not(feature = "serde"), allow(dead_code))]
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

        /// 对称 int2 的默认格式：32 元素分组。
        pub const INT2: Self = Self::int2(32);
        /// 对称 int4 的默认格式：32 元素分组。
        pub const INT4: Self = Self::int4(32);
        /// 对称 int8 的默认格式：32 元素分组。
        pub const INT8: Self = Self::int8(32);

        /// 无分组编码；非量化时 `group` 固定为零，避免 `Option<usize>` 扩大布局。
        const fn plain(dtype: DType) -> Self {
            Self {
                dtype,
                group: 0,
                quantized: false,
            }
        }

        /// 有分组量化；布尔标记与 `group` 分离，分组为零仍保持可声明语义。
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

        /// 由逻辑量化位宽和分组构造；未知位宽返回 `None`，不猜测默认格式。
        #[must_use]
        pub fn from_quantized(bits: u8, group: usize) -> Option<Self> {
            FORMAT_DESCRIPTORS
                .iter()
                .find(|descriptor| {
                    descriptor.quantized && descriptor.dtype.bits() == u32::from(bits)
                })
                .map(|descriptor| Self::scaled(descriptor.dtype, group))
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

        /// 量化分组；非量化格式返回 `None`，调用方不需要匹配各个量化变体。
        pub const fn group(self) -> Option<usize> {
            if self.quantized {
                Some(self.group)
            } else {
                None
            }
        }

        /// 稳定配置名，由同一张描述表正向查出。
        #[cfg_attr(not(feature = "serde"), allow(dead_code))]
        fn wire_name(self) -> Option<&'static str> {
            FORMAT_DESCRIPTORS
                .iter()
                .find(|descriptor| {
                    descriptor.dtype == self.dtype && descriptor.quantized == self.quantized
                })
                .map(|descriptor| descriptor.name)
        }

        /// 从稳定配置名和分组还原；无分组格式拒绝多余的 `group`，量化格式强制要求它。
        #[cfg_attr(not(feature = "serde"), allow(dead_code))]
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

    #[cfg(feature = "serde")]
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
    #[cfg(feature = "serde")]
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
    #[cfg(feature = "serde")]
    fn serialize_group<S: SerializeStruct>(
        format: &WeightFormat,
        wire: &mut S,
    ) -> Result<(), S::Error> {
        if format.quantized {
            wire.serialize_field("group", &format.group)?;
        }
        Ok(())
    }

    /// 配置反序列化的临时结构；借用输入文本，不给 `no_std` 引入分配。
    #[cfg(feature = "serde")]
    #[derive(Deserialize)]
    #[serde(deny_unknown_fields)]
    struct WeightFormatWire<'a> {
        #[serde(rename = "type", borrow)]
        name: &'a str,
        #[serde(default)]
        group: Option<usize>,
    }

    #[cfg(feature = "serde")]
    impl<'de> Deserialize<'de> for WeightFormat {
        fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
        where
            D: serde::Deserializer<'de>,
        {
            let wire = WeightFormatWire::deserialize(deserializer)?;
            Self::from_wire(wire.name, wire.group).ok_or_else(|| {
                <D::Error as serde::de::Error>::custom("未知权重编码或分组字段不匹配")
            })
        }
    }
}
