//! 四个格式闭集共用的泛型视角：[`Format`]（身份）、[`BlockFormat`]（块布局）与 [`Layout`]。
//!
//! `DType`/`QType`/`MxType`/`Element` 各持自己的行表（`SPECS`/`ELEMENT_SPECS`），但「怎么读
//! 一行」完全一样：闭集全量、判别值即下标、规范名、位宽、按名解析、按元素数算字节。实现只有
//! 一份，trait 只负责把它交给泛型代码：几何走各枚举的 inherent `const fn`（常量求值需要），
//! trait 方法逐个转发；`parse_name` 反过来由本文件的 trait 默认实现唯一提供（stable 无 const
//! trait），inherent 方法转发。两条路径读同一张行表，由 `tests/identity.rs` 一处对四个闭集
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
//! 这个分工由 `tests/identity.rs` 对四个闭集逐项对拍（trait 视角 == `const fn` 视角），不是靠
//! 约定；`tests/geometry.rs` 只管性质（整块取整、单调、饱和与 `Layout` 两条手臂）。
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
///
/// 不用 `#[derive(Serialize)]` 的 `rename_all`：`MxType` 的规范名带下划线（`mxfp8_e4m3`），
/// 而变体名没有（`MxFp8E4M3`）——逐变体写 `rename` 就是又一份名字表。这里直接读行表，
/// 于是 serde 的线上名字、`Display` 与 `FromStr` 永远是同一个词。
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
///
/// 四个格式闭集各自实现它，于是「遍历闭集 / 名字往返 / 按名查找」只需写一次泛型代码。
/// `Ord + Hash` 不是格式语义，而是位集（`prof` 的 `Set<E, WORDS>` 用判别值做下标、
/// 用排序做诊断顺序）需要的边界：同一个闭集既能当枚举用，也能当集合元素用。
///
/// 标准互操作由这里派生：每个实现它的枚举同时得到 [`core::str::FromStr`]
/// （见 `impl_from_str!`），`Display` 则输出 [`Format::name`]。`Display` 本身是本契约的
/// 一部分（见下面的 supertrait）而不是各 impl 的额外约定：闭集一定有对外名字，泛型代码
/// 因此不必再写一条 `where F: Display`。
pub trait Format:
    Copy + Eq + Ord + core::hash::Hash + core::fmt::Debug + core::fmt::Display + 'static
{
    /// 闭集全量，顺序即 [`Format::index`]，也是内部行表的下标。
    const ALL: &'static [Self];

    /// 判别值（`repr(u8)` 的自然数），必须等于本类型在 [`Format::ALL`] 里的位置。
    #[must_use]
    fn index(self) -> usize;

    /// 规范小写名，用于档案、注解与诊断。
    #[must_use]
    fn name(self) -> &'static str;

    /// 单个元素（块格式为标称元素）的位宽。
    #[must_use]
    fn bits(self) -> u32;

    /// 由规范名还原；未知名字返回 `None`。
    ///
    /// 唯一实现在这里：四个闭集的 inherent `parse_name` 只是转调，负责让
    /// `DType::parse_name(..)` 这类路径调用不必先 `use llm_dtype::Format`。
    #[must_use]
    fn parse_name(text: &str) -> Option<Self> {
        Self::ALL
            .iter()
            .copied()
            .find(|format| format.name() == text)
    }
}

/// 块格式在 [`Format`] 之上多出来的布局：每块元素数、每块字节数与字节数算法。
///
/// 标量 [`DType`] 不实现它：亚字节类型按位跨元素打包（`U4` 的两个元素共用一个字节），
/// 「每块字节」在标量轴上没有意义，硬套 `block_len = 1` 会把 `U4` 的字节数算成两倍。
/// 四个方法都不带默认实现：几何的唯一实现在各枚举的 inherent `const fn`（常量求值需要），
/// 这里若再写一份公式，就会出现「同一算法两处实现、靠测试对拍」的第二事实源。
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
    ///
    /// 块数乘每块字节时**饱和**到 `u64::MAX`，不回绕：`numel` 可能来自权重文件头这类
    /// 不受信任的输入，回绕会把巨大张量报成很小的字节数，让「缓冲够长」的判定被绕过。
    #[must_use]
    fn bytes_for(self, numel: u64) -> u64;
}

/// 一个张量的布局来源：标量元素（[`DType`]）或块格式（`B`）。
///
/// `weights` 的 GGUF 支持表用它表达「这个外部码按什么计宽」，标量手臂与块手臂的几何只有
/// 这一份实现。块参数留在泛型上，因此同一套算法将来可直接用于 [`crate::MxType`]
/// （MXFP4/NVFP4 解码接入时不必再写一遍 `elems`/`span`）。
///
/// 下面的几何方法走 [`BlockFormat`]，所以不是 `const fn`；需要在常量上下文里算几何时，
/// 直接用具体格式的 inherent `const fn`（如 [`crate::QType::bytes_for`]）。
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
