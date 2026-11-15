//! GGML 块量化格式 [`QType`]：块内共享 scale 的权重量化身份表。
//!
//! 与标量 [`DType`](crate::DType) 的区别：`DType` 描述**一个元素怎么存**（无 scale），
//! 本模块的 `QType` 描述**一块元素怎么打包**——每块带自己的 scale（有时还带 min/sum），
//! 反量化时必须先按块解出 scale 再乘。两者是正交轴：同一份权重可以是 `Q4_K`（`QType`）
//! 解到 `F32`（`DType`）。
//!
//! # 两个不可互相推导的事实
//!
//! - [`Scale`]：块内 **scale 前缀的形状**（要先把哪几个字段读出来才有 scale）。
//! - [`Family`]：**值的编码族**（同一个 scale 形状可以出现在不同编码族里）。
//!
//! 二者刻意分开，因为 `GGML` 源码里有反例：`q8_K` 的 scale 是单个 `float`（不是 f16 加
//! sub-scale），却仍属 K-quant 族；`iq4_nl` 只有单个 `ggml_half d`，却属 IQ 查表族。
//! 若把两者压成一个字段，这两类格式必然被描述错。
//!
//! # 唯一事实源
//!
//! 全仓库的块量化「格式身份 + 布局元数据」只有这一张表。`weights` 的 GGUF 解码只保留
//! 「GGML 数字码 → 类型」的薄映射（ADR 0018 决策 5 的外部码镜像），每块元素数与字节数
//! 全部取自 [`QType::block_len`] / [`QType::block_bytes`]；反量化算法本身留在 `weights`
//! （本模块零依赖、不引入任何解码代码）。
//!
//! # 表驱动
//!
//! 每个格式只声明**不可从别处推出**的事实：规范名、每块元素数、每块字节数、标称位宽、
//! scale 形状、编码族。派生量（块数、总字节）全部由这几个字段算出，不另存一份。
//!
//! # 来源
//!
//! 块布局与字节数逐项来自 `ggml` 的 `src/ggml-common.h`（每条 `static_assert(sizeof(block_*)==…)`
//! 锁死），类型名来自 `include/ggml.h` 的 `ggml_type` 枚举。

use crate::blocked::blocked_accessors;
#[cfg(feature = "serde")]
use crate::format::impl_serde;
use crate::format::{BlockFormat, Format, impl_from_str};

/// 块内 scale 前缀的形状——决定反量化时先读哪几个字段。
///
/// 这里只描述 **scale 本身**，不描述值怎么编码（那是 [`Family`]）。块字节数里已经含了
/// scale 的空间，但「scale 是什么形状」推不出来，必须显式声明。
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub enum Scale {
    /// 单个 `f16` 缩放因子（`q1_0`/`q2_0`/`q4_0`/`q5_0`/`q8_0`，以及只有 `d` 的 IQ/TQ 格式）。
    F16,
    /// `f16` scale + 第二个 `f16`（`q4_1`/`q5_1` 的 min，`q8_1` 的 sum）。
    F16Pair,
    /// 单个 `f32` 缩放因子（只有 `q8_K`：它用于中间精度与点积，块首是 `float d`）。
    F32,
    /// super-block：`f16` 的 `d`（或 `d`+`dmin`）加每 16/32 元素的显式 sub-scale 数组。
    SubScale,
}

/// 值的编码族——决定反量化走哪一条解码路径。
///
/// 与 [`Scale`] 正交：`Family::KQuant` 里既有 `SubScale`（`q2_K`–`q6_K`）也有 `F32`
/// （`q8_K`）；`Family::Grid` 里既有 `SubScale`（`iq2_xs` 等）也有 `F16`（`iq1_s`/`iq4_nl`）。
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub enum Family {
    /// 传统块量化（`q1_0`–`q8_1`）：固定块 32/64/128，线性 scale。
    Plain,
    /// K-quant super-block（`q2_K`–`q8_K`）：块 256，量化后的 sub-scale。
    KQuant,
    /// IQ 系列（`iq1_s`–`iq4_nl`）：值由网格/码表解码。
    Grid,
    /// 三元编码（`tq1_0`/`tq2_0`）：每元素三态，1.6875/2.0625 bpw。
    Ternary,
}

/// 一个块量化格式的布局元数据。
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub struct QTypeSpec {
    /// 规范小写名（与 GGML `type_name` 一致，如 `q4_K` 归一为 `q4_k`）。
    pub name: &'static str,
    /// 每块元素数（GGML 的 `blck_size`：传统 Q 为 32/64/128，K/IQ/TQ 为 256）。
    pub block_len: u32,
    /// 每块字节数（GGML 的 `type_size`，含 scale 与 min/sum）。
    pub block_bytes: u32,
    /// 标称元素位宽（量化位数，不含 scale 摊薄；`Q4_0` 为 4，而非 4.5 bpw）。
    pub bits: u32,
    /// 块内 scale 前缀的形状。
    pub scale: Scale,
    /// 值的编码族。
    pub family: Family,
}

/// GGML 块量化格式闭集：传统 Q、K-quant、IQ 查表、TQ 三元四族。
///
/// 变体顺序与 [`QType::ALL`]、内部 `SPECS` 表一致；新增格式只能追加在末尾。
///
/// `repr(u8)` 让「判别值 = 行表下标」成为类型保证，`SPECS[self as usize]` 不会越界。
///
/// 变体名必须与 GGML 官方码名逐字一致（`Q4_K`/`IQ2_XXS`），camel 化会与外部规范脱节，
/// 所以只在这个枚举上放宽 `non_camel_case_types`：模块级 `#![allow]` 会把以后新增的类型
/// 一并豁免。
#[allow(non_camel_case_types)]
#[repr(u8)]
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug, Hash)]
#[cfg_attr(
    feature = "bytemuck",
    derive(bytemuck::NoUninit, bytemuck::CheckedBitPattern)
)]
pub enum QType {
    /// 1 位、块 128（GGML `q1_0`）。
    Q1_0,
    /// 2 位、块 64（GGML `q2_0`）。
    Q2_0,
    /// 4 位、块 32、单 f16 scale（GGML `q4_0`）。
    Q4_0,
    /// 4 位、块 32、f16 scale + min（GGML `q4_1`）。
    Q4_1,
    /// 5 位、块 32、单 f16 scale（GGML `q5_0`）。
    Q5_0,
    /// 5 位、块 32、f16 scale + min（GGML `q5_1`）。
    Q5_1,
    /// 8 位、块 32、单 f16 scale（GGML `q8_0`）。
    Q8_0,
    /// 8 位、块 32、f16 scale + sum（GGML `q8_1`）。
    Q8_1,
    /// 2 位 K-quant、块 256（GGML `q2_K`）。
    Q2_K,
    /// 3 位 K-quant、块 256（GGML `q3_K`）。
    Q3_K,
    /// 4 位 K-quant、块 256（GGML `q4_K`，事实上的部署主力）。
    Q4_K,
    /// 5 位 K-quant、块 256（GGML `q5_K`）。
    Q5_K,
    /// 6 位 K-quant、块 256（GGML `q6_K`）。
    Q6_K,
    /// 8 位 K-quant、块 256、单个 f32 scale，供矩阵乘中间精度（GGML `q8_K`）。
    Q8_K,
    /// IQ 1 位查表、块 256（GGML `iq1_s`）。
    IQ1_S,
    /// IQ 1 位查表、块 256（GGML `iq1_m`）。
    IQ1_M,
    /// IQ 2 位查表、块 256（GGML `iq2_xxs`）。
    IQ2_XXS,
    /// IQ 2 位查表、块 256（GGML `iq2_xs`）。
    IQ2_XS,
    /// IQ 2 位查表、块 256（GGML `iq2_s`）。
    IQ2_S,
    /// IQ 3 位查表、块 256（GGML `iq3_xxs`）。
    IQ3_XXS,
    /// IQ 3 位查表、块 256（GGML `iq3_s`）。
    IQ3_S,
    /// IQ 4 位查表、块 256（GGML `iq4_xs`）。
    IQ4_XS,
    /// IQ 4 位查表、块 32（GGML `iq4_nl`，非线性 16 码本）。
    IQ4_NL,
    /// 三元 1.6875 bpw、块 256（GGML `tq1_0`）。
    TQ1_0,
    /// 三元 2.0625 bpw、块 256（GGML `tq2_0`）。
    TQ2_0,
}

/// 每格式的布局，按 [`QType::ALL`] 的声明顺序对齐（索引即判别值）。
///
/// `scale` 与 `family` 分别取自 `ggml-common.h` 的块字段与 `ggml_type_traits` 的解码归属：
/// `q8_K` 的 `float d` 是唯一的 `F32`，`iq4_nl`/`iq1_s`/`iq2_xxs`/`iq3_xxs` 只有 `d`，
/// 其余 IQ 格式块内另带 sub-scale 数组。
const SPECS: [QTypeSpec; QType::ALL.len()] = [
    QTypeSpec {
        name: "q1_0",
        block_len: 128,
        block_bytes: 18,
        bits: 1,
        scale: Scale::F16,
        family: Family::Plain,
    },
    QTypeSpec {
        name: "q2_0",
        block_len: 64,
        block_bytes: 18,
        bits: 2,
        scale: Scale::F16,
        family: Family::Plain,
    },
    QTypeSpec {
        name: "q4_0",
        block_len: 32,
        block_bytes: 18,
        bits: 4,
        scale: Scale::F16,
        family: Family::Plain,
    },
    QTypeSpec {
        name: "q4_1",
        block_len: 32,
        block_bytes: 20,
        bits: 4,
        scale: Scale::F16Pair,
        family: Family::Plain,
    },
    QTypeSpec {
        name: "q5_0",
        block_len: 32,
        block_bytes: 22,
        bits: 5,
        scale: Scale::F16,
        family: Family::Plain,
    },
    QTypeSpec {
        name: "q5_1",
        block_len: 32,
        block_bytes: 24,
        bits: 5,
        scale: Scale::F16Pair,
        family: Family::Plain,
    },
    QTypeSpec {
        name: "q8_0",
        block_len: 32,
        block_bytes: 34,
        bits: 8,
        scale: Scale::F16,
        family: Family::Plain,
    },
    QTypeSpec {
        name: "q8_1",
        block_len: 32,
        block_bytes: 36,
        bits: 8,
        scale: Scale::F16Pair,
        family: Family::Plain,
    },
    QTypeSpec {
        name: "q2_k",
        block_len: 256,
        block_bytes: 84,
        bits: 2,
        scale: Scale::SubScale,
        family: Family::KQuant,
    },
    QTypeSpec {
        name: "q3_k",
        block_len: 256,
        block_bytes: 110,
        bits: 3,
        scale: Scale::SubScale,
        family: Family::KQuant,
    },
    QTypeSpec {
        name: "q4_k",
        block_len: 256,
        block_bytes: 144,
        bits: 4,
        scale: Scale::SubScale,
        family: Family::KQuant,
    },
    QTypeSpec {
        name: "q5_k",
        block_len: 256,
        block_bytes: 176,
        bits: 5,
        scale: Scale::SubScale,
        family: Family::KQuant,
    },
    QTypeSpec {
        name: "q6_k",
        block_len: 256,
        block_bytes: 210,
        bits: 6,
        scale: Scale::SubScale,
        family: Family::KQuant,
    },
    QTypeSpec {
        name: "q8_k",
        block_len: 256,
        block_bytes: 292,
        bits: 8,
        scale: Scale::F32,
        family: Family::KQuant,
    },
    QTypeSpec {
        name: "iq1_s",
        block_len: 256,
        block_bytes: 50,
        bits: 1,
        scale: Scale::F16,
        family: Family::Grid,
    },
    QTypeSpec {
        name: "iq1_m",
        block_len: 256,
        block_bytes: 56,
        bits: 1,
        scale: Scale::SubScale,
        family: Family::Grid,
    },
    QTypeSpec {
        name: "iq2_xxs",
        block_len: 256,
        block_bytes: 66,
        bits: 2,
        scale: Scale::F16,
        family: Family::Grid,
    },
    QTypeSpec {
        name: "iq2_xs",
        block_len: 256,
        block_bytes: 74,
        bits: 2,
        scale: Scale::SubScale,
        family: Family::Grid,
    },
    QTypeSpec {
        name: "iq2_s",
        block_len: 256,
        block_bytes: 82,
        bits: 2,
        scale: Scale::SubScale,
        family: Family::Grid,
    },
    QTypeSpec {
        name: "iq3_xxs",
        block_len: 256,
        block_bytes: 98,
        bits: 3,
        scale: Scale::F16,
        family: Family::Grid,
    },
    QTypeSpec {
        name: "iq3_s",
        block_len: 256,
        block_bytes: 110,
        bits: 3,
        scale: Scale::SubScale,
        family: Family::Grid,
    },
    QTypeSpec {
        name: "iq4_xs",
        block_len: 256,
        block_bytes: 136,
        bits: 4,
        scale: Scale::SubScale,
        family: Family::Grid,
    },
    QTypeSpec {
        name: "iq4_nl",
        block_len: 32,
        block_bytes: 18,
        bits: 4,
        scale: Scale::F16,
        family: Family::Grid,
    },
    QTypeSpec {
        name: "tq1_0",
        block_len: 256,
        block_bytes: 54,
        bits: 1,
        scale: Scale::F16,
        family: Family::Ternary,
    },
    QTypeSpec {
        name: "tq2_0",
        block_len: 256,
        block_bytes: 66,
        bits: 2,
        scale: Scale::F16,
        family: Family::Ternary,
    },
];

impl QType {
    /// 全部块量化格式，按声明顺序；供遍历、子集表与门禁使用。
    pub const ALL: &[Self] = &[
        Self::Q1_0,
        Self::Q2_0,
        Self::Q4_0,
        Self::Q4_1,
        Self::Q5_0,
        Self::Q5_1,
        Self::Q8_0,
        Self::Q8_1,
        Self::Q2_K,
        Self::Q3_K,
        Self::Q4_K,
        Self::Q5_K,
        Self::Q6_K,
        Self::Q8_K,
        Self::IQ1_S,
        Self::IQ1_M,
        Self::IQ2_XXS,
        Self::IQ2_XS,
        Self::IQ2_S,
        Self::IQ3_XXS,
        Self::IQ3_S,
        Self::IQ4_XS,
        Self::IQ4_NL,
        Self::TQ1_0,
        Self::TQ2_0,
    ];

    blocked_accessors!(SPECS, QTypeSpec);

    /// 由规范名还原；未知名字返回 `None`。
    ///
    /// 转调 [`Format::parse_name`]：解析的写法只有一份，inherent 这个方法只负责路径调用。
    #[must_use]
    pub fn parse_name(text: &str) -> Option<Self> {
        <Self as Format>::parse_name(text)
    }

    /// 每块字节数（含 scale 与 min/sum）。
    #[must_use]
    pub const fn block_bytes(self) -> u32 {
        self.spec().block_bytes
    }

    /// 标称元素位宽（量化位数，不含 scale 摊薄）。
    #[must_use]
    pub const fn bits(self) -> u32 {
        self.spec().bits
    }

    /// 块内 scale 前缀的形状。
    #[must_use]
    pub const fn scale(self) -> Scale {
        self.spec().scale
    }

    /// 值的编码族。
    #[must_use]
    pub const fn family(self) -> Family {
        self.spec().family
    }

    /// `numel` 个元素按本格式打包需要的字节数（含末块补齐与 scale）。
    ///
    /// 块数乘每块字节时**饱和**到 `u64::MAX`，不回绕：`numel` 可能来自权重文件头这类
    /// 不受信任的输入，回绕会把巨大张量报成很小的字节数，让"缓冲够长"的判定被绕过。
    ///
    /// ```
    /// assert_eq!(llm_dtype::QType::Q4_K.bytes_for(256), 144);
    /// assert_eq!(llm_dtype::QType::Q4_K.bytes_for(255), 144);
    /// assert_eq!(llm_dtype::QType::Q8_0.bytes_for(64), 68);
    /// ```
    #[must_use]
    pub const fn bytes_for(self, numel: u64) -> u64 {
        self.block_count(numel)
            .saturating_mul(self.block_bytes() as u64)
    }
}

// 泛型视角：与上面的 inherent `const fn` 读同一张 `SPECS`，不引入第二份事实。
impl Format for QType {
    const ALL: &'static [Self] = QType::ALL;

    fn index(self) -> usize {
        self as usize
    }

    fn name(self) -> &'static str {
        QType::name(self)
    }

    fn bits(self) -> u32 {
        QType::bits(self)
    }
}

// 纯转发：几何的唯一实现是上面的 inherent `const fn`，这里不再重写公式。
impl BlockFormat for QType {
    fn block_len(self) -> u32 {
        QType::block_len(self)
    }

    fn block_bytes(self) -> u32 {
        QType::block_bytes(self)
    }

    fn block_count(self, numel: u64) -> u64 {
        QType::block_count(self, numel)
    }

    fn bytes_for(self, numel: u64) -> u64 {
        QType::bytes_for(self, numel)
    }
}

impl core::fmt::Display for QType {
    /// 输出规范小写名（与 [`QType::name`] 一致）。
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter.write_str(self.name())
    }
}

impl_from_str!(QType);
#[cfg(feature = "serde")]
impl_serde!(QType);

#[cfg(test)]
mod tests;
