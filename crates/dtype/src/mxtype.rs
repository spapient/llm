//! 微缩放格式 [`MxType`]：OCP MX 规范的元素格式与 NVIDIA NVFP4 厂商扩展。
//!
//! 第三类精度轴，与 [`DType`]、[`QType`](crate::QType) 正交：
//!
//! - `DType` 描述单个元素，**没有共享 scale**；
//! - `QType` 描述 GGML 块（`f16`/`f32` 的 scale，可带 min/sum），块内是它的私有编码；
//! - `MxType` 描述「低精度元素 + 块内共享指数」的微缩放——块固定 32 元素（OCP MX）
//!   或 16 元素（NVFP4），scale 是**纯指数**或高精度 FP8，而不是 GGML 的 f16 线性 scale。
//!
//! # 与 `QType` 的分界
//!
//! MX 不是 GGML 的子集：它的 scale 是 E8M0（8 位纯指数、无尾数）且块大小由 OCP 规范固定，
//! 与 GGML 的 f16 线性 scale、可变块（32/128/256）是两套独立标准。混进同一个枚举会让
//! 反量化分支互相污染，故独立成文件。
//!
//! # 位宽与块字节都是派生量，不是事实
//!
//! 与 [`QType`](crate::QType) 不同，这里**不存** `bits` 与 `block_bytes`：MX 的标称位宽恒等于
//! 元素位宽，块字节恒等于 `ceil(block_len × element.bits / 8)` 再加 1 字节 scale，存一份就是
//! 第二事实源。`QType` 之所以显式存，是因为 `Q8_1` 的量化位（8）不等于含 scale 摊薄的 bpw（9），
//! 且块字节要按 `ggml-common.h` 逐格式锁定——两个模块的不对称是语义决定的，不是疏漏。
//!
//! # 来源
//!
//! `OCP MicroScaling (MX) Specification v1.1.0` 定义 [`MxType::MxFp8E4M3`] 等六个标准格式；
//! NVFP4 属 NVIDIA Blackwell 厂商扩展，不写入 OCP 标准码表。
//!
//! # 消费者
//!
//! 当前**还没有**解码器消费 [`MxType`]：7 个格式与 [`Element::dtype`] 只有本 crate 内的测试
//! 覆盖。布局已经有规范对照——`tests/mx_spec.rs` 用 OCP v1.1 的数值表与手写字节块核了块长、
//! 每块字节与 E2M1/E4M3 解码——但**还没有一条真实模型的权重字节流**校过。因此：接上
//! MX/NVFP4 反量化之前不再扩格式；接上时先补一条真实字节流的对拍，再谈加格式。

use crate::blocked::blocked_accessors;
use crate::dtype::DType;
#[cfg(feature = "serde")]
use crate::format::impl_serde;
use crate::format::{BlockFormat, Format, impl_from_str};

/// 一个 MX 元素格式的身份：规范名、Rust 变体名、位宽与标量对应。
///
/// 四者都必须显式声明：位宽不能从名字推出（`E2M1` 是 4 位，但 `Int8` 的 8 与
/// 尾数/指数切分无关），而「有没有标量对应」是身份而非布局——所以和 [`MxType`] 的
/// `SPECS` 一样，一行一个元素，位宽与映射都读这一行。
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub struct ElementSpec {
    /// 规范小写名（`e2m1`），用于档案与诊断。
    pub name: &'static str,
    /// Rust 变体名（`E2M1`），与 [`Debug`] 输出逐字一致。
    pub variant: &'static str,
    /// 元素标称位宽。
    pub bits: u32,
    /// 对应的标量元素类型；MX 专有的 6 位格式为 `None`。
    pub dtype: Option<DType>,
}

/// MX 元素格式闭集（低精度浮点或整数），决定元素的位宽与指数/尾数切分。
///
/// 变体顺序与 [`Element::ALL`]、内部 `ELEMENT_SPECS` 表一致；`repr(u8)` 让判别值即行表下标。
#[repr(u8)]
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug, Hash)]
#[cfg_attr(
    feature = "bytemuck",
    derive(bytemuck::NoUninit, bytemuck::CheckedBitPattern)
)]
pub enum Element {
    /// 4 位浮点 E2M1（1 符号 + 2 指数 + 1 尾数），MXFP4 与 NVFP4 的元素。
    E2M1,
    /// 6 位浮点 E2M3（1 符号 + 2 指数 + 3 尾数）。
    E2M3,
    /// 6 位浮点 E3M2（1 符号 + 3 指数 + 2 尾数）。
    E3M2,
    /// 8 位浮点 E4M3（1 符号 + 4 指数 + 3 尾数）。
    E4M3,
    /// 8 位浮点 E5M2（1 符号 + 5 指数 + 2 尾数）。
    E5M2,
    /// 8 位有符号整数（MXINT8）。
    Int8,
}

/// 每元素的身份与布局，按 [`Element::ALL`] 的声明顺序对齐（索引即判别值）。
const ELEMENT_SPECS: [ElementSpec; Element::ALL.len()] = [
    ElementSpec {
        name: "e2m1",
        variant: "E2M1",
        bits: 4,
        dtype: Some(DType::F4),
    },
    ElementSpec {
        name: "e2m3",
        variant: "E2M3",
        bits: 6,
        dtype: None,
    },
    ElementSpec {
        name: "e3m2",
        variant: "E3M2",
        bits: 6,
        dtype: None,
    },
    ElementSpec {
        name: "e4m3",
        variant: "E4M3",
        bits: 8,
        dtype: Some(DType::F8E4M3),
    },
    ElementSpec {
        name: "e5m2",
        variant: "E5M2",
        bits: 8,
        dtype: Some(DType::F8E5M2),
    },
    ElementSpec {
        name: "int8",
        variant: "Int8",
        bits: 8,
        dtype: Some(DType::I8),
    },
];

impl Element {
    /// 全部元素格式，按声明顺序；供遍历与门禁使用。
    pub const ALL: &[Self] = &[
        Self::E2M1,
        Self::E2M3,
        Self::E3M2,
        Self::E4M3,
        Self::E5M2,
        Self::Int8,
    ];

    /// 身份与布局元数据（规范名、变体名、位宽、标量对应）。
    #[must_use]
    pub const fn spec(self) -> ElementSpec {
        ELEMENT_SPECS[self as usize]
    }

    /// 规范小写名，用于档案与诊断；与 [`Element::parse_name`] 双向。
    #[must_use]
    pub const fn name(self) -> &'static str {
        self.spec().name
    }

    /// Rust 变体名（与 [`Debug`] 输出一致），供代码生成与门禁对齐用。
    #[must_use]
    pub const fn variant(self) -> &'static str {
        self.spec().variant
    }

    /// 元素的标称位宽。
    #[must_use]
    pub const fn bits(self) -> u32 {
        self.spec().bits
    }

    /// 由规范名还原；未知名字返回 `None`。
    ///
    /// 转调 [`Format::parse_name`]：解析的写法只有一份，inherent 这个方法只负责路径调用。
    #[must_use]
    pub fn parse_name(text: &str) -> Option<Self> {
        <Self as Format>::parse_name(text)
    }

    /// 对应的标量元素类型；MX 专有的 6 位格式没有标量对应，返回 `None`。
    ///
    /// `E2M1`/`E4M3`/`E5M2`/`Int8` 与 [`DType::F4`]/[`DType::F8E4M3`]/[`DType::F8E5M2`]/
    /// [`DType::I8`] 是同一批格式的两个名字；`E2M3`/`E3M2` 没有对应的 Rust 标量类型，
    /// 与 [`crate::QType::bits`] 的取舍相同：不对应就明说 `None`，不在下游另写一份映射。
    #[must_use]
    pub const fn dtype(self) -> Option<DType> {
        self.spec().dtype
    }
}

// 泛型视角：与上面的 inherent `const fn` 读同一张 `ELEMENT_SPECS`。
impl Format for Element {
    const ALL: &'static [Self] = Element::ALL;

    fn index(self) -> usize {
        self as usize
    }

    fn name(self) -> &'static str {
        Element::name(self)
    }

    fn bits(self) -> u32 {
        Element::bits(self)
    }
}

impl core::fmt::Display for Element {
    /// 输出规范小写名（与 [`Element::name`] 一致）。
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter.write_str(self.name())
    }
}

/// 块内微缩放的种类。
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub enum Scale {
    /// E8M0：8 位无符号纯指数（OCP MX 标准共享 scale，块内只有 2 的幂因子）。
    E8M0,
    /// E4M3（FP8）微块 scale，另有 FP32 每张量第二级 scale（NVFP4 的两级缩放）。
    E4M3,
}

impl Scale {
    /// scale 字段的字节宽度（E8M0 与 E4M3 都是 1 字节）。
    const fn bytes(self) -> u32 {
        match self {
            Self::E8M0 | Self::E4M3 => 1,
        }
    }
}

/// 格式家族：OCP 标准格式，或厂商扩展。
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub enum Family {
    /// OCP MX 标准格式（块 32 元素 + E8M0 scale）。
    Ocp,
    /// 厂商扩展（NVFP4，块 16 元素 + 两级缩放）。
    Vendor,
}

/// 一个微缩放格式的布局元数据。
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub struct MxTypeSpec {
    /// 规范名（`mxfp8_e4m3` 等，小写无前缀）。
    pub name: &'static str,
    /// 每块元素数（OCP MX 为 32，NVFP4 为 16）。
    pub block_len: u32,
    /// 元素格式。
    pub element: Element,
    /// 块内 scale 种类。
    pub scale: Scale,
    /// 家族归属。
    pub family: Family,
}

/// 微缩放格式闭集：OCP MX 六个标准格式 + NVFP4 厂商扩展。
///
/// 变体顺序与 [`MxType::ALL`]、内部 `SPECS` 表一致；新增格式只能追加在末尾。
///
/// `repr(u8)` 让「判别值 = 行表下标」成为类型保证，`SPECS[self as usize]` 不会越界。
#[repr(u8)]
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug, Hash)]
#[cfg_attr(
    feature = "bytemuck",
    derive(bytemuck::NoUninit, bytemuck::CheckedBitPattern)
)]
pub enum MxType {
    /// MXFP8 E4M3：8 位元素、块 32、E8M0 scale。
    MxFp8E4M3,
    /// MXFP8 E5M2：8 位元素、块 32、E8M0 scale。
    MxFp8E5M2,
    /// MXFP6 E3M2：6 位元素、块 32、E8M0 scale。
    MxFp6E3M2,
    /// MXFP6 E2M3：6 位元素、块 32、E8M0 scale。
    MxFp6E2M3,
    /// MXFP4 E2M1：4 位元素、块 32、E8M0 scale（OCP 标准）。
    MxFp4E2M1,
    /// MXINT8：8 位整数、块 32、E8M0 scale。
    MxInt8,
    /// NVFP4：4 位 E2M1 元素、块 16、E4M3 微块 scale + FP32 每张量 scale（NVIDIA 厂商扩展）。
    ///
    /// 这里是 NVIDIA 原生的 16 元素微块（9 字节）。GGML 的 `QK_NVFP4=64` 把 4 个微块拼成
    /// 一块（4×9=36 字节），所以本格式在 64 元素上算出的 `bytes_for` 正好也是 36，
    /// 两种粒度不冲突。
    NvFp4,
}

/// 每格式的布局，按 [`MxType::ALL`] 的声明顺序对齐（索引即判别值）。
const SPECS: [MxTypeSpec; MxType::ALL.len()] = [
    MxTypeSpec {
        name: "mxfp8_e4m3",
        block_len: 32,
        element: Element::E4M3,
        scale: Scale::E8M0,
        family: Family::Ocp,
    },
    MxTypeSpec {
        name: "mxfp8_e5m2",
        block_len: 32,
        element: Element::E5M2,
        scale: Scale::E8M0,
        family: Family::Ocp,
    },
    MxTypeSpec {
        name: "mxfp6_e3m2",
        block_len: 32,
        element: Element::E3M2,
        scale: Scale::E8M0,
        family: Family::Ocp,
    },
    MxTypeSpec {
        name: "mxfp6_e2m3",
        block_len: 32,
        element: Element::E2M3,
        scale: Scale::E8M0,
        family: Family::Ocp,
    },
    MxTypeSpec {
        name: "mxfp4_e2m1",
        block_len: 32,
        element: Element::E2M1,
        scale: Scale::E8M0,
        family: Family::Ocp,
    },
    MxTypeSpec {
        name: "mxint8",
        block_len: 32,
        element: Element::Int8,
        scale: Scale::E8M0,
        family: Family::Ocp,
    },
    MxTypeSpec {
        name: "nvfp4",
        block_len: 16,
        element: Element::E2M1,
        scale: Scale::E4M3,
        family: Family::Vendor,
    },
];

impl MxType {
    /// 全部微缩放格式，按声明顺序；供遍历与门禁使用。
    pub const ALL: &[Self] = &[
        Self::MxFp8E4M3,
        Self::MxFp8E5M2,
        Self::MxFp6E3M2,
        Self::MxFp6E2M3,
        Self::MxFp4E2M1,
        Self::MxInt8,
        Self::NvFp4,
    ];

    blocked_accessors!(SPECS, MxTypeSpec);

    /// 由规范名还原；未知名字返回 `None`。
    ///
    /// 转调 [`Format::parse_name`]：解析的写法只有一份，inherent 这个方法只负责路径调用。
    #[must_use]
    pub fn parse_name(text: &str) -> Option<Self> {
        <Self as Format>::parse_name(text)
    }

    /// 每块字节数（含 scale 占位）：由块长、元素位宽与 scale 宽度派生，不另存一份。
    #[must_use]
    pub const fn block_bytes(self) -> u32 {
        let spec = self.spec();
        (spec.block_len * spec.element.bits()).div_ceil(8) + spec.scale.bytes()
    }

    /// 元素格式。
    #[must_use]
    pub const fn element(self) -> Element {
        self.spec().element
    }

    /// 块内 scale 的种类。
    #[must_use]
    pub const fn scale(self) -> Scale {
        self.spec().scale
    }

    /// 家族归属。
    #[must_use]
    pub const fn family(self) -> Family {
        self.spec().family
    }

    /// 元素的标称位宽（由 [`MxType::element`] 派生，不另存）。
    #[must_use]
    pub const fn bits(self) -> u32 {
        self.element().bits()
    }

    /// `numel` 个元素按本格式需要的字节数（不足一块按整块计）。
    ///
    /// 块数乘每块字节时**饱和**到 `u64::MAX`，不回绕：`numel` 可能来自权重文件头这类
    /// 不受信任的输入，回绕会把巨大张量报成很小的字节数，让"缓冲够长"的判定被绕过。
    ///
    /// ```
    /// assert_eq!(llm_dtype::MxType::MxFp4E2M1.bytes_for(32), 17);
    /// assert_eq!(llm_dtype::MxType::NvFp4.bytes_for(64), 36);
    /// ```
    #[must_use]
    pub const fn bytes_for(self, numel: u64) -> u64 {
        self.block_count(numel)
            .saturating_mul(self.block_bytes() as u64)
    }
}

// 泛型视角：与上面的 inherent `const fn` 读同一张 `SPECS`；`bits` 仍从元素派生，不另存。
impl Format for MxType {
    const ALL: &'static [Self] = MxType::ALL;

    fn index(self) -> usize {
        self as usize
    }

    fn name(self) -> &'static str {
        MxType::name(self)
    }

    fn bits(self) -> u32 {
        MxType::bits(self)
    }
}

// 纯转发：几何的唯一实现是上面的 inherent `const fn`（`block_bytes` 仍由元素位宽派生），
// 这里不再重写公式。
impl BlockFormat for MxType {
    fn block_len(self) -> u32 {
        MxType::block_len(self)
    }

    fn block_bytes(self) -> u32 {
        MxType::block_bytes(self)
    }

    fn block_count(self, numel: u64) -> u64 {
        MxType::block_count(self, numel)
    }

    fn bytes_for(self, numel: u64) -> u64 {
        MxType::bytes_for(self, numel)
    }
}

impl core::fmt::Display for MxType {
    /// 输出规范名（与 [`MxType::name`] 一致）。
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter.write_str(self.name())
    }
}

impl_from_str!(MxType);
impl_from_str!(Element);
#[cfg(feature = "serde")]
impl_serde!(MxType);
#[cfg(feature = "serde")]
impl_serde!(Element);

#[cfg(test)]
mod tests {
    use super::*;
    use std::{format, mem::size_of};

    #[test]
    fn mx_type_discriminant_fits_one_byte() {
        assert_eq!(size_of::<MxType>(), 1, "repr(u8) 加 7 个格式应正好一字节");
    }

    #[test]
    fn element_table_is_locked() {
        // 逐行锁死元素身份：位宽、名字或标量对应被误改时失败在这一行。
        const EXPECTED: [(&str, &str, u32, Option<DType>); Element::ALL.len()] = [
            ("e2m1", "E2M1", 4, Some(DType::F4)),
            ("e2m3", "E2M3", 6, None),
            ("e3m2", "E3M2", 6, None),
            ("e4m3", "E4M3", 8, Some(DType::F8E4M3)),
            ("e5m2", "E5M2", 8, Some(DType::F8E5M2)),
            ("int8", "Int8", 8, Some(DType::I8)),
        ];
        for (element, (name, variant, bits, dtype)) in Element::ALL.iter().zip(EXPECTED) {
            let spec = element.spec();
            assert_eq!(
                (spec.name, spec.variant, spec.bits, spec.dtype),
                (name, variant, bits, dtype),
                "{element:?} 的身份行被改动"
            );
            assert_eq!(element.name(), variant.to_ascii_lowercase());
            assert_eq!(element.variant(), format!("{element:?}"));
            assert_eq!(Element::parse_name(name), Some(*element));
        }
    }

    #[test]
    fn element_discriminant_fits_one_byte() {
        assert_eq!(size_of::<Element>(), 1, "repr(u8) 加 6 个元素应正好一字节");
    }

    #[test]
    fn accessors_read_the_element_row() {
        // 行表锁字段，这里锁「取哪一列」：`Element::bits()`/`dtype()` 没有其它读取点。
        for element in Element::ALL {
            let spec = element.spec();
            assert_eq!(element.name(), spec.name);
            assert_eq!(element.variant(), spec.variant);
            assert_eq!(element.bits(), spec.bits);
            assert_eq!(element.dtype(), spec.dtype);
        }
    }

    #[test]
    fn layout_table_is_locked() {
        // 一行一个格式：规范名、块长、块字节、元素、scale、家族。位宽由元素派生，不在这里
        // 重复一列；块字节是派生量，但派生公式的等价改写不该改变这里锁死的数值——事实先于
        // 公式。表格是给人逐行核对的，所以跳过格式化、保持一行一个格式。
        #[rustfmt::skip]
        const LOCKED: [(&str, u32, u32, Element, Scale, Family); MxType::ALL.len()] = [
            ("mxfp8_e4m3", 32, 33, Element::E4M3, Scale::E8M0, Family::Ocp),
            ("mxfp8_e5m2", 32, 33, Element::E5M2, Scale::E8M0, Family::Ocp),
            ("mxfp6_e3m2", 32, 25, Element::E3M2, Scale::E8M0, Family::Ocp),
            ("mxfp6_e2m3", 32, 25, Element::E2M3, Scale::E8M0, Family::Ocp),
            ("mxfp4_e2m1", 32, 17, Element::E2M1, Scale::E8M0, Family::Ocp),
            ("mxint8", 32, 33, Element::Int8, Scale::E8M0, Family::Ocp),
            ("nvfp4", 16, 9, Element::E2M1, Scale::E4M3, Family::Vendor),
        ];
        for (mx, (name, block_len, block_bytes, element, scale, family)) in
            MxType::ALL.iter().zip(LOCKED)
        {
            let spec = mx.spec();
            assert_eq!(
                (
                    spec.name,
                    spec.block_len,
                    mx.block_bytes(),
                    spec.element,
                    spec.scale,
                    spec.family,
                ),
                (name, block_len, block_bytes, element, scale, family),
                "{mx:?} 的布局行被改动"
            );
        }
        // 位宽是元素的函数：元素一旦锁死，这三个 `bits()` 就只能是 4/6/8。
        assert_eq!(
            (
                MxType::MxFp4E2M1.bits(),
                MxType::MxFp6E3M2.bits(),
                MxType::MxInt8.bits(),
            ),
            (4, 6, 8)
        );
    }

    #[test]
    fn bytes_for_pads_to_whole_blocks() {
        assert_eq!(MxType::MxFp4E2M1.bytes_for(32), 17);
        assert_eq!(MxType::MxFp4E2M1.bytes_for(64), 34);
        assert_eq!(MxType::MxFp4E2M1.bytes_for(31), 17, "不足一块按整块计");
        assert_eq!(MxType::NvFp4.bytes_for(16), 9);
        // GGML 的 NVFP4 容器按 64 元素成块，4 个原生微块正好 36 字节。
        assert_eq!(MxType::NvFp4.bytes_for(64), 36);
    }
}
