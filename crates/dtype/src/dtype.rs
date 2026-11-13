//! 全精度闭集 [`DType`]：全仓库元素类型的唯一事实源。
//!
//! 各层不再手写类型枚举与跨层转换，需要表达"本层只支持其中一部分"时用子集切片或位集，
//! 而不是另造一个 enum。统一的三套旧枚举是 `tensor::DType`（5 变体）、`prof::atom::DType`
//! （9 变体）与 `weights::dtype::Wire`（21 变体）。
//!
//! # 线上 code 的稳定性
//!
//! [`DType::code`] 的 `1..=21` 与旧 `Wire` 的判别值逐位相同，已有 LLPK 容器字节完全兼容；
//! 新增的 `F4`（4 位浮点）刻意追加在末尾取 `22`，`FNUZ` 两个变体取 `23`/`24`，均不插入中间，
//! 避免重排任何既有 code。
//!
//! # 零成本与单一事实源
//!
//! 24 个变体判别值装得下一个 `u8`；位宽、分类谓词、名字与变体名映射全是 `const fn`，
//! 编译期求值，没有任何分配或运行时转换路径。每个变体的全部身份（规范名、Rust 变体名、
//! 位宽、数值类别）住在 `SPECS` 的**同一行**，按判别值索引，不存在两张表各写一半的可能。

#[cfg(feature = "serde")]
use crate::format::impl_serde;
use crate::format::{Format, UnknownCode, impl_from_str};

/// 数值类别：决定 [`DType::is_float`]、[`DType::is_integer`] 与 [`DType::is_signed`]。
/// 类别不能从名字或位宽推出（`Bool` 是 1 位但不是整数，`F4` 是 4 位但不是整数），
/// 所以和位宽一样显式声明，让三个谓词共享同一份事实而不是各写一份 `matches!` 清单。
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub enum Class {
    /// 布尔（既不是整数也不是浮点）。
    Bool,
    /// 无符号整数。
    UInt,
    /// 有符号整数。
    SInt,
    /// 浮点（含 FP4/FP8 与 FNUZ 变体）。
    Float,
}

/// 一个标量类型的身份与布局：规范名、Rust 变体名、位宽、数值类别。
///
/// 四者都必须显式声明、都不能从别处推出：`bf16` 的变体名是 `BF16` 而不是机械首字母大写，
/// 位宽也不等于名字里的数字（`F4` 是 4、`Bool` 是 1）。
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub struct DTypeSpec {
    /// 规范小写名（`f16`、`bf16`、`f8e4m3fnuz`），用于档案、注解与诊断。
    pub name: &'static str,
    /// Rust 变体名（`BF16`），与 [`Debug`] 输出逐字一致。
    pub variant: &'static str,
    /// 单个元素的位宽。
    pub bits: u32,
    /// 数值类别，三个分类谓词的唯一来源。
    pub class: Class,
}

/// 元素类型闭集：布尔、无符号/有符号整数、浮点与亚字节打包类型。
///
/// 变体顺序就是 [`DType::code`] 的编码顺序；加新类型只能追加在 [`DType::ALL`] 末尾，
/// 重排会破坏容器字节兼容。
///
/// `repr(u8)` 把「判别值 = 行表下标」从测试观察升级为类型保证：`SPECS[self as usize]`
/// 不可能越界，判别值也不可能被默认成别的宽度。
#[repr(u8)]
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug, Hash)]
#[cfg_attr(
    feature = "bytemuck",
    derive(bytemuck::NoUninit, bytemuck::CheckedBitPattern)
)]
pub enum DType {
    /// 布尔（1 位）。
    Bool,
    /// 无符号 1 位整数。
    U1,
    /// 无符号 2 位整数。
    U2,
    /// 无符号 4 位整数（按位打包）。
    U4,
    /// 无符号 8 位整数。
    U8,
    /// 无符号 16 位整数。
    U16,
    /// 无符号 32 位整数。
    U32,
    /// 无符号 64 位整数。
    U64,
    /// 有符号 1 位整数。
    I1,
    /// 有符号 2 位整数。
    I2,
    /// 有符号 4 位整数（按位打包）。
    I4,
    /// 有符号 8 位整数。
    I8,
    /// 有符号 16 位整数。
    I16,
    /// 有符号 32 位整数。
    I32,
    /// 有符号 64 位整数。
    I64,
    /// 8 位浮点 E4M3（1 符号 + 4 指数 + 3 尾数）。
    F8E4M3,
    /// 8 位浮点 E5M2（1 符号 + 5 指数 + 2 尾数）。
    F8E5M2,
    /// IEEE 16 位浮点。
    F16,
    /// bfloat16。
    BF16,
    /// IEEE 32 位浮点。
    F32,
    /// IEEE 64 位浮点。
    F64,
    /// 4 位浮点 E2M1（MXFP4 与 NVFP4 的元素，code 22）。
    F4,
    /// 8 位浮点 E4M3 的 FNUZ 变体（无 inf、零为无符号，AMD/OCP 变体，code 23）。
    F8E4M3FNUZ,
    /// 8 位浮点 E5M2 的 FNUZ 变体（无 inf、零为无符号，AMD/OCP 变体，code 24）。
    F8E5M2FNUZ,
}

/// 每格式的身份与布局，按 [`DType::ALL`] 的声明顺序对齐（索引即判别值）。
///
/// 一行一个类型，改名、改位宽或改类别只动这一行；[`DType::name`]、[`DType::variant`]、
/// [`DType::bits`] 与三个分类谓词都读这张表，四者不可能各持一份。
const SPECS: [DTypeSpec; DType::ALL.len()] = [
    DTypeSpec {
        name: "bool",
        variant: "Bool",
        bits: 1,
        class: Class::Bool,
    },
    DTypeSpec {
        name: "u1",
        variant: "U1",
        bits: 1,
        class: Class::UInt,
    },
    DTypeSpec {
        name: "u2",
        variant: "U2",
        bits: 2,
        class: Class::UInt,
    },
    DTypeSpec {
        name: "u4",
        variant: "U4",
        bits: 4,
        class: Class::UInt,
    },
    DTypeSpec {
        name: "u8",
        variant: "U8",
        bits: 8,
        class: Class::UInt,
    },
    DTypeSpec {
        name: "u16",
        variant: "U16",
        bits: 16,
        class: Class::UInt,
    },
    DTypeSpec {
        name: "u32",
        variant: "U32",
        bits: 32,
        class: Class::UInt,
    },
    DTypeSpec {
        name: "u64",
        variant: "U64",
        bits: 64,
        class: Class::UInt,
    },
    DTypeSpec {
        name: "i1",
        variant: "I1",
        bits: 1,
        class: Class::SInt,
    },
    DTypeSpec {
        name: "i2",
        variant: "I2",
        bits: 2,
        class: Class::SInt,
    },
    DTypeSpec {
        name: "i4",
        variant: "I4",
        bits: 4,
        class: Class::SInt,
    },
    DTypeSpec {
        name: "i8",
        variant: "I8",
        bits: 8,
        class: Class::SInt,
    },
    DTypeSpec {
        name: "i16",
        variant: "I16",
        bits: 16,
        class: Class::SInt,
    },
    DTypeSpec {
        name: "i32",
        variant: "I32",
        bits: 32,
        class: Class::SInt,
    },
    DTypeSpec {
        name: "i64",
        variant: "I64",
        bits: 64,
        class: Class::SInt,
    },
    DTypeSpec {
        name: "f8e4m3",
        variant: "F8E4M3",
        bits: 8,
        class: Class::Float,
    },
    DTypeSpec {
        name: "f8e5m2",
        variant: "F8E5M2",
        bits: 8,
        class: Class::Float,
    },
    DTypeSpec {
        name: "f16",
        variant: "F16",
        bits: 16,
        class: Class::Float,
    },
    DTypeSpec {
        name: "bf16",
        variant: "BF16",
        bits: 16,
        class: Class::Float,
    },
    DTypeSpec {
        name: "f32",
        variant: "F32",
        bits: 32,
        class: Class::Float,
    },
    DTypeSpec {
        name: "f64",
        variant: "F64",
        bits: 64,
        class: Class::Float,
    },
    DTypeSpec {
        name: "f4",
        variant: "F4",
        bits: 4,
        class: Class::Float,
    },
    DTypeSpec {
        name: "f8e4m3fnuz",
        variant: "F8E4M3FNUZ",
        bits: 8,
        class: Class::Float,
    },
    DTypeSpec {
        name: "f8e5m2fnuz",
        variant: "F8E5M2FNUZ",
        bits: 8,
        class: Class::Float,
    },
];

impl DType {
    /// 全部元素类型，按 code 顺序；供遍历、子集表与测试使用。
    pub const ALL: &[Self] = &[
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
        Self::F4,
        Self::F8E4M3FNUZ,
        Self::F8E5M2FNUZ,
    ];

    /// 身份与布局元数据（规范名、变体名、位宽）。
    #[must_use]
    pub const fn spec(self) -> DTypeSpec {
        SPECS[self as usize]
    }

    /// 稳定线上 code（`1..=24`；`0` 保留不分配）。
    ///
    /// 判别值与 [`DType::ALL`] 的声明顺序一致（`Bool` 取 `1`，最后追加的 `F8E5M2FNUZ` 取
    /// `24`），因此 code 就是判别值加一；`codes_are_a_stable_bijection` 锁死这份编码。
    #[must_use]
    pub const fn code(self) -> u16 {
        (self as u16) + 1
    }

    /// 由 code 还原；`0` 与未知 code 返回 `None`。
    #[must_use]
    pub const fn from_code(code: u16) -> Option<Self> {
        let index = (code as usize).checked_sub(1);
        match index {
            Some(value) if value < Self::ALL.len() => Some(Self::ALL[value]),
            _ => None,
        }
    }

    /// 单个元素的位宽。
    #[must_use]
    pub const fn bits(self) -> u32 {
        self.spec().bits
    }

    /// 规范小写名，用于档案、注解与诊断；与 [`DType::parse_name`] 双向。
    #[must_use]
    pub const fn name(self) -> &'static str {
        self.spec().name
    }

    /// Rust 变体名（与 [`Debug`] 输出一致），供代码生成与门禁对齐用。
    ///
    /// 缩写词用全大写（`bf16 → BF16`），因此**不能**由 [`DType::name`] 机械首字母大写推出：
    /// `bf16` 首字母大写会得到不存在的 `Bf16`。变体名与规范名住在同一行 `SPECS` 里，
    /// 两个事实源不可能分家——annotation 的 `#[model(dtype = "..")]` 与 `tests/contracts.rs`
    /// 都从这里取变体名，不再各自手写。
    #[must_use]
    pub const fn variant(self) -> &'static str {
        self.spec().variant
    }

    /// 由规范名还原；未知名字返回 `None`。
    ///
    /// 转调 [`Format::parse_name`]：解析的写法只有一份；inherent 这个方法只是让
    /// `DType::parse_name(..)` 这种路径调用不必先 `use llm_dtype::Format`（`annotation`
    /// 生成的代码与 `prof` 都按这个路径用）。
    #[must_use]
    pub fn parse_name(text: &str) -> Option<Self> {
        <Self as Format>::parse_name(text)
    }

    /// 是否为浮点类型（含 FP4/FP8 与 FNUZ 变体）。
    #[must_use]
    pub const fn is_float(self) -> bool {
        matches!(self.spec().class, Class::Float)
    }

    /// 是否为整数类型（布尔不归入）。
    #[must_use]
    pub const fn is_integer(self) -> bool {
        matches!(self.spec().class, Class::UInt | Class::SInt)
    }

    /// 是否为有符号整数。
    #[must_use]
    pub const fn is_signed(self) -> bool {
        matches!(self.spec().class, Class::SInt)
    }

    /// 是否为亚字节类型（位宽小于 8，存储按位打包）。
    #[must_use]
    pub const fn is_sub_byte(self) -> bool {
        self.bits() < 8
    }

    /// `numel` 个元素占用的字节数（亚字节按位向上取整）。
    ///
    /// 乘法溢出时**饱和**到 `u64::MAX`，不回绕：`numel` 可能来自权重文件头这类不受信任的
    /// 输入，回绕会把巨大张量报成很小的字节数，让"缓冲够长"的判定被绕过。
    ///
    /// ```
    /// assert_eq!(llm_dtype::DType::U4.bytes_for(6), 3);
    /// assert_eq!(llm_dtype::DType::F32.bytes_for(2), 8);
    /// ```
    #[must_use]
    pub const fn bytes_for(self, numel: u64) -> u64 {
        let total = numel.saturating_mul(self.bits() as u64);
        total.div_ceil(8)
    }
}

// `DType` 的泛型视角：与上面的 inherent `const fn` 读同一张 `SPECS`，只多一个判别值。
// `bytes_for` 留在 inherent 一侧——标量按位跨元素打包，与块格式的「整块补齐」不是同一条公式。
impl Format for DType {
    const ALL: &'static [Self] = DType::ALL;

    fn index(self) -> usize {
        self as usize
    }

    fn name(self) -> &'static str {
        DType::name(self)
    }

    fn bits(self) -> u32 {
        DType::bits(self)
    }
}

impl core::fmt::Display for DType {
    /// 输出规范小写名（与 [`DType::name`] 一致）。
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter.write_str(self.name())
    }
}

impl_from_str!(DType);
#[cfg(feature = "serde")]
impl_serde!(DType);

impl From<DType> for u16 {
    /// 稳定线上 code（与 [`DType::code`] 同一事实）。
    fn from(dtype: DType) -> Self {
        dtype.code()
    }
}

impl TryFrom<u16> for DType {
    type Error = UnknownCode;

    /// 由线上 code 还原；`0` 与未知 code 报 [`UnknownCode`]（与 [`DType::from_code`] 同一事实）。
    fn try_from(code: u16) -> Result<Self, Self::Error> {
        Self::from_code(code).ok_or(UnknownCode { code })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{format, mem::size_of};

    #[test]
    fn discriminant_fits_one_byte() {
        assert_eq!(size_of::<DType>(), 1, "repr(u8) 加 24 变体应正好一字节");
    }

    #[test]
    fn codes_are_a_stable_bijection() {
        for dtype in DType::ALL {
            assert_eq!(DType::from_code(dtype.code()), Some(*dtype));
        }
        // 边界由闭集长度推出，而不是写死 25：加类型时这里会自动挪到新的末位。
        let last = u16::try_from(DType::ALL.len()).expect("DType 变体数远小于 u16 上限");
        assert_eq!(DType::from_code(last), DType::ALL.last().copied());
        assert_eq!(DType::from_code(last + 1), None);
        assert_eq!(DType::from_code(0), None);
        assert_eq!(DType::from_code(u16::MAX), None);
    }

    #[test]
    fn every_u16_is_either_a_valid_code_or_rejected() {
        // 穷举全部 65536 个 code：合法集恰好是 `1..=ALL.len()`，且往返一致。
        let last = u16::try_from(DType::ALL.len()).expect("DType 变体数远小于 u16 上限");
        let valid = 1..=last;
        for code in u16::MIN..=u16::MAX {
            match DType::from_code(code) {
                Some(dtype) => {
                    assert!(valid.contains(&code), "{code} 不应被接受");
                    assert_eq!(dtype.code(), code);
                    assert_eq!(DType::try_from(code), Ok(dtype));
                }
                None => assert!(!valid.contains(&code), "{code} 不应被拒绝"),
            }
        }
    }

    #[test]
    fn from_str_uses_the_canonical_table() {
        for dtype in DType::ALL {
            assert_eq!(dtype.name().parse::<DType>(), Ok(*dtype));
        }
        assert_eq!("unknown".parse::<DType>(), Err(crate::format::UnknownName));
    }

    #[test]
    fn legacy_wire_codes_remain_unchanged() {
        // 与旧 weights::dtype::Wire 的判别值逐位对齐，保证容器字节兼容。
        const EXPECTED: [(DType, u16); 22] = [
            (DType::Bool, 1),
            (DType::U1, 2),
            (DType::U2, 3),
            (DType::U4, 4),
            (DType::U8, 5),
            (DType::U16, 6),
            (DType::U32, 7),
            (DType::U64, 8),
            (DType::I1, 9),
            (DType::I2, 10),
            (DType::I4, 11),
            (DType::I8, 12),
            (DType::I16, 13),
            (DType::I32, 14),
            (DType::I64, 15),
            (DType::F8E4M3, 16),
            (DType::F8E5M2, 17),
            (DType::F16, 18),
            (DType::BF16, 19),
            (DType::F32, 20),
            (DType::F64, 21),
            (DType::F4, 22),
        ];
        for (dtype, code) in EXPECTED {
            assert_eq!(dtype.code(), code, "{dtype:?} 的线上 code 漂移");
        }
    }

    #[test]
    fn names_round_trip() {
        for dtype in DType::ALL {
            assert_eq!(DType::parse_name(dtype.name()), Some(*dtype));
        }
        assert_eq!(DType::parse_name("unknown"), None);
    }

    #[test]
    fn names_and_variants_are_unique() {
        for (index, dtype) in DType::ALL.iter().enumerate() {
            for other in &DType::ALL[index + 1..] {
                assert_ne!(dtype.name(), other.name(), "规范名重复");
                assert_ne!(dtype.variant(), other.variant(), "变体名重复");
            }
        }
    }

    #[test]
    fn identity_table_is_locked() {
        // 逐行锁死身份事实：位宽或名字被误改时失败在这一行，而不是靠人眼扫表。
        const EXPECTED: [(&str, u32, &str, Class); 24] = [
            ("bool", 1, "Bool", Class::Bool),
            ("u1", 1, "U1", Class::UInt),
            ("u2", 2, "U2", Class::UInt),
            ("u4", 4, "U4", Class::UInt),
            ("u8", 8, "U8", Class::UInt),
            ("u16", 16, "U16", Class::UInt),
            ("u32", 32, "U32", Class::UInt),
            ("u64", 64, "U64", Class::UInt),
            ("i1", 1, "I1", Class::SInt),
            ("i2", 2, "I2", Class::SInt),
            ("i4", 4, "I4", Class::SInt),
            ("i8", 8, "I8", Class::SInt),
            ("i16", 16, "I16", Class::SInt),
            ("i32", 32, "I32", Class::SInt),
            ("i64", 64, "I64", Class::SInt),
            ("f8e4m3", 8, "F8E4M3", Class::Float),
            ("f8e5m2", 8, "F8E5M2", Class::Float),
            ("f16", 16, "F16", Class::Float),
            ("bf16", 16, "BF16", Class::Float),
            ("f32", 32, "F32", Class::Float),
            ("f64", 64, "F64", Class::Float),
            ("f4", 4, "F4", Class::Float),
            ("f8e4m3fnuz", 8, "F8E4M3FNUZ", Class::Float),
            ("f8e5m2fnuz", 8, "F8E5M2FNUZ", Class::Float),
        ];
        assert_eq!(EXPECTED.len(), DType::ALL.len());
        for (dtype, (name, bits, variant, class)) in DType::ALL.iter().zip(EXPECTED) {
            let spec = dtype.spec();
            assert_eq!(
                (spec.name, spec.bits, spec.variant, spec.class),
                (name, bits, variant, class),
                "{dtype:?} 的身份行被改动"
            );
        }
    }

    #[test]
    fn names_are_the_lowercase_form_of_variants() {
        // 规范名与变体名只差大小写，唯一例外是 `bool`/`Bool`：拼错（如 `i7`）不能全绿。
        for dtype in DType::ALL {
            if *dtype == DType::Bool {
                assert_eq!(dtype.name(), "bool");
                continue;
            }
            assert_eq!(
                dtype.name().to_ascii_uppercase(),
                dtype.variant(),
                "{dtype:?} 的规范名与变体名不一致"
            );
        }
    }

    #[test]
    fn variant_names_match_the_derived_debug_identifiers() {
        for dtype in DType::ALL {
            assert_eq!(
                dtype.variant(),
                format!("{dtype:?}"),
                "变体名与 Debug 输出必须一致，annotation 生成的 `DType::` 变体才编得过"
            );
        }
    }

    #[test]
    fn display_prints_the_canonical_name() {
        for dtype in DType::ALL {
            assert_eq!(format!("{dtype}"), dtype.name());
        }
    }

    #[test]
    fn wire_code_traits_agree_with_the_const_accessors() {
        for dtype in DType::ALL {
            assert_eq!(u16::from(*dtype), dtype.code());
            assert_eq!(DType::try_from(dtype.code()), Ok(*dtype));
        }
        assert_eq!(
            DType::try_from(0),
            Err(crate::format::UnknownCode { code: 0 })
        );
        let past_last = u16::try_from(DType::ALL.len()).expect("DType 变体数远小于 u16 上限") + 1;
        assert_eq!(
            DType::try_from(past_last),
            Err(crate::format::UnknownCode { code: past_last })
        );
    }

    #[test]
    fn table_rows_align_with_the_variant_order() {
        // 索引必须与判别值同序，表与枚举不能分家；整行相等，避免只对上名字。
        for (index, dtype) in DType::ALL.iter().enumerate() {
            assert_eq!(SPECS[index], dtype.spec());
            assert_eq!(SPECS[*dtype as usize], dtype.spec());
        }
    }

    #[test]
    fn widths_and_predicates_are_consistent() {
        assert_eq!(DType::U4.bytes_for(6), 3);
        assert_eq!(DType::U1.bytes_for(8), 1);
        assert_eq!(DType::I2.bytes_for(5), 2);
        assert_eq!(DType::F32.bytes_for(2), 8);
        assert!(DType::F4.is_float());
        assert!(DType::I4.is_signed() && DType::I4.is_sub_byte());
        assert!(DType::U8.is_integer() && !DType::U8.is_signed());
        assert!(DType::Bool.is_sub_byte() && !DType::Bool.is_integer());
        assert_eq!(DType::F64.bits(), 64);
        assert!(DType::F8E4M3FNUZ.is_float() && DType::F8E5M2FNUZ.is_float());
    }

    #[test]
    fn bytes_for_rounds_sub_byte_types_up() {
        // 逐类型核对：字节数 = ceil(numel * bits / 8)，亚字节不留半字节。
        for dtype in DType::ALL {
            for numel in 0..=3u64 {
                let want = (numel * u64::from(dtype.bits())).div_ceil(8);
                assert_eq!(dtype.bytes_for(numel), want, "{dtype:?} / {numel}");
            }
        }
    }

    #[test]
    fn floats_and_integers_partition_the_closed_set() {
        let floats = DType::ALL.iter().filter(|d| d.is_float()).count();
        let integers = DType::ALL.iter().filter(|d| d.is_integer()).count();
        assert_eq!(
            floats + integers + 1,
            DType::ALL.len(),
            "只剩 Bool 不属于两类"
        );
        assert_eq!((floats, integers), (9, 14));
        assert_eq!(
            DType::ALL.iter().filter(|d| d.is_signed()).count(),
            7,
            "有符号整数全在 I 族"
        );
    }
}
