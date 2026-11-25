//! 全精度闭集 [`DType`]：全仓库元素类型的唯一事实源。
//!
//! 各层不再手写类型枚举与跨层转换，需要表达"本层只支持其中一部分"时用子集切片或位集，
//! 而不是另造一个 enum。统一的三套旧枚举是 `tensor::DType`（5 变体）、`prof::atom::DType`
//! （9 变体）与 `weights::dtype::Wire`（21 变体）。
//!
//! # 线上 code 的稳定性
//!
//! [`DType::code`] 的 `1..=21` 与旧 `Wire` 的判别值逐位相同，既有容器字节完全兼容；
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

/// 浮点格式的位域切分：符号位恒为 1，其余由指数位宽、尾数位宽与指数偏置决定。
///
/// 这是 [`DTypeSpec`] 与 [`crate::mxtype::ElementSpec`] 的一列，位域因此和规范名、位宽一样
/// 只有一个事实源：FP4/FP8 的指数/尾数切分不必在下游反量化里硬编码，规范漂移只改一行。
///
/// ```
/// let layout = llm_dtype::DType::BF16.float_layout().expect("bf16 是浮点");
/// assert_eq!(
///     (layout.exp_bits, layout.mantissa_bits, layout.exponent_bias),
///     (8, 7, 127)
/// );
/// ```
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub struct FloatLayout {
    /// 指数字段位宽。
    pub exp_bits: u32,
    /// 尾数字段位宽（不含隐含的整数位）。
    pub mantissa_bits: u32,
    /// 指数偏置：真指数 = 位域值 − 偏置；FNUZ 变体比同形 IEEE 浮点多 1。
    pub exponent_bias: u32,
}

impl FloatLayout {
    /// 按位域宽度与指数偏置构造；符号位不在这里，恒为 1。
    #[must_use]
    pub const fn new(exp_bits: u32, mantissa_bits: u32, exponent_bias: u32) -> Self {
        Self {
            exp_bits,
            mantissa_bits,
            exponent_bias,
        }
    }

    /// 全宽：符号 1 位 + 指数 + 尾数。恒等于所在格式的位宽。
    ///
    /// 符号位不在字段里（恒为 1），所以全宽是派生量而不是另一列；`bits()` 是三段位域的
    /// 唯一求和公式，测试里不再手写 `1 + exp + mantissa`。
    ///
    /// ```
    /// use llm_dtype::DType;
    ///
    /// assert_eq!(DType::BF16.float_layout().unwrap().bits(), 16);
    /// assert_eq!(DType::F4.float_layout().unwrap().bits(), 4);
    /// ```
    #[must_use]
    pub const fn bits(self) -> u32 {
        1 + self.exp_bits + self.mantissa_bits
    }
}

/// 一个标量类型的身份与布局：规范名、Rust 变体名、位宽、数值类别与浮点位域。
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
    /// 浮点位域；非浮点类型为 `None`（与 `class == Float` 同真同假）。
    pub float: Option<FloatLayout>,
}

/// 元素类型闭集：布尔、无符号/有符号整数、浮点与亚字节打包类型。
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
const SPECS: [DTypeSpec; DType::ALL.len()] = [
    DTypeSpec {
        name: "bool",
        variant: "Bool",
        bits: 1,
        class: Class::Bool,
        float: None,
    },
    DTypeSpec {
        name: "u1",
        variant: "U1",
        bits: 1,
        class: Class::UInt,
        float: None,
    },
    DTypeSpec {
        name: "u2",
        variant: "U2",
        bits: 2,
        class: Class::UInt,
        float: None,
    },
    DTypeSpec {
        name: "u4",
        variant: "U4",
        bits: 4,
        class: Class::UInt,
        float: None,
    },
    DTypeSpec {
        name: "u8",
        variant: "U8",
        bits: 8,
        class: Class::UInt,
        float: None,
    },
    DTypeSpec {
        name: "u16",
        variant: "U16",
        bits: 16,
        class: Class::UInt,
        float: None,
    },
    DTypeSpec {
        name: "u32",
        variant: "U32",
        bits: 32,
        class: Class::UInt,
        float: None,
    },
    DTypeSpec {
        name: "u64",
        variant: "U64",
        bits: 64,
        class: Class::UInt,
        float: None,
    },
    DTypeSpec {
        name: "i1",
        variant: "I1",
        bits: 1,
        class: Class::SInt,
        float: None,
    },
    DTypeSpec {
        name: "i2",
        variant: "I2",
        bits: 2,
        class: Class::SInt,
        float: None,
    },
    DTypeSpec {
        name: "i4",
        variant: "I4",
        bits: 4,
        class: Class::SInt,
        float: None,
    },
    DTypeSpec {
        name: "i8",
        variant: "I8",
        bits: 8,
        class: Class::SInt,
        float: None,
    },
    DTypeSpec {
        name: "i16",
        variant: "I16",
        bits: 16,
        class: Class::SInt,
        float: None,
    },
    DTypeSpec {
        name: "i32",
        variant: "I32",
        bits: 32,
        class: Class::SInt,
        float: None,
    },
    DTypeSpec {
        name: "i64",
        variant: "I64",
        bits: 64,
        class: Class::SInt,
        float: None,
    },
    DTypeSpec {
        name: "f8e4m3",
        variant: "F8E4M3",
        bits: 8,
        class: Class::Float,
        float: Some(FloatLayout::new(4, 3, 7)),
    },
    DTypeSpec {
        name: "f8e5m2",
        variant: "F8E5M2",
        bits: 8,
        class: Class::Float,
        float: Some(FloatLayout::new(5, 2, 15)),
    },
    DTypeSpec {
        name: "f16",
        variant: "F16",
        bits: 16,
        class: Class::Float,
        float: Some(FloatLayout::new(5, 10, 15)),
    },
    DTypeSpec {
        name: "bf16",
        variant: "BF16",
        bits: 16,
        class: Class::Float,
        float: Some(FloatLayout::new(8, 7, 127)),
    },
    DTypeSpec {
        name: "f32",
        variant: "F32",
        bits: 32,
        class: Class::Float,
        float: Some(FloatLayout::new(8, 23, 127)),
    },
    DTypeSpec {
        name: "f64",
        variant: "F64",
        bits: 64,
        class: Class::Float,
        float: Some(FloatLayout::new(11, 52, 1023)),
    },
    DTypeSpec {
        name: "f4",
        variant: "F4",
        bits: 4,
        class: Class::Float,
        float: Some(FloatLayout::new(2, 1, 1)),
    },
    DTypeSpec {
        name: "f8e4m3fnuz",
        variant: "F8E4M3FNUZ",
        bits: 8,
        class: Class::Float,
        float: Some(FloatLayout::new(4, 3, 8)),
    },
    DTypeSpec {
        name: "f8e5m2fnuz",
        variant: "F8E5M2FNUZ",
        bits: 8,
        class: Class::Float,
        float: Some(FloatLayout::new(5, 2, 16)),
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
    #[must_use]
    pub const fn code(self) -> u16 {
        (self as u16) + 1
    }

    /// 由 code 还原；`0` 与未知 code 返回 `None`。
    #[must_use]
    pub const fn from_code(code: u16) -> Option<Self> {
        let idx = (code as usize).checked_sub(1);
        match idx {
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
    #[must_use]
    pub const fn variant(self) -> &'static str {
        self.spec().variant
    }

    /// 浮点位域（指数/尾数位宽与偏置）；非浮点类型为 `None`。
    ///
    /// 与 [`DType::is_float`] 同真同假：浮点 ⟺ `Some`，由测试跨列锁住。
    #[must_use]
    pub const fn float_layout(self) -> Option<FloatLayout> {
        self.spec().float
    }

    /// 指数字段位宽；非浮点类型为 `None`。
    #[must_use]
    pub const fn exp_bits(self) -> Option<u32> {
        match self.spec().float {
            Some(layout) => Some(layout.exp_bits),
            None => None,
        }
    }

    /// 尾数字段位宽（不含隐含的整数位）；非浮点类型为 `None`。
    #[must_use]
    pub const fn mantissa_bits(self) -> Option<u32> {
        match self.spec().float {
            Some(layout) => Some(layout.mantissa_bits),
            None => None,
        }
    }

    /// 指数偏置；非浮点类型为 `None`。
    #[must_use]
    pub const fn exponent_bias(self) -> Option<u32> {
        match self.spec().float {
            Some(layout) => Some(layout.exponent_bias),
            None => None,
        }
    }

    /// 由规范名还原；未知名字返回 `None`。
    #[must_use]
    pub fn parse_name(text: &str) -> Option<Self> {
        <Self as Format>::parse_name(text)
    }

    /// 是否为布尔（1 位，但既不是整数也不是浮点）。
    #[must_use]
    pub const fn is_bool(self) -> bool {
        matches!(self.spec().class, Class::Bool)
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

    /// 是否为无符号整数（布尔不归入，与 [`DType::is_integer`] 同口径）。
    #[must_use]
    pub const fn is_unsigned(self) -> bool {
        matches!(self.spec().class, Class::UInt)
    }

    /// 是否为有符号整数。
    #[must_use]
    pub const fn is_signed(self) -> bool {
        matches!(self.spec().class, Class::SInt)
    }

    /// 符号位宽：浮点与有符号整数各占 1 位，布尔与无符号整数为 0。
    ///
    /// 对浮点，[`FloatLayout::bits`]（`sign_bits + exp_bits + mantissa_bits`）恒等于
    /// [`DType::bits`]。
    #[must_use]
    pub const fn sign_bits(self) -> u32 {
        match self.spec().class {
            Class::Float | Class::SInt => 1,
            Class::Bool | Class::UInt => 0,
        }
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

    fn idx(self) -> usize {
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
#[path = "../tests/dtype_tests.rs"]
mod tests;
