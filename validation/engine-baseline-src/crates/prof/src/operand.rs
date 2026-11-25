//! 操作数、元素位宽闭集与一次计算的标量代价。

use crate::atom::Closed;

/// 元素位宽闭集（bit）。
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Bits {
    /// 4 位（打包的 int4 权重）。
    Four,
    /// 8 位。
    Eight,
    /// 16 位（f16 / bf16 / i16）。
    Sixteen,
    /// 32 位（f32 / i32）。
    ThirtyTwo,
    /// 64 位（f64 / i64）。
    SixtyFour,
    /// 2 位（打包的 int2 权重，每字节四个码）。
    Two,
}

/// 与 [`Bits::ALL`] 同序的名字表；下标即判别值，加变体时与 `ALL`、[`WIDTHS`] 三处对齐。
const NAMES: [&str; Bits::ALL.len()] = ["4", "8", "16", "32", "64", "2"];

/// 与 [`Bits::ALL`] 同序的位宽表；下标即判别值。
const WIDTHS: [u32; Bits::ALL.len()] = [4, 8, 16, 32, 64, 2];

/// `Bits` 的词汇表由 [`crate::atom::Closed`] 单点声明，`ALL` / `as_str` / `parse` 由
/// [`crate::inherent_names!`] 以同名固有项转发。
impl Closed for Bits {
    const ALL: &'static [Self] = &[
        Self::Four,
        Self::Eight,
        Self::Sixteen,
        Self::ThirtyTwo,
        Self::SixtyFour,
        Self::Two,
    ];

    fn as_str(self) -> &'static str {
        NAMES[self as usize]
    }

    fn idx(self) -> usize {
        self as usize
    }
}

crate::inherent_names!(Bits);

impl Bits {
    /// 每元素占多少位。
    pub const fn bits(self) -> u32 {
        WIDTHS[self as usize]
    }

    /// 给定元素个数需要多少字节，**向上取整**。
    pub const fn bytes_for(self, elements: u128) -> u128 {
        (elements * self.bits() as u128).div_ceil(8)
    }
}

// ---------------------------------------------------------------------------
// 操作数：名字、元素位宽与读/写方向。

/// 操作数的数据流向。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Access {
    /// 只读：算子入口的输入。
    Read,
    /// 只写：算子出口的输出。写入同样计费，因为"至少写一遍输出"是逃不掉的流量。
    Write,
}

/// 表达式里一个操作数的**编译期**形态：名字、元素位宽与读/写方向。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OperandSpec {
    /// 缓冲名（与 `.k` 的形参名一致，也是诊断里显示的名字）。
    pub name: &'static str,
    /// 元素位宽。
    pub bits: Bits,
    /// 读写方向。
    pub access: Access,
}

impl OperandSpec {
    /// 只读操作数。
    #[must_use]
    pub const fn read(name: &'static str, bits: Bits) -> Self {
        Self {
            name,
            bits,
            access: Access::Read,
        }
    }

    /// 只写操作数。
    #[must_use]
    pub const fn write(name: &'static str, bits: Bits) -> Self {
        Self {
            name,
            bits,
            access: Access::Write,
        }
    }
}

/// f32 一读一写的标准操作数：`a` 读、`y` 写（一元与归约共用同一份）。
#[must_use]
pub const fn f32_io_operands() -> [OperandSpec; 2] {
    [
        OperandSpec::read("a", Bits::ThirtyTwo),
        OperandSpec::write("y", Bits::ThirtyTwo),
    ]
}

/// f16 一读一写的标准操作数：`a` 读、`y` 写，元素 16 位（半精度一元/归约共用）。
#[must_use]
pub const fn f16_io_operands() -> [OperandSpec; 2] {
    [
        OperandSpec::read("a", Bits::Sixteen),
        OperandSpec::write("y", Bits::Sixteen),
    ]
}

/// bf16 一读一写的标准操作数：`a` 读、`y` 写，元素 16 位（brain 浮点一元/归约共用）。
#[must_use]
pub const fn bf16_io_operands() -> [OperandSpec; 2] {
    [
        OperandSpec::read("a", Bits::Sixteen),
        OperandSpec::write("y", Bits::Sixteen),
    ]
}

/// f32 两读一写的标准操作数：`a`、`b` 读、`y` 写（二元逐元素）。
#[must_use]
pub const fn f32_binary_operands() -> [OperandSpec; 3] {
    [
        OperandSpec::read("a", Bits::ThirtyTwo),
        OperandSpec::read("b", Bits::ThirtyTwo),
        OperandSpec::write("y", Bits::ThirtyTwo),
    ]
}

/// 一个操作数：名字、元素位宽与读/写方向。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Operand {
    name: String,
    bits: Bits,
    access: Access,
}

impl Operand {
    /// 只读操作数。
    pub fn read(name: impl Into<String>, bits: Bits) -> Self {
        Self {
            name: name.into(),
            bits,
            access: Access::Read,
        }
    }

    /// 只写操作数。
    pub fn write(name: impl Into<String>, bits: Bits) -> Self {
        Self {
            name: name.into(),
            bits,
            access: Access::Write,
        }
    }

    /// 按方向构造（方向来自数据时用这个）。
    pub fn new(name: impl Into<String>, bits: Bits, access: Access) -> Self {
        Self {
            name: name.into(),
            bits,
            access,
        }
    }

    /// 名字。
    pub fn name(&self) -> &str {
        &self.name
    }

    /// 元素位宽。
    pub fn bits(&self) -> Bits {
        self.bits
    }

    /// 流向。
    pub fn access(&self) -> Access {
        self.access
    }
}

// ---------------------------------------------------------------------------
// 一次计算的标量代价，roofline 的分子。

/// 一次计算的标量代价，roofline 的分子（cutile README §12.1）。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Cost {
    /// 浮点运算次数。
    pub flops: f64,
    /// 最少读入字节：算子入口的输入。
    pub read: u64,
    /// 最少写出字节：算子出口的输出。
    pub write: u64,
}

impl Cost {
    /// 零代价：空的表达式或纯标量计算。
    pub const ZERO: Self = Self {
        flops: 0.0,
        read: 0,
        write: 0,
    };

    /// 由三个分量构造（便于测试与适配旧的调用点）。
    pub const fn new(flops: f64, read: u64, write: u64) -> Self {
        Self { flops, read, write }
    }

    /// 最少流量（字节）：读与写合计。
    pub fn bytes(&self) -> u64 {
        self.read.saturating_add(self.write)
    }

    /// 两笔代价相加：复合算子的总代价。
    pub fn plus(self, other: Self) -> Self {
        Self {
            flops: self.flops + other.flops,
            read: self.read.saturating_add(other.read),
            write: self.write.saturating_add(other.write),
        }
    }
}
