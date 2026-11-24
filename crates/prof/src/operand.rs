//! 操作数：名字、元素位宽与读/写方向。

use super::bits::Bits;

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
