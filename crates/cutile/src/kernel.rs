//! Typed, target-neutral IR for the `#[dsl]` kernel family.
//!
//! 一张 [`Kernel`] 覆盖三种写法：逐点 `output[idx] = value`、归约 `output[at] += value`、以及由
//! 前两者组合出的融合核（[`Kernel::fuse`]）。这一层做校验、规范化、常量折叠、死形参消除，并发射
//! 两种目标文本：可移植 C（[`Kernel::emit_c`]）与设备 lowering 的 `.k` 正文（[`Kernel::emit_k`]，
//! 经 `shard::ir` 桥接入 `Lower`/设备 dispatch）。`Image` 尚未接上，所以「性能提升」只能由桥接后
//! 的设备路径宣称，不由这一层单独宣称。

use core::fmt;

use crate::pass::Pipeline;
use crate::validate::verify_core;

/// Scalar element types represented by the initial pointwise IR.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScalarType {
    /// IEEE-754 single precision.
    F32,
    /// Rust `usize`, used for iteration extents.
    Usize,
}

impl ScalarType {
    /// C spelling used in a kernel signature.
    fn c_name(self) -> &'static str {
        match self {
            ScalarType::F32 => "float",
            ScalarType::Usize => "size_t",
        }
    }

    /// Whether this type may be held in a buffer parameter. `usize` is an extent scalar only.
    pub(crate) fn buf_ok(self) -> bool {
        matches!(self, ScalarType::F32)
    }
}

/// Buf access required by a kernel parameter.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Access {
    /// Read-only buffer.
    Read,
    /// Mutable buffer, which may be read and written.
    ReadWrite,
}

impl Access {
    /// C pointer declaration for a buffer of the given element spelling, held with this access.
    fn c_pointer(self, element: &str) -> String {
        match self {
            Access::Read => format!("const {element} *"),
            Access::ReadWrite => format!("{element} *"),
        }
    }
}

/// A checked value type in the kernel signature.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ValueType {
    /// Scalar value.
    Scalar(ScalarType),
    /// Contiguous buffer and its element/access contract.
    Buf {
        /// Buf element type.
        element: ScalarType,
        /// Required access.
        access: Access,
    },
}

impl ValueType {
    /// C declaration for this parameter, or `None` when the element type has no buffer spelling.
    pub(crate) fn c_type(self) -> Option<String> {
        match self {
            ValueType::Scalar(element) => Some(element.c_name().to_string()),
            ValueType::Buf { element, access } => {
                element.buf_ok().then(|| access.c_pointer(element.c_name()))
            }
        }
    }

    pub(crate) fn is_buf(self) -> bool {
        matches!(self, ValueType::Buf { .. })
    }

    /// 标量总能发射；缓冲元素必须在 [`ScalarType::buf_ok`] 闭集内，否则校验阶段就打回。
    pub(crate) fn element_ok(self) -> bool {
        match self {
            ValueType::Scalar(_) => true,
            ValueType::Buf { element, .. } => element.buf_ok(),
        }
    }
}

/// A named kernel parameter.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Parameter {
    /// Rust parameter name.
    pub name: String,
    /// Scalar or buffer type.
    pub ty: ValueType,
}

/// Arithmetic operations supported by the pointwise expression IR.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Arith {
    /// Addition.
    Add,
    /// Subtraction.
    Sub,
    /// Multiplication.
    Mul,
    /// Division.
    Div,
}

impl Arith {
    /// Infix symbol used when emitting C.
    pub(crate) fn c_symbol(self) -> &'static str {
        match self {
            Arith::Add => "+",
            Arith::Sub => "-",
            Arith::Mul => "*",
            Arith::Div => "/",
        }
    }

    /// Apply the operation to two `f32` operands.
    pub(crate) fn apply(self, left: f32, right: f32) -> f32 {
        match self {
            Arith::Add => left + right,
            Arith::Sub => left - right,
            Arith::Mul => left * right,
            Arith::Div => left / right,
        }
    }
}

/// 一元 f32 数学 / 取整组合子闭集。
///
/// 每个算子只在这一处登记 C 发射名与 Rust 主机求值；校验、遍历、折叠、解释执行都从这张表取，
/// 不为单个变体各写一份 match 臂。结果仍是 `f32`——比较 / 选择这类出布尔值的组合子不在本闭集。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnaryFn {
    /// `|x|`
    Abs,
    /// `√x`
    Sqrt,
    /// `e^x`
    Exp,
    /// `ln x`
    Ln,
    /// `sin x`
    Sin,
    /// `cos x`
    Cos,
    /// 向正无穷取整。
    Ceil,
    /// 向负无穷取整。
    Floor,
    /// 四舍五入到最近整数，半入远离零。
    Round,
    /// 截断到整数。
    Trunc,
}

impl UnaryFn {
    /// C `<math.h>` 函数名。
    pub(crate) const fn c_name(self) -> &'static str {
        match self {
            UnaryFn::Abs => "fabsf",
            UnaryFn::Sqrt => "sqrtf",
            UnaryFn::Exp => "expf",
            UnaryFn::Ln => "logf",
            UnaryFn::Sin => "sinf",
            UnaryFn::Cos => "cosf",
            UnaryFn::Ceil => "ceilf",
            UnaryFn::Floor => "floorf",
            UnaryFn::Round => "roundf",
            UnaryFn::Trunc => "truncf",
        }
    }

    /// 主机侧 f32 求值：解释执行、字面量折叠与参考对拍共用同一口径，避免两边各写一套数学定义。
    pub fn apply(self, value: f32) -> f32 {
        match self {
            UnaryFn::Abs => value.abs(),
            UnaryFn::Sqrt => value.sqrt(),
            UnaryFn::Exp => value.exp(),
            UnaryFn::Ln => value.ln(),
            UnaryFn::Sin => value.sin(),
            UnaryFn::Cos => value.cos(),
            UnaryFn::Ceil => value.ceil(),
            UnaryFn::Floor => value.floor(),
            UnaryFn::Round => value.round(),
            UnaryFn::Trunc => value.trunc(),
        }
    }
}

/// Typed expression nodes. Buf loads use the kernel's implicit loop index.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Expr {
    /// An f32 literal stored by its exact IEEE-754 bit pattern.
    F32(u32),
    /// An f32 scalar parameter.
    Scalar(String),
    /// Load one f32 value from a buffer at the current loop index.
    Load(String),
    /// Negate an f32 expression.
    Neg(Box<Expr>),
    /// A unary math / rounding combinator over one f32 expression.
    Unary {
        /// Registered unary operator.
        op: UnaryFn,
        /// Operand.
        child: Box<Expr>,
    },
    /// Binary f32 arithmetic.
    Binary {
        /// Operation.
        op: Arith,
        /// Left operand.
        left: Box<Expr>,
        /// Right operand.
        right: Box<Expr>,
    },
    /// Fused multiply-add `a * b + c`, evaluated with a single rounding.
    Fma {
        /// Multiplicand.
        a: Box<Expr>,
        /// Multiplier.
        b: Box<Expr>,
        /// Addend.
        c: Box<Expr>,
    },
    /// Element-wise minimum of two f32 expressions.
    Min {
        /// Left operand.
        left: Box<Expr>,
        /// Right operand.
        right: Box<Expr>,
    },
    /// Element-wise maximum of two f32 expressions.
    Max {
        /// Left operand.
        left: Box<Expr>,
        /// Right operand.
        right: Box<Expr>,
    },
}

/// Verification failure for malformed or unsupported typed IR.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VerificationError {
    message: String,
}

impl VerificationError {
    pub(crate) fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

impl fmt::Display for VerificationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.message)
    }
}

impl std::error::Error for VerificationError {}

/// What a kernel stores on every iteration.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Write {
    /// Pointwise store: `output[idx] = value`.
    Map(Expr),
    /// Reduction: `output[at] += value`.
    Reduce {
        /// Constant slot accumulated into.
        at: u32,
        /// Value added to the accumulator on each iteration.
        value: Expr,
    },
}

impl Write {
    /// Expression evaluated once per iteration.
    pub(crate) fn expr(&self) -> &Expr {
        match self {
            Write::Map(value) => value,
            Write::Reduce { value, .. } => value,
        }
    }

    /// Mutable access to the expression evaluated once per iteration.
    pub(crate) fn expr_mut(&mut self) -> &mut Expr {
        match self {
            Write::Map(value) => value,
            Write::Reduce { value, .. } => value,
        }
    }

    /// Fixed slot a reduction accumulates into; `None` for pointwise stores.
    pub(crate) fn slot(&self) -> Option<u32> {
        match self {
            Write::Reduce { at, .. } => Some(*at),
            Write::Map(_) => None,
        }
    }
}

/// One verified kernel: a single loop that stores into `output` as described by [`Write`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Kernel {
    /// Kernel name.
    pub name: String,
    /// Ordered scalar and buffer parameters.
    pub params: Vec<Parameter>,
    /// Loop index variable.
    pub idx: String,
    /// Name of the `usize` parameter providing the loop bound.
    pub extent: String,
    /// Name of the mutable output buffer.
    pub output: String,
    /// Store performed on every iteration.
    pub write: Write,
}

impl Kernel {
    /// Construct a kernel from an explicit [`Write`].
    pub fn new(
        name: impl Into<String>,
        params: Vec<Parameter>,
        idx: impl Into<String>,
        extent: impl Into<String>,
        output: impl Into<String>,
        write: Write,
    ) -> Self {
        Self {
            name: name.into(),
            params,
            idx: idx.into(),
            extent: extent.into(),
            output: output.into(),
            write,
        }
    }

    /// Construct a pointwise kernel: `output[idx] = value`.
    pub fn map(
        name: impl Into<String>,
        params: Vec<Parameter>,
        idx: impl Into<String>,
        extent: impl Into<String>,
        output: impl Into<String>,
        value: Expr,
    ) -> Self {
        Self::new(name, params, idx, extent, output, Write::Map(value))
    }

    /// Construct a reduction kernel: `output[at] += value`.
    pub fn reduce(
        name: impl Into<String>,
        params: Vec<Parameter>,
        idx: impl Into<String>,
        extent: impl Into<String>,
        output: impl Into<String>,
        at: u32,
        value: Expr,
    ) -> Self {
        Self::new(
            name,
            params,
            idx,
            extent,
            output,
            Write::Reduce { at, value },
        )
    }

    /// Validate names, parameter types, memory access, and expression types.
    pub fn verify(&self) -> Result<(), VerificationError> {
        verify_core(
            &self.name,
            &self.idx,
            &self.params,
            &self.extent,
            &self.output,
            self.write.expr(),
        )
    }

    /// Run the standard pipeline to a fixpoint: verify, then simplify (algebraic laws + constant
    /// folding), then drop dead parameters, sweeping until a whole round leaves the kernel
    /// unchanged, then verify again. The result is a valid kernel in normal form.
    ///
    /// 轮次不用魔法数：上限是初始 `cost`（形参数 + 节点数），严格收缩的 pass 必然在此之前收敛，耗尽
    /// 即视为流水线震荡并报错。
    pub fn optimize(&self) -> Result<Self, VerificationError> {
        Pipeline::standard().run(self.clone())
    }

    /// Run the contracted pipeline: standard simplification plus multiply-add contraction
    /// (`a * b + c` → a single fused multiply-add). Contraction changes rounding, so it is opt-in;
    /// use it for inner-product / affine kernels that want the fused instruction.
    pub fn optimize_contracted(&self) -> Result<Self, VerificationError> {
        Pipeline::contracted().run(self.clone())
    }

    /// Emit portable C; return status 1 when a guarded length is violated.
    pub fn emit_c(&self) -> Result<String, VerificationError> {
        crate::printer::render(self, crate::printer::Target::C)
    }

    /// Emit the `.k`-format serial body the device lowering seam consumes (the `shard::ir` bridge
    /// feeds it to the vendor backends' `Codegen::lower`): a `void` entry
    /// `cutile_{name}_kernel` with plain `const float *`/`float *`/`int` parameters and a single
    /// guard-free loop. Unlike [`Self::emit_c`], it pairs no buffer lengths, emits no guards and no
    /// `#include` — the vendor backend wraps this body into its grid / chunk / serial skeleton.
    pub fn emit_k(&self) -> Result<String, VerificationError> {
        crate::printer::render(self, crate::printer::Target::K)
    }
}

#[cfg(test)]
#[path = "../tests/ir_tests.rs"]
mod ir_tests;

#[cfg(test)]
#[path = "../tests/unary_tests.rs"]
mod unary_tests;
