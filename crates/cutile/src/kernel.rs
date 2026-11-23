//! Typed, target-neutral IR for the `#[dsl]` kernel family.
//!
//! 一张 [`Kernel`] 覆盖三种写法：逐点 `output[idx] = value`、归约 `output[at] += value`、以及由
//! 前两者组合出的融合核（[`Kernel::fuse`]）。这一层做校验、规范化、常量折叠、死形参消除，并发射
//! 两种目标文本：可移植 C（[`Kernel::emit_c`]）与设备 lowering 的 `.k` 正文（[`Kernel::emit_k`]，
//! 经 `shard::ir` 桥接入 `Lower`/设备 dispatch）。`Image` 尚未接上，所以「性能提升」只能由桥接后
//! 的设备路径宣称，不由这一层单独宣称。

use core::fmt::{self, Write as _};

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
    fn c_type(self) -> Option<String> {
        match self {
            ValueType::Scalar(element) => Some(element.c_name().to_string()),
            ValueType::Buf { element, access } => element
                .buf_ok()
                .then(|| access.c_pointer(element.c_name())),
        }
    }

    pub(crate) fn is_buf(self) -> bool {
        matches!(self, ValueType::Buf { .. })
    }

    /// 标量总能发射；缓冲元素必须在 [`ScalarType::buffer_ok`] 闭集内，否则校验阶段就打回。
    pub(crate) fn element_ok(self) -> bool {
        match self {
            ValueType::Scalar(_) => true,
            ValueType::Buf { element, .. } => element.buffer_ok(),
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
    fn c_symbol(self) -> &'static str {
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
    /// Absolute value of an f32 expression.
    Abs(Box<Expr>),
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
        self.emit(Target::C)
    }

    /// Emit the `.k`-format serial body the device lowering seam consumes (the `shard::ir` bridge
    /// feeds it to the vendor backends' `Codegen::lower`): a `void` entry
    /// `cutile_{name}_kernel` with plain `const float *`/`float *`/`int` parameters and a single
    /// guard-free loop. Unlike [`Self::emit_c`], it pairs no buffer lengths, emits no guards and no
    /// `#include` — the vendor backend wraps this body into its grid / chunk / serial skeleton.
    pub fn emit_k(&self) -> Result<String, VerificationError> {
        self.emit(Target::K)
    }

    /// 两种目标共用的骨架：先优化到范式，再拼签名、守卫、循环。
    fn emit(&self, target: Target) -> Result<String, VerificationError> {
        let kernel = self.optimize()?;
        let mut source = String::new();
        target.prologue(&mut source);
        write_signature(&mut source, target, &kernel.name, &kernel.params)?;
        target.guards(&mut source, &kernel);
        write_loop(&mut source, target, &kernel);
        Ok(source)
    }
}

/// 发射目标。两边的正文是同一段循环，只差四处——入口签名、长度配对与守卫、循环变量类型、
/// 循环收尾。把这四处收进一个枚举，正文就只写一遍；少一份副本就少一处「改了一边忘了另一边」。
#[derive(Clone, Copy)]
enum Target {
    /// 可移植 C：配对 `size_t` 长度、带守卫、`return 0` 收尾。
    C,
    /// 设备 `.k` 正文：无长度配对、无守卫，入口与返回交给 vendor 骨架。
    K,
}

impl Target {
    /// 循环变量的 C 拼写：C 用 `size_t` 对齐 `size_t` 长度；`.k` 用 `int` 对齐 `int` 长度标量，
    /// 规避有符号 / 无符号比较。
    fn int(self) -> &'static str {
        match self {
            Target::C => "size_t",
            Target::K => "int",
        }
    }

    /// 入口签名前段：C 用 `int` 原名，`.k` 用 `void` 加前缀名。
    fn entry(self, name: &str) -> String {
        match self {
            Target::C => format!("int {name}("),
            Target::K => format!("void cutile_{name}_kernel("),
        }
    }

    /// 形参的 C 拼写。`.k` 把 `usize` 长度标量收敛成 `int`（签名解析认 `int`），其余同
    /// [`ValueType::c_type`]。
    fn declaration(self, ty: ValueType) -> Option<String> {
        match (self, ty) {
            (Target::K, ValueType::Scalar(ScalarType::Usize)) => Some("int".to_string()),
            _ => ty.c_type(),
        }
    }

    /// 是否给每个缓冲补一个隐式长度参数。只有 C 需要：设备后端的长度就是标量形参本身。
    fn pairs_lengths(self) -> bool {
        matches!(self, Target::C)
    }

    /// 头文件。`.k` 不要——vendor 骨架已经 include 过。`<math.h>` 供 `fabsf` / `fminf` / `fmaxf`。
    fn prologue(self, source: &mut String) {
        if let Target::C = self {
            source.push_str("#include <stddef.h>\n#include <math.h>\n");
        }
    }

    /// 越界守卫。`.k` 不要——设备骨架接管入口与返回，没有 `return 1` 可写。
    fn guards(self, source: &mut String, kernel: &Kernel) {
        if let Target::C = self {
            write_guards(source, kernel);
        }
    }

    /// 循环收尾：C 的入口要自己的返回值，`.k` 的返回交给骨架。
    fn trailer(self) -> &'static str {
        match self {
            Target::C => ";\n    }\n    return 0;\n}\n",
            Target::K => ";\n    }\n}\n",
        }
    }
}

/// 形参声明，C 还给每个缓冲补一个隐式长度参数：`buffer` 与其长度成对出现，顺序与
/// `check_generated` 对齐。
fn write_signature(
    source: &mut String,
    target: Target,
    name: &str,
    params: &[Parameter],
) -> Result<(), VerificationError> {
    source.push_str(&target.entry(name));
    let mut bufs = 0;
    for (idx, parameter) in params.iter().enumerate() {
        if idx != 0 {
            source.push_str(", ");
        }
        bufs += write_parameter(source, target, parameter, bufs)?;
    }
    source.push_str(") {\n");
    Ok(())
}

fn write_parameter(
    source: &mut String,
    target: Target,
    parameter: &Parameter,
    bufs: usize,
) -> Result<usize, VerificationError> {
    let Some(declaration) = target.declaration(parameter.ty) else {
        return Err(VerificationError::new(format!(
            "unsupported buffer type for `{}`",
            parameter.name
        )));
    };
    let separator = if declaration.ends_with('*') { "" } else { " " };
    let _ = write!(source, "{declaration}{separator}{}", parameter.name);
    if !(target.pairs_lengths() && parameter.ty.is_buf()) {
        return Ok(0);
    }
    let _ = write!(source, ", size_t {}", buf_length_name(bufs));
    Ok(1)
}

/// 归约只碰 `output[at]` 一个槽：输出守卫看 `at` 是否越界，其余缓冲仍看 `extent`。
fn write_guards(source: &mut String, kernel: &Kernel) {
    let slot = kernel.write.slot();
    let bufs = kernel
        .params
        .iter()
        .filter(|parameter| parameter.ty.is_buf());
    for (idx, parameter) in bufs.enumerate() {
        let guard = match slot {
            Some(at) if parameter.name == kernel.output => format!("{at} >="),
            _ => format!("{} >", kernel.extent),
        };
        let _ = writeln!(
            source,
            "    if ({guard} {}) return 1;",
            buf_length_name(idx)
        );
    }
}

/// 循环正文：头、写回值、收尾。两种目标共用同一份正文，差异全在 `target` 里。
fn write_loop(source: &mut String, target: Target, kernel: &Kernel) {
    write_header(source, target, kernel, kernel.write.slot());
    write_expr(source, kernel.write.expr(), &kernel.idx);
    source.push_str(target.trailer());
}

/// 循环头加上写回语句的左半段：循环变量类型由 `target` 定，归约多一个常量槽。
fn write_header(source: &mut String, target: Target, kernel: &Kernel, at: Option<u32>) {
    let ty = target.int();
    let Kernel {
        idx,
        extent,
        output,
        ..
    } = kernel;
    let _ = write!(
        source,
        "    for ({ty} {idx} = 0; {idx} < {extent}; ++{idx}) {{\n        {output}["
    );
    match at {
        Some(at) => {
            let _ = write!(source, "{at}] += ");
        }
        None => {
            let _ = write!(source, "{idx}] = ");
        }
    }
}

pub(crate) fn buf_length_name(buf: usize) -> String {
    format!("cutile_len_{buf}")
}

fn write_expr(source: &mut String, expression: &Expr, idx: &str) {
    match expression {
        Expr::F32(bits) => {
            let value = f32::from_bits(*bits);
            let _ = write!(source, "{value:?}f");
        }
        Expr::Scalar(name) => source.push_str(name),
        Expr::Load(name) => {
            let _ = write!(source, "{name}[{idx}]");
        }
        other => write_compound(source, other, idx),
    }
}

fn write_compound(source: &mut String, expression: &Expr, idx: &str) {
    match expression {
        Expr::Neg(value) => {
            source.push_str("-(");
            write_expr(source, value, idx);
            source.push(')');
        }
        Expr::Abs(value) => write_call(source, "fabsf", &[value.as_ref()], idx),
        Expr::Binary { op, left, right } => write_binary(source, *op, left, right, idx),
        Expr::Fma { a, b, c } => {
            write_call(source, "fmaf", &[a.as_ref(), b.as_ref(), c.as_ref()], idx)
        }
        Expr::Min { left, right } => {
            write_call(source, "fminf", &[left.as_ref(), right.as_ref()], idx)
        }
        Expr::Max { left, right } => {
            write_call(source, "fmaxf", &[left.as_ref(), right.as_ref()], idx)
        }
        // 叶子由 [`write_expr`] 前置分流到本函数之外；落到这里即空体，不再发射。
        Expr::F32(_) | Expr::Scalar(_) | Expr::Load(_) => {}
    }
}

fn write_binary(source: &mut String, op: Arith, left: &Expr, right: &Expr, idx: &str) {
    source.push('(');
    write_expr(source, left, idx);
    let _ = write!(source, " {} ", op.c_symbol());
    write_expr(source, right, idx);
    source.push(')');
}

/// Emit a C function call over already-rendered argument expressions, comma-separated.
fn write_call(source: &mut String, function: &str, args: &[&Expr], idx: &str) {
    source.push_str(function);
    source.push('(');
    for (position, argument) in args.iter().enumerate() {
        if position != 0 {
            source.push_str(", ");
        }
        write_expr(source, argument, idx);
    }
    source.push(')');
}

#[cfg(test)]
#[path = "../tests/ir_tests.rs"]
mod ir_tests;
