//! Typed, target-neutral IR for the `#[dsl]` kernel family.
//!
//! 一张 [`Kernel`] 覆盖三种写法：逐点 `output[idx] = value`、归约 `output[at] += value`、以及由
//! 前两者组合出的融合核（[`Kernel::fuse`]）。这一层只做校验、常量折叠与可移植 C 发射；它还没接
//! `Lower`/`Image`，所以任何「性能提升」都不该由它来宣称。

use core::fmt::{self, Write as _};

mod validate;
use validate::verify_core;

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
    /// C pointer declaration for an `f32` buffer held with this access.
    fn c_pointer(self) -> &'static str {
        match self {
            Access::Read => "const float *",
            Access::ReadWrite => "float *",
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
    /// C declaration for this parameter, or `None` when the element type has no C spelling.
    fn c_type(self) -> Option<&'static str> {
        match self {
            ValueType::Scalar(element) => Some(element.c_name()),
            ValueType::Buf {
                element: ScalarType::F32,
                access,
            } => Some(access.c_pointer()),
            ValueType::Buf { .. } => None,
        }
    }

    fn is_buf(self) -> bool {
        matches!(self, ValueType::Buf { .. })
    }

    /// `f32` 缓冲是唯一能发射成 C 的缓冲类型；`usize` 缓冲在校验阶段就被打回。
    fn element_ok(self) -> bool {
        match self {
            ValueType::Scalar(_) => true,
            ValueType::Buf {
                element: ScalarType::F32,
                ..
            } => true,
            ValueType::Buf { .. } => false,
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
pub enum BinaryOp {
    /// Addition.
    Add,
    /// Subtraction.
    Sub,
    /// Multiplication.
    Mul,
    /// Division.
    Div,
}

impl BinaryOp {
    /// Infix symbol used when emitting C.
    fn c_symbol(self) -> &'static str {
        match self {
            BinaryOp::Add => "+",
            BinaryOp::Sub => "-",
            BinaryOp::Mul => "*",
            BinaryOp::Div => "/",
        }
    }

    /// Apply the operation to two `f32` operands.
    fn apply(self, left: f32, right: f32) -> f32 {
        match self {
            BinaryOp::Add => left + right,
            BinaryOp::Sub => left - right,
            BinaryOp::Mul => left * right,
            BinaryOp::Div => left / right,
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
    /// Binary f32 arithmetic.
    Binary {
        /// Operation.
        op: BinaryOp,
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
    fn new(message: impl Into<String>) -> Self {
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
    fn expr(&self) -> &Expr {
        match self {
            Write::Map(value) => value,
            Write::Reduce { value, .. } => value,
        }
    }

    /// Mutable access to the expression evaluated once per iteration.
    fn expr_mut(&mut self) -> &mut Expr {
        match self {
            Write::Map(value) => value,
            Write::Reduce { value, .. } => value,
        }
    }

    /// Fixed slot a reduction accumulates into; `None` for pointwise stores.
    fn slot(&self) -> Option<u32> {
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
        Self::new(name, params, idx, extent, output, Write::Reduce { at, value })
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

    /// Fold constant-only subexpressions, retaining operations whose result is non-finite.
    pub fn optimize(&self) -> Result<Self, VerificationError> {
        self.verify()?;
        let mut optimized = self.clone();
        *optimized.write.expr_mut() = optimize_expr(self.write.expr().clone());
        optimized.verify()?;
        Ok(optimized)
    }

    /// Fuse a pointwise consumer into this pointwise producer.
    ///
    /// 消费者把 `self.output` 声明成只读缓冲时，中间缓冲被消掉：消费者里每一处对它的取值都换成
    /// `self` 的值表达式，于是两步落进同一个循环、只写最终输出。归约链、缺失的链接、可写链接与
    /// 长度不一致一律拒绝，而不是猜一个意思。
    ///
    /// # Errors
    ///
    /// 任一侧校验失败、写回不是逐点、消费者没有把 `self.output` 声明成只读 `f32` 缓冲、
    /// 或两侧 `extent` 不是同一个参数名时返回说明。
    pub fn fuse(&self, next: &Kernel) -> Result<Self, VerificationError> {
        verify_link(self, next)?;
        let (Write::Map(value), Write::Map(after)) = (&self.write, &next.write) else {
            return Err(VerificationError::new("fusion needs two pointwise kernels"));
        };
        let fused = Kernel::map(
            format!("{}_{}", self.name, next.name),
            merged_params(self, next),
            self.idx.clone(),
            self.extent.clone(),
            next.output.clone(),
            substitute(after, &self.output, value),
        );
        fused.verify()?;
        Ok(fused)
    }

    /// Emit portable C; return status 1 when a guarded length is violated.
    pub fn emit_c(&self) -> Result<String, VerificationError> {
        let kernel = self.optimize()?;
        let mut source = String::from("#include <stddef.h>\n");
        write_signature(&mut source, &kernel.name, &kernel.params)?;
        write_guards(&mut source, &kernel);
        write_loop(&mut source, &kernel);
        Ok(source)
    }
}

/// 先各自校验，再确认链路成立：把三次提前返回收进一个入口，压低 `fuse` 的分支复杂度。
fn verify_link(producer: &Kernel, consumer: &Kernel) -> Result<(), VerificationError> {
    producer.verify()?;
    consumer.verify()?;
    check_link(producer, consumer)
}

/// 融合前提：两侧都是逐点核、共享同一个 `extent`，且消费者把生产者的输出当只读缓冲读。
fn check_link(producer: &Kernel, consumer: &Kernel) -> Result<(), VerificationError> {
    check_pointwise(producer, consumer)?;
    check_extent(producer, consumer)?;
    check_input(producer, consumer)
}

fn check_pointwise(producer: &Kernel, consumer: &Kernel) -> Result<(), VerificationError> {
    if producer.write.slot().is_some() || consumer.write.slot().is_some() {
        return Err(VerificationError::new("fusion needs two pointwise kernels"));
    }
    Ok(())
}

fn check_extent(producer: &Kernel, consumer: &Kernel) -> Result<(), VerificationError> {
    if producer.extent != consumer.extent {
        return Err(VerificationError::new(format!(
            "fused kernels must share extent `{}`",
            producer.extent
        )));
    }
    Ok(())
}

fn check_input(producer: &Kernel, consumer: &Kernel) -> Result<(), VerificationError> {
    let link = consumer
        .params
        .iter()
        .find(|parameter| parameter.name == producer.output);
    match link {
        Some(Parameter { ty: ValueType::Buf { element: ScalarType::F32, access: Access::Read }, .. }) => Ok(()),
        Some(_) => Err(VerificationError::new(format!(
            "fused input `{}` must be a read-only f32 buffer",
            producer.output
        ))),
        None => Err(VerificationError::new(format!(
            "consumer does not read `{}`",
            producer.output
        ))),
    }
}

/// 融合核的形参：两侧都去掉中间缓冲；重名不在这里处理，交给 `verify` 的「参数不重名」闸。
fn merged_params(producer: &Kernel, consumer: &Kernel) -> Vec<Parameter> {
    producer
        .params
        .iter()
        .chain(&consumer.params)
        .filter(|parameter| parameter.name != producer.output)
        .cloned()
        .collect()
}

/// 把消费者里对中间缓冲的每一次取值换成生产者的值表达式。
fn substitute(expression: &Expr, name: &str, replacement: &Expr) -> Expr {
    match expression {
        Expr::Load(loaded) => substitute_load(loaded, name, replacement),
        other => substitute_compound(other, name, replacement),
    }
}

fn substitute_load(loaded: &str, name: &str, replacement: &Expr) -> Expr {
    if loaded == name {
        return replacement.clone();
    }
    Expr::Load(loaded.to_string())
}

fn substitute_compound(expression: &Expr, name: &str, replacement: &Expr) -> Expr {
    match expression {
        Expr::Neg(value) => Expr::Neg(Box::new(substitute(value, name, replacement))),
        Expr::Binary { op, left, right } => Expr::Binary {
            op: *op,
            left: Box::new(substitute(left, name, replacement)),
            right: Box::new(substitute(right, name, replacement)),
        },
        leaf => leaf.clone(),
    }
}

/// 形参声明 + 每个缓冲的隐式长度参数：`buffer` 与其长度成对出现，顺序与 `check_generated` 对齐。
fn write_signature(
    source: &mut String,
    name: &str,
    params: &[Parameter],
) -> Result<(), VerificationError> {
    let _ = write!(source, "int {name}(");
    let mut bufs = 0;
    for (idx, parameter) in params.iter().enumerate() {
        if idx != 0 {
            source.push_str(", ");
        }
        bufs += write_parameter(source, parameter, bufs)?;
    }
    source.push_str(") {\n");
    Ok(())
}

fn write_parameter(
    source: &mut String,
    parameter: &Parameter,
    bufs: usize,
) -> Result<usize, VerificationError> {
    let Some(declaration) = parameter.ty.c_type() else {
        return Err(VerificationError::new(format!(
            "unsupported buffer type for `{}`",
            parameter.name
        )));
    };
    let separator = if declaration.ends_with('*') { "" } else { " " };
    let _ = write!(source, "{declaration}{separator}{}", parameter.name);
    if !parameter.ty.is_buf() {
        return Ok(0);
    }
    let _ = write!(source, ", size_t {}", buf_length_name(bufs));
    Ok(1)
}

/// 归约只碰 `output[at]` 一个槽：输出守卫看 `at` 是否越界，其余缓冲仍看 `extent`。
fn write_guards(source: &mut String, kernel: &Kernel) {
    let slot = kernel.write.slot();
    let mut bufs = 0;
    for parameter in kernel.params.iter().filter(|parameter| parameter.ty.is_buf()) {
        match slot {
            Some(at) if parameter.name == kernel.output => {
                let _ = writeln!(
                    source,
                    "    if ({at} >= {}) return 1;",
                    buf_length_name(bufs)
                );
            }
            _ => {
                let _ = writeln!(
                    source,
                    "    if ({} > {}) return 1;",
                    kernel.extent,
                    buf_length_name(bufs)
                );
            }
        }
        bufs += 1;
    }
}

fn write_loop(source: &mut String, kernel: &Kernel) {
    match &kernel.write {
        Write::Map(_) => write_map_header(source, kernel),
        Write::Reduce { at, .. } => write_reduce_header(source, kernel, *at),
    }
    write_expr(source, kernel.write.expr(), &kernel.idx);
    source.push_str(";\n    }\n    return 0;\n}\n");
}

fn write_map_header(source: &mut String, kernel: &Kernel) {
    let _ = write!(
        source,
        "    for (size_t {0} = 0; {0} < {1}; ++{0}) {{\n        {2}[{0}] = ",
        kernel.idx, kernel.extent, kernel.output
    );
}

fn write_reduce_header(source: &mut String, kernel: &Kernel, at: u32) {
    let _ = write!(
        source,
        "    for (size_t {0} = 0; {0} < {1}; ++{0}) {{\n        {2}[{3}] += ",
        kernel.idx, kernel.extent, kernel.output, at
    );
}

fn optimize_expr(expression: Expr) -> Expr {
    match expression {
        Expr::Neg(value) => fold_negation(*value),
        Expr::Binary { op, left, right } => fold_binary(op, *left, *right),
        expression => expression,
    }
}

fn fold_negation(value: Expr) -> Expr {
    let value = optimize_expr(value);
    let Expr::F32(bits) = value else {
        return Expr::Neg(Box::new(value));
    };
    let result = -f32::from_bits(bits);
    if result.is_finite() {
        return Expr::F32(result.to_bits());
    }
    Expr::Neg(Box::new(Expr::F32(bits)))
}

fn fold_binary(op: BinaryOp, left: Expr, right: Expr) -> Expr {
    let left = optimize_expr(left);
    let right = optimize_expr(right);
    if let (Expr::F32(left), Expr::F32(right)) = (&left, &right) {
        let result = op.apply(f32::from_bits(*left), f32::from_bits(*right));
        if result.is_finite() {
            return Expr::F32(result.to_bits());
        }
    }
    Expr::Binary {
        op,
        left: Box::new(left),
        right: Box::new(right),
    }
}

fn buf_length_name(buf: usize) -> String {
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
        Expr::Binary { op, left, right } => write_binary(source, *op, left, right, idx),
        _ => {}
    }
}

fn write_binary(source: &mut String, op: BinaryOp, left: &Expr, right: &Expr, idx: &str) {
    source.push('(');
    write_expr(source, left, idx);
    let _ = write!(source, " {} ", op.c_symbol());
    write_expr(source, right, idx);
    source.push(')');
}

#[cfg(test)]
#[path = "../../tests/ir_tests.rs"]
mod ir_tests;
