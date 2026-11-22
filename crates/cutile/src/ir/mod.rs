//! Typed, target-neutral IR for the `#[dsl]` kernel families: pointwise and reduction.
//!
//! 这一层只做三件事：校验形状（名字合法、类型对得上、缓冲只按循环变量取值）、折叠常量、
//! 发射可移植 C。它还没有接 `Lower`/`Image`，所以任何「性能提升」都不该由它来宣称。

use core::fmt::{self, Write as _};

mod reduce;
pub use reduce::Reduce;

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

/// One verified pointwise kernel: loop extent, mutable output, and a scalar expression.
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
    /// Value written to each output element.
    pub value: Expr,
}

impl Kernel {
    /// Construct a pointwise kernel IR node.
    pub fn new(
        name: impl Into<String>,
        params: Vec<Parameter>,
        idx: impl Into<String>,
        extent: impl Into<String>,
        output: impl Into<String>,
        value: Expr,
    ) -> Self {
        Self {
            name: name.into(),
            params,
            idx: idx.into(),
            extent: extent.into(),
            output: output.into(),
            value,
        }
    }

    /// Validate names, parameter types, memory access, and expression types.
    pub fn verify(&self) -> Result<(), VerificationError> {
        verify_core(
            &self.name,
            &self.idx,
            &self.params,
            &self.extent,
            &self.output,
            &self.value,
        )
    }

    /// Fold constant-only subexpressions, retaining operations whose result is non-finite.
    pub fn optimize(&self) -> Result<Self, VerificationError> {
        self.verify()?;
        let mut optimized = self.clone();
        optimized.value = optimize_expr(optimized.value);
        optimized.verify()?;
        Ok(optimized)
    }

    /// Emit portable C; return status 1 when the extent exceeds any buffer length.
    pub fn emit_c(&self) -> Result<String, VerificationError> {
        let kernel = self.optimize()?;
        let mut source = String::from("#include <stddef.h>\n");
        write_signature(&mut source, &kernel.name, &kernel.params)?;
        write_guards(&mut source, &kernel.params, &kernel.extent);
        write_loop(&mut source, &kernel);
        Ok(source)
    }
}

/// 逐点与归约共用的六道校验：名字/形参/生成名决定 C 声明，之后才验表达式是否算 f32 输出。
fn verify_core(
    name: &str,
    idx: &str,
    params: &[Parameter],
    extent: &str,
    output: &str,
    value: &Expr,
) -> Result<(), VerificationError> {
    check_shape(name, idx, params)?;
    check_semantics(params, extent, output, value)
}

/// 名字与形参决定 C 声明，必须是最前面的闸。
fn check_shape(name: &str, idx: &str, params: &[Parameter]) -> Result<(), VerificationError> {
    check_names(name, idx, params)?;
    check_params(params)?;
    check_generated(params)
}

/// 声明合法之后才谈得上「表达式到底算不算 f32 输出」。
fn check_semantics(
    params: &[Parameter],
    extent: &str,
    output: &str,
    value: &Expr,
) -> Result<(), VerificationError> {
    check_extent(params, extent)?;
    check_output(params, output)?;
    check_value(params, value)
}

fn check_names(name: &str, idx: &str, params: &[Parameter]) -> Result<(), VerificationError> {
    valid_identifier(name)?;
    valid_identifier(idx)?;
    if params.iter().any(|param| param.name == idx) {
        return Err(VerificationError::new(format!(
            "loop index `{idx}` shadows a kernel parameter"
        )));
    }
    Ok(())
}

/// 逐条检查参数：名字合法、不重名、缓冲元素是 `f32`。
fn check_params(params: &[Parameter]) -> Result<(), VerificationError> {
    for (idx, param) in params.iter().enumerate() {
        valid_identifier(&param.name)?;
        if params[..idx].iter().any(|prior| prior.name == param.name) {
            return Err(VerificationError::new(format!(
                "duplicate parameter `{}`",
                param.name
            )));
        }
        if !param.ty.element_ok() {
            return Err(VerificationError::new(format!(
                "buffer `{}` must contain f32 elements",
                param.name
            )));
        }
    }
    Ok(())
}

/// 生成的长度名不许和用户形参撞车，否则 C 签名里会出现两个同名参数。
fn check_generated(params: &[Parameter]) -> Result<(), VerificationError> {
    for (idx, _) in params.iter().filter(|param| param.ty.is_buf()).enumerate() {
        let generated = buf_length_name(idx);
        if params.iter().any(|other| other.name == generated) {
            return Err(VerificationError::new(format!(
                "parameter `{generated}` conflicts with a generated buffer length"
            )));
        }
    }
    Ok(())
}

fn check_extent(params: &[Parameter], extent: &str) -> Result<(), VerificationError> {
    if parameter(params, extent) != Some(ValueType::Scalar(ScalarType::Usize)) {
        return Err(VerificationError::new(format!(
            "loop extent `{extent}` must be a `usize` parameter"
        )));
    }
    Ok(())
}

fn check_output(params: &[Parameter], output: &str) -> Result<(), VerificationError> {
    let expected = ValueType::Buf {
        element: ScalarType::F32,
        access: Access::ReadWrite,
    };
    if parameter(params, output) != Some(expected) {
        return Err(VerificationError::new(format!(
            "output `{output}` must be a mutable f32 buffer"
        )));
    }
    Ok(())
}

fn check_value(params: &[Parameter], value: &Expr) -> Result<(), VerificationError> {
    if value_type(params, value)? != ScalarType::F32 {
        return Err(VerificationError::new(
            "kernel output expression must have type f32",
        ));
    }
    Ok(())
}

fn parameter(params: &[Parameter], name: &str) -> Option<ValueType> {
    params
        .iter()
        .find(|parameter| parameter.name == name)
        .map(|parameter| parameter.ty)
}

fn value_type(params: &[Parameter], expression: &Expr) -> Result<ScalarType, VerificationError> {
    match expression {
        Expr::F32(bits) => literal_type(*bits),
        Expr::Scalar(name) => scalar_type(params, name),
        Expr::Load(name) => load_type(params, name),
        other => compound_type(params, other),
    }
}

fn compound_type(params: &[Parameter], expression: &Expr) -> Result<ScalarType, VerificationError> {
    match expression {
        Expr::Neg(value) => negated_type(params, value),
        Expr::Binary { left, right, .. } => binary_type(params, left, right),
        _ => Err(VerificationError::new("unsupported expression node")),
    }
}

fn scalar_type(params: &[Parameter], name: &str) -> Result<ScalarType, VerificationError> {
    match parameter(params, name) {
        Some(ValueType::Scalar(ty)) => Ok(ty),
        Some(ValueType::Buf { .. }) => Err(VerificationError::new(format!(
            "buffer `{name}` must be loaded at the loop index"
        ))),
        None => Err(VerificationError::new(format!(
            "unknown scalar parameter `{name}`"
        ))),
    }
}

fn load_type(params: &[Parameter], name: &str) -> Result<ScalarType, VerificationError> {
    match parameter(params, name) {
        Some(ValueType::Buf {
            element: ScalarType::F32,
            ..
        }) => Ok(ScalarType::F32),
        Some(ValueType::Buf { .. }) => Err(VerificationError::new(format!(
            "buffer `{name}` does not contain f32 elements"
        ))),
        Some(ValueType::Scalar(_)) => Err(VerificationError::new(format!(
            "scalar `{name}` cannot be loaded as a buffer"
        ))),
        None => Err(VerificationError::new(format!("unknown buffer `{name}`"))),
    }
}

fn negated_type(params: &[Parameter], value: &Expr) -> Result<ScalarType, VerificationError> {
    let ty = value_type(params, value)?;
    if ty == ScalarType::F32 {
        return Ok(ty);
    }
    Err(VerificationError::new("unary negation requires f32"))
}

fn binary_type(
    params: &[Parameter],
    left: &Expr,
    right: &Expr,
) -> Result<ScalarType, VerificationError> {
    let left = value_type(params, left)?;
    let right = value_type(params, right)?;
    if left == right && left == ScalarType::F32 {
        return Ok(left);
    }
    Err(VerificationError::new(
        "pointwise arithmetic operands must both be f32",
    ))
}

fn literal_type(bits: u32) -> Result<ScalarType, VerificationError> {
    if f32::from_bits(bits).is_finite() {
        return Ok(ScalarType::F32);
    }
    Err(VerificationError::new(
        "f32 literals in typed IR must be finite",
    ))
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

fn write_guards(source: &mut String, params: &[Parameter], extent: &str) {
    for (idx, _) in params
        .iter()
        .filter(|parameter| parameter.ty.is_buf())
        .enumerate()
    {
        let _ = writeln!(
            source,
            "    if ({extent} > {}) return 1;",
            buf_length_name(idx)
        );
    }
}

fn write_loop(source: &mut String, kernel: &Kernel) {
    let _ = write!(
        source,
        "    for (size_t {0} = 0; {0} < {1}; ++{0}) {{\n        {2}[{0}] = ",
        kernel.idx, kernel.extent, kernel.output
    );
    write_expr(source, &kernel.value, &kernel.idx);
    source.push_str(";\n    }\n    return 0;\n}\n");
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

fn valid_identifier(name: &str) -> Result<(), VerificationError> {
    if identifier_shape_ok(name) && !is_c_keyword(name) {
        return Ok(());
    }
    Err(VerificationError::new(format!(
        "`{name}` is not a portable C identifier"
    )))
}

fn identifier_shape_ok(name: &str) -> bool {
    let mut bytes = name.bytes();
    let start = bytes
        .next()
        .is_some_and(|byte| byte.is_ascii_alphabetic() || byte == b'_');
    start && bytes.all(|byte| byte.is_ascii_alphanumeric() || byte == b'_')
}

/// C11 关键字闭集：被发射的名字要原样写进 C 片段，撞上它就不是合法标识符。
/// 写成空白分隔的单串比四十二行数组短，`split_ascii_whitespace` 不分配。
const C_KEYWORDS: &str = "auto break case char const continue default do double else enum extern \
     float for goto if inline int long register restrict return short signed sizeof static struct \
     switch typedef union unsigned void volatile while _Alignas _Atomic _Bool _Complex _Generic \
     _Imaginary _Noreturn _Static_assert _Thread_local";

fn is_c_keyword(name: &str) -> bool {
    C_KEYWORDS
        .split_ascii_whitespace()
        .any(|keyword| keyword == name)
}

#[cfg(test)]
#[path = "../tests/ir_tests.rs"]
mod ir_tests;
