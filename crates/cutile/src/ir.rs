//! Typed, target-neutral IR for the first `#[dsl]` pointwise kernel family.
//!
//! 这一层只做三件事：校验形状（名字合法、类型对得上、缓冲只按循环变量取值）、折叠常量、
//! 发射可移植 C。它还没有接 `Lower`/`Image`，所以任何「性能提升」都不该由它来宣称。

use core::fmt::{self, Write as _};

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
        self.check_shape()?;
        self.check_semantics()
    }

    /// 名字与形参决定了 C 声明长什么样，必须是排在最前面的两道闸。
    fn check_shape(&self) -> Result<(), VerificationError> {
        self.check_names()?;
        self.check_params()?;
        self.check_generated()
    }

    /// 名字合法之后才谈得上"这个表达式到底算不算 f32 输出"。
    fn check_semantics(&self) -> Result<(), VerificationError> {
        self.check_extent()?;
        self.check_output()?;
        self.check_value()
    }

    fn check_names(&self) -> Result<(), VerificationError> {
        valid_identifier(&self.name)?;
        valid_identifier(&self.idx)?;
        if self.params.iter().any(|param| param.name == self.idx) {
            return Err(VerificationError::new(format!(
                "loop index `{}` shadows a kernel parameter",
                self.idx
            )));
        }
        Ok(())
    }

    /// 逐条检查参数：名字合法、不重名、缓冲元素是 `f32`。
    fn check_params(&self) -> Result<(), VerificationError> {
        for (idx, param) in self.params.iter().enumerate() {
            valid_identifier(&param.name)?;
            if self.params[..idx]
                .iter()
                .any(|prior| prior.name == param.name)
            {
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
    fn check_generated(&self) -> Result<(), VerificationError> {
        for (idx, _) in self
            .params
            .iter()
            .filter(|param| param.ty.is_buf())
            .enumerate()
        {
            let generated = buf_length_name(idx);
            if self.params.iter().any(|other| other.name == generated) {
                return Err(VerificationError::new(format!(
                    "parameter `{generated}` conflicts with a generated buffer length"
                )));
            }
        }
        Ok(())
    }

    fn check_extent(&self) -> Result<(), VerificationError> {
        let extent = self.parameter(&self.extent);
        if extent != Some(ValueType::Scalar(ScalarType::Usize)) {
            return Err(VerificationError::new(format!(
                "loop extent `{}` must be a `usize` parameter",
                self.extent
            )));
        }
        Ok(())
    }

    fn check_output(&self) -> Result<(), VerificationError> {
        let output = self.parameter(&self.output);
        let expected = ValueType::Buf {
            element: ScalarType::F32,
            access: Access::ReadWrite,
        };
        if output != Some(expected) {
            return Err(VerificationError::new(format!(
                "output `{}` must be a mutable f32 buffer",
                self.output
            )));
        }
        Ok(())
    }

    fn check_value(&self) -> Result<(), VerificationError> {
        if self.value_type(&self.value)? != ScalarType::F32 {
            return Err(VerificationError::new(
                "kernel output expression must have type f32",
            ));
        }
        Ok(())
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
        write_signature(&mut source, &kernel)?;
        write_guards(&mut source, &kernel);
        write_loop(&mut source, &kernel);
        Ok(source)
    }

    fn parameter(&self, name: &str) -> Option<ValueType> {
        self.params
            .iter()
            .find(|parameter| parameter.name == name)
            .map(|parameter| parameter.ty)
    }

    fn value_type(&self, expression: &Expr) -> Result<ScalarType, VerificationError> {
        match expression {
            Expr::F32(bits) => literal_type(*bits),
            Expr::Scalar(name) => self.scalar_type(name),
            Expr::Load(name) => self.load_type(name),
            other => self.compound_type(other),
        }
    }

    fn compound_type(&self, expression: &Expr) -> Result<ScalarType, VerificationError> {
        match expression {
            Expr::Neg(value) => self.negated_type(value),
            Expr::Binary { left, right, .. } => self.binary_type(left, right),
            _ => Err(VerificationError::new("unsupported expression node")),
        }
    }

    fn scalar_type(&self, name: &str) -> Result<ScalarType, VerificationError> {
        match self.parameter(name) {
            Some(ValueType::Scalar(ty)) => Ok(ty),
            Some(ValueType::Buf { .. }) => Err(VerificationError::new(format!(
                "buffer `{name}` must be loaded at the loop index"
            ))),
            None => Err(VerificationError::new(format!(
                "unknown scalar parameter `{name}`"
            ))),
        }
    }

    fn load_type(&self, name: &str) -> Result<ScalarType, VerificationError> {
        match self.parameter(name) {
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

    fn negated_type(&self, value: &Expr) -> Result<ScalarType, VerificationError> {
        let ty = self.value_type(value)?;
        if ty == ScalarType::F32 {
            return Ok(ty);
        }
        Err(VerificationError::new("unary negation requires f32"))
    }

    fn binary_type(&self, left: &Expr, right: &Expr) -> Result<ScalarType, VerificationError> {
        let left = self.value_type(left)?;
        let right = self.value_type(right)?;
        if left == right && left == ScalarType::F32 {
            return Ok(left);
        }
        Err(VerificationError::new(
            "pointwise arithmetic operands must both be f32",
        ))
    }
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
fn write_signature(source: &mut String, kernel: &Kernel) -> Result<(), VerificationError> {
    let _ = write!(source, "int {}(", kernel.name);
    let mut bufs = 0;
    for (idx, parameter) in kernel.params.iter().enumerate() {
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

fn write_guards(source: &mut String, kernel: &Kernel) {
    for (idx, _) in kernel
        .params
        .iter()
        .filter(|parameter| parameter.ty.is_buf())
        .enumerate()
    {
        let _ = writeln!(
            source,
            "    if ({} > {}) return 1;",
            kernel.extent,
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
