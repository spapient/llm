//! Typed, target-neutral IR for the first `#[dsl]` pointwise kernel family.

use core::fmt::{self, Write as _};

/// Scalar element types represented by the initial pointwise IR.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScalarType {
    /// IEEE-754 single precision.
    F32,
    /// Rust `usize`, used for iteration extents.
    Usize,
}

/// Buffer access required by a kernel parameter.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Access {
    /// Read-only buffer.
    Read,
    /// Mutable buffer, which may be read and written.
    ReadWrite,
}

/// A checked value type in the kernel signature.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ValueType {
    /// Scalar value.
    Scalar(ScalarType),
    /// Contiguous buffer and its element/access contract.
    Buffer {
        /// Buffer element type.
        element: ScalarType,
        /// Required access.
        access: Access,
    },
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

/// Typed expression nodes. Buffer loads use the kernel's implicit loop index.
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
    pub index: String,
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
        index: impl Into<String>,
        extent: impl Into<String>,
        output: impl Into<String>,
        value: Expr,
    ) -> Self {
        Self {
            name: name.into(),
            params,
            index: index.into(),
            extent: extent.into(),
            output: output.into(),
            value,
        }
    }

    /// Validate names, parameter types, memory access, and expression types.
    pub fn verify(&self) -> Result<(), VerificationError> {
        valid_identifier(&self.name)?;
        valid_identifier(&self.index)?;
        if self.params.iter().any(|param| param.name == self.index) {
            return Err(VerificationError::new(format!(
                "loop index `{}` shadows a kernel parameter",
                self.index
            )));
        }

        for (index, param) in self.params.iter().enumerate() {
            valid_identifier(&param.name)?;
            if self.params[..index]
                .iter()
                .any(|prior| prior.name == param.name)
            {
                return Err(VerificationError::new(format!(
                    "duplicate parameter `{}`",
                    param.name
                )));
            }
            if matches!(
                param.ty,
                ValueType::Buffer {
                    element: ScalarType::Usize,
                    ..
                }
            ) {
                return Err(VerificationError::new(format!(
                    "buffer `{}` must contain f32 elements",
                    param.name
                )));
            }
        }
        for (index, param) in self
            .params
            .iter()
            .filter(|param| is_buffer(param.ty))
            .enumerate()
        {
            let generated = buffer_length_name(index);
            if self.params.iter().any(|other| other.name == generated) {
                return Err(VerificationError::new(format!(
                    "parameter `{generated}` conflicts with a generated buffer length"
                )));
            }
            valid_identifier(&param.name)?;
        }
        if !matches!(
            self.parameter(&self.extent),
            Some(ValueType::Scalar(ScalarType::Usize))
        ) {
            return Err(VerificationError::new(format!(
                "loop extent `{}` must be a `usize` parameter",
                self.extent
            )));
        }
        if !matches!(
            self.parameter(&self.output),
            Some(ValueType::Buffer {
                element: ScalarType::F32,
                access: Access::ReadWrite,
            })
        ) {
            return Err(VerificationError::new(format!(
                "output `{}` must be a mutable f32 buffer",
                self.output
            )));
        }
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
        let mut source = String::new();
        source.push_str("#include <stddef.h>\n");
        write!(source, "int {}(", kernel.name).expect("writing to String cannot fail");
        let mut buffer_index = 0;
        for (index, parameter) in kernel.params.iter().enumerate() {
            if index != 0 {
                source.push_str(", ");
            }
            match parameter.ty {
                ValueType::Scalar(ScalarType::F32) => {
                    write!(source, "float {}", parameter.name)
                        .expect("writing to String cannot fail");
                }
                ValueType::Scalar(ScalarType::Usize) => {
                    write!(source, "size_t {}", parameter.name)
                        .expect("writing to String cannot fail");
                }
                ValueType::Buffer {
                    element: ScalarType::F32,
                    access: Access::Read,
                } => {
                    write!(source, "const float *{}", parameter.name)
                        .expect("writing to String cannot fail");
                    write!(source, ", size_t {}", buffer_length_name(buffer_index))
                        .expect("writing to String cannot fail");
                    buffer_index += 1;
                }
                ValueType::Buffer {
                    element: ScalarType::F32,
                    access: Access::ReadWrite,
                } => {
                    write!(source, "float *{}", parameter.name)
                        .expect("writing to String cannot fail");
                    write!(source, ", size_t {}", buffer_length_name(buffer_index))
                        .expect("writing to String cannot fail");
                    buffer_index += 1;
                }
                ValueType::Buffer { .. } => {
                    return Err(VerificationError::new(format!(
                        "unsupported buffer type for `{}`",
                        parameter.name
                    )));
                }
            }
        }
        source.push_str(") {\n");
        buffer_index = 0;
        for _parameter in kernel.params.iter().filter(|param| is_buffer(param.ty)) {
            let length = buffer_length_name(buffer_index);
            buffer_index += 1;
            writeln!(source, "    if ({} > {}) return 1;", kernel.extent, length)
                .expect("writing to String cannot fail");
        }
        write!(
            source,
            "    for (size_t {} = 0; {} < {}; ++{}) {{\n        {}[{}] = ",
            kernel.index, kernel.index, kernel.extent, kernel.index, kernel.output, kernel.index
        )
        .expect("writing to String cannot fail");
        write_expr(&mut source, &kernel.value, &kernel.index);
        source.push_str(";\n    }\n    return 0;\n}\n");
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
            Expr::F32(bits) if f32::from_bits(*bits).is_finite() => Ok(ScalarType::F32),
            Expr::F32(_) => Err(VerificationError::new(
                "f32 literals in typed IR must be finite",
            )),
            Expr::Scalar(name) => match self.parameter(name) {
                Some(ValueType::Scalar(ty)) => Ok(ty),
                Some(ValueType::Buffer { .. }) => Err(VerificationError::new(format!(
                    "buffer `{name}` must be loaded at the loop index"
                ))),
                None => Err(VerificationError::new(format!(
                    "unknown scalar parameter `{name}`"
                ))),
            },
            Expr::Load(name) => match self.parameter(name) {
                Some(ValueType::Buffer {
                    element: ScalarType::F32,
                    ..
                }) => Ok(ScalarType::F32),
                Some(ValueType::Buffer { .. }) => Err(VerificationError::new(format!(
                    "buffer `{name}` does not contain f32 elements"
                ))),
                Some(ValueType::Scalar(_)) => Err(VerificationError::new(format!(
                    "scalar `{name}` cannot be loaded as a buffer"
                ))),
                None => Err(VerificationError::new(format!("unknown buffer `{name}`"))),
            },
            Expr::Neg(value) => {
                let ty = self.value_type(value)?;
                if ty == ScalarType::F32 {
                    Ok(ty)
                } else {
                    Err(VerificationError::new("unary negation requires f32"))
                }
            }
            Expr::Binary { left, right, .. } => {
                let left = self.value_type(left)?;
                let right = self.value_type(right)?;
                if left == right && left == ScalarType::F32 {
                    Ok(left)
                } else {
                    Err(VerificationError::new(
                        "pointwise arithmetic operands must both be f32",
                    ))
                }
            }
        }
    }
}

fn valid_identifier(name: &str) -> Result<(), VerificationError> {
    let mut bytes = name.bytes();
    let valid_start = bytes
        .next()
        .is_some_and(|byte| byte.is_ascii_alphabetic() || byte == b'_');
    let valid_chars = valid_start && bytes.all(|byte| byte.is_ascii_alphanumeric() || byte == b'_');
    let keyword = [
        "auto",
        "break",
        "case",
        "char",
        "const",
        "continue",
        "default",
        "do",
        "double",
        "else",
        "enum",
        "extern",
        "float",
        "for",
        "goto",
        "if",
        "inline",
        "int",
        "long",
        "register",
        "restrict",
        "return",
        "short",
        "signed",
        "sizeof",
        "static",
        "struct",
        "switch",
        "typedef",
        "union",
        "unsigned",
        "void",
        "volatile",
        "while",
        "_Alignas",
        "_Atomic",
        "_Bool",
        "_Complex",
        "_Generic",
        "_Imaginary",
        "_Noreturn",
        "_Static_assert",
        "_Thread_local",
    ]
    .contains(&name);
    if valid_chars && !keyword {
        Ok(())
    } else {
        Err(VerificationError::new(format!(
            "`{name}` is not a portable C identifier"
        )))
    }
}

fn optimize_expr(expression: Expr) -> Expr {
    match expression {
        Expr::Neg(value) => match optimize_expr(*value) {
            Expr::F32(bits) => {
                let result = -f32::from_bits(bits);
                if result.is_finite() {
                    Expr::F32(result.to_bits())
                } else {
                    Expr::Neg(Box::new(Expr::F32(bits)))
                }
            }
            value => Expr::Neg(Box::new(value)),
        },
        Expr::Binary { op, left, right } => {
            let left = optimize_expr(*left);
            let right = optimize_expr(*right);
            if let (Expr::F32(left), Expr::F32(right)) = (&left, &right) {
                let left = f32::from_bits(*left);
                let right = f32::from_bits(*right);
                let result = match op {
                    BinaryOp::Add => left + right,
                    BinaryOp::Sub => left - right,
                    BinaryOp::Mul => left * right,
                    BinaryOp::Div => left / right,
                };
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
        expression => expression,
    }
}

fn is_buffer(ty: ValueType) -> bool {
    matches!(ty, ValueType::Buffer { .. })
}

fn buffer_length_name(index: usize) -> String {
    format!("cutile_len_{index}")
}

fn write_expr(source: &mut String, expression: &Expr, index: &str) {
    match expression {
        Expr::F32(bits) => {
            let value = f32::from_bits(*bits);
            write!(source, "{value:?}f").expect("writing to String cannot fail");
        }
        Expr::Scalar(name) => source.push_str(name),
        Expr::Load(name) => {
            write!(source, "{name}[{index}]").expect("writing to String cannot fail");
        }
        Expr::Neg(value) => {
            source.push_str("-(");
            write_expr(source, value, index);
            source.push(')');
        }
        Expr::Binary { op, left, right } => {
            source.push('(');
            write_expr(source, left, index);
            source.push_str(match op {
                BinaryOp::Add => " + ",
                BinaryOp::Sub => " - ",
                BinaryOp::Mul => " * ",
                BinaryOp::Div => " / ",
            });
            write_expr(source, right, index);
            source.push(')');
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[crate::dsl]
    #[allow(dead_code)]
    fn affine(input: &[f32], output: &mut [f32], scale: f32, extent: usize) {
        for position in 0..extent {
            output[position] = input[position] * scale + (2.0 + 1.0);
        }
    }

    #[test]
    fn dsl_builds_verified_optimized_ir_and_portable_source() {
        let kernel = __dsl_ir_affine();
        kernel.verify().expect("macro output has valid types");
        let source = kernel.emit_c().expect("verified IR emits C");
        assert!(source.contains("int affine(const float *input, size_t cutile_len_0, float *output, size_t cutile_len_1, float scale, size_t extent)"));
        assert!(source.contains("if (extent > cutile_len_0) return 1;"));
        assert!(source.contains("output[position] = ((input[position] * scale) + 3.0f);"));
    }

    #[test]
    fn verifier_rejects_non_mutable_outputs_and_invalid_extents() {
        let mut kernel = Kernel::new(
            "bad",
            vec![
                Parameter {
                    name: "output".to_string(),
                    ty: ValueType::Buffer {
                        element: ScalarType::F32,
                        access: Access::Read,
                    },
                },
                Parameter {
                    name: "extent".to_string(),
                    ty: ValueType::Scalar(ScalarType::Usize),
                },
            ],
            "i",
            "extent",
            "output",
            Expr::F32(0.0f32.to_bits()),
        );
        let error = kernel
            .verify()
            .expect_err("wrong output access is rejected");
        assert!(error.to_string().contains("mutable f32 buffer"));

        kernel.params[0].ty = ValueType::Buffer {
            element: ScalarType::F32,
            access: Access::ReadWrite,
        };
        kernel.verify().expect("valid signature passes");
        kernel.params[1].ty = ValueType::Scalar(ScalarType::F32);
        let error = kernel.verify().expect_err("non-usize extent is rejected");
        assert!(error.to_string().contains("loop extent"));
    }
}
