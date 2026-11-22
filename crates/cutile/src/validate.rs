//! Validation for every [`Kernel`], regardless of its [`Write`].
//!
//! 六道闸按序排布：名字 → 形参 → 生成名 → extent → 输出 → 表达式。前一半决定 C 声明能否
//! 合法发射，后一半判定表达式到底算不算 `f32` 输出；逐点与归约只有写回形状不同，闸口完全一样。

use crate::kernel::{
    Access, Expr, Parameter, ScalarType, ValueType, VerificationError, buf_length_name,
};

/// 逐点与归约共用的六道校验：名字/形参/生成名决定 C 声明，之后才验表达式是否算 f32 输出。
pub(super) fn verify_core(
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
