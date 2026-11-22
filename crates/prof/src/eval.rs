//! 职责：推导式求值：数值字面量与四则/括号。
use crate::schema::invalid;

/// 求值一个只用数值字面量、`+ - * /` 与括号的推导式。
pub(crate) fn eval_expression(text: &str) -> diagnostic::Result<f64> {
    let bytes = text.as_bytes();
    let mut cursor = 0usize;
    let value = eval_sum(text, bytes, &mut cursor)?;
    skip_spaces(bytes, &mut cursor);
    if cursor != bytes.len() {
        return Err(invalid(format!("位置 {cursor} 起有多余内容")));
    }
    Ok(value)
}
fn skip_spaces(bytes: &[u8], cursor: &mut usize) {
    while bytes.get(*cursor).is_some_and(u8::is_ascii_whitespace) {
        *cursor += 1;
    }
}
/// 加减层：左结合的 `+`/`-`。
fn eval_sum(text: &str, bytes: &[u8], cursor: &mut usize) -> diagnostic::Result<f64> {
    let mut value = eval_product(text, bytes, cursor)?;
    while let Some(operator) = operator(bytes, cursor, b"+-") {
        let operand = eval_product(text, bytes, cursor)?;
        value = added(value, operator, operand);
    }
    Ok(value)
}
/// 乘除层：`*` 与 `/` 优先于加减，除数为 0 是声明写错。
fn eval_product(text: &str, bytes: &[u8], cursor: &mut usize) -> diagnostic::Result<f64> {
    let mut value = eval_atom(text, bytes, cursor)?;
    while let Some(operator) = operator(bytes, cursor, b"*/") {
        let operand = eval_atom(text, bytes, cursor)?;
        value = multiplied(value, operator, operand)?;
    }
    Ok(value)
}
/// 当前位置的二元运算符（消费掉它）；不在 `allowed` 里就返回 `None` 且不动游标。
fn operator(bytes: &[u8], cursor: &mut usize, allowed: &[u8]) -> Option<u8> {
    skip_spaces(bytes, cursor);
    let found = *bytes.get(*cursor)?;
    if !allowed.contains(&found) {
        return None;
    }
    *cursor += 1;
    Some(found)
}
fn added(value: f64, operator: u8, operand: f64) -> f64 {
    if operator == b'+' {
        value + operand
    } else {
        value - operand
    }
}
/// 一次乘除；除数为 0 时给出诊断，否则原样折叠。
fn multiplied(value: f64, operator: u8, operand: f64) -> diagnostic::Result<f64> {
    if operator == b'*' {
        return Ok(value * operand);
    }
    if operand == 0.0 {
        return Err(invalid("除数为 0"));
    }
    Ok(value / operand)
}
/// 原子层：括号、数字字面量，以及"这里不该有东西"。
fn eval_atom(text: &str, bytes: &[u8], cursor: &mut usize) -> diagnostic::Result<f64> {
    skip_spaces(bytes, cursor);
    match bytes.get(*cursor) {
        Some(b'(') => eval_group(text, bytes, cursor),
        None => Err(invalid("表达式意外结束")),
        Some(byte) => eval_scalar(text, bytes, cursor, *byte),
    }
}
/// 括号表达式：括号必须闭合，里面按完整表达式求值。
fn eval_group(text: &str, bytes: &[u8], cursor: &mut usize) -> diagnostic::Result<f64> {
    *cursor += 1;
    let value = eval_sum(text, bytes, cursor)?;
    skip_spaces(bytes, cursor);
    if bytes.get(*cursor) != Some(&b')') {
        return Err(invalid("括号没有闭合"));
    }
    *cursor += 1;
    Ok(value)
}
/// 一个数字、或者一个不允许出现在这里的字符。
fn eval_scalar(text: &str, bytes: &[u8], cursor: &mut usize, byte: u8) -> diagnostic::Result<f64> {
    if byte.is_ascii_digit() || byte == b'.' {
        return eval_number(text, bytes, cursor);
    }
    let found = char::from(byte);
    Err(invalid(format!("位置 {cursor} 处不允许字符 `{found}`")))
}
/// 数值字面量：按最长前缀切出来再交给 `f64`，非法字面量连位置一起报出。
fn eval_number(text: &str, bytes: &[u8], cursor: &mut usize) -> diagnostic::Result<f64> {
    let start = *cursor;
    scan_number(bytes, cursor);
    let literal = &text[start..*cursor];
    let illegal = |error: std::num::ParseFloatError| {
        invalid(format!("位置 {start} 的数字 `{literal}` 非法：{error}"))
    };
    literal.parse::<f64>().map_err(illegal)
}
/// 扫描数值字面量：指数只认第一个 `e`/`E`，其后的 `+`/`-` 属于指数。
fn scan_number(bytes: &[u8], cursor: &mut usize) {
    let mut exponent = false;
    while let Some(&byte) = bytes.get(*cursor) {
        match byte {
            b'0'..=b'9' | b'.' => *cursor += 1,
            b'e' | b'E' if !exponent => {
                exponent = true;
                *cursor += 1;
                skip_sign(bytes, cursor);
            }
            _ => break,
        }
    }
}
fn skip_sign(bytes: &[u8], cursor: &mut usize) {
    let sign = bytes.get(*cursor);
    if sign.is_some_and(|s| matches!(s, b'+' | b'-')) {
        *cursor += 1;
    }
}
