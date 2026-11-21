//! 职责：写回尾块（epilogue）闭集文法——把 `bias; act:gelu; quant:i8` 解析成闭集步骤，并可逆地渲染回文本。
//!
//! 语法（闭集，超出即 [`diagnostic::Code::Invalid`]）：
//!
//! ```text
//! epilogue := step ((';' | ',') step)*
//! step     := op (':' arg)?
//! op       := bias | act | residual | scale | quant | store   （闭集 [`Epilogue`]）
//! arg      := act → 激活闭集名；quant → i8 / i4 / i2；scale → 有限十进制数
//! ```
//!
//! 与 [`crate::tpl`] + [`prof::atom::Prim`] 同构：闭集词汇在 `prof::atom`（[`Epilogue`] / [`ActKind`]），
//! 文法在此层。执行核（`gemm` / `conv` / `device` 的写回阶段）只消费解析结果，不各自再解析一遍。
//! 各步骤的数值语义（列偏置广播、激活、量化）归 `gemm` 等算术拥有方：本层只描述「做了什么、按什么顺序」，
//! 不复制算术。

use diagnostic::{Error, Result};
use prof::atom::{ActKind, DType, Epilogue};

/// 一个写回步骤：闭集操作 + 它的参数。每个变体自带参数，错误状态不可表达。
#[derive(Debug, Clone, PartialEq)]
pub enum Step {
    /// `bias`：加列偏置。
    Bias,
    /// `act:<kind>`：按 [`ActKind`] 逐元素变换。
    Act(ActKind),
    /// `residual`：加残差输入。
    Residual,
    /// `scale:<α>`：乘标量。
    Scale(f32),
    /// `quant:<i8|i4|i2>`：量化到整数位宽再反量化。
    Quant(DType),
    /// `store`：写回（无参，无附加状态）。
    Store,
}

/// 解析一段 epilogue 文本为闭集步骤序列。
///
/// # Errors
///
/// 任何一步超出闭集（未知步骤名、未知激活、`quant` 非 `i8/i4/i2`、`scale` 非有限数、
/// 无参步骤带了参数）都返回 [`diagnostic::Code::Invalid`]，错误挂「第 N 行」上下文。
pub fn parse(text: &str) -> Result<Vec<Step>> {
    let mut steps = Vec::new();
    for (line, piece) in split_steps(text) {
        steps.push(parse_step(piece, line)?);
    }
    Ok(steps)
}

/// 把步骤序列渲染回文本（与 [`parse`] 互逆：`parse(render(s)) == s`）。
pub fn render(steps: &[Step]) -> String {
    let mut text = String::new();
    for (index, step) in steps.iter().enumerate() {
        if index > 0 {
            text.push_str("; ");
        }
        write_step(&mut text, step);
    }
    text
}

/// 按 `;` / `,` 切步骤，返回 `(行号, 去空白后的步骤文本)`，跳过空段。
fn split_steps(text: &str) -> Vec<(usize, &str)> {
    let mut out = Vec::new();
    let mut start = 0usize;
    for (index, ch) in text.char_indices() {
        if ch == ';' || ch == ',' {
            push_step(&mut out, text, start, index);
            start = index + ch.len_utf8();
        }
    }
    push_step(&mut out, text, start, text.len());
    out
}

/// 把一个 `[start, end)` 字节区间收进结果；空段丢弃，行号按去空白后的真实起点算。
fn push_step<'a>(out: &mut Vec<(usize, &'a str)>, text: &'a str, start: usize, end: usize) {
    let raw = &text[start..end];
    let leading = raw.len() - raw.trim_start().len();
    let piece = raw.trim();
    if piece.is_empty() {
        return;
    }
    let line = crate::tpl::parse::count_newlines(&text[..start + leading]) + 1;
    out.push((line, piece));
}

/// 解析一个步骤：`op` 与可选 `:arg`。
fn parse_step(piece: &str, line: usize) -> Result<Step> {
    let (name, arg) = split_name_arg(piece);
    let op = Epilogue::parse(name).ok_or_else(|| error(line, format!("未知写回步骤 `{name}`")))?;
    step_of(op, arg, line)
}

/// 步骤名与参数的切分：`name[:arg]`。
fn split_name_arg(piece: &str) -> (&str, Option<&str>) {
    match piece.split_once(':') {
        Some((name, arg)) => (name.trim(), Some(arg.trim())),
        None => (piece.trim(), None),
    }
}

/// 由闭集操作与可选参数造一个步骤，校验参数是否符合该操作。
fn step_of(op: Epilogue, arg: Option<&str>, line: usize) -> Result<Step> {
    match op {
        Epilogue::Bias | Epilogue::Residual | Epilogue::Store => plain(op, arg, line),
        Epilogue::Act => act_of(arg, line).map(Step::Act),
        Epilogue::Scale => scale_of(arg, line).map(Step::Scale),
        Epilogue::Quant => quant_of(arg, line).map(Step::Quant),
    }
}

/// 无参步骤：带了参数即错。
fn plain(op: Epilogue, arg: Option<&str>, line: usize) -> Result<Step> {
    if arg.is_some() {
        return Err(error(line, format!("`{}` 不接受参数", op.as_str())));
    }
    Ok(match op {
        Epilogue::Bias => Step::Bias,
        Epilogue::Residual => Step::Residual,
        _ => Step::Store,
    })
}

/// `act:<kind>`：参数必填，且必须在激活闭集内。
fn act_of(arg: Option<&str>, line: usize) -> Result<ActKind> {
    let text = arg.ok_or_else(|| error(line, "`act` 需要参数（激活名）"))?;
    ActKind::parse(text).ok_or_else(|| error(line, format!("未知激活 `{text}`")))
}

/// `scale:<α>`：参数必填，且是有限十进制数。
fn scale_of(arg: Option<&str>, line: usize) -> Result<f32> {
    let text = arg.ok_or_else(|| error(line, "`scale` 需要参数（标量）"))?;
    let value: f32 = text
        .parse()
        .map_err(|_| error(line, format!("`scale` 参数不是十进制数：`{text}`")))?;
    if !value.is_finite() {
        return Err(error(line, format!("`scale` 参数必须是有限数：`{text}`")));
    }
    Ok(value)
}

/// `quant:<i8|i4|i2>`：参数必填，且是三者之一。
fn quant_of(arg: Option<&str>, line: usize) -> Result<DType> {
    let text = arg.ok_or(error(line, "`quant` 需要参数（i8 / i4 / i2）"))?;
    let dtype = DType::parse_name(text).ok_or(error(line, format!("未知 dtype `{text}`")))?;
    if !is_quant(dtype) {
        return Err(error(
            line,
            format!("`quant` 只接受 i8 / i4 / i2，`{text}` 不是"),
        ));
    }
    Ok(dtype)
}

/// `quant` 只接受 i8 / i4 / i2 三种整数位宽。
fn is_quant(dtype: DType) -> bool {
    matches!(dtype, DType::I8 | DType::I4 | DType::I2)
}

/// 一个步骤的渲染。
fn write_step(text: &mut String, step: &Step) {
    match step {
        Step::Bias | Step::Residual | Step::Store => write_plain(text, step),
        Step::Act(kind) => write_act(text, *kind),
        Step::Scale(value) => write_scale(text, *value),
        Step::Quant(dtype) => write_quant(text, *dtype),
    }
}

/// 无参步骤名。
fn write_plain(text: &mut String, step: &Step) {
    text.push_str(match step {
        Step::Bias => "bias",
        Step::Residual => "residual",
        _ => "store",
    });
}

/// `act:<kind>`。
fn write_act(text: &mut String, kind: ActKind) {
    text.push_str("act:");
    text.push_str(kind.as_str());
}

/// `scale:<α>`。
fn write_scale(text: &mut String, factor: f32) {
    text.push_str("scale:");
    text.push_str(&factor.to_string());
}

/// `quant:<dtype>`。
fn write_quant(text: &mut String, dtype: DType) {
    text.push_str("quant:");
    text.push_str(dtype.name());
}

/// 带行号的文法错误。
fn error(line: usize, message: impl Into<String>) -> Error {
    crate::tpl::parse::line_error(line, message)
}

#[cfg(test)]
#[path = "../tests/epilogue_tests.rs"]
mod tests;
