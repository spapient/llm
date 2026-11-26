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
//! 与 [`crate::tpl`] + [`crate::atom::Prim`] 同构：闭集词汇在 `crate::atom`（[`Epilogue`] / [`ActKind`]），
//! 文法在此层。执行核（`gemm` / `conv` / `device` 的写回阶段）只消费解析结果，不各自再解析一遍。
//! 各步骤的数值语义（列偏置广播、激活、量化）归 `gemm` 等算术拥有方：本层只描述「做了什么、按什么顺序」，
//! 不复制算术。

use crate::atom::{ActKind, DType, Epilogue};
use crate::tpl::error;
use diagnostic::Result;

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

/// 量化步长：`quant` 往返的网格间隔，也是对称整数量化的反量化系数。
///
/// 唯一事实源放在这里：算术侧执行（`nn::epilogue`）与设备侧装配
/// （`cutile::substitute::epilogue`）都读这一个常数，免得两处各写一份、日后悄悄分叉。
pub const QUANT_SCALE: f32 = 0.25;

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
    for (idx, step) in steps.iter().enumerate() {
        if idx > 0 {
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
    for (idx, ch) in text.char_indices() {
        if ch == ';' || ch == ',' {
            push_step(&mut out, text, start, idx);
            start = idx + ch.len_utf8();
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
    let line = crate::text::count_newlines(&text[..start + leading]) + 1;
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

#[cfg(test)]
mod tests {
    use super::*;
    use diagnostic::Code;

    testkit::test_cases! {
    /// 一整条写回链：六个步骤各归其位，参数被正确装箱。
    fn a_full_chain_parses_into_six_steps() {
        let steps = parse("bias; act:gelu; residual; scale:0.5; quant:i8; store").expect("合法链");
        assert_eq!(
            steps,
            [
                Step::Bias,
                Step::Act(ActKind::Gelu),
                Step::Residual,
                Step::Scale(0.5),
                Step::Quant(DType::I8),
                Step::Store,
            ],
        );
    }

    /// `parse(render(s)) == s`：渲染不丢任何参数，是可逆的规范形。
    fn render_is_the_parse_inverse() {
        let cases: &[&[Step]] = &[
            &[],
            &[Step::Store],
            &[Step::Bias, Step::Act(ActKind::Silu), Step::Residual],
            &[
                Step::Scale(2.5),
                Step::Quant(DType::I4),
                Step::Act(ActKind::LeakyRelu),
            ],
            &[
                Step::Bias,
                Step::Act(ActKind::Gelu),
                Step::Residual,
                Step::Scale(0.5),
                Step::Quant(DType::I2),
                Step::Store,
            ],
        ];
        for steps in cases {
            assert_eq!(parse(&render(steps)).expect("回环可解析"), steps.to_vec());
        }
    }

    /// 渲染的规范形固定：分号分隔、无参步骤不拼冒号。
    fn render_is_canonical() {
        let steps = [
            Step::Bias,
            Step::Act(ActKind::Relu),
            Step::Scale(1.5),
            Step::Store,
        ];
        assert_eq!(render(&steps), "bias; act:relu; scale:1.5; store");
    }

    /// 闭集之外的步骤名必须被拒，错误指向行号。
    fn an_unknown_step_is_rejected_with_its_line() {
        let err = parse("bias; norm; store").expect_err("`norm` 不在闭集");
        assert_eq!(err.code(), Code::Invalid);
        assert!(err.to_string().contains("未知写回步骤 `norm`"), "{err}");
        assert!(err.to_string().contains("第 1 行"), "{err}");
    }

    /// 未知激活被拒。
    fn an_unknown_activation_is_rejected() {
        let err = parse("act:swish").expect_err("`swish` 不在激活闭集");
        assert_eq!(err.code(), Code::Invalid);
        assert!(err.to_string().contains("未知激活 `swish`"), "{err}");
    }

    /// `quant` 只收 `i8 / i4 / i2`，浮点与更宽整数都被拒。
    fn quant_accepts_only_i8_i4_i2() {
        assert!(parse("quant:i8").is_ok());
        assert!(parse("quant:i4").is_ok());
        assert!(parse("quant:i2").is_ok());
        for bad in ["quant:f32", "quant:i16", "quant:bf16"] {
            let err = parse(bad).expect_err(bad);
            assert_eq!(err.code(), Code::Invalid);
        }
    }

    /// `scale` 只收有限十进制数。
    fn scale_accepts_only_finite_decimals() {
        assert_eq!(parse("scale:0.25").expect("有限"), [Step::Scale(0.25)]);
        for bad in ["scale:nan", "scale:inf", "scale:-inf", "scale:abc"] {
            let err = parse(bad).expect_err(bad);
            assert_eq!(err.code(), Code::Invalid);
        }
    }

    /// 无参步骤带参数即错；有参步骤缺参数即错。
    fn arguments_must_match_the_step() {
        assert!(parse("bias:1").is_err(), "bias 无参");
        assert!(parse("residual:x").is_err(), "residual 无参");
        assert!(parse("store:y").is_err(), "store 无参");
        assert!(parse("act").is_err(), "act 缺参");
        assert!(parse("scale").is_err(), "scale 缺参");
        assert!(parse("quant").is_err(), "quant 缺参");
    }

    /// 分号与逗号都可作分隔；空段被跳过；行号随换行前进。
    fn separators_whitespace_and_lines() {
        let steps = parse("  bias ,  act:relu  ;\n  scale:2  , store ").expect("混合分隔");
        assert_eq!(
            steps,
            [
                Step::Bias,
                Step::Act(ActKind::Relu),
                Step::Scale(2.0),
                Step::Store,
            ],
        );
        assert!(parse("").expect("空输入").is_empty());
        assert!(parse(";;;,").expect("全空段").is_empty());
        let err = parse("bias;\n\nnorm").expect_err("第三行的坏步骤");
        assert!(err.to_string().contains("第 3 行"), "{err}");
    }
    }
}
