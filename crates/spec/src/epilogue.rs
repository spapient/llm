//! 职责：写回尾块（epilogue）闭集文法——把 `bias; act:gelu; quant:i8` 解析成闭集步骤并给出标量参考。
//!
//! 语法（闭集，超出即 [`Code::Invalid`]）：
//!
//! ```text
//! epilogue := step ((';' | ',') step)*
//! step     := op (':' arg)?
//! op       := bias | act | residual | scale | quant | store   （闭集 [`Epilogue`]）
//! arg      := act → 激活闭集名；quant → i8 / i4 / i2；scale → 有限十进制数
//! ```
//!
//! 与 [`crate::tpl`] + [`prof::atom::Prim`] 同构：闭集词汇在 `prof::atom`（[`Epilogue`] / [`ActKind`]），
//! 文法与标量参考在此层。执行核（`gemm` / `conv` / `device` 的写回阶段）只消费解析结果，不各自再解析
//! 一遍；`apply` 是唯一的标量参考，也是误差界（γₙ 与量化半格）的落点。
//!
//! # 误差界
//!
//! - `bias` / `residual`：两数相加，前向误差 ≤ γ₂·|x|（γ₂ = 2u/(1−2u)，u 为 f32 机器精度）。
//! - `scale`：单次乘法，前向误差 ≤ γ₁·|x|（γ₁ = u）。
//! - `act`：`relu` / `leaky_relu` / `hard_sigmoid` 无超越函数，逐点精确；`gelu` 用
//!   Abramowitz–Stegun 7.1.26 近似 erf（绝对误差 < 1.5e-7），`silu` 用 `exp`（≤ 1 ulp）。
//! - `quant`：round-half-away-from-zero 后夹到目标整数范围，误差 ≤ 0.5 格（饱和点外为截断）。

use diagnostic::{Code, Error, Result};
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
    /// `store`：写回（标量参考里是空操作，结果就是瓦片）。
    Store,
}

/// 标量参考的输入：累加器之外还需要列偏置与残差（缺省视作全 0）。
#[derive(Debug, Clone, Copy)]
pub struct Ctx<'a> {
    /// 瓦片列数（决定 `bias` 的下标周期）。
    pub n: usize,
    /// 列偏置（长度 `n`）；`None` 视作全 0。
    pub bias: Option<&'a [f32]>,
    /// 残差（与瓦片同形状）；`None` 视作全 0。
    pub residual: Option<&'a [f32]>,
}

/// 解析一段 epilogue 文本为闭集步骤序列。
///
/// # Errors
///
/// 任何一步超出闭集（未知步骤名、未知激活、`quant` 非 `i8/i4/i2`、`scale` 非有限数、
/// 无参步骤带了参数）都返回 [`Code::Invalid`]，错误挂「第 N 行」上下文。
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

/// 标量参考：把步骤序列依次施加到行主序瓦片上。所有形状校验先做一遍，之后的每步都不再失败。
pub fn apply(steps: &[Step], tile: &mut [f32], ctx: Ctx<'_>) -> Result<()> {
    check_ctx(tile, ctx)?;
    for step in steps {
        run(step, tile, ctx);
    }
    Ok(())
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

/// 把一个 `[start, end)` 字节区间收进结果；空段丢弃，行号按起点算。
fn push_step<'a>(out: &mut Vec<(usize, &'a str)>, text: &'a str, start: usize, end: usize) {
    let piece = text[start..end].trim();
    if piece.is_empty() {
        return;
    }
    let line = crate::tpl::parse::count_newlines(&text[..start]) + 1;
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
    let text = arg.ok_or_else(|| error(line, "`quant` 需要参数（i8 / i4 / i2）"))?;
    let dtype = DType::parse_name(text).ok_or_else(|| error(line, format!("未知 dtype `{text}`")))?;
    if !matches!(dtype, DType::I8 | DType::I4 | DType::I2) {
        return Err(error(line, format!("`quant` 只接受 i8 / i4 / i2，`{text}` 不是")));
    }
    Ok(dtype)
}

/// 形状校验：`bias` 长度 = 列数、`residual` 长度 = 瓦片长度。
fn check_ctx(tile: &[f32], ctx: Ctx<'_>) -> Result<()> {
    check_len(ctx.bias, ctx.n, "bias")?;
    check_len(ctx.residual, tile.len(), "residual")?;
    Ok(())
}

/// 一条可选缓冲的长度校验。
fn check_len(buf: Option<&[f32]>, want: usize, name: &str) -> Result<()> {
    if let Some(buf) = buf {
        if buf.len() != want {
            return Err(Error::new(
                Code::Invalid,
                format!("{name} 缓冲 {}，需要 {want}", buf.len()),
            ));
        }
    }
    Ok(())
}

/// 施加一个步骤（形状已在上游校验，这里不再失败）。
fn run(step: &Step, tile: &mut [f32], ctx: Ctx<'_>) {
    match step {
        Step::Bias | Step::Residual => add_tile(tile, step, ctx),
        Step::Act(kind) => activate(tile, *kind),
        Step::Scale(value) => scale_all(tile, *value),
        Step::Quant(dtype) => quant_all(tile, *dtype),
        Step::Store => {}
    }
}

/// `bias`（按列广播）与 `residual`（同形状相加）分派到各自实现。
fn add_tile(tile: &mut [f32], step: &Step, ctx: Ctx<'_>) {
    match step {
        Step::Bias => add_bias(tile, ctx.bias),
        _ => add_residual(tile, ctx.residual),
    }
}

/// 列偏置按列广播：`tile[row * cols + col] += bias[col]`（缺省即不变）。
fn add_bias(tile: &mut [f32], bias: Option<&[f32]>) {
    let Some(bias) = bias else { return };
    let cols = bias.len();
    if cols == 0 {
        return;
    }
    for (index, out) in tile.iter_mut().enumerate() {
        *out += bias[index % cols];
    }
}

/// 残差同形状相加（缺省即不变）。
fn add_residual(tile: &mut [f32], residual: Option<&[f32]>) {
    let Some(residual) = residual else { return };
    for (out, value) in tile.iter_mut().zip(residual) {
        *out += *value;
    }
}

/// 逐元素激活。
fn activate(tile: &mut [f32], kind: ActKind) {
    for value in tile.iter_mut() {
        *value = apply_act(*value, kind);
    }
}

/// 单个激活的标量参考。
fn apply_act(x: f32, kind: ActKind) -> f32 {
    match kind {
        ActKind::Relu => x.max(0.0),
        ActKind::Gelu => 0.5 * x * (1.0 + erf(x * std::f32::consts::FRAC_1_SQRT_2)),
        ActKind::Silu => x / (1.0 + (-x).exp()),
        ActKind::LeakyRelu => leaky(x),
        ActKind::HardSigmoid => (x / 6.0 + 0.5).clamp(0.0, 1.0),
    }
}

/// LeakyReLU：非负恒等，负区按斜率 `0.01` 缩放（与 `gemm::unary` 的 `SLOPE` 一致）。
fn leaky(x: f32) -> f32 {
    if x >= 0.0 {
        x
    } else {
        0.01 * x
    }
}

/// 标量缩放。
fn scale_all(tile: &mut [f32], factor: f32) {
    for value in tile.iter_mut() {
        *value *= factor;
    }
}

/// 量化到整数位宽再反量化（round-half-away-from-zero，夹到目标范围）。
fn quant_all(tile: &mut [f32], dtype: DType) {
    for value in tile.iter_mut() {
        *value = round_trip(*value, dtype);
    }
}

/// 单值的量化往返：四舍五入后夹到整数范围。
fn round_trip(x: f32, dtype: DType) -> f32 {
    let (lo, hi) = quant_bounds(dtype);
    x.round().clamp(lo, hi)
}

/// `i8` / `i4` / `i2` 的整数范围。
fn quant_bounds(dtype: DType) -> (f32, f32) {
    match dtype {
        DType::I8 => (-128.0, 127.0),
        DType::I4 => (-8.0, 7.0),
        _ => (-2.0, 1.0),
    }
}

/// 标量误差函数：Abramowitz–Stegun 7.1.26（`|误差| < 1.5e-7`），与 `gemm::unary::gelu_f32::erf` 同源。
fn erf(x: f32) -> f32 {
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    let p = 0.327_591_1;
    let a1 = 0.254_829_6;
    let a2 = -0.284_496_72;
    let a3 = 1.421_413_8;
    let a4 = -1.453_152_1;
    let a5 = 1.061_405_4;
    let t = 1.0 / (1.0 + p * x);
    let series = t * (a1 + t * (a2 + t * (a3 + t * (a4 + t * a5))));
    sign * (1.0 - series * (-x * x).exp())
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
