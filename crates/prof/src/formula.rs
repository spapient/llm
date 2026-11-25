//! 一份可估算的语义：表达式 + 每个操作数的位宽与方向，推导 flops 与读 / 写流量。

use diagnostic::{Code, Error};
use std::sync::OnceLock;

use super::axis::{Axes, Axis, AxisRole, Extents, multiply, overflow};
use super::expr::Expr;
use super::operand::{Access, Cost, Operand, OperandSpec};

/// 一份可估算的语义：表达式 + 每个操作数的位宽与方向。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Formula {
    expr: Expr,
    operands: Vec<Operand>,
}

impl Formula {
    /// 把表达式与操作数配对。
    ///
    /// # Errors
    ///
    /// - 操作数个数与表达式的项数不一致；
    /// - 操作数重名（同一块缓冲会被重复计费）；
    /// - 写入操作数多于一个，或写入的轴不在输出里（写回中间量不是算子语义）。
    pub fn new(expr: Expr, operands: Vec<Operand>) -> Result<Self, Error> {
        if expr.arity() != operands.len() {
            return Err(Error::new(
                Code::Invalid,
                format!(
                    "表达式 `{expr}` 有 {} 项，但给了 {} 个操作数",
                    expr.arity(),
                    operands.len()
                ),
            )
            .with("einsum::Formula::new"));
        }
        unique(&operands)?;
        single_write(&operands)?;
        writes_only_outputs(&expr, &operands)?;
        Ok(Self { expr, operands })
    }

    /// 表达式。
    pub fn expr(&self) -> &Expr {
        &self.expr
    }

    /// 操作数（顺序与表达式的项一致）。
    pub fn operands(&self) -> &[Operand] {
        &self.operands
    }

    /// 轴分解：每条轴的长度与角色。
    ///
    /// # Errors
    ///
    /// 表达式里的字母没有给长度、给了长度的字母没出现在表达式里，或乘积超出 `u128`。
    pub fn axes(&self, extents: &Extents) -> Result<Axes, Error> {
        let mut axes = self.output_axes(extents)?;
        self.push_reduced(extents, &mut axes)?;
        self.check_used(extents, &axes)?;
        Ok(Axes { axes })
    }

    /// 输出轴：角色看"有几个读操作数共享它"（多于一个即 batch）。
    fn output_axes(&self, extents: &Extents) -> Result<Vec<Axis>, Error> {
        let mut axes: Vec<Axis> = Vec::with_capacity(extents.len());
        for &label in self.expr.output() {
            let Some((base, divisor)) = extents.grouped(label) else {
                let extent = extent(extents, label)?;
                axes.push(Axis {
                    label,
                    extent,
                    role: self.output_role(label),
                });
                continue;
            };
            return Err(Error::new(
                Code::Invalid,
                format!("轴 `{label}` 是 `{base}` 按 {divisor} 分组的派生轴，不能出现在输出里"),
            )
            .with("einsum::Formula::axes"));
        }
        Ok(axes)
    }

    /// 输出轴的角色：被多个读操作数共享的是 batch，否则是 free。
    fn output_role(&self, label: char) -> AxisRole {
        let shared = self
            .operands
            .iter()
            .zip(self.expr.terms())
            .filter(|(operand, term)| operand.access() == Access::Read && term.contains(&label))
            .count();
        if shared > 1 {
            AxisRole::Batch
        } else {
            AxisRole::Free
        }
    }

    /// 输入项里还没被收进来的轴：派生轴是 [`AxisRole::Grouped`]，其余是 [`AxisRole::Reduced`]。
    fn push_reduced(&self, extents: &Extents, axes: &mut Vec<Axis>) -> Result<(), Error> {
        for term in self.expr.terms() {
            for &label in term {
                if axes.iter().any(|axis| axis.label() == label) {
                    continue;
                }
                let role = extents
                    .grouped(label)
                    .map_or(AxisRole::Reduced, |_| AxisRole::Grouped);
                axes.push(Axis {
                    label,
                    extent: extent(extents, label)?,
                    role,
                });
            }
        }
        Ok(())
    }

    /// 给了长度却没出现在表达式里的轴：多半是笔误。
    fn check_used(&self, extents: &Extents, axes: &[Axis]) -> Result<(), Error> {
        for label in extents.labels() {
            if axes.iter().any(|axis| axis.label() == label) {
                continue;
            }
            return Err(Error::new(
                Code::Invalid,
                format!(
                    "轴 `{label}` 有长度 {} 但没出现在表达式 `{}` 里：多半是笔误",
                    extent(extents, label)?,
                    self.expr
                ),
            )
            .with("einsum::Formula::axes"));
        }
        Ok(())
    }

    /// 计算代价。
    ///
    /// # Errors
    ///
    /// 轴分解失败（见 [`Formula::axes`]）、维度乘积溢出，或流量超出 `u64`。
    pub fn cost(&self, extents: &Extents) -> Result<Cost, Error> {
        let axes = self.axes(extents)?;
        self.estimate(extents, &axes)
    }

    /// 一次算完：flops 与两个方向的流量。
    fn estimate(&self, extents: &Extents, axes: &Axes) -> Result<Cost, Error> {
        let flops = axes.flops()?;
        let (read, write) = self.traffic(extents)?;
        Ok(Cost {
            flops,
            read: bytes(read, "读")?,
            write: bytes(write, "写")?,
        })
    }

    /// 流量：每个操作数按自己的项长 × 位宽计费，读与写分开累加。
    fn traffic(&self, extents: &Extents) -> Result<(u128, u128), Error> {
        let mut read: u128 = 0;
        let mut write: u128 = 0;
        for (operand, term) in self.operands.iter().zip(self.expr.terms()) {
            let elements = term_elements(extents, term)?;
            let bytes = operand.bits().bytes_for(elements);
            let total = match operand.access() {
                Access::Read => &mut read,
                Access::Write => &mut write,
            };
            accumulate(total, bytes)?;
        }
        Ok((read, write))
    }
}

/// 同名缓冲只允许出现一次：否则同一块内存会被重复计费。
fn unique(operands: &[Operand]) -> Result<(), Error> {
    let mut names: Vec<&str> = Vec::with_capacity(operands.len());
    for operand in operands {
        if names.contains(&operand.name()) {
            return Err(Error::new(
                Code::Invalid,
                format!(
                    "操作数 `{}` 出现了两次：同名缓冲会被重复计费",
                    operand.name()
                ),
            )
            .with("einsum::Formula::new"));
        }
        names.push(operand.name());
    }
    Ok(())
}

/// 一次计算只有一个出口。
fn single_write(operands: &[Operand]) -> Result<(), Error> {
    let writes = operands
        .iter()
        .filter(|operand| operand.access() == Access::Write)
        .count();
    if writes > 1 {
        return Err(
            Error::new(Code::Invalid, "有多个写入操作数：一次计算只有一个出口")
                .with("einsum::Formula::new"),
        );
    }
    Ok(())
}

/// 写入操作数只能写输出轴：写回中间量不是算子语义。
fn writes_only_outputs(expr: &Expr, operands: &[Operand]) -> Result<(), Error> {
    for (operand, term) in operands.iter().zip(expr.terms()) {
        if operand.access() != Access::Write {
            continue;
        }
        if let Some(label) = term.iter().find(|label| !expr.output().contains(label)) {
            return Err(Error::new(
                Code::Invalid,
                format!(
                    "写入操作数 `{}` 的轴 `{label}` 不在输出里：写回中间量不是算子语义",
                    operand.name()
                ),
            )
            .with("einsum::Formula::new"));
        }
    }
    Ok(())
}

/// 一项的元素数：该项每条轴长度的乘积。
fn term_elements(extents: &Extents, term: &[char]) -> Result<u128, Error> {
    term.iter().try_fold(1u128, |total, label| {
        multiply(total, extent(extents, *label)?)
    })
}

/// 按方向累加：两个计数器各自溢出都要报。
fn accumulate(total: &mut u128, bytes: u128) -> Result<(), Error> {
    *total = total.checked_add(bytes).ok_or_else(overflow)?;
    Ok(())
}

/// 流量收口到 `u64`：`Cost` 的字段是 `u64`，装不下说明形状不可能是真的。
fn bytes(value: u128, kind: &str) -> Result<u64, Error> {
    u64::try_from(value).map_err(|_| {
        Error::new(Code::Invalid, format!("{kind}流量 {value} 字节超出 u64"))
            .with("einsum::Formula::cost")
    })
}

fn extent(extents: &Extents, label: char) -> Result<usize, Error> {
    extents.get(label).ok_or_else(|| {
        Error::new(
            Code::Invalid,
            format!("轴 `{label}` 没有长度：表达式里的每个字母都必须给出轴长度"),
        )
        .with("einsum::Formula::axes")
    })
}

// ---------------------------------------------------------------------------
// 表达式缓存：构造一次，之后复用到任意形状。

/// 一份表达式缓存：构造一次，之后复用到任意形状。
pub struct FormulaCache(OnceLock<Result<Formula, Error>>);

impl FormulaCache {
    /// 空缓存（`static` 初始化用，必须是 `const fn`）。
    #[must_use]
    pub const fn new() -> Self {
        Self(OnceLock::new())
    }

    /// 取这条表达式：第一次调用时解析并校验，之后直接复用。
    ///
    /// `site` 是错误帧（如 `prof::gemv::formula`），由**骨架**给出而不是由格式给出：同一条骨架的
    /// 所有精度共享同一个位置，重复写在每个实现里只会多出几份可以写歪的事实。
    ///
    /// # Errors
    ///
    /// 表达式或操作数表不自洽时返回第一次得到的说明（带 `site` 帧）。
    pub fn get(
        &'static self,
        expr: &'static str,
        operands: &'static [OperandSpec],
        site: &'static str,
    ) -> Result<&'static Formula, Error> {
        self.0
            .get_or_init(|| {
                let operands = operands
                    .iter()
                    .map(|spec| Operand::new(spec.name, spec.bits, spec.access))
                    .collect();
                Expr::parse(expr)
                    .and_then(|expr| Formula::new(expr, operands))
                    .map_err(|error| error.with(site))
            })
            .as_ref()
            .map_err(Clone::clone)
    }
}

impl Default for FormulaCache {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// 设备峰值的数值推导式：与求和语义共用公式模块，不再经过档案动态树。

use crate::atom::invalid;

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

/// 加减层：左结合的 `+` / `-`。
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

/// 当前位置的二元运算符；不在 `allowed` 里就返回 `None` 且不动游标。
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

fn multiplied(value: f64, operator: u8, operand: f64) -> diagnostic::Result<f64> {
    if operator == b'*' {
        return Ok(value * operand);
    }
    if operand == 0.0 {
        return Err(invalid("除数为 0"));
    }
    Ok(value / operand)
}

/// 原子层：括号、数字字面量，以及“这里不该有东西”。
fn eval_atom(text: &str, bytes: &[u8], cursor: &mut usize) -> diagnostic::Result<f64> {
    skip_spaces(bytes, cursor);
    match bytes.get(*cursor) {
        Some(b'(') => eval_group(text, bytes, cursor),
        None => Err(invalid("表达式意外结束")),
        Some(byte) => eval_scalar(text, bytes, cursor, *byte),
    }
}

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

/// 扫描数值字面量：指数只认第一个 `e` / `E`，其后的 `+` / `-` 属于指数。
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
