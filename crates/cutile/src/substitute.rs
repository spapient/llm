//! 职责：片段拼接（README §4.2、§10.1、§10.2 的 ④）。

use crate::decl::{OpDecl, TileCombo};
use prof::atom::{ActKind, DType, Epilogue};
use prof::epilogue::{QUANT_SCALE, Step};
use std::collections::BTreeMap;

/// 词法与行号唯一的错误构造器直接转发：词法只有一份，声明解析与拼接看同一套 `{{…}}` 规则。
pub use prof::tpl::{Arg, Segment, error, lex};

// 片段层的事实源在 `prof`（`<vendor>/<prim>.<ext>`）：本模块只做拼接，不持有片段。
pub use prof::FragmentSet;

/// 片段层缺少原语片段时的统一错误：ast 展开与本模块拼接都走这条，免得两处各写一遍格式串。
pub(crate) fn missing_fragment(
    line: usize,
    prim: crate::atom::Prim,
    fragments: &FragmentSet,
) -> diagnostic::Error {
    error(
        line,
        format!(
            "片段层缺少 `{}/{}.{}`：该设备尚未实现 `{prim}`",
            fragments.vendor(),
            prim,
            fragments.ext()
        ),
    )
}

/// ④ substitute：模板槽位 ← 轴/布局取值，`@原语(...)` ← 片段文本。
pub fn render(
    op: &OpDecl,
    combo: &TileCombo,
    fragments: &FragmentSet,
) -> diagnostic::Result<String> {
    Ok(crate::ast::render(&crate::ast::build(
        op, combo, fragments,
    )?))
}

/// 把一段片段放进输出：去掉模板已提供的重复 `;`，并按调用点当前缩进对齐多行片段。
pub(crate) fn place(text: &mut String, expanded: &str) {
    let trimmed = expanded.trim_end();
    let trimmed = trimmed.strip_suffix(';').unwrap_or(trimmed);
    // 缩进取自插入点的当前行；单行片段只追加一次，不 `repeat` 缩进、也不走 `replace`。
    let indent = text
        .rsplit('\n')
        .next()
        .map_or(0, |prefix| prefix.chars().count());
    let mut lines = trimmed.split('\n');
    text.push_str(lines.next().unwrap_or_default());
    for line in lines {
        text.push('\n');
        for _ in 0..indent {
            text.push(' ');
        }
        text.push_str(line);
    }
}

/// 片段占位符替换：`$0`..`$9` 是位置参数，`$name` 是命名参数。
pub(crate) fn splice(fragment: &str, args: &[Arg<'_>], line: usize) -> diagnostic::Result<String> {
    let table = Args::new(args);
    let mut out = String::with_capacity(fragment.len());
    let mut rest = fragment;
    while let Some((head, name, tail)) = next_hole(rest) {
        out.push_str(head);
        out.push_str(table.resolve(name, line)?);
        rest = tail;
    }
    out.push_str(rest);
    Ok(out)
}

/// 一段片段里的占位符切成“字面量 + 名字”：位置参数是 `$0`，命名参数是 `$name`。
fn next_hole(text: &str) -> Option<(&str, &str, &str)> {
    let mut from = 0;
    loop {
        let dollar = from + text[from..].find('$')?;
        let start = dollar + 1;
        let len = text[start..]
            .bytes()
            .take_while(|byte| byte.is_ascii_alphanumeric() || *byte == b'_')
            .count();
        if len > 0 {
            return Some((
                &text[..dollar],
                &text[start..start + len],
                &text[start + len..],
            ));
        }
        // 孤立的 `$` 是字面量，从它后面继续找。
        from = start;
    }
}

/// 片段的实参表：位置参数按序号、命名参数按名字，拼接时各查一次。
struct Args<'a> {
    positional: Vec<&'a str>,
    named: BTreeMap<&'a str, &'a str>,
}

impl<'a> Args<'a> {
    fn new(args: &'a [Arg<'a>]) -> Self {
        let mut positional = Vec::new();
        let mut named = BTreeMap::new();
        for arg in args {
            match arg {
                Arg::Expr(text) => positional.push(*text),
                Arg::Named { key, value } => {
                    named.insert(*key, *value);
                }
            }
        }
        Self { positional, named }
    }

    /// 一个占位符的取值；缺参是模板错误，而不是空串。
    fn resolve(&self, name: &str, line: usize) -> diagnostic::Result<&'a str> {
        if let Ok(idx) = name.parse::<usize>() {
            return self.positional.get(idx).copied().ok_or_else(|| {
                error(
                    line,
                    format!(
                        "片段占用了 `${idx}`，但模板只给了 {} 个位置参数",
                        self.positional.len()
                    ),
                )
            });
        }
        self.named
            .get(name)
            .copied()
            .ok_or_else(|| error(line, format!("片段引用了未提供的命名参数 `${name}`")))
    }
}

/// 写回尾块的装配实参：设备文本里 `$0`..`$3` 的取值，由调用点一次给足。
///
/// 与 `@原语(...)` 的实参同形：装配不猜任何名字，缺参即错误。
#[derive(Debug, Clone, Copy)]
pub struct EpiArgs<'a> {
    /// `$0`：输出缓冲（行主序）。
    pub out: &'a str,
    /// `$1`：`bias` 步的列偏置数组，长度是列数。
    pub bias: &'a str,
    /// `$1`：`residual` 步的残差数组，长度是元素数。
    pub residual: &'a str,
    /// `$2`：元素数——除 `bias` 外的各步都按它走整块。
    pub count: &'a str,
    /// `$2`：列宽——`bias` 按 `[行][列]` 两层走，`$3` 是行数。
    pub cols: &'a str,
    /// `$3`：`bias` 的行数。
    pub rows: &'a str,
}

/// 可直接注入表达式的激活：三段文本是闭集成员在 C 类语言里的写法，操作数名固定为 `_v`。
///
/// `gelu` / `silu` 不在表里：它们要 `exp` / `erf`，而设备侧数学函数的拼写各语言不同
/// （OpenCL 只有 `exp`、CUDA 有 `expf`、纯 C 要 `<math.h>`），所以那两种形式必须先在
/// `epilogue.<ext>` 里落成设备自己的段落，不能由一个跨设备的表达式表假装通用。
const ACT_EXPR: [(ActKind, &str); 3] = [
    (ActKind::Relu, "(_v > 0.0f) ? _v : 0.0f"),
    (ActKind::LeakyRelu, "(_v >= 0.0f) ? _v : 0.01f * _v"),
    (
        ActKind::HardSigmoid,
        "(_v / 6.0f + 0.5f < 0.0f) ? 0.0f : \
         ((_v / 6.0f + 0.5f > 1.0f) ? 1.0f : (_v / 6.0f + 0.5f))",
    ),
];

/// 量化位宽：`quant` 步骤的 `$3`，与 `nn::epilogue` 的对称网格边界一一对应。
const QUANT_BITS: [(DType, &str); 3] = [(DType::I8, "8"), (DType::I4, "4"), (DType::I2, "2")];

/// 写回尾块装配：按 `steps` 的顺序逐段取设备文本、填占位符、拼成一段可直接放进内核尾部的语句块。
///
/// 六段文本的事实源是设备的 `epilogue.<ext>`（[`prof::Device::epilogue`]）：本函数只做「按形式取段 →
/// 填 `$0`..`$3` → 按声明顺序拼接」，不持有第二份算术。唯一的算术是激活表达式（`ACT_EXPR`），
/// 它与 `nn::unary` 的标量核由执行对拍钉住。
///
/// # Errors
///
/// 设备缺该形式的段、激活形式还没有设备侧段落、`quant` 位宽超出 `i8/i4/i2` 时返回 `invalid`。
pub fn epilogue(
    device: &prof::Device,
    steps: &[Step],
    args: &EpiArgs<'_>,
) -> diagnostic::Result<String> {
    let mut text = String::new();
    for step in steps {
        if !text.is_empty() {
            text.push('\n');
        }
        let block = block(device, step, args)?;
        place(&mut text, &block);
    }
    Ok(text)
}

/// 一步的段文本：先定位形式与四个槽位，再交给 [`splice`] 填。
fn block(device: &prof::Device, step: &Step, args: &EpiArgs<'_>) -> diagnostic::Result<String> {
    let (form, slots) = slots(step, args)?;
    let section = device
        .epilogue_of(form)
        .ok_or_else(|| missing_section(device, form))?;
    let refs: Vec<Arg<'_>> = slots.iter().map(|slot| Arg::Expr(slot)).collect();
    splice(section, &refs, 0)
}

/// 形式与四个槽位：前三步交给 [`slots_head`]、其余交给 [`slots_tail`]，把六臂 match 压成两段、
/// 每段都远低于圈复杂度上限。
fn slots(step: &Step, args: &EpiArgs<'_>) -> diagnostic::Result<(Epilogue, [String; 4])> {
    match step {
        Step::Bias | Step::Act(_) | Step::Residual => slots_head(step, args),
        _ => slots_tail(step, args),
    }
}

/// 前三步的槽位：只有 `act` 的 `$3` 不是调用点给的（那是注入的激活表达式）。
fn slots_head(step: &Step, args: &EpiArgs<'_>) -> diagnostic::Result<(Epilogue, [String; 4])> {
    use Epilogue;
    use Step;
    Ok(match step {
        Step::Bias => (
            Epilogue::Bias,
            [args.out, args.bias, args.cols, args.rows].map(str::to_string),
        ),
        Step::Act(kind) => (
            Epilogue::Act,
            [args.out, args.out, args.count, act(*kind)?].map(str::to_string),
        ),
        Step::Residual => (
            Epilogue::Residual,
            [args.out, args.residual, args.count, args.out].map(str::to_string),
        ),
        _ => return Err(unreachable()),
    })
}

/// 后三步的槽位：`scale` 的 `$1` 是标量字面量，`quant` 的 `$1` / `$3` 是步长与位宽。
fn slots_tail(step: &Step, args: &EpiArgs<'_>) -> diagnostic::Result<(Epilogue, [String; 4])> {
    use Epilogue;
    use Step;
    Ok(match step {
        Step::Scale(alpha) => (
            Epilogue::Scale,
            [
                args.out.to_string(),
                format!("{alpha:?}f"),
                args.count.to_string(),
                args.out.to_string(),
            ],
        ),
        Step::Quant(dtype) => (
            Epilogue::Quant,
            [
                args.out.to_string(),
                format!("{QUANT_SCALE:?}f"),
                args.count.to_string(),
                bits(*dtype)?.to_string(),
            ],
        ),
        Step::Store => (
            Epilogue::Store,
            [args.out, args.out, args.count, args.out].map(str::to_string),
        ),
        _ => return Err(unreachable()),
    })
}

/// 激活形式 → 注入表达式；不在 [`ACT_EXPR`] 里的形式说明它还没有设备段落。
fn act(kind: ActKind) -> diagnostic::Result<&'static str> {
    ACT_EXPR
        .iter()
        .find(|(have, _)| *have == kind)
        .map(|(_, text)| *text)
        .ok_or_else(|| {
            diagnostic::Error::new(
                diagnostic::Code::Invalid,
                format!(
                    "`act:{}` 需要设备侧数学函数（`exp` / `erf`），各语言拼写不同：\
                     先在 `epilogue.<ext>` 里补这一形式的段落，再由本层取用",
                    kind.as_str()
                ),
            )
        })
}

/// `quant` 位宽 → 位数字面量。
fn bits(dtype: DType) -> diagnostic::Result<&'static str> {
    QUANT_BITS
        .iter()
        .find(|(have, _)| *have == dtype)
        .map(|(_, text)| *text)
        .ok_or_else(|| {
            diagnostic::Error::new(
                diagnostic::Code::Invalid,
                format!("`quant:{}` 只接受 i8/i4/i2", dtype.name()),
            )
        })
}

/// 设备文本缺这一形式时的错误：捎上 vendor 与扩展名，一眼能定位是哪份 `epilogue.<ext>`。
fn missing_section(device: &prof::Device, form: Epilogue) -> diagnostic::Error {
    diagnostic::Error::new(
        diagnostic::Code::Invalid,
        format!(
            "设备 `{}` 的 `epilogue.{}` 缺少 `/* epi: {} */` 段",
            device.vendor,
            device.ext,
            form.as_str()
        ),
    )
}

/// 分段守卫：六臂已在前一步分派干净，走到这里是装配器的内部错误。
fn unreachable() -> diagnostic::Error {
    diagnostic::Error::new(diagnostic::Code::Invalid, "写回装配分派不可达")
}

#[cfg(test)]
mod tests {
    use super::*;
    use diagnostic::Code;

    testkit::test_cases! {
    fn splice_substitutes_positional_and_named_holes() {
        let args = [
            Arg::Expr("32"),
            Arg::Named {
                key: "acc",
                value: "out",
            },
        ];
        let out = splice("g=$0; s=$acc", &args, 1).expect("占位符可解析");
        assert_eq!(out, "g=32; s=out");
    }

    fn splice_repeats_the_same_hole() {
        let args = [Arg::Expr("x")];
        let out = splice("$0 -> $0", &args, 1).expect("重复占位");
        assert_eq!(out, "x -> x");
    }

    fn missing_positional_arg_is_an_error() {
        let args = [Arg::Expr("x")];
        let err = splice("$1", &args, 7).expect_err("缺位置参数");
        assert!(err.to_string().contains("$1"), "{err}");
        assert_eq!(err.code(), Code::Invalid, "缺绑定是模板/片段契约不合法");
    }

    fn missing_named_arg_is_an_error() {
        let err = splice("$nope", &[], 7).expect_err("缺命名参数");
        assert!(err.to_string().contains("nope"), "{err}");
        assert_eq!(err.code(), Code::Invalid, "缺绑定是模板/片段契约不合法");
    }

    fn isolated_dollar_stays_literal() {
        let out = splice("cost $ 5", &[], 1).expect("孤立美元号不是占位符");
        assert_eq!(out, "cost $ 5");
    }

    fn place_strips_trailing_semicolon_and_aligns_lines() {
        let mut text = String::from("    call(");
        place(&mut text, "a;\nb;");
        // 缩进对齐到当前整行宽度（`    call(` 共 9 个字符），第二行续到同宽。
        assert_eq!(text, "    call(a;\n         b");
    }

    fn place_without_indent_keeps_flat_text() {
        let mut text = String::new();
        place(&mut text, "only;");
        assert_eq!(text, "only");
    }
    }
}
