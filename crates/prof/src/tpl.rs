//! 模板词法：把 `<name>.k` 切成文本、`{{槽位}}` 与 `@原语(...)`（cutile README §4.2）。
//!
//! 词法属于声明层：它解释的是 `ops/` 里的文本格式，声明解析（[`crate::decl::OpDecl`]）要用它提前
//! 拦住 `{{GMAXX}}` 这类笔误。片段拼接（`@原语` → 片段文本）才是机制，留在 `cutile`。
//!
//! 位置信息只记行号——这是"诊断精度换解析器规模"的刻意取舍。
//!
//! # 切一次，查三遍
//!
//! 装载期要把同一份模板校验三遍：槽位都得有定义、每条 `@原语` 都得被申请、调用点的能力参数
//! 都得落在该变体的能力签名之内。三次校验看的是同一份切分结果，所以 [`Segment`] **借用**模板
//! 正文——文本段与槽位名都是 `&str`，一个字符都不拷贝。只有 `@原语` 的参数表需要一次小分配
//! （每处调用一次，与模板里写了几条 `@` 同阶）。

use crate::atom::Prim;
use crate::text::{matching_paren, split_top_level};
use diagnostic::Error;

/// 模板片段。
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Segment<'a> {
    /// 原样保留的文本。
    Text(&'a str),
    /// `{{槽位}}`。
    Slot {
        /// 槽位名（已去掉首尾空白）。
        name: &'a str,
        /// 行号。
        line: usize,
    },
    /// `@原语(参数)`。
    Call {
        /// 原语。
        prim: Prim,
        /// 参数。
        args: Vec<Arg<'a>>,
        /// 行号。
        line: usize,
    },
}

/// `@原语` 的一个参数。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Arg<'a> {
    /// 位置参数，原样拼入片段。
    Expr(&'a str),
    /// 命名参数（`along: BN`、`op: sum`），作为 `$key` 占位符的值。
    Named {
        /// 键。
        key: &'a str,
        /// 值。
        value: &'a str,
    },
}

/// 一个参数的两种写法。
impl Arg<'_> {
    /// 命名参数解成 `(键, 值)`；位置参数返回 `None`。
    pub fn named(&self) -> Option<(&str, &str)> {
        match self {
            Self::Expr(_) => None,
            Self::Named { key, value } => Some((key, value)),
        }
    }
}

/// 一次词法的结果：切好的片段。
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Template<'a> {
    segments: Vec<Segment<'a>>,
}

impl<'a> Template<'a> {
    /// 切分模板正文。
    pub fn parse(template: &'a str) -> Self {
        let mut segments = Vec::new();
        let mut cursor = 0usize;
        let mut text_start = 0usize;
        let mut line = 1usize;

        while cursor < template.len() {
            let rest = &template[cursor..];
            let Some((consumed, segment)) = construct(rest, line) else {
                // 普通字符：推进一个字符，顺带数换行；文本段不逐字拷贝，最后整段切走。
                let width = rest.chars().next().map_or(0, char::len_utf8);
                line += rest[..width].matches('\n').count();
                cursor += width;
                continue;
            };
            if text_start < cursor {
                segments.push(Segment::Text(&template[text_start..cursor]));
            }
            segments.push(segment);
            cursor = advance(template, cursor, consumed, &mut line);
            text_start = cursor;
        }
        if text_start < template.len() {
            segments.push(Segment::Text(&template[text_start..]));
        }
        Self { segments }
    }

    /// 切好的片段。
    pub fn segments(&self) -> &[Segment<'a>] {
        &self.segments
    }

    /// 拆成片段（[`lex`] 的等价形式）。
    pub fn into_segments(self) -> Vec<Segment<'a>> {
        self.segments
    }

    /// 逐个槽位：`(行号, 名字)`。
    pub fn slots(&self) -> impl Iterator<Item = (usize, &'a str)> + '_ {
        self.segments.iter().filter_map(|segment| match segment {
            Segment::Slot { name, line } => Some((*line, *name)),
            _ => None,
        })
    }

    /// 逐个原语调用：`(行号, 原语, 参数)`。
    pub fn calls<'s>(&'s self) -> impl Iterator<Item = (usize, Prim, &'s [Arg<'a>])> + use<'s, 'a> {
        self.segments.iter().filter_map(|segment| match segment {
            Segment::Call { prim, args, line } => Some((*line, *prim, args.as_slice())),
            _ => None,
        })
    }
}

/// 模板拼接错误（只给行号，不给列）。
pub fn error(line: usize, message: impl Into<String>) -> Error {
    crate::text::line_error(line, message)
}

/// 词法：把模板切成 [`Segment`]。
pub fn lex(template: &str) -> Vec<Segment<'_>> {
    Template::parse(template).into_segments()
}

/// 跳过一个构造：推进 `consumed` 个字节，并把其中的换行计入行号。
fn advance(template: &str, from: usize, consumed: usize, line: &mut usize) -> usize {
    let to = from + consumed;
    *line += crate::text::count_newlines(&template[from..to]);
    to
}

/// 认一个构造：`(消费的字节数, 片段)`。槽位在前、原语调用在后，都不是就返回 `None`。
fn construct<'a>(rest: &'a str, line: usize) -> Option<(usize, Segment<'a>)> {
    if rest.starts_with("{{") {
        return slot(rest, line);
    }
    let (consumed, prim, args) = call(rest)?;
    Some((consumed, Segment::Call { prim, args, line }))
}

/// `{{槽位}}`：名字去首尾空白，行号取构造起点；未闭合的写法退回文本。
fn slot<'a>(rest: &'a str, line: usize) -> Option<(usize, Segment<'a>)> {
    let end = rest[2..].find("}}")?;
    let name = rest.get(2..2 + end)?.trim();
    Some((2 + end + 2, Segment::Slot { name, line }))
}

/// 识别 `@原语(...)`，返回 `(消费的字节数, 原语, 参数)`。
fn call<'a>(rest: &'a str) -> Option<(usize, Prim, Vec<Arg<'a>>)> {
    let after = rest.strip_prefix('@')?;
    let (name_len, prim) = named(after)?;
    let open = after.get(name_len..).filter(|tail| tail.starts_with('('))?;
    let end = matching_paren(open, 0)?;
    Some((1 + name_len + end + 1, prim, parse_args(&open[1..end])))
}

/// 调用名：名字占的字节数与它在原语闭集里的身份。
fn named(text: &str) -> Option<(usize, Prim)> {
    let skip = |ch: char| !ch.is_ascii_alphanumeric() && ch != '_';
    let len = text.find(skip).unwrap_or(text.len());
    Some((len, Prim::parse(&text[..len])?))
}

/// 按顶层逗号切分参数；`key: value` 且键是标识符时是命名参数。切分原语是 [`split_top_level`]。
fn parse_args(inner: &str) -> Vec<Arg<'_>> {
    let mut args = Vec::new();
    for (_, piece) in split_top_level(inner) {
        push_arg(&mut args, piece);
    }
    args
}

/// 一个参数：去空白后空则丢弃，`键: 值` 且键是标识符则是命名参数，否则是位置参数。
fn push_arg<'a>(args: &mut Vec<Arg<'a>>, piece: &'a str) {
    let piece = piece.trim();
    if piece.is_empty() {
        return;
    }
    let named = piece.split_once(':').and_then(|(key, value)| {
        let key = key.trim();
        is_key(key).then_some(Arg::Named {
            key,
            value: value.trim(),
        })
    });
    args.push(named.unwrap_or(Arg::Expr(piece)));
}

/// 命名参数的键：非空、且只由字母数字下划线组成。
fn is_key(key: &str) -> bool {
    !key.is_empty()
        && key
            .chars()
            .all(|ch| ch.is_ascii_alphanumeric() || ch == '_')
}

pub mod formula {
    //! 骨架的表达式协议：一条 einsum 表达式、一张操作数表、一个按实现分开的缓存单元格。
    use crate::einsum::{Formula, FormulaCache, OperandSpec};
    use diagnostic::Error;

    /// 表达式的主人：一副骨架怎么声明表达式、缓存在哪、解析失败报什么帧。
    pub trait FormulaOwner {
        /// 一条 einsum 表达式。
        const EXPR: &'static str;

        /// 表达式各项的位宽与方向，顺序与 [`FormulaOwner::EXPR`] 的项一一对应。
        const OPERANDS: &'static [OperandSpec];

        /// 解析失败时的诊断帧：点名是哪一族在读表达式（如 `prof::unary::formula`）。
        const FRAME: &'static str;

        /// 本实现的缓存单元格。
        fn cache() -> &'static FormulaCache;

        /// 表达式：第一次调用时解析并校验，之后复用。
        ///
        /// # Errors
        ///
        /// 表达式与操作数表不自洽时返回说明（帧即 [`FormulaOwner::FRAME`]）。
        fn formula() -> Result<&'static Formula, Error> {
            Self::cache().get(Self::EXPR, Self::OPERANDS, Self::FRAME)
        }
    }

    /// 一个实现自己的表达式缓存单元格。
    #[macro_export]
    macro_rules! formula_cell {
        () => {{
            static CACHE: $crate::einsum::FormulaCache = $crate::einsum::FormulaCache::new();
            &CACHE
        }};
    }

    /// 装配一份 [`FormulaOwner`]：表达式、操作数表、诊断帧与缓存单元格一次写全，
    /// 避免每族骨架把同一份 impl 样板抄一遍。
    #[macro_export]
    macro_rules! formula_owner {
        ($ty:ident, $expr:expr, $operands:expr, $frame:expr) => {
            impl $crate::formula::FormulaOwner for $ty {
                const EXPR: &'static str = $expr;

                const OPERANDS: &'static [$crate::einsum::OperandSpec] = &$operands;

                const FRAME: &'static str = $frame;

                fn cache() -> &'static $crate::einsum::FormulaCache {
                    $crate::formula_cell!()
                }
            }
        };
    }
}

pub mod guard {
    //! 缓冲长度校验：`(实际, 需要, 名称)` 逐条比对，失败即带名称与两个数字返回。
    use diagnostic::{Code, Error, Result};

    /// 一条长度约束：缓冲不足时报出名称、实际长度与下限。
    fn check_count(actual: usize, required: usize, what: &str) -> Result<()> {
        if actual < required {
            return Err(Error::new(
                Code::Invalid,
                format!("{what}: 缓冲 {actual}，需要 {required}"),
            ));
        }
        Ok(())
    }

    /// 一组约束，逐条短路；把多处 `?` 收进一个出口，调用点只写一次。
    pub fn check_all(bounds: &[(usize, usize, &str)]) -> Result<()> {
        for &(actual, required, what) in bounds {
            check_count(actual, required, what)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    testkit::test_cases! {
    fn text_slots_and_calls_are_split_without_copying() {
        let template = "void f() {\n    float d[{{GMAX}}];\n    @dot(acc, d, xs, gsz);\n}\n";
        let parsed = Template::parse(template);

        assert_eq!(
            parsed.slots().collect::<Vec<_>>(),
            [(2, "GMAX")],
            "槽位名借用正文，不去空白之外做任何改写"
        );
        let calls: Vec<_> = parsed.calls().map(|(line, prim, _)| (line, prim)).collect();
        assert_eq!(calls, [(3, Prim::Dot)]);
        // 文本段拼回去必须还原原文：切分只拿走两种构造，不能吃掉任何字符。
        let text_only: String = parsed
            .segments()
            .iter()
            .filter_map(|segment| match segment {
                Segment::Text(text) => Some(*text),
                _ => None,
            })
            .collect();
        assert_eq!(
            text_only,
            template
                .replace("{{GMAX}}", "")
                .replace("@dot(acc, d, xs, gsz)", "")
        );
        assert_eq!(lex(template), parsed.into_segments());
    }

    /// 行号必须跟着被跳过的构造一起前进：否则跨行调用的下一条诊断指错行。
    fn the_line_number_survives_a_multi_line_call() {
        let template = "void f() {\n    @bar(\n        tok,\n    );\n    {{typo}}\n}\n";
        let parsed = Template::parse(template);
        assert_eq!(parsed.slots().collect::<Vec<_>>(), [(5, "typo")]);
        let (line, prim, _) = parsed.calls().next().expect("一条调用");
        assert_eq!(prim, Prim::Bar);
        assert_eq!(line, 2);
    }

    fn named_and_positional_arguments_keep_their_text() {
        let parsed = Template::parse("@cvt(v, acc, sat: off, along: 32)");
        let (_, prim, args) = parsed.calls().next().expect("一条调用");
        assert_eq!(prim, Prim::Cvt);
        assert_eq!(
            args,
            [
                Arg::Expr("v"),
                Arg::Expr("acc"),
                Arg::Named {
                    key: "sat",
                    value: "off"
                },
                Arg::Named {
                    key: "along",
                    value: "32"
                },
            ]
        );
    }

    fn an_unknown_primitive_stays_text_instead_of_being_guessed() {
        let parsed = Template::parse("@nosuchprim(y, n, out);");
        assert!(parsed.calls().next().is_none(), "闭集外不猜");
        assert!(matches!(
            parsed.segments(),
            [Segment::Text("@nosuchprim(y, n, out);")]
        ));
    }

    fn an_unclosed_slot_or_call_is_left_alone() {
        for template in ["{{GMAX", "@dot(acc, d"] {
            let parsed = Template::parse(template);
            assert_eq!(
                parsed.segments(),
                [Segment::Text(template)],
                "写坏了应当退回文本，而不是产生半个构造"
            );
        }
    }
    }
}
