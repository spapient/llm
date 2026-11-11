//! 模板词法与片段拼接（DESIGN §4.2、§10.1、§10.2 的 ④）。
//!
//! 解析器只做三件事：切词、识别 `{{槽位}}` 与 `@原语(...)`、其余文本原样保留。位置信息
//! 只记行号——这是"诊断精度换解析器规模"的刻意取舍。

use crate::atom::Prim;
use crate::lower::expand::{OpDecl, TileCombo, Variant};
use core::fmt;
use std::collections::BTreeMap;
use std::path::Path;

/// 模板片段。
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Segment {
    /// 原样保留的文本。
    Text(String),
    /// `{{槽位}}`。
    Slot {
        /// 槽位名。
        name: String,
        /// 行号。
        line: usize,
    },
    /// `@原语(参数)`。
    Call {
        /// 原语。
        prim: Prim,
        /// 参数。
        args: Vec<Arg>,
        /// 行号。
        line: usize,
    },
}

/// `@原语` 的一个参数。
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Arg {
    /// 位置参数，原样拼入片段。
    Expr(String),
    /// 命名参数（`along: BN`、`op: sum`），作为 `$key` 占位符的值。
    Named {
        /// 键。
        key: String,
        /// 值。
        value: String,
    },
}

/// 片段拼接错误（只给行号，不给列）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SubstituteError {
    /// 1 起始的行号。
    pub line: usize,
    /// 说明。
    pub message: String,
}

impl fmt::Display for SubstituteError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "第 {} 行：{}", self.line, self.message)
    }
}

impl std::error::Error for SubstituteError {}

fn error(line: usize, message: impl Into<String>) -> SubstituteError {
    SubstituteError {
        line,
        message: message.into(),
    }
}

/// 列出模板里出现的全部槽位（声明层提前校验用）。
pub fn slots(template: &str) -> Vec<(usize, String)> {
    lex(template)
        .into_iter()
        .filter_map(|segment| match segment {
            Segment::Slot { name, line } => Some((line, name)),
            _ => None,
        })
        .collect()
}

/// 词法：把模板切成 [`Segment`]。
pub fn lex(template: &str) -> Vec<Segment> {
    let mut segments = Vec::new();
    let mut text = String::new();
    let mut line = 1usize;
    let mut chars = template.char_indices().peekable();

    while let Some((index, ch)) = chars.next() {
        if ch == '\n' {
            line += 1;
        }
        if ch == '{'
            && template[index..].starts_with("{{")
            && let Some(end) = template[index + 2..].find("}}")
        {
            let name = template[index + 2..index + 2 + end].trim().to_string();
            if !text.is_empty() {
                segments.push(Segment::Text(std::mem::take(&mut text)));
            }
            segments.push(Segment::Slot { name, line });
            let consumed = 2 + end + 2;
            for _ in 0..consumed - 1 {
                chars.next();
            }
            continue;
        }
        if ch == '@' {
            let rest = &template[index + 1..];
            let name_len = rest
                .chars()
                .take_while(|c| c.is_ascii_alphanumeric() || *c == '_')
                .count();
            if name_len > 0 && rest[name_len..].starts_with('(') {
                let prim_name = &rest[..name_len];
                if let Some(end) = matching_paren(&rest[name_len..])
                    && let Some(prim) = Prim::parse(prim_name)
                {
                    if !text.is_empty() {
                        segments.push(Segment::Text(std::mem::take(&mut text)));
                    }
                    let inner = &rest[name_len + 1..name_len + end];
                    segments.push(Segment::Call {
                        prim,
                        args: parse_args(inner),
                        line,
                    });
                    let consumed = 1 + name_len + end + 1;
                    for _ in 0..consumed - 1 {
                        chars.next();
                    }
                    continue;
                }
            }
        }
        text.push(ch);
    }
    if !text.is_empty() {
        segments.push(Segment::Text(text));
    }
    segments
}

fn matching_paren(text: &str) -> Option<usize> {
    let mut depth = 0usize;
    for (index, ch) in text.char_indices() {
        match ch {
            '(' => depth += 1,
            ')' => {
                depth = depth.checked_sub(1)?;
                if depth == 0 {
                    return Some(index);
                }
            }
            _ => {}
        }
    }
    None
}

fn parse_args(inner: &str) -> Vec<Arg> {
    let mut args = Vec::new();
    let mut current = String::new();
    let mut depth = 0usize;
    let mut pieces = Vec::new();
    for ch in inner.chars() {
        match ch {
            '(' | '[' | '{' => {
                depth += 1;
                current.push(ch);
            }
            ')' | ']' | '}' => {
                depth = depth.saturating_sub(1);
                current.push(ch);
            }
            ',' if depth == 0 => pieces.push(std::mem::take(&mut current)),
            _ => current.push(ch),
        }
    }
    if !current.trim().is_empty() {
        pieces.push(current);
    }
    for piece in pieces {
        let trimmed = piece.trim();
        if trimmed.is_empty() {
            continue;
        }
        if let Some((key, value)) = trimmed.split_once(':') {
            let key = key.trim();
            if !key.is_empty() && key.chars().all(|c| c.is_ascii_alphanumeric() || c == '_') {
                args.push(Arg::Named {
                    key: key.to_string(),
                    value: value.trim().to_string(),
                });
                continue;
            }
        }
        args.push(Arg::Expr(trimmed.to_string()));
    }
    args
}

/// 某个 vendor 的片段集合。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FragmentSet {
    vendor: String,
    ext: String,
    fragments: BTreeMap<Prim, String>,
}

impl FragmentSet {
    /// 新建空集合。
    pub fn new(vendor: &str, ext: &str) -> Self {
        Self {
            vendor: vendor.to_string(),
            ext: ext.to_string(),
            fragments: BTreeMap::new(),
        }
    }

    /// 内置的 CPU 片段（随 crate 一起编译，不依赖工作目录）。
    pub fn builtin_cpu() -> Self {
        let mut set = Self::new("cpu", "c");
        for (prim, source) in [
            (Prim::Load, include_str!("../../prim/cpu/load.c")),
            (Prim::LoadAsync, include_str!("../../prim/cpu/load_async.c")),
            (
                Prim::CommitWait,
                include_str!("../../prim/cpu/commit_wait.c"),
            ),
            (Prim::Store, include_str!("../../prim/cpu/store.c")),
            (Prim::Quant, include_str!("../../prim/cpu/quant.c")),
            (Prim::Dot, include_str!("../../prim/cpu/dot.c")),
            (Prim::Bar, include_str!("../../prim/cpu/bar.c")),
            (Prim::Cvt, include_str!("../../prim/cpu/cvt.c")),
            (Prim::Reduce, include_str!("../../prim/cpu/reduce.c")),
        ] {
            set.insert(prim, source);
        }
        set
    }

    /// 从 `prim/<vendor>/<prim>.<ext>` 读取所有片段。
    pub fn from_dir(root: &Path, vendor: &str, ext: &str) -> Result<Self, String> {
        let mut set = Self::new(vendor, ext);
        let dir = root.join("prim").join(vendor);
        let entries = std::fs::read_dir(&dir)
            .map_err(|error| format!("读取 {} 失败：{error}", dir.display()))?;
        for entry in entries {
            let entry = entry.map_err(|error| error.to_string())?;
            let path = entry.path();
            if path.extension().and_then(|value| value.to_str()) != Some(ext) {
                continue;
            }
            let Some(stem) = path.file_stem().and_then(|value| value.to_str()) else {
                continue;
            };
            let Some(prim) = Prim::parse(stem) else {
                return Err(format!(
                    "`{}` 的文件名 `{stem}` 不在原语闭集内",
                    path.display()
                ));
            };
            let source = std::fs::read_to_string(&path)
                .map_err(|error| format!("读取 {} 失败：{error}", path.display()))?;
            set.insert(prim, source);
        }
        Ok(set)
    }

    /// 加入一段片段。
    pub fn insert(&mut self, prim: Prim, source: impl Into<String>) {
        self.fragments.insert(prim, source.into());
    }

    /// 取一段片段。
    pub fn get(&self, prim: Prim) -> Option<&str> {
        self.fragments.get(&prim).map(String::as_str)
    }

    /// 设备名。
    pub fn vendor(&self) -> &str {
        &self.vendor
    }

    /// 片段所在的扩展名。
    pub fn ext(&self) -> &str {
        &self.ext
    }

    /// 片段数。
    pub fn len(&self) -> usize {
        self.fragments.len()
    }

    /// 是否为空。
    pub fn is_empty(&self) -> bool {
        self.fragments.is_empty()
    }
}

/// 拼接结果：文本 + 参与拼接的片段哈希（缓存指纹用，DESIGN §10.3）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Rendered {
    /// 生成的源文本。
    pub text: String,
    /// 参与拼接的片段与内容哈希。
    pub fragments: Vec<(Prim, u64)>,
}

/// ④ substitute：模板槽位 ← 轴/布局取值，`@原语(...)` ← 片段文本。
pub fn render(
    op: &OpDecl,
    variant: &Variant,
    combo: &TileCombo,
    fragments: &FragmentSet,
) -> Result<Rendered, SubstituteError> {
    let mut text = String::new();
    let mut used: Vec<(Prim, u64)> = Vec::new();

    for segment in lex(&op.template) {
        match segment {
            Segment::Text(part) => text.push_str(&part),
            Segment::Slot { name, line } => {
                if name == "op" {
                    text.push_str(&op.name);
                } else if let Some(value) = combo.get(&name) {
                    text.push_str(&value.to_string());
                } else if let Some(spec) = op.layout(&name) {
                    text.push_str(&spec.as_str());
                } else {
                    return Err(error(
                        line,
                        format!("槽位 `{{{{{name}}}}}` 既不是轴也不是布局槽位"),
                    ));
                }
            }
            Segment::Call { prim, args, line } => {
                let Some(fragment) = fragments.get(prim) else {
                    return Err(error(
                        line,
                        format!(
                            "缺少片段 `prim/{}/{}.{}`：该设备尚未实现 `{prim}`",
                            fragments.vendor(),
                            prim,
                            fragments.ext()
                        ),
                    ));
                };
                let expanded = splice(fragment, &args, line)?;
                // 片段是语句块；模板已经提供了 `;`，这里去掉重复的终结符并对齐缩进。
                let trimmed = expanded.trim_end();
                let trimmed = trimmed.strip_suffix(';').unwrap_or(trimmed);
                let indent: String = text
                    .rsplit('\n')
                    .next()
                    .map(|prefix| " ".repeat(prefix.chars().count()))
                    .unwrap_or_default();
                text.push_str(&trimmed.replace('\n', &format!("\n{indent}")));
                if !used.iter().any(|(have, _)| *have == prim) {
                    used.push((prim, crate::lower::emit::fnv1a(fragment)));
                }
            }
        }
    }

    // 变体只在调度层参与选择；这里显式引用一次，避免"声明了却没生效"。
    let _ = variant;
    Ok(Rendered {
        text,
        fragments: used,
    })
}

/// 片段占位符替换：`$0`..`$9` 是位置参数，`$name` 是命名参数。
fn splice(fragment: &str, args: &[Arg], line: usize) -> Result<String, SubstituteError> {
    let positional: Vec<&str> = args
        .iter()
        .filter_map(|arg| match arg {
            Arg::Expr(text) => Some(text.as_str()),
            Arg::Named { .. } => None,
        })
        .collect();
    let named: BTreeMap<&str, &str> = args
        .iter()
        .filter_map(|arg| match arg {
            Arg::Named { key, value } => Some((key.as_str(), value.as_str())),
            Arg::Expr(_) => None,
        })
        .collect();

    let mut out = String::with_capacity(fragment.len());
    let mut chars = fragment.chars().peekable();
    while let Some(ch) = chars.next() {
        if ch != '$' {
            out.push(ch);
            continue;
        }
        let mut name = String::new();
        while let Some(next) = chars.peek() {
            if next.is_ascii_alphanumeric() || *next == '_' {
                name.push(*next);
                chars.next();
            } else {
                break;
            }
        }
        if name.is_empty() {
            out.push('$');
            continue;
        }
        if let Ok(index) = name.parse::<usize>() {
            let Some(value) = positional.get(index) else {
                return Err(error(
                    line,
                    format!(
                        "片段占用了 `${index}`，但模板只给了 {} 个位置参数",
                        positional.len()
                    ),
                ));
            };
            out.push_str(value);
            continue;
        }
        let Some(value) = named.get(name.as_str()) else {
            return Err(error(line, format!("片段引用了未提供的命名参数 `${name}`")));
        };
        out.push_str(value);
    }
    Ok(out)
}
