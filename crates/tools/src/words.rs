//! 一概念一词（N5）：同一个概念只准一种拼写。
//!
//! 判据落在**标识符**上而不是文本上：`lex::parse` 产出的 `Ident` 天然不含字符串字面量与
//! 注释，所以设备源里的 Metal 选择器 `[[buffer(0)]]`、文档里的历史提法都不会被误伤——
//! 这也正是旧 `name-lint.sh` 只能查文件名的原因，它拿 grep 看文本，看见什么都算数。
//!
//! 检查顺序是「整名先过外来白名单，再过分段」：`MTLBuffer` / `char_indices` 这类外来 API
//! 由厂商或标准库定名，整名豁免；`isBuffers` / `cb_buffer` 这类本项目自造的名字则拆段后
//! 逐段比对规范拼写。

use crate::lex::{self, Token};

/// 非规范拼写 → 规范拼写。比较统一转小写，因此 `Buffer` / `BUFFER` 一并命中。
///
/// 词表与 `docs/PROMPT-S++-AUDIT.md` 的 Unix 缩写白名单对齐：`buf` 与 `idx` 在名单内，
/// 全拼不在，所以收敛方向是「向缩写靠」而不是反过来。
///
/// 只收**同一个概念**的两种拼法。`buffered` 不在表内：它是"已缓冲的量"，一个独立概念，
/// 不是 `buffer` 的另一种写法——把它折成 `buf` 反而制造两个概念共用一个词。
const CANON: &[(&str, &str)] = &[
    ("buffer", "buf"),
    ("buffers", "bufs"),
    ("index", "idx"),
    ("indexes", "idxs"),
    ("indices", "idxs"),
];

/// 外来 API 的整名白名单：段里含非规范词，但拼写归厂商或标准库，本项目无权改写。
///
/// 每一条都是**对方头文件里逐字写死**的名字：改了就调不到那个符号。本项目自有的
/// `MetalBuffer` 不在名单里——它已收敛成 `MetalBuf`，正是这条规则的目标。
pub const FOREIGN: &[&str] = &[
    "MTLBuffer",
    "MTLCommandBuffer",
    "MTLCommandBufferStatusCompleted",
    "commandBuffer",
    "setBuffer",
    "setBuffers",
    "newBufferWithLength",
    "newBufferWithBytes",
    "atIndex",
    "char_indices",
    "match_indices",
    "Index",
    "IndexMut",
];

/// 外来**成员**名：只在 `.成员` 位置豁免（`fn index` / `index:` 仍照查）。
///
/// 唯一一条是 proptest 的 `sample::Index::index`——测试里 `at.index(len)` 取随机下标，
/// 换了就没有收缩能力。它不在本项目名下，词表管不到它。
const FOREIGN_MEMBERS: &[&str] = &["index"];

/// 一处违规：位置、原文与该概念的规范拼写。
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct Hit {
    /// 行号（1-based）。
    pub line: usize,
    /// 违规标识符原文。
    pub found: String,
    /// 该概念应有的拼写。
    pub canon: &'static str,
}

impl Hit {
    /// 一行诊断：文件路径由调用方拼在前面。
    #[must_use]
    pub fn describe(&self) -> String {
        format!("第 {} 行 `{}` 应写成 `{}`", self.line, self.found, self.canon)
    }
}

/// 扫一段 Rust 源码，按出现顺序返回所有非规范标识符。
///
/// 源码无法解析时返回词法错误而不是空表：静默返回空表会让门禁在坏文件上变绿。
pub fn scan(src: &str) -> diagnostic::Result<Vec<Hit>> {
    let tokens = lex::parse(src)?;
    let mut hits = Vec::new();
    walk(&tokens, &mut hits);
    Ok(hits)
}

/// 递归遍历 token 流：分组里的标识符同样要查，`.成员` 位置要传给 [`checked`]。
fn walk(tokens: &[Token], hits: &mut Vec<Hit>) {
    let mut after_dot = false;
    for token in tokens {
        match token {
            Token::Ident { text, pos } => {
                if !exempt(text, after_dot) {
                    hits.extend(offences(text, pos.line));
                }
            }
            Token::Group { items, .. } => walk(items, hits),
            _ => {}
        }
        after_dot = matches!(token, Token::Punct { ch: '.', .. });
    }
}

/// 这个标识符是不是外来拼写：整名在 [`FOREIGN`] 里，或成员名在 [`FOREIGN_MEMBERS`] 里。
fn exempt(ident: &str, after_dot: bool) -> bool {
    FOREIGN.contains(&ident) || (after_dot && FOREIGN_MEMBERS.contains(&ident))
}

/// 一个标识符里的每一段非规范拼写各记一条。
fn offences(ident: &str, line: usize) -> Vec<Hit> {
    let mut out = Vec::new();
    for segment in segments(ident) {
        let lower = segment.to_ascii_lowercase();
        let hit = CANON.iter().find(|(bad, _)| *bad == lower.as_str());
        if let Some((_, canon)) = hit {
            out.push(Hit {
                line,
                found: ident.to_owned(),
                canon,
            });
        }
    }
    out
}

/// 拆成词段：先按 `_` 拆蛇形，再按大小写拆驼峰。
fn segments(ident: &str) -> Vec<&str> {
    ident.split('_').flat_map(camel).collect()
}

/// 把一段驼峰/全大写/小写串切成词：`MTLBuffer` → `MTL`、`Buffer`。
fn camel(part: &str) -> Vec<&str> {
    let bytes = part.as_bytes();
    let mut out = Vec::new();
    let mut start = 0;
    for idx in 1..bytes.len() {
        if splits(bytes[idx - 1], bytes[idx], bytes.get(idx + 1).copied()) {
            out.push(&part[start..idx]);
            start = idx;
        }
    }
    if start < part.len() {
        out.push(&part[start..]);
    }
    out
}

/// 这一对相邻字节是不是词边界。
fn splits(prev: u8, cur: u8, next: Option<u8>) -> bool {
    leaves_snake(prev, cur) || leaves_acronym(prev, cur, next)
}

/// 小写或数字接大写：`isBuffer` 在这里断开。
fn leaves_snake(prev: u8, cur: u8) -> bool {
    (prev.is_ascii_lowercase() || prev.is_ascii_digit()) && cur.is_ascii_uppercase()
}

/// 全大写串的末尾接小写开头的新词：`MTLBuffer` 在这里断开。
fn leaves_acronym(prev: u8, cur: u8, next: Option<u8>) -> bool {
    prev.is_ascii_uppercase()
        && cur.is_ascii_uppercase()
        && next.is_some_and(|after| after.is_ascii_lowercase())
}
