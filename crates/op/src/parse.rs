//! 声明层解析原语：括号深度、行号与带行号的错误。
//!
//! `decl` / `sig` / `sch` / `tpl` 四个解析器各自解析一种声明文本（TOML 表、C 签名、
//! 调度串、模板词法），但共用同一套「括号配对 / 行号 / 行号错误」的底层原语。这里是它们
//! 的唯一实现；新增声明解析器只从这里取原语，不再各自写一遍。

use diagnostic::{Code, Error};

/// 括号深度的一步：`(` 加深、`)` 回退（退到 0 之前即不配对），其它字符不动。
pub(crate) fn depth_step(depth: usize, ch: char) -> Option<usize> {
    match ch {
        '(' => Some(depth + 1),
        ')' => depth.checked_sub(1),
        _ => Some(depth),
    }
}

/// 带行号的解析错误：`Error::new(Code::Invalid, message)` 再挂一个「第 N 行」上下文。
///
/// 绝对行号（`sig`）与增量行号（`tpl`）各自跟踪行，但报错的构造方式只有这一份。
pub(crate) fn line_error(line: usize, message: impl Into<String>) -> Error {
    Error::new(Code::Invalid, message).with(format!("第 {line} 行"))
}

/// 一段文本里的换行数：绝对行号（`sig::line_of`）与增量行号（`tpl::advance`）共用。
pub(crate) fn count_newlines(text: &str) -> usize {
    text.bytes().filter(|byte| *byte == b'\n').count()
}

/// 从 `open` 处的 `(` 找配对的 `)`：深度归零的那一格就是答案，返回绝对下标。
///
/// 调用方保证 `text[open..]` 以 `(` 开头；配对失败（右括号不够）返回 `None`。
pub(crate) fn matching_paren(text: &str, open: usize) -> Option<usize> {
    let mut depth = 0usize;
    for (index, ch) in text[open..].char_indices() {
        depth = depth_step(depth, ch)?;
        if depth == 0 {
            return Some(open + index);
        }
    }
    None
}

/// 按顶层逗号切分：括号内的逗号不算顶层。返回每个片段的 `(字节偏移, 原文)`。
///
/// 括号不配对按「没看见」处理（切分不校验，配对交给 [`matching_paren`]）；片段不裁剪、
/// 不丢弃，由调用方决定各自的 trim / 过滤口径。
pub(crate) fn split_top_level(inner: &str) -> Vec<(usize, &str)> {
    let mut parts = Vec::new();
    let mut depth = 0usize;
    let mut start = 0usize;
    for (index, ch) in inner.char_indices() {
        let at_comma = ch == ',' && depth == 0;
        depth = depth_step(depth, ch).unwrap_or_default();
        if at_comma {
            parts.push((start, &inner[start..index]));
            start = index + 1;
        }
    }
    parts.push((start, &inner[start..]));
    parts
}
