//! 声明层共享文本扫描原语：括号深度、顶层逗号切分、行号计数与带行号的诊断错误。
//!
//! 这些原语不绑定模板或签名格式：[`crate::sig`] 解析入口形参，[`crate::tpl`] 解析模板调用，
//! [`crate::epilogue`] 计算换行位置，共用同一份扫描口径，避免括号/逗号/行号在三处漂移。

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
pub(crate) fn line_error(line: usize, message: impl Into<String>) -> Error {
    Error::new(Code::Invalid, message).with(format!("第 {line} 行"))
}

/// 一段文本里的换行数：绝对行号与增量行号共用。
pub(crate) fn count_newlines(text: &str) -> usize {
    text.bytes().filter(|byte| *byte == b'\n').count()
}

/// 从 `open` 处的 `(` 找配对的 `)`：深度归零的那一格就是答案，返回绝对下标。
pub(crate) fn matching_paren(text: &str, open: usize) -> Option<usize> {
    let mut depth = 0usize;
    for (idx, ch) in text[open..].char_indices() {
        depth = depth_step(depth, ch)?;
        if depth == 0 {
            return Some(open + idx);
        }
    }
    None
}

/// 按顶层逗号切分：括号内的逗号不算顶层。返回每个片段的 `(字节偏移, 原文)`。
pub(crate) fn split_top_level(inner: &str) -> Vec<(usize, &str)> {
    let mut parts = Vec::new();
    let mut depth = 0usize;
    let mut start = 0usize;
    for (idx, ch) in inner.char_indices() {
        let at_comma = ch == ',' && depth == 0;
        depth = depth_step(depth, ch).unwrap_or_default();
        if at_comma {
            parts.push((start, &inner[start..idx]));
            start = idx + 1;
        }
    }
    parts.push((start, &inner[start..]));
    parts
}
