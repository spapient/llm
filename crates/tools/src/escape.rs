//! JSON 字符串转义：归档写出唯一需要的文本机制。
//!
//! 纯函数，零依赖。协议卡（[`crate::card`]）与基准记录（[`crate::bench::json`]）共用这一份：
//! 两处各留一张转义表时，同一份归档会在两条路径上写出不同的写法（`\b` 与 `\u0008` 都是合法
//! JSON，但同一套台账里出现两种就不该）。

/// 结构字符与控制字符的短转义：JSON 允许这两种写法，短的那个更好读。
const SHORT: [(char, &str); 5] = [
    ('"', "\\\""),
    ('\\', "\\\\"),
    ('\n', "\\n"),
    ('\r', "\\r"),
    ('\t', "\\t"),
];

/// 写一个 JSON 字符串（连两侧引号）：短转义查表，其余控制字符按 `\u00xx`，其他字符原样透出。
pub fn quote(out: &mut String, text: &str) {
    out.push('"');
    for ch in text.chars() {
        match SHORT.iter().find(|(from, _)| *from == ch) {
            Some((_, literal)) => out.push_str(literal),
            None if ch.is_control() => out.push_str(&format!("\\u{:04x}", ch as u32)),
            None => out.push(ch),
        }
    }
    out.push('"');
}

#[cfg(test)]
mod tests {
    use super::*;

    fn quoted(text: &str) -> String {
        let mut out = String::new();
        quote(&mut out, text);
        out
    }

    #[test]
    fn quotes_and_backslashes_are_escaped_and_unicode_passes_through() {
        assert_eq!(quoted("a\"b\\c"), "\"a\\\"b\\\\c\"");
        assert_eq!(quoted("延迟"), "\"延迟\"");
    }

    #[test]
    fn short_escapes_win_for_the_common_controls() {
        assert_eq!(quoted("a\nb\tc\rd"), "\"a\\nb\\tc\\rd\"");
    }

    /// 其余控制字符（含 `\x07` 与 U+0085）一律走 `\u00XX`：裸放进 JSON 会让回读器读到非法字节。
    #[test]
    fn other_controls_become_unicode_escapes() {
        assert_eq!(quoted("\u{7}"), "\"\\u0007\"");
        assert_eq!(quoted("\u{85}"), "\"\\u0085\"");
    }
}
