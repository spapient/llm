//! JSON 字符串转义：归档写出唯一需要的文本机制。
//!
//! 纯函数，零依赖。协议卡（[`crate::card`]）归档时共用这一份：多留一张转义表时，同一份
//! 归档会在两条路径上写出不同的写法（`\b` 与 `\u0008` 都是合法 JSON，但同一套台账里
//! 出现两种就不该）。

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
