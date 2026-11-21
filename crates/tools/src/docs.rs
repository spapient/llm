//! 文档漂移：markdown 里相对链接的目标文件是否存在。

/// 一个待校验的相对链接：目标路径与所在行号。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Link {
    /// 目标（去掉 `#anchor` 部分）。
    pub target: String,
    /// 文档内行号（从 1 起）。
    pub line: usize,
}

/// 提取全部相对链接。
#[must_use]
pub fn links(text: &str) -> Vec<Link> {
    let mut out = Vec::new();
    for (line_no, line) in text.lines().enumerate() {
        line_links(line, line_no + 1, &mut out);
    }
    out
}

/// 一行的全部相对链接；外部链接、纯锚点与空路径都不计。
fn line_links(line: &str, line_no: usize, out: &mut Vec<Link>) {
    let mut rest = line;
    while let Some(open) = rest.find("](") {
        let after = &rest[open + 2..];
        let Some(close) = after.find(')') else {
            return;
        };
        let raw = after[..close].trim();
        let target = raw.split('#').next().unwrap_or(raw).trim();
        if is_relative(raw) && !target.is_empty() {
            out.push(Link {
                target: target.to_string(),
                line: line_no,
            });
        }
        rest = &after[close + 1..];
    }
}

/// 是否是仓库内相对路径；协议头与纯锚点都不是。
fn is_relative(raw: &str) -> bool {
    let external = ["http://", "https://", "mailto:", "ftp://"];
    !external.iter().any(|prefix| raw.starts_with(prefix)) && !raw.starts_with('#')
}
