//! 文档漂移：markdown 里相对链接的目标文件是否存在。
//!
//! 纯函数：提取文档中的相对链接（排除 http/https/mailto 与纯锚点），由调用方核对目标文件。
//! 不自己读盘，因此在测试里用任意虚拟文档即可验证，不依赖仓库布局。

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn relative_links_are_collected_and_anchors_stripped() {
        let doc = "见 [架构](ARCHITECTURE.md#二) 与 [代码](../crates/tools/src/lib.rs)。";
        assert_eq!(
            links(doc),
            vec![
                Link {
                    target: "ARCHITECTURE.md".to_string(),
                    line: 1
                },
                Link {
                    target: "../crates/tools/src/lib.rs".to_string(),
                    line: 1
                },
            ]
        );
    }

    #[test]
    fn external_and_pure_anchor_links_are_skipped() {
        let doc = "[site](https://x.io/a) [top](#section) [mail](mailto:a@b.c)";
        assert!(links(doc).is_empty());
    }
}
