//! 单文件行数棘轮。
//!
//! 重复块检测**不在本模块**：主门禁用 jscpd（`.jscpd.json` 的 `threshold` 重复率硬阈值），
//! 语义巡检用 `cargo dupes`（AST 归一化识别近重复）。这里只保留两者都不管的单文件行数上限。
//! 纯函数：调用方传入文本，超限时返回实际行数，不读盘、不打印。

use crate::gate::Gate;

/// 文件行数超限时返回实际行数。
#[must_use]
pub fn file_lines(text: &str, gate: &Gate) -> Option<usize> {
    let lines = text.lines().count();
    (lines > gate.max_file_lines).then_some(lines)
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use super::*;

    fn gate() -> Gate {
        Gate {
            max_ccn: 5,
            max_nloc: 60,
            simd_nloc: 96,
            max_file_lines: 10,
            debt: BTreeMap::new(),
        }
    }

    #[test]
    fn an_oversized_file_is_flagged() {
        let text = (0..12).map(|_| "x").collect::<Vec<_>>().join("\n");
        assert_eq!(file_lines(&text, &gate()), Some(12));
    }

    #[test]
    fn a_file_within_budget_is_silent() {
        let text = (0..8).map(|_| "x").collect::<Vec<_>>().join("\n");
        assert_eq!(file_lines(&text, &gate()), None);
    }
}
