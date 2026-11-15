//! 单文件行数棘轮。
//!
//! 重复块检测**不在本模块**：主门禁用 jscpd（`.jscpd.json` 的 `threshold` 重复率硬阈值），
//! 语义巡检用 `cargo dupes`（AST 归一化识别近重复）。这里只保留两者都不管的单文件行数上限。
//! 纯函数：调用方传入文本，超限时返回实际行数，不读盘、不打印。

use crate::gate::{Gate, Metric};

/// 文件行数超限时返回实际行数；限值由门禁给出，本函数不持有阈值。
#[must_use]
pub fn file_lines(text: &str, gate: &Gate) -> Option<usize> {
    let lines = text.lines().count();
    (lines > gate.limit(Metric::Lines)).then_some(lines)
}

#[cfg(test)]
#[path = "dup_tests.rs"]
mod tests;
