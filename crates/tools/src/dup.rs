//! 单文件行数棘轮。

use crate::gate::{Gate, Metric};

/// 文件行数超限时返回实际行数；限值由门禁给出，本函数不持有阈值。
#[must_use]
pub fn file_lines(text: &str, gate: &Gate) -> Option<usize> {
    let lines = text.lines().count();
    (lines > gate.limit(Metric::Lines)).then_some(lines)
}
