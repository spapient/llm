//! `ledger` 命令的台账矩阵排版：内核只给结构化结论，状态文本与 markdown 在这里出（L7 前端）。

use cutile::atom::Prim;
use cutile::ledger::{BackendRow, LedgerCell, Status};

/// 台账矩阵：证据汇总表 + 原语×后端对照表 + 逐条证据说明。
pub fn render(rows: &[BackendRow]) -> String {
    let mut out = String::from("# 后端能力台账\n\n");
    out.push_str(&evidence_table(rows));
    out.push_str(&prim_matrix(rows));
    out.push_str(&evidence_notes(rows));
    out
}

/// 设备证据表：本轮执行到的依据、设备/自检计数、语言门结论与本机可用性。
fn evidence_table(rows: &[BackendRow]) -> String {
    let mut out = String::new();
    out.push_str("## 设备证据（本轮执行）\n\n");
    out.push_str("| 设备 | 型号 | emit | 登记工具链 | 依据 | 设备见证 | 自检 | 语言门 | 本机 |\n");
    out.push_str("|---|---|---|---|---|---:|---:|---|---|\n");
    for row in rows {
        let meta = &row.profile.meta;
        out.push_str(&format!(
            "| `{}` | {} | {} | {} {} | {} | {} | {} | {} | {} |\n",
            meta.vendor,
            meta.id,
            meta.emit,
            meta.tool,
            meta.tool_ver,
            row.basis,
            row.device,
            row.host,
            row.gate,
            if row.available { "是" } else { "否" },
        ));
    }
    out
}

/// 原语 × 后端对照表：一格一条供给，多条用 `<br>` 叠在格里。
fn prim_matrix(rows: &[BackendRow]) -> String {
    let vendors: Vec<&str> = rows
        .iter()
        .map(|row| row.profile.meta.vendor.as_str())
        .collect();

    let mut out = String::new();
    out.push_str("\n## 原语 × 后端\n\n| 原语 | ");
    out.push_str(&vendors.join(" | "));
    out.push_str(" |\n| --- | ");
    for _ in &vendors {
        out.push_str("--- | ");
    }
    out.push('\n');

    for prim in Prim::ALL {
        out.push_str(&format!("| `{prim}` | "));
        for row in rows {
            let cell = row
                .cells
                .iter()
                .filter(|cell| cell.prim == *prim)
                .map(cell_text)
                .collect::<Vec<_>>()
                .join("<br>");
            out.push_str(&format!("{cell} | "));
        }
        out.push('\n');
    }
    out
}

/// 逐条证据说明：未声明的格子不进说明，实现名与理由只在这里展开。
fn evidence_notes(rows: &[BackendRow]) -> String {
    let mut out = String::new();
    out.push_str("\n## 证据说明\n\n");
    for row in rows {
        let vendor = row.profile.meta.vendor.as_str();
        for cell in &row.cells {
            if cell.status == Status::Unprovided {
                continue;
            }
            out.push_str(&format!(
                "- `{}` {} · {} {} — {}\n",
                vendor, cell.prim, cell.impl_name, cell.grade, cell.note
            ));
        }
    }
    out
}

/// 矩阵格：状态标签 + 档案等级；实现名与理由放在证据说明里，避免格子塞字。
fn cell_text(cell: &LedgerCell) -> String {
    format!("{} · {}", cell.status.label(), cell.grade)
}
