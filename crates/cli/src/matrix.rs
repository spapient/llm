//! 台账的表格排版：证据汇总表、原语 × 后端矩阵、逐条说明与档案头/原语表。

use cutile::atom::Prim;
use cutile::plan::Verdict;
use cutile::prof::Profile;
use diagnostic::render::{Align, Sheet};
use llm_optim::ledger::{BackendRow, LedgerCell, Status};
use llm_optim::verify::Verified;
use prof::decl::Fusion;
use std::fmt::Write as _;

/// 台账矩阵：证据汇总表 + 原语×后端对照表 + 逐条证据说明。
pub(super) fn ledger(rows: &[BackendRow]) -> String {
    let mut out = String::from("# 后端能力台账\n\n");
    out.push_str(&ledger_evidence_table(rows));
    out.push_str(&ledger_prim_matrix(rows));
    out.push_str(&ledger_evidence_notes(rows));
    out
}

/// 设备证据表：本轮执行到的依据、设备/自检计数、语言门结论与本机可用性。
fn ledger_evidence_table(rows: &[BackendRow]) -> String {
    let mut table = Sheet::new([
        "设备",
        "型号",
        "emit",
        "登记工具链",
        "依据",
        "设备见证",
        "自检",
        "语言门",
        "本机",
    ])
    .align([
        Align::Left,
        Align::Left,
        Align::Left,
        Align::Left,
        Align::Left,
        Align::Right,
        Align::Right,
        Align::Left,
        Align::Left,
    ]);
    for row in rows {
        let meta = &row.profile.meta;
        table = table.row([
            format!("`{}`", meta.vendor),
            meta.id.clone(),
            meta.emit.to_string(),
            format!("{} {}", meta.tool, meta.tool_ver),
            row.basis.to_string(),
            row.device.to_string(),
            row.host.to_string(),
            row.gate.clone(),
            if row.available { "是" } else { "否" }.to_string(),
        ]);
    }
    format!("## 设备证据（本轮执行）\n\n{}", table.render())
}

/// 原语 × 后端对照表：一格一条供给，多条用 `<br>` 叠在格里。
fn ledger_prim_matrix(rows: &[BackendRow]) -> String {
    let mut head = vec![String::from("原语")];
    head.extend(rows.iter().map(|row| row.profile.meta.vendor.clone()));
    let mut table = Sheet::new(head);
    for prim in Prim::ALL {
        let mut cells = vec![format!("`{prim}`")];
        for row in rows {
            cells.push(
                row.cells
                    .iter()
                    .filter(|cell| cell.prim == *prim)
                    .map(ledger_cell_text)
                    .collect::<Vec<_>>()
                    .join("<br>"),
            );
        }
        table = table.row(cells);
    }
    format!("\n## 原语 × 后端\n\n{}", table.render())
}

/// 逐条证据说明：未声明的格子不进说明，实现名与理由只在这里展开。
fn ledger_evidence_notes(rows: &[BackendRow]) -> String {
    let mut out = String::new();
    out.push_str("\n## 证据说明\n\n");
    for row in rows {
        let vendor = row.profile.meta.vendor.as_str();
        for cell in &row.cells {
            if cell.status == Status::Unprovided {
                continue;
            }
            let _ = writeln!(
                out,
                "- `{}` {} · {} {} — {}",
                vendor, cell.prim, cell.impl_name, cell.grade, cell.note
            );
        }
    }
    out
}

/// 矩阵格：状态标签 + 档案等级；实现名与理由放在证据说明里，避免格子塞字。
fn ledger_cell_text(cell: &LedgerCell) -> String {
    format!("{} · {}", cell.status.label(), cell.grade)
}

/// 台账档案头：厂商、档案编号、工具链实测/登记一致性与结论。
pub(super) fn ledger_header(profile: &Profile, verified: &Verified, conclusion: &str) -> String {
    let mut text = format!(
        "# {} 能力台账（由 `cutile verify --emit-ledger` 生成）\n\n",
        profile.meta.vendor
    );
    let _ = writeln!(
        text,
        "- 档案：`{}` ({})",
        profile.meta.id, profile.meta.checked
    );
    // 台账要能回答"这条证据是谁编出来的"：实测值来自 `--version`，登记值是人写的出处。
    let actual = verified
        .toolchain
        .as_deref()
        .unwrap_or("未知（没有编译器，只能复用已落盘产物）");
    let agrees = if verified.toolchain_agrees {
        ""
    } else {
        "，与登记值不一致"
    };
    let _ = writeln!(
        text,
        "- 工具链：实测 `{actual}`，档案登记 `{}`{agrees}",
        verified.tool_ver
    );
    let _ = write!(text, "- 结论：{}", conclusion.trim());
    text
}

/// 台账原语表：逐条列出实现、等级与见证出处。
pub(super) fn ledger_primitives(profile: &Profile) -> String {
    let mut table = Sheet::new(["原语", "实现", "等级", "见证"]);
    for have in &profile.provides {
        let witness = have.witness.as_ref().map_or_else(
            || "—".to_string(),
            |witness| format!("{}({})", witness.file, witness.func),
        );
        table = table.row([
            format!("`{}`", have.prim),
            have.impl_name.clone(),
            have.grade.to_string(),
            witness,
        ]);
    }
    format!("\n\n{}", table.render())
}

/// 融合候选打分表：能打分的按加速比降序在前，没条件打分的带原因排在后面。
pub(super) fn fusions(rows: &[(Fusion, Verdict)]) -> String {
    let mut table = Sheet::new(["生产者", "消费者", "加速比", "省下", "定界", "判决", "说明"])
        .align([
            Align::Left,
            Align::Left,
            Align::Right,
            Align::Right,
            Align::Left,
            Align::Left,
            Align::Left,
        ]);
    for (fusion, verdict) in rows {
        let (speedup, saved, bound) = match verdict {
            Verdict::Scored(payoff) => (
                format!("{:.2}×", payoff.speedup()),
                format!("{} B", payoff.saved_bytes),
                payoff.bound(),
            ),
            _ => (String::from("—"), String::from("—"), "—"),
        };
        table = table.row([
            format!("`{}`", fusion.producer),
            format!("`{}`", fusion.consumer),
            speedup,
            saved,
            bound.to_string(),
            verdict.as_str().to_string(),
            verdict.reason().unwrap_or("—").to_string(),
        ]);
    }
    format!(
        "## 候选融合（roofline 加速比）\n\n{}\n\n\
         加速比是结构性上界：只算中间量的往返，不算掩盖、寄存器压力与指令缓存。\
         拼不出来的那条带几何原因，缺投影或缺峰值的那条带缺失项——它们都留在表里。\n",
        table.render()
    )
}
