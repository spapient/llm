//! 行级排版原件：参数写法、计数措辞、片段判定、档案头与 `cache gc` 报告。

use cutile::cache::Gc;
use cutile::dirs::Dirs;
use cutile::emit::Fragment;
use diagnostic::render::Panel;
use llm_optim::doctor::Doctor;
use std::fmt::Write as _;

/// 编译参数的一行写法：空列表也要有话说，否则报告里会出现一对空括号。
pub(super) fn params_note(flags: &str) -> String {
    if flags.is_empty() {
        "无".to_string()
    } else {
        flags.to_string()
    }
}

/// 一条"跑了几条、用什么参数跑的"：零条时不带参数，免得给一个没有内容的括号。
pub(super) fn counted(count: usize, flags: &str) -> String {
    if count == 0 {
        "0 条".to_string()
    } else {
        format!("{count} 条（{}）", params_note(flags))
    }
}

/// `cache gc` 的整份报告：四行账 + 三条按需说明。
pub(super) fn cache_gc(report: &Gc, dirs: &Dirs) -> String {
    let mut out = Panel::new(format!(
        "cache gc | 上限 {}",
        ::diagnostic::render::human_bytes(report.limit)
    ))
    .row("目录", dirs.cache().display().to_string())
    .row("回收前", ::diagnostic::render::human_bytes(report.before))
    .row("回收后", ::diagnostic::render::human_bytes(report.after))
    .row("按 LRU 删除", format!("{} 项", report.removed))
    .render();
    if report.residue.items > 0 {
        let _ = writeln!(
            out,
            "  清理残留 {} 项（{}）：半截落盘，本来就不会命中",
            report.residue.items,
            ::diagnostic::render::human_bytes(report.residue.bytes)
        );
    }
    if report.skipped > 0 {
        let _ = writeln!(
            out,
            "  {} 项仍在使用，已跳过：锁释放后再跑一次",
            report.skipped
        );
    }
    if report.after > report.limit {
        let _ = writeln!(
            out,
            "  仍高于上限：还需再删 {}",
            ::diagnostic::render::human_bytes(report.after - report.limit)
        );
    }
    out
}

/// §16 的片段行数判定：上限管成败，内部目标只是余量。
/// 判定在库里（`cutile::emit::Fragment`），这里只选词；`doctor` 与 `verify` 用同一句，
/// 同一个数在两处不会被说成两种结论。
pub(super) fn fragment_note(fragment: Fragment) -> String {
    let limit = if fragment.within_limit() {
        "达标"
    } else {
        "超出上限：原语粒度错，先合并原语"
    };
    let goal = if fragment.within_goal() {
        "达标"
    } else {
        "未达（余量不足）"
    };
    format!(
        "片段 {} 行（上限 {}：{limit}；内部目标 {}：{goal}）",
        fragment.lines,
        fragment.limit(),
        fragment.goal()
    )
}

/// 一台设备的档案层头部行：厂商 / 语言 / 工具 / 工具链与参数。
pub(super) fn device_value(report: &Doctor) -> String {
    let toolchain = report
        .device
        .toolchain
        .as_deref()
        .unwrap_or("未知（没有编译器，只能复用已落盘产物）");
    format!(
        "{}({}) emit={} tools={} 档案 {} 实测 {} 参数 {}",
        report.device.vendor,
        report.device.id,
        report.device.emit,
        report.device.tool,
        report.device.tool_ver,
        toolchain,
        params_note(&report.device.flags)
    )
}

/// 执行单元行；空单元时不报这一行。
pub(super) fn units_value(report: &Doctor) -> Option<String> {
    (!report.device.units.is_empty()).then(|| {
        report
            .device
            .units
            .iter()
            .map(|(name, space)| format!("{name}({space})"))
            .collect::<Vec<_>>()
            .join(" ")
    })
}

/// 声明层头部行：需求 / 变体 / tile 数与生产形状。
pub(super) fn declaration_value(report: &Doctor) -> String {
    format!(
        "needs={} variants={} tiles={} | 生产形状 {}x{}",
        report.op.needs, report.op.variants, report.op.tiles, report.shape.0, report.shape.1
    )
}

/// 报告头部：schema、设备、容量档、units 可选行与声明行。
pub(super) fn doctor_head(report: &Doctor) -> Panel {
    let mut head = Panel::new(format!(
        "cutile schema v{} | op={} | skeleton={} | {}",
        cutile::SCHEMA_VER,
        report.op.name,
        report.op.skeleton,
        report.op.summary
    ))
    .row("device", device_value(report))
    .row(
        "spaces",
        format!(
            "GM={} FAST={} REG={}",
            report.device.spaces[0], report.device.spaces[1], report.device.spaces[2]
        ),
    );
    if let Some(units) = units_value(report) {
        head = head.row("units", units);
    }
    head.row("声明", declaration_value(report))
}
