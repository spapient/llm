//! cutile 命令的人可读排版聚簇（bin 内）：判定与数据都在 `cutile` 库，本模块只选词、缩进与分行。
//!
//! 这里不定义新类型，也不持有第二份判定规则；需要拍板的结论（首档 / 降级 / 口径无效 / 未达上限）
//! 一律读库给的枚举、计数与标志，本模块只把它们翻成一句话。台账（`ledger`）矩阵排版也在本模块。

use cutile::atom::Prim;
use cutile::bench::{Bench, Peak, Performance, Verdict};
use cutile::cache::Gc;
use cutile::cap::VariantCheck;
use cutile::dirs::Dirs;
use cutile::doctor::{Built, CacheState, CacheWhy, Doctor};
use cutile::emit::Fragment;
use cutile::ledger::{BackendRow, LedgerCell, Status};
use cutile::prof::Profile;
use cutile::verify::{Negotiation, Verified, VerifyReport};
use tool::Report;

/// 编译参数的一行写法：空列表也要有话说，否则报告里会出现一对空括号。
///
/// 参数是能力的第三个维度（片段、工具链、参数）：同一份片段换一组参数就是另一次代码生成，
/// 所以结论行、证据行与产物行都把它写出来，而不是只留一个哈希。
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
///
/// 哪条说明出现只看库给的计数（残留项、在用项、回收后与上限的差），本模块不重新判定
/// "该不该回收"——阈值的语义在 `cutile::cache`。
pub(super) fn cache_gc(report: &Gc, dirs: &Dirs) -> String {
    let mut out = Report::new(format!(
        "cache gc | 上限 {}",
        tool::human_bytes(report.limit)
    ))
    .row("目录", dirs.cache().display().to_string())
    .row("回收前", tool::human_bytes(report.before))
    .row("回收后", tool::human_bytes(report.after))
    .row("按 LRU 删除", format!("{} 项", report.removed))
    .render();
    if report.residue.items > 0 {
        out.push_str(&format!(
            "  清理残留 {} 项（{}）：半截落盘，本来就不会命中\n",
            report.residue.items,
            tool::human_bytes(report.residue.bytes)
        ));
    }
    if report.skipped > 0 {
        out.push_str(&format!(
            "  {} 项仍在使用，已跳过：锁释放后再跑一次\n",
            report.skipped
        ));
    }
    if report.after > report.limit {
        out.push_str(&format!(
            "  仍高于上限：还需再删 {}\n",
            tool::human_bytes(report.after - report.limit)
        ));
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
pub(super) fn doctor_head(report: &Doctor) -> Report {
    let mut head = Report::new(format!(
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

/// `[证据]` 段：依据类别、设备执行 / 本机自检条数与语言门；不成立的见证单列一行。
pub(super) fn evidence_block(report: &Doctor) -> String {
    let mut out = String::from("\n[证据] 依据 ");
    out.push_str(&report.evidence.basis.to_string());
    out.push_str(&format!(
        " | 设备执行 {} | 本机自检 {} | 语言门 {}\n",
        counted(report.evidence.device, &report.evidence.device_params),
        counted(report.evidence.host, &report.evidence.host_params),
        report.evidence.language
    ));
    if let Some(error) = &report.evidence.failure {
        out.push_str(&format!("  见证不成立：{error}\n"));
    }
    out
}

/// `[布局]` 段：每个槽位一条声明式。
pub(super) fn layouts_block(report: &Doctor) -> String {
    let mut out = String::from("\n[布局]\n");
    for row in &report.layouts {
        out.push_str(&format!("  {} = {}\n", row.slot, row.spec));
    }
    out
}

/// `[存储]` 段：容量与用量并排；没有声明用量的层不报。
pub(super) fn storage_block(report: &Doctor) -> Option<String> {
    (!report.storage.is_empty()).then(|| {
        let mut out = String::from("\n[存储]\n");
        for row in &report.storage {
            // 容量与用量并排：判定比较的就是这两个数，少一个就无法复核。
            let capacity = row
                .capacity
                .map_or_else(|| "不限".to_string(), |bytes| format!("{bytes} B"));
            let declared = if row.declared.is_empty() {
                "无".to_string()
            } else {
                row.declared
                    .iter()
                    .map(|usage| format!("{}<={} B", usage.prim, usage.bytes))
                    .collect::<Vec<_>>()
                    .join(" ")
            };
            out.push_str(&format!(
                "  {}={} 容量 {} | 声明 {}\n",
                row.space, row.physical, capacity, declared
            ));
        }
        out
    })
}

/// `[emit]` 段：片段口径、扩展名、缓存键与缓存目录。
pub(super) fn source_block(source: &Built) -> String {
    let mut out = String::from("\n[emit] ");
    out.push_str(&format!(
        "{} | {} | cache key {}\n  {}\n",
        fragment_note(source.fragment),
        source.ext,
        source.key.hex(),
        source.key.describe()
    ));
    if let Some(dir) = &source.cache_dir {
        out.push_str(&format!("  cache dir {}\n", dir.display()));
    }
    out
}

/// 可行路径之后的整串产物段：plan、布局、存储、产物。集中在一起，编排入口只判"有没有路径"。
pub(super) fn post_path_blocks(report: &Doctor) -> String {
    let mut out = String::new();
    if let Some(plan) = &report.plan {
        out.push_str(&format!("\n[plan] {plan}\n"));
    }
    out.push_str(&layouts_block(report));
    if let Some(storage) = storage_block(report) {
        out.push_str(&storage);
    }
    if let Some(source) = &report.source {
        out.push_str(&source_block(source));
    }
    out
}

/// `[协商]` 段：成功时逐变体给标记，失败时原样给出诊断。
///
/// 判定（`VariantCheck`）与错误文本（`NoPath`）都在库里，这里只负责标记与缩进。
pub(super) fn negotiation(report: &Doctor) -> String {
    if let Some(no_path) = &report.no_path {
        return no_path.to_string();
    }
    let Some(path) = &report.path else {
        return String::new();
    };
    let mut out = format!(
        "path: {} (schedule {}) mma_share {:.2} | 第 {}/{} 档{}\n",
        path.variant,
        path.schedule,
        path.mma_share,
        path.index + 1,
        report.variants.len(),
        if path.is_head() {
            "（首档）"
        } else {
            "（降级）"
        }
    );
    for (index, check) in report.variants.iter().enumerate() {
        out.push_str(&variant_line(check, path.index, index));
    }
    out
}

/// 单个变体的一行：选中标记 + 名称；有失败就逐条列需求与原因。
fn variant_line(check: &VariantCheck, selected: usize, index: usize) -> String {
    let mark = if index == selected {
        "✓ 选中"
    } else {
        "·"
    };
    if check.failures.is_empty() {
        format!("  {mark} {}\n", check.name)
    } else {
        let reasons = check
            .failures
            .iter()
            .map(|failure| format!("{}：{}", failure.need.prim, failure.reason()))
            .collect::<Vec<_>>()
            .join("；");
        format!("  {mark} {} ✗ {reasons}\n", check.name)
    }
}

/// `[协商]` 汇总：cutile README §16 的「协商 3/3 命中首档」。
///
/// 这段与逐设备结果正交：执行需要真机，协商只看声明与档案——所以"本机没有该设备"的行
/// 照样在这里有答案。判定（首档 / 降级 / 无路径 / 无档案）在库里（`Negotiation`），
/// 这里只排版。
pub(super) fn negotiation_summary(report: &VerifyReport) -> String {
    let (hits, total) = report.head_hits();
    let mut out = format!(
        "\n协商首档 {hits}/{total}（cutile README §16：首选没被选中说明 `needs` 粒度写错了）\n"
    );
    for (device, negotiation) in report.devices.iter().zip(&report.negotiations) {
        out.push_str(&format!(
            "  {device} = {}\n",
            describe_negotiation(negotiation)
        ));
    }
    out
}

fn describe_negotiation(negotiation: &Negotiation) -> String {
    match negotiation {
        Negotiation::Head { variant, of } => format!("1/{of} 首选 {variant}"),
        Negotiation::Fallback { variant, step, of } => {
            format!("{step}/{of} 降级到 {variant}：首选的能力没被满足")
        }
        Negotiation::NoPath { of, unmet } => {
            let missing = if unmet.is_empty() {
                String::new()
            } else {
                format!("；首选缺 {}", unmet.join(" "))
            };
            format!("0/{of} 链上无可行变体{missing}")
        }
        Negotiation::Unknown { reason } => format!("无档案：{reason}"),
    }
}

/// §16 的性能判据：每次都给一行结论。
///
/// 判定在 `Performance`（达成率连同尺子一起给），这里只负责把每种结论翻译成一句话。
/// 口径无效或没声明峰值时也要打印：一条只在失败时出声的判据，读者会把"什么都没说"当成通过。
pub(super) fn performance_note(performance: Performance, peak: Option<&Peak>) -> String {
    let ratio = ratio_text(&performance);
    let ruler = ruler_text(peak);
    match performance.verdict {
        Verdict::OverPeak => format!(
            "  ⚠ 口径无效：达成率 {ratio}（{ruler}）超过 100%，数据在缓存中；\
             这个数不参与 §16 判定\n"
        ),
        Verdict::CacheResident => format!(
            "  ⚠ 口径无效：达成率 {ratio}（{ruler}）——工作集装得进 FAST 容量，\
             测到的是缓存带宽；这个数不参与 §16 判定\n"
        ),
        Verdict::Reached => format!(
            "  带宽达成率 {ratio}（{ruler}）→ 达到 70% 门槛\
             （尺子是 roofline：实测 ÷ 档案峰值，不是“相对原生库的 85%”）\n"
        ),
        Verdict::Below => format!(
            "  带宽达成率 {ratio}（{ruler}）→ 低于 70% 门槛：\
             按 §16 这台设备承认需要特化代码（手写 + 片段复用）\n"
        ),
        Verdict::Unknown => format!("  ⚠ 未测量：{ruler}；§16 的性能判据不参与 Go/No-Go\n"),
    }
}

fn ratio_text(performance: &Performance) -> String {
    performance.ratio.map_or_else(
        || "未测到".to_string(),
        |ratio| format!("{:.1}%", ratio * 100.0),
    )
}

fn ruler_text(peak: Option<&Peak>) -> String {
    peak.map_or_else(
        || "档案没有声明带宽峰值".to_string(),
        |peak| format!("峰值 {:.1} GB/s，口径：{}", peak.value / 1e9, peak.src),
    )
}

/// 一台设备的 L2 结论行。
///
/// 形状矩阵与见证条数必须打印：它们才是"这条 L2 是跑出来的"的证据，而不是档案里的自述。
pub(super) fn passed_line(verified: &Verified) -> String {
    let shapes = shapes_text(verified);
    let artifacts = artifacts_text(verified);
    let toolchain = toolchain_line(verified);
    let checked = checked_text(verified);
    format!(
        "  {}({}): ✓ L2 | 通道 {} | variant {} | tile {} | 形状 {} | 见证 {} 条{checked} | max_rel_err {:.3e} | {} | {artifacts}{toolchain} | key {}\n",
        verified.vendor,
        verified.id,
        verified.channel.as_str(),
        verified.variant,
        verified.tile,
        shapes,
        verified.witnesses,
        verified.max_rel_err,
        fragment_note(verified.fragment),
        verified.key
    )
}

fn shapes_text(verified: &Verified) -> String {
    verified
        .shapes
        .iter()
        .map(|(m, n)| format!("{m}x{n}"))
        .collect::<Vec<_>>()
        .join(",")
}

/// 产物行是"编译过几次"的证据：全部复用说明这一轮没调用过编译器；编译器名说明这份产物
/// 是谁编出来的——复用的若是别人编的那份，它与本机装了什么无关。
fn artifacts_text(verified: &Verified) -> String {
    let reused = verified
        .artifacts
        .iter()
        .filter(|artifact| artifact.reused)
        .count();
    let compiler = verified
        .artifacts
        .first()
        .map_or("未知", |artifact| artifact.key.tool.as_str());
    format!(
        "产物 {} 份（复用 {reused}，{compiler}，{}）",
        verified.artifacts.len(),
        params_note(&verified.flags)
    )
}

/// 工具链身份进结论行：它是"这份 L2 是谁编出来的"的一部分，而不是档案里的自述。
/// 档案登记值与实测值不一致时明说——证据可以在别的机器上取得，但差异不能被藏起来。
fn toolchain_line(verified: &Verified) -> String {
    match (&verified.toolchain, verified.tool_ver.as_str()) {
        (Some(actual), declared) if verified.toolchain_agrees => {
            format!(" | 工具链 {actual}（档案登记 {declared}）")
        }
        (Some(actual), "") => format!(" | 工具链 {actual}"),
        (Some(actual), declared) => {
            format!(" | 工具链 {actual}（档案登记 {declared}，不一致）")
        }
        (None, declared) => {
            format!(" | 工具链 未知（档案登记 {declared}，本轮复用已落盘产物）")
        }
    }
}

/// 自检与设备执行分开报：前者只说明"见证源码算得对"，不能与 L2 证据合成一个数。
fn checked_text(verified: &Verified) -> String {
    if verified.self_checked == 0 {
        String::new()
    } else {
        format!("，自检 {} 条", verified.self_checked)
    }
}

/// `[缓存]` 段：本次构建的键，以及被问的那个键与它的关系。
///
/// 关系类别由 `CacheWhy::state` 给（半截落盘必须排在"差异为空"之前判），这里只选句子。
pub(super) fn cache_why(report: &CacheWhy) -> String {
    let mut out = format!("\n[缓存] 本次 key {}\n", report.current);
    let Some(entry) = &report.entry else {
        out.push_str("  被问的键从未落盘：没构建过，或已被 `cache gc` 回收\n");
        return out;
    };
    out.push_str(&format!("  被问的键 {}（{} B）\n", entry.hex, entry.bytes));
    match report.state() {
        CacheState::Match => {
            out.push_str("  与本次构建的键一致：命中的就是这份缓存\n");
            return out;
        }
        CacheState::Partial => {
            out.push_str("  只有半截落盘（缺 source）：写到一半就没了，重新构建即可\n");
            return out;
        }
        CacheState::Never | CacheState::Differs => {}
    }
    out.push_str(&format!(
        "  与本次构建不同（{} 项）：\n",
        report.changes.len()
    ));
    for change in &report.changes {
        // 只在一侧出现的项也要打印：新版本加的字段与已删除的字段都是"为什么没命中"。
        let stored = change.stored.as_deref().unwrap_or("—");
        let current = change.current.as_deref().unwrap_or("—");
        out.push_str(&format!(
            "    {}：缓存 {stored} | 本次 {current}\n",
            change.field
        ));
    }
    out
}

/// 工作集行：FAST 容量对照，决定重复执行会不会命中缓存。
pub(super) fn working_set_line(measured: &Bench) -> String {
    let verdict = if measured.in_fast() {
        "在容量内，重复执行会命中缓存"
    } else {
        "超出容量"
    };
    format!(
        "  工作集 {} / FAST {}（{verdict}）\n",
        tool::human_bytes(measured.working_set),
        tool::human_bytes(measured.fast_bytes)
    )
}

/// 产物行：复用还是新编译，以及落盘路径。
pub(super) fn artifact_line(measured: &Bench) -> String {
    match &measured.artifact {
        Some(artifact) => {
            let origin = if artifact.reused {
                format!("复用，未调用编译器 | {}", artifact.path.display())
            } else {
                format!("新编译并落盘 | {}", artifact.path.display())
            };
            format!(
                "  产物  {}（{} / {}：{origin}）\n",
                artifact.key.hex(),
                artifact.key.tool,
                params_note(&measured.flags)
            )
        }
        None => format!(
            "  产物  生产态设备源，临时编译、不缓存（工具链 {} / 参数 {}）\n",
            measured.toolchain.as_deref().unwrap_or("未知"),
            params_note(&measured.flags)
        ),
    }
}

/// 缓存目录行：未落盘时给一个占位符，不留空字段。
pub(super) fn cache_dir_line(measured: &Bench) -> String {
    let dir = measured
        .cache_dir
        .as_ref()
        .map(|path| path.display().to_string())
        .unwrap_or_else(|| "-".to_string());
    format!("  缓存  {dir}\n")
}

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
fn ledger_prim_matrix(rows: &[BackendRow]) -> String {
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
                .map(ledger_cell_text)
                .collect::<Vec<_>>()
                .join("<br>");
            out.push_str(&format!("{cell} | "));
        }
        out.push('\n');
    }
    out
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
            out.push_str(&format!(
                "- `{}` {} · {} {} — {}\n",
                vendor, cell.prim, cell.impl_name, cell.grade, cell.note
            ));
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
    text.push_str(&format!(
        "- 档案：`{}` ({})\n",
        profile.meta.id, profile.meta.checked
    ));
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
    text.push_str(&format!(
        "- 工具链：实测 `{actual}`，档案登记 `{}`{agrees}\n",
        verified.tool_ver
    ));
    text.push_str(&format!("- 结论：{}", conclusion.trim()));
    text
}

/// 台账原语表：逐条列出实现、等级与见证出处。
pub(super) fn ledger_primitives(profile: &Profile) -> String {
    let mut text = String::from("\n\n| 原语 | 实现 | 等级 | 见证 |\n|---|---|---|---|\n");
    for have in &profile.provides {
        let witness = have
            .witness
            .as_ref()
            .map(|witness| format!("{}({})", witness.file, witness.func))
            .unwrap_or_else(|| "—".to_string());
        text.push_str(&format!(
            "| `{}` | {} | {} | {} |\n",
            have.prim, have.impl_name, have.grade, witness
        ));
    }
    text
}
