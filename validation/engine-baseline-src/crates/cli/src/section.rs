//! 审计输出的**段落**：证据、布局、存储、产物、协商、性能与 L2 结论行。

use super::prose::{counted, fragment_note, params_note};
use cutile::cap::VariantCheck;
use shard::bench::{Bench, Peak, Performance, Verdict};
use shard::doctor::{Built, CacheState, CacheWhy, Doctor};
use shard::verify::{Negotiation, Verified, VerifyReport};
use std::fmt::Write as _;

/// `[证据]` 段：依据类别、设备执行 / 本机自检条数与语言门；不成立的见证单列一行。
pub(super) fn evidence_block(report: &Doctor) -> String {
    let mut out = String::from("\n[证据] 依据 ");
    out.push_str(&report.evidence.basis.to_string());
    let _ = writeln!(
        out,
        " | 设备执行 {} | 本机自检 {} | 语言门 {}",
        counted(report.evidence.device, &report.evidence.device_params),
        counted(report.evidence.host, &report.evidence.host_params),
        report.evidence.language
    );
    if let Some(error) = &report.evidence.failure {
        let _ = writeln!(out, "  见证不成立：{error}");
    }
    out
}

/// `[布局]` 段：每个槽位一条声明式。
pub(super) fn layouts_block(report: &Doctor) -> String {
    let mut out = String::from("\n[布局]\n");
    for row in &report.layouts {
        let _ = writeln!(out, "  {} = {}", row.slot, row.spec);
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
            let _ = writeln!(
                out,
                "  {}={} 容量 {} | 声明 {}",
                row.space, row.physical, capacity, declared
            );
        }
        out
    })
}

/// `[emit]` 段：片段口径、扩展名、缓存键与缓存目录。
pub(super) fn source_block(source: &Built) -> String {
    let mut out = String::from("\n[emit] ");
    let _ = writeln!(
        out,
        "{} | {} | cache key {}\n  {}",
        fragment_note(source.fragment),
        source.ext,
        source.key.hex(),
        source.key.describe()
    );
    if let Some(dir) = &source.cache_dir {
        let _ = writeln!(out, "  cache dir {}", dir.display());
    }
    out
}

/// 可行路径之后的整串产物段：plan、布局、存储、产物。集中在一起，编排入口只判"有没有路径"。
pub(super) fn post_path_blocks(report: &Doctor) -> String {
    let mut out = String::new();
    if let Some(plan) = &report.plan {
        let _ = writeln!(out, "\n[plan] {plan}");
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
        path.idx + 1,
        report.variants.len(),
        if path.is_head() {
            "（首档）"
        } else {
            "（降级）"
        }
    );
    for (idx, check) in report.variants.iter().enumerate() {
        out.push_str(&variant_line(check, path.idx, idx));
    }
    out
}

/// 单个变体的一行：选中标记 + 名称；有失败就逐条列需求与原因。
fn variant_line(check: &VariantCheck, selected: usize, idx: usize) -> String {
    let mark = if idx == selected { "✓ 选中" } else { "·" };
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
pub(super) fn negotiation_summary(report: &VerifyReport) -> String {
    let (hits, total) = report.head_hits();
    let mut out = format!(
        "\n协商首档 {hits}/{total}（cutile README §16：首选没被选中说明 `needs` 粒度写错了）\n"
    );
    for (device, negotiation) in report.devices.iter().zip(&report.negotiations) {
        let _ = writeln!(out, "  {device} = {}", describe_negotiation(negotiation));
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
pub(super) fn cache_why(report: &CacheWhy) -> String {
    let mut out = format!("\n[缓存] 本次 key {}\n", report.current);
    let Some(entry) = &report.entry else {
        out.push_str("  被问的键从未落盘：没构建过，或已被 `cache gc` 回收\n");
        return out;
    };
    let _ = writeln!(out, "  被问的键 {}（{} B）", entry.hex, entry.bytes);
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
    let _ = writeln!(out, "  与本次构建不同（{} 项）：", report.changes.len());
    for change in &report.changes {
        // 只在一侧出现的项也要打印：新版本加的字段与已删除的字段都是"为什么没命中"。
        let stored = change.stored.as_deref().unwrap_or("—");
        let current = change.current.as_deref().unwrap_or("—");
        let _ = writeln!(out, "    {}：缓存 {stored} | 本次 {current}", change.field);
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
        ::diagnostic::render::human_bytes(measured.working_set),
        ::diagnostic::render::human_bytes(measured.fast_bytes)
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
        .map_or_else(|| "-".to_string(), |path| path.display().to_string());
    format!("  缓存  {dir}\n")
}
