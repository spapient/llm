//! 内核层审计的驱动方：解析 → 执行 → 排版 → 退出码（L7 入口，不定义新类型）。
//!
//! 三件事分别有各自的归属：语法来自 [`cli`]（`cli::cutile`），计算来自
//! `cutile::report`，配置覆盖来自 `config`。本文件只把它们接起来并排版，
//! 因此 `cutile` 不再持有第二套命令行文法，`report` 也不再拼字符串。
//!
//! ```text
//! cargo run --bin cutile -- doctor --device cpu
//! cargo run --bin cutile -- doctor --device cpu --why 83a3ecdb31cd1225
//! cargo run --bin cutile -- verify --devices cpu,cuda --emit-ledger
//! cargo run --bin cutile -- bench --shape 4096x4096 --repeat 50
//! cargo run --bin cutile -- cache gc --limit 1073741824
//! ```

use cli::cutile::CutileCommand;
use config::{Cutile, Settings};
use cutile::builtin::Catalogue;
use cutile::report::{self, Bench, Doctor, Fragment, Peak, Performance, Verdict, Verified};
use diagnostic::{Code, Error};
use std::process::ExitCode;
use tool::Report;

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut settings = match Settings::<Cutile>::load() {
        Ok(settings) => settings,
        Err(error) => return fail(Error::new(Code::Invalid, format!("配置错误：{error}"))),
    };
    let rest = match take_overrides(&args, &mut settings) {
        Ok(rest) => rest,
        Err(error) => return fail(error),
    };
    let command = match CutileCommand::parse(&rest) {
        Ok(command) => command,
        Err(error) => return fail(error),
    };
    match run(&command, &settings) {
        Ok(text) => {
            print!("{text}");
            ExitCode::SUCCESS
        }
        Err(error) => fail(error),
    }
}

/// 失败：打印带类别的诊断，按 [`Code::exit_code`] 退出。
///
/// 类别 → 退出码只有一处定义（`diagnostic::Code::exit_code`）：`2` 输入/声明不合法、
/// `3` 缺能力（设备、原语、证据等级）、`4` 缺东西、`5` 读写失败、`70` 内部不变量被破坏。
/// 脚本因此能区分“这台设备就是没有这条能力”与“我把参数写错了”。
fn fail(error: Error) -> ExitCode {
    eprintln!("{error}");
    ExitCode::from(error.code().exit_code())
}

/// 把 `--set key=value` 摘出来写进设置，返回其余参数（配置覆盖属于配置层，不属于语法）。
fn take_overrides(
    args: &[String],
    settings: &mut Settings<Cutile>,
) -> diagnostic::Result<Vec<String>> {
    let mut rest = Vec::new();
    let mut index = 0;
    while index < args.len() {
        if args[index] == "--set" {
            let pair = args
                .get(index + 1)
                .ok_or_else(|| Error::new(Code::Invalid, "`--set` 缺少 `key=value`"))?;
            settings
                .apply_override(pair)
                .map_err(|error| Error::new(Code::Invalid, error.to_string()))?;
            index += 2;
            continue;
        }
        rest.push(args[index].clone());
        index += 1;
    }
    Ok(rest)
}

fn run(command: &CutileCommand, settings: &Settings<Cutile>) -> diagnostic::Result<String> {
    match command {
        CutileCommand::Doctor { op, device, why } => doctor(settings, op, device, why.as_deref()),
        CutileCommand::Verify {
            op,
            devices,
            ledger,
        } => verify(settings, op, devices, *ledger),
        CutileCommand::Bench { op, shape, repeat } => {
            bench(settings, op, shape.as_deref(), *repeat)
        }
        CutileCommand::Cache { limit } => cache_gc(settings, *limit),
    }
}

/// `cache gc`：回收结果全是数据，这里只负责排版与“还差多少”的一句话。
///
/// 残留与 LRU 删除分两行：残留不是"空间不够"的产物，它本来就不会命中，报在同一行里
/// 会让人以为上限调大就能少删几条。
fn cache_gc(settings: &Settings<Cutile>, limit: Option<u64>) -> diagnostic::Result<String> {
    let report = report::cache_gc(settings, limit)?;
    let mut out = Report::new(format!("cache gc | 上限 {}", human_bytes(report.limit)))
        .row("目录", settings.cache_dir().display().to_string())
        .row("回收前", human_bytes(report.before))
        .row("回收后", human_bytes(report.after))
        .row("按 LRU 删除", format!("{} 项", report.removed))
        .render();
    if report.residue.items > 0 {
        out.push_str(&format!(
            "  清理残留 {} 项（{}）：半截落盘，本来就不会命中\n",
            report.residue.items,
            human_bytes(report.residue.bytes)
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
            human_bytes(report.after - report.limit)
        ));
    }
    Ok(out)
}

fn doctor(
    settings: &Settings<Cutile>,
    op: &str,
    device: &str,
    why: Option<&str>,
) -> diagnostic::Result<String> {
    let catalogue = Catalogue::load(settings)?;
    let report: Doctor = report::doctor(&catalogue, settings, op, device)?;

    let mut head = Report::new(format!(
        "cutile schema v{} | op={} | skeleton={} | {}",
        cutile::SCHEMA_VER,
        report.op.name,
        report.op.skeleton,
        report.op.summary
    ))
    .row(
        "device",
        format!(
            "{}({}) emit={} tool={} 档案 {} 实测 {} 参数 {}",
            report.device.vendor,
            report.device.id,
            report.device.emit,
            report.device.tool,
            report.device.tool_ver,
            report
                .device
                .toolchain
                .as_deref()
                .unwrap_or("未知（没有编译器，只能复用已落盘产物）"),
            params_note(&report.device.flags)
        ),
    )
    .row(
        "spaces",
        format!(
            "GM={} FAST={} REG={}",
            report.device.spaces[0], report.device.spaces[1], report.device.spaces[2]
        ),
    );
    if !report.device.units.is_empty() {
        let units: Vec<String> = report
            .device
            .units
            .iter()
            .map(|(name, space)| format!("{name}({space})"))
            .collect();
        head = head.row("units", units.join(" "));
    }
    let mut out = head
        .row(
            "声明",
            format!(
                "needs={} variants={} tiles={} | 生产形状 {}x{}",
                report.op.needs,
                report.op.variants,
                report.op.tiles,
                report.shape.0,
                report.shape.1
            ),
        )
        .render();
    out.push('\n');
    out.push_str(&settings.describe());

    out.push_str("\n[证据] 依据 ");
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

    out.push_str("\n[协商]\n");
    out.push_str(&negotiation(&report));

    if report.path.is_none() {
        out.push_str("\n[emit] 无可行变体，未生成源文件\n");
        return Ok(out);
    }
    if let Some(plan) = &report.plan {
        out.push_str(&format!("\n[plan] {plan}\n"));
    }
    out.push_str("\n[布局]\n");
    for row in &report.layouts {
        out.push_str(&format!("  {} = {}\n", row.slot, row.spec));
    }
    if !report.storage.is_empty() {
        out.push_str("\n[存储]\n");
        for row in &report.storage {
            // 容量与用量并排：判定比较的就是这两个数，少一个就无法复核。
            let capacity = row
                .capacity
                .map_or_else(|| "不限".to_string(), |bytes| format!("{bytes} B"));
            let declared = if row.declared.is_empty() {
                "无".to_string()
            } else {
                let parts: Vec<String> = row
                    .declared
                    .iter()
                    .map(|usage| format!("{}<={} B", usage.prim, usage.bytes))
                    .collect();
                parts.join(" ")
            };
            out.push_str(&format!(
                "  {}={} 容量 {} | 声明 {}\n",
                row.space, row.physical, capacity, declared
            ));
        }
    }
    if let Some(source) = &report.source {
        out.push_str(&format!(
            "\n[emit] {} | {} | cache key {}\n  {}\n",
            fragment_note(source.fragment),
            source.ext,
            source.key.hex(),
            source.key.describe()
        ));
        if let Some(dir) = &source.cache_dir {
            out.push_str(&format!("  cache dir {}\n", dir.display()));
        }
    }
    if let Some(hex) = why {
        let built = report.source.as_ref().ok_or_else(|| {
            Error::new(
                Code::Unsupported,
                format!("没有可行变体，无法解释缓存键 `{hex}`"),
            )
        })?;
        out.push_str(&cache_why(&report::why(settings, &built.key, hex)));
    }
    Ok(out)
}

/// `[缓存]` 段：本次构建的键，以及被问的那个键与它的关系。
///
/// 关系类别由 `CacheWhy::state` 给（半截落盘必须排在"差异为空"之前判），这里只选句子。
fn cache_why(report: &report::CacheWhy) -> String {
    let mut out = format!("\n[缓存] 本次 key {}\n", report.current);
    let Some(entry) = &report.entry else {
        out.push_str("  被问的键从未落盘：没构建过，或已被 `cache gc` 回收\n");
        return out;
    };
    out.push_str(&format!("  被问的键 {}（{} B）\n", entry.hex, entry.bytes));
    match report.state() {
        report::CacheState::Match => {
            out.push_str("  与本次构建的键一致：命中的就是这份缓存\n");
            return out;
        }
        report::CacheState::Partial => {
            out.push_str("  只有半截落盘（缺 source）：写到一半就没了，重新构建即可\n");
            return out;
        }
        report::CacheState::Never | report::CacheState::Differs => {}
    }
    out.push_str(&format!(
        "  与本次构建不同（{} 项）：\n",
        report.changes.len()
    ));
    for change in &report.changes {
        // 只在一侧出现的项也要打印：新版本加的字段与已删除的字段都是“为什么没命中”。
        let stored = change.stored.as_deref().unwrap_or("—");
        let current = change.current.as_deref().unwrap_or("—");
        out.push_str(&format!(
            "    {}：缓存 {stored} | 本次 {current}\n",
            change.field
        ));
    }
    out
}

/// `[协商]` 段：成功时逐变体给标记，失败时原样给出诊断。
///
/// 判定（`VariantCheck`）与错误文本（`NoPath`）都在库里，这里只负责标记与缩进。
fn negotiation(report: &Doctor) -> String {
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
        let mark = if index == path.index {
            "✓ 选中"
        } else {
            "·"
        };
        if check.failures.is_empty() {
            out.push_str(&format!("  {mark} {}\n", check.name));
        } else {
            let reasons: Vec<String> = check
                .failures
                .iter()
                .map(|failure| format!("{}：{}", failure.need.prim, failure.reason()))
                .collect();
            out.push_str(&format!(
                "  {mark} {} ✗ {}\n",
                check.name,
                reasons.join("；")
            ));
        }
    }
    out
}

fn verify(
    settings: &Settings<Cutile>,
    op: &str,
    devices: &[String],
    ledger: bool,
) -> diagnostic::Result<String> {
    let catalogue = Catalogue::load(settings)?;
    let report = report::verify(&catalogue, settings, op, devices)?;
    let mut out = format!("verify {} @ [{}]\n", report.op, report.devices.join(","));

    for entry in &report.entries {
        match entry {
            report::VerifyEntry::Passed(verified) => {
                let line = passed_line(verified);
                out.push_str(&line);
                if ledger {
                    out.push_str(&write_ledger(&catalogue, verified, &line, settings)?);
                }
            }
            report::VerifyEntry::Failed { label, error } => {
                out.push_str(&format!("  {label}: ✗ {error}\n"));
            }
        }
    }
    out.push_str(&negotiation_summary(&report));

    let failures = report.failures();
    if failures.is_empty() {
        Ok(out)
    } else {
        // 缺设备（或设备不成立）是“能力不存在”：类别 `unsupported`，退出码 3。
        Err(Error::new(
            Code::Unsupported,
            format!(
                "{out}\n缺设备即失败（DESIGN §11.2）：{} 项未通过",
                failures.len()
            ),
        ))
    }
}

/// `[协商]` 汇总：DESIGN §16 的「协商 3/3 命中首档」。
///
/// 这段与上面的逐设备结果正交：执行需要真机，协商只看声明与档案——所以"本机没有该设备"的行
/// 照样在这里有答案。判定（首档 / 降级 / 无路径 / 无档案）在库里（`report::Negotiation`），
/// 这里只排版。
fn negotiation_summary(report: &report::VerifyReport) -> String {
    let (hits, total) = report.head_hits();
    let mut out =
        format!("\n协商首档 {hits}/{total}（DESIGN §16：首选没被选中说明 `needs` 粒度写错了）\n");
    for (device, negotiation) in report.devices.iter().zip(&report.negotiations) {
        out.push_str(&format!(
            "  {device} = {}\n",
            describe_negotiation(negotiation)
        ));
    }
    out
}

fn describe_negotiation(negotiation: &report::Negotiation) -> String {
    match negotiation {
        report::Negotiation::Head { variant, of } => format!("1/{of} 首选 {variant}"),
        report::Negotiation::Fallback { variant, step, of } => {
            format!("{step}/{of} 降级到 {variant}：首选的能力没被满足")
        }
        report::Negotiation::NoPath { of, unmet } if unmet.is_empty() => {
            format!("0/{of} 链上无可行变体")
        }
        report::Negotiation::NoPath { of, unmet } => {
            format!("0/{of} 链上无可行变体；首选缺 {}", unmet.join(" "))
        }
        report::Negotiation::Unknown { reason } => format!("无档案：{reason}"),
    }
}

/// §16 的片段行数判定：上限管成败，内部目标只是余量。
///
/// 判定在库里（`report::Fragment`），这里只选词；`doctor` 与 `verify` 用同一句，
/// 同一个数在两处不会被说成两种结论。
fn fragment_note(fragment: Fragment) -> String {
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

/// 编译参数的一行写法：空列表也要有话说，否则报告里会出现一对空括号。
///
/// 参数是能力的第三个维度（片段、工具链、参数）：同一份片段换一组参数就是另一次代码生成，
/// 所以结论行、证据行与产物行都把它写出来，而不是只留一个哈希。
fn params_note(flags: &str) -> String {
    if flags.is_empty() {
        "无".to_string()
    } else {
        flags.to_string()
    }
}

/// 一条"跑了几条、用什么参数跑的"：零条时不带参数，免得给一个没有内容的括号。
fn counted(count: usize, flags: &str) -> String {
    if count == 0 {
        "0 条".to_string()
    } else {
        format!("{count} 条（{}）", params_note(flags))
    }
}

/// §16 的性能判据：每次都给一行结论。
///
/// 判定在 `report::Performance`（达成率连同尺子一起给），这里只负责把每种结论翻译成一句话。
/// 口径无效或没声明峰值时也要打印：一条只在失败时出声的判据，读者会把“什么都没说”当成通过。
fn performance_note(performance: Performance, peak: Option<&Peak>) -> String {
    let ratio = performance.ratio.map_or_else(
        || "未测到".to_string(),
        |ratio| format!("{:.1}%", ratio * 100.0),
    );
    let ruler = peak.map_or_else(
        || "档案没有声明带宽峰值".to_string(),
        |peak| format!("峰值 {:.1} GB/s，口径：{}", peak.value / 1e9, peak.src),
    );
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

/// 一台设备的 L2 结论行。
///
/// 形状矩阵与见证条数必须打印：它们才是"这条 L2 是跑出来的"的证据，而不是档案里的自述。
fn passed_line(verified: &Verified) -> String {
    let shapes = verified
        .shapes
        .iter()
        .map(|(m, n)| format!("{m}x{n}"))
        .collect::<Vec<_>>()
        .join(",");
    // 产物行是"编译过几次"的证据：全部复用说明这一轮没调用过编译器；编译器名说明这份产物
    // 是谁编出来的——复用的若是别人编的那份，它与本机装了什么无关。
    let artifacts = verified.artifacts.len();
    let reused = verified
        .artifacts
        .iter()
        .filter(|artifact| artifact.reused)
        .count();
    let compiler = verified
        .artifacts
        .first()
        .map_or("未知", |artifact| artifact.key.tool.as_str());
    // 工具链身份进结论行：它是"这份 L2 是谁编出来的"的一部分，而不是档案里的自述。
    // 档案登记值与实测值不一致时明说——证据可以在别的机器上取得，但差异不能被藏起来。
    let toolchain = match (&verified.toolchain, verified.tool_ver.as_str()) {
        (Some(actual), declared) if verified.toolchain_agrees => {
            format!(" | 工具链 {actual}（档案登记 {declared}）")
        }
        (Some(actual), "") => format!(" | 工具链 {actual}"),
        (Some(actual), declared) => {
            format!(" | 工具链 {actual}（档案登记 {declared}，不一致）")
        }
        (None, declared) => format!(" | 工具链 未知（档案登记 {declared}，本轮复用已落盘产物）"),
    };
    // 自检与设备执行分开报：前者只说明“见证源码算得对”，不能与 L2 证据合成一个数。
    let checked = if verified.self_checked == 0 {
        String::new()
    } else {
        format!("，自检 {} 条", verified.self_checked)
    };
    format!(
        "  {}({}): ✓ L2 | variant {} | tile {} | 形状 {} | 见证 {} 条{checked} | max_rel_err {:.3e} | {} | 产物 {artifacts} 份（复用 {reused}，{compiler}，{}）{toolchain} | key {}\n",
        verified.vendor,
        verified.id,
        verified.variant,
        verified.tile,
        shapes,
        verified.witnesses,
        verified.max_rel_err,
        fragment_note(verified.fragment),
        params_note(&verified.flags),
        verified.key
    )
}

/// 能力台账：人可读的 markdown，属于呈现场景，所以留在前端。
fn write_ledger(
    catalogue: &Catalogue,
    verified: &Verified,
    conclusion: &str,
    settings: &Settings<Cutile>,
) -> diagnostic::Result<String> {
    let profile = catalogue.resolve(&verified.device)?;
    let dir = settings.work_dir().join("ledger");
    std::fs::create_dir_all(&dir)
        .map_err(|error| Error::new(Code::Io, format!("建 {} 失败：{error}", dir.display())))?;
    let path = dir.join(format!("{}.capability.md", profile.meta.vendor));
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
    text.push_str("\n\n| 原语 | 实现 | 等级 | 见证 |\n|---|---|---|---|\n");
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
    std::fs::write(&path, text)
        .map_err(|error| Error::new(Code::Io, format!("写 {} 失败：{error}", path.display())))?;
    Ok(format!("    ledger → {}\n", path.display()))
}

fn bench(
    settings: &Settings<Cutile>,
    op: &str,
    shape: Option<&str>,
    repeat: usize,
) -> diagnostic::Result<String> {
    let catalogue = Catalogue::load(settings)?;
    let measured: Bench = report::bench(&catalogue, settings, op, shape, repeat)?;
    let (m, n) = measured.shape;

    let mut out = format!("bench {} {m}x{n} repeat={}\n", measured.op, measured.repeat);
    out.push_str(&format!("  plan  {}\n", measured.plan));
    out.push_str(&format!(
        "  证据  {} 条见证（这次执行，协商用的就是它）\n",
        measured.proven
    ));
    out.push_str(&format!(
        "  实测  min {}, p50 {}（达成率按 p50）→ {:.1} GB/s, {:.1} GFLOP/s\n",
        per_run(measured.per_run_ns),
        per_run(measured.p50_ns),
        measured.measured.gb_s,
        measured.measured.gflops
    ));
    out.push_str(&format!(
        "  工作集 {} / FAST {}（{}）\n",
        human_bytes(measured.working_set),
        human_bytes(measured.fast_bytes),
        if measured.in_fast() {
            "在容量内，重复执行会命中缓存"
        } else {
            "超出容量"
        }
    ));
    out.push_str(&performance_note(
        measured.performance(),
        measured.peak.as_ref(),
    ));
    out.push_str(&format!(
        "  产物  {}（{} / {}：{}）\n",
        measured.artifact.key.hex(),
        measured.artifact.key.tool,
        params_note(&measured.flags),
        if measured.artifact.reused {
            format!("复用，未调用编译器 | {}", measured.artifact.path.display())
        } else {
            format!("新编译并落盘 | {}", measured.artifact.path.display())
        }
    ));
    out.push_str(&format!(
        "  缓存  {}\n",
        measured
            .cache_dir
            .as_ref()
            .map(|path| path.display().to_string())
            .unwrap_or_else(|| "-".to_string())
    ));
    Ok(out)
}

/// 单次耗时：亚微秒内核用 ns，否则用 ms（避免一串 `0.000`）。
fn per_run(ns: f64) -> String {
    if ns < 1e3 {
        format!("{ns:.1} ns/次")
    } else {
        format!("{:.3} ms/次", ns / 1e6)
    }
}

/// 人读的字节数（工作集与 FAST 容量对照用）。
fn human_bytes(bytes: u64) -> String {
    const UNITS: [&str; 4] = ["B", "KiB", "MiB", "GiB"];
    let mut value = bytes as f64;
    let mut unit = 0;
    while value >= 1024.0 && unit + 1 < UNITS.len() {
        value /= 1024.0;
        unit += 1;
    }
    if unit == 0 {
        format!("{bytes} B")
    } else {
        format!("{value:.1} {}", UNITS[unit])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn settings() -> Settings<Cutile> {
        let roots = config::Roots::new(std::env::temp_dir().join("cutile-driver-tests"));
        Settings::<Cutile>::defaults(&roots)
    }

    /// 性能判据每轮都要有一句结论：没声明峰值时说“未测量”，不能什么都不打印。
    #[test]
    fn the_performance_line_is_printed_even_when_nothing_was_measured() {
        let peak = Peak {
            value: 68.0e9,
            src: "2*8*2666e6".to_string(),
        };

        let reached = performance_note(Performance::new(Some(0.72), false), Some(&peak));
        assert!(reached.contains("72.0%"), "{reached}");
        assert!(reached.contains("达到 70% 门槛"), "{reached}");
        assert!(reached.contains("不是“相对原生库的 85%”"), "{reached}");

        let below = performance_note(Performance::new(Some(0.0069), false), Some(&peak));
        assert!(below.contains("0.7%"), "{below}");
        assert!(below.contains("低于 70% 门槛"), "{below}");
        assert!(below.contains("特化代码"), "{below}");

        let cached = performance_note(Performance::new(Some(0.19), true), Some(&peak));
        assert!(cached.contains("口径无效"), "{cached}");
        assert!(cached.contains("不参与 §16 判定"), "{cached}");

        // 没有峰值：判据没有尺子，仍要说清“未测量”，而不是静默通过。
        let unknown = performance_note(Performance::new(None, false), None);
        assert!(unknown.contains("未测量"), "{unknown}");
        assert!(unknown.contains("不参与 Go/No-Go"), "{unknown}");
        assert!(!unknown.contains("门槛"), "{unknown}");
    }

    #[test]
    fn doctor_prints_every_stage() {
        let text = doctor(&settings(), "gemv_i4", "cpu", None).expect("doctor 应成功");
        assert!(text.contains("path: pipe2"), "{text}");
        assert!(text.contains("[布局]"), "{text}");
        assert!(text.contains("[存储]"), "{text}");
        assert!(text.contains("[emit]"), "{text}");
        assert!(text.contains("cache key"), "{text}");
        assert!(text.contains("[配置]"), "{text}");
    }

    #[test]
    fn doctor_prints_the_capacity_next_to_the_declared_traffic() {
        // 打印的必须是档案容量（判定的分母）与声明用量（分子的来源），不是中转缓冲的字节数。
        let text = doctor(&settings(), "gemv_i4", "cpu", None).expect("doctor 应成功");
        assert!(text.contains("容量 524288 B"), "{text}");
        assert!(text.contains("声明 load<=4096 B"), "{text}");
        assert!(text.contains("GM=main 容量 不限"), "{text}");
    }

    #[test]
    fn doctor_explains_a_rejected_device_without_failing() {
        let text = doctor(&settings(), "gemv_i4", "nvidia", None).expect("仍应打印诊断");
        assert!(text.contains("没有可行变体"), "{text}");
        assert!(text.contains("L2"), "{text}");
    }

    #[test]
    fn doctor_explains_the_key_it_just_wrote() {
        // `--why` 要回答的是“这份缓存为什么没命中”：同一份键只能说一致，不能编差异。
        let settings = settings();
        let first = doctor(&settings, "gemv_i4", "cpu", None).expect("doctor 应成功");
        let hex = first
            .split("cache key ")
            .nth(1)
            .and_then(|rest| rest.split_whitespace().next())
            .expect("打印了键")
            .to_string();
        let text = doctor(&settings, "gemv_i4", "cpu", Some(&hex)).expect("doctor 应成功");
        assert!(text.contains("[缓存]"), "{text}");
        assert!(text.contains("与本次构建的键一致"), "{text}");
    }

    #[test]
    fn doctor_explains_a_key_that_never_landed() {
        let text =
            doctor(&settings(), "gemv_i4", "cpu", Some("0000000000000000")).expect("doctor 应成功");
        assert!(text.contains("[缓存]"), "{text}");
        assert!(text.contains("从未落盘"), "{text}");
    }

    #[test]
    fn doctor_reports_a_half_written_key_as_half_written() {
        // `doctor` 会先把本次的键补齐，所以“半截落盘”只能对另一个键观察到——
        // 那正是真实场景：另一个进程崩在 `put` 中间，留下 `meta.txt` 与 `source.tmp`。
        let settings = settings();
        let first = doctor(&settings, "gemv_i4", "cpu", None).expect("doctor 应成功");
        let hex = first
            .split("cache key ")
            .nth(1)
            .and_then(|rest| rest.split_whitespace().next())
            .expect("打印了键")
            .to_string();
        let source = settings.cache_dir().join(&hex[..2]).join(&hex);
        let half = settings.cache_dir().join("00").join("0000000000000001");
        std::fs::create_dir_all(&half).expect("建目录");
        std::fs::copy(source.join("meta.txt"), half.join("meta.txt")).expect("复制键快照");

        let text = doctor(&settings, "gemv_i4", "cpu", Some("0000000000000001")).expect("doctor");
        assert!(text.contains("只有半截落盘"), "{text}");
        assert!(
            !text.contains("与本次构建的键一致"),
            "没有 source 不能算命中：{text}"
        );
        let _ = std::fs::remove_dir_all(&half);
    }

    #[test]
    fn verify_reports_a_missing_device_as_a_failure() {
        let devices = vec!["nvidia".to_string()];
        let error = verify(&settings(), "gemv_i4", &devices, false).unwrap_err();
        assert_eq!(error.code(), Code::Unsupported);
        assert!(error.to_string().contains("缺设备即失败"), "{error}");
    }

    #[test]
    fn the_ledger_lists_every_declared_primitive() {
        let catalogue = Catalogue::load(&settings()).expect("内置数据");
        let profile = catalogue.resolve("cpu").expect("cpu 档案");
        let verified = Verified {
            device: "cpu".to_string(),
            vendor: profile.meta.vendor.to_string(),
            id: profile.meta.id.to_string(),
            variant: "pipe2".to_string(),
            tile: "GMAX=32".to_string(),
            shapes: vec![(32, 1), (4096, 4096)],
            witnesses: 9,
            self_checked: 0,
            max_rel_err: 0.0,
            fragment: Fragment::new(53),
            tool_ver: profile.meta.tool_ver.clone(),
            toolchain: Some("cc@21.0.0".to_string()),
            toolchain_agrees: true,
            flags: "-O3".to_string(),
            key: "0".to_string(),
            artifacts: Vec::new(),
        };
        let note = write_ledger(&catalogue, &verified, &passed_line(&verified), &settings())
            .expect("应写出台账");
        let path = note.trim().rsplit_once(' ').expect("路径").1;
        let text = std::fs::read_to_string(path).expect("台账可读");
        assert!(text.contains("| 原语 | 实现 | 等级 | 见证 |"), "{text}");
        assert!(text.contains("pipe2"), "{text}");
    }

    #[test]
    fn a_sub_microsecond_kernel_is_reported_in_nanoseconds() {
        assert_eq!(per_run(1.9), "1.9 ns/次");
        assert_eq!(per_run(13_300_000.0), "13.300 ms/次");
    }

    #[test]
    fn byte_sizes_are_printable_next_to_the_fast_capacity() {
        assert_eq!(human_bytes(148), "148 B");
        assert_eq!(human_bytes(10_485_760), "10.0 MiB");
        assert_eq!(human_bytes(524_288), "512.0 KiB");
    }
}
