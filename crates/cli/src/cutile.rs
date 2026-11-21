//! 内核层审计的驱动方：解析 → 执行 → 排版 → 退出码（L7 入口，不定义新类型）。
//!
//! 三件事分别有各自的归属：语法与配置来自 `llm_cli::kernel` / `llm_cli::settings`，计算来自
//! `cutile::{bench, doctor, verify}`，人可读行文聚簇在 [`prose`] 模块。本文件只把它们
//! 接起来并收退出码，因此 `cutile` 不再持有第二套命令行文法、也不决定缓存目录在哪，命令模块也不再
//! 拼字符串。
//!
//! ```text
//! cargo run --bin cutile -- doctor --device cpu
//! cargo run --bin cutile -- doctor --device cpu --why 83a3ecdb31cd1225
//! cargo run --bin cutile -- verify --devices cpu,cuda --emit-ledger
//! cargo run --bin cutile -- bench --shape 4096x4096 --repeat 50
//! cargo run --bin cutile -- cache gc --limit 1073741824
//! ```

use cutile::bench::{self, Bench};
use cutile::builtin::Catalogue;
use cutile::doctor::{self, Doctor};
use cutile::ledger;
use cutile::verify::{self, Verified, VerifyEntry};
use diagnostic::{Code, Error};
use llm_cli::kernel::CutileCommand;
use llm_cli::settings::Settings;
use std::fmt::Write as _;
use std::process::ExitCode;
mod matrix;
mod prose;
mod section;

fn main() -> ExitCode {
    // 装配设备后端：Metal / CUDA 由 engine 实现，cutile 只认 Backend 缝；在跑任何 verify / bench /
    // 真机对拍之前注入，机制层按 emit 从进程级表取后端。
    device::backends::install();
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut settings = match Settings::load() {
        Ok(settings) => settings,
        // 配置层自己不读文件，失败只剩 `invalid`（键未登记、`--set` 缺 `=`、值为空），
        // 这里只补一帧"这次失败发生在配置上"，类别不被折掉，退出码因此仍是 2。
        Err(error) => return fail(error.with("配置")),
    };
    let rest = match settings.apply_arguments(&args) {
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

fn run(command: &CutileCommand<'_>, settings: &Settings) -> diagnostic::Result<String> {
    match command {
        CutileCommand::Doctor { op, device, why } => doctor(settings, op, device, why.as_deref()),
        CutileCommand::Verify {
            op,
            devices,
            ledger,
        } => verify(settings, op, devices, *ledger),
        CutileCommand::Bench {
            op,
            shape,
            device,
            repeat,
        } => bench(settings, op, shape.as_deref(), device.as_deref(), *repeat),
        CutileCommand::Ledger { out } => ledger(settings, out.as_deref()),
        CutileCommand::Cache { limit } => cache_gc(settings, *limit),
    }
}

/// `cache gc`：回收结果全是数据，这里只负责排版与“还差多少”的一句话。
///
/// 残留与 LRU 删除分两行：残留不是"空间不够"的产物，它本来就不会命中，报在同一行里
/// 会让人以为上限调大就能少删几条。
///
/// 上限来源：`--limit`（命令行层）最高，其次是配置键 `cache.limit_bytes`（策略层投影），
/// 两者都没有时由内核 [`cutile::cache::CACHE_LIMIT`] 兜底。
fn cache_gc(settings: &Settings, limit: Option<u64>) -> diagnostic::Result<String> {
    let dirs = settings.dirs();
    let limit = limit.or_else(|| settings.cache_limit_bytes());
    let report = cutile::cache::cache_gc(&dirs, limit)?;
    Ok(prose::cache_gc(&report, &dirs))
}

/// 跨档案台账：聚合现有见证证据，渲染成矩阵 markdown。
fn ledger(settings: &Settings, out: Option<&str>) -> diagnostic::Result<String> {
    let dirs = settings.dirs();
    let catalogue = Catalogue::load(&dirs)?;
    let rows = ledger::ledger(&catalogue, &dirs);
    let text = matrix::ledger(&rows);
    match out {
        Some(path) => {
            let path = std::path::Path::new(path);
            std::fs::write(path, &text).map_err(|error| {
                Error::new(Code::Io, format!("写 {} 失败：{error}", path.display()))
            })?;
            Ok(format!("ledger → {}\n{text}", path.display()))
        }
        None => Ok(text),
    }
}

/// `--why` 缓存键解释段；没给键或没有可行变体时返回空串。
fn doctor_why(
    dirs: &cutile::dirs::Dirs,
    report: &Doctor,
    why: Option<&str>,
) -> diagnostic::Result<String> {
    let Some(hex) = why else {
        return Ok(String::new());
    };
    let built = report.source.as_ref().ok_or_else(|| {
        Error::new(
            Code::Unsupported,
            format!("没有可行变体，无法解释缓存键 `{hex}`"),
        )
    })?;
    Ok(section::cache_why(&doctor::why(dirs, &built.key, hex)))
}

fn doctor(
    settings: &Settings,
    op: &str,
    device: &str,
    why: Option<&str>,
) -> diagnostic::Result<String> {
    let dirs = settings.dirs();
    let catalogue = Catalogue::load(&dirs)?;
    let report: Doctor = doctor::doctor(&catalogue, &dirs, op, device)?;

    let mut out = prose::doctor_head(&report).render();
    out.push('\n');
    out.push_str(&settings.describe());

    out.push_str(&section::evidence_block(&report));

    out.push_str("\n[协商]\n");
    out.push_str(&section::negotiation(&report));

    if report.path.is_none() {
        out.push_str("\n[emit] 无可行变体，未生成源文件\n");
        return Ok(out);
    }
    out.push_str(&section::post_path_blocks(&report));
    out.push_str(&doctor_why(&dirs, &report, why)?);
    Ok(out)
}

/// 单条校验结果一行；通过且要求台账时顺带落盘。
fn entry_line(
    catalogue: &Catalogue,
    dirs: &cutile::dirs::Dirs,
    entry: &VerifyEntry,
    ledger: bool,
) -> diagnostic::Result<String> {
    match entry {
        VerifyEntry::Passed(verified) => {
            let line = section::passed_line(verified);
            let mut out = line.clone();
            if ledger {
                out.push_str(&write_ledger(catalogue, verified, &line, dirs)?);
            }
            Ok(out)
        }
        VerifyEntry::Failed { label, error } => Ok(format!("  {label}: ✗ {error}\n")),
    }
}

/// 收尾：有未通过项按"能力不存在"报 unsupported；否则原样返回报告。
fn finish_verify(out: String, failures: &[&diagnostic::Error]) -> diagnostic::Result<String> {
    if failures.is_empty() {
        return Ok(out);
    }
    // 缺设备（或设备不成立）是“能力不存在”：类别 `unsupported`，退出码 3。
    Err(Error::new(
        Code::Unsupported,
        format!(
            "{out}\n缺设备即失败（cutile README §11.2）：{} 项未通过",
            failures.len()
        ),
    ))
}

fn verify(
    settings: &Settings,
    op: &str,
    devices: &[String],
    ledger: bool,
) -> diagnostic::Result<String> {
    let dirs = settings.dirs();
    let catalogue = Catalogue::load(&dirs)?;
    let report = verify::verify(&catalogue, &dirs, op, devices)?;
    let mut out = format!("verify {} @ [{}]\n", report.op, report.devices.join(","));

    for entry in &report.entries {
        out.push_str(&entry_line(&catalogue, &dirs, entry, ledger)?);
    }
    out.push_str(&section::negotiation_summary(&report));

    let failures = report.failures();
    finish_verify(out, &failures)
}

/// 能力台账：人可读的 markdown，属于呈现场景，所以留在前端。
fn write_ledger(
    catalogue: &Catalogue,
    verified: &Verified,
    conclusion: &str,
    dirs: &cutile::dirs::Dirs,
) -> diagnostic::Result<String> {
    let profile = catalogue.resolve(&verified.device)?;
    let dir = dirs.work().join("ledger");
    std::fs::create_dir_all(&dir)
        .map_err(|error| Error::new(Code::Io, format!("建 {} 失败：{error}", dir.display())))?;
    let path = dir.join(format!("{}.capability.md", profile.meta.vendor));
    let mut text = matrix::ledger_header(profile, verified, conclusion);
    text.push_str(&matrix::ledger_primitives(profile));
    std::fs::write(&path, text)
        .map_err(|error| Error::new(Code::Io, format!("写 {} 失败：{error}", path.display())))?;
    Ok(format!("    ledger → {}\n", path.display()))
}

fn bench(
    settings: &Settings,
    op: &str,
    shape: Option<&str>,
    device: Option<&str>,
    repeat: usize,
) -> diagnostic::Result<String> {
    let dirs = settings.dirs();
    let catalogue = Catalogue::load(&dirs)?;
    let measured: Bench = bench::bench(&catalogue, &dirs, op, shape, device, repeat)?;
    let (m, n) = measured.shape;

    // 设备身份进首行：多设备机器上同一份台账会被并排读，缺了它无法归属。
    let mut out = format!(
        "bench {} {m}x{n} repeat={} [{}]\n",
        measured.op, measured.repeat, measured.device
    );
    let _ = writeln!(out, "  plan  {}", measured.plan);
    let _ = writeln!(
        out,
        "  证据  {} 条见证（这次执行，协商用的就是它）",
        measured.proven
    );
    let _ = writeln!(
        out,
        "  对拍  max_rel_err {:.3e}（与标量参考）",
        measured.max_rel_err
    );
    let _ = writeln!(
        out,
        "  实测  min {}, p50 {}（达成率按 p50）→ {:.1} GB/s, {:.1} GFLOP/s",
        ::diagnostic::render::per_run(measured.per_run_ns),
        ::diagnostic::render::per_run(measured.p50_ns),
        measured.measured.gb_s,
        measured.measured.gflops
    );
    out.push_str(&section::working_set_line(&measured));
    out.push_str(&section::performance_note(
        measured.performance(),
        measured.peak.as_ref(),
    ));
    out.push_str(&section::artifact_line(&measured));
    out.push_str(&section::cache_dir_line(&measured));
    Ok(out)
}

/// 驱动方的单元测试：分派、退出码、排版。
#[cfg(test)]
#[path = "../tests/cutile.rs"]
mod tests;
