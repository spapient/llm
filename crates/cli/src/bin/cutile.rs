//! 内核层审计的驱动方：解析 → 执行 → 排版 → 退出码（L7 入口，不定义新类型）。
//!
//! 三件事分别有各自的归属：语法与配置来自 `llm_cli::cutile_command` / `llm_cli::settings`，计算来自
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
use llm_cli::cutile_command::CutileCommand;
use llm_cli::settings::Settings;
use std::process::ExitCode;
#[path = "cutile/prose.rs"]
mod prose;

fn main() -> ExitCode {
    // 装配设备后端：Metal / CUDA 由 engine 实现，cutile 只认 Backend 缝；在跑任何 verify / bench /
    // 真机对拍之前注入，机制层按 emit 从进程级表取后端。
    engine::backends::install();
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

fn run(command: &CutileCommand, settings: &Settings) -> diagnostic::Result<String> {
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
    let text = prose::ledger(&rows);
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
    Ok(prose::cache_why(&doctor::why(dirs, &built.key, hex)))
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

    out.push_str(&prose::evidence_block(&report));

    out.push_str("\n[协商]\n");
    out.push_str(&prose::negotiation(&report));

    if report.path.is_none() {
        out.push_str("\n[emit] 无可行变体，未生成源文件\n");
        return Ok(out);
    }
    out.push_str(&prose::post_path_blocks(&report));
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
            let line = prose::passed_line(verified);
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
    out.push_str(&prose::negotiation_summary(&report));

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
    let mut text = prose::ledger_header(profile, verified, conclusion);
    text.push_str(&prose::ledger_primitives(profile));
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
    out.push_str(&format!("  plan  {}\n", measured.plan));
    out.push_str(&format!(
        "  证据  {} 条见证（这次执行，协商用的就是它）\n",
        measured.proven
    ));
    out.push_str(&format!(
        "  对拍  max_rel_err {:.3e}（与标量参考）\n",
        measured.max_rel_err
    ));
    out.push_str(&format!(
        "  实测  min {}, p50 {}（达成率按 p50）→ {:.1} GB/s, {:.1} GFLOP/s\n",
        ::diagnostic::render::per_run(measured.per_run_ns),
        ::diagnostic::render::per_run(measured.p50_ns),
        measured.measured.gb_s,
        measured.measured.gflops
    ));
    out.push_str(&prose::working_set_line(&measured));
    out.push_str(&prose::performance_note(
        measured.performance(),
        measured.peak.as_ref(),
    ));
    out.push_str(&prose::artifact_line(&measured));
    out.push_str(&prose::cache_dir_line(&measured));
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use cutile::emit::Fragment;

    fn settings() -> Settings {
        Settings::defaults(&llm_cli::settings::Roots::new(
            std::env::temp_dir().join("cutile-driver-tests"),
        ))
    }

    /// nvidia 在本机是否可执行（nvcc 在场）。T600 上为真，此时「缺设备即失败」的场景不成立，
    /// 用例主动让位——设备在场时跑的是真机对拍（更强的证据），不是「缺设备」文案。
    fn nvidia_present() -> bool {
        let catalogue = Catalogue::load(&settings().dirs()).expect("内置数据");
        catalogue
            .profile("nvidia")
            .map(cutile::witness::device_available)
            .unwrap_or(false)
    }

    /// 性能判据每轮都要有一句结论：没声明峰值时说“未测量”，不能什么都不打印。
    #[test]
    fn the_performance_line_is_printed_even_when_nothing_was_measured() {
        let peak = cutile::bench::Peak {
            value: 68.0e9,
            src: "2*8*2666e6".to_string(),
        };

        let reached = prose::performance_note(
            cutile::bench::Performance::new(Some(0.72), false),
            Some(&peak),
        );
        assert!(reached.contains("72.0%"), "{reached}");
        assert!(reached.contains("达到 70% 门槛"), "{reached}");
        assert!(reached.contains("不是“相对原生库的 85%”"), "{reached}");

        let below = prose::performance_note(
            cutile::bench::Performance::new(Some(0.0069), false),
            Some(&peak),
        );
        assert!(below.contains("0.7%"), "{below}");
        assert!(below.contains("低于 70% 门槛"), "{below}");
        assert!(below.contains("特化代码"), "{below}");

        let cached = prose::performance_note(
            cutile::bench::Performance::new(Some(0.19), true),
            Some(&peak),
        );
        assert!(cached.contains("口径无效"), "{cached}");
        assert!(cached.contains("不参与 §16 判定"), "{cached}");

        // 没有峰值：判据没有尺子，仍要说清“未测量”，而不是静默通过。
        let unknown = prose::performance_note(cutile::bench::Performance::new(None, false), None);
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
        if nvidia_present() {
            return; // nvidia 可执行时，缺设备被拒场景不成立
        }
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
        if nvidia_present() {
            return; // nvidia 可执行时，缺设备即失败场景不成立
        }
        let devices = vec!["nvidia".to_string()];
        let error = verify(&settings(), "gemv_i4", &devices, false).unwrap_err();
        assert_eq!(error.code(), Code::Unsupported);
        assert!(error.to_string().contains("缺设备即失败"), "{error}");
    }

    #[test]
    fn the_ledger_lists_every_declared_primitive() {
        let catalogue = Catalogue::load(&settings().dirs()).expect("内置数据");
        let profile = catalogue.resolve("cpu").expect("cpu 档案");
        let verified = Verified {
            device: "cpu".to_string(),
            channel: cutile::verify::Channel::Parallel,
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
        let note = write_ledger(
            &catalogue,
            &verified,
            &prose::passed_line(&verified),
            &settings().dirs(),
        )
        .expect("应写出台账");
        let path = note.trim().rsplit_once(' ').expect("路径").1;
        let text = std::fs::read_to_string(path).expect("台账可读");
        assert!(text.contains("| 原语 | 实现 | 等级 | 见证 |"), "{text}");
        assert!(text.contains("pipe2"), "{text}");
    }

    /// `cache gc` 的上限来源：`--limit` 最高，其次是配置键 `cache.limit_bytes`，内核常量
    /// 只在两者都缺时兜底；默认值与内核缺省同值（行为不变）。
    #[test]
    fn cache_gc_prefers_the_flag_then_the_settings_key() {
        let roots = llm_cli::settings::Roots::new(
            std::env::temp_dir().join(format!("cutile-gc-settings-{}", std::process::id())),
        );
        let mut settings = Settings::defaults(&roots);
        settings
            .apply_override("cache.limit_bytes=1024")
            .expect("合法覆盖");

        let text = cache_gc(&settings, None).expect("空缓存可回收");
        assert!(
            text.starts_with(&format!(
                "cache gc | 上限 {}",
                ::diagnostic::render::human_bytes(1024)
            )),
            "{text}"
        );

        let text = cache_gc(&settings, Some(2048)).expect("命令行覆盖");
        assert!(
            text.starts_with(&format!(
                "cache gc | 上限 {}",
                ::diagnostic::render::human_bytes(2048)
            )),
            "{text}"
        );

        let text = cache_gc(&Settings::defaults(&roots), None).expect("空缓存可回收");
        assert!(
            text.starts_with(&format!(
                "cache gc | 上限 {}",
                ::diagnostic::render::human_bytes(cutile::cache::CACHE_LIMIT)
            )),
            "{text}"
        );
    }
}
