//! `musa` 入口：把「生成 → 编译 → 计时 → 对拍」四段收成一条命令。
//!
//! 旧形态是四个文件手工串（生成器 + 编译 shell + 打补丁脚本 + 对拍脚本），中间产物落在固定
//! 路径上；现在每段一个子命令，`run` 把它们按数据流连起来，产物默认落临时目录、跑完删掉，
//! `--keep` 才留。判定失败走诊断码退出（对拍不过 = `Invalid`），成功只打印一份报告。

use std::path::{Path, PathBuf};
use std::process::{Command, ExitCode, Output};

use cutile::dirs::Dirs;
use diagnostic::render::Panel;
use diagnostic::{Code, Error, Result};

use llm_musa::{Job, Outcome, PROBES, Probe, Recipe, compare, lines, probe, program, shim, timed};

/// 缺省工具链根目录（`bin/mcc` 与 `lib/` 在它下面）。
const DEFAULT_ROOT: &str = "/usr/local/musa";

/// 缺省计时迭代次数：小形状下 launch 固定开销占比大，100 次够抹平抖动。
const DEFAULT_REPS: usize = 100;

/// 缺省对拍阈值：f32 级（tf32 张量核走 `--tol 1e-3`）。
const DEFAULT_TOL: f64 = 1e-6;

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match dispatch(args) {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("✗ {error}");
            ExitCode::from(error.code().exit_code())
        }
    }
}

/// 子命令分派；`run` 是四段合一，其余各管一段。
fn dispatch(mut args: Vec<String>) -> Result<()> {
    let command = take(&mut args, "子命令")?;
    if command == "list" {
        return list();
    }
    action(&command).map_or_else(
        || Err(usage(&format!("未知子命令 `{command}`"))),
        |step| step(&mut args),
    )
}

/// 一个子命令的入口：吃剩下的参数。
type Action = fn(&mut Vec<String>) -> Result<()>;

/// 子命令表：一段编排一个入口；`list` 不吃参数，故不进表。
const ACTIONS: [(&str, Action); 6] = [
    ("gen", generate),
    ("probe", probe_cmd),
    ("build", build),
    ("bench", bench),
    ("verify", verify),
    ("run", run),
];

/// 表里查子命令。
fn action(name: &str) -> Option<Action> {
    ACTIONS
        .iter()
        .find(|(command, _)| *command == name)
        .map(|(_, step)| *step)
}

/// 自描述：三条探针与生成方式。
fn list() -> Result<()> {
    println!(
        "{}",
        Panel::new("musa list")
            .row("生成", "声明式协商 → 规划 → 聚合 → 按 launch 形态装配")
            .row("探针", probe_names())
            .render()
    );
    Ok(())
}

/// 生成：写 `program.cu` 与 `reference.txt`，不在本机编译（生成不需要 MUSA）。
fn generate(args: &mut Vec<String>) -> Result<()> {
    let job = job(args)?;
    let out = dir_arg(args, "--out", "musa-gen")?;
    emit_into(&job, &out)?;
    println!("{}", generate_report(&job, &out).render());
    Ok(())
}

/// 写两份产物；生成只依赖声明自述，不依赖本机验证。
fn emit_into(job: &Job, out: &Path) -> Result<()> {
    let generated = program(job, &Dirs::under(out.join("build")))?;
    write(&out.join("program.cu"), &generated.device)?;
    write(&out.join("reference.txt"), &lines(&generated.reference))
}

/// `gen` 的报告。
fn generate_report(job: &Job, out: &Path) -> Panel {
    Panel::new("musa gen")
        .row("算子", &job.op)
        .row("形状", shape(job))
        .row("设备源", out.join("program.cu").display().to_string())
        .row("参考值", out.join("reference.txt").display().to_string())
}

/// 探针源码落到 `--out` 目录，交给目标机编译（仓库不存二进制）。
fn probe_cmd(args: &mut Vec<String>) -> Result<()> {
    let entry = entry_of(args)?;
    let out = dir_arg(args, "--out", "musa-probe")?;
    let path = out.join(format!("{}.cu", entry.name));
    write(&path, entry.source)?;
    println!("{}", probe_report(entry, &path).render());
    Ok(())
}

/// 名字 → 探针；未知名字由入口报（名字来自命令行）。
fn entry_of(args: &mut Vec<String>) -> Result<&'static Probe> {
    let name = take(args, "探针名")?;
    probe(&name).ok_or_else(|| usage(&format!("没有探针 `{name}`")))
}

/// `probe` 的报告。
fn probe_report(entry: &Probe, path: &Path) -> Panel {
    Panel::new("musa probe")
        .row("探针", entry.name)
        .row("内容", entry.about)
        .row("车道", entry.lane)
        .row("源码", path.display().to_string())
}

/// 编译一段设备源（四个脚本里的 `build.sh`）。
fn build(args: &mut Vec<String>) -> Result<()> {
    let source = PathBuf::from(take(args, "源文件")?);
    let binary = PathBuf::from(required(args, "--out")?);
    compile(args, &source, &binary)?;
    println!("✓ {}", binary.display());
    Ok(())
}

/// 编译一段设备源：转发头写在源旁边，编完只回二进制路径。
fn compile(args: &mut Vec<String>, source: &Path, binary: &Path) -> Result<()> {
    let headers = shim_dir(source)?;
    shim(&headers)?;
    recipe(args)?.compile(source, binary, &headers)
}

/// 工具链：根目录 + 车道。
fn recipe(args: &mut Vec<String>) -> Result<Recipe> {
    Recipe::at(&root(args)?, &lane(args, "plain")?)
}

/// 计时注入（四个脚本里的 `bench_patch.py`）：写一份包住整条链路的副本。
fn bench(args: &mut Vec<String>) -> Result<()> {
    let (source, out) = files(args)?;
    let reps = reps(args)?;
    write(&out, &timed_source(&source, reps)?)?;
    println!("✓ {}（{reps} 次迭代）", out.display());
    Ok(())
}

/// `bench` 的两条路径参数：源与产物。
fn files(args: &mut Vec<String>) -> Result<(PathBuf, PathBuf)> {
    Ok((
        PathBuf::from(take(args, "源文件")?),
        PathBuf::from(required(args, "--out")?),
    ))
}

/// 读原文件并夹入计时循环。
fn timed_source(path: &Path, reps: usize) -> Result<String> {
    timed(&read(path)?, reps)
}

/// `--reps` 或 `100`。
fn reps(args: &mut Vec<String>) -> Result<usize> {
    number(option(args, "--reps")?, DEFAULT_REPS, "--reps")
}

/// 对拍（四个脚本里的 `check_bits.py`）：逐元素相对误差，一个超阈就失败。
fn verify(args: &mut Vec<String>) -> Result<()> {
    let printed = file_arg(args, "程序输出")?;
    let reference = file_arg(args, "参考值")?;
    let tol = tolerance(args)?;
    report_outcome(&compare(&printed, &reference, tol)?, tol)
}

/// 读一个位置参数指向的文件。
fn file_arg(args: &mut Vec<String>, what: &str) -> Result<String> {
    read(Path::new(&take(args, what)?))
}

/// `--tol` 或 `1e-6`。
fn tolerance(args: &mut Vec<String>) -> Result<f64> {
    decimal(option(args, "--tol")?, DEFAULT_TOL, "--tol")
}

/// 四段合一：生成 → 写转发头 → 编译计时版 → 真机运行 → 对拍。
fn run(args: &mut Vec<String>) -> Result<()> {
    let session = session(args)?;
    let work = session.keep.clone().unwrap_or_else(scratch);
    let reference = stage(&session, &work)?;
    let output = measure(&session, &work)?;
    let printed = String::from_utf8_lossy(&output.stdout);
    let outcome = compare(&printed, &lines(&reference), session.tol)?;
    println!(
        "{}",
        run_report(&session, &work, &outcome, &output).render()
    );
    cleanup(&session, &work);
    verdict(&outcome, &output, session.tol)
}

/// `run` 的输入：作业 + 工具链 + 判定三项。
struct Session {
    job: Job,
    root: PathBuf,
    lane: String,
    reps: usize,
    tol: f64,
    keep: Option<PathBuf>,
}

/// `run` 的参数一次取齐（选项与位置参数可交错）。
fn session(args: &mut Vec<String>) -> Result<Session> {
    let job = job(args)?;
    let (root, lane) = tool(args)?;
    let (reps, tol) = rule(args)?;
    let keep = option(args, "--keep")?.map(PathBuf::from);
    Ok(Session {
        job,
        root,
        lane,
        reps,
        tol,
        keep,
    })
}

/// 工具链两项：根目录与编译车道。
fn tool(args: &mut Vec<String>) -> Result<(PathBuf, String)> {
    Ok((root(args)?, lane(args, "plain")?))
}

/// 判定两项：计时次数与对拍阈值。
fn rule(args: &mut Vec<String>) -> Result<(usize, f64)> {
    Ok((reps(args)?, tolerance(args)?))
}

/// 生成两版源并落盘，回参考值。
fn stage(session: &Session, work: &Path) -> Result<Vec<f32>> {
    headers(work)?;
    let generated = program(&session.job, &Dirs::under(work))?;
    emit_work(&generated, session.reps, work)?;
    Ok(generated.reference)
}

/// 转发头目录：`cuda*` 名字靠它编译（规则由库的 `shim` 定）。
fn headers(work: &Path) -> Result<PathBuf> {
    let dir = work.join("shim");
    std::fs::create_dir_all(&dir).map_err(|err| io(&dir, err))?;
    shim(&dir)?;
    Ok(dir)
}

/// 两版源写到 `work`：`program.cu` 原样，`bench.cu` 夹过计时。
fn emit_work(generated: &llm_musa::Program, reps: usize, work: &Path) -> Result<()> {
    write(&work.join("program.cu"), &generated.device)?;
    write(&work.join("bench.cu"), &timed(&generated.device, reps)?)
}

/// 编译计时版并运行；回进程输出（`stdout` 是结果，`stderr` 是计时行）。
fn measure(session: &Session, work: &Path) -> Result<Output> {
    let binary = work.join("program");
    Recipe::at(&session.root, &session.lane)?.compile(
        &work.join("bench.cu"),
        &binary,
        &headers(work)?,
    )?;
    Command::new(&binary)
        .output()
        .map_err(|err| io(&binary, err))
}

/// `run` 的报告。
fn run_report(session: &Session, work: &Path, outcome: &Outcome, output: &Output) -> Panel {
    Panel::new("musa run")
        .row("算子", &session.job.op)
        .row("形状", shape(&session.job))
        .row("车道", session.lane.as_str())
        .row("元素数", outcome.compared.to_string())
        .row("最大相对误差", format!("{:.3e}", outcome.worst))
        .row(
            "链路耗时",
            bench_line(&String::from_utf8_lossy(&output.stderr)),
        )
        .row("目录", work.display().to_string())
}

/// 跑完清理临时目录；`--keep` 时不清理。
fn cleanup(session: &Session, work: &Path) {
    if session.keep.is_none() {
        discard(work);
    }
}

/// 结论：进程成功且对拍全过才算通过；非零退出时不拿空输出对拍。
fn verdict(outcome: &Outcome, output: &Output, tol: f64) -> Result<()> {
    if !output.status.success() {
        return Err(Error::new(
            Code::Invalid,
            format!("设备程序退出 {}，没有输出可对拍", output.status),
        ));
    }
    report_outcome(outcome, tol)
}

/// 打印对拍结论；不过阈值就是失败（退出码 2）。
fn report_outcome(outcome: &llm_musa::Outcome, tol: f64) -> Result<()> {
    println!(
        "{}",
        Panel::new("musa verify")
            .row("元素数", outcome.compared.to_string())
            .row(
                "超阈元素",
                format!("{} / {}", outcome.over, outcome.compared)
            )
            .row("最大相对误差", format!("{:.3e}", outcome.worst))
            .row("出现位置", format!("第 {} 个", outcome.at))
            .row("阈值", format!("{tol:.0e}"))
            .render()
    );
    if outcome.pass() {
        return Ok(());
    }
    Err(Error::new(
        Code::Invalid,
        format!(
            "对拍失败：{} 个元素超阈，最大相对误差 {:.3e}（第 {} 个）",
            outcome.over, outcome.worst, outcome.at
        ),
    ))
}

/// 作业：算子名 + 可选 `m` `n`。缺省形状由 [`Job::of`] 给。
fn job(args: &mut Vec<String>) -> Result<Job> {
    let job = Job::of(&take(args, "算子名")?);
    shaped(args, job)
}

/// 两个可选位置参数：`m` 先、`n` 后，缺哪个就沿用缺省。
fn shaped(args: &mut Vec<String>, job: Job) -> Result<Job> {
    let Some(m) = positional(args) else {
        return Ok(job);
    };
    let m = number(Some(m), job.m, "m")?;
    let n = optional_number(args, job.n, "n")?;
    Ok(job.shape(m, n))
}

/// 下一个位置参数若是整数就采用，否则沿用缺省。
fn optional_number(args: &mut Vec<String>, fallback: usize, what: &str) -> Result<usize> {
    number(positional(args), fallback, what)
}

/// 工具链根目录：`--root` 或 `/usr/local/musa`。
fn root(args: &mut Vec<String>) -> Result<PathBuf> {
    Ok(option(args, "--root")?.map_or_else(|| PathBuf::from(DEFAULT_ROOT), PathBuf::from))
}

/// 编译车道：`--lane` 或调用方给的缺省。
fn lane(args: &mut Vec<String>, fallback: &str) -> Result<String> {
    Ok(option(args, "--lane")?.unwrap_or_else(|| fallback.to_string()))
}

/// 输出目录：`--out` 或名字；总是先建出来。
fn dir_arg(args: &mut Vec<String>, name: &str, fallback: &str) -> Result<PathBuf> {
    let dir = option(args, name)?.map_or_else(|| PathBuf::from(fallback), PathBuf::from);
    std::fs::create_dir_all(&dir).map_err(|err| io(&dir, err))?;
    Ok(dir)
}

/// 第一个位置参数（不以 `--` 起头）。
fn positional(args: &mut Vec<String>) -> Option<String> {
    let first = args.first()?;
    (!first.starts_with("--")).then(|| args.remove(0))
}

/// 必填的位置参数。
fn take(args: &mut Vec<String>, what: &str) -> Result<String> {
    positional(args).ok_or_else(|| usage(&format!("缺少{what}")))
}

/// 必填的 `--选项 值`。
fn required(args: &mut Vec<String>, name: &str) -> Result<String> {
    option(args, name)?.ok_or_else(|| usage(&format!("缺少 `{name}`")))
}

/// `--选项 值`：缺值报错（不静默退回缺省，否则打错一个选项名会跑出另一套参数）。
fn option(args: &mut Vec<String>, name: &str) -> Result<Option<String>> {
    let Some(index) = args.iter().position(|arg| arg == name) else {
        return Ok(None);
    };
    args.remove(index);
    if index >= args.len() {
        return Err(usage(&format!("`{name}` 缺值")));
    }
    Ok(Some(args.remove(index)))
}

/// 十进制整数（`0` 也接受：形状与次数由调用方判）。
fn number(text: Option<String>, fallback: usize, what: &str) -> Result<usize> {
    let Some(text) = text else {
        return Ok(fallback);
    };
    text.parse()
        .map_err(|_| usage(&format!("`{what}` 不是整数：`{text}`")))
}

/// 十进制实数（阈值）。
fn decimal(text: Option<String>, fallback: f64, what: &str) -> Result<f64> {
    let Some(text) = text else {
        return Ok(fallback);
    };
    text.parse()
        .map_err(|_| usage(&format!("`{what}` 不是数：`{text}`")))
}

/// `stderr` 里的计时行：打完补丁的程序只打印这一行，原样搬进报告。
fn bench_line(stderr: &str) -> String {
    stderr
        .lines()
        .find(|line| line.starts_with("bench:"))
        .map_or_else(
            || "（程序没有打印计时行）".to_string(),
            |line| line.trim().to_string(),
        )
}

/// 探针清单：`名字（说明）` 一行。
fn probe_names() -> String {
    PROBES
        .iter()
        .map(|entry| format!("{}（{}）", entry.name, entry.about))
        .collect::<Vec<_>>()
        .join("；")
}

/// 形状：`m×n`。
fn shape(job: &Job) -> String {
    format!("{}×{}", job.m, job.n)
}

/// 转发头目录：与源文件同级的 `shim/`，与旧 `-Ishim` 一致。
fn shim_dir(source: &Path) -> Result<PathBuf> {
    let parent = source
        .parent()
        .ok_or_else(|| usage(&format!("`{}` 没有上级目录", source.display())))?;
    Ok(parent.join("shim"))
}

/// 临时目录：`--keep` 没给时用，跑完删。
fn scratch() -> PathBuf {
    std::env::temp_dir().join(format!("llm-musa-{}", std::process::id()))
}

/// 清理临时目录；清理失败只提醒，不改判定（测量已经做完）。
fn discard(dir: &Path) {
    if let Err(err) = std::fs::remove_dir_all(dir) {
        eprintln!("! 临时目录没删掉 {}：{err}", dir.display());
    }
}

/// 读文件。
fn read(path: &Path) -> Result<String> {
    std::fs::read_to_string(path).map_err(|err| io(path, err))
}

/// 写文件。
fn write(path: &Path, text: &str) -> Result<()> {
    std::fs::write(path, text).map_err(|err| io(path, err))
}

/// IO 失败：带路径。
fn io(path: &Path, err: std::io::Error) -> Error {
    Error::new(Code::Io, format!("{}：{err}", path.display()))
}

/// 用法错误。
fn usage(message: &str) -> Error {
    Error::new(Code::Invalid, format!("{message}\n{USAGE}"))
}

/// 用法。
const USAGE: &str = "\
用法: musa <子命令> [参数]

  list                                        生成方式与探针
  gen   <op> [m] [n] [--out DIR]               写 program.cu 与 reference.txt
  probe <name> [--out DIR]                    写探针源码
  build <src.cu> --out BIN [--root DIR] [--lane LANE]
  bench <src.cu> --out PATCHED.cu [--reps N]  注入事件计时
  verify <printed.txt> <reference.txt> [--tol T]
  run   <op> [m] [n] [--root DIR] [--lane LANE] [--reps N] [--tol T] [--keep DIR]
";
