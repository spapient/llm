//! `musa` 入口：把「生成 → 编译 → 计时 → 对拍」四段收成一条命令。
//!
//! 旧形态是四个文件手工串（生成器 + 编译 shell + 打补丁脚本 + 对拍脚本），中间产物落在固定
//! 路径上；现在每段一个子命令，`run` 把它们按数据流连起来，产物默认落临时目录、跑完删掉，
//! `--keep` 才留。判定失败走诊断码退出（对拍不过 = `Invalid`），成功只打印一份报告。

use std::path::{Path, PathBuf};
use std::process::{Command, ExitCode};

use diagnostic::{Code, Error, Result};
use report::Report;

use llm_tools::musa::{Job, Recipe, compare, lanes, probe, program, shim, timed};

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
    if args.is_empty() {
        return Err(usage("缺少子命令"));
    }
    let command = args.remove(0);
    match command.as_str() {
        "list" => list(),
        "gen" => generate(&mut args),
        "probe" => probe_cmd(&mut args),
        "build" => build(&mut args),
        "bench" => bench(&mut args),
        "verify" => verify(&mut args),
        "run" => run(&mut args),
        other => Err(usage(&format!("未知子命令 `{other}`"))),
    }
}

/// 自描述：认领的车道前缀与三条探针。
fn list() -> Result<()> {
    let names = probe_names();
    println!(
        "{}",
        Report::new("musa list")
            .row("车道前缀", lanes().collect::<Vec<_>>().join(" / "))
            .row("（其余算子）", "按逐点车道处理")
            .row("探针", names)
            .render()
    );
    Ok(())
}

/// 生成：写 `program.cu` 与 `reference.txt`，不在本机编译（生成不需要 MUSA）。
fn generate(args: &mut Vec<String>) -> Result<()> {
    let job = job(args)?;
    let out = dir_arg(args, "--out", "musa-gen")?;
    let generated = program(&job)?;
    write(&out.join("program.cu"), &generated.device)?;
    write(&out.join("reference.txt"), &generated.reference)?;
    println!(
        "{}",
        Report::new("musa gen")
            .row("算子", &job.op)
            .row("形状", shape(&job))
            .row("设备源", out.join("program.cu").display().to_string())
            .row("参考值", out.join("reference.txt").display().to_string())
            .render()
    );
    Ok(())
}

/// 探针源码落到 `--out` 目录，交给目标机编译（仓库不存二进制）。
fn probe_cmd(args: &mut Vec<String>) -> Result<()> {
    let name = take(args, "探针名")?;
    let entry = probe(&name).ok_or_else(|| usage(&format!("没有探针 `{name}`")))?;
    let out = dir_arg(args, "--out", "musa-probe")?;
    let path = out.join(format!("{name}.cu"));
    write(&path, entry.source)?;
    println!(
        "{}",
        Report::new("musa probe")
            .row("探针", entry.name)
            .row("内容", entry.about)
            .row("车道", entry.lane)
            .row("源码", path.display().to_string())
            .render()
    );
    Ok(())
}

/// 编译一段设备源（四个脚本里的 `build.sh`）。
fn build(args: &mut Vec<String>) -> Result<()> {
    let source = PathBuf::from(take(args, "源文件")?);
    let binary = PathBuf::from(required(args, "--out")?);
    let recipe = Recipe::at(&root(args)?, &lane(args, "plain")?)?;
    let headers = shim_dir(&source)?;
    shim(&headers)?;
    recipe.compile(&source, &binary, &headers)?;
    println!("✓ {}", binary.display());
    Ok(())
}

/// 计时注入（四个脚本里的 `bench_patch.py`）：写一份包住整条链路的副本。
fn bench(args: &mut Vec<String>) -> Result<()> {
    let source = PathBuf::from(take(args, "源文件")?);
    let out = PathBuf::from(required(args, "--out")?);
    let reps = number(option(args, "--reps")?, DEFAULT_REPS, "--reps")?;
    write(&out, &timed(&read(&source)?, reps)?)?;
    println!("✓ {}（{reps} 次迭代）", out.display());
    Ok(())
}

/// 对拍（四个脚本里的 `check_bits.py`）：逐元素相对误差，一个超阈就失败。
fn verify(args: &mut Vec<String>) -> Result<()> {
    let printed = read(Path::new(&take(args, "程序输出")?))?;
    let reference = read(Path::new(&take(args, "参考值")?))?;
    let tol = decimal(option(args, "--tol")?, DEFAULT_TOL, "--tol")?;
    report_outcome(&compare(&printed, &reference, tol)?, tol)
}

/// 四段合一：生成 → 写转发头 → 编译计时版 → 真机运行 → 对拍。
fn run(args: &mut Vec<String>) -> Result<()> {
    let job = job(args)?;
    let root = root(args)?;
    let lane = lane(args, if job.wmma { "wmma" } else { "plain" })?;
    let reps = number(option(args, "--reps")?, DEFAULT_REPS, "--reps")?;
    let tol = decimal(option(args, "--tol")?, DEFAULT_TOL, "--tol")?;
    let keep = option(args, "--keep")?.map(PathBuf::from);
    let work = keep.clone().unwrap_or_else(scratch);
    let headers = work.join("shim");
    std::fs::create_dir_all(&headers).map_err(|err| io(&headers, err))?;
    shim(&headers)?;

    let generated = program(&job)?;
    let source = work.join("program.cu");
    let patched = work.join("bench.cu");
    let binary = work.join("program");
    write(&source, &generated.device)?;
    write(&patched, &timed(&generated.device, reps)?)?;
    Recipe::at(&root, &lane)?.compile(&patched, &binary, &headers)?;

    let output = Command::new(&binary)
        .output()
        .map_err(|err| io(&binary, err))?;
    let printed = String::from_utf8_lossy(&output.stdout);
    let outcome = compare(&printed, &generated.reference, tol)?;
    let elapsed = bench_line(&String::from_utf8_lossy(&output.stderr));
    println!(
        "{}",
        Report::new("musa run")
            .row("算子", &job.op)
            .row("形状", shape(&job))
            .row("车道", lane.as_str())
            .row("元素数", outcome.compared.to_string())
            .row("最大相对误差", format!("{:.3e}", outcome.worst))
            .row("链路耗时", elapsed)
            .row("目录", work.display().to_string())
            .render()
    );
    if keep.is_none() {
        discard(&work);
    }
    if output.status.success() {
        report_outcome(&outcome, tol)
    } else {
        Err(Error::new(
            Code::Invalid,
            format!("设备程序退出 {}，没有输出可对拍", output.status),
        ))
    }
}

/// 打印对拍结论；不过阈值就是失败（退出码 2）。
fn report_outcome(outcome: &llm_tools::musa::Outcome, tol: f64) -> Result<()> {
    println!(
        "{}",
        Report::new("musa verify")
            .row("元素数", outcome.compared.to_string())
            .row("超阈元素", format!("{} / {}", outcome.over, outcome.compared))
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

/// 作业：算子名 + 可选 `m` `n` + `--tensor`。缺省形状由 [`Job::of`] 给。
fn job(args: &mut Vec<String>) -> Result<Job> {
    let mut job = Job::of(&take(args, "算子名")?);
    if let Some(m) = positional(args) {
        let m = number(Some(m), job.m, "m")?;
        let n = job.n;
        job = job.shape(m, n);
    }
    if let Some(n) = positional(args) {
        let m = job.m;
        let n = number(Some(n), job.n, "n")?;
        job = job.shape(m, n);
    }
    if switch(args, "--tensor") {
        job = job.tensor();
    }
    Ok(job)
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

/// `--开关`：出现即真，并从参数表里摘掉（剩下的位置参数才不会被它挡住）。
fn switch(args: &mut Vec<String>, name: &str) -> bool {
    let before = args.len();
    args.retain(|arg| arg != name);
    args.len() != before
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
        .map_or_else(|| "（程序没有打印计时行）".to_string(), |line| line.trim().to_string())
}

/// 探针清单：`名字（说明）` 一行。
fn probe_names() -> String {
    llm_tools::musa::PROBES
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

  list                                        车道前缀与探针
  gen   <op> [m] [n] [--tensor] [--out DIR]   写 program.cu 与 reference.txt
  probe <name> [--out DIR]                    写探针源码
  build <src.cu> --out BIN [--root DIR] [--lane LANE]
  bench <src.cu> --out PATCHED.cu [--reps N]  注入事件计时
  verify <printed.txt> <reference.txt> [--tol T]
  run   <op> [m] [n] [--tensor] [--root DIR] [--lane LANE] [--reps N] [--tol T] [--keep DIR]
";
