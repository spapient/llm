//! Metal 运行时胶水：一次性目录内编 `.metal → .air → .metallib`、链接 ObjC host 驱动、执行并回读位模式。

use cutile::dirs::Dirs;
use diagnostic::{Code, Error, Result};
use llm_optim::backend::Readback;
use llm_optim::driver;
use prof::prof::Profile;
use std::path::{Path, PathBuf};
use std::process::Command;

use crate::metal::witness::metal_driver;

/// 跑一次见证：一次性目录里编 `metallib`、链接见证驱动、回读位模式。
pub(crate) fn run_metal(
    program: &str,
    count: usize,
    flags: &[String],
    dirs: &Dirs,
) -> Result<Readback> {
    let dir = driver::scratch(dirs, "witness-metal");
    let outcome = metal_in_dir(&dir, program, count, flags);
    let _ = std::fs::remove_dir_all(&dir);
    outcome
}

fn metal_in_dir(dir: &Path, program: &str, count: usize, flags: &[String]) -> Result<Readback> {
    let metalib = metal_kernel(dir, program, flags)?;
    let exe = metal_driver_exe(dir)?;
    metal_readback(&exe, &metalib, count)
}

/// 写 MSL 并编成 `metallib`：`xcrun metal` 管语言层、`metallib` 管链接；诊断里带 `.metal` 路径。
pub fn metal_kernel(dir: &Path, program: &str, flags: &[String]) -> Result<PathBuf> {
    let msl = dir.join("witness.metal");
    let air = dir.join("witness.air");
    let metalib = dir.join("witness.metallib");
    driver::write_text(&msl, program)?;
    let mut compile = xcrun("metal", flags);
    compile.arg("-c").arg(&msl).arg("-o").arg(&air);
    driver::run_tool("xcrun metal", &mut compile, &msl)?;
    let mut link = xcrun("metallib", &[]);
    link.arg(&air).arg("-o").arg(&metalib);
    driver::run_tool("xcrun metallib", &mut link, &msl)?;
    Ok(metalib)
}

/// 编译并执行一次 Metal 生产态内核（设备源 + host 驱动），返回 `(stdout, stderr)` 原文。
/// 与 [`llm_optim::backend::Driver::compile_run`] 同口径：一次性目录，用完即删。
pub fn compile_run_metal(
    profile: &Profile,
    dirs: &Dirs,
    msl: &str,
    host: &str,
    label: &str,
) -> Result<(String, String)> {
    let dir = driver::scratch(dirs, &format!("{label}-metal"));
    let outcome = metal_kernel_in_dir(&dir, msl, host, &profile.flags);
    let _ = std::fs::remove_dir_all(&dir);
    outcome
}

fn metal_kernel_in_dir(
    dir: &Path,
    msl: &str,
    driver: &str,
    flags: &[String],
) -> Result<(String, String)> {
    let metalib = metal_kernel(dir, msl, flags)?;
    let exe = metal_driver_exe_from(dir, driver, "kernel_driver", "内核对拍")?;
    let mut command = Command::new(&exe);
    command.arg(&metalib);
    driver::run_exe(&mut command, &exe, "Metal 内核执行失败")
}

fn metal_driver_exe_from(
    dir: &Path,
    driver_source: &str,
    stem: &str,
    role: &str,
) -> Result<PathBuf> {
    let Some(compiler) = llm_optim::toolchain::detect() else {
        return Err(Error::new(
            Code::Unsupported,
            format!("缺少 host C 编译器：Metal {role}需要一个 host 驱动把内核发射到 GPU"),
        ));
    };
    let driver = dir.join(format!("{stem}.m"));
    let exe = dir.join(stem);
    driver::write_text(&driver, driver_source)?;
    let mut clang = Command::new(&compiler.command);
    clang
        .args(["-fobjc-arc", "-O2", "-o"])
        .arg(&exe)
        .arg(&driver);
    clang.args(["-framework", "Foundation", "-framework", "Metal"]);
    driver::run_tool(&compiler.command, &mut clang, &driver)?;
    Ok(exe)
}

fn xcrun(tool: &str, flags: &[String]) -> Command {
    let mut command = Command::new("xcrun");
    command.args(["-sdk", "macosx"]).arg(tool).args(flags);
    command
}

fn metal_driver_exe(dir: &Path) -> Result<PathBuf> {
    metal_driver_exe_from(dir, &metal_driver(crate::dialect::MAX_OUT), "driver", "见证")
}

fn metal_readback(exe: &Path, metalib: &Path, count: usize) -> Result<Readback> {
    let mut command = Command::new(exe);
    command.arg(metalib).arg(count.to_string());
    let (stdout, stderr) = driver::run_exe(&mut command, exe, "Metal 见证执行失败")?;
    Ok(driver::readback(&stdout, &stderr))
}
