//! 厂商工具链探测：`nvcc` / `ccec` / `mcc` / `xcrun metal` / MSVC `cl.exe`。

use llm_optim::toolchain::{Toolchain, parse_version, probe};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::OnceLock;

// ---------------------------------------------------------------------------
// 厂商工具链探测：nvcc / xcrun metal / MSVC cl.exe。
// ---------------------------------------------------------------------------

static NVCC: OnceLock<Option<Toolchain>> = OnceLock::new();
static METAL: OnceLock<Option<Toolchain>> = OnceLock::new();
static MUSA: OnceLock<Option<Toolchain>> = OnceLock::new();
static CCE: OnceLock<Option<Toolchain>> = OnceLock::new();

/// 探测 nvcc（CUDA 编译器）：`nvcc --version` 只打印版本、不调用宿主 cl，故可独立成功；宿主编译器由
/// 调用方用 `-ccbin` 另行指定。
pub(crate) fn detect_nvcc() -> Option<Toolchain> {
    NVCC.get_or_init(|| probe("nvcc")).clone()
}

/// 探测 ccec（昇腾编译器，随 CANN 的 Ascend Toolkit 安装，不随宿主发行版）。
pub(crate) fn detect_cce() -> Option<Toolchain> {
    CCE.get_or_init(|| probe("ccec")).clone()
}

/// 这台机器上有没有昇腾设备：`npu-smi info -l` 能列出 NPU 就算有（华为版 `nvidia-smi -L`）。
pub(crate) fn ascend_device() -> bool {
    let Ok(output) = Command::new("npu-smi").args(["info", "-l"]).output() else {
        return false;
    };
    if !output.status.success() {
        return false;
    }
    String::from_utf8_lossy(&output.stdout)
        .lines()
        .any(|line| line.trim_start().starts_with("NPU ID"))
}

/// 探测 mcc（MUSA 编译器，clang-14 驱动）。
pub(crate) fn detect_musa() -> Option<Toolchain> {
    MUSA.get_or_init(|| probe("mcc")).clone()
}

/// 这台机器上有没有 MUSA 设备：`mthreads-gmi -L`（摩尔线程的 `nvidia-smi -L` 同构物）能列出卡就算有。
pub(crate) fn musa_device() -> bool {
    let Ok(output) = Command::new("mthreads-gmi").arg("-L").output() else {
        return false;
    };
    if !output.status.success() {
        return false;
    }
    String::from_utf8_lossy(&output.stdout).lines().any(|line| {
        line.split(':')
            .next()
            .is_some_and(|head| head.starts_with("GPU"))
    })
}

/// 探测 Metal 编译器（`xcrun metal --version`）：命令可用、能自报版本时才算数。
pub(crate) fn detect_metal() -> Option<Toolchain> {
    METAL.get_or_init(probe_metal).clone()
}

fn probe_metal() -> Option<Toolchain> {
    let output = Command::new("xcrun")
        .args(["-sdk", "macosx", "metal", "--version"])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let text = String::from_utf8_lossy(&output.stdout).to_string()
        + &String::from_utf8_lossy(&output.stderr);
    Some(Toolchain {
        command: "xcrun metal".to_string(),
        version: parse_version(&text),
        banner: text
            .lines()
            .map(str::trim)
            .find(|line| !line.is_empty())?
            .to_string(),
    })
}

/// 定位 MSVC cl.exe（Windows 上 nvcc 的宿主编译器）：先问 `vswhere`，再退回环境变量登记的两个
/// 安装根，多版本共存时按**数值**取最新。
pub(crate) fn find_cl() -> Option<PathBuf> {
    if std::env::consts::OS != "windows" {
        return None;
    }
    candidates().into_iter().max_by_key(|cl| msvc_version(cl))
}

fn candidates() -> Vec<PathBuf> {
    let mut found = vswhere_cls();
    for key in ["VCINSTALLDIR", "VSINSTALLDIR"] {
        if let Some(root) = std::env::var_os(key) {
            found.extend(msvc_cls(Path::new(&root)));
        }
    }
    found
}

fn vswhere_cls() -> Vec<PathBuf> {
    let Some(program_files) = std::env::var_os("ProgramFiles(x86)") else {
        return Vec::new();
    };
    let vswhere = Path::new(&program_files).join("Microsoft Visual Studio/Installer/vswhere.exe");
    let Ok(output) = Command::new(vswhere)
        .args([
            "-latest",
            "-products",
            "*",
            "-find",
            r"VC\Tools\MSVC\**\bin\Hostx64\x64\cl.exe",
        ])
        .output()
    else {
        return Vec::new();
    };
    if !output.status.success() {
        return Vec::new();
    }
    String::from_utf8_lossy(&output.stdout)
        .lines()
        .map(|line| PathBuf::from(line.trim()))
        .filter(|cl| cl.is_file())
        .collect()
}

fn msvc_cls(root: &Path) -> Vec<PathBuf> {
    let mut found = Vec::new();
    for base in [root.join("Tools/MSVC"), root.join("VC/Tools/MSVC")] {
        let Ok(entries) = std::fs::read_dir(base) else {
            continue;
        };
        found.extend(
            entries
                .flatten()
                .map(|entry| entry.path().join("bin/Hostx64/x64/cl.exe")),
        );
    }
    found.into_iter().filter(|cl| cl.is_file()).collect()
}

pub(crate) fn msvc_version(cl: &Path) -> Vec<u32> {
    let text = cl.to_string_lossy().replace('/', "\\");
    let Some((_, rest)) = text.rsplit_once("MSVC\\") else {
        return Vec::new();
    };
    rest.split('\\')
        .next()
        .map(|version| {
            version
                .split('.')
                .map(|part| part.parse::<u32>().unwrap_or(0))
                .collect()
        })
        .unwrap_or_default()
}
