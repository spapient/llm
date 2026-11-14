//! 见证前端：C / Metal / CUDA 的程序生成与执行。
//!
//! 每个前端分三段：**头**（前置声明）、**内核包装**（分派与回读）、**主函数**（拼成文本）；
//! 三段共用的装载、命名、起进程与回读各只有一份，算不算证据由 [`crate::witness`] 决定。
//! 编译器探测与身份在 [`crate::toolchain`]，编译/执行原语在 [`crate::runner`]。
//!
//! 各前端实现见 [`cpu`]（host C）、[`metal`]（MSL + Metal 驱动）、[`cuda`]（nvcc 真机）。

mod cpu;
mod cuda;
mod metal;

pub(crate) use cpu::{CpuDevice, HOST_FLAGS};
pub(crate) use cuda::CudaDevice;
pub use cuda::compile_run_cuda;
pub(crate) use metal::MetalDevice;

use crate::atom::{Prim, Witness};
use crate::dirs::Dirs;
use crate::runner;
use crate::witness::Selected;
use crate::witness::source;
use diagnostic::{Code, Error, Result};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::Command;

/// 见证函数在 C 侧（C 与 CUDA 前端）的形参文本。
pub(crate) const C_PARAMS: &str = "float *";

/// 本次要拼进程序的见证源码：文件名 → 文本。
pub(crate) type Sources = BTreeMap<String, String>;

/// 一次执行回读出来的东西：每条见证写出的个数 + 按顺序回读的 f32 值。
pub(crate) type Readback = (Vec<i64>, Vec<f32>);

/// 一份见证源文件：读出来，并确认里面有这个前端能调用的函数（签名由编译器回答）。
/// 错误里带原语名，是因为同一个文件被多条见证共用时，光有文件名定位不了是哪一条。
fn witness_source(prim: &Prim, witness: &Witness, dirs: &Dirs, params: &str) -> Result<String> {
    let text = source(witness, dirs).map_err(|error| error.with(format!("`{prim}` 的")))?;
    if !text.contains(&format!("{}(", witness.func)) {
        let message = format!(
            "`{prim}` 的见证 `{}` 里没有函数 `{}({params})`",
            witness.file, witness.func
        );
        return Err(Error::new(Code::Invalid, message));
    }
    Ok(text)
}

/// 本次执行的见证源码：按文件去重，同一文件里的多条见证只出现一次。
fn witness_files(selected: &[Selected<'_>], dirs: &Dirs, params: &str) -> Result<Sources> {
    let mut files = Sources::new();
    for (prim, _, witness) in selected {
        if files.contains_key(&witness.file) {
            continue;
        }
        files.insert(
            witness.file.clone(),
            witness_source(prim, witness, dirs, params)?,
        );
    }
    Ok(files)
}

/// 源码段接在头之后：每段补齐换行再空一行，段与段之间因此总有分隔。
fn push_sources(program: &mut String, files: &Sources) {
    for text in files.values() {
        program.push_str(text);
        if !program.ends_with('\n') {
            program.push('\n');
        }
        program.push('\n');
    }
}

/// 让一段文本以换行结尾（空文本保持为空）：下一段可以直接接上。
fn normalize_newline(text: &mut String) {
    if !text.is_empty() && !text.ends_with('\n') {
        text.push('\n');
    }
}

/// 写一个文件，并把它的上层目录建出来：一次性构建目录都只有一层。
fn write_text(path: &Path, text: &str) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|error| {
            Error::new(Code::Io, format!("创建 {} 失败：{error}", parent.display()))
        })?;
    }
    std::fs::write(path, text)
        .map_err(|error| Error::new(Code::Io, format!("写 {} 失败：{error}", path.display())))
}

/// 跑一条编译器命令；非零退出码时把命令、完整输出与源文件路径一并带进诊断。
fn run_tool(name: &str, command: &mut Command, source: &Path) -> Result<()> {
    let output = command
        .output()
        .map_err(|error| Error::new(Code::Io, format!("调用 `{name}` 失败：{error}")))?;
    if output.status.success() {
        return Ok(());
    }
    Err(Error::new(
        Code::Invalid,
        format!(
            "`{name}` 执行失败：\n{}\n{}\n源文件：{}",
            String::from_utf8_lossy(&output.stdout).trim(),
            String::from_utf8_lossy(&output.stderr).trim(),
            source.display()
        ),
    ))
}

/// 跑一个已经编好的可执行文件，返回 `(stdout, stderr)` 原文。非零退出码报 [`Code::Unsupported`]：编得过却跑不起来说明这台机器现在跑不了，不是见证写错了。
fn run_exe(command: &mut Command, path: &Path, failure: &str) -> Result<(String, String)> {
    let output = command
        .output()
        .map_err(|error| Error::new(Code::Io, format!("运行 {} 失败：{error}", path.display())))?;
    let stderr = String::from_utf8_lossy(&output.stderr).to_string();
    if !output.status.success() {
        return Err(Error::new(
            Code::Unsupported,
            format!(
                "{failure}（退出码 {:?}）：{}",
                output.status.code(),
                stderr.trim()
            ),
        ));
    }
    Ok((String::from_utf8_lossy(&output.stdout).to_string(), stderr))
}

/// 从 stderr 读出每条见证声明的写入个数（按调用顺序）。
pub(crate) fn parse_counts(stderr: &str) -> Vec<i64> {
    stderr
        .lines()
        .filter_map(|line| line.trim().strip_prefix("witness "))
        .filter_map(|value| value.trim().parse::<i64>().ok())
        .collect()
}

/// 回读解码：个数走 stderr（`witness N`），值走 stdout 的 f32 位模式；两条通道必须分开，
/// 否则 [`runner::parse_bits`] 会把计数当成值解出来。
fn readback(stdout: &str, stderr: &str) -> Readback {
    (parse_counts(stderr), runner::parse_bits(stdout))
}

/// 一个新的一次性目录：`<work_dir>/<tag>-<pid>-<序号>`，调用方负责删除。pid 隔开进程、序号
/// 隔开同进程内的多次调用：同时跑的两个 `cutile` 不会互相覆盖对方的中间产物。
pub(crate) fn scratch(dirs: &Dirs, tag: &str) -> PathBuf {
    static SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    dirs.work().join(format!(
        "{tag}-{}-{}",
        std::process::id(),
        SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
    ))
}

/// 两个后端 kernel 共享的分派段：按 `tid` 的 `switch`、cap 截断与回写循环。`close` 是 kernel
/// 收尾文本（Metal 与 CUDA 的换行数不同）：`local` 声明留在各自函数里，这里只拼逐字相同的部分。
fn push_dispatch(program: &mut String, max: usize, selected: &[Selected<'_>], close: &str) {
    for (index, (_, _, witness)) in selected.iter().enumerate() {
        program.push_str(&format!(
            "        case {index}: n = {}(local); break;\n",
            witness.func
        ));
    }
    program.push_str(&format!(
        "        default: break;\n\
         \x20   }}\n\
         \x20   int cap = n < 0 ? 0 : (n > {max} ? {max} : n);\n\
         \x20   counts[tid] = cap;\n\
         \x20   for (int i = 0; i < {max}; ++i) out[tid * {max} + i] = i < cap ? local[i] : 0.0f;\n\
         {close}"
    ));
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::witness::{Device, MAX_OUT, Scope};

    /// 生成各前端的见证源码。默认只断言生成非空；`CUTILE_CAPTURE_GOLDEN` 显式打开时才把副本
    /// 写到 `/tmp/cutile-golden` 供人工比对，不常驻产物。
    #[test]
    fn golden_capture() {
        let capture = std::env::var("CUTILE_CAPTURE_GOLDEN").is_ok();
        let dir = std::path::PathBuf::from("/tmp/cutile-golden");
        let dirs = Dirs::under(std::env::temp_dir().join("cutile-golden-work"));
        let catalogue = crate::builtin::Catalogue::try_builtin().expect("内置数据");
        for (vendor, file) in [
            ("cpu", "witness_c.txt"),
            ("metal", "witness_metal.txt"),
            ("nvidia", "witness_cuda.txt"),
        ] {
            let profile = catalogue.profile(vendor).cloned().expect("档案");
            let selected = crate::witness::declared_witnesses(&profile).expect("见证");
            if selected.is_empty() {
                continue;
            }
            let text = match vendor {
                "cpu" => CpuDevice::program::<MAX_OUT>(&profile, &dirs, &selected, Scope::Device),
                "metal" => {
                    MetalDevice::program::<MAX_OUT>(&profile, &dirs, &selected, Scope::Device)
                }
                _ => CudaDevice::program::<MAX_OUT>(&profile, &dirs, &selected, Scope::Device),
            }
            .expect("可生成");
            assert!(!text.is_empty(), "{vendor} 生成文本不能为空");
            if capture {
                std::fs::create_dir_all(&dir).expect("目录");
                std::fs::write(dir.join(file), text).expect("写");
            }
        }
        let metal = catalogue.profile("metal").cloned().expect("档案");
        let selected = crate::witness::declared_witnesses(&metal).expect("见证");
        let text =
            CpuDevice::program::<MAX_OUT>(&metal, &dirs, &selected, Scope::Host).expect("可生成");
        assert!(!text.is_empty(), "metal 档案的 host 自检文本不能为空");
        if capture {
            std::fs::create_dir_all(&dir).expect("目录");
            std::fs::write(dir.join("witness_metal_host.txt"), text).expect("写");
        }
    }

    #[test]
    fn scratch_directories_do_not_collide() {
        let dirs = Dirs::under(std::env::temp_dir());
        let first = scratch(&dirs, "gate");
        let second = scratch(&dirs, "gate");
        assert_ne!(first, second);
        assert_eq!(first.parent(), Some(dirs.work()));
    }
}
