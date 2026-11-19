//! 编译 / 执行 / 回读的共享机制原语：装载见证源、拼程序文本、起子进程、解码回读。
//!
//! 这是 cutile 的「机制」层，不认识任何设备厂商：C 宿主参考后端（[`crate::device`]）与
//! `engine` 里的 Metal / CUDA 后端都从这取同一套原语，避免三段后端各写一份「写文件 / 起进程 /
//! 解回读」。厂商差异只落在「源文本怎么拼」与「哪条工具链来编」，不进这层。

use crate::atom::{Prim, Witness};
use crate::backend::Readback;
use crate::dirs::Dirs;
use crate::witness::Selected;
use diagnostic::{Code, Error, Result};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::Command;

/// 见证函数在 C 侧（C 与 CUDA 前端）的形参文本。
pub const C_PARAMS: &str = "float *";

/// 本次要拼进程序的见证源码：文件名 → 文本。
pub type Sources = BTreeMap<String, String>;

/// 一份见证源文件：读出来，并确认里面有这个前端能调用的函数（签名由编译器回答）。
/// 错误里带原语名，是因为同一个文件被多条见证共用时，光有文件名定位不了是哪一条。
fn witness_source(prim: &Prim, witness: &Witness, dirs: &Dirs, params: &str) -> Result<String> {
    let text = crate::witness::source(witness, dirs)
        .map_err(|error| error.with(format!("`{prim}` 的")))?;
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
pub fn witness_files(
    selected: &[Selected<'_>],
    dirs: &Dirs,
    params: &str,
) -> Result<Sources> {
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
pub fn push_sources(program: &mut String, files: &Sources) {
    for text in files.values() {
        program.push_str(text);
        if !program.ends_with('\n') {
            program.push('\n');
        }
        program.push('\n');
    }
}

/// 让一段文本以换行结尾（空文本保持为空）：下一段可以直接接上。
pub fn normalize_newline(text: &mut String) {
    if !text.is_empty() && !text.ends_with('\n') {
        text.push('\n');
    }
}

/// 写一个文件，并把它的上层目录建出来：一次性构建目录都只有一层。
pub fn write_text(path: &Path, text: &str) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|error| {
            Error::new(Code::Io, format!("创建 {} 失败：{error}", parent.display()))
        })?;
    }
    std::fs::write(path, text)
        .map_err(|error| Error::new(Code::Io, format!("写 {} 失败：{error}", path.display())))
}

/// 跑一条编译器命令；非零退出码时把命令、完整输出与源文件路径一并带进诊断。
pub fn run_tool(name: &str, command: &mut Command, source: &Path) -> Result<()> {
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
pub fn run_exe(
    command: &mut Command,
    path: &Path,
    failure: &str,
) -> Result<(String, String)> {
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
/// 否则 [`crate::runner::parse_bits`] 会把计数当成值解出来。
pub fn readback(stdout: &str, stderr: &str) -> Readback {
    (parse_counts(stderr), crate::runner::parse_bits(stdout))
}

/// 一个新的一次性目录：`<work_dir>/<tag>-<pid>-<序号>`，调用方负责删除。pid 隔开进程、序号
/// 隔开同进程内的多次调用：同时跑的两个 `cutile` 不会互相覆盖对方的中间产物。
pub fn scratch(dirs: &Dirs, tag: &str) -> PathBuf {
    static SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    dirs.work().join(format!(
        "{tag}-{}-{}",
        std::process::id(),
        SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
    ))
}

/// 两个后端 kernel 共享的分派段：按 `tid` 的 `switch`、cap 截断与回写循环。`close` 是 kernel
/// 收尾文本（Metal 与 CUDA 的换行数不同）：`local` 声明留在各自函数里，这里只拼逐字相同的部分。
pub fn push_dispatch(
    program: &mut String,
    max: usize,
    selected: &[Selected<'_>],
    close: &str,
) {
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
