//! 编译、执行与回读原语：把源文本编成可执行文件、跑一遍、把 stdout 位模式与 stderr 计时样本解出来。

use cutile::dirs::Dirs;
use cutile::prof::Profile;
use std::path::{Path, PathBuf};
use std::process::Command;

/// 一次执行的产物。
#[derive(Debug, Clone, PartialEq)]
pub struct Outcome {
    /// 内核输出（按位模式解码后的 f32）。
    pub bits: Vec<f32>,
    /// 计时样本；warmup 不计入，样本为空表示内核算不出耗时。
    pub timing: Timing,
    /// 编译器/运行期的 stderr。
    pub stderr: String,
}

/// 计时样本：每个样本是一次内核调用的墙钟耗时（纳秒），只在 Rust 侧算 `min`/`p50`（README §12.1）。
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Timing {
    /// 全部样本（纳秒，按执行顺序；warmup 已排除）。
    pub samples_ns: Vec<f64>,
}

/// 一组计时样本的完整摘要，三个读数由同一次排序派生。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TimingSummary {
    /// 最快样本。
    pub min_ns: f64,
    /// 中位数；偶数个样本取中间两个的平均值。
    pub p50_ns: f64,
    /// 95 分位，采用最近秩法。
    pub p95_ns: f64,
}

impl Timing {
    /// 最快的一次：只需线性扫描，不为单项读数分配内存。
    #[must_use]
    pub fn min_ns(&self) -> Option<f64> {
        self.samples_ns.iter().copied().reduce(f64::min)
    }

    /// 一次排序同时得到 min / p50 / p95，避免基准路径重复克隆并排序样本。
    #[must_use]
    pub fn summary(&self) -> Option<TimingSummary> {
        let mut sorted = self.samples_ns.clone();
        sorted.sort_by(f64::total_cmp);
        let min_ns = *sorted.first()?;
        let middle = sorted.len() / 2;
        let p50_ns = if sorted.len().is_multiple_of(2) {
            (sorted[middle - 1] + sorted[middle]) / 2.0
        } else {
            sorted[middle]
        };
        let rank = ((0.95 * sorted.len() as f64).ceil() as usize).min(sorted.len());
        Some(TimingSummary {
            min_ns,
            p50_ns,
            p95_ns: sorted[rank - 1],
        })
    }

    /// 中位数；同时需要其它分位时使用 [`Self::summary`]。
    #[must_use]
    pub fn p50_ns(&self) -> Option<f64> {
        self.summary().map(|summary| summary.p50_ns)
    }

    /// 95 分位；同时需要其它分位时使用 [`Self::summary`]。
    #[must_use]
    pub fn p95_ns(&self) -> Option<f64> {
        self.summary().map(|summary| summary.p95_ns)
    }
}

/// 生成编译参数：档案 `[compile].flags` 加上工具链头文件垫片的 `-I`。
/// 垫片按档案写入 `work.dir/include`，供算子、见证与基准共用。
///
/// # Errors
///
/// 垫片目录不可创建或文件不可写时返回 IO 说明。
pub fn compile_flags(profile: &Profile, dirs: &Dirs) -> diagnostic::Result<Vec<String>> {
    let mut flags = profile.flags.clone();
    if profile.headers.is_empty() {
        return Ok(flags);
    }
    let include = dirs.work().join("include");
    std::fs::create_dir_all(&include).map_err(|error| {
        reject!(
            Io,
            "shard::runner::compile_flags",
            "创建 {} 失败：{error}",
            include.display()
        )
    })?;
    for (name, text) in &profile.headers {
        let path = include.join(name);
        std::fs::write(&path, text).map_err(|error| {
            reject!(
                Io,
                "shard::runner::compile_flags",
                "写工具链垫片 {} 失败：{error}",
                path.display()
            )
        })?;
    }
    flags.push(format!("-I{}", include.display()));
    Ok(flags)
}

/// 把一份源文本交给 `tool` 编译成可执行文件，返回产物路径。
pub fn compile(
    tool: &str,
    flags: &[String],
    source: &str,
    dir: &Path,
) -> diagnostic::Result<PathBuf> {
    std::fs::create_dir_all(dir).map_err(|error| {
        reject!(
            Io,
            "shard::runner::compile",
            "创建 {} 失败：{error}",
            dir.display()
        )
    })?;
    let src = dir.join("kernel.c");
    let exe = dir.join("kernel");
    std::fs::write(&src, source).map_err(|error| {
        reject!(
            Io,
            "shard::runner::compile",
            "写 {} 失败：{error}",
            src.display()
        )
    })?;

    let mut compile = Command::new(tool);
    compile.args(flags).arg("-o").arg(&exe).arg(&src);
    driver::run_tool(&format!("{tool} {}", flags.join(" ")), &mut compile, &src)?;
    require_executable(dir, &exe)
}

/// 编译成功但产物没落盘：把目录内容交给诊断，而不是让下游在复制阶段拿到一个不存在的裸路径。
fn require_executable(dir: &Path, base: &Path) -> diagnostic::Result<PathBuf> {
    let Some(exe) = compiled_executable(dir, base) else {
        fail!(
            Invalid,
            "shard::runner::require_executable",
            "编译成功但产物未落盘到 {}：目录内找不到可执行文件",
            dir.display()
        );
    };
    Ok(exe)
}

/// 编译器把可执行文件写成什么名字由驱动决定：Windows 上驱动常自动补 `.exe`，`-o` 给的裸名
/// 并不会落盘。编译成功后以磁盘上真实存在的那份为准回传，避免下游复制/执行拿到不存在的裸路径。
fn compiled_executable(dir: &Path, base: &Path) -> Option<PathBuf> {
    if base.exists() {
        return Some(base.to_path_buf());
    }
    let with_extension = base.with_extension("exe");
    if with_extension.exists() {
        return Some(with_extension);
    }
    // 驱动另起了默认名（如 `a.exe`）：目录里除源文本外的第一个文件就是本次产物。
    let entries = std::fs::read_dir(dir).ok()?;
    entries
        .flatten()
        .map(|entry| entry.path())
        .find(|path| path.is_file() && path.extension().is_none_or(|ext| ext != "c"))
}

/// 只执行：读 stdout 的位模式与 stderr 的计时样本。
///
/// # Errors：可执行文件不可运行（权限、架构不符）或退出码非零，`stderr` 原样带进诊断。
pub fn run(exe: &Path) -> diagnostic::Result<Outcome> {
    let run = Command::new(exe).output().map_err(|error| {
        reject!(
            Io,
            "shard::runner::run",
            "运行 {} 失败：{error}",
            exe.display()
        )
    })?;
    let stderr = String::from_utf8_lossy(&run.stderr).to_string();
    if !run.status.success() {
        fail!(
            Internal,
            "shard::runner::run",
            "内核运行失败（退出码 {:?}）：{stderr}",
            run.status.code()
        );
    }
    Ok(Outcome {
        bits: parse_bits(&String::from_utf8_lossy(&run.stdout)),
        timing: Timing {
            samples_ns: parse_samples(&stderr),
        },
        stderr,
    })
}

/// 把 `printf("%u\n", …)` 的输出解码回 f32。
pub fn parse_bits(stdout: &str) -> Vec<f32> {
    stdout
        .lines()
        .filter_map(|line| line.trim().parse::<u32>().ok().map(f32::from_bits))
        .collect()
}

/// 解析设备/host 驱动在 stderr 上输出的纳秒样本。
pub fn parse_timing(stderr: &str) -> Timing {
    Timing {
        samples_ns: parse_samples(stderr),
    }
}

/// 每行 `ns <value>` 是一个计时样本；缺行就是没有样本（由调用方决定是否算失败）。
fn parse_samples(stderr: &str) -> Vec<f64> {
    stderr
        .lines()
        .filter_map(|line| {
            line.trim()
                .strip_prefix("ns ")
                .and_then(|value| value.trim().parse::<f64>().ok())
        })
        .collect()
}

// ---- 工具链探测与实测身份 ----

use std::fmt;
use std::sync::OnceLock;

#[cfg(test)]
use testkit;

/// 一条被真正解析出来的工具链。产物身份与报告用它，而不是档案里的登记版本。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Toolchain {
    /// 命令名（`cc` / `clang` / `gcc` / `nvcc` / `xcrun metal`）。
    pub command: String,
    /// `--version` 里解析出的第一段版本号；解析不出时为 `None`（不猜）。
    pub version: Option<String>,
    /// `--version` 的首个非空行（原文，诊断与台账里要能对得上）。
    pub banner: String,
}

impl Toolchain {
    /// 身份：`命令@实测版本`，进产物身份与报告；解析不出版本时用 `-`，不假装知道版本。
    #[must_use]
    pub fn id(&self) -> String {
        format!(
            "{}@{}",
            self.command,
            self.version.as_deref().unwrap_or("-")
        )
    }

    /// 档案登记的版本是否与实测相容：只比**主版本号**（`clang-17` 的族名不算差异），不改判定等级。
    #[must_use]
    pub fn agrees_with(&self, declared: &str) -> bool {
        if declared.trim() == "*" {
            return true;
        }
        let Some(actual) = self.version.as_deref() else {
            return false;
        };
        let actual_major = actual.split('.').next().unwrap_or(actual);
        // 登记值可能带工具族前缀（`clang-17`）：只看最后一段。
        let declared = declared.trim().rsplit(['-', ' ', '_']).next().unwrap_or("");
        let declared_major = declared.split('.').next().unwrap_or("");
        !declared_major.is_empty() && declared_major == actual_major
    }
}

impl fmt::Display for Toolchain {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.version {
            Some(version) => write!(formatter, "{} {version}", self.command),
            None => write!(formatter, "{}（版本未知：{}）", self.command, self.banner),
        }
    }
}

/// 从 `--version` 的输出里取第一段版本号：先看含 `version` / `release` 的行，再退回任意行；每行
/// 取第一段带点的"数字与点"——`-O2`、`sm_100` 里的孤立数字不是版本号。
pub fn parse_version(text: &str) -> Option<String> {
    let lines: Vec<&str> = text
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .collect();
    let keyed = lines.iter().copied().filter(|line| {
        let lower = line.to_ascii_lowercase();
        lower.contains("version") || lower.contains("release")
    });
    keyed.chain(lines.iter().copied()).find_map(|line| {
        line.split(|ch: char| !(ch.is_ascii_digit() || ch == '.'))
            .map(|token| token.trim_end_matches('.'))
            .find(|token| token.contains('.'))
            .map(str::to_string)
    })
}

/// 进程内缓存的探测结果：同一台机器的工具链不会变，反复 `--version` 起子进程是纯浪费。
static C_TOOLCHAIN: OnceLock<Option<Toolchain>> = OnceLock::new();

/// 探测可用的 C 编译器（`cc` → `clang` → `gcc`）并读出它自报的版本；结果当作进程级快照缓存。
pub fn detect() -> Option<Toolchain> {
    C_TOOLCHAIN
        .get_or_init(|| ["cc", "clang", "gcc"].into_iter().find_map(probe))
        .clone()
}

/// 按**档案声明的工具**探测：除便携 C 之外，产物只能由档案里写的那个编译器编（`nvcc` / `mcc` /
/// `ccec` / `xcrun metal`），宿主的 `cc` 不认识 `--cuda-gpu-algorithm`，拿它去编只会把"这台设备没有
/// 工具链"报成一条 C 语法错误。声明不可用时返回 `None`（如实"没有"），不退回合主编译器——那会把
/// 缺工具链伪装成编译失败。
#[must_use]
pub fn for_profile(profile: &cutile::prof::Profile) -> Option<Toolchain> {
    match profile.meta.emit {
        // 便携 C 的登记值是 `cc` 且版本记作 `*`：探测顺序（cc → clang → gcc）本身就是它的语义。
        cutile::prof::Emit::C => detect(),
        _ => probe_declared(&profile.meta.tool),
    }
}

/// 按 `[meta].tools` 的**声明**探测：声明可以带固定前缀参数（`xcrun metal`），这仍是档案写的那条
/// 命令，不是退回合主编译器。命令名与参数以空白分隔，身份里保留声明原文，报告才能与档案对账。
#[must_use]
pub fn probe_declared(declared: &str) -> Option<Toolchain> {
    let mut parts = declared.split_whitespace();
    let program = parts.next()?;
    let leading: Vec<&str> = parts.collect();
    let mut tool = probe_with(program, &leading)?;
    tool.command = declared.trim().to_string();
    Some(tool)
}

/// 本机的 clang：先问 `clang`，再退到 `cc`（macOS 与多数发行版的 `cc` 就是 clang）。判据是
/// "这个程序是不是 clang"：`-x cuda` / `-x cl` 是 clang 的前端，gcc 没有。
#[must_use]
pub fn clang() -> Option<Toolchain> {
    probe("clang").or_else(|| detect().filter(|toolchain| toolchain.banner.contains("clang")))
}

/// 探测指定命令：跑一次 `--version`，成功才返回（版本解析不出也算可用）。
pub fn probe(command: &str) -> Option<Toolchain> {
    probe_with(command, &[])
}

/// 探测的实作：`program` 之后先放档案声明的固定前缀参数，再问版本。
pub(crate) fn probe_with(program: &str, leading: &[&str]) -> Option<Toolchain> {
    let output = Command::new(program)
        .args(leading)
        .arg("--version")
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout).to_string();
    let text = if stdout.trim().is_empty() {
        String::from_utf8_lossy(&output.stderr).to_string()
    } else {
        stdout
    };
    let banner = text
        .lines()
        .map(str::trim)
        .find(|line| !line.is_empty())?
        .to_string();
    Some(Toolchain {
        command: program.to_string(),
        version: parse_version(&text),
        banner,
    })
}

/// 测试门控：没有主机 C 编译器时，编译/执行类用例在 CI 上不该伪装成通过；返回 `true` 表示跳过，
/// 原因经 [`testkit::skip`] 登记计数（标准测试框架没有正式 skip 信号）。
#[cfg(test)]
pub(crate) fn skip_without_host_compiler() -> bool {
    if detect().is_none() {
        testkit::skip("cutile: 未探测到主机 C 编译器（cc/clang/gcc），编译/执行用例未运行")
    } else {
        false
    }
}

#[cfg(test)]
#[path = "../tests/runner_tests.rs"]
mod tests;

pub mod driver {
    //! 编译 / 执行 / 回读的共享机制原语：装载见证源、拼程序文本、起子进程、解码回读。

    use crate::backend::Readback;
    use crate::witness::Selected;
    use cutile::atom::{Prim, Witness};
    use cutile::dirs::Dirs;
    use diagnostic::Result;
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
            fail!(
                Invalid,
                "shard::driver::witness_source",
                "`{prim}` 的见证 `{}` 里没有函数 `{}({params})`",
                witness.file,
                witness.func
            );
        }
        Ok(text)
    }

    /// 本次执行的见证源码：按文件去重，同一文件里的多条见证只出现一次。
    pub fn witness_files(selected: &[Selected<'_>], dirs: &Dirs, params: &str) -> Result<Sources> {
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
                reject!(
                    Io,
                    "shard::driver::write_text",
                    "创建 {} 失败：{error}",
                    parent.display()
                )
            })?;
        }
        std::fs::write(path, text).map_err(|error| {
            reject!(
                Io,
                "shard::driver::write_text",
                "写 {} 失败：{error}",
                path.display()
            )
        })
    }

    /// 跑一条编译器命令；非零退出码时把命令、完整输出与源文件路径一并带进诊断。
    pub fn run_tool(name: &str, command: &mut Command, source: &Path) -> Result<()> {
        let output = command.output().map_err(|error| {
            reject!(Io, "shard::driver::run_tool", "调用 `{name}` 失败：{error}")
        })?;
        if output.status.success() {
            return Ok(());
        }
        fail!(
            Invalid,
            "shard::driver::run_tool",
            "`{name}` 编译失败：\n{}\n{}\n源文件：{}",
            String::from_utf8_lossy(&output.stdout).trim(),
            String::from_utf8_lossy(&output.stderr).trim(),
            source.display()
        )
    }

    /// 跑一个已经编好的可执行文件，返回 `(stdout, stderr)` 原文。非零退出码报 [`diagnostic::Code::Unsupported`]：编得过却跑不起来说明这台机器现在跑不了，不是见证写错了。
    pub fn run_exe(command: &mut Command, path: &Path, failure: &str) -> Result<(String, String)> {
        let output = command.output().map_err(|error| {
            reject!(
                Io,
                "shard::driver::run_exe",
                "运行 {} 失败：{error}",
                path.display()
            )
        })?;
        let stderr = String::from_utf8_lossy(&output.stderr).to_string();
        if !output.status.success() {
            fail!(
                Unsupported,
                "shard::driver::run_exe",
                "{failure}（退出码 {:?}）：{}",
                output.status.code(),
                stderr.trim()
            );
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
    /// 否则 `crate::runner::parse_bits` 会把计数当成值解出来。
    pub fn readback(stdout: &str, stderr: &str) -> Readback {
        (parse_counts(stderr), super::parse_bits(stdout))
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

    /// 在唯一临时目录中执行动作，成功或失败都清理中间产物。
    pub fn with_scratch<T>(
        dirs: &Dirs,
        tag: &str,
        action: impl FnOnce(&Path) -> Result<T>,
    ) -> Result<T> {
        let dir = scratch(dirs, tag);
        let outcome = action(&dir);
        let _ = std::fs::remove_dir_all(&dir);
        outcome
    }

    /// 两个后端 kernel 共享的 witness 正文：`local` 清零、`switch` 分派、cap 截断与回写循环。
    /// `close` 是收尾文本（换行数 Metal/CUDA 不同）；`local` 声明留在各自函数里（Metal 加 `thread`）。
    pub fn push_dispatch(program: &mut String, max: usize, selected: &[Selected<'_>], close: &str) {
        program.push_str(&format!(
            "    for (int i = 0; i < {max}; ++i) local[i] = 0.0f;\n    int n = 0;\n    switch (tid) {{\n"
        ));
        for (idx, (_, _, witness)) in selected.iter().enumerate() {
            program.push_str(&format!(
                "        case {idx}: n = {}(local); break;\n",
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
}
