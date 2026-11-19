//! 编译、执行与回读原语：把源文本编成可执行文件、跑一遍、把 stdout 位模式与 stderr 计时样本解出来。
//!
//! 编译产物可以落进缓存（`crate::cache`）：命中时整步跳过，没有编译器也能跑已验证产物。这里不认识
//! 算子，也不生成程序文本——那是 `crate::harness` 与 `crate::device` 的职责。

use diagnostic::{Code, Error};
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

impl Timing {
    /// 最快的一次：达成率用它，避免被调度抖动与换页拖低。
    #[must_use]
    pub fn min_ns(&self) -> Option<f64> {
        self.samples_ns.iter().copied().reduce(f64::min)
    }

    /// 中位数（偶数个样本取中间两个的平均值）：与 [`Self::min_ns`] 的差距即抖动幅度。
    #[must_use]
    pub fn p50_ns(&self) -> Option<f64> {
        let sorted = self.sorted();
        if sorted.is_empty() {
            return None;
        }
        let middle = sorted.len() / 2;
        Some(if sorted.len().is_multiple_of(2) {
            (sorted[middle - 1] + sorted[middle]) / 2.0
        } else {
            sorted[middle]
        })
    }

    /// 95 分位（最近秩法）：尾部离群的上界；样本很少时它退化成最大值。达成率仍按 [`Self::p50_ns`]，
    /// 这里只把离群幅度摊开，不做静默剔除——剔除了哪几样本应当可见、可复核。
    #[must_use]
    pub fn p95_ns(&self) -> Option<f64> {
        let sorted = self.sorted();
        if sorted.is_empty() {
            return None;
        }
        let rank = ((0.95 * sorted.len() as f64).ceil() as usize).min(sorted.len());
        Some(sorted[rank - 1])
    }

    /// 升序样本：空表返回空，分位方法各自判空。
    fn sorted(&self) -> Vec<f64> {
        let mut sorted = self.samples_ns.clone();
        sorted.sort_by(f64::total_cmp);
        sorted
    }
}

/// 只编译：把源文本写到 `dir/kernel.c` 并编成 `dir/kernel`，返回可执行文件路径。与 [`run`] 分开
/// 是因为编译产物可以落进缓存：命中时整步跳过。
///
/// # Errors：目录不可建、源文件不可写、编译器不存在，或编译失败（诊断带命令、完整输出与路径）。
pub fn compile(
    tool: &str,
    flags: &[String],
    source: &str,
    dir: &Path,
) -> diagnostic::Result<PathBuf> {
    std::fs::create_dir_all(dir)
        .map_err(|error| Error::new(Code::Io, format!("创建 {} 失败：{error}", dir.display())))?;
    let src = dir.join("kernel.c");
    let exe = dir.join("kernel");
    std::fs::write(&src, source)
        .map_err(|error| Error::new(Code::Io, format!("写 {} 失败：{error}", src.display())))?;

    let compile = Command::new(tool)
        .args(flags)
        .arg("-o")
        .arg(&exe)
        .arg(&src)
        .output()
        .map_err(|error| Error::new(Code::Io, format!("调用 `{tool}` 失败：{error}")))?;
    if !compile.status.success() {
        // 编译失败是"声明/模板/片段里有一处写不对"，类别是 invalid；诊断带命令与完整输出。
        return Err(Error::new(
            Code::Invalid,
            format!(
                "`{tool} {}` 编译失败：\n{}\n{}\n源文件：{}",
                flags.join(" "),
                String::from_utf8_lossy(&compile.stdout),
                String::from_utf8_lossy(&compile.stderr),
                src.display()
            ),
        ));
    }
    require_executable(dir, &exe)
}

/// 编译成功但产物没落盘：把目录内容交给诊断，而不是让下游在复制阶段拿到一个不存在的裸路径。
fn require_executable(dir: &Path, base: &Path) -> diagnostic::Result<PathBuf> {
    let Some(exe) = compiled_executable(dir, base) else {
        return Err(Error::new(
            Code::Invalid,
            format!(
                "编译成功但产物未落盘到 {}：目录内找不到可执行文件",
                dir.display()
            ),
        ));
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
        .find(|path| path.is_file() && path.extension().map(|ext| ext != "c").unwrap_or(true))
}

/// 只执行：读 stdout 的位模式与 stderr 的计时样本。
///
/// # Errors：可执行文件不可运行（权限、架构不符）或退出码非零，`stderr` 原样带进诊断。
pub fn run(exe: &Path) -> diagnostic::Result<Outcome> {
    let run = Command::new(exe)
        .output()
        .map_err(|error| Error::new(Code::Io, format!("运行 {} 失败：{error}", exe.display())))?;
    let stderr = String::from_utf8_lossy(&run.stderr).to_string();
    if !run.status.success() {
        return Err(Error::new(
            Code::Internal,
            format!("内核运行失败（退出码 {:?}）：{stderr}", run.status.code()),
        ));
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_timed_run_is_its_own_sample() {
        let samples = parse_samples("noise\nns 120\nns 100\nns 140\n");
        assert_eq!(samples, vec![120.0, 100.0, 140.0]);
        assert!(parse_samples("").is_empty());
    }

    #[test]
    fn min_and_median_are_computed_from_all_samples() {
        let timing = Timing {
            samples_ns: vec![30.0, 10.0, 20.0, 40.0],
        };
        assert_eq!(timing.min_ns(), Some(10.0));
        assert_eq!(timing.p50_ns(), Some(25.0));
        assert_eq!(timing.p95_ns(), Some(40.0));
        assert_eq!(Timing::default().min_ns(), None);
        assert_eq!(Timing::default().p50_ns(), None);
        assert_eq!(Timing::default().p95_ns(), None);
    }

    #[test]
    fn bit_patterns_decode_back_to_floats() {
        let bits = f32::to_bits(1.5);
        assert_eq!(parse_bits(&format!("{bits}\n")), vec![1.5]);
    }

    #[test]
    fn bit_patterns_round_trip_for_random_bits() {
        let mut rng = testkit::Rng::new(0xB17);
        for _ in 0..64 {
            let bits = rng.below(u32::MAX);
            let decoded = parse_bits(&format!("{bits}\n"));
            // 按位比较：NaN 的位模式也要能往返，而 `f32 == f32` 对 NaN 不成立。
            assert_eq!(decoded[0].to_bits(), bits);
        }
    }

    #[test]
    fn quantiles_are_ordered_for_random_samples() {
        let mut rng = testkit::Rng::new(0xA17E);
        for _ in 0..32 {
            let count = rng.below(64) + 1;
            let samples: Vec<f64> = (0..count).map(|_| rng.below(1_000_000) as f64).collect();
            let timing = Timing {
                samples_ns: samples,
            };
            let min = timing.min_ns().expect("非空");
            let p50 = timing.p50_ns().expect("非空");
            let p95 = timing.p95_ns().expect("非空");
            assert!(min <= p50 && p50 <= p95, "分位必须有序：{min} {p50} {p95}");
        }
    }

    #[test]
    fn compiler_gets_the_flags_it_was_given() {
        // 真机用例：默认跳过，`LLM_TEST_DEVICE=cpu` 才真的编译一次。
        if testkit::device_or_skip(
            "cpu",
            "cutile: 未显式启用 CPU 真机编译用例（LLM_TEST_DEVICE=cpu）",
        ) {
            return;
        }
        let workdir = std::env::temp_dir().join(format!("cutile-runner-{}", std::process::id()));
        let source = "#include <stdio.h>\n\
int main(void) { printf(\"%u\\n\", (unsigned)FLAG_MARK); return 0; }\n";
        let exe = compile("cc", &["-DFLAG_MARK=7".to_string()], source, &workdir)
            .expect("带档案参数编译");
        let outcome = run(&exe).expect("运行");
        assert_eq!(outcome.bits, vec![f32::from_bits(7)]);
        let _ = std::fs::remove_dir_all(&workdir);
    }
}

pub mod driver {
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
    pub fn run_exe(command: &mut Command, path: &Path, failure: &str) -> Result<(String, String)> {
        let output = command.output().map_err(|error| {
            Error::new(Code::Io, format!("运行 {} 失败：{error}", path.display()))
        })?;
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

    /// 两个后端 kernel 共享的 witness 正文：`local` 清零、`switch` 分派、cap 截断与回写循环。
    /// `close` 是收尾文本（换行数 Metal/CUDA 不同）；`local` 声明留在各自函数里（Metal 加 `thread`）。
    pub fn push_dispatch(program: &mut String, max: usize, selected: &[Selected<'_>], close: &str) {
        program.push_str(&format!(
            "    for (int i = 0; i < {max}; ++i) local[i] = 0.0f;\n    int n = 0;\n    switch (tid) {{\n"
        ));
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
}
