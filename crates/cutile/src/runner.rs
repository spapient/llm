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
    let exe =
        compile("cc", &["-DFLAG_MARK=7".to_string()], source, &workdir).expect("带档案参数编译");
    let outcome = run(&exe).expect("运行");
    assert_eq!(outcome.bits, vec![f32::from_bits(7)]);
    let _ = std::fs::remove_dir_all(&workdir);
}
}

