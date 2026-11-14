//! ⑤ 编译与执行：`c` 后端通过子进程调用 `cc`，不涉及 FFI，也不需要 `unsafe`。
//!
//! 只有"能跑出数"才算 L2 证据（README §11.2）：编译、运行、把 f32 结果位模式读回来交给
//! [`crate::reference`] 对拍；编译参数由调用方传入档案 `[compile].flags`，与缓存指纹哈希的
//! 是同一份，代码里不写死优化级别。样本由 `op::model` 给出、实参顺序由 `op::sig` 给出：本模块
//! 只按名字与类型把缓冲渲染成 C 数组，不认识任何算子。

use diagnostic::{Code, Error};
use op::model::{Sample, Values};
use op::sig::{CType, Signature};
use std::fmt;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::OnceLock;

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

/// 把样本与内核源码拼成一个可执行的 `main()`：warmup 若干次后逐次计时、打印输出位模式。
///
/// 计量口径见 README §12.1：墙钟取样、warmup 不计入，批量校准在 C 侧、`min`/`p50` 在 Rust 侧。
///
/// # Errors：签名没有可写输出、样本缺输出缓冲、输出不是 f32，或签名与样本不一致。
pub fn harness(
    kernel_source: &str,
    signature: &Signature,
    sample: &Sample,
    repeat: usize,
) -> diagnostic::Result<String> {
    let (output, values) = output_of(signature, sample)?;
    let kernel = signature.kernel();
    let warmup = warmup_runs(repeat);
    let args = signature.call_args(sample)?;
    let mut text = prelude(kernel_source, &kernel, sample, repeat, warmup);
    let (decls, init) = input_buffers(sample, output);
    text.push_str(&decls);
    text.push_str(&main(
        output,
        values.len(),
        &kernel,
        &args,
        &init,
        repeat,
        warmup,
    ));
    Ok(text)
}

/// 可写输出与它的样本值：回读按 `float` 位模式解释，所以只认 `float *`；签名声明成别的类型时
/// C 编译器只会报一句 `incompatible pointer type`，不带算子名与缓冲名，故在生成程序之前先说清。
pub(crate) fn output_of<'a>(
    signature: &'a Signature,
    sample: &'a Sample,
) -> diagnostic::Result<(&'a str, &'a [f32])> {
    let output = output_name(signature)?;
    let out = sample
        .buffer(output)
        .ok_or_else(|| Error::new(Code::Missing, format!("样本里没有输出缓冲 `{output}`")))?;
    let Values::F32(values) = &out.values else {
        return Err(Error::new(
            Code::Invalid,
            format!("输出缓冲 `{output}` 必须是 f32"),
        ));
    };
    Ok((output, values))
}

/// 可写输出名：装载期已校验，这里只能是实现漂移。
fn output_name(signature: &Signature) -> diagnostic::Result<&str> {
    let output = signature.output().ok_or_else(|| {
        Error::new(
            Code::Internal,
            "内核签名没有可写输出缓冲（装载期已经校验过，这里只能是实现漂移）",
        )
    })?;
    if signature.output_type() != Some(CType::F32) {
        return Err(Error::new(
            Code::Unsupported,
            format!(
                "算子 `{}` 的可写输出 `{output}` 在 `.k` 里不是 `float *`（是 `{}`）；回读按 float 位模式解释，只认 `float`",
                signature.op(),
                signature.output_type().map_or("无", CType::c_name)
            ),
        ));
    }
    Ok(output)
}

/// 文件头：POSIX 声明、身份注释、内核源码与计时运行时。
fn prelude(
    kernel_source: &str,
    kernel: &str,
    sample: &Sample,
    repeat: usize,
    warmup: usize,
) -> String {
    let mut text = String::new();
    // `clock_gettime` 在 `-std=c99` 下需要先声明 POSIX 版本，且必须在任何系统头之前。
    text.push_str("#define _POSIX_C_SOURCE 200809L\n");
    text.push_str(&format!(
        "/* cutile harness | kernel={kernel} m={} n={} repeat={repeat} warmup={warmup} */\n",
        sample.m, sample.n
    ));
    text.push_str(kernel_source);
    if !text.ends_with('\n') {
        text.push('\n');
    }
    text.push_str(RUNTIME);
    text
}

/// 计时运行时：`CLOCK_MONOTONIC` 取样、编译器屏障、`f32` 位模式回读。
const RUNTIME: &str = "\n#include <stdio.h>\n#include <time.h>\n\n\
static double cutile_now_ns(void) {\n\
    struct timespec ts;\n\
    clock_gettime(CLOCK_MONOTONIC, &ts);\n\
    return (double)ts.tv_sec * 1e9 + (double)ts.tv_nsec;\n\
}\n\n\
/* 编译器屏障：没有它，优化器可以把“输入相同”的重复调用提到循环外或整段删掉，\n\
 * 量到的是空循环。clang/gcc 都支持；其他编译器退化为空操作。 */\n\
#if defined(__GNUC__) || defined(__clang__)\n\
#define CUTILE_BARRIER() __asm__ __volatile__(\"\" ::: \"memory\")\n\
#else\n\
#define CUTILE_BARRIER() ((void)0)\n\
#endif\n\n";

/// 输入缓冲的两段文本：声明，以及"初值 → 缓冲"的拷贝循环。初值先落在 `volatile const` 里再拷入，
/// 故 `-O3` 折不进内核（32x1 会被折成空循环）。
fn input_buffers(sample: &Sample, output: &str) -> (String, String) {
    let mut decls = String::new();
    let mut init = String::new();
    for buffer in &sample.buffers {
        if buffer.name == output {
            continue;
        }
        let ctype = c_type(&buffer.values);
        let size = buffer.values.len();
        decls.push_str(&format!(
            "static volatile const {ctype} cutile_init_{name}[{size}] = {{{literal}}};\n\
             static {ctype} {name}[{size}];\n",
            name = buffer.name,
            literal = c_literal(&buffer.values)
        ));
        init.push_str(&format!(
            "    for (int cutile_i = 0; cutile_i < {size}; ++cutile_i) \
             {{ {name}[cutile_i] = cutile_init_{name}[cutile_i]; }}\n",
            name = buffer.name
        ));
    }
    (decls, init)
}

/// 主函数：warmup、校准批量、逐样本计时，最后按位模式回读输出。`CUTILE_BARRIER()` 挡"提到循环外"，
/// `volatile` 出口读挡"只留最后一次写"：缺一，32x1 就退化成空循环。
fn main(
    output: &str,
    len: usize,
    kernel: &str,
    call_args: &str,
    init: &str,
    repeat: usize,
    warmup: usize,
) -> String {
    // 出口保活整个输出数组：只读 `y[0]` 时其余写入可能被优化掉，故逐元素写进 `volatile` sink。
    let sink = if len > 0 {
        format!(
            "        for (int cutile_s = 0; cutile_s < {len}; ++cutile_s) {{ cutile_sink = {output}[cutile_s]; }}\n"
        )
    } else {
        String::new()
    };
    format!(
        "int main(void) {{
    float {output}[{len}];
    volatile float cutile_sink = 0.0f;
{init}    /* warmup：代码、页表与缓存进稳态，不计入样本 */
    for (int cutile_warm = 0; cutile_warm < {warmup}; ++cutile_warm) {{
        for (int cutile_i = 0; cutile_i < {len}; ++cutile_i) {{ {output}[cutile_i] = 0.0f; }}
        {kernel}({call_args});
        CUTILE_BARRIER();
{sink}    }}
    /* 校准：16 次调用取最短单次，定批量，让亚微秒内核也能跨过时钟分辨率；校准本身不算样本 */
    double cutile_calib = 1e18;
    for (int cutile_c = 0; cutile_c < 16; ++cutile_c) {{
        double cutile_c0 = cutile_now_ns();
        {kernel}({call_args});
        CUTILE_BARRIER();
        double cutile_c1 = cutile_now_ns();
        double cutile_cd = cutile_c1 - cutile_c0;
        if (cutile_cd < cutile_calib) {{ cutile_calib = cutile_cd; }}
    }}
    int cutile_batch = (int)(1e6 / (cutile_calib > 1.0 ? cutile_calib : 1.0));
    if (cutile_batch < 1) {{ cutile_batch = 1; }}
    if (cutile_batch > 1024) {{ cutile_batch = 1024; }}
    for (int cutile_r = 0; cutile_r < {repeat}; ++cutile_r) {{
        for (int cutile_i = 0; cutile_i < {len}; ++cutile_i) {{ {output}[cutile_i] = 0.0f; }}
        double cutile_t0 = cutile_now_ns();
        for (int cutile_b = 0; cutile_b < cutile_batch; ++cutile_b) {{
            {kernel}({call_args});
            CUTILE_BARRIER();
{sink}        }}
        double cutile_t1 = cutile_now_ns();
        fprintf(stderr, \"ns %.3f\\n\", (cutile_t1 - cutile_t0) / cutile_batch);
    }}
    for (int cutile_i = 0; cutile_i < {len}; ++cutile_i) {{
        union {{ float f; unsigned int u; }} cutile_u;
        cutile_u.f = {output}[cutile_i];
        printf(\"%u\\n\", cutile_u.u);
    }}
    return 0;
}}
"
    )
}
/// warmup 次数：至少一次，约为测量次数的 1/10。
fn warmup_runs(repeat: usize) -> usize {
    (repeat / 10).max(1)
}

/// CUDA 算子封装生成器：实现见 `crate::cuda`（私有模块，只在文档里指名）；保留 `backend::cuda_harness` 这条既有调用路径。
pub use crate::cuda::cuda_harness;

/// 缓冲的 C 元素类型。
pub(crate) fn c_type(values: &Values) -> &'static str {
    match values {
        Values::F32(_) => "float",
        Values::U8(_) => "unsigned char",
    }
}

/// 缓冲的 C 静态数组初始化式。
pub(crate) fn c_literal(values: &Values) -> String {
    match values {
        Values::F32(values) => values
            .iter()
            .map(|value| format!("{value:?}f"))
            .collect::<Vec<_>>()
            .join(", "),
        Values::U8(values) => values
            .iter()
            .map(|value| format!("0x{value:02x}u"))
            .collect::<Vec<_>>()
            .join(", "),
    }
}

/// 一条被真正解析出来的工具链。档案里的 `[meta].tool_ver` 是人写的登记值，代码生成的差异却由
/// **真正调用的那个编译器**决定：产物身份与报告一律用实测值。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Toolchain {
    /// 命令名（`cc` / `clang` / `gcc`）。
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
static NVCC_TOOLCHAIN: OnceLock<Option<Toolchain>> = OnceLock::new();

/// 探测可用的 C 编译器（`cc` → `clang` → `gcc`）并读出它自报的版本；结果当作进程级快照缓存，
/// 进程启动后新装的编译器要重启才看得见。
pub fn detect() -> Option<Toolchain> {
    C_TOOLCHAIN
        .get_or_init(|| ["cc", "clang", "gcc"].into_iter().find_map(probe))
        .clone()
}

/// 探测 nvcc（CUDA 编译器）：`nvcc --version` 只打印版本、不调用宿主 cl，故可独立成功；宿主编译器由
/// 调用方用 `-ccbin` 另行指定。
#[must_use]
pub fn detect_nvcc() -> Option<Toolchain> {
    NVCC_TOOLCHAIN.get_or_init(|| probe("nvcc")).clone()
}

/// 测试门控：没有主机 C 编译器时，编译/执行类用例在 CI 上不该伪装成通过；返回 `true` 表示跳过，
/// 原因写进 stderr（标准测试框架没有正式 skip 信号，这里只让"空转通过"在输出里可见）。
#[cfg(test)]
pub(crate) fn skip_without_host_compiler() -> bool {
    if detect().is_none() {
        eprintln!("skip: 未探测到主机 C 编译器（cc/clang/gcc），本用例未运行");
        true
    } else {
        false
    }
}

/// 探测指定命令：跑一次 `--version`，成功才返回（版本解析不出也算可用）。
pub fn probe(command: &str) -> Option<Toolchain> {
    let output = Command::new(command).arg("--version").output().ok()?;
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
        command: command.to_string(),
        version: parse_version(&text),
        banner,
    })
}

/// 只编译：把源文本写到 `dir/kernel.c` 并编成 `dir/kernel`，返回可执行文件路径。与 [`run`] 分开
/// 是因为编译产物可以落进缓存（`cutile::cache`）：命中时整步跳过，没有编译器也能跑已验证产物。
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
        // 编译失败是“声明/模板/片段里有一处写不对”，类别是 invalid；诊断带命令与完整输出。
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
    Ok(exe)
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

/// 编译并运行；`flags` 是档案的 `[compile].flags`，`dir` 是这次编译的沙箱目录。
pub fn compile_and_run(
    tool: &str,
    flags: &[String],
    source: &str,
    dir: &Path,
) -> diagnostic::Result<Outcome> {
    run(&compile(tool, flags, source, dir)?)
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
    use op::model::Buffer;

    fn signature() -> Signature {
        Signature::parse(
            "x",
            "void cutile_x_kernel(const float *x, const unsigned char *w, const float *scales,\n\
             float *y, int M, int N, int gsz) {}\n",
        )
        .expect("签名")
    }

    fn sample() -> Sample {
        let buffers = [
            ("x", Values::F32(vec![1.0, 1.0, 1.0, 1.0])),
            ("w", Values::U8(vec![0x98, 0x76])),
            ("scales", Values::F32(vec![0.5, 0.5])),
            ("y", Values::F32(vec![0.0, 0.0])),
        ];
        Sample {
            m: 4,
            n: 2,
            scalars: vec![("M", 4), ("N", 2), ("gsz", 4)],
            buffers: buffers
                .into_iter()
                .map(|(name, values)| Buffer { name, values })
                .collect(),
        }
    }

    /// 生成 C/CUDA 封装文本；`CUTILE_CAPTURE_GOLDEN` 显式打开时才把副本写到 `/tmp/cutile-golden`。
    #[test]
    fn golden_capture() {
        let capture = std::env::var("CUTILE_CAPTURE_GOLDEN").is_ok();
        let dir = std::path::PathBuf::from("/tmp/cutile-golden");
        for repeat in [1usize, 3, 50] {
            let text = harness(
                "void cutile_x_kernel(void) {}\n",
                &signature(),
                &sample(),
                repeat,
            )
            .expect("可生成");
            assert!(!text.is_empty(), "封装文本不能为空");
            assert!(text.contains("cutile_x_kernel("), "封装里要调用内核");
            if capture {
                std::fs::create_dir_all(&dir).expect("目录");
                std::fs::write(dir.join(format!("c_harness_{repeat}.txt")), text).expect("写");
            }
        }
        let kernel = "void cutile_x_kernel(const float *x, const unsigned char *w, const float *scales,\n\
             float *y, int M, int N, int gsz) {\n}\n";
        let text = cuda_harness(kernel, &signature(), &sample()).expect("可生成");
        assert!(!text.is_empty(), "CUDA 封装文本不能为空");
        if capture {
            std::fs::create_dir_all(&dir).expect("目录");
            std::fs::write(dir.join("cuda_harness.txt"), text).expect("写");
        }
        let text = harness(kernel, &signature(), &sample(), 3).expect("可生成");
        assert!(
            text.contains("cutile_init_"),
            "输入初值要内联进封装，否则常量输入会被折叠"
        );
        if capture {
            std::fs::create_dir_all(&dir).expect("目录");
            std::fs::write(dir.join("c_harness_kernel.txt"), text).expect("写");
        }
    }

    #[test]
    fn harness_embeds_data_and_calls_the_kernel() {
        let source = harness(
            "void cutile_x_kernel(void) {}\n",
            &signature(),
            &sample(),
            3,
        )
        .expect("可生成");
        assert!(source.contains("static volatile const float cutile_init_x[4]"));
        assert!(source.contains("static volatile const unsigned char cutile_init_w[2]"));
        assert!(source.contains("static volatile const float cutile_init_scales[2]"));
        assert!(source.contains("x[cutile_i] = cutile_init_x[cutile_i];"));
        assert!(source.contains("#define CUTILE_BARRIER()"));
        assert!(source.contains("CUTILE_BARRIER();"));
        assert!(source.contains("volatile float cutile_sink"));
        assert!(
            source.contains("cutile_sink = y[cutile_s];"),
            "整个输出都要保活进 sink"
        );
        assert!(source.contains("float y[2]"));
        assert!(
            !source.contains("static volatile const float cutile_init_y"),
            "输出不能是只读"
        );
        assert!(source.contains("cutile_x_kernel(x, w, scales, y, 4, 2, 4);"));
        assert!(source.contains("for (int cutile_r = 0; cutile_r < 3; ++cutile_r)"));
        assert!(source.contains("clock_gettime(CLOCK_MONOTONIC"));
        assert!(
            source.contains("for (int cutile_warm = 0; cutile_warm < 1; ++cutile_warm)"),
            "warmup 至少一次，且不能与缓冲名撞车"
        );
        assert!(
            source.contains("int cutile_batch = (int)(1e6 /"),
            "小内核要批量计时，否则样本会被时钟分辨率抹平"
        );
        assert!(source.contains("fprintf(stderr, \"ns %.3f\\n\""));
        assert!(
            source.starts_with("#define _POSIX_C_SOURCE"),
            "POSIX 声明必须在系统头之前：{source}"
        );
    }

    #[test]
    fn warmup_is_a_tenth_of_the_measurement_and_never_zero() {
        assert_eq!(warmup_runs(1), 1);
        assert_eq!(warmup_runs(9), 1);
        assert_eq!(warmup_runs(50), 5);
    }

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
    fn harness_rejects_a_missing_or_mistyped_output() {
        let mut sample = sample();
        sample.buffers.retain(|buffer| buffer.name != "y");
        let error = harness("", &signature(), &sample, 1).unwrap_err();
        assert_eq!(error.code(), Code::Missing);
        assert!(error.to_string().contains("输出缓冲"), "{error}");

        let mut sample = self::sample();
        sample.buffers.retain(|buffer| buffer.name != "y");
        sample.buffers.push(Buffer {
            name: "y",
            values: Values::U8(vec![0]),
        });
        let error = harness("", &signature(), &sample, 1).unwrap_err();
        assert_eq!(error.code(), Code::Invalid);
        assert!(error.to_string().contains("必须是 f32"), "{error}");
    }

    /// 回读只认 `float`：可写指针不是 `float *` 时要在拼 C 之前报，否则只有编译器的一句 `incompatible pointer type`。
    #[test]
    fn a_non_float_output_is_rejected_before_any_c_is_generated() {
        let signature = Signature::parse(
            "x",
            "void cutile_x_kernel(const float *x, const unsigned char *w, const float *scales,\n\
             unsigned char *y, int M, int N, int gsz) {}\n",
        )
        .expect("u8 是闭集内的缓冲类型");
        let error = harness("", &signature, &sample(), 1).unwrap_err();
        assert_eq!(error.code(), Code::Unsupported);
        let text = error.to_string();
        assert!(text.contains("`x`"), "算子名要在诊断里：{text}");
        assert!(text.contains("`y`"), "缓冲名要在诊断里：{text}");
        assert!(
            text.contains("unsigned char"),
            "错误要说清 `.k` 实际写的是什么：{text}"
        );
        assert!(text.contains("float"), "{text}");
    }

    #[test]
    fn floats_round_trip_through_the_literal_format() {
        let value = 0.1f32;
        let literal = format!("{value:?}");
        let parsed: f64 = literal.parse().expect("可解析");
        assert_eq!(parsed as f32, value);
    }

    #[test]
    fn bit_patterns_decode_back_to_floats() {
        let bits = f32::to_bits(1.5);
        assert_eq!(parse_bits(&format!("{bits}\n")), vec![1.5]);
    }

    #[test]
    fn missing_compiler_is_not_an_error_by_itself() {
        // 只是探测：返回 None 也算合法结果，调用方负责报"缺设备即失败"。
        let _ = detect();
    }

    #[test]
    fn versions_are_read_from_what_the_compilers_actually_print() {
        // 三家真实输出：找的是同一段东西，而不是各自的字段名。
        assert_eq!(
            parse_version(
                "Apple clang version 21.0.0 (clang-2100.4.1.1)\nTarget: arm64-apple-darwin"
            ),
            Some("21.0.0".to_string())
        );
        assert_eq!(
            parse_version("gcc (Ubuntu 13.2.0-23ubuntu4) 13.2.0\nCopyright (C) 2023"),
            Some("13.2.0".to_string())
        );
        assert_eq!(
            parse_version(
                "nvcc: NVIDIA (R) Cuda compiler driver\nCuda compilation tools, release 13.3, V13.3.44"
            ),
            Some("13.3".to_string())
        );
        // 没有版本串时不猜：`-` 会出现在身份里，报告同时打印原文。
        assert_eq!(parse_version("cc (some vendor toolchain)\n"), None);
        // 孤立数字不是版本号（`-O2`、`sm_100` 常见于参数与设备名）。
        assert_eq!(parse_version("flags: -O2 sm_100"), None);
    }

    #[test]
    fn the_identity_carries_the_measured_version() {
        let tool = Toolchain {
            command: "cc".to_string(),
            version: Some("21.0.0".to_string()),
            banner: "Apple clang version 21.0.0".to_string(),
        };
        assert_eq!(tool.id(), "cc@21.0.0");
        assert!(tool.agrees_with("clang-21"), "登记值带族名也算一致");
        assert!(tool.agrees_with("21.4"), "只比主版本号");
        assert!(!tool.agrees_with("clang-17"), "换主版本就是另一条工具链");
        assert!(!tool.agrees_with(""), "空登记不能算一致");

        let unknown = Toolchain {
            command: "cc".to_string(),
            version: None,
            banner: "cc (some vendor toolchain)".to_string(),
        };
        assert_eq!(unknown.id(), "cc@-");
        assert!(!unknown.agrees_with("clang-21"), "不知道版本就不能声称一致");
    }

    #[cfg(unix)]
    #[test]
    fn a_probe_reads_the_version_of_the_command_it_was_given() {
        use std::os::unix::fs::PermissionsExt;
        let root = std::env::temp_dir().join(format!("cutile-probe-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).expect("临时目录");
        let fake = root.join("fake-cc.sh");
        std::fs::write(&fake, "#!/bin/sh\necho 'Fake clang version 99.1.2'\n").expect("写脚本");
        let mut perms = std::fs::metadata(&fake).expect("脚本").permissions();
        perms.set_mode(0o755);
        std::fs::set_permissions(&fake, perms).expect("可执行");
        let tool = probe(&fake.to_string_lossy()).expect("脚本可探测");
        assert_eq!(tool.version.as_deref(), Some("99.1.2"));
        assert_eq!(tool.id(), format!("{}@99.1.2", fake.to_string_lossy()));
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn compiler_gets_the_flags_it_was_given() {
        // 真机用例：默认跳过，`LLM_TEST_DEVICE=cpu` 才真的编译一次。
        if !testkit::device_enabled("cpu") {
            return;
        }
        let workdir = std::env::temp_dir().join(format!("cutile-backend-{}", std::process::id()));
        let source = "#include <stdio.h>\n\
int main(void) { printf(\"%u\\n\", (unsigned)FLAG_MARK); return 0; }\n";
        let outcome = compile_and_run("cc", &["-DFLAG_MARK=7".to_string()], source, &workdir)
            .expect("带档案参数编译并运行");
        assert_eq!(outcome.bits, vec![f32::from_bits(7)]);
        let _ = std::fs::remove_dir_all(&workdir);
    }
}
