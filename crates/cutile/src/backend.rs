//! ⑤ 编译与执行：`c` 后端通过子进程调用 `cc`，不涉及 FFI，也不需要 `unsafe`。
//!
//! 只有"能跑出数"才算 L2 证据（DESIGN §11.2）：这里负责把生成的源文件编成可执行文件、
//! 跑一遍、把 f32 结果的位模式读回来交给 [`crate::reference`] 对拍。
//!
//! 编译参数由调用方从档案的 `[compile].flags` 传进来——与缓存指纹里哈希的是**同一份**，
//! 代码里不写死优化级别，否则"缓存键描述了它、实际编译却没用它"。
//!
//! 样本由算子层给出（`op::model`），实参顺序与输出名由 `.k` 的签名给出（`op::sig`）：本模块
//! 只按名字与类型把缓冲渲染成 C 数组，不认识任何算子。

use diagnostic::{Code, Error};
use op::model::{Sample, Values};
use op::sig::Signature;
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

/// 计时样本：每个样本是一次内核调用的墙钟耗时（纳秒）。
///
/// 用 `CLOCK_MONOTONIC` 而不是 `clock()`：后者是 CPU 时间，跨核迁移、睡眠与频率变化下
/// 都不是墙钟，也无法与 `bytes/ns` 的口径对齐（DESIGN §12.1）。
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Timing {
    /// 全部样本（纳秒，按执行顺序；warmup 已排除）。
    pub samples_ns: Vec<f64>,
}

impl Timing {
    /// 最快的一次：达成率用它，避免被调度抖动与换页拖低。
    pub fn min_ns(&self) -> Option<f64> {
        self.samples_ns.iter().copied().reduce(f64::min)
    }

    /// 中位数（偶数个样本取中间两个的平均值）：与 [`Self::min_ns`] 的差距即抖动幅度。
    pub fn p50_ns(&self) -> Option<f64> {
        if self.samples_ns.is_empty() {
            return None;
        }
        let mut sorted = self.samples_ns.clone();
        sorted.sort_by(f64::total_cmp);
        let middle = sorted.len() / 2;
        Some(if sorted.len().is_multiple_of(2) {
            (sorted[middle - 1] + sorted[middle]) / 2.0
        } else {
            sorted[middle]
        })
    }
}

/// 把样本与内核源码拼成一个可执行的 `main()`：warmup 若干次后逐次计时、打印输出位模式。
///
/// 计量口径（DESIGN §12.1）：`CLOCK_MONOTONIC` 墙钟、warmup 不计入、每个样本是若干次内核调用
/// （校准后自动定批量：大内核 1 次，亚微秒内核放大到能跨过时钟分辨率），C 侧按批量折算成单次
/// 耗时，`min`/`p50` 在 Rust 侧算。缓冲按 [`Sample`] 的名字与类型声明：`output` 那个声明为可写
/// 并清零，其余是运行时从 `volatile const` 初值拷入的普通数组（避免常量折叠）。实参列表与
/// 输出名都由签名派生（`op::sig`），本函数不持有任何算子名或参数表。
///
/// # Errors
///
/// 签名没有可写输出、样本里没有输出缓冲、输出缓冲不是 f32，或签名与样本不一致时返回说明。
pub fn harness(
    kernel_source: &str,
    signature: &Signature,
    sample: &Sample,
    repeat: usize,
) -> diagnostic::Result<String> {
    let output = signature.output().ok_or_else(|| {
        Error::new(
            Code::Internal,
            "内核签名没有可写输出缓冲（装载期已经校验过，这里只能是实现漂移）",
        )
    })?;
    let out = sample
        .buffer(output)
        .ok_or_else(|| Error::new(Code::Missing, format!("样本里没有输出缓冲 `{output}`")))?;
    let Values::F32(out_values) = &out.values else {
        return Err(Error::new(
            Code::Invalid,
            format!("输出缓冲 `{output}` 必须是 f32"),
        ));
    };
    let call_args = signature.call_args(sample)?;
    let kernel = signature.kernel();
    let len = out_values.len();
    let warmup = warmup_runs(repeat);

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
    text.push_str(
        "\n#include <stdio.h>\n#include <time.h>\n\n\
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
#endif\n\n",
    );

    // 输入先落在 `volatile const` 的初值数组里，再在运行时拷进真正的缓冲：
    // 这样 `-O3` 无法把常量输入折进内核（32x1 会被折成空循环），每次调用的代价才是真的。
    let mut init = String::new();
    for buffer in &sample.buffers {
        if buffer.name == output {
            continue;
        }
        let ctype = c_type(&buffer.values);
        let size = buffer.values.len();
        text.push_str(&format!(
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

    // `CUTILE_BARRIER()` + `volatile` 出口读一起让每次调用都必须真的发生：
    // 只有屏障挡得住"提到循环外"，只有出口读挡得住"只保留最后一次的写"（两者缺一，
    // 32x1 都会退化成空循环或一次调用）。
    let sink = if len > 0 {
        format!("        cutile_sink = {output}[0];\n")
    } else {
        String::new()
    };
    text.push_str(&format!(
        "int main(void) {{
    float {output}[{len}];
    volatile float cutile_sink = 0.0f;
{init}    /* warmup：代码、页表与缓存进稳态，不计入样本 */
    for (int cutile_warm = 0; cutile_warm < {warmup}; ++cutile_warm) {{
        for (int cutile_i = 0; cutile_i < {len}; ++cutile_i) {{ {output}[cutile_i] = 0.0f; }}
        {kernel}({call_args});
        CUTILE_BARRIER();
{sink}    }}
    /* 校准：一次调用决定批量，让亚微秒内核也能跨过时钟分辨率；校准本身不算样本 */
    double cutile_c0 = cutile_now_ns();
    {kernel}({call_args});
    double cutile_calib = cutile_now_ns() - cutile_c0;
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
    ));
    Ok(text)
}

/// warmup 次数：至少一次，约为测量次数的 1/10。
fn warmup_runs(repeat: usize) -> usize {
    (repeat / 10).max(1)
}

/// 缓冲的 C 元素类型。
fn c_type(values: &Values) -> &'static str {
    match values {
        Values::F32(_) => "float",
        Values::U8(_) => "unsigned char",
    }
}

/// 缓冲的 C 静态数组初始化式。
fn c_literal(values: &Values) -> String {
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

/// 探测一个可用的 C 编译器（`cc` → `clang` → `gcc`）。
pub fn detect_toolchain() -> Option<String> {
    for candidate in ["cc", "clang", "gcc"] {
        let ok = Command::new(candidate)
            .arg("--version")
            .output()
            .map(|output| output.status.success())
            .unwrap_or(false);
        if ok {
            return Some(candidate.to_string());
        }
    }
    None
}

/// 只编译：把源文本写到 `dir/kernel.c` 并编成 `dir/kernel`，返回可执行文件路径。
///
/// 与 [`run`] 分开是因为编译产物可以落进缓存（`cutile::cache`）：命中时整步跳过，
/// 没有编译器也要能跑一次已经验证过的产物。
///
/// # Errors
///
/// 目录不可建、源文件不可写、编译器不存在，或编译失败（诊断带命令、完整输出与源文件路径）。
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
/// # Errors
///
/// 可执行文件不可运行（权限、架构不符）或退出码非零时返回说明，`stderr` 原样带进诊断。
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
        Sample {
            m: 4,
            n: 2,
            scalars: vec![("M", 4), ("N", 2), ("gsz", 4)],
            buffers: vec![
                Buffer {
                    name: "x",
                    values: Values::F32(vec![1.0, 1.0, 1.0, 1.0]),
                },
                Buffer {
                    name: "w",
                    values: Values::U8(vec![0x98, 0x76]),
                },
                Buffer {
                    name: "scales",
                    values: Values::F32(vec![0.5, 0.5]),
                },
                Buffer {
                    name: "y",
                    values: Values::F32(vec![0.0, 0.0]),
                },
            ],
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
        assert!(source.contains("cutile_sink = y[0];"));
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
        assert_eq!(Timing::default().min_ns(), None);
        assert_eq!(Timing::default().p50_ns(), None);
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
        let _ = detect_toolchain();
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
