//! 被测内核的宿主 C 基准封装：把一份 `.k` 内核与样本拼成可计时的 `main()`。
//!
//! 计量口径见 README §12.1：单调时钟取样（POSIX 或 Windows）、warmup 不计入，批量校准在 C 侧、`min`/`p50` 在 Rust 侧。
//! 样本由 `op::model` 给出、实参顺序由 `op::sig` 给出：本模块只按名字与类型把缓冲渲染成 C 数组，
//! 不认识任何算子。Metal / CUDA 的语义对拍方言与生产态 lowering 已搬到 `engine`（L6），cutile 只留
//! 这条宿主 C 参考车道与签名/样本渲染原语（后者也供 `engine` 的 vendor 后端复用）。

use diagnostic::{Code, Error};
use op::model::{Sample, Values};
use op::sig::{CType, Signature};

/// 把样本与内核源码拼成一个可执行的 `main()`：warmup 若干次后逐次计时、打印输出位模式。
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
pub fn output_of<'a>(
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
pub fn output_name(signature: &Signature) -> diagnostic::Result<&str> {
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

/// 文件头：POSIX 声明、身份注释、内核源码与跨平台计时运行时。
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
const RUNTIME: &str = "\n#include <stdio.h>\n\
#ifdef _WIN32\n\
#include <windows.h>\n\
static double cutile_now_ns(void) {\n\
    static LARGE_INTEGER frequency;\n\
    LARGE_INTEGER counter;\n\
    if (frequency.QuadPart == 0) QueryPerformanceFrequency(&frequency);\n\
    QueryPerformanceCounter(&counter);\n\
    return (double)counter.QuadPart * 1e9 / (double)frequency.QuadPart;\n\
}\n\
#else\n\
#include <time.h>\n\
static double cutile_now_ns(void) {\n\
    struct timespec ts;\n\
    clock_gettime(CLOCK_MONOTONIC, &ts);\n\
    return (double)ts.tv_sec * 1e9 + (double)ts.tv_nsec;\n\
}\n\
#endif\n\n\
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

/// 缓冲的 C 元素类型。
pub fn c_type(values: &Values) -> &'static str {
    match values {
        Values::F32(_) => "float",
        Values::U8(_) => "unsigned char",
    }
}

/// 缓冲的 C 静态数组初始化式。
pub fn c_literal(values: &Values) -> String {
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

/// `cutile_x_kernel` 的签名与样本：与 `engine` 的 vendor 内核对拍共用 `testkit` 的一份取值。
#[cfg(test)]
#[path = "../../testkit/tests/support/cutile_x_kernel.rs"]
mod cutile_x_kernel;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::harness::cutile_x_kernel::{sample, signature};
    use op::model::{Buffer, Values};

    /// 生成 C 封装文本；`CUTILE_CAPTURE_GOLDEN` 显式打开时才把副本写到 `/tmp/cutile-golden`。
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
        assert!(source.contains("QueryPerformanceCounter(&counter)"));
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
    fn f32_literals_round_trip_for_random_finite_values() {
        let mut rng = testkit::Rng::new(0xF17A);
        for _ in 0..48 {
            let value = f32::from_bits(rng.below(u32::MAX));
            if !value.is_finite() {
                continue;
            }
            let literal = format!("{value:?}f");
            let parsed: f32 = literal.trim_end_matches('f').parse().expect("字面量可解析");
            assert_eq!(parsed, value, "往返失真：{value}");
        }
    }
}
