//! 职责：被测内核的宿主 C 基准封装：把一份 `.k` 内核与样本拼成可计时的 `main()`。

use diagnostic::{Code, Error};
use prof::model::{Sample, Values};
use prof::sig::{CType, Signature};

/// 把样本与内核源码拼成一个可执行的 `main()`：warmup 若干次后逐次计时、打印输出位模式。
///
/// # Errors：签名没有可写输出、样本缺输出缓冲、输出不是 f32 标量或四车道档，或签名与样本不一致。
pub fn harness(
    kernel_source: &str,
    signature: &Signature,
    sample: &Sample,
    repeat: usize,
) -> diagnostic::Result<String> {
    let ctype = output_type(signature)?;
    let (output, values) = output_of(signature, sample)?;
    let kernel = signature.kernel();
    let warmup = warmup_runs(repeat);
    let args = signature.call_args(sample)?;
    let mut text = prelude(kernel_source, &kernel, sample, repeat, warmup);
    let (decls, init) = input_bufs(sample, output);
    text.push_str(&decls);
    text.push_str(&main(
        output,
        HarnessRun {
            len: values.len(),
            ctype,
            repeat,
            warmup,
        },
        &kernel,
        &args,
        &init,
    ));
    Ok(text)
}

/// 可写输出与它的扁平标量视图；`float4` 按底层 f32 车道回读。
pub fn output_of<'a>(
    signature: &'a Signature,
    sample: &'a Sample,
) -> diagnostic::Result<(&'a str, &'a [f32])> {
    let output = output_name(signature)?;
    let out = sample
        .buf(output)
        .ok_or_else(|| Error::new(Code::Missing, format!("样本里没有输出缓冲 `{output}`")))?;
    let values = out.values.as_f32().map_err(|_| {
        Error::new(
            Code::Invalid,
            format!("输出缓冲 `{output}` 必须是 f32 或 f32x4"),
        )
    })?;
    Ok((output, values))
}

/// 可写输出名：只允许可按 f32 车道回读的标量与四车道缓冲。
pub fn output_name(signature: &Signature) -> diagnostic::Result<&str> {
    let output = signature.output().ok_or_else(|| {
        Error::new(
            Code::Internal,
            "内核签名没有可写输出缓冲（装载期已经校验过，这里只能是实现漂移）",
        )
    })?;
    let ctype = output_type(signature)?;
    if matches!(ctype, CType::F32 | CType::F32x4) {
        return Ok(output);
    }
    Err(Error::new(
        Code::Unsupported,
        format!(
            "算子 `{}` 的可写输出 `{output}` 是 `{}`；回读只认 `float` 与 `float4`",
            signature.op(),
            ctype.c_name()
        ),
    ))
}

fn output_type(signature: &Signature) -> diagnostic::Result<CType> {
    signature
        .output_type()
        .ok_or_else(|| Error::new(Code::Internal, "内核签名有输出名却缺少输出类型"))
}

/// 文件头：POSIX 声明、身份注释、内核源码与跨平台计时运行时。
fn prelude(
    kernel_source: &str,
    kernel: &str,
    sample: &Sample,
    repeat: usize,
    warmup: usize,
) -> String {
    // `clock_gettime` 在 `-std=c99` 下需要先声明 POSIX 版本，且必须在任何系统头之前。
    let mut text = format!(
        "#define _POSIX_C_SOURCE 200809L\n/* cutile harness | kernel={kernel} m={} n={} repeat={repeat} warmup={warmup} */\n{kernel_source}",
        sample.m, sample.n
    );
    if !text.ends_with('\n') {
        text.push('\n');
    }
    text + RUNTIME
}

/// 计时运行时：`CLOCK_MONOTONIC` 取样、编译器屏障、`f32` 位模式回读。
const RUNTIME: &str = "\n#include <stdio.h>\n\
#include <stddef.h>\n\
#include <string.h>\n\
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
static float cutile_read_f32(const void *values, int idx) {\n\
    float value;\n\
    memcpy(&value, (const unsigned char *)values + (size_t)idx * sizeof value, sizeof value);\n\
    return value;\n\
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
fn input_bufs(sample: &Sample, output: &str) -> (String, String) {
    let mut decls = String::new();
    let mut init = String::new();
    for buf in &sample.bufs {
        if buf.name == output {
            continue;
        }
        let ctype = c_type(&buf.values);
        let size = buf.values.storage_len();
        // C/CUDA 不接受零长数组；存储至少一项，向量按组初始化，ABI 标量仍由样本给出。
        let storage = size.max(1);
        decls.push_str(&format!(
            "static volatile const {ctype} cutile_init_{name}[{storage}] = {{{literal}}};\n\
             static {ctype} {name}[{storage}];\n",
            name = buf.name,
            literal = c_literal(&buf.values)
        ));
        init.push_str(&format!(
            "    for (int cutile_i = 0; cutile_i < {size}; ++cutile_i) \
             {{ {name}[cutile_i] = cutile_init_{name}[cutile_i]; }}\n",
            name = buf.name
        ));
    }
    (decls, init)
}

/// 一次基准的运行参数：输出标量数、输出类型、样本数与预热次数。
struct HarnessRun {
    len: usize,
    ctype: CType,
    repeat: usize,
    warmup: usize,
}

/// 主函数：warmup、校准批量、逐样本计时，最后按位模式回读输出。`CUTILE_BARRIER()` 挡"提到循环外"，
/// `volatile` 出口读挡"只留最后一次写"：缺一，32x1 就退化成空循环。
fn main(output: &str, run: HarnessRun, kernel: &str, call_args: &str, init: &str) -> String {
    let (len, ctype, repeat, warmup) = (run.len, run.ctype, run.repeat, run.warmup);
    // 按底层标量车道保活与回读；memcpy 不把向量对象错误地别名成 float 数组。
    let sink = if len > 0 {
        format!(
            "        for (int cutile_s = 0; cutile_s < {len}; ++cutile_s) {{ cutile_sink = cutile_read_f32({output}, cutile_s); }}\n"
        )
    } else {
        String::new()
    };
    let storage = (len / ctype.lanes()).max(1);
    let ctype = ctype.c_name();
    format!(
        "int main(void) {{
    /* 大输出不能占宿主线程栈（Windows 默认 1 MiB）；与输入一样使用静态存储。 */
    static {ctype} {output}[{storage}];
    volatile float cutile_sink = 0.0f;
{init}    /* warmup：代码、页表与缓存进稳态，不计入样本 */
    for (int cutile_warm = 0; cutile_warm < {warmup}; ++cutile_warm) {{
        memset({output}, 0, sizeof({output}));
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
        memset({output}, 0, sizeof({output}));
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
        cutile_u.f = cutile_read_f32({output}, cutile_i);
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

/// 缓冲的 C 元素类型；与 `Values` 的档／车道描述唯一同源。
pub fn c_type(values: &Values) -> &'static str {
    values.describe()
}

/// 缓冲的 C 静态数组初始化式。
pub fn c_literal(values: &Values) -> String {
    match values {
        Values::F32(values) => literals(values.iter().copied(), float_literal),
        Values::U8(values) => literals(values.iter().copied(), byte_literal),
        Values::F16(values) => literals(values.iter().map(|value| value.to_f32()), float_literal),
        Values::Bf16(values) => literals(values.iter().map(|value| value.to_f32()), float_literal),
        other => vector_literal(other),
    }
}

fn vector_literal(values: &Values) -> String {
    match values {
        Values::F32x4(values) => vector_literals(values.as_arrays(), float_literal),
        Values::U8x4(values) => vector_literals(values.as_arrays(), byte_literal),
        Values::F16x4(values) => {
            vector_literals(values.as_arrays(), |value| float_literal(value.to_f32()))
        }
        Values::Bf16x4(values) => {
            vector_literals(values.as_arrays(), |value| float_literal(value.to_f32()))
        }
        other => c_literal(other),
    }
}

/// 每组单独加花括号，避免把 N 个标量误当成 N 个向量数组项。
fn vector_literals<T: Copy>(values: &[[T; 4]], mut literal: impl FnMut(T) -> String) -> String {
    values
        .iter()
        .map(|group| format!("{{{}}}", literals(group.iter().copied(), &mut literal)))
        .collect::<Vec<_>>()
        .join(", ")
}

fn literals<T>(values: impl IntoIterator<Item = T>, literal: impl FnMut(T) -> String) -> String {
    values
        .into_iter()
        .map(literal)
        .collect::<Vec<_>>()
        .join(", ")
}

fn float_literal(value: f32) -> String {
    format!("{value:?}f")
}

fn byte_literal(value: u8) -> String {
    format!("0x{value:02x}u")
}

/// `cutile_x_kernel` 的签名与样本：与 `engine` 的 vendor 内核对拍共用 `testkit` 的一份取值。
#[cfg(test)]
#[path = "../../testkit/tests/kernel.rs"]
mod cutile_x_kernel;

#[cfg(test)]
#[path = "../tests/harness_tests.rs"]
mod tests;
