//! 被测内核的基准封装：把一份 `.k` 内核与样本拼成可计时的 `main()`。
//!
//! 计量口径见 README §12.1：墙钟取样、warmup 不计入，批量校准在 C 侧、`min`/`p50` 在 Rust 侧。
//! 样本由 `op::model` 给出、实参顺序由 `op::sig` 给出：本模块只按名字与类型把缓冲渲染成 C 数组，
//! 不认识任何算子。同模块内的 [`cuda_harness`] 是同口径的 CUDA 语义对拍方言。

use diagnostic::{Code, Error};
use op::model::{Sample, Values};
use op::sig::{CType, Param, Signature};

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

/// 拼出可在 GPU 上真实执行的 CUDA 程序：`.k` 内核转 `__device__`，单线程 `__global__` 包装在
/// `<<<1,1>>>` 里调用，host `main` 负责分配/拷贝/回读——**语义对拍门**，性能另由 roofline 门承载。
///
/// # Errors：缺输出缓冲、输出不是 f32、样本缺缓冲，或签名与样本不一致。
pub fn cuda_harness(
    kernel_source: &str,
    signature: &Signature,
    sample: &Sample,
) -> diagnostic::Result<String> {
    let (output, values) = output_of(signature, sample)?;
    let kernel = signature.kernel();
    let params = signature.params();
    let launch_args = launch_args(sample, params)?;
    let mut text = cuda_prelude(kernel_source, &kernel, sample);
    text.push_str(&cuda_wrapper(&kernel, params, &launch_args));
    text.push_str("int main(void) {\n");
    text.push_str(&cuda_host_buffers(sample, params)?);
    // 设备侧报错就打印 `cuda:` 一行并以 1 退出。
    text.push_str(&format!(
        "    cutile_launch<<<1, 1>>>({launch_args});\n\
         \x20   cudaError_t error = cudaDeviceSynchronize();\n\
         \x20   if (error != cudaSuccess) {{ fprintf(stderr, \"cuda: %s\\n\", cudaGetErrorString(error)); return 1; }}\n"
    ));
    text.push_str(&cuda_readback(output, values.len()));
    for param in params.iter().filter(|param| param.ctype.is_buffer()) {
        text.push_str(&format!("    cudaFree(d_{});\n", param.name));
    }
    text.push_str("    return 0;\n}\n");
    Ok(text)
}

/// `__global__` 包装的形参声明（与 `__device__` 同签名）。
fn cuda_params(params: &[Param]) -> String {
    let declared = params.iter().map(|param| match param.ctype {
        CType::F32 => format!("float *{}", param.name),
        CType::U8 => format!("unsigned char *{}", param.name),
        CType::Int => format!("int {}", param.name),
    });
    declared.collect::<Vec<_>>().join(", ")
}

/// 文件头：CUDA 头、身份注释，以及整体转成 `__device__` 的内核源码。
fn cuda_prelude(kernel_source: &str, kernel: &str, sample: &Sample) -> String {
    let device = kernel_source.replace(
        &format!("void {kernel}("),
        &format!("__device__ void {kernel}("),
    );
    let mut text = format!(
        "#include <cuda_runtime.h>\n#include <cstdio>\n/* cutile cuda harness | kernel={kernel} m={} n={} | 单线程设备执行（语义门，非并行性能门） */\n{device}",
        sample.m, sample.n
    );
    if !device.ends_with('\n') {
        text.push('\n');
    }
    text
}

/// 设备侧包装：单线程 `<<<1,1>>>` 调内核，入口定义只有一处。
fn cuda_wrapper(kernel: &str, params: &[Param], launch_args: &str) -> String {
    format!(
        "\n__global__ void cutile_launch({}) {{\n    {kernel}({launch_args});\n}}\n\n",
        cuda_params(params)
    )
}

/// launch 实参：缓冲传设备指针 `d_<name>`，标量取样本值。
fn launch_args(sample: &Sample, params: &[Param]) -> diagnostic::Result<String> {
    let mut args = Vec::with_capacity(params.len());
    for param in params {
        if param.ctype.is_buffer() {
            args.push(format!("d_{}", param.name));
        } else {
            let value = sample.scalar(&param.name).ok_or_else(|| {
                Error::new(
                    Code::Internal,
                    format!("标量实参 `{}` 没有取值", param.name),
                )
            })?;
            args.push(value.to_string());
        }
    }
    Ok(args.join(", "))
}

/// 结果回读：设备 → 主机，再按 `float` 位模式逐元素打印。
fn cuda_readback(output: &str, len: usize) -> String {
    format!(
        "    cudaMemcpy(h_{output}, d_{output}, {len} * sizeof(float), cudaMemcpyDeviceToHost);\n\
         \x20   for (int i = 0; i < {len}; ++i) {{\n\
         \x20       union {{ float f; unsigned int u; }} v; v.f = h_{output}[i];\n\
         \x20       printf(\"%u\\n\", v.u);\n\
         \x20   }}\n"
    )
}

/// host 侧的缓冲声明、设备分配，以及只读缓冲的入向拷贝。
fn cuda_host_buffers(sample: &Sample, params: &[Param]) -> diagnostic::Result<String> {
    let mut text = String::new();
    for param in params.iter().filter(|param| param.ctype.is_buffer()) {
        let Some(buffer) = sample.buffer(&param.name) else {
            return Err(Error::new(Code::Internal, "签名与样本已对齐（漂移）"));
        };
        let (name, ctype, size) = (&param.name, c_type(&buffer.values), buffer.values.len());
        if param.mutable {
            text.push_str(&format!("    {ctype} h_{name}[{size}];\n"));
        } else {
            text.push_str(&format!(
                "    static const {ctype} h_{name}[{size}] = {{{}}};\n",
                c_literal(&buffer.values)
            ));
        }
        text.push_str(&format!(
            "    {ctype} *d_{name} = 0;\n    cudaMalloc((void **)&d_{name}, {size} * sizeof({ctype}));\n"
        ));
        if !param.mutable {
            text.push_str(&format!(
                "    cudaMemcpy(d_{name}, h_{name}, {size} * sizeof({ctype}), cudaMemcpyHostToDevice);\n"
            ));
        }
    }
    Ok(text)
}

#[cfg(test)]
mod tests {
    use super::*;
    use op::model::{Buffer, Values};

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

    #[test]
    fn cuda_harness_makes_device_kernel_launch_and_readback() {
        let source = cuda_harness("void cutile_x_kernel(void) {}\n", &signature(), &sample())
            .expect("可生成");
        assert!(source.contains("__device__ void cutile_x_kernel("));
        assert!(source.contains("__global__ void cutile_launch("));
        assert!(source.contains("cutile_launch<<<1, 1>>>("));
        assert!(source.contains("cudaMemcpy(h_y, d_y, 2 * sizeof(float)"));
        assert!(source.contains("cudaFree(d_x)"));
        assert!(source.contains("cudaFree(d_w)"));
        assert!(source.contains("cudaFree(d_scales)"));
    }

    #[test]
    fn cuda_missing_scalar_argument_is_an_internal_error() {
        let sample = Sample {
            m: 4,
            n: 2,
            scalars: vec![("M", 4), ("N", 2)],
            buffers: vec![
                Buffer {
                    name: "x",
                    values: Values::F32(vec![1.0, 1.0]),
                },
                Buffer {
                    name: "w",
                    values: Values::U8(vec![0x98]),
                },
                Buffer {
                    name: "scales",
                    values: Values::F32(vec![0.5]),
                },
                Buffer {
                    name: "y",
                    values: Values::F32(vec![0.0]),
                },
            ],
        };
        let error = cuda_harness("void cutile_x_kernel(void) {}\n", &signature(), &sample)
            .expect_err("缺 gsz 标量");
        assert!(error.to_string().contains("gsz"), "{error}");
    }

    #[test]
    fn cuda_missing_buffer_is_an_internal_error() {
        // 签名与样本漂移：某个缓冲在样本里不存在，host 分配段无法拼出声明。
        let sample = Sample {
            m: 4,
            n: 2,
            scalars: vec![("M", 4), ("N", 2), ("gsz", 4)],
            buffers: vec![
                Buffer {
                    name: "x",
                    values: Values::F32(vec![1.0, 1.0]),
                },
                Buffer {
                    name: "y",
                    values: Values::F32(vec![0.0]),
                },
            ],
        };
        let error = cuda_harness("void cutile_x_kernel(void) {}\n", &signature(), &sample)
            .expect_err("缺 w 缓冲");
        assert!(error.to_string().contains("漂移"), "{error}");
    }
}
