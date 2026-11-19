//! 被测内核的基准封装：把一份 `.k` 内核与样本拼成可计时的 `main()`。
//!
//! 计量口径见 README §12.1：墙钟取样、warmup 不计入，批量校准在 C 侧、`min`/`p50` 在 Rust 侧。
//! 样本由 `op::model` 给出、实参顺序由 `op::sig` 给出：本模块只按名字与类型把缓冲渲染成 C 数组，
//! 不认识任何算子。同模块内的 [`cuda_harness`] 是同口径的 CUDA 语义对拍方言。

use diagnostic::{Code, Error};
use op::decl::Grid;
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

/// 只读输入缓冲名（唯一非可写缓冲）：归约/扫描/循环注入三族 lowering 共用这一条查表。
fn read_input(signature: &Signature) -> diagnostic::Result<&str> {
    signature
        .params()
        .iter()
        .filter(|param| param.ctype.is_buffer())
        .find(|param| !param.mutable)
        .map(|param| param.name.as_str())
        .ok_or(Error::new(Code::Internal, "内核没有只读输入缓冲").with("cutile::harness::read_input"))
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

/// `__global__` 包装的形参声明：缓冲是设备指针，用 `d_` 前缀与 host 侧 `cuda_host_buffers` 的
/// `cudaMalloc(&d_<name>)` 同源；标量直接传值。`cutile_launch<<<1,1>>>(d_<name>...)` 与包装内
/// `kernel(d_<name>...)` 用同一份 `launch_args`，名字必须一致。
fn cuda_params(params: &[Param]) -> String {
    params
        .iter()
        .map(|param| {
            let base = param.ctype.c_name();
            if param.ctype.is_buffer() {
                format!("{base} *d_{}", param.name)
            } else {
                format!("{base} {}", param.name)
            }
        })
        .collect::<Vec<_>>()
        .join(", ")
}

/// 文件头：CUDA 头、身份注释，以及整体转成 `__device__` 的内核源码。
fn cuda_prelude(kernel_source: &str, kernel: &str, sample: &Sample) -> String {
    let device = cuda_device(kernel_source, kernel);
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

/// 拼出可在 GPU 上真实执行的 CUDA 网格程序：`kernel_main<<<grid,1>>>` 发射 + host 分配/拷贝/回读。
/// 与 [`cuda_harness`] 同口径，但发射是生产态网格（并行），输出缓冲长度由 `output_len` 显式给出——
/// 分块归约的输出是 `ceil(N/chunk)` 个部分结果，大于签名的单元素 `y`。
///
/// # Errors：缺输出缓冲、样本缺缓冲，或签名与样本不一致。
pub fn cuda_grid_driver(
    device_source: &str,
    signature: &Signature,
    sample: &Sample,
    grid: usize,
    output_len: usize,
) -> diagnostic::Result<String> {
    let output = output_name(signature)?;
    let params = signature.params();
    let launch = params
        .iter()
        .filter(|param| param.ctype.is_buffer())
        .map(|param| format!("d_{}", param.name))
        .collect::<Vec<_>>()
        .join(", ");
    let mut text = String::from("#include <cuda_runtime.h>\n#include <cstdio>\n");
    text.push_str(device_source);
    if !device_source.ends_with('\n') {
        text.push('\n');
    }
    text.push_str("\nint main(void) {\n");
    text.push_str(&cuda_host_buffers_grid(sample, params, output, output_len)?);
    text.push_str(&format!(
        "    kernel_main<<<{grid}, 1>>>({launch});\n\
         \x20   cudaError_t error = cudaDeviceSynchronize();\n\
         \x20   if (error != cudaSuccess) {{ fprintf(stderr, \"cuda: %s\\n\", cudaGetErrorString(error)); return 1; }}\n"
    ));
    text.push_str(&cuda_readback(output, output_len));
    for param in params.iter().filter(|param| param.ctype.is_buffer()) {
        text.push_str(&format!("    cudaFree(d_{});\n", param.name));
    }
    text.push_str("    return 0;\n}\n");
    Ok(text)
}

/// host 侧缓冲声明、设备分配与只读入向拷贝；输出缓冲长度用显式 `output_len`（分块归约的部分
/// 结果数），而非样本里的单元素。
fn cuda_host_buffers_grid(
    sample: &Sample,
    params: &[Param],
    output: &str,
    output_len: usize,
) -> diagnostic::Result<String> {
    let mut text = String::new();
    for param in params.iter().filter(|param| param.ctype.is_buffer()) {
        let Some(buffer) = sample.buffer(&param.name) else {
            return Err(Error::new(Code::Internal, "签名与样本已对齐（漂移）"));
        };
        text.push_str(&cuda_buffer_host(param, &buffer.values, output, output_len));
    }
    Ok(text)
}

/// 单个缓冲的 host 声明 + 设备分配 + （只读时）入向拷贝；`output` 与 `output_len` 只用于把输出
/// 缓冲的尺寸换成部分结果数，其余缓冲尺寸取样本原值。
fn cuda_buffer_host(param: &Param, values: &Values, output: &str, output_len: usize) -> String {
    let ctype = c_type(values);
    let size = if param.name == output {
        output_len
    } else {
        values.len()
    };
    let mut text = if param.mutable {
        format!("    {ctype} h_{}[{size}];\n", param.name)
    } else {
        format!(
            "    static const {ctype} h_{}[{size}] = {{{}}};\n",
            param.name,
            c_literal(values)
        )
    };
    text.push_str(&format!(
        "    {ctype} *d_{} = 0;\n    cudaMalloc((void **)&d_{}, {size} * sizeof({ctype}));\n",
        param.name, param.name
    ));
    if !param.mutable {
        text.push_str(&format!(
            "    cudaMemcpy(d_{}, h_{}, {size} * sizeof({ctype}), cudaMemcpyHostToDevice);\n",
            param.name, param.name
        ));
    }
    text
}

// ---------------------------------------------------------------------------
// Metal 内核对拍：MSL 需要显式地址空间，`.k` 的 C 签名在 `xcrun metal` 下是语法错误。
// 与 CUDA 同口径：单线程语义门，`device` 缓冲 ↔ `thread` 局部数组，host 驱动发射与回读。
// ---------------------------------------------------------------------------

/// MSL 没有默认地址空间，`.k` 的裸指针（签名 + 函数体局部）都要补 `device`——这些指针指向全局
/// 缓冲，`thread` 栈放不下生产形状（`REG` 上限 64 KiB）。`\u{1}` 先占住 `*`、最后统一回填，避免
/// 「先替换 const 版后，结果里的 `TYPE *` 又被非 const 规则二次命中」。
fn msl_address_space(text: &str) -> String {
    text.replace("const float *", "device const float\u{1}")
        .replace("const unsigned char *", "device const unsigned char\u{1}")
        .replace("const u8 *", "device const u8\u{1}")
        .replace("float *", "device float\u{1}")
        .replace("unsigned char *", "device unsigned char\u{1}")
        .replace("u8 *", "device u8\u{1}")
        .replace("\u{1}", " *")
}

/// 拼出可在本机 GPU 上真实执行的 Metal 程序：`.k` 内核补 `thread` 地址空间后，`cutile_launch`
/// 把它包成 compute kernel（`device` 缓冲 ↔ `thread` 局部数组），host 驱动负责把样本发射到 GPU、
/// 回读输出。返回 `(msl, driver)` 两段文本：`msl` 交给 `xcrun metal`，`driver` 交给 clang。
///
/// # Errors：缺输出缓冲、输出不是 f32、样本缺缓冲，或签名与样本不一致。
pub fn metal_harness(
    kernel_source: &str,
    signature: &Signature,
    sample: &Sample,
) -> diagnostic::Result<(String, String)> {
    let (output, _) = output_of(signature, sample)?;
    let msl = format!(
        "{}\n{}",
        msl_address_space(kernel_source),
        metal_launch(&signature.kernel(), signature, sample)?
    );
    Ok((msl, metal_driver(signature, sample, output)?))
}

/// 生产态设备缓冲声明：按签名序 `device {ctype}* d_{name} [[buffer({slot})]]`，槽位序号与驱动侧
/// `setBuffer:atIndex:` 同源。逐点 / 串行 / 循环注入三条 lowering 共用，只写这一份。
fn buffer_decls(signature: &Signature) -> String {
    signature
        .params()
        .iter()
        .filter(|param| param.ctype.is_buffer())
        .enumerate()
        .map(|(slot, param)| {
            format!(
                "    device {}* d_{} [[buffer({slot})]]",
                param.ctype.c_name(),
                param.name
            )
        })
        .collect::<Vec<_>>()
        .join(",\n")
}

/// 后端包装差异表：同一条 kernel body 拼成不同后端的 `kernel_main`。差异全是语法，机制层只按表插值、
/// 不按后端写 `match`、不单态化。加后端 = 加一行常量，与 `device::SPECS` 同一套登记方式。
struct Wrap {
    /// 把 `.k` 内核 body 转成该后端可编译的翻译单元（Metal 补 `device` 地址空间，CUDA 补 `__device__`）。
    prelude: fn(&str, &str) -> String,
    /// 缓冲声明段（按签名序；Metal 带 `[[buffer(slot)]]`，CUDA 不带）。
    buffers: fn(&Signature) -> String,
    /// `kernel_main` 声明关键字：`kernel` / `__global__`。
    decl: &'static str,
    /// 逐点网格线程形参（Metal `uint n [[thread_position_in_grid]]`；CUDA 空，用 `point_stmt` 语句）。
    point_param: &'static str,
    /// 逐点网格线程下标语句（CUDA `int n = blockIdx.x * blockDim.x + threadIdx.x;`；Metal 空）。
    point_stmt: &'static str,
    /// 分块/循环网格线程形参（Metal `uint3 tid [[thread_position_in_grid]]`；CUDA 空）。
    grid_param: &'static str,
    /// 分块/循环网格线程下标语句（CUDA `int g = blockIdx.x * blockDim.x + threadIdx.x;`；Metal 空）。
    grid_stmt: &'static str,
}

/// 把翻译单元拼成 `kernel_main`：prelude 变换后的 body 接后端声明、缓冲段、线程形参与线程下标语句。
fn kernel_main(
    wrap: &Wrap,
    source: &str,
    kernel: &str,
    buffers: &str,
    tid_param: &str,
    tid_stmt: &str,
    body: &str,
) -> String {
    let param = if tid_param.is_empty() {
        String::new()
    } else {
        format!(",\n    {tid_param}")
    };
    let stmt = if tid_stmt.is_empty() {
        String::new()
    } else {
        format!("    {tid_stmt}\n")
    };
    format!(
        "{}\n\n{} void kernel_main(\n{buffers}{param}) {{\n{stmt}{body}\n}}",
        (wrap.prelude)(source, kernel),
        wrap.decl,
    )
}

/// Metal 前置：`.k` 的 C 指针补 `device` 地址空间（MSL 无默认地址空间）。
fn metal_prelude(text: &str, _kernel: &str) -> String {
    msl_address_space(text)
}

/// CUDA 前置：`.k` 的内核函数与 `static` 辅助函数统一补 `__device__`（否则默认 `__host__`）。
fn cuda_device(text: &str, kernel: &str) -> String {
    text.replace(
        &format!("void {kernel}("),
        &format!("__device__ void {kernel}("),
    )
    .replace("static ", "__device__ static ")
}

/// CUDA 缓冲声明：`{ctype}* d_{name}`（无地址空间、无槽位标签，指针即全局内存）。
fn cuda_buffers(signature: &Signature) -> String {
    signature
        .params()
        .iter()
        .filter(|param| param.ctype.is_buffer())
        .map(|param| format!("    {}* d_{}", param.ctype.c_name(), param.name))
        .collect::<Vec<_>>()
        .join(",\n")
}

const METAL: Wrap = Wrap {
    prelude: metal_prelude,
    buffers: buffer_decls,
    decl: "kernel",
    point_param: "uint n [[thread_position_in_grid]]",
    point_stmt: "",
    grid_param: "uint3 tid [[thread_position_in_grid]]",
    grid_stmt: "int g = (int)tid.x;",
};

const CUDA: Wrap = Wrap {
    prelude: cuda_device,
    buffers: cuda_buffers,
    decl: "__global__",
    point_param: "",
    point_stmt: "int n = blockIdx.x * blockDim.x + threadIdx.x;",
    grid_param: "",
    grid_stmt: "int g = blockIdx.x * blockDim.x + threadIdx.x;",
};

/// 生产态逐点网格内核：把逐点（pointwise）串行内核包成按网格派发的 `kernel_main`，后端差异收进
/// [`Wrap`]，同一条 body 只拼一次。
///
/// 与 [`metal_harness`] 的单线程见证（入口 `cutile_launch`、`if (tid != 0) return`）不同，这里每个
/// 线程处理一个元素：网格索引 `n` 落到每个缓冲的 `+ n` 偏移、唯一的长度标量传 `1`。因此**只对
/// 逐点族成立**——所有缓冲按元素步长 1 索引、唯一标量是长度 `N`（`add` / `mul` / `scale` / `silu` /
/// `gelu` / `residual_add`）。gemv/gemm 的行跨步与分组标量、reduce 的跨元素归约、embed 的行 gather、
/// rmsnorm 的两趟归一化都要各自的 lowering，不能套这条。
///
/// # Errors：签名不是逐点形（标量不止长度一项）。
fn grid_kernel(
    kernel_source: &str,
    signature: &Signature,
    wrap: &Wrap,
) -> diagnostic::Result<String> {
    let scalars = signature
        .params()
        .iter()
        .filter(|param| !param.ctype.is_buffer())
        .count();
    if scalars != 1 {
        return Err(Error::new(
            Code::Invalid,
            format!(
                "`{}` 不是逐点签名（标量 {} 项）：行跨步/归约 lowering 不在此列",
                signature.op(),
                scalars
            ),
        )
        .with("cutile::harness::grid_kernel"));
    }
    let call = grid_args(signature.params());
    let buffers = (wrap.buffers)(signature);
    let body = format!("    {}({call});", signature.kernel());
    Ok(kernel_main(
        wrap,
        kernel_source,
        &signature.kernel(),
        &buffers,
        wrap.point_param,
        wrap.point_stmt,
        &body,
    ))
}

/// 生产态 Metal 逐点网格内核（[`grid_kernel`] 的 Metal 方言）。
pub fn metal_grid_kernel(kernel_source: &str, signature: &Signature) -> diagnostic::Result<String> {
    grid_kernel(kernel_source, signature, &METAL)
}

/// 生产态 CUDA 逐点网格内核（[`grid_kernel`] 的 CUDA 方言）。
pub fn cuda_grid_kernel(kernel_source: &str, signature: &Signature) -> diagnostic::Result<String> {
    grid_kernel(kernel_source, signature, &CUDA)
}

/// 生产态实参：缓冲取 `d_{name} + n`（线程处理的元素）、标量取 `1`（每线程一个元素）。
fn grid_args(params: &[Param]) -> String {
    params
        .iter()
        .map(|param| {
            if param.ctype.is_buffer() {
                format!("d_{} + n", param.name)
            } else {
                "1".to_owned()
            }
        })
        .collect::<Vec<_>>()
        .join(", ")
}

/// 串行实参：缓冲取 `d_{name}`（原样透传，单线程处理全部），标量取样本里的实际值。
fn serial_args(params: &[Param], sample: &Sample) -> diagnostic::Result<String> {
    params
        .iter()
        .map(|param| {
            if param.ctype.is_buffer() {
                Ok(format!("d_{}", param.name))
            } else {
                let value = sample.scalar(&param.name).ok_or_else(|| {
                    Error::new(
                        Code::Invalid,
                        format!("串行内核标量 `{}` 没有样本取值", param.name),
                    )
                    .with("cutile::harness::metal_loop_kernel")
                })?;
                Ok(value.to_string())
            }
        })
        .collect::<diagnostic::Result<Vec<_>>>()
        .map(|args| args.join(", "))
}

/// 生产态循环注入网格内核：把模板里的串行循环头绑定到线程下标（N 维网格，`tid.x` / `tid.y` …）。
///
/// 网格绑定来自 `[grid]` 声明（[`Grid`]），不是对模板的索引分析：每个网格维把模板里的串行循环
/// `for (int t = 0; t < S; ++t)` 的头绑定到线程下标（`t = tid.x`、循环只跑一次），正文里的跨步
/// （`i*K`、`p*N`、`i*D`）原样保留、标量取样本原值。这比「压缩标量 + 缓冲偏移」更泛：gemm 右
/// 操作数的规约维跨步 `p*N` 不会因压缩 `N` 而错位，注意力里 `i*D` 的行跨步同样保留。`tid` 声明在
/// `kernel_main`，而循环在内部函数里，故先给内部函数追加 `uint3 tid` 形参、调用点透传。gemv 一维、
/// gemm 二维、attention 按查询行一维共用这一条 lowering，差异全在声明里。
///
/// 空网格（`dims` 为空）退化为**串行**：不追加 `tid`、不绑循环、缓冲原样透传、标量取样本值——
/// 串行就是循环注入的零维特例，不再单列一条 lowering。
///
/// # Errors：网格维超过 3、模板里找不到声明的循环、标量缺样本取值时返回说明。
pub fn metal_loop_kernel(
    kernel_source: &str,
    signature: &Signature,
    sample: &Sample,
    grid: &Grid,
) -> diagnostic::Result<String> {
    if grid.dims.is_empty() {
        return serial_kernel(kernel_source, signature, sample);
    }
    injected_kernel(kernel_source, signature, sample, grid)
}

/// 串行（零维网格）特例：不追加 `tid`、不绑循环，缓冲原样透传、标量取样本值。
fn serial_kernel(
    kernel_source: &str,
    signature: &Signature,
    sample: &Sample,
) -> diagnostic::Result<String> {
    let call = serial_args(signature.params(), sample)?;
    let buffers = buffer_decls(signature);
    Ok(format!(
        "{}\n\nkernel void kernel_main(\n{buffers}) {{\n    {}({call});\n}}",
        msl_address_space(kernel_source),
        signature.kernel(),
    ))
}

/// 循环注入（非零网格）：给内部内核追加 `uint3 tid`、绑循环头到线程下标、调用点透传 `tid`。
fn injected_kernel(
    kernel_source: &str,
    signature: &Signature,
    sample: &Sample,
    grid: &Grid,
) -> diagnostic::Result<String> {
    let mut body = msl_address_space(kernel_source);
    body = thread_param(&body, &signature.kernel())?;
    body = bind_grid_loops(body, grid)?;
    let call = format!("{}, tid", serial_args(signature.params(), sample)?);
    let buffers = buffer_decls(signature);
    Ok(format!(
        "{body}\n\nkernel void kernel_main(\n{buffers},\n    uint3 tid [[thread_position_in_grid]]) {{\n    {}({call});\n}}",
        signature.kernel(),
    ))
}

/// 生产态分块归约内核：把长轴切成 `chunk` 元素一块，网格 = `ceil(N / chunk)`，每线程用设备
/// preamble 里的软件归约原语（`_cutile_sum_f32` / `_cutile_max_f32`）串行归约自己那一块，把
/// **块级部分结果**写到输出缓冲的第 `g` 个槽（`y[g]`）。主机折叠这些部分结果得到最终标量——
/// 归约的跨元素依赖不能靠逐元素拆线程（每线程会各算一整遍），只能「切块 + 主机折叠」，这是与
/// [`metal_loop_kernel`]（循环头绑定）不同的另一条 lowering，差异只在「规约轴交给谁」。
///
/// 输出语义因此从「单元素 `y[0]`」变成「`ceil(N/chunk)` 个部分结果」，折叠由调用方在主机侧做；
/// 归约函数名由分派层从算子声明推导（`reduce_sum_f32` → `_cutile_sum_f32`），不是对模板的字符串
/// 猜测。块长来自 `[chunk]` 声明，机制层只做 `start = g*size` / `len = min(size, N-start)` 的切块。
///
/// # Errors：签名不是归约形（缺唯一输入缓冲 / 可写输出 / 长度标量）或标量缺样本取值时返回说明。
pub fn metal_chunk_reduce(
    kernel_source: &str,
    signature: &Signature,
    sample: &Sample,
    chunk: usize,
    reduce_func: &str,
) -> diagnostic::Result<String> {
    chunk_reduce(kernel_source, signature, sample, chunk, reduce_func, &METAL)
}

/// 生产态 CUDA 分块归约内核（[`chunk_reduce`] 的 CUDA 方言）。
pub fn cuda_chunk_reduce(
    kernel_source: &str,
    signature: &Signature,
    sample: &Sample,
    chunk: usize,
    reduce_func: &str,
) -> diagnostic::Result<String> {
    chunk_reduce(kernel_source, signature, sample, chunk, reduce_func, &CUDA)
}

/// 生产态分块归约内核（后端无关）：把长轴切成 `chunk` 元素一块，网格 = `ceil(N / chunk)`，每线程用
/// 设备 preamble 里的软件归约原语（`_cutile_sum_f32` / `_cutile_max_f32`）串行归约自己那一块，把
/// **块级部分结果**写到输出缓冲的第 `g` 个槽（`y[g]`）。主机折叠这些部分结果得到最终标量——
/// 归约的跨元素依赖不能靠逐元素拆线程（每线程会各算一整遍），只能「切块 + 主机折叠」。后端差异收进
/// [`Wrap`]（缓冲声明、线程下标、`kernel_main` 声明关键字），与 [`metal_chunk_reduce`] / [`cuda_chunk_reduce`]
/// 共用同一条 body。
///
/// # Errors：签名不是归约形（缺唯一输入缓冲 / 可写输出 / 长度标量）或标量缺样本取值时返回说明。
fn chunk_reduce(
    kernel_source: &str,
    signature: &Signature,
    sample: &Sample,
    chunk: usize,
    reduce_func: &str,
    wrap: &Wrap,
) -> diagnostic::Result<String> {
    let input = read_input(signature)?;
    let output = output_name(signature)?;
    let len = length_scalar(signature, sample)?;
    let buffers = (wrap.buffers)(signature);
    let body = format!(
        "    int start = g * {chunk};\n    int len = min({chunk}, {len} - start);\n    d_{output}[g] = {reduce_func}(d_{input} + start, len, 1);"
    );
    Ok(kernel_main(
        wrap,
        kernel_source,
        &signature.kernel(),
        &buffers,
        wrap.grid_param,
        wrap.grid_stmt,
        &body,
    ))
}

/// 生产态分块扫描内核：把时间轴串行递推切成 `chunk` 元素一块，返回两段内核源 `(局部扫描, 校正)`。
///
/// 扫描 `s_t = a·s_{t-1} + b·x_t` 的跨元素依赖不能靠逐元素拆线程（每线程会各算一整遍），但它
/// 是**线性递推**，可以「切块 + 块级携带前缀」并行，与 [`metal_chunk_reduce`] 同属「切块 + 主机
/// 折叠」族，差异只在主机折叠从「求和/求最大」换成「携带前缀」且多一段校正内核：
///
/// 1. 局部扫描：每线程 `g` 对块 `[g·chunk, min(chunk, N-g·chunk))` 做基 0 扫描，写 `y`（块末态
///    即 `y[start+len-1]`，主机直接从这里读出，不另占缓冲）；
/// 2. 主机携带前缀：`C_0 = 0`、`C_{g+1} = a^len·C_g + y[start+len-1]`（只有 `ceil(N/chunk)` 项）；
/// 3. 校正：每线程 `g` 把携带态的衰减贡献 `C_g·a^(j+1)` 加回 `y[start+j]`（就地累加，`C` 是主机
///    注入的携带态缓冲）。
///
/// 衰减 `a` 由分派层从算子 `[consts]` 读出，机制层只按它拼校正内核的几何级数，不猜算子语义。
///
/// # Errors：签名不是扫描形（缺唯一只读输入 / 可写输出 / 长度标量）或标量缺样本取值、模板里
/// 找不到时间轴循环时返回说明。
pub fn metal_chunk_scan(
    kernel_source: &str,
    signature: &Signature,
    sample: &Sample,
    chunk: usize,
    decay: f32,
) -> diagnostic::Result<(String, String)> {
    let output = output_name(signature)?;
    let scalar = scalar_name(signature)?;
    let len = length_scalar(signature, sample)?;
    let local = chunk_scan_local(kernel_source, signature, sample, chunk, scalar)?;
    let correct = chunk_scan_correct(output, chunk, len, decay);
    Ok((local, correct))
}

/// 局部扫描内核：把模板里的时间轴循环头改写到块内 `[start, start+len)`，`tid` 透传进内部内核。
fn chunk_scan_local(
    kernel_source: &str,
    signature: &Signature,
    sample: &Sample,
    chunk: usize,
    scalar: &str,
) -> diagnostic::Result<String> {
    let kernel = signature.kernel();
    let mut body = msl_address_space(kernel_source);
    body = thread_param(&body, &kernel)?;
    body = inject_chunk_preamble(&body, &kernel, chunk, scalar)?;
    body = bind_chunk_loop(&body, scalar)?;
    let call = format!("{}, tid", serial_args(signature.params(), sample)?);
    let buffers = buffer_decls(signature);
    Ok(format!(
        "{body}\n\nkernel void kernel_main(\n{buffers},\n    uint3 tid [[thread_position_in_grid]]) {{\n    {kernel}({call});\n}}"
    ))
}

/// 校正内核：每线程 `g` 把携带态 `d_c[g]` 的衰减几何级数 `C·a^(j+1)` 加回 `y`（就地累加）。
/// `chunk`/`len`/`decay` 都是生成期常量，只有 `d_c`（只读）、`d_y`（就地读写）两块缓冲 + 线程下标。
/// 槽位按「输入后输出」：`d_c` 是输入在 `buffer(0)`、`d_y` 是输出在 `buffer(1)`，与 [`buffer_decls`]
/// 的同源约定一致。块尾用三元（不依赖 `min`），内核自包含、不引档案 preamble。
fn chunk_scan_correct(output: &str, chunk: usize, len: i64, decay: f32) -> String {
    format!(
        "kernel void kernel_main(\n    device float* d_c [[buffer(0)]],\n    device float* d_{output} [[buffer(1)]],\n    uint3 tid [[thread_position_in_grid]]) {{\n    int g = (int)tid.x;\n    int start = g * {chunk};\n    int rest = {len} - start;\n    int len = ({chunk} < rest) ? {chunk} : rest;\n    float c = d_c[g];\n    float pow = {decay}f;\n    for (int j = 0; j < len; ++j) {{\n        d_{output}[start + j] += c * pow;\n        pow *= {decay}f;\n    }}\n}}"
    )
}

/// 给内部内核函数体注入块头：`g`/`start`/`len` 三个局部量，供循环头改写与分块寻址引用。
fn inject_chunk_preamble(
    body: &str,
    kernel: &str,
    chunk: usize,
    scalar: &str,
) -> diagnostic::Result<String> {
    let marker = format!("void {kernel}(");
    let Some(signature_start) = body.find(&marker) else {
        return Err(Error::new(
            Code::Invalid,
            format!("找不到内核函数 `void {kernel}(`"),
        )
        .with("cutile::harness::metal_chunk_scan"));
    };
    let Some(close) = body[signature_start..].find(") {") else {
        return Err(Error::new(
            Code::Invalid,
            format!("内核函数 `{kernel}` 签名不完整"),
        )
        .with("cutile::harness::metal_chunk_scan"));
    };
    let brace = signature_start + close + 2;
    let preamble = format!(
        "    int g = (int)tid.x;\n    int start = g * {chunk};\n    int len = min({chunk}, {scalar} - start);\n"
    );
    let mut out = String::with_capacity(body.len() + preamble.len());
    out.push_str(&body[..=brace]);
    out.push_str(&preamble);
    out.push_str(&body[brace + 1..]);
    Ok(out)
}

/// 把以 `scalar` 为上界的串行循环头改写到块内边界：`for (int v = 0; v < {scalar}; ++v)` →
/// `for (int v = start; v < start + len; ++v)`。循环变量从模板读出，不对名字硬编码。
fn bind_chunk_loop(body: &str, scalar: &str) -> diagnostic::Result<String> {
    let bound = format!(" < {scalar}; ");
    let Some(bound_at) = body.find(&bound) else {
        return Err(Error::new(
            Code::Invalid,
            format!("模板里找不到以 `{scalar}` 为上界的串行循环"),
        )
        .with("cutile::harness::metal_chunk_scan"));
    };
    let prefix = &body[..bound_at];
    let Some(open) = prefix.rfind("for (int ") else {
        return Err(Error::new(
            Code::Invalid,
            "循环上界不在 `for (int …)` 头里",
        )
        .with("cutile::harness::metal_chunk_scan"));
    };
    let var = prefix[open + "for (int ".len()..]
        .split('=')
        .next()
        .unwrap_or("")
        .trim();
    let old_head = format!("for (int {var} = 0; {var} < {scalar}; ++{var})");
    let new_head = format!("for (int {var} = start; {var} < start + len; ++{var})");
    if !body.contains(&old_head) {
        return Err(Error::new(
            Code::Invalid,
            format!("模板里找不到串行循环 `{old_head}`"),
        )
        .with("cutile::harness::metal_chunk_scan"));
    }
    Ok(body.replace(&old_head, &new_head))
}

/// 生产态分块 RMSNorm：两段内核 + 主机折叠。
///
/// 1. **部分平方和**：复用 [`metal_chunk_reduce`]，把 `x` 切成块、每线程用 `_cutile_sumsq_f32`
///    归约一块、部分和写 `y[g]`（临时占 `y` 前 `G` 个槽）；
/// 2. **归一化**：每线程 `g` 用主机折叠出的 `rms`（`sqrt(Σx²/N + eps)`，由调用方算）归一化一块
///    （`y = gamma·x/rms`），覆盖 `y` 全行。
///
/// 归一化内核的缓冲序是「输入先、输出后」：`[rms, x, gamma, y]`，与 [`buffer_decls`] 同源约定
/// 一致；`rms` 是主机注入的 1 元素标量缓冲，`gamma`/`x` 是只读输入、`y` 是就地输出。
///
/// # Errors：签名缺长度标量或标量缺样本取值时返回说明。
pub fn metal_chunk_rmsnorm(
    kernel_source: &str,
    signature: &Signature,
    sample: &Sample,
    chunk: usize,
) -> diagnostic::Result<(String, String)> {
    let partial = metal_chunk_reduce(kernel_source, signature, sample, chunk, "_cutile_sumsq_f32")?;
    let len = length_scalar(signature, sample)?;
    let map = format!(
        "{}\n\nkernel void kernel_main(\n    device float* d_rms [[buffer(0)]],\n    device float* d_x [[buffer(1)]],\n    device float* d_gamma [[buffer(2)]],\n    device float* d_y [[buffer(3)]],\n    uint3 tid [[thread_position_in_grid]]) {{\n    int g = (int)tid.x;\n    int start = g * {chunk};\n    int rest = {len} - start;\n    int len = ({chunk} < rest) ? {chunk} : rest;\n    float rms = d_rms[0];\n    for (int j = 0; j < len; ++j) {{\n        d_y[start + j] = d_gamma[start + j] * d_x[start + j] / rms;\n    }}\n}}",
        msl_address_space(kernel_source),
    );
    Ok((partial, map))
}

/// 生产态分块 Softmax：三段内核 + 两次主机折叠。
///
/// 1. **部分最大值**：复用 [`metal_chunk_reduce`]，`_cutile_max_f32` 归约一块、部分最大写 `y[g]`；
/// 2. **部分指数和**：每线程 `g` 用主机折叠出的 `max` 做 `_cutile_exp_sum_f32(x, …, max)`，写 `y[g]`；
/// 3. **归一化**：每线程 `g` 用 `max`/`sum` 做 `_cutile_exp_norm_f32`，写 `y` 全行。
///
/// 后两段内核的缓冲序「输入先、输出后」：指数和 `[max, x, y]`、归一化 `[max, sum, x, y]`；
/// `max`/`sum` 是主机注入的 1 元素标量缓冲。
///
/// # Errors：签名缺长度标量或标量缺样本取值时返回说明。
pub fn metal_chunk_softmax(
    kernel_source: &str,
    signature: &Signature,
    sample: &Sample,
    chunk: usize,
) -> diagnostic::Result<(String, String, String)> {
    let max = metal_chunk_reduce(kernel_source, signature, sample, chunk, "_cutile_max_f32")?;
    let len = length_scalar(signature, sample)?;
    let preamble = msl_address_space(kernel_source);
    let expsum = format!(
        "{preamble}\n\nkernel void kernel_main(\n    device float* d_max [[buffer(0)]],\n    device float* d_x [[buffer(1)]],\n    device float* d_y [[buffer(2)]],\n    uint3 tid [[thread_position_in_grid]]) {{\n    int g = (int)tid.x;\n    int start = g * {chunk};\n    int rest = {len} - start;\n    int len = ({chunk} < rest) ? {chunk} : rest;\n    d_y[g] = _cutile_exp_sum_f32(d_x + start, len, 1, d_max[0]);\n}}"
    );
    let map = format!(
        "{preamble}\n\nkernel void kernel_main(\n    device float* d_max [[buffer(0)]],\n    device float* d_sum [[buffer(1)]],\n    device float* d_x [[buffer(2)]],\n    device float* d_y [[buffer(3)]],\n    uint3 tid [[thread_position_in_grid]]) {{\n    int g = (int)tid.x;\n    int start = g * {chunk};\n    int rest = {len} - start;\n    int len = ({chunk} < rest) ? {chunk} : rest;\n    _cutile_exp_norm_f32(d_y + start, d_x + start, len, 1, d_max[0], d_sum[0]);\n}}"
    );
    Ok((max, expsum, map))
}

/// 长度标量：签名里唯一非缓冲参数（扫描/归约/归一化的 `N`），取样本值。
fn length_scalar(signature: &Signature, sample: &Sample) -> diagnostic::Result<i64> {
    let scalar = scalar_name(signature)?;
    sample.scalar(scalar).ok_or(Error::new(
        Code::Invalid,
        format!("长度标量 `{scalar}` 没有样本取值"),
    ))
}

/// 唯一长度标量名（非缓冲参数）：归约/扫描/循环注入共用。
fn scalar_name(signature: &Signature) -> diagnostic::Result<&str> {
    signature
        .params()
        .iter()
        .find(|param| !param.ctype.is_buffer())
        .map(|param| param.name.as_str())
        .ok_or(Error::new(Code::Internal, "内核没有长度标量").with("cutile::harness::scalar_name"))
}

fn bind_grid_loops(body: String, grid: &Grid) -> diagnostic::Result<String> {
    const AXES: [&str; 3] = ["x", "y", "z"];
    let mut out = body;
    for (dim, (thread, scalar)) in grid.dims.iter().enumerate() {
        let axis = AXES.get(dim).ok_or(Error::new(
            Code::Invalid,
            format!("网格维 {} 超过 Metal 上限 3", grid.dims.len()),
        ))?;
        out = bind_grid_loop(&out, thread, scalar, axis)?;
    }
    Ok(out)
}

/// 给内部内核函数追加 `uint3 tid` 形参：`void {kernel}(...) {` → `void {kernel}(..., uint3 tid) {`，
/// 让循环头绑定能引用线程下标（`tid` 声明在 `kernel_main`，而循环在内部函数里，二者作用域不通）。
fn thread_param(body: &str, kernel: &str) -> diagnostic::Result<String> {
    let marker = format!("void {kernel}(");
    let Some(signature_start) = body.find(&marker) else {
        return Err(Error::new(
            Code::Invalid,
            format!("找不到内核函数 `void {kernel}(`"),
        )
        .with("cutile::harness::metal_loop_kernel"));
    };
    let Some(close) = body[signature_start..].find(") {") else {
        return Err(Error::new(
            Code::Invalid,
            format!("内核函数 `{kernel}` 签名不完整"),
        )
        .with("cutile::harness::metal_loop_kernel"));
    };
    let insert_at = signature_start + close;
    let mut out = String::with_capacity(body.len() + 12);
    out.push_str(&body[..insert_at]);
    out.push_str(", uint3 tid");
    out.push_str(&body[insert_at..]);
    Ok(out)
}

/// 把一个网格维的串行循环头绑定到线程下标：`for (int {thread} = 0; {thread} < {scalar}; ++{thread})`
/// → `for (int {thread} = (int)tid.{axis}; {thread} < (int)tid.{axis} + 1; ++{thread})`。循环只跑一次，
/// 循环变量即线程下标。找不到声明的循环时报错，避免静默生成串行语义的错误内核。
fn bind_grid_loop(
    body: &str,
    thread: &str,
    scalar: &str,
    axis: &str,
) -> diagnostic::Result<String> {
    let head = format!("for (int {thread} = 0; {thread} < {scalar}; ++{thread})");
    let bound = format!(
        "for (int {thread} = (int)tid.{axis}; {thread} < (int)tid.{axis} + 1; ++{thread})"
    );
    if !body.contains(&head) {
        return Err(Error::new(
            Code::Invalid,
            format!("模板里找不到网格循环 `{head}`"),
        )
        .with("cutile::harness::metal_loop_kernel"));
    }
    Ok(body.replace(&head, &bound))
}

/// `cutile_launch` compute kernel：按签名顺序声明 `device` 缓冲并原样透传给内核（内核指针已是
/// `device`，无需 thread 局部拷贝）；标量直接取样本值。槽位序号与驱动侧 `setBuffer:atIndex:` 同源。
fn metal_launch(
    kernel: &str,
    signature: &Signature,
    sample: &Sample,
) -> diagnostic::Result<String> {
    let call = launch_args(sample, signature.params())?;
    let buffers = signature
        .params()
        .iter()
        .filter(|param| param.ctype.is_buffer())
        .enumerate()
        .map(|(slot, param)| {
            format!(
                "device {} *d_{} [[buffer({slot})]]",
                param.ctype.c_name(),
                param.name
            )
        })
        .collect::<Vec<_>>()
        .join(",\n                          ");
    Ok(format!(
        "kernel void cutile_launch({buffers},\n                          uint tid [[thread_position_in_grid]]) {{\n    if (tid != 0) return;\n    {kernel}({call});\n}}"
    ))
}

/// host 驱动：把样本数据作为只读缓冲的初值，发射 `cutile_launch`，回读唯一可写输出的位模式。
/// 输出长度在生成期已知，直接嵌进回读循环；值与 stderr/stdout 口径与 C/CUDA harness 一致。
fn metal_driver(
    signature: &Signature,
    sample: &Sample,
    output: &str,
) -> diagnostic::Result<String> {
    let mut data = String::new();
    let mut allocs = String::new();
    let mut sets = String::new();
    let mut out_len = 0usize;
    for (slot, param) in signature
        .params()
        .iter()
        .filter(|p| p.ctype.is_buffer())
        .enumerate()
    {
        let buffer = sample
            .buffer(&param.name)
            .ok_or_else(|| Error::new(Code::Internal, "签名与样本已对齐（漂移）"))?;
        let base = param.ctype.c_name();
        let len = buffer.values.len();
        if param.mutable {
            out_len = len;
            allocs.push_str(&format!(
                "        id<MTLBuffer> b_{name} = [device newBufferWithLength:{len} * sizeof({base}) options:MTLResourceStorageModeShared];\n",
                name = param.name,
                len = len,
                base = base
            ));
        } else {
            data.push_str(&format!(
                "static const {base} init_{name}[] = {{{lit}}};\n",
                base = base,
                name = param.name,
                lit = c_literal(&buffer.values)
            ));
            allocs.push_str(&format!(
                "        id<MTLBuffer> b_{name} = [device newBufferWithBytes:init_{name} length:sizeof(init_{name}) options:MTLResourceStorageModeShared];\n",
                name = param.name
            ));
        }
        sets.push_str(&format!(
            "        [encoder setBuffer:b_{name} offset:0 atIndex:{slot}];\n",
            name = param.name,
            slot = slot
        ));
    }
    Ok(metal_driver_template(
        &data, &allocs, &sets, output, out_len,
    ))
}

/// 驱动正文模板：只拼字面与已算好的片段，不含循环与错误处理（那些留在 [`metal_driver`]）。
fn metal_driver_template(
    data: &str,
    allocs: &str,
    sets: &str,
    output: &str,
    out_len: usize,
) -> String {
    format!(
        r#"#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <stdio.h>
#include <stdlib.h>

{data}
int main(int argc, const char **argv) {{
    @autoreleasepool {{
        if (argc < 2) {{ fprintf(stderr, "usage: driver <metallib>\n"); return 2; }}
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) {{ fprintf(stderr, "本机没有 Metal 设备\n"); return 3; }}
        NSError *error = nil;
        NSURL *url = [NSURL fileURLWithPath:[NSString stringWithUTF8String:argv[1]]];
        id<MTLLibrary> library = [device newLibraryWithURL:url error:&error];
        if (!library) {{ fprintf(stderr, "metallib 装载失败：%s\n", error.localizedDescription.UTF8String); return 4; }}
        id<MTLFunction> function = [library newFunctionWithName:@"cutile_launch"];
        if (!function) {{ fprintf(stderr, "metallib 里没有 cutile_launch\n"); return 5; }}
        id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:function error:&error];
        if (!pipeline) {{ fprintf(stderr, "管线创建失败：%s\n", error.localizedDescription.UTF8String); return 6; }}
{allocs}        id<MTLCommandQueue> queue = [device newCommandQueue];
        id<MTLCommandBuffer> buffer = [queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [buffer computeCommandEncoder];
        [encoder setComputePipelineState:pipeline];
{sets}        [encoder dispatchThreads:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
        [encoder endEncoding];
        [buffer commit];
        [buffer waitUntilCompleted];
        if (buffer.error) {{ fprintf(stderr, "内核执行失败：%s\n", buffer.error.localizedDescription.UTF8String); return 7; }}
        const float *out = (const float *)b_{output}.contents;
        for (int i = 0; i < {out_len}; ++i) {{
            union {{ float f; unsigned int u; }} v;
            v.f = out[i];
            printf("%u\n", v.u);
        }}
    }}
    return 0;
}}
"#,
        data = data,
        allocs = allocs,
        sets = sets,
        output = output,
        out_len = out_len
    )
}

#[cfg(test)]
#[path = "../tests/harness_tests.rs"]
mod tests;
