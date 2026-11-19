//! 被测内核的基准封装：把一份 `.k` 内核与样本拼成可计时的 `main()`。
//!
//! 计量口径见 README §12.1：墙钟取样、warmup 不计入，批量校准在 C 侧、`min`/`p50` 在 Rust 侧。
//! 样本由 `op::model` 给出、实参顺序由 `op::sig` 给出：本模块只按名字与类型把缓冲渲染成 C 数组，
//! 不认识任何算子。同模块内的 [`cuda_harness`] 是同口径的 CUDA 语义对拍方言。

use diagnostic::{Code, Error};
use op::model::{Sample, Values};
use op::sig::{CType, Param, Signature};

use crate::dialect::{cuda_device, kernel_main, CUDA, Wrap};

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
pub(crate) fn output_name(signature: &Signature) -> diagnostic::Result<&str> {
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
pub(crate) fn read_input(signature: &Signature) -> diagnostic::Result<&str> {
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
pub(crate) fn launch_args(sample: &Sample, params: &[Param]) -> diagnostic::Result<String> {
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
// 生产态 lowering：后端无关的逐点/归约 lowering 与 CUDA 方言在此；Metal 方言见 `metal.rs`。
// ---------------------------------------------------------------------------

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
pub(crate) fn grid_kernel(
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
pub(crate) fn chunk_reduce(
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

/// 长度标量：签名里唯一非缓冲参数（扫描/归约/归一化的 `N`），取样本值。
pub(crate) fn length_scalar(signature: &Signature, sample: &Sample) -> diagnostic::Result<i64> {
    let scalar = scalar_name(signature)?;
    sample.scalar(scalar).ok_or(Error::new(
        Code::Invalid,
        format!("长度标量 `{scalar}` 没有样本取值"),
    ))
}

/// 唯一长度标量名（非缓冲参数）：归约/扫描/循环注入共用。
pub(crate) fn scalar_name(signature: &Signature) -> diagnostic::Result<&str> {
    signature
        .params()
        .iter()
        .find(|param| !param.ctype.is_buffer())
        .map(|param| param.name.as_str())
        .ok_or(Error::new(Code::Internal, "内核没有长度标量").with("cutile::harness::scalar_name"))
}

pub use crate::metal::{
    metal_chunk_reduce, metal_chunk_rmsnorm, metal_chunk_scan, metal_chunk_softmax,
    metal_grid_kernel, metal_harness, metal_loop_kernel,
};

#[cfg(test)]
#[path = "../tests/harness_tests.rs"]
mod tests;
