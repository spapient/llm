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
    let device = kernel_source
        .replace(
            &format!("void {kernel}("),
            &format!("__device__ void {kernel}("),
        )
        // 模板里的 `static` 辅助函数（如 gemm 的 cutile_code、gelu 的 gelu_erf）也是设备端代码，
        // 默认会按 `__host__` 编译而被内核调用报错，统一补 `__device__`。
        .replace("static ", "__device__ static ");
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

/// 生产态 Metal 计算内核：把逐点（pointwise）串行内核包成按网格派发的 `kernel_main`。
///
/// 与 [`metal_harness`] 的单线程见证（入口 `cutile_launch`、`if (tid != 0) return`）不同，这里每个
/// 线程处理一个元素：网格索引 `n` 落到每个缓冲的 `+ n` 偏移、唯一的长度标量传 `1`。因此**只对
/// 逐点族成立**——所有缓冲按元素步长 1 索引、唯一标量是长度 `N`（`add` / `mul` / `scale` / `silu` /
/// `gelu` / `residual_add`）。gemv/gemm 的行跨步与分组标量、reduce 的跨元素归约、embed 的行 gather、
/// rmsnorm 的两趟归一化都要各自的 lowering，不能套这条。
///
/// 本函数以「标量恰好一项」为可查的边界：多标量签名（gemv/gemm）在此被拒，而不是静默生成错的内核；
/// 逐点之外的 `reduce` / `embed` / `rmsnorm` 由装配层按算子骨架路由到各自的 lowering。
///
/// # Errors：签名不是逐点形（标量不止长度一项）。
pub fn metal_grid_kernel(kernel_source: &str, signature: &Signature) -> diagnostic::Result<String> {
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
        .with("cutile::harness::metal_grid_kernel"));
    }
    let call = grid_args(signature.params());
    let buffers = buffer_decls(signature);
    Ok(format!(
        "{}\n\nkernel void kernel_main(\n{buffers},\n    uint n [[thread_position_in_grid]]) {{\n    {}({call});\n}}",
        msl_address_space(kernel_source),
        signature.kernel(),
    ))
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
    if grid.dims.len() > 3 {
        return Err(Error::new(
            Code::Invalid,
            format!("网格维 {} 超过 Metal 上限 3", grid.dims.len()),
        )
        .with("cutile::harness::metal_loop_kernel"));
    }
    if grid.dims.is_empty() {
        let call = serial_args(signature.params(), sample)?;
        let buffers = buffer_decls(signature);
        return Ok(format!(
            "{}\n\nkernel void kernel_main(\n{buffers}) {{\n    {}({call});\n}}",
            msl_address_space(kernel_source),
            signature.kernel(),
        ));
    }
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
    let input = signature
        .params()
        .iter()
        .find(|param| param.ctype.is_buffer() && !param.mutable)
        .map(|param| param.name.as_str())
        .ok_or_else(|| {
            Error::new(Code::Internal, "归约内核没有只读输入缓冲")
                .with("cutile::harness::metal_chunk_reduce")
        })?;
    let output = signature.output().ok_or_else(|| {
        Error::new(Code::Internal, "归约内核没有可写输出缓冲")
            .with("cutile::harness::metal_chunk_reduce")
    })?;
    let scalar = signature
        .params()
        .iter()
        .find(|param| !param.ctype.is_buffer())
        .map(|param| param.name.as_str())
        .ok_or_else(|| {
            Error::new(Code::Internal, "归约内核没有长度标量")
                .with("cutile::harness::metal_chunk_reduce")
        })?;
    let len = sample.scalar(scalar).ok_or_else(|| {
        Error::new(Code::Invalid, format!("归约长度标量 `{scalar}` 没有样本取值"))
            .with("cutile::harness::metal_chunk_reduce")
    })?;
    let buffers = buffer_decls(signature);
    Ok(format!(
        "{}\n\nkernel void kernel_main(\n{buffers},\n    uint3 tid [[thread_position_in_grid]]) {{\n    int g = (int)tid.x;\n    int start = g * {chunk};\n    int len = min({chunk}, {len} - start);\n    d_{output}[g] = {reduce_func}(d_{input} + start, len, 1);\n}}",
        msl_address_space(kernel_source),
    ))
}

fn bind_grid_loops(body: String, grid: &Grid) -> diagnostic::Result<String> {
    const AXES: [&str; 3] = ["x", "y", "z"];
    let mut out = body;
    for (dim, (thread, scalar)) in grid.dims.iter().enumerate() {
        out = bind_grid_loop(&out, thread, scalar, AXES[dim])?;
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
