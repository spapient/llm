//! 生产态设备 codegen 方言与后端无关的 lowering 原语。
//!
//! 后端差异收进 `Wrap` 方言表——同一条 kernel body 只拼一次、只换语法；`metal` / `cuda` 两条
//! lowering 只从这里取方言，不各自持有语法差异。骨架分派留在 `cutile` 的机制层，
//! 它把算子降成中性的 [`cutile::backend::Lower`] 描述，经 [`cutile::backend::Codegen::lower`] 缝交给
//! 这里落地成 MSL / CUDA——机制层不出现厂商字面量（L3 契约），vendor 语法只在此处（L6）。
//!
//! 跨后端的「op → 文本」渲染助手（`output_of` / `output_name` / `c_type` / `c_literal`）由
//! `cutile::harness` 公开，这里直接复用，不在两端各写一份。

use diagnostic::{Code, Error, Result};
use op::decl::Grid;
use op::model::{Sample, Values};
use op::sig::{Param, Signature};

use cutile::harness::{c_literal, c_type, output_name, output_of};

/// 后端包装差异表：同一条 kernel body 拼成不同后端的 `kernel_main`，差异全是语法。
pub(crate) struct Wrap {
    /// 把 `.k` 内核 body 转成该后端可编译的翻译单元（Metal 补 `device` 地址空间，CUDA 补 `__device__`）。
    pub(crate) prelude: fn(&str, &str) -> String,
    /// 缓冲声明段（按签名序；Metal 带 `[[buffer(slot)]]`，CUDA 不带）。
    pub(crate) buffers: fn(&Signature) -> String,
    /// `kernel_main` 声明关键字：`kernel` / `__global__`。
    pub(crate) decl: &'static str,
    /// 逐点网格线程形参（Metal `uint n [[thread_position_in_grid]]`；CUDA 空，用 `point_stmt` 语句）。
    pub(crate) point_param: &'static str,
    /// 逐点网格线程下标语句（CUDA `int n = blockIdx.x * blockDim.x + threadIdx.x;`；Metal 空）。
    pub(crate) point_stmt: &'static str,
    /// 分块/循环网格线程形参（Metal `uint3 tid [[thread_position_in_grid]]`；CUDA 空）。
    pub(crate) grid_param: &'static str,
    /// 分块/循环网格线程下标语句（CUDA `int g = blockIdx.x * blockDim.x + threadIdx.x;`；Metal 空）。
    pub(crate) grid_stmt: &'static str,
}

/// 生产态设备缓冲声明：按签名序 `device {ctype}* d_{name} [[buffer({slot})]]`，槽位序号与驱动侧
/// `setBuffer:atIndex:` 同源。逐点 / 串行 / 循环注入三条 lowering 共用，只写这一份。
pub(crate) fn buffer_decls(signature: &Signature) -> String {
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

/// 把翻译单元拼成 `kernel_main`：prelude 变换后的 body 接后端声明、缓冲段、线程形参与线程下标语句。
pub(crate) fn kernel_main(
    wrap: &Wrap,
    source: &str,
    kernel: &str,
    buffers: &str,
    tid_param: &str,
    tid_stmt: &str,
    body: &str,
) -> String {
    kernel_entry(
        wrap,
        source,
        kernel,
        buffers,
        (tid_param, tid_stmt),
        body,
        "kernel_main",
    )
}

/// [`kernel_main`] 的具名版本：多段设备源同处一个文件时，各段入口名必须不同（`kernel_partial` /
/// `kernel_expsum` / `kernel_norm` 等）。其余拼装规则与 [`kernel_main`] 完全一致，只有一处。
pub(crate) fn kernel_entry(
    wrap: &Wrap,
    source: &str,
    kernel: &str,
    buffers: &str,
    tid: (&str, &str),
    body: &str,
    entry: &str,
) -> String {
    let (tid_param, tid_stmt) = tid;
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
        "{}\n\n{} void {entry}(\n{buffers}{param}) {{\n{stmt}{body}\n}}",
        (wrap.prelude)(source, kernel),
        wrap.decl,
    )
}

/// Metal 前置：`.k` 的 C 指针补 `device` 地址空间（MSL 无默认地址空间）。
fn metal_prelude(text: &str, _kernel: &str) -> String {
    msl_address_space(text)
}

/// CUDA 缓冲声明：`{ctype}* d_{name}`（无地址空间、无槽位标签，指针即全局内存）。
pub(crate) fn cuda_buffers(signature: &Signature) -> String {
    signature
        .params()
        .iter()
        .filter(|param| param.ctype.is_buffer())
        .map(|param| format!("    {}* d_{}", param.ctype.c_name(), param.name))
        .collect::<Vec<_>>()
        .join(",\n")
}

/// MSL 没有默认地址空间，`.k` 的裸指针（签名 + 函数体局部）都要补 `device`——这些指针指向全局
/// 缓冲，`thread` 栈放不下生产形状（`REG` 上限 64 KiB）。`\u{1}` 先占住 `*`、最后统一回填，避免
/// 「先替换 const 版后，结果里的 `TYPE *` 又被非 const 规则二次命中」。
pub(crate) fn msl_address_space(text: &str) -> String {
    text.replace("const float *", "device const float\u{1}")
        .replace("const unsigned char *", "device const unsigned char\u{1}")
        .replace("const u8 *", "device const u8\u{1}")
        .replace("float *", "device float\u{1}")
        .replace("unsigned char *", "device unsigned char\u{1}")
        .replace("u8 *", "device u8\u{1}")
        .replace("\u{1}", " *")
}

/// CUDA 前置：`.k` 的内核函数与 `static` 辅助函数统一补 `__device__`（否则默认 `__host__`）。
pub(crate) fn cuda_device(text: &str, kernel: &str) -> String {
    text.replace(
        &format!("void {kernel}("),
        &format!("__device__ void {kernel}("),
    )
    .replace("static ", "__device__ static ")
}

pub(crate) const METAL: Wrap = Wrap {
    prelude: metal_prelude,
    buffers: buffer_decls,
    decl: "kernel",
    point_param: "uint n [[thread_position_in_grid]]",
    point_stmt: "",
    grid_param: "uint3 tid [[thread_position_in_grid]]",
    grid_stmt: "int g = (int)tid.x;",
};

pub(crate) const CUDA: Wrap = Wrap {
    prelude: cuda_device,
    buffers: cuda_buffers,
    decl: "__global__",
    point_param: "",
    point_stmt: "int n = blockIdx.x * blockDim.x + threadIdx.x;",
    grid_param: "",
    grid_stmt: "int g = blockIdx.x * blockDim.x + threadIdx.x;",
};

// ---------------------------------------------------------------------------
// 后端无关的 lowering 原语：Metal / CUDA 共用同一条 body，只换方言表。
// ---------------------------------------------------------------------------

/// 生产态逐点网格内核：把逐点（pointwise）串行内核包成按网格派发的 `kernel_main`，后端差异收进
/// [`Wrap`]。每个线程处理一个元素：网格索引 `n` 落到每个缓冲的 `+ n` 偏移、唯一的长度标量传 `1`。
/// 因此**只对逐点族成立**——所有缓冲按元素步长 1 索引、唯一标量是长度 `N`。
pub(crate) fn grid_kernel(
    kernel_source: &str,
    signature: &Signature,
    wrap: &Wrap,
    guard_len: Option<i64>,
) -> Result<String> {
    let scalars = signature
        .params()
        .iter()
        .filter(|param| !param.ctype.is_buffer())
        .count();
    if scalars != 1 {
        return Err(Error::new(
            Code::Invalid,
            format!(
                "`{}` 不是逐点签名（标量 {scalars} 项）：行跨步/归约 lowering 不在此列",
                signature.op(),
            ),
        )
        .with("engine::codegen::grid_kernel"));
    }
    let call = grid_args(signature.params());
    let buffers = (wrap.buffers)(signature);
    // 网格可大于元素数（块内多线程时末块不满）：带守卫时越界线程直接跳过。
    let body = match guard_len {
        Some(len) => format!(
            "    if (n < {len}) {{\n        {}({call});\n    }}",
            signature.kernel()
        ),
        None => format!("    {}({call});", signature.kernel()),
    };
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

/// 生产态分块归约内核（后端无关）：把长轴切成 `chunk` 元素一块，网格 = `ceil(N / chunk)`，每线程用
/// 设备 preamble 里的软件归约原语串行归约自己那一块，把**块级部分结果**写到输出缓冲的第 `g` 个槽。
/// 归约的跨元素依赖不能靠逐元素拆线程，只能「切块 + 主机折叠」。后端差异收进 [`Wrap`]。
pub(crate) fn chunk_reduce(
    kernel_source: &str,
    signature: &Signature,
    sample: &Sample,
    chunk: usize,
    reduce_func: &str,
    wrap: &Wrap,
) -> Result<String> {
    chunk_reduce_entry(
        kernel_source,
        signature,
        sample,
        chunk,
        reduce_func,
        wrap,
        "kernel_main",
    )
}

/// [`chunk_reduce`] 的具名版本：多段归一化 lowering 里部分归约段的入口名必须与其他段区分。
pub(crate) fn chunk_reduce_entry(
    kernel_source: &str,
    signature: &Signature,
    sample: &Sample,
    chunk: usize,
    reduce_func: &str,
    wrap: &Wrap,
    entry: &str,
) -> Result<String> {
    let input = read_input(signature)?;
    let output = output_name(signature)?;
    let len = length_scalar(signature, sample)?;
    let buffers = (wrap.buffers)(signature);
    let body = format!(
        "    int start = g * {chunk};\n    int len = min({chunk}, {len} - start);\n    d_{output}[g] = {reduce_func}(d_{input} + start, len, 1);"
    );
    Ok(kernel_entry(
        wrap,
        kernel_source,
        &signature.kernel(),
        &buffers,
        (wrap.grid_param, wrap.grid_stmt),
        &body,
        entry,
    ))
}

/// 长度标量：签名里唯一非缓冲参数（扫描/归约/归一化的 `N`），取样本值。
pub(crate) fn length_scalar(signature: &Signature, sample: &Sample) -> Result<i64> {
    let scalar = scalar_name(signature)?;
    sample.scalar(scalar).ok_or(Error::new(
        Code::Invalid,
        format!("长度标量 `{scalar}` 没有样本取值"),
    ))
}

/// 唯一长度标量名（非缓冲参数）：归约/扫描/循环注入共用。
pub(crate) fn scalar_name(signature: &Signature) -> Result<&str> {
    signature
        .params()
        .iter()
        .find(|param| !param.ctype.is_buffer())
        .map(|param| param.name.as_str())
        .ok_or(Error::new(Code::Internal, "内核没有长度标量").with("engine::codegen::scalar_name"))
}

/// 只读输入缓冲名（唯一非可写缓冲）：归约/扫描/循环注入三族 lowering 共用这一条查表。
pub(crate) fn read_input(signature: &Signature) -> Result<&str> {
    signature
        .params()
        .iter()
        .filter(|param| param.ctype.is_buffer())
        .find(|param| !param.mutable)
        .map(|param| param.name.as_str())
        .ok_or(
            Error::new(Code::Internal, "内核没有只读输入缓冲").with("engine::codegen::read_input"),
        )
}

/// launch 实参：缓冲传设备指针 `d_<name>`，标量取样本值。
pub(crate) fn launch_args(sample: &Sample, params: &[Param]) -> Result<String> {
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

// ---------------------------------------------------------------------------
// Metal(MSL) lowering：`[grid]`/`[chunk]` 声明式骨架 → 设备源，只换 [`Wrap`] 方言。
// ---------------------------------------------------------------------------

/// 拼出可在本机 GPU 上真实执行的 Metal 程序：`.k` 内核补 `device` 地址空间后，`cutile_launch`
/// 把它包成 compute kernel（`device` 缓冲 ↔ 线程局部数组），host 驱动负责把样本发射到 GPU、
/// 回读输出。返回 `(msl, driver)` 两段文本：`msl` 交给 `xcrun metal`，`driver` 交给 clang。
///
/// # Errors：缺输出缓冲、输出不是 f32、样本缺缓冲，或签名与样本不一致。
pub fn metal_harness(
    kernel_source: &str,
    signature: &Signature,
    sample: &Sample,
) -> Result<(String, String)> {
    let (output, _) = output_of(signature, sample)?;
    let msl = format!(
        "{}\n{}",
        msl_address_space(kernel_source),
        metal_launch(&signature.kernel(), signature, sample)?
    );
    Ok((msl, metal_driver(signature, sample, output)?))
}

/// 生产态 Metal 逐点网格内核（`grid_kernel` 的 Metal 方言）。
pub fn metal_grid_kernel(kernel_source: &str, signature: &Signature) -> Result<String> {
    grid_kernel(kernel_source, signature, &METAL, None)
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
) -> Result<String> {
    if grid.dims.is_empty() {
        return serial_kernel(kernel_source, signature, sample);
    }
    injected_kernel(kernel_source, signature, sample, grid)
}

/// 串行（零维网格）特例：不追加 `tid`、不绑循环，缓冲原样透传、标量取样本值。
fn serial_kernel(kernel_source: &str, signature: &Signature, sample: &Sample) -> Result<String> {
    let call = launch_args(sample, signature.params())?;
    let buffers = buffer_decls(signature);
    Ok(format!(
        "{}\n\nkernel void kernel_main(\n{buffers}) {{\n    {}({call});\n}}",
        msl_address_space(kernel_source),
        signature.kernel(),
    ))
}

/// 带线程下标的内核包裹：给内部内核透传 `tid`、声明 `uint3 tid [[thread_position_in_grid]]`，
/// 循环注入与分块扫描两条 lowering 共用（串行零维用 [`serial_kernel`]，不追加 `tid`）。
fn tid_kernel_main(body: &str, signature: &Signature, sample: &Sample) -> Result<String> {
    let call = format!("{}, tid", launch_args(sample, signature.params())?);
    let buffers = buffer_decls(signature);
    Ok(format!(
        "{body}\n\nkernel void kernel_main(\n{buffers},\n    uint3 tid [[thread_position_in_grid]]) {{\n    {}({call});\n}}",
        signature.kernel(),
    ))
}

/// 循环注入（非零网格）：给内部内核追加 `uint3 tid`、绑循环头到线程下标、调用点透传 `tid`。
fn injected_kernel(
    kernel_source: &str,
    signature: &Signature,
    sample: &Sample,
    grid: &Grid,
) -> Result<String> {
    let mut body = msl_address_space(kernel_source);
    body = thread_param(&body, &signature.kernel())?;
    body = bind_grid_loops(body, grid)?;
    tid_kernel_main(&body, signature, sample)
}

/// 生产态 Metal 分块归约内核（`chunk_reduce` 的 Metal 方言）。
pub fn metal_chunk_reduce(
    kernel_source: &str,
    signature: &Signature,
    sample: &Sample,
    chunk: usize,
    reduce_func: &str,
) -> Result<String> {
    chunk_reduce(kernel_source, signature, sample, chunk, reduce_func, &METAL)
}

/// 生产态分块扫描内核：把时间轴串行递推切成 `chunk` 元素一块，返回两段内核源 `(局部扫描, 校正)`。
///
/// 1. 局部扫描：每线程 `g` 对块 `[g·chunk, min(chunk, N-g·chunk))` 做基 0 扫描，写 `y`；
/// 2. 主机携带前缀：`C_0 = 0`、`C_{g+1} = a^len·C_g + y[start+len-1]`；
/// 3. 校正：每线程 `g` 把携带态的衰减贡献 `C_g·a^(j+1)` 加回 `y[start+j]`（就地累加）。
pub fn metal_chunk_scan(
    kernel_source: &str,
    signature: &Signature,
    sample: &Sample,
    chunk: usize,
    decay: f32,
) -> Result<(String, String)> {
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
) -> Result<String> {
    let kernel = signature.kernel();
    let mut body = msl_address_space(kernel_source);
    body = thread_param(&body, &kernel)?;
    body = inject_chunk_preamble(&body, &kernel, chunk, scalar)?;
    body = bind_chunk_loop(&body, scalar)?;
    tid_kernel_main(&body, signature, sample)
}

/// 分块 map 内核：每线程 `g` 处理一块 `[start, start+len)`，块头（`g`/`start`/`rest`/`len`）与
/// `kernel_main` 包裹只写这一份；扫描校正、RMSNorm 归一化、Softmax 指数和/归一化四段共用，
/// 差异只在缓冲声明与块内正文。
fn chunk_map_kernel(preamble: &str, buffers: &str, chunk: usize, len: i64, body: &str) -> String {
    let head = if preamble.is_empty() {
        String::new()
    } else {
        format!("{preamble}\n\n")
    };
    format!(
        "{head}kernel void kernel_main(\n{buffers},\n    uint3 tid [[thread_position_in_grid]]) {{\n    int g = (int)tid.x;\n    int start = g * {chunk};\n    int rest = {len} - start;\n    int len = ({chunk} < rest) ? {chunk} : rest;\n{body}\n}}"
    )
}

/// 校正内核：每线程 `g` 把携带态 `d_c[g]` 的衰减几何级数 `C·a^(j+1)` 加回 `y`（就地累加）。
/// 槽位按「输入后输出」：`d_c` 是输入在 `buffer(0)`、`d_y` 是输出在 `buffer(1)`。
fn chunk_scan_correct(output: &str, chunk: usize, len: i64, decay: f32) -> String {
    let buffers =
        format!("    device float* d_c [[buffer(0)]],\n    device float* d_{output} [[buffer(1)]]");
    let body = format!(
        "    float c = d_c[g];\n    float pow = {decay}f;\n    for (int j = 0; j < len; ++j) {{\n        d_{output}[start + j] += c * pow;\n        pow *= {decay}f;\n    }}"
    );
    chunk_map_kernel("", &buffers, chunk, len, &body)
}

/// 定位内核签名：返回 `(右括号位置, 函数体起始位置)`，供追加形参（[`thread_param`]）与
/// 注入块头（[`inject_chunk_preamble`]）共用，避免两处各写一遍 `void {kernel}(` 查找。
/// 体起始是 `) {` 之后的第一个字符——注入时把 `{` 一并留在前缀里。
fn kernel_span(body: &str, kernel: &str) -> Result<(usize, usize)> {
    let marker = format!("void {kernel}(");
    let Some(start) = body.find(&marker) else {
        return Err(
            Error::new(Code::Invalid, format!("找不到内核函数 `void {kernel}(`"))
                .with("engine::codegen::metal::kernel_span"),
        );
    };
    let Some(close) = body[start..].find(") {") else {
        return Err(
            Error::new(Code::Invalid, format!("内核函数 `{kernel}` 签名不完整"))
                .with("engine::codegen::metal::kernel_span"),
        );
    };
    Ok((start + close, start + close + 3))
}

/// 给内部内核函数体注入块头：`g`/`start`/`len` 三个局部量，供循环头改写与分块寻址引用。
pub(crate) fn inject_chunk_preamble(
    body: &str,
    kernel: &str,
    chunk: usize,
    scalar: &str,
) -> Result<String> {
    let (_, body_at) = kernel_span(body, kernel)?;
    let preamble = format!(
        "    int g = (int)tid.x;\n    int start = g * {chunk};\n    int len = min({chunk}, {scalar} - start);\n"
    );
    let mut out = String::with_capacity(body.len() + preamble.len());
    out.push_str(&body[..body_at]);
    out.push_str(&preamble);
    out.push_str(&body[body_at..]);
    Ok(out)
}

/// 把以 `scalar` 为上界的串行循环头改写到块内边界：`for (int v = 0; v < {scalar}; ++v)` →
/// `for (int v = start; v < start + len; ++v)`。循环变量从模板读出，不对名字硬编码。
pub(crate) fn bind_chunk_loop(body: &str, scalar: &str) -> Result<String> {
    let bound = format!(" < {scalar}; ");
    let Some(bound_at) = body.find(&bound) else {
        return Err(Error::new(
            Code::Invalid,
            format!("模板里找不到以 `{scalar}` 为上界的串行循环"),
        )
        .with("engine::codegen::metal::metal_chunk_scan"));
    };
    let prefix = &body[..bound_at];
    let Some(open) = prefix.rfind("for (int ") else {
        return Err(Error::new(Code::Invalid, "循环上界不在 `for (int …)` 头里")
            .with("engine::codegen::metal::metal_chunk_scan"));
    };
    let var = prefix[open + "for (int ".len()..]
        .split('=')
        .next()
        .unwrap_or("")
        .trim();
    let old_head = format!("for (int {var} = 0; {var} < {scalar}; ++{var})");
    let new_head = format!("for (int {var} = start; {var} < start + len; ++{var})");
    if !body.contains(&old_head) {
        return Err(
            Error::new(Code::Invalid, format!("模板里找不到串行循环 `{old_head}`"))
                .with("engine::codegen::metal::metal_chunk_scan"),
        );
    }
    Ok(body.replace(&old_head, &new_head))
}

/// 生产态分块 RMSNorm：两段内核 + 主机折叠。
///
/// 1. **部分平方和**：复用 [`metal_chunk_reduce`]，把 `x` 切成块、每线程用 `_cutile_sumsq_f32`
///    归约一块、部分和写 `y[g]`；
/// 2. **归一化**：每线程 `g` 用主机折叠出的 `rms`（`sqrt(Σx²/N + eps)`）归一化一块。
pub fn metal_chunk_rmsnorm(
    kernel_source: &str,
    signature: &Signature,
    sample: &Sample,
    chunk: usize,
) -> Result<(String, String)> {
    let partial = metal_chunk_reduce(kernel_source, signature, sample, chunk, "_cutile_sumsq_f32")?;
    let len = length_scalar(signature, sample)?;
    let map = chunk_map_kernel(
        &msl_address_space(kernel_source),
        "    device float* d_rms [[buffer(0)]],\n    device float* d_x [[buffer(1)]],\n    device float* d_gamma [[buffer(2)]],\n    device float* d_y [[buffer(3)]]",
        chunk,
        len,
        "    float rms = d_rms[0];\n    for (int j = 0; j < len; ++j) {{\n        d_y[start + j] = d_gamma[start + j] * d_x[start + j] / rms;\n    }}",
    );
    Ok((partial, map))
}

/// 生产态分块 Softmax：三段内核 + 两次主机折叠。
///
/// 1. **部分最大值**：复用 [`metal_chunk_reduce`]，`_cutile_max_f32` 归约一块、部分最大写 `y[g]`；
/// 2. **部分指数和**：每线程 `g` 用主机折叠出的 `max` 做 `_cutile_exp_sum_f32(x, …, max)`，写 `y[g]`；
/// 3. **归一化**：每线程 `g` 用 `max`/`sum` 做 `_cutile_exp_norm_f32`，写 `y` 全行。
pub fn metal_chunk_softmax(
    kernel_source: &str,
    signature: &Signature,
    sample: &Sample,
    chunk: usize,
) -> Result<(String, String, String)> {
    let max = metal_chunk_reduce(kernel_source, signature, sample, chunk, "_cutile_max_f32")?;
    let len = length_scalar(signature, sample)?;
    let preamble = msl_address_space(kernel_source);
    let expsum = chunk_map_kernel(
        &preamble,
        "    device float* d_max [[buffer(0)]],\n    device float* d_x [[buffer(1)]],\n    device float* d_y [[buffer(2)]]",
        chunk,
        len,
        "    d_y[g] = _cutile_exp_sum_f32(d_x + start, len, 1, d_max[0]);",
    );
    let map = chunk_map_kernel(
        &preamble,
        "    device float* d_max [[buffer(0)]],\n    device float* d_sum [[buffer(1)]],\n    device float* d_x [[buffer(2)]],\n    device float* d_y [[buffer(3)]]",
        chunk,
        len,
        "    _cutile_exp_norm_f32(d_y + start, d_x + start, len, 1, d_max[0], d_sum[0]);",
    );
    Ok((max, expsum, map))
}

fn bind_grid_loops(body: String, grid: &Grid) -> Result<String> {
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

/// 给内部内核函数追加 `uint3 tid` 形参：`void {kernel}(...) {` → `void {kernel}(..., uint3 tid) {`。
fn thread_param(body: &str, kernel: &str) -> Result<String> {
    thread_param_typed(body, kernel, "uint3 tid")
}

/// [`thread_param`] 的方言版：Metal 的线程下标是 `uint3`，CUDA 侧用 `int3`（由 kernel_main 从
/// `blockIdx` 装配，见 [`cuda_injected_kernel`]），其余改写规则一致。
pub(crate) fn thread_param_typed(body: &str, kernel: &str, param: &str) -> Result<String> {
    let (paren_at, _) = kernel_span(body, kernel)?;
    let mut out = String::with_capacity(body.len() + param.len() + 2);
    out.push_str(&body[..paren_at]);
    out.push_str(", ");
    out.push_str(param);
    out.push_str(&body[paren_at..]);
    Ok(out)
}

/// 把一个网格维的串行循环头绑定到线程下标：`for (int {thread} = 0; {thread} < {scalar}; ++{thread})`
/// → `for (int {thread} = (int)tid.{axis}; {thread} < (int)tid.{axis} + 1; ++{thread})`。
fn bind_grid_loop(body: &str, thread: &str, scalar: &str, axis: &str) -> Result<String> {
    let head = format!("for (int {thread} = 0; {thread} < {scalar}; ++{thread})");
    let bound =
        format!("for (int {thread} = (int)tid.{axis}; {thread} < (int)tid.{axis} + 1; ++{thread})");
    if !body.contains(&head) {
        return Err(
            Error::new(Code::Invalid, format!("模板里找不到网格循环 `{head}`"))
                .with("engine::codegen::metal::metal_loop_kernel"),
        );
    }
    Ok(body.replace(&head, &bound))
}

/// `cutile_launch` compute kernel：按签名顺序声明 `device` 缓冲并原样透传给内核；标量直接取样本值。
fn metal_launch(kernel: &str, signature: &Signature, sample: &Sample) -> Result<String> {
    let call = launch_args(sample, signature.params())?;
    let buffers = buffer_decls(signature);
    Ok(format!(
        "kernel void cutile_launch({buffers},\n                          uint tid [[thread_position_in_grid]]) {{\n    if (tid != 0) return;\n    {kernel}({call});\n}}"
    ))
}

/// host 驱动：把样本数据作为只读缓冲的初值，发射 `cutile_launch`，回读唯一可写输出的位模式。
fn metal_driver(signature: &Signature, sample: &Sample, output: &str) -> Result<String> {
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

// ---------------------------------------------------------------------------
// CUDA lowering：声明式骨架 → CUDA 源（逐点网格 / 分块归约）+ verify 单线程 harness。
// ---------------------------------------------------------------------------

/// 拼出可在 GPU 上真实执行的 CUDA 程序：`.k` 内核转 `__device__`，单线程 `__global__` 包装在
/// `<<<1,1>>>` 里调用，host `main` 负责分配/拷贝/回读——**语义对拍门**，性能另由 roofline 门承载。
///
/// # Errors：缺输出缓冲、输出不是 f32、样本缺缓冲，或签名与样本不一致。
pub fn cuda_harness(kernel_source: &str, signature: &Signature, sample: &Sample) -> Result<String> {
    let (output, values) = output_of(signature, sample)?;
    let kernel = signature.kernel();
    let params = signature.params();
    let args = launch_args(sample, params)?;
    let head = format!(
        "{}{}",
        cuda_prelude(kernel_source, &kernel, sample),
        cuda_wrapper(&kernel, params, &args)
    );
    let emit = cuda_launch("cutile_launch", "1", 1, &args);
    Driver::new(signature, sample, output, values.len()).assemble(&head, &emit)
}

/// `__global__` 包装的形参声明：缓冲是设备指针，用 `d_` 前缀与 host 侧 `cuda_host_buffers` 的
/// `cudaMalloc(&d_<name>)` 同源；标量直接传值。
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
        "\n__global__ void cutile_launch({}) {{\n    {kernel}({launch_args});\n}}\n",
        cuda_params(params)
    )
}

/// host 驱动的公共骨架：文件头 → `main` → 缓冲分配 → 一次派发 → 回读 → 释放 → 返回。
///
/// 六个驱动（单线程语义门、逐点网格、循环注入、分块归约、分块归一化、分块扫描）只在「文件头」
/// 与「派发那一行」上不同；同步报错、回读与释放清单必须逐字同源——各写一份时，加一个 include
/// 就会漏掉其中一族。
pub(crate) struct Driver<'a> {
    signature: &'a Signature,
    sample: &'a Sample,
    output: &'a str,
    output_len: usize,
}

impl<'a> Driver<'a> {
    /// 由签名、样本与输出缓冲长度（分块归约取部分结果数，其余取样本长度）装配。
    pub(crate) fn new(
        signature: &'a Signature,
        sample: &'a Sample,
        output: &'a str,
        output_len: usize,
    ) -> Self {
        Self {
            signature,
            sample,
            output,
            output_len,
        }
    }

    /// 拼完整程序：`head` 是文件头（include + 身份注释 + 设备源，方言由调用方给），`emit` 是派发那一行。
    pub(crate) fn assemble(&self, head: &str, emit: &str) -> Result<String> {
        let mut text = String::from(head);
        text.push_str("\nint main(void) {\n");
        text.push_str(&cuda_host_buffers_grid(
            self.sample,
            self.signature.params(),
            self.output,
            self.output_len,
        )?);
        text.push_str(emit);
        text.push_str(&cuda_readback(self.output, self.output_len));
        text.push_str(&cuda_release(self.signature));
        text.push_str("    return 0;\n}\n");
        Ok(text)
    }
}

/// 一次派发 + 同步失败即报错：内核名、网格、线程数与实参由调用方给，报错那一行只有这一份。
pub(crate) fn cuda_launch(kernel: &str, grid: &str, threads: usize, args: &str) -> String {
    format!(
        "    {kernel}<<<{grid}, {threads}>>>({args});\n\
         \x20   cudaError_t error = cudaDeviceSynchronize();\n\
         \x20   if (error != cudaSuccess) {{ fprintf(stderr, \"cuda: %s\\n\", cudaGetErrorString(error)); return 1; }}\n"
    )
}

/// 结果回读：设备 → 主机，再按 `float` 位模式逐元素打印。
pub(crate) fn cuda_readback(output: &str, len: usize) -> String {
    format!(
        "    cudaMemcpy(h_{output}, d_{output}, {len} * sizeof(float), cudaMemcpyDeviceToHost);\n\
         \x20   for (int i = 0; i < {len}; ++i) {{\n\
         \x20       union {{ float f; unsigned int u; }} v; v.f = h_{output}[i];\n\
         \x20       printf(\"%u\\n\", v.u);\n\
         \x20   }}\n"
    )
}

/// 拼出可在 GPU 上真实执行的 CUDA 网格程序：`kernel_main<<<grid,1>>>` 发射 + host 分配/拷贝/回读。
/// 与 [`cuda_harness`] 同口径，但发射是生产态网格（并行），输出缓冲长度由 `output_len` 显式给出——
/// 分块归约的输出是 `ceil(N/chunk)` 个部分结果，大于签名的单元素 `y`。
pub fn cuda_grid_driver(
    device_source: &str,
    signature: &Signature,
    sample: &Sample,
    grid: usize,
    output_len: usize,
) -> Result<String> {
    let output = output_name(signature)?;
    let emit = cuda_launch("kernel_main", &grid.to_string(), 1, &buffer_args(signature));
    let head = cuda_prologue(device_source);
    Driver::new(signature, sample, output, output_len).assemble(&head, &emit)
}

/// host 侧缓冲声明、设备分配与只读入向拷贝；输出缓冲长度用显式 `output_len`（分块归约的部分
/// 结果数），而非样本里的单元素。
pub(crate) fn cuda_host_buffers_grid(
    sample: &Sample,
    params: &[Param],
    output: &str,
    output_len: usize,
) -> Result<String> {
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

/// 生产态 CUDA 逐点网格内核（`grid_kernel` 的 CUDA 方言）。
pub fn cuda_grid_kernel(
    kernel_source: &str,
    signature: &Signature,
    sample: &Sample,
) -> Result<String> {
    grid_kernel(
        kernel_source,
        signature,
        &CUDA,
        Some(length_scalar(signature, sample)?),
    )
}

/// `kernel_main` 的完整形参表：缓冲 + 标量（模板里的 `M` / `N` / `K` 是运行时值，循环注入与
/// 串行都要把它们透传给内部内核）。
pub(crate) fn cuda_launch_params(signature: &Signature) -> String {
    let scalars = signature
        .params()
        .iter()
        .filter(|param| !param.ctype.is_buffer())
        .map(|param| format!("{} {}", param.ctype.c_name(), param.name))
        .collect::<Vec<_>>()
        .join(", ");
    let buffers = cuda_buffers(signature);
    if scalars.is_empty() {
        buffers
    } else {
        format!("{buffers},\n    {scalars}")
    }
}

/// 逐点算子的 host 驱动：块内 256 线程、网格 `ceil(N/256)`，分配/回读与 [`cuda_grid_driver`]
/// 同口径（输出长度取样本值）。kernel 侧由 [`cuda_grid_kernel`] 的守卫挡住末块多余线程。
///
/// # Errors：缺输出缓冲、输出不是 f32、样本缺缓冲时返回说明。
pub fn cuda_point_driver(
    device_source: &str,
    signature: &Signature,
    sample: &Sample,
) -> Result<String> {
    let (output, values) = output_of(signature, sample)?;
    let blocks = values.len().div_ceil(POINT_THREADS);
    let emit = cuda_launch(
        "kernel_main",
        &blocks.to_string(),
        POINT_THREADS,
        &buffer_args(signature),
    );
    let head = cuda_head(
        "#include <cuda_runtime.h>\n#include <cstdio>\n",
        device_source,
    );
    Driver::new(signature, sample, output, values.len()).assemble(&head, &emit)
}

/// 逐点驱动每块线程数（128 线程下 16K 元素实测 -12%）。
const POINT_THREADS: usize = 256;

/// 生产态 CUDA 循环注入网格内核（`[grid]` 骨架的 CUDA 方言）：gemv 一维、gemm 二维共用。
///
/// 与 Metal 版同一条文本改写（循环头绑到线程下标、正文跨步原样保留），差异只有三处语法：
/// - 内部内核追加的是 `int3 tid`；
/// - `kernel_main` 里用 `blockIdx` 装配 `tid`；
/// - 设备函数用 `__device__` 而非 `device` 地址空间。
///
/// 空网格（`dims` 为空）退化为串行：不追加 `tid`、不绑循环、缓冲与标量原样透传。
///
/// # Errors：模板里找不到声明的循环、签名与样本不一致时返回说明。
pub fn cuda_loop_kernel(
    kernel_source: &str,
    signature: &Signature,
    sample: &Sample,
    grid: &Grid,
) -> Result<String> {
    if grid.dims.is_empty() {
        return cuda_serial_kernel(kernel_source, signature, sample);
    }
    cuda_injected_kernel(kernel_source, signature, sample, grid)
}

/// CUDA 串行（零维网格）：不追加 `tid`，一次调用跑完整条循环。
fn cuda_serial_kernel(
    kernel_source: &str,
    signature: &Signature,
    sample: &Sample,
) -> Result<String> {
    let call = launch_args(sample, signature.params())?;
    let buffers = cuda_launch_params(signature);
    Ok(format!(
        "{}\n\n__global__ void kernel_main(\n{buffers}) {{\n    {}({call});\n}}",
        cuda_device(kernel_source, &signature.kernel()),
        signature.kernel(),
    ))
}

/// CUDA 循环注入：内部内核追加 `int3 tid`、绑循环头，`kernel_main` 用 `blockIdx` 装配 `tid` 并透传。
fn cuda_injected_kernel(
    kernel_source: &str,
    signature: &Signature,
    sample: &Sample,
    grid: &Grid,
) -> Result<String> {
    let mut body = cuda_device(kernel_source, &signature.kernel());
    body = thread_param_typed(&body, &signature.kernel(), "int3 tid")?;
    body = bind_grid_loops(body, grid)?;
    let call = format!("{}, tid", launch_args(sample, signature.params())?);
    let buffers = cuda_launch_params(signature);
    let init = "    int3 tid; tid.x = blockIdx.x; tid.y = blockIdx.y; tid.z = blockIdx.z;\n";
    Ok(format!(
        "{body}\n\n__global__ void kernel_main(\n{buffers}) {{\n{init}    {}({call});\n}}",
        signature.kernel(),
    ))
}

/// 生产态 CUDA 循环注入的 host 驱动：按 `dims` 组成 `dim3` 网格发射 `kernel_main`，
/// 分配/拷贝/回读与 [`cuda_grid_driver`] 同一口径（输出长度取样本值）。
///
/// # Errors：缺输出缓冲、输出不是 f32、样本缺缓冲时返回说明。
pub fn cuda_loop_driver(
    device_source: &str,
    signature: &Signature,
    sample: &Sample,
    dims: &[usize],
) -> Result<String> {
    let (output, values) = output_of(signature, sample)?;
    let grid = cuda_grid(dims);
    let args = launch_args(sample, signature.params())?;
    let emit = cuda_launch(
        "kernel_main",
        &format!("dim3({}, {}, {})", grid[0], grid[1], grid[2]),
        1,
        &args,
    );
    let head = cuda_prologue(device_source);
    Driver::new(signature, sample, output, values.len()).assemble(&head, &emit)
}

/// 文件头一行 include 段 + 设备源码；设备源码缺尾换行时补一个，免得 `main` 粘在最后一行上。
pub(crate) fn cuda_head(includes: &str, device_source: &str) -> String {
    let mut text = String::from(includes);
    text.push_str(device_source);
    if !device_source.ends_with('\n') {
        text.push('\n');
    }
    text
}

/// CUDA 前导：运行时与 stdio 两行 include + 设备源。
fn cuda_prologue(device_source: &str) -> String {
    cuda_head(
        "#include <cuda_runtime.h>\n#include <cstdio>\n",
        device_source,
    )
}

/// `[grid]` 求出的各维大小 → `dim3` 的三维；缺的维保持 1（`dim3` 的缺省就是 1）。
fn cuda_grid(dims: &[usize]) -> [usize; 3] {
    let mut grid = [1usize; 3];
    for (slot, size) in grid.iter_mut().zip(dims) {
        *slot = *size;
    }
    grid
}

/// 启动实参：只传缓冲（形状是编译期常量，不进内核形参）。
pub(crate) fn buffer_args(signature: &Signature) -> String {
    signature
        .params()
        .iter()
        .filter(|param| param.ctype.is_buffer())
        .map(|param| format!("d_{}", param.name))
        .collect::<Vec<_>>()
        .join(", ")
}

/// 收尾：释放每个缓冲参数对应的设备端指针（只释放真分配过的，标量是值传递）。
pub(crate) fn cuda_release(signature: &Signature) -> String {
    let mut text = String::new();
    for param in signature
        .params()
        .iter()
        .filter(|param| param.ctype.is_buffer())
    {
        text.push_str(&format!("    cudaFree(d_{});\n", param.name));
    }
    text
}

/// 生产态 CUDA 分块归约内核（`chunk_reduce` 的 CUDA 方言）。
pub fn cuda_chunk_reduce(
    kernel_source: &str,
    signature: &Signature,
    sample: &Sample,
    chunk: usize,
    reduce_func: &str,
) -> Result<String> {
    chunk_reduce(kernel_source, signature, sample, chunk, reduce_func, &CUDA)
}
