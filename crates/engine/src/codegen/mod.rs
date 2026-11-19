//! 生产态设备 codegen 方言与后端无关的 lowering 原语。
//!
//! 后端差异收进 [`Wrap`] 方言表——同一条 kernel body 只拼一次、只换语法；Metal / CUDA 的 lowering
//! （[`metal`] / [`cuda`]）只从这里取方言，不各自持有语法差异。骨架分派留在 `cutile` 的机制层，
//! 它把算子降成中性的 [`cutile::backend::Lower`] 描述，经 [`cutile::backend::Codegen::lower`] 缝交给
//! 这里落地成 MSL / CUDA——机制层不出现厂商字面量（L3 契约），vendor 语法只在此处（L6）。
//!
//! 跨后端的「op → 文本」渲染助手（`output_of` / `output_name` / `c_type` / `c_literal`）由
//! `cutile::harness` 公开，这里直接复用，不在两端各写一份。

use diagnostic::{Code, Error, Result};
use op::model::Sample;
use op::sig::{Param, Signature};

use cutile::harness::output_name;

pub mod cuda;
pub mod metal;

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
pub(crate) fn length_scalar(signature: &Signature, sample: &Sample) -> Result<i64> {
    let scalar = scalar_name(signature)?;
    sample
        .scalar(scalar)
        .ok_or(Error::new(Code::Invalid, format!("长度标量 `{scalar}` 没有样本取值")))
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
        .ok_or(Error::new(Code::Internal, "内核没有只读输入缓冲").with("engine::codegen::read_input"))
}

/// launch 实参：缓冲传设备指针 `d_<name>`，标量取样本值。
pub(crate) fn launch_args(sample: &Sample, params: &[Param]) -> Result<String> {
    let mut args = Vec::with_capacity(params.len());
    for param in params {
        if param.ctype.is_buffer() {
            args.push(format!("d_{}", param.name));
        } else {
            let value = sample.scalar(&param.name).ok_or_else(|| {
                Error::new(Code::Internal, format!("标量实参 `{}` 没有取值", param.name))
            })?;
            args.push(value.to_string());
        }
    }
    Ok(args.join(", "))
}
