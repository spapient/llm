//! 后端 codegen 方言表：同一条 kernel body 拼成不同后端的 `kernel_main`，差异全是语法。
//!
//! 机制层只按表插值、不按后端写 `match`、不单态化。加后端 = 加一行常量，与 `device::specs()`
//! 同一套「按闭集登记」的方式。Metal / CUDA 的 lowering（[`crate::metal`] / [`crate::harness`]）
//! 只从这里取方言，不各自持有语法差异。

use op::sig::Signature;

/// 后端包装差异表：同一条 kernel body 拼成不同后端的 `kernel_main`。
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
