//! 厂商语法面：把「同一条内核 body 拼成不同后端」的语法差异收成一组方法与常量。
//!
//! 这一层是一个 trait，Metal 与 CUDA 两个 vendor 各给一份**手写实现**（`Metal` / `Cuda` 单元
//! 结构体）。trait 自身只声明**厂商差异面**，vendor 字面量（Metal 的 `device` / `kernel`、
//! CUDA 的 `__device__` / `__global__`）都住在这一个文件里；`crate::codegen` 里的 lowering 原语
//! 只面向 trait，不再持有任何厂商语法。
//!
//! 三条 lowering（逐点 / 循环注入 / 分块归约）只在「地址空间 / 设备函数 / 缓冲声明 / 入口
//! 关键字 / 线程形参与线程下标语句」这八处有 vendor 差异；循环注入与分块扫描还共用「内部内核
//! 追加的线程下标形参类型」与「`kernel_main` 里装配线程下标的语句」两处，trait 一并声明。

use op::sig::Param;

/// 厂商语法面：后端无关的 lowering 原语里需要厂商语法的那几处，抽成方法 + 关联常量。
///
/// 每个成员只回答一个问题：
///
/// - [`address_space`](Dialect::address_space)：给一条 C 声明补地址空间限定词（Metal 在指针类型
///   前插 `device`；CUDA 恒等）；
/// - [`device_fn`](Dialect::device_fn)：给设备函数名返回限定词（CUDA `__device__`；Metal 恒等）；
/// - [`buffer_decl`](Dialect::buffer_decl)：渲染一条缓冲形参声明（Metal 带 `[[buffer(slot)]]`）；
/// - 关联常量们：入口关键字、逐点 / 网格的线程形参与线程下标语句，以及循环注入 / 分块扫描共用
///   的线程下标形参与装配语句。
pub(crate) trait Dialect {
    /// 给一条 C 声明补地址空间：Metal 在指针类型前插 `device`（`\u{1}` 哨兵先占住 `*`），CUDA
    /// 原样返回。限定词为空即恒等，所以 CUDA 的「无地址空间」与 Metal 的「六型规则」共用同一份
    /// 调用代码。
    fn address_space(decl: &str) -> String;

    /// 设备函数限定词：CUDA 返回 `__device__`，Metal 返回空串（无操作）。
    fn device_fn(name: &str) -> String;

    /// 渲染一条缓冲形参声明；`slot` 是驱动侧 `setBuffer:atIndex:` / 拷贝的槽位序。
    fn buffer_decl(param: &Param, slot: usize) -> String;

    /// 内核入口关键字：Metal `kernel`，CUDA `__global__`。
    const ENTRY: &'static str;
    /// 逐点网格线程形参（Metal `uint n [[thread_position_in_grid]]`；CUDA 空，用 [`Dialect::POINT_STMT`]）。
    const POINT_PARAM: &'static str;
    /// 逐点网格线程下标语句（CUDA 装配 `blockIdx` / `threadIdx`；Metal 空）。
    const POINT_STMT: &'static str;
    /// 分块 / 循环网格线程形参（Metal `uint3 tid [[thread_position_in_grid]]`；CUDA 空）。
    const GRID_PARAM: &'static str;
    /// 分块 / 循环网格线程下标语句（CUDA 装配；Metal `int g = (int)tid.x;`）。
    const GRID_STMT: &'static str;
    /// 循环注入 / 分块扫描里追加给内部内核的线程下标形参（Metal `uint3 tid`；CUDA `int3 tid`）。
    const THREAD_PARAM: &'static str;
    /// `kernel_main` 里装配线程下标的语句：CUDA 从 `blockIdx` 拼 `int3 tid`；Metal 空（线程下标
    /// 由形参 [`Dialect::GRID_PARAM`] 提供）。
    const TID_INIT: &'static str;
}

/// Metal 方言标记：地址空间用 `device`、入口用 `kernel`，线程下标由 `thread_position_in_grid`
/// 形参提供。
pub(crate) struct Metal;

impl Dialect for Metal {
    fn address_space(decl: &str) -> String {
        format!("device {decl}").replace(" *", "\u{1}")
    }

    fn device_fn(_name: &str) -> String {
        String::new()
    }

    fn buffer_decl(param: &Param, slot: usize) -> String {
        format!(
            "    device {}* d_{} [[buffer({})]]",
            param.ctype.c_name(),
            param.name,
            slot
        )
    }

    const ENTRY: &'static str = "kernel";
    const POINT_PARAM: &'static str = "uint n [[thread_position_in_grid]]";
    const POINT_STMT: &'static str = "";
    const GRID_PARAM: &'static str = "uint3 tid [[thread_position_in_grid]]";
    const GRID_STMT: &'static str = "int g = (int)tid.x;";
    const THREAD_PARAM: &'static str = "uint3 tid";
    const TID_INIT: &'static str = "";
}

/// CUDA 方言标记：无地址空间、设备函数用 `__device__`、入口用 `__global__`，线程下标在
/// `kernel_main` 内由 `blockIdx` 装配。
pub(crate) struct Cuda;

impl Dialect for Cuda {
    fn address_space(decl: &str) -> String {
        decl.to_owned()
    }

    fn device_fn(_name: &str) -> String {
        "__device__".to_owned()
    }

    fn buffer_decl(param: &Param, _slot: usize) -> String {
        format!("    {}* d_{}", param.ctype.c_name(), param.name)
    }

    const ENTRY: &'static str = "__global__";
    const POINT_PARAM: &'static str = "";
    const POINT_STMT: &'static str = "int n = blockIdx.x * blockDim.x + threadIdx.x;";
    const GRID_PARAM: &'static str = "";
    const GRID_STMT: &'static str = "int g = blockIdx.x * blockDim.x + threadIdx.x;";
    const THREAD_PARAM: &'static str = "int3 tid";
    const TID_INIT: &'static str =
        "    int3 tid; tid.x = blockIdx.x; tid.y = blockIdx.y; tid.z = blockIdx.z;\n";
}
