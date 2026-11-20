//! 厂商语法面：把「同一条内核 body 拼成不同后端」的语法差异收成一组方法与常量。
//!
//! 与 [`crate::codegen::Wrap`] 那张「字段表 + 函数指针」的方言表不同，这一层是一个 trait，
//! 每个 vendor 的实现由 `#[annotation::dialect(..)]` 注解展开（见 `crate::codegen` 里的
//! `Metal` / `Cuda`）。trait 自身只声明**厂商差异面**，不出现任何厂商字面量：Metal 的
//! `device` / CUDA 的 `__device__` 都住在注解里，生成代码用绝对路径 `::llm_engine::dialect`
//! 指回这里，因此注解 crate 对 engine 没有依赖（与 `#[annotation::model]` 硬编码 `::llm_op`
//! 同一口径）。
//!
//! 三条 lowering（逐点 / 循环注入 / 分块归约）只在「地址空间 / 设备函数 / 缓冲声明 / 入口
//! 关键字 / 线程形参与线程下标语句」这八处有 vendor 差异，trait 就只声明这八处。

use op::sig::Param;

/// 厂商语法面：后端无关的 lowering 原语里需要厂商语法的那几处，抽成方法 + 关联常量。
///
/// 每个成员只回答一个问题，vendor 字面量由 `#[annotation::dialect(..)]` 展开的实现给出：
///
/// - [`address_space`](Dialect::address_space)：给一条 C 声明补地址空间限定词（Metal 在指针类型
///   前插 `device`；CUDA 恒等）；
/// - [`device_fn`](Dialect::device_fn)：给设备函数名返回限定词（CUDA `__device__`；Metal 恒等）；
/// - [`buffer_decl`](Dialect::buffer_decl)：渲染一条缓冲形参声明（Metal 带 `[[buffer(slot)]]`）；
/// - 关联常量们：入口关键字、逐点 / 网格的线程形参与线程下标语句。
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
}
