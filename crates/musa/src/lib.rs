//! MUSA 真机测量通道：把本仓流水线产出的设备程序送上 `mcc`，逐元素对拍并注入事件计时。
//!
//! 四步按数据流排列，每步只读写文件、不藏状态：
//!
//! 1. [`program`]：算子名 + 形状 → 设备源 + 参考值。生成走**声明式协商**（不要求本机执行证据），
//!    源文本由 `cutile` 的聚合给，host 驱动由 `engine::program` 按 `launch` 形态装——这条通道
//!    要回答的问题恰恰是「生产流水线在 MUSA 上是否成立」，自己再拼一份程序就答不了。
//! 2. [`Recipe`]：设备源 → `mcc` 命令行（车道决定开关）。
//! 3. [`timed`]：在首次发射与末次回读之间夹入事件计时循环。
//! 4. [`compare`]：程序按位打印的输出与参考值比对。
//!
//! 框架产不出的裸微基准放在 [`PROBES`]：那是给框架的数字一个外部参照，不是结论归档。
//!
//! **工具链要点**（MUSA 5.2 / `mcc`，全部来自真机实测）：
//!
//! - `mcc` 是 clang-14；普通 CUDA 语法的设备源要 `-x musa`。
//! - 生成的程序只写 `cuda*` 名字，靠 [`shim`] 的两个转发头编译；`cuda_fp16.h` 只是转发。
//! - 张量核要 `-D__MUSA_QY2_MMA__=1` 与 `--cuda-gpu-arch=mp_22`（默认 mp_21 没有 MMA），
//!   且 tf32 转换内建只在 `musa-5.2.0/include` 的完整头里声明。
//! - muBLAS 要 `-lmublas`。
//!
//! 硬件边界写在这里，免得每次重新发现：S4000 是 QY2/mp_22，**没有 `cp.async`**；
//! 5.2 工具链的 SIMT 调度相对 3.1 退化（旧树的 prefill 数无法复现）。
//!
//! **拥有**：四步编排（生成 / 编译 / 计时 / 对拍）、`mcc` 命令行与车道开关、计时注入的位置、
//! 逐元素对拍的口径，以及三条裸探针的源。**不拥有**：`cutile` 的协商与聚合、`engine` 的 host 装配、
//! `layout` 的排版——本成员只按它们的公开缝把一条链拼起来，不重写其中任何一段。
//! **依赖**：`cutile`（`execution` / `build`）、`engine`（`program`）、`op`（标量参考）、
//! `diagnostic`、`layout`。**被依赖**：无，只有自己的二进制。

#![deny(unsafe_code)]

mod bits;
mod compile;
mod emit;
mod probe;
mod timing;

pub use bits::{Outcome, compare};
pub use compile::{Recipe, shim};
pub use emit::{Job, Program, lines, program};
pub use probe::{PROBES, Probe};
pub use timing::timed;

/// 按名字取一支探针；未知名字返回 `None`（名字由命令行给，错误由入口报）。
pub fn probe(name: &str) -> Option<&'static Probe> {
    PROBES.iter().find(|probe| probe.name == name)
}

#[cfg(test)]
mod tests;
