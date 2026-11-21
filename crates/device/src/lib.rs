//! 设备源 lowering 与厂商后端：把一条内核 body 落成**可编译的设备源**，再经厂商工具链跑起来。
//!
//! **拥有**：方言层（[`dialect`]，把「同一条内核 body 拼成不同后端」的语法差异收进一个 trait）、
//! 各算子的源生成（[`norm`] / [`scan`] / [`tiled`] / [`wmma`] / [`program`]）、三条厂商链
//! （[`backends`] 经 `Codegen` 出源 + `Driver` 跑源）与厂商编译器探测（`toolchain`），
//! 以及六平台执行合同（[`platform`]，缺设备只能 `OPEN` 的那一格）。
//! **不拥有**：序列状态、页账、准入与提交契约（那些在 `engine`）——本 crate **依赖** `engine`
//! 的 [`Executor`](engine::exec::Executor) 缝来落地设备执行体，方向单边，无反向依赖。
//! **依赖**：`cutile`（L3 机制层：`Backend`/`Source`/`Profile` 缝）、`op`（内核契约）、`prof`、`diagnostic`。
//!
//! # 为什么与 `engine` 分家
//!
//! `engine` 的自述职责是"Serving state machine：序列 / KV 页 / 准入调度 / 提交契约"，而设备源
//! lowering 是另一件事：它不持有任何会跨请求演进的状态，输入是内核契约与样本，输出是源文本与
//! 可执行产物。两者混在一个 crate 里时，`engine` 有 60% 的行数在做与状态机无关的字符串装配，
//! 单 crate 密度预算（≤8,000 行）被这类装配撑破。
//!
//! 分层口径：**契约在 `engine`，实现在本 crate**。`metal::executor` 实现的
//! [`Executor`](engine::exec::Executor) 是 `engine` 定义的缝，本 crate 只填实现，因此
//! `device → engine` 是唯一方向。
//!
//! # 三条厂商链互不共享状态
//!
//! Metal 是「设备源（`.metal`）+ host 驱动（`.m`）」两段文本，CUDA 是单段 `.cu`，昇腾是 CCE。
//! 这个差异收在 [`cutile::backend::Source`] 里，由各自的 `Driver::compile_run` 解包，不外泄到机制层；
//! 三条链共用且只有两段字面（基准几何、见证回读），留在 [`backends`] 父模块，其余共用的导入面
//! 同样收在父模块的前导里，兄弟文件只写 `use super::*`。
//!
//! # 越界岛
//!
//! [`metal::executor`] 是全仓四个 `unsafe` 岛之一（Metal FFI，同 `storage::mmap` 形态），
//! 全仓默认 `deny(unsafe_code)`，只有该文件内 `#![allow(unsafe_code)]`。岛的行数与位置由
//! `testkit` 的 `unsafe_islands_are_the_ones_the_adr_lists` 逐个点名，ADR 0035 是唯一出处。

#![deny(unsafe_code)]

mod ascend;
pub mod backends;
pub mod cuda;
mod dialect;
pub mod metal;
pub mod norm;
pub mod platform;
pub mod program;
pub mod scan;
pub mod tiled;
mod toolchain;
pub mod wmma;

pub use platform::{Abi, Arch, Host, Machine, Row, Standing, Why, audit, require, standing};
