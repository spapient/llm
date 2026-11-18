//! 设备后端边界：Metal/CUDA 手写 FFI 与运行时四 trait 的实现落点（ADR 0035）。
//!
//! 这是全仓除 `storage::mmap` 之外唯一的越界 crate。依赖方向只指向运行时一侧三个 crate 的
//! trait——`engine::Executor`/`engine::Program`、`storage::Allocator`、`shard::Collective`——三者
//! 均不得反向持有本 crate 类型（ADR 0035 §3）。`engine::Device` 是数据 struct（厂商标签），不是
//! trait，边界只登记不实现。
//!
//! M3-1 第一刀只落「骨架 + 手写 FFI 首条符号」：Metal 设备探测在 [`metal`]（macOS 目标），CUDA
//! Driver API 在 `cuda`（Windows 目标，随 win 侧 CUDA 工程建立后接入）。真实设备内核与 trait 实现
//! 随后续刀填入，本层当前不铺 Metal/CUDA 内核（ADR 0035 §不做）。

// 全仓默认 `deny(unsafe_code)`；越界只出现在各边界模块（`metal`/`cuda`），本 crate 主体保持 deny。
#![deny(unsafe_code)]
#![deny(missing_docs)]

/// Metal 设备边界：手写 FFI 首条符号（仅 macOS 目标）。
#[cfg(target_os = "macos")]
pub mod metal;
