//! 设备运行时后端岛：提交契约、FFI 句柄、命令队列、缓冲、管线与执行器实现收在这里，不进上层编排。

#[path = "exec.rs"]
pub mod exec;

#[cfg(target_os = "macos")]
#[path = "metal.rs"]
pub mod metal;
