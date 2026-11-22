//! Metal 真机算子套件的装配根。
//!
//! `Cargo.toml` 是 `autotests = false`，所以这些用例必须由一个显式注册的根文件统一挂起，
//! 否则 `tests/` 下的文件一个都不会被编译——绿是假的绿。
//!
//! - `harness`：真机夹具（设备/队列、上传回读、CPU 对拍、计时）。
//! - `backends` / `metal`：被测面。

use llm_device::executor::*;

pub use llm_device::tiled;

/// 后端装配入口：把 Metal / CUDA 后端注入注册表（幂等）。
mod backends {
    pub use llm_device::backends::install;
}

#[path = "metal_executor_harness.rs"]
mod harness;
mod metal;
