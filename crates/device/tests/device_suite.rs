//! 后端/平台/程序层的契约套件装配根。
//!
//! `Cargo.toml` 是 `autotests = false`，`tests/` 下的文件必须由显式注册的根挂起才会编译。

pub use llm_device::tiled;

mod platform;
mod program;
