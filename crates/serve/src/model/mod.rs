//! 兼容旧路径：CPU 模型内核桥与装配宿主。

pub mod host;

pub use crate::kernels::register_cpu;
pub use crate::params::Params;
pub use host::CpuModel;
