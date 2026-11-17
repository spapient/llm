//! Metal 后端边界模块（ADR 0025 §1、ADR 0035 预留骨架）。
//!
//! # 越界白名单（本刀未打开）
//!
//! 未来可越过 Rust 安全边界的对象只收在本模块内，穷尽四类：裸指针解引用、Metal 设备缓冲 /
//! 命令队列 / 事件句柄、Metal FFI 调用、mmap 映射（ADR 0035 §1）。`#[allow(unsafe_code)]`
//! 的收敛点在本模块或边界函数；模块头注释须写明越界理由、指针生命周期 / 对齐 / 有效性 /
//! 线程归属不变量，以及对外暴露的安全类型（ADR 0035 §2/§4）。
//!
//! 本刀只落适配骨架：[`MetalBackend`] 持登记用 `Device` 标签与能力候选表；真实命令队列、
//! 设备缓冲映射、kernel launch 与事件同步留后续。入口仍按未挂载返回 `Unsupported`；不引入
//! Metal FFI crate、不写 `unsafe`，不暴露设备句柄类型。

use diagnostic::Result;
use engine::exec::{Device, Executor, Graph, Stream};
use storage::{Allocator, Buffer};

/// Metal 后端适配：持登记用设备标签（vendor 固定 `metal`），不持私有句柄。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MetalBackend {
    device: Device,
}

/// 候选挂载算子集：名字均在 `op::REGISTRY` 内；设备内核绑定尚未发生，仅作能力登记占位，
/// 不代表本后端已能执行这些算子。
pub const MOUNTED: &[&str] = &["gemv_f32", "softmax_f32"];

impl MetalBackend {
    /// 由架构标签构造；vendor 固定 `metal`，标签只作登记，不持有设备私有句柄。
    pub fn new(arch: impl Into<String>) -> Self {
        Self {
            device: Device::new("metal", arch),
        }
    }

    /// 候选挂载算子名（只读）。
    pub fn mounted(&self) -> &'static [&'static str] {
        MOUNTED
    }
}

impl Executor for MetalBackend {
    fn device(&self) -> &Device {
        &self.device
    }

    fn submit(&mut self, _graph: &Graph, _stream: Stream) -> Result<()> {
        Err(crate::unsupported("backend::metal::MetalBackend", "图提交"))
    }

    fn sync(&mut self, _stream: Stream) -> Result<()> {
        Err(crate::unsupported("backend::metal::MetalBackend", "流同步"))
    }
}

impl Allocator for MetalBackend {
    fn alloc(&mut self, _bytes: usize) -> Result<Buffer> {
        Err(crate::unsupported(
            "backend::metal::MetalBackend",
            "缓冲分配",
        ))
    }

    fn free(&mut self, _buffer: Buffer) {}

    fn copy_into(&self, _src: &[u8], _dst: &mut [u8]) -> Result<()> {
        Err(crate::unsupported(
            "backend::metal::MetalBackend",
            "显式拷贝",
        ))
    }
}
