//! [`Allocator`] trait 与 [`CpuAllocator`]。

use diagnostic::{Code, Error, Result};

use crate::Buf;

/// 缓冲分配器：分配、释放、显式拷贝。
pub trait Allocator {
    /// 分配 `bytes` 字节并清零；`0` 得到空缓冲。
    fn alloc(&mut self, bytes: usize) -> Result<Buf>;

    /// 归还一段缓冲。RAII 下向量 drop 即释放，显式归还点为未来设备回收预留钩子。
    fn free(&mut self, buf: Buf);

    /// 把 `src` 显式拷进 `dst`；两端长度必须一致，否则 `Invalid`。
    fn copy_into(&self, src: &[u8], dst: &mut [u8]) -> Result<()>;
}

/// CPU 内存分配器：安全 `Vec<u8>`，不含 `unsafe`。设备后端在边界 crate 实现同一 trait。
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct CpuAllocator;

impl CpuAllocator {
    /// 无状态构造。
    pub fn new() -> Self {
        Self
    }
}

impl Allocator for CpuAllocator {
    fn alloc(&mut self, bytes: usize) -> Result<Buf> {
        let mut owned = Vec::new();
        owned.try_reserve_exact(bytes).map_err(|_| {
            Error::new(
                Code::Invalid,
                format!("分配 {bytes} 字节：超出可用内存或地址空间"),
            )
            .with("storage::CpuAllocator::alloc")
        })?;
        owned.resize(bytes, 0);
        Ok(Buf::from_bytes(owned))
    }

    fn free(&mut self, _buf: Buf) {}

    fn copy_into(&self, src: &[u8], dst: &mut [u8]) -> Result<()> {
        if src.len() != dst.len() {
            return Err(Error::new(
                Code::Invalid,
                format!(
                    "拷贝长度不符：源 {} 字节，目标 {} 字节",
                    src.len(),
                    dst.len()
                ),
            )
            .with("storage::Allocator::copy_into"));
        }
        dst.copy_from_slice(src);
        Ok(())
    }
}

#[cfg(test)]
#[path = "../tests/alloc.rs"]
mod tests;
