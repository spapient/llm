//! 只读内存映射源：在 Unix + `mmap` feature 下实现 [`Reader`]，条目字节从映射区借用，零拷贝。
//!
//! 与 [`crate::safetensors::File`] 对称：`File` 是 [`Loader`]（seek + read 拷进调用方缓冲），
//! 本类型是 [`Reader`]（映射借用，不复制条目字节）。映射内存由 [`Mmap`] 拥有，Drop 时归还；
//! 本模块不引入第二处 `unsafe`——全部 FFI 与裸指针操作都收敛在 `storage::mmap` 唯一岛上。
//!
//! 非 Unix 目标没有 `mmap(2)`：本模块在 `lib.rs` 的 `cfg(all(feature = "mmap", unix))` 处
//! 整块裁掉，不提供假装映射的空实现，缺失类型在编译期报错。

use std::path::Path;

use diagnostic::{Code, Error, Result};
use storage::Mmap;

use crate::safetensors::{Archive, entry_slice};
use crate::{Entry, Header, Reader};

/// 只读映射权重源：整份文件映射后复用 safetensors 头部自证校验，条目字节作为映射借用。
///
/// 头部元数据（[`Header`]）在打开时解析并拥有下来；字节不复制，`read` 返回的切片落在映射区内，
/// 生命期不超过本类型。
#[derive(Debug)]
pub struct MmapSource {
    map: Mmap,
    header: Header,
}

impl MmapSource {
    /// 打开并校验一个权重文件，整份映射为只读。
    ///
    /// # Errors
    ///
    /// [`diagnostic::Code::Io`]：打不开、取长度失败或映射失败（透传自 [`Mmap::open`]）。
    ///
    /// [`diagnostic::Code::Invalid`] / [`diagnostic::Code::Unsupported`]：与 [`Archive::parse`]
    /// 同口径的头部校验失败（长度前缀、字段、区间、dtype 闭集）。
    pub fn open(path: &Path) -> Result<Self> {
        let map = Mmap::open(path)?;
        let archive = Archive::parse(map.as_slice())?;
        let header = archive.header().clone();
        Ok(Self { map, header })
    }

    /// 已校验的头部。
    #[must_use]
    pub fn header(&self) -> &Header {
        &self.header
    }

    /// 整份映射的只读字节。
    #[must_use]
    pub fn as_slice(&self) -> &[u8] {
        self.map.as_slice()
    }
}

impl Reader for MmapSource {
    fn read(&self, entry: &Entry) -> Result<&[u8]> {
        let all = self.map.as_slice();
        let data = all.get(self.header.data_start()..).ok_or_else(|| {
            Error::new(
                Code::Internal,
                format!(
                    "数据段起点 {} 超出映射长度 {}",
                    self.header.data_start(),
                    all.len()
                ),
            )
            .with("weights::mmap::MmapSource::read")
        })?;
        entry_slice(&self.header, data, entry)
    }
}

#[cfg(test)]
#[path = "../tests/mmap_source_tests.rs"]
mod tests;
