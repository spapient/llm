//! 只读内存映射源：在 Unix + `mmap` feature 下实现 [`Reader`]，条目字节从映射区借用，零拷贝。

use std::path::Path;

use diagnostic::{Code, Error, Result};
use storage::Mmap;

use crate::safetensors::{Archive, entry_slice};
use crate::{Entry, Header, Reader};

/// 只读映射权重源：整份文件映射后复用 safetensors 头部自证校验，条目字节作为映射借用。
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
#[path = "../tests/mmap.rs"]
mod tests;
