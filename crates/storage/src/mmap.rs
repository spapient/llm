//! 只读内存映射：全仓唯一的 `unsafe` 孤岛，对外只暴露一个拥有型、只读的字节视图。
//!
//! 本模块只在 `mmap` feature 开启时编译；默认构建不含任何 unsafe，storage 仍满足
//! `deny(unsafe_code)`。所有 `mmap`/`munmap` 的 FFI 与裸指针操作都收敛在本文件。
//! 映射以 `PROT_READ | MAP_PRIVATE` 建立，内容不可改、写时复制不触发，因此把它当作 `&[u8]`
//! 共享是线程安全的。[`Mmap`] 在 `Drop` 时自动 `munmap`，不泄漏映射。
//! 上层把该切片交给 `pack::Archive::parse` 做零拷贝打开。

// 工作区默认 deny(unsafe_code)；本模块是经架构决策隔离的唯一 unsafe 岛，对整个模块放开。
#![allow(unsafe_code)]

use core::ffi::{c_int, c_void};
use std::fs::File;
use std::os::unix::io::AsRawFd;
use std::path::Path;

use diagnostic::{Code, Error, Result};

/// 页保护：只读。
const PROT_READ: c_int = 1;

/// 映射标志：私有写时复制（本端只读，不会触发复制）。
const MAP_PRIVATE: c_int = 2;

unsafe extern "C" {
    fn mmap(
        addr: *mut c_void,
        len: usize,
        prot: c_int,
        flags: c_int,
        fd: c_int,
        offset: i64,
    ) -> *mut c_void;

    fn munmap(addr: *mut c_void, len: usize) -> c_int;
}

/// 一份只读文件映射：拥有映射内存，`Drop` 时归还。
#[derive(Debug)]
pub struct Mmap {
    /// 映射起始地址。
    ptr: *mut c_void,
    /// 映射长度（字节）。
    len: usize,
}

// 只读、私有映射：多个线程共享不可变引用是安全的。
unsafe impl Send for Mmap {}
unsafe impl Sync for Mmap {}

impl Mmap {
    /// 打开并完整映射一个文件为只读。
    pub fn open(path: &Path) -> Result<Self> {
        let file = File::open(path).map_err(|error| {
            Error::new(Code::Io, format!("打开 {} 失败：{error}", path.display()))
                .with("storage::mmap::Mmap::open")
        })?;
        let len = file
            .metadata()
            .map_err(|error| {
                Error::new(Code::Io, format!("取 {} 长度失败：{error}", path.display()))
                    .with("storage::mmap::Mmap::open")
            })?
            .len();

        // 空文件不映射：mmap 长度 0 的行为跨平台不可移植，直接返回空视图。
        if len == 0 {
            return Ok(Self {
                ptr: core::ptr::null_mut(),
                len: 0,
            });
        }

        let ptr = unsafe {
            mmap(
                core::ptr::null_mut(),
                len as usize,
                PROT_READ,
                MAP_PRIVATE,
                file.as_raw_fd(),
                0,
            )
        };

        // MAP_FAILED 是 (void*)-1；失败时读取系统错误并返回。
        if ptr == map_failed() {
            let error = std::io::Error::last_os_error();
            return Err(
                Error::new(Code::Io, format!("映射 {} 失败：{error}", path.display()))
                    .with("storage::mmap::Mmap::open"),
            );
        }

        Ok(Self {
            ptr,
            len: len as usize,
        })
    }

    /// 映射的只读字节。
    #[must_use]
    pub fn as_slice(&self) -> &[u8] {
        if self.len == 0 {
            return &[];
        }
        // SAFETY：ptr..ptr+len 是一次成功的 PROT_READ 映射，且 self 存活期间不会被 munmap
        // （Drop 是唯一释放点，且 self 在此借用中无法被释放）。
        unsafe { core::slice::from_raw_parts(self.ptr.cast::<u8>(), self.len) }
    }

    /// 映射字节数。
    #[must_use]
    pub fn len(&self) -> usize {
        self.len
    }

    /// 是否为空映射。
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

impl AsRef<[u8]> for Mmap {
    fn as_ref(&self) -> &[u8] {
        self.as_slice()
    }
}

impl Drop for Mmap {
    fn drop(&mut self) {
        if self.len == 0 || self.ptr.is_null() {
            return;
        }
        // munmap 失败仅能记录，无法在 Drop 中传播；映射随进程退出由内核回收。
        let rc = unsafe { munmap(self.ptr, self.len) };
        debug_assert_eq!(rc, 0, "munmap 应成功");
    }
}

/// `MAP_FAILED` 指针值 `(void*)-1`。
fn map_failed() -> *mut c_void {
    #[allow(clippy::cast_possible_wrap)]
    #[cfg(target_pointer_width = "64")]
    {
        (-1isize) as usize as *mut c_void
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn maps_file_bytes_and_length() {
        let path = std::env::temp_dir().join(format!(
            "llm_mmap_{}_{}.bin",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let payload: Vec<u8> = (0..=250).collect();
        std::fs::write(&path, &payload).expect("写文件");

        let map = Mmap::open(&path).expect("映射");
        assert_eq!(map.len(), payload.len());
        assert_eq!(map.as_slice(), payload.as_slice());
        assert_eq!(map.as_ref(), payload.as_slice());

        std::fs::remove_file(&path).expect("清理");
    }

    #[test]
    fn an_empty_file_maps_to_empty() {
        let path = std::env::temp_dir().join(format!(
            "llm_mmap_empty_{}.bin",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::write(&path, b"").expect("建空文件");
        let map = Mmap::open(&path).expect("空文件可映射");
        assert!(map.is_empty());
        assert_eq!(map.as_slice(), b"");
        std::fs::remove_file(&path).expect("清理");
    }

    #[test]
    fn a_missing_file_is_an_error() {
        let map = Mmap::open(Path::new("/nonexistent/path/llm_mmap_no_such_file"));
        assert!(map.is_err());
    }
}
