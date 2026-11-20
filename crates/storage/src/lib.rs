//! 物理布局、搬运与缓冲分配：[`Layout`] + [`remap`] + [`Buffer`] + [`Allocator`]。
//!
//! **拥有**：`Layout`、`remap`、`Buffer`、`Allocator`、`CpuAllocator`。
//! **不拥有**：排列顺序与步幅累积（归 `tensor`，这里再导出 [`Order`] 保持路径兼容）、
//! 逻辑张量元数据（归 `tensor`）、地址空间（`GM` / `FAST` / `REG` 归
//! `cutile` / `prof`）、设备私有句柄（藏在后端边界 crate 后，ADR 0025 §2）。
//! **依赖**：`tensor`、`diagnostic`。**被依赖**：`weights`。
//!
//! 布局是"同一组元素在扁平缓冲里的排列方式"，所以它只需要形状、不需要数据类型：搬运在这里
//! 是元素级别的重排，按位打包（int4）由 `tensor::DType` 与具体算子负责。
//!
//! `mmap` feature 只在 **Unix** 目标上生效：`Mmap` 是 `mmap(2)` 的薄封装（内联 `mmap` 模块，
//! 全仓唯一 unsafe 岛），非 Unix 目标没有这套 FFI。其它目标上该 feature 是空操作、`Mmap`
//! 不存在——这是刻意的取舍：用"整文件读入"冒充映射会让"不占内存"这个前提失真，而缺失的
//! 类型在消费者那里是编译错误，不是静默降级。
//!
//! # 例子
//!
//! ```
//! use llm_storage::{remap, Layout, Order};
//! use tensor::Shape;
//!
//! let shape = Shape::new([2, 3]);
//! let row_major = Layout::new(shape.clone(), Order::RowMajor);
//! let col_major = Layout::new(shape, Order::ColMajor);
//! let source = [0u8, 1, 2, 3, 4, 5];
//! let moved = remap(&source, &row_major, &col_major).expect("形状一致");
//! assert_eq!(moved, [0, 3, 1, 4, 2, 5]);
//! ```

#![deny(unsafe_code)]

#[cfg(all(feature = "mmap", unix))]
pub use mmap::Mmap;

use diagnostic::{Code, Error, Result};
use tensor::{Shape, Strides};

/// 排列顺序的主人在 `tensor`；这里再导出，保持 `storage::Order` 的既有路径。
pub use tensor::Order;

/// 形状 + 排列顺序；步长由 [`Strides::ordered`] 推出，累积规则只有一份。
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Layout {
    shape: Shape,
    order: Order,
}

impl Layout {
    /// 由形状与顺序构造。
    pub fn new(shape: Shape, order: Order) -> Self {
        Self { shape, order }
    }

    /// 形状。
    pub fn shape(&self) -> &Shape {
        &self.shape
    }

    /// 排列顺序。
    pub fn order(&self) -> Order {
        self.order
    }

    /// 按顺序推出的步长。
    pub fn strides(&self) -> Strides {
        Strides::ordered(&self.shape, self.order)
    }
}

/// 把 `src` 从 `from` 布局重排成 `to` 布局。
///
/// 两种布局必须覆盖同一个形状；`src` 至少要够长。实现是逐元素搬运，所以它既能用于
/// 行↔列主序的转换，也能用于任何由形状推出的布局——不需要为每种转换写一条路径。
pub fn remap<T: Copy>(src: &[T], from: &Layout, to: &Layout) -> Result<Vec<T>> {
    if from.shape() != to.shape() {
        return Err(Error::new(
            Code::Invalid,
            format!(
                "布局形状不一致：{:?} vs {:?}",
                from.shape().dims(),
                to.shape().dims()
            ),
        )
        .with("storage::remap"));
    }
    let count = from.shape().numel();
    if count == 0 {
        return Ok(Vec::new());
    }
    if src.len() < count {
        return Err(Error::new(
            Code::Invalid,
            format!("源缓冲 {} 个元素，形状需要 {count} 个", src.len()),
        )
        .with("storage::remap"));
    }
    let from_strides = from.strides();
    let to_strides = to.strides();
    let mut out = vec![src[0]; count];
    for linear in 0..count {
        let source = offset(&from_strides, from.shape(), linear);
        let target = offset(&to_strides, to.shape(), linear);
        out[target] = src[source];
    }
    Ok(out)
}

/// 行主序线性下标 → 该布局下的元素偏移。
fn offset(strides: &Strides, shape: &Shape, linear: usize) -> usize {
    let dims = shape.dims();
    let mut rest = linear;
    let mut offset = 0usize;
    for axis in (0..dims.len()).rev() {
        let dim = dims[axis];
        if dim == 0 {
            return 0;
        }
        let index = rest % dim;
        rest /= dim;
        offset = offset.saturating_add(index.saturating_mul(strides.get(axis).unwrap_or(0)));
    }
    offset
}

/// 一段拥有的扁平字节缓冲：CPU 后端的不透明句柄。
///
/// 设备后端的指针/句柄藏在各自边界 crate 内，本层只暴露字节读写，不泄漏设备地址（ADR 0025 §2）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Buffer {
    bytes: Vec<u8>,
}

impl Buffer {
    /// 由拥有的字节向量构造；长度即缓冲容量。
    pub fn from_bytes(bytes: Vec<u8>) -> Self {
        Self { bytes }
    }

    /// 只读字节。
    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }

    /// 可写字节。
    pub fn bytes_mut(&mut self) -> &mut [u8] {
        &mut self.bytes
    }

    /// 字节长度。
    pub fn len(&self) -> usize {
        self.bytes.len()
    }

    /// 是否为空。
    pub fn is_empty(&self) -> bool {
        self.bytes.is_empty()
    }

    /// 拆回拥有的字节向量（归还分配器前取出数据用）。
    pub fn into_bytes(self) -> Vec<u8> {
        self.bytes
    }
}

/// 缓冲分配器：分配、释放、显式拷贝。
///
/// 分配与释放只经过本 trait；跨设备拷贝是同一方法的后端实现，框架层不发生隐式拷贝（ADR 0025 §3）。
/// 具体后端实现放在独立边界 crate，依赖方向只能从边界 crate 指向本 trait（见 `docs/ARCHITECTURE.md` 第二节）。
pub trait Allocator {
    /// 分配 `bytes` 字节并清零；`0` 得到空缓冲。
    fn alloc(&mut self, bytes: usize) -> Result<Buffer>;

    /// 归还一段缓冲。RAII 下向量 drop 即释放，显式归还点为未来设备回收预留钩子。
    fn free(&mut self, buffer: Buffer);

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
    fn alloc(&mut self, bytes: usize) -> Result<Buffer> {
        let mut owned = Vec::new();
        owned.try_reserve_exact(bytes).map_err(|_| {
            Error::new(
                Code::Invalid,
                format!("分配 {bytes} 字节：超出可用内存或地址空间"),
            )
            .with("storage::CpuAllocator::alloc")
        })?;
        owned.resize(bytes, 0);
        Ok(Buffer::from_bytes(owned))
    }

    fn free(&mut self, _buffer: Buffer) {}

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

/// 只读内存映射：全仓唯一的 `unsafe` 孤岛，对外只暴露一个拥有型、只读的字节视图。
///
/// 本模块只在 `mmap` feature 开启**且目标为 Unix** 时编译：`mmap`/`munmap` 是 POSIX 系统调用，
/// 非 Unix 目标没有这套 FFI。默认构建不含任何 unsafe，storage 仍满足 `deny(unsafe_code)`。
/// 所有 `mmap`/`munmap` 的 FFI 与裸指针操作都收敛在本模块。
///
/// 映射以 `PROT_READ | MAP_PRIVATE` 建立，内容不可改、写时复制不触发，因此把它当作 `&[u8]`
/// 共享是线程安全的。[`Mmap`] 在 `Drop` 时自动 `munmap`，不泄漏映射。
/// 上层把该切片交给 `weights::Archive::parse` 做零拷贝打开。
#[cfg(all(feature = "mmap", unix))]
mod mmap {
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
            let (file, len) = open_readonly(path)?;
            let len = mapped_len(path, len)?;

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
                    len,
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

            Ok(Self { ptr, len })
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

    /// 打开文件并取出长度；两类 `io` 错误各自说清是"打不开"还是"读不到长度"。
    fn open_readonly(path: &Path) -> Result<(File, u64)> {
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
        Ok((file, len))
    }

    /// 文件长度 → `usize`：32 位目标上一个 >4 GiB 的文件必须报错，而不是悄悄只映射前缀。
    fn mapped_len(path: &Path, len: u64) -> Result<usize> {
        usize::try_from(len).map_err(|_| {
            Error::new(
                Code::Io,
                format!("{} 超过本目标可映射的长度（usize 上限）", path.display()),
            )
            .with("storage::mmap::Mmap::open")
        })
    }

    /// `MAP_FAILED` 指针值 `(void*)-1`。
    ///
    /// 用 `without_provenance_mut(usize::MAX)` 而不是 `(-1isize) as *mut c_void`：后者在非 64 位
    /// 目标上曾被 `cfg` 整块裁掉，函数体变成空 `{}`、返回类型不符，交叉构建直接编译失败。
    fn map_failed() -> *mut c_void {
        core::ptr::without_provenance_mut::<c_void>(usize::MAX)
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
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn row_major_is_the_tensor_default_and_col_major_is_its_mirror() {
        let shape = Shape::new([2, 3]);
        assert_eq!(
            Layout::new(shape.clone(), Order::RowMajor)
                .strides()
                .as_slice(),
            [3, 1]
        );
        assert_eq!(
            Layout::new(shape, Order::ColMajor).strides().as_slice(),
            [1, 2]
        );
    }

    #[test]
    fn remap_transposes_a_row_major_buffer() {
        let shape = Shape::new([2, 3]);
        let from = Layout::new(shape.clone(), Order::RowMajor);
        let to = Layout::new(shape.clone(), Order::ColMajor);
        let moved = remap(&[0u8, 1, 2, 3, 4, 5], &from, &to).expect("形状一致");
        assert_eq!(moved, [0, 3, 1, 4, 2, 5]);
        let back = remap(&moved, &to, &from).expect("回到行主序");
        assert_eq!(back, [0, 1, 2, 3, 4, 5]);
    }

    #[test]
    fn an_identity_layout_change_is_a_copy() {
        let shape = Shape::new([4]);
        let layout = Layout::new(shape, Order::RowMajor);
        let moved = remap(&[1.0f32, 2.0, 3.0, 4.0], &layout, &layout).expect("相同布局");
        assert_eq!(moved, [1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn mismatched_shapes_and_short_buffers_are_rejected() {
        let from = Layout::new(Shape::new([2, 3]), Order::RowMajor);
        let to = Layout::new(Shape::new([3, 2]), Order::ColMajor);
        assert_eq!(
            remap(&[0u8; 6], &from, &to).unwrap_err().code(),
            Code::Invalid
        );
        let same = Layout::new(Shape::new([2, 3]), Order::ColMajor);
        assert_eq!(
            remap(&[0u8; 5], &from, &same).unwrap_err().code(),
            Code::Invalid
        );
    }

    #[test]
    fn an_empty_shape_maps_to_nothing() {
        let shape = Shape::new([0, 3]);
        let from = Layout::new(shape.clone(), Order::RowMajor);
        let to = Layout::new(shape, Order::ColMajor);
        assert_eq!(remap::<u8>(&[], &from, &to), Ok(Vec::new()));
    }

    #[test]
    fn cpu_allocator_allocates_zeroed_buffer() {
        let mut alloc = CpuAllocator::new();
        let buffer = alloc.alloc(5).expect("分配");
        assert_eq!(buffer.len(), 5);
        assert_eq!(buffer.bytes(), &[0; 5]);
        alloc.free(buffer);
    }

    #[test]
    fn allocator_buffer_is_writable() {
        let mut alloc = CpuAllocator::new();
        let mut buffer = alloc.alloc(3).expect("分配");
        buffer.bytes_mut().copy_from_slice(&[1, 2, 3]);
        assert_eq!(buffer.bytes(), &[1, 2, 3]);
        let bytes = buffer.into_bytes();
        assert_eq!(bytes, vec![1, 2, 3]);
    }

    #[test]
    fn zero_allocation_yields_an_empty_buffer() {
        let mut alloc = CpuAllocator::new();
        let buffer = alloc.alloc(0).expect("零长度");
        assert!(buffer.is_empty());
    }

    #[test]
    fn copy_into_matches_lengths_and_rejects_mismatch() {
        let alloc = CpuAllocator::new();
        let mut dst = vec![0u8; 4];
        alloc.copy_into(&[1, 2, 3, 4], &mut dst).expect("等长拷贝");
        assert_eq!(dst, [1, 2, 3, 4]);
        let err = alloc.copy_into(&[1, 2], &mut dst).unwrap_err();
        assert_eq!(err.code(), Code::Invalid);
    }

    #[test]
    fn an_unaffordable_allocation_is_an_error_not_an_abort() {
        let mut alloc = CpuAllocator::new();
        let error = alloc.alloc(usize::MAX).expect_err("地址空间装不下");
        assert_eq!(error.code(), Code::Invalid);
    }
}
