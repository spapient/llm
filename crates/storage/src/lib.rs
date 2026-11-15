//! 物理布局、搬运与缓冲分配：[`Order`] + [`Layout`] + [`remap`] + [`Buffer`] + [`Allocator`]。
//!
//! **拥有**：`Layout`、`Order`、`remap`、`Buffer`、`Allocator`、`CpuAllocator`。
//! **不拥有**：逻辑张量元数据（归 `tensor`）、地址空间（`GM` / `FAST` / `REG` 归
//! `cutile` / `prof`）、设备私有句柄（藏在后端边界 crate 后，ADR 0025 §2）。
//! **依赖**：`tensor`、`diagnostic`。**被依赖**：`weights`。
//!
//! 布局是"同一组元素在扁平缓冲里的排列方式"，所以它只需要形状、不需要数据类型：搬运在这里
//! 是元素级别的重排，按位打包（int4）由 `tensor::DType` 与具体算子负责。
//!
//! `mmap` feature 只在 **Unix** 目标上生效：`Mmap` 是 `mmap(2)` 的薄封装，非 Unix 目标没有
//! 这套 FFI。其它目标上该 feature 是空操作、`Mmap` 不存在——这是刻意的取舍：用"整文件读入"
//! 冒充映射会让"不占内存"这个前提失真，而缺失的类型在消费者那里是编译错误，不是静默降级。
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
mod mmap;
#[cfg(all(feature = "mmap", unix))]
pub use mmap::Mmap;

use diagnostic::{Code, Error, Result};
use tensor::{Shape, Strides};

/// 元素在扁平缓冲里的排列顺序。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Order {
    /// 行主序：最后一轴相邻（C / numpy 默认）。
    RowMajor,
    /// 列主序：第一轴相邻（Fortran / cuBLAS 的 `T` 位）。
    ColMajor,
}

/// 形状 + 排列顺序；步长由它推出，避免两处各写一份。
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
        let dims = self.shape.dims();
        let mut strides = vec![0; dims.len()];
        let mut acc = 1usize;
        match self.order {
            Order::RowMajor => {
                for axis in (0..dims.len()).rev() {
                    strides[axis] = acc;
                    acc = acc.saturating_mul(dims[axis]);
                }
            }
            Order::ColMajor => {
                for axis in 0..dims.len() {
                    strides[axis] = acc;
                    acc = acc.saturating_mul(dims[axis]);
                }
            }
        }
        Strides::new(strides)
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
/// 具体后端实现放在独立边界 crate，依赖方向只能从边界 crate 指向本 trait（ADR 0001）。
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

#[cfg(test)]
mod tests;
