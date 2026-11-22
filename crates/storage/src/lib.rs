//! 扁平缓冲与缓冲分配：[`Buf`] + [`Allocator`] + `CpuAllocator`。
//!
//! **拥有**：`Buf`、`Allocator`、`CpuAllocator`。
//! **不拥有**：形状与字节数（归 `tensor`：`Shape::numel` / `Shape::element_bytes`）、
//! 布局代数与重排（本仓没有这一层：排布由算子契约与设备 Fragment 描述，见 ADR 0053）、
//! 地址空间（`GM` / `FAST` / `REG` 归
//! `cutile` / `prof`）、设备私有句柄（藏在后端边界 crate 后，ADR 0025 §2）。
//! **依赖**：`tensor`、`diagnostic`。**被依赖**：`weights`、`engine`。
//!
//! `mmap` feature 只在 **Unix** 目标上生效：`Mmap` 是 `mmap(2)` 的薄封装（见 [`mmap`] 模块，
//! 全仓唯一 unsafe 岛），非 Unix 目标没有这套 FFI。其它目标上该 feature 是空操作、`Mmap`
//! 不存在——这是刻意的取舍：用"整文件读入"冒充映射会让"不占内存"这个前提失真，而缺失的
//! 类型在消费者那里是编译错误，不是静默降级。
//!
//! # 例子
//!
//! ```
//! use llm_storage::Buf;
//! use tensor::{DType, Shape};
//!
//! let buf = Buf::from_bytes(vec![0, 1, 2, 3, 4, 5]);
//! assert_eq!(buf.len(), 6);
//!
//! // 同一个形状，元素宽度不同则字节数不同：亚字节精度按位向上取整。
//! let shape = Shape::new([2, 3]);
//! assert_eq!(shape.numel(), 6);
//! assert_eq!(shape.element_bytes(DType::F32), 24);
//! assert_eq!(shape.element_bytes(DType::U4), 3);
//! ```

#![deny(unsafe_code)]

mod alloc;
mod buf;

#[cfg(all(feature = "mmap", unix))]
mod mmap;

#[cfg(all(feature = "mmap", unix))]
pub use mmap::Mmap;

pub use alloc::{Allocator, CpuAllocator};
pub use buf::Buf;
