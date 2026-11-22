//! 物理布局、搬运与缓冲分配：[`Layout`] + [`remap`] + [`Buf`] + [`Allocator`]。
//!
//! **拥有**：`Layout`、`remap`、`Buf`、`Allocator`、`CpuAllocator`。
//! **不拥有**：排列顺序与步幅累积（归 `tensor`，这里再导出 [`Order`] 保持路径兼容）、
//! 逻辑张量元数据（归 `tensor`）、地址空间（`GM` / `FAST` / `REG` 归
//! `cutile` / `prof`）、设备私有句柄（藏在后端边界 crate 后，ADR 0025 §2）。
//! **依赖**：`tensor`、`diagnostic`。**被依赖**：`weights`。
//!
//! 布局是"同一组元素在扁平缓冲里的排列方式"，所以它只需要形状、不需要数据类型：搬运在这里
//! 是元素级别的重排，按位打包（int4）由 `tensor::DType` 与具体算子负责。
//!
//! `mmap` feature 只在 **Unix** 目标上生效：`Mmap` 是 `mmap(2)` 的薄封装（见 [`mmap`] 模块，
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

mod alloc;
mod buf;
mod layout;

#[cfg(all(feature = "mmap", unix))]
mod mmap;

#[cfg(all(feature = "mmap", unix))]
pub use mmap::Mmap;

pub use alloc::{Allocator, CpuAllocator};
pub use buf::Buf;
pub use layout::{Layout, remap};

/// 排列顺序的主人在 `tensor`；这里再导出，保持 `storage::Order` 的既有路径。
pub use tensor::Order;
