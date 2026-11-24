//! 扁平缓冲、缓冲分配与检查点字节格式：[`Buf`] + [`Allocator`] + `CpuAllocator` + [`snapshot`]，
//! 以及 [`LaneBuf`] / [`LaneView`] / [`LaneViewMut`] 两条 `f32`/打包 `u8` lane 的公共缓冲与视图；
//! 另含与领域无关的字节码闭集：CRC-32 [`crc`]、LEB128 变长整数 [`leb`]、统一字节输出接收器
//! [`sink`]，以及对齐落位 [`align_up`]、`Invalid` 错误构造点 [`bad`] / [`or_bad`]（原 `wire` crate 并入）。
//!
//! **拥有**：`Buf`、`Lane`、`LaneBuf`、`LaneView`、`LaneViewMut`、`Allocator`、`CpuAllocator`、
//! `snapshot` 的字节布局与两条契约（`Manifest` / `ManifestEntry`），以及 `crc` / `leb` / `sink` /
//! `fault` 与 `align_up` 这组纯字节格式基础件。
//! **不拥有**：形状与字节数（归 `tensor`：`Shape::numel` / `Shape::element_bytes`）、
//! 布局代数与重排（本仓没有这一层：排布由算子契约与设备 Fragment 描述，见 ADR 0053）、
//! 地址空间（`GM` / `FAST` / `REG` 归
//! `cutile` / `prof`）、设备私有句柄（藏在后端边界 crate 后，ADR 0025 §2）。
//! **依赖**：`tensor`、`diagnostic`。**被依赖**：`weights`、`engine`、`train`。
//!
//! `snapshot` 是**泛型**的：字节格式只需要"有序条目 + 数据段起点"，不需要知道权重文件怎么校验
//! 自己。`weights::Header` 实现 `Manifest`，于是 `weights::Snapshot` 只是
//! `snapshot::Snapshot<Header>` 的别名——`storage` 不必反向依赖 `weights`，成环的边不存在。
//!
//! `mmap` feature 只在 **Unix** 目标上生效：`Mmap` 是 `mmap(2)` 的薄封装（见 `mmap` 模块，
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
mod lane;

pub mod snapshot;

pub mod crc;
pub mod fault;
pub mod leb;
pub mod sink;

mod codec;
mod format;
#[cfg(all(feature = "mmap", unix))]
mod mmap;
mod state;

#[cfg(all(feature = "mmap", unix))]
pub use mmap::Mmap;

pub use alloc::{Allocator, CpuAllocator};
pub use buf::Buf;
pub use fault::{bad, or_bad};
pub use lane::{Lane, LaneBuf, LaneView, LaneViewMut};
pub use snapshot::{
    Facts, Format, GradAccum, Manifest, ManifestEntry, Moments, SlotMoments, Snapshot, Ties,
};

/// 向上取整到 `align` 的倍数；`align` 非 2 的幂或结果溢出时返回 `None`。
#[must_use]
pub fn align_up(value: usize, align: usize) -> Option<usize> {
    if align == 0 || !align.is_power_of_two() {
        return None;
    }
    value.div_ceil(align).checked_mul(align)
}
