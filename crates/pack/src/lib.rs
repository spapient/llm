//! 自家权重容器 `pack`：版本化、可对齐、可校验的存储与传输字节格式。
//!
//! **拥有**：容器字节布局（[`Writer`] / [`Archive`]）、全精度线上类型 [`Wire`]、显式量化方案
//! [`Quant`]、对齐与两级 CRC-32。**不拥有**：safetensors 适配（归 `weights`）、张量元素与内核
//! （归 `tensor`）、mmap（后续 feature-gated 独立 crate）。
//! **依赖**：`diagnostic`、`tensor`，无第三方 crate。
//!
//! 容器解决三件 `weights` 覆盖不了的事：**读写对称**（能产出也能读回）、**全精度与显式量化**
//! （INT1–FP64、F8，分组/零点是文件事实而非文件名猜测）、**对齐与完整性**（每段可对齐到
//! 2 的幂边界，段级 + 全局 CRC 把静默损坏拦在打开期）。
//!
//! 同一份逻辑容器只有一种字节形态：描述符走 canonical LEB128，对齐填充必须为 0，读端重算
//! 段区间并核对校验。格式细节见
//! [`docs/adr/0015-pack-weight-container.md`](../../docs/adr/0015-pack-weight-container.md)。
//!
//! # 例子
//!
//! ```
//! use llm_pack::{Archive, Segment, Wire, Writer};
//!
//! let segment = Segment::new(
//!     "blk.0.attn_q.weight",
//!     Wire::F32,
//!     vec![2, 3],
//!     None,
//!     vec![0u8; 24],
//! )
//! .expect("段自洽");
//!
//! let mut writer = Writer::new();
//! writer.push(segment).expect("追加");
//! let bytes = writer.finish(6).expect("64 B 对齐");
//!
//! let archive = Archive::parse(&bytes).expect("自证合法");
//! let stored = archive.get("blk.0.attn_q.weight").expect("按名查");
//! assert_eq!(archive.read(stored).len(), 24);
//! ```

#![deny(unsafe_code)]

mod crc;
mod dtype;
mod leb;
mod quant;
mod reader;
mod seg;
mod writer;

pub use dtype::Wire;
pub use quant::Quant;
pub use reader::{Archive, Stored};
pub use seg::Segment;
pub use writer::Writer;
