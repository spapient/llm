//! 权重文件的结构：[`Entry`]、[`Header`] 与两条读取契约（[`Reader`] / [`Loader`]），以及
//! safetensors 的两种后端（[`safetensors::Archive`] 借用整份字节、[`safetensors::File`] 只读头部）。
//!
//! **拥有**：`Header`/`Entry`/`Reader`/`Loader` 三条契约，`safetensors` 的两种后端，自有的 `.u`
//! 容器 v2（`writer`/`archive`/`seg`/`proof`/`rans`；CRC/LEB/Sink 已归 L1 `wire`）以及只能由
//! 本层判断的格式适配器与量化方案（`bin` 栈机含 ZIP_STORED 读取、`gguf`、`json`、`quant`）。
//! **不拥有**：dtype 闭集（`dtype`）、张量元数据（`tensor`）、物理布局（`storage`）。
//! **依赖**：`dtype`、`diagnostic`、`tensor`、`wire`（`storage` 经可选 `mmap` feature 引入，
//! 只在 Unix 上提供映射源；`+dev`：`testkit`）。
//! **被依赖**：`train`、`nn`、`llm`（`+dev`：`fuzz`）。
//!
//! 两条读取契约的分工是所有权而不是风格：[`Reader`] 把字节**借**出来（内存切片、映射），
//! [`Loader`] 把字节**拷进**调用方给的缓冲区——只有后者能由文件实现，因为不引入 `unsafe` 就无法
//! 自己映射，也无法承诺"借出的引用在函数返回后仍然有效"。真实权重动辄十几 GB，靠的正是后者。
//!
//! 这一层先解决“权重文件能不能自证合法”：条目名唯一、字节区间不重叠、区间起点不早于
//! 数据段，以及 safetensors 的“声明区间必须等于形状推出的字节数”。这几条不成立时，任何
//! 后续读写都会在别处以更难查的方式失败。
//!
//! 解析器只填这三个类型，不新增契约：换个文件格式（GGUF）只需再写一个“字节 → `Header`”，
//! 上层（`nn` 的层清单、`train` 的检查点）不需要跟着改。
//!
//! # 例子
//!
//! ```
//! use tensor::{DType, Shape};
//! use llm_weights::{Entry, Header};
//!
//! let entry = Entry::new("blk.0.attn_q.weight", DType::F32, Shape::new([2, 3])).at(64);
//! let header = Header::new(vec![entry], 64).expect("区间合法");
//! assert_eq!(header.data_bytes(), 24);
//! assert!(header.get("blk.0.attn_q.weight").is_some());
//! ```

#![deny(unsafe_code)]

/// 全 crate 的“非法输入”构造点转发自 L1 [`wire`]：判据与上下文标注各只有一处实现。
pub(crate) use wire::{bad, or_bad};

// —— 自有 `.u` 容器 v2：描述符、读/写、帧编码、增量补丁与双模式 ——
mod archive;
mod container;
mod lifecycle;
mod metrics;
mod parse;
mod rans;
mod seg;
mod stored;
mod writer;

// —— 量化方案：仿射方案与外部量化条目 ——
pub mod quant;

// —— 2/4/8-bit 对称 PTQ：f32 → 打包码的正向量化与其反解（离线换算，ADR 0018） ——
pub mod ptq;

// 自适应档位分配：ptq 的子域，平铺到顶层以过 N3（原 `ptq/adaptive`）。
mod adaptive;

// —— 完整性证明：Merkle + SHA-256 ——
mod proof;

// —— 谱隙证书链：f64 单边 Jacobi SVD、谱隙判定账本、五轴无损选码（.u v2 张量级压缩侧车）——
pub mod axes;
pub mod ledger;
pub mod svd;

// —— 沉降账本：LLD §4.7 的逐张量体积 / 原因 / 无损判决，全部由落盘字节现算 ——
pub mod settle;

// —— L3 合成律：LLD §4.5 的逐段证书 → 网络级误差上界与适用域 ——
pub mod l3;

// —— 外部格式只读适配：safetensors/gguf/bin/json ——
pub mod bin;
pub mod gguf;
mod pickle;
mod zip;
pub use lifecycle::import;
pub mod json;
pub mod safetensors;

// 只读内存映射源：feature + Unix 双门控，复用 `storage` 的唯一 unsafe 岛。
#[cfg(all(feature = "mmap", unix))]
mod mmap;

// 纯字节码基础件归 L1 `wire`；这里只做薄转发，保持容器内部路径不变。
/// CRC-32（转发自 `wire::crc`）。
pub mod crc {
    pub use wire::crc::*;
}

mod leb {
    pub(crate) use wire::leb::*;
}

mod sink {
    pub(crate) use wire::sink::*;
}

/// 生命周期模式薄转发：实现见内部 `lifecycle` 模块。
pub mod mode {
    pub use crate::lifecycle::{MODE_NAME, Mode, is_opt, mode_of, release};
}

/// 增量补丁薄转发：实现见内部 `lifecycle` 模块。
pub mod patch {
    pub use crate::lifecycle::{diff, join};
}

#[cfg(test)]
#[path = "../tests/testutil.rs"]
mod testutil;

/// 词表段保留名（字节容器占位，不接 tokenizer）。
pub use crate::archive::VOCAB_NAME;

pub use archive::Archive;
pub use dtype::DType as Wire;
pub use l3::{Bracket, Chain, Ruling, Rung};
pub use metrics::Metrics;
pub use proof::Proof;
pub use ptq::{
    Adaptive, AdaptiveCompensated, Compensated, I2, I3, I4, I8, Scheme, adaptive_quantize,
    compensate, compensate_adaptive, dequantize, quantize,
};
pub use quant::{Quant, Quantized};
pub use seg::Codec;
pub use seg::Segment;
pub use settle::{Reason, Settled, Settlement};
pub use stored::Stored;
pub use writer::Writer;

pub use container::{
    Entry, Header, Loader, Reader, decode_entry_f32, decode_f32, load_f32, read_f32,
};
pub(crate) use container::{align_up, check_capacity, read_wire};

#[cfg(all(feature = "mmap", unix))]
pub use mmap::MmapSource;

#[cfg(test)]
#[path = "../tests/lib.rs"]
mod tests;
