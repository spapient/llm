#![no_std]
#![forbid(unsafe_code)]
#![deny(missing_docs)]
#![deny(missing_debug_implementations)]

//! Element-type floor for the `llm` runtime: one closed set per precision axis — scalar
//! [`DType`], GGML block-quantized [`QType`] and micro-scaled [`MxType`] — where every identity
//! fact (canonical name, variant, bit width, layout) lives in a single table. Zero dependencies,
//! unconditional `no_std` and no `unsafe`; every accessor that can be is a `const fn`, while name
//! parsing delegates to a trait (stable Rust has no const traits).
//!
//! This crate answers “what is this format and how many bytes does it take”, never “how do I
//! decode it”: dequantization and every conversion algorithm stay in `weights` (ADR 0018).
//!
//! ---
//!
//! L1 精度类型基座：标量 [`DType`]、块量化 [`QType`]、微缩放 [`MxType`] 与 MX 元素 [`Element`]，
//! 零工作区依赖。
//!
//! 全仓库的精度/数值类型只有这一个事实源，按三个正交轴分文件，MX 元素作为 `MxType` 的子闭集
//! 住在同一文件：
//!
//! - [`DType`](dtype.rs)：**一个元素怎么存**（无 scale），bool、INT1–INT64、FP4–FP64；
//! - [`QType`](qtype.rs)：**一块元素怎么打包**（GGML 块量化，`f16`/`f32`/sub-scale），如 `Q4_K`；
//! - [`MxType`](mxtype.rs)：**低精度元素 + 块共享指数**（OCP MX 与 NVFP4 厂商扩展）。
//! - [`Element`](mxtype.rs)：**MX 块内是哪种元素**（E2M1/E2M3/E3M2/E4M3/E5M2/INT8），只回答位宽与
//!   标量对应。
//!
//! `QType`/`MxType` 的布局表不引用 `DType`；唯一的跨轴**身份**映射是 [`mxtype::Element::dtype`]——
//! 元素格式到标量 `DType` 的身份对应（6 位 MX 格式没有标量对应时为 `None`）。本文件只是薄入口与
//! 再导出，保持与 `audit` / `witness` 一致的「薄 lib + 分文件」结构。各层以再导出（`tensor`、
//! `weights` 的 `Wire`）或子集表表达局部能力，不再手写类型枚举与跨层映射。行表类型
//! （`dtype::DTypeSpec`/`qtype::QTypeSpec`/`mxtype::MxTypeSpec`/`mxtype::ElementSpec`）只走
//! 模块路径，不在根重复导出：同一个类型只应有一个公共名字。
//!
//! # 泛型视角：`Format` / `BlockFormat` / `Layout<B>`
//!
//! 四个闭集的身份读法（闭集全量、判别值即下标、规范名、位宽、按名解析）完全一样，所以只写一次：
//! [`Format`] 是闭集契约（`prof` 的位集直接拿它当 `ALL` + 判别值 + 名字的来源），
//! [`BlockFormat`] 在它之上补每块元素数/字节数与饱和的字节数算法，[`Layout<B>`] 则是
//! 「标量元素或块格式」的统一布局来源（`weights` 的 GGUF 支持表用它，将来接 MXFP4 时
//! 只换泛型参数）。trait 方法不是 `const fn`（stable 无 const trait）：几何与身份的唯一实现
//! 是各枚举的 inherent `const fn`，trait 只转发；`parse_name` 反过来由 trait 默认实现唯一
//! 提供，inherent 只转发。两条路径读同一张行表，由 `tests/identity.rs` 一处对四个闭集逐项
//! 对拍（身份与块几何都在那里）。
//!
//! # 互操作：标准 trait 与可选 feature
//!
//! - 四个闭集（`DType`/`QType`/`MxType`/[`Element`]）都实现 `Display` 与 [`core::str::FromStr`]：
//!   `"q4_k".parse::<QType>()` 与 `QType::parse_name("q4_k")` 读同一张表；失败类型是零分配的
//!   [`UnknownName`]（不为了报错而分配）。
//! - [`DType`] 另实现 `From<DType> for u16` 与 `TryFrom<u16>`（稳定线上 code），失败类型是回显
//!   code 的 [`UnknownCode`]。`QType`/`MxType` 的判别值只是行表下标、不是线上码，所以不提供
//!   数字转换——给一个不存在的字节码一个名字才是撒谎。
//! - **刻意不提供 `Default`**：默认精度会静默宣布一个未被声明的类型；本 crate 的取舍是
//!   读不到就失败（`UnknownName`/`UnknownCode`），不拿一个“差不多”的值顶上。
//! - `serde` feature（**默认关闭**）：四个闭集按规范名（小写）序列化/反序列化，与 `Display`
//!   逐字一致；默认特性集保持零依赖。
//! - `bytemuck` feature（**默认关闭**）：`NoUninit` + `CheckedBitPattern`。`repr(u8)` 的合法
//!   判别值只有 24/25/7/6 个，其余字节是非法位模式，所以给的是受检读取而不是 `Pod`；
//!   写入可直接 `bytemuck::bytes_of`，读取必须走 `bytemuck::checked`。
//!
//! # 基座约束
//!
//! - **零依赖**：`[dependencies]` 默认为空；`serde`/`bytemuck` 是默认关闭的可选 feature；
//! - **`no_std`**：**无条件** `no_std`，只用 `core`（`u64::div_ceil`、`core::fmt`），可直接嵌进
//!   裸机与 wasm 目标；测试模块显式 `use std::` 取测试工具，生产代码用不到 `std` 由编译器保证；
//! - **零 unsafe**：`#![forbid(unsafe_code)]`，无法被下游放宽；
//! - **零运行时**：除 `parse_name`（转调 trait 默认实现）外全部 inherent 访问器是 `const fn`，
//!   判别值 `repr(u8)` 且等于行表下标；trait 只提供泛型视角，不额外引入分支或分配。
//!
//! ```
//! let dtype = llm_dtype::DType::F32;
//! assert_eq!(dtype.bits(), 32);
//! assert_eq!(dtype.code(), 20);
//! assert_eq!(llm_dtype::DType::from_code(20), Some(dtype));
//!
//! let qtype = llm_dtype::QType::Q4_K;
//! assert_eq!(qtype.block_len(), 256);
//! assert_eq!(qtype.block_bytes(), 144);
//! assert_eq!(qtype.bytes_for(256), 144);
//!
//! let mx = llm_dtype::MxType::MxFp4E2M1;
//! assert_eq!(mx.block_len(), 32);
//! assert_eq!(mx.bits(), 4);
//!
//! assert_eq!("q4_k".parse::<llm_dtype::QType>(), Ok(llm_dtype::QType::Q4_K));
//! assert_eq!(u16::from(llm_dtype::DType::F32), 20);
//! assert_eq!(llm_dtype::DType::try_from(20), Ok(llm_dtype::DType::F32));
//! ```

#[cfg(test)]
extern crate std;

mod blocked;
pub mod dtype;
mod format;
pub mod mxtype;
pub mod qtype;

pub use dtype::DType;
pub use format::{BlockFormat, Format, Layout, UnknownCode, UnknownName};
pub use mxtype::{Element, MxType};
pub use qtype::QType;
