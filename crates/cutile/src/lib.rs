//! cutile —— 设备无关的替换式 kernel DSL。
//!
//! 本 crate 实现 [`DESIGN.md`](../DESIGN.md) 的 P0 纵切面：
//!
//! ```text
//! 声明层 ops/<op>.toml + ops/<op>.k   档案层 devices/<vendor>.toml   片段层 prim/<vendor>/<prim>.<ext>
//!        ① expand  →  ② negotiate  →  ③ select  →  ④ substitute  →  ⑤ emit + cache
//! ```
//!
//! 五个步骤的中间产物全部是可见的数据结构（[`lower::Variant`]、[`cap::Path`]、
//! [`plan::Plan`]、[`lower::Source`]），没有任何隐藏的优化阶段——这是"无 IR"的具体含义。
//!
//! # 例子
//!
//! 同一份算子声明、同一份模板，在三份档案上协商出不同的能力路径：
//!
//! ```
//! use cutile::builtin::Catalogue;
//!
//! let cat = Catalogue::builtin();
//! let op = cat.op("gemv_i4").expect("内置算子");
//! let cpu = cat.profile("cpu").expect("内置档案");
//! let path = cutile::cap::negotiate(op, cpu).expect("CPU 应命中首档");
//! assert_eq!(path.variant, "pipe2");
//!
//! // NVIDIA 档案在 P0 只有 L0/L1 证据：未验证的能力视为不具备（DESIGN §11.3）。
//! let nvidia = cat.profile("nvidia").expect("内置档案");
//! assert!(cutile::cap::negotiate(op, nvidia).is_err());
//! ```
//!
//! # 工程条款（DESIGN §3.2）
//!
//! - `unsafe` 在本 crate 被整体禁止；P0 的 `c` 后端通过子进程调用 `cc`，不需要 FFI。
//! - 公开项必须有文档；私有项不豁免 clippy。

#![deny(unsafe_code)]

pub mod atom;
pub mod backend;
pub mod builtin;
pub mod cache;
pub mod cap;
pub mod cli;
pub mod host;
pub mod layout;
pub mod lower;
pub mod plan;
pub mod probe;
pub mod prof;
pub mod reference;
pub mod sch;
pub mod space;
pub mod toml;

/// crate 自身的 schema 版本；闭集变更时递增，旧 `plan` 表与缓存随即失效（DESIGN §6.4）。
pub const SCHEMA_VER: u16 = 2;
