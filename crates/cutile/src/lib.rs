//! cutile —— 设备无关的替换式 kernel DSL。
//!
//! 本 crate 实现 [`DESIGN.md`](../DESIGN.md) 的 P0 纵切面：
//!
//! ```text
//! 声明层 crates/op/ops/<op>.{toml,k}   档案层 crates/prof/devices/<vendor>.toml
//! 片段层 prim/<vendor>/<prim>.<ext>（数据在 `crates/prim`，本 crate 只拼接与 emit）
//!        ① expand  →  ② negotiate  →  ③ select  →  ④ substitute  →  ⑤ emit + cache
//! ```
//!
//! 五个步骤的中间产物全部是可见的数据结构（[`decl::Variant`]、[`cap::Path`]、
//! [`plan::Plan`]、[`emit::Source`]），没有任何隐藏的优化阶段——这是"无 IR"的具体含义。
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
//! 上面用的是**按档案自述**的协商：§16 的判据问的是声明粒度，不该随本机有没有这块卡而变。
//! 执行路径（`audit` 的 `doctor` / `verify` / `bench`）用 [`cap::negotiate_with`]，它还要求供给
//! 出现在这次执行得到的 [`cap::Proof`] 里——`grade` 是执行的结论，不是 TOML 的结论。
//!
//! # 工程条款（DESIGN §3.2）
//!
//! - `unsafe` 在本 crate 被整体禁止；P0 的 `c` 后端通过子进程调用 `cc`，不需要 FFI。
//! - 公开项必须有文档；私有项不豁免 clippy。

#![deny(unsafe_code)]

// 词汇表与档案 schema 的事实源在 `prof` crate：`cutile` 只用它们，不改写闭集。
// 用 `::prof` 前缀强制指向 crate（本 crate 也导出同名模块，避免路径歧义）。
pub use ::prof::atom;
pub use ::prof::prof;

// 声明层的词汇表（布局构造子、调度模板、模板词法）与解析结果的事实源在 `op`：
// `cutile` 只消费它们，因此原地转发路径，机制里没有第二份闭集。
pub use ::op::decl;
pub use ::op::{layout, sch};

pub mod backend;
pub mod build;
pub mod builtin;
pub mod cache;
pub mod cap;
pub mod emit;
pub mod plan;
pub mod probe;
pub mod reference;
pub mod space;
pub mod substitute;

/// TOML 解析器由 `llm-config` 提供：实现只有一份，`cutile` 只转发路径。
pub use llm_config::toml;

/// crate 自身的 schema 版本；闭集变更时递增，旧 `plan` 表与缓存随即失效（DESIGN §6.4）。
///
/// v3：属性键闭集增加 `exec`，档案增加 `[units.<name>]`（执行单元与其可访问的存储层）。
pub const SCHEMA_VER: u16 = 3;
