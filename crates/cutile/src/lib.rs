//! cutile —— 设备无关的替换式 kernel DSL。
//!
//! 本 crate 实现 [`README.md`](../README.md) 的 P0 纵切面：
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
//! # 模块地图
//!
//! | 职责 | 模块 |
//! |---|---|
//! | 三层数据的装载与交叉校验 | [`builtin`] |
//! | ② 协商、降级链、缺项诊断 | [`cap`] |
//! | ③ 变体选择与 roofline 估值 | [`plan`] |
//! | ④ 槽位与 `@原语` 替换 | [`substitute`]、[`ast`] |
//! | ⑤ 成文与片段行数判据 | [`emit`] |
//! | ⑤ 指纹、锁、落盘 | [`build`] |
//! | ⑤ 指纹、锁、落盘与容量回收 | [`cache`] |
//! | 设备探测、存储容量与寻址 | [`probe`]、[`space`] |
//! | 对拍口径（相对误差 + 按 dtype 阈值） | [`mod@reference`] |
//! | 工具链探测与身份 | [`toolchain`] |
//! | 编译/执行/回读原语 | `runner` |
//! | 被测内核的基准封装（`cc` / `cuda`） | `harness` |
//! | 统一设备：按 `Emit` 分发的描述符表与编排行 | `device` |
//! | 见证执行与语言门 | [`witness`]、`gate` |
//! | 四条命令共用的流水线 | `pipeline` |
//! | 命令策略与结构化报告 | [`doctor`]、[`verify`]、[`mod@bench`]、[`execution`] |
//! | 输入形状：三个目录，纯数据 | [`dirs`] |
//!
//! 上表按"机制 / 命令策略"分界读：`runner`、`harness`、`device`、`gate`、`pipeline` 是私有模块，因为它们是接缝
//! 而不是 API——外部只该看到"能跑/不能跑"的结论，不该直接拼一次编译。
//!
//! # 例子
//!
//! 同一份算子声明、同一份模板，在三份档案上协商出不同的能力路径：
//!
//! ```
//! use cutile::builtin::Catalogue;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let cat = Catalogue::try_builtin()?;
//! let op = cat.op("gemv_i4").expect("内置算子");
//! let cpu = cat.profile("cpu").expect("内置档案");
//! let path = cutile::cap::negotiate(op, cpu).expect("CPU 应命中首档");
//! assert_eq!(path.variant, "pipe2");
//!
//! // NVIDIA 档案在 P0 只有 L0/L1 证据：未验证的能力视为不具备（README §11.3）。
//! let nvidia = cat.profile("nvidia").expect("内置档案");
//! assert!(cutile::cap::negotiate(op, nvidia).is_err());
//! # Ok(())
//! # }
//! ```
//!
//! 上面用的是**按档案自述**的协商：§16 的判据问的是声明粒度，不该随本机有没有这块卡而变。
//! 执行路径（本 crate 的 `doctor` / `verify` / `bench`）用 [`cap::negotiate_with`]，它还要求供给
//! 出现在这次执行得到的 [`cap::Proof`] 里——`grade` 是执行的结论，不是 TOML 的结论。
//!
//! # 工程条款（README §3.2）
//!
//! - `unsafe` 在本 crate 被整体禁止；P0 的 C 通道通过子进程调用 `cc`，不需要 FFI。
//! - 公开项必须有文档；私有项不豁免 clippy。
//! - **配置是输入，不是职责**：本 crate 只声明我要哪三个目录（[`dirs::Dirs`]），默认值、环境
//!   变量与 `--set` 的解析住在调用方（`cli::settings`，ADR 0023）。因此内核里没有一处读
//!   `CUTILE_*`，换一个嵌入方不需要改这里任何一行。

#![deny(unsafe_code)]

// 词汇表与档案 schema 的事实源在 `prof` crate：`cutile` 只用它们，不改写闭集。
// 用 `::prof` 前缀强制指向 crate（本 crate 也导出同名模块，避免路径歧义）。
pub use ::prof::atom;
pub use ::prof::prof;

// 声明层的词汇表（布局构造子、调度模板、模板词法）与解析结果的事实源在 `op`：
// `cutile` 只消费它们，因此原地转发路径，机制里没有第二份闭集。
pub use ::op::decl;
pub use ::op::{layout, sch};

pub mod ast;
pub mod bench;
pub mod build;
pub mod builtin;
pub mod cache;
mod cache_key;
pub mod cap;
pub mod dirs;
pub mod doctor;
pub mod emit;
pub mod execution;
pub mod plan;
pub mod probe;
pub mod reference;
pub mod space;
pub mod substitute;
pub mod toolchain;
pub mod verify;
pub mod witness;

#[cfg(test)]
mod cap_tests;

mod device;
mod gate;
mod harness;
mod pipeline;
mod runner;

/// crate 自身的 schema 版本；闭集变更时递增，旧 `plan` 表与缓存随即失效（README §6.4）。
///
/// v3：属性键闭集增加 `exec`，档案增加 `[units.<name>]`（执行单元与其可访问的存储层）。
/// v4：缓存指纹锚定结构化 AST（`FNV(ast::encode)`），旧的裸模板哈希条目随即失效。
pub const SCHEMA_VER: u16 = 4;
