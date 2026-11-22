//! 职责：cutile —— 设备无关的 kernel DSL 与编译前端。
//!
//! 现有生产链路实现 [`README.md`](../README.md) 的模板式 P0 纵切面；`kernel` 是开始迁移的 typed 内核
//! （写回分逐点 / 归约，并可把两个核拼成一个），当前不取代旧模板，也不宣称具备通用优化或 GPU 执行能力。
//!
//! ```text
//! 声明层 crates/op/ops/<op>.{toml,k}   档案层 crates/prof/devices/<vendor>.toml
//! 片段层 prim/<vendor>/<prim>.<ext>（数据在 `crates/prim`，本 crate 只拼接与 emit）
//!        ① expand  →  ② negotiate  →  ③ select  →  ④ substitute  →  ⑤ emit + cache
//! ```
//!
//! 模板链路的五步中间产物全部是可见数据结构（[`decl::Variant`]、[`cap::Path`]、
//! [`plan::Plan`]、[`emit::Source`]）；新 typed IR 逐算子替换这条链路，待设备验证后再迁移执行入口。
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
//! | typed IR：内核 + 写回 + 双目标发射、校验、融合、优化通道 | [`kernel`]、[`validate`]、[`fuse`]、[`pass`] |
//! | ⑤ 指纹、锁、落盘与容量回收 | [`cache`] |
//! | 设备探测与存储寻址 | [`probe`]、[`space`] |
//! | 对拍口径（相对误差 + 按 dtype 阈值） | [`mod@reference`] |
//! | 被测内核的基准封装（`cc` / `cuda`） | [`harness`] |
//! | 输入形状：三个目录，纯数据 | [`dirs`] |
//!
//! 上表到 `emit` + `cache` 为止：旧模板链路把三份输入折成源文本与缓存键，
//! 不碰进程、不碰设备、不跑内核。凡是要拼一次编译、跑一次设备、或给出"能跑/不能跑"结论的模块
//! （`backend`、`build`、`device`、`runner`、`gate`、`pipeline`、`toolchain`、`witness`、
//! `verify`、`execution` 与 `doctor` / `bench` / `ledger` 三份报告）住在 `optim`——那才是
//! "机制 / 命令策略"的分界线：跨过去就必须有证据，而证据来自执行。
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
//! // Intel 档案（gen12/OpenCL）在 P0 仍是 L1 证据：未验证的能力视为不具备（README §11.3）。
//! let intel = cat.profile("intel").expect("内置档案");
//! assert!(cutile::cap::negotiate(op, intel).is_err());
//! # Ok(())
//! # }
//! ```
//!
//! 上面用的是**按档案自述**的协商：§16 的判据问的是声明粒度，不该随本机有没有这块卡而变。
//! 执行路径（`optim` 的 `doctor` / `verify` / `bench`）用 [`cap::negotiate_with`]，它还要求供给
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

#[cfg(test)]
extern crate self as cutile;

pub use annotation::dsl;

// 词汇表与档案 schema 的事实源在 `prof` crate：`cutile` 只用它们，不改写闭集。
// 用 `::prof` 前缀强制指向 crate（本 crate 也导出同名模块，避免路径歧义）。
pub use ::prof::atom;
pub use ::prof::prof;

// 声明层的词汇表（布局构造子、调度模板、模板词法）与解析结果的事实源在 `prof`：
// `cutile` 只消费它们，因此原地转发路径，机制里没有第二份闭集。
// 同样用 `::prof`：本 crate 根上已有一个同名模块（`pub use ::prof::prof`）。
pub use ::prof::decl;
pub use ::prof::sch::layout;
pub use ::prof::{model, sch, sig};

pub mod ast;
pub mod builtin;
pub mod cache;
pub mod cap;
pub mod dirs;
pub mod emit;
pub mod fuse;
pub mod harness;
pub mod kernel;
mod lru;
pub mod pass;
pub mod plan;
pub mod probe;
pub mod reference;
pub mod space;
pub mod substitute;
pub mod validate;
// 生产态内核发射与基准封装（M3-2 / M3-10）已搬到 `engine`（L6）：Metal / CUDA 的 vendor 语法
// 不进 cutile，机制层只经 `llm_optim::backend::Codegen::lower` / `llm_optim::backend::Codegen::harness`
// 缝分派（`docs/ARCHITECTURE.md` 的 L3 契约：cutile 不出现厂商字面量）。
//
// 接缝与命令策略（`backend`、`build`、`device`、`gate`、`pipeline`、`runner`、`toolchain`、
// `witness`、`verify`、`execution`）住在 `optim`：它们要拼一次编译、跑一次设备，而本 crate
// 只做"把三层数据变成源文本"这段旧模板前端（①②③④⑤）。

#[cfg(test)]
#[path = "../tests/ir_exec_tests.rs"]
mod ir_exec_tests;

/// crate 自身的 schema 版本；闭集变更时递增，旧 `plan` 表与缓存随即失效（README §6.4）。
pub const SCHEMA_VER: u16 = 4;
