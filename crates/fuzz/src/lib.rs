//! 模糊测试的**契约层**：定义"被测入口"、确定性用例生成与"什么算违约"，不绑定任何具体解析器。
//!
//! **拥有**：[`Target`]、[`Seed`]、[`Behaviour`]、[`Finding`]、[`check_target`] / [`sweep`] /
//! [`sweep_all`]。**不拥有**：具体解析器（`cutile` 的模板与片段名）、语料库与进程管理
//! （属于 CI 的 fuzz job）。**依赖**：`diagnostic`（运行时只此一个）。**被依赖**：
//! `tests/parsers.rs`（本 crate 自己的集成测试，以 `+dev` 依赖接上 `cli` / `weights` 等
//! 真实解析器）与 `src/fuzz_*.rs`（libfuzzer harness 入口；判据与种子在 `src/drivers.rs` /
//! `src/seeds.rs`）。
//!
//! 为什么表单独立在一个 crate：真正需要它的不是一个解析器，而是**每一个**解析器。把
//! "什么算失败"写在一处，才能保证命令行文法与 safetensors 头部被同一套判据检查，而不是各写
//! 一份 `#[test]` 里再手抄一遍。
//!
//! # 三条契约
//!
//! 一次运行是否"符合预期"不是感觉，而是下面三条之一被违反：
//!
//! 1. **不 panic**：任何输入都不该打死宿主进程；
//! 2. **结论稳定**：同一输入跑两次必须给出同样的接受/拒绝、同样的类别与消息；
//! 3. **拒绝要有话说**：拒绝时消息为空，等于"失败了但不知道为什么"，这是诊断缺陷。
//!
//! 三条都由 [`check_target`] 核对。第 1 条靠 [`std::panic::catch_unwind`]：panic 被记成一条带
//! 目标名、用例号与输入的 [`Behaviour::Panicked`] 证据，而不是让整次遍历连同已跑过的用例
//! 一起消失。这并不与"不吞掉崩溃"矛盾——崩溃仍然被报告成 [`Finding`]，只是报告里带着"哪条
//! 输入打死了它"。若交给驱动方自己 `catch_unwind`，每个驱动都会重写一遍证据格式，还得自己
//! 保住用例号。
//!
//! panic 自身的消息仍会按宿主默认行为打到 stderr；[`Behaviour::Panicked`] 里带的是从
//! payload 取回的那一份，便于写进报告。本 crate 不装 `unsafe`。
//!
//! # 模块
//!
//! - [`contract`]：确定性种子、变异调度、失败证据与三条契约核对入口同住一处。
//!
//! # 例子
//!
//! ```
//! use diagnostic::{Code, Error};
//! use llm_fuzz::{Seed, Target, sweep};
//!
//! struct Bytes;
//!
//! impl Target for Bytes {
//!     fn name(&self) -> &str {
//!         "bytes"
//!     }
//!
//!     fn run(&self, input: &[u8]) -> Result<(), Error> {
//!         if input.is_empty() {
//!             return Err(Error::new(Code::Invalid, "输入不能为空"));
//!         }
//!         Ok(())
//!     }
//! }
//!
//! let seed = Seed::new(b"abc");
//! assert!(sweep(&Bytes, &seed, 4).is_ok(), "拒绝本身是合法结果");
//! ```

#![deny(unsafe_code)]

// 让本 crate 以库名自引用：`drivers.rs` 被三个外部目标 `#[path]` 包含（`tests/parsers.rs`、
// `src/fuzz_*.rs`、`tests/gen_corpus.rs`）时，它们通过 `llm_fuzz::Target` 取判据 trait；
// 库自身的测试模块也走同一条路径，避免再要求每个包含方在 crate 根各补一个 `Target`。
// 自引用只在 `#[cfg(test)]` 包含的 `drivers` 里被消费，纯 lib 构建不引用，故放开未用告警。
#[allow(unused_extern_crates)]
extern crate self as llm_fuzz;

pub mod contract;
pub use contract::Seed;
pub use contract::Target;
pub use contract::{Behaviour, Finding, Repro, repro};
pub use contract::{check_target, sweep, sweep_all};
// 契约单测与共享定义：`tests/parsers.rs`、`src/fuzz_*.rs` 与库内单测取同一份驱动，
// 不再各留一份。`drivers` 在测试构建里把 `seeds` 一起编进来（`#[path]`），并给出
// 「名称/适配器/种子」三合一的登记表，供遍历与语料生成共用。
#[cfg(test)]
#[allow(dead_code)]
mod drivers;
#[cfg(test)]
#[path = "acceptance.rs"]
mod fuzz_tests;
