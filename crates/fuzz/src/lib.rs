//! 模糊测试的**契约层**：定义"被测入口"、确定性用例生成与"什么算违约"，不绑定任何具体解析器。
//!
//! **拥有**：[`Target`]、[`Seed`]、[`Behaviour`]、[`Finding`]、[`check`] / [`sweep`] /
//! [`sweep_all`]。**不拥有**：具体解析器（`cutile` 的模板与片段名）、语料库与进程管理
//! （属于 CI 的 fuzz job）。**依赖**：`diagnostic`（运行时只此一个）。**被依赖**：
//! `tests/parsers.rs`（本 crate 自己的集成测试，以 `+dev` 依赖接上 `cli` / `weights` 等
//! 真实解析器）与 `tests/fuzz_*.rs`（libfuzzer harness）。
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
//! 三条都由 [`check`] 核对。第 1 条靠 [`std::panic::catch_unwind`]：panic 被记成一条带
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
//! - [`seed`]：确定性种子与变异调度（SplitMix64 流 + 位/字节翻转、边界字节、长度增减）；
//! - [`finding`]：结论、证据与可复现的输入呈现；
//! - [`contract`]：三条契约的核对入口，sweep 与 harness 共用。
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

use diagnostic::Error;

pub mod contract;
pub mod finding;
pub mod seed;

pub use contract::{check, sweep, sweep_all};
pub use finding::{Behaviour, Finding, Repro, repro};
pub use seed::Seed;

/// 被测入口：把任意字节变成"接受 / 拒绝"。
pub trait Target {
    /// 目标名：写进失败证据，便于把问题定位到具体解析器。
    fn name(&self) -> &str;

    /// 跑一次输入。
    ///
    /// 实现不得 `panic`；接受与拒绝都必须能被调用方原样复现（见 [`check`] 的三条契约）。
    /// 万一真的 `panic` 了，[`contract::judge`] 会把它记成 [`Behaviour::Panicked`] 证据，
    /// 而不是让整次遍历断在那里——这条兜底不改变"实现不得 `panic`"，只是让违约可定位、可复现。
    fn run(&self, input: &[u8]) -> Result<(), Error>;

    /// 单条输入跑完整三条契约核对。
    ///
    /// libfuzzer harness 调这条而不是直接调 [`run`](Self::run)：于是覆盖引导跑到的输入
    /// 同样受"结论稳定 / 拒绝有话说"约束，不只是不 panic。违约以 [`Finding`] 返回，用例号记
    /// 0——模糊输入不是派生用例。
    fn exercise(&self, input: &[u8]) -> Result<(), Finding>
    where
        Self: Sized,
    {
        check(self, 0, input.to_vec())
    }

    /// harness 入口：与 [`Self::exercise`] 同一套判据，违约时把 [`Finding`] 转成 panic。
    ///
    /// harness 没有上报通道，违约必须终止进程；libFuzzer 会留下触发输入作为复现样本。panic
    /// 转换只此一处，各 harness 不再各写一遍 `if let Err(...) { panic!(...) }`。
    fn exercise_or_panic(&self, input: &[u8])
    where
        Self: Sized,
    {
        if let Err(finding) = self.exercise(input) {
            panic!("{finding}");
        }
    }
}

#[cfg(test)]
mod tests;
