//! 模糊测试的**契约层**：定义"被测入口"与"确定性用例"，不绑定任何具体解析器。
//!
//! **拥有**：`Target`、`Seed`、`Behaviour`、`Finding`、`sweep`。**不拥有**：具体解析器
//! （`llm-config` 的 TOML、`cutile` 的模板与片段名）、语料库与进程管理（属于 CI 的 fuzz job）。
//! **依赖**：`diagnostic`。**被依赖**：暂无（驱动方将是 CI 的 fuzz job）。
//!
//! 为什么表单独立在一个 crate：真正需要它的不是一个解析器，而是**每一个**解析器。把
//! "什么算失败"写在一处，才能保证 TOML 解析器与模板解析器被同一套判据检查，而不是各写一份
//! `#[test]` 里再手抄一遍。
//!
//! # 三条契约
//!
//! 一次运行是否"符合预期"不是感觉，而是下面三条之一被违反：
//!
//! 1. **不 panic**：`panic` 无法被这里捕获，因此驱动方必须把它当作失败；
//! 2. **结论稳定**：同一输入跑两次必须给出同样的接受/拒绝、同样的类别与消息；
//! 3. **拒绝要有话说**：拒绝时消息为空，等于"失败了但不知道为什么"，这是诊断缺陷。
//!
//! 第 2、3 条由 [`sweep`] 检查；第 1 条由调用方负责——本 crate 不装 `unsafe`，
//! 也不替调用方吞掉崩溃。
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

use diagnostic::{Code, Error};

/// 被测入口：把任意字节变成"接受 / 拒绝"。
pub trait Target {
    /// 目标名：写进失败证据，便于把问题定位到具体解析器。
    fn name(&self) -> &str;

    /// 跑一次输入。
    ///
    /// 实现不得 `panic`；接受与拒绝都必须能被调用方原样复现（见 [`sweep`] 的三条契约）。
    fn run(&self, input: &[u8]) -> Result<(), Error>;
}

/// 确定性种子：同一个种子永远派生同一批用例，失败才能复现。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Seed(Vec<u8>);

impl Seed {
    /// 由初始字节构造；空种子是合法输入（"空输入"本身就是一道用例）。
    pub fn new(bytes: impl Into<Vec<u8>>) -> Self {
        Self(bytes.into())
    }

    /// 初始字节。
    pub fn bytes(&self) -> &[u8] {
        &self.0
    }

    /// 初始字节数。
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// 是否为空种子。
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// 第 `index` 个用例。
    ///
    /// 只做整数混合与逐字节异或：跨平台确定、无随机源、无全局状态，因此同一 `(seed, index)`
    /// 在任何机器上都得到同一串字节——这是"失败可复现"的前提。
    pub fn derive(&self, index: u64) -> Vec<u8> {
        let mut state = index ^ 0x9e37_79b9_7f4a_7c15;
        let mut out = Vec::with_capacity(self.0.len());
        for (position, byte) in self.0.iter().enumerate() {
            state = state
                .wrapping_mul(0x0100_0000_01b3)
                .wrapping_add(position as u64);
            state ^= state >> 29;
            out.push(byte ^ (state >> ((position % 8) * 8)) as u8);
        }
        out
    }
}

/// 一次运行的结论。
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Behaviour {
    /// 输入被接受。
    Accepted,
    /// 输入被拒绝，附错误类别与消息。
    Rejected {
        /// 错误类别（日志与退出码的对外契约）。
        code: Code,
        /// 一句话说明；空消息属于违约。
        message: String,
    },
}

/// 失败证据：第几个用例、输入是什么、违反了哪条契约。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Finding {
    case: u64,
    input: Vec<u8>,
    behaviour: Behaviour,
    contract: &'static str,
}

impl Finding {
    /// 用例序号（与 [`Seed::derive`] 的参数一致）。
    pub fn case(&self) -> u64 {
        self.case
    }

    /// 触发问题的输入。
    pub fn input(&self) -> &[u8] {
        &self.input
    }

    /// 这条输入得到的结论。
    pub fn behaviour(&self) -> &Behaviour {
        &self.behaviour
    }

    /// 被违反的契约，直接写进报告。
    pub fn contract(&self) -> &'static str {
        self.contract
    }
}

/// 跑 `cases` 个确定性用例；全部符合契约返回 `Ok(())`，否则返回第一条证据。
///
/// `cases` 为 0 时不做任何事：遍历范围由调用方决定，本函数不猜"默认跑多少个"。
pub fn sweep(target: &dyn Target, seed: &Seed, cases: u64) -> Result<(), Finding> {
    for case in 0..cases {
        let input = seed.derive(case);
        let first = judge(target, &input);
        if let Behaviour::Rejected { message, .. } = &first
            && message.trim().is_empty()
        {
            return Err(Finding {
                case,
                input,
                behaviour: first,
                contract: "拒绝时必须给出非空消息",
            });
        }
        let second = judge(target, &input);
        if first != second {
            return Err(Finding {
                case,
                input,
                behaviour: second,
                contract: "同一输入必须给出同一个结论",
            });
        }
    }
    Ok(())
}

/// 跑一次并把 `Result` 归一成 [`Behaviour`]，让"接受/拒绝"的比较不依赖 `Error` 的内部结构。
fn judge(target: &dyn Target, input: &[u8]) -> Behaviour {
    match target.run(input) {
        Ok(()) => Behaviour::Accepted,
        Err(error) => Behaviour::Rejected {
            code: error.code(),
            message: error.message().to_string(),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell;

    /// 永远接受的入口。
    struct Accept;

    impl Target for Accept {
        fn name(&self) -> &str {
            "accept"
        }

        fn run(&self, _input: &[u8]) -> Result<(), Error> {
            Ok(())
        }
    }

    /// 永远拒绝、但给出说明的入口。
    struct Reject;

    impl Target for Reject {
        fn name(&self) -> &str {
            "reject"
        }

        fn run(&self, _input: &[u8]) -> Result<(), Error> {
            Err(Error::new(Code::Invalid, "这段字节不是一个合法的键值对"))
        }
    }

    /// 同一次 `sweep` 里对同一输入给出两种结论——正是第 2 条契约要抓的东西。
    struct Unstable(Cell<bool>);

    impl Target for Unstable {
        fn name(&self) -> &str {
            "unstable"
        }

        fn run(&self, _input: &[u8]) -> Result<(), Error> {
            let flip = self.0.get();
            self.0.set(!flip);
            if flip {
                Err(Error::new(Code::Invalid, "第二次才拒绝"))
            } else {
                Ok(())
            }
        }
    }

    #[test]
    fn a_seed_derives_the_same_case_every_time() {
        let seed = Seed::new(b"gemv_i4");
        assert_eq!(seed.derive(7), seed.derive(7), "确定性是复现的前提");
        assert_ne!(seed.derive(7), seed.derive(8), "不同用例不该撞在一起");
    }

    #[test]
    fn an_empty_seed_is_a_legal_case() {
        let seed = Seed::new(b"");
        assert!(seed.is_empty());
        assert_eq!(seed.derive(3), Vec::<u8>::new());
        assert_eq!(seed.len(), 0);
    }

    #[test]
    fn acceptance_and_rejection_are_both_legal_outcomes() {
        let seed = Seed::new(b"abc");
        assert!(sweep(&Accept, &seed, 3).is_ok());
        assert!(sweep(&Reject, &seed, 3).is_ok(), "拒绝不是失败");
    }

    #[test]
    fn zero_cases_touches_nothing() {
        assert!(sweep(&Unstable(Cell::new(false)), &Seed::new(b"x"), 0).is_ok());
    }

    #[test]
    fn an_empty_rejection_message_is_a_finding() {
        struct Silent;

        impl Target for Silent {
            fn name(&self) -> &str {
                "silent"
            }

            fn run(&self, _input: &[u8]) -> Result<(), Error> {
                Err(Error::new(Code::Invalid, "   "))
            }
        }

        let finding = sweep(&Silent, &Seed::new(b"abc"), 1).expect_err("空消息");
        assert_eq!(finding.contract(), "拒绝时必须给出非空消息");
        assert_eq!(
            finding.behaviour(),
            &Behaviour::Rejected {
                code: Code::Invalid,
                message: "   ".to_string()
            }
        );
    }

    #[test]
    fn an_unstable_target_is_caught_with_its_case_and_input() {
        let seed = Seed::new(b"abc");
        let finding = sweep(&Unstable(Cell::new(false)), &seed, 2).expect_err("结论不稳定");
        assert_eq!(finding.contract(), "同一输入必须给出同一个结论");
        assert_eq!(finding.case(), 0, "第一条证据就是第 0 个用例");
        assert_eq!(finding.input(), seed.derive(0).as_slice());
    }
}
