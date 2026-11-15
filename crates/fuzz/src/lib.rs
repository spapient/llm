//! 模糊测试的**契约层**：定义"被测入口"与"确定性用例"，不绑定任何具体解析器。
//!
//! **拥有**：`Target`、`Seed`、`Behaviour`、`Finding`、`sweep`。**不拥有**：具体解析器
//! （`cutile` 的模板与片段名）、语料库与进程管理（属于 CI 的 fuzz job）。
//! **依赖**：`diagnostic`（运行时只此一个）。**被依赖**：`tests/parsers.rs`（本 crate 自己的
//! 集成测试，以 `+dev` 依赖接上 `cli` / `weights` 的真实解析器）。
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
//! 三条都由 [`sweep`] 检查。第 1 条靠 [`std::panic::catch_unwind`]：panic 被记成一条带
//! 用例号与输入的 [`Behaviour::Panicked`] 证据，而不是让整次遍历连同已跑过的用例一起消失。
//! 这并不与"不吞掉崩溃"矛盾——崩溃仍然被报告成 [`Finding`]，只是报告里带着"哪条输入
//! 打死了它"。若交给驱动方自己 `catch_unwind`，每个驱动都会重写一遍证据格式，还得自己
//! 保住用例号。
//!
//! panic 自身的消息仍会按宿主默认行为打到 stderr；[`Behaviour::Panicked`] 里带的是从
//! payload 取回的那一份，便于写进报告。本 crate 不装 `unsafe`。
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
    /// 万一真的 `panic` 了，[`sweep`] 会把它记成 [`Behaviour::Panicked`] 证据，而不是让整次
    /// 遍历断在那里——这条兜底不改变"实现不得 `panic`"，只是让违约可定位、可复现。
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
    /// 入口用 panic 结束——违约，附从 payload 取回的消息。
    ///
    /// `panic!` 不带字符串（如 `panic!()` 或断言失败的非字符串 payload）时是占位说明，
    /// 同样是违约：能打死宿主的输入本身就该修。
    Panicked {
        /// payload 里的说明；取不到时是占位文本。
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
    /// 一条证据：哪条用例、什么输入、什么结论、违反了哪条契约。
    fn new(case: u64, input: Vec<u8>, behaviour: Behaviour, contract: &'static str) -> Self {
        Self {
            case,
            input,
            behaviour,
            contract,
        }
    }

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
    (0..cases).try_for_each(|case| check(target, case, seed.derive(case)))
}

/// 一条用例：判两次，并逐条核对三条契约。
///
/// 先 panic、再空消息、最后确定性：顺序就是报告的优先级，也是 [`Finding`] 只带第一条证据的理由。
fn check(target: &dyn Target, case: u64, input: Vec<u8>) -> Result<(), Finding> {
    let first = judge(target, &input);
    if matches!(first, Behaviour::Panicked { .. }) {
        let contract = "不 panic：任何输入都必须被解析成结论，而不是打死宿主";
        return Err(Finding::new(case, input, first, contract));
    }
    if let Behaviour::Rejected { message, .. } = &first
        && message.trim().is_empty()
    {
        return Err(Finding::new(case, input, first, "拒绝时必须给出非空消息"));
    }
    let second = judge(target, &input);
    if first != second {
        let contract = "同一输入必须给出同一个结论";
        return Err(Finding::new(case, input, second, contract));
    }
    Ok(())
}

/// 跑一次并把 `Result` 归一成 [`Behaviour`]，让"接受/拒绝"的比较不依赖 `Error` 的内部结构。
///
/// panic 也在这里归一：入口是不受信任输入的边界，`catch_unwind` 是唯一能让"第 7 条输入打死
/// 进程"变成"第 7 条输入是一条证据"的手段。`AssertUnwindSafe` 的前提是 [`Target::run`] 在
/// panic 之后不会被观察到半截状态——它是 `&self` 的纯判定，没有跨用例的状态需要恢复。
fn judge(target: &dyn Target, input: &[u8]) -> Behaviour {
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| target.run(input))) {
        Ok(Ok(())) => Behaviour::Accepted,
        Ok(Err(error)) => Behaviour::Rejected {
            code: error.code(),
            message: error.message().to_string(),
        },
        Err(payload) => Behaviour::Panicked {
            message: panic_message(&payload),
        },
    }
}

/// 从 panic payload 取回说明：`panic!("…")` 给 `&str`，`panic!("{x}")` 给 `String`。
fn panic_message(payload: &Box<dyn std::any::Any + Send>) -> String {
    payload
        .downcast_ref::<&str>()
        .map(|message| (*message).to_string())
        .or_else(|| payload.downcast_ref::<String>().cloned())
        .unwrap_or_else(|| "panic 没有携带字符串说明".to_string())
}

#[cfg(test)]
mod tests;
