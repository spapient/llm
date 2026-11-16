//! 一次运行的结论与失败证据：判据违约定型成可复现、可贴进 issue 的值。

use core::fmt;

use diagnostic::Code;

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

impl fmt::Display for Behaviour {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Accepted => formatter.write_str("accepted"),
            Self::Rejected { code, message } => write!(formatter, "rejected {code}: {message}"),
            Self::Panicked { message } => write!(formatter, "panicked: {message}"),
        }
    }
}

/// 失败证据：目标名、用例序号、输入是什么、违反了哪条契约。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Finding {
    target: String,
    case: u64,
    input: Vec<u8>,
    behaviour: Behaviour,
    contract: &'static str,
}

impl Finding {
    /// 一条证据：哪条用例、什么输入、什么结论、违反了哪条契约。
    pub(crate) fn new(
        target: &str,
        case: u64,
        input: Vec<u8>,
        behaviour: Behaviour,
        contract: &'static str,
    ) -> Self {
        Self {
            target: target.to_string(),
            case,
            input,
            behaviour,
            contract,
        }
    }

    /// 被测目标名：把问题定位到具体解析器。
    pub fn target(&self) -> &str {
        &self.target
    }

    /// 用例序号（与 [`crate::Seed::derive`] 的参数一致；libfuzzer 喂来的输入记 0）。
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

impl fmt::Display for Finding {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            formatter,
            "fuzz::{} case #{}: {}",
            self.target, self.case, self.contract
        )?;
        writeln!(formatter, "  verdict: {}", self.behaviour)?;
        write!(formatter, "  input: {}", repro(&self.input))
    }
}

/// 输入的可读复现形式：能当 UTF-8 读就给 Rust 字符串字面量（带转义），否则给紧凑十六进制。
///
/// 两种形态都能直接贴进 issue 或 harness 复现命令：字符串字面量保留可读字符，十六进制覆盖
/// 任意字节，不依赖终端能不能显示。
#[derive(Debug, Clone, Copy)]
pub struct Repro<'a>(&'a [u8]);

/// 输入的可读复现形式（见 [`Repro`]）。
pub fn repro(input: &[u8]) -> Repro<'_> {
    Repro(input)
}

impl fmt::Display for Repro<'_> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match std::str::from_utf8(self.0) {
            Ok(text) => write!(formatter, "{text:?}"),
            Err(_) => {
                const NIBBLE: [u8; 16] = *b"0123456789abcdef";
                let mut hex = String::with_capacity(self.0.len() * 2);
                for &byte in self.0 {
                    hex.push(NIBBLE[(byte >> 4) as usize] as char);
                    hex.push(NIBBLE[(byte & 0x0F) as usize] as char);
                }
                write!(formatter, "0x{hex}")
            }
        }
    }
}
