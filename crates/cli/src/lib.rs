//! 命令行前端：把 `argv` 解析成 [`Command`]，**不执行任何业务**。
//!
//! **拥有**：`Command`、`Invocation`、`USAGE`、`parse`，以及内核层审计的子命令文法 [`cutile::CutileCommand`]。
//! **不拥有**：业务实现（各层自己）、输出排版（`tool`）、配置（`llm-config`）、退出码语义（`llm` 决定）。
//! **依赖**：`diagnostic`。**被依赖**：`llm`（`llm` 与它的 `cutile` 二进制）。
//!
//! 解析与执行分开的理由：`llm` 是组装点，它的 `main` 只该做"解析 → 分派 → 退出码"三件事。
//! 把参数文法写进组装点或写进被驱动的层（`cutile` 曾经自己认 `doctor` / `verify` / `bench`），
//! 等于多出一套文法：两个入口对同一个选项的解释会慢慢分叉。
//!
//! # 例子
//!
//! ```
//! use llm_cli::{Command, Invocation};
//!
//! let invocation = Invocation::parse(&[]).expect("空参数即默认命令");
//! assert_eq!(invocation.command(), &Command::Status);
//!
//! let verify = Invocation::parse(&["verify".to_string(), "--op".to_string(), "gemv_i4".to_string()])
//!     .expect("合法参数");
//! assert_eq!(verify.command(), &Command::Verify { op: Some("gemv_i4".to_string()) });
//!
//! assert!(Invocation::parse(&["frobnicate".to_string()]).is_err(), "未知命令不猜");
//! ```

#![deny(unsafe_code)]

pub mod cutile;

use diagnostic::{Code, Error, Result};

/// 用法文本：唯一一份，`help` 与错误信息都引用它。
pub const USAGE: &str = "\
用法：llm [命令] [选项]

命令：
  status                 打印各层状态（默认）
  verify --op <算子>      校验一个算子的协商与对拍
  bench  --op <算子>      测量一个算子的带宽与达成率
  help                   打印本说明

选项：
  --op <算子>             算子名（`verify` / `bench` 用，缺省为 `gemv_i4`）
  -h, --help             等价于 `help`
";

/// 顶层命令；只描述"要做什么"，不描述"怎么做"。
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Command {
    /// 打印各层状态。
    Status,
    /// 校验一个算子。
    Verify {
        /// 算子名；`None` 表示用命令的默认值。
        op: Option<String>,
    },
    /// 测量一个算子。
    Bench {
        /// 算子名；`None` 表示用命令的默认值。
        op: Option<String>,
    },
    /// 打印用法。
    Help,
}

/// 一次调用：解析后的命令。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Invocation {
    command: Command,
}

impl Invocation {
    /// 解析参数（不含 `argv[0]`）。
    ///
    /// # Errors
    ///
    /// 未知命令、未知选项、缺少取值、重复选项都返回 [`Code::Invalid`]，
    /// 并在消息里给出 [`USAGE`]——错误信息本身就是下一条命令的提示。
    pub fn parse(args: &[String]) -> Result<Self> {
        let mut words = args.iter().map(String::as_str);
        let command = match words.next() {
            None | Some("status") => Command::Status,
            Some("help" | "-h" | "--help") => Command::Help,
            Some("verify") => Command::Verify {
                op: parse_op(&mut words)?,
            },
            Some("bench") => Command::Bench {
                op: parse_op(&mut words)?,
            },
            Some(other) => {
                return Err(Error::new(Code::Invalid, format!("未知命令 `{other}`"))
                    .with("cli::Invocation::parse"));
            }
        };
        if let Some(extra) = words.next() {
            return Err(Error::new(Code::Invalid, format!("多余参数 `{extra}`"))
                .with("cli::Invocation::parse"));
        }
        Ok(Self { command })
    }

    /// 解析后的命令。
    pub fn command(&self) -> &Command {
        &self.command
    }
}

/// `--op <name>`：最多出现一次，取值不能为空或看起来像另一个选项。
fn parse_op<'a>(words: &mut impl Iterator<Item = &'a str>) -> Result<Option<String>> {
    let Some(flag) = words.next() else {
        return Ok(None);
    };
    if flag != "--op" {
        return Err(
            Error::new(Code::Invalid, format!("期望 `--op <算子>`，得到 `{flag}`"))
                .with("cli::Invocation::parse"),
        );
    }
    let value = words.next().ok_or_else(|| {
        Error::new(Code::Invalid, "`--op` 缺少取值").with("cli::Invocation::parse")
    })?;
    if value.is_empty() || value.starts_with('-') {
        return Err(
            Error::new(Code::Invalid, format!("`--op` 的取值不合法：`{value}`"))
                .with("cli::Invocation::parse"),
        );
    }
    Ok(Some(value.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(list: &[&str]) -> Vec<String> {
        list.iter().map(|item| item.to_string()).collect()
    }

    #[test]
    fn the_default_command_is_status() {
        assert_eq!(
            Invocation::parse(&[]).expect("默认").command(),
            &Command::Status
        );
        assert_eq!(
            Invocation::parse(&args(&["status"]))
                .expect("显式")
                .command(),
            &Command::Status
        );
    }

    #[test]
    fn help_has_three_spellings() {
        for spelling in ["help", "-h", "--help"] {
            assert_eq!(
                Invocation::parse(&args(&[spelling]))
                    .expect("帮助")
                    .command(),
                &Command::Help,
                "{spelling}"
            );
        }
    }

    #[test]
    fn an_operator_is_carried_through_without_defaulting_here() {
        assert_eq!(
            Invocation::parse(&args(&["verify", "--op", "gemv_i4"]))
                .expect("合法")
                .command(),
            &Command::Verify {
                op: Some("gemv_i4".to_string())
            }
        );
        assert_eq!(
            Invocation::parse(&args(&["bench"]))
                .expect("无选项")
                .command(),
            &Command::Bench { op: None }
        );
    }

    #[test]
    fn unknown_words_and_flags_are_rejected_with_the_usage_hint() {
        let unknown = Invocation::parse(&args(&["frobnicate"])).expect_err("未知命令");
        assert_eq!(unknown.code(), Code::Invalid);
        assert!(unknown.to_string().contains("frobnicate"), "{unknown}");

        let flag = Invocation::parse(&args(&["verify", "--op"])).expect_err("缺取值");
        assert!(flag.to_string().contains("缺少取值"), "{flag}");

        let extra = Invocation::parse(&args(&["status", "now"])).expect_err("多余参数");
        assert!(extra.to_string().contains("now"), "{extra}");
    }

    #[test]
    fn the_usage_text_lists_every_command() {
        for command in ["status", "verify", "bench", "help"] {
            assert!(USAGE.contains(command), "{command}");
        }
    }
}
