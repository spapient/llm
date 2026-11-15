//! 命令行前端：把 `argv` 与环境变量解析成 [`Command`] 与 [`settings::Settings`]，**不执行任何业务**。
//!
//! **拥有**：`Command`、`Invocation`、`USAGE`、`parse`、内核层审计的子命令文法 [`cutile::CutileCommand`]，
//! 以及三个目录开关的解析策略 [`settings`]（默认值、`CUTILE_*`、`--set`、来源追踪）。
//! **不拥有**：业务实现（各层自己）、输出排版（`tool`）、退出码语义（`llm` 决定）。
//! **依赖**：`diagnostic`。**被依赖**：`llm`（`llm` 与它的 `cutile` 二进制）。
//!
//! 配置为什么在这一层：默认目录是**部署策略**（面向发布后的二进制），`--set` 是**命令行文法**；
//! 两者都是"这次调用怎么来的"，而不是"内核怎么实现"。内核只收 `cutile::dirs::Dirs` 这份纯数据，
//! 于是它可以被测试、服务或别的宿主按各自的约定接管（ADR 0023）。
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
//! assert_eq!(invocation.command(), &Command::Status { weights: None });
//!
//! let verify = Invocation::parse(&["verify".to_string(), "--op".to_string(), "gemv_i4".to_string()])
//!     .expect("合法参数");
//! assert_eq!(verify.command(), &Command::Verify { op: Some("gemv_i4".to_string()) });
//!
//! assert!(Invocation::parse(&["frobnicate".to_string()]).is_err(), "未知命令不猜");
//! ```

#![deny(unsafe_code)]

pub mod cutile;
pub mod settings;

use diagnostic::{Code, Error, Result};

/// 用法文本：唯一一份，`help` 与错误信息都引用它。
pub const USAGE: &str = "\
用法：llm [命令] [选项]

命令：
  status                 打印各层状态（默认）
  run    --weights <文件> 用真实权重算一次前向（每条链一层内核）
  verify --op <算子>      校验一个算子的协商与对拍
  bench  --op <算子>      测量一个算子的带宽与达成率
  help                   打印本说明

选项：
  --op <算子>             算子名（`verify` / `bench` 用，缺省为 `gemv_i4`）
  --weights <文件>        safetensors 权重文件（`status` / `run` 用；`run` 必需）
  --input <a,b,c>         `run` 的输入向量（逗号分隔 f32；不给则用确定性图案）
  --device <设备>         `run` 的设备名（缺省 `auto`，按指纹选）
  -h, --help             等价于 `help`
";

/// 顶层命令；只描述"要做什么"，不描述"怎么做"。
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Command {
    /// 打印各层状态。
    Status {
        /// safetensors 权重文件路径；`None` 表示只用声明层。
        weights: Option<String>,
    },
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
    /// 用真实权重算一次。
    Run {
        /// safetensors 权重文件路径（必需）。
        weights: String,
        /// 输入向量（逗号分隔的 f32）；`None` 表示用确定性图案。
        input: Option<String>,
        /// 设备名；`None` 表示 `auto`（按指纹选）。
        device: Option<String>,
    },
    /// 打印用法。
    Help,
}

/// 一次调用：解析后的命令。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Invocation {
    command: Command,
}

/// 参数游标：按词推进，`peek` 用于「下一个词是不是这个选项」。
type Words<'a> =
    std::iter::Peekable<std::iter::Map<std::slice::Iter<'a, String>, fn(&String) -> &str>>;

/// 只接受一个选项的子命令：名字、唯一选项、取值说明、命令构造。
///
/// 表即文法：`status` 只认 `--weights`，`verify`/`bench` 只认 `--op`；新增这类子命令只加一行。
const SINGLE: &[Single] = &[
    ("status", "--weights", "文件", |weights| Command::Status {
        weights,
    }),
    ("verify", "--op", "算子", |op| Command::Verify { op }),
    ("bench", "--op", "算子", |op| Command::Bench { op }),
];

/// 只接受一个选项的子命令：名字、唯一选项、取值说明、命令构造。
type Single = (
    &'static str,
    &'static str,
    &'static str,
    fn(Option<String>) -> Command,
);

/// `run` 的选项与取值说明：顺序自由，权重文件必需。
const RUN: &[(&str, &str)] = &[
    ("--weights", "文件"),
    ("--input", "数值列表"),
    ("--device", "设备"),
];

impl Invocation {
    /// 解析参数（不含 `argv[0]`）。
    ///
    /// # Errors
    ///
    /// 未知命令、未知选项、缺少取值、重复选项都返回 [`Code::Invalid`]，
    /// 并在消息里给出 [`USAGE`]——错误信息本身就是下一条命令的提示。
    pub fn parse(args: &[String]) -> Result<Self> {
        let mut words = args
            .iter()
            .map(String::as_str as fn(&String) -> &str)
            .peekable();
        let head = words.next();
        let command = command(head, &mut words)?;
        if let Some(extra) = words.next() {
            return Err(invalid(format!("多余参数 `{extra}`")));
        }
        Ok(Self { command })
    }

    /// 解析后的命令。
    pub fn command(&self) -> &Command {
        &self.command
    }
}

/// 子命令名 → 命令；缺名字即默认命令 `status`。
fn command(name: Option<&str>, words: &mut Words<'_>) -> Result<Command> {
    let name = name.unwrap_or("status");
    if matches!(name, "help" | "-h" | "--help") {
        return Ok(Command::Help);
    }
    if name == "run" {
        return run(words);
    }
    let Some((_, flag, what, build)) = SINGLE.iter().find(|(known, ..)| *known == name) else {
        return Err(invalid(format!("未知命令 `{name}`")));
    };
    Ok(build(parse_value(words, flag, what)?))
}

/// `run`：三个选项顺序自由，但 `--weights` 必需——猜一个默认文件只会把"写错了"变成"跑出别的结果"。
fn run(words: &mut Words<'_>) -> Result<Command> {
    let mut slots: [Option<String>; 3] = [None, None, None];
    while let Some(flag) = words.peek().copied() {
        let Some(slot) = RUN.iter().position(|(known, _)| *known == flag) else {
            return Err(invalid(format!("`run` 不接受 `{flag}`")));
        };
        slots[slot] = parse_value(words, RUN[slot].0, RUN[slot].1)?;
    }
    let [weights, input, device] = slots;
    Ok(Command::Run {
        weights: weights.ok_or_else(missing_weights)?,
        input,
        device,
    })
}

/// 一个“带取值的选项”：没出现返回 `None`，出现了就必须有合法取值。
fn parse_value(words: &mut Words<'_>, flag: &str, what: &str) -> Result<Option<String>> {
    let Some(given) = words.next() else {
        return Ok(None);
    };
    if given != flag {
        return Err(invalid(format!("期望 `{flag} <{what}>`，得到 `{given}`")));
    }
    value(words.next(), flag).map(Some)
}

/// 选项的取值：不能缺失、不能为空、不能看起来像另一个选项。
fn value(given: Option<&str>, flag: &str) -> Result<String> {
    let Some(value) = given else {
        return Err(invalid(format!("`{flag}` 缺少取值")));
    };
    if value.is_empty() || value.starts_with('-') {
        return Err(invalid(format!("`{flag}` 的取值不合法：`{value}`")));
    }
    Ok(value.to_string())
}

/// `run` 缺权重文件：不猜一个默认文件，那只会把"写错了"变成"跑出别的结果"。
fn missing_weights() -> Error {
    invalid("`run` 需要 `--weights <文件>`")
}

/// 一条顶层命令文法的诊断：上下文固定在这一层。
fn invalid(message: impl Into<String>) -> Error {
    Error::new(Code::Invalid, message.into()).with("cli::Invocation::parse")
}

#[cfg(test)]
mod tests;
