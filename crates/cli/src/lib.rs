//! 组合根（L7）与命令行文法层：把 `argv` 与环境变量解析成 [`Command`] 与 [`settings::Settings`]，
//! 再由 [`entry`] 分派到 `run` / `status` / `train` 编排入口；两个二进制 `llm` 与 `cutile` 都在本包
//! （入口见 `src/main.rs` 与 `src/cutile_bin.rs`）。
//!
//! **拥有**：`Command`、`Invocation`、`USAGE`、`parse`、内核层审计的子命令文法 [`cutile_command::CutileCommand`]，
//! 三个目录开关的解析策略 [`settings`]（默认值、`CUTILE_*`、`--set`、来源追踪），以及编排分派。
//! **不拥有**：内核业务实现（各层库自己）、输出排版（`tool`）。编排入口只装配、取值解析与人可读排版，不定义新类型；
//! `main` 把 [`entry`] 的 [`Result`] 转成退出码。
//! **依赖**：各内核层与 `diagnostic`；本包即入口，不被更高层库依赖。
//!
//! 配置为什么在这一层：默认目录是**部署策略**（面向发布后的二进制），`--set` 是**命令行文法**；
//! 两者都是"这次调用怎么来的"，而不是"内核怎么实现"。内核只收 `cutile::dirs::Dirs` 这份纯数据，
//! 于是它可以被测试、服务或别的宿主按各自的约定接管（ADR 0023）。
//!
//! 解析与执行分开的理由：本包是组装点，它的 `main` 只该做"解析 → 分派 → 退出码"三件事。
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

mod archive;
mod run;
mod status;
mod train;

pub mod cutile_command;
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
  train [选项]            跑训练闭环（配置数值由组装点解析，本层只搬字符串）
  import --in <源> --out <文件> 把 safetensors / gguf / bin 条目流写成自有 `.u` 容器
  inspect --in <文件>     读 `.u` 容器：条目清单与现场现算的度量
  check --in <文件>       逐段解码还原 `.u` 容器，校验完整性
  help                   打印本说明

选项：
  --op <算子>             算子名（`verify` / `bench` 用，缺省为 `gemv_i4`）
  --weights <文件>        safetensors 权重文件（`status` / `run` 用；`run` 必需）
  --in <文件>             输入路径（`import` 的 safetensors 源；`inspect` / `check` 的 `.u` 容器；三者都必需）
  --out <文件>            输出路径（`import` 的 `.u` 容器，必需）
  --input <a,b,c>         `run` 的输入向量（逗号分隔 f32；不给则用确定性图案）
  --device <设备>         `run` 的设备名（缺省 `auto`，按指纹选）
  --lr <数值>             训练峰值学习率（`train` 用；缺省由调度表定）
  --warmup <步数>         学习率预热步数（`train` 用）
  --steps <步数>         总训练步数（`train` 用）
  --clip <范数>          全局梯度裁剪范数（`train` 用）
  --accumulation <步数>  梯度累积步数（`train` 用；缺省 1）
  --load <检查点>         从快照恢复参数、矩状态与步数（`train` 用）
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
    /// 跑训练闭环。配置只搬字符串：f32 / 整数的解析在组装点，文法层不猜默认值。
    Train {
        /// 峰值学习率；`None` 取调度表默认。
        lr: Option<String>,
        /// 学习率预热步数；`None` 取调度表默认。
        warmup: Option<String>,
        /// 总训练步数；`None` 取配置默认。
        steps: Option<String>,
        /// 全局梯度裁剪范数；`None` 表示不裁剪。
        clip: Option<String>,
        /// 梯度累积步数；`None` 表示每步更新一次。
        accumulation: Option<String>,
        /// 恢复用快照路径；`None` 表示从零开始。
        load: Option<String>,
    },
    /// 从 safetensors 源导入自有 `.u` 容器；两个路径都必需。
    Import {
        /// 源权重文件路径（按扩展名分派适配通路）。
        input: String,
        /// 输出 `.u` 容器路径。
        output: String,
    },
    /// 检查 `.u` 容器条目清单与现场度量；路径必需。
    Inspect {
        /// 待读 `.u` 容器路径。
        input: String,
    },
    /// 逐段解码校验 `.u` 容器；路径必需。
    Check {
        /// 待校验 `.u` 容器路径。
        input: String,
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

/// `train` 的选项与取值说明：全部可选，顺序自由；数值解析留给组装点。
const TRAIN: &[(&str, &str)] = &[
    ("--lr", "数值"),
    ("--warmup", "步数"),
    ("--steps", "步数"),
    ("--clip", "范数"),
    ("--accumulation", "步数"),
    ("--load", "检查点"),
];

/// `inspect` / `check` 的选项：只有 `--in`，必需。
const INSPECT: &[(&str, &str)] = &[("--in", "文件")];

/// `import` 的选项：`--in`（safetensors 源）与 `--out`（`.u` 容器），两者都必需。
const IMPORT: &[(&str, &str)] = &[("--in", "源"), ("--out", "文件")];

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
    match name {
        "help" | "-h" | "--help" => Ok(Command::Help),
        "run" => run(words),
        "train" => train(words),
        other => archive_command(other, words),
    }
}

/// 归档容器子命令（`import` / `inspect` / `check`）与单选项命令表的分派。
fn archive_command(name: &str, words: &mut Words<'_>) -> Result<Command> {
    match name {
        "import" => import(words),
        "inspect" => single_required(words, INSPECT, "inspect", |input| Command::Inspect {
            input,
        }),
        "check" => single_required(words, INSPECT, "check", |input| Command::Check { input }),
        known => single(known, words),
    }
}

/// 单选项必需命令：表内唯一槽位，缺省即 [`missing_required`]；`--in` 不允许悬空。
fn single_required(
    words: &mut Words<'_>,
    table: &[(&str, &str)],
    command: &str,
    build: fn(String) -> Command,
) -> Result<Command> {
    let slots = parse_slots(words, table, command)?;
    let input = slots[0]
        .clone()
        .ok_or_else(|| missing_required(table[0].0, command))?;
    Ok(build(input))
}

/// `import`：`--in` / `--out` 顺序自由，两者都必需——缺源或缺输出都是写错了，不猜默认路径。
fn import(words: &mut Words<'_>) -> Result<Command> {
    let slots = parse_slots(words, IMPORT, "import")?;
    let input = required_slot(&slots, 0, IMPORT[0].0, "import")?;
    let output = required_slot(&slots, 1, IMPORT[1].0, "import")?;
    Ok(Command::Import { input, output })
}

/// 取一个必需槽位；缺省点名对应开关，不替调用方补默认值。
fn required_slot(
    slots: &[Option<String>],
    index: usize,
    flag: &str,
    command: &str,
) -> Result<String> {
    slots[index]
        .clone()
        .ok_or_else(|| missing_required(flag, command))
}

/// 单参数命令表命中；未知名命令报 [`Code::Invalid`]。
fn single(known: &str, words: &mut Words<'_>) -> Result<Command> {
    let Some((_, flag, what, build)) = SINGLE.iter().find(|(candidate, ..)| *candidate == known)
    else {
        return Err(invalid(format!("未知命令 `{known}`")));
    };
    Ok(build(parse_value(words, flag, what)?))
}

/// 按文法表把逐个出现的开关填进对应槽位；未知开关交 [`invalid`]，缺取值交 [`parse_value`]。
fn parse_slots(
    words: &mut Words<'_>,
    table: &[(&str, &str)],
    command: &str,
) -> Result<Vec<Option<String>>> {
    let mut slots: Vec<Option<String>> = table.iter().map(|_| None).collect();
    while let Some(flag) = words.peek().copied() {
        let Some(slot) = table.iter().position(|(known, _)| *known == flag) else {
            return Err(invalid(format!("`{command}` 不接受 `{flag}`")));
        };
        slots[slot] = parse_value(words, table[slot].0, table[slot].1)?;
    }
    Ok(slots)
}

/// `run`：三个选项顺序自由，但 `--weights` 必需——猜一个默认文件只会把"写错了"变成"跑出别的结果"。
fn run(words: &mut Words<'_>) -> Result<Command> {
    let slots = parse_slots(words, RUN, "run")?;
    Ok(Command::Run {
        weights: slots[0].clone().ok_or_else(missing_weights)?,
        input: slots[1].clone(),
        device: slots[2].clone(),
    })
}

/// `train`：六个选项顺序自由，全部可缺省；文法层只搬字符串，不解析 f32 / 步数。
fn train(words: &mut Words<'_>) -> Result<Command> {
    let slots = parse_slots(words, TRAIN, "train")?;
    Ok(Command::Train {
        lr: slots[0].clone(),
        warmup: slots[1].clone(),
        steps: slots[2].clone(),
        clip: slots[3].clone(),
        accumulation: slots[4].clone(),
        load: slots[5].clone(),
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
/// 命令行两处解析（`cli::Invocation` 与 `cli::cutile_command`）共用这一份口径。
pub(crate) fn value(given: Option<&str>, flag: &str) -> Result<String> {
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

/// 命令缺必需选项：缺哪个选项就点名哪个，不替调用方补默认值。
fn missing_required(flag: &str, command: &str) -> Error {
    invalid(format!("`{command}` 需要 `{flag} <文件>`"))
}

/// 一条顶层命令文法的诊断：上下文固定在这一层。
fn invalid(message: impl Into<String>) -> Error {
    Error::new(Code::Invalid, message.into()).with("cli::Invocation::parse")
}

/// 命令分派：解析后的命令接到对应编排；输出由各编排函数落到 stdout，
/// 错误交给二进制入口汇总成退出码。
pub fn entry(invocation: &Invocation) -> Result<()> {
    match invocation.command() {
        Command::Help => {
            print!("{USAGE}");
            Ok(())
        }
        Command::Status { weights } => status::status(weights.as_deref()),
        Command::Run {
            weights,
            input,
            device,
        } => run::run(weights, input.as_deref(), device.as_deref()),
        Command::Verify { .. } | Command::Bench { .. } => audit_note(),
        other => entry_archive_or_train(other),
    }
}

/// 归档容器命令与训练闭环的分派；其余命令已在 [`entry`] 主匹配里收口。
fn entry_archive_or_train(command: &Command) -> Result<()> {
    match command {
        Command::Import { input, output } => archive::import(input, output),
        Command::Inspect { input } => archive::inspect(input),
        Command::Check { input } => archive::check(input),
        Command::Train {
            lr,
            warmup,
            steps,
            clip,
            accumulation,
            load,
        } => train::train_run(
            lr.as_deref(),
            warmup.as_deref(),
            steps.as_deref(),
            clip.as_deref(),
            accumulation.as_deref(),
            load.as_deref(),
        ),
        other => Err(invalid(format!("命令 `{other:?}` 未接入编排"))),
    }
}

/// 组装点不复制审计实现：这两个命令由 `cutile` 二进制提供。
fn audit_note() -> Result<()> {
    Err(Error::new(
        Code::Invalid,
        "该命令当前由 `cutile` 二进制提供（`llm` 只负责组装）",
    )
    .with("cli::entry"))
}
