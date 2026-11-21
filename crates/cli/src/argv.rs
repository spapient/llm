//! 顶层命令文法：`llm` 的一级子命令、选项表与解析。

use crate::flag::{Flags, Grammar, Words, required, value, words};
use diagnostic::{Code, Error, Result};

/// 用法文本：唯一一份，`help` 与错误信息都引用它。
pub const USAGE: &str = "\
用法：llm [命令] [选项]

命令：
  status                 打印各层状态（默认）
  platform               打印六平台执行合同的六行裁决（设备 + 编译器两件证据）
  run    --weights <文件> 用真实权重算一次前向（每条链一层内核）
  train [选项]            跑训练闭环（配置数值由组装点解析，本层只搬字符串）
  import --in <源> --out <文件> 把 safetensors / gguf / bin 条目流写成自有 `.u` 容器
  inspect --in <文件>     读 `.u` 容器：条目清单与现场现算的度量
  check --in <文件>       逐段解码还原 `.u` 容器，校验完整性
  deploy --in <文件>      E8 裁决：逐段解码 + 参数级体积门（24× 无损），不达门退出码非零
  settle --in <文件> [--out <文件>]  逐张量沉降账本（体积 / 原因 / 无损）；`--out` 落成 JSON
  audit  --in <文件> --ledger <账本> 把落盘账本读回，与同一份字节现算的账逐格核对
  help                   打印本说明

内核审计（逐算子协商 / 对拍 / 带宽）由本包的 `cutile` 入口提供：
  cutile verify --op <算子>   cutile bench --op <算子>   cutile doctor

选项：
  --weights <文件>        safetensors 权重文件（`status` / `run` 用；`run` 必需）
  --in <文件>             输入路径（`import` 的 safetensors 源；`inspect` / `check` / `deploy` / `settle` / `audit` 的 `.u` 容器；六者都必需）
  --out <文件>            输出路径（`import` 的 `.u` 容器，必需）；`settle` 的账本文件（可选）
  --ledger <文件>         `audit` 待复核的账本（必需）：`settle --out` 写下的那一份
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

/// 顶层命令；只描述"要做什么"，不描述"怎么做"。取值一律借用 `argv`，本层不分配。
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Command<'a> {
    /// 打印各层状态。
    Status {
        /// safetensors 权重文件路径；`None` 表示只用声明层。
        weights: Option<&'a str>,
    },
    /// 用真实权重算一次。
    Run {
        /// safetensors 权重文件路径（必需）。
        weights: &'a str,
        /// 输入向量（逗号分隔的 f32）；`None` 表示用确定性图案。
        input: Option<&'a str>,
        /// 设备名；`None` 表示 `auto`（按指纹选）。
        device: Option<&'a str>,
    },
    /// 跑训练闭环。配置只搬字符串：f32 / 整数的解析在组装点，文法层不猜默认值。
    Train {
        /// 峰值学习率；`None` 取调度表默认。
        lr: Option<&'a str>,
        /// 学习率预热步数；`None` 取调度表默认。
        warmup: Option<&'a str>,
        /// 总训练步数；`None` 取配置默认。
        steps: Option<&'a str>,
        /// 全局梯度裁剪范数；`None` 表示不裁剪。
        clip: Option<&'a str>,
        /// 梯度累积步数；`None` 表示每步更新一次。
        accumulation: Option<&'a str>,
        /// 恢复用快照路径；`None` 表示从零开始。
        load: Option<&'a str>,
    },
    /// 从 safetensors 源导入自有 `.u` 容器；两个路径都必需。
    Import {
        /// 源权重文件路径（按扩展名分派适配通路）。
        input: &'a str,
        /// 输出 `.u` 容器路径。
        output: &'a str,
    },
    /// 检查 `.u` 容器条目清单与现场度量；路径必需。
    Inspect {
        /// 待读 `.u` 容器路径。
        input: &'a str,
    },
    /// 逐段解码校验 `.u` 容器；路径必需。
    Check {
        /// 待校验 `.u` 容器路径。
        input: &'a str,
    },
    /// E8 部署裁决（LLD §E8）：逐段解码 + 参数级体积门，两件证据同时成立才算通过。
    Deploy {
        /// 待裁决 `.u` 容器路径。
        input: &'a str,
    },
    /// 逐张量沉降账本（LLD §4.7）：体积、原因与无损判定，全部由落盘字节现算。
    Settle {
        /// 待记账 `.u` 容器路径。
        input: &'a str,
        /// 账本 JSON 的输出路径；`None` 表示打到 stdout（不猜产物目录旁的默认文件名）。
        output: Option<&'a str>,
    },
    /// 账本复核：落盘账本与同一份字节现算的账逐格比对，不符即错。
    Audit {
        /// 待复核 `.u` 容器路径。
        input: &'a str,
        /// 待复核的账本路径（`settle --out` 写下的那一份）。
        ledger: &'a str,
    },
    /// 打印六平台执行合同的六行裁决；零选项。
    Platform,
    /// 打印用法。
    Help,
}

/// 命令表的一行：名字、选项表、组装。
type Build = for<'a> fn(&mut Flags<'a>) -> Result<Command<'a>>;

type Row = (&'static str, Grammar, Build);

/// 顶层子命令表。必需哪几个选项由 [`required`] 标明，组装函数因此直接取值。
const COMMANDS: &[Row] = &[
    ("status", &[value("--weights", "文件")], |flags| {
        Ok(Command::Status {
            weights: flags.take_text("--weights"),
        })
    }),
    (
        "run",
        &[
            required("--weights", "文件"),
            value("--input", "数值列表"),
            value("--device", "设备"),
        ],
        |flags| {
            Ok(Command::Run {
                weights: flags.take_need("--weights")?,
                input: flags.take_text("--input"),
                device: flags.take_text("--device"),
            })
        },
    ),
    (
        "train",
        &[
            value("--lr", "数值"),
            value("--warmup", "步数"),
            value("--steps", "步数"),
            value("--clip", "范数"),
            value("--accumulation", "步数"),
            value("--load", "检查点"),
        ],
        |flags| {
            Ok(Command::Train {
                lr: flags.take_text("--lr"),
                warmup: flags.take_text("--warmup"),
                steps: flags.take_text("--steps"),
                clip: flags.take_text("--clip"),
                accumulation: flags.take_text("--accumulation"),
                load: flags.take_text("--load"),
            })
        },
    ),
    (
        "import",
        &[required("--in", "源"), required("--out", "文件")],
        |flags| {
            Ok(Command::Import {
                input: flags.take_need("--in")?,
                output: flags.take_need("--out")?,
            })
        },
    ),
    ("inspect", &[required("--in", "文件")], |flags| {
        Ok(Command::Inspect {
            input: flags.take_need("--in")?,
        })
    }),
    ("check", &[required("--in", "文件")], |flags| {
        Ok(Command::Check {
            input: flags.take_need("--in")?,
        })
    }),
    ("deploy", &[required("--in", "文件")], |flags| {
        Ok(Command::Deploy {
            input: flags.take_need("--in")?,
        })
    }),
    (
        "settle",
        &[required("--in", "文件"), value("--out", "文件")],
        |flags| {
            Ok(Command::Settle {
                input: flags.take_need("--in")?,
                output: flags.take_text("--out"),
            })
        },
    ),
    ("platform", &[], |_| Ok(Command::Platform)),
    (
        "audit",
        &[required("--in", "文件"), required("--ledger", "账本")],
        |flags| {
            Ok(Command::Audit {
                input: flags.take_need("--in")?,
                ledger: flags.take_need("--ledger")?,
            })
        },
    ),
];

/// 诊断帧：本层是文法层。
const CONTEXT: &str = "cli::argv::parse";

/// 一次调用：解析后的命令。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Invocation<'a> {
    command: Command<'a>,
}

impl<'a> Invocation<'a> {
    /// 解析参数（不含 `argv[0]`）。
    ///
    /// # Errors
    ///
    /// 未知命令、未知选项、缺少取值、缺少必需选项、多余参数都返回 [`Code::Invalid`]，
    /// 消息里点名出错的词——错误信息本身就是下一条命令的提示。
    pub fn parse(args: &[String]) -> Result<Invocation<'_>> {
        let mut words = words(args);
        let command = command(words.next(), &mut words)?;
        if let Some(extra) = words.next() {
            return Err(invalid(format!("多余参数 `{extra}`")));
        }
        Ok(Invocation { command })
    }

    /// 解析后的命令。
    pub fn command(&self) -> &Command<'a> {
        &self.command
    }
}

/// 子命令名 → 命令；缺名字即默认命令 `status`。
fn command<'a>(head: Option<&'a str>, words: &mut Words<'a>) -> Result<Command<'a>> {
    let name = head.unwrap_or("status");
    if matches!(name, "help" | "-h" | "--help") {
        return Ok(Command::Help);
    }
    table(name, words)
}

/// 表内子命令：按名找行、收选项、组装；表外的名字不猜。
fn table<'a>(name: &'a str, words: &mut Words<'a>) -> Result<Command<'a>> {
    let Some((_, grammar, build)) = COMMANDS.iter().find(|(known, ..)| *known == name) else {
        let known: Vec<&str> = COMMANDS.iter().map(|(known, ..)| *known).collect();
        return Err(invalid(format!(
            "未知命令 `{name}`：已知命令 {}",
            known.join("、")
        )));
    };
    build(&mut Flags::parse(words, grammar, name, CONTEXT)?)
}

/// 一条顶层命令文法的诊断：上下文固定在这一层。
fn invalid(message: impl Into<String>) -> Error {
    Error::new(Code::Invalid, message.into()).with(CONTEXT)
}

/// 一级命令文法的单元测试：文法与类型同文件，进程级端到端在 `tests/cli.rs`。
#[cfg(test)]
#[path = "../tests/grammar.rs"]
mod tests;
