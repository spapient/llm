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
  --config <文件>        读取 TOML 配置（默认 < 文件 < CUTILE_* < 命令行）
  --set <键=值>          覆盖登记的配置键，如 train.learning_rate=0.001
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
const CTX: &str = "cli::argv::parse";

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
    build(&mut Flags::parse(words, grammar, name, CTX)?)
}

/// 一条顶层命令文法的诊断：上下文固定在这一层。
fn invalid(message: impl Into<String>) -> Error {
    Error::new(Code::Invalid, message.into()).with(CTX)
}

/// 一级命令文法的单元测试：文法与类型同文件，进程级端到端在 `tests/cli.rs`。
#[cfg(test)]
mod tests {
    use super::*;
    use diagnostic::Code;

    fn args(list: &[&str]) -> Vec<String> {
        list.iter().map(std::string::ToString::to_string).collect()
    }

    testkit::test_cases! {
    fn the_default_command_is_status() {
        assert_eq!(
            Invocation::parse(&[]).expect("默认").command(),
            &Command::Status { weights: None }
        );
        assert_eq!(
            Invocation::parse(&args(&["status"]))
                .expect("显式")
                .command(),
            &Command::Status { weights: None }
        );
    }

    fn platform_is_a_zero_option_command() {
        assert_eq!(
            Invocation::parse(&args(&["platform"]))
                .expect("零选项")
                .command(),
            &Command::Platform
        );
        let extra = Invocation::parse(&args(&["platform", "--op", "gemv_i4"])).expect_err("不接受参数");
        assert!(extra.to_string().contains("platform"), "{extra}");
    }

    fn a_weights_file_is_carried_through_without_being_opened() {
        assert_eq!(
            Invocation::parse(&args(&["status", "--weights", "model.safetensors"]))
                .expect("合法")
                .command(),
            &Command::Status {
                weights: Some("model.safetensors")
            }
        );
        let missing = Invocation::parse(&args(&["status", "--weights"])).expect_err("缺取值");
        assert!(missing.to_string().contains("缺少取值"), "{missing}");
        let wrong = Invocation::parse(&args(&["status", "--op", "gemv_i4"]))
            .expect_err("status 只接受 --weights");
        assert!(wrong.to_string().contains("--weights"), "{wrong}");
    }

    fn run_carries_its_options_in_any_order_and_refuses_to_guess_the_file() {
        assert_eq!(
            Invocation::parse(&args(&[
                "run",
                "--device",
                "cpu",
                "--weights",
                "model.safetensors",
                "--input",
                "1,2,3",
            ]))
            .expect("合法")
            .command(),
            &Command::Run {
                weights: "model.safetensors",
                input: Some("1,2,3"),
                device: Some("cpu"),
            }
        );
        let missing = Invocation::parse(&args(&["run", "--input", "1,2"])).expect_err("缺 --weights");
        assert!(missing.to_string().contains("--weights"), "{missing}");
        let unknown = Invocation::parse(&args(&["run", "--weights", "m", "--op", "gemv_i4"]))
            .expect_err("未知选项");
        assert!(unknown.to_string().contains("--op"), "{unknown}");
    }

    fn train_carries_its_options_without_parsing_numbers() {
        assert_eq!(
            Invocation::parse(&args(&[
                "train",
                "--lr",
                "0.01",
                "--warmup",
                "10",
                "--steps",
                "100",
                "--clip",
                "1.0",
                "--accumulation",
                "2",
                "--load",
                "ckpt.snap",
            ]))
            .expect("合法")
            .command(),
            &Command::Train {
                lr: Some("0.01"),
                warmup: Some("10"),
                steps: Some("100"),
                clip: Some("1.0"),
                accumulation: Some("2"),
                load: Some("ckpt.snap"),
            }
        );
        // 不带任何选项也是合法命令：数值缺省由组装点的训练配置决定。
        assert_eq!(
            Invocation::parse(&args(&["train"]))
                .expect("无选项")
                .command(),
            &Command::Train {
                lr: None,
                warmup: None,
                steps: None,
                clip: None,
                accumulation: None,
                load: None,
            }
        );
        let unknown = Invocation::parse(&args(&["train", "--op", "gemv_i4"])).expect_err("未知选项");
        assert!(unknown.to_string().contains("--op"), "{unknown}");
        let missing = Invocation::parse(&args(&["train", "--lr"])).expect_err("缺取值");
        assert!(missing.to_string().contains("缺少取值"), "{missing}");
    }

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

    fn archive_commands_carry_paths_through_without_opening_here() {
        assert_eq!(
            Invocation::parse(&args(&[
                "import",
                "--in",
                "model.safetensors",
                "--out",
                "model.u"
            ]))
            .expect("合法")
            .command(),
            &Command::Import {
                input: "model.safetensors",
                output: "model.u",
            }
        );
        assert_eq!(
            Invocation::parse(&args(&["inspect", "--in", "model.u"]))
                .expect("合法")
                .command(),
            &Command::Inspect { input: "model.u" }
        );
        assert_eq!(
            Invocation::parse(&args(&["check", "--in", "model.u"]))
                .expect("合法")
                .command(),
            &Command::Check { input: "model.u" }
        );
        assert_eq!(
            Invocation::parse(&args(&["deploy", "--in", "model.u"]))
                .expect("合法")
                .command(),
            &Command::Deploy { input: "model.u" },
            "E8 裁决与 check 同形：只差一次体积门"
        );

        assert_eq!(
            Invocation::parse(&args(&["settle", "--in", "model.u"]))
                .expect("合法")
                .command(),
            &Command::Settle {
                input: "model.u",
                output: None,
            },
            "不给 --out 就是打到 stdout，不猜默认路径"
        );
        assert_eq!(
            Invocation::parse(&args(&[
                "settle",
                "--out",
                "config.json",
                "--in",
                "model.u"
            ]))
            .expect("顺序自由")
            .command(),
            &Command::Settle {
                input: "model.u",
                output: Some("config.json"),
            }
        );
        assert_eq!(
            Invocation::parse(&args(&[
                "audit",
                "--in",
                "model.u",
                "--ledger",
                "ledger.json"
            ]))
            .expect("合法")
            .command(),
            &Command::Audit {
                input: "model.u",
                ledger: "ledger.json",
            }
        );
        assert_eq!(
            Invocation::parse(&args(&[
                "audit",
                "--ledger",
                "ledger.json",
                "--in",
                "model.u"
            ]))
            .expect("顺序自由")
            .command(),
            &Command::Audit {
                input: "model.u",
                ledger: "ledger.json",
            }
        );

        let no_ledger =
            Invocation::parse(&args(&["audit", "--in", "model.u"])).expect_err("缺 --ledger");
        assert!(no_ledger.to_string().contains("--ledger"), "{no_ledger}");
        let missing_out =
            Invocation::parse(&args(&["import", "--in", "model.safetensors"])).expect_err("缺 --out");
        assert!(missing_out.to_string().contains("--out"), "{missing_out}");
        let missing_in = Invocation::parse(&args(&["check"])).expect_err("缺 --in");
        assert!(missing_in.to_string().contains("--in"), "{missing_in}");
        let unknown = Invocation::parse(&args(&["inspect", "--op", "gemv_i4"])).expect_err("未知选项");
        assert!(unknown.to_string().contains("--op"), "{unknown}");
    }

    fn unknown_words_and_flags_are_rejected_with_the_usage_hint() {
        let unknown = Invocation::parse(&args(&["frobnicate"])).expect_err("未知命令");
        assert_eq!(unknown.code(), Code::Invalid);
        assert!(unknown.to_string().contains("frobnicate"), "{unknown}");

        let flag = Invocation::parse(&args(&["run", "--weights"])).expect_err("缺取值");
        assert!(flag.to_string().contains("缺少取值"), "{flag}");

        let extra = Invocation::parse(&args(&["status", "now"])).expect_err("多余参数");
        assert!(extra.to_string().contains("now"), "{extra}");
    }

    fn the_usage_text_lists_every_command() {
        for command in [
            "status", "platform", "run", "train", "import", "inspect", "check", "settle", "audit",
            "deploy", "help",
        ] {
            assert!(USAGE.contains(command), "{command}");
        }
    }

    /// 编译期自证：`Command` 的每个变体都必须在这里有落点，且这里**不写 `_` 兜底臂**。
    ///
    /// `lib.rs` 的分派为了守住 `Ccn ≤ 5` 分成四层，末端各有一个 `_` 臂；那些臂走不走得到不由
    /// 文档声明，而由这张平表证明：新增变体时这里编译失败，`cargo check --all-targets` 因此点名
    /// 要求接线。名字同时充当"这条命令确实出现在用法里"的清单。
    fn every_command_variant_has_a_destination() {
        fn destination(command: &Command<'_>) -> &'static str {
            match command {
                Command::Status { .. } => "status",
                Command::Run { .. } => "run",
                Command::Train { .. } => "train",
                Command::Import { .. } => "import",
                Command::Inspect { .. } => "inspect",
                Command::Check { .. } => "check",
                Command::Settle { .. } => "settle",
                Command::Audit { .. } => "audit",
                Command::Deploy { .. } => "deploy",
                Command::Platform => "platform",
                Command::Help => "help",
            }
        }
        let all = [
            Command::Status { weights: None },
            Command::Platform,
            Command::Help,
        ];
        for command in &all {
            assert!(USAGE.contains(destination(command)), "{command:?}");
        }
        let mut names: Vec<&str> = all.iter().map(destination).collect();
        names.sort_unstable();
        names.dedup();
        assert_eq!(names.len(), all.len(), "落点名重复：{names:?}");
    }
    }
}
