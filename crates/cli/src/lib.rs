//! 组合根（L7）与命令行文法层：把 `argv` 与环境变量解析成 [`Command`] 与 [`settings::Settings`]，
//! 再由 [`entry`] 分派到 `run` / `status` / `train` 编排入口；两个二进制 `llm` 与 `cutile` 都在本包
//! （入口见 `src/main.rs` 与 `src/bin/cutile.rs`）。
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

pub(crate) use self::train::archive;
mod platform;
mod run;
mod status;
mod train;

pub mod cutile_command {
    //! `cutile` 子命令：内核层审计（doctor / verify / bench / cache）的**参数文法**。
    //!
    //! 语法只有一份：内核层的驱动方（`cargo run --bin cutile`）与 `llm cutile ...` 都走这里，
    //! 执行在 `cutile::{doctor, verify, bench, gc}`，排版在驱动方。`--set key=value` 属于配置层（`crate::settings`），
    //! 由驱动方在交给本模块之前摘走。

    use diagnostic::{Code, Error, Result};

    /// 用法文本：唯一一份，`cutile help` 与错误信息都引用它。
    pub const USAGE: &str = "\
用法：cutile <子命令> [选项]

子命令：
  doctor                  打印协商链、布局声明、emit 结果与缓存指纹
  verify                  逐设备验证：协商 + 拼接 + 编译 + 与标量参考对拍
  bench                   跑一次并报告带宽达成率
  ledger                  跨档案能力台账：每条供给这次执行到的结论
  cache gc                按 LRU 把缓存回收到容量上限以内

选项：
  --op <算子>             算子名（缺省 `gemv_i4`）
  --device <设备>         doctor 用；`auto` 表示按指纹选
  --why <缓存键>          doctor 用；解释该键与本次构建的键差在哪
  --devices <列表>        verify 用；逗号分隔（缺省 `cpu`）
  --emit-ledger           把能力台账写到 `<work_dir>/ledger/`
  --out <文件>            ledger 用；给出后把 markdown 台账写到该文件
  --shape <MxN>           bench 用（缺省：算子的生产形状）
  --device <设备>          bench 用（缺省 auto：按本机指纹挑；也可写 cpu / nvidia / musa …）
  --repeat <次数>         bench 用（缺省 50）
  --limit <字节>          cache gc 用（缺省 2 GiB）
  -h, --help              打印本说明
";

    /// 缺省算子名。
    pub const DEFAULT_OP: &str = "gemv_i4";

    /// 内核层审计命令。
    #[derive(Debug, Clone, PartialEq, Eq)]
    pub enum CutileCommand {
        /// 打印整条流水线的每一步。
        Doctor {
            /// 算子名。
            op: String,
            /// 设备名（`auto` 表示按指纹选）。
            device: String,
            /// 要解释的缓存键（十六进制）；`None` 表示只打印本次构建的键。
            why: Option<String>,
        },
        /// 逐设备验证。
        Verify {
            /// 算子名。
            op: String,
            /// 设备列表。
            devices: Vec<String>,
            /// 是否写能力台账。
            ledger: bool,
        },
        /// 跑一次并报告达成率。
        Bench {
            /// 算子名。
            op: String,
            /// 形状 `MxN`；`None` 表示用算子的生产形状。
            shape: Option<String>,
            /// 设备名或缺省（`auto` 按本机指纹挑）；与 `doctor --device` 同一套取值。
            device: Option<String>,
            /// 重复次数。
            repeat: usize,
        },
        /// 跨档案能力台账。
        Ledger {
            /// 台账 markdown 落盘路径；`None` 只打印到 stdout。
            out: Option<String>,
        },
        /// 缓存维护：把缓存回收到容量上限以内。
        Cache {
            /// 容量上限（字节）；`None` 表示用缺省上限。
            limit: Option<u64>,
        },
    }

    /// 选项形态：带一个取值，还是不带取值的开关。
    #[derive(Clone, Copy)]
    enum Kind {
        Value,
        Switch,
    }

    use Kind::{Switch, Value};

    /// 一个子命令的选项文法：选项名与它要不要取值。
    type Grammar = &'static [(&'static str, Kind)];

    /// 一个子命令的组装：把解析后的选项变成命令（取值非法时返回诊断）。
    type Build = fn(&Flags) -> Result<CutileCommand>;

    /// `doctor` 的选项文法。
    const DOCTOR: Grammar = &[("--op", Value), ("--device", Value), ("--why", Value)];

    /// `verify` 的选项文法：`--emit-ledger` 是不带取值的开关。
    const VERIFY: Grammar = &[
        ("--op", Value),
        ("--devices", Value),
        ("--emit-ledger", Switch),
    ];

    /// `bench` 的选项文法：`--device` 与 `doctor` 同义（缺省 `auto`，按本机指纹挑）。
    const BENCH: Grammar = &[
        ("--op", Value),
        ("--shape", Value),
        ("--device", Value),
        ("--repeat", Value),
    ];

    /// `ledger` 的选项文法。
    const LEDGER: Grammar = &[("--out", Value)];

    /// `cache gc` 的选项文法：上限是字节数，与 `--repeat` 的次数不同。
    const CACHE: Grammar = &[("--limit", Value)];

    /// 子命令表：名字、选项文法、命令组装。表即文法，新增子命令只加一行。
    ///
    /// `cache` 还要先吃一个动作词（`gc`），它不在表里，由 [`dispatch`] 单独分派。
    const SUBCOMMANDS: &[(&str, Grammar, Build)] = &[
        ("doctor", DOCTOR, doctor),
        ("verify", VERIFY, verify),
        ("bench", BENCH, bench),
        ("ledger", LEDGER, ledger),
    ];

    impl CutileCommand {
        /// 解析参数（不含 `argv[0]`）。
        ///
        /// # Errors
        ///
        /// 缺子命令、未知子命令/选项、缺取值或取值非法都返回 [`Code::Invalid`] 并附 [`USAGE`]。
        pub fn parse(args: &[String]) -> Result<Self> {
            let Some((head, rest)) = args.split_first() else {
                return Err(invalid("缺子命令"));
            };
            let Some(out) = dispatch(head, rest) else {
                return Err(invalid(format!("未知子命令 `{head}`")));
            };
            out
        }
    }

    /// 子命令名 → 解析结果；`None` 表示这个名字不在文法里。
    fn dispatch(head: &str, rest: &[String]) -> Option<Result<CutileCommand>> {
        if matches!(head, "help" | "-h" | "--help") {
            return Some(Err(
                Error::new(Code::Invalid, USAGE).with("cli::cutile_command::parse")
            ));
        }
        if head == "cache" {
            return Some(cache(rest));
        }
        let (_, grammar, build) = SUBCOMMANDS.iter().find(|(name, ..)| *name == head)?;
        Some(Flags::parse(rest, grammar).and_then(|flags| build(&flags)))
    }

    /// `doctor`：打印整条流水线。
    fn doctor(flags: &Flags) -> Result<CutileCommand> {
        Ok(CutileCommand::Doctor {
            op: flags.text("--op").unwrap_or_else(|| DEFAULT_OP.to_string()),
            device: flags.text("--device").unwrap_or_else(|| "auto".to_string()),
            why: flags.text("--why"),
        })
    }

    /// `verify`：逐设备验证。
    fn verify(flags: &Flags) -> Result<CutileCommand> {
        Ok(CutileCommand::Verify {
            op: flags.text("--op").unwrap_or_else(|| DEFAULT_OP.to_string()),
            devices: devices(flags.text("--devices")),
            ledger: flags.switch("--emit-ledger"),
        })
    }

    /// `bench`：跑一次并报告达成率。
    fn bench(flags: &Flags) -> Result<CutileCommand> {
        Ok(CutileCommand::Bench {
            op: flags.text("--op").unwrap_or_else(|| DEFAULT_OP.to_string()),
            shape: flags.text("--shape"),
            device: flags.text("--device"),
            repeat: flags
                .number("--repeat", "`--repeat` 需要正整数")?
                .unwrap_or(50),
        })
    }

    /// `ledger`：跨档案能力台账。
    fn ledger(flags: &Flags) -> Result<CutileCommand> {
        Ok(CutileCommand::Ledger {
            out: flags.text("--out"),
        })
    }

    /// `cache`：唯一动作是 `gc`；缺动作与写错动作分开报错，不静默当成 `gc`。
    fn cache(rest: &[String]) -> Result<CutileCommand> {
        let Some(action) = rest.first().filter(|word| !word.starts_with('-')) else {
            return Err(invalid("缺少子命令"));
        };
        if action != "gc" {
            return Err(invalid(format!("未知子命令 `cache {action}`")));
        }
        Ok(CutileCommand::Cache {
            limit: Flags::parse(&rest[1..], CACHE)?
                .number("--limit", "`--limit` 需要字节数（十进制）")?,
        })
    }

    /// 设备列表：逗号分隔、去空白、丢弃空项；一个都没有时退回 `cpu`（与缺省一致）。
    fn devices(text: Option<String>) -> Vec<String> {
        let mut names: Vec<String> = text
            .unwrap_or_default()
            .split(',')
            .map(str::trim)
            .filter(|name| !name.is_empty())
            .map(str::to_string)
            .collect();
        if names.is_empty() {
            names.push("cpu".to_string());
        }
        names
    }

    /// 解析后的选项：名字与取值（开关没有取值），同名重复以最后一次为准。
    struct Flags(Vec<(String, Option<String>)>);

    impl Flags {
        /// 按文法收选项：名字必须在 `grammar` 里，带取值的必须后面跟着合法取值。
        fn parse(rest: &[String], grammar: Grammar) -> Result<Self> {
            let mut out = Vec::new();
            let mut words = rest.iter();
            while let Some(word) = words.next() {
                let value = match kind_of(grammar, word)? {
                    Kind::Value => Some(crate::value(words.next().map(String::as_str), word)?),
                    Kind::Switch => None,
                };
                out.push((word.clone(), value));
            }
            Ok(Self(out))
        }

        /// 某个选项的取值；没出现就是 `None`。
        fn text(&self, flag: &str) -> Option<String> {
            self.0.iter().rfind(|(name, _)| name == flag)?.1.clone()
        }

        /// 开关选项是否出现。
        fn switch(&self, flag: &str) -> bool {
            self.0.iter().any(|(name, _)| name == flag)
        }

        /// 数值选项：十进制解析失败按 `message` 报错（单位由选项自己给）。
        fn number<T: std::str::FromStr>(&self, flag: &str, message: &str) -> Result<Option<T>> {
            match self.text(flag) {
                Some(text) => text.parse().map(Some).map_err(|_| invalid(message)),
                None => Ok(None),
            }
        }
    }

    /// 选项形态：`-` 开头且在文法里才算选项，其余按「多余参数」报错。
    fn kind_of(grammar: Grammar, word: &str) -> Result<Kind> {
        if !word.starts_with('-') {
            return Err(invalid(format!("多余参数 `{word}`")));
        }
        grammar
            .iter()
            .find(|(name, _)| *name == word)
            .map(|(_, kind)| *kind)
            .ok_or_else(|| invalid(format!("未知参数 `{word}`")))
    }
    fn invalid(message: impl Into<String>) -> Error {
        Error::new(Code::Invalid, format!("{}\n{USAGE}", message.into()))
            .with("cli::cutile_command::parse")
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        fn args(list: &[&str]) -> Vec<String> {
            list.iter().map(|item| item.to_string()).collect()
        }

        #[test]
        fn parses_each_subcommand_with_its_defaults() {
            assert_eq!(
                CutileCommand::parse(&args(&["doctor"])).unwrap(),
                CutileCommand::Doctor {
                    op: DEFAULT_OP.to_string(),
                    device: "auto".to_string(),
                    why: None
                }
            );
            assert_eq!(
                CutileCommand::parse(&args(&[
                    "doctor",
                    "--device",
                    "cpu",
                    "--why",
                    "83a3ecdb31cd1225"
                ]))
                .unwrap(),
                CutileCommand::Doctor {
                    op: DEFAULT_OP.to_string(),
                    device: "cpu".to_string(),
                    why: Some("83a3ecdb31cd1225".to_string())
                }
            );
            assert_eq!(
                CutileCommand::parse(&args(&["cache", "gc"])).unwrap(),
                CutileCommand::Cache { limit: None }
            );
            assert_eq!(
                CutileCommand::parse(&args(&["cache", "gc", "--limit", "1048576"])).unwrap(),
                CutileCommand::Cache {
                    limit: Some(1_048_576)
                }
            );
            assert_eq!(
                CutileCommand::parse(&args(&["verify", "--devices", "cpu,cuda", "--emit-ledger"]))
                    .unwrap(),
                CutileCommand::Verify {
                    op: DEFAULT_OP.to_string(),
                    devices: vec!["cpu".to_string(), "cuda".to_string()],
                    ledger: true
                }
            );
            assert_eq!(
                CutileCommand::parse(&args(&["bench", "--shape", "1024x64", "--repeat", "3"]))
                    .unwrap(),
                CutileCommand::Bench {
                    op: DEFAULT_OP.to_string(),
                    shape: Some("1024x64".to_string()),
                    device: None,
                    repeat: 3
                }
            );
        }

        #[test]
        fn bench_without_shape_leaves_it_to_the_operator() {
            // 缺省形状是算子的生产形状：文法里不再有一份方阵。
            assert_eq!(
                CutileCommand::parse(&args(&["bench"])).unwrap(),
                CutileCommand::Bench {
                    op: DEFAULT_OP.to_string(),
                    shape: None,
                    device: None,
                    repeat: 50
                }
            );
        }

        #[test]
        fn unknown_arguments_are_rejected() {
            let error = CutileCommand::parse(&args(&["doctor", "--nope"])).unwrap_err();
            assert!(error.to_string().contains("未知参数"), "{error}");
            let error = CutileCommand::parse(&args(&["nope"])).unwrap_err();
            assert!(error.to_string().contains("未知子命令"), "{error}");
            assert!(CutileCommand::parse(&[]).is_err(), "缺少子命令不猜");
        }

        #[test]
        fn options_belong_to_their_subcommand() {
            // `--shape` 只对 bench 有意义：写错子命令时报错，而不是静默忽略。
            let error = CutileCommand::parse(&args(&["verify", "--shape", "1024x64"])).unwrap_err();
            assert!(error.to_string().contains("未知参数"), "{error}");
        }

        #[test]
        fn a_flag_without_a_value_is_rejected() {
            let error = CutileCommand::parse(&args(&["doctor", "--device"])).unwrap_err();
            assert!(error.to_string().contains("缺少取值"), "{error}");
            let error = CutileCommand::parse(&args(&["bench", "--repeat", "many"])).unwrap_err();
            assert!(error.to_string().contains("正整数"), "{error}");
            let error =
                CutileCommand::parse(&args(&["cache", "gc", "--limit", "2GiB"])).unwrap_err();
            assert!(error.to_string().contains("字节数"), "{error}");
        }

        #[test]
        fn cache_needs_a_maintenance_action() {
            // `cache` 自己不是动作：缺子命令与未实现的子命令都报错，不静默当成 gc。
            let error = CutileCommand::parse(&args(&["cache"])).unwrap_err();
            assert!(error.to_string().contains("缺少子命令"), "{error}");
            let error = CutileCommand::parse(&args(&["cache", "rm"])).unwrap_err();
            assert!(error.to_string().contains("未知子命令"), "{error}");
        }
    }
}
pub mod settings;

use diagnostic::{Code, Error, Result};

/// 用法文本：唯一一份，`help` 与错误信息都引用它。
pub const USAGE: &str = "\
用法：llm [命令] [选项]

命令：
  status                 打印各层状态（默认）
  platform               打印六平台执行合同的六行裁决（设备 + 编译器两件证据）
  run    --weights <文件> 用真实权重算一次前向（每条链一层内核）
  verify --op <算子>      校验一个算子的协商与对拍
  bench  --op <算子>      测量一个算子的带宽与达成率
  train [选项]            跑训练闭环（配置数值由组装点解析，本层只搬字符串）
  import --in <源> --out <文件> 把 safetensors / gguf / bin 条目流写成自有 `.u` 容器
  inspect --in <文件>     读 `.u` 容器：条目清单与现场现算的度量
  check --in <文件>       逐段解码还原 `.u` 容器，校验完整性
  settle --in <文件> [--out <文件>]  逐张量沉降账本（体积 / 原因 / 无损）；`--out` 落成 JSON
  audit  --in <文件> --ledger <账本> 把落盘账本读回，与同一份字节现算的账逐格核对
  help                   打印本说明

选项：
  --op <算子>             算子名（`verify` / `bench` 用，缺省为 `gemv_i4`）
  --weights <文件>        safetensors 权重文件（`status` / `run` 用；`run` 必需）
  --in <文件>             输入路径（`import` 的 safetensors 源；`inspect` / `check` / `settle` / `audit` 的 `.u` 容器；五者都必需）
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
    /// 逐张量沉降账本（LLD §4.7）：体积、原因与无损判定，全部由落盘字节现算。
    Settle {
        /// 待记账 `.u` 容器路径。
        input: String,
        /// 账本 JSON 的输出路径；`None` 表示打到 stdout（不猜产物目录旁的默认文件名）。
        output: Option<String>,
    },
    /// 账本复核：落盘账本与同一份字节现算的账逐格比对，不符即错。
    Audit {
        /// 待复核 `.u` 容器路径。
        input: String,
        /// 待复核的账本路径（`settle --out` 写下的那一份）。
        ledger: String,
    },
    /// 打印六平台执行合同的六行裁决；零选项。
    Platform,
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
///
/// 表即文法：这两条都只说一个文件，与 `SINGLE` 的差别只在“必填”——构造器因此吃 `String`
/// 而不是 `Option<String>`，类型上就区分了“可以有默认”与“必须点名文件”。
type PathRow = (
    &'static str,
    &'static str,
    &'static str,
    fn(String) -> Command,
);

const PATHS: &[PathRow] = &[
    ("inspect", "--in", "文件", |input| Command::Inspect {
        input,
    }),
    ("check", "--in", "文件", |input| Command::Check { input }),
];

/// `import` 的选项：`--in`（safetensors 源）与 `--out`（`.u` 容器），两者都必需。
const IMPORT: &[(&str, &str)] = &[("--in", "源"), ("--out", "文件")];

/// `settle` 的选项：`--in` 必需，`--out` 可选——不给就把账本打到 stdout。
const SETTLE: &[(&str, &str)] = &[("--in", "文件"), ("--out", "文件")];

/// `audit` 的选项：`--in`（容器）与 `--ledger`（账本）都必需——缺要么是漏了账本，要么是漏了被
/// 复核的那份字节，两种都算不出来。
const AUDIT: &[(&str, &str)] = &[("--in", "文件"), ("--ledger", "账本")];

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
        "platform" => platform(words),
        "run" => run(words),
        "train" => train(words),
        other => archive_command(other, words),
    }
}

/// 归档容器子命令（`import` / `inspect` / `check`）与单选项命令表的分派。
fn archive_command(name: &str, words: &mut Words<'_>) -> Result<Command> {
    match name {
        "import" => import(words),
        "settle" => settle(words),
        "audit" => audit(words),
        known => path_command(known, words),
    }
}

/// 表内单路径命令；表外名字再交给 `SINGLE`。
///
/// “表外”不是错：`SINGLE` 里的名字各有默认，两者合起来才是全部归档命令。
fn path_command(name: &str, words: &mut Words<'_>) -> Result<Command> {
    let Some((.., flag, what, build)) = PATHS.iter().find(|(candidate, ..)| *candidate == name)
    else {
        return single(name, words);
    };
    let slots = parse_slots(words, &[(flag, what)], name)?;
    Ok(build(required_slot(&slots, 0, flag, name)?))
}

/// `platform`：零选项命令，多给一个词就是写错了（不静默忽略）。
fn platform(words: &mut Words<'_>) -> Result<Command> {
    match words.next() {
        Some(word) => Err(invalid(format!("`platform` 不接受参数，得到 `{word}`"))),
        None => Ok(Command::Platform),
    }
}

/// `import`：`--in` / `--out` 顺序自由，两者都必需——缺源或缺输出都是写错了，不猜默认路径。
fn import(words: &mut Words<'_>) -> Result<Command> {
    let slots = parse_slots(words, IMPORT, "import")?;
    let input = required_slot(&slots, 0, IMPORT[0].0, "import")?;
    let output = required_slot(&slots, 1, IMPORT[1].0, "import")?;
    Ok(Command::Import { input, output })
}

/// `settle`：`--in` 顺序自由但必需，`--out` 可选——缺输入是写错了，缺输出只是换个落点。
fn settle(words: &mut Words<'_>) -> Result<Command> {
    let slots = parse_slots(words, SETTLE, "settle")?;
    let input = required_slot(&slots, 0, SETTLE[0].0, "settle")?;
    Ok(Command::Settle {
        input,
        output: slots[1].clone(),
    })
}

/// `audit`：`--in` / `--ledger` 顺序自由，两者都必需。
fn audit(words: &mut Words<'_>) -> Result<Command> {
    let slots = parse_slots(words, AUDIT, "audit")?;
    let input = required_slot(&slots, 0, AUDIT[0].0, "audit")?;
    let ledger = required_slot(&slots, 1, AUDIT[1].0, "audit")?;
    Ok(Command::Audit { input, ledger })
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
    // 设备后端是进程级注册表，装配点必须在产品路径上（ADR 0025 §1、ADR 0035）：
    // 只在测试里注册会让 `llm run --device cuda` 把“没装配”误报成“没有该后端”。
    engine::backends::install();
    match invocation.command() {
        Command::Help => {
            print!("{USAGE}");
            Ok(())
        }
        Command::Platform => platform::report(),
        Command::Status { .. } | Command::Run { .. } => entry_compute(invocation.command()),
        Command::Verify { .. } | Command::Bench { .. } => audit_note(),
        other => entry_archive_or_train(other),
    }
}

/// 归档容器命令与训练闭环的分派；其余命令已在 [`entry`] 主匹配里收口。
/// 两条读权重却不写状态的命令：`status` 只报各层几何、`run` 真跑一次前向。
fn entry_compute(command: &Command) -> Result<()> {
    match command {
        Command::Status { weights } => status::status(weights.as_deref()),
        Command::Run {
            weights,
            input,
            device,
        } => run::run(weights, input.as_deref(), device.as_deref()),
        other => Err(invalid(format!("命令 `{other:?}` 不是计算命令"))),
    }
}

fn entry_archive_or_train(command: &Command) -> Result<()> {
    match command {
        Command::Import { input, output } => archive::import(input, output),
        Command::Inspect { input } | Command::Check { input } => container(input, command),
        Command::Settle { input, output } => archive::settle(input, output.as_deref()),
        Command::Audit { input, ledger } => archive::audit(input, ledger),
        other => entry_train(other),
    }
}

/// `inspect` 与 `check` 都只说一个文件：差别只在 `check` 多跑一遍验码，入口因此只问一次
/// “是不是验码”，而不是各写一臂。
fn container(input: &str, command: &Command) -> Result<()> {
    if matches!(command, Command::Check { .. }) {
        return archive::check(input);
    }
    archive::inspect(input)
}

/// 训练闭环只有 `train` 一个命令：走到这里还不是它，就是"没接入编排"而不是配置错误。
fn entry_train(command: &Command) -> Result<()> {
    let Command::Train {
        lr,
        warmup,
        steps,
        clip,
        accumulation,
        load,
    } = command
    else {
        return Err(invalid(format!("命令 `{command:?}` 未接入编排")));
    };
    train::train_run(
        lr.as_deref(),
        warmup.as_deref(),
        steps.as_deref(),
        clip.as_deref(),
        accumulation.as_deref(),
        load.as_deref(),
    )
}

/// 组装点不复制审计实现：这两个命令由 `cutile` 二进制提供。
fn audit_note() -> Result<()> {
    Err(Error::new(
        Code::Invalid,
        "该命令当前由 `cutile` 二进制提供（`llm` 只负责组装）",
    )
    .with("cli::entry"))
}

/// `Invocation::parse` 与 [`USAGE`] 的单元测试：文法与用法文本与生产类型同文件，
/// 进程级端到端在 `tests/cli.rs`。
#[cfg(test)]
mod tests {
    use super::*;
    use diagnostic::Code;

    fn args(list: &[&str]) -> Vec<String> {
        list.iter().map(|item| item.to_string()).collect()
    }

    #[test]
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

    #[test]
    fn platform_is_a_zero_option_command() {
        assert_eq!(
            Invocation::parse(&args(&["platform"]))
                .expect("零选项")
                .command(),
            &Command::Platform
        );
        let extra =
            Invocation::parse(&args(&["platform", "--op", "gemv_i4"])).expect_err("不接受参数");
        assert!(extra.to_string().contains("platform"), "{extra}");
    }

    #[test]
    fn a_weights_file_is_carried_through_without_being_opened() {
        assert_eq!(
            Invocation::parse(&args(&["status", "--weights", "model.safetensors"]))
                .expect("合法")
                .command(),
            &Command::Status {
                weights: Some("model.safetensors".to_string())
            }
        );
        let missing = Invocation::parse(&args(&["status", "--weights"])).expect_err("缺取值");
        assert!(missing.to_string().contains("缺少取值"), "{missing}");
        let wrong = Invocation::parse(&args(&["status", "--op", "gemv_i4"]))
            .expect_err("status 只接受 --weights");
        assert!(wrong.to_string().contains("--weights"), "{wrong}");
    }

    #[test]
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
                weights: "model.safetensors".to_string(),
                input: Some("1,2,3".to_string()),
                device: Some("cpu".to_string()),
            }
        );
        let missing =
            Invocation::parse(&args(&["run", "--input", "1,2"])).expect_err("缺 --weights");
        assert!(missing.to_string().contains("--weights"), "{missing}");
        let unknown = Invocation::parse(&args(&["run", "--weights", "m", "--op", "gemv_i4"]))
            .expect_err("未知选项");
        assert!(unknown.to_string().contains("--op"), "{unknown}");
    }

    #[test]
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
                lr: Some("0.01".to_string()),
                warmup: Some("10".to_string()),
                steps: Some("100".to_string()),
                clip: Some("1.0".to_string()),
                accumulation: Some("2".to_string()),
                load: Some("ckpt.snap".to_string()),
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
        let unknown =
            Invocation::parse(&args(&["train", "--op", "gemv_i4"])).expect_err("未知选项");
        assert!(unknown.to_string().contains("--op"), "{unknown}");
        let missing = Invocation::parse(&args(&["train", "--lr"])).expect_err("缺取值");
        assert!(missing.to_string().contains("缺少取值"), "{missing}");
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
                input: "model.safetensors".to_string(),
                output: "model.u".to_string(),
            }
        );
        assert_eq!(
            Invocation::parse(&args(&["inspect", "--in", "model.u"]))
                .expect("合法")
                .command(),
            &Command::Inspect {
                input: "model.u".to_string()
            }
        );
        assert_eq!(
            Invocation::parse(&args(&["check", "--in", "model.u"]))
                .expect("合法")
                .command(),
            &Command::Check {
                input: "model.u".to_string()
            }
        );

        assert_eq!(
            Invocation::parse(&args(&["settle", "--in", "model.u"]))
                .expect("合法")
                .command(),
            &Command::Settle {
                input: "model.u".to_string(),
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
                input: "model.u".to_string(),
                output: Some("config.json".to_string()),
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
                input: "model.u".to_string(),
                ledger: "ledger.json".to_string(),
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
                input: "model.u".to_string(),
                ledger: "ledger.json".to_string(),
            }
        );

        let no_ledger =
            Invocation::parse(&args(&["audit", "--in", "model.u"])).expect_err("缺 --ledger");
        assert!(no_ledger.to_string().contains("--ledger"), "{no_ledger}");
        let missing_out = Invocation::parse(&args(&["import", "--in", "model.safetensors"]))
            .expect_err("缺 --out");
        assert!(missing_out.to_string().contains("--out"), "{missing_out}");
        let missing_in = Invocation::parse(&args(&["check"])).expect_err("缺 --in");
        assert!(missing_in.to_string().contains("--in"), "{missing_in}");
        let unknown =
            Invocation::parse(&args(&["inspect", "--op", "gemv_i4"])).expect_err("未知选项");
        assert!(unknown.to_string().contains("--op"), "{unknown}");
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
        for command in [
            "status", "platform", "run", "verify", "bench", "train", "import", "inspect", "check",
            "settle", "audit", "help",
        ] {
            assert!(USAGE.contains(command), "{command}");
        }
    }
}
