//! `cutile` 子命令：内核层审计（doctor / verify / bench / cache）的**参数文法**。
//!
//! 语法只有一份：内核层的驱动方（`cargo run --bin cutile`）与 `llm cutile ...` 都走这里，
//! 执行在 `llm-audit`，排版在驱动方。`--set key=value` 属于配置层（`llm-config`），
//! 由驱动方在交给本模块之前摘走。

use diagnostic::{Code, Error, Result};

/// 用法文本：唯一一份，`cutile help` 与错误信息都引用它。
pub const USAGE: &str = "\
用法：cutile <子命令> [选项]

子命令：
  doctor                  打印协商链、布局声明、emit 结果与缓存指纹
  verify                  逐设备验证：协商 + 拼接 + 编译 + 与标量参考对拍
  bench                   跑一次并报告带宽达成率
  cache gc                按 LRU 把缓存回收到容量上限以内

选项：
  --op <算子>             算子名（缺省 `gemv_i4`）
  --device <设备>         doctor 用；`auto` 表示按指纹选
  --why <缓存键>          doctor 用；解释该键与本次构建的键差在哪
  --devices <列表>        verify 用；逗号分隔（缺省 `cpu`）
  --emit-ledger           把能力台账写到 `<work_dir>/ledger/`
  --shape <MxN>           bench 用（缺省：算子的生产形状）
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
        /// 重复次数。
        repeat: usize,
    },
    /// 缓存维护：把缓存回收到容量上限以内。
    Cache {
        /// 容量上限（字节）；`None` 表示用缺省上限。
        limit: Option<u64>,
    },
}

impl CutileCommand {
    /// 解析参数（不含 `argv[0]`）。
    ///
    /// # Errors
    ///
    /// 缺少子命令、未知子命令、未知选项、缺少取值或取值非法都返回 [`Code::Invalid`]，
    /// 并在消息里给出 [`USAGE`]。
    pub fn parse(args: &[String]) -> Result<Self> {
        let (head, rest) = args.split_first().ok_or_else(|| invalid("缺子命令"))?;
        match head.as_str() {
            "doctor" => {
                let mut words = Words::new(rest);
                let mut op = DEFAULT_OP.to_string();
                let mut device = "auto".to_string();
                let mut why = None;
                while let Some(flag) = words.next_flag()? {
                    match flag.as_str() {
                        "--op" => op = words.value(&flag)?,
                        "--device" => device = words.value(&flag)?,
                        "--why" => why = Some(words.value(&flag)?),
                        other => return Err(invalid(format!("未知参数 `{other}`"))),
                    }
                }
                Ok(Self::Doctor { op, device, why })
            }
            "verify" => {
                let mut words = Words::new(rest);
                let mut op = DEFAULT_OP.to_string();
                let mut devices = Vec::new();
                let mut ledger = false;
                while let Some(flag) = words.next_flag()? {
                    match flag.as_str() {
                        "--op" => op = words.value(&flag)?,
                        "--devices" => {
                            devices = words
                                .value(&flag)?
                                .split(',')
                                .map(str::trim)
                                .filter(|name| !name.is_empty())
                                .map(str::to_string)
                                .collect();
                        }
                        "--emit-ledger" => ledger = true,
                        other => return Err(invalid(format!("未知参数 `{other}`"))),
                    }
                }
                if devices.is_empty() {
                    devices.push("cpu".to_string());
                }
                Ok(Self::Verify {
                    op,
                    devices,
                    ledger,
                })
            }
            "bench" => {
                let mut words = Words::new(rest);
                let mut op = DEFAULT_OP.to_string();
                let mut shape = None;
                let mut repeat = 50usize;
                while let Some(flag) = words.next_flag()? {
                    match flag.as_str() {
                        "--op" => op = words.value(&flag)?,
                        "--shape" => shape = Some(words.value(&flag)?),
                        "--repeat" => {
                            repeat = words
                                .value(&flag)?
                                .parse::<usize>()
                                .map_err(|_| invalid("`--repeat` 需要正整数"))?;
                        }
                        other => return Err(invalid(format!("未知参数 `{other}`"))),
                    }
                }
                Ok(Self::Bench { op, shape, repeat })
            }
            "cache" => {
                let mut words = Words::new(rest);
                let action = words.positional("子命令")?;
                if action != "gc" {
                    return Err(invalid(format!("未知子命令 `cache {action}`")));
                }
                let mut limit = None;
                while let Some(flag) = words.next_flag()? {
                    match flag.as_str() {
                        "--limit" => {
                            limit = Some(
                                words
                                    .value(&flag)?
                                    .parse::<u64>()
                                    .map_err(|_| invalid("`--limit` 需要字节数（十进制）"))?,
                            );
                        }
                        other => return Err(invalid(format!("未知参数 `{other}`"))),
                    }
                }
                Ok(Self::Cache { limit })
            }
            "help" | "-h" | "--help" => {
                Err(Error::new(Code::Invalid, USAGE).with("cli::cutile::parse"))
            }
            other => Err(invalid(format!("未知子命令 `{other}`"))),
        }
    }
}

fn invalid(message: impl Into<String>) -> Error {
    Error::new(Code::Invalid, format!("{}\n{USAGE}", message.into())).with("cli::cutile::parse")
}

/// 选项游标：`next_flag` 取下一个 `-` 开头的词，`value` 取其取值。
struct Words<'a> {
    rest: &'a [String],
    index: usize,
}

impl<'a> Words<'a> {
    fn new(rest: &'a [String]) -> Self {
        Self { rest, index: 0 }
    }

    /// 位置参数（子命令名之类）：缺词或以 `-` 开头都算缺。
    fn positional(&mut self, what: &str) -> Result<String> {
        let word = self
            .rest
            .get(self.index)
            .ok_or_else(|| invalid(format!("缺少{what}")))?;
        if word.starts_with('-') {
            return Err(invalid(format!("缺少{what}")));
        }
        self.index += 1;
        Ok(word.clone())
    }

    fn next_flag(&mut self) -> Result<Option<String>> {
        let Some(word) = self.rest.get(self.index) else {
            return Ok(None);
        };
        if !word.starts_with('-') {
            return Err(invalid(format!("多余参数 `{word}`")));
        }
        self.index += 1;
        Ok(Some(word.clone()))
    }

    fn value(&mut self, flag: &str) -> Result<String> {
        let value = self
            .rest
            .get(self.index)
            .ok_or_else(|| invalid(format!("`{flag}` 缺少取值")))?;
        if value.is_empty() || value.starts_with('-') {
            return Err(invalid(format!("`{flag}` 的取值不合法：`{value}`")));
        }
        self.index += 1;
        Ok(value.clone())
    }
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
            CutileCommand::parse(&args(&["bench", "--shape", "1024x64", "--repeat", "3"])).unwrap(),
            CutileCommand::Bench {
                op: DEFAULT_OP.to_string(),
                shape: Some("1024x64".to_string()),
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
        let error = CutileCommand::parse(&args(&["cache", "gc", "--limit", "2GiB"])).unwrap_err();
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
