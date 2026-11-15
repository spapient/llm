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

/// `bench` 的选项文法。
const BENCH: Grammar = &[("--op", Value), ("--shape", Value), ("--repeat", Value)];

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
            Error::new(Code::Invalid, USAGE).with("cli::cutile::parse")
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
                Kind::Value => Some(value(words.next(), word)?),
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

/// 选项取值：不能缺失，不能为空，也不能看起来像另一个选项。
fn value(given: Option<&String>, flag: &str) -> Result<String> {
    let Some(value) = given else {
        return Err(invalid(format!("`{flag}` 缺少取值")));
    };
    if value.is_empty() || value.starts_with('-') {
        return Err(invalid(format!("`{flag}` 的取值不合法：`{value}`")));
    }
    Ok(value.clone())
}

fn invalid(message: impl Into<String>) -> Error {
    Error::new(Code::Invalid, format!("{}\n{USAGE}", message.into())).with("cli::cutile::parse")
}

#[cfg(test)]
#[path = "cutile_tests.rs"]
mod tests;
