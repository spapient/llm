//! `cutile` 子命令的参数文法：内核层审计（doctor / verify / bench / ledger / cache gc）。
//!
//! 语法只有一份：内核层的驱动方（`cargo run --bin cutile`）与 `llm cutile ...` 都走这里，
//! 执行在 `cutile::{doctor, verify, bench, cache}`，排版在驱动方。`--set key=value` 属于配置层
//! （[`crate::settings`]），由驱动方在交给本模块之前摘走。
//!
//! 表即文法：一个子命令一行（名字、选项表、组装），新增子命令只加一行；解析器本体与一级命令共用
//! [`crate::flag`]，因此两层对「缺取值 / 未知参数 / 缺必需项」的说法逐字一致。

use crate::flag::{Flags, Grammar, Words, switch, value, words};
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

/// 一个子命令的组装：把解析后的选项变成命令（取值非法时返回诊断）。
type Build = fn(&Flags<'_>) -> Result<CutileCommand>;

/// 子命令表：名字、选项表、组装。
///
/// `cache` 还要先吃一个动作词（`gc`），它不在表里，由 [`CutileCommand::parse`] 单独分派。
const SUBCOMMANDS: &[(&str, Grammar, Build)] = &[
    (
        "doctor",
        &[
            value("--op", "算子"),
            value("--device", "设备"),
            value("--why", "缓存键"),
        ],
        doctor,
    ),
    (
        "verify",
        &[
            value("--op", "算子"),
            value("--devices", "设备列表"),
            switch("--emit-ledger"),
        ],
        verify,
    ),
    (
        "bench",
        &[
            value("--op", "算子"),
            value("--shape", "形状"),
            value("--device", "设备"),
            value("--repeat", "次数"),
        ],
        bench,
    ),
    ("ledger", &[value("--out", "文件")], ledger),
];

/// `cache gc` 的选项表：上限是字节数，与 `--repeat` 的次数不同。
const CACHE: Grammar = &[value("--limit", "字节数")];

/// 诊断帧：本层是内核审计命令的文法层。
const CONTEXT: &str = "cli::kernel::parse";

impl CutileCommand {
    /// 解析参数（不含 `argv[0]`）。
    ///
    /// # Errors
    ///
    /// 缺子命令、未知子命令/选项、缺取值或取值非法都返回 [`Code::Invalid`]。
    pub fn parse(args: &[String]) -> Result<Self> {
        let mut words = words(args);
        let Some(head) = words.next() else {
            return Err(invalid("缺子命令"));
        };
        match head {
            "help" | "-h" | "--help" => Err(help()),
            "cache" => cache(&mut words),
            name => table(name, &mut words),
        }
    }
}

/// 表内子命令：按名找行、收选项、组装；表外的名字不猜。
fn table(name: &str, words: &mut Words<'_>) -> Result<CutileCommand> {
    let Some((_, grammar, build)) = SUBCOMMANDS.iter().find(|(known, ..)| *known == name) else {
        return Err(invalid(format!("未知子命令 `{name}`")));
    };
    build(&Flags::parse(words, grammar, name, CONTEXT)?)
}

/// `doctor`：打印整条流水线。
fn doctor(flags: &Flags<'_>) -> Result<CutileCommand> {
    Ok(CutileCommand::Doctor {
        op: flags.text("--op").unwrap_or_else(|| DEFAULT_OP.to_string()),
        device: flags.text("--device").unwrap_or_else(|| "auto".to_string()),
        why: flags.text("--why"),
    })
}

/// `verify`：逐设备验证。
fn verify(flags: &Flags<'_>) -> Result<CutileCommand> {
    Ok(CutileCommand::Verify {
        op: flags.text("--op").unwrap_or_else(|| DEFAULT_OP.to_string()),
        devices: devices(flags.text("--devices")),
        ledger: flags.switch("--emit-ledger"),
    })
}

/// `bench`：跑一次并报告达成率。
fn bench(flags: &Flags<'_>) -> Result<CutileCommand> {
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
fn ledger(flags: &Flags<'_>) -> Result<CutileCommand> {
    Ok(CutileCommand::Ledger {
        out: flags.text("--out"),
    })
}

/// `cache`：唯一动作是 `gc`；缺动作与写错动作分开报错，不静默当成 `gc`。
fn cache(words: &mut Words<'_>) -> Result<CutileCommand> {
    let Some(action) = words.next().filter(|word| !word.starts_with('-')) else {
        return Err(invalid("缺少子命令"));
    };
    if action != "gc" {
        return Err(invalid(format!("未知子命令 `cache {action}`")));
    }
    let flags = Flags::parse(words, CACHE, "cache gc", CONTEXT)?;
    Ok(CutileCommand::Cache {
        limit: limit(&flags)?,
    })
}

/// 上限是字节数（与 `--repeat` 的次数不同），因此单独说明单位。
fn limit(flags: &Flags<'_>) -> Result<Option<u64>> {
    flags.number("--limit", "`--limit` 需要字节数（十进制）")
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

/// `help`：用法文本本身就是诊断消息（不带前缀），类别仍是「输入不合法」。
fn help() -> Error {
    Error::new(Code::Invalid, USAGE).with(CONTEXT)
}

/// 一条子命令文法的诊断：上下文固定在这一层。
fn invalid(message: impl Into<String>) -> Error {
    Error::new(Code::Invalid, message.into()).with(CONTEXT)
}

/// 子命令文法的单元测试：文法与用法文本与生产类型同文件，进程级端到端在 `tests/cli.rs`。
#[cfg(test)]
#[path = "../tests/kernel.rs"]
mod tests;
