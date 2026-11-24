//! `cutile` 子命令的参数文法：内核层审计（doctor / verify / bench / ledger / cache gc）。

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
  fusions                 候选融合按 roofline 加速比排名（能打分的在前）
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
  --profile <档案>        fusions 用（缺省 `cpu`）
  --budget <条数>         fusions 用（缺省 12）
  --limit <字节>          cache gc 用（缺省 2 GiB）
  -h, --help              打印本说明
";

/// 缺省算子名。
pub const DEFAULT_OP: &str = "gemv_i4";

/// 缺省设备名：按指纹选。
pub const AUTO: &str = "auto";

/// 缺省档案名：`fusions` 打分要用一个真实的峰值：缺了就无从定界。
pub const DEFAULT_PROFILE: &str = "cpu";

/// 内核层审计命令。
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum CutileCommand<'a> {
    /// 打印整条流水线的每一步。
    Doctor {
        /// 算子名。
        op: &'a str,
        /// 设备名（`auto` 表示按指纹选）。
        device: &'a str,
        /// 要解释的缓存键（十六进制）；`None` 表示只打印本次构建的键。
        why: Option<&'a str>,
    },
    /// 逐设备验证。
    Verify {
        /// 算子名。
        op: &'a str,
        /// 设备列表。
        devices: Vec<String>,
        /// 是否写能力台账。
        ledger: bool,
    },
    /// 跑一次并报告达成率。
    Bench {
        /// 算子名。
        op: &'a str,
        /// 形状 `MxN`；`None` 表示用算子的生产形状。
        shape: Option<&'a str>,
        /// 设备名或缺省（`auto` 按本机指纹挑）；与 `doctor --device` 同一套取值。
        device: Option<&'a str>,
        /// 重复次数。
        repeat: usize,
    },
    /// 跨档案能力台账。
    Ledger {
        /// 台账 markdown 落盘路径；`None` 只打印到 stdout。
        out: Option<&'a str>,
    },
    /// 候选融合打分。
    Fusions {
        /// 档案名（`auto` 表示按本机指纹选）。
        profile: &'a str,
        /// 候选条数上限。
        budget: usize,
    },
    /// 缓存维护：把缓存回收到容量上限以内。
    Cache {
        /// 容量上限（字节）；`None` 表示用缺省上限。
        limit: Option<u64>,
    },
}

/// 一个子命令的组装：把解析后的选项变成命令（取值非法时返回诊断）。
/// 组装函数：对每个 `'a` 都把借来的取值搬进同一寿命的命令。
type Build = for<'a> fn(&mut Flags<'a>) -> Result<CutileCommand<'a>>;

/// 子命令表：名字、选项表、组装。
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
    (
        "fusions",
        &[value("--profile", "档案"), value("--budget", "条数")],
        fusions,
    ),
];

/// `cache gc` 的选项表：上限是字节数，与 `--repeat` 的次数不同。
const CACHE: Grammar = &[value("--limit", "字节数")];

/// 诊断帧：本层是内核审计命令的文法层。
const CTX: &str = "cli::kernel::parse";

impl<'a> CutileCommand<'a> {
    /// 解析参数（不含 `argv[0]`）。
    ///
    /// # Errors
    ///
    /// 缺子命令、未知子命令/选项、缺取值或取值非法都返回 [`Code::Invalid`]。
    pub fn parse(args: &'a [String]) -> Result<Self> {
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
fn table<'a>(name: &'a str, words: &mut Words<'a>) -> Result<CutileCommand<'a>> {
    let Some((_, grammar, build)) = SUBCOMMANDS.iter().find(|(known, ..)| *known == name) else {
        return Err(invalid(format!("未知子命令 `{name}`")));
    };
    build(&mut Flags::parse(words, grammar, name, CTX)?)
}

/// `doctor`：打印整条流水线。
fn doctor<'a>(flags: &mut Flags<'a>) -> Result<CutileCommand<'a>> {
    Ok(CutileCommand::Doctor {
        op: flags.take_text("--op").unwrap_or(DEFAULT_OP),
        device: flags.take_text("--device").unwrap_or(AUTO),
        why: flags.take_text("--why"),
    })
}

/// `verify`：逐设备验证。
fn verify<'a>(flags: &mut Flags<'a>) -> Result<CutileCommand<'a>> {
    Ok(CutileCommand::Verify {
        op: flags.take_text("--op").unwrap_or(DEFAULT_OP),
        devices: devices(flags.take_text("--devices")),
        ledger: flags.switch("--emit-ledger"),
    })
}

/// `bench`：跑一次并报告达成率。
fn bench<'a>(flags: &mut Flags<'a>) -> Result<CutileCommand<'a>> {
    Ok(CutileCommand::Bench {
        op: flags.take_text("--op").unwrap_or(DEFAULT_OP),
        shape: flags.take_text("--shape"),
        device: flags.take_text("--device"),
        repeat: flags
            .number("--repeat", "`--repeat` 需要正整数")?
            .unwrap_or(50),
    })
}

/// `ledger`：跨档案能力台账。
fn ledger<'a>(flags: &mut Flags<'a>) -> Result<CutileCommand<'a>> {
    Ok(CutileCommand::Ledger {
        out: flags.take_text("--out"),
    })
}

/// `fusions`：候选融合按加速比排名。
fn fusions<'a>(flags: &mut Flags<'a>) -> Result<CutileCommand<'a>> {
    Ok(CutileCommand::Fusions {
        profile: flags.take_text("--profile").unwrap_or(DEFAULT_PROFILE),
        budget: flags
            .number("--budget", "`--budget` 需要正整数")?
            .unwrap_or(12),
    })
}

/// `cache`：唯一动作是 `gc`；缺动作与写错动作分开报错，不静默当成 `gc`。
fn cache<'a>(words: &mut Words<'a>) -> Result<CutileCommand<'a>> {
    let Some(action) = words.next().filter(|word| !word.starts_with('-')) else {
        return Err(invalid("缺少子命令"));
    };
    if action != "gc" {
        return Err(invalid(format!("未知子命令 `cache {action}`")));
    }
    let mut flags = Flags::parse(words, CACHE, "cache gc", CTX)?;
    Ok(CutileCommand::Cache {
        limit: limit(&mut flags)?,
    })
}

/// 上限是字节数（与 `--repeat` 的次数不同），因此单独说明单位。
fn limit(flags: &mut Flags<'_>) -> Result<Option<u64>> {
    flags.number("--limit", "`--limit` 需要字节数（十进制）")
}

/// 设备列表：逗号分隔、去空白、丢弃空项；一个都没有时退回 `cpu`（与缺省一致）。
fn devices(text: Option<&str>) -> Vec<String> {
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
    Error::new(Code::Invalid, USAGE).with(CTX)
}

/// 一条子命令文法的诊断：上下文固定在这一层。
fn invalid(message: impl Into<String>) -> Error {
    Error::new(Code::Invalid, message.into()).with(CTX)
}

/// 子命令文法的单元测试：文法与用法文本与生产类型同文件，进程级端到端在 `tests/cli.rs`。
#[cfg(test)]
#[path = "../tests/kernel.rs"]
mod tests;
