//! 选项文法的一次解析：选项形态、选项表与已收下的选项。
//!
//! 本包有**两层**文法——`llm` 的一级命令（[`crate::argv`]）与 `cutile` 的子命令
//! （[`crate::kernel`]）——形状完全相同：表即文法、按名收词、报错点名选项、必需项在收完时核对。
//! 因此解析器只有这一份（N7），两层各自只提供表与组装函数。
//!
//! **不拥有**：取值语义（哪个缺省算默认、数值范围、路径含义）。本模块只保证「出现了就有一个非空、
//! 不像另一个选项的词」，翻译成域类型是各层组装函数的事。

use diagnostic::{Code, Error, Result};
use std::iter::Peekable;
use std::slice;

/// 参数游标：按词推进，`peek` 用于「下一个词是不是某个选项」。
///
/// 只借出 `&'a str`：文法层不持有 `String`，取值由需要它的组装函数克隆。
pub(crate) type Words<'a> = Peekable<std::iter::Map<slice::Iter<'a, String>, fn(&String) -> &str>>;

/// 把一次调用的参数切成本层认识的游标。
pub(crate) fn words(args: &[String]) -> Words<'_> {
    args.iter()
        .map(String::as_str as fn(&String) -> &str)
        .peekable()
}

/// 选项形态。
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum Kind {
    /// 带取值；缺省由组装方决定（文法层不猜默认值）。
    Value,
    /// 带取值且必需：少了就是写错了，不替调用方补。
    Required,
    /// 不带取值的开关。
    Switch,
}

/// 一张选项表的容量上限：`train` 的六个选项是当前最宽的一张。
///
/// 定长数组让「按表收选项」不必堆分配；超过这里宽度由 [`Flags::parse`] 报内部不变量，
/// 而不是静默截断。
pub(crate) const WIDTH: usize = 8;

/// 一个选项：名字、取值说明（只用于报错）、形态。
#[derive(Clone, Copy)]
pub(crate) struct Flag {
    name: &'static str,
    what: &'static str,
    kind: Kind,
}

/// 带取值、可缺省的选项。
pub(crate) const fn value(name: &'static str, what: &'static str) -> Flag {
    Flag {
        name,
        what,
        kind: Kind::Value,
    }
}

/// 带取值、必需的选项。
pub(crate) const fn required(name: &'static str, what: &'static str) -> Flag {
    Flag {
        name,
        what,
        kind: Kind::Required,
    }
}

/// 不带取值的开关。
pub(crate) const fn switch(name: &'static str) -> Flag {
    Flag {
        name,
        what: "",
        kind: Kind::Switch,
    }
}

/// 一个命令的选项表。表即文法：新增选项只加一行。
pub(crate) type Grammar = &'static [Flag];

/// 按表收下的选项：与文法同序；出现与否存在位图里，开关因此没有取值。
pub(crate) struct Flags<'a> {
    grammar: Grammar,
    present: u8,
    slots: [Option<String>; WIDTH],
    /// 命令名：错误信息用它点名「谁不接受这个参数」。
    owner: &'a str,
    /// 诊断帧：这次失败发生在哪一层解析器。
    context: &'static str,
}

impl<'a> Flags<'a> {
    /// 收完一个命令的全部选项，收完核对必需项。
    ///
    /// # Errors
    ///
    /// 表外的词、缺取值、取值不合法、缺必需项都返回 [`Code::Invalid`]。
    pub(crate) fn parse(
        words: &mut Words<'_>,
        grammar: Grammar,
        owner: &'a str,
        context: &'static str,
    ) -> Result<Self> {
        debug_assert!(grammar.len() <= WIDTH, "选项表比 WIDTH 宽");
        let mut flags = Self {
            grammar,
            present: 0,
            slots: [const { None }; WIDTH],
            owner,
            context,
        };
        while let Some(word) = words.peek().copied() {
            let index = flags.index(word).ok_or_else(|| flags.unknown(word))?;
            let flag = grammar[index];
            words.next();
            flags.present |= 1 << index;
            if flag.kind != Kind::Switch {
                flags.slots[index] = Some(flags.take(words.next(), flag.name)?);
            }
        }
        flags.check_required()?;
        Ok(flags)
    }

    /// 可缺省选项的取值；没出现就是 `None`。
    pub(crate) fn text(&self, name: &str) -> Option<String> {
        self.index(name).and_then(|index| self.slots[index].clone())
    }

    /// 必需选项的取值：[`Flags::parse`] 已核对过，缺了不可能发生。
    pub(crate) fn need(&self, name: &str) -> String {
        self.text(name)
            .expect("parse 已核对必需项，且 name 来自文法表")
    }

    /// 开关是否出现。
    pub(crate) fn switch(&self, name: &str) -> bool {
        self.index(name)
            .is_some_and(|index| self.present & (1 << index) != 0)
    }

    /// 数值选项：十进制解析失败按 `message` 报错（单位由选项自己给）。
    pub(crate) fn number<T: std::str::FromStr>(
        &self,
        name: &str,
        message: &str,
    ) -> Result<Option<T>> {
        match self.text(name) {
            Some(text) => text.parse().map(Some).map_err(|_| self.invalid(message)),
            None => Ok(None),
        }
    }

    /// 一条选项诊断：帧固定在本层解析器。
    pub(crate) fn invalid(&self, message: impl Into<String>) -> Error {
        Error::new(Code::Invalid, message.into()).with(self.context)
    }

    /// 表外的词：`-` 开头是打错的选项，否则是多余参数——两者的修法不同，所以分开说。
    /// 提示直接列出本命令接受的选项，比再贴一遍整段用法更短、更对症。
    fn unknown(&self, word: &str) -> Error {
        let what = if word.starts_with('-') {
            "未知参数"
        } else {
            "多余参数"
        };
        let mut accepted = String::new();
        for flag in self.grammar {
            if !accepted.is_empty() {
                accepted.push_str("、");
            }
            accepted.push_str(flag.name);
        }
        let owner = self.owner;
        if accepted.is_empty() {
            return self.invalid(format!("{what} `{word}`：`{owner}` 不接受选项"));
        }
        self.invalid(format!("{what} `{word}`：`{owner}` 只接受 {accepted}"))
    }

    /// 名字 → 表内下标；表外名字是组装方的编程错误，不是调用方输入错误。
    fn index(&self, name: &str) -> Option<usize> {
        self.grammar.iter().position(|flag| flag.name == name)
    }

    /// 收一个取值：不能缺失、不能为空、不能看起来像另一个选项。
    fn take(&self, given: Option<&str>, flag: &str) -> Result<String> {
        let Some(text) = given else {
            return Err(self.invalid(format!("`{flag}` 缺少取值")));
        };
        if text.is_empty() || text.starts_with('-') {
            return Err(self.invalid(format!("`{flag}` 的取值不合法：`{text}`")));
        }
        Ok(text.to_string())
    }

    /// 必需项核对：缺哪个就点名哪个（含取值说明），不替调用方补默认值。
    fn check_required(&self) -> Result<()> {
        for (index, flag) in self.grammar.iter().enumerate() {
            if flag.kind == Kind::Required && self.slots[index].is_none() {
                let (name, what) = (flag.name, flag.what);
                return Err(self.invalid(format!("`{}` 需要 `{name} <{what}>`", self.owner)));
            }
        }
        Ok(())
    }
}
