//! 缓存身份：源文本指纹、产物身份与逐项差异。
//!
//! 存储/回收在 [`crate::cache`]；本模块只回答"一条输入长什么样、一份产物由哪几项指认"，
//! 不碰目录、锁与时间戳。键的字段是 `meta.txt`、哈希与差异比较的唯一出处。

use crate::atom::{DTypeSet, Prim};
use crate::emit::fnv1a;
use crate::prof::Emit;
use std::collections::BTreeMap;

/// 缓存指纹（源文本的身份）：只有声明侧的输入；工具链身份在 [`ArtifactKey`] 里，并成同一个键会让换编译器复用旧代码生成。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CacheKey {
    /// cutile 自身版本。
    pub schema_ver: u16,
    /// 输出语言。
    pub emit: Emit,
    /// 编译参数哈希。
    pub flags_hash: u64,
    /// 架构键。
    pub arch: String,
    /// AST 规范编码的哈希：`FNV(ast::encode(&nodes))`。
    pub ast_hash: u64,
    /// 参与拼接的片段与内容哈希。
    pub fragments: Vec<(Prim, u64)>,
    /// 布局声明式（槽位 → `row` / `col` / `swz<…>` / `pad(p)`）。
    pub layouts: Vec<(String, String)>,
    /// 变体名。
    pub variant: String,
    /// tile 取值。
    pub tile: String,
    /// 数据类型集合（声明 `[needs].dt` 的并集）。
    pub dtype: DTypeSet,
}

impl CacheKey {
    /// 逐项字段（名字, 值）：哈希、`meta.txt` 与差异共用这一份；片段与布局排序，否则同样输入会算出两个键。
    #[must_use]
    pub fn fields(&self) -> Vec<(&'static str, String)> {
        let mut fragments = self.fragments.clone();
        fragments.sort_by_key(|(prim, _)| *prim);
        let mut layouts = self.layouts.clone();
        layouts.sort();
        vec![
            ("schema", self.schema_ver.to_string()),
            ("emit", self.emit.to_string()),
            ("flags", format!("{:016x}", self.flags_hash)),
            ("arch", self.arch.clone()),
            ("ast", format!("{:016x}", self.ast_hash)),
            ("fragments", format!("{fragments:?}")),
            ("layouts", format!("{layouts:?}")),
            ("variant", self.variant.clone()),
            ("tile", self.tile.clone()),
            ("dtype", self.dtype.to_string()),
        ]
    }
    /// 规范化描述：`fields` 拼成一行，`meta.txt` 与诊断文本都用它。
    #[must_use]
    pub fn describe(&self) -> String {
        let fields = self.fields();
        fields
            .iter()
            .map(|(n, v)| format!("{n}={v}"))
            .collect::<Vec<_>>()
            .join(";")
    }
    /// 与一份已落盘的键快照比较，只返回不同的项：只在一侧出现的也是差异，不能静默当成没变。
    #[must_use]
    pub fn changes(&self, stored: &[(String, String)]) -> Vec<Changed> {
        let current = self.fields();
        let present = |field: &str| current.iter().any(|(name, _)| *name == field);
        let old = |field: &str| stored.iter().find(|(name, _)| name == field);
        let mut changes = Vec::new();
        for (field, value) in &current {
            let cached = old(field).map(|(_, value)| value.clone());
            if cached.as_deref() != Some(value.as_str()) {
                changes.push(Changed::of(field, cached, Some(value.clone())));
            }
        }
        let extra = stored.iter().filter(|(field, _)| !present(field));
        changes.extend(extra.map(|(f, v)| Changed::of(f, Some(v.clone()), None)));
        changes
    }
    /// 指纹哈希。
    pub fn hash(&self) -> u64 {
        fnv1a(&self.describe())
    }
    /// 十六进制形式（目录名）。
    pub fn hex(&self) -> String {
        format!("{:016x}", self.hash())
    }
}

/// 两个键的一处差异：字段、缓存里记的值、当前键的值；两侧都是 `Option`，只在一侧出现的字段也要说清。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Changed {
    /// 字段名。
    pub field: String,
    /// 缓存里记的值；`None` 表示旧版本还没有这一项。
    pub stored: Option<String>,
    /// 当前键的值；`None` 表示这一项已不再参与指纹。
    pub current: Option<String>,
}

impl Changed {
    fn of(field: &str, stored: Option<String>, current: Option<String>) -> Self {
        Self {
            field: field.to_string(),
            stored,
            current,
        }
    }
}

/// 一个编译产物的身份：源文本 + 实测工具链（[`crate::toolchain::Toolchain::id`]）+ 编译参数 + 主函数；缺一项都会复用成错误的产物。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ArtifactKey {
    /// 源文本的缓存键（十六进制）。
    pub source: String,
    /// 工具链身份（`命令@实测版本`）。
    pub tool: String,
    /// 编译参数的指纹（见 [`ArtifactKey::params`]）。
    pub params: u64,
    /// 拼进去的主函数的哈希。
    pub harness: u64,
}

impl ArtifactKey {
    /// 编译参数的指纹：算子走 [`CacheKey::flags_hash`]，见证程序的参数由调用方给；按顺序连接不排序，顺序本身就是命令行的一部分。
    #[must_use]
    pub fn params(flags: &[String]) -> u64 {
        fnv1a(&flags.join(" "))
    }
    /// 由任意源文本身份构造：算子靠声明指纹、见证靠程序文本，只是"源文本怎么指认"不同。
    #[must_use]
    pub fn keyed(source: impl Into<String>, tool: &str, params: u64, harness: &str) -> Self {
        Self {
            source: source.into(),
            tool: tool.to_string(),
            params,
            harness: fnv1a(harness),
        }
    }
    /// 从 [`Self::describe`] 的文本还原身份；解析失败或缺 `params=`（用哪组参数已无从查起）都算没有身份。
    pub(crate) fn parse(text: &str) -> Option<Self> {
        let fields: BTreeMap<&str, &str> = text
            .split(';')
            .filter_map(|item| item.split_once('='))
            .collect();
        let hex = |n: &str| fields.get(n).and_then(|t| u64::from_str_radix(t, 16).ok());
        Some(Self {
            source: fields.get("source")?.to_string(),
            tool: fields.get("tool")?.to_string(),
            params: hex("params")?,
            harness: hex("harness")?,
        })
    }

    /// 规范化描述。
    #[must_use]
    pub fn describe(&self) -> String {
        format!(
            "source={};tool={};params={:016x};harness={:016x}",
            self.source, self.tool, self.params, self.harness
        )
    }

    /// 十六进制形式（产物文件名）。
    #[must_use]
    pub fn hex(&self) -> String {
        format!("{:016x}", fnv1a(&self.describe()))
    }
}
