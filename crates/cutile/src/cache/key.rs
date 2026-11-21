//! 缓存的身份：键怎么由代码与参数算出、条目怎么判定新旧。
//!
//! 与生命周期（[`super`]）分开是因为两者变更理由不同：这里判「这是不是同一个产物」，
//! 那里判「这个产物还留不留」。身份算错会命中错产物，回收算错只会多花磁盘。

use crate::atom::{DTypeSet, Prim};
use crate::emit::fnv1a;
use crate::prof::Emit;
use std::collections::BTreeMap;
use std::path::PathBuf;
use std::time::{Duration, SystemTime};

fn fingerprint_hex(hash: u64) -> String {
    format!("{:016x}", hash)
}

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
    /// 内核实现标签（与档案 `provides.impl_name` 同源）；同原语不同实现必须是两条独立条目（ADR 0027 §6）。
    pub impl_name: String,
    /// 分块宽度（tile 尺寸 / 量化块宽）；标量实现记 `1`，P1 plan 层填真实块宽。
    pub block: String,
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
            ("algorithm", self.arch.clone()),
            ("impl", self.impl_name.clone()),
            ("block", self.block.clone()),
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
        self.fields()
            .iter()
            .map(|(n, v)| format!("{n}={v}"))
            .collect::<Vec<_>>()
            .join(";")
    }
    /// 与一份已落盘的键快照比较，只返回不同的项：只在一侧出现的也是差异，不能静默当成没变。
    #[must_use]
    pub fn changes(&self, stored: &[(String, String)]) -> Vec<Changed> {
        let current = self.fields();
        // 快照侧先建一次索引，逐项比对走查表；多出的字段仍按落盘顺序列出。
        let by_name: BTreeMap<&str, &str> = stored
            .iter()
            .map(|(n, v)| (n.as_str(), v.as_str()))
            .collect();
        let mut changes = Vec::new();
        for (field, value) in &current {
            let cached = by_name.get(*field).map(|v| v.to_string());
            if cached.as_deref() != Some(value.as_str()) {
                changes.push(Changed::of(field, cached, Some(value.clone())));
            }
        }
        for (field, stored_value) in stored {
            if !current.iter().any(|(n, _)| *n == field.as_str()) {
                changes.push(Changed::of(field, Some(stored_value.clone()), None));
            }
        }
        changes
    }
    /// 指纹哈希。
    pub fn hash(&self) -> u64 {
        fnv1a(&self.describe())
    }
    /// 十六进制形式（目录名）。
    pub fn hex(&self) -> String {
        fingerprint_hex(self.hash())
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

/// 一个编译产物的身份：源文本 + 实测工具链（`Toolchain::id`（在 `cutile-run`））+ 编译参数 + 主函数；缺一项都会复用成错误的产物。
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
            tool: fields.get("tools")?.to_string(),
            params: hex("params")?,
            harness: hex("harness")?,
        })
    }

    /// 规范化描述。
    #[must_use]
    pub fn describe(&self) -> String {
        format!(
            "source={};tools={};params={:016x};harness={:016x}",
            self.source, self.tool, self.params, self.harness
        )
    }

    /// 十六进制形式（产物文件名）。
    #[must_use]
    pub fn hex(&self) -> String {
        fingerprint_hex(fnv1a(&self.describe()))
    }
}

/// 陈旧判据：一次 emit 是毫秒级、等锁上限 5 s，一分钟没被动过的锁与半截目录只可能是崩溃遗留。
pub const STALE_LOCK: Duration = Duration::from_secs(60);

/// 缓存容量上限的兜底缺省值（README §10.3）：2 GiB。
///
/// 策略层（`cli::settings` 的 `cache.limit_bytes`，默认值与此相同）负责把配置投影成显式
/// 上限；内核不读配置，只在调用方没有接线时用这个常量兜底。
pub const CACHE_LIMIT: u64 = 2 * 1024 * 1024 * 1024;

/// 一份缓存目录的状态。
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub enum State {
    /// 有 `source`：可以直接复用。
    Complete,
    /// 没有 `source`：写到一半就没了，永远不会命中，只会占空间。
    #[default]
    Partial,
}

/// 缓存里的一条记录：目录、占用、最后一次使用时间与落盘时的键快照。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Entry {
    /// 键的十六进制形式（也是目录名）。
    pub hex: String,
    /// 该键的目录。
    pub dir: PathBuf,
    /// 目录内全部文件的字节数（递归，含 `bin/` 与残留临时文件）。
    pub bytes: u64,
    /// 最后一次使用（读命中或写入）的时间：GC 的 LRU 顺序用它。
    pub used: SystemTime,
    /// 落盘时的 [`CacheKey::fields`] 快照。
    pub fields: Vec<(String, String)>,
    /// 目录内容：有 `source` 才是可复用的一条记录。
    pub state: State,
    /// 目录里认得出身份的编译产物个数（[`Cache::artifacts_in`]）。
    pub artifacts: usize,
    /// 目录里是否还有 `build.lock`：有表示有人正在重新生成它。
    pub locked: bool,
}

impl Entry {
    /// 是否可复用（有 `source`）。
    #[must_use]
    pub fn complete(&self) -> bool {
        self.state == State::Complete
    }
    /// 是否正在被使用、因而不能回收：完整条目看活锁，半截落盘看 [`STALE_LOCK`] 窗口过没过。
    #[must_use]
    pub fn building(&self, now: SystemTime) -> bool {
        let active = now.duration_since(self.used).unwrap_or_default() < STALE_LOCK;
        active && (self.state == State::Partial || self.locked)
    }
}
