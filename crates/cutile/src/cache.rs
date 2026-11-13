//! ⑤ 缓存：指纹、编译产物、锁、原子落盘、容量回收（DESIGN §10.3）。
//!
//! 指纹必须是"改任何输入都换 key"：schema 版本、emit 语言、编译参数、架构、模板哈希、
//! 参与拼接的片段哈希、布局声明式、变体与 tile、数据类型。**工具链不在其中**：源文本与
//! 编译器无关，换编译器只换产物——那件事由 [`ArtifactKey`] 的 `tool` 字段承担，而它是
//! **实测**的工具链身份（命令名@版本），不是档案里的登记值。
//!
//! 键的逐项字段只有一个出处（[`CacheKey::fields`]）：哈希、`meta.txt` 与"为什么没命中"的
//! 差异都从它派生。存量管理同理只有一份（[`Cache::entries`]）：GC 删除的条目与 `doctor`
//! 报告的条目是同一批目录。
//!
//! 一个条目里有两种东西：源文本（`source` + `meta.txt`，由 [`CacheKey`] 定名）与编译产物
//! （`bin/<hex>`，由 [`ArtifactKey`] 定名）。产物必须单独定名，因为它还取决于主函数——
//! 同一个源文本在三个形状下就是三个可执行文件。产物进了缓存，命中才能连 `cc` 一起省掉。
//!
//! 每份产物旁边还写一行身份（`bin/<hex>.key`，[`ArtifactKey::describe`] 的原文）：编译器名
//! 因此躺在产物身边，本机装着哪个编译器都能把已有的产物认回来（[`Cache::artifacts_in`]）。
//!
//! "这条记录还能不能用"只有两个答案，而且都由目录内容决定：[`State::Complete`] 有 `source`，
//! 可以复用；[`State::Partial`] 没有，永远不会命中。后者不是理论情况——写到一半被 `kill`、
//! 断点停在 `put` 中间、磁盘写满，都会留下它；旧实现把它当成"别的工具建的目录"跳过，于是
//! 它既不计入占用也永远不被回收。同理，`build.lock` 是 `O_EXCL` 文件而不是 flock，持有者
//! 崩溃后没人能释放它，那一个键就会永远卡在"等待超时"上：两处的判据都是 [`STALE_LOCK`]。

use crate::atom::{DTypeSet, Prim};
use crate::emit::fnv1a;
use crate::prof::Emit;
use diagnostic::{Code, Error};
use std::fmt;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, SystemTime};

/// 陈旧判据：超过这个时间没被动过的锁文件与半截目录都算崩溃遗留。
///
/// 一次 emit 是毫秒级（生成的是几十行源文本），等锁的上限也只有 5 s，因此一分钟前创建的锁
/// 不可能还有活着的持有者；反过来，把判据取得更大只会让崩溃的代价拖得更久。
pub const STALE_LOCK: Duration = Duration::from_secs(60);

/// 缓存指纹（源文本的身份）。
///
/// 它描述的是"emit 出哪段文本"，所以只有声明侧的输入：换编译器不换这段文本，只换由它
/// 编出来的产物。工具链身份在 [`ArtifactKey`] 里。
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
    /// AST 规范编码的哈希：`FNV(ast::encode(&nodes))`，替代旧的裸模板文本哈希。
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
    /// 逐项字段（名字, 值）：哈希、`meta.txt` 与差异比较共用这一份。
    ///
    /// 片段与布局先排序，保证同一输入得到同一顺序（否则同样的输入会算出两个键）。
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
        self.fields()
            .into_iter()
            .map(|(name, value)| format!("{name}={value}"))
            .collect::<Vec<_>>()
            .join(";")
    }

    /// 与一份已落盘的键快照比较：只返回不同的项（含只在一侧出现的项）。
    #[must_use]
    pub fn changes(&self, stored: &[(String, String)]) -> Vec<Changed> {
        let current = self.fields();
        let mut changes = Vec::new();
        for (name, value) in &current {
            match stored.iter().find(|(field, _)| field == name) {
                Some((_, old)) if old == value => {}
                Some((_, old)) => changes.push(Changed {
                    field: (*name).to_string(),
                    stored: Some(old.clone()),
                    current: Some(value.clone()),
                }),
                None => changes.push(Changed {
                    field: (*name).to_string(),
                    stored: None,
                    current: Some(value.clone()),
                }),
            }
        }
        for (name, value) in stored {
            if !current.iter().any(|(field, _)| field == name) {
                changes.push(Changed {
                    field: name.clone(),
                    stored: Some(value.clone()),
                    current: None,
                });
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
        format!("{:016x}", self.hash())
    }
}

impl fmt::Display for CacheKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.describe())
    }
}

/// 一份缓存目录的状态。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum State {
    /// 有 `source`：可以直接复用。
    Complete,
    /// 没有 `source`：写到一半就没了（被 `kill`、磁盘写满、断点停在 `put` 中间）。
    /// 它永远不会命中，只会占空间——回收它是维护命令的义务，不是容量问题。
    Partial,
}

/// 缓存里的一条记录：目录、占用、最后一次使用时间与落盘时的键快照。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Entry {
    /// 键的十六进制形式（也是目录名）。
    pub hex: String,
    /// 该键的目录。
    pub dir: PathBuf,
    /// 目录内全部文件的字节数（递归，含 `bin/` 下的产物与残留的临时文件）。
    pub bytes: u64,
    /// 最后一次使用（读命中或写入）的时间：GC 的 LRU 顺序用它。
    pub used: SystemTime,
    /// 落盘时的 [`CacheKey::fields`] 快照。
    pub fields: Vec<(String, String)>,
    /// 目录内容：有 `source` 才是可复用的一条记录。
    pub state: State,
    /// 目录里已落盘的编译产物个数（`bin/` 下以产物键命名的文件）。
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

    /// 是否最近被动过（写入、读命中或取锁）。
    ///
    /// 超过 [`STALE_LOCK`] 还没被动过的，都是崩溃遗留：`O_EXCL` 锁没有持有者可以来
    /// 释放它，半截的 `source.tmp` 也没有谁会接着写完。
    #[must_use]
    pub fn active(&self, now: SystemTime) -> bool {
        now.duration_since(self.used).unwrap_or_default() < STALE_LOCK
    }

    /// 是否正在被使用、因而不能回收。
    ///
    /// 两种状态分开判：完整条目只要还有活着的锁就不能动（对方马上要覆写它），但"最近被
    /// 命中的完整条目"该不该删是容量问题，交给 LRU；半截落盘的没有可丢弃的 `source`，
    /// 只要最近被动过就可能还在写，等窗口过去再回收。
    #[must_use]
    pub fn building(&self, now: SystemTime) -> bool {
        match self.state {
            State::Complete => self.locked && self.active(now),
            State::Partial => self.active(now),
        }
    }
}

/// 两个键的一处差异：字段、缓存里记的值、当前键的值。
///
/// 两侧都是 `Option`：只会慢慢变多的字段（新版本加的）与已被删掉的字段都要能说清楚，
/// 不能静默当成"没变"。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Changed {
    /// 字段名。
    pub field: String,
    /// 缓存里记的值；`None` 表示旧版本还没有这一项。
    pub stored: Option<String>,
    /// 当前键的值；`None` 表示这一项已不再参与指纹。
    pub current: Option<String>,
}

/// 残留：写到一半就没了的缓存目录，永远不会命中。
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Residue {
    /// 回收掉的目录数。
    pub items: usize,
    /// 它们占过的字节数。
    pub bytes: u64,
}

/// 一次容量回收的结果。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Gc {
    /// 容量上限（字节）。
    pub limit: u64,
    /// 回收前的占用（含残留）。
    pub before: u64,
    /// 回收后的占用：仍高于上限时说明剩下的条目都在用。
    pub after: u64,
    /// 按 LRU 删除的可复用条目数。
    pub removed: usize,
    /// 回收掉的残留：它们本来就不会命中，所以与上限无关、一律回收。
    pub residue: Residue,
    /// 因为仍在被使用（[`Entry::active`]）而跳过的目录数。
    pub skipped: usize,
}

/// 一个编译产物的身份：源文本 + **实测工具链身份** + **编译参数** + 拼进去的主函数。
///
/// 四者缺一都会被复用成错误的产物：源文本决定内核，工具链决定代码生成，参数决定同一份源码
/// 被优化成什么，主函数决定样本、形状与重复次数。工具链这一项是
/// [`crate::backend::Toolchain::id`] 给出的 `命令@版本`（例如 `cc@21.0.0`）——只写命令名
/// 不够：升级一次编译器就换了一套代码生成，旧产物不能再冒充新编译器的产物。档案声明的
/// `[meta].tool_ver` 是人写的登记值，不参与这里。
///
/// 参数这一项是 [`ArtifactKey::params`] 的指纹：它不改变源码，却能改变产物（`-O0` 与 `-O3`
/// 是两份不同的代码生成）。见证这类不在声明里的程序原本只按 `标记;程序文本` 定名，于是改一次
/// 档案的 `[compile].flags` 会复用旧参数编出来的可执行文件——「这条见证跑通过」于是对着一份
/// 不再会被编出来的程序成立。把参数并进身份之后，参数的每次改动都换一份产物，证据的适用条件
/// 与协商的输入（片段、工具链、参数）对齐。
///
/// 产物因此有自己的哈希，但在目录里跟随父条目（`<条目>/bin/<hash>`）：它的 `source` 与
/// `meta.txt` 就是父条目的那一份，不该为它复制一套。
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
    /// 由缓存键、编译器与主函数文本构造。
    ///
    /// 参数取自 [`CacheKey::flags_hash`]：算子的缓存键已经把 `[compile].flags` 与设备
    /// preamble 一起哈希过，产物身份不必另算一份。
    #[must_use]
    pub fn new(key: &CacheKey, tool: &str, harness: &str) -> Self {
        Self::keyed(key.hex(), tool, key.flags_hash, harness)
    }

    /// 编译参数的指纹。
    ///
    /// 算子走 [`CacheKey::flags_hash`]，而见证程序没有缓存键，参数由调用方给：两者是同一件事
    /// ——"这次传给编译器的参数是哪一组"。按顺序用空格连接，是因为编译器看到的就是这一行；
    /// 参数顺序本身也是命令行的一部分（`-I` 之类会出现先到先得），所以不排序。
    #[must_use]
    pub fn params(flags: &[String]) -> u64 {
        fnv1a(&flags.join(" "))
    }

    /// 由任意源文本身份构造：算子的缓存键、见证程序的条目键……
    ///
    /// 两者只是"源文本怎么指认"不同（算子靠声明指纹，见证靠程序文本），产物身份的形状一致。
    #[must_use]
    pub fn keyed(source: impl Into<String>, tool: &str, params: u64, harness: &str) -> Self {
        Self {
            source: source.into(),
            tool: tool.to_string(),
            params,
            harness: fnv1a(harness),
        }
    }

    /// 从 [`Self::describe`] 的文本还原身份（产物旁的 `.key` 记的就是这一行）。
    ///
    /// 解析失败即当作"没有身份"：宁可不复用，也不把一份来历不明的可执行文件当成这里的产物。
    ///
    /// 缺 `params=` 的旧产物同样算没有身份：它是在参数还没进身份的时候编出来的，那一轮用的是
    /// 哪组参数已经无从查起。缓存按定义可重建，重编一次的代价远小于把证据挂到未知参数上。
    fn parse(text: &str) -> Option<Self> {
        let mut source = None;
        let mut tool = None;
        let mut params = None;
        let mut harness = None;
        for item in text.split(';') {
            let (name, value) = item.split_once('=')?;
            match name {
                "source" => source = Some(value.to_string()),
                "tool" => tool = Some(value.to_string()),
                "params" => params = Some(u64::from_str_radix(value, 16).ok()?),
                "harness" => harness = Some(u64::from_str_radix(value, 16).ok()?),
                _ => {}
            }
        }
        Some(Self {
            source: source?,
            tool: tool?,
            params: params?,
            harness: harness?,
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

/// 一份编译产物在缓存里的位置与来历。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Artifact {
    /// 产物身份。
    pub key: ArtifactKey,
    /// 可执行文件路径（`<条目>/bin/<hex>`）。
    pub path: PathBuf,
    /// 文件字节数。
    pub bytes: u64,
    /// 这次是命中复用（没调用编译器），还是刚编出来的。
    pub reused: bool,
}

/// 缓存目录。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Cache {
    root: PathBuf,
}

impl Cache {
    /// 指定根目录。
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }

    /// 由配置层解析出的缓存目录构造。
    ///
    /// 目录本身的默认值与来源（`CUTILE_CACHE_DIR`、`CUTILE_CACHE`、配置文件的 `cache.dir`）
    /// 统一由 `llm-config` 决定，这里不再自己读环境变量或拼相对路径。
    pub fn from_settings(settings: &llm_config::Settings) -> Self {
        Self::new(settings.cache_dir())
    }

    /// 根目录。
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// 某个 key 的目录：`<root>/<前两位>/<hex>`。
    pub fn dir_for(&self, key: &CacheKey) -> PathBuf {
        self.entry_dir(&key.hex())
    }

    /// 源文本键对应的条目目录（不检查存在性）。
    fn entry_dir(&self, hex: &str) -> PathBuf {
        let shard = hex.get(..2).unwrap_or(hex);
        self.root.join(shard).join(hex)
    }

    /// 一份产物的路径：`<条目>/bin/<产物 hex>`。
    #[must_use]
    pub fn artifact_path(&self, key: &ArtifactKey) -> PathBuf {
        self.entry_dir(&key.source).join("bin").join(key.hex())
    }

    /// 读产物：命中即返回，并顺手把条目的时间戳刷新（LRU 靠它，见 [`Self::get`]）。
    #[must_use]
    pub fn artifact(&self, key: &ArtifactKey) -> Option<Artifact> {
        let path = self.artifact_path(key);
        let bytes = std::fs::metadata(&path).ok()?.len();
        touch(&self.entry_dir(&key.source).join("meta.txt"));
        Some(Artifact {
            key: key.clone(),
            path,
            bytes,
            reused: true,
        })
    }

    /// 一个条目里已落盘的每一份产物，带"谁编译的、拼了哪个主函数"。
    ///
    /// 没有编译器时靠它复用产物：身份写在产物旁边（`bin/<hex>.key`），所以不必重新探测
    /// 工具链才能拼出文件名。没有身份文件的不在此列——不知道它由谁编译、
    /// 拼了哪个主函数，复用它是赌；只有身份文件、可执行文件已不在的也不算产物。
    #[must_use]
    pub fn artifacts_in(&self, source: &str) -> Vec<Artifact> {
        let bin = self.entry_dir(source).join("bin");
        let Ok(files) = std::fs::read_dir(&bin) else {
            return Vec::new();
        };
        files
            .flatten()
            .filter_map(|file| {
                let name = file.file_name().to_str()?.to_string();
                if !is_key(&name) {
                    return None;
                }
                let text = std::fs::read_to_string(bin.join(format!("{name}.key"))).ok()?;
                let key = ArtifactKey::parse(text.trim())?;
                let path = file.path();
                let bytes = std::fs::metadata(&path).ok()?.len();
                Some(Artifact {
                    key,
                    path,
                    bytes,
                    reused: true,
                })
            })
            .collect()
    }

    /// 落一份产物：先拷到同目录的临时名再 `rename`，并发写同一个键不会读到半截可执行文件。
    ///
    /// 落盘值与身份文件分两步写：可执行文件先到位，身份随后补上。中途崩溃会留下"没有身份"的
    /// 产物——精确键仍能用它，按身份扫描则看不见它，正是这里想要的保守方向。
    ///
    /// # Errors
    ///
    /// `bin/` 不可建、拷贝失败或改名失败时返回系统错误（调用方给出带路径的说明）。
    pub fn put_artifact(&self, key: &ArtifactKey, exe: &Path) -> std::io::Result<Artifact> {
        let path = self.artifact_path(key);
        let bin = path
            .parent()
            .map(Path::to_path_buf)
            .unwrap_or_else(|| path.clone());
        std::fs::create_dir_all(&bin)?;
        let temp = temp_path(&bin, &key.hex());
        // `copy` 保留可执行位：直接 `rename` 会丢掉它，产物就再也跑不起来。
        std::fs::copy(exe, &temp)?;
        std::fs::rename(&temp, &path)?;
        // 身份写在产物旁边：没有编译器的机器靠它把产物认回来（[`Self::artifacts_in`]），
        // 而不是让工具链探测猜出一个文件名。
        let identity = bin.join(format!("{}.key", key.hex()));
        let temp = temp_path(&bin, "key");
        std::fs::write(&temp, format!("{}\n", key.describe()))?;
        std::fs::rename(&temp, &identity)?;
        let bytes = std::fs::metadata(&path).map(|meta| meta.len()).unwrap_or(0);
        touch(&self.entry_dir(&key.source).join("meta.txt"));
        Ok(Artifact {
            key: key.clone(),
            path,
            bytes,
            reused: false,
        })
    }

    /// 读缓存（命中即返回源文本）。
    ///
    /// 命中就是"用过了"：顺手更新 `meta.txt` 的时间戳，否则 GC 的 LRU 只会退化成
    /// "最近写入"，一份天天命中的缓存反而先被删。
    pub fn get(&self, key: &CacheKey) -> Option<String> {
        let dir = self.dir_for(key);
        let text = std::fs::read_to_string(dir.join("source")).ok()?;
        touch(&dir.join("meta.txt"));
        Some(text)
    }

    /// 写缓存：先写临时文件再 `rename`，避免读到半截文件。
    ///
    /// `meta.txt` 逐行 `字段=值`：`doctor --why` 直接拿它与当前键逐项比对，不需要再解析
    /// 一个为哈希定制的单行格式。
    pub fn put(&self, key: &CacheKey, source: &str) -> std::io::Result<PathBuf> {
        let dir = self.dir_for(key);
        std::fs::create_dir_all(&dir)?;
        let temp = temp_path(&dir, "source");
        let final_path = dir.join("source");
        std::fs::write(&temp, source)?;
        let meta: String = key
            .fields()
            .into_iter()
            .map(|(name, value)| format!("{name}={value}\n"))
            .collect();
        std::fs::write(dir.join("meta.txt"), meta)?;
        std::fs::rename(&temp, &final_path)?;
        Ok(final_path)
    }

    /// 写一份不来自算子声明的程序（见证、探测）：条目按程序文本定名，`source` 就是程序本身。
    ///
    /// 条目名只由标签与程序文本决定，编译器不在其中：编译器是**产物**身份的一部分
    /// （[`ArtifactKey`]），不是源文本身份的一部分——同一份程序的 `cc` 产物与 `clang` 产物
    /// 共用一个条目、各占 `bin/` 下一份。返回条目键，它与 [`CacheKey::hex`] 同一形状，于是
    /// 扫描、LRU 回收与 [`Self::entry`] 都不需要第二套规则——见证的编译产物因此和算子的产物
    /// 一样受容量管理，而不是缓存根下一个扫不到也删不掉的角落。`meta.txt` 记的是"这份程序
    /// 是什么"，不是算子指纹。
    ///
    /// # Errors
    ///
    /// 目录不可建或文件不可写时返回系统错误（调用方给出带路径的说明）。
    pub fn put_program(&self, label: &str, program: &str) -> std::io::Result<String> {
        let hex = format!("{:016x}", fnv1a(&format!("{label};{program}")));
        let dir = self.entry_dir(&hex);
        std::fs::create_dir_all(&dir)?;
        let temp = temp_path(&dir, "program");
        std::fs::write(&temp, program)?;
        let meta = format!("program={label}\nprogram_hash={:016x}\n", fnv1a(program));
        std::fs::write(dir.join("meta.txt"), meta)?;
        std::fs::rename(&temp, dir.join("source"))?;
        Ok(hex)
    }

    /// 全部目录（含半截落盘的残留）。
    ///
    /// 扫描只认目录名是 16 位十六进制的那一层：那是 `key.hex()` 的形状，其它目录（别人在
    /// 同一个缓存根下放的东西）不归这里管，也不该被回收。读不到的子目录直接跳过。
    #[must_use]
    pub fn entries(&self) -> Vec<Entry> {
        let mut entries = Vec::new();
        let Ok(shards) = std::fs::read_dir(&self.root) else {
            return entries;
        };
        for shard in shards.flatten() {
            if !is_dir(&shard.path()) {
                continue;
            }
            let Ok(dirs) = std::fs::read_dir(shard.path()) else {
                continue;
            };
            for dir in dirs.flatten() {
                if let Some(entry) = self.read_entry(&dir.path()) {
                    entries.push(entry);
                }
            }
        }
        entries
    }

    /// 按十六进制键查一条记录；键不是 16 位十六进制时返回 `None`。
    #[must_use]
    pub fn entry(&self, hex: &str) -> Option<Entry> {
        let hex = hex.trim().to_ascii_lowercase();
        if !is_key(&hex) {
            return None;
        }
        self.read_entry(&self.root.join(&hex[..2]).join(&hex))
    }

    /// 回收：先清残留，再按 LRU 删到容量上限以内。
    ///
    /// 两件事分开：残留（没有 `source` 的目录）本来就不会命中，所以与上限无关、一律清掉——
    /// 它们只会一直占着磁盘，等不到任何一次「超限」。可复用条目才按 LRU 排序并删到上限内。
    ///
    /// 仍在用的（[`Entry::building`]）一律跳过：删掉它可能把另一个进程正要落盘的产物变成
    /// 半截目录，而它自己马上就要写好。跳过的数目在结果里如实计数。
    ///
    /// # Errors
    ///
    /// 删除某个目录失败时返回带路径的说明。
    pub fn gc(&self, limit: u64) -> diagnostic::Result<Gc> {
        let now = SystemTime::now();
        let entries = self.entries();
        let before: u64 = entries.iter().map(|entry| entry.bytes).sum();
        let mut after = before;
        let mut removed = 0;
        let mut residue = Residue::default();
        let mut skipped = 0;
        let mut complete = Vec::new();
        for entry in entries {
            if entry.complete() {
                complete.push(entry);
                continue;
            }
            if entry.building(now) {
                skipped += 1;
                continue;
            }
            remove(&entry.dir)?;
            after -= entry.bytes;
            residue.items += 1;
            residue.bytes += entry.bytes;
        }
        complete.sort_by_key(|entry| entry.used);
        for entry in complete {
            if after <= limit {
                break;
            }
            if entry.building(now) {
                skipped += 1;
                continue;
            }
            remove(&entry.dir)?;
            after -= entry.bytes;
            removed += 1;
        }
        Ok(Gc {
            limit,
            before,
            after,
            removed,
            residue,
            skipped,
        })
    }

    /// 删除一个陈旧锁：持有者已经崩溃，没人能来释放它。
    ///
    /// 返回是否真的删掉了一份陈旧锁（删失败也返回 `false`：多半是别人先删了）。
    /// 活着的锁（[`STALE_LOCK`] 内创建）一律不动，等待方继续等。
    pub fn break_stale_lock(&self, key: &CacheKey) -> bool {
        let path = self.dir_for(key).join("build.lock");
        let Ok(meta) = std::fs::metadata(&path) else {
            return false;
        };
        let stale = meta
            .modified()
            .map(|time| SystemTime::now().duration_since(time).unwrap_or_default() >= STALE_LOCK)
            .unwrap_or(true);
        stale && std::fs::remove_file(&path).is_ok()
    }

    /// 读一条目录：`bytes` 覆盖整套文件（含 `source.tmp`、残留的锁与 `bin/` 下的产物），
    /// `used` 取最新的时间戳。
    ///
    /// 目录要递归算：编译产物在 `bin/` 里，漏掉它就等于 GC 看着 2 GiB 上限却删不掉真正
    /// 占空间的东西。没有 `source` 的目录同样是记录（[`State::Partial`]），否则它们永远不会
    /// 被扇到。
    fn read_entry(&self, dir: &Path) -> Option<Entry> {
        let hex = dir.file_name()?.to_str()?.to_string();
        if !is_key(&hex) {
            return None;
        }
        let mut bytes = 0;
        let mut used = None;
        let mut complete = false;
        let mut locked = false;
        let mut pending = vec![dir.to_path_buf()];
        while let Some(path) = pending.pop() {
            for file in std::fs::read_dir(&path).ok()?.flatten() {
                let Ok(meta) = file.metadata() else {
                    continue;
                };
                if meta.is_dir() {
                    pending.push(file.path());
                    continue;
                }
                bytes += meta.len();
                if let Ok(time) = meta.modified() {
                    used = Some(used.map_or(time, |old: SystemTime| old.max(time)));
                }
                if path == dir {
                    match file.file_name().to_str() {
                        Some("source") => complete = true,
                        Some("build.lock") => locked = true,
                        _ => {}
                    }
                }
            }
        }
        // 空目录没有任何文件时间戳：退回目录自身的时间（删旧的空目录仍要靠它排 LRU）。
        let used = used
            .or_else(|| std::fs::metadata(dir).ok()?.modified().ok())
            .unwrap_or(std::time::UNIX_EPOCH);
        let fields = std::fs::read_to_string(dir.join("meta.txt"))
            .map(|text| parse_fields(&text))
            .unwrap_or_default();
        Some(Entry {
            hex,
            dir: dir.to_path_buf(),
            bytes,
            used,
            fields,
            state: if complete {
                State::Complete
            } else {
                State::Partial
            },
            locked,
            artifacts: artifacts(dir),
        })
    }

    /// 取锁（`O_EXCL` 语义）；调用方 `drop` 即释放。
    pub fn lock(&self, key: &CacheKey) -> std::io::Result<Lock> {
        let dir = self.dir_for(key);
        std::fs::create_dir_all(&dir)?;
        let path = dir.join("build.lock");
        std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&path)?;
        Ok(Lock { path, held: true })
    }
}

fn is_dir(path: &Path) -> bool {
    std::fs::metadata(path)
        .map(|meta| meta.is_dir())
        .unwrap_or(false)
}

/// 目录名是否是 `key.hex()` 的形状（16 位十六进制）：缓存只回收这些目录。
fn is_key(hex: &str) -> bool {
    hex.len() == 16 && hex.bytes().all(|byte| byte.is_ascii_hexdigit())
}

/// 条目里已落盘的编译产物个数（`bin/` 下以产物键命名的文件）。
///
/// 写到一半的临时名（`.…-<pid>`）不算：它们不是可复用的产物，只会在 GC 里算作字节。
fn artifacts(dir: &Path) -> usize {
    std::fs::read_dir(dir.join("bin"))
        .map(|files| {
            files
                .flatten()
                .filter(|file| file.path().is_file())
                .filter(|file| file.file_name().to_str().is_some_and(is_key))
                .count()
        })
        .unwrap_or(0)
}

/// 删一个缓存目录；失败时带上路径。
fn remove(dir: &Path) -> diagnostic::Result<()> {
    std::fs::remove_dir_all(dir).map_err(|error| {
        Error::new(
            Code::Io,
            format!("删除缓存 {} 失败：{error}", dir.display()),
        )
    })
}

/// 原子落盘用的临时名：同一个目录可能被多个线程同时写（同一份 key 的两个并发构建），
/// 名字必须唯一——否则先 `rename` 的那个会把后来者的临时文件搬走，后者拿到 `ENOENT`。
fn temp_path(dir: &Path, tag: &str) -> PathBuf {
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let seq = NEXT.fetch_add(1, Ordering::Relaxed);
    dir.join(format!(".{tag}-{}-{seq}.tmp", std::process::id()))
}

/// 更新修改时间（时间戳是 GC 的 LRU 依据）；只读文件系统上失败也不影响命中。
fn touch(path: &Path) {
    if let Ok(file) = std::fs::OpenOptions::new().write(true).open(path) {
        let _ = file.set_modified(SystemTime::now());
    }
}

/// 解析 `meta.txt`：兼容逐行 `字段=值`（当前写法）与分号分隔的单行（旧写法）。
fn parse_fields(text: &str) -> Vec<(String, String)> {
    text.split(['\n', ';'])
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .filter_map(|line| line.split_once('='))
        .map(|(name, value)| (name.trim().to_string(), value.trim().to_string()))
        .collect()
}

/// 缓存锁；`drop` 时删除锁文件。
#[derive(Debug)]
pub struct Lock {
    path: PathBuf,
    held: bool,
}

impl Drop for Lock {
    fn drop(&mut self) {
        if self.held {
            let _ = std::fs::remove_file(&self.path);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::atom::DType;

    fn key() -> CacheKey {
        CacheKey {
            schema_ver: 1,
            emit: Emit::C,
            flags_hash: 0x1234,
            arch: "aarch64".to_string(),
            ast_hash: 0xabcd,
            fragments: vec![(Prim::Dot, 1), (Prim::Load, 2)],
            layouts: vec![("w".to_string(), "row".to_string())],
            variant: "pipe2".to_string(),
            tile: "GMAX=32".to_string(),
            dtype: DTypeSet::of([DType::F32]),
        }
    }

    #[test]
    fn identical_inputs_share_a_hash() {
        assert_eq!(key().hash(), key().hash());
        assert_eq!(key().hex().len(), 16);
    }

    #[test]
    fn changing_any_field_changes_the_hash() {
        let mut other = key();
        other.tile = "GMAX=64".to_string();
        assert_ne!(key().hash(), other.hash());

        let mut other = key();
        other.variant = "direct".to_string();
        assert_ne!(key().hash(), other.hash());
    }

    #[test]
    fn the_source_key_ignores_the_archive_tool_registration() {
        // 工具链不属于源文本：它的身份在 [`ArtifactKey`] 里，而且是实测值。这条断言盯的是
        // "换编译器会不会把源文本缓存也弄失效"——不会，也不该。
        let names: Vec<&str> = key().fields().into_iter().map(|(name, _)| name).collect();
        assert!(!names.contains(&"tool"), "{names:?}");
        assert!(!names.contains(&"tool_ver"), "{names:?}");
    }

    #[test]
    fn the_artifact_identity_carries_the_measured_toolchain() {
        let cc17 = ArtifactKey::keyed("abc", "cc@17.0.0", 0, "main");
        let cc21 = ArtifactKey::keyed("abc", "cc@21.0.0", 0, "main");
        assert_ne!(cc17, cc21, "换编译器版本必须换一份产物");
        assert_ne!(cc17.hex(), cc21.hex());
        assert_eq!(cc17, ArtifactKey::parse(&cc17.describe()).expect("可还原"));
    }

    /// 编译参数也决定产物：同一份见证在 `-O0` 与 `-O3` 下是两份不同的代码生成。
    ///
    /// 少了这一项，改一次档案的 `[compile].flags` 会静默复用旧参数的二进制，而「这条见证
    /// 跑通过」是对着一份不再会被编出来的程序成立的。
    #[test]
    fn the_artifact_identity_carries_the_compile_parameters() {
        let o3 = ArtifactKey::keyed(
            "abc",
            "cc@21.0.0",
            ArtifactKey::params(&args(&["-O3"])),
            "main",
        );
        let o0 = ArtifactKey::keyed(
            "abc",
            "cc@21.0.0",
            ArtifactKey::params(&args(&["-O0"])),
            "main",
        );
        assert_ne!(o3, o0, "换编译参数必须换一份产物");
        assert_ne!(o3.hex(), o0.hex());
        assert_eq!(o3, ArtifactKey::parse(&o3.describe()).expect("可还原"));
        // 参数顺序是命令行的一部分：不排序，`-I a -I b` 与 `-I b -I a` 不是同一行。
        assert_ne!(
            ArtifactKey::params(&args(&["-I", "a", "-I", "b"])),
            ArtifactKey::params(&args(&["-I", "b", "-I", "a"]))
        );
    }

    /// 缺 `params=` 的产物不认：它是在参数还没进身份的那一轮编出来的，来历无从查起。
    #[test]
    fn an_artifact_without_parameters_has_no_identity() {
        let text = "source=abc;tool=cc@21.0.0;harness=0000000000000001";
        assert!(ArtifactKey::parse(text).is_none(), "{text}");
    }

    /// 参数列表的字面量便利构造。
    fn args(values: &[&str]) -> Vec<String> {
        values.iter().map(|value| (*value).to_string()).collect()
    }

    #[test]
    fn fragment_order_does_not_change_the_hash() {
        let mut other = key();
        other.fragments.reverse();
        assert_eq!(key().hash(), other.hash());
    }

    #[test]
    fn put_and_get_roundtrip() {
        let root = std::env::temp_dir().join(format!("cutile-cache-{}", std::process::id()));
        let cache = Cache::new(&root);
        let key = key();
        assert!(cache.get(&key).is_none());
        cache.put(&key, "int x;\n").expect("写缓存");
        assert_eq!(cache.get(&key), Some("int x;\n".to_string()));
        let _ = std::fs::remove_dir_all(&root);
    }

    fn root(name: &str) -> std::path::PathBuf {
        let root = std::env::temp_dir().join(format!("cutile-cache-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        root
    }

    /// 把条目的时间戳定住：GC 的 LRU 与陈旧判据都靠它，靠测试自己 `put` 的先后顺序测不出来。
    fn stamp(cache: &Cache, key: &CacheKey, secs: u64) {
        age(&cache.dir_for(key), secs);
    }

    /// 把一个目录里所有文件的时间戳退到 `secs`（含锁与半截文件：陈旧判据看的是最新的那个）。
    fn age(dir: &Path, secs: u64) {
        for file in std::fs::read_dir(dir).expect("目录可读").flatten() {
            let file = std::fs::OpenOptions::new()
                .write(true)
                .open(file.path())
                .expect("缓存文件可写");
            file.set_modified(std::time::UNIX_EPOCH + std::time::Duration::from_secs(secs))
                .expect("可设置时间戳");
        }
    }

    fn stored(key: &CacheKey) -> Vec<(String, String)> {
        key.fields()
            .into_iter()
            .map(|(name, value)| (name.to_string(), value))
            .collect()
    }

    #[test]
    fn meta_keeps_the_key_field_by_field() {
        let root = root("meta");
        let cache = Cache::new(&root);
        let key = key();
        cache.put(&key, "int x;\n").expect("写缓存");
        let entry = cache.entry(&key.hex()).expect("有记录");
        assert_eq!(entry.fields.len(), key.fields().len(), "{entry:?}");
        assert!(
            entry
                .fields
                .iter()
                .any(|(name, value)| name == "variant" && value == "pipe2"),
            "{entry:?}"
        );
        assert!(key.changes(&entry.fields).is_empty(), "同一份键不应有差异");
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn changes_name_every_field_that_differs() {
        let mut other = key();
        other.variant = "direct".to_string();
        other.tile = "GMAX=64".to_string();
        let changes = other.changes(&stored(&key()));
        let fields: Vec<&str> = changes.iter().map(|change| change.field.as_str()).collect();
        assert_eq!(fields, ["variant", "tile"], "{changes:?}");
        assert_eq!(changes[0].stored.as_deref(), Some("pipe2"));
        assert_eq!(changes[0].current.as_deref(), Some("direct"));
        assert!(
            other.changes(&stored(&other)).is_empty(),
            "自己与自己不存在差异"
        );
    }

    #[test]
    fn gc_evicts_the_least_recently_used_first() {
        let root = root("gc");
        let cache = Cache::new(&root);
        let mut keys = Vec::new();
        for (index, secs) in [100u64, 200, 300].into_iter().enumerate() {
            let mut key = key();
            key.variant = format!("v{index}");
            cache.put(&key, "int x;\n").expect("写缓存");
            stamp(&cache, &key, secs);
            keys.push(key);
        }
        let oldest = cache.entry(&keys[0].hex()).expect("有记录").bytes;
        let total: u64 = cache.entries().iter().map(|entry| entry.bytes).sum::<u64>();
        let report = cache.gc(total - oldest).expect("可回收");
        assert_eq!(report.removed, 1, "{report:?}");
        assert!(report.after <= report.limit, "{report:?}");
        assert!(cache.get(&keys[0]).is_none(), "最久未使用的被删");
        assert!(cache.get(&keys[1]).is_some(), "较新的保留");
        assert!(cache.get(&keys[2]).is_some(), "最新的保留");
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn gc_skips_a_key_that_is_being_built() {
        let root = root("gc-lock");
        let cache = Cache::new(&root);
        let key = key();
        cache.put(&key, "int x;\n").expect("写缓存");
        let lock = cache.lock(&key).expect("取锁");
        let report = cache.gc(0).expect("可回收");
        assert_eq!(report.removed, 0, "{report:?}");
        assert_eq!(report.skipped, 1, "{report:?}");
        assert!(report.after > 0, "锁着的条目不能算作已回收：{report:?}");
        drop(lock);
        let report = cache.gc(0).expect("可回收");
        assert_eq!(report.removed, 1, "{report:?}");
        assert_eq!(report.after, 0, "{report:?}");
        assert!(cache.get(&key).is_none());
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn a_half_written_directory_is_residue_not_a_hit() {
        // 写到一半就没了（被 kill、断点停在 put 中间）的目录没有 `source`：它永远不该命中，
        // 但它确实存在、确实占磁盘，所以它必须是一条记录而不是被扫描跳过。
        let root = root("residue");
        let cache = Cache::new(&root);
        let key = key();
        let dir = cache.dir_for(&key);
        std::fs::create_dir_all(&dir).expect("建目录");
        std::fs::write(dir.join("source.tmp"), "int x;\n").expect("半截落盘");
        let entry = cache.entry(&key.hex()).expect("半截目录也是记录");
        assert_eq!(entry.state, State::Partial, "{entry:?}");
        assert!(!entry.complete(), "{entry:?}");
        assert!(entry.bytes > 0, "半截文件也算占用：{entry:?}");
        assert!(cache.get(&key).is_none(), "没有 source 不能命中");
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn gc_reclaims_residue_even_below_the_limit() {
        // 残留不是“空间不够”的产物：上限调多大都不该让它留下来。
        let root = root("gc-residue");
        let cache = Cache::new(&root);
        let key = key();
        cache.put(&key, "int x;\n").expect("写缓存");
        let mut half = key.clone();
        half.variant = "direct".to_string();
        let dir = cache.dir_for(&half);
        std::fs::create_dir_all(&dir).expect("建目录");
        std::fs::write(dir.join("source.tmp"), "int y;\n").expect("半截落盘");

        // 刚写了一半的不动：那可能是另一个进程正在落盘。
        let report = cache.gc(u64::MAX).expect("可回收");
        assert_eq!(report.residue.items, 0, "{report:?}");
        assert_eq!(report.skipped, 1, "{report:?}");

        age(&dir, 100);
        let before: u64 = cache.entries().iter().map(|entry| entry.bytes).sum();
        let report = cache.gc(u64::MAX).expect("可回收");
        assert_eq!(report.residue.items, 1, "{report:?}");
        assert!(report.residue.bytes > 0, "{report:?}");
        assert_eq!(report.removed, 0, "上限很大，不该动可复用条目：{report:?}");
        assert!(report.after < before, "{report:?}");
        assert!(cache.get(&key).is_some(), "完整条目保留");
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn a_crashed_lock_does_not_shield_a_record_forever() {
        // `O_EXCL` 锁是文件：持有者崩溃后没人释放它，旧实现会让这个 key 永远回收不掉。
        let root = root("gc-stale-lock");
        let cache = Cache::new(&root);
        let key = key();
        cache.put(&key, "int x;\n").expect("写缓存");
        let dir = cache.dir_for(&key);
        std::fs::write(dir.join("build.lock"), "").expect("模拟崩溃留下的锁");
        age(&dir, 100);
        let report = cache.gc(0).expect("可回收");
        assert_eq!(report.removed, 1, "陈旧锁不是“正在构建”：{report:?}");
        assert_eq!(report.skipped, 0, "{report:?}");
        assert!(cache.get(&key).is_none());
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn break_stale_lock_leaves_a_live_lock_alone() {
        let root = root("break-lock");
        let cache = Cache::new(&root);
        let key = key();
        let lock = cache.lock(&key).expect("取锁");
        assert!(!cache.break_stale_lock(&key), "活着的锁不能抢");
        assert!(cache.dir_for(&key).join("build.lock").exists());
        drop(lock);

        let lock = cache.lock(&key).expect("释放后可再次取锁");
        age(&cache.dir_for(&key), 100);
        assert!(cache.break_stale_lock(&key), "陈旧锁要能删掉");
        assert!(!cache.dir_for(&key).join("build.lock").exists());
        drop(lock);
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn entry_rejects_a_key_that_is_not_hexadecimal() {
        let root = root("entry");
        let cache = Cache::new(&root);
        assert!(cache.entry("not-a-key").is_none());
        assert!(cache.entry("83a3ecdb31cd122").is_none(), "少一位也不行");
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn an_artifact_lives_in_its_entry_and_is_counted_by_the_gc() {
        let root = root("artifact");
        let cache = Cache::new(&root);
        let key = key();
        cache.put(&key, "int x;\n").expect("写缓存");
        let built = root.join("exe");
        std::fs::create_dir_all(&root).expect("临时目录");
        std::fs::write(&built, vec![0u8; 4096]).expect("写假产物");

        let artifact_key = ArtifactKey::new(&key, "cc", "int main(void){return 0;}\n");
        assert!(cache.artifact(&artifact_key).is_none(), "还没编译过");
        let artifact = cache.put_artifact(&artifact_key, &built).expect("落产物");
        assert!(!artifact.reused, "{artifact:?}");
        assert!(
            artifact.path.starts_with(cache.dir_for(&key)),
            "产物必须落在它自己的条目里：{artifact:?}"
        );

        let hit = cache.artifact(&artifact_key).expect("命中");
        assert!(hit.reused, "{hit:?}");
        assert_eq!(hit.bytes, 4096, "{hit:?}");

        let entry = cache.entry(&key.hex()).expect("有记录");
        assert_eq!(entry.artifacts, 1, "{entry:?}");
        assert!(
            entry.bytes >= 4096,
            "产物必须计入占用，否则 GC 删不掉真正占空间的东西：{entry:?}"
        );
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn an_artifact_records_who_compiled_it() {
        // 身份写在产物旁边：没有编译器的机器靠它把产物认回来，而不是重新猜一个文件名。
        let root = root("artifact-identity");
        let cache = Cache::new(&root);
        let key = key();
        cache.put(&key, "int x;\n").expect("写缓存");
        let built = root.join("exe");
        std::fs::create_dir_all(&root).expect("临时目录");
        std::fs::write(&built, vec![0u8; 16]).expect("写假产物");
        let harness = "int main(void){return 0;}\n";

        let cc = cache
            .put_artifact(&ArtifactKey::new(&key, "cc", harness), &built)
            .expect("落产物");
        let listed = cache.artifacts_in(&key.hex());
        assert_eq!(listed.len(), 1, "{listed:?}");
        assert_eq!(listed[0].key, cc.key);
        assert!(listed[0].reused, "{listed:?}");
        assert_eq!(listed[0].bytes, 16, "{listed:?}");

        cache
            .put_artifact(&ArtifactKey::new(&key, "clang", harness), &built)
            .expect("落另一份");
        let mut tools: Vec<String> = cache
            .artifacts_in(&key.hex())
            .into_iter()
            .map(|artifact| artifact.key.tool)
            .collect();
        tools.sort();
        assert_eq!(
            tools,
            vec![String::from("cc"), String::from("clang")],
            "身份里带着编译器名"
        );
        assert_eq!(
            cache.entry(&key.hex()).expect("有记录").artifacts,
            2,
            "身份文件不算第二份产物"
        );

        std::fs::remove_file(&cc.path).expect("删掉可执行文件");
        let left: Vec<String> = cache
            .artifacts_in(&key.hex())
            .into_iter()
            .map(|artifact| artifact.key.tool)
            .collect();
        assert_eq!(
            left,
            vec![String::from("clang")],
            "只有身份文件、可执行文件已不在的不算产物"
        );
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn a_different_harness_compiler_or_dtype_is_a_different_artifact() {
        let key = key();
        let harness = "int main(void){return 0;}\n";
        let other = "int main(void){return 1;}\n";
        assert_eq!(
            ArtifactKey::new(&key, "cc", harness).hex(),
            ArtifactKey::new(&key, "cc", harness).hex(),
            "同一份输入必须得到同一份产物"
        );
        assert_ne!(
            ArtifactKey::new(&key, "cc", harness).hex(),
            ArtifactKey::new(&key, "clang", harness).hex(),
            "换个编译器就换了代码生成，不能复用"
        );
        assert_ne!(
            ArtifactKey::new(&key, "cc", harness).hex(),
            ArtifactKey::new(&key, "cc", other).hex(),
            "形状与样本都在主函数里，换一份就不能复用"
        );
        let mut retyped = key.clone();
        retyped.dtype = DTypeSet::of([DType::U8]);
        assert_ne!(
            ArtifactKey::new(&key, "cc", harness).hex(),
            ArtifactKey::new(&retyped, "cc", harness).hex(),
            "输出精度变了，源文本与产物都必须换键"
        );
    }

    #[test]
    fn lock_is_exclusive_until_dropped() {
        let root = std::env::temp_dir().join(format!("cutile-lock-{}", std::process::id()));
        let cache = Cache::new(&root);
        let key = key();
        let lock = cache.lock(&key).expect("第一次取锁");
        assert!(cache.lock(&key).is_err(), "同一 key 不能并发编译");
        drop(lock);
        assert!(cache.lock(&key).is_ok(), "释放后可再次取锁");
        let _ = std::fs::remove_dir_all(&root);
    }
}
