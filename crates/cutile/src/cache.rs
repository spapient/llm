//! 职责：编译产物缓存的键、锁与落盘。

use crate::atom::{DTypeSet, Prim};
use crate::emit::fnv1a;
use crate::lru::{Tally, directory_time, evict, reclaim};
use crate::prof::Emit;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, SystemTime};

pub use crate::lru::{Entry, Gc, Residue, STALE_LOCK, State};

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
    /// 由配置层解析出的缓存目录构造：来源由 `cutile::dirs` 决定。
    pub fn from_dirs(dirs: &crate::dirs::Dirs) -> Self {
        Self::new(dirs.cache())
    }
    /// 某个 key 的目录：`<root>/<前两位>/<hex>`。
    pub fn dir_for(&self, key: &CacheKey) -> PathBuf {
        self.entry_dir(&key.hex())
    }
    fn entry_dir(&self, hex: &str) -> PathBuf {
        self.root.join(hex.get(..2).unwrap_or(hex)).join(hex)
    }
    /// 一份产物的路径：`<条目>/bin/<产物 hex>`。
    #[must_use]
    pub fn artifact_path(&self, key: &ArtifactKey) -> PathBuf {
        self.entry_dir(&key.source).join("bin").join(key.hex())
    }
    /// 读产物：命中即返回，并顺手刷新条目的时间戳（LRU 靠它，见 [`Self::get`]）。
    #[must_use]
    pub fn artifact(&self, key: &ArtifactKey) -> Option<Artifact> {
        let path = self.artifact_path(key);
        let bytes = std::fs::metadata(&path).ok()?.len();
        touch(&self.entry_dir(&key.source).join("meta.txt"));
        Some(Artifact::new(key.clone(), path, bytes, true))
    }
    /// 条目里已落盘的产物：身份在产物旁的 `bin/<hex>.key`，无身份文件或可执行文件已不在的都不算。
    #[must_use]
    pub fn artifacts_in(&self, source: &str) -> Vec<Artifact> {
        let bin = self.entry_dir(source).join("bin");
        let files = std::fs::read_dir(&bin).into_iter().flatten().flatten();
        files.filter_map(|file| artifact_at(&bin, &file)).collect()
    }
    /// 落一份产物：可执行文件与身份文件都先写到同目录临时名再 `rename`（`copy` 保留可执行位），并发写同一个键不会读到半截文件。
    pub fn put_artifact(&self, key: &ArtifactKey, exe: &Path) -> std::io::Result<Artifact> {
        let path = self.artifact_path(key);
        let bin = path.parent().unwrap_or(&path).to_path_buf();
        std::fs::create_dir_all(&bin)?;
        let staged = temp_path(&bin, &key.hex());
        std::fs::copy(exe, &staged)?;
        std::fs::rename(&staged, &path)?;
        sign(&bin, key)?;
        let bytes = std::fs::metadata(&path).map_or(0, |meta| meta.len());
        touch(&self.entry_dir(&key.source).join("meta.txt"));
        Ok(Artifact::new(key.clone(), path, bytes, false))
    }
    /// 读缓存；命中就顺手刷新 `meta.txt` 时间戳，否则 GC 的 LRU 会退化成"最近写入"。
    pub fn get(&self, key: &CacheKey) -> Option<String> {
        let dir = self.dir_for(key);
        let text = std::fs::read_to_string(dir.join("source")).ok()?;
        touch(&dir.join("meta.txt"));
        Some(text)
    }
    /// 写缓存：先写临时文件再 `rename`（避免半截文件）；`meta.txt` 逐行 `字段=值`，便于 `doctor --why` 比对。
    pub fn put(&self, key: &CacheKey, source: &str) -> std::io::Result<PathBuf> {
        let dir = self.dir_for(key);
        let meta = format!("{}\n", key.describe().replace(';', "\n"));
        self.write_source(&dir, "source", source, &meta)?;
        Ok(dir.join("source"))
    }
    /// 写一份不来自算子声明的程序（见证、探测）：条目按程序文本定名，返回的键与 [`CacheKey::hex`] 同形。
    pub fn put_program(&self, label: &str, program: &str) -> std::io::Result<String> {
        let hex = format!("{:016x}", fnv1a(&format!("{label};{program}")));
        let dir = self.entry_dir(&hex);
        let hash = format!("program={label}\nprogram_hash={:016x}\n", fnv1a(program));
        self.write_source(&dir, "program", program, &hash)?;
        Ok(hex)
    }
    /// 原子落一份条目：先写临时名再 `rename`，读路径只拿到完整旧份或新份；两份写入共用这一条。
    fn write_source(
        &self,
        dir: &Path,
        temp_tag: &str,
        source: &str,
        meta: &str,
    ) -> std::io::Result<()> {
        std::fs::create_dir_all(dir)?;
        let temp = temp_path(dir, temp_tag);
        std::fs::write(&temp, source)?;
        std::fs::write(dir.join("meta.txt"), meta)?;
        std::fs::rename(&temp, dir.join("source"))?;
        Ok(())
    }
    /// 回收：先清残留，再按 LRU 删到上限以内；仍在用的（[`Entry::building`]）跳过——删它会把别人的产物变半截。
    pub fn gc(&self, limit: u64) -> diagnostic::Result<Gc> {
        let now = SystemTime::now();
        let entries = self.entries();
        let before: u64 = entries.iter().map(|e| e.bytes).sum();
        let mut gc = Gc {
            limit,
            before,
            after: before,
            ..Default::default()
        };
        reclaim(&entries, now, &mut gc)?;
        evict(&entries, now, &mut gc)?;
        Ok(gc)
    }
    /// 产物锁路径：与产物同目录、按产物键定名；同一份产物在同一时刻只允许一个写入者。
    pub fn artifact_lock_path(&self, key: &ArtifactKey) -> PathBuf {
        self.artifact_path(key)
            .with_file_name(format!("{}.build.lock", key.hex()))
    }
    /// 删陈旧锁：持有者已崩溃，没人能来释放它（删失败也返回 `false`，多半别人先删了）；活着的锁不动。
    pub fn break_stale_lock(&self, key: &CacheKey) -> bool {
        self.break_stale_lock_at(&self.dir_for(key).join("build.lock"))
    }
    /// 删陈旧锁（按锁路径）：见 [`Self::break_stale_lock`]。
    pub fn break_stale_lock_at(&self, path: &Path) -> bool {
        let stale = match std::fs::metadata(path).and_then(|meta| meta.modified()) {
            Ok(time) => SystemTime::now().duration_since(time).unwrap_or_default() >= STALE_LOCK,
            // 读不到时间戳只可能是锁已经不在了：`remove_file` 会失败，结果仍是 `false`。
            Err(_) => true,
        };
        stale && std::fs::remove_file(path).is_ok()
    }
    /// 取锁（`O_EXCL` 语义）；调用方 `drop` 即释放。
    pub fn lock(&self, key: &CacheKey) -> std::io::Result<Lock> {
        self.lock_at(&self.dir_for(key).join("build.lock"))
    }
    /// 取锁（按锁路径）：见 [`Self::lock`]。
    pub fn lock_at(&self, path: &Path) -> std::io::Result<Lock> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::File::create_new(path)?;
        Ok(Lock {
            path: path.to_path_buf(),
        })
    }
    /// 全部目录（含半截落盘的残留）。
    #[must_use]
    pub fn entries(&self) -> Vec<Entry> {
        self.shards()
            .flat_map(|shard| self.entries_in(&shard).collect::<Vec<_>>())
            .collect()
    }
    /// 按十六进制键查一条记录；键不是 16 位十六进制时返回 `None`。
    #[must_use]
    pub fn entry(&self, hex: &str) -> Option<Entry> {
        let hex = hex.trim().to_ascii_lowercase();
        let key = is_key(&hex).then_some(hex)?;
        self.read_entry(&self.root.join(&key[..2]).join(key))
    }
    fn shards(&self) -> impl Iterator<Item = PathBuf> + '_ {
        let dirs = std::fs::read_dir(&self.root).into_iter().flatten();
        dirs.flatten().map(|e| e.path()).filter(|p| p.is_dir())
    }
    fn entries_in(&self, shard: &Path) -> impl Iterator<Item = Entry> + '_ {
        let dirs = std::fs::read_dir(shard).into_iter().flatten();
        dirs.flatten().filter_map(|e| self.read_entry(&e.path()))
    }
    fn read_entry(&self, dir: &Path) -> Option<Entry> {
        let hex = dir.file_name()?.to_str().filter(|name| is_key(name))?;
        let tally = Tally::measure(dir)?;
        Some(Entry {
            artifacts: self.artifacts_in(hex).len(),
            fields: fields_of(dir),
            used: tally.used.unwrap_or_else(|| directory_time(dir)),
            hex: hex.to_string(),
            dir: dir.to_path_buf(),
            bytes: tally.bytes,
            state: tally.state,
            locked: tally.locked,
        })
    }
}

/// `cache gc`：按 LRU 回收到上限以内；`limit` 为 `None` 时用 [`CACHE_LIMIT`]。删除失败返回带路径的说明。
pub fn cache_gc(dirs: &crate::dirs::Dirs, limit: Option<u64>) -> diagnostic::Result<Gc> {
    Cache::from_dirs(dirs).gc(limit.unwrap_or(CACHE_LIMIT))
}

impl Artifact {
    fn new(key: ArtifactKey, path: PathBuf, bytes: u64, reused: bool) -> Self {
        Self {
            key,
            path,
            bytes,
            reused,
        }
    }
}

fn sign(bin: &Path, key: &ArtifactKey) -> std::io::Result<()> {
    let temp = temp_path(bin, "key");
    std::fs::write(&temp, format!("{}\n", key.describe()))?;
    std::fs::rename(&temp, bin.join(format!("{}.key", key.hex())))
}

fn artifact_at(bin: &Path, file: &std::fs::DirEntry) -> Option<Artifact> {
    let path = file.path();
    let name = file.file_name().into_string().ok().filter(|n| is_key(n))?;
    let text = std::fs::read_to_string(bin.join(format!("{name}.key"))).ok()?;
    let key = ArtifactKey::parse(text.trim())?;
    let bytes = std::fs::metadata(&path).ok()?.len();
    Some(Artifact::new(key, path, bytes, true))
}

/// 原子落盘用的临时名：多线程写同一目录时名字必须唯一，否则先 `rename` 的会把后来者的临时文件搬走。
fn temp_path(dir: &Path, tag: &str) -> PathBuf {
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let seq = NEXT.fetch_add(1, Ordering::Relaxed);
    dir.join(format!(".{tag}-{}-{seq}.tmp", std::process::id()))
}
/// LRU 时间戳刷新窗口：LRU 只要粗粒度先后，每次命中都写文件会把读路径拖成写路径。
const TOUCH_WINDOW: Duration = Duration::from_secs(5);

/// 更新修改时间（GC 的 LRU 依据）；只读文件系统上失败也不影响命中。
fn touch(path: &Path) {
    let Ok(last) = std::fs::metadata(path).and_then(|m| m.modified()) else {
        return;
    };
    let now = SystemTime::now();
    if now.duration_since(last).unwrap_or_default() < TOUCH_WINDOW {
        return;
    }
    if let Ok(file) = std::fs::OpenOptions::new().write(true).open(path) {
        let _ = file.set_modified(now);
    }
}
/// 缓存锁；`drop` 时删除锁文件。
#[derive(Debug)]
pub struct Lock {
    path: PathBuf,
}

impl Drop for Lock {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.path);
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

pub(super) fn is_key(hex: &str) -> bool {
    hex.len() == 16 && hex.bytes().all(|byte| byte.is_ascii_hexdigit())
}
fn fields_of(dir: &Path) -> Vec<(String, String)> {
    let text = std::fs::read_to_string(dir.join("meta.txt")).unwrap_or_default();
    text.split(['\n', ';'])
        .filter_map(|line| line.split_once('='))
        .map(|(name, value)| (name.trim().to_string(), value.trim().to_string()))
        .collect()
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
            let cached = by_name.get(*field).map(std::string::ToString::to_string);
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

/// 一个编译产物的身份：源文本 + 实测工具链（`Toolchain::id`（在 `shard`））+ 编译参数 + 主函数；缺一项都会复用成错误的产物。
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
        format!("{:016x}", fnv1a(&self.describe()))
    }
}

/// 缓存容量上限的兜底缺省值（README §10.3）：2 GiB。
pub const CACHE_LIMIT: u64 = 2 * 1024 * 1024 * 1024;

#[cfg(test)]
mod tests {
    //! `cutile::cache` 的用例：键、条目与淘汰。
    //!
    //! 生产文件只留执行与判据，用例集中于此。

    use super::*;
    use crate::atom::{DType, DTypeSet, Prim};
    use crate::prof::Emit;
    fn key() -> CacheKey {
        CacheKey {
            schema_ver: 1,
            emit: Emit::C,
            flags_hash: 0x1234,
            arch: "aarch64".to_string(),
            impl_name: "scalar".to_string(),
            block: "1".to_string(),
            ast_hash: 0xabcd,
            fragments: vec![(Prim::Dot, 1), (Prim::Load, 2)],
            layouts: vec![("w".to_string(), "row".to_string())],
            variant: "pipe2".to_string(),
            tile: "GMAX=32".to_string(),
            dtype: DTypeSet::of([DType::F32]),
        }
    }
    /// 一个独占的临时缓存根；`Drop` 负责删干净，`Deref` 让测试直接调 [`Cache`] 的方法。
    struct Fixture(PathBuf, Cache);
    impl Fixture {
        fn new(name: &str) -> Self {
            let root = std::env::temp_dir().join(format!("cutile-{name}-{}", std::process::id()));
            let _ = std::fs::remove_dir_all(&root);
            Self(root.clone(), Cache::new(root))
        }
        fn seeded(name: &str) -> (Self, CacheKey) {
            let cache = Self::new(name);
            let key = key();
            cache.put(&key, "int x;\n").expect("写缓存");
            (cache, key)
        }
    }
    impl std::ops::Deref for Fixture {
        type Target = Cache;
        fn deref(&self) -> &Cache {
            &self.1
        }
    }
    impl Drop for Fixture {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }
    /// 把条目的时间戳定住：GC 的 LRU 与陈旧判据都靠它，靠测试自己 `put` 的先后顺序测不出来。
    fn age(dir: &Path, secs: u64) {
        let when = std::time::UNIX_EPOCH + std::time::Duration::from_secs(secs);
        for file in std::fs::read_dir(dir).expect("目录可读").flatten() {
            let opened = std::fs::OpenOptions::new()
                .write(true)
                .open(file.path())
                .expect("可写");
            opened.set_modified(when).expect("可设时间戳");
        }
    }
    fn stored(key: &CacheKey) -> Vec<(String, String)> {
        key.fields()
            .into_iter()
            .map(|(n, v)| (n.to_string(), v))
            .collect()
    }
    fn tools(cache: &Cache, hex: &str) -> Vec<String> {
        let mut tools: Vec<_> = cache
            .artifacts_in(hex)
            .into_iter()
            .map(|a| a.key.tool)
            .collect();
        tools.sort();
        tools
    }
    testkit::test_cases! {
    fn identical_inputs_share_a_hash() {
        assert_eq!((key().hash(), key().hex().len()), (key().hash(), 16));
    }
    fn changing_any_field_changes_the_hash() {
        let mut other = key();
        other.tile = "GMAX=64".to_string();
        assert_ne!(key().hash(), other.hash());
        other.variant = "direct".to_string();
        assert_ne!(key().hash(), other.hash());
    }

    /// 内核身份字段（实现名 / 块宽）必须进指纹：同原语不同实现或不同块宽不能互相命中。
    fn kernel_identity_fields_change_the_hash() {
        let mut other = key();
        other.impl_name = "dp4a".to_string();
        assert_ne!(key().hash(), other.hash(), "换实现必须换键");
        other.impl_name = key().impl_name;
        other.block = "32".to_string();
        assert_ne!(key().hash(), other.hash(), "换块宽必须换键");
    }
    /// 工具链不属于源文本：它的身份在 [`ArtifactKey`] 里，而且是实测值。换编译器不该让源文本失效。
    fn the_source_key_ignores_the_archive_tool_registration() {
        let names: Vec<&str> = key().fields().into_iter().map(|(n, _)| n).collect();
        assert!(!names.contains(&"tools"), "{names:?}");
        assert!(!names.contains(&"tool_ver"), "{names:?}");
    }
    fn the_artifact_identity_carries_the_measured_toolchain() {
        let cc17 = ArtifactKey::keyed("abc", "cc@17.0.0", 0, "main");
        let cc21 = ArtifactKey::keyed("abc", "cc@21.0.0", 0, "main");
        assert_ne!(cc17, cc21, "换编译器版本必须换产物");
        assert_ne!(cc17.hex(), cc21.hex());
        assert_eq!(cc17, ArtifactKey::parse(&cc17.describe()).expect("可还原"));
    }
    /// 编译参数也决定产物：少了它，改一次 `[compile].flags` 会静默复用旧参数的二进制，而"这条见证跑通过"就对不上。
    fn the_artifact_identity_carries_the_compile_parameters() {
        let o3 = ArtifactKey::keyed(
            "abc",
            "cc",
            ArtifactKey::params(&["-O3".to_string()]),
            "main",
        );
        let o0 = ArtifactKey::keyed(
            "abc",
            "cc",
            ArtifactKey::params(&["-O0".to_string()]),
            "main",
        );
        assert_ne!(o3, o0, "换编译参数必须换产物");
        assert_ne!(o3.hex(), o0.hex());
        assert_eq!(o3, ArtifactKey::parse(&o3.describe()).expect("可还原"));
        // 参数顺序是命令行的一部分：不排序，`-I a -I b` 与 `-I b -I a` 不是同一行。
        let order = |flag: &str| ["-I".to_string(), flag.to_string()];
        assert_ne!(
            ArtifactKey::params(&order("a")),
            ArtifactKey::params(&order("b"))
        );
    }
    /// 缺 `params=` 的产物不认：它是在参数还没进身份的那一轮编出来的，来历无从查起。
    fn an_artifact_without_parameters_has_no_identity() {
        let text = "source=abc;tools=cc@21.0.0;harness=0000000000000001";
        assert!(ArtifactKey::parse(text).is_none(), "{text}");
    }
    fn fragment_order_does_not_change_the_hash() {
        let mut other = key();
        other.fragments.reverse();
        assert_eq!(key().hash(), other.hash());
    }
    fn put_and_get_roundtrip() {
        let cache = Fixture::new("roundtrip");
        assert!(cache.get(&key()).is_none());
        cache.put(&key(), "int x;\n").expect("写缓存");
        assert_eq!(cache.get(&key()), Some("int x;\n".to_string()));
    }
    fn meta_keeps_the_key_field_by_field() {
        let (cache, key) = Fixture::seeded("meta");
        let entry = cache.entry(&key.hex()).expect("有记录");
        assert_eq!(entry.fields.len(), key.fields().len(), "{entry:?}");
        let pipe2 = |(n, v): &(String, String)| n == "variant" && v == "pipe2";
        assert!(entry.fields.iter().any(pipe2), "{entry:?}");
        assert!(key.changes(&entry.fields).is_empty(), "同一份键没有差异");
    }
    fn changes_name_every_field_that_differs() {
        let mut other = key();
        other.variant = "direct".to_string();
        other.tile = "GMAX=64".to_string();
        let changes = other.changes(&stored(&key()));
        let fields: Vec<&str> = changes.iter().map(|c| c.field.as_str()).collect();
        assert_eq!(fields, ["variant", "tile"], "{changes:?}");
        assert_eq!(changes[0].stored.as_deref(), Some("pipe2"));
        assert_eq!(changes[0].current.as_deref(), Some("direct"));
        assert!(other.changes(&stored(&other)).is_empty(), "自比无差异");
    }
    fn gc_evicts_the_least_recently_used_first() {
        let cache = Fixture::new("gc");
        let mut keys = Vec::new();
        for (idx, secs) in [100u64, 200, 300].into_iter().enumerate() {
            let mut key = key();
            key.variant = format!("v{idx}");
            cache.put(&key, "int x;\n").expect("写缓存");
            age(&cache.dir_for(&key), secs);
            keys.push(key);
        }
        let oldest = cache.entry(&keys[0].hex()).expect("有记录").bytes;
        let total: u64 = cache.entries().iter().map(|e| e.bytes).sum();
        let report = cache.gc(total - oldest).expect("可回收");
        assert_eq!(report.removed, 1, "{report:?}");
        assert!(report.after <= report.limit, "{report:?}");
        let hits: Vec<bool> = keys.iter().map(|key| cache.get(key).is_some()).collect();
        assert_eq!(hits, [false, true, true], "最久未使用的先删");
    }
    fn gc_skips_a_key_that_is_being_built() {
        let (cache, key) = Fixture::seeded("gc-lock");
        let lock = cache.lock(&key).expect("取锁");
        let report = cache.gc(0).expect("可回收");
        assert_eq!((report.removed, report.skipped), (0, 1), "{report:?}");
        assert!(report.after > 0, "锁着的不算已回收：{report:?}");
        drop(lock);
        let report = cache.gc(0).expect("可回收");
        assert!(report.after == 0 && cache.get(&key).is_none(), "{report:?}");
    }
    fn a_half_written_directory_is_residue_not_a_hit() {
        // 写到一半被 kill 的目录没有 `source`：它永不命中，却确实占磁盘——必须是一条记录而不是跳过。
        let cache = Fixture::new("residue");
        let key = key();
        let dir = cache.dir_for(&key);
        std::fs::create_dir_all(&dir).expect("建目录");
        std::fs::write(dir.join("source.tmp"), "int x;\n").expect("半截落盘");
        let entry = cache.entry(&key.hex()).expect("也是记录");
        assert_eq!(entry.state, State::Partial, "{entry:?}");
        assert!(!entry.complete());
        assert!(entry.bytes > 0, "半截文件也算占用：{entry:?}");
        assert!(cache.get(&key).is_none(), "没有 source 不能命中");
    }
    fn gc_reclaims_residue_even_below_the_limit() {
        // 残留不是"空间不够"的产物：上限调多大都不该让它留下来。
        let (cache, key) = Fixture::seeded("gc-residue");
        let mut half = key.clone();
        half.variant = "direct".to_string();
        let dir = cache.dir_for(&half);
        std::fs::create_dir_all(&dir).expect("建目录");
        std::fs::write(dir.join("source.tmp"), "int y;\n").expect("半截落盘");
        // 刚写了一半的不动：那可能是另一个进程正在落盘。
        let report = cache.gc(u64::MAX).expect("可回收");
        assert_eq!((report.residue.items, report.skipped), (0, 1), "{report:?}");
        age(&dir, 100);
        let before: u64 = cache.entries().iter().map(|e| e.bytes).sum();
        let report = cache.gc(u64::MAX).expect("可回收");
        assert_eq!(report.residue.items, 1, "{report:?}");
        assert!(report.residue.bytes > 0, "{report:?}");
        assert_eq!(report.removed, 0, "上限很大，不动可复用条目");
        assert!(report.after < before, "完整条目保留");
        assert!(cache.get(&key).is_some());
    }
    fn a_crashed_lock_does_not_shield_a_record_forever() {
        // `O_EXCL` 锁是文件：持有者崩溃后没人释放它，旧实现会让这个 key 永远回收不掉。
        let (cache, key) = Fixture::seeded("gc-stale-lock");
        let dir = cache.dir_for(&key);
        std::fs::write(dir.join("build.lock"), "").expect("模拟崩溃的锁");
        age(&dir, 100);
        let report = cache.gc(0).expect("可回收");
        assert_eq!((report.removed, report.skipped), (1, 0), "{report:?}");
        assert!(cache.get(&key).is_none());
    }
    fn break_stale_lock_leaves_a_live_lock_alone() {
        let cache = Fixture::new("break-lock");
        let key = key();
        let lock = cache.lock(&key).expect("取锁");
        let path = cache.dir_for(&key).join("build.lock");
        assert!(!cache.break_stale_lock(&key), "活着的锁不能抢");
        assert!(path.exists());
        drop(lock);
        let lock = cache.lock(&key).expect("释放后可再次取锁");
        age(&cache.dir_for(&key), 100);
        assert!(cache.break_stale_lock(&key), "陈旧锁要能删掉");
        assert!(!path.exists());
        drop(lock);
    }
    fn entry_rejects_a_key_that_is_not_hexadecimal() {
        let cache = Fixture::new("entry");
        assert!(cache.entry("not-a-key").is_none());
        assert!(cache.entry("83a3ecdb31cd122").is_none(), "少一位也不行");
    }
    fn an_artifact_lives_in_its_entry_and_is_counted_by_the_gc() {
        let (cache, key) = Fixture::seeded("artifact");
        let built = cache.0.join("exe");
        std::fs::write(&built, vec![0u8; 4096]).expect("写假产物");
        let artifact_key = ArtifactKey::keyed(key.hex(), "cc", key.flags_hash, "main");
        assert!(cache.artifact(&artifact_key).is_none(), "还没编译过");
        let artifact = cache.put_artifact(&artifact_key, &built).expect("落产物");
        assert!(!artifact.reused, "{artifact:?}");
        assert!(artifact.path.starts_with(cache.dir_for(&key)), "落在条目里");
        let hit = cache.artifact(&artifact_key).expect("命中");
        assert_eq!((hit.reused, hit.bytes), (true, 4096), "{hit:?}");
        let entry = cache.entry(&key.hex()).expect("有记录");
        assert_eq!(entry.artifacts, 1, "{entry:?}");
        assert!(entry.bytes >= 4096, "产物要计入占用");
    }
    fn an_artifact_records_who_compiled_it() {
        // 身份写在产物旁边：没有编译器的机器靠它把产物认回来，而不是重新猜一个文件名。
        let (cache, key) = Fixture::seeded("artifact-identity");
        let hex = key.hex();
        let built = cache.0.join("exe");
        std::fs::write(&built, vec![0u8; 16]).expect("写假产物");
        let cc = ArtifactKey::keyed(hex.clone(), "cc", key.flags_hash, "main");
        let listed = cache.put_artifact(&cc, &built).expect("落产物");
        let found = cache.artifacts_in(&hex);
        assert_eq!((found.len(), &found[0].key), (1, &listed.key), "{found:?}");
        assert_eq!((found[0].reused, found[0].bytes), (true, 16), "{found:?}");
        let clang = ArtifactKey::keyed(hex.clone(), "clang", key.flags_hash, "main");
        cache.put_artifact(&clang, &built).expect("落另一份");
        assert_eq!(tools(&cache, &hex), ["cc", "clang"], "身份带着编译器名");
        let entry = cache.entry(&hex).expect("有记录");
        assert_eq!(entry.artifacts, 2, "身份文件不算产物");
        std::fs::remove_file(&listed.path).expect("删掉可执行文件");
        assert_eq!(tools(&cache, &hex), ["clang"], "缺可执行文件不算");
    }
    fn a_different_harness_compiler_or_dtype_is_a_different_artifact() {
        let key = key();
        let hex = key.hex();
        let harness = "int main(void){return 0;}\n";
        let other = "int main(void){return 1;}\n";
        let base = ArtifactKey::keyed(hex.clone(), "cc", key.flags_hash, harness);
        let same = ArtifactKey::keyed(hex.clone(), "cc", key.flags_hash, harness);
        let clang = ArtifactKey::keyed(hex.clone(), "clang", key.flags_hash, harness);
        let shaped = ArtifactKey::keyed(hex, "cc", key.flags_hash, other);
        assert_eq!(base.hex(), same.hex(), "同一输入得到同一产物");
        assert_ne!(base.hex(), clang.hex(), "换编译器的代码生成不同");
        assert_ne!(base.hex(), shaped.hex(), "主函数换了不能复用");
        let mut retyped = key;
        retyped.dtype = DTypeSet::of([DType::U8]);
        let dtype = ArtifactKey::keyed(retyped.hex(), "cc", retyped.flags_hash, harness);
        assert_ne!(base.hex(), dtype.hex(), "输出精度变了必须换键");
    }
    fn lock_is_exclusive_until_dropped() {
        let cache = Fixture::new("lock");
        let lock = cache.lock(&key()).expect("第一次取锁");
        assert!(cache.lock(&key()).is_err(), "同一 key 不能并发编译");
        drop(lock);
        assert!(cache.lock(&key()).is_ok(), "释放后可再次取锁");
    }
    }
}
