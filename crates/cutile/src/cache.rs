//! ⑤ 缓存：指纹、编译产物、锁、原子落盘、容量回收（README §10.3）。
//!
//! 指纹必须"改任何输入都换 key"，但**工具链不在其中**：源文本与编译器无关，换编译器只换产物，那由
//! [`ArtifactKey::tool`] 的实测身份承担。[`CacheKey::fields`] 是逐项字段的唯一出处。
use crate::atom::{DTypeSet, Prim};
use crate::emit::fnv1a;
use crate::prof::Emit;
use diagnostic::{Code, Error};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, SystemTime};

/// 陈旧判据：一次 emit 是毫秒级、等锁上限 5 s，一分钟没被动过的锁与半截目录只可能是崩溃遗留。
pub const STALE_LOCK: Duration = Duration::from_secs(60);

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

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
/// 回收掉的残留：本来就不会命中，所以与上限无关。
pub struct Residue {
    /// 回收掉的目录数。
    pub items: usize,
    /// 它们占过的字节数。
    pub bytes: u64,
}

/// 一次容量回收的结果。
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct Gc {
    /// 容量上限（字节）。
    pub limit: u64,
    /// 回收前的占用（含残留）。
    pub before: u64,
    /// 回收后的占用：仍高于上限时说明剩下的条目都在用。
    pub after: u64,
    /// 按 LRU 删除的可复用条目数。
    pub removed: usize,
    /// 回收掉的残留：本来就不会命中，所以与上限无关。
    pub residue: Residue,
    /// 因为仍在被使用（[`Entry::building`]）而跳过的目录数。
    pub skipped: usize,
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
        let bytes = std::fs::metadata(&path).map(|meta| meta.len()).unwrap_or(0);
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
        std::fs::create_dir_all(&dir)?;
        let temp = temp_path(&dir, "source");
        std::fs::write(&temp, source)?;
        let fields = key.fields();
        let meta: String = fields.iter().map(|(n, v)| format!("{n}={v}\n")).collect();
        std::fs::write(dir.join("meta.txt"), meta)?;
        std::fs::rename(&temp, dir.join("source"))?;
        Ok(dir.join("source"))
    }
    /// 写一份不来自算子声明的程序（见证、探测）：条目按程序文本定名，返回的键与 [`CacheKey::hex`] 同形。
    pub fn put_program(&self, label: &str, program: &str) -> std::io::Result<String> {
        let hex = format!("{:016x}", fnv1a(&format!("{label};{program}")));
        let dir = self.entry_dir(&hex);
        std::fs::create_dir_all(&dir)?;
        let temp = temp_path(&dir, "program");
        std::fs::write(&temp, program)?;
        let hash = format!("program={label}\nprogram_hash={:016x}\n", fnv1a(program));
        std::fs::write(dir.join("meta.txt"), hash)?;
        std::fs::rename(&temp, dir.join("source"))?;
        Ok(hex)
    }
    /// 回收：先清残留，再按 LRU 删到上限以内；仍在用的（[`Entry::building`]）跳过——删它会把别人的产物变半截。
    pub fn gc(&self, limit: u64) -> diagnostic::Result<Gc> {
        let now = SystemTime::now();
        let entries = self.entries();
        let before: u64 = entries.iter().map(|e| e.bytes).sum();
        let mut gc = Gc::default();
        (gc.limit, gc.before, gc.after) = (limit, before, before);
        reclaim(&entries, now, &mut gc)?;
        evict(&entries, now, &mut gc)?;
        Ok(gc)
    }
    /// 删陈旧锁：持有者已崩溃，没人能来释放它（删失败也返回 `false`，多半别人先删了）；活着的锁不动。
    pub fn break_stale_lock(&self, key: &CacheKey) -> bool {
        let path = self.dir_for(key).join("build.lock");
        let stale = match std::fs::metadata(&path).and_then(|meta| meta.modified()) {
            Ok(time) => SystemTime::now().duration_since(time).unwrap_or_default() >= STALE_LOCK,
            // 读不到时间戳只可能是锁已经不在了：`remove_file` 会失败，结果仍是 `false`。
            Err(_) => true,
        };
        stale && std::fs::remove_file(&path).is_ok()
    }
    /// 取锁（`O_EXCL` 语义）；调用方 `drop` 即释放。
    pub fn lock(&self, key: &CacheKey) -> std::io::Result<Lock> {
        let dir = self.dir_for(key);
        std::fs::create_dir_all(&dir)?;
        let path = dir.join("build.lock");
        std::fs::File::create_new(&path)?;
        Ok(Lock { path })
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
        let mut tally = Tally::default();
        tally.walk(dir, dir)?;
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

/// 残留一律回收：它们本来就不会命中，等不到任何一次"超限"。
fn reclaim(entries: &[Entry], now: SystemTime, gc: &mut Gc) -> diagnostic::Result<()> {
    for entry in entries.iter().filter(|entry| !entry.complete()) {
        if entry.building(now) {
            gc.skipped += 1;
            continue;
        }
        remove(&entry.dir)?;
        gc.after -= entry.bytes;
        gc.residue.items += 1;
        gc.residue.bytes += entry.bytes;
    }
    Ok(())
}
fn evict(entries: &[Entry], now: SystemTime, gc: &mut Gc) -> diagnostic::Result<()> {
    let mut complete: Vec<&Entry> = entries.iter().filter(|entry| entry.complete()).collect();
    complete.sort_by_key(|entry| entry.used);
    for entry in complete {
        if gc.after <= gc.limit {
            break;
        }
        if entry.building(now) {
            gc.skipped += 1;
            continue;
        }
        remove(&entry.dir)?;
        gc.after -= entry.bytes;
        gc.removed += 1;
    }
    Ok(())
}
fn remove(dir: &Path) -> diagnostic::Result<()> {
    std::fs::remove_dir_all(dir)
        .map_err(|error| Error::new(Code::Io, format!("删除 {} 失败：{error}", dir.display())))
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

/// 一个编译产物的身份：源文本 + 实测工具链（[`crate::backend::Toolchain::id`]）+ 编译参数 + 主函数；缺一项都会复用成错误的产物。
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

#[derive(Default)]
struct Tally {
    bytes: u64,
    used: Option<SystemTime>,
    state: State,
    locked: bool,
}

impl Tally {
    fn walk(&mut self, root: &Path, dir: &Path) -> Option<()> {
        for file in std::fs::read_dir(dir).ok()?.flatten() {
            let Ok(meta) = file.metadata() else {
                continue;
            };
            if meta.is_dir() {
                let _ = self.walk(root, &file.path());
            } else {
                self.add(dir == root, file.file_name().to_str(), &meta);
            }
        }
        Some(())
    }

    fn add(&mut self, at_root: bool, name: Option<&str>, meta: &std::fs::Metadata) {
        self.bytes += meta.len();
        self.used = self.used.max(meta.modified().ok());
        match (at_root, name) {
            (true, Some("source")) => self.state = State::Complete,
            (true, Some("build.lock")) => self.locked = true,
            _ => {}
        }
    }
}

/// 空目录没有任何文件时间戳：退回目录自身的时间（删旧的空目录仍要靠它排 LRU）。
fn directory_time(dir: &Path) -> SystemTime {
    let meta = std::fs::metadata(dir).and_then(|meta| meta.modified());
    meta.unwrap_or(std::time::UNIX_EPOCH)
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
    #[test]
    fn identical_inputs_share_a_hash() {
        assert_eq!((key().hash(), key().hex().len()), (key().hash(), 16));
    }
    #[test]
    fn changing_any_field_changes_the_hash() {
        let mut other = key();
        other.tile = "GMAX=64".to_string();
        assert_ne!(key().hash(), other.hash());
        other.variant = "direct".to_string();
        assert_ne!(key().hash(), other.hash());
    }
    /// 工具链不属于源文本：它的身份在 [`ArtifactKey`] 里，而且是实测值。换编译器不该让源文本失效。
    #[test]
    fn the_source_key_ignores_the_archive_tool_registration() {
        let names: Vec<&str> = key().fields().into_iter().map(|(n, _)| n).collect();
        assert!(!names.contains(&"tool"), "{names:?}");
        assert!(!names.contains(&"tool_ver"), "{names:?}");
    }
    #[test]
    fn the_artifact_identity_carries_the_measured_toolchain() {
        let cc17 = ArtifactKey::keyed("abc", "cc@17.0.0", 0, "main");
        let cc21 = ArtifactKey::keyed("abc", "cc@21.0.0", 0, "main");
        assert_ne!(cc17, cc21, "换编译器版本必须换产物");
        assert_ne!(cc17.hex(), cc21.hex());
        assert_eq!(cc17, ArtifactKey::parse(&cc17.describe()).expect("可还原"));
    }
    /// 编译参数也决定产物：少了它，改一次 `[compile].flags` 会静默复用旧参数的二进制，而"这条见证跑通过"就对不上。
    #[test]
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
    #[test]
    fn an_artifact_without_parameters_has_no_identity() {
        let text = "source=abc;tool=cc@21.0.0;harness=0000000000000001";
        assert!(ArtifactKey::parse(text).is_none(), "{text}");
    }
    #[test]
    fn fragment_order_does_not_change_the_hash() {
        let mut other = key();
        other.fragments.reverse();
        assert_eq!(key().hash(), other.hash());
    }
    #[test]
    fn put_and_get_roundtrip() {
        let cache = Fixture::new("roundtrip");
        assert!(cache.get(&key()).is_none());
        cache.put(&key(), "int x;\n").expect("写缓存");
        assert_eq!(cache.get(&key()), Some("int x;\n".to_string()));
    }
    #[test]
    fn meta_keeps_the_key_field_by_field() {
        let (cache, key) = Fixture::seeded("meta");
        let entry = cache.entry(&key.hex()).expect("有记录");
        assert_eq!(entry.fields.len(), key.fields().len(), "{entry:?}");
        let pipe2 = |(n, v): &(String, String)| n == "variant" && v == "pipe2";
        assert!(entry.fields.iter().any(pipe2), "{entry:?}");
        assert!(key.changes(&entry.fields).is_empty(), "同一份键没有差异");
    }
    #[test]
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
    #[test]
    fn gc_evicts_the_least_recently_used_first() {
        let cache = Fixture::new("gc");
        let mut keys = Vec::new();
        for (index, secs) in [100u64, 200, 300].into_iter().enumerate() {
            let mut key = key();
            key.variant = format!("v{index}");
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
    #[test]
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
    #[test]
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
    #[test]
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
    #[test]
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
    #[test]
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
    #[test]
    fn entry_rejects_a_key_that_is_not_hexadecimal() {
        let cache = Fixture::new("entry");
        assert!(cache.entry("not-a-key").is_none());
        assert!(cache.entry("83a3ecdb31cd122").is_none(), "少一位也不行");
    }
    #[test]
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
    #[test]
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
    #[test]
    fn a_different_harness_compiler_or_dtype_is_a_different_artifact() {
        let key = key();
        let hex = key.hex();
        let harness = "int main(void){return 0;}\n";
        let other = "int main(void){return 1;}\n";
        let base = ArtifactKey::keyed(hex.clone(), "cc", key.flags_hash, harness);
        let same = ArtifactKey::keyed(hex.clone(), "cc", key.flags_hash, harness);
        let clang = ArtifactKey::keyed(hex.clone(), "clang", key.flags_hash, harness);
        let shaped = ArtifactKey::keyed(hex.clone(), "cc", key.flags_hash, other);
        assert_eq!(base.hex(), same.hex(), "同一输入得到同一产物");
        assert_ne!(base.hex(), clang.hex(), "换编译器的代码生成不同");
        assert_ne!(base.hex(), shaped.hex(), "主函数换了不能复用");
        let mut retyped = key.clone();
        retyped.dtype = DTypeSet::of([DType::U8]);
        let dtype = ArtifactKey::keyed(retyped.hex(), "cc", retyped.flags_hash, harness);
        assert_ne!(base.hex(), dtype.hex(), "输出精度变了必须换键");
    }
    #[test]
    fn lock_is_exclusive_until_dropped() {
        let cache = Fixture::new("lock");
        let lock = cache.lock(&key()).expect("第一次取锁");
        assert!(cache.lock(&key()).is_err(), "同一 key 不能并发编译");
        drop(lock);
        assert!(cache.lock(&key()).is_ok(), "释放后可再次取锁");
    }
}
