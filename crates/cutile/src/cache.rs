//! 编译产物缓存：指纹、锁、落盘与容量回收。
//!
//! 身份（键与新旧判定）在 [`key`]；本文件只管「留不留」，以及一次落盘的原子性。

mod key;

pub use key::{ArtifactKey, CACHE_LIMIT, CacheKey, Changed, Entry, STALE_LOCK, State};

use crate::emit::fnv1a;
use diagnostic::{Code, Error};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, SystemTime};

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
        let meta: String = key
            .fields()
            .iter()
            .map(|(n, v)| format!("{n}={v}\n"))
            .collect();
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
    /// 原子落一份条目：目录建好后，源文本先写临时名、`meta.txt` 同目录写好，再把临时名换成 `source`；
    /// 读路径因此要么拿到完整旧份，要么拿到完整新份，不会读到半截。两份写入（算子缓存 / 见证程序）共用这一条。
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
        let mut gc = Gc::default();
        (gc.limit, gc.before, gc.after) = (limit, before, before);
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

/// `cache gc`：把缓存按 LRU 回收到上限以内；`limit` 为 `None` 时用 [`CACHE_LIMIT`]。
///
/// # Errors
///
/// 删除某个条目失败时返回带路径的说明（缓存目录不可写之类）。
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
#[path = "../tests/cache_tests.rs"]
mod tests;
