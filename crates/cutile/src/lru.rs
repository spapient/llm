//! 职责：缓存条目计数与 LRU 淘汰。

use crate::cache::{Cache, CacheKey};
use diagnostic::{Code, Error};
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime};

/// 陈旧判据：一次 emit 是毫秒级、等锁上限 5 s，一分钟没被动过的锁与半截目录只可能是崩溃遗留。
pub const STALE_LOCK: Duration = Duration::from_secs(60);

/// 回收掉的残留：本来就不会命中，所以与上限无关。
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
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
    /// 残留部分：定义见 [`Residue`]。
    pub residue: Residue,
    /// 因为仍在被使用（[`Entry::building`]）而跳过的目录数。
    pub skipped: usize,
}

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

/// 一次目录遍历的累计量：字节数、最近修改时间，以及目录里有没有 `source` / `build.lock`。
#[derive(Default)]
pub(crate) struct Tally {
    pub(crate) bytes: u64,
    pub(crate) used: Option<SystemTime>,
    pub(crate) state: State,
    pub(crate) locked: bool,
}

impl Tally {
    /// 递归测量一个条目目录；目录列表读不动时返回 `None`（该目录当不存在）。
    pub(crate) fn measure(dir: &Path) -> Option<Self> {
        let mut tally = Self::default();
        tally.walk(dir, dir)?;
        Some(tally)
    }

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
pub(crate) fn directory_time(dir: &Path) -> SystemTime {
    let meta = std::fs::metadata(dir).and_then(|meta| meta.modified());
    meta.unwrap_or(std::time::UNIX_EPOCH)
}

/// 删一个目录并扣账：仍在用的只记一次跳过。返回是否真的删掉了。
fn retire(entry: &Entry, now: SystemTime, gc: &mut Gc) -> diagnostic::Result<bool> {
    if entry.building(now) {
        gc.skipped += 1;
        return Ok(false);
    }
    remove(&entry.dir)?;
    gc.after -= entry.bytes;
    Ok(true)
}

/// 残留一律回收：它们本来就不会命中，等不到任何一次"超限"。
pub(crate) fn reclaim(entries: &[Entry], now: SystemTime, gc: &mut Gc) -> diagnostic::Result<()> {
    for entry in entries.iter().filter(|entry| !entry.complete()) {
        if retire(entry, now, gc)? {
            gc.residue.items += 1;
            gc.residue.bytes += entry.bytes;
        }
    }
    Ok(())
}

/// 最久没用过的先删：排到旧的在前面，删到上限以内为止。
pub(crate) fn evict(entries: &[Entry], now: SystemTime, gc: &mut Gc) -> diagnostic::Result<()> {
    let mut complete: Vec<&Entry> = entries.iter().filter(|entry| entry.complete()).collect();
    complete.sort_by_key(|entry| entry.used);
    for entry in complete {
        if gc.after <= gc.limit {
            break;
        }
        if retire(entry, now, gc)? {
            gc.removed += 1;
        }
    }
    Ok(())
}

fn remove(dir: &Path) -> diagnostic::Result<()> {
    std::fs::remove_dir_all(dir)
        .map_err(|error| Error::new(Code::Io, format!("删除 {} 失败：{error}", dir.display())))
}
