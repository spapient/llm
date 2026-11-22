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

/// 内核身份字段（实现名 / 块宽）必须进指纹：同原语不同实现或不同块宽不能互相命中。
#[test]
fn kernel_identity_fields_change_the_hash() {
    let mut other = key();
    other.impl_name = "dp4a".to_string();
    assert_ne!(key().hash(), other.hash(), "换实现必须换键");
    other.impl_name = key().impl_name.clone();
    other.block = "32".to_string();
    assert_ne!(key().hash(), other.hash(), "换块宽必须换键");
}
/// 工具链不属于源文本：它的身份在 [`ArtifactKey`] 里，而且是实测值。换编译器不该让源文本失效。
#[test]
fn the_source_key_ignores_the_archive_tool_registration() {
    let names: Vec<&str> = key().fields().into_iter().map(|(n, _)| n).collect();
    assert!(!names.contains(&"tools"), "{names:?}");
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
    let text = "source=abc;tools=cc@21.0.0;harness=0000000000000001";
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
