//! ⑤ 缓存：指纹、锁、原子落盘（DESIGN §10.3）。
//!
//! 指纹必须是"改任何输入都换 key"：schema 版本、emit 语言、工具版本、编译参数、架构、
//! 模板哈希、参与拼接的片段哈希、布局正规形、变体与 tile、数据类型。

use crate::atom::Prim;
use crate::lower::emit::fnv1a;
use crate::prof::Emit;
use std::fmt;
use std::path::{Path, PathBuf};

/// 缓存指纹。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CacheKey {
    /// cutile 自身版本。
    pub schema_ver: u16,
    /// 输出语言。
    pub emit: Emit,
    /// 工具名。
    pub tool: String,
    /// 工具版本。
    pub tool_ver: String,
    /// 编译参数哈希。
    pub flags_hash: u64,
    /// 架构键。
    pub arch: String,
    /// 模板文本哈希。
    pub template_hash: u64,
    /// 参与拼接的片段与内容哈希。
    pub fragments: Vec<(Prim, u64)>,
    /// 布局正规形（槽位 → 签名）。
    pub layouts: Vec<(String, String)>,
    /// 变体名。
    pub variant: String,
    /// tile 取值。
    pub tile: String,
    /// 数据类型。
    pub dtype: String,
}

impl CacheKey {
    /// 规范化描述：片段与布局先排序，保证同一输入得到同一文本。
    pub fn describe(&self) -> String {
        let mut fragments = self.fragments.clone();
        fragments.sort_by_key(|(prim, _)| *prim);
        let mut layouts = self.layouts.clone();
        layouts.sort();
        format!(
            "schema={};emit={};tool={}@{};flags={:016x};arch={};template={:016x};fragments={:?};layouts={:?};variant={};tile={};dtype={}",
            self.schema_ver,
            self.emit,
            self.tool,
            self.tool_ver,
            self.flags_hash,
            self.arch,
            self.template_hash,
            fragments,
            layouts,
            self.variant,
            self.tile,
            self.dtype
        )
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

    /// `CUTILE_CACHE` 或默认的 `target/cutile-cache`。
    pub fn from_env() -> Self {
        match std::env::var_os("CUTILE_CACHE") {
            Some(value) => Self::new(PathBuf::from(value)),
            None => Self::new(PathBuf::from("target").join("cutile-cache")),
        }
    }

    /// 根目录。
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// 某个 key 的目录：`<root>/<前两位>/<hex>`。
    pub fn dir_for(&self, key: &CacheKey) -> PathBuf {
        let hex = key.hex();
        self.root.join(&hex[..2]).join(hex)
    }

    /// 读缓存（命中即返回源文本）。
    pub fn get(&self, key: &CacheKey) -> Option<String> {
        let path = self.dir_for(key).join("source");
        std::fs::read_to_string(path).ok()
    }

    /// 写缓存：先写临时文件再 `rename`，避免读到半截文件。
    pub fn put(&self, key: &CacheKey, source: &str) -> std::io::Result<PathBuf> {
        let dir = self.dir_for(key);
        std::fs::create_dir_all(&dir)?;
        let temp = dir.join("source.tmp");
        let final_path = dir.join("source");
        std::fs::write(&temp, source)?;
        std::fs::write(dir.join("meta.txt"), key.describe())?;
        std::fs::rename(&temp, &final_path)?;
        Ok(final_path)
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

    fn key() -> CacheKey {
        CacheKey {
            schema_ver: 1,
            emit: Emit::C,
            tool: "cc".to_string(),
            tool_ver: "17".to_string(),
            flags_hash: 0x1234,
            arch: "aarch64".to_string(),
            template_hash: 0xabcd,
            fragments: vec![(Prim::Dot, 1), (Prim::Load, 2)],
            layouts: vec![("w".to_string(), "row[16,64]".to_string())],
            variant: "pipe2".to_string(),
            tile: "GMAX=32".to_string(),
            dtype: "f32".to_string(),
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
        other.tool_ver = "18".to_string();
        assert_ne!(key().hash(), other.hash());
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
