//! 档案层：设备能力档案（[`devices/`](../devices)）与它的 schema（[`prof`]）。
//!
//! 本 crate 是"设备差异只有数据"的容身之处：
//!
//! - [`devices/`](../devices) 里的 `<name>.toml` 是**内容**：存储层物理名、输出语言、工具链版本、
//!   执行单元及其可访问的存储层、每条能力区间，全部在里面；
//! - [`prof`] 是**唯一的 schema**：解析、闭集字段、见证与 `[units]` 一致性校验；
//! - [`atom`] 是**唯一的能力词汇表**：16 个原语、类型/存储位集、参数化签名 `Need`/`Provide`
//!   与属性键闭集 [`atom::AttrKey`]。
//!
//! `cutile` 只做机制（协商、降级、拼接、emit），因此本 crate 不反向依赖它：依赖方向
//! `cutile → prof`，而 `prof` 只依赖 `std` 与同属基座的 [`llm_config::toml`]。
//!
//! 档案里 `witness`（L2 证据）指向片段层（`crates/prim`）：路径相对**片段层根**，仓库里是
//! `crates/prim`，换数据根时是 `<data.root>/prim`。本 crate 只在装载期做"文件在不在"的校验，
//! 不解析片段内容，也就不至于把两层耦合成一个 crate。
//!
//! # 例子
//!
//! ```
//! let archives = llm_prof::builtin();
//! assert_eq!(archives.len(), llm_prof::BUILTIN.len());
//! assert!(llm_prof::embedded("cpu").is_some_and(|text| text.contains("[meta]")));
//! ```

#![deny(unsafe_code)]

use std::path::{Path, PathBuf};

pub mod atom;
pub mod prof;

pub use llm_config::toml;

/// 编译期内嵌的档案层：`(文件名, 正文)`。
///
/// 新增设备档案要在这里补一行，否则发布出去的二进制里没有它；`builtin_matches_disk` 测试会把
/// "磁盘上有、内嵌里没有"直接判为失败。顺序与磁盘一致（按文件名排序），否则装载结果与内嵌
/// 副本无法逐项对拍。
pub const BUILTIN: [(&str, &str); 4] = [
    ("ascend", include_str!("../devices/ascend.toml")),
    ("cpu", include_str!("../devices/cpu.toml")),
    ("intel", include_str!("../devices/intel.toml")),
    ("nvidia", include_str!("../devices/nvidia.toml")),
];

/// 一份设备档案：`<name>.toml` 的文件名与正文。
///
/// 名字是**文件名**，不是 `[meta] vendor`：同一 vendor 可以有多份档案（按 `id` 区分），
/// 因此这里不做"名字等于 vendor"的校验——那属于 `cutile` 侧的语义。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Archive {
    name: String,
    text: String,
}

impl Archive {
    /// 由文件名与正文构造。
    pub fn new(name: impl Into<String>, text: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            text: text.into(),
        }
    }

    /// 档案名（与 `<name>.toml` 的文件名一致）。
    pub fn name(&self) -> &str {
        &self.name
    }

    /// 档案正文：`[meta]` / `[fingerprint]` / `[space]` / `[[provides]]`。
    pub fn text(&self) -> &str {
        &self.text
    }
}

/// 内嵌档案，顺序与 [`BUILTIN`] 一致。
pub fn builtin() -> Vec<Archive> {
    BUILTIN
        .iter()
        .map(|(name, text)| Archive::new(*name, *text))
        .collect()
}

/// 按名字取内嵌档案正文；没有同名档案时返回 `None`。
pub fn embedded(name: &str) -> Option<&'static str> {
    BUILTIN
        .iter()
        .find(|(builtin, _)| *builtin == name)
        .map(|(_, text)| *text)
}

/// 档案层目录：本 crate 根下的 `devices/`，即磁盘上的事实源。
pub fn dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("devices")
}

/// 从目录装载档案：只取 `*.toml`，结果按文件名排序。
///
/// # Errors
///
/// 目录不可读、文件名不是合法 UTF-8、或某个文件读不出来时返回带路径的说明。
pub fn load(dir: &Path) -> Result<Vec<Archive>, String> {
    let entries =
        std::fs::read_dir(dir).map_err(|error| format!("读取 {} 失败：{error}", dir.display()))?;
    let mut paths: Vec<PathBuf> = entries
        .filter_map(|entry| entry.ok().map(|entry| entry.path()))
        .filter(|path| path.is_file())
        .collect();
    paths.sort();

    let mut archives = Vec::new();
    for path in paths {
        if path.extension().and_then(|value| value.to_str()) != Some("toml") {
            continue;
        }
        let name = path
            .file_stem()
            .and_then(|value| value.to_str())
            .ok_or_else(|| format!("`{}` 的文件名不是合法 UTF-8", path.display()))?;
        archives.push(Archive::new(name, read(&path)?));
    }
    Ok(archives)
}

fn read(path: &Path) -> Result<String, String> {
    std::fs::read_to_string(path).map_err(|error| format!("读取 {} 失败：{error}", path.display()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_dir(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("prof-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("建临时目录");
        dir
    }

    #[test]
    fn builtin_matches_disk() {
        let disk = load(&dir()).expect("磁盘上的档案层可解析");
        let embedded = builtin();
        assert_eq!(
            disk.iter().map(Archive::name).collect::<Vec<_>>(),
            embedded.iter().map(Archive::name).collect::<Vec<_>>(),
            "`devices/` 里新增或删除了档案，但 `BUILTIN` 没跟着改"
        );
        assert_eq!(disk, embedded);
    }

    #[test]
    fn every_builtin_entry_is_a_device_archive() {
        for archive in builtin() {
            assert!(!archive.name().is_empty());
            assert!(archive.text().contains("[meta]"), "{}", archive.name());
            assert!(
                archive.text().contains("[[provides]]"),
                "{}",
                archive.name()
            );
            assert_eq!(embedded(archive.name()), Some(archive.text()));
        }
    }

    #[test]
    fn embedded_reports_a_missing_archive() {
        assert_eq!(embedded("no-such-device"), None);
    }

    #[test]
    fn load_skips_non_toml_files_and_sorts_by_name() {
        let dir = temp_dir("order");
        for name in ["b", "a"] {
            std::fs::write(dir.join(format!("{name}.toml")), "[meta]\n").expect("写");
        }
        std::fs::write(dir.join("notes.md"), "ignored\n").expect("写");
        let names: Vec<String> = load(&dir)
            .expect("装载")
            .into_iter()
            .map(|archive| archive.name().to_string())
            .collect();
        assert_eq!(names, ["a", "b"]);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_missing_directory_names_the_path() {
        let dir = temp_dir("missing");
        let _ = std::fs::remove_dir_all(&dir);
        let error = load(&dir).unwrap_err();
        assert!(error.contains("prof-missing"), "{error}");
    }
}
