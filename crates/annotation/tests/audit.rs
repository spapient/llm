//! 性能与供应链静态门禁（`annotation` 局部）。
//!
//! 过程宏在编译期运行，输入只是一小段属性；它的性能风险不是绝对毫秒数，而是文法退化成关联
//! 容器查找、正则回溯或引入重型初始化。下面两条测试把这几类回退挡在合并之前：
//!
//! - 生产源码不出现关联容器、正则或惰性/全局初始化；
//! - 运行期依赖为空（`MODEL` 改为投影 `Operator` 后，连 `dtype` 基座也不再需要）。

use std::fs;
use std::path::PathBuf;

/// 生产源码里禁止出现的重型结构：关联容器、正则、惰性 / 全局初始化。
const FORBIDDEN: [&str; 10] = [
    "HashMap",
    "HashSet",
    "BTreeMap",
    "BTreeSet",
    "regex",
    "Regex",
    "lazy_static",
    "once_cell",
    "DashMap",
    "phf",
];

fn walk(dir: &PathBuf, files: &mut Vec<PathBuf>) {
    let Ok(entries) = fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            walk(&path, files);
        } else if path.extension().is_some_and(|ext| ext == "rs") {
            files.push(path);
        }
    }
}

/// 生产源码不引入关联容器或正则：文法是游标单趟扫描，查名字只在闭集数组上线性包含。
#[test]
fn production_source_has_no_associative_container_or_regex() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src");
    let mut files = Vec::new();
    walk(&root, &mut files);

    let mut offenders: Vec<String> = Vec::new();
    for path in files {
        let text = fs::read_to_string(&path).expect("源码可读");
        for token in FORBIDDEN {
            if text.contains(token) {
                offenders.push(format!("{} 含 `{token}`", path.display()));
            }
        }
    }
    assert!(offenders.is_empty(), "{offenders:#?}");
}

/// 取出某个 manifest 小节的键（小节头到下一个 `[` 之间）。
fn section_keys<'a>(manifest: &'a str, section: &str) -> Vec<&'a str> {
    let mut inside = false;
    let mut keys = Vec::new();
    for line in manifest.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with('[') {
            inside = trimmed == section;
            continue;
        }
        if !inside {
            continue;
        }
        let Some(key) = trimmed.split('=').next().map(str::trim) else {
            continue;
        };
        if !key.is_empty() {
            keys.push(key);
        }
    }
    keys
}

/// 运行期依赖为空：注解文法手写解析，`MODEL` 由 `Operator` 投影，不再需要任何基座 crate。
#[test]
fn runtime_dependency_surface_is_empty() {
    let manifest_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("Cargo.toml");
    let manifest = fs::read_to_string(&manifest_path).expect("manifest 可读");
    let mut dependencies = section_keys(&manifest, "[dependencies]");
    dependencies.sort_unstable();
    assert!(
        dependencies.is_empty(),
        "运行期依赖面必须为空：{dependencies:?}"
    );
}
