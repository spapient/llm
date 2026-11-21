//! 性质测试的**工作区纪律**：[`crate::prop`] 是唯一配置来源，谁都不能自建一套。
//!
//! [`super::prop_config`] 验"装置本身对不对"，这里验"有没有人绕过装置"。
//! 分两份是因为前者改装置时红、后者改任何 crate 时红，故障定位不同。
//!
//! 为什么值得一道门：`nn` 与 `algorithm` 曾经各有一份手抄的 `ProptestConfig`，
//! 两份的 `cases` 不同、`max_shrink_iters` 也不同，于是同一性质在两个 crate 里
//! 的失败率与最小反例都不一样。把配置收成一处只是修好当下；让第二处**无法出现**
//! 才是纪律。手抄配置不会让任何测试变红——它只会让性质测试静默地变弱。

use crate::architecture::{rust_files, source_text, src_root, workspace, workspace_root};
use std::path::{Path, PathBuf};

/// 唯一允许出现 `ProptestConfig` 构造的位置。
const HARNESS: &str = "crates/testkit/src/prop.rs";

fn relative(path: &Path) -> String {
    let root = workspace_root();
    let root = std::fs::canonicalize(&root).unwrap_or(root);
    let path = std::fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf());
    path.strip_prefix(&root)
        .unwrap_or(&path)
        .to_string_lossy()
        .replace('\\', "/")
}

/// 成员名 → 需要扫描的源码目录（`src`、`tests`、`benches`，存在才扫）。
fn scan_dirs(member: &str) -> Vec<PathBuf> {
    let src = src_root(member);
    let Some(manifest_dir) = src.parent() else {
        return Vec::new();
    };
    ["src", "tests", "benches"]
        .iter()
        .map(|name| manifest_dir.join(name))
        .filter(|dir| dir.is_dir())
        .collect()
}

/// 成员名 → 该成员全部 Rust 源文件的 `(相对路径, 文本)`。
fn member_sources(member: &str) -> Vec<(String, String)> {
    scan_dirs(member)
        .iter()
        .flat_map(|dir| rust_files(dir))
        .filter_map(|path| source_text(&path).map(|text| (relative(&path), text)))
        .collect()
}

/// 成员名 → 清单文本；读不到时返回 `None`（虚拟成员没有清单）。
fn manifest_text(member: &str) -> Option<String> {
    let manifest = src_root(member).parent()?.join("Cargo.toml");
    source_text(&manifest)
}

/// 工作区里声明了 `proptest` 的成员。
fn property_crates() -> Vec<String> {
    let (members, _) = workspace();
    members
        .iter()
        .filter(|member| {
            manifest_text(member).is_some_and(|text| text.contains("proptest"))
        })
        .cloned()
        .collect()
}

/// 去掉字符串字面量与注释，只留下真正的代码。
///
/// 剥字符串不是为了更聪明，是为了让这道门**可以讨论自己**：这个文件的断言语里就写着
/// `"ProptestConfig {"`，不剥的话门会被自己的说明文字打红，然后就有人去把它改成空转。
fn code_of(line: &str) -> String {
    let mut out = String::new();
    let mut chars = line.chars();
    let mut in_string = false;
    let mut escaped = false;
    while let Some(current) = chars.next() {
        if in_string {
            if escaped {
                escaped = false;
            } else if current == '\\' {
                escaped = true;
            } else if current == '"' {
                in_string = false;
            }
            continue;
        }
        if current == '"' {
            in_string = true;
            continue;
        }
        if current == '/' && matches!(chars.clone().next(), Some('/') | Some('*')) {
            break;
        }
        out.push(current);
    }
    out
}

/// 只认**构造**，不认提及：散文里写 `ProptestConfig` 是合法的，
/// 真绕过装置的是 `ProptestConfig { .. }` 或 `ProptestConfig::default()`。
fn constructs_a_config(line: &str) -> bool {
    let code = code_of(line);
    code.contains("ProptestConfig {")
        || code.contains("ProptestConfig{")
        || code.contains("ProptestConfig::")
}

#[test]
fn no_crate_hand_rolls_its_own_proptest_config() {
    let (members, _) = workspace();
    let mut offenders = Vec::new();
    for member in &members {
        for (path, text) in member_sources(member) {
            if path == HARNESS {
                continue;
            }
            for (index, line) in text.lines().enumerate() {
                if constructs_a_config(line) {
                    offenders.push(format!("{path}:{}", index + 1));
                }
            }
        }
    }
    assert!(
        offenders.is_empty(),
        "这些位置自建了 `ProptestConfig`，请改用 `testkit::prop::config()`：{offenders:?}\n\
         手抄配置不会让测试变红，只会让性质测试静默变弱（样本数与收缩预算各不相同）"
    );
}

#[test]
fn every_property_crate_depends_on_the_harness() {
    let mut offenders = Vec::new();
    for member in property_crates() {
        let Some(manifest) = manifest_text(&member) else {
            continue;
        };
        if !manifest.contains("testkit") {
            offenders.push(member);
        }
    }
    assert!(
        offenders.is_empty(),
        "这些成员声明了 `proptest` 却没有 `testkit` 依赖，因此拿不到共享配置：{offenders:?}"
    );
}

#[test]
fn every_property_crate_reaches_the_shared_configuration() {
    let mut offenders = Vec::new();
    for member in property_crates() {
        // 装置自己的 lib 内部写 `crate::prop`，只有它免检。
        if member == "testkit" {
            continue;
        }
        let reached = member_sources(&member)
            .iter()
            .any(|(_, text)| text.contains("testkit::prop"));
        if !reached {
            offenders.push(member);
        }
    }
    assert!(
        offenders.is_empty(),
        "这些成员有 `testkit` 依赖却从不调用 `testkit::prop::*`，性质测试没走共享装置：{offenders:?}"
    );
}

/// 纪律门的自检：扫描面必须真的覆盖到已经收口的四个 crate。
///
/// 没有这条，"扫描恒返回空集"这种失效会和"纪律完好"长得一模一样——
/// `mirror` 的注解扫描就因为写死 `crate_name == "op"` 而在拆分后静默归零过。
#[test]
fn the_scan_actually_sees_the_property_crates() {
    let found = property_crates();
    for expected in ["gemm", "conv", "nn", "algorithm"] {
        assert!(
            found.contains(&expected.to_string()),
            "扫描面漏了 `{expected}`：清单里确实声明了 proptest，说明这条路没有真的读文件"
        );
    }
    let total: usize = found.iter().map(|member| member_sources(member).len()).sum();
    assert!(total > 0, "一个源文件都没读到，扫描面是空的");
}
