//! 架构契约的事实面：`docs/ARCHITECTURE.md` 的三张表与 `cargo metadata` 的成员/边。
//!
//! 层表、依赖边与所有权表过去靠评审清单核对，30 个成员 × 每次改动 = 必然漂移。这里把
//! `cargo metadata` 的事实与文档逐条解析出来，由 [`crate::layer`] / [`crate::ownership`] /
//! [`crate::boundary`] 三个门禁各自比对一侧：改文档或改 `Cargo.toml` 而不同步另一侧即失败。
//!
//! 只读 `cargo metadata` 与源码，不修改任何东西；`cargo test` 因此就是守门人。
//!
//! 三张表的解析是**同一份**事实源（`DOC` 编译期嵌入），所以集中在这个模块里：门禁各自只持有
//! 自己的判据，解析口径不复制第二遍。

#[cfg(test)]
pub(crate) mod tests {
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::process::Command;

/// 仓根：本模块住在 `crates/<crate>/src/`，仓根是清单目录上两级。
pub fn workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

/// 递归列出 `dir` 下的全部 `.rs` 文件（顺序无关，只用于扫描）。
///
/// 目录不存在或读不到时返回空：扫描集派生方（如层表）必须另守"至少扫到一个文件"，
/// 否则合并/删除成员会让守卫静默通过。
pub fn rust_files(dir: &Path) -> Vec<PathBuf> {
    let mut found = Vec::new();
    walk(dir, &mut found);
    found
}

pub(crate) fn walk(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            walk(&path, out);
        } else if path.extension().and_then(|value| value.to_str()) == Some("rs") {
            out.push(path);
        }
    }
}

/// 契约文本编译期嵌入：测试不依赖当前工作目录，也不会读到另一份副本。
pub(crate) const DOC: &str = include_str!("../../../docs/ARCHITECTURE.md");

/// 层表：成员 → (层号, 层内位置)；同层位置决定“可以依赖谁”。
pub(crate) struct Contract {
    pub(crate) layers: BTreeMap<String, (usize, usize)>,
    pub(crate) tools: BTreeSet<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct Edge {
    pub(crate) from: String,
    pub(crate) to: String,
    pub(crate) dev: bool,
}

/// `llm-op` 与文档里的 `op` 指同一个成员（包名带 `llm-` 前缀，层表用短名）。
pub(crate) fn short(name: &str) -> String {
    let name = name.trim();
    name.strip_prefix("llm-").unwrap_or(name).to_string()
}

pub(crate) fn contract() -> Contract {
    let mut layers = BTreeMap::new();
    for line in DOC.lines() {
        let Some(rest) = line.strip_prefix("| L") else {
            continue;
        };
        let Some((level, tail)) = rest.split_once(' ') else {
            continue;
        };
        let Ok(level) = level.parse::<usize>() else {
            continue;
        };
        let cells: Vec<&str> = tail.split('|').collect();
        let Some(members) = cells.get(1) else {
            continue;
        };
        // `L7` 的成员格还带了 bin 说明（`llm`（bin：`llm` 与 `cutile`）），括号内不是成员。
        let members = members.split('（').next().unwrap_or_default();
        for (position, member) in backticks(members).into_iter().enumerate() {
            layers.insert(short(&member), (level, position));
        }
    }
    let tools = DOC
        .lines()
        .find(|line| line.contains("是**工具成员**"))
        .map(|line| {
            backticks(line)
                .into_iter()
                .map(|name| short(&name))
                .collect()
        })
        .unwrap_or_default();
    Contract { layers, tools }
}


/// 解析“允许的边”代码块，返回边集与文档声称的总数 / `+dev` 数。
pub(crate) fn declared_edges() -> (BTreeSet<Edge>, usize, usize) {
    let mut edges = BTreeSet::new();
    let mut entries: Vec<String> = Vec::new();
    let mut inside = false;
    let mut opened = false;
    let mut total = 0;
    let mut dev = 0;
    for line in DOC.lines() {
        if !inside {
            if line.contains("允许的边（当前全量") {
                inside = true;
                total = count_after(line, "共 ");
                dev = count_after(line, "含 ");
            }
            continue;
        }
        let trimmed = line.trim();
        if trimmed == "```" {
            if opened {
                break;
            }
            opened = true;
            continue;
        }
        if trimmed.is_empty() {
            continue;
        }
        if line.contains('→') {
            entries.push(trimmed.to_string());
        } else if let Some(last) = entries.last_mut() {
            last.push(' ');
            last.push_str(trimmed);
        }
    }
    for entry in entries {
        let (lhs, rhs) = entry.split_once('→').expect("每条边都有箭头");
        let froms: Vec<String> = lhs.split('/').map(short).collect();
        let (normal, dev_part) = match rhs.split_once("+dev:") {
            Some((normal, dev_part)) => (normal, dev_part),
            None => (rhs, ""),
        };
        // 「叶」标记只表示没有运行时出边；同一行仍可以登记 `+dev` 测试夹具边。
        let normal_is_leaf = normal.contains("（叶");
        for from in &froms {
            if !normal_is_leaf {
                for to in names(normal) {
                    edges.insert(Edge {
                        from: from.clone(),
                        to,
                        dev: false,
                    });
                }
            }
            for to in names(dev_part) {
                edges.insert(Edge {
                    from: from.clone(),
                    to,
                    dev: true,
                });
            }
        }
    }
    (edges, total, dev)
}

pub(crate) fn names(part: &str) -> Vec<String> {
    part.split(',')
        .map(|name| name.split('（').next().unwrap_or_default())
        .map(short)
        .filter(|name| !name.is_empty())
        .collect()
}

pub(crate) fn backticks(text: &str) -> Vec<String> {
    text.split('`')
        .skip(1)
        .step_by(2)
        .map(str::to_string)
        .collect()
}


pub(crate) fn count_after(text: &str, prefix: &str) -> usize {
    let start = text
        .find(prefix)
        .unwrap_or_else(|| panic!("文档缺 `{prefix}`"))
        + prefix.len();
    text[start..]
        .chars()
        .take_while(char::is_ascii_digit)
        .collect::<String>()
        .parse()
        .unwrap_or_else(|_| panic!("`{prefix}` 后不是数字"))
}

/// 所有权表：主人 → 它声称拥有的每个类型名。
pub(crate) fn owned_types() -> Vec<(String, String)> {
    let mut rows = Vec::new();
    let mut inside = false;
    for line in DOC.lines() {
        if line.starts_with("## 三、所有权表") {
            inside = true;
            continue;
        }
        if inside && line.starts_with("## ") {
            break;
        }
        let trimmed = line.trim();
        if !inside || !trimmed.starts_with('|') || trimmed.starts_with("|---") {
            continue;
        }
        let cells: Vec<&str> = trimmed.split('|').map(str::trim).collect();
        if cells.len() < 4 {
            continue;
        }
        let Some(owner) = backticks(cells[2]).into_iter().next() else {
            continue;
        };
        for name in backticks(cells[1]) {
            rows.push((short(&owner), name));
        }
    }
    rows
}

/// 工作区成员与成员之间的依赖边（`cargo metadata` 是唯一事实源）。
///
/// 只取 `packages[].name` / `packages[].dependencies[].{name,kind}` 四个字段：成员集合来自前者，
/// 边来自后者（`kind == "dev"` 即 `+dev` 边）。解析交给 `serde_json`，不自己写第二份 JSON 文法。
pub(crate) fn workspace() -> (BTreeSet<String>, BTreeSet<Edge>) {
    let cargo = std::env::var_os("CARGO").unwrap_or_else(|| "cargo".into());
    let output = Command::new(cargo)
        .args(["metadata", "--no-deps", "--format-version", "1"])
        .current_dir(workspace_root())
        .output()
        .expect("cargo metadata 可执行");
    assert!(
        output.status.success(),
        "cargo metadata 失败：{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let metadata: serde_json::Value =
        serde_json::from_slice(&output.stdout).expect("`cargo metadata` 输出 JSON");
    let packages = metadata["packages"].as_array().expect("metadata 有 packages");
    let named = |value: &serde_json::Value| short(value["name"].as_str().expect("包名"));
    let members: BTreeSet<String> = packages.iter().map(named).collect();
    let mut edges = BTreeSet::new();
    for package in packages {
        let from = named(package);
        for dependency in package["dependencies"]
            .as_array()
            .expect("每个包都有 dependencies")
        {
            let to = named(dependency);
            if !members.contains(&to) {
                continue;
            }
            edges.insert(Edge {
                from: from.clone(),
                to,
                dev: dependency["kind"].as_str() == Some("dev"),
            });
        }
    }
    (members, edges)
}
}
