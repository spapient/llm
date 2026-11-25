//! 架构契约的事实面：`docs/ARCHITECTURE.md` 的三张表与 `cargo metadata` 的成员/边。

use std::borrow::Borrow;
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::{Arc, Mutex, OnceLock};

/// 仓根：本模块住在 `crates/<crate>/src/`，仓根是清单目录上两级。
pub fn workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

/// 目录扫描与文本读取共享缓存；后续门禁只克隆 `Arc` 句柄，不复制内容。
pub fn rust_files(dir: &Path) -> Arc<[PathBuf]> {
    with_cache(|cache| {
        cached(&mut cache.dirs, dir, || {
            let mut found = Vec::new();
            walk(dir, &mut found);
            found.into()
        })
    })
}

/// 读一份源文件；读不到时返回 `None`。
pub fn source_text(path: &Path) -> Option<Arc<str>> {
    with_cache(|cache| {
        cached(&mut cache.texts, path, || {
            std::fs::read_to_string(path).ok().map(Arc::from)
        })
    })
}

/// 读命中时只克隆值句柄；未命中时计算一次并写回同一张缓存表。
fn cached<K, Q, V>(map: &mut BTreeMap<K, V>, key: &Q, load: impl FnOnce() -> V) -> V
where
    K: Borrow<Q> + Ord,
    Q: Ord + ToOwned<Owned = K> + ?Sized,
    V: Clone,
{
    map.get(key).cloned().unwrap_or_else(|| {
        let value = load();
        map.insert(key.to_owned(), value.clone());
        value
    })
}

/// 扫描缓存：门禁分散在多个模块里，扫的是同一批目录、读的是同一批文件。
#[derive(Default)]
struct SourceCache {
    dirs: BTreeMap<PathBuf, Arc<[PathBuf]>>,
    texts: BTreeMap<PathBuf, Option<Arc<str>>>,
}

fn with_cache<T>(body: impl FnOnce(&mut SourceCache) -> T) -> T {
    static CACHE: OnceLock<Mutex<SourceCache>> = OnceLock::new();
    let cache = CACHE.get_or_init(|| Mutex::new(SourceCache::default()));
    // 门禁里的断言会 panic，但都不在持锁期间：这里按中毒恢复处理，不让一个门禁的失败
    // 连带把其余门禁判成"缓存中毒"。
    let mut guard = cache
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    body(&mut guard)
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

/// `llm-nn` 与文档里的 `nn` 指同一个成员（包名带 `llm-` 前缀，层表用短名）。
pub(crate) fn short(name: &str) -> String {
    let name = name.trim();
    name.strip_prefix("llm-").unwrap_or(name).to_string()
}

/// 依赖 `dependency` 的成员集合（只算普通边，`+dev` 不算）。
pub(crate) fn dependents(dependency: &str) -> BTreeSet<String> {
    workspace()
        .1
        .iter()
        .filter(|edge| !edge.dev && edge.to == dependency)
        .map(|edge| edge.from.clone())
        .collect()
}

/// 成员短名 → 源码根目录。
pub(crate) fn src_root(member: &str) -> PathBuf {
    let dir = dirs()
        .get(member)
        .map_or_else(|| PathBuf::from("crates").join(member), |path| path.clone());
    workspace_root().join(dir).join("src")
}

/// 成员短名 → 相对仓根的清单目录（`crates/gemm`）。
fn dirs() -> &'static BTreeMap<String, PathBuf> {
    static CACHE: OnceLock<BTreeMap<String, PathBuf>> = OnceLock::new();
    CACHE.get_or_init(|| {
        // `workspace_root()` 含 `../..`，与 `cargo metadata` 给的真实路径不是词法前缀，
        // 不归一化的话 `strip_prefix` 恒失败、映射恒回落成 `crates/<短名>`——即静默失效。
        let root = std::fs::canonicalize(workspace_root()).unwrap_or_else(|error| {
            panic!("仓根可归一化：{error}");
        });
        let mut dirs = BTreeMap::new();
        for package in packages() {
            let name = package["name"].as_str().expect("包名");
            let manifest = package["manifest_path"].as_str().expect("清单路径");
            let dir = Path::new(manifest)
                .parent()
                .and_then(|parent| parent.strip_prefix(&root).ok())
                .map_or_else(
                    || PathBuf::from("crates").join(short(name)),
                    Path::to_path_buf,
                );
            dirs.insert(short(name), dir);
        }
        dirs
    })
}

/// DOC 层表只解析一次；所有边界门禁共享同一份只读契约。
pub(crate) fn contract() -> &'static Contract {
    static CACHE: OnceLock<Contract> = OnceLock::new();
    CACHE.get_or_init(parse_contract)
}

fn parse_contract() -> Contract {
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
        .map_or_default(|line| {
            backticks(line)
                .into_iter()
                .map(|name| short(&name))
                .collect()
        });
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
        let sources: Vec<String> = lhs.split('/').map(short).collect();
        let (normal, dev_part) = match rhs.split_once("+dev:") {
            Some((normal, dev_part)) => (normal, dev_part),
            None => (rhs, ""),
        };
        // 「叶」标记只表示没有运行时出边；同一行仍可以登记 `+dev` 测试夹具边。
        let normal_is_leaf = normal.contains("（叶");
        for source in &sources {
            if !normal_is_leaf {
                for to in names(normal) {
                    edges.insert(Edge {
                        from: source.clone(),
                        to,
                        dev: false,
                    });
                }
            }
            for to in names(dev_part) {
                edges.insert(Edge {
                    from: source.clone(),
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

/// 工作区成员与依赖边只读共享；调用方不得为了一次遍历克隆整张图。
pub(crate) fn workspace() -> &'static (BTreeSet<String>, BTreeSet<Edge>) {
    metadata()
}

fn metadata() -> &'static (BTreeSet<String>, BTreeSet<Edge>) {
    static CACHE: OnceLock<(BTreeSet<String>, BTreeSet<Edge>)> = OnceLock::new();
    CACHE.get_or_init(load_workspace)
}

/// `cargo metadata` 的包数组：一次调用，成员表、边集与目录映射共用同一份事实。
fn packages() -> &'static [serde_json::Value] {
    static CACHE: OnceLock<Vec<serde_json::Value>> = OnceLock::new();
    CACHE.get_or_init(|| {
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
        metadata["packages"]
            .as_array()
            .expect("metadata 有 packages")
            .clone()
    })
}

fn load_workspace() -> (BTreeSet<String>, BTreeSet<Edge>) {
    let packages = packages();
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

/// 层内顺序来自 DOC 表；同层只允许依赖排在自己前面的成员。
fn strip(members: &BTreeSet<String>, edges: &BTreeSet<Edge>) -> BTreeSet<Edge> {
    edges
        .iter()
        .filter(|edge| members.contains(&edge.from) && members.contains(&edge.to))
        .cloned()
        .collect()
}

#[test]
fn every_workspace_member_is_in_the_table() {
    let (members, _) = workspace();
    let contract = contract();
    let missing: Vec<&String> = members
        .iter()
        .filter(|name| !contract.layers.contains_key(*name) && !contract.tools.contains(*name))
        .collect();
    assert!(
        missing.is_empty(),
        "有工作区成员不在 docs/ARCHITECTURE.md 的层表里：{missing:?}"
    );
}

#[test]
fn every_member_in_the_table_is_a_workspace_member() {
    let (members, _) = workspace();
    let stale: Vec<&String> = contract()
        .layers
        .keys()
        .filter(|name| !members.contains(*name))
        .collect();
    assert!(stale.is_empty(), "层表里有不是工作区成员的条目：{stale:?}");
}

#[test]
fn documented_edges_and_counts_match_cargo_metadata() {
    let (declared, total, dev) = declared_edges();
    let (_, actual) = workspace();
    assert_eq!(
        declared, *actual,
        "docs/ARCHITECTURE.md 的允许边与 cargo metadata 不一致"
    );
    assert_eq!(total, actual.len(), "文档的边总数");
    assert_eq!(
        dev,
        actual.iter().filter(|edge| edge.dev).count(),
        "文档的 +dev 边数"
    );
}

#[test]
fn runtime_edges_only_point_to_the_same_or_a_lower_layer() {
    let (_, edges) = workspace();
    let contract = contract();
    for edge in edges.iter().filter(|edge| !edge.dev) {
        let from = contract.layers.get(&edge.from);
        let to = contract.layers.get(&edge.to);
        let (Some(from), Some(to)) = (from, to) else {
            continue;
        };
        assert!(
            to.0 < from.0 || (to.0 == from.0 && to.1 <= from.1),
            "非法边：{} (L{}) → {} (L{})",
            edge.from,
            from.0,
            edge.to,
            to.0
        );
    }
}

#[test]
fn a_cross_layer_dev_dependency_never_hides_a_layer_violation() {
    let (members, edges) = workspace();
    let contract = contract();
    let stripped = strip(members, edges);
    for edge in stripped {
        let Some(from) = contract.layers.get(&edge.from) else {
            continue;
        };
        let Some(to) = contract.layers.get(&edge.to) else {
            continue;
        };
        assert!(
            to.0 < from.0 || (to.0 == from.0 && to.1 <= from.1),
            "层间禁止依赖（含 dev）：{} → {}",
            edge.from,
            edge.to
        );
    }
}
