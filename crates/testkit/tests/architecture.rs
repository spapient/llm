//! 架构契约的可执行版本（`docs/ARCHITECTURE.md`）。
//!
//! 层表、依赖边与所有权表过去靠评审清单核对，30 个成员 × 每次改动 = 必然漂移。
//! 这里把 `cargo metadata` 的事实与文档里的三张表逐条比对：成员集合、边集合与方向、
//! 每个类型的 `pub` 定义都必须对得上，改文档或改 `Cargo.toml` 而不同步另一侧即失败。
//!
//! 只读 `cargo metadata` 与源码，不修改任何东西；`cargo test` 因此就是守门人。

use std::collections::{BTreeMap, BTreeSet};
use std::process::Command;

/// 契约文本编译期嵌入：测试不依赖当前工作目录，也不会读到另一份副本。
const DOC: &str = include_str!("../../../docs/ARCHITECTURE.md");

/// 仓根：本测试住在 `crates/testkit/tests/`，仓根是清单目录上两级。
fn workspace_root() -> std::path::PathBuf {
    std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

/// 层表：成员 → (层号, 层内位置)；同层位置决定“可以依赖谁”。
struct Contract {
    layers: BTreeMap<String, (usize, usize)>,
    tools: BTreeSet<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct Edge {
    from: String,
    to: String,
    dev: bool,
}

#[test]
fn the_layer_table_lists_exactly_the_workspace_members() {
    let (members, _) = workspace();
    let contract = contract();
    let listed: BTreeSet<String> = contract
        .layers
        .keys()
        .chain(contract.tools.iter())
        .cloned()
        .collect();
    assert_eq!(
        members,
        listed,
        "层表与 Cargo.toml 的成员不一致：多 {:?}，缺 {:?}",
        members.difference(&listed).collect::<Vec<_>>(),
        listed.difference(&members).collect::<Vec<_>>()
    );
    let overlap: Vec<&String> = contract
        .tools
        .iter()
        .filter(|tool| contract.layers.contains_key(*tool))
        .collect();
    assert!(
        overlap.is_empty(),
        "工具成员不能同时出现在运行时层里：{overlap:?}"
    );
}

#[test]
fn the_declared_edges_are_exactly_the_workspace_edges() {
    let (_, actual) = workspace();
    let (declared, total, dev) = declared_edges();
    let missing: Vec<&Edge> = actual.difference(&declared).collect();
    let extra: Vec<&Edge> = declared.difference(&actual).collect();
    assert!(
        missing.is_empty() && extra.is_empty(),
        "文档与 cargo metadata 的边集不一致：文档缺 {missing:?}，文档多 {extra:?}"
    );
    assert_eq!(
        actual.len(),
        total,
        "文档声称共 {total} 条边，实际 {} 条",
        actual.len()
    );
    let actual_dev = actual.iter().filter(|edge| edge.dev).count();
    assert_eq!(
        actual_dev, dev,
        "文档声称 {dev} 条 +dev 边，实际 {actual_dev} 条"
    );
}

#[test]
fn every_edge_points_to_a_lower_layer_or_to_the_left() {
    let (_, edges) = workspace();
    let contract = contract();
    for edge in &edges {
        // 工具成员没有运行时层级，它们的边不参与方向判定（ARCHITECTURE §一）。
        // `+dev` 边只服务测试/夹具，不进任何产品二进制，同样不约束运行时分层方向。
        if edge.dev || contract.tools.contains(&edge.from) || contract.tools.contains(&edge.to) {
            continue;
        }
        let from = contract
            .layers
            .get(&edge.from)
            .unwrap_or_else(|| panic!("层表缺成员 {}", edge.from));
        let to = contract
            .layers
            .get(&edge.to)
            .unwrap_or_else(|| panic!("层表缺成员 {}", edge.to));
        assert!(
            to.0 < from.0 || (to.0 == from.0 && to.1 < from.1),
            "{} → {} 违反依赖方向：只能指向更低的层或同层更靠左的成员",
            edge.from,
            edge.to
        );
    }
}

#[test]
fn every_member_is_consumed_or_consumes_something() {
    let (members, edges) = workspace();
    let contract = contract();
    let used: BTreeSet<&String> = edges
        .iter()
        .flat_map(|edge| [&edge.from, &edge.to])
        .collect();
    for member in &members {
        if contract.tools.contains(member) {
            continue;
        }
        assert!(
            used.contains(member),
            "{member} 没有任何依赖边：它只是一个目录，不是模块（ARCHITECTURE §五）"
        );
    }
}

#[test]
fn every_owned_type_is_public_in_the_claimed_crate() {
    let (members, _) = workspace();
    let contract = contract();
    for (owner, name) in owned_types() {
        assert!(
            contract.layers.contains_key(&owner) || contract.tools.contains(&owner),
            "所有权表的主人 {owner} 不在层表里"
        );
        assert!(
            members.contains(&owner),
            "所有权表的主人 {owner} 不是工作区成员"
        );
        assert!(
            declares_public(&owner, &name),
            "所有权表称 {name} 属于 {owner}，但 {owner} 的源码里没有 `pub` 定义"
        );
    }
}

/// 所有权表必须是**双向**的：上一条只保证「表里写的在源码里 `pub`」，这条保证反方向——
/// 每个成员源码里的 `pub struct|enum|trait` 要么进表，要么登记在 `lints/vocabulary.toml`
/// 的 `[[ownership.unlisted]]` 存量基线里。
///
/// 基线**不允许过期**（与 `[closed]` 白名单同一判据）：某个名字后来进了表、或不再是 `pub`，
/// 对应条目就必须删掉，否则「存量冻结」会慢慢退化成「表永远落后于代码」——那正是这条门禁
/// 之前发现不了 `Chunks` 的原因。
#[test]
fn every_public_type_is_listed_or_baselined() {
    let contract = contract();
    let listed: BTreeSet<String> = owned_types().into_iter().map(|(_, name)| name).collect();
    let baseline = unlisted_baseline();

    let mut unlisted = Vec::new();
    for member in contract.layers.keys().chain(contract.tools.iter()) {
        let root = workspace_root().join("crates").join(member).join("src");
        let files = rust_files(&root);
        if contract.layers.contains_key(member) {
            assert!(
                !files.is_empty(),
                "层表成员 `{member}` 在 {} 下一个源文件都没有：扫描集由层表派生，这里必须扫得到东西",
                root.display()
            );
        }
        for name in public_types(&files) {
            if listed.contains(&name) {
                continue;
            }
            if baseline
                .get(member)
                .is_some_and(|names| names.contains(&name))
            {
                continue;
            }
            unlisted.push(format!("{member}:{name}"));
        }
    }
    assert!(
        unlisted.is_empty(),
        "这些公开类型既不在 `docs/ARCHITECTURE.md` 第三节的所有权表里，也没登记在 \
         `lints/vocabulary.toml` 的 `[[ownership.unlisted]]`：{unlisted:?}。\
         新类型应进表；存量缺口才登记为基线，并写明理由"
    );

    let mut stale = Vec::new();
    for (member, names) in &baseline {
        let root = workspace_root().join("crates").join(member).join("src");
        let present = public_types(&rust_files(&root));
        for name in names {
            if listed.contains(name) {
                stale.push(format!("{member}:{name} 已进所有权表"));
            } else if !present.contains(name) {
                stale.push(format!(
                    "{member}:{name} 不再是源码里的 `pub struct|enum|trait`"
                ));
            }
        }
    }
    assert!(
        stale.is_empty(),
        "存量基线过期，必须删掉这些条目（基线与 `[closed]` 白名单同一判据：白名单不能过期）：{stale:?}"
    );
}

/// 组合根是 `cli`：编排入口「不得定义新类型」。
///
/// 上一个组装点里的类型（`LmHead`）就是这条规则失效的证据：它既不在所有权表里，也不违反
/// 任何既有断言。类型属于产出它的层，编排入口只分派。
#[test]
fn the_entry_point_defines_no_types() {
    let cli_src = workspace_root().join("crates/cli/src");
    let files = ["run.rs", "status.rs", "train.rs"]
        .iter()
        .map(|name| cli_src.join(name))
        .collect::<Vec<_>>();
    assert!(
        !files.is_empty(),
        "组装点的源码必须扫得到：{}",
        cli_src.display()
    );
    let mut defined = Vec::new();
    for path in &files {
        let text = std::fs::read_to_string(path).expect("源文件可读");
        for (index, line) in text.lines().enumerate() {
            // 只拦对外公开的领域类型；模块内私有枚举/结构体是编排实现细节，允许保留。
            if let Some((public, _)) = type_definition(line)
                && public
            {
                defined.push(format!("{}:{}", path.display(), index + 1));
            }
        }
    }
    assert!(
        defined.is_empty(),
        "L7 组装点不得定义类型（把它放回所属的层）：{defined:?}"
    );
}

/// 一行顶层的 `struct|enum|trait|union` 定义，返回（是否是 `pub`，名字）。
fn type_definition(line: &str) -> Option<(bool, &str)> {
    let trimmed = line.trim_start();
    let (public, rest) = match trimmed.strip_prefix("pub ") {
        Some(rest) => (true, rest.trim_start()),
        None => (false, trimmed),
    };
    let rest = ["struct", "enum", "trait", "union"]
        .iter()
        .find_map(|kind| {
            rest.strip_prefix(kind)
                .and_then(|tail| tail.strip_prefix(' '))
        })?;
    let name = rest.trim_start();
    let end = name
        .find(|value: char| !(value.is_alphanumeric() || value == '_'))
        .unwrap_or(name.len());
    let name = &name[..end];
    (!name.is_empty()).then_some((public, name))
}

/// `files` 里出现的 `pub struct|enum|trait` 名字（`union` 不在所有权表的口径内）。
///
/// `vocabulary! { Name { ... } }` 宏展开出的 `pub enum`：名字在宏块后一行、行首没有 `pub`，
/// 逐行正则看不到，这里按宏块结构补一份识别（prof 的四个闭集都走这个生成器）。
fn public_types(files: &[std::path::PathBuf]) -> BTreeSet<String> {
    let mut names = BTreeSet::new();
    for path in files {
        let Ok(text) = std::fs::read_to_string(path) else {
            continue;
        };
        let mut in_vocabulary = false;
        for line in text.lines() {
            if let Some((true, name)) = type_definition(line) {
                names.insert(name.to_string());
            }
            let trimmed = line.trim();
            if trimmed.starts_with("vocabulary!") {
                in_vocabulary = true;
                continue;
            }
            if in_vocabulary && !trimmed.is_empty() && !trimmed.starts_with("///") {
                if let Some(name) = macro_block_name(trimmed) {
                    names.insert(name.to_string());
                }
                in_vocabulary = false;
            }
        }
    }
    names
}

/// 宏块开头行 `Name {`：取 `{` 前的标识符，首字母大写才算类型名。
fn macro_block_name(line: &str) -> Option<&str> {
    let name = line.split('{').next()?.trim();
    (!name.is_empty() && name.chars().next()?.is_ascii_uppercase()).then_some(name)
}

/// `lints/vocabulary.toml` 里的反向所有权存量基线：crate → 登记为「暂未进表」的公开类型名。
///
/// 读 `llm_tool` 构建期生成的词汇表：同一份文件只解析一次，运行期没有解析器；与
/// `tests/contracts.rs` 读的是同一棵树，不另写一份。
fn unlisted_baseline() -> BTreeMap<String, BTreeSet<String>> {
    let entries = tool::vocabulary()
        .table("ownership")
        .and_then(|ownership| ownership.table_array("unlisted"))
        .unwrap_or_default();
    let mut baseline: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    for entry in entries {
        let crate_name = entry
            .string("crate")
            .unwrap_or_else(|| panic!("`ownership.unlisted` 的条目缺 `crate`"));
        let reason = entry
            .string("reason")
            .unwrap_or_else(|| panic!("`ownership.unlisted` 的 `{crate_name}` 缺 `reason`"));
        assert!(
            reason.chars().count() >= 8,
            "`ownership.unlisted` 的 `{crate_name}` 理由太短：一份没有理由的豁免等于把规则删掉一半"
        );
        let names = entry
            .string_array("names")
            .unwrap_or_else(|| panic!("`ownership.unlisted` 的 `{crate_name}` 缺 `names`"));
        assert!(
            !names.is_empty(),
            "`ownership.unlisted` 的 `{crate_name}` 一个名字都没登记"
        );
        let registered = baseline.entry(crate_name.to_string()).or_default();
        for name in names {
            assert!(
                registered.insert(name.to_string()),
                "`{name}` 在 `ownership.unlisted` 的 `{crate_name}` 里重复登记"
            );
        }
    }
    baseline
}

/// 允许用 `String` 做**消息片段**的文件（不是"允许用 `String` 报错"）。
///
/// 现在是空的：唯一一份"把失败原因写成字符串"的实现是手写 TOML 解析器，它随 ADR 0022 一起
/// 删除（语料改在构建期解析）。留着这张表是为了让下一条同类豁免有地方登记，而不是散在断言里。
const MESSAGE_FRAGMENTS: &[&str] = &[];

/// 错误边界不得退回字符串（`docs/ARCHITECTURE.md` 第三节）：凡是被 L7 直接调用的层——数据层
/// （`op` / `prof` / `prim`）、内核层（`cutile`，含 `report`）——失败类型必须是
/// `diagnostic::Error`，否则类别与退出码就失去了来源。
///
/// 扫描集**从层表派生**，不手写目录名：手写的清单在成员被合并或删除后会指向不存在的目录，
/// 而 [`rust_files`] 对不存在的目录返回空——那会让本测试在"少扫一个成员"时静默通过。
/// 反过来，层表里每个成员都必须扫出至少一个文件，否则同样失败。
///
/// 扫描的是"`Result<…, String>`"这一形态本身，不是某个函数：新加的函数同样被覆盖。
/// 豁免表 [`MESSAGE_FRAGMENTS`] 现在是空的：随着手写 TOML 解析器删除，最后一处"把失败原因
/// 写成字符串"的实现也没有了。
#[test]
fn the_error_boundary_does_not_fall_back_to_strings() {
    let contract = contract();
    let mut paths = Vec::new();
    for member in contract.layers.keys() {
        let root = workspace_root().join("crates").join(member).join("src");
        let files = rust_files(&root);
        assert!(
            !files.is_empty(),
            "层表成员 `{member}` 在 {} 下一个源文件都没有：扫描集由层表派生，这里必须扫得到东西，\
             否则合并/删除成员会让本测试静默通过",
            root.display()
        );
        paths.extend(files);
    }
    let cli_src = workspace_root().join("crates/cli/src");
    for entry in ["cutile_bin.rs", "cutile_report.rs"] {
        paths.push(cli_src.join(entry));
    }

    let mut offences = Vec::new();
    for path in paths {
        if MESSAGE_FRAGMENTS
            .iter()
            .any(|exempt| path == std::path::Path::new(exempt))
        {
            continue;
        }
        let text = std::fs::read_to_string(&path).expect("源文件可读");
        for (index, line) in text.lines().enumerate() {
            if line.trim_start().starts_with("//") {
                continue;
            }
            // 只认错误位是 `String` 的形态；`Result<Vec<String>, Error>` 这类是合法的。
            if line.contains("Result<") && line.contains(", String>") {
                offences.push(format!("{}:{}", path.display(), index + 1));
            }
        }
    }
    assert!(
        offences.is_empty(),
        "这些位置仍以 `String` 报错，应改为 `diagnostic::Error`：{offences:?}"
    );
}

/// 递归列出 `dir` 下的 `.rs` 文件（顺序无关，只用于扫描）。
fn rust_files(dir: &std::path::Path) -> Vec<std::path::PathBuf> {
    let mut found = Vec::new();
    let Ok(entries) = std::fs::read_dir(dir) else {
        return found;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            found.extend(rust_files(&path));
        } else if path.extension().and_then(|value| value.to_str()) == Some("rs") {
            found.push(path);
        }
    }
    found
}

/// `llm-op` 与文档里的 `op` 指同一个成员（包名带 `llm-` 前缀，层表用短名）。
fn short(name: &str) -> String {
    let name = name.trim();
    name.strip_prefix("llm-").unwrap_or(name).to_string()
}

fn contract() -> Contract {
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
fn declared_edges() -> (BTreeSet<Edge>, usize, usize) {
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

fn names(part: &str) -> Vec<String> {
    part.split(',')
        .map(|name| name.split('（').next().unwrap_or_default())
        .map(short)
        .filter(|name| !name.is_empty())
        .collect()
}

fn backticks(text: &str) -> Vec<String> {
    text.split('`')
        .skip(1)
        .step_by(2)
        .map(str::to_string)
        .collect()
}

fn count_after(text: &str, prefix: &str) -> usize {
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
fn owned_types() -> Vec<(String, String)> {
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
fn workspace() -> (BTreeSet<String>, BTreeSet<Edge>) {
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
    let metadata = json::parse(std::str::from_utf8(&output.stdout).expect("元数据是 UTF-8"));
    let packages = metadata
        .get("packages")
        .and_then(json::Value::as_array)
        .expect("metadata 有 packages");
    let members: BTreeSet<String> = packages
        .iter()
        .map(|package| {
            short(
                package
                    .get("name")
                    .and_then(json::Value::as_str)
                    .expect("包名"),
            )
        })
        .collect();
    let mut edges = BTreeSet::new();
    for package in packages {
        let from = short(
            package
                .get("name")
                .and_then(json::Value::as_str)
                .expect("包名"),
        );
        for dependency in package
            .get("dependencies")
            .and_then(json::Value::as_array)
            .expect("每个包都有 dependencies")
        {
            let to = short(
                dependency
                    .get("name")
                    .and_then(json::Value::as_str)
                    .expect("依赖名"),
            );
            if !members.contains(&to) {
                continue;
            }
            let dev = dependency.get("kind").and_then(json::Value::as_str) == Some("dev");
            edges.insert(Edge {
                from: from.clone(),
                to,
                dev,
            });
        }
    }
    (members, edges)
}

/// `name` 是否在 `crate_name` 的源码里以 `pub` 形式定义。
fn declares_public(crate_name: &str, name: &str) -> bool {
    const KINDS: [&str; 8] = [
        "struct", "enum", "trait", "type", "fn", "const", "union", "mod",
    ];
    let root = workspace_root().join("crates").join(crate_name).join("src");
    let mut pending = vec![root];
    while let Some(directory) = pending.pop() {
        let Ok(entries) = std::fs::read_dir(directory) else {
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                pending.push(path);
            } else if path.extension().is_some_and(|extension| extension == "rs") {
                let Ok(text) = std::fs::read_to_string(&path) else {
                    continue;
                };
                if text
                    .lines()
                    .any(|line| is_public_definition(line, name, &KINDS))
                {
                    return true;
                }
            }
        }
    }
    false
}

fn is_public_definition(line: &str, name: &str, kinds: &[&str]) -> bool {
    let Some(rest) = line.trim_start().strip_prefix("pub ") else {
        return false;
    };
    let rest = rest.trim_start();
    kinds.iter().any(|kind| {
        rest.strip_prefix(kind)
            .and_then(|tail| tail.strip_prefix(' '))
            .is_some_and(|tail| {
                tail.strip_prefix(name).is_some_and(|after| {
                    after
                        .chars()
                        .next()
                        .is_none_or(|next| !(next.is_alphanumeric() || next == '_'))
                })
            })
    })
}

/// `cargo metadata` 只输出 JSON；本测试只取四个字段，因此用手写解析器避免引入 `serde`。
mod json {
    /// JSON 值：只保留解析与取值需要的能力。
    #[derive(Debug, Clone, PartialEq)]
    pub enum Value {
        /// `null`。
        Null,
        /// `true` / `false`。
        Bool(bool),
        /// 数字（本测试不读取数值）。
        Number(f64),
        /// 字符串。
        String(String),
        /// 数组。
        Array(Vec<Value>),
        /// 对象（保序，便于诊断）。
        Object(Vec<(String, Value)>),
    }

    impl Value {
        /// 按字段名取值；非对象返回 `None`。
        pub fn get(&self, key: &str) -> Option<&Value> {
            match self {
                Self::Object(fields) => fields
                    .iter()
                    .find(|(field, _)| field == key)
                    .map(|(_, value)| value),
                _ => None,
            }
        }

        /// 字符串视图。
        pub fn as_str(&self) -> Option<&str> {
            match self {
                Self::String(text) => Some(text),
                _ => None,
            }
        }

        /// 数组视图。
        pub fn as_array(&self) -> Option<&[Value]> {
            match self {
                Self::Array(items) => Some(items),
                _ => None,
            }
        }
    }

    /// 解析一段完整 JSON。
    pub fn parse(text: &str) -> Value {
        let mut parser = Parser {
            bytes: text.as_bytes(),
            at: 0,
        };
        let value = parser.value();
        parser.skip_whitespace();
        assert_eq!(parser.at, parser.bytes.len(), "JSON 末尾有多余内容");
        value
    }

    struct Parser<'a> {
        bytes: &'a [u8],
        at: usize,
    }

    impl Parser<'_> {
        fn skip_whitespace(&mut self) {
            while matches!(self.bytes.get(self.at), Some(b' ' | b'\n' | b'\r' | b'\t')) {
                self.at += 1;
            }
        }

        fn value(&mut self) -> Value {
            self.skip_whitespace();
            match self.bytes.get(self.at).copied() {
                Some(b'{') => self.object(),
                Some(b'[') => self.array(),
                Some(b'"') => Value::String(self.string()),
                Some(b't') => {
                    self.literal("true");
                    Value::Bool(true)
                }
                Some(b'f') => {
                    self.literal("false");
                    Value::Bool(false)
                }
                Some(b'n') => {
                    self.literal("null");
                    Value::Null
                }
                Some(_) => self.number(),
                None => panic!("JSON 意外结束"),
            }
        }

        fn object(&mut self) -> Value {
            self.at += 1;
            let mut fields = Vec::new();
            self.skip_whitespace();
            if self.bytes.get(self.at) == Some(&b'}') {
                self.at += 1;
                return Value::Object(fields);
            }
            loop {
                self.skip_whitespace();
                let key = self.string();
                self.skip_whitespace();
                self.expect(b':');
                fields.push((key, self.value()));
                self.skip_whitespace();
                match self.bytes.get(self.at) {
                    Some(b',') => self.at += 1,
                    Some(b'}') => {
                        self.at += 1;
                        return Value::Object(fields);
                    }
                    _ => panic!("JSON 对象缺少逗号或右花括号，位置 {}", self.at),
                }
            }
        }

        fn array(&mut self) -> Value {
            self.at += 1;
            let mut items = Vec::new();
            self.skip_whitespace();
            if self.bytes.get(self.at) == Some(&b']') {
                self.at += 1;
                return Value::Array(items);
            }
            loop {
                items.push(self.value());
                self.skip_whitespace();
                match self.bytes.get(self.at) {
                    Some(b',') => self.at += 1,
                    Some(b']') => {
                        self.at += 1;
                        return Value::Array(items);
                    }
                    _ => panic!("JSON 数组缺少逗号或右方括号，位置 {}", self.at),
                }
            }
        }

        fn string(&mut self) -> String {
            self.expect(b'"');
            let mut text = String::new();
            loop {
                match self.bytes.get(self.at).copied() {
                    None => panic!("JSON 字符串未闭合"),
                    Some(b'"') => {
                        self.at += 1;
                        return text;
                    }
                    Some(b'\\') => {
                        self.at += 1;
                        let escape = self.bytes.get(self.at).copied().expect("转义符后不应结束");
                        self.at += 1;
                        match escape {
                            b'"' => text.push('"'),
                            b'\\' => text.push('\\'),
                            b'/' => text.push('/'),
                            b'b' => text.push('\u{0008}'),
                            b'f' => text.push('\u{000c}'),
                            b'n' => text.push('\n'),
                            b'r' => text.push('\r'),
                            b't' => text.push('\t'),
                            b'u' => {
                                let hex = std::str::from_utf8(&self.bytes[self.at..self.at + 4])
                                    .expect("\\u 后是四个字节");
                                let code = u32::from_str_radix(hex, 16).expect("\\u 后是十六进制");
                                self.at += 4;
                                text.push(char::from_u32(code).expect("合法码点"));
                            }
                            other => panic!("未知转义 `\\{}`", char::from(other)),
                        }
                    }
                    Some(byte) => {
                        let width = utf8_width(byte);
                        let chunk = std::str::from_utf8(&self.bytes[self.at..self.at + width])
                            .expect("元数据是 UTF-8");
                        text.push_str(chunk);
                        self.at += width;
                    }
                }
            }
        }

        fn number(&mut self) -> Value {
            let start = self.at;
            while matches!(
                self.bytes.get(self.at),
                Some(b'-' | b'+' | b'0'..=b'9' | b'.' | b'e' | b'E')
            ) {
                self.at += 1;
            }
            let text = std::str::from_utf8(&self.bytes[start..self.at]).expect("数字是 ASCII");
            Value::Number(text.parse().unwrap_or_else(|_| panic!("非法数字 `{text}`")))
        }

        fn literal(&mut self, word: &str) {
            assert!(
                self.bytes[self.at..].starts_with(word.as_bytes()),
                "位置 {} 处不是 `{word}`",
                self.at
            );
            self.at += word.len();
        }

        fn expect(&mut self, byte: u8) {
            self.skip_whitespace();
            assert_eq!(
                self.bytes.get(self.at),
                Some(&byte),
                "位置 {} 处缺少 `{}`",
                self.at,
                char::from(byte)
            );
            self.at += 1;
        }
    }

    fn utf8_width(byte: u8) -> usize {
        match byte {
            0x00..=0x7f => 1,
            0xc0..=0xdf => 2,
            0xe0..=0xef => 3,
            _ => 4,
        }
    }
}
