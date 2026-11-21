//! 所有权表：`docs/ARCHITECTURE.md` 第三节与源码的**双向**核对。
//!
//! 一条方向「表里写的必须真的 `pub`」，另一条「源码里的 `pub` 类型必须在表里或基线里」；
//! 再加一条组合根判据：`cli` 是编排入口，不得定义新类型。三条合起来，所有权表既不能过期，
//! 也不能漏项——只写一侧的话，漂移会从另一侧静默穿过。

use std::collections::{BTreeMap, BTreeSet};

use crate::architecture::{
    contract, owned_types, rust_files, workspace, workspace_root,
};

/// 一行顶层的 `struct|enum|trait|union` 定义，返回（是否是 `pub`，名字）。
pub(crate) fn type_definition(line: &str) -> Option<(bool, &str)> {
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
pub(crate) fn public_types(files: &[std::path::PathBuf]) -> BTreeSet<String> {
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
pub(crate) fn macro_block_name(line: &str) -> Option<&str> {
    let name = line.split('{').next()?.trim();
    (!name.is_empty() && name.chars().next()?.is_ascii_uppercase()).then_some(name)
}

/// `lints/vocabulary.toml` 里的反向所有权存量基线：crate → 登记为「暂未进表」的公开类型名。
///
/// 读 `llm_tool` 构建期生成的词汇表：同一份文件只解析一次，运行期没有解析器；与
/// `tests/contracts.rs` 读的是同一棵树，不另写一份。
pub(crate) fn unlisted_baseline() -> BTreeMap<String, BTreeSet<String>> {
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

#[test]
pub(crate) fn every_owned_type_is_public_in_the_claimed_crate() {
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
pub(crate) fn every_public_type_is_listed_or_baselined() {
    let contract = contract();
    let owned: BTreeSet<(String, String)> = owned_types().into_iter().collect();
    let listed: BTreeSet<String> = owned.iter().map(|(_, name)| name.clone()).collect();
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
            // 过期判定按「主人 + 名字」认，不能只认名字：`prof::Witness` 与 `testkit::Witness`
            // 是两个类型，只看名字会让另一个 crate 进表时静默注销本条基线，等于偷偷放宽门禁。
            // 反过来，上面的未登记判定仍只认名字——存量表里有 42 个历史缺口靠名字对齐，
            // 收紧成限定名要重写整张表，属于另一件事。
            if owned.contains(&(member.clone(), name.clone())) {
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
pub(crate) fn the_entry_point_defines_no_types() {
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

/// 一行 `pub` 定义的名字；不是 `pub` 定义就是 `None`。
pub(crate) fn public_item_name(line: &str) -> Option<&str> {
    const KINDS: [&str; 8] = [
        "struct", "enum", "trait", "type", "fn", "const", "union", "mod",
    ];
    let rest = line.trim_start().strip_prefix("pub ")?;
    let rest = KINDS.iter().find_map(|kind| {
        rest.trim_start()
            .strip_prefix(kind)
            .and_then(|tail| tail.strip_prefix(' '))
    })?;
    let name = rest.trim_start();
    let end = name
        .find(|value: char| !(value.is_alphanumeric() || value == '_'))
        .unwrap_or(name.len());
    (!name.is_empty()).then(|| &name[..end])
}

/// `name` 是否在 `crate_name` 的源码里以 `pub` 形式定义。
pub(crate) fn declares_public(crate_name: &str, name: &str) -> bool {
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

pub(crate) fn is_public_definition(line: &str, name: &str, kinds: &[&str]) -> bool {
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
