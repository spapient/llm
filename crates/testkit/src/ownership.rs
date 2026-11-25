//! 所有权表：`docs/ARCHITECTURE.md` 第三节与源码的**双向**核对。

use std::collections::{BTreeMap, BTreeSet};

use crate::architecture::{
    contract, owned_types, rust_files, source_text, src_root, workspace, workspace_root,
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

/// 类型声明宏的形态：一个块声明单个类型，或一张表声明多个类型。
#[derive(Clone, Copy, PartialEq, Eq)]
enum TypeMacro {
    Single,
    Table,
}

/// 一段源码里的顶层 `pub` 类型名，包括声明宏展开出的公开类型。
/// `axis!` / `planar!` 表中的每一行都声明一个类型；`vocabulary!` / `closed_enum!` / `stock!`
/// 只声明一个。正文扫描与 `pub` 校验共用这一份口径，避免正向与反向所有权检查互相矛盾。
pub(crate) fn public_type_names(text: &str) -> BTreeSet<&str> {
    let mut names = BTreeSet::new();
    let mut macro_block = None;
    for line in text.lines() {
        if let Some((true, name)) = type_definition(line) {
            names.insert(name);
        }
        let trimmed = line.trim();
        if let Some(kind) = type_macro_open(trimmed) {
            match inline_type_macro_arg(trimmed) {
                Some(name) => {
                    names.insert(name);
                }
                None if trimmed.ends_with('{') || trimmed.ends_with('(') => {
                    macro_block = Some(kind);
                }
                None => {}
            }
            continue;
        }
        let Some(kind) = macro_block else {
            continue;
        };
        if kind == TypeMacro::Table && trimmed == "}" {
            macro_block = None;
            continue;
        }
        if trimmed.is_empty()
            || trimmed.starts_with("///")
            || trimmed.starts_with("//!")
            || trimmed.starts_with('#')
            || trimmed.starts_with('@')
        {
            continue;
        }
        if let Some(name) = macro_type_arg(trimmed) {
            names.insert(name);
        }
        if kind == TypeMacro::Single {
            macro_block = None;
        }
    }
    names
}

/// 展开成公开类型的声明宏；调用点可带 `crate::` / `$crate::` 前缀。
fn type_macro_open(trimmed: &str) -> Option<TypeMacro> {
    [
        ("vocabulary!", TypeMacro::Single),
        ("closed_enum!", TypeMacro::Single),
        ("stock!", TypeMacro::Single),
        ("axis!", TypeMacro::Table),
        ("planar!", TypeMacro::Table),
    ]
    .iter()
    .find_map(|(open, kind)| {
        trimmed
            .split_once(open)
            .is_some_and(|(head, _)| head.is_empty() || head.ends_with("::"))
            .then_some(*kind)
    })
}

/// 内联形式 `axis!(Name, ...)` 的 `Name`；块形式返回 `None`。
fn inline_type_macro_arg(trimmed: &str) -> Option<&str> {
    let open = trimmed.find('(')?;
    macro_type_arg(&trimmed[open + 1..])
}

/// `files` 里出现的 `pub struct|enum|trait|union` 名字，口径由 [`public_type_names`] 定义。
pub(crate) fn public_types(files: &[std::path::PathBuf]) -> BTreeSet<String> {
    let mut names = BTreeSet::new();
    for path in files {
        let Some(text) = source_text(path) else {
            continue;
        };
        names.extend(public_type_names(&text).into_iter().map(str::to_string));
    }
    names
}

/// 宏声明中的类型名：取行首标识符，首字母大写才算类型名。
pub(crate) fn macro_type_arg(line: &str) -> Option<&str> {
    let name = line.trim_start();
    let end = name
        .find(|value: char| !(value.is_alphanumeric() || value == '_'))
        .unwrap_or(name.len());
    let name = &name[..end];
    (!name.is_empty() && name.chars().next()?.is_ascii_uppercase()).then_some(name)
}

#[cfg(test)]
mod tests {
    use super::public_type_names;

    #[test]
    fn declaration_tables_expose_every_type() {
        let names = public_type_names(
            r#"axis! {
                /// 文档不属于类型名。
                RmsNorm => "rmsnorm_f32", Norm;
                Sigmoid => "sigmoid_f32";
                Tanh => "tanh_f32";
            }"#,
        );
        let expected = ["RmsNorm", "Sigmoid", "Tanh"].into_iter().collect();
        assert_eq!(names, expected);
    }

    #[test]
    fn enum_declarations_do_not_expose_variants() {
        let names = public_type_names(
            r#"closed_enum! { @named
                /// 枚举类型与变体必须分开。
                pub enum Reason {
                    Stored = 0 => "stored",
                    Coded = 1 => "coded",
                }
            }"#,
        );
        let expected = ["Reason"].into_iter().collect();
        assert_eq!(names, expected);
    }
}

/// `lints/vocabulary.toml` 里的反向所有权存量基线：crate → 登记为「暂未进表」的公开类型名。
pub(crate) fn unlisted_baseline() -> BTreeMap<String, BTreeSet<String>> {
    let entries = tool::vocabulary().ownership_unlisted;
    let mut baseline: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    for entry in entries {
        let crate_name = entry.crate_name;
        let reason = entry.reason;
        assert!(
            reason.chars().count() >= 8,
            "`ownership.unlisted` 的 `{crate_name}` 理由太短：一份没有理由的豁免等于把规则删掉一半"
        );
        assert!(
            !entry.names.is_empty(),
            "`ownership.unlisted` 的 `{crate_name}` 一个名字都没登记"
        );
        let registered = baseline.entry(crate_name.to_string()).or_default();
        for name in entry.names {
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
#[test]
pub(crate) fn every_public_type_is_listed_or_baselined() {
    let contract = contract();
    let owned: BTreeSet<(String, String)> = owned_types().into_iter().collect();
    let listed: BTreeSet<String> = owned.iter().map(|(_, name)| name.clone()).collect();
    let baseline = unlisted_baseline();

    let mut unlisted = Vec::new();
    for member in contract.layers.keys().chain(contract.tools.iter()) {
        let root = src_root(member);
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
        let root = src_root(member);
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
        for (idx, line) in text.lines().enumerate() {
            // 只拦对外公开的领域类型；模块内私有枚举/结构体是编排实现细节，允许保留。
            if let Some((public, _)) = type_definition(line)
                && public
            {
                defined.push(format!("{}:{}", path.display(), idx + 1));
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
    /// 所有权表除类型外还允许登记这几种公开条目（类型别名、函数、常量、模块）。
    const NON_TYPE_KINDS: [&str; 4] = ["type", "fn", "const", "mod"];
    for path in rust_files(&src_root(crate_name)).iter() {
        let Some(text) = source_text(&path) else {
            continue;
        };
        if public_type_names(&text).contains(name)
            || text
                .lines()
                .any(|line| is_public_definition(line, name, &NON_TYPE_KINDS))
        {
            return true;
        }
    }
    false
}

pub(crate) fn is_public_definition(line: &str, name: &str, kinds: &[&str]) -> bool {
    let Some(rest) = line.trim_start().strip_prefix("pub ") else {
        return false;
    };
    let rest = rest.trim_start();
    // `pub const fn name(..)` 在 kind 前还有一层 `const`：两个候选前缀都试，普通 `pub const NAME`
    // 仍由 `const` kind 命中，`const fn` 则剥掉 `const` 后由 `fn` kind 命中。
    let rest_no_const = rest.strip_prefix("const ").unwrap_or(rest);
    kinds.iter().any(|kind| {
        [rest, rest_no_const].iter().any(|candidate| {
            candidate
                .strip_prefix(kind)
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
    })
}
