//! 词汇表契约：闭集只声明一次、位集只有一份实现、设备前端都被登记。
//!
//! 每条契约都是「同一件事不许有第二份写法」，判据是扫 `crates/*/src` 的文本并对照
//! `lints/vocabulary.toml` 的豁免表；豁免必须仍然命中，否则说明它已经过期。

use std::fs;
use std::path::PathBuf;

use crate::architecture::{rust_files, source_text, workspace_root as root};

/// 扫描下限：近 30 个 crate、80+ 个源文件；低于此数说明遍历逻辑坏了（而不是"没有违规"）。
pub(crate) const MIN_SOURCES: usize = 70;

/// 一份扫描结果：`(crate, 文件, 行号, 该行原文)`。
pub(crate) type Hit = (String, PathBuf, usize, String);

/// `crates/<crate>/src/**/*.rs` 下的全部源文件。
pub(crate) fn sources() -> Vec<(String, PathBuf)> {
    // `testkit` 自己不在扫描集里：它的 `src/` 全部是检验器自身的代码（每个文件都是
    // `#[cfg(test)] mod tests`），不是被检验的生产词汇表；扫进来只会让判据命中自己的谓词文本。
    let crates = root().join("crates");
    let mut out = Vec::new();
    let Ok(entries) = fs::read_dir(&crates) else {
        return out;
    };
    let mut names: Vec<String> = entries
        .flatten()
        .filter(|entry| entry.path().is_dir())
        .map(|entry| entry.file_name().to_string_lossy().into_owned())
        .collect();
    names.retain(|name| name != "testkit");
    names.sort();
    for name in names {
        let mut files = rust_files(&crates.join(&name).join("src"));
        files.sort();
        for file in files {
            out.push((name.clone(), file));
        }
    }
    out
}

/// 找到算子模块的实体文件：先按 `pub use` 别名换成实体名，再在骨架旁的嵌套目录里找实体。
///
/// 注册表的模块名有三种来源，都得落到"带注解的那个文件"上：
/// - 别名：`pub use classic::{adaptive_pool}` + `pub use pool::{adaptive as adaptive_pool}`；
/// - 嵌套：`pub use softmax::f32::{softmax as softmax_f32}`，实体在 `softmax/f32/softmax.rs`；
/// - 直连：模块名就是文件名。
pub(crate) fn locate_module(module: &str) -> Option<PathBuf> {
    locate(module, 0)
}

/// 别名链上限：`classic::adaptive_pool` → `classic::pool::adaptive` 只有一跳，留余量是为了让
/// 成环的别名表停在判定上而不是栈溢出。
const ALIAS_DEPTH: usize = 4;

fn locate(module: &str, depth: usize) -> Option<PathBuf> {
    if depth > ALIAS_DEPTH {
        return None;
    }
    let src = root().join("crates/op/src");
    // 带路径的模块名（`classic::pool::adaptive`）按目录走就是实体文件。
    if module.contains("::") {
        let explicit = src.join(format!("{}.rs", module.replace("::", "/")));
        if explicit.exists() {
            return Some(explicit);
        }
    }
    let direct = src.join(format!("{module}.rs"));
    if direct.exists() {
        return Some(direct);
    }
    for (crate_name, file) in sources() {
        if crate_name != "op" {
            continue;
        }
        let Some(text) = source_text(&file) else {
            continue;
        };
        if !text.contains(&format!("pub mod {module}")) {
            continue;
        }
        // 聚合布局：骨架文件 `classic/pool.rs` 声明 `pub mod adaptive;`，实体在骨架旁的
        // `classic/pool/adaptive.rs`——注册表要的注解读的是实体，不是只做声明的骨架。
        let body = file.with_extension("").join(format!("{module}.rs"));
        return Some(if body.exists() { body } else { file });
    }
    let source = aliases().get(module.rsplit("::").next().unwrap_or(module))?;
    (source != module).then(|| locate(source, depth + 1)).flatten()
}

pub(crate) fn aliases_debug() -> &'static std::collections::BTreeMap<String, String> {
    aliases()
}

/// `pub use` 表的别名：`adaptive as adaptive_pool` 说明注册名由实体名重导出。聚合后注册表的
/// 模块名几乎都不是文件名（45 个里 44 个），所以解析必须跨过这一层，否则判据只会看到重导出行。
///
/// 路径按 `use` 树的前缀补全：`pub use softmax::f32::{softmax as softmax_f32}` 里的
/// `softmax` 是 `softmax::f32::softmax`，不是 crate 根的 `softmax` 骨架文件。
fn aliases() -> &'static std::collections::BTreeMap<String, String> {
    static ALIASES: std::sync::OnceLock<std::collections::BTreeMap<String, String>> =
        std::sync::OnceLock::new();
    ALIASES.get_or_init(|| {
        let mut out = std::collections::BTreeMap::new();
        for (crate_name, file) in sources() {
            if crate_name != "op" {
                continue;
            }
            let Some(text) = source_text(&file) else {
                continue;
            };
            let owner = module_path_of(&file);
            for statement in use_statements(&text) {
                for (source, alias) in use_tree(&statement, &owner) {
                    // 近处的重导出更接近实体（`classic.rs` 的 `pool::adaptive` 胜过 `lib.rs` 的
                    // `classic::adaptive_pool`），扫描集按文件路径排序，先到先得即可。
                    out.entry(alias).or_insert(source);
                }
            }
        }
        out
    })
}

/// 文件自己的模块路径（`classic/pool.rs` → `classic::pool`，`lib.rs` → 空）。
fn module_path_of(file: &std::path::Path) -> String {
    let rel = file.strip_prefix(root().join("crates/op/src")).unwrap_or(file);
    let parts: Vec<String> = rel
        .with_extension("")
        .components()
        .map(|part| part.as_os_str().to_string_lossy().into_owned())
        .filter(|part| part != "lib" && part != "mod")
        .collect();
    parts.join("::")
}

/// 文件里的每条 `use` 语句（多行也算一条，直到分号）。
fn use_statements(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut buffer = String::new();
    for line in text.lines() {
        let line = line.trim();
        let starts = line.starts_with("pub use ") || line.starts_with("use ");
        if buffer.is_empty() && !starts {
            continue;
        }
        buffer.push_str(line);
        if line.ends_with(';') {
            out.push(std::mem::take(&mut buffer));
        } else {
            buffer.push(' ');
        }
    }
    out
}

/// 一条 `use` 语句拆成 `(实体路径, 注册名)`：支持花括号别名与直接别名。
fn use_tree(statement: &str, owner: &str) -> Vec<(String, String)> {
    let body = statement
        .split_once("use ")
        .map(|(_, rest)| rest.trim_end_matches(';').trim())
        .unwrap_or("");
    let join = |prefix: &str, name: &str| {
        if prefix.is_empty() {
            return name.to_string();
        }
        let absolute = prefix
            .strip_prefix("crate::")
            .or_else(|| (prefix == "crate").then_some(""));
        match absolute {
            Some(tail) if tail.is_empty() => name.to_string(),
            Some(tail) => format!("{tail}::{name}"),
            None if owner.is_empty() => format!("{prefix}::{name}"),
            None => format!("{owner}::{prefix}::{name}"),
        }
    };
    let mut out = Vec::new();
    match body.split_once('{') {
        Some((prefix, rest)) => {
            let prefix = prefix.trim_end_matches(':').trim_end_matches("::");
            for item in rest.split('}').next().unwrap_or("").split(',') {
                let item = item.trim();
                if item.is_empty() || item.contains('{') {
                    continue;
                }
                let (source, alias) = split_alias(item);
                out.push((join(prefix, source), alias.to_string()));
            }
        }
        None => {
            let (source, alias) = split_alias(body);
            let (prefix, name) = match source.rsplit_once("::") {
                Some((prefix, name)) => (prefix, name),
                None => ("", source),
            };
            out.push((join(prefix, name), alias.to_string()));
        }
    }
    out
}

/// `adaptive as adaptive_pool` → `("adaptive", "adaptive_pool")`；没有 `as` 时别名就是尾名。
fn split_alias(item: &str) -> (&str, &str) {
    let item = item.trim();
    match item.rsplit_once(" as ") {
        Some((source, alias)) => (ident(source), ident(alias)),
        None => (ident(item), ident(item)),
    }
}

/// 取词尾的标识符：`adaptive,` → `adaptive`，`gemm::f32` → `f32`，`{adaptive` → `adaptive`。
fn ident(word: &str) -> &str {
    word.rsplit(|c: char| !c.is_alphanumeric() && c != '_')
        .find(|part| !part.is_empty())
        .unwrap_or("")
}

/// 命中 `predicate` 的行。
pub(crate) fn hits(predicate: impl Fn(&str) -> bool) -> Vec<Hit> {
    let mut out = Vec::new();
    for (name, file) in sources() {
        let Some(text) = source_text(&file) else {
            continue;
        };
        for (index, line) in text.lines().enumerate() {
            if predicate(line) {
                out.push((
                    name.clone(),
                    file.strip_prefix(root()).unwrap_or(&file).to_path_buf(),
                    index + 1,
                    line.trim().to_string(),
                ));
            }
        }
    }
    out
}

/// 白名单里的 crate 与理由；配置来自 `tools` 构建期生成的词汇表（同一份文件只解析一次）。
pub(crate) fn exempt(section: &str) -> Vec<(String, String)> {
    let table = tool::vocabulary()
        .table(section)
        .unwrap_or_else(|| panic!("`lints/vocabulary.toml` 缺 `[{section}]`"));
    table
        .table_array("exempt")
        .unwrap_or_else(|| panic!("`lints/vocabulary.toml` 缺 `{section}.exempt`"))
        .iter()
        .map(|entry| {
            let name = entry
                .string("crate")
                .unwrap_or_else(|| panic!("`{section}.exempt` 的条目缺 `crate`"));
            let reason = entry
                .string("reason")
                .unwrap_or_else(|| panic!("`{section}.exempt` 的 `{name}` 缺 `reason`"));
            assert!(
                reason.chars().count() >= 8,
                "`{section}.exempt` 的 `{name}` 理由太短（{reason:?}）：豁免必须说清为什么"
            );
            (name.to_string(), reason.to_string())
        })
        .collect()
}

/// 注解文法里的闭集表：`const <名字>: [&str; N] = [ "..", .. ];`。
///
/// 只能读源码：proc-macro crate 不导出非宏项，`decl::DTYPES` 在 crate 外不可见。按**名字**在
/// 注解 crate 的 `src/` 全部文件里找，不预设它在哪个文件——crate 平铺后文件边界随时可动，
/// 判据只认声明本身。扫描只认 `= [ .. ]` 之间的逗号分隔字面量，排版怎么折行都不影响；声明
/// 里的 `N` 一并核对，防止“表删了一项、长度还写着 9”。
pub(crate) fn annotation_list(name: &str) -> Vec<String> {
    let head = format!("const {name}: [&str;");
    let (_, text) = rust_files(&root().join("crates/annotation/src"))
        .into_iter()
        .find_map(|path| {
            let body = fs::read_to_string(&path).ok()?;
            body.contains(&head).then_some((path, body))
        })
        .unwrap_or_else(|| {
            panic!("注解文法里没有 `{name}`：它是注解与词汇表之间契约的一半，不能改名或删除")
        });
    let start = text.find(&head).expect("上面刚按同一个串找到");

    let after = &text[start..];
    let open = start + after.find("= [").expect("闭集表必须写成 `= [ .. ]`") + 3;
    let close = open + text[open..].find(']').expect("闭集表必须闭合");
    let items: Vec<String> = text[open..close]
        .split(',')
        .map(str::trim)
        .filter(|item| !item.is_empty())
        .map(|item| {
            item.strip_prefix('"')
                .and_then(|rest| rest.strip_suffix('"'))
                .unwrap_or_else(|| panic!("`{name}` 的条目必须是字符串字面量，得到 {item:?}"))
                .to_string()
        })
        .collect();
    assert!(!items.is_empty(), "`{name}` 是空表：闭集表不能为空");

    let length_start = start + after.find("; ").expect("长度必须写在 `[&str; N]` 里") + 2;
    let length_end = length_start + text[length_start..].find(']').expect("长度必须闭合");
    let declared: usize = text[length_start..length_end]
        .trim()
        .parse()
        .unwrap_or_else(|_| panic!("`{name}` 的长度不是整数"));
    assert_eq!(
        declared,
        items.len(),
        "`{name}` 声明了 {declared} 个元素，实际写了 {} 个",
        items.len()
    );
    items
}

pub(crate) fn describe(hits: &[Hit]) -> String {
    hits.iter()
        .map(|(name, file, line, text)| format!("\n  {name} {}:{line}: {text}", file.display()))
        .collect()
}

/// 命中里除去 `vocabulary.toml` 该节豁免的名字，剩下的就是要报的违规项；白名单一并返回，
/// 供"豁免不能过期"那一半契约复用。
pub(crate) fn offenders_of(section: &str, hits: &[Hit]) -> (Vec<(String, String)>, Vec<Hit>) {
    let allowed = exempt(section);
    let names: Vec<&str> = allowed.iter().map(|(name, _)| name.as_str()).collect();
    let offenders = hits
        .iter()
        .filter(|(name, ..)| !names.contains(&name.as_str()))
        .cloned()
        .collect();
    (allowed, offenders)
}

#[test]
pub(crate) fn closed_vocabularies_are_declared_once() {
    // 手写的闭集表：数组形式 `pub const ALL: [Self; N]`。切片形式 `&'static [Self]` 是
    // `Closed` 的实现（`inherent_names!` 转发的那一份），不在此列。
    // 行内任意位置都算：`impl X { pub const ALL: [Self; N] = ..; }` 这种一行写法不该逃过。
    // 切片形式 `const ALL: &'static [Self]` 不以 `[` 紧接，所以不会被误判。
    let hits = hits(|line| line.contains("const ALL: ["));
    assert!(
        sources().len() >= MIN_SOURCES,
        "扫描器没看到源文件：{} < {MIN_SOURCES}",
        sources().len()
    );

    let (allowed, offenders) = offenders_of("closed", &hits);
    assert!(
        offenders.is_empty(),
        "这些闭集表该用 `impl prof::atom::Closed for ..` + `prof::inherent_names!`：{}",
        describe(&offenders)
    );

    // 白名单不能过期：每条豁免都必须在今天仍然真的手写着闭集表。
    for (name, _) in &allowed {
        assert!(
            root().join("crates").join(name).is_dir(),
            "`lints/vocabulary.toml` 里的 `{name}` 不是成员"
        );
        assert!(
            hits.iter().any(|(crate_name, ..)| crate_name == name),
            "`lints/vocabulary.toml` 里的 `{name}` 已经没有手写闭集表，删除这条豁免"
        );
    }

    // 契约的另一半：`Closed` 的实现必须写成切片，这样才能与 `ALL` 的顺序、`index` 的下标一致。
    let prof = root().join("crates/prof/src/atom.rs");
    let text = fs::read_to_string(&prof).expect("`crates/prof/src/atom.rs` 必须存在");
    assert!(
        text.contains("const ALL: &'static [Self]"),
        "`prof::atom` 的闭集表必须写成切片"
    );
    assert!(
        text.contains("macro_rules! inherent_names"),
        "转发宏 `inherent_names!` 是这份契约的入口"
    );
}

/// 执行后端必须落进 `cutile::backend` 的 trait 缝（`Backend = Codegen + Driver`），四个能力槽
/// 由 trait 方法**结构性**保证：`emit`/`program` 在 `Codegen`，`available`/`run`/`toolchain` 在
/// `Driver`。宿主 C 参考后端在 `device` 的注册表里种子就位，Metal / CUDA 经 `register` 由 L7
/// 注入。文本契约守的是：缝在、C 后端种子在、注入点在——"写了实现却忘了接线"只剩"没调 `register`"
/// 这一种，且只发生在装配层（L7），不在机制层。
#[test]
pub(crate) fn every_device_front_end_is_entered() {
    let device = root().join("crates/cutile/src/device.rs");
    let text = fs::read_to_string(&device).expect("`crates/cutile/src/device.rs` 必须存在");
    assert!(
        text.contains("static REGISTRY"),
        "后端注册表 `static REGISTRY` 必须存在"
    );
    assert!(
        text.contains("&CBackend"),
        "注册表必须种子内置 host C 参考后端"
    );
    assert!(
        text.contains("pub(crate) fn register"),
        "注册表必须有注入点 `register`（L7 装配 Metal/CUDA）"
    );

    let backend = root().join("crates/cutile/src/backend.rs");
    let text = fs::read_to_string(&backend).expect("`crates/cutile/src/backend.rs` 必须存在");
    assert!(text.contains("pub trait Codegen"), "Codegen 缝必须存在");
    assert!(text.contains("fn emit"), "Codegen 必须要求语言身份 `emit`");
    assert!(
        text.contains("fn program"),
        "Codegen 必须要求程序生成 `program`"
    );
    assert!(text.contains("pub trait Driver"), "Driver 缝必须存在");
    assert!(
        text.contains("fn available"),
        "Driver 必须要求探测 `available`"
    );
    assert!(text.contains("fn run"), "Driver 必须要求执行回读 `run`");
    assert!(
        text.contains("fn toolchain"),
        "Driver 必须要求工具链身份 `toolchain`"
    );
    assert!(
        text.contains("pub trait Backend: Codegen + Driver"),
        "Backend 必须是 Codegen + Driver 的并集"
    );
}

#[test]
pub(crate) fn bitsets_have_exactly_one_implementation() {
    // 手写位集：`struct FooSet(u32)` 一类。
    let hits = hits(|line| line.contains("struct ") && line.contains("Set(u32)"));
    let (_, offenders) = offenders_of("bitset", &hits);
    assert!(
        offenders.is_empty(),
        "位集只允许 `prof::atom::Set<E, WORDS>` 一份实现：{}",
        describe(&offenders)
    );

    let atom = fs::read_to_string(root().join("crates/prof/src/atom.rs"))
        .expect("`crates/prof/src/atom.rs` 必须存在");
    assert!(
        atom.contains("pub struct Set<E: Closed, const WORDS: usize>"),
        "唯一实现 `Set<E, WORDS>` 不见了：这条规则的替代品是空的"
    );
}

/// 注解的闭集必须由词汇表与对拍口径**推出来**，而不是第三份手抄表。
///
/// `DTYPES` 与「[`cutile::reference::tolerance`] 给出非零阈值」的 `DType` 集合逐项相等——
/// 注解能写出的输出类型必须正好是对拍能给结论的那些：多一个，`verify` 会拿 0 当阈值，
/// 少一个，那门输出就永远无法被声明成算子。
///
/// 变体名再核一遍：`render` 按 `DType::variant` 把名字翻成变体，
/// 名字与变体一旦分家，生成代码会指向不存在的变体——错误会出现在被标注的 crate 里，
/// 而不是写错字的那张表上。
#[test]
pub(crate) fn annotation_names_come_from_the_vocabulary() {
    // 输出类型：与「对拍有阈值」的那一组相等。
    let dtypes = annotation_list("DTYPES");
    let mut comparable: Vec<&str> = cutile::atom::DType::ALL
        .iter()
        .filter(|dt| cutile::reference::tolerance(**dt) > 0.0)
        .map(|dt| dt.name())
        .collect();
    comparable.sort_unstable();
    let mut declared: Vec<&str> = dtypes.iter().map(String::as_str).collect();
    declared.sort_unstable();
    assert_eq!(
        declared, comparable,
        "`DTYPES` 必须与 `cutile::reference::tolerance` 给出非零阈值的类型逐项相等"
    );
    for dtype in &dtypes {
        let dt = cutile::atom::DType::parse_name(dtype)
            .unwrap_or_else(|| panic!("`DTYPES` 里的 `{dtype}` 不在 `DType` 闭集里"));
        assert_eq!(dt.name(), *dtype, "`{dtype}` 必须能往返解析");
        // 变体名的唯一事实源是 `DType::variant`（缩写词全大写规则住在 dtype 里）。
        // `Debug` 输出必须与它一致，否则 annotation 生成的 `DType::{variant}` 编不过。
        assert_eq!(
            format!("{dt:?}"),
            dt.variant(),
            "`{dtype}` 的 Debug 名与 variant 不一致"
        );
    }
}
