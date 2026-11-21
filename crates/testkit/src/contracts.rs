//! 词汇表契约：闭集只声明一次、位集只有一份实现、设备前端都被登记。
//!
//! 每条契约都是「同一件事不许有第二份写法」，判据是扫 `crates/*/src` 的文本并对照
//! `lints/vocabulary.toml` 的豁免表；豁免必须仍然命中，否则说明它已经过期。

#[cfg(test)]
mod tests {
use std::fs;
use std::path::PathBuf;

use crate::architecture::tests::{rust_files, workspace_root as root};

const MIN_SOURCES: usize = 70;

/// 一份扫描结果：`(crate, 文件, 行号, 该行原文)`。
type Hit = (String, PathBuf, usize, String);

/// `crates/<crate>/src/**/*.rs` 下的全部源文件。
fn sources() -> Vec<(String, PathBuf)> {
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

/// 找到声明 `pub mod {module}` 的文件：优先同名文件；聚合后的算子声明在骨架文件里。
fn locate_module(module: &str) -> Option<PathBuf> {
    let direct = root().join("crates/op/src").join(format!("{module}.rs"));
    if direct.exists() {
        return Some(direct);
    }
    for (crate_name, file) in sources() {
        if crate_name != "op" {
            continue;
        }
        let Ok(text) = fs::read_to_string(&file) else {
            continue;
        };
        if text.contains(&format!("pub mod {module}")) {
            return Some(file);
        }
    }
    None
}

/// 命中 `predicate` 的行。
fn hits(predicate: impl Fn(&str) -> bool) -> Vec<Hit> {
    let mut out = Vec::new();
    for (name, file) in sources() {
        let Ok(text) = fs::read_to_string(&file) else {
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
fn exempt(section: &str) -> Vec<(String, String)> {
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
/// 只能读源码：proc-macro crate 不导出非宏项，`syntax::DTYPES` 在 crate 外
/// 不可见。扫描只认 `= [ .. ]` 之间的逗号分隔字面量，排版怎么折行都不影响；声明里的 `N`
/// 一并核对，防止“表删了一项、长度还写着 9”。
fn annotation_list(name: &str) -> Vec<String> {
    let path = root().join("crates/annotation/src/syntax.rs");
    let text = fs::read_to_string(&path).expect("`crates/annotation/src/syntax.rs` 必须存在");
    let head = format!("const {name}: [&str;");
    let start = text.find(&head).unwrap_or_else(|| {
        panic!("注解文法里没有 `{name}`：它是注解与词汇表之间契约的一半，不能改名或删除")
    });

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

fn describe(hits: &[Hit]) -> String {
    hits.iter()
        .map(|(name, file, line, text)| format!("\n  {name} {}:{line}: {text}", file.display()))
        .collect()
}

/// 命中里除去 `vocabulary.toml` 该节豁免的名字，剩下的就是要报的违规项；白名单一并返回，
/// 供"豁免不能过期"那一半契约复用。
fn offenders_of(section: &str, hits: &[Hit]) -> (Vec<(String, String)>, Vec<Hit>) {
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
fn closed_vocabularies_are_declared_once() {
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
fn every_device_front_end_is_entered() {
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
fn bitsets_have_exactly_one_implementation() {
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
fn annotation_names_come_from_the_vocabulary() {
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

/// 语义表的**唯一生产者**：注册表里的 `MODEL` 必须由注解生成，注解标注的格式必须在注册表里。
///
/// 类型系统守住了"注解说的是哪份格式"（`<别名 as gemv::Format>`，写错就编译不过），守不住两件
/// 纯文本事故：
///
/// 1. 标注了 `#[annotation::model(..)]` 却忘了在 `REGISTRY` 里补一行——`MODEL` 常量照样生成，
///    但注册表里没有这条算子，`verify --op <名字>` 报"未知算子"，而写算子的人以为已经接上了；
/// 2. 注册表里的 `model:` 手写成 `Model { .. }` 字面量——语义表就有了第二份来源，生成的那一份
///    与它漂移时没有任何判据（这正是 P0-4 之前 `call_args` 的老毛病）。
///
/// 两条规则互为反面，都没有合法例外（标注了却不注册永远错；注册的语义表不是常量也永远错），
/// 所以不进白名单：出现例外说明契约本身该改，而不是这个 crate 该豁免。
}
