//! 词汇表契约：闭集只声明一次、位集只有一份实现、设备前端都被登记。

use std::collections::BTreeSet;
use std::fs;
use std::path::PathBuf;
use std::sync::OnceLock;

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
pub(crate) fn locate_module(module: &str) -> Option<PathBuf> {
    locate(module, 0)
}

/// 内核机制与运行期的 `src` 根。机制（协商 / 拼接 / emit）在 `cutile`，执行面
/// （编译 / doctor / bench / verify）并进 `shard` 后，契约文件会在两家之间搬；判据守的是结构契约
/// （缝在、注册表在、注入点在），不是文件坐标，所以按两家探测，谁家有就用谁家的。
fn cutile_path(relative: &str) -> PathBuf {
    let candidates: Vec<PathBuf> = ["cutile", "shard"]
        .iter()
        .map(|name| root().join("crates").join(name).join("src").join(relative))
        .collect();
    candidates
        .iter()
        .find(|path| path.exists())
        .cloned()
        .unwrap_or_else(|| {
            panic!(
                "`{relative}` 在 cutile 与 shard 的 `src` 里都不存在（查过 {}）",
                candidates
                    .iter()
                    .map(|path| path.display().to_string())
                    .collect::<Vec<_>>()
                    .join("、")
            )
        })
}

/// 算子家族的判据：**在 `src` 里写了 `#[annotation::model(..)]` 的 crate**。
pub(crate) fn in_op_family(crate_name: &str) -> bool {
    family().contains(crate_name)
}

/// 算子家族的成员（目录名），只扫一遍：`sources()` 的文本已按文件缓存。
fn family() -> &'static BTreeSet<String> {
    static CACHE: OnceLock<BTreeSet<String>> = OnceLock::new();
    CACHE.get_or_init(|| {
        sources()
            .into_iter()
            .filter(|(_, file)| {
                source_text(file).is_some_and(|text| text.contains(MODEL_ANNOTATION))
            })
            .map(|(name, _)| name)
            .collect()
    })
}

/// 语义表的注解前缀：`#[annotation::model(` 必须带左括号，`#[annotation::model_of]` 这类
/// 只提名字的散文不算声明。
const MODEL_ANNOTATION: &str = "#[annotation::model(";

/// 算子家族的 `src` 根。算子实体可能散在多家，路径探测逐家去试；按名字排序只为稳定。
fn op_roots() -> Vec<PathBuf> {
    let mut names: Vec<&String> = family().iter().collect();
    names.sort();
    names
        .into_iter()
        .map(|name| root().join("crates").join(name).join("src"))
        .filter(|path| path.is_dir())
        .collect()
}

/// 别名链上限：`classic::adaptive_pool` → `classic::pool::adaptive` 只有一跳，留余量是为了让
/// 成环的别名表停在判定上而不是栈溢出。
const ALIAS_DEPTH: usize = 4;

fn locate(module: &str, depth: usize) -> Option<PathBuf> {
    if depth > ALIAS_DEPTH {
        return None;
    }
    // 声明是唯一真相：文件名可以自由取（`#[path = "kernel_gemv.rs"] pub mod gemv;`），
    // 所以「模块路径 → 文件」只能由声明处推导，不能由文件名反推。
    if let Some(found) = module_index().get(module) {
        return Some(found.clone());
    }
    let roots = op_roots();
    // 按相对路径在家族各家根下探测：名字在谁家就返回谁家的实体。
    let find = |relative: String| {
        roots
            .iter()
            .map(|src| src.join(&relative))
            .find(|cand| cand.exists())
    };
    // 带路径的模块名（`classic::pool::adaptive`）按目录走就是实体文件。
    if module.contains("::") {
        if let Some(explicit) = find(format!("{}.rs", module.replace("::", "/"))) {
            return Some(explicit);
        }
        // 内联模块（`activation.rs` 里的 `pub mod embed { … }`、`classic.rs` 里的
        // `pub mod pool { pub mod adaptive { … } }`）没有自己的文件，注解与实体都在外层骨架里：
        // 逐级上溯到第一个**确实声明了下一段**的文件。这条判据是必要的——只按"文件存在"上溯，
        // `kernel::fast::i8::gemm_i8` 这类纯函数重导出会被错认成算子实体所在的文件。
        let segments: Vec<&str> = module.split("::").collect();
        let leaf = segments.last().copied().unwrap_or_default();
        // 判据是「这个文件声明了**目标段**」而不是「声明了中间段」：`conv.rs` 声明了
        // `pub mod classic`，实体在同一文件的更深一层内联 `pub mod batch` 里，
        // 认中间段会把只声明模块名的位置当成实体位置。
        for cut in (1..segments.len()).rev() {
            let Some(candidate) = find(format!("{}.rs", segments[..cut].join("/"))) else {
                continue;
            };
            let Some(text) = source_text(&candidate) else {
                continue;
            };
            if text.lines().any(|line| {
                let line = line.trim_start();
                let Some(rest) = line
                    .strip_prefix("pub mod ")
                    .or_else(|| line.strip_prefix("pub(crate) mod "))
                else {
                    return false;
                };
                let name = rest
                    .trim_start()
                    .trim_end_matches('{')
                    .trim_end_matches(';')
                    .trim();
                name == leaf
            }) {
                return Some(candidate);
            }
        }
    }
    if let Some(direct) = find(format!("{module}.rs")) {
        return Some(direct);
    }
    for (crate_name, file) in sources() {
        if !in_op_family(&crate_name) {
            continue;
        }
        let Some(text) = source_text(&file) else {
            continue;
        };
        if !text.contains(&format!("pub mod {module}")) {
            continue;
        }
        // 聚合布局：骨架文件 `gemv.rs` 声明 `pub mod f32;`，实体在骨架旁的
        // `gemv/f32.rs`——注册表要的注解读的是实体，不是只做声明的骨架。
        let body = file.with_extension("").join(format!("{module}.rs"));
        return Some(if body.exists() { body } else { file });
    }
    // 候选按路径段数从少到多试：`gemm_i8` 既是 `gemm::{i8 as gemm_i8}` 的算子实体，也是
    // `fast::i8::gemm_i8` 这个纯函数的重导出，短的才是注册表要的那份。
    let tail = module.rsplit("::").next().unwrap_or(module);
    let mut candidates: Vec<&String> = aliases().get(tail).into_iter().flatten().collect();
    candidates.sort_by_key(|source| source.matches("::").count());
    for source in candidates {
        if source != module
            && let Some(found) = locate(source, depth + 1)
        {
            return Some(found);
        }
    }
    None
}

/// `pub use` 表的别名：`adaptive as adaptive_pool` 说明注册名由实体名重导出。聚合后注册表的
/// 模块名几乎都不是文件名（45 个里 44 个），所以解析必须跨过这一层，否则判据只会看到重导出行。
fn aliases() -> &'static std::collections::BTreeMap<String, Vec<String>> {
    static ALIASES: std::sync::OnceLock<std::collections::BTreeMap<String, Vec<String>>> =
        std::sync::OnceLock::new();
    ALIASES.get_or_init(|| {
        let mut out: std::collections::BTreeMap<String, Vec<String>> =
            std::collections::BTreeMap::new();
        for (crate_name, file) in sources() {
            if !in_op_family(&crate_name) {
                continue;
            }
            let Some(text) = source_text(&file) else {
                continue;
            };
            let owner = module_path_of(&file);
            for statement in use_statements(&text) {
                for (source, alias) in use_tree(&statement, &owner) {
                    let candidates = out.entry(alias).or_default();
                    if !candidates.contains(&source) {
                        candidates.push(source);
                    }
                }
            }
        }
        out
    })
}

/// 模块路径 → 实体文件的索引，由**声明处**递归推导（`OnceLock` 只建一次）。
///
/// 三条判据，全部来自 Rust 的模块解析规则本身，不看文件名与模块名是否同名：
/// - `#[path = "X.rs"] pub mod name;`：`X.rs` 相对**声明所在文件**解析（`src/` 下的扁平文件即由此接入）；
/// - `pub mod name;`：`name.rs` / `name/mod.rs`（`lib.rs` 与 `mod.rs` 的子模块与自身同目录）；
/// - `pub mod name { … }`：内联模块，实体就在本文件里。
///
/// 这条索引让「源码怎么拆、文件叫什么」与「模块路径」彻底解耦——否则每拆一次文件，
/// 契约门禁就得跟着改一遍路径猜测。
fn module_index() -> &'static std::collections::BTreeMap<String, PathBuf> {
    static INDEX: OnceLock<std::collections::BTreeMap<String, PathBuf>> = OnceLock::new();
    INDEX.get_or_init(|| {
        let mut out = std::collections::BTreeMap::new();
        for (crate_name, file) in sources() {
            if !in_op_family(&crate_name) || file.file_stem().is_some_and(|stem| stem != "lib") {
                continue;
            }
            walk_declarations(&file, &[], &mut out);
        }

        out
    })
}

/// `#[path]` 与 `pub mod name;` 的解析：把声明所在文件的子模块文件找出来。
fn child_file(owner: &std::path::Path, name: &str, explicit: Option<&str>) -> Option<PathBuf> {
    let dir = owner.parent()?;
    if let Some(relative) = explicit {
        let candidate = dir.join(relative);
        return candidate.exists().then_some(candidate);
    }
    // `lib.rs` / `mod.rs` 的子模块与自身同目录，其余文件（`foo.rs`）的子模块在 `foo/` 里。
    let stem = owner.file_stem()?.to_string_lossy().into_owned();
    let base = if stem == "lib" || stem == "mod" {
        dir.to_path_buf()
    } else {
        dir.join(&stem)
    };
    [
        base.join(format!("{name}.rs")),
        base.join(name).join("mod.rs"),
    ]
    .into_iter()
    .find(|candidate| candidate.exists())
}

/// 递归跟随声明：内联模块只换栈不换文件，`#[path]`/同名文件换文件。
///
/// 缩进栈取代花括号计数：全仓 `cargo fmt` 门禁保证「内联模块的深度 = 缩进 / 4」，
/// 而花括号计数会被字符串里的 `{`（格式化参数）打乱。
fn walk_declarations(
    file: &std::path::Path,
    prefix: &[String],
    out: &mut std::collections::BTreeMap<String, PathBuf>,
) {
    let Some(text) = source_text(file) else {
        return;
    };
    let mut stack: Vec<(usize, String)> = Vec::new();
    let mut explicit: Option<String> = None;
    for line in text.lines() {
        let trimmed = line.trim_start();
        let indent = line.len() - trimmed.len();
        if let Some(rest) = trimmed.strip_prefix("#[path = \"") {
            explicit = rest.split('"').next().map(str::to_string);
            continue;
        }
        if !trimmed.starts_with('#') && !trimmed.is_empty() {
            // 属性行之外的任何一行都清掉待用属性：`#[path]` 只对它紧邻的那条 `mod` 生效。
            if !trimmed.starts_with("//") {
                let pending = explicit.take();
                if let Some((name, inline)) = module_declaration(trimmed) {
                    while stack.last().is_some_and(|(depth, _)| *depth >= indent) {
                        stack.pop();
                    }
                    stack.push((indent, name.clone()));
                    let mut path = prefix.to_vec();
                    path.extend(stack.iter().map(|(_, name)| name.clone()));
                    let joined = path.join("::");
                    out.entry(joined.clone())
                        .or_insert_with(|| file.to_path_buf());
                    if !inline {
                        stack.pop();
                        if let Some(child) = child_file(file, &name, pending.as_deref()) {
                            // 非内联模块的实体是子文件（`#[path]` / 同名 `.rs` / 目录 `mod.rs`）：
                            // 父路径回填到实体，声明处骨架只在找不到子文件时兜底。
                            out.insert(joined.clone(), child.clone());
                            walk_declarations(&child, &path, out);
                        }
                    }
                }
            }
        }
    }
}

/// 一行是不是模块声明：返回 `(模块名, 是否内联)`。
fn module_declaration(line: &str) -> Option<(String, bool)> {
    let rest = line
        .strip_prefix("pub mod ")
        .or_else(|| line.strip_prefix("pub(crate) mod "))
        .or_else(|| line.strip_prefix("mod "))?;
    let inline = rest.ends_with('{');
    let name = rest.trim_end_matches('{').trim_end_matches(';').trim();
    let valid = !name.is_empty()
        && name
            .chars()
            .all(|ch| ch.is_ascii_alphanumeric() || ch == '_');
    valid.then(|| (name.to_string(), inline))
}

/// 文件 → 声明出的模块路径（取最短的一条：一个文件被多处声明时，最短的是它的规范位置）。
fn declared_paths() -> &'static std::collections::BTreeMap<PathBuf, String> {
    static REVERSE: OnceLock<std::collections::BTreeMap<PathBuf, String>> = OnceLock::new();
    REVERSE.get_or_init(|| {
        let mut out: std::collections::BTreeMap<PathBuf, String> =
            std::collections::BTreeMap::new();
        for (module, file) in module_index() {
            let key = normalize(file);
            match out.get(&key) {
                Some(existing) if existing.len() <= module.len() => {}
                _ => {
                    out.insert(key, module.clone());
                }
            }
        }
        out
    })
}

/// 词法规整：去掉 `.` 与可消的 `..`，让「同一条路径的不同写法」比较得上。
fn normalize(path: &std::path::Path) -> PathBuf {
    let mut out = PathBuf::new();
    for component in path.components() {
        match component {
            std::path::Component::CurDir => {}
            std::path::Component::ParentDir => {
                out.pop();
            }
            other => out.push(other),
        }
    }
    out
}

/// 文件自己的模块路径（`gemv/f32.rs` → `gemv::f32`，`lib.rs` → 空）。
fn module_path_of(file: &std::path::Path) -> String {
    // 声明索引优先：`#[path = "kernel_matmul.rs"] pub mod matmul;` 之后文件名不再等于模块路径，
    // 只有声明处知道这个文件是谁的实体。
    if let Some(declared) = declared_paths().get(&normalize(file)) {
        return declared.clone();
    }
    // 家族有多家，模块路径一律相对各自的 `crates/<crate>/src` 算：写死其中某一家 crate 名会让其余各家的
    // 路径多出一层 crate 名，`gemv/f32.rs` 就不再等于 `gemv::f32`。
    let rel = file
        .strip_prefix(root().join("crates"))
        .ok()
        .and_then(|rest| {
            let mut parts = rest.components();
            parts.next()?;
            parts.next().filter(|part| part.as_os_str() == "src")?;
            Some(parts.as_path().to_path_buf())
        })
        .unwrap_or_else(|| file.to_path_buf());
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
    let mut buf = String::new();
    for line in text.lines() {
        let line = line.trim();
        let starts = line.starts_with("pub use ") || line.starts_with("use ");
        if buf.is_empty() && !starts {
            continue;
        }
        buf.push_str(line);
        if line.ends_with(';') {
            out.push(std::mem::take(&mut buf));
        } else {
            buf.push(' ');
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
    // `use` 的路径是相对**所在模块**解析的：`super::` 逐层上溯、`self::` 原地、`crate::` 绝对。
    // 声明索引让 owner 与文件名解耦（`kernel_matmul.rs` 的 owner 是 `kernel::matmul`），
    // 这条解析才可能对——按文件名算 owner 时，`super::` 会被拼成 `kernel_matmul::super::…`。
    let join = |prefix: &str, name: &str| {
        let mut parts: Vec<&str> = owner.split("::").filter(|seg| !seg.is_empty()).collect();
        let mut rest = prefix;
        loop {
            if rest == "super" {
                parts.pop();
                rest = "";
            } else if let Some(tail) = rest.strip_prefix("super::") {
                parts.pop();
                rest = tail;
            } else if rest == "self" {
                rest = "";
            } else if let Some(tail) = rest.strip_prefix("self::") {
                rest = tail;
            } else if rest == "crate" {
                parts.clear();
                rest = "";
            } else if let Some(tail) = rest.strip_prefix("crate::") {
                parts.clear();
                rest = tail;
            } else {
                break;
            }
        }
        parts.extend(rest.split("::").filter(|seg| !seg.is_empty()));
        parts.push(name);
        parts.join("::")
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
            // 无花括号时 source 可能带 `::` 前缀（`use decode::semantics as decode_attention`），
            // 前缀必须保留，不能像花括号条目那样按词尾标识符截断。
            let (source, alias) = match body.rsplit_once(" as ") {
                Some((source, alias)) => (source.trim(), ident(alias)),
                None => (body.trim(), body.trim()),
            };
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
        for (idx, line) in text.lines().enumerate() {
            if predicate(line) {
                out.push((
                    name.clone(),
                    file.strip_prefix(root()).unwrap_or(&file).to_path_buf(),
                    idx + 1,
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
        "这些闭集表该用 `impl prof::atom::Closed for ..` + `inherent_names!`：{}",
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

    // 契约的另一半：`Closed` 的实现必须写成切片，这样才能与 `ALL` 的顺序、`idx` 的下标一致。
    let text = fs::read_to_string(root().join("crates/prof/src/atom.rs"))
        .expect("`crates/prof/src/atom.rs` 必须存在");
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
    let device = cutile_path("device.rs");
    let text = fs::read_to_string(&device).expect("`device.rs` 必须可读");
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

    let backend = cutile_path("backend.rs");
    let text = fs::read_to_string(&backend).expect("`backend.rs` 必须可读");
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
