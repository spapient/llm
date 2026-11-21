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

/// 只留下代码：字符串字面量与行注释都剔掉。生成文本里就写着 `format!` / `.to_string()`，
/// 直接搜全文会把"被生成的代码"误判成"生成器的实现"。
fn code_only(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut chars = text.chars().peekable();
    while let Some(current) = chars.next() {
        match current {
            '"' => {
                out.push('"');
                while let Some(inner) = chars.next() {
                    match inner {
                        '\\' => {
                            chars.next();
                        }
                        '"' => break,
                        _ => {}
                    }
                }
            }
            '/' if chars.peek() == Some(&'/') => {
                while chars.peek().is_some_and(|next| *next != '\n') {
                    chars.next();
                }
            }
            _ => out.push(current),
        }
    }
    out
}

/// 某个函数的函数体：从 `fn 名字` 起找第一个 `{`，再按花括号配平。泛型参数（`fn unique<'a>(`）
/// 也认，但 `fn render` 不会命中 `fn render_model`。
fn body<'a>(code: &'a str, name: &str) -> &'a str {
    let marker = format!("fn {name}");
    let start = code
        .match_indices(&marker)
        .find(|(offset, _)| {
            matches!(
                code[offset + marker.len()..].chars().next(),
                Some('(' | '<')
            )
        })
        .unwrap_or_else(|| panic!("`render.rs` 里没有 `{name}`"))
        .0
        + marker.len();
    let rest = &code[start..];
    let open = rest.find('{').expect("函数有函数体");
    let mut depth = 0usize;
    for (offset, character) in rest[open..].char_indices() {
        match character {
            '{' => depth += 1,
            '}' => {
                depth -= 1;
                if depth == 0 {
                    return &rest[open..=open + offset];
                }
            }
            _ => {}
        }
    }
    panic!("`{name}` 的函数体没有配平");
}

/// 热路径每次生成**只分配一个输出缓冲**：三个入口各是一处 `String::with_capacity`，且除错误
/// 路径 [`closed`] 外不再出现 `format!` / `to_string` / `collect` 这类中间分配。
///
/// 这条判据比计时更硬：`forbid(unsafe_code)` 下挂不了计数分配器，而毫秒数在 CI 上会抖；"每份声明
/// 只分配一次"是结构事实，代码一退化就红。
#[test]
fn rendering_allocates_only_its_output_buffer() {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src/render.rs");
    let code = code_only(&fs::read_to_string(&path).expect("源码可读"));
    for entry in ["render", "render_model", "render_runtime"] {
        let count = body(&code, entry).matches("String::with_capacity").count();
        assert_eq!(count, 1, "`{entry}` 必须只分配一个输出缓冲，实际 {count}");
    }
    let hot = [
        "spec",
        "contract",
        "check",
        "render_runtime",
        "render_model",
        "asserts",
        "registry",
        "dtype",
        "unique",
        "holes",
        "shape_of",
        "comma",
        "literal",
    ];
    for name in hot {
        let text = body(&code, name);
        for forbidden in ["format!", ".to_string()", ".collect()", "String::new()"] {
            assert!(
                !text.contains(forbidden),
                "`{name}` 在热路径上，不得出现 `{forbidden}`：中间 `String` 会让每份声明多分配几次"
            );
        }
    }
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
