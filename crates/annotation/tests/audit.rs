//! 性能与供应链静态门禁（`annotation` 局部）。
//!
//! 过程宏在编译期运行，输入只是一小段属性；它的性能风险不是绝对毫秒数，而是文法退化成关联
//! 容器查找、正则回溯或引入重型初始化，渲染热路径上每次多分配几个中间 `String`。下面的门禁
//! 把这几类回退挡在合并之前：
//!
//! - 生产源码不出现关联容器、正则或惰性/全局初始化；
//! - 渲染热路径每次只分配一个输出缓冲；
//! - 运行期依赖为空（`MODEL` 改为投影 `Operator` 后，连 `dtype` 基座也不再需要）。
//!
//! 前两条靠 [`code_only`] 先剔掉注释与字面量：生成文本里本身就写着 `format!`，不剔会把"被生成
//! 的代码"当成"生成器的实现"。剔除器自己有等价类测试，认错一次就会让上面两条静默漏判。

use std::fs;
use std::path::{Path, PathBuf};

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

fn walk(dir: &Path, files: &mut Vec<PathBuf>) {
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
/// 只算代码——注释里提到 `HashMap`、字面量里写着 `"regex"` 都不构成依赖。
#[test]
fn production_source_has_no_associative_container_or_regex() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src");
    let mut files = Vec::new();
    walk(&root, &mut files);

    let mut offenders: Vec<String> = Vec::new();
    for path in files {
        let text = code_only(&fs::read_to_string(&path).expect("源码可读"));
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

/// 注释与字面量的三种开头；`/` 也可能是除号、`'` 也可能是生命周期，都要先看见再判定。
const OPENERS: &[char] = &['"', '\'', '/'];

/// 只留下代码：注释与字面量都剔掉。生成文本里就写着 `format!` / `.to_string()`，直接搜全文会把
/// "被生成的代码"误判成"生成器的实现"；`'{'`、`'"'` 这类字面量还会把 [`body`] 的花括号配平带偏。
///
/// 每层只处理"下一个真正要跳过的开头"：它之前那段是纯代码，整段追加；它之后的部分交给下一层。
/// 除号与生命周期不算跳过，就地留在代码里，不额外占一层。单趟线性，不分配中间 `String`。
fn code_only(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    code_into(text, &mut out);
    out
}

/// [`code_only`] 的一层：把 `rest` 开头那段纯代码追进 `out`，跳过它之后的注释 / 字面量再自递归。
///
/// 递归深度等于"注释 + 字面量"的个数（不是字节数，也不是候选个数），每层 `rest` 都严格变短，
/// 所以必然终止；也因为没有可变游标，"跳过量算错导致原地打转"这类失败模式不存在。
fn code_into(rest: &str, out: &mut String) {
    // 候选逐个试：`skip` 返回 `None` 说明那个 `/` 或 `'` 其实是代码，继续往后找下一个候选。
    let opening = rest
        .match_indices(OPENERS)
        .find_map(|(at, _)| skip(&rest[at..], &rest[..at]).map(|len| (at, len)));
    let Some((at, len)) = opening else {
        out.push_str(rest);
        return;
    };
    out.push_str(&rest[..at]);
    code_into(&rest[at + len..], out);
}

/// 从注释 / 字面量开头算起要吞掉的字节数；裸 `/` 与生命周期 `'a` 都不是注释或字面量，返回 `None`。
///
/// `code` 是已经收下的代码：raw string 的 `r##` 前缀都已按代码追加，数它尾随的 `#` 就是收尾定界符。
fn skip(rest: &str, code: &str) -> Option<usize> {
    let hashes = code.len() - code.trim_end_matches('#').len();
    match *rest.as_bytes() {
        [b'/', b'/', ..] => Some(rest.find('\n').unwrap_or(rest.len())),
        [b'/', b'*', ..] => Some(comment_end(rest)),
        [b'"', ..] => Some(quoted_end(rest, hashes)),
        [b'\'', ..] => char_end(rest),
        _ => None,
    }
}

/// 块注释可嵌套（Rust 特性）：`/* a /* b */ c */` 整体只是一条注释。
fn comment_end(rest: &str) -> usize {
    let bytes = rest.as_bytes();
    let mut depth = 0u32;
    let mut at = 0;
    while at + 1 < bytes.len() {
        match (bytes[at], bytes[at + 1]) {
            (b'/', b'*') => {
                depth += 1;
                at += 2;
            }
            (b'*', b'/') => {
                depth -= 1;
                at += 2;
                if depth == 0 {
                    return at;
                }
            }
            _ => at += 1,
        }
    }
    rest.len()
}

/// 字符串的结尾；`hashes` 是开引子后的 `#` 数：`r##"…"##` 为 2，普通 `"…"` 为 0。
fn quoted_end(rest: &str, hashes: usize) -> usize {
    let bytes = rest.as_bytes();
    let mut at = 1;
    while at < bytes.len() {
        match bytes[at] {
            b'\\' if hashes == 0 => at += 2,
            b'"' if closes_with(&bytes[at + 1..], hashes) => return at + 1 + hashes,
            _ => at += 1,
        }
    }
    rest.len()
}

/// raw string 的收尾定界符是 `"` 后跟与开引子同数量的 `#`。
fn closes_with(rest: &[u8], hashes: usize) -> bool {
    rest.len() >= hashes && rest[..hashes].iter().all(|byte| *byte == b'#')
}

/// 字符字面量的结尾：`'{'` / `'\''` / `'\u{7f}'`；`'a` 是生命周期，返回 `None`。
///
/// 长度必须卡死：放开长度就会让 `for<'a, 'b>` 里的 `'a, '` 被当成字面量，进而吞掉真实代码。
fn char_end(rest: &str) -> Option<usize> {
    /// `'\u{10FFFF}'` 是最长的字符字面量（13 字节），收尾引号不会晚于第 12 个字节。
    const LONGEST: usize = 12;
    let bytes = rest.as_bytes();
    if *bytes.get(1)? == b'\\' {
        let mut at = 1;
        while at <= LONGEST {
            match *bytes.get(at)? {
                b'\\' => at += 2,
                b'\'' => return Some(at + 1),
                _ => at += 1,
            }
        }
        return None;
    }
    let width = rest[1..].chars().next()?.len_utf8();
    (bytes.get(1 + width) == Some(&b'\'')).then_some(2 + width)
}

/// 某个函数的函数体：从 `fn 名字` 起找第一个 `{`，再按花括号配平。泛型参数（`fn unique<'a>(`）
/// 也认，但 `fn render` 不会命中 `fn render_model`。
fn body<'a>(code: &'a str, name: &str) -> &'a str {
    let start = code
        .match_indices("fn ")
        .find_map(|(offset, marker)| {
            let tail = code[offset + marker.len()..].strip_prefix(name)?;
            let joined = tail
                .chars()
                .next()
                .is_some_and(|next| next.is_alphanumeric() || next == '_');
            (!joined).then_some(offset + marker.len() + name.len())
        })
        .unwrap_or_else(|| panic!("`render.rs` 里没有 `fn {name}`"));
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

/// [`code_only`] 的等价类：五种形态各自要吞干净，生命周期和跨行代码一个字都不能少。
///
/// 剔除器认错一次，上面两条门禁就静默失效（`'"'` 会把下一行的代码当字符串吞掉，`'{'` 会让
/// 花括号配平多出一层），所以它自带测试，而不是只靠门禁"看起来还是绿的"。
#[test]
fn code_only_strips_every_literal_form() {
    let source = r##"
        // format! 行注释
        /* format! /* 嵌套块注释 */ format! */
        let text = "format! \" //";
        let raw = r#"format! "quoted" //"#;
        let brace = '{';
        let quote = '"';
        let escaped = '\'';
        fn unique<'a>(names: &'a str) -> &'a str { names }
    "##;
    let code = code_only(source);
    for leaked in ["format!", "quoted", "//", "/*", "嵌套"] {
        assert!(!code.contains(leaked), "`{leaked}` 漏进了代码：{code}");
    }
    for kept in [
        "let brace",
        "let quote",
        "let escaped",
        "fn unique<'a>",
        "&'a str",
    ] {
        assert!(code.contains(kept), "`{kept}` 被误剔：{code}");
    }
    assert_eq!(
        code.matches('{').count(),
        code.matches('}').count(),
        "字面量里的花括号不得参与配平：{code}"
    );
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
