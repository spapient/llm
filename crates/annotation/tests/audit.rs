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

/// 注释与字面量的三种开头字节；`/` 也可能是除号、`'` 也可能是生命周期，都要先看见再判定。
const OPENERS: [u8; 3] = [b'"', b'\'', b'/'];

/// 16 字节一步的字节搜索，语义等同 `bytes[from..].iter().position(needles.contains)`。
///
/// 逐字节交给迭代器是把速度押在"编译器愿意自动向量化"上；这里把比较写成每道放一个字节的位并行：
/// `word ^ 目标` 之后，借位链正好只在"该道为 0"处留下 `0x80`。三个目标的位图先 OR 再取最低命中道，
/// 比逐个目标取最小快一倍。误命中只会出现在真命中右侧（借位从低往高传），所以最低那一位永远是真的。
fn scan<const N: usize>(bytes: &[u8], from: usize, needles: [u8; N]) -> Option<usize> {
    let patterns = needles.map(spread);
    let mut cursor = from;
    while let Some(chunk) = bytes.get(cursor..cursor + 16) {
        let word = u128::from_ne_bytes(chunk.try_into().expect("切片长度恰为 16"));
        let lanes = patterns
            .iter()
            .fold(0, |lanes, pattern| lanes | zero_lanes(word ^ pattern));
        if lanes != 0 {
            return Some(cursor + (lanes.trailing_zeros() / 8) as usize);
        }
        cursor += 16;
    }
    bytes[cursor..]
        .iter()
        .position(|byte| needles.contains(byte))
        .map(|offset| cursor + offset)
}

/// 每道放一个 `0x01`：借位链的起点。
const LANE_ONES: u128 = u128::from_ne_bytes([0x01; 16]);

/// 每道放一个 `0x80`：借位链的终点，只留下"该道已归零"的标记位。
const LANE_HIGHS: u128 = u128::from_ne_bytes([0x80; 16]);

/// 哪几道是 0：减一之后仍小于 `0x80` 的道（减的借位传到它自己身上）就是 0。
fn zero_lanes(word: u128) -> u128 {
    word.wrapping_sub(LANE_ONES) & !word & LANE_HIGHS
}

fn spread(byte: u8) -> u128 {
    u128::from_ne_bytes([byte; 16])
}

/// 只留下代码：注释与字面量都剔掉。生成文本里就写着 `format!` / `.to_string()`，直接搜全文会把
/// "被生成的代码"误判成"生成器的实现"；`'{'`、`'"'` 这类字面量还会把 [`body`] 的花括号配平带偏。
///
/// 单趟字节游标：每圈用 [`scan`] 跳到下一个可能的开头，认成注释 / 字面量就把它之前那段代码整段
/// 追加、游标跳到它后面；`/` 与 `'` 若不是注释或字面量就只挪一个字节。游标每圈严格前进（被吞下的
/// 片段至少 1 字节），终止性由构造保证：不需要递归兜底，栈深与输入长度无关，中间也不分配 `String`。
/// 行注释的换行留在代码里，剔除后行数与原文件一一对应。
fn code_only(text: &str) -> String {
    let bytes = text.as_bytes();
    let mut out = String::with_capacity(text.len());
    let mut flushed = 0;
    let mut cursor = 0;
    while let Some(at) = scan(bytes, cursor, OPENERS) {
        match skip(text, at) {
            Some(end) => {
                debug_assert!(end > at, "被吞下的片段必须至少推进一个字节");
                out.push_str(&text[flushed..at]);
                flushed = end;
                cursor = end;
            }
            None => cursor = at + 1,
        }
    }
    out.push_str(&text[flushed..]);
    out
}

/// `at` 处的注释 / 字面量结束后游标该停在哪；裸 `/`（除号）与生命周期 `'a` 都不是，返回 `None`。
///
/// 按首两字节分派，不再切子串：`&text[at..]` 每次都要重算边界与长度，这里只读字节。
fn skip(text: &str, at: usize) -> Option<usize> {
    let bytes = text.as_bytes();
    match bytes[at..] {
        [b'/', b'/', ..] => Some(scan(bytes, at, [b'\n']).unwrap_or(bytes.len())),
        [b'/', b'*', ..] => Some(comment_end(bytes, at)),
        [b'"', ..] => Some(quoted_end(bytes, at, raw_hashes(bytes, at))),
        [b'\'', ..] => char_end(text, at),
        _ => None,
    }
}

/// 块注释可嵌套（Rust 特性）：`/* a /* b */ c */` 整体只是一条注释。
///
/// 开合定界符都从 `*` 或 `/` 开始，所以正文一次 [`scan`] 跳过，不逐字节走状态机。
fn comment_end(bytes: &[u8], at: usize) -> usize {
    let mut depth = 1u32;
    let mut cursor = at + 2;
    loop {
        let Some(marker) = scan(bytes, cursor, [b'*', b'/']) else {
            return bytes.len();
        };
        let next = bytes.get(marker + 1);
        let opens = bytes[marker] == b'/' && next == Some(&b'*');
        let closes = bytes[marker] == b'*' && next == Some(&b'/');
        if opens {
            depth += 1;
            cursor = marker + 2;
        } else if closes {
            depth -= 1;
            cursor = marker + 2;
            if depth == 0 {
                return cursor;
            }
        } else {
            cursor = marker + 1;
        }
    }
}

/// 字符串的结束位置；`hashes` 是开引子后的 `#` 数：`r##"…"##` 为 2，普通 `"…"` 为 0。
fn quoted_end(bytes: &[u8], at: usize, hashes: usize) -> usize {
    let escapes = hashes == 0;
    let mut cursor = at + 1;
    while let Some(stop) = if escapes {
        scan(bytes, cursor, [b'"', b'\\'])
    } else {
        scan(bytes, cursor, [b'"'])
    } {
        if bytes[stop] == b'\\' {
            cursor = (stop + 2).min(bytes.len());
        } else if closes_with(&bytes[stop + 1..], hashes) {
            return stop + 1 + hashes;
        } else {
            cursor = stop + 1;
        }
    }
    bytes.len()
}

/// raw string 的收尾定界符是 `"` 后跟与开引子同数量的 `#`。
fn closes_with(rest: &[u8], hashes: usize) -> bool {
    rest.len() >= hashes && rest[..hashes].iter().all(|byte| *byte == b'#')
}

/// `"` 前紧邻的 `#` 数：`r##"…"##` 是 2，普通 `"…"` 是 0。只给 `"` 候选算，除号与生命周期不白扫。
fn raw_hashes(bytes: &[u8], at: usize) -> usize {
    bytes[..at]
        .iter()
        .rev()
        .take_while(|byte| **byte == b'#')
        .count()
}

/// 字符字面量的结束位置：`'{'` / `'\''` / `'\u{7f}'`；`'a` 是生命周期，返回 `None`。
///
/// 长度必须卡死：放开长度就会让 `for<'a, 'b>` 里的 `'a, '` 被当成字面量，进而吞掉真实代码。
fn char_end(text: &str, at: usize) -> Option<usize> {
    /// `'\u{10FFFF}'` 是最长的字符字面量（13 字节），收尾引号不会晚于第 12 个字节。
    const LONGEST: usize = 12;
    let rest = &text[at..];
    let bytes = rest.as_bytes();
    if *bytes.get(1)? == b'\\' {
        let mut cursor = 1;
        while cursor <= LONGEST {
            match *bytes.get(cursor)? {
                b'\\' => cursor += 2,
                b'\'' => return Some(at + cursor + 1),
                _ => cursor += 1,
            }
        }
        return None;
    }
    let width = rest[1..].chars().next()?.len_utf8();
    (bytes.get(1 + width) == Some(&b'\'')).then_some(at + 2 + width)
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

/// 游标是单趟的：十万段字面量也不占一帧栈。旧的自递归实现在这里会把测试线程的栈撑爆，而栈溢出是
/// abort 而不是断言失败，所以"栈深与输入无关"这条只能靠这种规模用例观测。
#[test]
fn code_only_is_flat_over_many_literals() {
    const COUNT: usize = 100_000;
    let mut source = "\"x\" ".repeat(COUNT);
    source.push_str("done");
    let code = code_only(&source);
    assert_eq!(
        code.len(),
        COUNT + "done".len(),
        "每段字面量只应剩下一个分隔空格"
    );
    assert!(code.ends_with("done"), "字面量之后的代码不得丢失");
}

/// 没有收尾的注释 / 字面量只能吞到文件末尾：游标不许越过末尾，否则切片就 panic。
#[test]
fn code_only_stops_at_end_on_unterminated_forms() {
    assert_eq!(code_only("a /* b"), "a ");
    assert_eq!(code_only("let s = \"abc"), "let s = ");
    assert_eq!(code_only("let raw = r\"###tail"), "let raw = r");
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
