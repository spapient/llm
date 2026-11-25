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

#![deny(clippy::pedantic)]

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
        if !inside || trimmed.starts_with('#') {
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
const OPENERS: &[u8] = b"\"'/";

/// 16 字节一步地找"第一个被 `accept` 接受的位置"；`accept` 为假就清掉该命中道、在同一窗口里续找。
///
/// 逐字节交给迭代器是把速度押在"编译器愿意自动向量化"上；这里把比较写成每道放一个字节的位并行：
/// `word ^ 目标` 之后，借位链正好只在"该道为 0"处留下 `0x80`。多个目标的位图先 OR 再取最低命中道，
/// 比逐个目标取最小快两成。误命中只会出现在真命中右侧（借位从低往高传），所以最低那一位永远是真的。
///
/// "清位续找"是不退化的关键：源码里 `/` 多为除号、`'` 多为生命周期，假候选若回到外层重开窗口，
/// 每个 `/` 都要重读一整块；就地清位则整块只读一次。
///
/// 一律按小端读：内存里第一个字节必须是"最低有效道"，`trailing_zeros() / 8` 才等于块内偏移。
/// 用本机序在 big-endian 上会把索引反过来——这条判据不做端序假设。
fn find(
    bytes: &[u8],
    from: usize,
    needles: &[u8],
    mut accept: impl FnMut(usize) -> bool,
) -> Option<usize> {
    let mut cursor = from;
    while let Some(chunk) = bytes.get(cursor..cursor + 16) {
        let word = u128::from_le_bytes(chunk.try_into().expect("切片长度恰为 16"));
        let mut lanes = needles.iter().fold(0, |lanes, &needle| {
            lanes | zero_lanes(word ^ spread(needle))
        });
        while lanes != 0 {
            let at = cursor + (lanes.trailing_zeros() / 8) as usize;
            if accept(at) {
                return Some(at);
            }
            lanes &= lanes - 1;
        }
        cursor += 16;
    }
    (cursor..bytes.len())
        .filter(|at| needles.contains(&bytes[*at]))
        .find(|at| accept(*at))
}

/// 每道放一个 `0x01`：借位链的起点。
const LANE_ONES: u128 = u128::from_le_bytes([0x01; 16]);

/// 每道放一个 `0x80`：借位链的终点，只留下"该道已归零"的标记位。
const LANE_HIGHS: u128 = u128::from_le_bytes([0x80; 16]);

/// 哪几道是 0：减一之后仍小于 `0x80` 的道（减的借位传到它自己身上）就是 0。
fn zero_lanes(word: u128) -> u128 {
    word.wrapping_sub(LANE_ONES) & !word & LANE_HIGHS
}

/// 把一个字节铺满 16 道，好跟整个窗口一次异或。
fn spread(byte: u8) -> u128 {
    u128::from_le_bytes([byte; 16])
}

/// 只要位置、不做判定的 [`find`]：注释与字符串内部只关心"下一个定界符在哪"。
fn scan(bytes: &[u8], from: usize, needles: &[u8]) -> Option<usize> {
    find(bytes, from, needles, |_| true)
}

/// 要剔掉的一段：形态由开头几字节定，正文长度等真要跳过时再算。
///
/// 分成"定性"与"量长"两半，是为了让 [`find`] 的窗口循环保持小：把正文扫描（可能是几十 KB 的
/// 字符串或嵌套注释）内联进主循环会抬高寄存器压力，实测真实语料反而慢三成。
#[derive(Clone, Copy)]
enum Lexeme {
    /// `//` 到行尾；换行本身留在代码里。
    Line,
    /// `/* … */`，可嵌套。
    Block,
    /// `"…"`，参数是开引子后的 `#` 数（raw string）。
    Str(usize),
    /// `'…'`。
    Char,
}

impl Lexeme {
    /// `at` 处的形态；裸 `/`（除号）与生命周期 `'a` 都不是注释或字面量，返回 `None`。
    ///
    /// 按首两字节分派，不切子串：`&text[at..]` 每次都要重算边界与长度，这里只读字节。
    fn of(bytes: &[u8], at: usize) -> Option<Self> {
        match bytes[at..] {
            [b'/', b'/', ..] => Some(Self::Line),
            [b'/', b'*', ..] => Some(Self::Block),
            [b'"', ..] => Some(Self::Str(raw_hashes(bytes, at))),
            [b'\'', ..] => char_end(bytes, at).map(|_| Self::Char),
            _ => None,
        }
    }

    /// 游标该停在哪。
    fn end(self, bytes: &[u8], at: usize) -> usize {
        match self {
            Self::Line => scan(bytes, at, b"\n").unwrap_or(bytes.len()),
            Self::Block => comment_end(bytes, at),
            Self::Str(hashes) => quoted_end(bytes, at, hashes),
            Self::Char => char_end(bytes, at).expect("定性时已确认是字符字面量"),
        }
    }
}

/// `from` 之后第一个要剔掉的注释 / 字面量：`(开头, 游标该停的位置)`。
fn next_token(text: &str, from: usize) -> Option<(usize, usize)> {
    let bytes = text.as_bytes();
    let at = find(bytes, from, OPENERS, |at| Lexeme::of(bytes, at).is_some())?;
    // 位置找回来之后再定性一次：`find` 只回传位置，让判据把形态带出来会给窗口循环多一个返回值，
    // 而定性只看开头两字节，重复一次比多一层交接便宜。
    Some((at, Lexeme::of(bytes, at)?.end(bytes, at)))
}

/// 只留下代码：注释与字面量都剔掉。生成文本里就写着 `format!` / `.to_string()`，直接搜全文会把
/// "被生成的代码"误判成"生成器的实现"；`'{'`、`'"'` 这类字面量还会把 [`body`] 的花括号配平带偏。
///
/// 单趟字节游标：每圈用 [`next_token`] 跳到下一个要剔掉的注释 / 字面量，把它之前那段代码整段追加、
/// 游标跳到它后面。终止性由构造保证（被吞下的片段至少 1 字节），不需要递归兜底：栈深与输入长度无关，
/// 中间也不分配 `String`。行注释的换行留在代码里，剔除后行数与原文件一一对应。
fn code_only(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut flushed = 0;
    let mut cursor = 0;
    while let Some((at, end)) = next_token(text, cursor) {
        debug_assert!(end > at, "被吞下的片段必须至少推进一个字节");
        out.push_str(&text[flushed..at]);
        flushed = end;
        cursor = end;
    }
    out.push_str(&text[flushed..]);
    out
}

/// 块注释可嵌套（Rust 特性）：`/* a /* b */ c */` 整体只是一条注释。
///
/// 开合定界符都从 `*` 或 `/` 开始，所以正文一次 [`scan`] 跳过，不逐字节走状态机。
fn comment_end(bytes: &[u8], at: usize) -> usize {
    let mut depth = 1u32;
    let mut cursor = at + 2;
    loop {
        let Some(marker) = scan(bytes, cursor, b"*/") else {
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
        scan(bytes, cursor, b"\"\\")
    } else {
        scan(bytes, cursor, b"\"")
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
fn char_end(bytes: &[u8], at: usize) -> Option<usize> {
    /// `'\u{10FFFF}'` 是最长的字符字面量（13 字节），收尾引号不会晚于第 12 个字节。
    const LONGEST: usize = 12;
    if *bytes.get(at + 1)? == b'\\' {
        let mut cursor = at + 1;
        while cursor <= at + LONGEST {
            match *bytes.get(cursor)? {
                b'\\' => cursor += 2,
                b'\'' => return Some(cursor + 1),
                _ => cursor += 1,
            }
        }
        return None;
    }
    let width = utf8_width(*bytes.get(at + 1)?);
    (bytes.get(at + 1 + width) == Some(&b'\'')).then_some(at + 2 + width)
}

/// UTF-8 首字节的字符宽度：`'` 之后不是 ASCII 时要整字跳过才看得见收尾引号。
fn utf8_width(first: u8) -> usize {
    match first {
        0x00..=0x7F => 1,
        0xC0..=0xDF => 2,
        0xE0..=0xEF => 3,
        _ => 4,
    }
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
fn rendering_allocates_only_its_output_buf() {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src/render.rs");
    let code = code_only(&fs::read_to_string(&path).expect("源码可读"));
    for entry in ["render", "render_model", "render_runtime"] {
        let count = body(&code, entry).matches("String::with_capacity").count();
        assert_eq!(count, 1, "`{entry}` 必须只分配一个输出缓冲，实际 {count}");
    }
    let hot = [
        "spec",
        "contract",
        "check_module",
        "render_runtime",
        "render_model",
        "asserts",
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

/// 只依赖编译期工具：Rust AST 工具与语料 TOML 解析都只在宏进程跑，展开后的目标 crate
/// 不承接这些运行期依赖（见 ADR 0022 补记：编译期生成器依赖 ≠ 运行期解析器）。
#[test]
fn macro_dependencies_are_compile_time_only() {
    let manifest_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("Cargo.toml");
    let manifest = fs::read_to_string(&manifest_path).expect("manifest 可读");
    let mut dependencies = section_keys(&manifest, "[dependencies]");
    dependencies.sort_unstable();
    assert_eq!(dependencies, ["proc-macro2", "quote", "syn", "toml"]);
}
