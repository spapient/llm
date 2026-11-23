//! 扫描前预处理：剔除 `#[cfg(test)]` 条目并保留行号。

use crate::lex::{Delim, Token, parse};
use diagnostic::Result;

/// 剔除 cfg(test) 条目；源码无法解析返回 invalid 错误。
pub fn strip_tests(src: &str) -> Result<String> {
    let tokens = parse(src)?;
    let mut blank = vec![false; src.lines().count()];
    collect(&tokens, &mut blank);

    let out: Vec<&str> = src
        .lines()
        .enumerate()
        .map(|(at, line)| if blank[at] { "" } else { line })
        .collect();
    Ok(out.join("\n"))
}

/// 一条 `mod 名字;` 挂载：模块名与可选的显式 `#[path]` 相对路径。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Mount {
    /// 模块名（`mod` 后的标识符）。
    pub name: String,
    /// `#[path = "..."]` 显式路径；无属性时为 `None`，按模块名推断目录。
    pub path: Option<String>,
}

/// 文件顶层以 `#[cfg(test)] mod 名字;` 挂载的子模块名。
pub fn test_modules(src: &str) -> Result<Vec<Mount>> {
    modules(src, true)
}

/// 文件顶层 `mod 名字;` 声明的全部子模块名，不看 cfg 口径。
pub fn submodules(src: &str) -> Result<Vec<Mount>> {
    modules(src, false)
}

/// 顶层 `mod 名字;` 声明；`only_test` 为真时只收 cfg(test) 标注过的。
fn modules(src: &str, only_test: bool) -> Result<Vec<Mount>> {
    let tokens = parse(src)?;
    let mut out = Vec::new();
    split_items(&tokens, |attr, item| {
        if let Some(mount) = declared(item, only_test, attr.is_test, attr.path.clone()) {
            out.push(mount);
        }
    });
    Ok(out)
}

/// 一条条目贡献的模块名：`mod 名字;` 形态，且（未限定口径或该条目标注了 cfg(test)）。
fn declared(item: &[Token], only_test: bool, test: bool, path: Option<String>) -> Option<Mount> {
    let wanted = !only_test || test;
    let name = declared_module(item)?;
    wanted.then_some(Mount { name, path })
}

/// 条目是否 `mod 名字;` 形态：返回模块名，带体形式（`mod 名字 { .. }`）与其他条目返回 `None`。
fn declared_module(item: &[Token]) -> Option<String> {
    let last = item.last()?;
    if !matches!(last, Token::Punct { ch: ';', .. }) {
        return None;
    }
    item.windows(2).find_map(|pair| match pair {
        [
            Token::Ident { text: keyword, .. },
            Token::Ident { text: name, .. },
        ] if keyword == "mod" => Some(name.clone()),
        _ => None,
    })
}

/// 一层里的属性归属：首个属性的起始行与是否 `cfg(test)`。
#[derive(Default)]
struct Attr {
    start: Option<usize>,
    is_test: bool,
    path: Option<String>,
}

impl Attr {
    /// 吸收一个属性：首个属性的起始行与 `cfg(test)` 口径；`#[path]` 字面量一并取出。
    fn absorb(&mut self, hash: &Token, bracket: &Token) {
        self.start = self.start.or(Some(hash.start_line()));
        if let Token::Group { items, .. } = bracket {
            self.is_test |= is_cfg_test(items);
            if self.path.is_none() {
                self.path = path_attr(items);
            }
        }
    }

    /// 口径为 test 时，把条目覆盖的行置空。
    fn blank_if_test(&self, last: &Token, blank: &mut [bool]) {
        if !self.is_test {
            return;
        }
        let from = self.start.unwrap_or(last.start_line());
        for line in from..=last.end_line() {
            if let Some(flag) = blank.get_mut(line - 1) {
                *flag = true;
            }
        }
    }
}

/// 递归遍历一层 token：标记本层 cfg(test) 条目，并进入分组处理嵌套的 cfg(test)。
fn collect(tokens: &[Token], blank: &mut [bool]) {
    split_items(tokens, |attr, item| {
        let Some(last) = item.last() else {
            return;
        };
        attr.blank_if_test(last, blank);
        for token in item {
            if let Token::Group { items, .. } = token {
                collect(items, blank);
            }
        }
    });
}

/// 按「属性 → 条目」切分一层 token：连续属性先吸收进 [`Attr`]，遇到条目把已吸收的属性
/// 与条目 token 段交给 `visit`；`modules` 与 `collect` 共用这一份下标推进。
fn split_items(tokens: &[Token], mut visit: impl FnMut(Attr, &[Token])) {
    let mut attr = Attr::default();
    let mut idx = 0;
    while idx < tokens.len() {
        if let Some((hash, bracket)) = attr_at(tokens, idx) {
            attr.absorb(&tokens[hash], &tokens[bracket]);
            idx = bracket + 1;
            continue;
        }
        let end = item_end(tokens, idx);
        visit(attr, &tokens[idx..=end]);
        attr = Attr::default();
        idx = end + 1;
    }
}

/// 属性起始：返回 `#` 下标与方括号分组下标。
fn attr_at(tokens: &[Token], idx: usize) -> Option<(usize, usize)> {
    let Some(Token::Punct { ch: '#', .. }) = tokens.get(idx) else {
        return None;
    };
    match tokens.get(idx + 1) {
        Some(Token::Group {
            delim: Delim::Bracket,
            ..
        }) => Some((idx, idx + 1)),
        Some(Token::Punct { ch: '!', .. }) => match tokens.get(idx + 2) {
            Some(Token::Group {
                delim: Delim::Bracket,
                ..
            }) => Some((idx, idx + 2)),
            _ => None,
        },
        _ => None,
    }
}

/// 从条目首个 token 起，定位其结束 token：带体的条目结束于花括号分组，否则结束于分号。
fn item_end(tokens: &[Token], first: usize) -> usize {
    let mut idx = first;
    while idx < tokens.len() {
        match &tokens[idx] {
            Token::Group {
                delim: Delim::Brace,
                ..
            }
            | Token::Punct { ch: ';', .. } => return idx,
            _ => idx += 1,
        }
    }
    first
}

/// `#[path = "..."]` 属性里的相对路径字面量；无该属性返回 `None`。
fn path_attr(items: &[Token]) -> Option<String> {
    let mut cur = 0;
    while cur + 2 < items.len() {
        if path_literal(&items[cur], &items[cur + 1], &items[cur + 2]) {
            let Token::Literal { text: lit, .. } = &items[cur + 2] else {
                return None;
            };
            return Some(unquote(lit));
        }
        cur += 1;
    }
    None
}

/// 三连 token 是否为 `path = "..."`：标识符、等号、字符串字面量依次就位。
fn path_literal(ident: &Token, eq: &Token, lit: &Token) -> bool {
    let Token::Ident { text, .. } = ident else {
        return false;
    };
    let is_path = text == "path";
    let is_eq = matches!(eq, Token::Punct { ch: '=', .. });
    let is_lit = matches!(lit, Token::Literal { .. });
    is_path && is_eq && is_lit
}

/// 剥字符串字面量外层引号（兼容 `r"` / `br"` 前缀），取中间路径文本。
fn unquote(literal: &str) -> String {
    let Some(first) = literal.find('"') else {
        return literal.to_owned();
    };
    let last = literal.rfind('"').unwrap_or(first);
    literal[first + 1..last].to_owned()
}

/// 方括号分组内容是否是 `cfg(..)` 且口径为 test。
fn is_cfg_test(items: &[Token]) -> bool {
    let mut cursor = 0;
    while cursor + 1 < items.len() {
        if let Token::Ident { text, .. } = &items[cursor]
            && text == "cfg"
            && let Token::Group {
                delim: Delim::Paren,
                items: body,
                ..
            } = &items[cursor + 1]
        {
            return is_test_body(&render(body));
        }
        cursor += 1;
    }
    false
}

/// 口径：纯 `test`、`all(.. test ..)`、`any(.. test ..)` 都算测试编译口径；
/// 只要出现 `not(test)` 就不剔除（`any(not(test), ..)` 也不剔）。
fn is_test_body(body: &str) -> bool {
    let compact: String = body.chars().filter(|ch| !ch.is_whitespace()).collect();
    if compact.contains("not(test)") {
        return false;
    }
    compact == "test" || wrapped_in_test_cfg(&compact)
}

/// `all(..)` / `any(..)` 包装里是否含独立 `test` 分支。
fn wrapped_in_test_cfg(compact: &str) -> bool {
    ["all(", "any("]
        .iter()
        .any(|prefix| compact.starts_with(prefix) && contains_token(compact, "test"))
}

/// `test` 是否作为独立 token 出现（两侧边界都不是标识符字符）。
fn contains_token(body: &str, token: &str) -> bool {
    let boundary = |ch: Option<char>| ch.is_none_or(|c| !c.is_alphanumeric() && c != '_');
    body.match_indices(token).any(|(at, _)| {
        let before = body[..at].chars().next_back();
        let after = body[at + token.len()..].chars().next();
        boundary(before) && boundary(after)
    })
}

/// 把 token 渲染回紧凑文本（字面量保留原文，分组补括号）。
fn render(tokens: &[Token]) -> String {
    tokens.iter().map(piece).collect()
}

/// 单个 token 的紧凑文本。
fn piece(token: &Token) -> String {
    match token {
        Token::Ident { text, .. } | Token::Literal { text, .. } => text.clone(),
        Token::Punct { ch, .. } => ch.to_string(),
        Token::Group { delim, items, .. } => {
            format!("{}{}{}", delim.open(), render(items), delim.close())
        }
    }
}
