use super::*;

/// 入口函数的括号区间：`cutile_{op}_kernel(` 的 `(` 与配对的 `)`。
///
/// 模板保留 `{{op}}` 插值，已展开的文本也接受（测试与将来的内联模板）。
pub(super) fn entry(op: &str, template: &str) -> diagnostic::Result<(usize, usize)> {
    let marker = format!("cutile_{op}_kernel(");
    // 一次定位：`find` 已经给出起点，再 `contains` 一遍只是把同一段文本扫两次。
    let (start, width) = [marker.as_str(), "cutile_{{op}}_kernel("]
        .into_iter()
        .find_map(|entry| template.find(entry).map(|start| (start, entry.len())))
        .ok_or_else(|| {
            at(
                1,
                format!("算子 `{op}` 的模板缺少内核入口 `cutile_{{op}}_kernel(`"),
            )
        })?;
    let open = start + width - 1;
    let close = matching_paren(template, open).ok_or_else(|| {
        at(
            line_of(template, open),
            format!("算子 `{op}` 的内核签名括号不配对"),
        )
    })?;
    Ok((open, close))
}

/// 括号里的形参表：`void` 与空表都是「没有形参」，其余按顶层逗号切开逐个解析。
pub(super) fn params(
    op: &str,
    template: &str,
    open: usize,
    close: usize,
) -> diagnostic::Result<Vec<Param>> {
    let inner = &template[open + 1..close];
    if inner.trim() == "void" || inner.trim().is_empty() {
        return Ok(Vec::new());
    }
    split(inner)
        .into_iter()
        .map(|(offset, text)| {
            // 参数可能从下一行开始：行号要落在非空白字符上，而不是逗号之后。
            let indent = inner[offset..].len() - inner[offset..].trim_start().len();
            parse_param(op, text, line_of(template, open + 1 + offset + indent))
        })
        .collect()
}

/// 按顶层逗号切分形参：切分原语是 [`split_top_level`]，这里只加「裁剪 + 丢空尾」的口径。
pub(super) fn split(inner: &str) -> Vec<(usize, &str)> {
    let mut parts: Vec<(usize, &str)> = split_top_level(inner)
        .into_iter()
        .map(|(offset, text)| (offset, text.trim()))
        .collect();
    if parts.last().is_some_and(|(_, tail)| tail.is_empty()) {
        parts.pop();
    }
    parts
}

/// 一个形参：先认名字（最后一个词），再按「指针 / 标量」两条闭集解析类型。
pub(super) fn parse_param(op: &str, text: &str, line: usize) -> diagnostic::Result<Param> {
    let pointer = text.contains('*');
    let words = words(text);
    let name = param_name(op, &words, text, line)?;
    let ty = type_text(&words);
    let ctype = ctype(op, name, &ty, line, pointer)?;
    Ok(Param {
        name: name.to_string(),
        ctype,
        mutable: pointer && !words.contains(&"const"),
    })
}

/// C 类型：指针读声明表里的缓冲类型，其余是标量类型。
pub(super) fn ctype(
    op: &str,
    name: &str,
    ty: &str,
    line: usize,
    pointer: bool,
) -> diagnostic::Result<CType> {
    if pointer {
        buffer_ctype(op, name, ty, line)
    } else {
        scalar_ctype(op, name, ty, line)
    }
}

/// 声明里的词：按空白与 `*` 切开，`const` 这类限定词与类型名都在里面。
pub(super) fn words(text: &str) -> Vec<&str> {
    text.split(|ch: char| ch.is_whitespace() || ch == '*')
        .filter(|token| !token.is_empty())
        .collect()
}

/// 形参名：最后一个词，必须是 C 标识符。
pub(super) fn param_name<'a>(
    op: &str,
    words: &'a [&'a str],
    text: &str,
    line: usize,
) -> diagnostic::Result<&'a str> {
    let Some(name) = words.last() else {
        return Err(at(
            line,
            format!("算子 `{op}` 的签名里有一个空的形参声明 `{text}`"),
        ));
    };
    if name
        .chars()
        .all(|ch| ch.is_ascii_alphanumeric() || ch == '_')
    {
        return Ok(name);
    }
    Err(at(
        line,
        format!("算子 `{op}` 的形参名 `{name}` 不是 C 标识符"),
    ))
}

/// 类型文本：去掉形参名与 `const`/`volatile`/`restrict` 这些限定词。
pub(super) fn type_text(words: &[&str]) -> String {
    words
        .iter()
        .take(words.len().saturating_sub(1))
        .filter(|token| !matches!(**token, "const" | "volatile" | "restrict" | "__restrict"))
        .copied()
        .collect::<Vec<_>>()
        .join(" ")
}

/// 缓冲形参的类型：只有 `float` 与 `unsigned char`/`u8`。
pub(super) fn buffer_ctype(
    op: &str,
    name: &str,
    ty: &str,
    line: usize,
) -> diagnostic::Result<CType> {
    match ty {
        "float" => Ok(CType::F32),
        "u8" | "unsigned char" => Ok(CType::U8),
        _ => Err(at(
            line,
            format!(
                "算子 `{op}` 的缓冲形参 `{name}` 用类型 `{ty} *`；支持的缓冲类型只有 `float`（f32）与 `unsigned char`/`u8`"
            ),
        )),
    }
}

/// 标量形参的类型：只支持整数标量。
pub(super) fn scalar_ctype(
    op: &str,
    name: &str,
    ty: &str,
    line: usize,
) -> diagnostic::Result<CType> {
    match ty {
        "int" | "long" | "long long" | "unsigned" | "unsigned int" | "unsigned long" | "size_t" => {
            Ok(CType::Int)
        }
        _ => Err(at(
            line,
            format!(
                "算子 `{op}` 的标量形参 `{name}` 用类型 `{ty}`；只支持整数标量（int、long、unsigned、size_t）"
            ),
        )),
    }
}
