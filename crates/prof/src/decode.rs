//! 表值解码：`diagnostic::data::Value` → T 的单一口径。
//!
//! 档案层（[`crate::schema`]）与声明层（[`crate::ingest`]）读的是同一种树，失败也是同一套形态：
//! 缺、非法、闭集外。这两边原本各写一遍，于是同一个形状在两处漂移——一处为名字列表克隆出
//! `Vec<String>`、另一处逐项借用；一处报闭集内容、另一处只报一个词。
//!
//! 现在解码只有一份，失败形态也只有一个来源（[`Fail`]）：哪一层读表，就把三种形态落到哪一层的
//! 错误类型上，所以档案的「缺字段」与声明的「缺字段」各自保留自己的 `code()` 与文案，
//! 而「怎么读出一张表」不再有第二份实现。
//!
//! 名字一律**借用**着交出去：`Value` 树在整段解析里都活着，逐项克隆只会让同一个字符串
//! 多分配一次。需要所有权的调用点（`Attr` 的取值）在自己的那一行做最后一次 `to_string`。
//!
//! 段读取也只写一份：[`rows`] 把「`[X]` 每行一个条目」的两件事（缺段算空还是算错、逐行怎么解）
//! 变成两个参数，`[space]` / `[units]` / `[axes]` / `[needs]` / `[layouts]` / `[consts]`
//! 因此不再各抄一遍同一个 `let Some(section) = … else`。

use crate::atom::{Closed, Set, Span};
use core::fmt;
use diagnostic::data::{Table, Value};

// ── 失败口径 ──────────────────────────────────────────────────────────────

/// 解码失败的三种形态。哪一层读表，就把它们落到哪一层的错误类型上：
/// 档案要 `ProfileError`（缺字段与闭集外可分辨），声明要 `diagnostic::Error`（一切非法）。
pub(crate) trait Fail: Sized {
    /// 键在，值不成形。
    fn bad(path: &str, detail: &str) -> Self;
    /// 闭集外取值；`allowed` 是闭集名字，已按 ` | ` 连好（只在出错时拼）。
    fn unknown(path: &str, value: &str, allowed: &str) -> Self;
}

/// 一条"值不成形"的失败；`path` 只在出错时才渲染成字符串。
fn bad<F: Fail>(path: &impl fmt::Display, detail: &str) -> F {
    F::bad(&path.to_string(), detail)
}

// ── 段 ────────────────────────────────────────────────────────────────────

/// 缺段的口径：可选段缺了就是空，必填段缺了要给一句说明。
#[derive(Debug, Clone, Copy)]
pub(crate) enum Absent<'a> {
    /// 缺段即空。
    Empty,
    /// 缺段即错（说明即错因）。
    Required(&'a str),
}

/// 一个按键的段：缺段按 `absent` 处理，每行交给 `each`（连同段本身，需要的调用点自取）。
pub(crate) fn rows<'t, F: Fail, T>(
    table: &'t Table,
    key: &str,
    absent: Absent<'_>,
    each: impl Fn(&'t Table, &str, &'t Value) -> Result<T, F>,
) -> Result<Vec<T>, F> {
    let Some(section) = table.table(key) else {
        return match absent {
            Absent::Empty => Ok(Vec::new()),
            Absent::Required(detail) => Err(F::bad(key, detail)),
        };
    };
    section
        .entries()
        .iter()
        .map(|(name, value)| each(section, name, value))
        .collect()
}

// ── 值 ────────────────────────────────────────────────────────────────────

/// 一个字符串：只收 `Value::Str`，数字与布尔都算"不成形"。
pub(crate) fn text<'a, F: Fail>(path: impl fmt::Display, value: &'a Value) -> Result<&'a str, F> {
    match value {
        Value::Str(text) => Ok(text),
        _ => Err(bad(&path, "必须是字符串")),
    }
}

/// 一组名字：字符串原样，其余标量按十进制渲染，数组逐项。
pub(crate) fn names<F: Fail>(path: impl fmt::Display, value: &Value) -> Result<Vec<String>, F> {
    let mut out = Vec::new();
    each_text(&path, value, |name| {
        out.push(name.to_string());
        Ok(())
    })?;
    Ok(out)
}

/// 一个标量的十进制渲染：字符串原样，整数/浮点/布尔按十进制写；数组与表不是标量。
pub(crate) fn scalar_text<F: Fail>(path: impl fmt::Display, value: &Value) -> Result<String, F> {
    crate::prof::scalar_texts(value)
        .and_then(|texts| texts.into_iter().next())
        .ok_or_else(|| bad(&path, "只接受标量"))
}

/// 一个闭集里的名字；闭集外时把全量名字一并报出。
pub(crate) fn closed<E: Closed, F: Fail>(path: impl fmt::Display, name: &str) -> Result<E, F> {
    E::parse(name).ok_or_else(|| {
        let allowed: Vec<&str> = E::ALL.iter().map(|item| item.as_str()).collect();
        F::unknown(&path.to_string(), name, &allowed.join(" | "))
    })
}

/// 一个闭集位集：标量或字符串数组都接受，逐项必须是闭集里的名字。
pub(crate) fn closed_set<E: Closed, F: Fail, const WORDS: usize>(
    path: impl fmt::Display,
    value: &Value,
) -> Result<Set<E, WORDS>, F> {
    let mut set = Set::<E, WORDS>::empty();
    each_text(&path, value, |name| {
        set.insert(closed(&path, name)?);
        Ok(())
    })?;
    Ok(set)
}

/// 一个闭区间：文本按 [`Span::parse`] 的写法解析，解析失败的原话一并带上。
pub(crate) fn span<F: Fail>(path: impl fmt::Display, text: &str) -> Result<Span, F> {
    Span::parse(text).map_err(|error| bad(&path, error.message()))
}

/// 一个正整数：`0`、负数、非整数与装不进 `u32` 的取值都是"不成形"。
pub(crate) fn positive<F: Fail>(path: impl fmt::Display, value: &Value) -> Result<u32, F> {
    let Value::Int(number) = value else {
        return Err(bad(&path, "每个取值必须是正整数"));
    };
    u32::try_from(*number)
        .ok()
        .filter(|number| *number > 0)
        .ok_or_else(|| bad(&path, "每个取值必须是正整数"))
}

/// 一个标量或一组名字：标量按十进制渲染（[`crate::prof::scalar_texts`] 是唯一渲染口径）。
fn each_text<F: Fail>(
    path: &impl fmt::Display,
    value: &Value,
    mut each: impl FnMut(&str) -> Result<(), F>,
) -> Result<(), F> {
    if let Value::Array(items) = value {
        for item in items {
            let Value::Str(text) = item else {
                return Err(bad(path, "数组里只能是字符串"));
            };
            each(text)?;
        }
        return Ok(());
    }
    let Some(texts) = crate::prof::scalar_texts(value) else {
        return Err(bad(path, "只接受字符串或字符串数组"));
    };
    texts.iter().try_for_each(|text| each(text))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::atom::Prim;

    /// 测试用的失败口径：三种形态渲染成一句话，断言只看这句话。
    #[derive(Debug, PartialEq, Eq)]
    struct Note(String);
    impl Fail for Note {
        fn bad(path: &str, detail: &str) -> Self {
            Self(format!("`{path}` 非法：{detail}"))
        }
        fn unknown(path: &str, value: &str, allowed: &str) -> Self {
            Self(format!("`{path} = {value}` 不在闭集内（闭集：{allowed}）"))
        }
    }

    #[test]
    fn rows_tell_absent_sections_apart() {
        let table = Table::new();
        let optional: Vec<()> =
            rows(&table, "space", Absent::Empty, |_, _, _| Ok::<(), Note>(())).unwrap();
        assert!(optional.is_empty());
        let required = rows(&table, "axes", Absent::Required("至少需要一个轴"), |_, _, _| {
            Ok::<(), Note>(())
        });
        assert_eq!(
            required.unwrap_err(),
            Note("`axes` 非法：至少需要一个轴".to_string())
        );
    }

    #[test]
    fn rows_hand_the_section_to_every_row() {
        let mut table = Table::new();
        table.set(
            "axes",
            Value::table(vec![
                ("m", Value::Array(vec![Value::Int(1), Value::Int(2)])),
                ("n", Value::Array(vec![Value::Int(3)])),
            ]),
        );
        let counts: Vec<usize> = rows(&table, "axes", Absent::Empty, |section, _, value| {
            assert_eq!(section.entries().len(), 2, "每一行都拿到同一段");
            Ok::<usize, Note>(value_array_len(value))
        })
        .unwrap();
        assert_eq!(counts, [2, 1]);
    }

    #[test]
    fn names_render_scalars_and_refuse_mixed_arrays() {
        assert_eq!(
            names::<Note>("dt", &Value::Str("f16".to_string())).unwrap(),
            ["f16"]
        );
        assert_eq!(names::<Note>("sat", &Value::Int(3)).unwrap(), ["3"]);
        let mixed = Value::Array(vec![Value::Str("a".to_string()), Value::Int(1)]);
        assert_eq!(
            names::<Note>("x", &mixed).unwrap_err(),
            Note("`x` 非法：数组里只能是字符串".to_string())
        );
    }

    #[test]
    fn closed_set_reports_the_whole_closed_set() {
        let value = Value::Array(vec![
            Value::Str("dot".to_string()),
            Value::Str("nope".to_string()),
        ]);
        let error = closed_set::<Prim, Note, 1>("prim", &value).unwrap_err();
        let allowed: Vec<&str> = Prim::ALL.iter().map(|prim| prim.as_str()).collect();
        let expected = format!("`prim = nope` 不在闭集内（闭集：{}）", allowed.join(" | "));
        assert_eq!(error, Note(expected));
    }

    #[test]
    fn positive_refuses_zero_negative_and_overflow() {
        for bad in [
            Value::Int(0),
            Value::Int(-1),
            Value::Int(1 << 40),
            Value::Bool(true),
        ] {
            assert!(positive::<Note>("n", &bad).is_err(), "{bad:?} 不该通过");
        }
        assert_eq!(positive::<Note>("n", &Value::Int(16)).unwrap(), 16);
    }

    fn value_array_len(value: &Value) -> usize {
        match value {
            Value::Array(items) => items.len(),
            _ => 0,
        }
    }
}
