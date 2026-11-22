//! 表值解码：`diagnostic::data::Value` → T 的单一口径。
//!
//! 档案层（[`crate::schema`]）与声明层（[`crate::ingest`]）读的是同一种树，失败也是同一套形态：
//! 缺、非法、闭集外。这两边原本各写一遍，于是同一个形状在两处漂移——一处为名字列表克隆出
//! `Vec<String>`、另一处逐项借用；一处报闭集内容、另一处只报一个词。
//!
//! 现在解码只有一份，失败也只有一个值类型 [`Rejected`]：它记着路径与形态，文案只有一个口径
//! （`缺字段 x` / `x` 非法：…/ `x = v` 不在闭集内（闭集：…））。要落到哪一层的错误类型上由
//! `From` 决定——档案转成 [`ProfileError`](crate::prof::ProfileError)，三种形态各自成变体、
//! `code()` 仍能分辨"缺字段"；声明转成 [`diagnostic::Error`]，一切非法都是同一类。
//!
//! 名字一律**借用**着交出去：`Value` 树在整段解析里都活着，逐项克隆只会让同一个字符串
//! 多分配一次。需要所有权的调用点（`Attr` 的取值）在自己的那一行做最后一次 `to_string`。
//!
//! 段读取也只写一份：[`rows`] 把「`[X]` 每行一个条目」的两件事（缺段算空还是算错、逐行怎么解）
//! 变成两个参数，`[space]` / `[units]` / `[axes]` / `[needs]` / `[layouts]` / `[consts]`
//! 因此不再各抄一遍同一个 `let Some(section) = … else`。

use crate::atom::{Closed, Set, Span};
use crate::prof::ProfileError;
use core::fmt;
use diagnostic::Code;
use diagnostic::data::{Table, Value};

// ── 失败 ──────────────────────────────────────────────────────────────────

/// 一条解码失败：哪条路径、哪一种形态。
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Rejected {
    path: String,
    kind: Rejection,
}

/// 失败的三种形态：缺、非法、闭集外。
#[derive(Debug, Clone, PartialEq, Eq)]
enum Rejection {
    /// 键不存在。
    Missing,
    /// 键在，值不成形。
    Bad(String),
    /// 闭集外取值。
    Unknown { value: String, allowed: String },
}

impl fmt::Display for Rejected {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let path = &self.path;
        match &self.kind {
            Rejection::Missing => write!(f, "缺字段 `{path}`"),
            Rejection::Bad(detail) => write!(f, "`{path}` 非法：{detail}"),
            Rejection::Unknown { value, allowed } => {
                write!(f, "`{path} = {value}` 不在闭集内（闭集：{allowed}）")
            }
        }
    }
}

impl std::error::Error for Rejected {}

impl From<Rejected> for ProfileError {
    /// 三种形态各自成变体：`Missing` 与 `BadValue` / `Unknown` 的可分辨性是档案层的契约。
    fn from(rejected: Rejected) -> Self {
        let path = rejected.path;
        match rejected.kind {
            Rejection::Missing => Self::Missing { field: path },
            Rejection::Bad(detail) => Self::BadValue {
                field: path,
                detail,
            },
            Rejection::Unknown { value, allowed } => Self::Unknown {
                field: path,
                value,
                allowed,
            },
        }
    }
}

impl From<Rejected> for diagnostic::Error {
    /// 声明层的一切非法都是同一类，只有"缺"能被 `code()` 分辨。
    fn from(rejected: Rejected) -> Self {
        let code = match rejected.kind {
            Rejection::Missing => Code::Missing,
            _ => Code::Invalid,
        };
        Self::new(code, rejected.to_string())
    }
}

/// 缺字段。
fn missing<X>(path: impl fmt::Display) -> Result<X, Rejected> {
    Err(Rejected {
        path: path.to_string(),
        kind: Rejection::Missing,
    })
}

/// 键在但值不成形。
fn bad<X>(path: impl fmt::Display, detail: impl Into<String>) -> Result<X, Rejected> {
    Err(Rejected {
        path: path.to_string(),
        kind: Rejection::Bad(detail.into()),
    })
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
pub(crate) fn rows<'t, T>(
    table: &'t Table,
    key: &str,
    absent: Absent<'_>,
    each: impl Fn(&'t Table, &str, &'t Value) -> Result<T, Rejected>,
) -> Result<Vec<T>, Rejected> {
    let Some(section) = table.table(key) else {
        return match absent {
            Absent::Empty => Ok(Vec::new()),
            Absent::Required(detail) => bad(key, detail),
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
pub(crate) fn text<'a>(path: impl fmt::Display, value: &'a Value) -> Result<&'a str, Rejected> {
    match value {
        Value::Str(text) => Ok(text),
        _ => bad(&path, "必须是字符串"),
    }
}

/// 一组名字：字符串原样，其余标量按十进制渲染，数组逐项。
pub(crate) fn names(path: impl fmt::Display, value: &Value) -> Result<Vec<String>, Rejected> {
    let mut out = Vec::new();
    each_text(&path, value, |name| {
        out.push(name.to_string());
        Ok(())
    })?;
    Ok(out)
}

/// 一个标量的十进制渲染：字符串原样，整数/浮点/布尔按十进制写；数组与表不是标量。
pub(crate) fn scalar_text(path: impl fmt::Display, value: &Value) -> Result<String, Rejected> {
    crate::prof::scalar_texts(value)
        .and_then(|texts| texts.into_iter().next())
        .ok_or_else(|| mismatch(&path, "只接受标量"))
}

/// 一个闭集里的名字；闭集外时把全量名字一并报出。
pub(crate) fn closed<E: Closed>(path: impl fmt::Display, name: &str) -> Result<E, Rejected> {
    E::parse(name).ok_or_else(|| {
        let allowed: Vec<&str> = E::ALL.iter().map(|item| item.as_str()).collect();
        Rejected {
            path: path.to_string(),
            kind: Rejection::Unknown {
                value: name.to_string(),
                allowed: allowed.join(" | "),
            },
        }
    })
}

/// 一个闭集位集：标量或字符串数组都接受，逐项必须是闭集里的名字。
pub(crate) fn closed_set<E: Closed, const WORDS: usize>(
    path: impl fmt::Display,
    value: &Value,
) -> Result<Set<E, WORDS>, Rejected> {
    let mut set = Set::<E, WORDS>::empty();
    each_text(&path, value, |name| {
        set.insert(closed(&path, name)?);
        Ok(())
    })?;
    Ok(set)
}

/// 一个闭区间：文本按 [`Span::parse`] 的写法解析，解析失败的原话一并带上。
pub(crate) fn span(path: impl fmt::Display, text: &str) -> Result<Span, Rejected> {
    Span::parse(text).map_err(|error| mismatch(&path, error.message()))
}

/// 一个正整数：`0`、负数、非整数与装不进 `u32` 的取值都是"不成形"。
pub(crate) fn positive(path: impl fmt::Display, value: &Value) -> Result<u32, Rejected> {
    let Value::Int(number) = value else {
        return bad(&path, "每个取值必须是正整数");
    };
    u32::try_from(*number)
        .ok()
        .filter(|number| *number > 0)
        .ok_or_else(|| mismatch(&path, "每个取值必须是正整数"))
}

/// 一个标量或一组名字：标量按十进制渲染（[`crate::prof::scalar_texts`] 是唯一渲染口径）。
fn each_text(
    path: &impl fmt::Display,
    value: &Value,
    mut each: impl FnMut(&str) -> Result<(), Rejected>,
) -> Result<(), Rejected> {
    if let Value::Array(items) = value {
        for item in items {
            let Value::Str(text) = item else {
                return bad(path, "数组里只能是字符串");
            };
            each(text)?;
        }
        return Ok(());
    }
    let Some(texts) = crate::prof::scalar_texts(value) else {
        return bad(path, "只接受字符串或字符串数组");
    };
    texts.iter().try_for_each(&mut each)
}

/// 一条"不成形"的失败（`Result` 形态，`ok_or_else` 用）。
fn mismatch(path: &impl fmt::Display, detail: &str) -> Rejected {
    Rejected {
        path: path.to_string(),
        kind: Rejection::Bad(detail.to_string()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::atom::Prim;

    #[test]
    fn rows_tell_absent_sections_apart() {
        let table = Table::new();
        let optional: Vec<()> = rows(&table, "space", Absent::Empty, |_, _, _| Ok(())).unwrap();
        assert!(optional.is_empty());
        let required = rows(&table, "axes", Absent::Required("至少需要一个轴"), |_, _, _| {
            Ok::<(), Rejected>(())
        });
        assert_eq!(required.unwrap_err().to_string(), "`axes` 非法：至少需要一个轴");
    }

    #[test]
    fn rows_hand_the_section_to_every_row() {
        let mut table = Table::new();
        table.set("axes.m", Value::Array(vec![Value::Int(1), Value::Int(2)]));
        table.set("axes.n", Value::Array(vec![Value::Int(3)]));
        let counts: Vec<usize> = rows(&table, "axes", Absent::Empty, |section, _, value| {
            assert_eq!(section.entries().len(), 2, "每一行都拿到同一段");
            Ok(value_array_len(value))
        })
        .unwrap();
        assert_eq!(counts, [2, 1]);
    }

    #[test]
    fn names_render_scalars_and_refuse_mixed_arrays() {
        assert_eq!(names("dt", &Value::Str("f16".to_string())).unwrap(), ["f16"]);
        assert_eq!(names("sat", &Value::Int(3)).unwrap(), ["3"]);
        let mixed = Value::Array(vec![Value::Str("a".to_string()), Value::Int(1)]);
        assert_eq!(
            names("x", &mixed).unwrap_err().to_string(),
            "`x` 非法：数组里只能是字符串"
        );
    }

    #[test]
    fn closed_set_reports_the_whole_closed_set() {
        let value = Value::Array(vec![
            Value::Str("dot".to_string()),
            Value::Str("nope".to_string()),
        ]);
        let error = closed_set::<Prim, 1>("prim", &value).unwrap_err();
        let allowed: Vec<&str> = Prim::ALL.iter().map(|prim| prim.as_str()).collect();
        assert_eq!(
            error.to_string(),
            format!("`prim = nope` 不在闭集内（闭集：{}）", allowed.join(" | "))
        );
    }

    #[test]
    fn positive_refuses_zero_negative_and_overflow() {
        for bad in [
            Value::Int(0),
            Value::Int(-1),
            Value::Int(1 << 40),
            Value::Bool(true),
        ] {
            assert!(positive("n", &bad).is_err(), "{bad:?} 不该通过");
        }
        assert_eq!(positive("n", &Value::Int(16)).unwrap(), 16);
    }

    fn value_array_len(value: &Value) -> usize {
        match value {
            Value::Array(items) => items.len(),
            _ => 0,
        }
    }
}
