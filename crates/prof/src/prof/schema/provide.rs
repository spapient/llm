//! 职责：名字集合字段解析：标量或数组逐项解析。
use super::super::*;

/// 一个"名字集合"字段：标量字符串或字符串数组，逐项交给 `each`。
/// `dt` 与 `space` 是同一形状，且都只把名字解析成位集——这里不克隆 `String`。
pub(super) fn each_name(
    field: impl fmt::Display,
    value: &Value,
    mut each: impl FnMut(&str) -> Result<(), ProfileError>,
) -> Result<(), ProfileError> {
    match value {
        Value::Str(text) => each(text),
        Value::Array(items) => each_in_array(field, items, each),
        _ => Err(bad(field, "只接受字符串或字符串数组")),
    }
}
/// 数组形态：逐项必须是字符串，遇到别的类型就报错（不静默跳过）。
fn each_in_array(
    field: impl fmt::Display,
    items: &[Value],
    mut each: impl FnMut(&str) -> Result<(), ProfileError>,
) -> Result<(), ProfileError> {
    for item in items {
        let Value::Str(text) = item else {
            return Err(bad(&field, "数组里只能是字符串"));
        };
        each(text)?;
    }
    Ok(())
}
/// 一组存储层名字：每个都必须在 [`Space`] 闭集里。
pub(super) fn parse_space_set(
    field: impl fmt::Display,
    items: &[&str],
) -> Result<SpaceSet, ProfileError> {
    let mut set = SpaceSet::empty();
    for item in items {
        set.insert(closed::<Space>(item, &field)?);
    }
    Ok(set)
}
/// 一条 `[[provides]]`：认得的键各有分支，其余一律当属性（键还要过 [`AttrKey`] 闭集）。
pub(super) fn parse_provide(entry: &Table, index: usize) -> Result<Provide, ProfileError> {
    let mut provide = head_of(entry, index)?;
    provide_axes(&mut provide, entry, index)?;
    provide_types(&mut provide, entry, index)?;
    provide_tail(&mut provide, entry, index)?;
    Ok(provide)
}
/// 一条 `[[provides]]` 的头两项：`prim` 必填且必须在闭集里，`impl` 缺省为 `-`。
fn head_of(entry: &Table, index: usize) -> Result<Provide, ProfileError> {
    let field = at(index, "prim");
    let prim = closed::<Prim>(text(entry, "prim", field)?, field)?;
    Ok(Provide::of(prim, entry.string("impl").unwrap_or("-")))
}
/// 三条区间：主维度（`m` / `bytes` / `extent` 三选一）与 `n` / `k`。
fn provide_axes(provide: &mut Provide, entry: &Table, index: usize) -> Result<(), ProfileError> {
    provide.extent = span_of(entry, index, &["m", "bytes", "extent"], EXTENT_DETAIL)?;
    provide.n = span_of(entry, index, &["n"], AXIS_DETAIL)?;
    provide.k = span_of(entry, index, &["k"], AXIS_DETAIL)?;
    Ok(())
}
/// `dt` / `space` 缺省为空集，`grade` 必填。
fn provide_types(provide: &mut Provide, entry: &Table, index: usize) -> Result<(), ProfileError> {
    provide.dt = dtype_set(entry, index)?;
    provide.space = space_set(entry, index)?;
    provide.grade = grade_of(entry, index)?;
    Ok(())
}
/// 见证与其余属性：两者都只在签名写完之后才看得懂。
fn provide_tail(provide: &mut Provide, entry: &Table, index: usize) -> Result<(), ProfileError> {
    provide.witness = witness_of(entry, index)?;
    attrs(entry, index, provide)
}
/// 一组同义维度：没写就是不限，多写就是错，写了就要能解析。
pub(super) fn span_of(
    entry: &Table,
    index: usize,
    names: &[&str],
    detail: &str,
) -> Result<Span, ProfileError> {
    let hit = |name: &&str| entry.contains(name);
    let used: Vec<&str> = names.iter().copied().filter(hit).collect();
    let Some(alias) = used.first().copied() else {
        return Ok(Span::any());
    };
    if used.len() > 1 {
        let count = used.len();
        let aliases = "主维度有三个同义名字（`m` / `bytes` / `extent`），只能写一个";
        let detail = format!("{aliases}，这里写了 {count}");
        return Err(bad(at(index, alias), detail));
    }
    let field = at(index, alias);
    let Some(text) = entry.string(alias) else {
        return Err(bad(field, detail));
    };
    parse_span(field, text)
}
/// `dt`：标量或数组都接受，闭集外取值报 `Unknown`。
pub(super) fn dtype_set(entry: &Table, index: usize) -> Result<DTypeSet, ProfileError> {
    let Some(value) = entry.get("dt") else {
        return Ok(DTypeSet::empty());
    };
    let field = at(index, "dt");
    let mut set = DTypeSet::empty();
    each_name(field, value, |name| {
        set.insert(closed_dtype(field, name)?);
        Ok(())
    })?;
    Ok(set)
}
/// 一个数据类型名：`DType` 是 L1 外部类型，按它的规范名解析。
pub(super) fn closed_dtype(path: impl fmt::Display, name: &str) -> Result<DType, ProfileError> {
    DType::parse_name(name).ok_or_else(|| unknown(path, name, &dtype_names()))
}
/// `space`：标量或数组都接受，缺省是空集（无存储层约束）。
pub(super) fn space_set(entry: &Table, index: usize) -> Result<SpaceSet, ProfileError> {
    let Some(value) = entry.get("space") else {
        return Ok(SpaceSet::empty());
    };
    let field = at(index, "space");
    let mut set = SpaceSet::empty();
    each_name(field, value, |name| {
        set.insert(closed::<Space>(name, field)?);
        Ok(())
    })?;
    Ok(set)
}
/// `grade`：必填，闭集外取值报 `Unknown`。
pub(super) fn grade_of(entry: &Table, index: usize) -> Result<Grade, ProfileError> {
    let path = at(index, "grade");
    closed::<Grade>(text(entry, "grade", path)?, &path)
}
/// `witness`：内联表 `{ file, func, expect }`；只有路径的见证无法执行。
pub(super) fn witness_of(entry: &Table, index: usize) -> Result<Option<Witness>, ProfileError> {
    let Some(table) = entry.table("witness") else {
        if entry.contains("witness") {
            return Err(bad(at(index, "witness"), WITNESS_DETAIL));
        }
        return Ok(None);
    };
    Ok(Some(witness_fields(table, |key| witness_at(index, key))?))
}
/// 见证的三个字段：`file` / `func` 必填，`expect` 是数值数组。字段路径由调用方给：
/// `[[provides]]` 与 `[[epilogue]]` 共用这一段，报错位置却必须落在各自的段里。
pub(super) fn witness_fields(
    table: &Table,
    at: impl Fn(&'static str) -> Field<'static>,
) -> Result<Witness, ProfileError> {
    let file = text(table, "file", at("file"))?.to_string();
    let func = text(table, "func", at("func"))?.to_string();
    let expect = expect_values(table, at("expect"))?;
    Ok(Witness { file, func, expect })
}
/// `expect`：数值数组，按顺序与见证输出对拍。
pub(super) fn expect_values(table: &Table, path: Field<'_>) -> Result<Vec<f32>, ProfileError> {
    let items = table.array("expect").ok_or_else(|| missing(path))?;
    let number = |value: &Value| expect_number(value, path);
    items.iter().map(number).collect()
}
pub(super) fn expect_number(value: &Value, path: impl fmt::Display) -> Result<f32, ProfileError> {
    match value {
        Value::Float(number) => Ok(*number as f32),
        Value::Int(number) => Ok(*number as f32),
        _ => Err(bad(path, "只能是数值")),
    }
}
/// 其余键一律视为枚举型属性；闭集外的键直接报错，避免"写了但没人看"。
pub(super) fn attrs(
    entry: &Table,
    index: usize,
    provide: &mut Provide,
) -> Result<(), ProfileError> {
    for (key, value) in entry.entries() {
        if !parsed_key(key) {
            provide.attrs.push(one_attr(index, key, value)?);
        }
    }
    Ok(())
}
/// 已经有专属解析分支的键：再当属性处理就会重复计入（与诊断同一种 `a | b` 写法）。
pub(super) fn parsed_key(key: &str) -> bool {
    let parsed = "prim|impl|m|bytes|extent|n|k|dt|space|grade|witness";
    parsed.split('|').any(|name| name == key)
}
/// 一条属性：键在 [`AttrKey`] 闭集里，取值按标量或数组展开。
pub(super) fn one_attr(index: usize, key: &str, value: &Value) -> Result<Attr, ProfileError> {
    let attr_key = closed::<AttrKey>(key, &at(index, key))?;
    Ok(Attr::new(attr_key, attr_values(index, key, value)?))
}
/// 属性取值：标量或数组；表在这里没有意义。
pub(super) fn attr_values(
    index: usize,
    key: &str,
    value: &Value,
) -> Result<Vec<String>, ProfileError> {
    let path = at(index, key);
    match value {
        Value::Array(items) => items.iter().map(|item| one_string(path, item)).collect(),
        Value::Table(_) => Err(bad(path, "属性不能是表")),
        scalar => one_scalar(path, scalar),
    }
}
/// 单个标量属性值：字符串原样，整数/浮点/布尔按字面写。
pub(super) fn one_scalar(
    path: impl fmt::Display,
    value: &Value,
) -> Result<Vec<String>, ProfileError> {
    scalar_texts(value).ok_or_else(|| bad(path, "属性不能是表"))
}
pub(super) fn one_string(path: impl fmt::Display, item: &Value) -> Result<String, ProfileError> {
    match item {
        Value::Str(text) => Ok(text.clone()),
        _ => Err(bad(path, "属性数组里只能是字符串")),
    }
}
/// 区间文本：解析失败时把原话带上，字段路径由调用方给全。
pub(super) fn parse_span(field: impl fmt::Display, text: &str) -> Result<Span, ProfileError> {
    Span::parse(text).map_err(|error| bad(&field, error.message().to_string()))
}
