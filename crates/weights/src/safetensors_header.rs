//! safetensors 头部 JSON → [`Entry`] 构造：字段槽位收集、必填校验、形状与区间自洽。

use crate::bad;
use crate::json::{self, Value};
use crate::safetensors::{DTYPES, METADATA};
use crate::{Entry, Header};
use diagnostic::{Code, Error, Result};
use tensor::{DType, Shape};

/// 由头部 JSON 构造并校验 [`Header`]：[`crate::safetensors::Archive`] 与
/// [`crate::safetensors::File`] 共用这一份。
pub(crate) fn parse_header(text: &[u8], data_start: usize) -> Result<Header> {
    let root = root_value(text)?;
    let fields = root
        .object()
        .map_err(|error| error.with(format!("safetensors 头部（{} 字节）的顶层", text.len())))?;
    let entries = collect_entries(fields, data_start)?;
    Header::new(entries, data_start)
}

/// 按文件顺序：元数据只校验，其余解析成条目。
fn collect_entries(fields: &[(String, Value)], data_start: usize) -> Result<Vec<Entry>> {
    let mut entries = Vec::new();
    for (name, value) in fields {
        if name.as_str() == METADATA {
            check_metadata(value)?;
            continue;
        }
        entries.push(entry(name, value, data_start)?);
    }
    Ok(entries)
}

/// 头部 JSON 根值；不是 UTF-8 或文法坏都在这里报错。
fn root_value(text: &[u8]) -> Result<Value> {
    let text = std::str::from_utf8(text).map_err(|_| bad("头部不是 UTF-8"))?;
    json::parse(text).map_err(|error| error.with("weights::safetensors 头部"))
}

/// 元数据不参与布局，但它的值必须是对象。
fn check_metadata(value: &Value) -> Result<()> {
    value
        .object()
        .map_err(|error| error.with(format!("`{METADATA}` 必须是对象")))?;
    Ok(())
}

/// 把一条 JSON 条目解析成 [`Entry`]：字段闭集、形状与区间必须自洽。
fn entry(name: &str, value: &Value, data_start: usize) -> Result<Entry> {
    if name.is_empty() {
        return Err(bad("条目名不能为空").with("weights::safetensors"));
    }
    let ctx = format!("权重条目 `{name}`");
    let fields = value.object().map_err(|error| error.with(&ctx))?;
    let (dtype, shape, offsets) = collect_fields(fields, &ctx)?;
    let (dtype, shape, offsets) = required_fields(dtype, shape, offsets, &ctx)?;
    build(name, dtype, shape, offsets, data_start, &ctx)
}

// —— 头部字段机制（原 `header` 平铺进 `safetensors`）：JSON 字段槽位收集、必填校验与 `Entry` 组装 ——

/// 条目三个字段的可选槽位。
type Slots = (Option<DType>, Option<Shape>, Option<(usize, usize)>);

/// 三个字段；重复与未知字段按文件顺序拒绝。
pub(crate) fn collect_fields(fields: &[(String, Value)], ctx: &str) -> Result<Slots> {
    let mut dtype = None;
    let mut shape = None;
    let mut offsets = None;
    for (key, value) in fields {
        field(key, value, ctx, &mut dtype, &mut shape, &mut offsets)?;
    }
    Ok((dtype, shape, offsets))
}

/// 一条字段写进对应槽位；未知字段直接报错。
fn field(
    key: &str,
    value: &Value,
    ctx: &str,
    dtype: &mut Option<DType>,
    shape: &mut Option<Shape>,
    offsets: &mut Option<(usize, usize)>,
) -> Result<()> {
    match key {
        "dtype" => dtype_of(value).and_then(|value| set_once(dtype, value, key, ctx)),
        "shape" => shape_of(value).and_then(|value| set_once(shape, value, key, ctx)),
        "data_offsets" => offsets_of(value).and_then(|value| set_once(offsets, value, key, ctx)),
        other => Err(bad(format!(
            "未知字段 `{other}`（条目只允许 dtype / shape / data_offsets）"
        ))
        .with(ctx)),
    }
}

/// 字段只允许出现一次。
fn set_once<T>(slot: &mut Option<T>, value: T, key: &str, ctx: &str) -> Result<()> {
    if slot.is_some() {
        return Err(bad(format!("字段 `{key}` 重复")).with(ctx));
    }
    *slot = Some(value);
    Ok(())
}

/// 三个必填字段，缺一即报错并指出名字。
pub(crate) fn required_fields(
    dtype: Option<DType>,
    shape: Option<Shape>,
    offsets: Option<(usize, usize)>,
    ctx: &str,
) -> Result<(DType, Shape, (usize, usize))> {
    let dtype = required(dtype, "dtype", ctx)?;
    let shape = required(shape, "shape", ctx)?;
    let offsets = required(offsets, "data_offsets", ctx)?;
    Ok((dtype, shape, offsets))
}

fn required<T>(slot: Option<T>, key: &str, ctx: &str) -> Result<T> {
    slot.ok_or_else(|| bad(format!("缺字段 `{key}`")).with(ctx))
}

/// 区间长度与形状自洽，并把相对起点转成绝对偏移。
pub(crate) fn build(
    name: &str,
    dtype: DType,
    shape: Shape,
    (begin, end): (usize, usize),
    data_start: usize,
    ctx: &str,
) -> Result<Entry> {
    let expected = shape.element_bytes(dtype);
    if end < begin {
        return Err(bad(format!("区间 {begin}..{end} 起点晚于终点")).with(ctx));
    }
    if end - begin != expected {
        return Err(bad(format!(
            "区间 {begin}..{end} 是 {} 字节，但 {:?} {:?} 推出 {expected} 字节",
            end - begin,
            dtype,
            shape.dims()
        ))
        .with(ctx));
    }
    let offset = data_start
        .checked_add(begin)
        .ok_or_else(|| bad(format!("偏移 {begin} 与数据段起点相加溢出")).with(ctx))?;
    Ok(Entry::new(name, dtype, shape).at(offset))
}

/// `dtype` → [`DType`]；不认识的名字直接报错并列出闭集。
fn dtype_of(value: &Value) -> Result<DType> {
    let name = value.string().map_err(|error| error.with("字段 `dtype`"))?;
    DTYPES
        .iter()
        .find(|(known, _)| *known == name)
        .map(|(_, dtype)| *dtype)
        .ok_or_else(|| {
            let known: Vec<&str> = DTYPES.iter().map(|(name, _)| *name).collect();
            Error::new(
                Code::Unsupported,
                format!("`dtype` 是 `{name}`，闭集里只有 {}", known.join(" / ")),
            )
            .with("字段 `dtype`")
        })
}

fn shape_of(value: &Value) -> Result<Shape> {
    let items = value.array().map_err(|error| error.with("字段 `shape`"))?;
    let mut dims = Vec::with_capacity(items.len());
    for item in items {
        let number = item
            .integer()
            .map_err(|error| error.with("字段 `shape` 的轴长度"))?;
        let axis = nonnegative(number, "轴长度", "字段 `shape`")?;
        dims.push(axis);
    }
    Ok(Shape::new(dims))
}

/// 一个非负整数换算成 `usize`；负数（或超出可寻址范围）按 `Invalid` 拒绝，给出字段站点。
fn nonnegative(number: i64, what: &str, ctx: &str) -> Result<usize> {
    usize::try_from(number).map_err(|_| bad(format!("{what} {number} 不能为负")).with(ctx))
}

fn offsets_of(value: &Value) -> Result<(usize, usize)> {
    let (begin, end) = pair_of(value)?;
    let begin = offset(begin)?;
    let end = offset(end)?;
    if begin > end {
        return Err(bad(format!("`data_offsets` 的起点 {begin} 晚于终点 {end}"))
            .with("字段 `data_offsets`"));
    }
    Ok((begin, end))
}

/// `data_offsets` 必须是恰好两个元素的数组。
fn pair_of(value: &Value) -> Result<(&Value, &Value)> {
    let items = value
        .array()
        .map_err(|error| error.with("字段 `data_offsets`"))?;
    match items {
        [begin, end] => Ok((begin, end)),
        _ => Err(bad(format!(
            "`data_offsets` 必须是两个整数，实际 {} 个",
            items.len()
        ))
        .with("字段 `data_offsets`")),
    }
}

/// 一个非负偏移值；负数越界即报错。
fn offset(value: &Value) -> Result<usize> {
    let number = value
        .integer()
        .map_err(|error| error.with("字段 `data_offsets`"))?;
    nonnegative(number, "偏移", "字段 `data_offsets`")
}
