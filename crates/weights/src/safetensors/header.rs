//! `safetensors` 的头部字段机制：JSON 字段槽位收集、必填校验与 `Entry` 组装。
//!
//! 从 [`crate::safetensors`] 拆出：公开面只管内存/文件两种读取路径，这里只管「一条 JSON
//! 张量记录怎么落成 [`Entry`]」的纯解析，两边通过少量 `pub(crate)` 入口相连。

use crate::bad;
use diagnostic::{Code, Error, Result};
use tensor::{DType, Shape};

use crate::json::Value;
use crate::safetensors::{DTYPES, split_prefix};
use crate::{Entry, Header};

/// 条目三个字段的可选槽位。
type Slots = (Option<DType>, Option<Shape>, Option<(usize, usize)>);

/// 三个字段；重复与未知字段按文件顺序拒绝。
pub(crate) fn collect_fields(fields: &[(String, Value)], context: &str) -> Result<Slots> {
    let mut dtype = None;
    let mut shape = None;
    let mut offsets = None;
    for (key, value) in fields {
        field(key, value, context, &mut dtype, &mut shape, &mut offsets)?;
    }
    Ok((dtype, shape, offsets))
}

/// 一条字段写进对应槽位；未知字段直接报错。
fn field(
    key: &str,
    value: &Value,
    context: &str,
    dtype: &mut Option<DType>,
    shape: &mut Option<Shape>,
    offsets: &mut Option<(usize, usize)>,
) -> Result<()> {
    match key {
        "dtype" => dtype_slot(dtype, value, key, context),
        "shape" => shape_slot(shape, value, key, context),
        "data_offsets" => offsets_slot(offsets, value, key, context),
        other => Err(bad(format!(
            "未知字段 `{other}`（条目只允许 dtype / shape / data_offsets）"
        ))
        .with(context)),
    }
}

fn dtype_slot(slot: &mut Option<DType>, value: &Value, key: &str, context: &str) -> Result<()> {
    set_once(slot, dtype_of(value)?, key, context)
}

fn shape_slot(slot: &mut Option<Shape>, value: &Value, key: &str, context: &str) -> Result<()> {
    set_once(slot, shape_of(value)?, key, context)
}

fn offsets_slot(
    slot: &mut Option<(usize, usize)>,
    value: &Value,
    key: &str,
    context: &str,
) -> Result<()> {
    set_once(slot, offsets_of(value)?, key, context)
}

/// 字段只允许出现一次。
fn set_once<T>(slot: &mut Option<T>, value: T, key: &str, context: &str) -> Result<()> {
    if slot.is_some() {
        return Err(bad(format!("字段 `{key}` 重复")).with(context));
    }
    *slot = Some(value);
    Ok(())
}

/// 三个必填字段，缺一即报错并指出名字。
pub(crate) fn required_fields(
    dtype: Option<DType>,
    shape: Option<Shape>,
    offsets: Option<(usize, usize)>,
    context: &str,
) -> Result<(DType, Shape, (usize, usize))> {
    let dtype = required(dtype, "dtype", context)?;
    let shape = required(shape, "shape", context)?;
    let offsets = required(offsets, "data_offsets", context)?;
    Ok((dtype, shape, offsets))
}

fn required<T>(slot: Option<T>, key: &str, context: &str) -> Result<T> {
    slot.ok_or_else(|| bad(format!("缺字段 `{key}`")).with(context))
}

/// 区间长度与形状自洽，并把相对起点转成绝对偏移。
pub(crate) fn build(
    name: &str,
    dtype: DType,
    shape: Shape,
    (begin, end): (usize, usize),
    data_start: usize,
    context: &str,
) -> Result<Entry> {
    let expected = shape.element_bytes(dtype);
    if end - begin != expected {
        return Err(bad(format!(
            "区间 {begin}..{end} 是 {} 字节，但 {:?} {:?} 推出 {expected} 字节",
            end - begin,
            dtype,
            shape.dims()
        ))
        .with(context));
    }
    let offset = data_start
        .checked_add(begin)
        .ok_or_else(|| bad(format!("偏移 {begin} 与数据段起点相加溢出")).with(context))?;
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
fn nonnegative(number: i64, what: &str, context: &str) -> Result<usize> {
    usize::try_from(number).map_err(|_| bad(format!("{what} {number} 不能为负")).with(context))
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

/// 8 字节长度前缀与头部 JSON 切片；两者都必须落在文件内。
pub(crate) fn header_slice(bytes: &[u8]) -> Result<(usize, &[u8])> {
    let Some(prefix) = bytes.get(..8) else {
        return Err(bad(format!(
            "文件只有 {} 字节，读不出 8 字节头部长度",
            bytes.len()
        )));
    };
    let mut word = [0u8; 8];
    word.copy_from_slice(prefix);
    let (declared, header_end) = split_prefix(word)?;
    let Some(text) = bytes.get(8..header_end) else {
        return Err(bad(format!(
            "头部声明 {declared} 字节，但 {header_end} 已超出文件长度 {}",
            bytes.len()
        )));
    };
    Ok((header_end, text))
}

/// 数据段：必须正好被条目覆盖，不许多余字节。
pub(crate) fn data_after<'b>(
    bytes: &'b [u8],
    header: &Header,
    header_end: usize,
) -> Result<&'b [u8]> {
    let data = &bytes[header_end..];
    let covered = header.data_bytes();
    if covered == data.len() {
        return Ok(data);
    }
    Err(bad(format!(
        "数据段应为 {covered} 字节（由条目的最大结束偏移推出），实际 {} 字节",
        data.len()
    ))
    .with("weights::safetensors::Archive"))
}

/// 读端内部错误：偏移越界只可能是头部与数据段漂移。
pub(crate) fn internal(message: String) -> Error {
    Error::new(Code::Internal, message).with("weights::safetensors::Archive::read")
}
