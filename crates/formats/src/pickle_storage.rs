//! 职责：pickle 的对象解码——把栈上的 [`Value`] 读成张量的五要素（storage、dtype、形状、字节区间）。

use diagnostic::Result;
use dtype::DType;

use crate::bin::Tensor;
use crate::pickle::Value;
use wire::{bad, or_bad};

/// 元组视图：`REDUCE` 的实参与 persistent id 都必须是元组。
pub(crate) fn as_tuple(value: Value, message: &str) -> Result<Vec<Value>> {
    match value {
        Value::Tuple(items) => Ok(items),
        _ => Err(bad(message)),
    }
}

pub(crate) fn into_str(value: &Value) -> Result<String> {
    match value {
        Value::Str(text) => Ok(text.clone()),
        _ => Err(bad("期望字符串")),
    }
}

fn item<'v>(items: &'v [Value], index: usize, missing: &str) -> Result<&'v Value> {
    or_bad(items.get(index), missing)
}

fn tuple_at<'v>(items: &'v [Value], index: usize, what: &str) -> Result<&'v [Value]> {
    match item(items, index, &format!("rebuild 缺 {what}"))? {
        Value::Tuple(fields) => Ok(fields),
        _ => Err(bad(format!("rebuild 缺 {what}"))),
    }
}

/// persistent id：`("storage", <类名>, <key>)` → 存库名与元素类型。
pub(crate) fn storage_handle(items: &[Value]) -> Result<(String, DType)> {
    let tag = string_at(items, 0, "pid 空")?;
    if tag != "storage" {
        return Err(bad(format!("不支持的 persistent id `{tag}`")));
    }
    let dtype = storage_dtype(items)?;
    let key = string_at(items, 2, "pid 缺 key")?;
    Ok((key, dtype))
}

/// 元组第 `index` 项的字符串视图。
fn string_at(items: &[Value], index: usize, missing: &str) -> Result<String> {
    into_str(item(items, index, missing)?)
}

/// storage 类名 → 元素类型。
fn storage_dtype(items: &[Value]) -> Result<DType> {
    let class = match item(items, 1, "pid 缺 storage 类")? {
        Value::Global { name, .. } => name.as_str(),
        _ => return Err(bad("storage 类不是 GLOBAL")),
    };
    STORAGE_DTYPES
        .iter()
        .find_map(|&(name, dtype)| (name == class).then_some(dtype))
        .ok_or_else(|| bad(format!("不支持的 storage 类型 `{class}`")))
}

/// 形状维度：非负整数，负数与错型分别是溢出与类型错误。
fn dim(value: &Value) -> Result<u64> {
    match value {
        Value::Int(wide) => u64::try_from(*wide).map_err(|_| bad("形状维度溢出")),
        _ => Err(bad("形状维度不是整数")),
    }
}

/// stride 与维度是同一概念（非负整数），只多一步收窄到 `usize`。
fn step(value: &Value) -> Result<usize> {
    let wide = dim(value).map_err(|_| bad("stride 不是非负整数"))?;
    usize::try_from(wide).map_err(|_| bad("stride 溢出"))
}

/// row-major 连续判据：尺寸为 1 的维不参与（torch 允许其 stride 任意）；返回元素总数。
/// 非连续张量无法用单个连续字节区间表示，显式失败而不是按连续布局读出错误字节。
fn contiguous(shape: &[u64], steps: &[usize]) -> Result<usize> {
    if shape.len() != steps.len() {
        return Err(bad("stride 与 shape 维数不符"));
    }
    let mut count = 1usize;
    for (&size, &step) in shape.iter().zip(steps).rev() {
        count = advance_dim(count, size, step)?;
    }
    Ok(count)
}

/// 一维的连续判据：宽度为 1 的维不参与，其余维 stride 必须等于当前累计长度。
fn advance_dim(count: usize, size: u64, step: usize) -> Result<usize> {
    let width = usize::try_from(size).map_err(|_| bad("形状维度溢出"))?;
    if width > 1 && step != count {
        return Err(bad(format!(
            "非连续张量：stride {step}，连续布局应为 {count}"
        )));
    }
    or_bad(count.checked_mul(width), "形状乘积溢出")
}

/// `_rebuild_tensor_v2` 的实参 → 张量：storage、offset、shape、stride 齐备且布局连续。
pub(crate) fn tensor_of(items: &[Value]) -> Result<Tensor> {
    let (key, dtype, offset, shape, steps) = tensor_parts(items)?;
    let count = contiguous(&shape, &steps)?;
    let (offset, len) = byte_range(dtype, offset, count)?;
    Ok(Tensor {
        name: String::new(),
        dtype,
        shape,
        storage_key: key,
        offset,
        len,
    })
}

/// 张量的五个要素：storage 键、元素类型、offset、形状、stride。
fn tensor_parts(items: &[Value]) -> Result<(String, DType, usize, Vec<u64>, Vec<usize>)> {
    let (key, dtype) = storage_at(items)?;
    let offset = offset_at(items)?;
    let shape = dims(items, 2, "shape")?;
    let steps = dims_as_steps(items)?;
    Ok((key, dtype, offset, shape, steps))
}

/// 首项必须是 storage。
fn storage_at(items: &[Value]) -> Result<(String, DType)> {
    let Some(Value::Storage { key, dtype }) = items.first() else {
        return Err(bad("rebuild 缺 storage"));
    };
    Ok((key.clone(), *dtype))
}

/// 次项必须是 offset（非负整数收窄到 `usize`）。
fn offset_at(items: &[Value]) -> Result<usize> {
    let Some(Value::Int(wide)) = items.get(1) else {
        return Err(bad("rebuild 缺 offset"));
    };
    usize::try_from(*wide).map_err(|_| bad("offset 溢出"))
}

/// 第 `index` 项元组 → 维度向量。
fn dims(items: &[Value], index: usize, what: &str) -> Result<Vec<u64>> {
    tuple_at(items, index, what)?
        .iter()
        .map(dim)
        .collect::<Result<Vec<u64>>>()
}

/// 第 3 项元组 → stride 向量。
fn dims_as_steps(items: &[Value]) -> Result<Vec<usize>> {
    tuple_at(items, 3, "stride")?
        .iter()
        .map(step)
        .collect::<Result<Vec<usize>>>()
}

/// 元素字节区间：offset 与元素总数各乘宽度，溢出显式失败。
fn byte_range(dtype: DType, offset: usize, count: usize) -> Result<(usize, usize)> {
    let width = dtype.bytes_for(1) as usize;
    let offset = or_bad(offset.checked_mul(width), "offset 溢出")?;
    let len = or_bad(count.checked_mul(width), "长度溢出")?;
    Ok((offset, len))
}

/// torch 的 storage 类名 → 元素类型（Torch 与 `dtype` 的命名对应表）。
pub(crate) const STORAGE_DTYPES: [(&str, DType); 9] = [
    ("FloatStorage", DType::F32),
    ("HalfStorage", DType::F16),
    ("BFloat16Storage", DType::BF16),
    ("DoubleStorage", DType::F64),
    ("LongStorage", DType::I64),
    ("IntStorage", DType::I32),
    ("ShortStorage", DType::I16),
    ("CharStorage", DType::I8),
    ("ByteStorage", DType::U8),
];
