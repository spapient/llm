//! 职责：pickle 对象图里的 torch 张量——persistent id 与 `_rebuild_tensor_v2` 的实参 → 张量。
//!
//! 两个入口各管一件事：[`storage_handle`] 解 `BINPERSID` 的三元组，[`tensor_of`] 解重建实参的
//! 五要素（storage、offset、shape、stride、元素总数）。布局判据只有一条：[`contiguous`]，
//! 非连续张量无法用单个连续字节区间表示，显式失败好过按连续布局读出错误字节。

use diagnostic::Result;
use dtype::DType;

use crate::pickle::Value;
use crate::torch::Tensor;
use wire::{bad, or_bad};

/// 元组视图：`REDUCE` 的实参与 persistent id 都必须是元组。
pub(crate) fn as_tuple(value: Value, message: &str) -> Result<Vec<Value>> {
    match value {
        Value::Tuple(items) => Ok(items),
        _ => Err(bad(message)),
    }
}

/// 字符串视图。
pub(crate) fn into_str(value: &Value) -> Result<String> {
    match value {
        Value::Str(text) => Ok(text.clone()),
        _ => Err(bad("期望字符串")),
    }
}

/// persistent id：`("storage", <类名>, <key>)` → 存库名与元素类型。
pub(crate) fn storage_handle(items: &[Value]) -> Result<(String, DType)> {
    let Some(Value::Str(tag)) = items.first() else {
        return Err(bad("pid 空"));
    };
    if tag != "storage" {
        return Err(bad(format!("不支持的 persistent id `{tag}`")));
    }
    let Some(Value::Global { name: class, .. }) = items.get(1) else {
        return Err(bad("pid 缺 storage 类"));
    };
    let dtype = STORAGE_DTYPES
        .iter()
        .find_map(|&(name, dtype)| (name == class).then_some(dtype))
        .ok_or_else(|| bad(format!("不支持的 storage 类型 `{class}`")))?;
    Ok((into_str(field(items, 2, "key")?)?, dtype))
}

/// `_rebuild_tensor_v2` 的实参 → 张量：五要素齐备且布局连续。
pub(crate) fn tensor_of(items: &[Value]) -> Result<Tensor> {
    let Some(Value::Storage { key, dtype }) = items.first() else {
        return Err(bad("rebuild 缺 storage"));
    };
    let Some(Value::Int(wide)) = items.get(1) else {
        return Err(bad("rebuild 缺 offset"));
    };
    let width = dtype.bytes_for(1) as usize;
    let shape = tuple_at(items, 2, "shape")?
        .iter()
        .map(dim)
        .collect::<Result<Vec<u64>>>()?;
    let steps = tuple_at(items, 3, "stride")?
        .iter()
        .map(step)
        .collect::<Result<Vec<usize>>>()?;
    let count = contiguous(&shape, &steps)?;
    let offset = or_bad(
        usize::try_from(*wide)
            .ok()
            .and_then(|n| n.checked_mul(width)),
        "offset 溢出",
    )?;
    Ok(Tensor {
        name: String::new(),
        dtype: *dtype,
        shape,
        storage_key: key.clone(),
        offset,
        len: or_bad(count.checked_mul(width), "长度溢出")?,
    })
}

/// 第 `index` 项。
fn field<'v>(items: &'v [Value], index: usize, missing: &str) -> Result<&'v Value> {
    items
        .get(index)
        .ok_or_else(|| bad(format!("rebuild 缺 {missing}")))
}

/// 第 `index` 项元组视图。
fn tuple_at<'v>(items: &'v [Value], index: usize, what: &str) -> Result<&'v [Value]> {
    match field(items, index, what)? {
        Value::Tuple(fields) => Ok(fields),
        _ => Err(bad(format!("rebuild 缺 {what}"))),
    }
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
fn contiguous(shape: &[u64], steps: &[usize]) -> Result<usize> {
    if shape.len() != steps.len() {
        return Err(bad("stride 与 shape 维数不符"));
    }
    let mut count = 1usize;
    for (&size, &step) in shape.iter().zip(steps).rev() {
        count = advance(count, size, step)?;
    }
    Ok(count)
}

/// 单一维的连续校验与累乘：尺寸为 1 跳过 stride 校验，乘积溢出显式失败。
fn advance(count: usize, size: u64, step: usize) -> Result<usize> {
    let width = usize::try_from(size).map_err(|_| bad("形状维度溢出"))?;
    if width > 1 && step != count {
        return Err(bad(format!(
            "非连续张量：stride {step}，连续布局应为 {count}"
        )));
    }
    or_bad(count.checked_mul(width), "形状乘积溢出")
}

/// torch 的 storage 类名 → 元素类型（Torch 与 `dtype` 的命名对应表）。
const STORAGE_DTYPES: [(&str, DType); 9] = [
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
