//! pickle 的对象解码助手：把栈上的 [`crate::pickle::Value`] 读成张量的四要素。

use diagnostic::Result;
use dtype::DType;

use crate::bin::Tensor;
use crate::pickle::Value;
use wire::{bad, or_bad};

pub(crate) fn as_tuple(value: Value, message: &str) -> Result<Vec<Value>> {
    match value {
        Value::Tuple(items) => Ok(items),
        _ => Err(bad(message)),
    }
}

pub(crate) fn at<'v>(items: &'v [Value], index: usize, message: &str) -> Result<&'v Value> {
    or_bad(items.get(index), message)
}

pub(crate) fn storage_handle(items: &[Value]) -> Result<(String, DType)> {
    let tag = into_str(at(items, 0, "pid 空")?)?;
    if tag != "storage" {
        return Err(bad(format!("不支持的 persistent id `{tag}`")));
    }
    let class = match at(items, 1, "pid 缺 storage 类")? {
        Value::Global { name, .. } => name.as_str(),
        _ => return Err(bad("storage 类不是 GLOBAL")),
    };
    let dtype = storage_dtype(class)?;
    let key = into_str(at(items, 2, "pid 缺 key")?)?;
    Ok((key, dtype))
}

pub(crate) fn storage_of(items: &[Value]) -> Result<(String, DType)> {
    match items.first() {
        Some(Value::Storage { key, dtype }) => Ok((key.clone(), *dtype)),
        _ => Err(bad("rebuild 缺 storage")),
    }
}

pub(crate) fn offset_of(items: &[Value]) -> Result<usize> {
    if let Some(Value::Int(n)) = items.get(1) {
        return usize::try_from(*n).map_err(|_| bad("offset 溢出"));
    }
    Err(bad("rebuild 缺 offset"))
}

pub(crate) fn shape_of(items: &[Value]) -> Result<(Vec<u64>, usize)> {
    let dims = match at(items, 2, "rebuild 缺 shape")? {
        Value::Tuple(dims) => dims,
        _ => return Err(bad("rebuild 缺 shape")),
    };
    let shape = dims.iter().map(dim).collect::<Result<Vec<u64>>>()?;
    let count = shape.iter().try_fold(1usize, |acc, &d| {
        usize::try_from(d).ok().and_then(|n| acc.checked_mul(n))
    });
    Ok((shape, or_bad(count, "形状乘积溢出")?))
}

pub(crate) fn dim(value: &Value) -> Result<u64> {
    match value {
        Value::Int(n) => u64::try_from(*n).map_err(|_| bad("形状维度溢出")),
        _ => Err(bad("形状维度不是整数")),
    }
}

/// stride 与维度是同一概念（非负整数），只多一步收窄到 `usize`。
pub(crate) fn stride(value: &Value) -> Result<usize> {
    let step = dim(value).map_err(|_| bad("stride 不是非负整数"))?;
    usize::try_from(step).map_err(|_| bad("stride 溢出"))
}

pub(crate) fn stride_of(items: &[Value]) -> Result<Vec<usize>> {
    let steps = match at(items, 3, "rebuild 缺 stride")? {
        Value::Tuple(steps) => steps,
        _ => return Err(bad("rebuild 缺 stride")),
    };
    steps.iter().map(stride).collect()
}

/// 连续（row-major）判据：尺寸为 1 的维不参与（torch 允许其 stride 任意）。
/// 非连续张量无法用单个连续字节区间表示，显式失败而不是按连续布局读出错误字节。
pub(crate) fn check_contiguous(shape: &[u64], steps: &[usize]) -> Result<()> {
    if shape.len() != steps.len() {
        return Err(bad("stride 与 shape 维数不符"));
    }
    let mut expected = 1usize;
    for (&size, &step) in shape.iter().zip(steps).rev() {
        expected = contiguous_dim(size, step, expected)?;
    }
    Ok(())
}

/// 单维连续判据：尺寸为 1 的维不参与；返回累计的期望 stride。
fn contiguous_dim(size: u64, step: usize, expected: usize) -> Result<usize> {
    let width = usize::try_from(size).map_err(|_| bad("形状维度溢出"))?;
    if width > 1 && step != expected {
        return Err(bad(format!(
            "非连续张量：stride {step}，连续布局应为 {expected}"
        )));
    }
    or_bad(expected.checked_mul(width), "形状乘积溢出")
}

pub(crate) fn tensor_of(items: &[Value]) -> Result<Tensor> {
    let (storage_key, dtype) = storage_of(items)?;
    let offset = offset_of(items)?;
    let (shape, count) = shape_of(items)?;
    check_contiguous(&shape, &stride_of(items)?)?;
    let (offset, len) = span(offset, count, dtype.bytes_for(1) as usize)?;
    Ok(Tensor {
        name: String::new(),
        dtype,
        shape,
        storage_key,
        offset,
        len,
    })
}

pub(crate) fn into_str(value: &Value) -> Result<String> {
    match value {
        Value::Str(s) => Ok(s.clone()),
        _ => Err(bad("期望字符串")),
    }
}

pub(crate) fn span(offset: usize, count: usize, width: usize) -> Result<(usize, usize)> {
    let offset = or_bad(offset.checked_mul(width), "offset 溢出")?;
    let len = or_bad(count.checked_mul(width), "长度溢出")?;
    Ok((offset, len))
}

pub(crate) fn storage_dtype(name: &str) -> Result<DType> {
    STORAGE_DTYPES
        .iter()
        .find_map(|&(class, dtype)| (class == name).then_some(dtype))
        .ok_or_else(|| bad(format!("不支持的 storage 类型 `{name}`")))
}

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
