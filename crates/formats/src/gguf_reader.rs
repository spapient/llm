//! 职责：GGUF 的字节读取与表解析——值/张量记录的消费、对齐与数据区定位。
//!
//! 字节层（越界检查、小端拼装）统一走 [`Cursor`]；本模块只保留 GGUF 自己的
//! 布局判据：数组嵌套深度、张量维数上限、值类型码表与 `general.alignment`。

use diagnostic::Result;

use wire::bad;

use crate::bytes::Cursor;
use crate::gguf::{DEFAULT_ALIGN, Kind, MAGIC, Meta, Tensor, VERSION, Value};

/// 数组最大嵌套深度（防未信任输入递归耗尽栈）。
const DEPTH: usize = 8;
/// 张量最大维数。
const MAX_RANK: u32 = 8;
/// 表与数组的预分配上限：谎报的巨大计数不会换来一次巨额分配。
const MAX_PREALLOC: usize = 4096;

/// 值类型码 → `class << 4 | 字节宽度`；`0` 表示这不是标量码（字符串/数组/表外码）。
const NUMBERS: [u8; 13] = [
    0x01, 0x11, 0x02, 0x12, 0x04, 0x14, 0x24, 0, 0, 0, 0x08, 0x18, 0x28,
];

/// GGUF 字符串：8 字节长度前缀 + UTF-8 正文。
fn text(cursor: &mut Cursor<'_>) -> Result<String> {
    let len = usize::try_from(cursor.wide()?).map_err(|_| bad("字符串长度溢出"))?;
    cursor.text(len)
}

/// 一个值：布尔、字符串、数组各有自己的布局，其余按 [`NUMBERS`] 当标量读。
fn value(cursor: &mut Cursor<'_>, kind: u32, depth: usize) -> Result<Value> {
    match kind {
        7 => Ok(Value::Bool(cursor.uint(1)? != 0)),
        8 => text(cursor).map(Value::Str),
        9 => list(cursor, depth),
        _ => scalar(cursor, kind),
    }
}

fn list(cursor: &mut Cursor<'_>, depth: usize) -> Result<Value> {
    if depth >= DEPTH {
        return Err(bad("数组嵌套过深"));
    }
    let kind = cursor.word()?;
    let count = usize::try_from(cursor.wide()?).map_err(|_| bad("数组长度溢出"))?;
    if count > cursor.rest() {
        return Err(bad("数组元素数超过剩余字节"));
    }
    let mut out = Vec::with_capacity(count.min(MAX_PREALLOC));
    for _ in 0..count {
        out.push(value(cursor, kind, depth + 1)?);
    }
    Ok(Value::Arr(out))
}

/// 标量：`class << 4 | 宽度`，整数按有无符号扩展，浮点 f32 升 f64。
fn scalar(cursor: &mut Cursor<'_>, kind: u32) -> Result<Value> {
    let packed = NUMBERS.get(kind as usize).copied().unwrap_or(0);
    if packed == 0 {
        return Err(bad("未知值类型码"));
    }
    let width = usize::from(packed & 0x0F);
    let raw = cursor.take(width)?;
    let mut buf = [0u8; 8];
    buf[..width].copy_from_slice(raw);
    Ok(number(packed >> 4, u64::from_le_bytes(buf), width))
}

fn number(class: u8, bits: u64, width: usize) -> Value {
    let shift = 64 - width * 8;
    match class {
        0 => Value::Uint(bits),
        1 => Value::Int(((bits << shift) as i64) >> shift),
        _ => Value::Float(match width {
            4 => f64::from(f32::from_bits(bits as u32)),
            _ => f64::from_bits(bits),
        }),
    }
}

/// 读 `count` 条记录；容量按 [`MAX_PREALLOC`] 封顶——谎报的巨大计数不会换来一次巨额分配。
fn records<Item>(
    cursor: &mut Cursor<'_>,
    count: u64,
    overflow: &str,
    item: impl Fn(&mut Cursor<'_>) -> Result<Item>,
) -> Result<Vec<Item>> {
    let count = usize::try_from(count).map_err(|_| bad(overflow))?;
    let mut out = Vec::with_capacity(count.min(MAX_PREALLOC));
    for _ in 0..count {
        out.push(item(cursor)?);
    }
    Ok(out)
}

/// 一对 KV：键、类型码与值。
fn pair(cursor: &mut Cursor<'_>) -> Result<(String, Value)> {
    let key = text(cursor)?;
    let kind = cursor.word()?;
    Ok((key, value(cursor, kind, 0)?))
}

/// 一条张量信息：名字、形状、元素类型与数据区偏移（偏移相对数据区起点）。
fn tensor(cursor: &mut Cursor<'_>) -> Result<Tensor> {
    let name = text(cursor)?;
    let shape = shape(cursor)?;
    let kind = Kind::from_code(cursor.word()?)?;
    Ok(Tensor {
        name,
        kind,
        shape,
        offset: cursor.wide()?,
    })
}

/// 形状：维数上限 `MAX_RANK`，每维一个 `u64`。
fn shape(cursor: &mut Cursor<'_>) -> Result<Vec<u64>> {
    let rank = cursor.word()?;
    if rank > MAX_RANK {
        return Err(bad("张量维数超限"));
    }
    let mut shape = Vec::with_capacity(rank as usize);
    for _ in 0..rank {
        shape.push(cursor.wide()?);
    }
    Ok(shape)
}

/// 头部 magic 与版本：两个都对得上才继续，不做向后兼容猜测。
fn stamp(cursor: &mut Cursor<'_>) -> Result<()> {
    if cursor.take(4)? != MAGIC {
        return Err(bad("magic 不匹配"));
    }
    let version = cursor.word()?;
    if version != VERSION {
        return Err(bad(format!("版本 {version}，期望 {VERSION}")));
    }
    Ok(())
}

/// 头部与两张表：magic/版本、两个计数（张量条目数、KV 对数），接着 KV 表与张量信息表。
pub(crate) fn tables(cursor: &mut Cursor<'_>) -> Result<(Vec<Tensor>, Meta)> {
    stamp(cursor)?;
    let tensor_count = cursor.wide()?;
    let kv_count = cursor.wide()?;
    let meta = records(cursor, kv_count, "KV 计数溢出", pair)?;
    Ok((records(cursor, tensor_count, "张量计数溢出", tensor)?, meta))
}

/// 数据区起点：按 `general.alignment` 对齐（未声明或非法时退回 [`DEFAULT_ALIGN`]），且不得越过末尾。
pub(crate) fn data_start(meta: &[(String, Value)], pos: usize, len: usize) -> Result<usize> {
    let align = meta
        .iter()
        .find(|(key, _)| key == "general.alignment")
        .and_then(|(_, value)| value.uint())
        .and_then(|n| usize::try_from(n).ok())
        .filter(|&n| n > 0)
        .unwrap_or(DEFAULT_ALIGN);
    let Some(data) = wire::align_up(pos, align) else {
        return Err(bad("对齐必须是 2 的幂且对齐后不溢出"));
    };
    if data <= len {
        return Ok(data);
    }
    Err(bad("数据区偏移超出文件"))
}
