//! segment 描述符的解码路径：帧头、形状、量化参数与三条长度判据。
//!
//! 从 [`crate::seg`] 拆出：那边只管「一个 `Segment` 怎么被写出」，这里只管「字节怎么被读回」，
//! 编码与解码各有唯一一处，交叉核对时不必在同一份文件里来回跳。

use diagnostic::Result;

use crate::Wire;
use crate::bad;
use crate::leb::Cursor;
use crate::quant::Quant;
use crate::seg::{Codec, Descriptor};

pub(crate) fn decode_meta(cur: &mut Cursor<'_>) -> Result<(Descriptor, u32)> {
    let (name, dtype) = decode_head(cur)?;
    let (shape, rank) = decode_shape_head(cur)?;
    Ok((
        Descriptor {
            name,
            dtype,
            shape,
            quant: None,
        },
        rank,
    ))
}

/// 段名：LEB 长度 + UTF-8 + 非空。
pub(crate) fn decode_name(cur: &mut Cursor<'_>) -> Result<String> {
    let len = cur.read::<u64>()?;
    let name = name_bytes(cur, len)?;
    if name.is_empty() {
        return Err(bad("段名不能为空"));
    }
    Ok(name.to_owned())
}

/// 名字字节：长度可表示且是 UTF-8。
pub(crate) fn name_bytes<'c>(cur: &mut Cursor<'c>, len: u64) -> Result<&'c str> {
    let len = usize::try_from(len).map_err(|_| bad("名字长度超 usize"))?;
    std::str::from_utf8(cur.take(len)?).map_err(|_| bad("段名不是 UTF-8"))
}

/// 段名与元素类型；两者都先于形状出现。
pub(crate) fn decode_head(cur: &mut Cursor<'_>) -> Result<(String, Wire)> {
    let name = decode_name(cur)?;
    Ok((name, crate::read_wire(cur)?))
}

/// 张量轴数上限：真实张量的维数远小于此；谎报的 rank 不换来巨额预分配。
pub(crate) const MAX_RANK: u32 = 8;

/// 形状：rank 轴长度；codec 形态由调用方按描述符形态分派读取。
pub(crate) fn decode_shape_head(cur: &mut Cursor<'_>) -> Result<(Vec<u64>, u32)> {
    let rank = cur.read::<u32>()?;
    let shape = decode_shape(cur, rank)?;
    Ok((shape, rank))
}

/// `rank` 个轴长度；维数先过 [`MAX_RANK`] 守卫。
pub(crate) fn decode_shape(cur: &mut Cursor<'_>, rank: u32) -> Result<Vec<u64>> {
    if rank > MAX_RANK {
        return Err(bad(format!("张量维数 {rank} 超过 {MAX_RANK}")));
    }
    let mut shape = Vec::with_capacity(rank as usize);
    for _ in 0..rank {
        shape.push(cur.read::<u64>()?);
    }
    Ok(shape)
}

/// 量化方案与长度字段；`rank` 用于校验量化轴。
pub(crate) fn decode_tail(cur: &mut Cursor<'_>, rank: u32) -> Result<(Option<Quant>, u64, u32)> {
    let quant = decode_quant(cur, rank)?;
    Ok((quant, cur.read::<u64>()?, cur.read::<u32>()?))
}

/// 量化方案：present 标志只能是 0/1。
pub(crate) fn decode_quant(cur: &mut Cursor<'_>, rank: u32) -> Result<Option<Quant>> {
    match cur.read_u8()? {
        0 => Ok(None),
        1 => Ok(Some(Quant::decode(cur, rank)?)),
        other => Err(bad(format!("quant present 标志只能是 0/1，实际 {other}"))),
    }
}

/// 存储帧长度必须与 codec 的口径一致：store 恰好相等，rans/axes 严格更小。
pub(crate) fn check_frame(name: &str, codec: Codec, data_len: u64, expected: u64) -> Result<()> {
    let valid = match codec {
        Codec::Store => data_len == expected,
        Codec::Rans | Codec::Axes => data_len < expected,
    };
    if valid {
        Ok(())
    } else {
        Err(bad(format!(
            "段 `{name}` 声明 {data_len} 字节，{codec:?} 段不符合 {expected} 字节口径"
        )))
    }
}

/// 量化轴不得超出 rank；没有量化方案时无事可做。
pub(crate) fn check_quant(quant: Option<&Quant>, rank: u32) -> Result<()> {
    if let Some(quant) = quant
        && quant.axis > rank
    {
        return Err(bad("量化轴超出 rank"));
    }
    Ok(())
}
