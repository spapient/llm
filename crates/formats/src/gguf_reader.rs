//! GGUF 的字节读取与表解析：`Reader`、值/张量记录的消费、对齐与数据区定位。

use diagnostic::Result;

use wire::bad;

use crate::gguf::{DEFAULT_ALIGN, Kind, MAGIC, Meta, Tensor, VERSION, Value};

/// 数组最大嵌套深度（防未信任输入递归耗尽栈）。
const DEPTH: usize = 8;
/// 张量最大维数。
const MAX_RANK: u32 = 8;

/// 小端读取器，所有消费经边界检查。
pub(crate) struct Reader<'a> {
    bytes: &'a [u8],
    pub(crate) pos: usize,
    depth: usize,
}

impl<'a> Reader<'a> {
    pub(crate) fn new(bytes: &'a [u8]) -> Self {
        Self {
            bytes,
            pos: 0,
            depth: 0,
        }
    }

    fn take(&mut self, len: usize) -> Result<&'a [u8]> {
        let stop = self.pos.checked_add(len).ok_or_else(|| bad("偏移溢出"))?;
        let out = self
            .bytes
            .get(self.pos..stop)
            .ok_or_else(|| bad("越界读取"))?;
        self.pos = stop;
        Ok(out)
    }

    fn word(&mut self) -> Result<u32> {
        let raw = self.take(4)?;
        Ok(u32::from_le_bytes(
            raw.try_into().map_err(|_| bad("越界读取"))?,
        ))
    }

    fn wide(&mut self) -> Result<u64> {
        let raw = self.take(8)?;
        Ok(u64::from_le_bytes(
            raw.try_into().map_err(|_| bad("越界读取"))?,
        ))
    }

    fn text(&mut self) -> Result<String> {
        let len = usize::try_from(self.wide()?).map_err(|_| bad("字符串长度溢出"))?;
        let raw = self.take(len)?;
        String::from_utf8(raw.to_vec()).map_err(|_| bad("字符串不是 UTF-8"))
    }

    fn array_count(&mut self) -> Result<usize> {
        let count = usize::try_from(self.wide()?).map_err(|_| bad("数组长度溢出"))?;
        if count <= self.bytes.len().saturating_sub(self.pos) {
            return Ok(count);
        }
        Err(bad("数组元素数超过剩余字节"))
    }

    fn value(&mut self, kind: u32) -> Result<Value> {
        match kind {
            7 => Ok(Value::Bool(self.take(1)?[0] != 0)),
            8 => self.text().map(Value::Str),
            9 => self.list(),
            _ => self.scalar(kind),
        }
    }

    fn list(&mut self) -> Result<Value> {
        if self.depth >= DEPTH {
            return Err(bad("数组嵌套过深"));
        }
        let (kind, count) = self.header()?;
        self.depth += 1;
        let mut out = Vec::with_capacity(count.min(4096));
        for _ in 0..count {
            out.push(self.value(kind)?);
        }
        self.depth -= 1;
        Ok(Value::Arr(out))
    }

    /// 数组头：元素类型码与元素数（元素数已按剩余字节粗查过）。
    fn header(&mut self) -> Result<(u32, usize)> {
        Ok((self.word()?, self.array_count()?))
    }

    fn scalar(&mut self, kind: u32) -> Result<Value> {
        let packed = NUMBERS.get(kind as usize).copied().unwrap_or(0);
        if packed == 0 {
            return Err(bad("未知值类型码"));
        }
        let width = usize::from(packed & 0x0F);
        let raw = self.take(width)?;
        let mut buf = [0u8; 8];
        buf[..width].copy_from_slice(raw);
        Ok(number(packed >> 4, u64::from_le_bytes(buf), width))
    }
}

/// 值类型码 → `class << 4 | 字节宽度`；`0` 表示这不是标量码（字符串/数组/表外码）。
const NUMBERS: [u8; 13] = [
    0x01, 0x11, 0x02, 0x12, 0x04, 0x14, 0x24, 0, 0, 0, 0x08, 0x18, 0x28,
];

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

/// 读 `count` 条记录；容量按 4096 封顶——谎报的巨大计数不会换来一次巨额分配。
fn records<Item>(
    reader: &mut Reader<'_>,
    count: u64,
    overflow: &str,
    item: impl Fn(&mut Reader<'_>) -> Result<Item>,
) -> Result<Vec<Item>> {
    let count = usize::try_from(count).map_err(|_| bad(overflow))?;
    let mut out = Vec::with_capacity(count.min(4096));
    for _ in 0..count {
        out.push(item(reader)?);
    }
    Ok(out)
}

/// 一对 KV：键、类型码与值。
fn pair(reader: &mut Reader<'_>) -> Result<(String, Value)> {
    let key = reader.text()?;
    let kind = reader.word()?;
    Ok((key, reader.value(kind)?))
}

/// 一条张量信息：名字、形状、元素类型与数据区偏移（偏移相对数据区起点）。
fn tensor(reader: &mut Reader<'_>) -> Result<Tensor> {
    let name = reader.text()?;
    let shape = shape(reader)?;
    let kind = kind_of(reader)?;
    Ok(Tensor {
        name,
        kind,
        shape,
        offset: reader.wide()?,
    })
}

/// 形状：维数上限 `MAX_RANK`，每维一个 `u64`。
fn shape(reader: &mut Reader<'_>) -> Result<Vec<u64>> {
    let rank = reader.word()?;
    if rank > MAX_RANK {
        return Err(bad("张量维数超限"));
    }
    let mut shape = Vec::with_capacity(rank as usize);
    for _ in 0..rank {
        shape.push(reader.wide()?);
    }
    Ok(shape)
}

/// 元素类型码：闭集之外的码显式失败（判据在 [`Kind::from_code`]，这里只做“先读再解”）。
fn kind_of(reader: &mut Reader<'_>) -> Result<Kind> {
    Kind::from_code(reader.word()?)
}

/// 头部 magic 与版本：两个都对得上才继续，不做向后兼容猜测。
fn stamp(reader: &mut Reader<'_>) -> Result<()> {
    let magic = reader.take(4)?;
    if magic != MAGIC {
        return Err(bad("magic 不匹配"));
    }
    let version = reader.word()?;
    match version == VERSION {
        true => Ok(()),
        false => Err(bad(format!("版本 {version}，期望 {VERSION}"))),
    }
}

/// 头部两个计数：张量信息表条目数与 KV 对数。
fn counts(reader: &mut Reader<'_>) -> Result<(u64, u64)> {
    Ok((reader.wide()?, reader.wide()?))
}

/// 头部与两张表：magic/版本、两个计数，接着 KV 表与张量信息表。
pub(crate) fn tables(reader: &mut Reader<'_>) -> Result<(Vec<Tensor>, Meta)> {
    stamp(reader)?;
    let (tensor_count, kv_count) = counts(reader)?;
    let meta = records(reader, kv_count, "KV 计数溢出", pair)?;
    Ok((records(reader, tensor_count, "张量计数溢出", tensor)?, meta))
}

/// `general.alignment` 声明的正整数，未声明或非法时退回 [`DEFAULT_ALIGN`]。
pub(crate) fn alignment(meta: &[(String, Value)]) -> usize {
    meta.iter()
        .find(|(key, _)| *key == "general.alignment")
        .and_then(|(_, value)| value.uint())
        .and_then(|n| usize::try_from(n).ok())
        .filter(|&n| n > 0)
        .unwrap_or(DEFAULT_ALIGN)
}

/// 数据区起点：按 `general.alignment` 对齐，且不能越过文件末尾。
pub(crate) fn data_start(meta: &[(String, Value)], pos: usize, len: usize) -> Result<usize> {
    let Some(data) = wire::align_up(pos, alignment(meta)) else {
        return Err(bad("对齐必须是 2 的幂且对齐后不溢出"));
    };
    if data <= len {
        return Ok(data);
    }
    Err(bad("数据区偏移超出文件"))
}
