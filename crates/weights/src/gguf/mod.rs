//! 职责：GGUF v3 只读解析——头部、KV 元数据表与张量信息表、数据区定位，以及分块反量化入口。

use diagnostic::{Code, Error, Result};
use dtype::{DType, QType};
use storage::bad;

use storage::leb::Cursor;

mod dequant;
pub(crate) use dequant::dequantize;

/// 文件 magic。
pub const MAGIC: [u8; 4] = *b"GGUF";
/// 支持的容器版本。
pub const VERSION: u32 = 3;
/// 未声明 `general.alignment` 时的数据区对齐。
pub const DEFAULT_ALIGN: usize = 32;

/// GGML 磁盘类型码的薄映射（ADR 0018 决策 5 的外部码镜像）：只声明本解码器支持哪些类型，
/// 每块元素数与每块字节数一律委托 [`QType`]（块格式）或 [`DType`]（标量元素）的显式布局，
/// 本模块不再自带 `SPAN`/`ELEM` 表。
#[allow(non_camel_case_types)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    /// 32 位浮点（GGML 码 0）。
    F32 = 0,
    /// 16 位浮点（GGML 码 1）。
    F16 = 1,
    /// 4 位传统块（GGML 码 2，f16 尺度 + 32×4bit）。
    Q4_0 = 2,
    /// 5 位传统块（GGML 码 6，f16 尺度 + 32×5bit）。
    Q5_0 = 6,
    /// 5 位传统块（GGML 码 7，f16 尺度 + min + 32×5bit）。
    Q5_1 = 7,
    /// 8 位量化块（GGML 码 8，f16 尺度 + 32×i8）。
    Q8_0 = 8,
    /// 2 位 K 超块（GGML 码 10）。
    Q2_K = 10,
    /// 3 位 K 超块（GGML 码 11）。
    Q3_K = 11,
    /// 4 位 K 超块（GGML 码 12）。
    Q4_K = 12,
    /// 5 位 K 超块（GGML 码 13）。
    Q5_K = 13,
    /// 6 位 K 超块（GGML 码 14）。
    Q6_K = 14,
}

/// 一个已支持格式的布局来源：标量元素按 [`DType`] 计宽，块格式全部委托 [`QType`]。
pub type Layout = dtype::Layout<QType>;

/// 本解码器**已实现**反量化的格式表：GGML 码 → 布局来源。
pub const SUPPORTED: &[(Kind, Layout)] = &[
    (Kind::F32, Layout::Scalar(DType::F32)),
    (Kind::F16, Layout::Scalar(DType::F16)),
    (Kind::Q4_0, Layout::Block(QType::Q4_0)),
    (Kind::Q5_0, Layout::Block(QType::Q5_0)),
    (Kind::Q5_1, Layout::Block(QType::Q5_1)),
    (Kind::Q8_0, Layout::Block(QType::Q8_0)),
    (Kind::Q2_K, Layout::Block(QType::Q2_K)),
    (Kind::Q3_K, Layout::Block(QType::Q3_K)),
    (Kind::Q4_K, Layout::Block(QType::Q4_K)),
    (Kind::Q5_K, Layout::Block(QType::Q5_K)),
    (Kind::Q6_K, Layout::Block(QType::Q6_K)),
];

impl Kind {
    /// 磁盘类型码。
    pub const fn code(self) -> u32 {
        self as u32
    }

    /// 由磁盘码解析；未支持的码显式失败。
    pub fn from_code(code: u32) -> Result<Self> {
        SUPPORTED
            .iter()
            .find_map(|(kind, _)| (kind.code() == code).then_some(*kind))
            .ok_or_else(|| bad("不支持的 ggml 类型码"))
    }

    /// 布局来源；每个变体都必须在 [`SUPPORTED`] 里出现（由 `supported_table_covers_kind` 锁定）。
    pub fn layout(self) -> Result<Layout> {
        SUPPORTED
            .iter()
            .find(|(kind, _)| *kind == self)
            .map(|(_, layout)| *layout)
            .ok_or_else(|| {
                Error::new(Code::Internal, "Kind 的每个变体都必须在 SUPPORTED 里").with("gguf")
            })
    }

    /// 每块元素数：块格式取 [`QType::block_len`]，标量元素按 1。
    pub fn elems(self) -> Result<usize> {
        Ok(self.layout()?.elems() as usize)
    }

    /// 每块字节数：块格式取 [`QType::block_bytes`]，标量元素取 [`DType::bytes_for`] 的单元素值。
    pub fn span(self) -> Result<usize> {
        Ok(self.layout()?.span() as usize)
    }

    /// 元素数 → 磁盘字节数：块布局必须整除元素数，乘法溢出显式失败。
    /// 反量化与 [`Tensor::bytes`] 共用这一个判据，避免两处各写一遍整除。
    pub fn bytes_of(self, count: usize) -> Result<usize> {
        let elems = self.elems()?;
        if !count.is_multiple_of(elems) {
            return Err(bad("元素数不是量化块长的整数倍"));
        }
        (count / elems)
            .checked_mul(self.span()?)
            .ok_or_else(|| bad("张量大小溢出"))
    }
}

/// 一条元数据：键与值；表按书写顺序排列。
pub(crate) type Meta = (String, Value);

/// 元数据值：整数按有无符号归类，浮点统一 f64。
#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    /// 无符号整数。
    Uint(u64),
    /// 有符号整数。
    Int(i64),
    /// 浮点（f32 升 f64）。
    Float(f64),
    /// 布尔。
    Bool(bool),
    /// UTF-8 字符串。
    Str(String),
    /// 同类型数组。
    Arr(Vec<Value>),
}

impl Value {
    /// 字符串视图。
    pub fn text(&self) -> Option<&str> {
        match self {
            Value::Str(text) => Some(text),
            _ => None,
        }
    }

    /// 无符号整数视图。
    pub fn uint(&self) -> Option<u64> {
        match self {
            Value::Uint(value) => Some(*value),
            _ => None,
        }
    }

    /// 有符号整数视图。
    pub fn int(&self) -> Option<i64> {
        match self {
            Value::Int(value) => Some(*value),
            _ => None,
        }
    }

    /// 浮点视图。
    pub fn float(&self) -> Option<f64> {
        match self {
            Value::Float(value) => Some(*value),
            _ => None,
        }
    }

    /// 布尔视图。
    pub fn bool(&self) -> Option<bool> {
        match self {
            Value::Bool(value) => Some(*value),
            _ => None,
        }
    }

    /// 数组视图。
    pub fn arr(&self) -> Option<&[Value]> {
        match self {
            Value::Arr(items) => Some(items),
            _ => None,
        }
    }

    /// 字符串数组：元素必须同为 [`Value::Str`]，混型或缺失返回 `None`。
    pub fn text_array(&self) -> Option<Vec<&str>> {
        self.arr()?.iter().map(Value::text).collect()
    }

    /// 浮点数组：元素必须同为 [`Value::Float`]，混型或缺失返回 `None`。
    pub fn float_array(&self) -> Option<Vec<f64>> {
        self.arr()?.iter().map(Value::float).collect()
    }
}

/// 张量信息表条目；offset 相对数据区起点。
#[derive(Debug, Clone)]
pub struct Tensor {
    /// 张量名。
    pub name: String,
    /// 元素类型。
    pub kind: Kind,
    /// 形状。
    pub shape: Vec<u64>,
    /// 数据区相对偏移。
    pub offset: u64,
}

impl Tensor {
    /// 元素总数，乘法溢出显式失败。
    pub fn count(&self) -> Result<usize> {
        self.shape
            .iter()
            .try_fold(1usize, |acc, &dim| {
                acc.checked_mul(usize::try_from(dim).ok()?)
            })
            .ok_or_else(|| bad("张量形状溢出"))
    }

    /// 磁盘字节数；量化块必须整除元素数。
    pub fn bytes(&self) -> Result<usize> {
        self.kind.bytes_of(self.count()?)
    }
}

/// 已解析的 GGUF 文件，持有调用方字节切片的借用。
pub struct Gguf<'a> {
    bytes: &'a [u8],
    meta: Vec<Meta>,
    tensors: Vec<Tensor>,
    data: usize,
}

impl<'a> Gguf<'a> {
    /// 解析头部、KV 与张量信息表，定位数据区。
    pub fn parse(bytes: &'a [u8]) -> Result<Self> {
        let mut reader = Cursor::new(bytes);
        let (tensors, meta) = tables(&mut reader)?;
        let data = data_start(&meta, reader.pos(), bytes.len())?;
        Ok(Self {
            bytes,
            meta,
            tensors,
            data,
        })
    }

    /// 按 key 取元数据。
    pub fn meta(&self, key: &str) -> Option<&Value> {
        self.meta
            .iter()
            .find(|(name, _)| name == key)
            .map(|(_, value)| value)
    }

    /// 全部元数据。
    #[must_use]
    pub fn metas(&self) -> &[Meta] {
        &self.meta
    }

    /// 全部张量信息。
    #[must_use]
    pub fn tensors(&self) -> &[Tensor] {
        &self.tensors
    }

    /// 按名取张量信息。
    pub fn tensor(&self, name: &str) -> Option<&Tensor> {
        self.tensors.iter().find(|t| t.name == name)
    }

    /// `general.name`。
    pub fn name(&self) -> Option<&str> {
        self.meta("general.name").and_then(Value::text)
    }

    /// `general.architecture`。
    pub fn arch(&self) -> Option<&str> {
        self.meta("general.architecture").and_then(Value::text)
    }

    /// 张量原始字节视图：区间判据走共享的 `storage::leb::slice`，偏移越界与长度越界
    /// 都由它一处拒。
    pub fn raw(&self, tensor: &Tensor) -> Result<&'a [u8]> {
        let offset = usize::try_from(tensor.offset).map_err(|_| bad("张量偏移溢出"))?;
        let start = self
            .data
            .checked_add(offset)
            .ok_or_else(|| bad("张量偏移溢出"))?;
        storage::leb::slice(self.bytes, start, tensor.bytes()?)
    }

    /// 按名反量化为 f32。
    pub fn floats(&self, name: &str) -> Result<Vec<f32>> {
        let tensor = self.required(name)?;
        let raw = self.raw(tensor)?;
        let count = tensor.count()?;
        let mut out = vec![0.0f32; count];
        dequantize(tensor.kind, raw, &mut out)?;
        Ok(out)
    }

    /// 按名取张量信息，缺失显式失败。
    fn required(&self, name: &str) -> Result<&Tensor> {
        self.tensor(name)
            .ok_or_else(|| bad(format!("无张量 `{name}`")))
    }
}

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
    let (kind, count) = length(cursor)?;
    Ok(Value::Arr(records(
        cursor,
        count,
        "数组长度溢出",
        |at| value(at, kind, depth + 1),
    )?))
}

/// 数组头：元素类型码 + 计数，计数不得超过剩余字节（每元素至少 1 字节）。
fn length(cursor: &mut Cursor<'_>) -> Result<(u32, u64)> {
    let kind = cursor.word()?;
    let count = cursor.wide()?;
    if count > cursor.remaining().len() as u64 {
        return Err(bad("数组元素数超过剩余字节"));
    }
    Ok((kind, count))
}

/// 标量：`class << 4 | 宽度`，整数按有无符号扩展，浮点 f32 升 f64。
fn scalar(cursor: &mut Cursor<'_>, kind: u32) -> Result<Value> {
    let packed = packed_code(kind)?;
    let width = usize::from(packed & 0x0F);
    let raw = cursor.take(width)?;
    let mut buf = [0u8; 8];
    buf[..width].copy_from_slice(raw);
    Ok(decode(packed, u64::from_le_bytes(buf)))
}

/// 值类型码 → `class << 4 | 宽度`；`0` 表示这不是标量码。
fn packed_code(kind: u32) -> Result<u8> {
    match NUMBERS.get(kind as usize).copied() {
        Some(0) | None => Err(bad("未知值类型码")),
        Some(packed) => Ok(packed),
    }
}

/// 定宽位模式 → 值：整数左移后算术右移即符号扩展。
fn decode(packed: u8, bits: u64) -> Value {
    let width = usize::from(packed & 0x0F);
    let shift = 64 - width * 8;
    match (packed >> 4, width) {
        (0, _) => Value::Uint(bits),
        (1, _) => Value::Int(((bits << shift) as i64) >> shift),
        (_, 4) => Value::Float(f64::from(f32::from_bits(bits as u32))),
        _ => Value::Float(f64::from_bits(bits)),
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
    let (shape, kind) = geometry(cursor)?;
    Ok(Tensor {
        name,
        kind,
        shape,
        offset: cursor.wide()?,
    })
}

/// 名字之后的几何信息：形状与元素类型，两者都在偏移之前。
fn geometry(cursor: &mut Cursor<'_>) -> Result<(Vec<u64>, Kind)> {
    let shape = shape(cursor)?;
    Ok((shape, Kind::from_code(cursor.word()?)?))
}

/// 形状：维数上限 `MAX_RANK`，每维一个 `u64`。
fn shape(cursor: &mut Cursor<'_>) -> Result<Vec<u64>> {
    let rank = cursor.word()?;
    if rank > MAX_RANK {
        return Err(bad("张量维数超限"));
    }
    records(cursor, u64::from(rank), "张量维数超限", |c| c.wide())
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
pub(crate) fn tables(cursor: &mut Cursor<'_>) -> Result<(Vec<Tensor>, Vec<Meta>)> {
    let (tensor_count, kv_count) = counts(cursor)?;
    let meta = records(cursor, kv_count, "KV 计数溢出", pair)?;
    Ok((records(cursor, tensor_count, "张量计数溢出", tensor)?, meta))
}

/// magic 与版本之后是两个计数：张量条目数在前、KV 对数在后。
fn counts(cursor: &mut Cursor<'_>) -> Result<(u64, u64)> {
    stamp(cursor)?;
    Ok((cursor.wide()?, cursor.wide()?))
}

/// 数据区起点：按 `general.alignment` 对齐（未声明或非法时退回 [`DEFAULT_ALIGN`]），且不得越过末尾。
pub(crate) fn data_start(meta: &[Meta], pos: usize, len: usize) -> Result<usize> {
    let align = meta
        .iter()
        .find(|(key, _)| key == "general.alignment")
        .and_then(|(_, value)| value.uint())
        .and_then(|n| usize::try_from(n).ok())
        .filter(|&n| n > 0)
        .unwrap_or(DEFAULT_ALIGN);
    let Some(data) = storage::align_up(pos, align) else {
        return Err(bad("对齐必须是 2 的幂且对齐后不溢出"));
    };
    if data <= len {
        return Ok(data);
    }
    Err(bad("数据区偏移超出文件"))
}


#[cfg(test)]
#[path = "../../tests/gguf.rs"]
mod tests;
