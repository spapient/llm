//! 职责：GGUF v3 只读解析——头部、KV 元数据表与张量信息表、数据区定位，以及分块反量化入口。

use crate::codec::{decode_scalar, f16_to_f32, half_at};
use diagnostic::{Code, Error, Result};
use dtype::{DType, QType};
use storage::bad;

use storage::leb::Cursor;

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

// —— 分块反量化：块内位域 → f32。纯数学函数，不持有文件/游标状态；分派表代替多臂 `match` ——

/// 把一个张量的原始字节反量化进 `out`；长度不符或布局不符显式报错。
pub(crate) fn dequantize(kind: Kind, raw: &[u8], out: &mut [f32]) -> Result<()> {
    if raw.len() != kind.bytes_of(out.len())? {
        return Err(bad("张量字节长度与块布局不符"));
    }
    decode_kind(kind, raw, out)
}

/// 按类型分派：两种标量格式留在这里，分块格式下沉一层。
fn decode_kind(kind: Kind, raw: &[u8], out: &mut [f32]) -> Result<()> {
    match kind {
        Kind::F32 => decode_scalar::<4>(raw, out, "f32", f32::from_le_bytes),
        Kind::F16 => decode_scalar::<2>(raw, out, "f16", |b| f16_to_f32(u16::from_le_bytes(b))),
        _ => blocked(kind, raw, out),
    }
}

/// 分块格式的位解包：传统 Q（`Q4_0`/`Q5_0`/`Q5_1`/`Q8_0`）与 K 超块（`Q2_K`~`Q6_K`）各自分派。
/// 分块格式的解码函数。
type Decoder = fn(&[u8], &mut [f32]) -> Result<()>;

/// 分块格式 → 解码函数。一张表代替九臂 `match`：加一种格式只加一行，
/// 分派本身没有分支，也就不会随格式数量长出复杂度。
const BLOCKED: &[(Kind, Decoder)] = &[
    (Kind::Q4_0, decode_q4_0),
    (Kind::Q5_0, decode_q5_0),
    (Kind::Q5_1, decode_q5_1),
    (Kind::Q8_0, decode_q8),
    (Kind::Q2_K, decode_q2k),
    (Kind::Q3_K, decode_q3k),
    (Kind::Q4_K, decode_k::<false>),
    (Kind::Q5_K, decode_k::<true>),
    (Kind::Q6_K, decode_q6k),
];

fn blocked(kind: Kind, raw: &[u8], out: &mut [f32]) -> Result<()> {
    match BLOCKED.iter().find(|(known, _)| *known == kind) {
        Some((_, decode)) => decode(raw, out),
        None => Err(bad(format!("{kind:?} 不是分块格式"))),
    }
}

fn decode_q8(raw: &[u8], out: &mut [f32]) -> Result<()> {
    let (blocks, _) = raw.as_chunks::<34>();
    let (slots, _) = out.as_chunks_mut::<32>();
    for (slot, block) in slots.iter_mut().zip(blocks) {
        let d = half_at(block, 0);
        for (value, &q) in slot.iter_mut().zip(&block[2..]) {
            *value = d * f32::from(q as i8);
        }
    }
    Ok(())
}

/// Q4_K 的 6 位 scale/min 解包。
fn scale_min(scales: &[u8], idx: usize) -> (u8, u8) {
    if idx < 4 {
        return (scales[idx] & 63, scales[idx + 4] & 63);
    }
    let high = (scales[idx - 4] >> 6) << 4;
    let low = (scales[idx] >> 6) << 4;
    (
        (scales[idx + 4] & 0x0F) | high,
        (scales[idx + 4] >> 4) | low,
    )
}

/// Q4_K / Q5_K 共用的解码：Q5_K 多一段 `qh`（每个元素的第 5 位），块长度多 32 字节。
/// 两者每块都是 4 个 64 元素的组，组内高低半字节分别写 `lane` 与 `lane + 32`。
///
/// `HIGH` 为编译期常量而不是运行时开关：Q4_K 是磁盘上最常见的方向，它的内循环里
/// 因此没有 `qh` 的取值与越界检查（`high_bit` 只剩 Q5_K 一份下游）。
fn decode_k<const HIGH: bool>(raw: &[u8], out: &mut [f32]) -> Result<()> {
    let (block, high_bytes) = if HIGH { (176, 32) } else { (144, 0) };
    let (slots, _) = out.as_chunks_mut::<256>();
    for (slot, bytes) in slots.iter_mut().zip(raw.chunks_exact(block)) {
        let d = half_at(bytes, 0);
        let min = half_at(bytes, 2);
        let scales = &bytes[4..16];
        let (high, quantized) = bytes[16..].split_at(high_bytes);
        let (groups, _) = quantized.as_chunks::<32>();
        for (group, qs) in groups.iter().enumerate() {
            let (s0, m0) = scale_min(scales, group * 2);
            let (s1, m1) = scale_min(scales, group * 2 + 1);
            let (ds0, mm0) = (d * f32::from(s0), min * f32::from(m0));
            let (ds1, mm1) = (d * f32::from(s1), min * f32::from(m1));
            let row = &mut slot[group * 64..];
            for lane in 0..32 {
                let high0 = high_bit::<HIGH>(high, lane, group * 2);
                let high1 = high_bit::<HIGH>(high, lane, group * 2 + 1);
                let q0 = (qs[lane] & 0x0F) + high0;
                let q1 = (qs[lane] >> 4) + high1;
                row[lane] = ds0 * f32::from(q0) - mm0;
                row[lane + 32] = ds1 * f32::from(q1) - mm1;
            }
        }
    }
    Ok(())
}

/// Q5_K 的第 5 位；`HIGH` 为 `false`（Q4_K）时整条取值路径在编译期消失，恒为 0。
fn high_bit<const HIGH: bool>(high: &[u8], lane: usize, bit: usize) -> u8 {
    match high.get(lane) {
        Some(byte) if HIGH => u8::from(byte & (1 << bit) != 0) << 4,
        _ => 0,
    }
}

fn decode_q6k(raw: &[u8], out: &mut [f32]) -> Result<()> {
    let (blocks, _) = raw.as_chunks::<210>();
    let (slots, _) = out.as_chunks_mut::<256>();
    for (slot, block) in slots.iter_mut().zip(blocks) {
        let d = half_at(block, 208);
        for part in 0..2 {
            let ql = &block[part * 64..][..64];
            let qh = &block[128 + part * 32..][..32];
            let scales = &block[192 + part * 8..][..8];
            let mut ds = [0.0f32; 8];
            for (slot, &s) in ds.iter_mut().zip(scales) {
                *slot = d * f32::from(s as i8);
            }
            let row = &mut slot[part * 128..][..128];
            for lane in 0..32 {
                let base = lane / 16;
                let q1 = (ql[lane] & 0x0F) | ((qh[lane] & 0x03) << 4);
                let q2 = (ql[lane + 32] & 0x0F) | (((qh[lane] >> 2) & 0x03) << 4);
                let q3 = (ql[lane] >> 4) | (((qh[lane] >> 4) & 0x03) << 4);
                let q4 = (ql[lane + 32] >> 4) | (((qh[lane] >> 6) & 0x03) << 4);
                row[lane] = ds[base] * (q1 as i32 - 32) as f32;
                row[lane + 32] = ds[base + 2] * (q2 as i32 - 32) as f32;
                row[lane + 64] = ds[base + 4] * (q3 as i32 - 32) as f32;
                row[lane + 96] = ds[base + 6] * (q4 as i32 - 32) as f32;
            }
        }
    }
    Ok(())
}

/// `Q4_0`/`Q5_0`/`Q5_1` 共用的「f16 缩放 + 低 4 位量化点」解码：三者只是块长、有无 f16 min 与
/// 有无第 5 位不同。`QH_AT = usize::MAX` 表示无第 5 位（Q4_0），`SIGNED` 区分「减偏置」
/// （Q4_0 减 8、Q5_0 减 16）与「加 min」（Q5_1）；参数在实例化时被常量折叠，无运行时分支。
fn decode_low4<const BLOCK: usize, const QH_AT: usize, const SIGNED: bool>(
    raw: &[u8],
    out: &mut [f32],
) -> Result<()> {
    let (blocks, _) = raw.as_chunks::<BLOCK>();
    let (slots, _) = out.as_chunks_mut::<32>();
    let (qs_at, bias) = low4_geom::<QH_AT, SIGNED>();
    for (slot, block) in slots.iter_mut().zip(blocks) {
        decode_low4_block::<QH_AT, SIGNED>(slot, block, qs_at, bias);
    }
    Ok(())
}

/// 量化点起点与符号偏置：无 qh 时紧跟 f16 缩放并减 8，有 qh 时越过 4 字节第 5 位并减 16。
fn low4_geom<const QH_AT: usize, const SIGNED: bool>() -> (usize, f32) {
    if QH_AT == usize::MAX {
        (2, if SIGNED { 8.0 } else { 0.0 })
    } else {
        (QH_AT + 4, if SIGNED { 16.0 } else { 0.0 })
    }
}

/// 解一个 32 槽块：高低半字节拆出，按符号减偏置或无符号加 min 写回。
fn decode_low4_block<const QH_AT: usize, const SIGNED: bool>(
    slot: &mut [f32],
    block: &[u8],
    qs_at: usize,
    bias: f32,
) {
    let d = half_at(block, 0);
    let m = if SIGNED { 0.0 } else { half_at(block, 2) };
    let qh = if QH_AT == usize::MAX {
        0
    } else {
        u32::from_le_bytes([
            block[QH_AT],
            block[QH_AT + 1],
            block[QH_AT + 2],
            block[QH_AT + 3],
        ])
    };
    for (j, &q) in block[qs_at..].iter().enumerate() {
        let hi0 = (((qh >> j) & 1) as u8) << 4;
        let hi1 = (((qh >> (j + 16)) & 1) as u8) << 4;
        let (lo, hi) = ((q & 0x0F) | hi0, (q >> 4) | hi1);
        if SIGNED {
            slot[j] = d * (f32::from(lo) - bias);
            slot[j + 16] = d * (f32::from(hi) - bias);
        } else {
            slot[j] = d * f32::from(lo) + m;
            slot[j + 16] = d * f32::from(hi) + m;
        }
    }
}

/// `Q4_0`：块 32、18 字节，4-bit 量化点（减 8）。
fn decode_q4_0(raw: &[u8], out: &mut [f32]) -> Result<()> {
    decode_low4::<18, { usize::MAX }, true>(raw, out)
}

/// `Q5_0`：块 32、22 字节，4-bit 低半字节 + 第 5 位（减 16）。
fn decode_q5_0(raw: &[u8], out: &mut [f32]) -> Result<()> {
    decode_low4::<22, 2, true>(raw, out)
}

/// `Q5_1`：块 32、24 字节，无符号量化点加 f16 min。
fn decode_q5_1(raw: &[u8], out: &mut [f32]) -> Result<()> {
    decode_low4::<24, 4, false>(raw, out)
}

/// `Q2_K`：块 256、84 字节。f16 `d`+`dmin`、16 个 4-bit scale/min、64 字节 2-bit 量化点。
/// 每 16 元素一个 scale，低 4 位是尺度乘数、高 4 位是 min 乘数；量化点有符号（2-bit）。
fn decode_q2k(raw: &[u8], out: &mut [f32]) -> Result<()> {
    let (blocks, _) = raw.as_chunks::<84>();
    let (slots, _) = out.as_chunks_mut::<256>();
    for (slot, block) in slots.iter_mut().zip(blocks) {
        let (d, min) = (half_at(block, 0), half_at(block, 2));
        let scales = &block[4..20];
        let qs = &block[20..84];
        let mut is = 0usize;
        let mut y = 0usize;
        for n in 0..2 {
            let q = &qs[n * 32..];
            let mut shift = 0u32;
            for _j in 0..4 {
                let (dl, ml) = scale_pair(d, min, scales[is]);
                is += 1;
                lanes_2bit(slot, &mut y, q, shift, dl, ml);
                let (dl, ml) = scale_pair(d, min, scales[is]);
                is += 1;
                lanes_2bit(slot, &mut y, &q[16..], shift, dl, ml);
                shift += 2;
            }
        }
    }
    Ok(())
}

/// Q2_K 的一对缩放：低四位是 scale，高四位是 min。
fn scale_pair(d: f32, min: f32, sc: u8) -> (f32, f32) {
    (d * f32::from(sc & 0x0F), min * f32::from(sc >> 4))
}

/// 16 个 3-bit lane：高位掩码 `hm` 决定减不减偏置（Q3_K 的 `-4`）。
fn lanes_3bit(slot: &mut [f32], y: &mut usize, q: &[u8], shift: u32, dl: f32, hm: &[u8], m: u8) {
    for (at, value) in q.iter().take(16).enumerate() {
        let sub = if hm[at] & m == 0 { 4 } else { 0 };
        slot[*y] = dl * f32::from(((value >> shift) & 3) as i8 - sub);
        *y += 1;
    }
}

/// 16 个 2-bit lane：`out = dl * q - ml`。
fn lanes_2bit(slot: &mut [f32], y: &mut usize, q: &[u8], shift: u32, dl: f32, ml: f32) {
    for value in q.iter().take(16) {
        slot[*y] = dl * f32::from(((value >> shift) & 3) as i8) - ml;
        *y += 1;
    }
}

/// `Q3_K`：块 256、110 字节。32 字节 mask + 64 字节 2-bit 量化点 + 12 字节 6-bit scale +
/// 尾 2 字节 f16 `d`。mask 的第 `m` 位补每个量化点的第 3 位；scale 按交叉排列解包（减 32 偏置）。
fn decode_q3k(raw: &[u8], out: &mut [f32]) -> Result<()> {
    let (blocks, _) = raw.as_chunks::<110>();
    let (slots, _) = out.as_chunks_mut::<256>();
    for (slot, block) in slots.iter_mut().zip(blocks) {
        let d_all = half_at(block, 108);
        let hm = &block[0..32];
        let qs = &block[32..96];
        let scales = unpack_q3_scales(&block[96..108]);
        let mut is = 0usize;
        let mut y = 0usize;
        let mut m: u8 = 1;
        for n in 0..2 {
            let q = &qs[n * 32..];
            let mut shift = 0u32;
            for _j in 0..4 {
                let dl = d_all * f32::from(scales[is] - 32);
                is += 1;
                lanes_3bit(slot, &mut y, q, shift, dl, hm, m);
                let dl = d_all * f32::from(scales[is] - 32);
                is += 1;
                lanes_3bit(slot, &mut y, &q[16..], shift, dl, &hm[16..], m);
                shift += 2;
                m <<= 1;
            }
        }
    }
    Ok(())
}

/// `Q3_K` 的 12 字节 scales → 16 个 6-bit 尺度：交叉排列的位解包，逐字节对照 ggml-quants.c。
fn unpack_q3_scales(raw: &[u8]) -> [i8; 16] {
    const KMASK1: u32 = 0x0303_0303;
    const KMASK2: u32 = 0x0f0f_0f0f;
    let mut aux = [0u32; 4];
    aux[0] = u32::from_le_bytes([raw[0], raw[1], raw[2], raw[3]]);
    aux[1] = u32::from_le_bytes([raw[4], raw[5], raw[6], raw[7]]);
    aux[2] = u32::from_le_bytes([raw[8], raw[9], raw[10], raw[11]]);
    let tmp = aux[2];
    aux[2] = ((aux[0] >> 4) & KMASK2) | (((tmp >> 4) & KMASK1) << 4);
    aux[3] = ((aux[1] >> 4) & KMASK2) | (((tmp >> 6) & KMASK1) << 4);
    aux[0] = (aux[0] & KMASK2) | ((tmp & KMASK1) << 4);
    aux[1] = (aux[1] & KMASK2) | (((tmp >> 2) & KMASK1) << 4);
    let mut scales = [0i8; 16];
    for (i, word) in aux.iter().enumerate() {
        for (j, byte) in word.to_le_bytes().iter().enumerate() {
            scales[i * 4 + j] = *byte as i8;
        }
    }
    scales
}

#[cfg(test)]
mod dequant_tests {
    use super::*;

    /// f16 位模式 → 小端字节。
    fn f16_bytes(bits: u16) -> [u8; 2] {
        bits.to_le_bytes()
    }

    #[cfg(test)]
    testkit::test_cases! {
    fn q4_0_dequantizes_a_known_block() {
        let mut block = [0u8; 18];
        block[0..2].copy_from_slice(&f16_bytes(0x3C00)); // d = 1.0
        block[2] = 0x08; // 元素 0 = 8-8 = 0；元素 16 = 0-8 = -8
        block[3] = 0x19; // 元素 1 = 9-8 = 1；元素 17 = 1-8 = -7
        let mut out = [0.0f32; 32];
        decode_q4_0(&block, &mut out).expect("解码");
        assert_eq!(out[0], 0.0);
        assert_eq!(out[16], -8.0);
        assert_eq!(out[1], 1.0);
        assert_eq!(out[17], -7.0);
    }

    fn q5_0_dequantizes_a_known_block() {
        let mut block = [0u8; 22];
        block[0..2].copy_from_slice(&f16_bytes(0x3C00)); // d = 1.0
        block[6] = 0x10; // 元素 0 = 0-16 = -16；元素 16 = 1-16 = -15
        block[2] = 0x01; // qh bit 0 = 1 → 元素 0 第 5 位 = 1 → 16-16 = 0
        let mut out = [0.0f32; 32];
        decode_q5_0(&block, &mut out).expect("解码");
        assert_eq!(out[0], 0.0);
        assert_eq!(out[16], -15.0);
    }

    fn q5_1_dequantizes_a_known_block() {
        let mut block = [0u8; 24];
        block[0..2].copy_from_slice(&f16_bytes(0x3C00)); // d = 1.0
        block[2..4].copy_from_slice(&f16_bytes(0x3800)); // m = 0.5
        block[8] = 0x10; // 元素 0 = 0，元素 16 = 1
        let mut out = [0.0f32; 32];
        decode_q5_1(&block, &mut out).expect("解码");
        assert_eq!(out[0], 0.5);
        assert_eq!(out[16], 1.5);
    }

    fn q2_k_dequantizes_a_known_block() {
        let mut block = [0u8; 84];
        block[0..2].copy_from_slice(&f16_bytes(0x3C00)); // d = 1.0
        block[2..4].copy_from_slice(&f16_bytes(0x3800)); // dmin = 0.5
        block[4..20].fill(0x11); // 每个 scale：低 4 位 = 1（dl=1.0）、高 4 位 = 1（ml=0.5）
        // qs 全 0 → 量化点 0 → 输出 = 1.0*0 - 0.5 = -0.5
        let mut out = [0.0f32; 256];
        decode_q2k(&block, &mut out).expect("解码");
        assert!(out.iter().all(|&v| (v + 0.5).abs() < 1e-6));
    }

    fn q3_k_scales_unpack_crosses_6bit_fields() {
        // 全 0xFF 的 6-bit 交叉排列解包后是 16 个 63。
        assert_eq!(unpack_q3_scales(&[0xFF; 12]), [63i8; 16]);
    }

    fn q3_k_dequantizes_a_zero_block() {
        let mut block = [0u8; 110];
        block[108..110].copy_from_slice(&f16_bytes(0x3C00)); // d = 1.0
        block[0..32].fill(0xFF); // mask 全 1，不扣 4
        // scales 全 0 → dl = 1.0*(0-32) = -32；qs 全 0 → 量化点 0 → 输出 0
        let mut out = [0.0f32; 256];
        decode_q3k(&block, &mut out).expect("解码");
        assert!(out.iter().all(|&v| v == 0.0));
    }
    }
}

#[cfg(test)]
#[path = "../tests/gguf.rs"]
mod tests;
