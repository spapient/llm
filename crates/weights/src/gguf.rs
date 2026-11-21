//! GGUF v3 只读解析：头部、KV 元数据、张量信息表与反量化。
//!
//! 布局：`magic "GGUF" + version(u32) + tensor_count(u64) + kv_count(u64)` → KV 表 → 张量信息表
//! → 按 `general.alignment`（默认 32）对齐 → 数据区。本模块只消费调用方持有的字节切片，不做写入、
//! 不做 I/O；所有偏移与长度都经过边界检查，未信任输入不会触发越界。
//!
//! 支持 GGML 类型码：`F32=0`、`F16=1`、`Q8_0=8`、`Q4_K=12`、`Q5_K=13`、`Q6_K=14`。
//! 反量化只求正确，不做 SIMD；性能路径由上层决定。零第三方依赖，纯安全 Rust。

use diagnostic::{Code, Error, Result};
use dtype::{DType, QType};

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
    /// 8 位量化块（GGML 码 8，f16 尺度 + 32×i8）。
    Q8_0 = 8,
    /// 4 位 K 超块（GGML 码 12）。
    Q4_K = 12,
    /// 5 位 K 超块（GGML 码 13）。
    Q5_K = 13,
    /// 6 位 K 超块（GGML 码 14）。
    Q6_K = 14,
}

/// 一个已支持格式的布局来源：标量元素按 [`DType`] 计宽，块格式全部委托 [`QType`]。
///
/// 几何（`elems`/`span`/`bytes_for`）只有一份实现，住在 `dtype::Layout<B>`：这里只把块参数固化成
/// [`QType`]，标量手臂与块手臂的几何都不再另写一份。
pub type Layout = dtype::Layout<QType>;

/// 本解码器**已实现**反量化的格式表：GGML 码 → 布局来源。
///
/// 这是「支持哪些格式」的唯一事实：[`Kind::from_code`]、[`Kind::elems`] 与 [`Kind::span`]
/// 都读这张表；新增一种可解码格式只加一行。`dtype` 的 [`QType`] 声明了 25 个块格式，
/// 这里只实现其中 4 个，其余 21 个尚无解码器——这张表让「表里有、解不了」成为可读清单，
/// 而不是散在几个 `match` 里的隐式事实。
pub const SUPPORTED: &[(Kind, Layout)] = &[
    (Kind::F32, Layout::Scalar(DType::F32)),
    (Kind::F16, Layout::Scalar(DType::F16)),
    (Kind::Q8_0, Layout::Block(QType::Q8_0)),
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
}

/// 元数据表：键值对按书写顺序。
pub(crate) type Meta = Vec<(String, Value)>;

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
        let count = self.count()?;
        let elem = self.kind.elems()?;
        self.blocks(count, elem)
    }

    /// 块数 × 每块字节数；块大小必须整除元素数，乘法溢出显式失败。
    fn blocks(&self, count: usize, elem: usize) -> Result<usize> {
        if !count.is_multiple_of(elem) {
            return Err(bad("量化块大小不整除元素数"));
        }
        (count / elem)
            .checked_mul(self.kind.span()?)
            .ok_or_else(|| bad("张量大小溢出"))
    }
}

/// 已解析的 GGUF 文件，持有调用方字节切片的借用。
pub struct Gguf<'a> {
    bytes: &'a [u8],
    meta: Vec<(String, Value)>,
    tensors: Vec<Tensor>,
    data: usize,
}

impl<'a> Gguf<'a> {
    /// 解析头部、KV 与张量信息表，定位数据区。
    pub fn parse(bytes: &'a [u8]) -> Result<Self> {
        let mut reader = Reader::new(bytes);
        let (tensors, meta) = tables(&mut reader)?;
        let data = data_start(&meta, reader.pos, bytes.len())?;
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
    pub fn metas(&self) -> &[(String, Value)] {
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

    /// 张量原始字节视图。
    pub fn raw(&self, tensor: &Tensor) -> Result<&'a [u8]> {
        let offset = usize::try_from(tensor.offset).ok();
        let start = offset
            .and_then(|offset| self.data.checked_add(offset))
            .ok_or_else(|| bad("张量偏移溢出"))?;
        let len = tensor.bytes()?;
        self.bytes
            .get(start..start.saturating_add(len))
            .ok_or_else(|| bad("张量字节越界"))
    }

    /// 按名反量化为 f32。
    pub fn floats(&self, name: &str) -> Result<Vec<f32>> {
        let tensor = self.named(name)?;
        let raw = self.raw(tensor)?;
        let count = tensor.count()?;
        let mut out = vec![0.0f32; count];
        dequantize(tensor.kind, raw, &mut out)?;
        Ok(out)
    }

    /// 按名取张量信息；缺失时显式报错（[`Gguf::tensor`] 的 `Option` 留给调用方自己决定）。
    fn named(&self, name: &str) -> Result<&Tensor> {
        self.tensor(name)
            .ok_or_else(|| bad(&format!("无张量 `{name}`")))
    }
}

pub(crate) fn bad(message: &str) -> Error {
    Error::new(Code::Invalid, message).with("gguf")
}

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
        false => Err(bad(&format!("版本 {version}，期望 {VERSION}"))),
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
fn alignment(meta: &[(String, Value)]) -> usize {
    meta.iter()
        .find(|(key, _)| *key == "general.alignment")
        .and_then(|(_, value)| value.uint())
        .and_then(|n| usize::try_from(n).ok())
        .filter(|&n| n > 0)
        .unwrap_or(DEFAULT_ALIGN)
}

/// 数据区起点：按 `general.alignment` 对齐，且不能越过文件末尾。
pub(crate) fn data_start(meta: &[(String, Value)], pos: usize, len: usize) -> Result<usize> {
    let Some(data) = crate::align_up(pos, alignment(meta)) else {
        return Err(bad("对齐必须是 2 的幂且对齐后不溢出"));
    };
    if data <= len {
        return Ok(data);
    }
    Err(bad("数据区偏移超出文件"))
}


/// 把一个张量的原始字节反量化进 `out`；长度不符或布局不符显式报错。
pub(crate) fn dequantize(kind: Kind, raw: &[u8], out: &mut [f32]) -> Result<()> {
    check_len(kind, raw.len(), out.len())?;
    match kind {
        Kind::F32 => decode_scalar::<4>(raw, out, "f32", f32::from_le_bytes),
        Kind::F16 => decode_scalar::<2>(raw, out, "f16", |b| f16_to_f32(u16::from_le_bytes(b))),
        block => decode_block(block, raw, out),
    }
}

/// 块格式（Q8_0 与 K 超块）的分派；两个标量格式已在 [`dequantize`] 就地分派。
fn decode_block(kind: Kind, raw: &[u8], out: &mut [f32]) -> Result<()> {
    match kind {
        Kind::Q8_0 => decode_q8(raw, out),
        Kind::Q4_K => decode_k(raw, out, 144, 0),
        Kind::Q5_K => decode_k(raw, out, 176, 32),
        _ => decode_q6k(raw, out),
    }
}

/// 字节长度必须正好等于块布局推出的值：短了会读出半个张量，长了会把下一条混进来。
fn check_len(kind: Kind, raw: usize, count: usize) -> Result<()> {
    let expect = kind.span()? * (count / kind.elems()?);
    if raw == expect {
        return Ok(());
    }
    Err(bad("张量字节长度与块布局不符"))
}

/// 标量格式：固定宽度小端，逐元素转换；长度不符即报错（`what` 是类型名）。
fn decode_scalar<const N: usize>(
    raw: &[u8],
    out: &mut [f32],
    what: &str,
    convert: impl Fn([u8; N]) -> f32,
) -> Result<()> {
    let (chunks, tail) = raw.as_chunks::<N>();
    if !tail.is_empty() || chunks.len() != out.len() {
        return Err(bad(&format!("{what} 长度不符")));
    }
    for (slot, block) in out.iter_mut().zip(chunks) {
        *slot = convert(*block);
    }
    Ok(())
}

/// 半精度位模式 → f32，含次正规、无穷与 NaN。
pub(crate) fn f16_to_f32(bits: u16) -> f32 {
    let sign = u32::from(bits >> 15) << 31;
    let exp = u32::from((bits >> 10) & 0x1F);
    let frac = u32::from(bits & 0x03FF);
    if exp == 31 {
        f32::from_bits(sign | 0x7F80_0000 | (frac << 13))
    } else if exp == 0 {
        let mag = frac as f32 * 5.960_464_5e-8;
        f32::from_bits(sign | mag.to_bits())
    } else {
        f32::from_bits(sign | ((exp + 112) << 23) | (frac << 13))
    }
}

fn half_at(raw: &[u8], at: usize) -> f32 {
    f16_to_f32(u16::from_le_bytes([raw[at], raw[at + 1]]))
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
fn scale_min(scales: &[u8], index: usize) -> (u8, u8) {
    if index < 4 {
        return (scales[index] & 63, scales[index + 4] & 63);
    }
    let high = (scales[index - 4] >> 6) << 4;
    let low = (scales[index] >> 6) << 4;
    (
        (scales[index + 4] & 0x0F) | high,
        (scales[index + 4] >> 4) | low,
    )
}

/// Q4_K / Q5_K 共用的解码：Q5_K 多一段 `qh`（每个元素的第 5 位），块长度多 32 字节。
/// 两者每块都是 4 个 64 元素的组，组内高低半字节分别写 `lane` 与 `lane + 32`。
fn decode_k(raw: &[u8], out: &mut [f32], block: usize, high_bytes: usize) -> Result<()> {
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
            let row = &mut slot[group * 64..];
            for lane in 0..32 {
                let high0 = high_bit(high, lane, group * 2);
                let high1 = high_bit(high, lane, group * 2 + 1);
                let q0 = (qs[lane] & 0x0F) + high0;
                let q1 = (qs[lane] >> 4) + high1;
                row[lane] = d * f32::from(s0) * f32::from(q0) - min * f32::from(m0);
                row[lane + 32] = d * f32::from(s1) * f32::from(q1) - min * f32::from(m1);
            }
        }
    }
    Ok(())
}

/// Q5_K 的第 5 位：`high` 为空（Q4_K）时恒为 0。
fn high_bit(high: &[u8], lane: usize, bit: usize) -> u8 {
    high.get(lane)
        .map_or(0, |byte| u8::from(byte & (1 << bit) != 0) << 4)
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
            let row = &mut slot[part * 128..][..128];
            for lane in 0..32 {
                let base = lane / 16;
                let q1 = (ql[lane] & 0x0F) | ((qh[lane] & 0x03) << 4);
                let q2 = (ql[lane + 32] & 0x0F) | (((qh[lane] >> 2) & 0x03) << 4);
                let q3 = (ql[lane] >> 4) | (((qh[lane] >> 4) & 0x03) << 4);
                let q4 = (ql[lane + 32] >> 4) | (((qh[lane] >> 6) & 0x03) << 4);
                row[lane] = d * f32::from(scales[base] as i8) * (q1 as i32 - 32) as f32;
                row[lane + 32] = d * f32::from(scales[base + 2] as i8) * (q2 as i32 - 32) as f32;
                row[lane + 64] = d * f32::from(scales[base + 4] as i8) * (q3 as i32 - 32) as f32;
                row[lane + 96] = d * f32::from(scales[base + 6] as i8) * (q4 as i32 - 32) as f32;
            }
        }
    }
    Ok(())
}


#[cfg(test)]
#[path = "../tests/gguf.rs"]
mod tests;
