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
/// 数组最大嵌套深度（防未信任输入递归耗尽栈）。
const DEPTH: usize = 8;
/// 张量最大维数。
const MAX_RANK: u32 = 8;

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
/// 几何（`elems`/`span`/`bytes_for`）只有一份实现，住在 `dtype::Layout<B>`；这里只把块参数
/// 固化成 [`QType`]。MXFP4/NVFP4 解码接入时换成 `dtype::Layout<MxType>` 即可，不必再写一份
/// 「标量手臂 + 块手臂」的几何。
pub type Layout = dtype::Layout<QType>;

/// 本解码器**已实现**反量化的格式表：GGML 码 → 布局来源。
///
/// 这是「支持哪些格式」的唯一事实，[`Kind::from_code`]、[`Kind::elems`] 与 [`Kind::span`]
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
    #[must_use]
    pub const fn code(self) -> u32 {
        self as u32
    }

    /// 由磁盘码解析；未支持的码显式失败。
    pub fn from_code(code: u32) -> Result<Self> {
        SUPPORTED
            .iter()
            .map(|(kind, _)| *kind)
            .find(|kind| kind.code() == code)
            .ok_or_else(|| bad("不支持的 ggml 类型码"))
    }

    /// 布局来源；每个变体都必须在 [`SUPPORTED`] 里出现（由 `supported_table_covers_kind` 锁定）。
    #[must_use]
    pub const fn layout(self) -> Layout {
        let mut index = 0;
        while index < SUPPORTED.len() {
            let (kind, layout) = SUPPORTED[index];
            if kind as u32 == self as u32 {
                return layout;
            }
            index += 1;
        }
        panic!("Kind 的每个变体都必须在 SUPPORTED 里")
    }

    /// 每块元素数：块格式取 [`QType::block_len`]，标量元素按 1。
    #[must_use]
    pub fn elems(self) -> usize {
        self.layout().elems() as usize
    }

    /// 每块字节数：块格式取 [`QType::block_bytes`]，标量元素取 [`DType::bytes_for`] 的单元素值。
    #[must_use]
    pub fn span(self) -> usize {
        self.layout().span() as usize
    }
}

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
    #[must_use]
    pub fn text(&self) -> Option<&str> {
        match self {
            Value::Str(text) => Some(text),
            _ => None,
        }
    }

    /// 无符号整数视图。
    #[must_use]
    pub fn uint(&self) -> Option<u64> {
        match self {
            Value::Uint(value) => Some(*value),
            _ => None,
        }
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
                usize::try_from(dim).ok().and_then(|n| acc.checked_mul(n))
            })
            .ok_or_else(|| bad("张量形状溢出"))
    }

    /// 磁盘字节数；量化块必须整除元素数。
    pub fn bytes(&self) -> Result<usize> {
        let count = self.count()?;
        let elem = self.kind.elems();
        if count % elem != 0 {
            return Err(bad("量化块大小不整除元素数"));
        }
        (count / elem)
            .checked_mul(self.kind.span())
            .ok_or_else(|| bad("张量大小溢出"))
    }
}

/// 小端读取器，所有消费经边界检查。
struct Reader<'a> {
    bytes: &'a [u8],
    pos: usize,
    depth: usize,
}

impl<'a> Reader<'a> {
    fn new(bytes: &'a [u8]) -> Self {
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
        Ok(u32::from_le_bytes(raw.try_into().expect("4")))
    }

    fn wide(&mut self) -> Result<u64> {
        let raw = self.take(8)?;
        Ok(u64::from_le_bytes(raw.try_into().expect("8")))
    }

    fn text(&mut self) -> Result<String> {
        let len = usize::try_from(self.wide()?).map_err(|_| bad("字符串长度溢出"))?;
        let raw = self.take(len)?;
        String::from_utf8(raw.to_vec()).map_err(|_| bad("字符串不是 UTF-8"))
    }

    fn array_count(&mut self) -> Result<usize> {
        let count = usize::try_from(self.wide()?).map_err(|_| bad("数组长度溢出"))?;
        if count > self.bytes.len() - self.pos {
            return Err(bad("数组元素数超过剩余字节"));
        }
        Ok(count)
    }

    fn value(&mut self, kind: u32) -> Result<Value> {
        match kind {
            7 => Ok(Value::Bool(self.take(1)?[0] != 0)),
            8 => Ok(Value::Str(self.text()?)),
            9 => self.list(),
            _ => self.scalar(kind),
        }
    }

    fn list(&mut self) -> Result<Value> {
        if self.depth >= DEPTH {
            return Err(bad("数组嵌套过深"));
        }
        let kind = self.word()?;
        let count = self.array_count()?;
        self.depth += 1;
        let mut out = Vec::with_capacity(count.min(4096));
        for _ in 0..count {
            out.push(self.value(kind)?);
        }
        self.depth -= 1;
        Ok(Value::Arr(out))
    }

    fn scalar(&mut self, kind: u32) -> Result<Value> {
        let (width, class) = NUMBERS
            .get(kind as usize)
            .copied()
            .flatten()
            .ok_or_else(|| bad("未知值类型码"))?;
        let raw = self.take(width)?;
        let mut buf = [0u8; 8];
        buf[..width].copy_from_slice(raw);
        Ok(number(class, u64::from_le_bytes(buf), width))
    }
}

/// 值类型码 → (字节宽度, 0=无符号/1=有符号/2=浮点)。
const NUMBERS: [Option<(usize, u8)>; 13] = [
    Some((1, 0)),
    Some((1, 1)),
    Some((2, 0)),
    Some((2, 1)),
    Some((4, 0)),
    Some((4, 1)),
    Some((4, 2)),
    None,
    None,
    None,
    Some((8, 0)),
    Some((8, 1)),
    Some((8, 2)),
];

fn number(class: u8, bits: u64, width: usize) -> Value {
    let shift = 64 - width * 8;
    match class {
        0 => Value::Uint(bits),
        1 => Value::Int(((bits << shift) as i64) >> shift),
        _ => Value::Float(if width == 4 {
            f64::from(f32::from_bits(bits as u32))
        } else {
            f64::from_bits(bits)
        }),
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
        let magic = reader.take(4)?;
        if magic != MAGIC {
            return Err(bad("magic 不匹配"));
        }
        let version = reader.word()?;
        if version != VERSION {
            return Err(bad(&format!("版本 {version}，期望 {VERSION}")));
        }
        let tensor_count = reader.wide()?;
        let kv_count = reader.wide()?;

        let meta = read_kv(&mut reader, kv_count)?;
        let tensors = read_tensors(&mut reader, tensor_count)?;
        let align = alignment(&meta);
        let data = align_up(reader.pos, align)?;
        if data > bytes.len() {
            return Err(bad("数据区偏移超出文件"));
        }
        Ok(Self {
            bytes,
            meta,
            tensors,
            data,
        })
    }

    /// 按 key 取元数据。
    #[must_use]
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
    #[must_use]
    pub fn tensor(&self, name: &str) -> Option<&Tensor> {
        self.tensors.iter().find(|t| t.name == name)
    }

    /// `general.name`。
    #[must_use]
    pub fn name(&self) -> Option<&str> {
        self.meta("general.name").and_then(Value::text)
    }

    /// `general.architecture`。
    #[must_use]
    pub fn arch(&self) -> Option<&str> {
        self.meta("general.architecture").and_then(Value::text)
    }

    /// 张量原始字节视图。
    pub fn raw(&self, tensor: &Tensor) -> Result<&'a [u8]> {
        let offset = usize::try_from(tensor.offset).map_err(|_| bad("张量偏移溢出"))?;
        let start = self
            .data
            .checked_add(offset)
            .ok_or_else(|| bad("张量偏移溢出"))?;
        let len = tensor.bytes()?;
        self.bytes
            .get(start..start + len)
            .ok_or_else(|| bad("张量字节越界"))
    }

    /// 按名反量化为 f32。
    pub fn floats(&self, name: &str) -> Result<Vec<f32>> {
        let tensor = self
            .tensor(name)
            .ok_or_else(|| bad(&format!("无张量 `{name}`")))?;
        let raw = self.raw(tensor)?;
        let count = tensor.count()?;
        let mut out = vec![0.0f32; count];
        dequantize(tensor.kind, raw, &mut out)?;
        Ok(out)
    }
}

fn read_kv(reader: &mut Reader<'_>, count: u64) -> Result<Vec<(String, Value)>> {
    let count = usize::try_from(count).map_err(|_| bad("KV 计数溢出"))?;
    let mut out = Vec::with_capacity(count.min(4096));
    for _ in 0..count {
        let key = reader.text()?;
        let kind = reader.word()?;
        out.push((key, reader.value(kind)?));
    }
    Ok(out)
}

fn read_tensors(reader: &mut Reader<'_>, count: u64) -> Result<Vec<Tensor>> {
    let count = usize::try_from(count).map_err(|_| bad("张量计数溢出"))?;
    let mut out = Vec::with_capacity(count.min(4096));
    for _ in 0..count {
        let name = reader.text()?;
        let rank = reader.word()?;
        if rank > MAX_RANK {
            return Err(bad("张量维数超限"));
        }
        let mut shape = Vec::with_capacity(rank as usize);
        for _ in 0..rank {
            shape.push(reader.wide()?);
        }
        let kind = Kind::from_code(reader.word()?)?;
        let offset = reader.wide()?;
        out.push(Tensor {
            name,
            kind,
            shape,
            offset,
        });
    }
    Ok(out)
}

fn alignment(meta: &[(String, Value)]) -> usize {
    meta.iter()
        .find(|(key, _)| key == "general.alignment")
        .and_then(|(_, value)| value.uint())
        .and_then(|n| usize::try_from(n).ok())
        .filter(|&n| n > 0)
        .unwrap_or(DEFAULT_ALIGN)
}

fn align_up(pos: usize, align: usize) -> Result<usize> {
    let last = align.checked_sub(1).ok_or_else(|| bad("对齐为零"))?;
    pos.checked_add(last)
        .map(|rounded| rounded / align * align)
        .ok_or_else(|| bad("对齐溢出"))
}

// ------------------------------------------------ 反量化

fn dequantize(kind: Kind, raw: &[u8], out: &mut [f32]) -> Result<()> {
    let expect = kind.span() * (out.len() / kind.elems());
    if raw.len() != expect {
        return Err(bad("张量字节长度与块布局不符"));
    }
    match kind {
        Kind::F32 => decode_f32(raw, out),
        Kind::F16 => decode_f16(raw, out),
        Kind::Q8_0 => decode_q8(raw, out),
        Kind::Q4_K => decode_q4k(raw, out),
        Kind::Q5_K => decode_q5k(raw, out),
        Kind::Q6_K => decode_q6k(raw, out),
    }
}

fn decode_f32(raw: &[u8], out: &mut [f32]) -> Result<()> {
    let (chunks, tail) = raw.as_chunks::<4>();
    if !tail.is_empty() || chunks.len() != out.len() {
        return Err(bad("f32 长度不符"));
    }
    for (slot, block) in out.iter_mut().zip(chunks) {
        *slot = f32::from_le_bytes(*block);
    }
    Ok(())
}

fn decode_f16(raw: &[u8], out: &mut [f32]) -> Result<()> {
    let (chunks, tail) = raw.as_chunks::<2>();
    if !tail.is_empty() || chunks.len() != out.len() {
        return Err(bad("f16 长度不符"));
    }
    for (slot, block) in out.iter_mut().zip(chunks) {
        *slot = f16_to_f32(u16::from_le_bytes(*block));
    }
    Ok(())
}

/// 半精度位模式 → f32，含次正规、无穷与 NaN。
fn f16_to_f32(bits: u16) -> f32 {
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
        (scales[index] & 63, scales[index + 4] & 63)
    } else {
        let high = (scales[index - 4] >> 6) << 4;
        let low = (scales[index] >> 6) << 4;
        (
            (scales[index + 4] & 0x0F) | high,
            (scales[index + 4] >> 4) | low,
        )
    }
}

fn decode_q4k(raw: &[u8], out: &mut [f32]) -> Result<()> {
    let (blocks, _) = raw.as_chunks::<144>();
    let (slots, _) = out.as_chunks_mut::<256>();
    for (slot, block) in slots.iter_mut().zip(blocks) {
        let d = half_at(block, 0);
        let min = half_at(block, 2);
        let scales = &block[4..16];
        let nibbles = &block[16..];
        let (groups, _) = nibbles.as_chunks::<32>();
        for (group, qs) in groups.iter().enumerate() {
            let (s0, m0) = scale_min(scales, group * 2);
            let (s1, m1) = scale_min(scales, group * 2 + 1);
            let row = &mut slot[group * 64..];
            for lane in 0..32 {
                row[lane] = d * f32::from(s0) * f32::from(qs[lane] & 0x0F) - min * f32::from(m0);
                row[lane + 32] = d * f32::from(s1) * f32::from(qs[lane] >> 4) - min * f32::from(m1);
            }
        }
    }
    Ok(())
}

fn decode_q5k(raw: &[u8], out: &mut [f32]) -> Result<()> {
    let (blocks, _) = raw.as_chunks::<176>();
    let (slots, _) = out.as_chunks_mut::<256>();
    for (slot, block) in slots.iter_mut().zip(blocks) {
        let d = half_at(block, 0);
        let min = half_at(block, 2);
        let scales = &block[4..16];
        let qh = &block[16..48];
        let ql = &block[48..];
        let (groups, _) = ql.as_chunks::<32>();
        for (group, qs) in groups.iter().enumerate() {
            let (s0, m0) = scale_min(scales, group * 2);
            let (s1, m1) = scale_min(scales, group * 2 + 1);
            let bit0 = 1u8 << (group * 2);
            let bit1 = 1u8 << (group * 2 + 1);
            let row = &mut slot[group * 64..];
            for lane in 0..32 {
                let high0 = u8::from(qh[lane] & bit0 != 0) << 4;
                let high1 = u8::from(qh[lane] & bit1 != 0) << 4;
                row[lane] =
                    d * f32::from(s0) * f32::from((qs[lane] & 0x0F) + high0) - min * f32::from(m0);
                row[lane + 32] =
                    d * f32::from(s1) * f32::from((qs[lane] >> 4) + high1) - min * f32::from(m1);
            }
        }
    }
    Ok(())
}

fn six_bit(q: u8) -> f32 {
    (q as i32 - 32) as f32
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
                row[lane] = d * f32::from(scales[base] as i8) * six_bit(q1);
                row[lane + 32] = d * f32::from(scales[base + 2] as i8) * six_bit(q2);
                row[lane + 64] = d * f32::from(scales[base + 4] as i8) * six_bit(q3);
                row[lane + 96] = d * f32::from(scales[base + 6] as i8) * six_bit(q4);
            }
        }
    }
    Ok(())
}

fn bad(message: &str) -> Error {
    Error::new(Code::Invalid, message).with("gguf")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kind_maps_ggml_codes_and_delegates_layout() {
        // 锁定的码表：这 6 个码是当前唯一能解码的格式。
        const CODES: [(u32, Kind); 6] = [
            (0, Kind::F32),
            (1, Kind::F16),
            (8, Kind::Q8_0),
            (12, Kind::Q4_K),
            (13, Kind::Q5_K),
            (14, Kind::Q6_K),
        ];
        assert_eq!(SUPPORTED.len(), CODES.len(), "支持表与锁定码表必须同规模");
        for (code, kind) in CODES {
            assert_eq!(kind.code(), code);
            assert_eq!(Kind::from_code(code).unwrap(), kind);
            let hits = SUPPORTED.iter().filter(|(other, _)| *other == kind).count();
            assert_eq!(hits, 1, "{kind:?} 在 SUPPORTED 里必须恰好出现一次");
        }
        assert!(Kind::from_code(2).is_err(), "未支持的 ggml 码必须显式失败");
        assert!(Kind::from_code(15).is_err(), "未支持的 ggml 码必须显式失败");

        // 布局事实只有 dtype crate 一份：块格式取 QType，标量取 DType 的位宽。
        assert_eq!(Kind::F32.layout(), Layout::Scalar(DType::F32));
        assert_eq!(Kind::F16.layout(), Layout::Scalar(DType::F16));
        assert_eq!(Kind::Q8_0.layout(), Layout::Block(QType::Q8_0));
        assert_eq!(Kind::Q4_K.layout(), Layout::Block(QType::Q4_K));
        assert_eq!(Kind::Q5_K.layout(), Layout::Block(QType::Q5_K));
        assert_eq!(Kind::Q6_K.layout(), Layout::Block(QType::Q6_K));
        assert_eq!(Kind::Q8_0.elems(), QType::Q8_0.block_len() as usize);
        assert_eq!(Kind::Q8_0.span(), QType::Q8_0.block_bytes() as usize);
        assert_eq!(Kind::Q4_K.elems(), QType::Q4_K.block_len() as usize);
        assert_eq!(Kind::Q4_K.span(), QType::Q4_K.block_bytes() as usize);
        assert_eq!(Kind::F32.span(), DType::F32.bytes_for(1) as usize);
        assert_eq!(Kind::F16.span(), DType::F16.bytes_for(1) as usize);
    }

    #[test]
    fn undecoded_qtypes_are_explicit() {
        // dtype 声明 25 个块格式，本解码器只实现 4 个；差额是一份可读清单，不是隐式事实。
        let decoded: Vec<QType> = SUPPORTED
            .iter()
            .filter_map(|(_, layout)| match layout {
                Layout::Scalar(_) => None,
                Layout::Block(qtype) => Some(*qtype),
            })
            .collect();
        assert_eq!(
            decoded,
            vec![QType::Q8_0, QType::Q4_K, QType::Q5_K, QType::Q6_K]
        );
        assert_eq!(
            QType::ALL.len() - decoded.len(),
            21,
            "其余 QType 尚无解码器；接上时要同步这张表与反量化分支"
        );
        for qtype in &decoded {
            assert!(
                QType::ALL.contains(qtype),
                "{qtype:?} 必须在 dtype 的闭集里"
            );
        }
    }

    fn text(out: &mut Vec<u8>, value: &str) {
        out.extend_from_slice(&(value.len() as u64).to_le_bytes());
        out.extend_from_slice(value.as_bytes());
    }
    fn word(out: &mut Vec<u8>, value: u32) {
        out.extend_from_slice(&value.to_le_bytes());
    }
    fn wide(out: &mut Vec<u8>, value: u64) {
        out.extend_from_slice(&value.to_le_bytes());
    }
    fn key(out: &mut Vec<u8>, name: &str, kind: u32) {
        text(out, name);
        word(out, kind);
    }

    fn q8_block() -> Vec<u8> {
        let mut out = 0x3C00u16.to_le_bytes().to_vec();
        out.extend((0..32).map(|i| (i as i8) - 16).map(|q| q as u8));
        out
    }
    fn q4_block() -> Vec<u8> {
        let mut out = 0x3C00u16.to_le_bytes().to_vec();
        out.extend(0u16.to_le_bytes());
        out.extend([1u8; 12]);
        out.extend([0x21u8; 128]);
        out
    }
    fn q6_block() -> Vec<u8> {
        let mut out = vec![0x21u8; 128];
        out.extend([0u8; 64]);
        out.extend([1u8; 16]);
        out.extend(0x3C00u16.to_le_bytes());
        out
    }

    type Piece = (&'static str, Vec<u64>, Kind, Vec<u8>);
    fn pieces() -> Vec<Piece> {
        let f32s: Vec<u8> = [1.0f32, -0.5, 3.25, 0.0]
            .into_iter()
            .flat_map(f32::to_le_bytes)
            .collect();
        let halves: Vec<u8> = [0x3800u16, 0xC000, 0, 0x3E00]
            .into_iter()
            .flat_map(u16::to_le_bytes)
            .collect();
        vec![
            ("f32.weight", vec![2, 2], Kind::F32, f32s),
            ("half.weight", vec![4], Kind::F16, halves),
            ("q8.weight", vec![32], Kind::Q8_0, q8_block()),
            ("q4.weight", vec![256], Kind::Q4_K, q4_block()),
            ("q6.weight", vec![256], Kind::Q6_K, q6_block()),
        ]
    }

    fn build() -> Vec<u8> {
        let pieces = pieces();
        let mut head = Vec::new();
        head.extend_from_slice(&MAGIC);
        word(&mut head, VERSION);
        wide(&mut head, pieces.len() as u64);
        wide(&mut head, 4);
        key(&mut head, "general.name", 8);
        text(&mut head, "fixture");
        key(&mut head, "general.architecture", 8);
        text(&mut head, "llama");
        key(&mut head, "general.alignment", 4);
        word(&mut head, DEFAULT_ALIGN as u32);
        key(&mut head, "llama.block_count", 4);
        word(&mut head, 1);

        let mut offset = 0u64;
        let mut infos = Vec::new();
        let mut body = Vec::new();
        for (name, shape, kind, bytes) in &pieces {
            offset = offset.div_ceil(DEFAULT_ALIGN as u64) * DEFAULT_ALIGN as u64;
            text(&mut infos, name);
            word(&mut infos, shape.len() as u32);
            for dim in shape {
                wide(&mut infos, *dim);
            }
            word(&mut infos, kind.code());
            wide(&mut infos, offset);
            body.extend(std::iter::repeat_n(0, offset as usize - body.len()));
            body.extend_from_slice(bytes);
            offset += bytes.len() as u64;
        }
        head.extend(infos);
        let fill = (DEFAULT_ALIGN - head.len() % DEFAULT_ALIGN) % DEFAULT_ALIGN;
        head.extend(std::iter::repeat_n(0, fill));
        head.extend(body);
        head
    }

    #[test]
    fn half_specials() {
        assert_eq!(f16_to_f32(0x0001), 5.960_464_5e-8);
        assert!(f16_to_f32(0xFC00).is_infinite() && f16_to_f32(0xFC00) < 0.0);
        assert!(f16_to_f32(0x7C01).is_nan());
        assert_eq!(f16_to_f32(0), 0.0);
    }

    #[test]
    fn reads_header_and_metadata() {
        let raw = build();
        let file = Gguf::parse(&raw).unwrap();
        assert_eq!(file.name(), Some("fixture"));
        assert_eq!(file.arch(), Some("llama"));
        assert_eq!(file.metas().len(), 4);
        assert_eq!(file.tensors().len(), 5);
        assert_eq!(file.tensor("q4.weight").unwrap().kind, Kind::Q4_K);
    }

    #[test]
    fn decodes_plain_types() {
        let raw = build();
        let file = Gguf::parse(&raw).unwrap();
        assert_eq!(
            file.floats("f32.weight").unwrap(),
            vec![1.0, -0.5, 3.25, 0.0]
        );
        assert_eq!(
            file.floats("half.weight").unwrap(),
            vec![0.5, -2.0, 0.0, 1.5]
        );
        assert_eq!(
            file.floats("q8.weight").unwrap(),
            (0..32).map(|i| i as f32 - 16.0).collect::<Vec<_>>()
        );
    }

    #[test]
    fn decodes_q4k_and_q6k() {
        let raw = build();
        let file = Gguf::parse(&raw).unwrap();
        let q4 = file.floats("q4.weight").unwrap();
        assert_eq!(q4.len(), 256);
        let q6 = file.floats("q6.weight").unwrap();
        assert_eq!(q6.len(), 256);
    }

    #[test]
    fn rejects_bad_magic_and_overflow() {
        let mut raw = build();
        raw[0] = b'X';
        assert!(Gguf::parse(&raw).is_err());
        let tensor = Tensor {
            name: "huge".into(),
            kind: Kind::F32,
            shape: vec![1u64 << 62],
            offset: 0,
        };
        assert!(tensor.bytes().is_err());
    }
}
