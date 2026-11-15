//! 段：一个权重张量的名字、类型、形状、可选量化方案与它的逻辑字节。
//!
//! 描述符走紧凑 LEB128；数据字节本身在数据区单独写。P1 只有 `store` 一种 codec（code 0），
//! 因此 `data_len` 必须正好等于形状与类型推出的字节数。

use diagnostic::{Code, Error, Result};

use crate::Wire;
use crate::leb::{self, Cursor};
use crate::quant::Quant;
use crate::sink::Sink;

/// 段级编解码闭集：码位稳定，未知码在打开期拒绝。
///
/// - [`Codec::Store`]：原样存储，读端零拷贝；
/// - [`Codec::Rans`]：`rans` 熵编码帧，仅在严格变小时才会被写端选用。
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Codec {
    /// 原样字节（code 0）。
    Store = 0,
    /// rANS 熵编码帧（code 1）。
    Rans = 1,
}

impl Codec {
    /// 线上码位。
    pub(crate) fn code(self) -> u8 {
        self as u8
    }

    /// 从码位解析；未知码显式报错而非静默回退。
    pub(crate) fn from_code(code: u8) -> Result<Self> {
        match code {
            0 => Ok(Self::Store),
            1 => Ok(Self::Rans),
            other => Err(
                Error::new(Code::Unsupported, format!("未知段 codec {other}")).with("weights::seg"),
            ),
        }
    }
}

/// 段元数据：名字、类型、形状与可选量化。写端段、解码描述符与读视图三份共用这一份，
/// 避免同一组字段在三处各自漂移。
#[derive(Clone, PartialEq, Eq, Debug)]
pub(crate) struct Descriptor {
    pub(crate) name: String,
    pub(crate) dtype: Wire,
    pub(crate) shape: Vec<u64>,
    pub(crate) quant: Option<Quant>,
}

impl Descriptor {
    /// 元素个数（乘积，溢出饱和）。
    fn numel(&self) -> u64 {
        numel(&self.shape)
    }

    /// 形状推出的字节数。
    fn expected_bytes(&self) -> u64 {
        self.dtype.bytes_for(self.numel())
    }

    /// 段名非空，rank 可表示；不回答字节长度。
    fn rank(&self) -> Result<u32> {
        if self.name.is_empty() {
            return Err(bad("段名不能为空"));
        }
        u32::try_from(self.shape.len()).map_err(|_| bad("rank 超出 u32"))
    }
}

/// 一个权重段（写端的输入）。
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Segment {
    desc: Descriptor,
    /// 逻辑字节。
    pub data: Vec<u8>,
}

impl Segment {
    /// 构造并校验：名字非空，字节长度与形状/类型自洽，量化方案与 rank 一致。
    pub fn new(
        name: impl Into<String>,
        dtype: Wire,
        shape: Vec<u64>,
        quant: Option<Quant>,
        data: Vec<u8>,
    ) -> Result<Self> {
        let segment = Self {
            desc: Descriptor {
                name: name.into(),
                dtype,
                shape,
                quant,
            },
            data,
        };
        segment.validate()?;
        Ok(segment)
    }

    /// 权重名。
    pub fn name(&self) -> &str {
        &self.desc.name
    }

    /// 元素类型。
    pub fn dtype(&self) -> Wire {
        self.desc.dtype
    }

    /// 形状（各轴长度）。
    pub fn shape(&self) -> &[u64] {
        &self.desc.shape
    }

    /// 可选量化方案。
    pub fn quant(&self) -> Option<&Quant> {
        self.desc.quant.as_ref()
    }

    /// 元素个数（乘积，溢出饱和）。
    pub fn numel(&self) -> u64 {
        self.desc.numel()
    }

    /// 形状推出的字节数。
    pub fn expected_bytes(&self) -> u64 {
        self.desc.expected_bytes()
    }

    /// 自洽校验。
    ///
    /// 逐轴没有可判的东西：轴长度允许 0（零元素张量），`u64` 本来就非负，越界形状由 `numel` 的
    /// 饱和乘与下面的字节数交叉校验兜住。真正有主体的只有三条：段名、rank 与数据长度。
    pub fn validate(&self) -> Result<()> {
        let rank = self.desc.rank()?;
        self.check_len()?;
        check_quant(self.desc.quant.as_ref(), rank)
    }

    /// 数据字节数必须等于类型 × 元素个数。
    fn check_len(&self) -> Result<()> {
        let actual = u64::try_from(self.data.len()).map_err(|_| bad("段长度超 u64"))?;
        let expected = self.desc.expected_bytes();
        if actual == expected {
            return Ok(());
        }
        Err(bad(&format!(
            "段 `{}` 的数据是 {} 字节，{:?} {:?} 推出 {expected} 字节",
            self.desc.name,
            self.data.len(),
            self.desc.dtype,
            self.desc.shape
        )))
    }

    /// 编码描述符（不含数据字节）；`stored_len` 是实际写入数据区的字节数（可能已压缩）。
    pub(crate) fn encode_desc(
        &self,
        out: &mut impl Sink,
        codec: Codec,
        stored_len: u64,
        seg_crc: u32,
    ) -> Result<()> {
        let name = self.desc.name.as_bytes();
        leb::write(out, name.len());
        out.put(name);
        leb::write(out, self.desc.dtype.code());
        leb::write(out, self.desc.shape.len());
        for axis in &self.desc.shape {
            leb::write(out, *axis);
        }
        out.put(&[codec.code()]);
        match &self.desc.quant {
            None => out.put(&[0]),
            Some(quant) => {
                out.put(&[1]);
                quant.encode(out, self.desc.shape.len() as u32)?;
            }
        }
        leb::write(out, stored_len);
        leb::write(out, seg_crc);
        Ok(())
    }
}

/// 解码后的描述符（数据区间在解析数据区时再确定）。
#[derive(Debug)]
pub(crate) struct Desc {
    desc: Descriptor,
    codec: Codec,
    data_len: u64,
    seg_crc: u32,
}

impl Desc {
    pub(crate) fn decode(cur: &mut Cursor<'_>) -> Result<Self> {
        let (name, dtype) = decode_head(cur)?;
        let (shape, rank, codec) = decode_body(cur)?;
        let (quant, data_len, seg_crc) = decode_tail(cur, rank)?;
        check_frame(&name, codec, data_len, dtype.bytes_for(numel(&shape)))?;
        Ok(Self {
            desc: Descriptor {
                name,
                dtype,
                shape,
                quant,
            },
            codec,
            data_len,
            seg_crc,
        })
    }

    pub(crate) fn into_parts(self) -> (Descriptor, Codec, u64, u32) {
        (self.desc, self.codec, self.data_len, self.seg_crc)
    }
}

/// 段名：LEB 长度 + UTF-8 + 非空。
fn decode_name(cur: &mut Cursor<'_>) -> Result<String> {
    let len = cur.read::<u64>()?;
    let name = name_bytes(cur, len)?;
    if name.is_empty() {
        return Err(bad("段名不能为空"));
    }
    Ok(name.to_owned())
}

/// 名字字节：长度可表示且是 UTF-8。
fn name_bytes<'c>(cur: &mut Cursor<'c>, len: u64) -> Result<&'c str> {
    let len = usize::try_from(len).map_err(|_| bad("名字长度超 usize"))?;
    std::str::from_utf8(cur.take(len)?).map_err(|_| bad("段名不是 UTF-8"))
}

/// 段名与元素类型；两者都先于形状出现。
fn decode_head(cur: &mut Cursor<'_>) -> Result<(String, Wire)> {
    let name = decode_name(cur)?;
    Ok((name, read_wire(cur)?))
}

/// 形状与 codec：rank 轴长度，随后一个 codec 码。
fn decode_body(cur: &mut Cursor<'_>) -> Result<(Vec<u64>, u32, Codec)> {
    let rank = cur.read::<u32>()?;
    let shape = decode_shape(cur, rank)?;
    Ok((shape, rank, Codec::from_code(cur.read_u8()?)?))
}

/// `rank` 个轴长度。
fn decode_shape(cur: &mut Cursor<'_>, rank: u32) -> Result<Vec<u64>> {
    let capacity = usize::try_from(rank).map_err(|_| bad("rank 超 usize"))?;
    let mut shape = Vec::with_capacity(capacity);
    for _ in 0..rank {
        shape.push(cur.read::<u64>()?);
    }
    Ok(shape)
}

/// 量化方案与长度字段；`rank` 用于校验量化轴。
fn decode_tail(cur: &mut Cursor<'_>, rank: u32) -> Result<(Option<Quant>, u64, u32)> {
    let quant = decode_quant(cur, rank)?;
    Ok((quant, cur.read::<u64>()?, cur.read::<u32>()?))
}

/// 量化方案：present 标志只能是 0/1。
fn decode_quant(cur: &mut Cursor<'_>, rank: u32) -> Result<Option<Quant>> {
    match cur.read_u8()? {
        0 => Ok(None),
        1 => Ok(Some(Quant::decode(cur, rank)?)),
        other => Err(bad(&format!("quant present 标志只能是 0/1，实际 {other}"))),
    }
}

/// 存储帧长度必须与 codec 的口径一致：store 恰好相等，rans 严格更小。
fn check_frame(name: &str, codec: Codec, data_len: u64, expected: u64) -> Result<()> {
    match codec {
        Codec::Store if data_len != expected => Err(bad(&format!(
            "段 `{name}` 声明 {data_len} 字节，store 段应为 {expected} 字节"
        ))),
        Codec::Rans if data_len >= expected => Err(bad(&format!(
            "段 `{name}` 的 rans 帧 {data_len} 字节未小于原始 {expected} 字节"
        ))),
        _ => Ok(()),
    }
}

/// 量化轴不得超出 rank；没有量化方案时无事可做。
fn check_quant(quant: Option<&Quant>, rank: u32) -> Result<()> {
    if let Some(quant) = quant
        && quant.axis > rank
    {
        return Err(bad("量化轴超出 rank"));
    }
    Ok(())
}

/// 形状的元素乘积，溢出饱和。
pub(crate) fn numel(shape: &[u64]) -> u64 {
    shape.iter().fold(1u64, |acc, dim| acc.saturating_mul(*dim))
}

fn read_wire(cur: &mut Cursor<'_>) -> Result<Wire> {
    let code = cur.read::<u32>()?;
    Wire::from_code(u16::try_from(code).map_err(|_| bad("wire code 超 u16"))?)
        .ok_or_else(|| bad("未知 wire code"))
}

fn bad(message: &str) -> Error {
    Error::new(Code::Invalid, message).with("weights::seg")
}
