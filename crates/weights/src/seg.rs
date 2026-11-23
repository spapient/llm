//! 段：权重张量的元数据、存储帧与容器帧解析。
//!
//! 段元数据（`Descriptor`/`Segment`）与容器帧本是同一件事的两半：容器帧的每条描述符就是一段
//! 的元数据，落位结果就是 `Stored`。两者原先互相 import，现在合成一个模块。

use crate::bad;
use diagnostic::{Code, Error, Result};

use crate::quant::Quant;
use crate::Wire;
use std::borrow::Cow;
use wire::leb::{self, Cursor};
use wire::sink::Sink;

/// 段级编解码闭集：码位稳定，未知码在打开期拒绝。
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Codec {
    /// 原样字节（code 0）。
    Store = 0,
    /// rANS 熵编码帧（code 1）。
    Rans = 1,
    /// 五轴无损选码帧（code 2）。
    Axes = 2,
}

impl Codec {
    /// 线上码位。
    pub(crate) fn code(self) -> u8 {
        self as u8
    }

    /// 段体解码：codec 码位与解码器的对应只此一处。
    pub(crate) fn decode<'a>(self, raw: &'a [u8], want: usize) -> Result<Cow<'a, [u8]>> {
        Ok(match self {
            Self::Store => Cow::Borrowed(raw),
            Self::Rans => Cow::Owned(crate::rans::decode(raw, want)?),
            Self::Axes => Cow::Owned(crate::axes::expand(raw)?),
        })
    }

    /// 从码位解析；未知码显式报错而非静默回退。
    pub(crate) fn from_code(code: u8) -> Result<Self> {
        match code {
            0 => Ok(Self::Store),
            1 => Ok(Self::Rans),
            2 => Ok(Self::Axes),
            other => Err(
                Error::new(Code::Unsupported, format!("未知段 codec {other}")).with("weights::seg"),
            ),
        }
    }
}

/// 描述符形态：内联数据，数据区写该段自己的存储帧。
const FORM_INLINE: u8 = 0;
/// 描述符形态：引用代表段，数据区不重复写字节，段体解引用到被引用的内联段。
const FORM_REF: u8 = 1;

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
        tensor::product(&self.shape)
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

    /// 引用段与代表段的取值元数据必须一致：类型、形状、量化方案同名同构；段名允许不同
    ///（两个条目共享一份字节），因此名字不参与比对。
    pub(crate) fn same_value(&self, other: &Descriptor) -> bool {
        self.dtype == other.dtype && self.shape == other.shape && self.quant == other.quant
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

    /// 取值元数据是否与另一篇一致（类型/形状/量化）；名字不参与，引用段与代表段名字本就不同。
    pub(crate) fn same_value(&self, other: &Segment) -> bool {
        self.desc.same_value(&other.desc)
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
        Err(bad(format!(
            "段 `{}` 的数据是 {} 字节，{:?} {:?} 推出 {expected} 字节",
            self.desc.name,
            self.data.len(),
            self.desc.dtype,
            self.desc.shape
        )))
    }

    /// 编码内联描述符（不含数据字节）；`stored_len` 是实际写入数据区的字节数（可能已压缩）。
    pub(crate) fn encode_desc(
        &self,
        out: &mut impl Sink,
        codec: Codec,
        stored_len: u64,
        seg_crc: u32,
    ) -> Result<()> {
        out.put(&[FORM_INLINE]);
        self.encode_head(out)?;
        out.put(&[codec.code()]);
        self.encode_quant(out)?;
        leb::write(out, stored_len);
        leb::write(out, seg_crc);
        Ok(())
    }

    /// 编码引用描述符：只记代表段下标，不写 codec/长度/CRC；存储帧与校验都归代表段。
    pub(crate) fn encode_ref_desc(&self, out: &mut impl Sink, rep: u32) -> Result<()> {
        out.put(&[FORM_REF]);
        self.encode_head(out)?;
        self.encode_quant(out)?;
        leb::write(out, rep);
        Ok(())
    }

    /// 量化方案编码：存在标志后接方案本身；内联与引用两种描述符共用这一份。
    fn encode_quant(&self, out: &mut impl Sink) -> Result<()> {
        match &self.desc.quant {
            None => out.put(&[0]),
            Some(quant) => {
                out.put(&[1]);
                quant.encode(out, self.desc.shape.len() as u32)?;
            }
        }
        Ok(())
    }

    /// 描述符公共头部：名字、类型、形状。内联与引用两种形态共用这一份。
    fn encode_head(&self, out: &mut impl Sink) -> Result<()> {
        let name = self.desc.name.as_bytes();
        leb::write(out, name.len());
        out.put(name);
        leb::write(out, self.desc.dtype.code());
        leb::write(out, self.desc.shape.len());
        for axis in &self.desc.shape {
            leb::write(out, *axis);
        }
        Ok(())
    }
}

/// 解码后的描述符体：内联段带自己的存储帧与段级 CRC；引用段只记代表段下标。
#[derive(Debug)]
pub(crate) enum DescBody {
    /// 内联段：存储 codec、帧长与段级 CRC。
    Inline {
        codec: Codec,
        data_len: u64,
        seg_crc: u32,
    },
    /// 引用段：代表段在描述符表中的下标。
    Ref { rep: u32 },
}

/// 解码后的描述符（数据区间在解析数据区时再确定）。
#[derive(Debug)]
pub(crate) struct Desc {
    desc: Descriptor,
    body: DescBody,
}

impl Desc {
    pub(crate) fn decode(cur: &mut Cursor<'_>) -> Result<Self> {
        let form = cur.read_u8()?;
        let (desc, rank) = decode_meta(cur)?;
        Self::decode_body(cur, form, desc, rank)
    }

    /// 按描述符形态分派到内联或引用解码；未知形态显式拒绝。
    fn decode_body(cur: &mut Cursor<'_>, form: u8, desc: Descriptor, rank: u32) -> Result<Self> {
        match form {
            FORM_INLINE => Self::decode_inline(cur, desc, rank),
            FORM_REF => Self::decode_ref(cur, desc, rank),
            other => Err(bad(format!("未知描述符形态 {other}"))),
        }
    }

    /// 内联描述符：codec、量化方案、帧长与段级 CRC；帧长口径按 codec 校验。
    fn decode_inline(cur: &mut Cursor<'_>, mut desc: Descriptor, rank: u32) -> Result<Self> {
        let codec = Codec::from_code(cur.read_u8()?)?;
        let (quant, data_len, seg_crc) = decode_tail(cur, rank)?;
        check_frame(
            &desc.name,
            codec,
            data_len,
            desc.dtype.bytes_for(tensor::product(&desc.shape)),
        )?;
        desc.quant = quant;
        Ok(Self {
            desc,
            body: DescBody::Inline {
                codec,
                data_len,
                seg_crc,
            },
        })
    }

    /// 引用描述符：量化方案 + 代表段下标；无 codec/帧长/CRC 字段。
    fn decode_ref(cur: &mut Cursor<'_>, mut desc: Descriptor, rank: u32) -> Result<Self> {
        let quant = decode_quant(cur, rank)?;
        let rep = cur.read::<u32>()?;
        desc.quant = quant;
        Ok(Self {
            desc,
            body: DescBody::Ref { rep },
        })
    }

    /// 描述符元数据视图。
    pub(crate) fn descriptor(&self) -> &Descriptor {
        &self.desc
    }

    /// 描述符体（内联帧或引用下标）。
    pub(crate) fn body(&self) -> &DescBody {
        &self.body
    }
}

// —— 已解析段条目（原 `stored` 平铺进 `seg`）：共享描述符 + 重算出的绝对字节区间 ——

/// 一个已解析的段：共享描述符 + 重算出的绝对字节区间。
#[derive(Debug, Clone)]
pub struct Stored {
    pub(crate) desc: Descriptor,
    pub(crate) inner: StoredInner,
}

/// 段的存储形态：内联帧在本段名下；引用段解引用到代表段，区间/编码/CRC 在布局时复制自代表段。
#[derive(Debug, Clone)]
pub(crate) enum StoredInner {
    Inline {
        codec: Codec,
        start: usize,
        len: usize,
        crc: u32,
    },
    Ref {
        rep: usize,
        codec: Codec,
        start: usize,
        len: usize,
        crc: u32,
    },
}

impl Stored {
    /// 段名。
    pub fn name(&self) -> &str {
        &self.desc.name
    }

    /// 存储帧的取值视图（codec、绝对起点、帧长、段级 CRC）；内联与引用两种形态共用一次解构。
    fn span(&self) -> (Codec, usize, usize, u32) {
        match &self.inner {
            StoredInner::Inline {
                codec,
                start,
                len,
                crc,
            }
            | StoredInner::Ref {
                codec,
                start,
                len,
                crc,
                ..
            } => (*codec, *start, *len, *crc),
        }
    }

    /// 存储编解码方式；引用段沿用代表段的 codec。
    pub fn codec(&self) -> Codec {
        self.span().0
    }

    /// 元素类型。
    pub fn dtype(&self) -> Wire {
        self.desc.dtype
    }

    /// 形状。
    pub fn shape(&self) -> &[u64] {
        &self.desc.shape
    }

    /// 量化方案（若有）。
    pub fn quant(&self) -> Option<&Quant> {
        self.desc.quant.as_ref()
    }

    /// 数据起点；引用段沿用代表段的区间。
    pub fn start(&self) -> usize {
        self.span().1
    }

    /// 存储字节数；引用段沿用代表段的帧长。
    pub fn len(&self) -> usize {
        self.span().2
    }

    /// 是否零字节。
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// 是否为引用段（共享代表段字节，自身数据区不占帧）。
    pub fn is_ref(&self) -> bool {
        matches!(self.inner, StoredInner::Ref { .. })
    }

    /// 代表段下标；内联段返回 `None`。
    pub fn rep_idx(&self) -> Option<usize> {
        match &self.inner {
            StoredInner::Ref { rep, .. } => Some(*rep),
            StoredInner::Inline { .. } => None,
        }
    }

    /// 逻辑字节数：元素类型 × 元素个数，也是 `rans::decode` 的分配上限（来自描述符而非帧）。
    pub fn logical_len(&self) -> Result<usize> {
        usize::try_from(self.desc.dtype.bytes_for(tensor::product(&self.desc.shape)))
            .map_err(|_| bad("段的逻辑长度超 usize"))
    }

    /// 段级 CRC；引用段复制自代表段，不另存校验字段。
    pub(crate) fn crc(&self) -> u32 {
        self.span().3
    }
}

/// 单段是否无损：不带量化方案，且 codec 属于无损的 `Store`/`Rans`/`Axes`。
pub(crate) fn lossless_entry(seg: &Stored) -> bool {
    seg.quant().is_none() && matches!(seg.codec(), Codec::Store | Codec::Rans | Codec::Axes)
}

#[path = "seg_layout.rs"]
pub(crate) mod seg_layout;

pub(crate) use seg_layout::{fixed, layout, read_descs, FOOTER_LEN};


// —— 描述符解码路径：帧头、形状、量化参数与三条长度判据 ——

/// 描述符元数据：名字/精度/形状先读成中性描述符，形态分支再决定尾部字段。
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

#[cfg(test)]
#[path = "../tests/seg.rs"]
mod tests;
