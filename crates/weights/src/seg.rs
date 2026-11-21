//! 段：一个权重张量的名字、类型、形状、可选量化方案与它的逻辑字节。
//!
//! 描述符走紧凑 LEB128；数据字节本身在数据区单独写。P1 只有 `store` 一种 codec（code 0），
//! 因此 `data_len` 必须正好等于形状与类型推出的字节数。

use crate::bad;
use diagnostic::{Code, Error, Result};

use crate::seg_decode::{check_frame, check_quant, decode_meta, decode_quant, decode_tail};

use crate::Wire;
use crate::leb::{self, Cursor};
use crate::quant::Quant;
#[cfg(test)]
pub(crate) use crate::seg_decode::{MAX_RANK, decode_shape};
use crate::sink::Sink;

/// 段级编解码闭集：码位稳定，未知码在打开期拒绝。
///
/// - [`Codec::Store`]：原样存储，读端零拷贝；
/// - [`Codec::Rans`]：`rans` 熵编码帧，仅在严格变小时才会被写端选用；
/// - [`Codec::Axes`]：五轴无损选码帧（仅 f32 二维稠密权重），逐位还原且严格变小才选用。
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

    /// 编码引用描述符：只记代表段下标，不写 codec/长度/CRC；存储帧与校验都归代表段。
    pub(crate) fn encode_ref_desc(&self, out: &mut impl Sink, rep: u32) -> Result<()> {
        out.put(&[FORM_REF]);
        self.encode_head(out)?;
        match &self.desc.quant {
            None => out.put(&[0]),
            Some(quant) => {
                out.put(&[1]);
                quant.encode(out, self.desc.shape.len() as u32)?;
            }
        }
        leb::write(out, rep);
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

#[cfg(test)]
#[path = "../tests/seg.rs"]
mod tests;
