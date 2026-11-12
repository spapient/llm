//! 段：一个权重张量的名字、类型、形状、可选量化方案与它的逻辑字节。
//!
//! 描述符走紧凑 LEB128；数据字节本身在数据区单独写。P1 只有 `store` 一种 codec（code 0），
//! 因此 `data_len` 必须正好等于形状与类型推出的字节数。

use diagnostic::{Code, Error, Result};

use crate::dtype::Wire;
use crate::leb::{self, Cursor};
use crate::quant::Quant;

/// 一个权重段（写端的输入）。
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Segment {
    /// 权重名（非空）。
    pub name: String,
    /// 元素类型。
    pub dtype: Wire,
    /// 形状（各轴长度）。
    pub shape: Vec<u64>,
    /// 可选量化方案。
    pub quant: Option<Quant>,
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
            name: name.into(),
            dtype,
            shape,
            quant,
            data,
        };
        segment.validate()?;
        Ok(segment)
    }

    /// 元素个数（乘积，溢出饱和）。
    pub fn numel(&self) -> u64 {
        numel(&self.shape)
    }

    /// 形状推出的字节数。
    pub fn expected_bytes(&self) -> u64 {
        self.dtype.element_bytes(self.numel())
    }

    /// 自洽校验。
    pub fn validate(&self) -> Result<()> {
        if self.name.is_empty() {
            return Err(bad("段名不能为空"));
        }
        let rank = u32::try_from(self.shape.len()).map_err(|_| bad("rank 超出 u32"))?;
        for axis in &self.shape {
            // 轴长度允许 0（零元素张量），不允许负数（u64 本来就非负）。
            let _ = axis;
        }
        let expected = self.expected_bytes();
        if u64::try_from(self.data.len()).expect("usize <= u64") != expected {
            return Err(bad(&format!(
                "段 `{}` 的数据是 {} 字节，{:?} {:?} 推出 {expected} 字节",
                self.name,
                self.data.len(),
                self.dtype,
                self.shape
            )));
        }
        if let Some(quant) = &self.quant
            && quant.axis > rank
        {
            return Err(bad("量化轴超出 rank"));
        }
        Ok(())
    }

    /// 编码描述符（不含数据字节）。
    pub(crate) fn encode_desc(&self, buf: &mut Vec<u8>, seg_crc: u32) -> Result<()> {
        let name = self.name.as_bytes();
        leb::write_u64(buf, name.len() as u64);
        buf.extend_from_slice(name);
        leb::write_u32(buf, u32::from(self.dtype.code()));
        leb::write_u32(buf, self.shape.len() as u32);
        for axis in &self.shape {
            leb::write_u64(buf, *axis);
        }
        buf.push(0); // codec = store
        match &self.quant {
            None => buf.push(0),
            Some(quant) => {
                buf.push(1);
                quant.encode(buf, self.shape.len() as u32)?;
            }
        }
        leb::write_u64(buf, self.data.len() as u64);
        leb::write_u32(buf, seg_crc);
        Ok(())
    }
}

/// 解码后的描述符（数据区间在解析数据区时再确定）。
#[derive(Debug)]
pub(crate) struct Desc {
    name: String,
    dtype: Wire,
    shape: Vec<u64>,
    quant: Option<Quant>,
    data_len: u64,
    seg_crc: u32,
}

impl Desc {
    pub(crate) fn decode(cur: &mut Cursor<'_>) -> Result<Self> {
        let name_len = cur.read_u64()?;
        let name_bytes =
            cur.take(usize::try_from(name_len).map_err(|_| bad("名字长度超 usize"))?)?;
        let name = std::str::from_utf8(name_bytes)
            .map_err(|_| bad("段名不是 UTF-8"))?
            .to_owned();
        if name.is_empty() {
            return Err(bad("段名不能为空"));
        }
        let dtype = read_wire(cur)?;
        let rank = cur.read_u32()?;
        let mut shape = Vec::with_capacity(usize::try_from(rank).expect("u32 -> usize"));
        for _ in 0..rank {
            shape.push(cur.read_u64()?);
        }
        let codec = cur.read_u8()?;
        if codec != 0 {
            return Err(Error::new(
                Code::Unsupported,
                format!("段 codec {codec} 尚未实现，P1 只有 store(0)"),
            )
            .with("pack::seg"));
        }
        let present = cur.read_u8()?;
        let quant = match present {
            0 => None,
            1 => Some(Quant::decode(cur, rank)?),
            other => return Err(bad(&format!("quant present 标志只能是 0/1，实际 {other}"))),
        };
        let data_len = cur.read_u64()?;
        let seg_crc = cur.read_u32()?;

        let expected = dtype.element_bytes(numel(&shape));
        if data_len != expected {
            return Err(bad(&format!(
                "段 `{name}` 声明 {data_len} 字节，形状推出 {expected} 字节"
            )));
        }
        Ok(Self {
            name,
            dtype,
            shape,
            quant,
            data_len,
            seg_crc,
        })
    }

    pub(crate) fn into_parts(self) -> (String, Wire, Vec<u64>, Option<Quant>, u64, u32) {
        (
            self.name,
            self.dtype,
            self.shape,
            self.quant,
            self.data_len,
            self.seg_crc,
        )
    }
}

/// 形状的元素乘积，溢出饱和。
pub(crate) fn numel(shape: &[u64]) -> u64 {
    shape.iter().fold(1u64, |acc, dim| acc.saturating_mul(*dim))
}

fn read_wire(cur: &mut Cursor<'_>) -> Result<Wire> {
    let code = cur.read_u32()?;
    Wire::from_code(u16::try_from(code).map_err(|_| bad("wire code 超 u16"))?)
        .ok_or_else(|| bad("未知 wire code"))
}

fn bad(message: &str) -> Error {
    Error::new(Code::Invalid, message).with("pack::seg")
}
