//! 读端：解析并自证合法——固定头、描述符、对齐填充、两级 CRC，一处都不放宽。
//!
//! 段的绝对偏移不进文件，由"数据区起点 + 对齐规则 + 各段长度"顺序重算；重算出的区间再核对
//! 填充为 0、段级 CRC 与全局 CRC。截断、改一个比特、尾部多余字节、非零填充都在打开期失败。

use std::borrow::Cow;

use diagnostic::{Code, Error, Result};

use crate::affine::Quant;
use crate::codec::Codec;
use crate::dtype::Wire;
use crate::leb::Cursor;
use crate::merkle::Proof;
use crate::rans;
use crate::seg::Desc;
use crate::sha256::Digest;
use crate::writer::{GLOBAL_CRC, MAGIC, MAX_ALIGN_LOG, SEG_CRC, VERSION};

/// 固定头长度：magic 4 + version 2 + flags 4 + align 1（count 是 LEB，不计入）。
const FIXED_HEAD: usize = 11;

/// 尾部全局 CRC 长度。
const FOOTER_LEN: usize = 4;

/// 一个已解析的段：描述符 + 重算出的绝对字节区间。
#[derive(Debug, Clone)]
pub struct Stored {
    name: String,
    dtype: Wire,
    shape: Vec<u64>,
    quant: Option<Quant>,
    codec: Codec,
    start: usize,
    len: usize,
    crc: u32,
}

impl Stored {
    /// 段名。
    pub fn name(&self) -> &str {
        &self.name
    }

    /// 存储编解码方式。
    pub fn codec(&self) -> Codec {
        self.codec
    }

    /// 元素类型。
    pub fn dtype(&self) -> Wire {
        self.dtype
    }

    /// 形状。
    pub fn shape(&self) -> &[u64] {
        &self.shape
    }

    /// 量化方案（若有）。
    pub fn quant(&self) -> Option<&Quant> {
        self.quant.as_ref()
    }

    /// 数据起点。
    pub fn start(&self) -> usize {
        self.start
    }

    /// 存储字节数。
    pub fn len(&self) -> usize {
        self.len
    }

    /// 是否零字节。
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

/// 内存容器：绑定在输入字节上，不复制数据。
#[derive(Debug)]
pub struct Archive<'a> {
    bytes: &'a [u8],
    segs: Vec<Stored>,
    flags: u32,
}

impl<'a> Archive<'a> {
    /// 解析并完整校验。
    pub fn parse(bytes: &'a [u8]) -> Result<Self> {
        if bytes.len() < FIXED_HEAD + FOOTER_LEN + 1 {
            return Err(invalid(format!(
                "文件只有 {} 字节，读不出固定头、段计数与尾部",
                bytes.len()
            )));
        }
        let magic = u32::from_le_bytes(bytes[0..4].try_into().expect("4 字节"));
        if magic != MAGIC {
            return Err(invalid(format!(
                "魔数不匹配：期望 {MAGIC:#010x}，得到 {magic:#010x}"
            )));
        }
        let version = u16::from_le_bytes(bytes[4..6].try_into().expect("2 字节"));
        if version > VERSION {
            return Err(Error::new(
                Code::Unsupported,
                format!("版本 {version} 高于本端支持的 {VERSION}"),
            )
            .with("pack::Archive"));
        }
        if version < VERSION {
            return Err(invalid(format!(
                "版本 {version} 已不再支持，当前为 {VERSION}"
            )));
        }
        let flags = u32::from_le_bytes(bytes[6..10].try_into().expect("4 字节"));
        if flags & !(GLOBAL_CRC | SEG_CRC) != 0 {
            return Err(invalid("flags 含非零预留位"));
        }
        let align_log = bytes[10];
        if align_log > MAX_ALIGN_LOG {
            return Err(invalid(format!(
                "align_log {align_log} 超过 {MAX_ALIGN_LOG}"
            )));
        }
        let align = 1usize << align_log;

        let mut cur = Cursor::new(bytes);
        let _ = cur.take(FIXED_HEAD)?;
        let count = cur.read_u32()?;
        let mut descs = Vec::with_capacity(usize::try_from(count).expect("u32 -> usize"));
        for _ in 0..count {
            descs.push(Desc::decode(&mut cur)?);
        }

        let footer_pos = bytes.len() - FOOTER_LEN;
        let data_start = cur.pos();
        let mut cursor = data_start;
        let mut stored = Vec::with_capacity(descs.len());
        for desc in descs {
            let (name, dtype, shape, quant, codec, data_len, seg_crc) = desc.into_parts();
            let start = mem::align_up(cursor, align)?;
            // 对齐填充必须全 0。
            if !bytes
                .get(cursor..start)
                .map(|pad| pad.iter().all(|b| *b == 0))
                .unwrap_or(false)
            {
                return Err(invalid(format!("段 `{name}` 之前的对齐填充含非零字节")));
            }
            let len = usize::try_from(data_len).map_err(|_| invalid("段长度超 usize"))?;
            let end = start
                .checked_add(len)
                .ok_or_else(|| invalid("段区间相加溢出"))?;
            if end > footer_pos {
                return Err(invalid(format!("段 `{name}` 的区间超出数据区，文件被截断")));
            }
            stored.push(Stored {
                name,
                dtype,
                shape,
                quant,
                codec,
                start,
                len,
                crc: seg_crc,
            });
            cursor = end;
        }

        // 数据区末尾必须正好接上 footer，不许多余字节。
        if cursor != footer_pos {
            return Err(invalid(format!(
                "数据区结束于 {cursor}，尾部在 {footer_pos}：存在多余字节"
            )));
        }

        let archive = Self {
            bytes,
            segs: stored,
            flags,
        };
        archive.verify_checksums(footer_pos)?;
        Ok(archive)
    }

    /// 两级 CRC 校验；关闭时对应字段必须为 0。
    fn verify_checksums(&self, footer_pos: usize) -> Result<()> {
        let want_global = self.flags & GLOBAL_CRC != 0;
        let want_seg = self.flags & SEG_CRC != 0;
        let stored_global = u32::from_le_bytes(
            self.bytes[footer_pos..footer_pos + 4]
                .try_into()
                .expect("4 字节"),
        );
        if want_global {
            let actual = crate::crc::crc32(&self.bytes[..footer_pos]);
            if actual != stored_global {
                return Err(invalid(format!(
                    "全局 CRC 不匹配：文件声明 {stored_global:#010x}，实算 {actual:#010x}"
                )));
            }
        } else if stored_global != 0 {
            return Err(invalid("全局 CRC 已关闭却写了非零值"));
        }
        if want_seg {
            for seg in &self.segs {
                let actual = crate::crc::crc32(&self.bytes[seg.start..seg.start + seg.len]);
                if actual != seg.crc {
                    return Err(invalid(format!(
                        "段 `{}` 的 CRC 不匹配：声明 {:#010x}，实算 {actual:#010x}",
                        seg.name, seg.crc
                    )));
                }
            }
        } else if self.segs.iter().any(|seg| seg.crc != 0) {
            return Err(invalid("段级 CRC 已关闭却写了非零值"));
        }
        Ok(())
    }

    /// 全部段。
    pub fn segments(&self) -> &[Stored] {
        &self.segs
    }

    /// 按名查段。
    pub fn get(&self, name: &str) -> Option<&Stored> {
        self.segs.iter().find(|seg| seg.name == name)
    }

    /// 借用段的原始存储字节（零拷贝，不做解压）。
    pub fn read(&self, seg: &Stored) -> &'a [u8] {
        &self.bytes[seg.start..seg.start + seg.len]
    }

    /// 加载段的逻辑字节：`store` 段零拷贝借用，`rans` 段解压为新缓冲。
    pub fn load(&self, seg: &Stored) -> Result<Cow<'a, [u8]>> {
        let raw = self.read(seg);
        match seg.codec {
            Codec::Store => Ok(Cow::Borrowed(raw)),
            Codec::Rans => Ok(Cow::Owned(rans::decode(raw)?)),
        }
    }

    /// 收集每段逻辑字节的叶子哈希；用解压后字节，证明独立于压缩选择。
    fn leaves(&self) -> Result<Vec<Digest>> {
        let mut leaves = Vec::with_capacity(self.segs.len());
        for seg in &self.segs {
            let logical = self.load(seg)?;
            leaves.push(crate::merkle::leaf(&seg.name, &logical));
        }
        Ok(leaves)
    }

    /// 逐张量 Merkle 根。
    pub fn merkle_root(&self) -> Result<Digest> {
        let leaves = self.leaves()?;
        Ok(crate::merkle::root(&leaves))
    }

    /// 为下标生成包含证明；下标越界返回错误。
    pub fn prove(&self, index: usize) -> Result<Proof> {
        let leaves = self.leaves()?;
        crate::merkle::prove(&leaves, index)
            .ok_or_else(|| invalid(format!("段下标 {index} 越界，共 {} 段", leaves.len())))
    }

    /// 仅用下标、证明与根验证归属，不接触其他张量。
    pub fn verify(&self, index: usize, proof: &Proof, root: &Digest) -> Result<bool> {
        let leaves = self.leaves()?;
        let Some(leaf_hash) = leaves.get(index) else {
            return Ok(false);
        };
        Ok(crate::merkle::verify_proof(leaf_hash, proof, root))
    }
}

fn invalid(message: impl Into<String>) -> Error {
    Error::new(Code::Invalid, message).with("pack::Archive")
}
