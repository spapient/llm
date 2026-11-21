//! 内存容器：绑定在输入字节上，不复制数据，负责两级 CRC 校验、段读取与 Merkle 证明。

use std::borrow::Cow;
use std::sync::OnceLock;

use diagnostic::{Code, Error, Result};

use crate::metrics::{Metrics, MetricsAcc};
use crate::parse::{FOOTER_LEN, fixed, layout, read_descs};
use crate::proof::{Digest, Proof};
use crate::stored::Stored;
use crate::writer::{GLOBAL_CRC, SEG_CRC};
use crate::{Codec, rans};

/// 内存容器：绑定在输入字节上，不复制数据。
#[derive(Debug)]
pub struct Archive<'a> {
    bytes: &'a [u8],
    segs: Vec<Stored>,
    flags: u32,
    /// 叶子摘要惰性缓存：同一份不可变字节只解压+哈希一次，root/prove/verify 复用。
    /// 内层带错误：`get_or_init` 是稳定接口，解码失败存下后原样克隆复返。
    leaves: OnceLock<Result<Vec<Digest>>>,
}

impl<'a> Archive<'a> {
    /// 解析并完整校验。
    pub fn parse(bytes: &'a [u8]) -> Result<Self> {
        let (flags, align) = fixed(bytes)?;
        let (descs, data_start) = read_descs(bytes)?;
        let footer_pos = bytes.len() - FOOTER_LEN;
        let segs = layout(descs, bytes, data_start, footer_pos, align)?;
        let archive = Self {
            bytes,
            segs,
            flags,
            leaves: OnceLock::new(),
        };
        archive.verify_checksums(footer_pos)?;
        Ok(archive)
    }

    /// 两级 CRC 校验；关闭时对应字段必须为 0。
    fn verify_checksums(&self, footer_pos: usize) -> Result<()> {
        self.verify_global(footer_pos)?;
        self.verify_segments()
    }

    /// 全局 CRC：开启时必须匹配，关闭时必须为 0。
    fn verify_global(&self, footer_pos: usize) -> Result<()> {
        let stored = u32::from_le_bytes([
            self.bytes[footer_pos],
            self.bytes[footer_pos + 1],
            self.bytes[footer_pos + 2],
            self.bytes[footer_pos + 3],
        ]);
        if self.flags & GLOBAL_CRC == 0 {
            if stored != 0 {
                return Err(bad("全局 CRC 已关闭却写了非零值"));
            }
            return Ok(());
        }
        let actual = crate::crc::crc32(&self.bytes[..footer_pos]);
        if actual != stored {
            return Err(bad(format!(
                "全局 CRC 不匹配：文件声明 {stored:#010x}，实算 {actual:#010x}"
            )));
        }
        Ok(())
    }

    /// 段级 CRC：开启时逐段匹配，关闭时所有段字段必须为 0。引用段复用代表段的校验，
    /// 不单独重算。
    fn verify_segments(&self) -> Result<()> {
        if self.flags & SEG_CRC == 0 {
            return self.check_disabled_segment_crc();
        }
        for seg in &self.segs {
            self.check_segment_crc(seg)?;
        }
        Ok(())
    }

    /// 关闭段级 CRC 时，所有段的 CRC 字段必须为 0。
    fn check_disabled_segment_crc(&self) -> Result<()> {
        let nonzero = self.segs.iter().any(|seg| seg.crc() != 0);
        if nonzero {
            return Err(bad("段级 CRC 已关闭却写了非零值"));
        }
        Ok(())
    }

    /// 单条内联段的帧 CRC 比对；引用段跳过，校验归代表段。
    fn check_segment_crc(&self, seg: &Stored) -> Result<()> {
        if seg.is_ref() {
            return Ok(());
        }
        let actual = crate::crc::crc32(&self.bytes[seg.start()..seg.start() + seg.len()]);
        if actual != seg.crc() {
            return Err(bad(format!(
                "段 `{}` 的 CRC 不匹配：声明 {:#010x}，实算 {actual:#010x}",
                seg.name(),
                seg.crc()
            )));
        }
        Ok(())
    }

    /// 全部段。
    pub fn segments(&self) -> &[Stored] {
        &self.segs
    }

    /// 按名查段。
    pub fn get(&self, name: &str) -> Option<&Stored> {
        self.segs.iter().find(|seg| seg.name() == name)
    }

    /// 借用段的原始存储字节（零拷贝，不做解压）。引用段返回代表段的帧。
    pub fn read(&self, seg: &Stored) -> &'a [u8] {
        &self.bytes[seg.start()..seg.start() + seg.len()]
    }

    /// 加载段的逻辑字节：`store` 段零拷贝借用，`rans`/`axes` 段解压为新缓冲。引用段解引用到代表段。
    pub fn load(&self, seg: &Stored) -> Result<Cow<'a, [u8]>> {
        let raw = self.read(seg);
        decode_frame(raw, seg.codec(), seg.logical_len()?)
    }

    /// 收集每段逻辑字节的叶子哈希；用解压后字节，证明独立于压缩选择。
    fn collect_leaves(&self) -> Result<Vec<Digest>> {
        let mut leaves = Vec::with_capacity(self.segs.len());
        for seg in &self.segs {
            let logical = self.load(seg)?;
            leaves.push(crate::proof::leaf(seg.name(), &logical));
        }
        Ok(leaves)
    }

    /// 惰性取叶子摘要：首次调用解压+哈希全段，之后复用缓存，不再重复解压。
    fn cached_leaves(&self) -> Result<&[Digest]> {
        match self.leaves.get_or_init(|| self.collect_leaves()) {
            Ok(leaves) => Ok(leaves.as_slice()),
            Err(error) => Err(error.clone()),
        }
    }

    /// 逐张量 Merkle 根。
    pub fn merkle_root(&self) -> Result<Digest> {
        let leaves = self.cached_leaves()?;
        Ok(crate::proof::root(leaves))
    }

    /// 为下标生成包含证明；下标越界返回错误。
    pub fn prove(&self, index: usize) -> Result<Proof> {
        let leaves = self.cached_leaves()?;
        crate::proof::prove(leaves, index)
            .ok_or_else(|| bad(format!("段下标 {index} 越界，共 {} 段", leaves.len())))
    }

    /// 仅用下标、证明与根验证归属，不接触其他张量。
    pub fn verify(&self, index: usize, proof: &Proof, root: &Digest) -> Result<bool> {
        let leaves = self.cached_leaves()?;
        let Some(leaf_hash) = leaves.get(index) else {
            return Ok(false);
        };
        Ok(crate::proof::verify_proof(leaf_hash, proof, root))
    }

    /// 容器度量：打开后由现场字节与段表现算，不解析文件内保存的任何度量字段。
    pub fn metrics(&self) -> Result<Metrics> {
        let mut acc = MetricsAcc::default();
        for seg in &self.segs {
            acc.add(seg)?;
        }
        Ok(acc.finish(self.segs.len(), self.bytes.len()))
    }
}


/// 按段 codec 把原始存储帧解为逻辑字节：`store` 零拷贝借用，`rans`/`axes` 解压为新缓冲。
fn decode_frame<'a>(raw: &'a [u8], codec: Codec, want: usize) -> Result<Cow<'a, [u8]>> {
    match codec {
        Codec::Store => Ok(Cow::Borrowed(raw)),
        Codec::Rans => Ok(Cow::Owned(rans::decode(raw, want)?)),
        Codec::Axes => Ok(Cow::Owned(decode_axes(raw, want)?)),
    }
}

/// 解五轴无损帧为逻辑字节，并核对长度与描述符推出的一致。
fn decode_axes(raw: &[u8], want: usize) -> Result<Vec<u8>> {
    let bytes = crate::axes::expand(raw)?;
    if bytes.len() != want {
        return Err(bad("axes 帧解出的字节数与形状不符"));
    }
    Ok(bytes)
}

/// 词表段保留名：字节容器占位；字节 → `VocabSource` 解析留给后续刀，本层不接 tokenizer。
pub const VOCAB_NAME: &str = "llm.vocab";

#[cfg(test)]
#[path = "../tests/archive.rs"]
mod tests;
