//! 词表中性字节布局：`VocabSource` ↔ 字节的自描述编码。
//!
//! 布局（全部小端）：
//!
//! ```text
//! magic(4) | version(u8) | family(u8) | vocab_size(u32) | special_count(u32)
//! | pieces:  vocab_size × (len(u32) + bytes)
//! | special: special_count × (kind(u8) + id(u32))
//! | [Bpe] merges:  count(u32) + count × (len_l(u32)+left | len_r(u32)+right)
//! | [Unigram] scores: count(u32) + count × f32
//! | digest(u64)
//! ```
//!
//! 附件块是否出现由 [`VocabFamily::requires_merges`] / [`VocabFamily::requires_scores`] 声明式
//! 决定；除这两个布尔外没有按族名 `match` 的扇出。`digest` 是期望的词表指纹摘要（FNV-1a），由
//! 调用方经 [`Vocab::try_new`] 算得传入，读回时对拍；本模块不复制指纹算法，也不做 CRC（段字节
//! 完整性由 `.u` 容器段级 CRC 承担）。

use crate::{Merge, SpecialTokens, VocabFamily, VocabSource};
use diagnostic::{Code, Error, Result};

const MAGIC: [u8; 4] = *b"LVOC";
const VERSION: u8 = 1;

/// 把中性词表值对象序列化成自描述字节；`digest` 是期望指纹摘要，随段一起存。
pub fn encode_vocab(source: &VocabSource, digest: u64) -> Vec<u8> {
    let mut w = Writer(Vec::new());
    w.raw(&MAGIC);
    w.u8(VERSION);
    w.u8(source.family.id());
    w.u32(source.pieces.len() as u32);
    let slots = source.special.slots();
    let special_count = slots.iter().filter(|(_, id)| id.is_some()).count();
    w.u32(special_count as u32);
    for piece in &source.pieces {
        w.bytes(piece);
    }
    for (kind, id) in slots {
        if let Some(id) = id {
            w.u8(kind);
            w.u32(id);
        }
    }
    if source.family.requires_merges() {
        w.u32(source.merges.len() as u32);
        for merge in &source.merges {
            w.bytes(&merge.left);
            w.bytes(&merge.right);
        }
    }
    if source.family.requires_scores() {
        w.u32(source.scores.len() as u32);
        for score in &source.scores {
            w.f32(*score);
        }
    }
    w.u64(digest);
    w.0
}

/// 把自描述字节解码回中性词表值对象与期望指纹摘要。
pub fn decode_vocab(bytes: &[u8]) -> Result<(VocabSource, u64)> {
    let mut c = Cursor { bytes, pos: 0 };
    if c.take(4)? != MAGIC {
        return Err(invalid("词表段魔数不符"));
    }
    if c.u8()? != VERSION {
        return Err(invalid("词表段版本不支持"));
    }
    let family = VocabFamily::from_id(c.u8()?).ok_or_else(|| invalid("未知词表族序号"))?;
    let vocab_size = c.u32()? as usize;
    let special_count = c.u32()? as usize;
    let mut pieces = Vec::with_capacity(vocab_size);
    for _ in 0..vocab_size {
        pieces.push(c.bytes()?);
    }
    let mut special = SpecialTokens::EMPTY;
    for _ in 0..special_count {
        let kind = c.u8()?;
        let id = c.u32()?;
        special.set_slot(kind, id)?;
    }
    let mut merges = Vec::new();
    if family.requires_merges() {
        let count = c.u32()? as usize;
        merges.reserve(count);
        for _ in 0..count {
            merges.push(Merge {
                left: c.bytes()?,
                right: c.bytes()?,
            });
        }
    }
    let mut scores = Vec::new();
    if family.requires_scores() {
        let count = c.u32()? as usize;
        scores.reserve(count);
        for _ in 0..count {
            scores.push(c.f32()?);
        }
    }
    let digest = c.u64()?;
    if !c.is_done() {
        return Err(invalid("词表段尾部有残留字节"));
    }
    Ok((
        VocabSource {
            family,
            pieces,
            special,
            scores,
            merges,
        },
        digest,
    ))
}

/// 累积写入器：按小端顺序追加固定宽度与长度前缀字段。
struct Writer(Vec<u8>);

impl Writer {
    fn raw(&mut self, bytes: &[u8]) {
        self.0.extend_from_slice(bytes);
    }

    fn u8(&mut self, value: u8) {
        self.0.push(value);
    }

    fn u32(&mut self, value: u32) {
        self.0.extend_from_slice(&value.to_le_bytes());
    }

    fn u64(&mut self, value: u64) {
        self.0.extend_from_slice(&value.to_le_bytes());
    }

    fn f32(&mut self, value: f32) {
        self.0.extend_from_slice(&value.to_le_bytes());
    }

    fn bytes(&mut self, bytes: &[u8]) {
        self.u32(bytes.len() as u32);
        self.raw(bytes);
    }
}

/// 只进游标：按声明长度取字节，越界统一报 `Invalid`，不做整段快照之外的任何分配。
struct Cursor<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        let end = self
            .pos
            .checked_add(n)
            .ok_or_else(|| invalid("词表段越界"))?;
        if end > self.bytes.len() {
            return Err(invalid("词表段截断"));
        }
        let slice = &self.bytes[self.pos..end];
        self.pos = end;
        Ok(slice)
    }

    fn u8(&mut self) -> Result<u8> {
        Ok(self.take(1)?[0])
    }

    fn u32(&mut self) -> Result<u32> {
        let b = self.take(4)?;
        Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }

    fn u64(&mut self) -> Result<u64> {
        let b = self.take(8)?;
        Ok(u64::from_le_bytes([
            b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7],
        ]))
    }

    fn f32(&mut self) -> Result<f32> {
        let b = self.take(4)?;
        Ok(f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }

    fn bytes(&mut self) -> Result<Vec<u8>> {
        let len = self.u32()? as usize;
        Ok(self.take(len)?.to_vec())
    }

    fn is_done(&self) -> bool {
        self.pos == self.bytes.len()
    }
}

fn invalid(message: impl Into<String>) -> Error {
    Error::new(Code::Invalid, message).with("tokenizer::vocab_codec")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn roundtrip(source: VocabSource) {
        let digest = 0xdead_beef_1234_5678u64;
        let bytes = encode_vocab(&source, digest);
        let (decoded, got) = decode_vocab(&bytes).expect("解码");
        assert_eq!(decoded, source, "往返一致");
        assert_eq!(got, digest, "摘要透传");
    }

    #[test]
    fn byte_family_round_trips() {
        roundtrip(VocabSource {
            family: VocabFamily::Byte,
            pieces: vec![b"a".to_vec(), b"b".to_vec()],
            special: SpecialTokens::EMPTY,
            scores: Vec::new(),
            merges: Vec::new(),
        });
    }

    #[test]
    fn bpe_with_merges_and_special_round_trips() {
        roundtrip(VocabSource {
            family: VocabFamily::Bpe,
            pieces: vec![b"a".to_vec(), b"b".to_vec(), b"ab".to_vec()],
            special: SpecialTokens::EMPTY.with_bos(0).with_eos(2),
            scores: Vec::new(),
            merges: vec![Merge {
                left: b"a".to_vec(),
                right: b"b".to_vec(),
            }],
        });
    }

    #[test]
    fn unigram_with_scores_round_trips() {
        roundtrip(VocabSource {
            family: VocabFamily::Unigram,
            pieces: vec![b"a".to_vec(), b"b".to_vec()],
            special: SpecialTokens::EMPTY.with_unk(0),
            scores: vec![0.5, -0.25],
            merges: Vec::new(),
        });
    }

    #[test]
    fn bad_magic_is_rejected() {
        let mut bytes = encode_vocab(&VocabSource::default(), 0);
        bytes[0] ^= 0xff;
        assert_eq!(decode_vocab(&bytes).unwrap_err().code(), Code::Invalid);
    }

    #[test]
    fn truncation_is_rejected() {
        let bytes = encode_vocab(&VocabSource::default(), 0);
        let cut = &bytes[..bytes.len() - 1];
        assert_eq!(decode_vocab(cut).unwrap_err().code(), Code::Invalid);
    }
}
