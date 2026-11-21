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
    w.u32(slots.iter().filter(|(_, id)| id.is_some()).count() as u32);
    write_pieces(&mut w, &source.pieces);
    write_special(&mut w, slots);
    write_merges(&mut w, source);
    write_scores(&mut w, source);
    w.u64(digest);
    w.0
}

fn write_pieces(w: &mut Writer, pieces: &[Vec<u8>]) {
    for piece in pieces {
        w.bytes(piece);
    }
}

fn write_special(w: &mut Writer, slots: [(u8, Option<u32>); 7]) {
    for (kind, id) in slots {
        if let Some(id) = id {
            w.u8(kind);
            w.u32(id);
        }
    }
}

fn write_merges(w: &mut Writer, source: &VocabSource) {
    if !source.family.requires_merges() {
        return;
    }
    w.u32(source.merges.len() as u32);
    for merge in &source.merges {
        w.bytes(&merge.left);
        w.bytes(&merge.right);
    }
}

fn write_scores(w: &mut Writer, source: &VocabSource) {
    if !source.family.requires_scores() {
        return;
    }
    w.u32(source.scores.len() as u32);
    for score in &source.scores {
        w.f32(*score);
    }
}

/// 把自描述字节解码回中性词表值对象与期望指纹摘要。
pub fn decode_vocab(bytes: &[u8]) -> Result<(VocabSource, u64)> {
    let mut c = Cursor { bytes, pos: 0 };
    let prologue = prologue(&mut c)?;
    let source = source(&mut c, prologue)?;
    let digest = c.u64()?;
    c.finish()?;
    Ok((source, digest))
}

/// 前导固定 14 字节：magic(4)+version(1)+family(1)+pieces(4)+specials(4)。
const PROLOGUE: usize = 14;

struct Prologue {
    family: VocabFamily,
    pieces: usize,
    specials: usize,
}

fn prologue(c: &mut Cursor<'_>) -> Result<Prologue> {
    let raw = c.take(PROLOGUE)?;
    check_prologue(raw)?;
    let family = VocabFamily::from_id(raw[5]).ok_or_else(|| invalid("未知词表族序号"))?;
    Ok(Prologue {
        family,
        pieces: read_u32(raw, 6) as usize,
        specials: read_u32(raw, 10) as usize,
    })
}

fn check_prologue(raw: &[u8]) -> Result<()> {
    if raw[..4] != MAGIC {
        return Err(invalid("词表段魔数不符"));
    }
    if raw[4] != VERSION {
        return Err(invalid("词表段版本不支持"));
    }
    Ok(())
}

fn source(c: &mut Cursor<'_>, p: Prologue) -> Result<VocabSource> {
    let pieces = read_n(c, p.pieces, |cur| cur.bytes())?;
    let special = specials(c, p.specials)?;
    let merges = read_merges(c, p.family)?;
    let scores = read_scores(c, p.family)?;
    Ok(VocabSource {
        family: p.family,
        pieces,
        special,
        scores,
        merges,
    })
}

fn specials(c: &mut Cursor<'_>, count: usize) -> Result<SpecialTokens> {
    let mut special = SpecialTokens::EMPTY;
    read_n(c, count, |cur| {
        let kind = cur.u8()?;
        let id = cur.u32()?;
        special.set_slot(kind, id)
    })?;
    Ok(special)
}

fn read_merges(c: &mut Cursor<'_>, family: VocabFamily) -> Result<Vec<Merge>> {
    if !family.requires_merges() {
        return Ok(Vec::new());
    }
    let count = c.u32()? as usize;
    read_n(c, count, |cur| {
        Ok(Merge {
            left: cur.bytes()?,
            right: cur.bytes()?,
        })
    })
}

fn read_scores(c: &mut Cursor<'_>, family: VocabFamily) -> Result<Vec<f32>> {
    if !family.requires_scores() {
        return Ok(Vec::new());
    }
    let count = c.u32()? as usize;
    read_n(c, count, |cur| cur.f32())
}

/// 读 `n` 个同类元素；元素解码器由闭包提供，`n` 个 `Result` 短路收束为一个 `Result`。
fn read_n<T>(
    c: &mut Cursor<'_>,
    n: usize,
    mut read: impl FnMut(&mut Cursor<'_>) -> Result<T>,
) -> Result<Vec<T>> {
    (0..n).map(|_| read(c)).collect()
}

/// 固定偏移读小端 `u32`；偏移由前导布局常量保证，不越界。
fn read_u32(raw: &[u8], at: usize) -> u32 {
    u32::from_le_bytes([raw[at], raw[at + 1], raw[at + 2], raw[at + 3]])
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

    /// 读毕后核对游标恰好耗尽；有残留字节即报 `Invalid`。
    fn finish(&self) -> Result<()> {
        if self.is_done() {
            Ok(())
        } else {
            Err(invalid("词表段尾部有残留字节"))
        }
    }
}

fn invalid(message: impl Into<String>) -> Error {
    Error::new(Code::Invalid, message).with("tokenizer::vocab::codec")
}

#[cfg(test)]
#[path = "../../tests/codec.rs"]
mod tests;
