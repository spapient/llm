//! 读端容器与生命周期变换：解析自证、模式判定、发布剥离与增量补丁。
//!
//! 解析固定头、描述符、对齐填充与两级 CRC，一处都不放宽；段的绝对偏移不进文件，由「数据区
//! 起点 + 对齐规则 + 各段长度」重算，再核对填充、段级与全局 CRC。截断、改一个比特、尾部多余
//! 字节、非零填充都在打开期失败。
//!
//! 在此之上挂两类只依赖已读 `Archive` 的变换：训练/发布模式判定与 `opt.` 状态剥离；以及逐字节
//! XOR 的增量补丁（自逆，base 缺失或不等长即按全量保留）。

use std::borrow::Cow;

use diagnostic::{Code, Error, Result};

use crate::Codec;
use crate::Wire;
use crate::leb::Cursor;
use crate::proof::{Digest, Proof};
use crate::quant::Quant;
use crate::rans;
use crate::seg::{Desc, Descriptor, Segment, numel};
use crate::writer::{GLOBAL_CRC, MAGIC, MAX_ALIGN_LOG, SEG_CRC, VERSION};

/// 固定头长度：magic 4 + version 2 + flags 4 + align 1（count 是 LEB，不计入）。
const FIXED_HEAD: usize = 11;

const FOOTER_LEN: usize = 4;

/// 一个已解析的段：共享描述符 + 重算出的绝对字节区间。
#[derive(Debug, Clone)]
pub struct Stored {
    desc: Descriptor,
    codec: Codec,
    start: usize,
    len: usize,
    crc: u32,
}

impl Stored {
    /// 段名。
    pub fn name(&self) -> &str {
        &self.desc.name
    }

    /// 存储编解码方式。
    pub fn codec(&self) -> Codec {
        self.codec
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

    /// 逻辑字节数：元素类型 × 元素个数，也是 `rans::decode` 的分配上限（来自描述符而非帧）。
    pub fn logical_len(&self) -> Result<usize> {
        usize::try_from(self.desc.dtype.bytes_for(numel(&self.desc.shape)))
            .map_err(|_| invalid("段的逻辑长度超 usize"))
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
        let (flags, align) = fixed(bytes)?;
        let (descs, data_start) = read_descs(bytes)?;
        let footer_pos = bytes.len() - FOOTER_LEN;
        let segs = layout(descs, bytes, data_start, footer_pos, align)?;
        let archive = Self { bytes, segs, flags };
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
                return Err(invalid("全局 CRC 已关闭却写了非零值"));
            }
            return Ok(());
        }
        let actual = crate::crc::crc32(&self.bytes[..footer_pos]);
        if actual != stored {
            return Err(invalid(format!(
                "全局 CRC 不匹配：文件声明 {stored:#010x}，实算 {actual:#010x}"
            )));
        }
        Ok(())
    }

    /// 段级 CRC：开启时逐段匹配，关闭时所有段字段必须为 0。
    fn verify_segments(&self) -> Result<()> {
        if self.flags & SEG_CRC == 0 {
            return if self.segs.iter().any(|seg| seg.crc != 0) {
                Err(invalid("段级 CRC 已关闭却写了非零值"))
            } else {
                Ok(())
            };
        }
        for seg in &self.segs {
            let actual = crate::crc::crc32(&self.bytes[seg.start..seg.start + seg.len]);
            if actual != seg.crc {
                return Err(invalid(format!(
                    "段 `{}` 的 CRC 不匹配：声明 {:#010x}，实算 {actual:#010x}",
                    seg.name(),
                    seg.crc
                )));
            }
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

    /// 借用段的原始存储字节（零拷贝，不做解压）。
    pub fn read(&self, seg: &Stored) -> &'a [u8] {
        &self.bytes[seg.start..seg.start + seg.len]
    }

    /// 加载段的逻辑字节：`store` 段零拷贝借用，`rans` 段解压为新缓冲。
    pub fn load(&self, seg: &Stored) -> Result<Cow<'a, [u8]>> {
        let raw = self.read(seg);
        match seg.codec {
            Codec::Store => Ok(Cow::Borrowed(raw)),
            Codec::Rans => Ok(Cow::Owned(rans::decode(raw, seg.logical_len()?)?)),
        }
    }

    /// 收集每段逻辑字节的叶子哈希；用解压后字节，证明独立于压缩选择。
    fn leaves(&self) -> Result<Vec<Digest>> {
        let mut leaves = Vec::with_capacity(self.segs.len());
        for seg in &self.segs {
            let logical = self.load(seg)?;
            leaves.push(crate::proof::leaf(seg.name(), &logical));
        }
        Ok(leaves)
    }

    /// 逐张量 Merkle 根。
    pub fn merkle_root(&self) -> Result<Digest> {
        let leaves = self.leaves()?;
        Ok(crate::proof::root(&leaves))
    }

    /// 为下标生成包含证明；下标越界返回错误。
    pub fn prove(&self, index: usize) -> Result<Proof> {
        let leaves = self.leaves()?;
        crate::proof::prove(&leaves, index)
            .ok_or_else(|| invalid(format!("段下标 {index} 越界，共 {} 段", leaves.len())))
    }

    /// 仅用下标、证明与根验证归属，不接触其他张量。
    pub fn verify(&self, index: usize, proof: &Proof, root: &Digest) -> Result<bool> {
        let leaves = self.leaves()?;
        let Some(leaf_hash) = leaves.get(index) else {
            return Ok(false);
        };
        Ok(crate::proof::verify_proof(leaf_hash, proof, root))
    }
}

/// 固定头：长度、魔数、版本、预留位与对齐逐项检查，不回答段布局。
fn fixed(bytes: &[u8]) -> Result<(u32, usize)> {
    if bytes.len() < FIXED_HEAD + FOOTER_LEN + 1 {
        return Err(invalid(format!(
            "文件只有 {} 字节，读不出固定头、段计数与尾部",
            bytes.len()
        )));
    }
    let magic = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
    if magic != MAGIC {
        return Err(invalid(format!(
            "魔数不匹配：期望 {MAGIC:#010x}，得到 {magic:#010x}"
        )));
    }
    check_version(u16::from_le_bytes([bytes[4], bytes[5]]))?;
    flags_align(bytes)
}

/// 版本：高于本端是 Unsupported，低于本端是已不支持。
fn check_version(version: u16) -> Result<()> {
    if version > VERSION {
        return Err(Error::new(
            Code::Unsupported,
            format!("版本 {version} 高于本端支持的 {VERSION}"),
        )
        .with("weights::Archive"));
    }
    if version < VERSION {
        return Err(invalid(format!(
            "版本 {version} 已不再支持，当前为 {VERSION}"
        )));
    }
    Ok(())
}

/// 预留位必须为 0，对齐指数不得超出上限。
fn flags_align(bytes: &[u8]) -> Result<(u32, usize)> {
    let flags = u32::from_le_bytes([bytes[6], bytes[7], bytes[8], bytes[9]]);
    if flags & !(GLOBAL_CRC | SEG_CRC) != 0 {
        return Err(invalid("flags 含非零预留位"));
    }
    let align_log = bytes[10];
    if align_log > MAX_ALIGN_LOG {
        return Err(invalid(format!(
            "align_log {align_log} 超过 {MAX_ALIGN_LOG}"
        )));
    }
    Ok((flags, 1usize << align_log))
}

/// 段计数与描述符；返回数据区起点（绝对偏移）。
fn read_descs(bytes: &[u8]) -> Result<(Vec<Desc>, usize)> {
    let mut cur = Cursor::new(&bytes[FIXED_HEAD..]);
    let count = usize::try_from(cur.read::<u32>()?).map_err(|_| invalid("段数量超 usize"))?;
    let mut descs = Vec::with_capacity(count);
    for _ in 0..count {
        descs.push(Desc::decode(&mut cur)?);
    }
    Ok((descs, FIXED_HEAD + cur.pos()))
}

/// 重算每段的绝对区间；末尾必须正好接上 footer。
fn layout(
    descs: Vec<Desc>,
    bytes: &[u8],
    data_start: usize,
    footer_pos: usize,
    align: usize,
) -> Result<Vec<Stored>> {
    let mut cursor = data_start;
    let mut stored = Vec::with_capacity(descs.len());
    for desc in descs {
        let (segment, end) = place(desc, bytes, cursor, align)?;
        if end > footer_pos {
            return Err(invalid(format!(
                "段 `{}` 的区间超出数据区，文件被截断",
                segment.name()
            )));
        }
        stored.push(segment);
        cursor = end;
    }
    if cursor != footer_pos {
        return Err(invalid(format!(
            "数据区结束于 {cursor}，尾部在 {footer_pos}：存在多余字节"
        )));
    }
    Ok(stored)
}

/// 一个描述符落到字节区间：填充为 0、长度可表示、相加不溢出。
fn place(desc: Desc, bytes: &[u8], cursor: usize, align: usize) -> Result<(Stored, usize)> {
    let (meta, codec, data_len, crc) = desc.into_parts();
    let Some(start) = crate::align_up(cursor, align) else {
        return Err(invalid("对齐必须是 2 的幂且对齐后不溢出"));
    };
    let pad_ok = bytes
        .get(cursor..start)
        .is_some_and(|pad| pad.iter().all(|b| *b == 0));
    if !pad_ok {
        return Err(invalid(format!(
            "段 `{}` 之前的对齐填充含非零字节",
            meta.name
        )));
    }
    let len = usize::try_from(data_len).map_err(|_| invalid("段长度超 usize"))?;
    let Some(end) = start.checked_add(len) else {
        return Err(invalid("段区间相加溢出"));
    };
    Ok((
        Stored {
            desc: meta,
            codec,
            start,
            len,
            crc,
        },
        end,
    ))
}

fn invalid(message: impl Into<String>) -> Error {
    Error::new(Code::Invalid, message).with("weights::Archive")
}

// —— 容器生命周期模式：训练断点 / 推理发布 ——

/// 模式标记段的保留名。
pub const MODE_NAME: &str = "llm.mode";
/// 优化器状态段的命名空间前缀。
const OPT_PREFIX: &str = "opt.";

/// 容器生命周期模式。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    /// 训练断点：含优化器/动量/方差等 `opt.` 状态。
    Train,
    /// 推理发布：剥离训练状态的精简产物。
    Infer,
}

impl Mode {
    fn code(self) -> u8 {
        match self {
            Mode::Train => 0,
            Mode::Infer => 1,
        }
    }

    fn from_code(code: u8) -> Result<Self> {
        match code {
            0 => Ok(Mode::Train),
            1 => Ok(Mode::Infer),
            other => Err(bad(&format!("mode: 未知码 {other}"))),
        }
    }

    /// 编码为可写进容器的标记段：保留名、`U8` 单轴、1 字节。
    pub fn to_segment(self) -> Result<Segment> {
        Segment::new(MODE_NAME, Wire::U8, vec![1], None, vec![self.code()])
    }
}

/// 读取模式；无标记段的旧文件按推理模式处理。
pub fn mode_of(archive: &Archive<'_>) -> Mode {
    archive
        .get(MODE_NAME)
        .and_then(|seg| archive.read(seg).first().copied())
        .map_or(Mode::Infer, |code| {
            Mode::from_code(code).unwrap_or(Mode::Infer)
        })
}

/// 段名是否属于优化器状态。
pub fn is_opt(name: &str) -> bool {
    name.starts_with(OPT_PREFIX)
}

/// 发布转换：剥离所有 `opt.` 段，保留其余段并标记为推理模式。
pub fn release(src: &Archive<'_>) -> Result<Vec<Segment>> {
    let mut out = Vec::new();
    for seg in src.segments() {
        if !is_opt(seg.name()) {
            out.push(keep_segment(src, seg)?);
        }
    }
    out.push(Mode::Infer.to_segment()?);
    Ok(out)
}

/// 把读端段还原成写端段：加载逻辑字节，沿用名字/类型/形状/量化方案。
fn keep_segment(src: &Archive<'_>, seg: &Stored) -> Result<Segment> {
    let data = src.load(seg)?.into_owned();
    Segment::new(
        seg.name(),
        seg.dtype(),
        seg.shape().to_vec(),
        seg.quant().cloned(),
        data,
    )
}

fn bad(message: &str) -> Error {
    Error::new(Code::Invalid, message).with("mode")
}

// —— 增量补丁：逐字节 XOR，读端按 base 自逆合成 ——

/// 计算 cur 相对 base 的补丁段（逻辑字节，未压缩；交给 Writer 落盘并按需 rans）。
pub fn diff(base: &Archive<'_>, cur: &Archive<'_>) -> Result<Vec<Segment>> {
    rebuild(base, cur)
}

/// 用 base 与补丁归档合成完整段。
pub fn join(base: &Archive<'_>, patch: &Archive<'_>) -> Result<Vec<Segment>> {
    rebuild(base, patch)
}

/// diff 与 join 同构：对 right 每个段，base 有同名等长段就 XOR，否则按全量保留。
fn rebuild(base: &Archive<'_>, right: &Archive<'_>) -> Result<Vec<Segment>> {
    let mut out = Vec::new();
    for seg in right.segments() {
        let bytes = right.load(seg)?.into_owned();
        let data = apply(base, seg.name(), bytes)?;
        out.push(Segment::new(
            seg.name(),
            seg.dtype(),
            seg.shape().to_vec(),
            seg.quant().cloned(),
            data,
        )?);
    }
    Ok(out)
}

/// 同名等长则 XOR；base 缺失或长度不同则原样返回（全量）。
fn apply(base: &Archive<'_>, name: &str, bytes: Vec<u8>) -> Result<Vec<u8>> {
    let Some(prev) = base.get(name) else {
        return Ok(bytes);
    };
    let prev_bytes = base.load(prev)?;
    if prev_bytes.len() != bytes.len() {
        return Ok(bytes);
    }
    Ok(xor(&bytes, &prev_bytes))
}

fn xor(a: &[u8], b: &[u8]) -> Vec<u8> {
    a.iter().zip(b).map(|(x, y)| x ^ y).collect()
}

#[cfg(test)]
mod mode_tests {
    use super::*;
    use crate::testutil::{archive, f32_segment};

    #[test]
    fn release_strips_opt_and_marks_infer() {
        let bytes = archive(vec![
            f32_segment("w", &[1.0, 2.0]),
            f32_segment("opt.m.w", &[0.5, 0.5]),
        ]);
        let src = Archive::parse(&bytes).expect("src");

        let released = release(&src).expect("release");
        let names: Vec<&str> = released.iter().map(|s| s.name()).collect();
        assert!(names.contains(&"w"));
        assert!(!names.iter().any(|n| is_opt(n)));

        let released_bytes = archive(released);
        let released_archive = Archive::parse(&released_bytes).expect("x");
        assert_eq!(mode_of(&released_archive), Mode::Infer);
    }

    #[test]
    fn missing_marker_defaults_infer() {
        let bytes = archive(vec![f32_segment("w", &[1.0])]);
        assert_eq!(mode_of(&Archive::parse(&bytes).expect("a")), Mode::Infer);
    }

    #[test]
    fn train_marker_roundtrips() {
        let mut segments = vec![f32_segment("w", &[1.0])];
        segments.push(Mode::Train.to_segment().expect("标记"));
        let bytes = archive(segments);
        let parsed = Archive::parse(&bytes).expect("a");
        assert_eq!(mode_of(&parsed), Mode::Train);
    }
}

#[cfg(test)]
mod patch_tests {
    use super::*;
    use crate::testutil::{archive, f32_segment};
    use crate::writer::Writer;

    #[test]
    fn xor_roundtrip_recovers_cur() {
        let base_bytes = archive(vec![f32_segment("w", &[1.0, 2.0, 3.0])]);
        let base = Archive::parse(&base_bytes).expect("base");
        let cur_bytes = archive(vec![f32_segment("w", &[1.0, 2.0, 3.5])]);
        let cur = Archive::parse(&cur_bytes).expect("cur");

        let patch_bytes = archive(diff(&base, &cur).expect("diff"));
        let patch = Archive::parse(&patch_bytes).expect("patch");

        let joined = join(&base, &patch).expect("join");
        let expected = cur.load(&cur.segments()[0]).expect("cur 字节");
        assert_eq!(joined[0].data, &expected[..]);
    }

    #[test]
    fn small_change_patch_is_smaller() {
        let base_bytes = archive(vec![f32_segment("w", &[1.0; 4096])]);
        let base = Archive::parse(&base_bytes).expect("base");
        let mut changed = vec![1.0f32; 4096];
        changed[10] = 1.0001;
        let cur_bytes = archive(vec![f32_segment("w", &changed)]);
        let cur = Archive::parse(&cur_bytes).expect("cur");

        let mut w = Writer::new();
        w.compress(true);
        for seg in diff(&base, &cur).expect("diff") {
            w.push(seg).expect("push");
        }
        assert!(w.finish(6).expect("patch").len() < cur_bytes.len());
    }

    #[test]
    fn missing_base_is_full_not_garbage() {
        let base_bytes = archive(vec![f32_segment("a", &[1.0])]);
        let base = Archive::parse(&base_bytes).expect("base");
        let cur_bytes = archive(vec![f32_segment("b", &[9.0])]);
        let cur = Archive::parse(&cur_bytes).expect("cur");

        let patch = diff(&base, &cur).expect("diff");
        assert_eq!(patch[0].name(), "b");
        assert_eq!(patch[0].data, f32_segment("b", &[9.0]).data);
    }
}
