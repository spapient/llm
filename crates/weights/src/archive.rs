//! 读端容器与生命周期变换：解析自证、模式判定、发布剥离与增量补丁。
//!
//! 解析固定头、描述符、对齐填充与两级 CRC，一处都不放宽；段的绝对偏移不进文件，由「数据区
//! 起点 + 对齐规则 + 各段长度」重算，再核对填充、段级与全局 CRC。截断、改一个比特、尾部多余
//! 字节、非零填充都在打开期失败。
//!
//! 在此之上挂两类只依赖已读 `Archive` 的变换：训练/发布模式判定与 `opt.` 状态剥离；以及逐字节
//! XOR 的增量补丁（自逆，base 缺失或不等长即按全量保留）。

use std::borrow::Cow;
use std::sync::OnceLock;

use diagnostic::{Code, Error, Result};

use crate::Codec;
use crate::Wire;
use crate::leb::Cursor;
use crate::proof::{Digest, Proof};
use crate::quant::Quant;
use crate::rans;
use crate::seg::{Desc, DescBody, Descriptor, numel};
use crate::writer::{GLOBAL_CRC, MAGIC, MAX_ALIGN_LOG, SEG_CRC, VERSION};

/// 固定头长度：magic 4 + version 2 + flags 4 + align 1（count 是 LEB，不计入）。
const FIXED_HEAD: usize = 11;

const FOOTER_LEN: usize = 4;

/// 一个已解析的段：共享描述符 + 重算出的绝对字节区间。
#[derive(Debug, Clone)]
pub struct Stored {
    desc: Descriptor,
    inner: StoredInner,
}

/// 段的存储形态：内联帧在本段名下；引用段解引用到代表段，区间/编码/CRC 在布局时复制自代表段。
#[derive(Debug, Clone)]
enum StoredInner {
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

    /// 存储编解码方式；引用段沿用代表段的 codec。
    pub fn codec(&self) -> Codec {
        match &self.inner {
            StoredInner::Inline { codec, .. } | StoredInner::Ref { codec, .. } => *codec,
        }
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
        match &self.inner {
            StoredInner::Inline { start, .. } | StoredInner::Ref { start, .. } => *start,
        }
    }

    /// 存储字节数；引用段沿用代表段的帧长。
    pub fn len(&self) -> usize {
        match &self.inner {
            StoredInner::Inline { len, .. } | StoredInner::Ref { len, .. } => *len,
        }
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
    pub fn rep_index(&self) -> Option<usize> {
        match &self.inner {
            StoredInner::Ref { rep, .. } => Some(*rep),
            StoredInner::Inline { .. } => None,
        }
    }

    /// 逻辑字节数：元素类型 × 元素个数，也是 `rans::decode` 的分配上限（来自描述符而非帧）。
    pub fn logical_len(&self) -> Result<usize> {
        usize::try_from(self.desc.dtype.bytes_for(numel(&self.desc.shape)))
            .map_err(|_| invalid("段的逻辑长度超 usize"))
    }

    /// 压缩比：存储字节 / 逻辑字节；逻辑字节为 0 时返回 `None`。
    pub fn compression_ratio(&self) -> Option<f64> {
        let logical = self.logical_len().ok()?;
        (logical != 0).then(|| self.len() as f64 / logical as f64)
    }

    /// 段级 CRC；引用段复制自代表段，不另存校验字段。
    fn crc(&self) -> u32 {
        match &self.inner {
            StoredInner::Inline { crc, .. } | StoredInner::Ref { crc, .. } => *crc,
        }
    }
}

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
            return Err(invalid("段级 CRC 已关闭却写了非零值"));
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
            return Err(invalid(format!(
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

    /// 加载段的逻辑字节：`store` 段零拷贝借用，`rans` 段解压为新缓冲。引用段解引用到代表段。
    pub fn load(&self, seg: &Stored) -> Result<Cow<'a, [u8]>> {
        let raw = self.read(seg);
        match seg.codec() {
            Codec::Store => Ok(Cow::Borrowed(raw)),
            Codec::Rans => Ok(Cow::Owned(rans::decode(raw, seg.logical_len()?)?)),
        }
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
            .ok_or_else(|| invalid(format!("段下标 {index} 越界，共 {} 段", leaves.len())))
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
        let ratio = if acc.v_fp16 == 0 {
            0.0
        } else {
            self.bytes.len() as f64 / acc.v_fp16 as f64
        };
        Ok(Metrics {
            entries: self.segs.len(),
            params: acc.params,
            v_artifact: self.bytes.len(),
            v_fp16: acc.v_fp16,
            ratio,
            lossless: acc.lossless,
            ref_hits: acc.ref_hits,
            ref_saved_bytes: acc.ref_saved_bytes,
        })
    }
}

/// 度量累加器：逐段把 numel、FP16 参考体积、无损判定与 REF 命中/节省字节收拢到一处。
struct MetricsAcc {
    params: u64,
    v_fp16: u64,
    ref_hits: usize,
    ref_saved_bytes: u64,
    lossless: bool,
}

impl Default for MetricsAcc {
    /// 空容器按无损计：无条目即无有损段；逐段再用「与」收口，任一段有损即整体有损。
    fn default() -> Self {
        Self {
            params: 0,
            v_fp16: 0,
            ref_hits: 0,
            ref_saved_bytes: 0,
            lossless: true,
        }
    }
}

impl MetricsAcc {
    /// 累加单段；引用段计入命中数与逻辑字节节省。
    fn add(&mut self, seg: &Stored) -> Result<()> {
        let numel = numel(seg.shape());
        self.params = self.params.saturating_add(numel);
        self.v_fp16 = self.v_fp16.saturating_add(numel.saturating_mul(2));
        self.lossless = self.lossless && self.is_lossless(seg);
        if seg.is_ref() {
            self.ref_hits += 1;
            let logical = u64::try_from(seg.logical_len()?).unwrap_or(u64::MAX);
            self.ref_saved_bytes = self.ref_saved_bytes.saturating_add(logical);
        }
        Ok(())
    }

    /// 单段是否无损：不带量化方案，且 codec 属于无损的 `Store`/`Rans`。
    fn is_lossless(&self, seg: &Stored) -> bool {
        seg.quant().is_none() && matches!(seg.codec(), Codec::Store | Codec::Rans)
    }
}

/// 容器度量：条目数、参数总数、体积对比与无损口径；全部在打开容器后现算。
///
/// 不读文件内保存的度量字段，也不替量化条目下"无损"结论：带量化方案的条目按既有
/// [`Quant`] 语义判为有损，未识别的 codec 同样不计入无损。
#[derive(Debug, Clone)]
pub struct Metrics {
    entries: usize,
    params: u64,
    v_artifact: usize,
    v_fp16: u64,
    ratio: f64,
    lossless: bool,
    ref_hits: usize,
    ref_saved_bytes: u64,
}

impl Metrics {
    /// 条目总数（段表行数，含引用段）。
    pub fn entries(&self) -> usize {
        self.entries
    }

    /// 参数总数：各段 `numel` 乘积之和，溢出饱和。
    pub fn params(&self) -> u64 {
        self.params
    }

    /// 容器文件字节数（含头部、对齐填充与两级 CRC）。
    pub fn v_artifact(&self) -> usize {
        self.v_artifact
    }

    /// 全量 FP16 参考体积：Σ `numel × 2`，溢出饱和。
    pub fn v_fp16(&self) -> u64 {
        self.v_fp16
    }

    /// 体积比：`V_artifact / V_FP16`；`V_FP16` 为 0 时记 0。
    pub fn ratio(&self) -> f64 {
        self.ratio
    }

    /// 无损标志：所有条目都不带量化方案，且 codec 属于无损的 `Store`/`Rans`。
    pub fn lossless(&self) -> bool {
        self.lossless
    }

    /// REF 命中数：被去重写成引用的段数。
    pub fn ref_hits(&self) -> usize {
        self.ref_hits
    }

    /// REF 节省字节数：引用段逻辑字节之和（未重复落盘的部分），溢出饱和。
    pub fn ref_saved_bytes(&self) -> u64 {
        self.ref_saved_bytes
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

/// 单条描述符占有的最小字节（名字长度 LEB 1 + 名字 1 + dtype 1 + rank 1 + 量化标志 1 +
/// 引用下标 1；内联帧另带 codec/长度/CRC）；计数先按剩余字节粗查。
const MIN_DESC_BYTES: usize = 6;
/// 描述符表预分配上限：真实归档段数远小于此，超出部分由 `Vec` 按需增长；谎报的巨大
/// 计数不会换来一次巨额分配。
const MAX_DESC_PREALLOC: usize = 4096;

/// 段计数：读 u32 计数并按剩余字节粗查；谎报的巨大计数不换来巨额预分配。
fn read_count(cur: &mut Cursor<'_>) -> Result<usize> {
    let count = usize::try_from(cur.read::<u32>()?).map_err(|_| invalid("段数量超 usize"))?;
    let plausible = cur.remaining().len().saturating_div(MIN_DESC_BYTES);
    if count > plausible {
        return Err(invalid("段数量超出剩余字节能容纳的规模"));
    }
    Ok(count)
}

/// 段计数与描述符；返回数据区起点（绝对偏移）。
fn read_descs(bytes: &[u8]) -> Result<(Vec<Desc>, usize)> {
    let mut cur = Cursor::new(&bytes[FIXED_HEAD..]);
    let count = read_count(&mut cur)?;
    let mut descs = Vec::with_capacity(count.min(MAX_DESC_PREALLOC));
    for _ in 0..count {
        descs.push(Desc::decode(&mut cur)?);
    }
    Ok((descs, FIXED_HEAD + cur.pos()))
}

/// 重算每段的绝对区间；末尾必须正好接上 footer。引用段不占数据区，区间在代表段落位后回填。
fn layout(
    descs: Vec<Desc>,
    bytes: &[u8],
    data_start: usize,
    footer_pos: usize,
    align: usize,
) -> Result<Vec<Stored>> {
    validate_names(&descs)?;
    validate_refs(&descs)?;
    let mut stored = place_all(descs, bytes, data_start, align, footer_pos)?;
    resolve_refs(&mut stored)?;
    Ok(stored)
}

/// 逐条落位到数据区，并校验数据区正好接到 footer；引用段区间待 `resolve_refs` 回填。
fn place_all(
    descs: Vec<Desc>,
    bytes: &[u8],
    mut cursor: usize,
    align: usize,
    footer_pos: usize,
) -> Result<Vec<Stored>> {
    let mut stored: Vec<Stored> = Vec::with_capacity(descs.len());
    for desc in &descs {
        let (segment, end) = place_next(desc, bytes, cursor, align, footer_pos)?;
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

/// 单条描述符落到数据区：内联段占一帧，引用段不占字节、区间待回填。
fn place_next(
    desc: &Desc,
    bytes: &[u8],
    cursor: usize,
    align: usize,
    footer_pos: usize,
) -> Result<(Stored, usize)> {
    match desc.body() {
        DescBody::Inline {
            codec,
            data_len,
            seg_crc,
        } => {
            let (segment, end) = place_inline(
                desc.descriptor(),
                bytes,
                cursor,
                align,
                *codec,
                *data_len,
                *seg_crc,
            )?;
            if end > footer_pos {
                return Err(invalid(format!(
                    "段 `{}` 的区间超出数据区，文件被截断",
                    segment.name()
                )));
            }
            Ok((segment, end))
        }
        DescBody::Ref { rep } => Ok((
            Stored {
                desc: desc.descriptor().clone(),
                inner: StoredInner::Ref {
                    rep: *rep as usize,
                    codec: Codec::Store,
                    start: 0,
                    len: 0,
                    crc: 0,
                },
            },
            cursor,
        )),
    }
}

/// 段名唯一：重名条目在打开期拒绝，不交给上层按名查找时静默覆盖。
fn validate_names(descs: &[Desc]) -> Result<()> {
    let mut seen = std::collections::BTreeSet::new();
    for desc in descs {
        let name = desc.descriptor().name.as_str();
        if !seen.insert(name) {
            return Err(invalid(format!("段名 `{name}` 重复")));
        }
    }
    Ok(())
}

/// 引用图校验：悬空/越界下标、自引用、代表段是引用、元数据不一致、引用成环，一律 `Invalid`。
fn validate_refs(descs: &[Desc]) -> Result<()> {
    let mut colors = vec![0u8; descs.len()];
    for index in 0..descs.len() {
        check_ref(index, descs, &mut colors)?;
    }
    Ok(())
}

/// 单色 DFS：0 未访 / 1 在途 / 2 完成；回到在途节点即成环。
fn check_ref(index: usize, descs: &[Desc], colors: &mut [u8]) -> Result<()> {
    match colors[index] {
        2 => return Ok(()),
        1 => return Err(invalid("引用段成环")),
        _ => {
            colors[index] = 1;
            walk_ref(index, descs, colors)?;
            colors[index] = 2;
        }
    }
    Ok(())
}

/// 在途节点沿代表下标递归；内联段直接收口，引用段交代表段继续 DFS。
fn walk_ref(index: usize, descs: &[Desc], colors: &mut [u8]) -> Result<()> {
    let DescBody::Ref { rep } = descs[index].body() else {
        return Ok(());
    };
    let rep = *rep as usize;
    validate_ref_target(index, rep, descs)?;
    check_ref(rep, descs, colors)
}

/// 引用下标与代表段元数据校验：越界、自引用、代表段是引用、取值元数据不一致一律 `Invalid`。
fn validate_ref_target(index: usize, rep: usize, descs: &[Desc]) -> Result<()> {
    let name = descs[index].descriptor().name.clone();
    if rep >= descs.len() {
        return Err(invalid(format!("引用段 `{name}` 指向越界下标 {rep}")));
    }
    if rep == index {
        return Err(invalid(format!("引用段 `{name}` 自引用")));
    }
    let target = &descs[rep];
    if !matches!(target.body(), DescBody::Inline { .. }) {
        return Err(invalid(format!(
            "引用段 `{name}` 的代表段 `{rep}` 本身是引用"
        )));
    }
    if !descs[index].descriptor().same_value(target.descriptor()) {
        return Err(invalid(format!(
            "引用段 `{name}` 与代表段 `{rep}` 的类型/形状/量化元数据不一致"
        )));
    }
    Ok(())
}

/// 代表段落位后，把其区间、codec 与 CRC 回填到引用段。先收集回填决议再落地，
/// 避免同一轮迭代里可变借用与不可变借用叠在同一份段表上。
fn resolve_refs(stored: &mut [Stored]) -> Result<()> {
    let resolutions = collect_resolutions(stored)?;
    apply_resolutions(stored, resolutions);
    Ok(())
}

/// 一条回填决议：引用段下标 + 代表段的 codec / 起点 / 长度 / CRC 快照。
type Resolution = (usize, (Codec, usize, usize, u32));

/// 先收集每条引用段的代表段区间快照；与落地分成两轮，避免可变借用叠在段表上。
fn collect_resolutions(stored: &[Stored]) -> Result<Vec<Resolution>> {
    let mut resolutions = Vec::new();
    for (index, seg) in stored.iter().enumerate() {
        let StoredInner::Ref { rep, .. } = &seg.inner else {
            continue;
        };
        let resolution = resolve_one(stored, index, *rep)?;
        resolutions.push((index, resolution));
    }
    Ok(resolutions)
}

/// 把代表段区间、codec 与 CRC 回填到引用段槽位。
fn apply_resolutions(stored: &mut [Stored], resolutions: Vec<Resolution>) {
    for (index, (codec, start, len, crc)) in resolutions {
        if let StoredInner::Ref {
            codec: c,
            start: s,
            len: l,
            crc: r,
            ..
        } = &mut stored[index].inner
        {
            *c = codec;
            *s = start;
            *l = len;
            *r = crc;
        }
    }
}

/// 取一条引用段的代表段区间快照；代表段仍是引用则拒绝。
fn resolve_one(stored: &[Stored], index: usize, rep: usize) -> Result<(Codec, usize, usize, u32)> {
    match &stored[rep].inner {
        StoredInner::Inline {
            codec,
            start,
            len,
            crc,
        } => Ok((*codec, *start, *len, *crc)),
        StoredInner::Ref { .. } => Err(invalid(format!(
            "引用段 `{index}` 的代表段 `{rep}` 本身是引用"
        ))),
    }
}

/// 一个描述符落到字节区间：填充为 0、长度可表示、相加不溢出。
fn place_inline(
    meta: &Descriptor,
    bytes: &[u8],
    cursor: usize,
    align: usize,
    codec: Codec,
    data_len: u64,
    seg_crc: u32,
) -> Result<(Stored, usize)> {
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
            desc: meta.clone(),
            inner: StoredInner::Inline {
                codec,
                start,
                len,
                crc: seg_crc,
            },
        },
        end,
    ))
}

fn invalid(message: impl Into<String>) -> Error {
    Error::new(Code::Invalid, message).with("weights::Archive")
}

/// 词表段保留名：字节容器占位；字节 → `VocabSource` 解析留给后续刀，本层不接 tokenizer。
pub const VOCAB_NAME: &str = "llm.vocab";

#[cfg(test)]
#[path = "../tests/archive_tests.rs"]
mod tests;
