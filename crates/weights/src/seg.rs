//! 段：权重张量的元数据、存储帧与容器帧解析。
//!
//! 段元数据（`Descriptor`/`Segment`）与容器帧本是同一件事的两半：容器帧的每条描述符就是一段
//! 的元数据，落位结果就是 `Stored`。两者原先互相 import，现在合成一个模块。

use crate::align_up;
use crate::bad;
use diagnostic::{Code, Error, Result};

use crate::Wire;
use crate::quant::Quant;
use crate::writer::{GLOBAL_CRC, MAGIC, MAX_ALIGN_LOG, SEG_CRC, VERSION};
use std::borrow::Cow;
use storage::leb::{self, Cursor, u16_at, u32_at};
use storage::sink::Sink;

closed_enum! { @coded_unsupported
    /// 段级编解码闭集：码位稳定，未知码在打开期按 `Unsupported` 拒绝。
    pub enum Codec {
        /// 原样字节（code 0）。
        Store = 0,
        /// rANS 熵编码帧（code 1）。
        Rans = 1,
        /// 五轴无损选码帧（code 2）。
        Axes = 2,
    }
}

impl Codec {
    /// 段体解码：codec 码位与解码器的对应只此一处。
    pub(crate) fn decode<'a>(self, raw: &'a [u8], want: usize) -> Result<Cow<'a, [u8]>> {
        Ok(match self {
            Self::Store => Cow::Borrowed(raw),
            Self::Rans => Cow::Owned(crate::codec::decode_rans(raw, want)?),
            Self::Axes => Cow::Owned(crate::axes::expand(raw)?),
        })
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

// —— 容器帧解析（原 `layout` 平铺进 `seg`）：固定头、描述符表、对齐落位、引用图校验与回填。
// 操作 [`Desc`]/[`Stored`] 切片，不持有游标字节流状态。——

/// 固定头长度：magic 4 + version 2 + flags 4 + align 1（count 是 LEB，不计入）。
const FIXED_HEAD: usize = 11;

/// 尾部长度：全局 CRC 4 字节。
pub(crate) const FOOTER_LEN: usize = 4;

/// 固定头：长度、魔数、版本、预留位与对齐逐项检查，不回答段布局。
pub(crate) fn fixed(bytes: &[u8]) -> Result<(u32, usize)> {
    check_prologue(bytes)?;
    check_version(u16_at(bytes, 4)?)?;
    flags_align(bytes)
}

/// 固定头的前两道：长度够读出尾部，且魔数一致。
fn check_prologue(bytes: &[u8]) -> Result<()> {
    check_len(bytes)?;
    check_magic(bytes)?;
    Ok(())
}

/// 至少固定头 + 段计数 1 字节 + 尾部。
fn check_len(bytes: &[u8]) -> Result<()> {
    if bytes.len() < FIXED_HEAD + FOOTER_LEN + 1 {
        return Err(bad(format!(
            "文件只有 {} 字节，读不出固定头、段计数与尾部",
            bytes.len()
        )));
    }
    Ok(())
}

/// 魔数。
fn check_magic(bytes: &[u8]) -> Result<()> {
    let magic = u32_at(bytes, 0)?;
    if magic != MAGIC {
        return Err(bad(format!(
            "魔数不匹配：期望 {MAGIC:#010x}，得到 {magic:#010x}"
        )));
    }
    Ok(())
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
        return Err(bad(format!("版本 {version} 已不再支持，当前为 {VERSION}")));
    }
    Ok(())
}

/// 预留位必须为 0，对齐指数不得超出上限。
fn flags_align(bytes: &[u8]) -> Result<(u32, usize)> {
    let flags = u32_at(bytes, 6)?;
    if flags & !(GLOBAL_CRC | SEG_CRC) != 0 {
        return Err(bad("flags 含非零预留位"));
    }
    let align_log = bytes[10];
    if align_log > MAX_ALIGN_LOG {
        return Err(bad(format!("align_log {align_log} 超过 {MAX_ALIGN_LOG}")));
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
    let count = usize::try_from(cur.read::<u32>()?).map_err(|_| bad("段数量超 usize"))?;
    let plausible = cur.remaining().len().saturating_div(MIN_DESC_BYTES);
    if count > plausible {
        return Err(bad("段数量超出剩余字节能容纳的规模"));
    }
    Ok(count)
}

/// 段计数与描述符；返回数据区起点（绝对偏移）。
pub(crate) fn read_descs(bytes: &[u8]) -> Result<(Vec<Desc>, usize)> {
    let mut cur = Cursor::new(&bytes[FIXED_HEAD..]);
    let count = read_count(&mut cur)?;
    let mut descs = Vec::with_capacity(count.min(MAX_DESC_PREALLOC));
    for _ in 0..count {
        descs.push(Desc::decode(&mut cur)?);
    }
    Ok((descs, FIXED_HEAD + cur.pos()))
}

/// 重算每段的绝对区间；末尾必须正好接上 footer。引用段不占数据区，区间在代表段落位后回填。
pub(crate) fn layout(
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
        return Err(bad(format!(
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
                return Err(bad(format!(
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
            return Err(bad(format!("段名 `{name}` 重复")));
        }
    }
    Ok(())
}

/// 引用图校验：悬空/越界下标、自引用、代表段是引用、元数据不一致、引用成环，一律 `Invalid`。
fn validate_refs(descs: &[Desc]) -> Result<()> {
    let mut colors = vec![0u8; descs.len()];
    for idx in 0..descs.len() {
        check_ref(idx, descs, &mut colors)?;
    }
    Ok(())
}

/// 单色 DFS：0 未访 / 1 在途 / 2 完成；回到在途节点即成环。
fn check_ref(idx: usize, descs: &[Desc], colors: &mut [u8]) -> Result<()> {
    match colors[idx] {
        2 => return Ok(()),
        1 => return Err(bad("引用段成环")),
        _ => {
            colors[idx] = 1;
            walk_ref(idx, descs, colors)?;
            colors[idx] = 2;
        }
    }
    Ok(())
}

/// 在途节点沿代表下标递归；内联段直接收口，引用段交代表段继续 DFS。
fn walk_ref(idx: usize, descs: &[Desc], colors: &mut [u8]) -> Result<()> {
    let DescBody::Ref { rep } = descs[idx].body() else {
        return Ok(());
    };
    let rep = *rep as usize;
    validate_ref_target(idx, rep, descs)?;
    check_ref(rep, descs, colors)
}

/// 引用下标与代表段元数据校验：越界、自引用、代表段是引用、取值元数据不一致一律 `Invalid`。
fn validate_ref_target(idx: usize, rep: usize, descs: &[Desc]) -> Result<()> {
    let name = descs[idx].descriptor().name.clone();
    if rep >= descs.len() {
        return Err(bad(format!("引用段 `{name}` 指向越界下标 {rep}")));
    }
    if rep == idx {
        return Err(bad(format!("引用段 `{name}` 自引用")));
    }
    let target = &descs[rep];
    if !matches!(target.body(), DescBody::Inline { .. }) {
        return Err(bad(format!("引用段 `{name}` 的代表段 `{rep}` 本身是引用")));
    }
    if !descs[idx].descriptor().same_value(target.descriptor()) {
        return Err(bad(format!(
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
    for (idx, seg) in stored.iter().enumerate() {
        let StoredInner::Ref { rep, .. } = &seg.inner else {
            continue;
        };
        let resolution = resolve_one(stored, idx, *rep)?;
        resolutions.push((idx, resolution));
    }
    Ok(resolutions)
}

/// 把代表段区间、codec 与 CRC 回填到引用段槽位。
fn apply_resolutions(stored: &mut [Stored], resolutions: Vec<Resolution>) {
    for (idx, (codec, start, len, crc)) in resolutions {
        if let StoredInner::Ref {
            codec: c,
            start: s,
            len: l,
            crc: r,
            ..
        } = &mut stored[idx].inner
        {
            *c = codec;
            *s = start;
            *l = len;
            *r = crc;
        }
    }
}

/// 取一条引用段的代表段区间快照；代表段仍是引用则拒绝。
fn resolve_one(stored: &[Stored], idx: usize, rep: usize) -> Result<(Codec, usize, usize, u32)> {
    match &stored[rep].inner {
        StoredInner::Inline {
            codec,
            start,
            len,
            crc,
        } => Ok((*codec, *start, *len, *crc)),
        StoredInner::Ref { .. } => Err(bad(format!("引用段 `{idx}` 的代表段 `{rep}` 本身是引用"))),
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
    let Some(start) = align_up(cursor, align) else {
        return Err(bad("对齐必须是 2 的幂且对齐后不溢出"));
    };
    let pad_ok = bytes
        .get(cursor..start)
        .is_some_and(|pad| pad.iter().all(|b| *b == 0));
    if !pad_ok {
        return Err(bad(format!("段 `{}` 之前的对齐填充含非零字节", meta.name)));
    }
    let len = usize::try_from(data_len).map_err(|_| bad("段长度超 usize"))?;
    let Some(end) = start.checked_add(len) else {
        return Err(bad("段区间相加溢出"));
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
