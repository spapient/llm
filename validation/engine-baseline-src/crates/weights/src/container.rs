//! 容器核心：权重条目 [`Entry`]、头部 [`Header`]、读端 trait、度量与通用 ZIP 读取。
//! 名字唯一、区间不重叠、长度/类型判据都在这里收口，格式适配层只做外部格式映射，
//! 不重复这些判据。

use std::collections::HashMap;

use crate::seg::{Stored, lossless_entry};
use diagnostic::{Code, Error, Result};
use storage::snapshot::{Manifest, ManifestEntry};
use tensor::{DType, Shape};

use storage::bad;
use storage::leb::{Cursor, slice, u16_at, u32_at, u64_at};

const SITE: &str = "weights::Header::new";

pub(crate) use storage::align_up;

/// 一个权重条目：名字、元素类型、形状与它在数据段里的字节区间。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Entry {
    name: String,
    dtype: DType,
    shape: Shape,
    offset: usize,
}

impl Entry {
    /// 由名字、类型与形状构造（偏移默认 0，用 [`Entry::at`] 指定）。
    pub fn new(name: impl Into<String>, dtype: DType, shape: Shape) -> Self {
        Self {
            name: name.into(),
            dtype,
            shape,
            offset: 0,
        }
    }

    /// 指定在数据段里的起始偏移。
    #[must_use]
    pub fn at(mut self, offset: usize) -> Self {
        self.offset = offset;
        self
    }

    /// 条目名（权重文件里的键）。
    pub fn name(&self) -> &str {
        &self.name
    }

    /// 元素类型。
    pub fn dtype(&self) -> DType {
        self.dtype
    }

    /// 形状。
    pub fn shape(&self) -> &Shape {
        &self.shape
    }

    /// 起始偏移。
    pub fn offset(&self) -> usize {
        self.offset
    }

    /// 字节长度（由形状与类型推出）。
    pub fn len(&self) -> usize {
        self.shape.element_bytes(self.dtype)
    }

    /// 是否为零字节条目。
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// 结束偏移（不含）。
    pub fn end(&self) -> usize {
        self.offset.saturating_add(self.len())
    }
}

/// 权重文件的头部：条目表 + 数据段起点。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Header {
    entries: Vec<Entry>,
    idx: HashMap<String, usize>,
    data_start: usize,
}

impl Header {
    /// 校验并构造：名字唯一、区间按偏移有序且不重叠、起点不早于数据段。
    pub fn new(entries: Vec<Entry>, data_start: usize) -> Result<Self> {
        let mut sorted = entries;
        sorted.sort_by_key(Entry::offset);
        check_start(&sorted, data_start)?;
        check_pairs(&sorted)?;
        let idx = build_idx(&sorted)?;
        Ok(Self {
            entries: sorted,
            idx,
            data_start,
        })
    }

    /// 全部条目，按偏移排序。
    pub fn entries(&self) -> &[Entry] {
        &self.entries
    }

    /// 按名字查条目（一次解析期建名索引，之后每次查询 O(1)）。
    pub fn get(&self, name: &str) -> Option<&Entry> {
        self.idx
            .get(name)
            .and_then(|&position| self.entries.get(position))
    }

    /// 数据段起点（头部长度）。
    pub fn data_start(&self) -> usize {
        self.data_start
    }

    /// 数据段实际占用的字节数：最后一个条目结束位置减数据段起点。
    pub fn data_bytes(&self) -> usize {
        self.entries
            .last()
            .map_or(0, |entry| entry.end().saturating_sub(self.data_start))
    }
}

// —— 检查点快照的接入点：字节格式归 L2 `storage::snapshot`，本 crate 只回答"进入字节的四个字段
// 从哪来"与"读回来的条目表怎么复判"，于是快照不必反向认识 `Header`，`storage → weights → storage`
// 的环也就不存在。——
impl ManifestEntry for Entry {
    fn name(&self) -> &str {
        &self.name
    }

    fn dtype(&self) -> DType {
        self.dtype
    }

    fn dims(&self) -> &[usize] {
        self.shape.dims()
    }

    fn offset(&self) -> usize {
        self.offset
    }

    fn from_parts(name: String, dtype: DType, dims: Vec<usize>, offset: usize) -> Self {
        Self::new(name, dtype, Shape::new(dims)).at(offset)
    }
}

impl Manifest for Header {
    type Entry = Entry;

    fn entries(&self) -> &[Entry] {
        &self.entries
    }

    fn data_start(&self) -> usize {
        self.data_start
    }

    /// 解码期复判：字节里的条目表必须再过一遍 [`Header::new`] 的三条判据。
    fn build(entries: Vec<Entry>, data_start: usize) -> Result<Self> {
        Self::new(entries, data_start)
    }
}

/// 相邻两条目（已按偏移排序）不能重叠：重叠意味着后者的起点落在前者的字节区间里。
fn check_overlap(left: &Entry, right: &Entry) -> Result<()> {
    if left.end() <= right.offset() {
        return Ok(());
    }
    Err(Error::new(
        Code::Invalid,
        format!(
            "条目 `{}` 的区间与 `{}` 重叠（{} > {}）",
            left.name(),
            right.name(),
            left.end(),
            right.offset()
        ),
    )
    .with(SITE))
}

/// 每条目起点不得早于数据段起点。
fn check_start(sorted: &[Entry], data_start: usize) -> Result<()> {
    for entry in sorted {
        if entry.offset < data_start {
            return Err(Error::new(
                Code::Invalid,
                format!(
                    "条目 `{}` 的偏移 {} 早于数据段起点 {data_start}",
                    entry.name(),
                    entry.offset
                ),
            )
            .with(SITE));
        }
    }
    Ok(())
}

/// 相邻两条目（已按偏移排序）逐一校验区间不重叠。
fn check_pairs(sorted: &[Entry]) -> Result<()> {
    for pair in sorted.windows(2) {
        check_overlap(&pair[0], &pair[1])?;
    }
    Ok(())
}

/// 名字 → 序号索引；重名报错。
fn build_idx(sorted: &[Entry]) -> Result<HashMap<String, usize>> {
    let mut idx = HashMap::with_capacity(sorted.len());
    for (position, entry) in sorted.iter().enumerate() {
        let name = entry.name();
        if idx.insert(name.to_owned(), position).is_some() {
            return Err(Error::new(Code::Invalid, format!("权重条目 `{name}` 重复")).with(SITE));
        }
    }
    Ok(idx)
}

/// 零拷贝后端：把条目字节**借**出来（内存切片、映射或测试夹具）。
pub trait Reader {
    /// 读出条目的原始字节；长度必须与 `entry.len()` 一致。
    fn read(&self, entry: &Entry) -> Result<&[u8]>;
}

/// 按需后端：把条目字节**拷进**调用方给的缓冲区（文件、压缩包、远端分块）。
pub trait Loader {
    /// 把条目字节写进 `out`；`out` 必须正好等于 `entry.len()` 字节。
    ///
    /// # Errors
    ///
    /// 长度不符时返回 [`Code::Invalid`]（这是调用方的错，不能截断也不能补零）；
    /// 读不出来时由具体后端给出（文件是 [`Code::Io`]，借用的后端是 [`Code::Internal`]）。
    fn copy_into(&self, entry: &Entry, out: &mut [u8]) -> Result<()>;

    /// 分配并读出条目字节。
    fn load(&self, entry: &Entry) -> Result<Vec<u8>> {
        let mut buf = vec![0u8; entry.len()];
        self.copy_into(entry, &mut buf)?;
        Ok(buf)
    }
}

/// 把一段小端 f32 条目字节解包成 `Vec<f32>`；字节数必须是 4 的整数倍。
///
/// 元素解包是权重条目进入内存缓冲的唯一入口：读端不另写 `from_le_bytes` 循环，
/// 半字节目节在此报 [`Code::Invalid`]，不带进装配期。
///
/// # Errors
///
/// 余字节非零时返回 [`Code::Invalid`]（条目不是完整 f32）。
pub fn decode_f32(bytes: &[u8]) -> Result<Vec<f32>> {
    let (words, rest) = bytes.as_chunks::<4>();
    if !rest.is_empty() {
        return Err(Error::new(
            Code::Invalid,
            format!(
                "条目长度 {} 不是 f32 的整数倍（余 {} 字节）",
                bytes.len(),
                rest.len()
            ),
        )
        .with("weights::decode_f32"));
    }
    Ok(words.iter().map(|word| f32::from_le_bytes(*word)).collect())
}

/// 把一条 F32 条目解码进驻留 `f32` 缓冲：装载期一次性读出并解包，生成循环内不再读文件、
/// 也不二次分配元素缓冲。
pub fn load_f32(entry: &Entry, loader: &impl Loader) -> Result<Vec<f32>> {
    let bytes = loader.load(entry)?;
    decode_entry_f32(entry, &bytes)
}

/// 借用后端（内存归档、只读映射）的对称入口：条目字节从 [`Reader`] 借出来直接解包，
/// 省掉 [`Loader::load`] 那一次 `Vec<u8>` 字节拷贝；f32 主机序解包仍是必须的，驻留缓冲
/// 拥有形态不变。类型/边界判据与 [`load_f32`] 走同一个 [`decode_entry_f32`]，两条路径不各写一份。
pub fn read_f32(entry: &Entry, reader: &impl Reader) -> Result<Vec<f32>> {
    let bytes = reader.read(entry)?;
    decode_entry_f32(entry, bytes)
}

/// 按条目类型把一段已读回的字节解包成 `f32` 驻留缓冲；`Reader` 借用路径与 `Loader` 拷贝
/// 路径共用这一份类型/边界判据。
pub fn decode_entry_f32(entry: &Entry, bytes: &[u8]) -> Result<Vec<f32>> {
    if entry.dtype() != DType::F32 {
        return Err(Error::new(
            Code::Unsupported,
            format!("条目 `{}` 不是 F32，不能进 f32 驻留缓冲", entry.name()),
        )
        .with("weights::decode_entry_f32"));
    }
    decode_f32(bytes)
}

/// 从游标读一个线上类型码并解析成 [`DType`]：seg 描述符与 quant 方案共用这一份解码，
/// 避免两份"u32 → u16 → from_code"的同构步骤漂移。
pub(crate) fn read_wire(cur: &mut Cursor<'_>) -> Result<DType> {
    let code = cur.read::<u32>()?;
    let code = u16::try_from(code)
        .map_err(|_| Error::new(Code::Invalid, "wire code 超出 u16").with("weights"))?;
    DType::from_code(code)
        .ok_or_else(|| Error::new(Code::Invalid, "未知 wire code").with("weights"))
}

/// 缓冲区长度必须与条目声明一致：短了会读出半个张量，长了会把相邻条目混进来。
pub(crate) fn check_capacity(entry: &Entry, out: &[u8]) -> Result<()> {
    if out.len() == entry.len() {
        return Ok(());
    }
    Err(Error::new(
        Code::Invalid,
        format!(
            "缓冲区 {} 字节，条目 `{}` 需要 {} 字节",
            out.len(),
            entry.name(),
            entry.len()
        ),
    )
    .with("weights::Loader::copy_into"))
}

// —— 容器度量：条目、参数、体积与无损判据 ——

/// E8 体积门：`ratio_param ≤ 1/24`（LLD §E8「24× 无损」）。
pub const E8_RATIO: f64 = 1.0 / 24.0;

/// 度量累加器：逐段把 numel、FP16 参考体积、无损判定与 REF 命中/节省字节收拢到一处。
pub(crate) struct MetricsAcc {
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
    pub(crate) fn add(&mut self, seg: &Stored) -> Result<()> {
        let numel = tensor::product(seg.shape());
        self.params = self.params.saturating_add(numel);
        self.v_fp16 = self.v_fp16.saturating_add(numel.saturating_mul(2));
        self.lossless = self.lossless && lossless_entry(seg);
        if seg.is_ref() {
            self.ref_hits += 1;
            let logical = u64::try_from(seg.logical_len()?).unwrap_or(u64::MAX);
            self.ref_saved_bytes = self.ref_saved_bytes.saturating_add(logical);
        }
        Ok(())
    }

    /// 收口为 [`Metrics`]：条目数与容器体积由调用方提供，体积比在此处现算。
    pub(crate) fn finish(self, entries: usize, v_artifact: usize) -> Metrics {
        let ratio = if self.v_fp16 == 0 {
            0.0
        } else {
            v_artifact as f64 / self.v_fp16 as f64
        };
        Metrics {
            entries,
            params: self.params,
            v_artifact,
            v_fp16: self.v_fp16,
            ratio,
            lossless: self.lossless,
            ref_hits: self.ref_hits,
            ref_saved_bytes: self.ref_saved_bytes,
        }
    }
}

/// 容器度量：条目数、参数总数、体积对比与无损口径；全部在打开容器后现算。
#[derive(Debug, Clone)]
pub struct Metrics {
    pub(crate) entries: usize,
    pub(crate) params: u64,
    pub(crate) v_artifact: usize,
    pub(crate) v_fp16: u64,
    pub(crate) ratio: f64,
    pub(crate) lossless: bool,
    pub(crate) ref_hits: usize,
    pub(crate) ref_saved_bytes: u64,
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

    /// 参数级体积比 `V_artifact / (params × 2)`，即 LLD §E8 的 `ratio_param`。
    pub fn ratio_param(&self) -> f64 {
        let denom = self.params.saturating_mul(2);
        if denom == 0 {
            return 0.0;
        }
        self.v_artifact as f64 / denom as f64
    }

    /// E8 裁决：24× 体积门与逐位无损**同时**成立才算通过（LLD §E8「必须且不可降级」）。
    pub fn e8_pass(&self) -> bool {
        self.params > 0 && self.lossless && self.ratio_param() <= E8_RATIO
    }

    /// 无损标志：所有条目都不带量化方案，且 codec 属于无损的 `Store`/`Rans`/`Axes`。
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

// —— 通用 ZIP 读取：STORED 条目与 ZIP64 回退 ——

// —— ZIP 归档解析（原 `zip`）：STORED 条目，含 ZIP64 回退 ——
//
// 每个函数对应 APPNOTE 的一段：`directory`（尾部 EOCD 或 ZIP64 回退）、`entry`（一条中央目录
// 记录）、`sizes`（32 位尺寸字段与其 ZIP64 槽位）、`zip64_extra`（extra 字段头遍历）。

const EOCD_SIG: u32 = 0x0605_4B50;
const CD_SIG: u32 = 0x0201_4B50;
const LOCAL_SIG: u32 = 0x0403_4B50;
const ZIP64_SIG: u32 = 0x0606_4B50;
const EOCD64_LOCATOR_SIG: u32 = 0x0706_4B50;
/// extra 字段里 ZIP64 的 header id。
const ZIP64_EXTRA_ID: u16 = 0x0001;
const STORED: u16 = 0;
/// 条目最小 46+1 字节；预分配上限 4096，谎报计数先按剩余字节粗查。
const MIN_ENTRY_BYTES: usize = 47;
const MAX_ENTRY_PREALLOC: usize = 4096;
/// 中央目录尺寸字段被截断时的 32 位哨兵。
const WIDE_SENTINEL: u32 = u32::MAX;

#[derive(Debug, Clone)]
pub(crate) struct ZipEntry {
    name: String,
    method: u16,
    start: usize,
    len: usize,
}

pub(crate) struct Zip<'a> {
    bytes: &'a [u8],
    entries: Vec<ZipEntry>,
}

impl<'a> Zip<'a> {
    pub(crate) fn parse(bytes: &'a [u8]) -> Result<Self> {
        let (mut at, total) = directory(bytes)?;
        if total > bytes.len() / MIN_ENTRY_BYTES {
            return Err(bad("条目数超出归档能容纳的规模"));
        }
        let mut entries = Vec::with_capacity(total.min(MAX_ENTRY_PREALLOC));
        for _ in 0..total {
            let (entry, next) = entry(bytes, at)?;
            entries.push(entry);
            at = next;
        }
        Ok(Self { bytes, entries })
    }

    #[must_use]
    pub(crate) fn get_suffix(&self, suffix: &str) -> Option<&ZipEntry> {
        self.entries.iter().find(|e| e.name.ends_with(suffix))
    }

    pub(crate) fn data_suffix(&self, suffix: &str) -> Result<&'a [u8]> {
        let entry = self
            .get_suffix(suffix)
            .ok_or_else(|| bad(format!("无后缀条目 `{suffix}`")))?;
        self.data(entry)
    }

    fn data(&self, entry: &ZipEntry) -> Result<&'a [u8]> {
        if entry.method != STORED {
            return Err(bad("条目经过压缩，零依赖读取器只支持 STORED"));
        }
        slice(self.bytes, entry.start, entry.len)
    }
}

/// 中央目录起点与条目总数：EOCD 的两个字段任一为 `0xFFFF…` 就走 ZIP64 回退。
fn directory(bytes: &[u8]) -> Result<(usize, usize)> {
    let eocd = locate_eocd(bytes)?;
    narrow_or_wide(
        bytes,
        eocd,
        u32_at(bytes, eocd + 16)?,
        u16_at(bytes, eocd + 10)?,
    )
}

/// 32 位字段够用就按它换算，任一被截断（`0xFFFF…`）就走 ZIP64 回退。
fn narrow_or_wide(bytes: &[u8], eocd: usize, wide: u32, total: u16) -> Result<(usize, usize)> {
    if wide == WIDE_SENTINEL || total == u16::MAX {
        return zip64_directory(bytes, eocd);
    }
    Ok((
        narrow(u64::from(wide), "中央目录偏移超出 usize")?,
        usize::from(total),
    ))
}

/// ZIP64 回退：定位器在 EOCD 前 20 字节，定位器 +8 指向 64 位 EOCD；
/// 条目数与该 EOCD 的 +32、中央目录偏移在 +48。
fn zip64_directory(bytes: &[u8], eocd: usize) -> Result<(usize, usize)> {
    let z64 = zip64_eocd(bytes, eocd)?;
    let total = wide_value(bytes, z64 + 32, "条目数超出 usize")?;
    let at = wide_value(bytes, z64 + 48, "中央目录偏移超出 usize")?;
    Ok((at, total))
}

/// 定位器的第一跳：它在 EOCD 前 20 字节，定位器 +8 指向 64 位 EOCD。
fn zip64_eocd(bytes: &[u8], eocd: usize) -> Result<usize> {
    let locator = locator_at(eocd)?;
    check_sig(bytes, locator, EOCD64_LOCATOR_SIG, "ZIP64 定位器签名不符")?;
    let z64 = wide_value(bytes, locator + 8, "ZIP64 EOCD 偏移超出 usize")?;
    check_sig(bytes, z64, ZIP64_SIG, "ZIP64 EOCD 签名不符")?;
    Ok(z64)
}

/// 定位器起点：EOCD 前 20 字节，越界说明这个归档没有 ZIP64 段。
fn locator_at(eocd: usize) -> Result<usize> {
    match eocd.checked_sub(20) {
        Some(at) => Ok(at),
        None => Err(bad("ZIP64 定位器不完整")),
    }
}

/// 一个 64 位字段换算成 `usize`：越界即错，不截断。
fn wide_value(bytes: &[u8], at: usize, msg: &str) -> Result<usize> {
    narrow(u64_at(bytes, at)?, msg)
}

/// 64 位真值收窄成 `usize`：收不下就显式失败。
fn narrow(value: u64, msg: &str) -> Result<usize> {
    usize::try_from(value).map_err(|_| bad(msg))
}

fn check_sig(bytes: &[u8], at: usize, sig: u32, msg: &str) -> Result<()> {
    if u32_at(bytes, at)? == sig {
        return Ok(());
    }
    Err(bad(msg))
}

/// 一条中央目录记录：名字、方法、数据区间与下一条记录的起点。
/// 记录长 46 字节 + 名字 + extra + 注释依次排开；区间必须落在归档内，否则显式失败。
fn entry(bytes: &[u8], at: usize) -> Result<(ZipEntry, usize)> {
    let lens = Lens::of(cd_head(bytes, at)?)?;
    let (csize, local) = sizes(bytes, at, lens.name)?;
    build(bytes, at, lens, csize, local)
}

/// 固定 46 字节头：签名对得上才有后面的字段，所以先整块切出来。
fn cd_head(bytes: &[u8], at: usize) -> Result<&[u8]> {
    let fixed = slice(bytes, at, 46)?;
    check_sig(fixed, 0, CD_SIG, "中央目录条目签名不符")?;
    Ok(fixed)
}

/// 装配一条记录：名字、数据区间与下一条记录的起点。
fn build(
    bytes: &[u8],
    at: usize,
    lens: Lens,
    csize: u64,
    local: usize,
) -> Result<(ZipEntry, usize)> {
    let name = entry_name(bytes, at, lens.name)?;
    let (start, len) = span(bytes, local, csize)?;
    let next = lens.next(at)?;
    Ok((
        ZipEntry {
            name,
            method: lens.method,
            start,
            len,
        },
        next,
    ))
}

/// 固定头里的方法码与三段长度：条目名、extra、注释依次排开。
struct Lens {
    method: u16,
    name: usize,
    extra: usize,
    comment: usize,
}

impl Lens {
    fn of(fixed: &[u8]) -> Result<Self> {
        Ok(Self {
            method: u16_at(fixed, 10)?,
            name: usize::from(u16_at(fixed, 28)?),
            extra: usize::from(u16_at(fixed, 30)?),
            comment: usize::from(u16_at(fixed, 32)?),
        })
    }

    /// 下一条记录的起点：46 字节固定头 + 名字 + extra + 注释。
    fn next(&self, at: usize) -> Result<usize> {
        match at.checked_add(46 + self.name + self.extra + self.comment) {
            Some(next) => Ok(next),
            None => Err(bad("中央目录偏移溢出")),
        }
    }
}

/// 条目名：中央目录里的名字是 UTF-8 闭集，收不下就是归档坏了。
fn entry_name(bytes: &[u8], at: usize, len: usize) -> Result<String> {
    String::from_utf8(slice(bytes, at + 46, len)?.to_vec()).map_err(|_| bad("条目名不是 UTF-8"))
}

/// 数据区间（起点、长度）：本地头给出起点，区间必须落在归档内。
fn span(bytes: &[u8], local: usize, csize: u64) -> Result<(usize, usize)> {
    let start = local_start(bytes, local)?;
    let len = narrow(csize, "条目长度超出 usize")?;
    if start.checked_add(len).is_none_or(|stop| stop > bytes.len()) {
        return Err(bad("条目数据超出归档"));
    }
    Ok((start, len))
}

/// 中央目录的尺寸字段：32 位真值，或被截断成 `0xFFFF_FFFF` 时回退到 ZIP64 extra。
/// 真值按 APPNOTE 的固定顺序（原长、压缩长、本地头偏移）落在 extra 里，且**只**收下被
/// 截断的那些——所以槽位是「前面被截断的字段个数」，不是字段序号。返回（压缩长，本地头偏移）。
fn sizes(bytes: &[u8], at: usize, name_len: usize) -> Result<(u64, usize)> {
    let extra = at + 46 + name_len;
    let extra_len = usize::from(u16_at(bytes, at + 30)?);
    let mut slot = 0usize;
    let mut wide = [0u64; 3];
    for (idx, offset) in [24, 20, 42].into_iter().enumerate() {
        wide[idx] = wide_field(
            bytes,
            at + offset,
            Extra {
                extra,
                extra_len,
                slot: &mut slot,
            },
        )?;
    }
    Ok((wide[1], narrow(wide[2], "本地头偏移超出 usize")?))
}

/// ZIP64 extra 字段体的位置、声明长度与「前面截断过几个字段」的槽位。
struct Extra<'a> {
    extra: usize,
    extra_len: usize,
    slot: &'a mut usize,
}

/// 一个尺寸字段：32 位真值，或被截断成 `0xFFFF_FFFF` 时按槽位读 ZIP64 extra。
fn wide_field(bytes: &[u8], at: usize, extra: Extra<'_>) -> Result<u64> {
    let narrow = u32_at(bytes, at)?;
    if narrow != WIDE_SENTINEL {
        return Ok(u64::from(narrow));
    }
    let slot = *extra.slot;
    *extra.slot = slot + 1;
    zip64_extra(bytes, extra.extra, extra.extra_len, slot)
}

/// 本地头起点：签名对得上，30 字节固定段 + 名字 + extra 之后就是数据。
fn local_start(bytes: &[u8], local: usize) -> Result<usize> {
    check_sig(bytes, local, LOCAL_SIG, "本地头签名不符")?;
    let name_len = usize::from(u16_at(bytes, local + 26)?);
    let extra_len = usize::from(u16_at(bytes, local + 28)?);
    let Some(start) = local.checked_add(30 + name_len + extra_len) else {
        return Err(bad("本地头偏移溢出"));
    };
    Ok(start)
}

/// ZIP64 extra 里第 `field` 个 8 字节真值：逐个 extra 字段头（id u16 + 长 u16）往前挪，
/// 找到 id 0x0001 就按槽位取；声明长度装不下该槽位显式失败。
pub(crate) fn zip64_extra(
    bytes: &[u8],
    extra_start: usize,
    extra_len: usize,
    field: usize,
) -> Result<u64> {
    let stop = extra_start.saturating_add(extra_len);
    let mut at = extra_start;
    while at.checked_add(4).is_some_and(|end| end <= stop) {
        let id = u16_at(bytes, at)?;
        let size = usize::from(u16_at(bytes, at + 2)?);
        let body = at + 4;
        if id == ZIP64_EXTRA_ID {
            return slot_value(bytes, body, size, field);
        }
        at = body.saturating_add(size);
    }
    Err(bad("缺少 ZIP64 extra 字段"))
}

/// 第 `field` 个 8 字节真值：声明长度装不下该槽位就显式失败（截断不许读成 0）。
fn slot_value(bytes: &[u8], body: usize, size: usize, field: usize) -> Result<u64> {
    let end = field
        .checked_add(1)
        .and_then(|slots| slots.checked_mul(8))
        .ok_or_else(|| bad("ZIP64 字段序号溢出"))?;
    if end > size {
        return Err(bad("ZIP64 extra 声明长度不足，字段被截断"));
    }
    u64_at(bytes, body.saturating_add(end - 8))
}

/// EOCD 在文件尾部 22 + 65535 字节窗口内反向查找；窗口外一律拒（不猜其他签名）。
fn locate_eocd(bytes: &[u8]) -> Result<usize> {
    let from = bytes.len().saturating_sub(22 + 65535);
    let mut at = bytes.len().saturating_sub(4);
    while at >= from {
        if u32_at(bytes, at)? == EOCD_SIG {
            return Ok(at);
        }
        if at == 0 {
            break;
        }
        at -= 1;
    }
    Err(bad("找不到 ZIP 签名，可能不是 ZIP 归档"))
}
