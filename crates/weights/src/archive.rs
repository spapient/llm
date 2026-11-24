//! 内存容器：绑定在输入字节上，不复制数据，负责两级 CRC 校验、段读取与 Merkle 证明。

use std::borrow::Cow;
use std::sync::OnceLock;

use crate::bad;
use diagnostic::Result;

use crate::Codec;
use crate::Segment;
use crate::Wire;
use crate::proof::{Digest, Proof};
use crate::seg::{FOOTER_LEN, Stored, fixed, layout, lossless_entry, read_descs};
use crate::writer::{GLOBAL_CRC, SEG_CRC};
use storage::leb::u32_at;

/// 内存容器：绑定在输入字节上，不复制数据。
#[derive(Debug)]
pub struct Archive<'a> {
    bytes: &'a [u8],
    segs: Vec<Stored>,
    /// 段名 → 段下标：名字唯一由 `layout` 保证，按名查段为 O(1)。
    idx: std::collections::HashMap<String, usize>,
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
        let idx = segs
            .iter()
            .enumerate()
            .map(|(position, seg)| (seg.name().to_owned(), position))
            .collect();
        let archive = Self {
            bytes,
            segs,
            idx,
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
        let stored = u32_at(self.bytes, footer_pos)?;
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
        self.idx
            .get(name)
            .and_then(|&position| self.segs.get(position))
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
    pub fn prove(&self, idx: usize) -> Result<Proof> {
        let leaves = self.cached_leaves()?;
        crate::proof::prove(leaves, idx)
            .ok_or_else(|| bad(format!("段下标 {idx} 越界，共 {} 段", leaves.len())))
    }

    /// 仅用下标、证明与根验证归属，不接触其他张量。
    pub fn verify(&self, idx: usize, proof: &Proof, root: &Digest) -> Result<bool> {
        let leaves = self.cached_leaves()?;
        let Some(leaf_hash) = leaves.get(idx) else {
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
    let bytes = codec.decode(raw, want)?;
    if bytes.len() != want {
        return Err(bad(format!(
            "段解出的字节数 {} 与描述符推出的 {want} 不符",
            bytes.len()
        )));
    }
    Ok(bytes)
}

/// 词表段保留名：字节容器占位；字节 → `VocabSource` 解析留给后续刀，本层不接 tokenizer。
pub const VOCAB_NAME: &str = "llm.vocab";

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

// —— 容器生命周期模式、增量补丁与外部源导入桥接 ——

/// 模式标记段的保留名。
pub const MODE_NAME: &str = "llm.mode";
/// 优化器状态段的命名空间前缀。
const OPT_PREFIX: &str = "opt.";

closed_enum! { @coded
    /// 容器生命周期模式。
    pub enum Mode {
        /// 训练断点：含优化器/动量/方差等 `opt.` 状态。
        Train = 0,
        /// 推理发布：剥离训练状态的精简产物。
        Infer = 1,
    }
}

impl Mode {
    /// 编码为可写进容器的标记段：保留名、`U8` 单轴、1 字节。
    pub fn to_segment(self) -> Result<Segment> {
        Segment::new(MODE_NAME, Wire::U8, vec![1], None, vec![self.code()])
    }
}

/// 读取模式；无标记段的旧文件按推理模式处理。
pub fn mode_of(archive: &Archive<'_>) -> Result<Mode> {
    match archive
        .get(MODE_NAME)
        .and_then(|seg| archive.read(seg).first().copied())
    {
        None => Ok(Mode::Infer),
        Some(code) => Mode::from_code(code),
    }
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

/// 把读端段还原成写端段：沿用名字/类型/形状/量化方案，`data` 是已就位的逻辑字节。
/// `release` 与补丁重建共用这一份字段装配，避免四处各拼一遍 `Segment::new`。
fn to_segment(seg: &Stored, data: Vec<u8>) -> Result<Segment> {
    Segment::new(
        seg.name(),
        seg.dtype(),
        seg.shape().to_vec(),
        seg.quant().cloned(),
        data,
    )
}

/// 把读端段还原成写端段：先加载逻辑字节，再交 [`to_segment`] 装配。
fn keep_segment(src: &Archive<'_>, seg: &Stored) -> Result<Segment> {
    let data = src.load(seg)?.into_owned();
    to_segment(seg, data)
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
        out.push(to_segment(seg, data)?);
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

pub mod import {
    //! 中性条目流导入：外部权重源 → 既有 [`Segment`] 列表，不新增第二份条目类型。

    use std::path::Path;

    use diagnostic::{Code, Error, Result};

    use crate::codec::f32_bytes;
    use crate::gguf::{Gguf, Kind, Tensor};
    use crate::safetensors::{Archive as StArchive, File as StFile};
    use crate::torch::Checkpoint;
    use crate::{Entry, Loader, Reader, Segment, Wire};

    /// 按扩展名把源文件分派到对应适配通路；不认识的后缀显式 [`Code::Unsupported`]，不猜格式、
    /// 不把块量化伪装成标量段。
    ///
    /// - `.safetensors`：只读头部的 `File` 后端，按条目按需 seek+read；
    /// - `.gguf`：整份读入后交 GGUF 解析，标量 `F32`/`F16` 直映，块量化显式拒绝；
    /// - `.bin` / `.pt`：整份读入后交 bin（ZIP_STORED + pickle 栈机）解析。
    ///
    /// # Errors
    ///
    /// 文件 IO 失败按 [`Code::Io`]；后缀未登记、读不通或类型不在映射闭集内按各通路既有诊断拒绝。
    pub fn segments_from_path(path: &Path) -> Result<Vec<Segment>> {
        match path.extension().and_then(|ext| ext.to_str()) {
            Some("safetensors") => segments_from_safetensors_file(path),
            Some("gguf") => load_gguf(path),
            Some("bin" | "pt") => load_torch(path),
            other => Err(unsupported(other)),
        }
    }

    /// 后缀未登记：不猜格式，也不把块量化伪装成标量段。
    fn unsupported(other: Option<&str>) -> Error {
        Error::new(
            Code::Unsupported,
            format!(
                "不认识的源格式 `{}`：支持 .safetensors / .gguf / .bin / .pt",
                other.unwrap_or("无扩展名")
            ),
        )
        .with("weights::import::segments_from_path")
    }

    /// GGUF 通路：整份读入 → 解析 → 中性段流。
    fn load_gguf(path: &Path) -> Result<Vec<Segment>> {
        let bytes = read_file(path)?;
        segments_from_gguf(&Gguf::parse(&bytes)?)
    }

    /// torch 检查点通路（`data.pkl` + storage）：整份读入 → 解析 → 中性段流。
    fn load_torch(path: &Path) -> Result<Vec<Segment>> {
        let bytes = read_file(path)?;
        segments_from_torch(&Checkpoint::parse(&bytes)?)
    }

    /// 整份读入文件为自有字节；GGUF/bin 解析需要整份缓冲，safetensors 条目通路不走这里。
    fn read_file(path: &Path) -> Result<Vec<u8>> {
        std::fs::read(path).map_err(|error| {
            Error::new(Code::Io, format!("读取 {} 失败：{error}", path.display()))
                .with("weights::import::read_file")
        })
    }

    /// safetensors 条目 → 中性段：字段与长度判据同源，文件与内存归档共用。
    fn push_entry_segment(out: &mut Vec<Segment>, entry: &Entry, bytes: Vec<u8>) -> Result<()> {
        let dims: Vec<u64> = entry.shape().dims().iter().map(|&d| d as u64).collect();
        out.push(segment_from_entry(
            entry.name(),
            entry.dtype(),
            &dims,
            bytes,
        )?);
        Ok(())
    }

    /// safetensors 文件 → 中性段流：`File` 后端按条目按需 seek+read，不把整份文件读进内存。
    ///
    /// # Errors
    ///
    /// 文件打开/读取失败按后端诊断；条目字段与字节不自洽时按 [`Segment::new`] 的既有规则拒绝。
    pub fn segments_from_safetensors_file(path: &Path) -> Result<Vec<Segment>> {
        let file = StFile::open(path)?;
        let entries = file.header().entries();
        let mut out = Vec::with_capacity(entries.len());
        for entry in entries {
            push_entry_segment(&mut out, entry, file.load(entry)?)?;
        }
        Ok(out)
    }

    /// safetensors 内存归档 → 中性段流：借用路径与文件路径同口径，字节零拷贝读出后再落段。
    ///
    /// # Errors
    ///
    /// 与 [`segments_from_safetensors_file`] 同一套字段/长度判据。
    pub fn segments_from_safetensors_archive(arch: &StArchive<'_>) -> Result<Vec<Segment>> {
        let entries = arch.header().entries();
        let mut out = Vec::with_capacity(entries.len());
        for entry in entries {
            push_entry_segment(&mut out, entry, arch.read(entry)?.to_vec())?;
        }
        Ok(out)
    }

    /// GGUF 归档 → 中性段流：标量 `F32`/`F16` 直映 Wire；GGML 块量化（`Q8_0`/`Q4_K`/`Q5_K`/
    /// `Q6_K`）在导入时按元素反量化成 `F32` 段。
    ///
    /// 打包字节不能原样落段：中性 `Wire` 只描述标量与打包整数，段的长度口径又是
    /// `dtype × 元素数`，把块字节按标量宽度塞进去就是伪装。反量化的代价是块格式在导入时
    /// 展开（Q4_K 每 256 元素 144 字节 → 1024 字节），换来的是段里的值可按元素校验。
    ///
    /// # Errors
    ///
    /// 块的字节数与元素数不整除、或布局不符时由 `weights` 显式拒绝；其余读取错误透传解析诊断。
    pub fn segments_from_gguf(arch: &Gguf<'_>) -> Result<Vec<Segment>> {
        arch.tensors()
            .iter()
            .map(|tensor| segment_from_gguf(arch, tensor))
            .collect()
    }

    /// 一个 GGML 张量 → 中性段。
    fn segment_from_gguf(arch: &Gguf<'_>, tensor: &Tensor) -> Result<Segment> {
        let data = gguf_bytes(arch, tensor)?;
        segment_from_entry(&tensor.name, gguf_wire(tensor.kind), &tensor.shape, data)
    }

    /// 张量类型 → 中性 Wire：标量直映，块量化反量化后一律 `F32`。
    fn gguf_wire(kind: Kind) -> Wire {
        match kind {
            Kind::F16 => Wire::F16,
            _ => Wire::F32,
        }
    }

    /// 张量值 → 段存储字节：标量取原始小端字节（不重排），块量化按元素反量化成 f32。
    fn gguf_bytes(arch: &Gguf<'_>, tensor: &Tensor) -> Result<Vec<u8>> {
        match tensor.kind {
            Kind::F32 | Kind::F16 => Ok(arch.raw(tensor)?.to_vec()),
            _ => Ok(f32_bytes(&arch.floats(&tensor.name)?)),
        }
    }

    /// torch 检查点 → 同一中性段流；读不通的归档或对象流在 `torch` 解析层已显式失败，
    /// 这里只做字段归一，不再补第二份容错。
    ///
    /// # Errors
    ///
    /// 条目字节区间或字段不自洽时按 [`Segment::new`] 既有规则拒绝。
    pub fn segments_from_torch(file: &Checkpoint<'_>) -> Result<Vec<Segment>> {
        let mut out = Vec::with_capacity(file.tensors().len());
        for tensor in file.tensors() {
            let raw = file.data(tensor)?.to_vec();
            out.push(segment_from_entry(
                tensor.name(),
                tensor.dtype(),
                tensor.shape(),
                raw,
            )?);
        }
        Ok(out)
    }

    /// 外部源字段 → 中性段：形状轴长统一转 `u64`；量化方案不在导入侧挂载（标度配对仍是
    /// 源格式自己的条目）。类型/长度不自洽时透传 [`Segment::new`] 诊断。
    fn segment_from_entry(name: &str, wire: Wire, dims: &[u64], bytes: Vec<u8>) -> Result<Segment> {
        Segment::new(name, wire, dims.to_vec(), None, bytes)
    }
}

#[cfg(test)]
#[path = "../tests/lifecycle.rs"]
mod lifecycle_tests;
#[cfg(test)]
#[path = "../tests/archive.rs"]
mod tests;
