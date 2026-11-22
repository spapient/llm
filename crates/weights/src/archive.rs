//! 内存容器：绑定在输入字节上，不复制数据，负责两级 CRC 校验、段读取与 Merkle 证明。

use std::borrow::Cow;
use std::sync::OnceLock;

use crate::bad;
use diagnostic::Result;

use crate::Codec;
use crate::parse::{FOOTER_LEN, fixed, layout, read_descs};
use crate::proof::{Digest, Proof};
use crate::stored::{Stored, lossless_entry};
use crate::writer::{GLOBAL_CRC, SEG_CRC};

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

#[cfg(test)]
#[path = "../tests/archive.rs"]
mod tests;
