//! 沉降账本：LLD §4.7 的 `config.json`——逐张量把"落盘后到底占了多少字节、为什么用这个编码"
//! 写成一份可复核的账。
//!
//! 三条取舍写在这里，因为它们是判断这份账可不可信的全部依据：
//!
//! - **量的是落盘字节，不是写入意图**：条目从 [`Archive::parse`] 之后的段表与帧里现算，于是
//!   "写的时候打算压"与"最后确实省了"不会混为一谈；写侧的决策不参与记账。
//! - **`verify_bitwise_equal` 由解码往返判定**：`load` 成功且长度等于描述符给出的逻辑长度才算
//!   逐位还原；解不开就是 [`Reason::Mismatch`]（拒绝），而不是"没量到"。
//! - **`volume_ratio` 只算容器编码这一层**：量化省下的位宽记在 `quant_bits` 上，不重复计入体积比，
//!   否则一次 4 位量化会让"熵编码到底省了多少"变成无法回答的问题。
//!
//! [`Reason`] 是闭集。LLD §4.7 列了六个值（`raw` / `rans` / `below_gain` / `incompressible` /
//! `passthrough:small` / `fallback:mismatch`），本容器另有两种 LLD 未列、却必须如实命名的形态：
//! 轴字典帧（`axes`）与去重引用（`deduplicated`）。把前者记成 `rans` 是把事实说错，把后者并进
//! `raw` 是让"零字节"看起来像"没压动"——两者都会让账本失去复核价值，故显式登记。

use std::fmt::Write as _;

use diagnostic::Result;

use crate::axes;
use crate::json;
use crate::ledger::{self, SCHEMA};
use crate::settle_reason::{bits_per, entropy_of, layer_of, reason_of};
use crate::settle_text::{escape, verify_root};
use crate::stored::lossless_entry;
use crate::{Archive, Codec, Stored};

/// 小于该字节数的段不值得编码，原样落盘（`passthrough:small`）。
pub const SMALL_BYTES: usize = 128;

/// 至少省下这个比例才算真压了；不到就是 `below_gain`。
pub const GAIN_FLOOR: f64 = 0.02;

/// 逻辑字节的熵达到这个值（bit/byte）即视为没有冗余（`incompressible`）。
pub const MAX_ENTROPY: f64 = 7.9;

/// `codec_reason` 闭集：**为什么**这一条占了现在这些字节。
///
/// 判别值即 `NAMES` 的下标，于是"名字只有一份表"：增删变体时编译器会要求同时改表长，
/// 不会出现"加了变体忘了写名字"那种只有账本产物才看得见的偏差。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum Reason {
    /// `store` 帧，且内容有冗余但没压——写侧没开压缩。
    Raw = 0,
    /// `rans` 帧，且省下超过 [`GAIN_FLOOR`]。
    Rans = 1,
    /// `rans` 帧，但省下的不足以抵一次重编码的代价。
    BelowGain = 2,
    /// `store` 帧，熵已到 [`MAX_ENTROPY`]：内容真的没有冗余。
    Incompressible = 3,
    /// 段本身小于 [`SMALL_BYTES`]：编码的固定开销比省下的多。
    Small = 4,
    /// 解码往返不是逐位：压缩形态没能还原，必须拒绝。
    Mismatch = 5,
    /// 五轴无损选码帧（本容器特有形态）。
    Axes = 6,
    /// 去重引用：字节存在代表段里，本条目自身零字节。
    Deduplicated = 7,
}

/// 原因码的规范短名，下标即判别值：这是账本格式的一部分，不是排版选择。
const NAMES: [&str; 8] = [
    "raw",
    "rans",
    "below_gain",
    "incompressible",
    "passthrough:small",
    "fallback:mismatch",
    "axes",
    "deduplicated",
];

impl Reason {
    /// 账本里的字符串写法（闭集，码位稳定）。
    pub const fn as_str(self) -> &'static str {
        NAMES[self as usize]
    }
}

/// 证书列：这条量的压缩有没有一张自洽的证书背书（LLD §4.7 的账本要说清「凭什么这么压」）。
///
/// 判别值即 `CERT_NAMES` 的下标，与 [`Reason`] 同一口径：名字只有一份表。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum Cert {
    /// 不是秩轴段：这条量没经过谱隙证书，其余轴本来就不签发。
    None = 0,
    /// 秩轴段且证书自洽：界是 L0 诊断、`ε` 来自实测残差，不变式全部通过。
    Diagnostic = 1,
    /// 秩轴段却读不出自洽证书：自称压了但背书缺失或自相矛盾。
    Refused = 2,
}

/// 证书列的规范短名，下标即判别值。
const CERT_NAMES: [&str; 3] = ["none", "l0_diagnostic", "refused"];

impl Cert {
    /// 账本里的字符串写法（闭集，码位稳定）。
    pub const fn as_str(self) -> &'static str {
        CERT_NAMES[self as usize]
    }

    /// 是否构成拒压：证书不是装饰，矛盾就退回原样。
    pub const fn refused(self) -> bool {
        matches!(self, Self::Refused)
    }
}

/// 一张量的账：全部字段都在 [`Settlement::settle`] 里由落盘字节现算。
#[derive(Clone, Debug)]
pub struct Settled {
    pub(crate) name: String,
    pub(crate) layer: Option<u32>,
    pub(crate) raw_len: usize,
    pub(crate) volume: usize,
    pub(crate) entropy: f64,
    pub(crate) quant: Option<(u32, u32)>,
    pub(crate) bitwise: bool,
    pub(crate) lossless: bool,
    pub(crate) reason: Reason,
    pub(crate) cert: Cert,
}

impl Settled {
    /// 张量名（与段名同一份）。
    pub fn name(&self) -> &str {
        &self.name
    }

    /// 名字里第一个整数分量；没有就是 `None`（嵌入、词表这类不带层号的张量）。
    pub fn layer(&self) -> Option<u32> {
        self.layer
    }

    /// 逻辑长度：这张量不压缩、不解码时该占的字节数。
    pub fn raw_len(&self) -> usize {
        self.raw_len
    }

    /// 落盘字节；去重引用为 0（字节在代表段里）。
    pub fn volume(&self) -> usize {
        self.volume
    }

    /// 体积比 `volume / raw_len`；`raw_len` 为 0 时记 1（空张量既没省也没涨）。
    pub fn volume_ratio(&self) -> f64 {
        if self.raw_len == 0 {
            1.0
        } else {
            self.volume as f64 / self.raw_len as f64
        }
    }

    /// 逻辑字节的实测熵（bit/byte）；空段记 0。
    pub fn entropy(&self) -> f64 {
        self.entropy
    }

    /// 每元素实际占的位数与组宽；未量化时 `None`，组宽 0 表示 per-tensor。
    pub fn quant(&self) -> Option<(u32, u32)> {
        self.quant
    }

    /// 解码往返是否逐位。
    pub fn bitwise(&self) -> bool {
        self.bitwise
    }

    /// 是否无损：与容器度量读同一个谓词，且往返逐位。
    pub fn lossless(&self) -> bool {
        self.lossless
    }

    /// 为什么是这个编码方式。
    pub fn reason(&self) -> Reason {
        self.reason
    }

    /// 这张量是否带一张自洽的谱隙证书。
    pub fn cert(&self) -> Cert {
        self.cert
    }

    /// 账本里的判定：确实比原样小、且证书不自相矛盾，才叫 `compress`。
    ///
    /// 「省了字节」是度量事实，「有证书背书」是判决前提，两者缺一都退回 `reject`——否则一条
    /// 自称秩压缩却读不出证书的段，会因为字节更小就在账本上冒充「压成了」。
    pub fn decision(&self) -> &'static str {
        if self.cert.refused() || self.volume >= self.raw_len {
            "reject"
        } else {
            "compress"
        }
    }
}

/// 整份账：文件长度与逐张量条目。
#[derive(Clone, Debug)]
pub struct Settlement {
    file_len: usize,
    tensors: Vec<Settled>,
}

impl Settlement {
    /// 由落盘字节现算整份账；容器不合法就是错误，不退回"空账"。
    pub fn settle(bytes: &[u8]) -> Result<Self> {
        let archive = Archive::parse(bytes)?;
        let mut tensors = Vec::with_capacity(archive.segments().len());
        for seg in archive.segments() {
            tensors.push(settle_one(&archive, seg)?);
        }
        Ok(Self {
            file_len: bytes.len(),
            tensors,
        })
    }

    /// 文件字节数（含头部、对齐填充与两级 CRC）。
    pub fn file_len(&self) -> usize {
        self.file_len
    }

    /// 逐张量条目，顺序即段表顺序。
    pub fn tensors(&self) -> &[Settled] {
        &self.tensors
    }

    /// 全量体积比 `Σvolume / Σraw_len`；空账记 1（没有可省的字节）。
    pub fn total_ratio(&self) -> f64 {
        let raw: usize = self.tensors.iter().map(Settled::raw_len).sum();
        if raw == 0 {
            return 1.0;
        }
        let volume: usize = self.tensors.iter().map(Settled::volume).sum();
        volume as f64 / raw as f64
    }

    /// 整份账是否无损：每条都得无损；空账按无损计（与容器度量同口径）。
    pub fn lossless(&self) -> bool {
        self.tensors.iter().all(Settled::lossless)
    }

    /// 规范 JSON（键顺序固定、浮点六位），写盘与比对都用这一份文本。
    pub fn render(&self) -> String {
        let mut out = String::new();
        out.push_str("{\n");
        let _ = writeln!(out, "  \"schema\": \"{}\",", escape(SCHEMA));
        let _ = writeln!(out, "  \"file_len\": {},", self.file_len);
        match self.tensors.split_last() {
            None => out.push_str("  \"tensors\": []\n"),
            Some((last, head)) => {
                out.push_str("  \"tensors\": [\n");
                for entry in head {
                    let _ = writeln!(out, "{},", entry.render());
                }
                let _ = writeln!(out, "{}", last.render());
                out.push_str("  ]\n");
            }
        }
        out.push_str("}\n");
        out
    }

    /// 读回一份账本文本，并与同一份落盘字节现算的账**逐格**核对。
    ///
    /// 这是账本唯一的消费方式：[`Settlement::render`] 写出的每一格都得能被这里核回，否则那一格
    /// 就只是「写入者说了什么」。两处判据都是硬的：
    ///
    /// - **按规范写法比，不按数值比**：`volume_ratio` 与 `entropy_H` 是 `{:.6}` 的产物，读回的
    ///   `0.5` 与写出的 `0.500000` 不算相等。账本是同一份字节的规范产物，不是可自由排版的配置，
    ///   所以「差一点点也算对」在这里没有位置。
    /// - **条目与段表一一对应**：多一条、少一条、换次序都是 [`Code::Invalid`]。账本描述的是这一份
    ///   字节，错位的账没有意义。
    /// - **多一格也不行**：条目里出现现算不出的键（例如手写的 `"note"`），就是描述了这份字节里
    ///   不存在的东西；「不能少」与「不能多」是同一个判据的两面。
    ///
    /// 键序与空白不参与比对（按名取键），这是它比逐字节比对宽松的唯一一处。
    ///
    /// # Errors
    ///
    /// 文本不是合法 JSON、缺键、某格与现算不符，或容器本身解不开，一律 [`Code::Invalid`]；消息里
    /// 带上出错的那一格与它的主人，避免「账本有问题」这种无法定位的结论。
    pub fn verify(text: &str, bytes: &[u8]) -> Result<Self> {
        let settled = Self::settle(bytes)?;
        let parsed = json::parse_ledger(text)?;
        verify_root(parsed.object()?, &settled)?;
        Ok(settled)
    }
}

/// 单段记账：解码往返先定 `bitwise`，再由 codec 与实测字节定量给出原因。
pub(crate) fn settle_one(archive: &Archive<'_>, seg: &Stored) -> Result<Settled> {
    let raw_len = seg.logical_len()?;
    // 引用段的 `len()` 是代表段的帧长；它自己一个字节都不占，所以体积要按 0 记。
    let volume = if seg.is_ref() { 0 } else { seg.len() };
    let (entropy, bitwise) = match archive.load(seg) {
        Ok(bytes) => (entropy_of(&bytes), bytes.len() == raw_len),
        // 解不开也是账本的一部分：这一条就是"压缩形态没能还原"，而不是"没量到"。
        Err(_) => (0.0, false),
    };
    Ok(Settled {
        name: seg.name().to_string(),
        layer: layer_of(seg.name()),
        raw_len,
        volume,
        entropy,
        quant: seg
            .quant()
            .map(|quant| (bits_per(seg, volume), quant.block)),
        bitwise,
        lossless: lossless_entry(seg) && bitwise,
        reason: reason_of(seg, raw_len, volume, entropy, bitwise),
        cert: certificate_of(archive, seg),
    })
}

/// 证书列：段里带的证书自己说不说得通，与“省了多少字节”是两件事。
///
/// 只有秩轴段会签发证书，其余轴（字典 / 稀疏 / 量化）与去重引用本来就没有证书可言，记 `none`
/// 不是“没查到”而是“本不该有”。秩轴段却读不出自洽证书时记 `refused`：那条压缩自称有谱隙
/// 背书，而背书缺失或自相矛盾，字节再小也不能算“压成了”。
fn certificate_of(archive: &Archive<'_>, seg: &Stored) -> Cert {
    if seg.is_ref() || seg.codec() != Codec::Axes {
        return Cert::None;
    }
    match axes::certificate(archive.read(seg)) {
        Ok(Some(note)) => sound_cert(&note),
        // 能解成别的轴，说明这条根本没用秩选择。
        Ok(None) => Cert::None,
        // 自己的帧都读不成轴，而 `load` 又成功了：证书不在段里。
        Err(_) => Cert::Refused,
    }
}

/// 证书自洽才记 `l0_diagnostic`：不变式全过，且方法与界都是能当上界的那两种。
///
/// 写侧只产出这一种口径，所以读侧不再细分：落盘的不是实测残差或不是 L0 诊断界，都只可能是
/// 别人改过或写坏了，一律按“背不住”处理。
pub(crate) fn sound_cert(note: &ledger::Note) -> Cert {
    let usable =
        note.method == ledger::EpsMethod::Residual && note.bound == ledger::Bound::Diagnostic;
    if usable && ledger::check(note).is_ok() {
        Cert::Diagnostic
    } else {
        Cert::Refused
    }
}

/// 每元素实际占的位数：按打包后的字节数四舍五入，不读方案里的"名义位宽"。

#[cfg(test)]
#[path = "../tests/settle.rs"]
mod tests;
