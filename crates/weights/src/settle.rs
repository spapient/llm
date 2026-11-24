//! 沉降账本：LLD §4.7 的 `config.json`——逐张量把"落盘后到底占了多少字节、为什么用这个编码"
//! 写成一份可复核的账。

use std::fmt::Write as _;

use diagnostic::Result;

use crate::axes;
use crate::bad;
use crate::json;
use crate::json::Value;
use crate::ledger::{self, SCHEMA};
use crate::seg::lossless_entry;
use crate::{Archive, Codec, Stored};

/// 小于该字节数的段不值得编码，原样落盘（`passthrough:small`）。
pub const SMALL_BYTES: usize = 128;

/// 至少省下这个比例才算真压了；不到就是 `below_gain`。
pub const GAIN_FLOOR: f64 = 0.02;

/// 逻辑字节的熵达到这个值（bit/byte）即视为没有冗余（`incompressible`）。
pub const MAX_ENTROPY: f64 = 7.9;

/// 沉降账本的判定阈值。默认值与 [`SMALL_BYTES`] / [`GAIN_FLOOR`] / [`MAX_ENTROPY`]
/// 三个线上常量一致；只有显式调用 [`Settlement::settle_with`] 才改变判定口径。
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Thresholds {
    /// 小于该字节数的段不值得编码，原样落盘（`passthrough:small`）。
    pub small_bytes: usize,
    /// 至少省下这个比例才算真压了；不到就是 `below_gain`。
    pub gain_floor: f64,
    /// 逻辑字节的熵达到这个值（bit/byte）即视为没有冗余（`incompressible`）。
    pub max_entropy: f64,
}

impl Default for Thresholds {
    fn default() -> Self {
        Self {
            small_bytes: SMALL_BYTES,
            gain_floor: GAIN_FLOOR,
            max_entropy: MAX_ENTROPY,
        }
    }
}

closed_enum! { @named
    /// `codec_reason` 闭集：**为什么**这一条占了现在这些字节。
    pub enum Reason {
        /// `store` 帧，且内容有冗余但没压——写侧没开压缩。
        Raw = 0 => "raw",
        /// `rans` 帧，且省下超过 [`GAIN_FLOOR`]。
        Rans = 1 => "rans",
        /// `rans` 帧，但省下的不足以抵一次重编码的代价。
        BelowGain = 2 => "below_gain",
        /// `store` 帧，熵已到 [`MAX_ENTROPY`]：内容真的没有冗余。
        Incompressible = 3 => "incompressible",
        /// 段本身小于 [`SMALL_BYTES`]：编码的固定开销比省下的多。
        Small = 4 => "passthrough:small",
        /// 解码往返不是逐位：压缩形态没能还原，必须拒绝。
        Mismatch = 5 => "fallback:mismatch",
        /// 五轴无损选码帧（本容器特有形态）。
        Axes = 6 => "axes",
        /// 去重引用：字节存在代表段里，本条目自身零字节。
        Deduplicated = 7 => "deduplicated",
    }
}

closed_enum! { @named
    /// 证书列：这条量的压缩有没有一张自洽的证书背书（LLD §4.7 的账本要说清「凭什么这么压」）。
    pub enum Cert {
        /// 不是秩轴段：这条量没经过谱隙证书，其余轴本来就不签发。
        None = 0 => "none",
        /// 秩轴段且证书自洽：界是 L0 诊断、`ε` 来自实测残差，不变式全部通过。
        Diagnostic = 1 => "l0_diagnostic",
        /// 秩轴段却读不出自洽证书：自称压了但背书缺失或自相矛盾。
        Refused = 2 => "refused",
    }
}

impl Cert {
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
        Self::settle_with(bytes, Thresholds::default())
    }

    /// 按给定阈值现算整份账；只改变原因判定，不改变任何读端校验。
    pub fn settle_with(bytes: &[u8], thresholds: Thresholds) -> Result<Self> {
        let archive = Archive::parse(bytes)?;
        let mut tensors = Vec::with_capacity(archive.segments().len());
        for seg in archive.segments() {
            tensors.push(settle_one(&archive, seg, thresholds)?);
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
    /// - **条目与段表一一对应**：多一条、少一条、换次序都是 [`diagnostic::Code::Invalid`]。账本描述的是这一份
    ///   字节，错位的账没有意义。
    /// - **多一格也不行**：条目里出现现算不出的键（例如手写的 `"note"`），就是描述了这份字节里
    ///   不存在的东西；「不能少」与「不能多」是同一个判据的两面。
    ///
    /// 键序与空白不参与比对（按名取键），这是它比逐字节比对宽松的唯一一处。
    ///
    /// # Errors
    ///
    /// 文本不是合法 JSON、缺键、某格与现算不符，或容器本身解不开，一律 [`diagnostic::Code::Invalid`]；消息里
    /// 带上出错的那一格与它的主人，避免「账本有问题」这种无法定位的结论。
    pub fn verify(text: &str, bytes: &[u8]) -> Result<Self> {
        Self::verify_with(text, bytes, Thresholds::default())
    }

    /// 按同一份阈值核对账本：写账与核账必须使用同一口径。
    pub fn verify_with(text: &str, bytes: &[u8], thresholds: Thresholds) -> Result<Self> {
        let settled = Self::settle_with(bytes, thresholds)?;
        let parsed = json::parse_ledger(text)?;
        verify_root(parsed.object()?, &settled)?;
        Ok(settled)
    }
}

/// 单段记账：解码往返先定 `bitwise`，再由 codec 与实测字节定量给出原因。
pub(crate) fn settle_one(
    archive: &Archive<'_>,
    seg: &Stored,
    thresholds: Thresholds,
) -> Result<Settled> {
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
        reason: reason_of(seg, raw_len, volume, entropy, bitwise, thresholds),
        cert: certificate_of(archive, seg),
    })
}

/// 证书列：段里带的证书自己说不说得通，与“省了多少字节”是两件事。
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
pub(crate) fn sound_cert(note: &ledger::Note) -> Cert {
    let usable =
        note.method == ledger::EpsMethod::Residual && note.bound == ledger::Bound::Diagnostic;
    if usable && ledger::check_note(note).is_ok() {
        Cert::Diagnostic
    } else {
        Cert::Refused
    }
}

// —— 记账原因（原 `reason` 平铺进 `settle`）：为什么选了这个 codec、有没有真的更小 ——

/// 每元素实际占的位数：按打包后的字节数四舍五入，不读方案里的"名义位宽"。
pub(crate) fn bits_per(seg: &Stored, volume: usize) -> u32 {
    let numel = tensor::product(seg.shape());
    if numel == 0 {
        return 0;
    }
    // `+ numel / 2` 是四舍五入：1.5 bit 记 2，不把逐组打包的余量说成"没量化"。
    let bits = (volume as u64).saturating_mul(8).saturating_add(numel / 2) / numel;
    u32::try_from(bits).unwrap_or(u32::MAX)
}

/// 原因判定：先看往返，再按容器形态分开答，同一条只落进一档。
pub(crate) fn reason_of(
    seg: &Stored,
    raw_len: usize,
    volume: usize,
    entropy: f64,
    bitwise: bool,
    thresholds: Thresholds,
) -> Reason {
    if !bitwise {
        return Reason::Mismatch;
    }
    if seg.is_ref() {
        return Reason::Deduplicated;
    }
    // 小段先答：编码的固定开销与熵无关，先判大小才不会把 8 字节的偏置说成"不可压"。
    if raw_len < thresholds.small_bytes {
        return Reason::Small;
    }
    frame_reason(seg.codec(), raw_len, volume, entropy, thresholds)
}

/// 按帧形态答：轴字典是被选中的无损形态，`store`/`rans` 才有"省没省"的问题。
pub(crate) fn frame_reason(
    codec: Codec,
    raw_len: usize,
    volume: usize,
    entropy: f64,
    thresholds: Thresholds,
) -> Reason {
    match codec {
        Codec::Axes => Reason::Axes,
        Codec::Store => store_reason(entropy, thresholds.max_entropy),
        Codec::Rans => rans_reason(raw_len, volume, thresholds.gain_floor),
    }
}

/// `store` 帧：熵到顶说明内容真的没冗余，否则是"有冗余但没编"。
pub(crate) fn store_reason(entropy: f64, max_entropy: f64) -> Reason {
    if entropy >= max_entropy {
        Reason::Incompressible
    } else {
        Reason::Raw
    }
}

/// `rans` 帧：省下的比例够不够抵一次重编码。
pub(crate) fn rans_reason(raw_len: usize, volume: usize, gain_floor: f64) -> Reason {
    if gain(raw_len, volume) < gain_floor {
        Reason::BelowGain
    } else {
        Reason::Rans
    }
}

/// 省下的比例：`1 − volume / raw_len`；逻辑长度为 0 时视作没省。
pub(crate) fn gain(raw_len: usize, volume: usize) -> f64 {
    if raw_len == 0 {
        return 0.0;
    }
    1.0 - volume as f64 / raw_len as f64
}

/// 字节流的 Shannon 熵（bit/byte）：256 桶直方图，空流记 0。
pub(crate) fn entropy_of(bytes: &[u8]) -> f64 {
    let mut counts = [0u64; 256];
    for &byte in bytes {
        counts[byte as usize] += 1;
    }
    let total = bytes.len() as f64;
    if total == 0.0 {
        return 0.0;
    }
    counts
        .iter()
        .filter(|&&count| count > 0)
        .map(|&count| {
            let p = count as f64 / total;
            -p * p.log2()
        })
        .sum()
}

/// 名字里第一个整数分量：`backbone.layers.12.mlp.w` 给 12，`embed` 给 `None`。
pub(crate) fn layer_of(name: &str) -> Option<u32> {
    name.split('.').find_map(|part| part.parse().ok())
}

// —— 账本正文的文本形态（原 `text` 平铺进 `settle`）：一端写，一端核回 ——

impl Settled {
    /// 单条目的规范 JSON（六格缩进，与 [`Settlement::render`] 拼在一起）。
    pub(crate) fn render(&self) -> String {
        let mut out = String::new();
        out.push_str("    {\n");
        let _ = writeln!(out, "      \"name\": \"{}\",", escape(&self.name));
        let _ = writeln!(out, "      \"layer\": {},", layer_json(self.layer));
        let _ = writeln!(out, "      \"raw_len\": {},", self.raw_len);
        let _ = writeln!(out, "      \"volume_bytes\": {},", self.volume);
        let _ = writeln!(out, "      \"volume_ratio\": {:.6},", self.volume_ratio());
        let _ = writeln!(out, "      \"entropy_H\": {:.6},", self.entropy);
        let (bits, group) = self.quant.unwrap_or((0, 0));
        let _ = writeln!(out, "      \"quant_bits\": {bits},");
        let _ = writeln!(out, "      \"quant_group_size\": {group},");
        let _ = writeln!(out, "      \"verify_bitwise_equal\": {},", self.bitwise);
        let _ = writeln!(out, "      \"is_lossless\": {},", self.lossless);
        let _ = writeln!(out, "      \"decision\": \"{}\",", self.decision());
        let _ = writeln!(out, "      \"codec_reason\": \"{}\",", self.reason.as_str());
        let _ = writeln!(out, "      \"cert\": \"{}\"", self.cert.as_str());
        out.push_str("    }");
        out
    }
}

/// 层号的 JSON 写法：没有层号就写 `null`，不写 0（0 是真实层号，混同会丢信息）。
pub(crate) fn layer_json(layer: Option<u32>) -> String {
    match layer {
        Some(layer) => layer.to_string(),
        None => "null".to_string(),
    }
}

/// JSON 字符串转义：张量名来自文件，不能默认"里面没有引号与控制字符"。
pub(crate) fn escape(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    for ch in text.chars() {
        push_escaped(&mut out, ch);
    }
    out
}

/// 逐字符转义：引号与反斜杠必须转义，换行用短形式，其余控制字符留给 [`push_char`]。
fn push_escaped(out: &mut String, ch: char) {
    match ch {
        '"' => out.push_str("\\\""),
        '\\' => out.push_str("\\\\"),
        '\n' => out.push_str("\\n"),
        other => push_char(out, other),
    }
}

/// 控制字符走 `\u00xx`（JSON 合法形式），其余原样。
fn push_char(out: &mut String, ch: char) {
    if (ch as u32) < 0x20 {
        let _ = write!(out, "\\u{:04x}", ch as u32);
    } else {
        out.push(ch);
    }
}

/// 顶层三格 + 条目数：多一格、少一格、数目不符都在这里作答。
pub(crate) fn verify_root(root: &[(String, Value)], settled: &Settlement) -> Result<()> {
    verify_header(root, settled)?;
    let listed = field(root, "tensors", "账本")?.array()?;
    verify_count(listed, settled.tensors().len())?;
    verify_rows(listed, settled.tensors())
}

/// 段表逐条比对：数目已在本层之前对齐，这里只问「每条账与它那一行是否同源」。
fn verify_rows(listed: &[Value], entries: &[Settled]) -> Result<()> {
    entries
        .iter()
        .zip(listed)
        .try_for_each(|(entry, row)| verify_row(row.object()?, entry))
}

/// 顶层的两个标量格：`schema` 与 `file_len` 都得按规范写法与这份字节相等。
fn verify_header(root: &[(String, Value)], settled: &Settlement) -> Result<()> {
    only(root, &["schema", "file_len", "tensors"], "账本")?;
    check_field(root, "schema", SCHEMA, "账本")?;
    check_field(root, "file_len", &settled.file_len().to_string(), "账本")
}

/// 条目数必须与段表一致：多一条、少一条都是错位的账。
fn verify_count(listed: &[Value], actual: usize) -> Result<()> {
    if listed.len() == actual {
        return Ok(());
    }
    Err(bad(format!(
        "账本列了 {} 个张量，这份字节有 {actual} 段",
        listed.len()
    )))
}

/// 一个条目的十二格：先挡掉多余的格，再逐格比规范写法。
fn verify_row(fields: &[(String, Value)], entry: &Settled) -> Result<()> {
    let want = expected(entry);
    let names: Vec<&str> = want.iter().map(|(key, _)| *key).collect();
    only(fields, &names, entry.name())?;
    for (key, value) in &want {
        check_field(fields, key, value, entry.name())?;
    }
    Ok(())
}

/// 键集合不得超出 `allowed`：账本只能描述这份字节里有的格。
fn only(fields: &[(String, Value)], allowed: &[&str], owner: &str) -> Result<()> {
    match fields
        .iter()
        .find(|(name, _)| !allowed.contains(&name.as_str()))
    {
        None => Ok(()),
        Some((extra, _)) => Err(bad(format!(
            "{owner} 的账本多出 `{extra}`：这份字节没有这一格"
        ))),
    }
}

/// 一格账的比对：键必须在、且按**规范写法**逐字等于现算值。
fn check_field(fields: &[(String, Value)], key: &str, want: &str, owner: &str) -> Result<()> {
    let text = written(field(fields, key, owner)?)?;
    if text == want {
        return Ok(());
    }
    Err(bad(format!(
        "{owner} 的 `{key}` 是 `{text}`，这份字节现算的是 `{want}`"
    )))
}

/// 按名取一格；缺键就是账本不完整（不是「这格可以不写」）。
fn field<'a>(fields: &'a [(String, Value)], key: &str, owner: &str) -> Result<&'a Value> {
    fields
        .iter()
        .find(|(name, _)| name == key)
        .map(|(_, value)| value)
        .ok_or_else(|| bad(format!("{owner} 的账本缺 `{key}`")))
}

/// 值在文本里的写法：字符串取内容、布尔与 `null` 取字面、数字取原文。
fn written(value: &Value) -> Result<String> {
    match value {
        Value::String(text) => Ok(text.clone()),
        Value::Bool(flag) => Ok(flag.to_string()),
        Value::Null => Ok("null".to_string()),
        Value::Integer(_) | Value::Decimal(_) => value.number_text(),
        other => Err(bad(format!("账本的这一格必须是标量，得到{}", other.kind()))),
    }
}

/// 一条目期望的十三格：键名与 [`Settled::render`] 同源同序，值按规范写法成文。
fn expected(entry: &Settled) -> [(&'static str, String); 13] {
    let (bits, group) = entry.quant().unwrap_or((0, 0));
    [
        ("name", entry.name().to_string()),
        ("layer", layer_json(entry.layer())),
        ("raw_len", entry.raw_len().to_string()),
        ("volume_bytes", entry.volume().to_string()),
        ("volume_ratio", format!("{:.6}", entry.volume_ratio())),
        ("entropy_H", format!("{:.6}", entry.entropy())),
        ("quant_bits", bits.to_string()),
        ("quant_group_size", group.to_string()),
        ("verify_bitwise_equal", entry.bitwise().to_string()),
        ("is_lossless", entry.lossless().to_string()),
        ("decision", entry.decision().to_string()),
        ("codec_reason", entry.reason().as_str().to_string()),
        ("cert", entry.cert().as_str().to_string()),
    ]
}

#[cfg(test)]
#[path = "../tests/settle.rs"]
mod tests;
