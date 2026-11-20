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

use crate::archive::lossless_entry;
use crate::ledger::SCHEMA;
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

/// 一张量的账：全部字段都在 [`Settlement::settle`] 里由落盘字节现算。
#[derive(Clone, Debug)]
pub struct Settled {
    name: String,
    layer: Option<u32>,
    raw_len: usize,
    volume: usize,
    entropy: f64,
    quant: Option<(u32, u32)>,
    bitwise: bool,
    lossless: bool,
    reason: Reason,
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

    /// 账本里的判定：确实比原样小才叫 `compress`，否则 `reject`。
    pub fn decision(&self) -> &'static str {
        if self.volume < self.raw_len {
            "compress"
        } else {
            "reject"
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
}

impl Settled {
    /// 单条目的规范 JSON（六格缩进，与 [`Settlement::render`] 拼在一起）。
    fn render(&self) -> String {
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
        let _ = writeln!(out, "      \"codec_reason\": \"{}\"", self.reason.as_str());
        out.push_str("    }");
        out
    }
}

/// 层号的 JSON 写法：没有层号就写 `null`，不写 0（0 是真实层号，混同会丢信息）。
fn layer_json(layer: Option<u32>) -> String {
    match layer {
        Some(layer) => layer.to_string(),
        None => "null".to_string(),
    }
}

/// 单段记账：解码往返先定 `bitwise`，再由 codec 与实测字节定量给出原因。
fn settle_one(archive: &Archive<'_>, seg: &Stored) -> Result<Settled> {
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
    })
}

/// 每元素实际占的位数：按打包后的字节数四舍五入，不读方案里的"名义位宽"。
fn bits_per(seg: &Stored, volume: usize) -> u32 {
    let numel = tensor::product(seg.shape());
    if numel == 0 {
        return 0;
    }
    // `+ numel / 2` 是四舍五入：1.5 bit 记 2，不把逐组打包的余量说成"没量化"。
    let bits = (volume as u64).saturating_mul(8).saturating_add(numel / 2) / numel;
    u32::try_from(bits).unwrap_or(u32::MAX)
}

/// 原因判定：先看往返，再按容器形态分开答，同一条只落进一档。
fn reason_of(seg: &Stored, raw_len: usize, volume: usize, entropy: f64, bitwise: bool) -> Reason {
    if !bitwise {
        return Reason::Mismatch;
    }
    if seg.is_ref() {
        return Reason::Deduplicated;
    }
    // 小段先答：编码的固定开销与熵无关，先判大小才不会把 8 字节的偏置说成"不可压"。
    if raw_len < SMALL_BYTES {
        return Reason::Small;
    }
    frame_reason(seg.codec(), raw_len, volume, entropy)
}

/// 按帧形态答：轴字典是被选中的无损形态，`store`/`rans` 才有"省没省"的问题。
fn frame_reason(codec: Codec, raw_len: usize, volume: usize, entropy: f64) -> Reason {
    match codec {
        Codec::Axes => Reason::Axes,
        Codec::Store => store_reason(entropy),
        Codec::Rans => rans_reason(raw_len, volume),
    }
}

/// `store` 帧：熵到顶说明内容真的没冗余，否则是"有冗余但没编"。
fn store_reason(entropy: f64) -> Reason {
    if entropy >= MAX_ENTROPY {
        Reason::Incompressible
    } else {
        Reason::Raw
    }
}

/// `rans` 帧：省下的比例够不够抵一次重编码。
fn rans_reason(raw_len: usize, volume: usize) -> Reason {
    if gain(raw_len, volume) < GAIN_FLOOR {
        Reason::BelowGain
    } else {
        Reason::Rans
    }
}

/// 省下的比例：`1 − volume / raw_len`；逻辑长度为 0 时视作没省。
fn gain(raw_len: usize, volume: usize) -> f64 {
    if raw_len == 0 {
        return 0.0;
    }
    1.0 - volume as f64 / raw_len as f64
}

/// 字节流的 Shannon 熵（bit/byte）：256 桶直方图，空流记 0。
fn entropy_of(bytes: &[u8]) -> f64 {
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
fn layer_of(name: &str) -> Option<u32> {
    name.split('.').find_map(|part| part.parse().ok())
}

/// JSON 字符串转义：张量名来自文件，不能默认"里面没有引号与控制字符"。
fn escape(text: &str) -> String {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::seg::Segment;
    use crate::testutil::f32_segment;
    use crate::writer::Writer;
    use crate::{Quant, Wire};

    /// 写入并 settle：`compress` 决定写侧是否尝试熵编码。
    fn settled(segments: Vec<Segment>, compress: bool) -> Settlement {
        let mut writer = Writer::new();
        writer.compress(compress);
        for segment in segments {
            writer.push(segment).expect("入写");
        }
        let bytes = writer.finish(6).expect("落盘");
        Settlement::settle(&bytes).expect("记账")
    }

    /// 0/1 交替的 f32 流：重复度高，写侧会选中熵编码。
    fn compressible(len: usize) -> Vec<f32> {
        (0..len).map(|i| (i % 2) as f32).collect()
    }

    /// 只有一条时取出它：多条说明用例自己的构造错了。
    fn only(settlement: &Settlement) -> &Settled {
        let [entry] = settlement.tensors() else {
            panic!("应当只有一条，实际 {}", settlement.tensors().len());
        };
        entry
    }

    #[test]
    fn a_stored_tensor_reports_its_dense_length_and_no_gain() {
        let values: Vec<f32> = (1..=64).map(|i| i as f32).collect();
        let settlement = settled(vec![f32_segment("backbone.layers.3.mlp.w", &values)], false);
        let entry = only(&settlement);
        assert_eq!(entry.name(), "backbone.layers.3.mlp.w");
        assert_eq!(entry.layer(), Some(3), "层号取名字里第一个整数分量");
        assert_eq!(entry.raw_len(), 256, "64 个 f32");
        assert_eq!(entry.volume(), entry.raw_len(), "原样落盘");
        assert_eq!(entry.volume_ratio(), 1.0);
        assert_eq!(entry.decision(), "reject", "没省就是 reject");
        assert_eq!(entry.reason(), Reason::Raw);
        assert!(entry.bitwise() && entry.lossless());
    }

    #[test]
    fn a_compressed_tensor_reports_the_measured_gain() {
        let settlement = settled(vec![f32_segment("w", &compressible(1024))], true);
        let entry = only(&settlement);
        assert_eq!(entry.reason(), Reason::Rans);
        assert_eq!(entry.decision(), "compress");
        assert!(entry.volume() < entry.raw_len());
        let measured = entry.volume() as f64 / 4096.0;
        assert!((entry.volume_ratio() - measured).abs() < 1e-12);
        assert!(entry.bitwise(), "解出来的长度必须等于逻辑长度");
        assert!(entry.lossless());
    }

    #[test]
    fn a_tiny_tensor_is_passed_through_instead_of_coded() {
        let settlement = settled(vec![f32_segment("bias", &[1.0, 2.0, 3.0])], true);
        let entry = only(&settlement);
        assert_eq!(entry.raw_len(), 12);
        assert_eq!(entry.reason(), Reason::Small);
        assert_eq!(entry.decision(), "reject");
        assert!(entry.lossless(), "原样落盘仍然无损");
    }

    #[test]
    fn a_deduplicated_tensor_occupies_no_bytes_and_says_so() {
        let values = compressible(512);
        let settlement = settled(
            vec![f32_segment("w", &values), f32_segment("w.tied", &values)],
            true,
        );
        assert_eq!(settlement.tensors().len(), 2);
        let tied = &settlement.tensors()[1];
        assert_eq!(tied.volume(), 0, "引用段自身零字节");
        assert_eq!(tied.volume_ratio(), 0.0);
        assert_eq!(tied.reason(), Reason::Deduplicated);
        assert_eq!(tied.decision(), "compress");
        assert_eq!(tied.raw_len(), 2048, "逻辑长度仍是这张量自己的");
    }

    #[test]
    fn a_quantized_tensor_is_never_lossless_and_reports_real_bits() {
        let quant = Quant::new(1, 1, 0, false, Wire::F16, None).expect("方案");
        let segment =
            Segment::new("q", Wire::U8, vec![64], Some(quant), vec![0u8; 64]).expect("段");
        let settlement = settled(vec![segment], false);
        let entry = only(&settlement);
        assert_eq!(entry.quant(), Some((8, 0)), "64 字节 64 元素即 8 bit");
        assert!(!entry.lossless(), "带量化方案的条目不是无损");
        assert!(!settlement.lossless());
    }

    #[test]
    fn entropy_is_measured_on_the_logical_bytes() {
        let settlement = settled(vec![f32_segment("zeros", &[0.0; 64])], false);
        let entry = only(&settlement);
        assert!(
            entry.entropy() < 1.0,
            "全零字节的熵接近 0：{}",
            entry.entropy()
        );
    }

    #[test]
    fn an_empty_tensor_does_not_divide_by_zero() {
        let segment = Segment::new("empty", Wire::F32, vec![0], None, Vec::new()).expect("段");
        let settlement = settled(vec![segment], true);
        let entry = only(&settlement);
        assert_eq!(entry.raw_len(), 0);
        assert_eq!(entry.volume_ratio(), 1.0);
        assert_eq!(entry.entropy(), 0.0);
        assert_eq!(settlement.total_ratio(), 1.0);
    }

    #[test]
    fn the_ledger_carries_every_contracted_key_in_a_stable_order() {
        let settlement = settled(vec![f32_segment("l.0.w", &compressible(256))], true);
        let text = settlement.render();
        assert_eq!(text, settlement.render(), "同一份账要给出同一份文本");
        assert!(text.starts_with("{\n  \"schema\": \"llm.cert.v1\",\n  \"file_len\": "));
        for key in [
            "name",
            "layer",
            "raw_len",
            "volume_bytes",
            "volume_ratio",
            "entropy_H",
            "quant_bits",
            "quant_group_size",
            "verify_bitwise_equal",
            "is_lossless",
            "decision",
            "codec_reason",
        ] {
            assert!(text.contains(&format!("\"{key}\": ")), "缺字段 {key}");
        }
        assert!(text.ends_with("  ]\n}\n"));
    }

    #[test]
    fn a_name_without_a_layer_index_maps_to_null() {
        let settlement = settled(vec![f32_segment("embed", &compressible(64))], true);
        assert_eq!(only(&settlement).layer(), None);
        assert!(settlement.render().contains("\"layer\": null"));
    }

    #[test]
    fn the_ledger_refuses_bytes_that_are_not_a_container() {
        let error = Settlement::settle(&[0u8; 8]).expect_err("不是容器");
        assert_eq!(error.code(), diagnostic::Code::Invalid);
    }

    #[test]
    fn the_ledger_agrees_with_the_container_metrics() {
        let values = compressible(1024);
        let mut writer = Writer::new();
        writer
            .push(f32_segment("a", &values))
            .expect("入写")
            .push(f32_segment("b", &values))
            .expect("入写");
        let bytes = writer.finish(6).expect("落盘");
        let settlement = Settlement::settle(&bytes).expect("记账");
        let archive = Archive::parse(&bytes).expect("打开");
        let metrics = archive.metrics().expect("度量");
        assert_eq!(settlement.lossless(), metrics.lossless());
        assert_eq!(settlement.file_len(), bytes.len());
    }

    #[test]
    fn every_reason_has_a_stable_name() {
        let names: Vec<&str> = [
            Reason::Raw,
            Reason::Rans,
            Reason::BelowGain,
            Reason::Incompressible,
            Reason::Small,
            Reason::Mismatch,
            Reason::Axes,
            Reason::Deduplicated,
        ]
        .iter()
        .map(|reason| reason.as_str())
        .collect();
        assert_eq!(
            names,
            [
                "raw",
                "rans",
                "below_gain",
                "incompressible",
                "passthrough:small",
                "fallback:mismatch",
                "axes",
                "deduplicated",
            ]
        );
    }

    #[test]
    fn strings_from_the_file_are_escaped() {
        assert_eq!(escape("a\"b\\c\n"), "a\\\"b\\\\c\\n");
        assert_eq!(escape("\u{1}"), "\\u0001");
    }
}
