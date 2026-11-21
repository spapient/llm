//! [`settle`](crate::settle) 的机制层：单段记账、证书判定与账本逐格复核，
//! 公开面由 [`settle`](crate::settle) 以 `pub(crate)` 再导出。

use std::fmt::Write as _;

use diagnostic::Result;

use crate::axes;
use crate::json::Value;
use crate::ledger::{self, SCHEMA};
use crate::settle::{
    bad, Cert, Reason, Settled, Settlement, GAIN_FLOOR, MAX_ENTROPY, SMALL_BYTES,
};
use crate::stored::lossless_entry;
use crate::{Archive, Codec, Stored};

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
    check(root, "schema", SCHEMA, "账本")?;
    check(root, "file_len", &settled.file_len().to_string(), "账本")
}

/// 条目数必须与段表一致：多一条、少一条都是错位的账。
fn verify_count(listed: &[Value], actual: usize) -> Result<()> {
    if listed.len() == actual {
        return Ok(());
    }
    Err(bad(&format!(
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
        check(fields, key, value, entry.name())?;
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
        Some((extra, _)) => Err(bad(&format!(
            "{owner} 的账本多出 `{extra}`：这份字节没有这一格"
        ))),
    }
}

/// 一格账的比对：键必须在、且按**规范写法**逐字等于现算值。
fn check(fields: &[(String, Value)], key: &str, want: &str, owner: &str) -> Result<()> {
    let text = written(field(fields, key, owner)?)?;
    if text == want {
        return Ok(());
    }
    Err(bad(&format!(
        "{owner} 的 `{key}` 是 `{text}`，这份字节现算的是 `{want}`"
    )))
}

/// 按名取一格；缺键就是账本不完整（不是「这格可以不写」）。
fn field<'a>(fields: &'a [(String, Value)], key: &str, owner: &str) -> Result<&'a Value> {
    fields
        .iter()
        .find(|(name, _)| name == key)
        .map(|(_, value)| value)
        .ok_or_else(|| bad(&format!("{owner} 的账本缺 `{key}`")))
}

/// 值在文本里的写法：字符串取内容、布尔与 `null` 取字面、数字取原文。
///
/// 与 [`Settled::render`] 的写法是同一份约定——两边各写一套，逐格比对就变成两种格式之争。
fn written(value: &Value) -> Result<String> {
    match value {
        Value::String(text) => Ok(text.clone()),
        Value::Bool(flag) => Ok(flag.to_string()),
        Value::Null => Ok("null".to_string()),
        Value::Integer(_) | Value::Decimal(_) => value.number_text(),
        other => Err(bad(&format!(
            "账本的这一格必须是标量，得到{}",
            other.kind()
        ))),
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
