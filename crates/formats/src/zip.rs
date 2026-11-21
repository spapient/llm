//! ZIP_STORED 归档只读解析：定位中央目录（含 ZIP64 回退）、逐条读出本地头与数据区间。

use diagnostic::Result;

use wire::bad;

use crate::bytes::{slice, u16_at, u32_at, u64_at};

const EOCD_SIG: u32 = 0x0605_4B50;
const CD_SIG: u32 = 0x0201_4B50;
const LOCAL_SIG: u32 = 0x0403_4B50;
const ZIP64_SIG: u32 = 0x0606_4B50;
const STORED: u16 = 0;
const EOCD64_LOCATOR_SIG: u32 = 0x0706_4B50;
/// 条目最小 46+1 字节；预分配上限 4096，谎报计数先按剩余字节粗查。
const MIN_ENTRY_BYTES: usize = 47;
const MAX_ENTRY_PREALLOC: usize = 4096;

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

fn directory(bytes: &[u8]) -> Result<(usize, usize)> {
    let eocd = locate_eocd(bytes)?;
    let at = u32_at(bytes, eocd + 16)?;
    let total = u16_at(bytes, eocd + 10)?;
    if at == 0xFFFF_FFFF || total == 0xFFFF {
        return zip64_directory(bytes, eocd);
    }
    Ok((at as usize, usize::from(total)))
}

fn zip64_directory(bytes: &[u8], eocd: usize) -> Result<(usize, usize)> {
    let locator = eocd
        .checked_sub(20)
        .ok_or_else(|| bad("ZIP64 定位器不完整"))?;
    check_sig(bytes, locator, EOCD64_LOCATOR_SIG, "ZIP64 定位器签名不符")?;
    let z64 = usize::try_from(u64_at(bytes, locator + 8)?)
        .map_err(|_| bad("ZIP64 EOCD 偏移超出 usize"))?;
    check_sig(bytes, z64, ZIP64_SIG, "ZIP64 EOCD 签名不符")?;
    let total = usize::try_from(u64_at(bytes, z64 + 32)?).map_err(|_| bad("条目数超出 usize"))?;
    let at =
        usize::try_from(u64_at(bytes, z64 + 48)?).map_err(|_| bad("中央目录偏移超出 usize"))?;
    Ok((at, total))
}

fn check_sig(bytes: &[u8], at: usize, sig: u32, msg: &str) -> Result<()> {
    if u32_at(bytes, at)? == sig {
        return Ok(());
    }
    Err(bad(msg))
}

/// 一条中央目录记录：签名、名字、方法、ZIP64 尺寸与本地头偏移，并给出下一条的起点。
fn entry(bytes: &[u8], at: usize) -> Result<(ZipEntry, usize)> {
    check_sig(bytes, at, CD_SIG, "中央目录条目签名不符")?;
    let name_len = usize::from(u16_at(bytes, at + 28)?);
    let extra_len = usize::from(u16_at(bytes, at + 30)?);
    let comment_len = usize::from(u16_at(bytes, at + 32)?);
    let name = String::from_utf8(slice(bytes, at + 46, name_len)?.to_vec())
        .map_err(|_| bad("条目名不是 UTF-8"))?;
    let (csize, local) = zip64_sizes(bytes, at, name_len)?;
    let start = local_start(bytes, local)?;
    let len = usize::try_from(csize).map_err(|_| bad("条目长度超出 usize"))?;
    if start.checked_add(len).is_none_or(|stop| stop > bytes.len()) {
        return Err(bad("条目数据超出归档"));
    }
    let next = at
        .checked_add(46 + name_len + extra_len + comment_len)
        .ok_or_else(|| bad("中央目录偏移溢出"))?;
    let method = u16_at(bytes, at + 10)?;
    let record = ZipEntry {
        name,
        method,
        start,
        len,
    };
    Ok((record, next))
}

/// 中央目录的 32 位尺寸字段超限时写成 `0xFFFF_FFFF`，真值按 APPNOTE 的固定顺序
/// （原长、压缩长、本地头偏移）落在 ZIP64 extra 里，且**只**收下被截断的那些——
/// 于是槽位是「前面被截断的字段个数」，不是字段序号。
fn zip64_sizes(bytes: &[u8], at: usize, name_len: usize) -> Result<(u64, usize)> {
    let extra = at + 46 + name_len;
    let extra_len = usize::from(u16_at(bytes, at + 30)?);
    let mut slot = 0usize;
    let mut wide = [0u64; 3];
    for (index, offset) in [24, 20, 42].into_iter().enumerate() {
        let narrow = u32_at(bytes, at + offset)?;
        wide[index] = if narrow == 0xFFFF_FFFF {
            let value = zip64_extra(bytes, extra, extra_len, slot)?;
            slot += 1;
            value
        } else {
            u64::from(narrow)
        };
    }
    let local = usize::try_from(wide[2]).map_err(|_| bad("本地头偏移超出 usize"))?;
    Ok((wide[1], local))
}

fn local_start(bytes: &[u8], local: usize) -> Result<usize> {
    check_sig(bytes, local, LOCAL_SIG, "本地头签名不符")?;
    let name_len = usize::from(u16_at(bytes, local + 26)?);
    let extra_len = usize::from(u16_at(bytes, local + 28)?);
    let Some(start) = local.checked_add(30 + name_len + extra_len) else {
        return Err(bad("本地头偏移溢出"));
    };
    Ok(start)
}

fn zip64_extra(bytes: &[u8], extra_start: usize, extra_len: usize, field: usize) -> Result<u64> {
    let stop = extra_start.saturating_add(extra_len);
    let mut at = extra_start;
    while at.checked_add(4).is_some_and(|end| end <= stop) {
        let header_id = u16_at(bytes, at)?;
        let size = usize::from(u16_at(bytes, at + 2)?);
        let body = at + 4;
        if header_id == 0x0001 {
            return zip64_field(bytes, body, size, field);
        }
        at = body.saturating_add(size);
    }
    Err(bad("缺少 ZIP64 extra 字段"))
}

fn zip64_field(bytes: &[u8], body: usize, size: usize, field: usize) -> Result<u64> {
    let end = field
        .checked_add(1)
        .and_then(|slots| slots.checked_mul(8))
        .ok_or_else(|| bad("ZIP64 字段序号溢出"))?;
    if end > size {
        return Err(bad("ZIP64 extra 声明长度不足，字段被截断"));
    }
    u64_at(bytes, body.saturating_add(end - 8))
}

fn locate_eocd(bytes: &[u8]) -> Result<usize> {
    let window = bytes.len().min(22 + 65535);
    find_sig_back(bytes, EOCD_SIG, bytes.len() - window)
}

fn find_sig_back(bytes: &[u8], sig: u32, from: usize) -> Result<usize> {
    let mut at = bytes.len().saturating_sub(4);
    while at >= from {
        if u32_at(bytes, at)? == sig {
            return Ok(at);
        }
        if at == 0 {
            break;
        }
        at -= 1;
    }
    Err(bad("找不到 ZIP 签名，可能不是 ZIP 归档"))
}

#[cfg(test)]
#[path = "../tests/bin_zip.rs"]
mod tests;
