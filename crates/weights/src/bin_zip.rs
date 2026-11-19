//! ZIP_STORED 归档只读解析：定位中央目录（含 ZIP64 回退）、逐条读出本地头与数据区间。
//!
//! 只支持未压缩（STORED）条目；压缩条目显式报错，不假装解压。字节区间全部走越界检查。

use diagnostic::Result;

use super::bad;

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
            .ok_or_else(|| bad(&format!("无后缀条目 `{suffix}`")))?;
        self.data(entry)
    }

    fn data(&self, entry: &ZipEntry) -> Result<&'a [u8]> {
        if entry.method != STORED {
            return Err(bad("条目经过压缩，零依赖读取器只支持 STORED"));
        }
        self.bytes
            .get(entry.start..entry.start + entry.len)
            .ok_or_else(|| bad("条目数据越界"))
    }
}

fn directory(bytes: &[u8]) -> Result<(usize, usize)> {
    let eocd = locate_eocd(bytes)?;
    let at = u32::from_le_bytes(le(bytes, eocd + 16)?) as usize;
    let total = u16::from_le_bytes(le(bytes, eocd + 10)?) as usize;
    if needs_zip64(at, total) {
        return zip64_directory(bytes, eocd);
    }
    Ok((at, total))
}

fn needs_zip64(at: usize, total: usize) -> bool {
    at == 0xFFFF_FFFF || total == 0xFFFF
}

fn zip64_directory(bytes: &[u8], eocd: usize) -> Result<(usize, usize)> {
    let z64 = zip64_record(bytes, eocd)?;
    let total = u64::from_le_bytes(le(bytes, z64 + 32)?) as usize;
    let at = u64::from_le_bytes(le(bytes, z64 + 48)?) as usize;
    Ok((at, total))
}

fn zip64_record(bytes: &[u8], eocd: usize) -> Result<usize> {
    let locator = find_sig_back(bytes, EOCD64_LOCATOR_SIG, eocd.saturating_sub(20))?;
    let z64 = u32::from_le_bytes(le(bytes, locator + 8)?) as usize;
    if u32::from_le_bytes(le(bytes, z64)?) != ZIP64_SIG {
        return Err(bad("ZIP64 EOCD 签名不符"));
    }
    Ok(z64)
}

fn entry(bytes: &[u8], at: usize) -> Result<(ZipEntry, usize)> {
    let central = Central::read(bytes, at)?;
    let start = local_start(bytes, central.local)?;
    let len = data_range(bytes, start, central.csize)?;
    let entry = ZipEntry {
        name: central.name,
        method: central.method,
        start,
        len,
    };
    Ok((entry, central.next))
}

struct Central {
    name: String,
    method: u16,
    csize: u64,
    local: usize,
    next: usize,
}

impl Central {
    fn read(bytes: &[u8], at: usize) -> Result<Self> {
        let fixed = fixed_fields(bytes, at)?;
        let name_len = u16::from_le_bytes([fixed[28], fixed[29]]) as usize;
        let extra_len = u16::from_le_bytes([fixed[30], fixed[31]]) as usize;
        let comment_len = u16::from_le_bytes([fixed[32], fixed[33]]) as usize;
        let name = entry_name(bytes, at, name_len)?;
        let (csize, local) = zip64_sizes(bytes, at, name_len, extra_len, fixed)?;
        Ok(Self {
            name,
            method: u16::from_le_bytes([fixed[10], fixed[11]]),
            csize,
            local,
            next: next_offset(at, name_len, extra_len, comment_len)?,
        })
    }
}

fn fixed_fields(bytes: &[u8], at: usize) -> Result<&[u8]> {
    let fixed = slice(bytes, at, 46)?;
    if u32::from_le_bytes([fixed[0], fixed[1], fixed[2], fixed[3]]) != CD_SIG {
        return Err(bad("中央目录条目签名不符"));
    }
    Ok(fixed)
}

fn entry_name(bytes: &[u8], at: usize, len: usize) -> Result<String> {
    String::from_utf8(slice(bytes, at + 46, len)?.to_vec()).map_err(|_| bad("条目名不是 UTF-8"))
}

fn zip64_sizes(
    bytes: &[u8],
    at: usize,
    name_len: usize,
    extra_len: usize,
    fixed: &[u8],
) -> Result<(u64, usize)> {
    let extra = at + 46 + name_len;
    let local_off = u32::from_le_bytes([fixed[42], fixed[43], fixed[44], fixed[45]]) as usize;
    let local = if local_off == 0xFFFF_FFFF {
        zip64_extra(bytes, extra, extra_len, 0)?
    } else {
        local_off
    };
    let csize = u32::from_le_bytes([fixed[20], fixed[21], fixed[22], fixed[23]]);
    let data_len = if csize == 0xFFFF_FFFF {
        zip64_extra(bytes, extra, extra_len, 1)? as u64
    } else {
        u64::from(csize)
    };
    Ok((data_len, local))
}

fn next_offset(at: usize, name_len: usize, extra_len: usize, comment_len: usize) -> Result<usize> {
    at.checked_add(46 + name_len + extra_len + comment_len)
        .ok_or_else(|| bad("中央目录偏移溢出"))
}

fn local_start(bytes: &[u8], local: usize) -> Result<usize> {
    if u32::from_le_bytes(le(bytes, local)?) != LOCAL_SIG {
        return Err(bad("本地头签名不符"));
    }
    let (name_len, extra_len) = local_lengths(bytes, local)?;
    let Some(start) = local.checked_add(30 + name_len + extra_len) else {
        return Err(bad("本地头偏移溢出"));
    };
    Ok(start)
}

fn local_lengths(bytes: &[u8], local: usize) -> Result<(usize, usize)> {
    Ok((
        u16::from_le_bytes(le(bytes, local + 26)?) as usize,
        u16::from_le_bytes(le(bytes, local + 28)?) as usize,
    ))
}

fn data_range(bytes: &[u8], start: usize, data_len: u64) -> Result<usize> {
    let len = data_len as usize;
    if start.checked_add(len).is_none_or(|stop| stop > bytes.len()) {
        return Err(bad("条目数据超出归档"));
    }
    Ok(len)
}

fn zip64_extra(bytes: &[u8], extra_start: usize, extra_len: usize, field: usize) -> Result<usize> {
    let stop = extra_start.saturating_add(extra_len);
    let mut at = extra_start;
    while at.checked_add(4).is_some_and(|end| end <= stop) {
        let header_id = u16::from_le_bytes(le(bytes, at)?);
        let size = u16::from_le_bytes(le(bytes, at + 2)?) as usize;
        let body = at + 4;
        if header_id == 0x0001 {
            return zip64_field(bytes, body, size, field);
        }
        at = body.saturating_add(size);
    }
    Err(bad("缺少 ZIP64 extra 字段"))
}

fn zip64_field(bytes: &[u8], body: usize, size: usize, field: usize) -> Result<usize> {
    let end = field
        .checked_add(1)
        .and_then(|slots| slots.checked_mul(8))
        .ok_or_else(|| bad("ZIP64 字段序号溢出"))?;
    if end > size {
        return Err(bad("ZIP64 extra 声明长度不足，字段被截断"));
    }
    Ok(u64::from_le_bytes(le(bytes, body.saturating_add(end.saturating_sub(8)))?) as usize)
}

fn locate_eocd(bytes: &[u8]) -> Result<usize> {
    let window = bytes.len().min(22 + 65535);
    find_sig_back(bytes, EOCD_SIG, bytes.len() - window)
}

fn find_sig_back(bytes: &[u8], sig: u32, from: usize) -> Result<usize> {
    let mut at = bytes.len().saturating_sub(4);
    while at >= from {
        if u32::from_le_bytes(le(bytes, at)?) == sig {
            return Ok(at);
        }
        if at == 0 {
            break;
        }
        at -= 1;
    }
    Err(bad("找不到 ZIP 签名，可能不是 ZIP 归档"))
}

fn slice(bytes: &[u8], at: usize, len: usize) -> Result<&[u8]> {
    let stop = at.checked_add(len).ok_or_else(|| bad("ZIP 字段偏移溢出"))?;
    bytes.get(at..stop).ok_or_else(|| bad("ZIP 字段越界"))
}

fn le<const N: usize>(bytes: &[u8], at: usize) -> Result<[u8; N]> {
    let s = slice(bytes, at, N)?;
    let mut out = [0; N];
    out.copy_from_slice(s);
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutil::real_bin_bytes;

    #[test]
    fn parses_real_torch_zip() {
        let Some(raw) = real_bin_bytes() else {
            return;
        };
        let zip = Zip::parse(&raw).expect("解析");
        assert!(zip.get_suffix("data.pkl").is_some());
        assert!(zip.get_suffix("/data/0").is_some());
        assert_eq!(zip.data_suffix("/data/0").expect("数据").len(), 96);
    }

    #[test]
    fn rejects_non_zip() {
        assert!(Zip::parse(b"not a zip archive at all").is_err());
    }

    #[test]
    fn slices_with_overflowing_offsets_are_errors_not_panics() {
        assert!(slice(b"abc", usize::MAX, 4).is_err());
        assert!(slice(b"abc", 1, usize::MAX).is_err());
    }

    #[test]
    fn zip64_fields_must_fit_the_declared_size() {
        let mut bytes = vec![0x01, 0x00, 8, 0];
        bytes.extend(7u64.to_le_bytes());
        bytes.extend(0xDEAD_BEEF_DEAD_BEEFu64.to_le_bytes());
        assert_eq!(zip64_extra(&bytes, 0, 12, 0).ok(), Some(7));
        assert!(zip64_extra(&bytes, 0, 12, 1).is_err());
    }

    #[test]
    fn zip64_extra_handles_absurd_lengths_without_overflow() {
        let bytes = vec![0x01, 0x00, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0];
        assert!(zip64_extra(&bytes, 0, usize::MAX, 0).is_ok());
        assert!(zip64_extra(&bytes, usize::MAX - 2, 12, 0).is_err());
    }
}
