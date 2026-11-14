//! 最小 ZIP 归档读取器，只服务于权重容器的读取需求：定位条目并把 STORED（不压缩）数据
//! 以借用切片交出，不复制。torch.save 把所有条目写成 ZIP_STORED，因此
//! 零依赖即可读；遇到 deflate 等压缩方法显式报错，不静默给错数据。
//!
//! 解析顺序：末尾的 End of Central Directory（EOCD）→ 中央目录逐条读元数据与本地头偏移 →
//! 每个本地头跳过文件名与 extra 字段定位数据起点。支持 ZIP64 的 0xFFFF/0xFFFFFFFF 哨兵探测。

use diagnostic::{Code, Error, Result};

const EOCD_SIG: u32 = 0x0605_4B50;
const CD_SIG: u32 = 0x0201_4B50;
const LOCAL_SIG: u32 = 0x0403_4B50;
const ZIP64_SIG: u32 = 0x0606_4B50;
const STORED: u16 = 0;
const EOCD64_LOCATOR_SIG: u32 = 0x0706_4B50;

/// 一个归档条目：数据在原始字节的 `[start, start+len)`。
#[derive(Debug, Clone)]
pub struct Entry {
    name: String,
    method: u16,
    start: usize,
    len: usize,
}

/// 已解析的 ZIP，持有调用方字节切片的借用。
pub struct Zip<'a> {
    bytes: &'a [u8],
    entries: Vec<Entry>,
}

impl<'a> Zip<'a> {
    /// 解析中央目录并定位全部条目。
    pub fn parse(bytes: &'a [u8]) -> Result<Self> {
        let (mut at, total) = directory(bytes)?;
        let mut entries = Vec::with_capacity(total);
        for _ in 0..total {
            let (entry, next) = entry(bytes, at)?;
            entries.push(entry);
            at = next;
        }
        Ok(Self { bytes, entries })
    }

    /// 按路径后缀取条目（顶层目录名随保存文件名变化，不能假设固定前缀）。
    #[must_use]
    pub fn get_suffix(&self, suffix: &str) -> Option<&Entry> {
        self.entries.iter().find(|e| e.name.ends_with(suffix))
    }

    /// 按后缀取数据切片。
    pub fn data_suffix(&self, suffix: &str) -> Result<&'a [u8]> {
        let entry = self
            .get_suffix(suffix)
            .ok_or_else(|| bad(&format!("无后缀条目 `{suffix}`")))?;
        self.data(entry)
    }

    /// 条目数据切片；仅支持 STORED，其他方法报错。
    pub fn data(&self, entry: &Entry) -> Result<&'a [u8]> {
        if entry.method != STORED {
            return Err(bad("条目经过压缩，零依赖读取器只支持 STORED"));
        }
        self.bytes
            .get(entry.start..entry.start + entry.len)
            .ok_or_else(|| bad("条目数据越界"))
    }
}

/// 中央目录起点与条目总数；ZIP64 哨兵把这两项改从扩展记录读。
fn directory(bytes: &[u8]) -> Result<(usize, usize)> {
    let eocd = locate_eocd(bytes)?;
    let at = u32le(bytes, eocd + 16)? as usize;
    let total = u16le(bytes, eocd + 10)? as usize;
    if needs_zip64(at, total) {
        return zip64_directory(bytes, eocd);
    }
    Ok((at, total))
}

/// 任一字段为全 1 就说明真实值在 ZIP64 记录里。
fn needs_zip64(at: usize, total: usize) -> bool {
    at == 0xFFFF_FFFF || total == 0xFFFF
}

/// ZIP64 目录：真实总数与中央目录偏移由扩展记录给出。
fn zip64_directory(bytes: &[u8], eocd: usize) -> Result<(usize, usize)> {
    let z64 = zip64_record(bytes, eocd)?;
    let total = u64le(bytes, z64 + 32)? as usize;
    let at = u64le(bytes, z64 + 48)? as usize;
    Ok((at, total))
}

/// ZIP64 EOCD 记录的位置：先找 locator，再核对记录签名。
fn zip64_record(bytes: &[u8], eocd: usize) -> Result<usize> {
    let locator = find_sig_back(bytes, EOCD64_LOCATOR_SIG, eocd.saturating_sub(20))?;
    let z64 = u32le(bytes, locator + 8)? as usize;
    if u32le(bytes, z64)? != ZIP64_SIG {
        return Err(bad("ZIP64 EOCD 签名不符"));
    }
    Ok(z64)
}

/// 一条中央目录记录：条目元数据与下一条记录偏移。
fn entry(bytes: &[u8], at: usize) -> Result<(Entry, usize)> {
    let central = Central::read(bytes, at)?;
    let start = local_start(bytes, central.local)?;
    let len = data_range(bytes, start, central.csize)?;
    let entry = Entry {
        name: central.name,
        method: central.method,
        start,
        len,
    };
    Ok((entry, central.next))
}

/// 中央目录记录的元数据；本地头偏移可能来自 ZIP64 extra。
struct Central {
    name: String,
    method: u16,
    csize: u64,
    local: usize,
    next: usize,
}

impl Central {
    /// 读一条记录：签名、字段、名字与 ZIP64 扩展。
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

/// 中央目录记录的 46 字节固定部分；签名必须在最前。
fn fixed_fields(bytes: &[u8], at: usize) -> Result<&[u8]> {
    let fixed = slice(bytes, at, 46)?;
    if u32::from_le_bytes([fixed[0], fixed[1], fixed[2], fixed[3]]) != CD_SIG {
        return Err(bad("中央目录条目签名不符"));
    }
    Ok(fixed)
}

/// 条目名：UTF-8；长度来自文件。
fn entry_name(bytes: &[u8], at: usize, len: usize) -> Result<String> {
    let name = slice(bytes, at + 46, len)?;
    String::from_utf8(name.to_vec()).map_err(|_| bad("条目名不是 UTF-8"))
}

/// 压缩后长度与本地头偏移；全 1 哨兵改从 ZIP64 extra 读。
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

/// 下一条中央目录记录的偏移；字段来自文件，加法必须检查。
fn next_offset(at: usize, name_len: usize, extra_len: usize, comment_len: usize) -> Result<usize> {
    at.checked_add(46 + name_len + extra_len + comment_len)
        .ok_or_else(|| bad("中央目录偏移溢出"))
}

/// 本地头之后的数据起点：跳过文件名与 extra。
fn local_start(bytes: &[u8], local: usize) -> Result<usize> {
    if u32le(bytes, local)? != LOCAL_SIG {
        return Err(bad("本地头签名不符"));
    }
    let (name_len, extra_len) = local_lengths(bytes, local)?;
    let Some(start) = local.checked_add(30 + name_len + extra_len) else {
        return Err(bad("本地头偏移溢出"));
    };
    Ok(start)
}

/// 本地头的文件名与 extra 长度。
fn local_lengths(bytes: &[u8], local: usize) -> Result<(usize, usize)> {
    Ok((
        u16le(bytes, local + 26)? as usize,
        u16le(bytes, local + 28)? as usize,
    ))
}

/// 条目数据长度；起点 + 长度必须落在归档内。
fn data_range(bytes: &[u8], start: usize, data_len: u64) -> Result<usize> {
    let len = data_len as usize;
    if start.checked_add(len).is_none_or(|stop| stop > bytes.len()) {
        return Err(bad("条目数据超出归档"));
    }
    Ok(len)
}

/// 在 ZIP64 extra 字段里按字段序号取 u64（0=未压缩大小，1=压缩大小，2/3=偏移/磁盘）。
///
/// 三个长度都直接来自文件，所以推进与取值一律用 `checked_add`/`saturating_add`：越界只会变成
/// 错误，不会变成算术溢出。
fn zip64_extra(bytes: &[u8], extra_start: usize, extra_len: usize, field: usize) -> Result<usize> {
    let stop = extra_start.saturating_add(extra_len);
    let mut at = extra_start;
    while at.checked_add(4).is_some_and(|end| end <= stop) {
        let header_id = u16le(bytes, at)?;
        let size = u16le(bytes, at + 2)? as usize;
        let body = at + 4;
        if header_id == 0x0001 {
            return zip64_field(bytes, body, size, field);
        }
        at = body.saturating_add(size);
    }
    Err(bad("缺少 ZIP64 extra 字段"))
}

/// 取 ZIP64 extra 的第 `field` 个 u64；字段必须完整落在子字段声明的 `size` 之内。
fn zip64_field(bytes: &[u8], body: usize, size: usize, field: usize) -> Result<usize> {
    let end = field
        .checked_add(1)
        .and_then(|slots| slots.checked_mul(8))
        .ok_or_else(|| bad("ZIP64 字段序号溢出"))?;
    if end > size {
        return Err(bad("ZIP64 extra 声明长度不足，字段被截断"));
    }
    Ok(u64le(bytes, body.saturating_add(end.saturating_sub(8)))? as usize)
}

fn locate_eocd(bytes: &[u8]) -> Result<usize> {
    // EOCD 无固定位置（前面有注释），从末尾在 22+65535 窗口内反向找签名。
    let window = bytes.len().min(22 + 65535);
    let from = bytes.len() - window;
    find_sig_back(bytes, EOCD_SIG, from)
}

fn find_sig_back(bytes: &[u8], sig: u32, from: usize) -> Result<usize> {
    let mut at = bytes.len().saturating_sub(4);
    while at >= from {
        if u32le(bytes, at)? == sig {
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

fn u16le(bytes: &[u8], at: usize) -> Result<u16> {
    let s = slice(bytes, at, 2)?;
    Ok(u16::from_le_bytes([s[0], s[1]]))
}

fn u32le(bytes: &[u8], at: usize) -> Result<u32> {
    let s = slice(bytes, at, 4)?;
    Ok(u32::from_le_bytes([s[0], s[1], s[2], s[3]]))
}

fn u64le(bytes: &[u8], at: usize) -> Result<u64> {
    let s = slice(bytes, at, 8)?;
    Ok(u64::from_le_bytes([
        s[0], s[1], s[2], s[3], s[4], s[5], s[6], s[7],
    ]))
}

fn bad(message: &str) -> Error {
    Error::new(Code::Invalid, message).with("zip")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_real_torch_zip() {
        // 真实 torch.save 归档由对拍脚本在 /tmp/pt/model.bin 生成；这里只在文件存在时校验。
        let path = "/tmp/pt/model.bin";
        let Ok(raw) = std::fs::read(path) else {
            return;
        };
        let zip = Zip::parse(&raw).expect("解析");
        assert!(zip.get_suffix("data.pkl").is_some());
        assert!(zip.get_suffix("/data/0").is_some());
        let data0 = zip.data_suffix("/data/0").expect("数据");
        assert_eq!(data0.len(), 96);
    }

    #[test]
    fn rejects_non_zip() {
        assert!(Zip::parse(b"not a zip archive at all").is_err());
    }

    #[test]
    fn slices_with_overflowing_offsets_are_errors_not_panics() {
        // 偏移来自文件字段（ZIP64 的 u64 原样转成 usize），所以 `at + len` 必须在 usize 上检查。
        assert!(slice(b"abc", usize::MAX, 4).is_err());
        assert!(slice(b"abc", 1, usize::MAX).is_err());
    }

    #[test]
    fn zip64_fields_must_fit_the_declared_size() {
        // 一个只声明 8 字节的 ZIP64 extra，后面故意跟 8 字节"邻居数据"。
        let mut bytes = vec![0x01, 0x00, 8, 0];
        bytes.extend(7u64.to_le_bytes());
        bytes.extend(0xDEAD_BEEF_DEAD_BEEFu64.to_le_bytes());
        assert_eq!(zip64_extra(&bytes, 0, 12, 0).ok(), Some(7));
        assert!(
            zip64_extra(&bytes, 0, 12, 1).is_err(),
            "字段 1 超出声明的 8 字节，不得把邻居数据当成字段值"
        );
    }

    #[test]
    fn zip64_extra_handles_absurd_lengths_without_overflow() {
        let bytes = vec![0x01, 0x00, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0];
        // 声明的 extra 长度是 u16，但函数是公开给内部调用的，按 usize 处理也不许溢出。
        assert!(zip64_extra(&bytes, 0, usize::MAX, 0).is_ok());
        assert!(zip64_extra(&bytes, usize::MAX - 2, 12, 0).is_err());
    }
}
