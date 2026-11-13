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
        let eocd = locate_eocd(bytes)?;
        let mut at = u32le(bytes, eocd + 16)? as usize;
        let mut total = u16le(bytes, eocd + 10)? as usize;

        // ZIP64 哨兵：真实计数/偏移在 ZIP64 EOCD 记录里。
        if at == 0xFFFF_FFFF || total == 0xFFFF {
            let locator = find_sig_back(bytes, EOCD64_LOCATOR_SIG, eocd.saturating_sub(20))?;
            let z64 = u32le(bytes, locator + 8)? as usize;
            if u32le(bytes, z64)? != ZIP64_SIG {
                return Err(bad("ZIP64 EOCD 签名不符"));
            }
            total = u64le(bytes, z64 + 32)? as usize;
            at = u64le(bytes, z64 + 48)? as usize;
        }

        let mut entries = Vec::with_capacity(total);
        for _ in 0..total {
            if u32le(bytes, at)? != CD_SIG {
                return Err(bad("中央目录条目签名不符"));
            }
            let method = u16le(bytes, at + 10)?;
            let csize = u32le(bytes, at + 20)? as u64;
            let name_len = u16le(bytes, at + 28)? as usize;
            let extra_len = u16le(bytes, at + 30)? as usize;
            let comment_len = u16le(bytes, at + 32)? as usize;
            let name_bytes = slice(bytes, at + 46, name_len)?;
            let name =
                String::from_utf8(name_bytes.to_vec()).map_err(|_| bad("条目名不是 UTF-8"))?;
            let local_off = u32le(bytes, at + 42)? as usize;
            let mut data_len = csize;
            let mut local = local_off;

            if local_off == 0xFFFF_FFFF {
                local = zip64_extra(bytes, at + 46 + name_len, extra_len, 0)?;
            }
            if csize == 0xFFFF_FFFF {
                data_len = zip64_extra(bytes, at + 46 + name_len, extra_len, 1)? as u64;
            }

            // 本地头：跳过文件名与 extra，得到数据起点。
            if u32le(bytes, local)? != LOCAL_SIG {
                return Err(bad("本地头签名不符"));
            }
            let local_name = u16le(bytes, local + 26)? as usize;
            let local_extra = u16le(bytes, local + 28)? as usize;
            let start = local
                .checked_add(30 + local_name + local_extra)
                .ok_or_else(|| bad("本地头偏移溢出"))?;
            let len = data_len as usize;
            if start.checked_add(len).is_none_or(|stop| stop > bytes.len()) {
                return Err(bad("条目数据超出归档"));
            }
            entries.push(Entry {
                name,
                method,
                start,
                len,
            });

            at = at
                .checked_add(46 + name_len + extra_len + comment_len)
                .ok_or_else(|| bad("中央目录偏移溢出"))?;
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

/// 在 ZIP64 extra 字段里按字段序号取 u64（0=未压缩大小，1=压缩大小，2/3=偏移/磁盘）。
fn zip64_extra(bytes: &[u8], extra_start: usize, extra_len: usize, field: usize) -> Result<usize> {
    let mut at = extra_start;
    let stop = extra_start + extra_len;
    while at + 4 <= stop {
        let header_id = u16le(bytes, at)?;
        let size = u16le(bytes, at + 2)? as usize;
        let body = at + 4;
        if header_id == 0x0001 {
            // ZIP64 extra 内字段按 u64 顺序排列，按需取。
            let off = body + field * 8;
            return Ok(u64le(bytes, off)? as usize);
        }
        at = body + size;
    }
    Err(bad("缺少 ZIP64 extra 字段"))
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
    bytes.get(at..at + len).ok_or_else(|| bad("ZIP 字段越界"))
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
}
